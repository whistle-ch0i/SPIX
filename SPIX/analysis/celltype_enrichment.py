from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import combine_pvalues, mannwhitneyu, norm, rankdata


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    if n == 0:
        return pvalues
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _fdr_bh_axis1(pvalue_matrix: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg correction across columns for each row."""
    p = np.asarray(pvalue_matrix, dtype=float)
    if p.ndim != 2:
        raise ValueError("pvalue_matrix must be 2D.")

    n_rows, n_cols = p.shape
    if n_cols == 0:
        return p.copy()

    order = np.argsort(p, axis=1)
    ranked = np.take_along_axis(p, order, axis=1)
    denom = np.arange(1, n_cols + 1, dtype=float)[None, :]
    q = ranked * n_cols / denom
    q = np.minimum.accumulate(q[:, ::-1], axis=1)[:, ::-1]
    q = np.clip(q, 0.0, 1.0)

    inverse_order = np.empty_like(order)
    row_idx = np.arange(n_rows)[:, None]
    inverse_order[row_idx, order] = np.arange(n_cols, dtype=order.dtype)
    return np.take_along_axis(q, inverse_order, axis=1)


def _resolve_gene_indices(
    var_names: np.ndarray,
    genes: Optional[Sequence[str]],
) -> Tuple[np.ndarray, list]:
    if genes is None:
        gene_idx = np.arange(var_names.size, dtype=np.int64)
        return gene_idx, var_names.tolist()

    var_to_idx = {g: i for i, g in enumerate(var_names.tolist())}
    seen = set()
    idx = []
    resolved_names = []
    for gene in genes:
        g = str(gene)
        if g in seen:
            continue
        seen.add(g)
        i = var_to_idx.get(g)
        if i is None:
            continue
        idx.append(i)
        resolved_names.append(g)

    if not idx:
        raise ValueError("No overlap between input genes and adata.var_names.")

    return np.asarray(idx, dtype=np.int64), resolved_names


def _resolve_background_gene_indices(
    var_names: np.ndarray,
    selected_gene_names: Sequence[str],
    background_genes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, list]:
    selected_set = set(str(g) for g in selected_gene_names)
    if background_genes is None:
        bg_names = [str(g) for g in var_names.tolist() if str(g) not in selected_set]
        bg_idx = np.asarray(
            [i for i, g in enumerate(var_names.tolist()) if str(g) not in selected_set],
            dtype=np.int64,
        )
        return bg_idx, bg_names

    bg_idx, bg_names = _resolve_gene_indices(var_names, background_genes)
    keep = np.asarray([str(g) not in selected_set for g in bg_names], dtype=bool)
    if not np.any(keep):
        raise ValueError("No usable background genes remain after excluding selected genes.")
    return bg_idx[keep], list(np.asarray(bg_names, dtype=object)[keep])


def _resolve_auto_min_cells(n_cells: int) -> int:
    return int(max(15, np.ceil(0.001 * float(n_cells))))


def _row_zscore_safe(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True, ddof=0)
    sd = np.maximum(sd, float(eps))
    return (x - mu) / sd


def _sample_expression_matched_gene_set(
    rng: np.random.Generator,
    selected_bin_counts: dict,
    bin_to_gene_idx: dict,
    fallback_gene_idx: np.ndarray,
    set_size: int,
) -> np.ndarray:
    sampled = []
    used = set()

    for bin_id, n_needed in selected_bin_counts.items():
        pool = np.asarray(bin_to_gene_idx.get(bin_id, []), dtype=np.int64)
        if pool.size == 0 or n_needed <= 0:
            continue
        pool = np.asarray([g for g in pool.tolist() if g not in used], dtype=np.int64)
        if pool.size == 0:
            continue
        take = min(int(n_needed), int(pool.size))
        chosen = rng.choice(pool, size=take, replace=False)
        sampled.extend(chosen.tolist())
        used.update(int(x) for x in chosen.tolist())

    n_missing = int(set_size) - len(sampled)
    if n_missing > 0:
        fallback = np.asarray([g for g in fallback_gene_idx.tolist() if int(g) not in used], dtype=np.int64)
        if fallback.size < n_missing:
            raise ValueError(
                "Background gene pool is too small to sample matched random gene sets without replacement."
            )
        chosen = rng.choice(fallback, size=n_missing, replace=False)
        sampled.extend(chosen.tolist())

    return np.asarray(sampled, dtype=np.int64)


def _compute_gene_set_score_vector(
    X_cells,
    gene_idx: np.ndarray,
    size_factors: np.ndarray,
) -> np.ndarray:
    if sparse.issparse(X_cells):
        raw = X_cells[:, gene_idx].toarray()
    else:
        raw = np.asarray(X_cells[:, gene_idx])
    raw = np.asarray(raw, dtype=np.float64)
    log_expr = np.log1p(raw * size_factors[:, None])
    return log_expr.mean(axis=1, dtype=np.float64)


def _compute_gene_level_celltype_log_stats(
    X_cells,
    gene_idx: np.ndarray,
    size_factors: np.ndarray,
    in_masks: Sequence[np.ndarray],
    n_in: np.ndarray,
    n_out: np.ndarray,
    *,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    n_genes = int(gene_idx.size)
    n_celltypes = len(in_masks)
    mean_in = np.empty((n_genes, n_celltypes), dtype=np.float64)
    mean_out = np.empty((n_genes, n_celltypes), dtype=np.float64)

    for start in range(0, n_genes, int(chunk_size)):
        stop = min(start + int(chunk_size), n_genes)
        chunk_idx = gene_idx[start:stop]
        if sparse.issparse(X_cells):
            raw_chunk = X_cells[:, chunk_idx].toarray()
        else:
            raw_chunk = np.asarray(X_cells[:, chunk_idx])
        raw_chunk = np.asarray(raw_chunk, dtype=np.float64)

        log_chunk = np.log1p(raw_chunk * size_factors[:, None])
        sum_all_log = log_chunk.sum(axis=0)

        for j, in_mask in enumerate(in_masks):
            sum_in_log = log_chunk[in_mask].sum(axis=0)
            mean_in[start:stop, j] = sum_in_log / float(n_in[j])
            mean_out[start:stop, j] = (sum_all_log - sum_in_log) / float(n_out[j])

    return mean_in, mean_out


def _permutation_batch_stats(
    bg_mean_diff_mat: np.ndarray,
    score_diff_obs: np.ndarray,
    selected_bin_counts: dict,
    bin_to_gene_idx: dict,
    fallback_gene_pos: np.ndarray,
    set_size: int,
    n_permutations: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_celltypes = int(bg_mean_diff_mat.shape[1])
    null_ge = np.zeros(n_celltypes, dtype=np.int64)
    null_sum = np.zeros(n_celltypes, dtype=np.float64)
    null_sq_sum = np.zeros(n_celltypes, dtype=np.float64)

    for _ in range(int(n_permutations)):
        rand_gene_idx = _sample_expression_matched_gene_set(
            rng,
            selected_bin_counts=selected_bin_counts,
            bin_to_gene_idx=bin_to_gene_idx,
            fallback_gene_idx=fallback_gene_pos,
            set_size=int(set_size),
        )
        rand_diff = bg_mean_diff_mat[rand_gene_idx].mean(axis=0, dtype=np.float64)
        null_ge += rand_diff >= score_diff_obs
        null_sum += rand_diff
        null_sq_sum += rand_diff * rand_diff

    return null_ge, null_sum, null_sq_sum


def _perm_celltype_null_for_all_genes(
    log_mat: np.ndarray,
    score_diff_obs: np.ndarray,
    n_in: int,
    n_out: int,
    n_permutations: int,
    seed: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_cells, n_genes = log_mat.shape
    sum_all = log_mat.sum(axis=0)
    null_ge = np.zeros(n_genes, dtype=np.int64)
    null_sum = np.zeros(n_genes, dtype=np.float64)
    null_sq_sum = np.zeros(n_genes, dtype=np.float64)

    for start in range(0, int(n_permutations), int(batch_size)):
        stop = min(start + int(batch_size), int(n_permutations))
        for _ in range(start, stop):
            idx = rng.choice(n_cells, size=int(n_in), replace=False)
            sum_in = log_mat[idx].sum(axis=0)
            diff = (sum_in / float(n_in)) - ((sum_all - sum_in) / float(n_out))
            null_ge += diff >= score_diff_obs
            null_sum += diff
            null_sq_sum += diff * diff

    return null_ge, null_sum, null_sq_sum


def permutation_celltype_gene_set_enrichment(
    adata,
    genes: Sequence[str],
    celltype_col: str = "predicted_labels",
    layer: Optional[str] = "counts",
    target_sum: float = 1e4,
    min_cells: int = 15,
    alpha: float = 0.05,
    min_score_diff: float = 0.05,
    n_permutations: int = 2000,
    background_genes: Optional[Sequence[str]] = None,
    match_expression: bool = True,
    n_expression_bins: int = 20,
    n_jobs: int = 1,
    permutation_batch_size: int = 128,
    random_state: Optional[int] = 0,
) -> Tuple[pd.DataFrame, list]:
    """Empirical celltype enrichment for one gene set using random matched gene sets.

    Notes
    -----
    This is intended for the no-replicate setting where per-cell hypothesis tests are
    prone to pseudoreplication. It tests whether the observed gene-set score difference
    for a celltype exceeds what is expected from random gene sets of the same size.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1.")
    if n_expression_bins < 2:
        raise ValueError("n_expression_bins must be >= 2.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if permutation_batch_size < 1:
        raise ValueError("permutation_batch_size must be >= 1.")
    if celltype_col not in adata.obs.columns:
        raise KeyError(f"'{celltype_col}' not found in adata.obs")

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        X_source = adata.layers[layer]
    else:
        X_source = adata.X

    var_names = np.asarray(adata.var_names).astype(str, copy=False)
    gene_idx, genes_resolved = _resolve_gene_indices(var_names, genes)
    if len(genes_resolved) == 0:
        raise ValueError("No overlap between input genes and adata.var_names.")

    bg_idx, bg_names = _resolve_background_gene_indices(
        var_names,
        genes_resolved,
        background_genes=background_genes,
    )
    if bg_idx.size < gene_idx.size:
        raise ValueError(
            "Background gene pool is smaller than the selected gene set. "
            "Provide more background genes or reduce the gene set size."
        )

    keep_cells = adata.obs[celltype_col].notna().to_numpy()
    if not np.any(keep_cells):
        raise ValueError(f"No non-null labels found in adata.obs['{celltype_col}'].")

    ct = adata.obs.loc[keep_cells, celltype_col].astype("category")
    ct_arr = ct.to_numpy()
    ct_levels = ct.cat.categories.tolist()

    valid_celltypes = []
    for ct_name in ct_levels:
        n_in = int((ct_arr == ct_name).sum())
        n_out = int((ct_arr != ct_name).sum())
        if n_in >= min_cells and n_out >= min_cells:
            valid_celltypes.append(ct_name)
    if not valid_celltypes:
        raise ValueError("No valid celltypes after min_cells filtering.")

    is_sparse = sparse.issparse(X_source)
    if is_sparse:
        X_csr = X_source.tocsr() if not sparse.isspmatrix_csr(X_source) else X_source
        X_cells = X_csr[keep_cells]
        row_sums = np.asarray(X_cells.sum(axis=1)).ravel().astype(np.float64, copy=False)
        raw_gene_means = np.asarray(X_cells.mean(axis=0)).ravel().astype(np.float64, copy=False)
    else:
        X_dense = np.asarray(X_source)
        X_cells = X_dense[keep_cells]
        row_sums = np.asarray(X_cells.sum(axis=1, dtype=np.float64)).ravel()
        raw_gene_means = np.asarray(X_cells.mean(axis=0), dtype=np.float64).ravel()

    size_factors = np.zeros_like(row_sums, dtype=np.float64)
    np.divide(float(target_sum), row_sums, out=size_factors, where=row_sums > 0)

    in_masks = [ct_arr == ct_name for ct_name in valid_celltypes]
    n_in = np.asarray([mask.sum() for mask in in_masks], dtype=np.int64)
    n_out = int(keep_cells.sum()) - n_in

    combined_gene_idx = np.concatenate([gene_idx, bg_idx], axis=0).astype(np.int64, copy=False)
    combined_mean_in, combined_mean_out = _compute_gene_level_celltype_log_stats(
        X_cells,
        gene_idx=combined_gene_idx,
        size_factors=size_factors,
        in_masks=in_masks,
        n_in=n_in,
        n_out=n_out,
        chunk_size=512,
    )
    n_sel = int(gene_idx.size)
    sel_mean_in = combined_mean_in[:n_sel]
    sel_mean_out = combined_mean_out[:n_sel]
    bg_mean_in = combined_mean_in[n_sel:]
    bg_mean_out = combined_mean_out[n_sel:]

    mean_in = sel_mean_in.mean(axis=0, dtype=np.float64)
    mean_out = sel_mean_out.mean(axis=0, dtype=np.float64)
    score_diff = mean_in - mean_out
    bg_mean_diff = bg_mean_in - bg_mean_out

    bg_gene_means = np.asarray(raw_gene_means[bg_idx], dtype=np.float64)
    sel_gene_means = np.asarray(raw_gene_means[gene_idx], dtype=np.float64)

    if match_expression:
        combined_means = np.concatenate([bg_gene_means, sel_gene_means], axis=0)
        combined_log = np.log1p(np.maximum(combined_means, 0.0))
        quantiles = np.linspace(0.0, 1.0, int(n_expression_bins) + 1)
        edges = np.unique(np.quantile(combined_log, quantiles))
        if edges.size < 2:
            match_expression = False
        else:
            bg_bins = np.digitize(np.log1p(np.maximum(bg_gene_means, 0.0)), edges[1:-1], right=False)
            sel_bins = np.digitize(np.log1p(np.maximum(sel_gene_means, 0.0)), edges[1:-1], right=False)
            selected_bin_counts = {
                int(b): int((sel_bins == b).sum()) for b in np.unique(sel_bins).tolist()
            }
            bin_to_gene_idx = {
                int(b): np.flatnonzero(bg_bins == b).astype(np.int64) for b in np.unique(bg_bins).tolist()
            }
    if not match_expression:
        selected_bin_counts = {-1: int(gene_idx.size)}
        bin_to_gene_idx = {-1: np.arange(bg_idx.size, dtype=np.int64)}

    null_ge = np.zeros(len(valid_celltypes), dtype=np.int64)
    null_sum = np.zeros(len(valid_celltypes), dtype=np.float64)
    null_sq_sum = np.zeros(len(valid_celltypes), dtype=np.float64)
    batch_sizes = []
    n_remaining = int(n_permutations)
    while n_remaining > 0:
        take = min(int(permutation_batch_size), n_remaining)
        batch_sizes.append(take)
        n_remaining -= take

    base_seed = 0 if random_state is None else int(random_state)
    if int(n_jobs) == 1 or len(batch_sizes) == 1:
        for batch_id, batch_n in enumerate(batch_sizes):
            ge_i, sum_i, sq_sum_i = _permutation_batch_stats(
                bg_mean_diff,
                score_diff_obs=score_diff,
                selected_bin_counts=selected_bin_counts,
                bin_to_gene_idx=bin_to_gene_idx,
                fallback_gene_pos=np.arange(bg_idx.size, dtype=np.int64),
                set_size=int(gene_idx.size),
                n_permutations=int(batch_n),
                seed=base_seed + batch_id,
            )
            null_ge += ge_i
            null_sum += sum_i
            null_sq_sum += sq_sum_i
    else:
        max_workers = min(int(n_jobs), len(batch_sizes))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _permutation_batch_stats,
                    bg_mean_diff,
                    score_diff,
                    selected_bin_counts,
                    bin_to_gene_idx,
                    np.arange(bg_idx.size, dtype=np.int64),
                    int(gene_idx.size),
                    int(batch_n),
                    base_seed + batch_id,
                )
                for batch_id, batch_n in enumerate(batch_sizes)
            ]
            for fut in futures:
                ge_i, sum_i, sq_sum_i = fut.result()
                null_ge += ge_i
                null_sum += sum_i
                null_sq_sum += sq_sum_i

    pvalue = (null_ge + 1.0) / (float(n_permutations) + 1.0)
    qvalue = fdr_bh(pvalue)
    null_mean = null_sum / float(n_permutations)
    null_var = np.maximum(
        (null_sq_sum / float(n_permutations)) - (null_mean * null_mean),
        0.0,
    )
    null_sd = np.sqrt(null_var)
    zscore = np.divide(
        score_diff - null_mean,
        null_sd,
        out=np.zeros_like(score_diff),
        where=null_sd > 0,
    )

    out = pd.DataFrame(
        {
            "celltype": np.asarray(valid_celltypes, dtype=object),
            "gene_set_size": int(gene_idx.size),
            "n_background_genes": int(bg_idx.size),
            "n_in": n_in.astype(int),
            "n_out": n_out.astype(int),
            "mean_score_in": mean_in,
            "mean_score_out": mean_out,
            "score_diff": score_diff,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "zscore": zscore,
            "pvalue": pvalue,
            "qvalue": qvalue,
        }
    )
    out["is_enriched"] = (
        (out["qvalue"] <= float(alpha))
        & (out["score_diff"] >= float(min_score_diff))
        & (out["zscore"] > 0)
    )
    out = out.sort_values(
        ["qvalue", "score_diff", "zscore"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return out, valid_celltypes


def summarize_permutation_celltype_gene_set_enrichment(
    celltype_gene_set_stats: pd.DataFrame,
    pan_fraction: float = 1.0,
) -> pd.DataFrame:
    if celltype_gene_set_stats.empty:
        return pd.DataFrame(
            columns=[
                "enrichment_class",
                "n_enriched_celltypes",
                "top_celltype",
                "top_qvalue",
                "top_score_diff",
                "enriched_celltypes",
            ]
        )
    if pan_fraction <= 0:
        raise ValueError("pan_fraction must be > 0.")

    enriched = celltype_gene_set_stats[celltype_gene_set_stats["is_enriched"]].copy()
    n_celltypes = int(celltype_gene_set_stats["celltype"].nunique())
    pan_threshold = max(1, int(np.ceil(float(pan_fraction) * n_celltypes)))
    n_enriched = int(enriched["celltype"].nunique())

    if n_enriched == 0:
        enrichment_class = "none"
        top_row = celltype_gene_set_stats.sort_values(
            ["qvalue", "score_diff", "zscore"],
            ascending=[True, False, False],
        ).iloc[0]
        enriched_celltypes = ""
    elif n_enriched == 1:
        enrichment_class = "single"
        top_row = enriched.sort_values(
            ["qvalue", "score_diff", "zscore"],
            ascending=[True, False, False],
        ).iloc[0]
        enriched_celltypes = ",".join(enriched["celltype"].tolist())
    elif n_enriched >= pan_threshold:
        enrichment_class = "pan"
        top_row = enriched.sort_values(
            ["qvalue", "score_diff", "zscore"],
            ascending=[True, False, False],
        ).iloc[0]
        enriched_celltypes = ",".join(
            enriched.sort_values(
                ["qvalue", "score_diff", "zscore"],
                ascending=[True, False, False],
            )["celltype"].tolist()
        )
    else:
        enrichment_class = "multi"
        top_row = enriched.sort_values(
            ["qvalue", "score_diff", "zscore"],
            ascending=[True, False, False],
        ).iloc[0]
        enriched_celltypes = ",".join(
            enriched.sort_values(
                ["qvalue", "score_diff", "zscore"],
                ascending=[True, False, False],
            )["celltype"].tolist()
        )

    return pd.DataFrame(
        [
            {
                "enrichment_class": enrichment_class,
                "n_enriched_celltypes": n_enriched,
                "top_celltype": str(top_row["celltype"]),
                "top_qvalue": float(top_row["qvalue"]),
                "top_score_diff": float(top_row["score_diff"]),
                "enriched_celltypes": enriched_celltypes,
            }
        ]
    )


def _tie_term_per_gene(values: np.ndarray) -> np.ndarray:
    """Compute sum(t_i^3 - t_i) per gene column for tie correction."""
    n_genes = values.shape[1]
    out = np.zeros(n_genes, dtype=np.float64)
    for gi in range(n_genes):
        counts = np.unique(values[:, gi], return_counts=True)[1]
        if counts.size:
            c = counts.astype(np.float64, copy=False)
            out[gi] = np.sum(c * c * c - c)
    return out


def _mw_pvalues_greater_prerank(
    log_values: np.ndarray,
    in_masks: Sequence[np.ndarray],
    n_in: np.ndarray,
    n_out: np.ndarray,
    *,
    use_continuity: bool = True,
    tie_correction: bool = True,
) -> np.ndarray:
    """Asymptotic one-sided MWU p-values using one rank pass per gene."""
    n_cells, n_genes = log_values.shape
    n_celltypes = len(in_masks)
    pvalue_mat = np.ones((n_genes, n_celltypes), dtype=np.float64)
    if n_cells < 2 or n_genes == 0 or n_celltypes == 0:
        return pvalue_mat

    const_mask = np.isclose(np.ptp(log_values, axis=0), 0.0, atol=1e-12)
    if np.all(const_mask):
        return pvalue_mat

    test_values = log_values[:, ~const_mask]
    ranks = rankdata(test_values, axis=0, method="average")

    if tie_correction:
        tie_term = _tie_term_per_gene(test_values)
        denom = float(n_cells * (n_cells - 1))
        tie_factor = (n_cells + 1.0) - (tie_term / denom)
        tie_factor = np.maximum(tie_factor, 0.0)
    else:
        tie_factor = np.full(test_values.shape[1], n_cells + 1.0, dtype=np.float64)

    for j, in_mask in enumerate(in_masks):
        n1 = float(n_in[j])
        n2 = float(n_out[j])
        if n1 <= 0 or n2 <= 0:
            continue

        rank_sum_in = ranks[in_mask].sum(axis=0)
        u1 = rank_sum_in - n1 * (n1 + 1.0) / 2.0
        mu = n1 * n2 / 2.0
        sigma = np.sqrt((n1 * n2 / 12.0) * tie_factor)

        if use_continuity:
            z_num = u1 - mu - 0.5
        else:
            z_num = u1 - mu

        z = np.empty_like(z_num, dtype=np.float64)
        np.divide(z_num, sigma, out=z, where=sigma > 0)
        p = norm.sf(z)
        p = np.where(sigma <= 0, 1.0, p)
        pvalue_mat[~const_mask, j] = p

    return pvalue_mat


def classify_gene_celltype_enrichment(
    adata,
    genes: Optional[Sequence[str]] = None,
    celltype_col: str = "predicted_labels",
    layer: Optional[str] = "counts",
    target_sum: float = 1e4,
    normalization_scope: str = "all_genes",
    min_cells: int = 15,
    alpha: float = 0.05,
    min_mean_diff: float = 0.10,
    min_pct_diff: float = 0.05,
    min_pct_overall: float = 0.0,
    pan_fraction: float = 1.0,
    chunk_size: int = 256,
    mw_method: str = "asymptotic",
    mwu_backend: str = "prerank",
    use_continuity: bool = True,
    tie_correction: bool = True,
):
    """Classify genes by celltype enrichment using vectorized, chunked statistics.

    Notes
    -----
    - If `genes=None`, all genes in `adata.var_names` are used.
    - Normalization and log1p are computed on the selected genes, but the default
      count-normalization denominator is each cell's total expression across all
      genes in ``layer``. Set ``normalization_scope='selected_genes'`` to normalize
      by only the tested genes.
    - `mwu_backend='prerank'` is substantially faster than repeated SciPy MWU calls
      and matches `method='asymptotic'` when `tie_correction=True`.
    - Use `min_pct_overall > 0` to prefilter rarely expressed genes for speed.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1.")
    if min_pct_overall < 0.0 or min_pct_overall > 1.0:
        raise ValueError("min_pct_overall must be in [0, 1].")
    if mwu_backend not in {"prerank", "scipy"}:
        raise ValueError("mwu_backend must be one of {'prerank', 'scipy'}.")
    if normalization_scope not in {"all_genes", "selected_genes"}:
        raise ValueError("normalization_scope must be one of {'all_genes', 'selected_genes'}.")
    if pan_fraction <= 0:
        raise ValueError("pan_fraction must be > 0.")

    if celltype_col not in adata.obs.columns:
        raise KeyError(f"'{celltype_col}' not found in adata.obs")

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        X_source = adata.layers[layer]
    else:
        X_source = adata.X

    var_names = np.asarray(adata.var_names).astype(str, copy=False)
    gene_idx, genes_resolved = _resolve_gene_indices(var_names, genes)
    n_genes = len(genes_resolved)

    keep_cells = adata.obs[celltype_col].notna().to_numpy()
    if not np.any(keep_cells):
        raise ValueError(f"No non-null labels found in adata.obs['{celltype_col}'].")

    ct = adata.obs.loc[keep_cells, celltype_col].astype("category")
    ct_arr = ct.to_numpy()
    ct_levels = ct.cat.categories.tolist()

    valid_celltypes = []
    for ct_name in ct_levels:
        n_in = int((ct_arr == ct_name).sum())
        n_out = int((ct_arr != ct_name).sum())
        if n_in >= min_cells and n_out >= min_cells:
            valid_celltypes.append(ct_name)

    if not valid_celltypes:
        raise ValueError("No valid celltypes after min_cells filtering.")

    n_cells = int(keep_cells.sum())
    n_celltypes = len(valid_celltypes)
    in_masks = [ct_arr == ct_name for ct_name in valid_celltypes]
    n_in = np.asarray([mask.sum() for mask in in_masks], dtype=np.int64)
    n_out = n_cells - n_in

    is_sparse = sparse.issparse(X_source)
    all_cells_selected = bool(np.all(keep_cells))
    all_genes_selected = bool(
        (gene_idx.size == var_names.size)
        and np.array_equal(gene_idx, np.arange(var_names.size, dtype=np.int64))
    )

    if is_sparse:
        X_csr = X_source.tocsr() if not sparse.isspmatrix_csr(X_source) else X_source
        if all_cells_selected and all_genes_selected:
            X_kept = X_csr
            X_sel = X_csr
        elif all_cells_selected:
            X_kept = X_csr
            X_sel = X_csr[:, gene_idx]
        elif all_genes_selected:
            X_kept = X_csr[keep_cells]
            X_sel = X_kept
        else:
            X_kept = X_csr[keep_cells]
            X_sel = X_kept[:, gene_idx]
        X_sel = X_sel.tocsr()
        all_gene_row_sums = np.asarray(X_kept.sum(axis=1)).ravel().astype(np.float64, copy=False)
    else:
        X_dense = np.asarray(X_source)
        if all_cells_selected and all_genes_selected:
            X_kept = X_dense
            X_sel = X_dense
        elif all_cells_selected:
            X_kept = X_dense
            X_sel = X_dense[:, gene_idx]
        elif all_genes_selected:
            X_kept = X_dense[keep_cells]
            X_sel = X_kept
        else:
            X_kept = X_dense[keep_cells]
            X_sel = X_kept[:, gene_idx]
        all_gene_row_sums = np.asarray(X_kept.sum(axis=1, dtype=np.float64)).ravel()

    if min_pct_overall > 0.0:
        if is_sparse:
            nnz_gene = np.asarray(X_sel.getnnz(axis=0)).ravel().astype(np.float64, copy=False)
        else:
            nnz_gene = np.count_nonzero(X_sel, axis=0).astype(np.float64, copy=False)
        pct_gene = nnz_gene / max(1.0, float(n_cells))
        keep_gene_mask = pct_gene >= float(min_pct_overall)
        if not np.any(keep_gene_mask):
            raise ValueError(
                "No genes remain after applying min_pct_overall filtering."
            )
        if not np.all(keep_gene_mask):
            if is_sparse:
                X_sel = X_sel[:, keep_gene_mask]
            else:
                X_sel = X_sel[:, keep_gene_mask]
            genes_resolved = list(np.asarray(genes_resolved, dtype=object)[keep_gene_mask])
            n_genes = len(genes_resolved)

    if normalization_scope == "all_genes":
        row_sums = all_gene_row_sums
    elif is_sparse:
        row_sums = np.asarray(X_sel.sum(axis=1)).ravel().astype(np.float64, copy=False)
    else:
        row_sums = np.asarray(X_sel.sum(axis=1, dtype=np.float64)).ravel()

    size_factors = np.zeros_like(row_sums, dtype=np.float64)
    np.divide(float(target_sum), row_sums, out=size_factors, where=row_sums > 0)

    celltype_array = np.asarray(valid_celltypes, dtype=object)
    chunk_frames = []

    for start in range(0, n_genes, int(chunk_size)):
        stop = min(start + int(chunk_size), n_genes)
        chunk_len = stop - start

        if is_sparse:
            raw_chunk = X_sel[:, start:stop].toarray()
        else:
            raw_chunk = X_sel[:, start:stop]
        raw_chunk = np.asarray(raw_chunk, dtype=np.float64)

        lin_chunk = raw_chunk * size_factors[:, None]
        log_chunk = np.log1p(lin_chunk)
        expr_chunk = lin_chunk > 0
        sum_all_log = log_chunk.sum(axis=0)
        sum_all_lin = lin_chunk.sum(axis=0)
        sum_all_expr = expr_chunk.sum(axis=0, dtype=np.float64)

        mean_in_mat = np.empty((chunk_len, n_celltypes), dtype=np.float64)
        mean_out_mat = np.empty((chunk_len, n_celltypes), dtype=np.float64)
        mean_diff_mat = np.empty((chunk_len, n_celltypes), dtype=np.float64)
        pct_in_mat = np.empty((chunk_len, n_celltypes), dtype=np.float64)
        pct_out_mat = np.empty((chunk_len, n_celltypes), dtype=np.float64)
        pct_diff_mat = np.empty((chunk_len, n_celltypes), dtype=np.float64)
        log2_fc_mat = np.empty((chunk_len, n_celltypes), dtype=np.float64)
        pvalue_mat = np.ones((chunk_len, n_celltypes), dtype=np.float64)
        const_mask = np.isclose(np.ptp(log_chunk, axis=0), 0.0, atol=1e-12)

        for j, in_mask in enumerate(in_masks):
            n1 = float(n_in[j])
            n2 = float(n_out[j])

            sum_in_log = log_chunk[in_mask].sum(axis=0)
            sum_in_lin = lin_chunk[in_mask].sum(axis=0)
            sum_in_expr = expr_chunk[in_mask].sum(axis=0, dtype=np.float64)

            mean_in = sum_in_log / n1
            mean_out = (sum_all_log - sum_in_log) / n2
            pct_in = sum_in_expr / n1
            pct_out = (sum_all_expr - sum_in_expr) / n2
            lin_in = sum_in_lin / n1
            lin_out = (sum_all_lin - sum_in_lin) / n2

            mean_in_mat[:, j] = mean_in
            mean_out_mat[:, j] = mean_out
            mean_diff_mat[:, j] = mean_in - mean_out
            pct_in_mat[:, j] = pct_in
            pct_out_mat[:, j] = pct_out
            pct_diff_mat[:, j] = pct_in - pct_out
            log2_fc_mat[:, j] = np.log2((lin_in + 1e-8) / (lin_out + 1e-8))

            if mwu_backend == "scipy":
                test_mask = ~const_mask
                try:
                    if np.any(test_mask):
                        mw = mannwhitneyu(
                            log_chunk[in_mask][:, test_mask],
                            log_chunk[~in_mask][:, test_mask],
                            alternative="greater",
                            axis=0,
                            method=mw_method,
                            use_continuity=use_continuity,
                        )
                        pvalue_mat[test_mask, j] = np.asarray(mw.pvalue, dtype=np.float64)
                except ValueError:
                    pvalue_mat[test_mask, j] = 1.0

        if mwu_backend == "prerank":
            pvalue_mat = _mw_pvalues_greater_prerank(
                log_chunk,
                in_masks,
                n_in=n_in,
                n_out=n_out,
                use_continuity=use_continuity,
                tie_correction=tie_correction,
            )

        qvalue_mat = _fdr_bh_axis1(pvalue_mat)
        is_enriched_mat = (
            (qvalue_mat <= float(alpha))
            & (mean_diff_mat >= float(min_mean_diff))
            & (pct_diff_mat >= float(min_pct_diff))
            & (log2_fc_mat > 0)
        )

        gene_col = np.repeat(np.asarray(genes_resolved[start:stop], dtype=object), n_celltypes)
        celltype_col_flat = np.tile(celltype_array, chunk_len)

        frame = pd.DataFrame(
            {
                "gene": gene_col,
                "celltype": celltype_col_flat,
                "n_in": np.tile(n_in, chunk_len),
                "n_out": np.tile(n_out, chunk_len),
                "mean_in": mean_in_mat.reshape(-1),
                "mean_out": mean_out_mat.reshape(-1),
                "mean_diff": mean_diff_mat.reshape(-1),
                "pct_in": pct_in_mat.reshape(-1),
                "pct_out": pct_out_mat.reshape(-1),
                "pct_diff": pct_diff_mat.reshape(-1),
                "log2_fc": log2_fc_mat.reshape(-1),
                "pvalue": pvalue_mat.reshape(-1),
                "qvalue": qvalue_mat.reshape(-1),
                "is_enriched": is_enriched_mat.reshape(-1),
            }
        )
        chunk_frames.append(frame)

    gene_celltype_stats = (
        pd.concat(chunk_frames, axis=0, ignore_index=True)
        if chunk_frames
        else pd.DataFrame(
            columns=[
                "gene",
                "celltype",
                "n_in",
                "n_out",
                "mean_in",
                "mean_out",
                "mean_diff",
                "pct_in",
                "pct_out",
                "pct_diff",
                "log2_fc",
                "pvalue",
                "qvalue",
                "is_enriched",
            ]
        )
    )

    pan_threshold = max(1, int(np.ceil(float(pan_fraction) * n_celltypes)))
    summary_rows = []

    for gene, gdf in gene_celltype_stats.groupby("gene", sort=False):
        gdf_sorted = gdf.sort_values(
            ["qvalue", "pvalue", "log2_fc"],
            ascending=[True, True, False],
        )
        enriched = gdf_sorted[gdf_sorted["is_enriched"]].copy()

        n_enriched = int(enriched["celltype"].nunique())
        if n_enriched == 0:
            enrichment_class = "none"
        elif n_enriched == 1:
            enrichment_class = "single"
        elif n_enriched >= pan_threshold:
            enrichment_class = "pan"
        else:
            enrichment_class = "multi"

        if n_enriched > 0:
            top_row = enriched.sort_values(
                ["qvalue", "log2_fc", "mean_diff"],
                ascending=[True, False, False],
            ).iloc[0]
            enriched_celltypes = ",".join(
                enriched.sort_values(
                    ["qvalue", "log2_fc"],
                    ascending=[True, False],
                )["celltype"].tolist()
            )
        else:
            top_row = gdf_sorted.iloc[0]
            enriched_celltypes = ""

        summary_rows.append(
            {
                "gene": gene,
                "enrichment_class": enrichment_class,
                "n_enriched_celltypes": n_enriched,
                "top_celltype": top_row["celltype"],
                "top_qvalue": float(top_row["qvalue"]),
                "top_log2_fc": float(top_row["log2_fc"]),
                "enriched_celltypes": enriched_celltypes,
            }
        )

    gene_enrichment_summary = pd.DataFrame(summary_rows)
    class_order = pd.CategoricalDtype(
        categories=["single", "multi", "pan", "none"],
        ordered=True,
    )
    gene_enrichment_summary["enrichment_class"] = gene_enrichment_summary[
        "enrichment_class"
    ].astype(class_order)
    gene_enrichment_summary = gene_enrichment_summary.sort_values(
        ["enrichment_class", "n_enriched_celltypes", "top_qvalue", "top_log2_fc"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)

    enriched_pairs = gene_celltype_stats[gene_celltype_stats["is_enriched"]].copy()
    if enriched_pairs.empty:
        celltype_enrichment_summary = pd.DataFrame(
            columns=[
                "n_enriched_lowres_genes",
                "n_single_genes",
                "n_multi_genes",
                "n_pan_genes",
            ]
        )
    else:
        tmp = enriched_pairs[["gene", "celltype"]].merge(
            gene_enrichment_summary[["gene", "enrichment_class"]],
            on="gene",
            how="left",
        )
        n_total = tmp.groupby("celltype")["gene"].nunique().rename(
            "n_enriched_lowres_genes"
        )
        n_single = (
            tmp[tmp["enrichment_class"] == "single"]
            .groupby("celltype")["gene"]
            .nunique()
            .rename("n_single_genes")
        )
        n_multi = (
            tmp[tmp["enrichment_class"] == "multi"]
            .groupby("celltype")["gene"]
            .nunique()
            .rename("n_multi_genes")
        )
        n_pan = (
            tmp[tmp["enrichment_class"] == "pan"]
            .groupby("celltype")["gene"]
            .nunique()
            .rename("n_pan_genes")
        )

        celltype_enrichment_summary = (
            pd.concat([n_total, n_single, n_multi, n_pan], axis=1)
            .fillna(0)
            .astype(int)
            .sort_values(
                ["n_enriched_lowres_genes", "n_single_genes"],
                ascending=False,
            )
        )

    return (
        gene_celltype_stats,
        gene_enrichment_summary,
        celltype_enrichment_summary,
        valid_celltypes,
    )


def score_gene_celltype_specificity(
    adata,
    genes: Optional[Sequence[str]] = None,
    celltype_col: str = "predicted_labels",
    layer: Optional[str] = "counts",
    target_sum: float = 1e4,
    normalization_scope: str = "all_genes",
    min_cells: int | str = "auto",
    min_mean_diff: float = 0.05,
    min_pct_diff: float = 0.05,
    min_log2_fc: float = 0.25,
    min_margin: float = 0.75,
    pan_fraction: float = 1.0,
    score_weights: Tuple[float, float, float] = (0.2, 0.3, 0.5),
):
    """Effect-size-based gene-to-celltype specificity scoring.

    This is intended as a practical annotation tool when strict inference is either
    unavailable or less useful than stable effect-size ranking. By default, count
    normalization uses each cell's total expression across all genes in ``layer``;
    set ``normalization_scope='selected_genes'`` to reproduce subset-only
    normalization.
    """
    if celltype_col not in adata.obs.columns:
        raise KeyError(f"'{celltype_col}' not found in adata.obs")
    if len(score_weights) != 3:
        raise ValueError("score_weights must have length 3: (mean_diff, pct_diff, log2_fc).")
    if normalization_scope not in {"all_genes", "selected_genes"}:
        raise ValueError("normalization_scope must be one of {'all_genes', 'selected_genes'}.")
    if pan_fraction <= 0:
        raise ValueError("pan_fraction must be > 0.")

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        X_source = adata.layers[layer]
    else:
        X_source = adata.X

    var_names = np.asarray(adata.var_names).astype(str, copy=False)
    gene_idx, genes_resolved = _resolve_gene_indices(var_names, genes)
    n_genes = len(genes_resolved)

    keep_cells = adata.obs[celltype_col].notna().to_numpy()
    if not np.any(keep_cells):
        raise ValueError(f"No non-null labels found in adata.obs['{celltype_col}'].")
    n_cells_kept = int(keep_cells.sum())

    if isinstance(min_cells, str):
        if min_cells != "auto":
            raise ValueError("min_cells must be an integer or 'auto'.")
        min_cells_eff = _resolve_auto_min_cells(n_cells_kept)
    else:
        min_cells_eff = int(min_cells)
    if min_cells_eff < 1:
        raise ValueError("min_cells must be >= 1.")

    ct = adata.obs.loc[keep_cells, celltype_col].astype("category")
    ct_arr = ct.to_numpy()
    ct_levels = ct.cat.categories.tolist()

    valid_celltypes = []
    for ct_name in ct_levels:
        n_in_ct = int((ct_arr == ct_name).sum())
        n_out_ct = int((ct_arr != ct_name).sum())
        if n_in_ct >= min_cells_eff and n_out_ct >= min_cells_eff:
            valid_celltypes.append(ct_name)
    if not valid_celltypes:
        raise ValueError("No valid celltypes after min_cells filtering.")

    is_sparse = sparse.issparse(X_source)
    if is_sparse:
        X_csr = X_source.tocsr() if not sparse.isspmatrix_csr(X_source) else X_source
        X_kept = X_csr[keep_cells]
        X_sel = X_kept[:, gene_idx].tocsr()
        row_sum_source = X_kept if normalization_scope == "all_genes" else X_sel
        row_sums = np.asarray(row_sum_source.sum(axis=1)).ravel().astype(np.float64, copy=False)
        raw_mat = X_sel.toarray()
    else:
        X_dense = np.asarray(X_source)
        X_kept = X_dense[keep_cells]
        raw_mat = np.asarray(X_kept[:, gene_idx], dtype=np.float64)
        row_sum_source = X_kept if normalization_scope == "all_genes" else raw_mat
        row_sums = np.asarray(row_sum_source.sum(axis=1, dtype=np.float64)).ravel()

    size_factors = np.zeros_like(row_sums, dtype=np.float64)
    np.divide(float(target_sum), row_sums, out=size_factors, where=row_sums > 0)

    lin_mat = raw_mat * size_factors[:, None]
    log_mat = np.log1p(lin_mat)
    expr_mat = lin_mat > 0
    sum_all_log = log_mat.sum(axis=0)
    sum_all_lin = lin_mat.sum(axis=0)
    sum_all_expr = expr_mat.sum(axis=0, dtype=np.float64)

    in_masks = [ct_arr == ct_name for ct_name in valid_celltypes]
    n_celltypes = len(valid_celltypes)
    n_in = np.asarray([mask.sum() for mask in in_masks], dtype=np.int64)
    n_out = int(log_mat.shape[0]) - n_in

    mean_in_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    mean_out_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    mean_diff_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    pct_in_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    pct_out_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    pct_diff_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    log2_fc_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)

    for j, in_mask in enumerate(in_masks):
        n1 = float(n_in[j])
        n2 = float(n_out[j])
        sum_in_log = log_mat[in_mask].sum(axis=0)
        sum_in_lin = lin_mat[in_mask].sum(axis=0)
        sum_in_expr = expr_mat[in_mask].sum(axis=0, dtype=np.float64)

        mean_in = sum_in_log / n1
        mean_out = (sum_all_log - sum_in_log) / n2
        pct_in = sum_in_expr / n1
        pct_out = (sum_all_expr - sum_in_expr) / n2
        lin_in = sum_in_lin / n1
        lin_out = (sum_all_lin - sum_in_lin) / n2

        mean_in_mat[:, j] = mean_in
        mean_out_mat[:, j] = mean_out
        mean_diff_mat[:, j] = mean_in - mean_out
        pct_in_mat[:, j] = pct_in
        pct_out_mat[:, j] = pct_out
        pct_diff_mat[:, j] = pct_in - pct_out
        log2_fc_mat[:, j] = np.log2((lin_in + 1e-8) / (lin_out + 1e-8))

    w_mean, w_pct, w_fc = [float(w) for w in score_weights]
    z_mean = _row_zscore_safe(mean_diff_mat)
    z_pct = _row_zscore_safe(pct_diff_mat)
    z_fc = _row_zscore_safe(log2_fc_mat)
    specificity_score_mat = (w_mean * z_mean) + (w_pct * z_pct) + (w_fc * z_fc)

    is_positive_effect_mat = (
        (mean_diff_mat >= float(min_mean_diff))
        & (pct_diff_mat >= float(min_pct_diff))
        & (log2_fc_mat >= float(min_log2_fc))
    )

    gene_col = np.repeat(np.asarray(genes_resolved, dtype=object), n_celltypes)
    celltype_col_flat = np.tile(np.asarray(valid_celltypes, dtype=object), n_genes)
    gene_celltype_scores = pd.DataFrame(
        {
            "gene": gene_col,
            "celltype": celltype_col_flat,
            "n_in": np.tile(n_in, n_genes),
            "n_out": np.tile(n_out, n_genes),
            "mean_in": mean_in_mat.reshape(-1),
            "mean_out": mean_out_mat.reshape(-1),
            "mean_diff": mean_diff_mat.reshape(-1),
            "pct_in": pct_in_mat.reshape(-1),
            "pct_out": pct_out_mat.reshape(-1),
            "pct_diff": pct_diff_mat.reshape(-1),
            "log2_fc": log2_fc_mat.reshape(-1),
            "z_mean_diff": z_mean.reshape(-1),
            "z_pct_diff": z_pct.reshape(-1),
            "z_log2_fc": z_fc.reshape(-1),
            "specificity_score": specificity_score_mat.reshape(-1),
            "is_positive_effect": is_positive_effect_mat.reshape(-1),
        }
    )

    pan_threshold = max(1, int(np.ceil(float(pan_fraction) * n_celltypes)))
    summary_rows = []
    for gi, gene in enumerate(genes_resolved):
        scores = specificity_score_mat[gi]
        pos_mask = is_positive_effect_mat[gi]
        order = np.argsort(scores)[::-1]
        best_idx = int(order[0])
        second_idx = int(order[1]) if order.size > 1 else int(order[0])
        best_score = float(scores[best_idx])
        second_score = float(scores[second_idx])
        margin = best_score - second_score
        positive_idx = np.flatnonzero(pos_mask)
        n_positive = int(positive_idx.size)

        if n_positive == 0:
            enrichment_class = "none"
            enriched_celltypes = ""
            top_idx = best_idx
        elif n_positive >= pan_threshold:
            enrichment_class = "pan"
            top_idx = int(positive_idx[np.argmax(scores[positive_idx])])
            enriched_celltypes = ",".join(
                np.asarray(valid_celltypes, dtype=object)[
                    positive_idx[np.argsort(scores[positive_idx])[::-1]]
                ].tolist()
            )
        elif n_positive == 1:
            enrichment_class = "single"
            top_idx = int(positive_idx[0])
            enriched_celltypes = str(np.asarray(valid_celltypes, dtype=object)[top_idx])
        else:
            enrichment_class = "multi"
            top_idx = int(positive_idx[np.argmax(scores[positive_idx])])
            enriched_celltypes = ",".join(
                np.asarray(valid_celltypes, dtype=object)[
                    positive_idx[np.argsort(scores[positive_idx])[::-1]]
                ].tolist()
            )

        summary_rows.append(
            {
                "gene": gene,
                "enrichment_class": enrichment_class,
                "n_enriched_celltypes": n_positive,
                "top_celltype": str(np.asarray(valid_celltypes, dtype=object)[top_idx]),
                "second_celltype": str(np.asarray(valid_celltypes, dtype=object)[second_idx]),
                "top_specificity_score": float(scores[top_idx]),
                "second_specificity_score": second_score,
                "specificity_margin": float(margin),
                "strong_single": bool((n_positive == 1) and (margin >= float(min_margin))),
                "top_log2_fc": float(log2_fc_mat[gi, top_idx]),
                "top_pct_diff": float(pct_diff_mat[gi, top_idx]),
                "top_mean_diff": float(mean_diff_mat[gi, top_idx]),
                "enriched_celltypes": enriched_celltypes,
            }
        )

    gene_specificity_summary = pd.DataFrame(summary_rows)
    class_order = pd.CategoricalDtype(
        categories=["single", "multi", "pan", "none"],
        ordered=True,
    )
    gene_specificity_summary["enrichment_class"] = gene_specificity_summary[
        "enrichment_class"
    ].astype(class_order)
    gene_specificity_summary = gene_specificity_summary.sort_values(
        ["enrichment_class", "strong_single", "top_specificity_score", "specificity_margin"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    positive_pairs = gene_celltype_scores[gene_celltype_scores["is_positive_effect"]].copy()
    if positive_pairs.empty:
        celltype_specificity_summary = pd.DataFrame(
            columns=[
                "n_positive_genes",
                "n_single_genes",
                "n_multi_genes",
                "n_pan_genes",
            ]
        )
    else:
        tmp = positive_pairs[["gene", "celltype"]].merge(
            gene_specificity_summary[["gene", "enrichment_class"]],
            on="gene",
            how="left",
        )
        n_total = tmp.groupby("celltype")["gene"].nunique().rename("n_positive_genes")
        n_single = (
            tmp[tmp["enrichment_class"] == "single"]
            .groupby("celltype")["gene"]
            .nunique()
            .rename("n_single_genes")
        )
        n_multi = (
            tmp[tmp["enrichment_class"] == "multi"]
            .groupby("celltype")["gene"]
            .nunique()
            .rename("n_multi_genes")
        )
        n_pan = (
            tmp[tmp["enrichment_class"] == "pan"]
            .groupby("celltype")["gene"]
            .nunique()
            .rename("n_pan_genes")
        )
        celltype_specificity_summary = (
            pd.concat([n_total, n_single, n_multi, n_pan], axis=1)
            .fillna(0)
            .astype(int)
            .sort_values(["n_positive_genes", "n_single_genes"], ascending=False)
        )

    return (
        gene_celltype_scores,
        gene_specificity_summary,
        celltype_specificity_summary,
        valid_celltypes,
    )


def permutation_classify_gene_celltype_enrichment(
    adata,
    genes: Optional[Sequence[str]] = None,
    celltype_col: str = "predicted_labels",
    layer: Optional[str] = "counts",
    target_sum: float = 1e4,
    normalization_scope: str = "all_genes",
    min_cells: int | str = "auto",
    alpha: float = 0.05,
    min_mean_diff: Optional[float] = None,
    min_pct_diff: Optional[float] = None,
    min_zscore: float = 2.0,
    min_log2_fc: float = 0.25,
    use_effect_thresholds: str = "recommended",
    pan_fraction: float = 1.0,
    n_permutations: int = 2000,
    n_jobs: int = 1,
    permutation_batch_size: int = 128,
    correction_scope: str = "global",
    random_state: Optional[int] = 0,
):
    """Gene-by-gene empirical celltype enrichment using one-vs-rest label permutations.

    Notes
    -----
    This is intended for small to moderate gene lists when per-gene asymptotic tests
    feel too optimistic. It preserves celltype sizes and computes empirical p-values
    for the observed mean log-expression difference. By default, count normalization
    uses each cell's total expression across all genes in ``layer``; set
    ``normalization_scope='selected_genes'`` to reproduce subset-only normalization.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if permutation_batch_size < 1:
        raise ValueError("permutation_batch_size must be >= 1.")
    if correction_scope not in {"global", "per_gene"}:
        raise ValueError("correction_scope must be one of {'global', 'per_gene'}.")
    if use_effect_thresholds not in {"recommended", "standard", "raw", "none"}:
        raise ValueError("use_effect_thresholds must be one of {'recommended', 'standard', 'raw', 'none'}.")
    if normalization_scope not in {"all_genes", "selected_genes"}:
        raise ValueError("normalization_scope must be one of {'all_genes', 'selected_genes'}.")
    if pan_fraction <= 0:
        raise ValueError("pan_fraction must be > 0.")
    if celltype_col not in adata.obs.columns:
        raise KeyError(f"'{celltype_col}' not found in adata.obs")

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        X_source = adata.layers[layer]
    else:
        X_source = adata.X

    var_names = np.asarray(adata.var_names).astype(str, copy=False)
    gene_idx, genes_resolved = _resolve_gene_indices(var_names, genes)
    n_genes = len(genes_resolved)

    keep_cells = adata.obs[celltype_col].notna().to_numpy()
    if not np.any(keep_cells):
        raise ValueError(f"No non-null labels found in adata.obs['{celltype_col}'].")
    n_cells_kept = int(keep_cells.sum())

    if isinstance(min_cells, str):
        if min_cells != "auto":
            raise ValueError("min_cells must be an integer or 'auto'.")
        min_cells_eff = _resolve_auto_min_cells(n_cells_kept)
    else:
        min_cells_eff = int(min_cells)
    if min_cells_eff < 1:
        raise ValueError("min_cells must be >= 1.")

    ct = adata.obs.loc[keep_cells, celltype_col].astype("category")
    ct_arr = ct.to_numpy()
    ct_levels = ct.cat.categories.tolist()

    valid_celltypes = []
    for ct_name in ct_levels:
        n_in_ct = int((ct_arr == ct_name).sum())
        n_out_ct = int((ct_arr != ct_name).sum())
        if n_in_ct >= min_cells_eff and n_out_ct >= min_cells_eff:
            valid_celltypes.append(ct_name)
    if not valid_celltypes:
        raise ValueError("No valid celltypes after min_cells filtering.")

    is_sparse = sparse.issparse(X_source)
    if is_sparse:
        X_csr = X_source.tocsr() if not sparse.isspmatrix_csr(X_source) else X_source
        X_kept = X_csr[keep_cells]
        X_sel = X_kept[:, gene_idx].tocsr()
        row_sum_source = X_kept if normalization_scope == "all_genes" else X_sel
        row_sums = np.asarray(row_sum_source.sum(axis=1)).ravel().astype(np.float64, copy=False)
        raw_mat = X_sel.toarray()
    else:
        X_dense = np.asarray(X_source)
        X_kept = X_dense[keep_cells]
        raw_mat = np.asarray(X_kept[:, gene_idx], dtype=np.float64)
        row_sum_source = X_kept if normalization_scope == "all_genes" else raw_mat
        row_sums = np.asarray(row_sum_source.sum(axis=1, dtype=np.float64)).ravel()

    size_factors = np.zeros_like(row_sums, dtype=np.float64)
    np.divide(float(target_sum), row_sums, out=size_factors, where=row_sums > 0)

    lin_mat = raw_mat * size_factors[:, None]
    log_mat = np.log1p(lin_mat)
    expr_mat = lin_mat > 0
    sum_all_log = log_mat.sum(axis=0)
    sum_all_lin = lin_mat.sum(axis=0)
    sum_all_expr = expr_mat.sum(axis=0, dtype=np.float64)

    in_masks = [ct_arr == ct_name for ct_name in valid_celltypes]
    n_celltypes = len(valid_celltypes)
    n_in = np.asarray([mask.sum() for mask in in_masks], dtype=np.int64)
    n_out = int(log_mat.shape[0]) - n_in

    mean_in_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    mean_out_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    mean_diff_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    pct_in_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    pct_out_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    pct_diff_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)
    log2_fc_mat = np.empty((n_genes, n_celltypes), dtype=np.float64)

    for j, in_mask in enumerate(in_masks):
        n1 = float(n_in[j])
        n2 = float(n_out[j])
        sum_in_log = log_mat[in_mask].sum(axis=0)
        sum_in_lin = lin_mat[in_mask].sum(axis=0)
        sum_in_expr = expr_mat[in_mask].sum(axis=0, dtype=np.float64)

        mean_in = sum_in_log / n1
        mean_out = (sum_all_log - sum_in_log) / n2
        pct_in = sum_in_expr / n1
        pct_out = (sum_all_expr - sum_in_expr) / n2
        lin_in = sum_in_lin / n1
        lin_out = (sum_all_lin - sum_in_lin) / n2

        mean_in_mat[:, j] = mean_in
        mean_out_mat[:, j] = mean_out
        mean_diff_mat[:, j] = mean_in - mean_out
        pct_in_mat[:, j] = pct_in
        pct_out_mat[:, j] = pct_out
        pct_diff_mat[:, j] = pct_in - pct_out
        log2_fc_mat[:, j] = np.log2((lin_in + 1e-8) / (lin_out + 1e-8))

    null_ge_mat = np.zeros((n_genes, n_celltypes), dtype=np.int64)
    null_sum_mat = np.zeros((n_genes, n_celltypes), dtype=np.float64)
    null_sq_sum_mat = np.zeros((n_genes, n_celltypes), dtype=np.float64)

    base_seed = 0 if random_state is None else int(random_state)
    if int(n_jobs) == 1 or n_celltypes == 1:
        results = []
        for j in range(n_celltypes):
            results.append(
                _perm_celltype_null_for_all_genes(
                    log_mat=log_mat,
                    score_diff_obs=mean_diff_mat[:, j],
                    n_in=int(n_in[j]),
                    n_out=int(n_out[j]),
                    n_permutations=int(n_permutations),
                    seed=base_seed + j,
                    batch_size=int(permutation_batch_size),
                )
            )
    else:
        max_workers = min(int(n_jobs), n_celltypes)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _perm_celltype_null_for_all_genes,
                    log_mat,
                    mean_diff_mat[:, j],
                    int(n_in[j]),
                    int(n_out[j]),
                    int(n_permutations),
                    base_seed + j,
                    int(permutation_batch_size),
                )
                for j in range(n_celltypes)
            ]
            results = [f.result() for f in futures]

    for j, (ge_j, sum_j, sq_sum_j) in enumerate(results):
        null_ge_mat[:, j] = ge_j
        null_sum_mat[:, j] = sum_j
        null_sq_sum_mat[:, j] = sq_sum_j

    pvalue_mat = (null_ge_mat.astype(np.float64) + 1.0) / (float(n_permutations) + 1.0)
    if correction_scope == "global":
        qvalue_mat = fdr_bh(pvalue_mat.reshape(-1)).reshape(pvalue_mat.shape)
    else:
        qvalue_mat = _fdr_bh_axis1(pvalue_mat)
    null_mean_mat = null_sum_mat / float(n_permutations)
    null_var_mat = np.maximum(
        (null_sq_sum_mat / float(n_permutations)) - (null_mean_mat * null_mean_mat),
        0.0,
    )
    null_sd_mat = np.sqrt(null_var_mat)
    zscore_mat = np.divide(
        mean_diff_mat - null_mean_mat,
        null_sd_mat,
        out=np.zeros_like(mean_diff_mat),
        where=null_sd_mat > 0,
    )

    effect_mask = (qvalue_mat <= float(alpha))
    if use_effect_thresholds == "recommended":
        rec_min_mean_diff = 0.05 if min_mean_diff is None else float(min_mean_diff)
        rec_min_pct_diff = 0.03 if min_pct_diff is None else float(min_pct_diff)
        effect_mask &= (mean_diff_mat >= rec_min_mean_diff)
        effect_mask &= (pct_diff_mat >= rec_min_pct_diff)
        effect_mask &= (log2_fc_mat > 0)
        effect_mask &= (zscore_mat > 0)
    elif use_effect_thresholds == "standard":
        effect_mask &= (zscore_mat >= float(min_zscore))
        effect_mask &= (log2_fc_mat >= float(min_log2_fc))
    elif use_effect_thresholds == "raw":
        if min_mean_diff is not None:
            effect_mask &= (mean_diff_mat >= float(min_mean_diff))
        if min_pct_diff is not None:
            effect_mask &= (pct_diff_mat >= float(min_pct_diff))
        effect_mask &= (log2_fc_mat > 0)
        effect_mask &= (zscore_mat > 0)
    else:
        effect_mask &= (log2_fc_mat > 0)
        effect_mask &= (zscore_mat > 0)
    is_enriched_mat = effect_mask

    gene_col = np.repeat(np.asarray(genes_resolved, dtype=object), n_celltypes)
    celltype_col_flat = np.tile(np.asarray(valid_celltypes, dtype=object), n_genes)
    gene_celltype_stats = pd.DataFrame(
        {
            "gene": gene_col,
            "celltype": celltype_col_flat,
            "n_in": np.tile(n_in, n_genes),
            "n_out": np.tile(n_out, n_genes),
            "mean_in": mean_in_mat.reshape(-1),
            "mean_out": mean_out_mat.reshape(-1),
            "mean_diff": mean_diff_mat.reshape(-1),
            "pct_in": pct_in_mat.reshape(-1),
            "pct_out": pct_out_mat.reshape(-1),
            "pct_diff": pct_diff_mat.reshape(-1),
            "log2_fc": log2_fc_mat.reshape(-1),
            "null_mean": null_mean_mat.reshape(-1),
            "null_sd": null_sd_mat.reshape(-1),
            "zscore": zscore_mat.reshape(-1),
            "pvalue": pvalue_mat.reshape(-1),
            "qvalue": qvalue_mat.reshape(-1),
            "correction_scope": correction_scope,
            "effect_mode": use_effect_thresholds,
            "is_enriched": is_enriched_mat.reshape(-1),
        }
    )

    pan_threshold = max(1, int(np.ceil(float(pan_fraction) * n_celltypes)))
    summary_rows = []
    for gene, gdf in gene_celltype_stats.groupby("gene", sort=False):
        gdf_sorted = gdf.sort_values(
            ["qvalue", "pvalue", "zscore", "log2_fc"],
            ascending=[True, True, False, False],
        )
        enriched = gdf_sorted[gdf_sorted["is_enriched"]].copy()
        n_enriched = int(enriched["celltype"].nunique())
        if n_enriched == 0:
            enrichment_class = "none"
        elif n_enriched == 1:
            enrichment_class = "single"
        elif n_enriched >= pan_threshold:
            enrichment_class = "pan"
        else:
            enrichment_class = "multi"

        if n_enriched > 0:
            top_row = enriched.sort_values(
                ["qvalue", "zscore", "log2_fc", "mean_diff"],
                ascending=[True, False, False, False],
            ).iloc[0]
            enriched_celltypes = ",".join(
                enriched.sort_values(
                    ["qvalue", "zscore", "log2_fc"],
                    ascending=[True, False, False],
                )["celltype"].tolist()
            )
        else:
            top_row = gdf_sorted.iloc[0]
            enriched_celltypes = ""

        top_fail_reasons = []
        if not bool(top_row["is_enriched"]):
            if float(top_row["qvalue"]) > float(alpha):
                top_fail_reasons.append("qvalue")
            if use_effect_thresholds == "recommended":
                rec_min_mean_diff = 0.05 if min_mean_diff is None else float(min_mean_diff)
                rec_min_pct_diff = 0.03 if min_pct_diff is None else float(min_pct_diff)
                if float(top_row["mean_diff"]) < rec_min_mean_diff:
                    top_fail_reasons.append("mean_diff")
                if float(top_row["pct_diff"]) < rec_min_pct_diff:
                    top_fail_reasons.append("pct_diff")
                if float(top_row["log2_fc"]) <= 0:
                    top_fail_reasons.append("log2_fc")
                if float(top_row["zscore"]) <= 0:
                    top_fail_reasons.append("zscore")
            elif use_effect_thresholds == "standard":
                if float(top_row["zscore"]) < float(min_zscore):
                    top_fail_reasons.append("zscore")
                if float(top_row["log2_fc"]) < float(min_log2_fc):
                    top_fail_reasons.append("log2_fc")
            elif use_effect_thresholds == "raw":
                if min_mean_diff is not None and float(top_row["mean_diff"]) < float(min_mean_diff):
                    top_fail_reasons.append("mean_diff")
                if min_pct_diff is not None and float(top_row["pct_diff"]) < float(min_pct_diff):
                    top_fail_reasons.append("pct_diff")
                if float(top_row["log2_fc"]) <= 0:
                    top_fail_reasons.append("log2_fc")
                if float(top_row["zscore"]) <= 0:
                    top_fail_reasons.append("zscore")
            else:
                if float(top_row["log2_fc"]) <= 0:
                    top_fail_reasons.append("log2_fc")
                if float(top_row["zscore"]) <= 0:
                    top_fail_reasons.append("zscore")

        summary_rows.append(
            {
                "gene": gene,
                "enrichment_class": enrichment_class,
                "n_enriched_celltypes": n_enriched,
                "top_celltype": top_row["celltype"],
                "top_is_enriched": bool(top_row["is_enriched"]),
                "top_fail_reasons": ",".join(top_fail_reasons),
                "top_pvalue": float(top_row["pvalue"]),
                "top_qvalue": float(top_row["qvalue"]),
                "top_mean_diff": float(top_row["mean_diff"]),
                "top_pct_diff": float(top_row["pct_diff"]),
                "top_log2_fc": float(top_row["log2_fc"]),
                "top_zscore": float(top_row["zscore"]),
                "enriched_celltypes": enriched_celltypes,
            }
        )

    gene_enrichment_summary = pd.DataFrame(summary_rows)
    class_order = pd.CategoricalDtype(
        categories=["single", "multi", "pan", "none"],
        ordered=True,
    )
    gene_enrichment_summary["enrichment_class"] = gene_enrichment_summary[
        "enrichment_class"
    ].astype(class_order)
    gene_enrichment_summary = gene_enrichment_summary.sort_values(
        ["enrichment_class", "n_enriched_celltypes", "top_qvalue", "top_zscore"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)

    enriched_pairs = gene_celltype_stats[gene_celltype_stats["is_enriched"]].copy()
    if enriched_pairs.empty:
        celltype_enrichment_summary = pd.DataFrame(
            columns=[
                "n_enriched_lowres_genes",
                "n_single_genes",
                "n_multi_genes",
                "n_pan_genes",
            ]
        )
    else:
        tmp = enriched_pairs[["gene", "celltype"]].merge(
            gene_enrichment_summary[["gene", "enrichment_class"]],
            on="gene",
            how="left",
        )
        n_total = tmp.groupby("celltype")["gene"].nunique().rename(
            "n_enriched_lowres_genes"
        )
        n_single = (
            tmp[tmp["enrichment_class"] == "single"]
            .groupby("celltype")["gene"]
            .nunique()
            .rename("n_single_genes")
        )
        n_multi = (
            tmp[tmp["enrichment_class"] == "multi"]
            .groupby("celltype")["gene"]
            .nunique()
            .rename("n_multi_genes")
        )
        n_pan = (
            tmp[tmp["enrichment_class"] == "pan"]
            .groupby("celltype")["gene"]
            .nunique()
            .rename("n_pan_genes")
        )
        celltype_enrichment_summary = (
            pd.concat([n_total, n_single, n_multi, n_pan], axis=1)
            .fillna(0)
            .astype(int)
            .sort_values(
                ["n_enriched_lowres_genes", "n_single_genes"],
                ascending=False,
            )
        )

    return (
        gene_celltype_stats,
        gene_enrichment_summary,
        celltype_enrichment_summary,
        valid_celltypes,
    )


def summarize_celltype_gene_set_enrichment(gene_celltype_stats: pd.DataFrame) -> pd.DataFrame:
    if gene_celltype_stats.empty:
        return pd.DataFrame()

    rows = []
    for ct_name, cdf in gene_celltype_stats.groupby("celltype", sort=False):
        pvals = np.clip(cdf["pvalue"].to_numpy(dtype=float), 1e-300, 1.0)
        fisher_stat, p_combined = combine_pvalues(pvals, method="fisher")
        rows.append(
            {
                "celltype": ct_name,
                "n_tested_genes": int(cdf["gene"].nunique()),
                "n_enriched_genes": int(cdf["is_enriched"].sum()),
                "mean_log2_fc": float(cdf["log2_fc"].mean()),
                "median_log2_fc": float(cdf["log2_fc"].median()),
                "fisher_stat": float(fisher_stat),
                "p_combined": float(p_combined),
            }
        )

    out = pd.DataFrame(rows)
    out["q_combined"] = fdr_bh(out["p_combined"].to_numpy(dtype=float))
    out = out.sort_values(
        ["q_combined", "n_enriched_genes", "median_log2_fc"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return out


def classify_scale_group_gene_celltype_enrichment(
    adata,
    scale_svg_table: pd.DataFrame,
    *,
    group_col: str = "scale_group",
    gene_col: str = "gene",
    pass_col: Optional[str] = "passes_rule",
    require_pass: bool = True,
    top_n_per_group: Optional[int] = 100,
    celltype_col: str = "predicted_labels",
    layer: Optional[str] = "counts",
    target_sum: float = 1e4,
    normalization_scope: str = "all_genes",
    min_cells: int | str = "auto",
    alpha: float = 0.05,
    min_mean_diff: Optional[float] = None,
    min_pct_diff: Optional[float] = None,
    min_zscore: float = 2.0,
    min_log2_fc: float = 0.25,
    use_effect_thresholds: str = "recommended",
    pan_fraction: float = 1.0,
    n_permutations: int = 2000,
    n_jobs: int = 1,
    permutation_batch_size: int = 128,
    correction_scope: str = "global",
    random_state: Optional[int] = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run gene-by-celltype enrichment separately for each scale-SVG group.

    Returns
    -------
    gene_celltype_stats
        One row per scale group, gene, and tested celltype.
    gene_enrichment_summary
        One row per scale group and gene, including ``single/multi/pan/none``.
    group_class_summary
        Scale-group level counts/fractions of enrichment classes.
    group_celltype_summary
        Scale-group by celltype counts of enriched genes.
    """
    from .scale_biology import make_scale_group_gene_sets

    gene_sets = make_scale_group_gene_sets(
        scale_svg_table,
        group_col=group_col,
        gene_col=gene_col,
        pass_col=pass_col,
        require_pass=require_pass,
        top_n_per_group=top_n_per_group,
    )
    if not gene_sets:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    stats_frames = []
    gene_summary_frames = []
    celltype_summary_frames = []
    group_rows = []

    for group, genes in gene_sets.items():
        genes = [str(g) for g in genes]
        if not genes:
            continue
        stats, gene_summary, celltype_summary, tested_celltypes = permutation_classify_gene_celltype_enrichment(
            adata,
            genes=genes,
            celltype_col=celltype_col,
            layer=layer,
            target_sum=target_sum,
            normalization_scope=normalization_scope,
            min_cells=min_cells,
            alpha=alpha,
            min_mean_diff=min_mean_diff,
            min_pct_diff=min_pct_diff,
            min_zscore=min_zscore,
            min_log2_fc=min_log2_fc,
            use_effect_thresholds=use_effect_thresholds,
            pan_fraction=pan_fraction,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            permutation_batch_size=permutation_batch_size,
            correction_scope=correction_scope,
            random_state=random_state,
        )
        stats = stats.copy()
        gene_summary = gene_summary.copy()
        celltype_summary = celltype_summary.copy()
        stats.insert(0, "scale_group", str(group))
        gene_summary.insert(0, "scale_group", str(group))
        if not celltype_summary.empty:
            celltype_summary = celltype_summary.reset_index().rename(columns={"index": "celltype"})
        else:
            celltype_summary = pd.DataFrame(
                columns=[
                    "celltype",
                    "n_enriched_lowres_genes",
                    "n_single_genes",
                    "n_multi_genes",
                    "n_pan_genes",
                ]
            )
        celltype_summary.insert(0, "scale_group", str(group))
        celltype_summary["n_tested_group_genes"] = int(len(gene_summary))
        celltype_summary["n_tested_celltypes"] = int(len(tested_celltypes))

        class_counts = gene_summary["enrichment_class"].astype(str).value_counts()
        n_genes = int(len(gene_summary))
        group_rows.append(
            {
                "scale_group": str(group),
                "n_tested_genes": n_genes,
                "n_tested_celltypes": int(len(tested_celltypes)),
                "n_single_genes": int(class_counts.get("single", 0)),
                "n_multi_genes": int(class_counts.get("multi", 0)),
                "n_pan_genes": int(class_counts.get("pan", 0)),
                "n_none_genes": int(class_counts.get("none", 0)),
                "fraction_single": float(class_counts.get("single", 0) / max(n_genes, 1)),
                "fraction_multi": float(class_counts.get("multi", 0) / max(n_genes, 1)),
                "fraction_pan": float(class_counts.get("pan", 0) / max(n_genes, 1)),
                "fraction_none": float(class_counts.get("none", 0) / max(n_genes, 1)),
                "n_any_enriched_genes": int(n_genes - class_counts.get("none", 0)),
                "fraction_any_enriched": float((n_genes - class_counts.get("none", 0)) / max(n_genes, 1)),
            }
        )

        stats_frames.append(stats)
        gene_summary_frames.append(gene_summary)
        celltype_summary_frames.append(celltype_summary)

    gene_celltype_stats = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()
    gene_enrichment_summary = pd.concat(gene_summary_frames, ignore_index=True) if gene_summary_frames else pd.DataFrame()
    group_class_summary = pd.DataFrame(group_rows)
    group_celltype_summary = pd.concat(celltype_summary_frames, ignore_index=True) if celltype_summary_frames else pd.DataFrame()

    if not group_celltype_summary.empty and "n_enriched_lowres_genes" in group_celltype_summary.columns:
        group_celltype_summary["fraction_enriched_group_genes"] = (
            pd.to_numeric(group_celltype_summary["n_enriched_lowres_genes"], errors="coerce")
            / pd.to_numeric(group_celltype_summary["n_tested_group_genes"], errors="coerce").replace(0, np.nan)
        )
        group_celltype_summary = group_celltype_summary.sort_values(
            ["scale_group", "n_enriched_lowres_genes", "n_single_genes"],
            ascending=[True, False, False],
            kind="mergesort",
        ).reset_index(drop=True)

    return (
        gene_celltype_stats,
        gene_enrichment_summary,
        group_class_summary,
        group_celltype_summary,
    )


def plot_scale_group_celltype_enrichment_heatmap(
    group_celltype_summary: pd.DataFrame,
    *,
    value_col: str = "n_enriched_lowres_genes",
    max_celltypes: Optional[int] = 30,
    group_order: Optional[Sequence[str]] = None,
    celltype_order: Optional[Sequence[str]] = None,
    title: str = "Celltype-enriched genes by scale group",
    cmap: str = "viridis",
    annotate: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Plot scale-group by celltype enriched-gene counts/fractions."""
    if group_celltype_summary is None or group_celltype_summary.empty:
        raise ValueError("group_celltype_summary must be non-empty.")
    required = {"scale_group", "celltype", value_col}
    missing = required.difference(group_celltype_summary.columns)
    if missing:
        raise KeyError(f"group_celltype_summary is missing required columns: {sorted(missing)}")

    import matplotlib.pyplot as plt

    work = group_celltype_summary.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    if group_order is None:
        group_order = work["scale_group"].astype(str).drop_duplicates().tolist()
    else:
        group_order = [str(x) for x in group_order]
    if celltype_order is None:
        top = (
            work.groupby("celltype", observed=True)[value_col]
            .sum()
            .sort_values(ascending=False)
        )
        if max_celltypes is not None:
            top = top.head(int(max_celltypes))
        celltype_order = top.index.astype(str).tolist()
    else:
        celltype_order = [str(x) for x in celltype_order]

    matrix = (
        work.assign(scale_group=work["scale_group"].astype(str), celltype=work["celltype"].astype(str))
        .pivot_table(index="celltype", columns="scale_group", values=value_col, aggfunc="max")
        .reindex(index=celltype_order, columns=group_order)
        .fillna(0.0)
    )
    if figsize is None:
        figsize = (max(4.5, 1.35 * len(group_order) + 2.0), max(4.0, 0.32 * len(celltype_order) + 1.7))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.to_numpy(float), aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=0)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=8)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    if annotate:
        vals = matrix.to_numpy(float)
        vmax = float(np.nanmax(vals)) if vals.size else 1.0
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                val = vals[i, j]
                if val <= 0:
                    continue
                color = "white" if val > 0.55 * max(vmax, 1.0) else "black"
                label = f"{val:.2f}" if "fraction" in value_col else f"{int(round(val))}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(value_col)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_scale_group_celltype_enrichment_dotplot(
    group_celltype_summary: pd.DataFrame,
    *,
    group_col: str = "scale_group",
    celltype_col: str = "celltype",
    size_col: str = "n_enriched_lowres_genes",
    color_col: str = "fraction_enriched_group_genes",
    group_order: Optional[Sequence[str]] = None,
    celltype_order: Optional[Sequence[str]] = None,
    max_celltypes: Optional[int] = 30,
    min_size: float = 1.0,
    order_by: str = "dominant_scale",
    dot_min: float = 35.0,
    dot_max: float = 420.0,
    cmap: str = "viridis",
    edgecolor: str = "#333333",
    linewidth: float = 0.45,
    annotate_counts: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Celltype-enriched genes by scale group",
    size_legend_title: Optional[str] = None,
    colorbar_title: Optional[str] = None,
    pretty_group_labels: bool = True,
    group_label_map: Optional[Mapping[str, str]] = None,
    xtick_rotation: float = 0.0,
    grid_color: str = "#e8e8e8",
    panel_background: str = "white",
    show_size_legend: bool = True,
    show_colorbar: bool = True,
    right_margin: float = 0.70,
):
    """Dotplot of celltype-enriched genes across scale groups.

    The default visual mapping is intentionally simple: dot size is the number
    of enriched group genes and dot color is the fraction of tested group genes.
    Celltypes are ordered by their dominant scale group by default, which makes
    scale-specific differences easier to read than an alphabetical order.
    """
    if group_celltype_summary is None or group_celltype_summary.empty:
        raise ValueError("group_celltype_summary must be non-empty.")
    required = {group_col, celltype_col, size_col, color_col}
    missing = required.difference(group_celltype_summary.columns)
    if missing:
        raise KeyError(f"group_celltype_summary is missing required columns: {sorted(missing)}")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    work = group_celltype_summary.copy()
    work[group_col] = work[group_col].astype(str)
    work[celltype_col] = work[celltype_col].astype(str)
    work[size_col] = pd.to_numeric(work[size_col], errors="coerce").fillna(0.0)
    work[color_col] = pd.to_numeric(work[color_col], errors="coerce").fillna(0.0)
    work = work.loc[work[size_col] >= float(min_size)].copy()
    if work.empty:
        raise ValueError("No rows remain after applying min_size.")
    default_group_label_map = {
        "fine_hclust": "Fine",
        "mid_hclust": "Mid",
        "coarse_hclust": "Coarse",
    }
    if group_label_map is not None:
        default_group_label_map.update({str(k): str(v) for k, v in group_label_map.items()})

    if group_order is None:
        canonical = ["fine_hclust", "mid_hclust", "coarse_hclust"]
        present = work[group_col].drop_duplicates().astype(str).tolist()
        if set(present).issubset(set(canonical)):
            group_order = [g for g in canonical if g in present]
        else:
            group_order = present
    else:
        group_order = [str(x) for x in group_order]
    group_rank = {g: i for i, g in enumerate(group_order)}
    work = work.loc[work[group_col].isin(group_order)].copy()
    if work.empty:
        raise ValueError("No rows match group_order.")

    if celltype_order is None:
        pivot = (
            work.pivot_table(index=celltype_col, columns=group_col, values=color_col, aggfunc="max", fill_value=0.0)
            .reindex(columns=group_order, fill_value=0.0)
        )
        size_pivot = (
            work.pivot_table(index=celltype_col, columns=group_col, values=size_col, aggfunc="max", fill_value=0.0)
            .reindex(columns=group_order, fill_value=0.0)
        )
        mode = str(order_by).lower()
        if mode in {"dominant_scale", "dominant", "scale"}:
            dominant = pivot.idxmax(axis=1)
            order_frame = pd.DataFrame(
                {
                    "_dominant_rank": dominant.map(group_rank).fillna(len(group_order)).astype(int),
                    "_max_color": pivot.max(axis=1),
                    "_total_size": size_pivot.sum(axis=1),
                },
                index=pivot.index,
            )
            order_frame = order_frame.sort_values(
                ["_dominant_rank", "_max_color", "_total_size"],
                ascending=[True, False, False],
                kind="mergesort",
            )
            celltype_order = order_frame.index.astype(str).tolist()
        elif mode == "max":
            celltype_order = pivot.max(axis=1).sort_values(ascending=False).index.astype(str).tolist()
        elif mode == "total":
            celltype_order = size_pivot.sum(axis=1).sort_values(ascending=False).index.astype(str).tolist()
        elif mode in {"alphabetical", "alpha", "name"}:
            celltype_order = sorted(pivot.index.astype(str).tolist())
        else:
            raise ValueError("order_by must be 'dominant_scale', 'max', 'total', or 'alphabetical'.")
        if max_celltypes is not None:
            celltype_order = celltype_order[: int(max_celltypes)]
    else:
        celltype_order = [str(x) for x in celltype_order]
    work = work.loc[work[celltype_col].isin(celltype_order)].copy()

    x_map = {g: i for i, g in enumerate(group_order)}
    y_map = {ct: i for i, ct in enumerate(celltype_order)}
    work["_x"] = work[group_col].map(x_map)
    work["_y"] = work[celltype_col].map(y_map)

    max_size = float(work[size_col].max())
    if max_size <= 0:
        max_size = 1.0
    sizes = dot_min + (work[size_col].to_numpy(float) / max_size) * (dot_max - dot_min)

    if figsize is None:
        figsize = (
            max(7.2, 1.55 * len(group_order) + 3.8),
            max(4.6, 0.42 * len(celltype_order) + 1.8),
        )
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(panel_background)
    sc = ax.scatter(
        work["_x"].to_numpy(float),
        work["_y"].to_numpy(float),
        s=sizes,
        c=work[color_col].to_numpy(float),
        cmap=cmap,
        edgecolors=edgecolor,
        linewidths=float(linewidth),
    )
    xticklabels = [
        default_group_label_map.get(g, g) if pretty_group_labels else g
        for g in group_order
    ]
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(
        xticklabels,
        rotation=float(xtick_rotation),
        ha="right" if float(xtick_rotation) else "center",
        fontweight="bold",
    )
    ax.set_yticks(np.arange(len(celltype_order)))
    ax.set_yticklabels(celltype_order)
    ax.set_xlim(-0.5, len(group_order) - 0.5)
    ax.set_ylim(len(celltype_order) - 0.5, -0.5)
    ax.set_xlabel("Scale group")
    ax.set_ylabel("Celltype")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(-0.5, len(group_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(celltype_order), 1), minor=True)
    ax.grid(which="major", axis="y", color=grid_color, linewidth=0.8)
    ax.grid(which="minor", axis="x", color=grid_color, linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    if annotate_counts:
        for _, row in work.iterrows():
            ax.text(
                row["_x"],
                row["_y"],
                str(int(round(float(row[size_col])))),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    fig.subplots_adjust(right=float(right_margin))
    if show_colorbar:
        cax = fig.add_axes([float(right_margin) + 0.19, 0.18, 0.025, 0.43])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_label(colorbar_title or "Fraction of group genes", fontweight="bold")

    legend_values = np.unique(
        np.clip(
            np.array([1, np.ceil(max_size / 3), np.ceil(2 * max_size / 3), np.ceil(max_size)], dtype=float),
            1,
            max_size,
        ).astype(int)
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=np.sqrt(dot_min + (v / max_size) * (dot_max - dot_min)),
            markerfacecolor="lightgray",
            markeredgecolor=edgecolor,
            markeredgewidth=float(linewidth),
            label=str(int(v)),
        )
        for v in legend_values
    ]
    if handles and show_size_legend:
        ax.legend(
            handles=handles,
            title=size_legend_title or "Enriched genes",
            frameon=False,
            bbox_to_anchor=(1.04, 0.98),
            loc="upper left",
            borderaxespad=0.0,
            labelspacing=1.0,
            handletextpad=1.2,
        )
    return fig, ax


def plot_scale_group_celltype_enrichment_contrast_dotplot(
    group_celltype_summary: pd.DataFrame,
    *,
    group_col: str = "scale_group",
    celltype_col: str = "celltype",
    size_col: str = "n_enriched_lowres_genes",
    value_col: str = "fraction_enriched_group_genes",
    group_order: Optional[Sequence[str]] = None,
    group_label_map: Optional[Mapping[str, str]] = None,
    max_celltypes: Optional[int] = 18,
    min_size: float = 1.0,
    min_contrast: float = 0.0,
    contrast_mode: str = "row_relative",
    dot_min: float = 28.0,
    dot_max: float = 430.0,
    cmap: str = "YlOrRd",
    edgecolor: str = "#333333",
    linewidth: float = 0.45,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Scale-biased celltype enrichment",
    annotate_counts: bool = False,
    show_dominant_scale: bool = False,
    size_legend_title: str = "Enriched genes",
    colorbar_title: Optional[str] = None,
    show_size_legend: bool = True,
    show_colorbar: bool = True,
    right_margin: float = 0.78,
    legend_bbox_to_anchor: Tuple[float, float] = (1.04, 0.98),
    colorbar_bbox: Tuple[float, float, float, float] = (0.86, 0.18, 0.025, 0.45),
    font_scale: float = 1.0,
):
    """Contrast dotplot emphasizing which scale group dominates each celltype.

    Dot size keeps the absolute support (number of enriched genes), while color
    shows the within-celltype scale bias.  This is useful when the claim is not
    just "which celltypes are enriched", but that enrichment differs across
    fine/mid/coarse SVG groups.
    """
    if group_celltype_summary is None or group_celltype_summary.empty:
        raise ValueError("group_celltype_summary must be non-empty.")
    required = {group_col, celltype_col, size_col, value_col}
    missing = required.difference(group_celltype_summary.columns)
    if missing:
        raise KeyError(f"group_celltype_summary is missing required columns: {sorted(missing)}")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    default_group_label_map = {
        "fine_hclust": "Fine",
        "mid_hclust": "Mid",
        "coarse_hclust": "Coarse",
    }
    default_group_colors = {
        "fine_hclust": "#4C78A8",
        "mid_hclust": "#F28E2B",
        "coarse_hclust": "#59A14F",
    }
    if group_label_map is not None:
        default_group_label_map.update({str(k): str(v) for k, v in group_label_map.items()})
    fs = float(font_scale)
    if fs <= 0:
        raise ValueError("font_scale must be positive.")
    title_size = 13.0 * fs
    label_size = 11.0 * fs
    tick_size = 10.0 * fs
    legend_size = 9.0 * fs
    dominant_size = 8.0 * fs
    annotation_size = 7.0 * fs

    work = group_celltype_summary.copy()
    work[group_col] = work[group_col].astype(str)
    work[celltype_col] = work[celltype_col].astype(str)
    work[size_col] = pd.to_numeric(work[size_col], errors="coerce").fillna(0.0)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work = work.loc[work[size_col] >= float(min_size)].copy()
    if work.empty:
        raise ValueError("No rows remain after applying min_size.")

    if group_order is None:
        canonical = ["fine_hclust", "mid_hclust", "coarse_hclust"]
        present = work[group_col].drop_duplicates().astype(str).tolist()
        group_order = [g for g in canonical if g in present] if set(present).issubset(set(canonical)) else present
    else:
        group_order = [str(x) for x in group_order]
    group_rank = {g: i for i, g in enumerate(group_order)}
    work = work.loc[work[group_col].isin(group_order)].copy()

    value_mat = (
        work.pivot_table(index=celltype_col, columns=group_col, values=value_col, aggfunc="max", fill_value=0.0)
        .reindex(columns=group_order, fill_value=0.0)
    )
    size_mat = (
        work.pivot_table(index=celltype_col, columns=group_col, values=size_col, aggfunc="max", fill_value=0.0)
        .reindex(columns=group_order, fill_value=0.0)
    )
    if value_mat.empty:
        raise ValueError("No values remain after filtering.")

    vmax_row = value_mat.max(axis=1).replace(0, np.nan)
    vmin_row = value_mat.min(axis=1)
    contrast = (value_mat.max(axis=1) - vmin_row).fillna(0.0)
    dominant = value_mat.idxmax(axis=1)
    order_frame = pd.DataFrame(
        {
            "_dominant_rank": dominant.map(group_rank).fillna(len(group_order)).astype(int),
            "_contrast": contrast,
            "_max_value": value_mat.max(axis=1),
            "_total_size": size_mat.sum(axis=1),
        },
        index=value_mat.index,
    )
    order_frame = order_frame.loc[order_frame["_contrast"] >= float(min_contrast)]
    order_frame = order_frame.sort_values(
        ["_dominant_rank", "_contrast", "_max_value", "_total_size"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    if max_celltypes is not None:
        order_frame = order_frame.head(int(max_celltypes))
    celltype_order = order_frame.index.astype(str).tolist()
    if not celltype_order:
        raise ValueError("No celltypes remain after applying min_contrast/max_celltypes.")

    value_mat = value_mat.reindex(index=celltype_order, columns=group_order).fillna(0.0)
    size_mat = size_mat.reindex(index=celltype_order, columns=group_order).fillna(0.0)
    dominant = value_mat.idxmax(axis=1)
    mode = str(contrast_mode).lower()
    if mode in {"row_relative", "relative", "rowmax"}:
        color_mat = value_mat.div(value_mat.max(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        color_label = colorbar_title or "Within-celltype relative enrichment"
        vmin, vmax = 0.0, 1.0
    elif mode in {"delta", "delta_from_min"}:
        color_mat = value_mat.sub(value_mat.min(axis=1), axis=0)
        color_label = colorbar_title or "Fraction difference from weakest scale"
        vmin, vmax = 0.0, None
    elif mode in {"absolute", "fraction"}:
        color_mat = value_mat.copy()
        color_label = colorbar_title or "Fraction of group genes"
        vmin, vmax = 0.0, None
    else:
        raise ValueError("contrast_mode must be 'row_relative', 'delta', or 'absolute'.")

    plot_rows = []
    for y, ct in enumerate(celltype_order):
        for x, group in enumerate(group_order):
            plot_rows.append(
                {
                    "_x": x,
                    "_y": y,
                    group_col: group,
                    celltype_col: ct,
                    "_size": float(size_mat.loc[ct, group]),
                    "_color": float(color_mat.loc[ct, group]),
                    "_value": float(value_mat.loc[ct, group]),
                }
            )
    plot_df = pd.DataFrame(plot_rows)
    max_size = float(plot_df["_size"].max())
    if max_size <= 0:
        max_size = 1.0
    dot_sizes = dot_min + (plot_df["_size"].to_numpy(float) / max_size) * (dot_max - dot_min)

    if figsize is None:
        figsize = (
            max(7.8, 1.55 * len(group_order) + 4.0),
            max(4.8, 0.44 * len(celltype_order) + 1.9),
        )
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        plot_df["_x"].to_numpy(float),
        plot_df["_y"].to_numpy(float),
        s=dot_sizes,
        c=plot_df["_color"].to_numpy(float),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors=edgecolor,
        linewidths=float(linewidth),
    )
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(
        [default_group_label_map.get(g, g) for g in group_order],
        fontweight="bold",
        fontsize=tick_size,
    )
    ax.set_yticks(np.arange(len(celltype_order)))
    ax.set_yticklabels(celltype_order, fontsize=tick_size, fontweight="bold")
    ax.set_xlim(-0.5, len(group_order) - 0.5)
    ax.set_ylim(len(celltype_order) - 0.5, -0.5)
    ax.set_xlabel("Scale group", fontsize=label_size, fontweight="bold")
    ax.set_ylabel("Celltype", fontsize=label_size, fontweight="bold")
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(-0.5, len(group_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(celltype_order), 1), minor=True)
    ax.grid(which="major", axis="y", color="#e6e6e6", linewidth=0.8)
    ax.grid(which="minor", axis="x", color="#e6e6e6", linewidth=0.8)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    if show_dominant_scale:
        strip_x = len(group_order) - 0.35
        for y, ct in enumerate(celltype_order):
            group = str(dominant.loc[ct])
            ax.add_patch(
                Rectangle(
                    (strip_x, y - 0.34),
                    0.10,
                    0.68,
                    color=default_group_colors.get(group, "#999999"),
                    clip_on=False,
                    linewidth=0,
                )
            )
        ax.text(
            strip_x + 0.05,
            -0.85,
            "dominant",
            ha="center",
            va="bottom",
            fontsize=dominant_size,
            rotation=90,
            fontweight="bold",
        )

    if annotate_counts:
        for _, row in plot_df.iterrows():
            if row["_size"] <= 0:
                continue
            ax.text(
                row["_x"],
                row["_y"],
                str(int(round(row["_size"]))),
                ha="center",
                va="center",
                fontsize=annotation_size,
                color="black" if row["_color"] < 0.65 else "white",
                fontweight="bold",
            )

    fig.subplots_adjust(right=float(right_margin))
    if show_colorbar:
        cax = fig.add_axes(list(colorbar_bbox))
        cbar = fig.colorbar(sc, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_label(color_label, fontsize=label_size, fontweight="bold")
        cbar.ax.tick_params(labelsize=tick_size)

    legend_values = np.unique(
        np.clip(
            np.array([1, np.ceil(max_size / 3), np.ceil(2 * max_size / 3), np.ceil(max_size)], dtype=float),
            1,
            max_size,
        ).astype(int)
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=np.sqrt(dot_min + (v / max_size) * (dot_max - dot_min)),
            markerfacecolor="lightgray",
            markeredgecolor=edgecolor,
            markeredgewidth=float(linewidth),
            label=str(int(v)),
        )
        for v in legend_values
    ]
    if show_size_legend:
        ax.legend(
            handles=handles,
            title=size_legend_title,
            frameon=False,
            bbox_to_anchor=legend_bbox_to_anchor,
            loc="upper left",
            borderaxespad=0.0,
            labelspacing=1.0,
            handletextpad=1.2,
            fontsize=legend_size,
            title_fontsize=label_size,
        )
    return fig, ax


def plot_scale_group_enrichment_class_stacked_bar(
    group_class_summary: pd.DataFrame,
    *,
    fraction: bool = True,
    title: str = "Gene celltype-enrichment class by scale group",
    figsize: Tuple[float, float] = (6.5, 3.6),
    colors: Optional[Mapping[str, str]] = None,
):
    """Plot single/multi/pan/none composition for each scale group."""
    if group_class_summary is None or group_class_summary.empty:
        raise ValueError("group_class_summary must be non-empty.")
    import matplotlib.pyplot as plt

    classes = ["single", "multi", "pan", "none"]
    default_colors = {
        "single": "#4C78A8",
        "multi": "#F28E2B",
        "pan": "#59A14F",
        "none": "#BAB0AC",
    }
    if colors is not None:
        default_colors.update({str(k): str(v) for k, v in colors.items()})
    work = group_class_summary.copy()
    groups = work["scale_group"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups), dtype=float)
    for cls in classes:
        col = f"fraction_{cls}" if fraction else f"n_{cls}_genes"
        vals = pd.to_numeric(work.get(col, 0), errors="coerce").fillna(0).to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=cls, color=default_colors.get(cls, "#999999"))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0)
    ax.set_ylabel("Fraction of genes" if fraction else "Number of genes")
    ax.set_ylim(0, 1.0 if fraction else max(bottom.max() * 1.08, 1.0))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def classify_scale_group_gene_set_celltype_enrichment(
    adata,
    scale_svg_table: pd.DataFrame,
    *,
    group_col: str = "scale_group",
    gene_col: str = "gene",
    pass_col: Optional[str] = "passes_rule",
    require_pass: bool = True,
    top_n_per_group: Optional[int] = 100,
    celltype_col: str = "predicted_labels",
    layer: Optional[str] = "counts",
    target_sum: float = 1e4,
    min_cells: int = 15,
    alpha: float = 0.05,
    min_score_diff: float = 0.05,
    n_permutations: int = 2000,
    background_genes: Optional[Sequence[str]] = None,
    match_expression: bool = True,
    n_expression_bins: int = 20,
    n_jobs: int = 1,
    permutation_batch_size: int = 128,
    random_state: Optional[int] = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Test each scale group as a gene-set/module against celltypes.

    This complements :func:`classify_scale_group_gene_celltype_enrichment`.
    The per-gene classifier asks whether individual genes are celltype markers;
    this function asks whether the whole group gene set has a higher module
    score in a celltype than expression-matched random gene sets.
    """
    from .scale_biology import make_scale_group_gene_sets

    gene_sets = make_scale_group_gene_sets(
        scale_svg_table,
        group_col=group_col,
        gene_col=gene_col,
        pass_col=pass_col,
        require_pass=require_pass,
        top_n_per_group=top_n_per_group,
    )
    if not gene_sets:
        return pd.DataFrame(), pd.DataFrame()

    stats_frames = []
    summary_frames = []
    for group, genes in gene_sets.items():
        genes = [str(g) for g in genes]
        if not genes:
            continue
        stats, tested_celltypes = permutation_celltype_gene_set_enrichment(
            adata,
            genes=genes,
            celltype_col=celltype_col,
            layer=layer,
            target_sum=target_sum,
            min_cells=int(min_cells),
            alpha=alpha,
            min_score_diff=min_score_diff,
            n_permutations=n_permutations,
            background_genes=background_genes,
            match_expression=match_expression,
            n_expression_bins=n_expression_bins,
            n_jobs=n_jobs,
            permutation_batch_size=permutation_batch_size,
            random_state=random_state,
        )
        stats = stats.copy()
        stats.insert(0, "scale_group", str(group))
        stats["n_tested_celltypes"] = int(len(tested_celltypes))
        summary = summarize_permutation_celltype_gene_set_enrichment(stats, pan_fraction=1.0)
        summary = summary.copy()
        summary.insert(0, "scale_group", str(group))
        summary["gene_set_size"] = int(len(genes))
        summary["n_tested_celltypes"] = int(len(tested_celltypes))
        stats_frames.append(stats)
        summary_frames.append(summary)

    stats_out = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()
    summary_out = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return stats_out, summary_out


def plot_scale_group_gene_set_celltype_heatmap(
    gene_set_celltype_stats: pd.DataFrame,
    *,
    value_col: str = "score_diff",
    only_enriched: bool = False,
    max_celltypes: Optional[int] = 30,
    group_order: Optional[Sequence[str]] = None,
    celltype_order: Optional[Sequence[str]] = None,
    title: str = "Scale-group gene-set celltype enrichment",
    cmap: str = "viridis",
    annotate: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Plot module-level scale-group celltype enrichment."""
    if gene_set_celltype_stats is None or gene_set_celltype_stats.empty:
        raise ValueError("gene_set_celltype_stats must be non-empty.")
    required = {"scale_group", "celltype", value_col}
    missing = required.difference(gene_set_celltype_stats.columns)
    if missing:
        raise KeyError(f"gene_set_celltype_stats is missing required columns: {sorted(missing)}")

    import matplotlib.pyplot as plt

    work = gene_set_celltype_stats.copy()
    if only_enriched and "is_enriched" in work.columns:
        work = work.loc[work["is_enriched"].astype(bool)].copy()
    if work.empty:
        raise ValueError("No rows remain after filtering.")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    if group_order is None:
        group_order = work["scale_group"].astype(str).drop_duplicates().tolist()
    else:
        group_order = [str(x) for x in group_order]
    if celltype_order is None:
        top = work.groupby("celltype", observed=True)[value_col].max().sort_values(ascending=False)
        if max_celltypes is not None:
            top = top.head(int(max_celltypes))
        celltype_order = top.index.astype(str).tolist()
    else:
        celltype_order = [str(x) for x in celltype_order]

    matrix = (
        work.assign(scale_group=work["scale_group"].astype(str), celltype=work["celltype"].astype(str))
        .pivot_table(index="celltype", columns="scale_group", values=value_col, aggfunc="max")
        .reindex(index=celltype_order, columns=group_order)
        .fillna(0.0)
    )
    if figsize is None:
        figsize = (max(4.5, 1.35 * len(group_order) + 2.0), max(4.0, 0.32 * len(celltype_order) + 1.7))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.to_numpy(float), aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist())
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=8)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    if annotate:
        vals = matrix.to_numpy(float)
        vmax = float(np.nanmax(vals)) if vals.size else 1.0
        qmat = None
        if "qvalue" in work.columns:
            qmat = (
                work.assign(scale_group=work["scale_group"].astype(str), celltype=work["celltype"].astype(str))
                .pivot_table(index="celltype", columns="scale_group", values="qvalue", aggfunc="min")
                .reindex(index=celltype_order, columns=group_order)
                .to_numpy(float)
            )
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                val = vals[i, j]
                if val <= 0:
                    continue
                label = f"{val:.2f}"
                if qmat is not None and np.isfinite(qmat[i, j]):
                    if qmat[i, j] <= 0.001:
                        label = "***"
                    elif qmat[i, j] <= 0.01:
                        label = "**"
                    elif qmat[i, j] <= 0.05:
                        label = "*"
                    else:
                        label = ""
                if not label:
                    continue
                color = "white" if val > 0.55 * max(vmax, 1.0) else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(value_col)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig
