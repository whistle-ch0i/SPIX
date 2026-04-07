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
    - Normalization and log1p are computed on the selected genes only (same intent
      as normalizing `adata[:, genes]` before testing).
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
            X_sel = X_csr
        elif all_cells_selected:
            X_sel = X_csr[:, gene_idx]
        elif all_genes_selected:
            X_sel = X_csr[keep_cells]
        else:
            X_sel = X_csr[keep_cells][:, gene_idx]
        X_sel = X_sel.tocsr()
    else:
        X_dense = np.asarray(X_source)
        if all_cells_selected and all_genes_selected:
            X_sel = X_dense
        elif all_cells_selected:
            X_sel = X_dense[:, gene_idx]
        elif all_genes_selected:
            X_sel = X_dense[keep_cells]
        else:
            X_sel = X_dense[keep_cells][:, gene_idx]

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

    if is_sparse:
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
    unavailable or less useful than stable effect-size ranking.
    """
    if celltype_col not in adata.obs.columns:
        raise KeyError(f"'{celltype_col}' not found in adata.obs")
    if len(score_weights) != 3:
        raise ValueError("score_weights must have length 3: (mean_diff, pct_diff, log2_fc).")

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
        X_sel = X_csr[keep_cells][:, gene_idx].tocsr()
        row_sums = np.asarray(X_sel.sum(axis=1)).ravel().astype(np.float64, copy=False)
        raw_mat = X_sel.toarray()
    else:
        X_dense = np.asarray(X_source)
        raw_mat = np.asarray(X_dense[keep_cells][:, gene_idx], dtype=np.float64)
        row_sums = np.asarray(raw_mat.sum(axis=1, dtype=np.float64)).ravel()

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
    for the observed mean log-expression difference.
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
        X_sel = X_csr[keep_cells][:, gene_idx].tocsr()
        row_sums = np.asarray(X_sel.sum(axis=1)).ravel().astype(np.float64, copy=False)
        raw_mat = X_sel.toarray()
    else:
        X_dense = np.asarray(X_source)
        raw_mat = np.asarray(X_dense[keep_cells][:, gene_idx], dtype=np.float64)
        row_sums = np.asarray(raw_mat.sum(axis=1, dtype=np.float64)).ravel()

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

        summary_rows.append(
            {
                "gene": gene,
                "enrichment_class": enrichment_class,
                "n_enriched_celltypes": n_enriched,
                "top_celltype": top_row["celltype"],
                "top_qvalue": float(top_row["qvalue"]),
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
