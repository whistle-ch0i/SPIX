from __future__ import annotations

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
