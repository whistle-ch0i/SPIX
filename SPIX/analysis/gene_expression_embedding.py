import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp
from typing import Literal

def add_gene_expression_embedding(
    adata: AnnData,
    genes: list[str],
    layer: str | None = None,
    segment_key: str | None = None,
    embedding_key: str = "X_gene_embedding",
    normalize_total: bool = False,
    target_sum: float = 1e4,
    log1p: bool = False,
    aggregation: Literal["sum", "mean"] = "sum",
) -> None:
    """Add a gene-expression-based embedding to `adata.obsm`.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression matrix.
    genes : list[str]
        List of gene names to aggregate. Order does not matter.
    layer : str or None, optional
        If provided, use `adata.layers[layer]` instead of `adata.X`.
    segment_key : str or None, optional
        If provided and exists in `adata.obs`, aggregate within each segment
        and assign the same value to all cells in that segment.
    embedding_key : str, optional
        Key under which to store the resulting embedding in `adata.obsm`.
    normalize_total : bool, optional (default=False)
        If True, normalize each cell (or segment) to `target_sum` total counts.
    target_sum : float, optional (default=1e4)
        Target sum for normalization when `normalize_total=True`.
    log1p : bool, optional (default=False)
        If True, apply `log1p` transform after normalization.
    aggregation : {"sum", "mean"}, optional (default="sum")
        Aggregation method across the selected genes:
        - "sum": sum of counts per cell (or segment)
        - "mean": mean per cell within each segment (divide segment‐sum by cell count)
    """
    # Select data matrix
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]

    # Find indices of requested genes
    gene_idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
    if not gene_idx:
        raise ValueError("None of the specified genes were found in adata.var_names")

    # Compute raw expression for selected genes and total counts per cell
    if sp.issparse(X):
        expr = np.asarray(X[:, gene_idx].sum(axis=1)).ravel()
        total = np.asarray(X.sum(axis=1)).ravel()
    else:
        expr = X[:, gene_idx].sum(axis=1)
        total = X.sum(axis=1)

    # Validate aggregation argument
    if aggregation not in ("sum", "mean"):
        raise ValueError("aggregation must be either 'sum' or 'mean'")

    # If segments are provided, aggregate per segment
    if segment_key is not None and segment_key in adata.obs:
        seg = pd.Categorical(adata.obs[segment_key])
        codes = seg.codes
        nseg = codes.max() + 1

        # Sum of expr and total counts within each segment
        seg_expr_sum = np.bincount(codes, weights=expr, minlength=nseg)
        seg_total_sum = np.bincount(codes, weights=total, minlength=nseg)
        # Number of cells in each segment
        seg_cell_counts = np.bincount(codes, minlength=nseg)

        if aggregation == "mean":
            # Divide by number of cells per segment
            seg_expr = seg_expr_sum / np.maximum(seg_cell_counts, 1)
            seg_total = seg_total_sum / np.maximum(seg_cell_counts, 1)
        else:  # sum
            seg_expr = seg_expr_sum
            seg_total = seg_total_sum

        # Assign back to each cell
        expr = seg_expr[codes]
        total = seg_total[codes]

    # Optional library-size normalization
    if normalize_total:
        denom = np.maximum(total, 1e-12)
        expr = expr / denom * target_sum

    # Optional log1p transform
    if log1p:
        expr = np.log1p(expr)

    # Store as 1D embedding in obsm
    adata.obsm[embedding_key] = expr.reshape(-1, 1)
