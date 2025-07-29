"""Utility to convert gene expression into a 1D embedding."""

import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp


def add_gene_expression_embedding(
    adata: AnnData,
    genes: list[str],
    layer: str | None = None,
    segment_key: str | None = None,
    embedding_key: str = "X_gene_embedding",
    normalize_total: bool = False,
    target_sum: float = 1e4,
    log1p: bool = False,
) -> None:
    """Add a gene expression based embedding to ``adata.obsm``.

    Parameters
    ----------
    adata : AnnData
        Object with expression data.
    genes : list of str
        Genes to sum for the embedding. Order of genes does not matter.
    layer : str or None, optional
        Use this layer of ``adata`` instead of ``adata.X``.
    segment_key : str or None, optional
        If provided, sum expression within each segment and assign the
        same aggregated value to all cells in that segment.
    embedding_key : str, optional
        Key under which to store the resulting embedding in ``adata.obsm``.
    normalize_total : bool, optional (default=False)
        If ``True``, divide expression by total counts and multiply by
        ``target_sum`` before storing, similar to ``scanpy.pp.normalize_total``.
    target_sum : float, optional (default=1e4)
        Target sum used when ``normalize_total`` is ``True``.
    log1p : bool, optional (default=False)
        Apply a ``log1p`` transform to the normalized values, matching
        ``scanpy.pp.log1p`` behaviour.
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]

    gene_idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
    if not gene_idx:
        raise ValueError("None of the specified genes were found in adata.var_names")

    if sp.issparse(X):
        expr = np.asarray(X[:, gene_idx].sum(axis=1)).ravel()
        total = np.asarray(X.sum(axis=1)).ravel()
    else:
        expr = X[:, gene_idx].sum(axis=1)
        total = X.sum(axis=1)

    if segment_key is not None and segment_key in adata.obs:
        seg = pd.Categorical(adata.obs[segment_key])
        seg_codes = seg.codes
        nseg = seg_codes.max() + 1
        sums = np.bincount(seg_codes, weights=expr, minlength=nseg)
        totals = np.bincount(seg_codes, weights=total, minlength=nseg)
        expr = sums[seg_codes]
        total = totals[seg_codes]

    if normalize_total:
        denom = np.maximum(total, 1e-12)
        expr = expr / denom * target_sum

    if log1p:
        expr = np.log1p(expr)

    adata.obsm[embedding_key] = expr.reshape(-1, 1)
