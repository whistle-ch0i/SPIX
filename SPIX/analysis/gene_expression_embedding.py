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
    else:
        expr = X[:, gene_idx].sum(axis=1)

    if segment_key is not None and segment_key in adata.obs:
        seg = pd.Categorical(adata.obs[segment_key])
        seg_codes = seg.codes
        nseg = seg_codes.max() + 1
        sums = np.bincount(seg_codes, weights=expr, minlength=nseg)
        expr = sums[seg_codes]

    adata.obsm[embedding_key] = expr.reshape(-1, 1)

