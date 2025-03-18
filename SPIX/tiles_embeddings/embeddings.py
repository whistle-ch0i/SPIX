import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
import logging
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.vectorized import contains
from collections import Counter
import itertools
import os
import scipy
from scipy.sparse import csr_matrix

from .tiles import generate_tiles
from ..utils.utils import _process_in_parallel_map, calculate_pca_loadings, run_lsi, smooth_polygon

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_embeddings(
    adata: AnnData,
    dim_reduction: str = 'PCA',
    normalization: str = 'log_norm',
    use_counts: str = 'raw',
    dimensions: int = 30,
    tensor_resolution: float = 1,
    filter_grid: float = 0.01,
    filter_threshold: float = 0.995,
    nfeatures: int = 2000,
    features: list = None,
    min_cutoff: str = 'q5',
    remove_lsi_1: bool = True,
    n_jobs: int = None,
    chunksize: int = 5000,
    verbose: bool = True
) -> AnnData:
    """
    Generate embeddings for the given AnnData object using specified parameters.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    dim_reduction : str, optional (default='PCA')
        Dimensionality reduction method to use ('PCA', 'UMAP', 'NMF', 'LSI', etc.).
    normalization : str, optional (default='log_norm')
        Normalization method to apply to the count data ('log_norm', 'SCT', 'TFIDF', 'none').
    use_counts : str, optional (default='raw')
        Which counts to use for embedding ('raw' for adata.raw.X, or layer name).
    dimensions : int, optional (default=30)
        Number of dimensions to reduce to.
    tensor_resolution : float, optional (default=1)
        Resolution parameter for tensor reduction.
    filter_grid : float, optional (default=0.01)
        Grid filtering threshold to remove outlier beads.
    filter_threshold : float, optional (default=0.995)
        Threshold to filter tiles based on area.
    nfeatures : int, optional (default=2000)
        Number of highly variable genes to select.
    features : list, optional (default=None)
        Specific features to use for embedding. If None, uses all selected features.
    min_cutoff : str, optional (default='q5')
        Minimum cutoff for feature selection (e.g., 'q5' for 5th percentile).
    remove_lsi_1 : bool, optional (default=True)
        Whether to remove the first component in LSI.
    n_jobs : int, optional (default=None)
        Number of parallel jobs to run.
    chunksize : int, optional (default=5000)
        Number of samples per chunk for parallel processing.
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    
    Returns
    -------
    AnnData
        The updated AnnData object with generated embeddings.
    """
    if verbose:
        logging.info("Starting generate_embeddings...")

    # Generate tiles if not already present
    if 'tiles_generated' not in adata.uns or not adata.uns['tiles_generated']:
        if verbose:
            logging.info("Tiles not found. Generating tiles...")
        adata = generate_tiles(
            adata,
            tensor_resolution=tensor_resolution,
            filter_grid=filter_grid,
            filter_threshold=filter_threshold,
            verbose=verbose,
            chunksize=chunksize,
            n_jobs=n_jobs
        )

    # Process counts with specified normalization
    if verbose:
        logging.info("Processing counts...")
    adata_proc = process_counts(
        adata,
        method=normalization,
        use_counts=use_counts,
        nfeatures=nfeatures,
        min_cutoff=min_cutoff,
        verbose=verbose
    )

    # Store processed counts in a new layer
    adata.layers['log_norm'] = adata_proc.X.copy()

    # Embed latent space using specified dimensionality reduction
    if verbose:
        logging.info("Embedding latent space...")
    embeds = embed_latent_space(
        adata_proc,
        dim_reduction=dim_reduction,
        dimensions=dimensions,
        features=features,
        remove_lsi_1=remove_lsi_1,
        verbose=verbose,
        n_jobs=n_jobs
    )

    # Store embeddings in AnnData object
    adata.obsm['X_embedding'] = embeds
    adata.uns['embedding_method'] = dim_reduction
    adata.uns['tensor_resolution'] = tensor_resolution
    if verbose:
        logging.info("generate_embeddings completed.")

    return adata


def process_counts(
    adata: AnnData,
    method: str = 'log_norm',
    use_counts: str = 'raw',
    nfeatures: int = 2000,
    min_cutoff: str = 'q5',
    verbose: bool = True
) -> AnnData:
    """
    Process count data with specified normalization and feature selection.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    method : str, optional (default='log_norm')
        Normalization method to apply ('log_norm', 'SCT', 'TFIDF', 'none').
    use_counts : str, optional (default='raw')
        Which counts to use ('raw' for adata.raw.X, or layer name).
    nfeatures : int, optional (default=2000)
        Number of highly variable genes to select.
    min_cutoff : str, optional (default='q5')
        Minimum cutoff for feature selection (e.g., 'q5' for 5th percentile).
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    
    Returns
    -------
    AnnData
        The processed AnnData object.
    """
    if verbose:
        logging.info(f"Processing counts with method: {method}")

    # Select counts based on 'use_counts' parameter
    if use_counts == 'raw':
        counts = adata.raw.X if adata.raw is not None else adata.X.copy()
        adata_proc = adata.copy()
    else:
        if use_counts not in adata.layers:
            raise ValueError(f"Layer '{use_counts}' not found in adata.layers.")
        counts = adata.layers[use_counts]
        adata_proc = AnnData(X=counts)
        adata_proc.var_names = adata.var_names.copy()
        adata_proc.obs_names = adata.obs_names.copy()

    # Apply normalization and feature selection
    if method == 'log_norm':
        sc.pp.normalize_total(adata_proc, target_sum=1e4)
        sc.pp.log1p(adata_proc)
        sc.pp.highly_variable_genes(adata_proc, n_top_genes=nfeatures)
    elif method == 'SCT':
        raise NotImplementedError("SCTransform normalization is not implemented in this code.")
    elif method == 'TFIDF':
        tf = counts / counts.sum(axis=1)
        idf = np.log(1 + counts.shape[0] / (1 + (counts > 0).sum(axis=0)))
        counts_tfidf = tf.multiply(idf)
        adata_proc.X = counts_tfidf

        if min_cutoff.startswith('q'):
            quantile = float(min_cutoff[1:]) / 100
            if scipy.sparse.issparse(counts_tfidf):
                counts_tfidf_dense = counts_tfidf.toarray()
            else:
                counts_tfidf_dense = counts_tfidf
            variances = np.var(counts_tfidf_dense, axis=0)
            cutoff = np.quantile(variances, quantile)
            selected_features = variances >= cutoff
            adata_proc = adata_proc[:, selected_features]
        else:
            raise ValueError("Invalid min_cutoff format.")
    elif method == 'none':
        pass
    else:
        raise ValueError(f"Normalization method '{method}' is not recognized.")

    if verbose:
        logging.info("Counts processing completed.")

    return adata_proc


def embed_latent_space(
    adata_proc: AnnData,
    dim_reduction: str = 'PCA',
    dimensions: int = 30,
    features: list = None,
    remove_lsi_1: bool = True,
    verbose: bool = True,
    n_jobs: int = None
) -> np.ndarray:
    """
    Embed the processed data into a latent space using specified dimensionality reduction.
    
    Parameters
    ----------
    adata_proc : AnnData
        The processed AnnData object.
    dim_reduction : str, optional (default='PCA')
        Dimensionality reduction method ('PCA', 'UMAP', 'NMF', 'LSI','Harmony', etc.).
    dimensions : int, optional (default=30)
        Number of dimensions to reduce to.
    features : list, optional (default=None)
        Specific features to use for embedding. If None, uses all selected features.
    remove_lsi_1 : bool, optional (default=True)
        Whether to remove the first component in LSI.
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    n_jobs : int, optional (default=None)
        Number of parallel jobs to run.
    
    Returns
    -------
    np.ndarray
        The embedded data.
    """
    if verbose:
        logging.info(f"Embedding latent space using {dim_reduction}...")

    if features is not None:
        adata_proc = adata_proc[:, features]

    embeds = None

    if dim_reduction == 'PCA':
        sc.tl.pca(adata_proc, n_comps=dimensions)
        embeds = adata_proc.obsm['X_pca']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'PCA_L':
        sc.tl.pca(adata_proc, n_comps=dimensions)
        loadings = adata_proc.varm['PCs']
        counts = adata_proc.X

        iterable = zip(range(counts.shape[0]), itertools.repeat(counts), itertools.repeat(loadings))
        desc = "Calculating PCA loadings"

        embeds_list = _process_in_parallel_map(
            calculate_pca_loadings,
            iterable,
            desc,
            n_jobs,
            chunksize=1000,
            total=counts.shape[0]
        )
        embeds = np.array(embeds_list)
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'UMAP':
        sc.pp.neighbors(adata_proc, n_pcs=dimensions)
        sc.tl.umap(adata_proc, n_components=3)
        embeds = adata_proc.obsm['X_umap']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'LSI':
        embeds = run_lsi(adata_proc, n_components=dimensions, remove_first=remove_lsi_1)
    elif dim_reduction == 'LSI_UMAP':
        lsi_embeds = run_lsi(adata_proc, n_components=dimensions, remove_first=remove_lsi_1)
        adata_proc.obsm['X_lsi'] = lsi_embeds
        sc.pp.neighbors(adata_proc, use_rep='X_lsi')
        sc.tl.umap(adata_proc, n_components=3)
        embeds = adata_proc.obsm['X_umap']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'NMF':
        nmf_model = NMF(n_components=dimensions, init='random', random_state=0)
        W = nmf_model.fit_transform(adata_proc.X)
        embeds = MinMaxScaler().fit_transform(W)
    elif dim_reduction == 'Harmony':
        # Perform PCA before Harmony integration
        sc.tl.pca(adata_proc, n_comps=dimensions)
        # Run Harmony integration using 'library_id' as the batch key
        sc.external.pp.harmony_integrate(adata_proc, key='library_id')
        # Retrieve the Harmony integrated embeddings; these are stored in 'X_pca_harmony'
        embeds = adata_proc.obsm['X_pca_harmony']
        embeds = MinMaxScaler().fit_transform(embeds)
    else:
        raise ValueError(f"Dimensionality reduction method '{dim_reduction}' is not recognized.")

    if verbose:
        logging.info("Latent space embedding completed.")

    return embeds
