import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA, NMF
try:
    from sklearn.decomposition import MiniBatchNMF  # sklearn >= 1.1
except Exception:  # pragma: no cover - optional
    MiniBatchNMF = None
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
import logging
from contextlib import contextmanager
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.vectorized import contains
from collections import Counter
import itertools
import os
import scipy
from scipy.sparse import csr_matrix
from typing import Optional

try:
    from threadpoolctl import threadpool_limits, threadpool_info
except Exception:  # pragma: no cover - optional
    # Fallback no-op context manager if threadpoolctl is unavailable
    @contextmanager
    def threadpool_limits(limits=None, user_api=None):
        yield


# Helper context manager to limit both BLAS and OpenMP threads safely
@contextmanager
def _limit_blas_openmp(threads: Optional[int]):
    if not threads:
        # No limiting requested
        yield
        return
    try:
        # Apply limits to BLAS and then OpenMP in nested contexts
        with threadpool_limits(limits=threads, user_api='blas'):
            with threadpool_limits(limits=threads, user_api='openmp'):
                yield
    except Exception:
        # If threadpoolctl is missing or does not support user_api, do nothing
        yield

from .tiles import generate_tiles
from ..utils.utils import _process_in_parallel_map, calculate_pca_loadings, run_lsi, smooth_polygon


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_embeddings(
    adata: AnnData,
    dim_reduction: str = 'PCA',
    normalization: str = 'log_norm',
    use_counts: str = 'raw',
    library_id : str = 'library_id',
    dimensions: int = 30,
    tensor_resolution: float = 1,
    filter_grid: float = 1,
    filter_threshold: float = 0.995,
    use_hvg_only: bool = True,
    nfeatures: int = 2000,
    features: list = None,
    min_cutoff: str = 'q5',
    remove_lsi_1: bool = True,
    n_jobs: int = None,
    chunksize: int = 5000,
    force: bool = False,
    verbose: bool = True,
    use_coords_as_tiles: bool = False,
    coords_max_gap_factor: Optional[float] = 10.0,
    coords_compaction: str = 'cap',
    coords_tight_step: float = 1.0,
    coords_rescale_to_nn: Optional[bool] = None,
    coords_filter: bool = False,
    coords_filter_k: int = 5,
    coords_filter_mode: str = 'knn',
    # Rasterisation sampling controls (Voronoi mode)
    raster_stride: int = 1,
    raster_sample_frac: Optional[float] = None,
    raster_max_pixels_per_tile: Optional[int] = None,
    raster_random_seed: Optional[int] = None,
    # Restoration toggles for tiles
    restore_zero_pixel_tiles: bool = True,
    restore_voronoi_dropped: bool = True,
    # NMF acceleration options (all optional; defaults keep previous behavior)
    nmf_solver: str = 'auto',               # 'auto'|'mu'|'cd'
    nmf_init: str = 'nndsvda',
    nmf_tol: float = 1e-3,
    nmf_max_iter: int = 200,
    nmf_alpha_W: float = 0.0,
    nmf_alpha_H: float = 0.0,
    nmf_l1_ratio: float = 0.0,
    nmf_use_minibatch: bool = False,
    nmf_batch_size: Optional[int] = None,
    nmf_threads: Optional[int] = None,         # control BLAS/OpenMP threads
) -> AnnData:
    """
    Generate embeddings for the given AnnData object using specified parameters.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    dim_reduction : str, optional (default='PCA')
        Dimensionality reduction method to use.
        Supported: 'PCA', 'UMAP', 'NMF', 'LSI', 'Harmony',
        and 'RAW' (no reduction; uses processed counts as embedding) or 'RAW_HVG'
        (no reduction; restrict to HVGs).
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
        For HVG workflows, number of highly variable genes to select.
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
    use_coords_as_tiles : bool, optional (default=False)
        If True, skip Voronoi and rasterisation during tile generation and
        directly write coordinates to ``adata.uns['tiles']`` (one row per barcode,
        origin=1). If ``tensor_resolution`` != 1, the coordinates are reduced
        using ``reduce_tensor_resolution`` before writing. Coordinates are then
        compacted to avoid excessively large gaps (see ``coords_max_gap_factor``).
    coords_max_gap_factor : float or None, optional (default=10.0)
        When using coordinates-as-tiles, limit 1D gaps along X/Y to at most
        this factor times the median gap on that axis. Set to ``None`` or
        ``<= 0`` to disable compaction.
    coords_compaction : {'cap','tight'}, optional (default='cap')
        Strategy for compaction of coordinates-as-tiles. 'tight' packs columns/rows
        contiguously using rank-based mapping; 'cap' limits large axis gaps.
    coords_tight_step : float, optional (default=1.0)
        Spacing between consecutive ranks when using 'tight' compaction.
    coords_rescale_to_nn : bool or None, optional (default=None)
        Rescale post-compaction to match original median 1-NN distance. If None,
        defaults to True for 'cap' and False for 'tight'.
    coords_filter : bool, optional (default=False)
        When using coordinates-as-tiles, optionally filter isolated points using
        kNN distance; points with mean kNN distance above the ``filter_threshold``
        quantile are dropped.
    coords_filter_k : int, optional (default=5)
        Number of neighbors for the coords-as-tiles kNN filtering.
    coords_filter_mode : {'knn','voronoi'}, optional (default='knn')
        Filtering strategy in coords-as-tiles mode. Use 'voronoi' to reproduce
        original Voronoi area-based filtering and match previous behavior.
    raster_stride : int, optional (default=1)
        In Voronoi+rasterise mode, sample the interior grid every ``stride``
        pixels along X and Y to reduce tile pixel count while preserving
        spatial structure. ``1`` keeps all pixels.
    raster_sample_frac : float or None, optional (default=None)
        Additional random subsampling fraction per tile in (0,1]. Applied after
        ``raster_stride``. ``None`` disables.
    raster_max_pixels_per_tile : int or None, optional (default=None)
        Cap the maximum number of pixels kept per tile. Applied after stride and
        fractional subsampling. ``None`` disables.
    raster_random_seed : int or None, optional (default=None)
        Random seed for reproducible subsampling of rasterised pixels.
    restore_zero_pixel_tiles : bool, optional (default=True)
        If True, spots that pass Voronoi filtering but yield zero pixels after
        rasterisation are restored as origin-only tiles.
    restore_voronoi_dropped : bool, optional (default=True)
        If True, spots dropped by Voronoi area filtering are restored as
        origin-only tiles.
    
    Returns
    -------
    AnnData
        The updated AnnData object with generated embeddings.
    use_hvg_only : bool, optional (default=False)
        When True and dim_reduction in {'RAW', 'PCA', 'NMF', 'Harmony'} with log_norm,
        restrict to highly variable genes. For 'RAW_HVG', this is implicitly True.
    nmf_* : optional
        Advanced controls for speeding up NMF on large/sparse matrices:
        - `nmf_solver`: choose 'mu' for sparse CSR, 'cd' for dense, or 'auto'.
        - `nmf_init`: initialization method (default 'nndsvda' for faster convergence).
        - `nmf_tol`, `nmf_max_iter`: convergence controls.
        - `nmf_alpha_W`, `nmf_alpha_H`, `nmf_l1_ratio`: regularization (defaults 0 for speed).
        - `nmf_use_minibatch`: try MiniBatchNMF if available in sklearn; else falls back.
        - `nmf_batch_size`: mini-batch size when using MiniBatchNMF.
        - `nmf_threads`: limit BLAS/OpenMP threads during NMF to avoid oversubscription
          and utilize cores efficiently.

    """
    if verbose:
        logging.info("Starting generate_embeddings...")

    # Generate tiles if not already present
    tiles_exist = adata.uns.get('tiles_generated', False)
    if force or not tiles_exist:
        if verbose:
            logging.info("Tiles not found. Generating tiles...")
        adata = generate_tiles(
            adata,
            tensor_resolution=tensor_resolution,
            filter_grid=filter_grid,
            filter_threshold=filter_threshold,
            verbose=verbose,
            chunksize=chunksize,
            force=force,
            n_jobs=n_jobs,
            use_coords_as_tiles=use_coords_as_tiles,
            coords_max_gap_factor=coords_max_gap_factor,
            coords_compaction=coords_compaction,
            coords_tight_step=coords_tight_step,
            coords_rescale_to_nn=coords_rescale_to_nn,
            coords_filter=coords_filter,
            coords_filter_k=coords_filter_k,
            coords_filter_mode=coords_filter_mode,
            raster_stride=raster_stride,
            raster_sample_frac=raster_sample_frac,
            raster_max_pixels_per_tile=raster_max_pixels_per_tile,
            raster_random_seed=raster_random_seed,
            restore_zero_pixel_tiles=restore_zero_pixel_tiles,
            restore_voronoi_dropped=restore_voronoi_dropped,
        )

    # Process counts with specified normalization
    if verbose:
        logging.info("Processing counts...")
    adata_proc = process_counts(
        adata,
        method=normalization,
        dim_reduction=dim_reduction,
        use_counts=use_counts,
        nfeatures=nfeatures,
        min_cutoff=min_cutoff,
        verbose=verbose
    )

    # Store processed counts in a new layer when shapes match the parent AnnData.
    # AnnData layers must have the same shape (n_obs, n_vars). In TFIDF mode we
    # may subset features for embedding; in that case, skip storing to avoid
    # alignment issues. Also, use a method-appropriate layer name.
    try:
        X_proc = adata_proc.X
        if scipy.sparse.issparse(X_proc):
            # Ensure a subscriptable sparse format (COO is not indexable)
            X_proc = X_proc.tocsr()
        if adata_proc.shape == adata.shape:
            layer_key = None
            if normalization == 'log_norm':
                layer_key = 'log_norm'
            elif normalization == 'TFIDF':
                layer_key = 'tfidf'
            if layer_key is not None:
                adata.layers[layer_key] = X_proc
    except Exception as e:
        if verbose:
            logging.warning(f"Skipping layer storage due to: {e}")

    # Embed latent space using specified dimensionality reduction
    if verbose:
        logging.info("Embedding latent space...")
    embeds = embed_latent_space(
        adata_proc,
        dim_reduction=dim_reduction,
        library_id=library_id,
        dimensions=dimensions,
        features=features,
        remove_lsi_1=remove_lsi_1,
        verbose=verbose,
        n_jobs=n_jobs,
        use_hvg_only=use_hvg_only,
        hvg_n_top=nfeatures,
        nmf_solver=nmf_solver,
        nmf_init=nmf_init,
        nmf_tol=nmf_tol,
        nmf_max_iter=nmf_max_iter,
        nmf_alpha_W=nmf_alpha_W,
        nmf_alpha_H=nmf_alpha_H,
        nmf_l1_ratio=nmf_l1_ratio,
        nmf_use_minibatch=nmf_use_minibatch,
        nmf_batch_size=nmf_batch_size,
        nmf_threads=nmf_threads,
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
    dim_reduction: str = 'PCA',
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
        if dim_reduction == 'Harmony':
            sc.pp.highly_variable_genes(adata_proc, batch_key='library_id', inplace=True)
        else:
            sc.pp.highly_variable_genes(adata_proc, n_top_genes=nfeatures)
    elif method == 'SCT':
        raise NotImplementedError("SCTransform normalization is not implemented in this code.")
    elif method == 'TFIDF':
        # Compute TF-IDF with sparse-friendly operations and ensure CSR/CSC storage
        if scipy.sparse.issparse(counts):
            counts = counts.tocsr()
            row_sums = np.asarray(counts.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            inv_row = 1.0 / row_sums
            tf = counts.multiply(inv_row[:, None])
            df = (counts > 0).astype(np.int32).sum(axis=0).A1
            idf = np.log(1 + counts.shape[0] / (1 + df))
            counts_tfidf = tf.multiply(idf)
            # Keep a subscriptable format for downstream slicing/SVD
            adata_proc.X = counts_tfidf.tocsr()
        else:
            tf = counts / counts.sum(axis=1, keepdims=True)
            df = (counts > 0).sum(axis=0)
            idf = np.log(1 + counts.shape[0] / (1 + df))
            counts_tfidf = tf * idf
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
    library_id : str = 'library_id',
    dimensions: int = 30,
    features: list = None,
    remove_lsi_1: bool = True,
    verbose: bool = True,
    n_jobs: int = None,
    use_hvg_only: bool = False,
    hvg_n_top: Optional[int] = None,
    # NMF options
    nmf_solver: str = 'auto',
    nmf_init: str = 'nndsvda',
    nmf_tol: float = 1e-3,
    nmf_max_iter: int = 200,
    nmf_alpha_W: float = 0.0,
    nmf_alpha_H: float = 0.0,
    nmf_l1_ratio: float = 0.0,
    nmf_use_minibatch: bool = False,
    nmf_batch_size: Optional[int] = None,
    nmf_threads: Optional[int] = None,
) -> np.ndarray:
    """
    Embed the processed data into a latent space using specified dimensionality reduction.
    
    Parameters
    ----------
    adata_proc : AnnData
        The processed AnnData object.
    dim_reduction : str, optional (default='PCA')
        Dimensionality reduction method ('PCA', 'UMAP', 'NMF', 'LSI','Harmony',
        'RAW', 'RAW_HVG').
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
    use_hvg_only : bool, optional (default=False)
        For 'RAW'/'PCA'/'NMF'/'Harmony' with log_norm counts, restrict to HVGs.
        Ignored when 'features' is provided. For 'RAW_HVG', this is always True.
    hvg_n_top : int or None
        Max number of HVGs to keep when HVG restriction is applied (default uses
        value from generate_embeddings.nfeatures when provided).
    nmf_* : optional
        See generate_embeddings docstring for details.
    
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
        # Optionally restrict to HVGs
        if use_hvg_only and 'highly_variable' in adata_proc.var:
            hv_mask = adata_proc.var['highly_variable'].values
            if hv_mask.any():
                adata_proc = adata_proc[:, hv_mask]
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
        # Optionally restrict to HVGs to reduce problem size
        if use_hvg_only and 'highly_variable' in adata_proc.var:
            hv_mask = adata_proc.var['highly_variable'].values
            if hv_mask.any():
                adata_proc = adata_proc[:, hv_mask]

        X = adata_proc.X
        is_sparse = scipy.sparse.issparse(X)
        if is_sparse:
            X = X.tocsr().astype(np.float32)
        else:
            X = X.astype(np.float32)

        # Choose solver automatically if requested
        solver = nmf_solver
        if solver == 'auto':
            solver = 'mu' if is_sparse else 'cd'

        # Prefer MiniBatchNMF when requested and available (fast on very large data)
        nmf_model = None
        if nmf_use_minibatch and MiniBatchNMF is not None:
            if verbose:
                logging.info("Using MiniBatchNMF for speed on large data.")
            nmf_model = MiniBatchNMF(
                n_components=dimensions,
                init=nmf_init,
                tol=nmf_tol,
                max_iter=nmf_max_iter,
                random_state=0,
                batch_size=nmf_batch_size,
                alpha_W=nmf_alpha_W,
                alpha_H=nmf_alpha_H,
                l1_ratio=nmf_l1_ratio,
            )
        else:
            nmf_model = NMF(
                n_components=dimensions,
                init=nmf_init,
                solver=solver,
                tol=nmf_tol,
                max_iter=nmf_max_iter,
                random_state=0,
                alpha_W=nmf_alpha_W,
                alpha_H=nmf_alpha_H,
                l1_ratio=nmf_l1_ratio,
            )

        # Fit on full matrix (no subset learning)
        if verbose:
            logging.info("NMF: fitting on full matrix with fit_transform().")
        threads = nmf_threads
        if verbose:
            try:
                logging.info(f"NMF: pre-fit_transform threadpools: {threadpool_info()}")
            except Exception:
                pass
        with _limit_blas_openmp(threads):
            W = nmf_model.fit_transform(X)

        embeds = MinMaxScaler().fit_transform(W)
    elif dim_reduction == 'Harmony':
        # Optionally restrict to HVGs
        if use_hvg_only and 'highly_variable' in adata_proc.var:
            hv_mask = adata_proc.var['highly_variable'].values
            if hv_mask.any():
                adata_proc = adata_proc[:, hv_mask]
        # Perform PCA before Harmony integration
        sc.tl.pca(adata_proc, n_comps=dimensions)
        # Run Harmony integration using 'library_id' as the batch key
        sc.external.pp.harmony_integrate(adata_proc, key=library_id)
        # Retrieve the Harmony integrated embeddings; these are stored in 'X_pca_harmony'
        embeds = adata_proc.obsm['X_pca_harmony']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction in ('RAW', 'RAW_HVG'):
        # No dimensionality reduction: use processed counts as embedding
        # Optionally restrict to HVGs
        hv_only = (dim_reduction == 'RAW_HVG') or use_hvg_only
        if hv_only:
            if 'highly_variable' not in adata_proc.var:
                try:
                    # Use module-level Scanpy import to avoid shadowing 'sc'
                    sc.pp.highly_variable_genes(adata_proc, n_top_genes=hvg_n_top or 2000)
                except Exception as e:
                    if verbose:
                        logging.warning(f"HVG computation failed; proceeding without HVG restriction: {e}")
            if 'highly_variable' in adata_proc.var:
                hv_mask = adata_proc.var['highly_variable'].values
                if hv_mask.any():
                    adata_proc = adata_proc[:, hv_mask]
        X = adata_proc.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        embeds = MinMaxScaler().fit_transform(X.astype('float32'))
    else:
        raise ValueError(f"Dimensionality reduction method '{dim_reduction}' is not recognized.")

    if verbose:
        logging.info("Latent space embedding completed.")

    return embeds
