import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA, NMF
try:
    from sklearn.decomposition import MiniBatchNMF  # sklearn >= 1.1
except Exception:  # pragma: no cover - optional
    MiniBatchNMF = None
from concurrent.futures import ProcessPoolExecutor
import logging
from contextlib import contextmanager
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.vectorized import contains
from collections import Counter
import itertools
import os
import hashlib
import scipy
from scipy.sparse import csr_matrix
from typing import Optional
from types import SimpleNamespace

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


def _resolve_thread_limit(n_jobs: Optional[int]) -> Optional[int]:
    """Resolve n_jobs-style values to a concrete thread cap for BLAS/OpenMP."""
    if n_jobs is None:
        return None
    try:
        n = int(n_jobs)
    except Exception:
        return None
    if n > 0:
        return n
    if n == -1:
        return os.cpu_count() or None
    return None


def _thread_cap_enabled() -> bool:
    """
    Whether to enforce BLAS/OpenMP thread limits via threadpoolctl.

    Default is disabled to preserve previous behavior and avoid per-call
    threadpool reconfiguration overhead on very large PCA workloads.
    Enable with SPIX_ENABLE_THREAD_CAP=1.
    """
    val = os.environ.get("SPIX_ENABLE_THREAD_CAP", "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


_LOG_NORM_CACHE_KEY = "_spix_log_norm_cache"
_LAPACK_INT32_MAX = np.iinfo(np.int32).max


def _var_names_signature(var_names: pd.Index) -> str:
    """Stable signature for var_names compatibility checks."""
    values = pd.Index(var_names).astype(str).to_numpy(dtype=str, copy=False)
    joined = "\x1f".join(values.tolist())
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _minmax_scale_dense(X: np.ndarray) -> np.ndarray:
    """Min-max scale a dense array in-place when possible."""
    arr = np.asarray(X, dtype=np.float32)
    if not arr.flags.writeable:
        arr = np.array(arr, dtype=np.float32, copy=True)
    mins = np.min(arr, axis=0, keepdims=True)
    maxs = np.max(arr, axis=0, keepdims=True)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    arr -= mins
    arr /= denom
    return arr


def _sparse_pca_lapack_overflow_risk(adata: AnnData, n_comps: int) -> bool:
    """
    Detect sparse PCA shapes that can overflow LP64 LAPACK indexing in SciPy's
    ARPACK-backed centered PCA path.
    """
    if not scipy.sparse.issparse(adata.X):
        return False
    try:
        k = int(n_comps)
    except Exception:
        return False
    if k <= 0:
        return False
    largest_dim = max(int(adata.n_obs), int(adata.n_vars))
    return largest_dim * k > _LAPACK_INT32_MAX


def _supports_centered_covariance_pca(
    adata_proc: AnnData,
    n_comps: int,
    max_vars: int,
) -> bool:
    """Whether exact centered covariance PCA is a practical sparse fallback."""
    return (
        scipy.sparse.issparse(adata_proc.X)
        and int(adata_proc.n_obs) > 1
        and int(adata_proc.n_vars) > 1
        and 0 < int(n_comps) < int(adata_proc.n_vars)
        and int(adata_proc.n_vars) <= max(1, int(max_vars))
    )


def _resolve_sparse_pca_strategy(
    adata_proc: AnnData,
    n_comps: int,
    sparse_strategy: str,
    centered_covariance_max_vars: int,
) -> str:
    """
    Resolve sparse PCA execution mode.

    Modes:
    - 'auto': use Scanpy by default; switch to exact centered covariance PCA when
      the ARPACK/LAPACK sparse path would overflow and the feature count is small
      enough; otherwise fall back to uncentered randomized SVD.
    - 'scanpy': always use Scanpy defaults.
    - 'uncentered': force zero_center=False randomized SVD.
    - 'centered_covariance': force exact centered covariance/eigendecomposition.
    """
    strategy = str(sparse_strategy or "auto").strip().lower()
    valid = {"auto", "scanpy", "uncentered", "centered_covariance"}
    if strategy not in valid:
        raise ValueError(
            f"Unrecognized pca_sparse_strategy={sparse_strategy!r}. Expected one of {sorted(valid)}."
        )
    if not scipy.sparse.issparse(adata_proc.X):
        return "scanpy"
    if strategy != "auto":
        if strategy == "centered_covariance" and not _supports_centered_covariance_pca(
            adata_proc,
            n_comps=n_comps,
            max_vars=centered_covariance_max_vars,
        ):
            raise ValueError(
                "pca_sparse_strategy='centered_covariance' requires sparse input with "
                f"1 < n_comps < n_vars <= {centered_covariance_max_vars}."
            )
        return strategy
    if _sparse_pca_lapack_overflow_risk(adata_proc, n_comps):
        if _supports_centered_covariance_pca(
            adata_proc,
            n_comps=n_comps,
            max_vars=centered_covariance_max_vars,
        ):
            return "centered_covariance"
        return "uncentered"
    return "scanpy"


def _score_dense_from_sparse_components(
    X: scipy.sparse.spmatrix,
    components_t: np.ndarray,
    mean_projection: np.ndarray,
    chunk_rows: int = 200_000,
) -> np.ndarray:
    """Project sparse rows onto dense components in chunks to avoid large temporaries."""
    n_obs = int(X.shape[0])
    n_comps = int(components_t.shape[1])
    scores = np.empty((n_obs, n_comps), dtype=np.float32)
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, n_obs, chunk_rows):
        stop = min(n_obs, start + chunk_rows)
        block_scores = X[start:stop] @ components_t
        block_scores = np.asarray(block_scores, dtype=np.float32)
        block_scores -= mean_projection[None, :]
        scores[start:stop] = block_scores
    return scores


def _run_centered_covariance_pca(
    adata_proc: AnnData,
    n_comps: int,
    blas_threads: Optional[int],
    feature_mask: Optional[np.ndarray] = None,
    verbose: bool = True,
    log_prefix: str = "PCA",
) -> None:
    """
    Exact centered PCA via covariance eigendecomposition for sparse HVG matrices.

    This avoids the sparse ARPACK -> LAPACK path that can overflow on extremely
    large ``n_obs`` while preserving the centered PCA objective.
    """
    X = adata_proc.X
    if not scipy.sparse.issparse(X):
        raise ValueError("Centered covariance PCA fallback requires sparse input.")
    feature_mask_bool = None
    if feature_mask is not None:
        feature_mask_bool = np.asarray(feature_mask, dtype=bool)
        if feature_mask_bool.shape[0] != int(adata_proc.n_vars):
            raise ValueError("feature_mask length must match adata_proc.n_vars.")
        if not np.any(feature_mask_bool):
            raise ValueError("feature_mask selected zero variables.")
        X = X[:, feature_mask_bool]
    X = X.tocsr().astype(np.float32, copy=False)
    n_obs = int(X.shape[0])
    n_vars = int(X.shape[1])
    if n_obs <= 1:
        raise ValueError("Centered covariance PCA requires at least 2 observations.")
    if not (0 < int(n_comps) < n_vars):
        raise ValueError("Centered covariance PCA requires 0 < n_comps < n_vars.")

    if verbose:
        logging.info(
            "%s sparse path selected with exact centered covariance eigendecomposition "
            "(shape=%s x %s, n_comps=%s, blas_threads=%s).",
            log_prefix,
            n_obs,
            n_vars,
            n_comps,
            blas_threads,
        )

    with _limit_blas_openmp(blas_threads):
        mean = np.asarray(X.sum(axis=0)).ravel().astype(np.float32, copy=False) / np.float32(n_obs)
        xtx = X.T @ X
        if scipy.sparse.issparse(xtx):
            cov = xtx.toarray().astype(np.float32, copy=False)
        else:
            cov = np.asarray(xtx, dtype=np.float32)
        cov -= np.float32(n_obs) * np.outer(mean, mean)
        cov /= np.float32(n_obs - 1)
        cov = 0.5 * (cov + cov.T)

        lo = max(0, n_vars - int(n_comps))
        try:
            eigvals, eigvecs = scipy.linalg.eigh(
                cov,
                subset_by_index=(lo, n_vars - 1),
                driver="evr",
                check_finite=False,
            )
        except Exception:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = eigvals[lo:]
            eigvecs = eigvecs[:, lo:]
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(np.asarray(eigvals[order], dtype=np.float32), 0.0)
        components = np.asarray(eigvecs[:, order], dtype=np.float32)
        projected_mean = np.asarray(mean @ components, dtype=np.float32)
        scores = _score_dense_from_sparse_components(
            X,
            components_t=components,
            mean_projection=projected_mean,
        )

    total_variance = float(np.trace(cov, dtype=np.float64))
    variance_ratio = (
        np.asarray(eigvals / total_variance, dtype=np.float64)
        if total_variance > 0
        else np.zeros_like(eigvals, dtype=np.float64)
    )
    adata_proc.obsm["X_pca"] = scores
    if feature_mask_bool is None:
        adata_proc.varm["PCs"] = components
    else:
        pcs_full = np.zeros((int(adata_proc.n_vars), int(n_comps)), dtype=np.float32)
        pcs_full[feature_mask_bool, :] = components
        adata_proc.varm["PCs"] = pcs_full
    adata_proc.uns["pca"] = {
        "variance": np.asarray(eigvals, dtype=np.float64),
        "variance_ratio": variance_ratio,
        "params": {
            "zero_center": True,
            "fallback": "centered_covariance",
            "used_feature_mask": bool(feature_mask_bool is not None),
        },
    }


def _run_scanpy_pca(
    adata_proc: AnnData,
    n_comps: int,
    blas_threads: Optional[int],
    sparse_strategy: str = "auto",
    centered_covariance_max_vars: int = 4096,
    feature_mask: Optional[np.ndarray] = None,
    verbose: bool = True,
    log_prefix: str = "PCA",
) -> None:
    """
    Run PCA with guarded sparse fallbacks for very large matrices.
    """
    is_sparse = scipy.sparse.issparse(adata_proc.X)
    effective_n_vars = int(adata_proc.n_vars)
    strategy_proxy = adata_proc
    if feature_mask is not None:
        effective_mask = np.asarray(feature_mask, dtype=bool)
        effective_n_vars = int(np.count_nonzero(effective_mask))
        strategy_proxy = SimpleNamespace(
            X=adata_proc.X,
            n_obs=int(adata_proc.n_obs),
            n_vars=effective_n_vars,
        )
    resolved_sparse_strategy = _resolve_sparse_pca_strategy(
        strategy_proxy,
        n_comps=n_comps,
        sparse_strategy=sparse_strategy,
        centered_covariance_max_vars=centered_covariance_max_vars,
    )

    if verbose and is_sparse:
        if resolved_sparse_strategy == "centered_covariance":
            logging.info(
                "%s sparse path selected with exact centered covariance fallback "
                "(shape=%s x %s, n_comps=%s, blas_threads=%s).",
                log_prefix,
                adata_proc.n_obs,
                effective_n_vars,
                n_comps,
                blas_threads,
            )
        elif resolved_sparse_strategy == "uncentered":
            logging.info(
                "%s sparse path selected with zero_center=False and svd_solver='randomized' "
                "(shape=%s x %s, n_comps=%s, blas_threads=%s).",
                log_prefix,
                adata_proc.n_obs,
                effective_n_vars,
                n_comps,
                blas_threads,
            )
        else:
            logging.info(
                "%s sparse path selected (shape=%s x %s, scanpy defaults preserved, blas_threads=%s).",
                log_prefix,
                adata_proc.n_obs,
                effective_n_vars,
                blas_threads,
            )

    if resolved_sparse_strategy == "centered_covariance":
        _run_centered_covariance_pca(
            adata_proc,
            n_comps=n_comps,
            blas_threads=blas_threads,
            feature_mask=feature_mask,
            verbose=False,
            log_prefix=log_prefix,
        )
        return

    with _limit_blas_openmp(blas_threads):
        if resolved_sparse_strategy == "uncentered":
            sc.tl.pca(
                adata_proc,
                n_comps=n_comps,
                zero_center=False,
                svd_solver='randomized',
                dtype='float32',
            )
            return
        try:
            sc.tl.pca(adata_proc, n_comps=n_comps)
        except ValueError as e:
            if is_sparse and "integer overflow in LAPACK" in str(e):
                overflow_proxy = strategy_proxy
                overflow_strategy = (
                    "centered_covariance"
                    if _supports_centered_covariance_pca(
                        overflow_proxy,
                        n_comps=n_comps,
                        max_vars=centered_covariance_max_vars,
                    )
                    else "uncentered"
                )
                if verbose:
                    logging.warning(
                        "%s sparse PCA hit LAPACK int32 overflow; retrying with %s fallback.",
                        log_prefix,
                        overflow_strategy,
                    )
                if overflow_strategy == "centered_covariance":
                    _run_centered_covariance_pca(
                        adata_proc,
                        n_comps=n_comps,
                        blas_threads=blas_threads,
                        feature_mask=feature_mask,
                        verbose=False,
                        log_prefix=log_prefix,
                    )
                else:
                    sc.tl.pca(
                        adata_proc,
                        n_comps=n_comps,
                        zero_center=False,
                        svd_solver='randomized',
                        dtype='float32',
                    )
            else:
                raise


def _cache_log_norm_hvg(
    adata: AnnData,
    adata_proc: AnnData,
    nfeatures: int,
    verbose: bool = False,
) -> None:
    """Cache HVG metadata for log-normalized layer reuse across repeated runs."""
    if 'highly_variable' not in adata_proc.var:
        return
    try:
        adata.uns[_LOG_NORM_CACHE_KEY] = {
            'nfeatures': int(nfeatures),
            'var_names_sig': _var_names_signature(adata_proc.var_names),
            'hvg_mask': adata_proc.var['highly_variable'].to_numpy(dtype=bool, copy=True),
        }
    except Exception as e:
        if verbose:
            logging.warning(f"Failed to cache log_norm HVG metadata: {e}")


def _try_restore_log_norm_cache(
    adata: AnnData,
    nfeatures: int,
    dim_reduction: str,
    compute_threads: Optional[int],
    verbose: bool = True,
) -> Optional[AnnData]:
    """
    Rebuild processed AnnData from cached log_norm layer and cached/recomputed HVGs.
    Returns None if cache cannot be used safely.
    """
    if 'log_norm' not in adata.layers:
        return None

    if adata.raw is not None:
        var_df = adata.raw.var.copy()
        var_names = adata.raw.var_names.copy()
    else:
        var_df = adata.var.copy()
        var_names = adata.var_names.copy()

    X_cached = adata.layers['log_norm']
    if X_cached.shape[0] != adata.n_obs or X_cached.shape[1] != len(var_names):
        return None

    adata_proc = AnnData(X=X_cached, obs=adata.obs.copy(), var=var_df)
    adata_proc.obs_names = pd.Index(adata.obs_names.astype(str))
    adata_proc.var_names = pd.Index(var_names.astype(str))

    restored_hvg = False
    cache = adata.uns.get(_LOG_NORM_CACHE_KEY)
    if isinstance(cache, dict):
        try:
            cached_nfeatures = int(cache.get('nfeatures', -1))
            cached_var_sig = cache.get('var_names_sig')
            cached_hvg_mask = cache.get('hvg_mask')
            if (
                cached_nfeatures == int(nfeatures)
                and cached_hvg_mask is not None
                and len(cached_hvg_mask) == adata_proc.n_vars
                and cached_var_sig is not None
                and str(cached_var_sig) == _var_names_signature(adata_proc.var_names)
            ):
                adata_proc.var['highly_variable'] = np.asarray(cached_hvg_mask, dtype=bool)
                restored_hvg = True
        except Exception:
            restored_hvg = False

    if restored_hvg:
        if verbose:
            logging.info("Reused cached log_norm layer and HVG mask.")
        return adata_proc

    # Cache exists but metadata is missing/stale; recompute HVG only (faster than full log_norm).
    with _limit_blas_openmp(compute_threads):
        if dim_reduction == 'Harmony':
            sc.pp.highly_variable_genes(adata_proc, batch_key='library_id', inplace=True)
        else:
            sc.pp.highly_variable_genes(adata_proc, n_top_genes=nfeatures)
    if verbose:
        logging.info("Reused log_norm layer and recomputed HVGs.")
    return adata_proc


def _coerce_string_indices_inplace(adata: AnnData) -> None:
    """Normalize obs/var names to plain string Index to avoid CategoricalIndex issues."""
    adata.obs_names = pd.Index(adata.obs_names.astype(str))
    adata.var_names = pd.Index(adata.var_names.astype(str))

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
    use_inferred_grid_tiles: bool = False,
    inferred_grid_mode: str = 'direct',
    inferred_grid_pitch: Optional[float] = None,
    inferred_grid_neighbor_radius: int = 1,
    inferred_grid_min_neighbors: int = 3,
    inferred_grid_closing_radius: int = 0,
    inferred_grid_fill_holes: bool = False,
    coords_max_gap_factor: Optional[float] = None,
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
    pca_blas_threads: Optional[int] = None,    # optional BLAS/OpenMP threads during PCA
    pca_sparse_strategy: str = 'auto',         # 'auto'|'scanpy'|'uncentered'|'centered_covariance'
    pca_centered_covariance_max_vars: int = 4096,
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
        using ``reduce_tensor_resolution`` before writing. Optional coordinate
        compaction can be enabled via ``coords_max_gap_factor`` / ``coords_compaction``.
    use_inferred_grid_tiles : bool, optional (default=False)
        If True, infer a post-QC regular lattice from the surviving coordinates
        and use one of the inferred-grid tile builders controlled by
        ``inferred_grid_mode``.
    inferred_grid_mode : {'direct','boundary_voronoi'}, optional (default='direct')
        Inferred-grid tiling mode.
        - 'direct': keep observed spots at their original coordinates and add
          only local synthetic support cells.
        - 'boundary_voronoi': infer a support boundary on the post-QC lattice,
          fill its interior, and assign every support cell to the nearest
          surviving barcode.
    inferred_grid_pitch : float or None, optional (default=None)
        Override the inferred lattice pitch. When None, X/Y pitches are inferred
        separately from the surviving coordinates.
    inferred_grid_neighbor_radius : int, optional (default=1)
        Radius of the local support neighborhood used to propose synthetic grid
        cells near observed ones.
    inferred_grid_min_neighbors : int, optional (default=3)
        Minimum number of observed neighbors within the support neighborhood for
        an empty inferred grid cell to be synthesized.
    inferred_grid_closing_radius : int, optional (default=0)
        Optional additional binary-closing radius on the inferred occupancy grid.
        Keep at 0 for the most conservative behavior.
    inferred_grid_fill_holes : bool, optional (default=False)
        If True, fill enclosed holes in the inferred occupancy grid before
        nearest-spot assignment. This is more aggressive and more artificial.
    coords_max_gap_factor : float or None, optional (default=None)
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
    pca_blas_threads : int or None, optional (default=None)
        Optional BLAS/OpenMP thread cap applied only during PCA. `None` preserves
        previous behavior; set to `1` if you need the most conservative/stable PCA path.
    pca_sparse_strategy : str, optional (default='auto')
        Sparse PCA execution strategy. 'auto' preserves Scanpy defaults unless a
        sparse centered PCA call would overflow LAPACK indexing, in which case it
        prefers exact centered covariance PCA when ``n_vars`` is small enough and
        otherwise falls back to ``zero_center=False`` randomized SVD.
    pca_centered_covariance_max_vars : int, optional (default=4096)
        Maximum feature count allowed for the exact centered covariance sparse
        PCA fallback.

    """
    _coerce_string_indices_inplace(adata)
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
            use_inferred_grid_tiles=use_inferred_grid_tiles,
            inferred_grid_mode=inferred_grid_mode,
            inferred_grid_pitch=inferred_grid_pitch,
            inferred_grid_neighbor_radius=inferred_grid_neighbor_radius,
            inferred_grid_min_neighbors=inferred_grid_min_neighbors,
            inferred_grid_closing_radius=inferred_grid_closing_radius,
            inferred_grid_fill_holes=inferred_grid_fill_holes,
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
    cap_threads = _thread_cap_enabled()
    compute_threads = _resolve_thread_limit(n_jobs) if cap_threads else None
    if verbose and compute_threads is not None:
        logging.info(f"Applying BLAS/OpenMP thread cap for normalization/PCA: {compute_threads}")
    elif verbose and n_jobs is not None and not cap_threads:
        logging.info("BLAS/OpenMP thread cap disabled (set SPIX_ENABLE_THREAD_CAP=1 to enable).")
    adata_proc = None
    if normalization == 'log_norm' and use_counts == 'raw':
        adata_proc = _try_restore_log_norm_cache(
            adata=adata,
            nfeatures=nfeatures,
            dim_reduction=dim_reduction,
            compute_threads=compute_threads,
            verbose=verbose,
        )
        if adata_proc is not None and verbose:
            logging.info("Counts processing completed (cache hit).")

    if adata_proc is None:
        adata_proc = process_counts(
            adata,
            method=normalization,
            dim_reduction=dim_reduction,
            use_counts=use_counts,
            nfeatures=nfeatures,
            min_cutoff=min_cutoff,
            verbose=verbose,
            compute_threads=compute_threads,
        )

    if normalization == 'log_norm' and use_counts == 'raw':
        _cache_log_norm_hvg(adata, adata_proc, nfeatures=nfeatures, verbose=verbose)

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
            pca_blas_threads=pca_blas_threads,
            pca_sparse_strategy=pca_sparse_strategy,
            pca_centered_covariance_max_vars=pca_centered_covariance_max_vars,
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
    verbose: bool = True,
    compute_threads: Optional[int] = None,
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
    compute_threads : int or None, optional (default=None)
        Optional BLAS/OpenMP thread cap for heavy normalization/HVG steps.
    
    Returns
    -------
    AnnData
        The processed AnnData object.
    """
    _coerce_string_indices_inplace(adata)
    if verbose:
        logging.info(f"Processing counts with method: {method}")

    # Select counts based on 'use_counts' parameter
    if use_counts == 'raw':
        # Avoid copying the full AnnData (obsm/uns/etc). We only need X/obs/var for
        # normalization + HVG/PCA. This is a major memory win on large objects.
        if adata.raw is not None:
            counts = adata.raw.X
            var_df = adata.raw.var.copy()
            var_names = adata.raw.var_names.copy()
        else:
            counts = adata.X
            var_df = adata.var.copy()
            var_names = adata.var_names.copy()

        X = counts.copy() if hasattr(counts, "copy") else counts
        adata_proc = AnnData(X=X, obs=adata.obs.copy(), var=var_df)
        adata_proc.obs_names = pd.Index(adata.obs_names.astype(str))
        adata_proc.var_names = pd.Index(var_names.astype(str))
    else:
        if use_counts not in adata.layers:
            raise ValueError(f"Layer '{use_counts}' not found in adata.layers.")
        counts = adata.layers[use_counts]
        X = counts.copy() if hasattr(counts, "copy") else counts
        adata_proc = AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())
        adata_proc.obs_names = pd.Index(adata.obs_names.astype(str))
        adata_proc.var_names = pd.Index(adata.var_names.astype(str))

    # Apply normalization and feature selection
    if method == 'log_norm':
        with _limit_blas_openmp(compute_threads):
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
    pca_blas_threads: Optional[int] = None,
    pca_sparse_strategy: str = 'auto',
    pca_centered_covariance_max_vars: int = 4096,
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
    pca_blas_threads : int or None
        Optional BLAS/OpenMP thread cap applied only during PCA.
    pca_sparse_strategy : str, optional
        Sparse PCA execution strategy. See generate_embeddings docstring.
    pca_centered_covariance_max_vars : int, optional
        Maximum feature count allowed for the exact centered covariance sparse
        PCA fallback.
    
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
    pca_threads = _resolve_thread_limit(n_jobs) if _thread_cap_enabled() else None

    if dim_reduction == 'PCA':
        # Optionally restrict to HVGs
        hv_mask = None
        if use_hvg_only and 'highly_variable' in adata_proc.var:
            hv_mask = adata_proc.var['highly_variable'].values
            if hv_mask.any():
                strategy_proxy = SimpleNamespace(
                    X=adata_proc.X,
                    n_obs=int(adata_proc.n_obs),
                    n_vars=int(np.count_nonzero(hv_mask)),
                )
                resolved_pca_strategy = _resolve_sparse_pca_strategy(
                    strategy_proxy,
                    n_comps=dimensions,
                    sparse_strategy=pca_sparse_strategy,
                    centered_covariance_max_vars=pca_centered_covariance_max_vars,
                )
                if resolved_pca_strategy != "centered_covariance":
                    # Non-centered-covariance paths still rely on a physical HVG subset.
                    adata_proc = adata_proc[:, hv_mask].copy()
                    hv_mask = None
        effective_pca_threads = pca_blas_threads if pca_blas_threads is not None else pca_threads
        _run_scanpy_pca(
            adata_proc,
            n_comps=dimensions,
            blas_threads=effective_pca_threads,
            sparse_strategy=pca_sparse_strategy,
            centered_covariance_max_vars=pca_centered_covariance_max_vars,
            feature_mask=hv_mask,
            verbose=verbose,
            log_prefix="PCA",
        )
        embeds = adata_proc.obsm['X_pca']
        embeds = _minmax_scale_dense(embeds)
    elif dim_reduction == 'PCA_L':
        _run_scanpy_pca(
            adata_proc,
            n_comps=dimensions,
            blas_threads=pca_threads,
            sparse_strategy=pca_sparse_strategy,
            centered_covariance_max_vars=pca_centered_covariance_max_vars,
            verbose=verbose,
            log_prefix="PCA_L",
        )
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
        embeds = _minmax_scale_dense(embeds)
    elif dim_reduction == 'UMAP':
        sc.pp.neighbors(adata_proc, n_pcs=dimensions)
        sc.tl.umap(adata_proc, n_components=3)
        embeds = adata_proc.obsm['X_umap']
        embeds = _minmax_scale_dense(embeds)
    elif dim_reduction == 'LSI':
        embeds = run_lsi(adata_proc, n_components=dimensions, remove_first=remove_lsi_1)
    elif dim_reduction == 'LSI_UMAP':
        lsi_embeds = run_lsi(adata_proc, n_components=dimensions, remove_first=remove_lsi_1)
        adata_proc.obsm['X_lsi'] = lsi_embeds
        sc.pp.neighbors(adata_proc, use_rep='X_lsi')
        sc.tl.umap(adata_proc, n_components=3)
        embeds = adata_proc.obsm['X_umap']
        embeds = _minmax_scale_dense(embeds)
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

        embeds = _minmax_scale_dense(W)
    elif dim_reduction == 'Harmony':
        # Optionally restrict to HVGs
        if use_hvg_only and 'highly_variable' in adata_proc.var:
            hv_mask = adata_proc.var['highly_variable'].values
            if hv_mask.any():
                adata_proc = adata_proc[:, hv_mask].copy()
        # Perform PCA before Harmony integration
        _run_scanpy_pca(
            adata_proc,
            n_comps=dimensions,
            blas_threads=pca_threads,
            sparse_strategy=pca_sparse_strategy,
            centered_covariance_max_vars=pca_centered_covariance_max_vars,
            verbose=verbose,
            log_prefix="Harmony PCA",
        )
        # Run Harmony integration using 'library_id' as the batch key
        sc.external.pp.harmony_integrate(adata_proc, key=library_id)
        # Retrieve the Harmony integrated embeddings; these are stored in 'X_pca_harmony'
        embeds = adata_proc.obsm['X_pca_harmony']
        embeds = _minmax_scale_dense(embeds)
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
        embeds = _minmax_scale_dense(X.astype('float32', copy=False))
    else:
        raise ValueError(f"Dimensionality reduction method '{dim_reduction}' is not recognized.")

    if verbose:
        logging.info("Latent space embedding completed.")

    return embeds
