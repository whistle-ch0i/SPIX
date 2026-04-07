import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply, LinearOperator
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import multiprocessing
import platform
import os
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from itertools import product

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_SMOOTH_PRECOMPUTED = None  # set per-process via initializer (or inherited via fork)


def smooth_image(
    adata,
    methods           = ['bilateral'],        # list of methods in order: 'knn', 'graph', 'bilateral', 'gaussian', 'guided', 'heat'
    embedding         = 'X_embedding',        # key for raw embeddings in adata.obsm
    embedding_dims    = None,                 # list of embedding dimensions to use (None for all)
    output            = 'X_embedding_smooth', # key for smoothed embeddings in adata.obsm
    obs_mask          = None,                 # optional mask (bool array) or adata.obs key (str) to smooth only a subset (e.g. in_tissue)
    # K-NN smoothing parameters
    knn_k             = 35,                   # Number of neighbors for KNN/gaussian
    knn_sigma         = 2.0,                  # Kernel width for KNN spatial distance
    knn_chunk         = 10_000,               # Chunk size for KNN/gaussian weighted averaging (memory control)
    # Graph-diffusion smoothing parameters
    graph_k           = 30,                   # Number of neighbors for graph construction
    graph_t           = 2,                    # Number of diffusion steps
    graph_tol         = None,                # Early-stop tolerance for graph diffusion (max abs change); None disables
    graph_check_every = 1,                   # Check graph convergence every N steps when graph_tol is set
    approx_mode       = None,                # None or 'grid' to approximate all methods on a coarse grid
    approx_target_n   = 200_000,             # Target number of bins for approx_mode='grid'
    approx_bin        = None,                # Bin size (coord units) for approx_mode='grid'
    approx_max_bins   = None,                # Hard cap on bin count for approx_mode='grid'
    graph_explicit    = 'explicit',                # 'explicit'|'implicit'|'auto' or bool to control graph operator
    graph_explicit_max_mb = None,            # Max MB allowed for explicit graph operator in 'auto' mode
    use_float32       = True,                # Override SPIX_SMOOTH_FLOAT32; bool or None
    fast_float32      = True,               # Use float32 weights/accumulation for speed
    # Bilateral filtering parameters
    bilateral_k       = 30,                   # Number of neighbors for bilateral filtering
    bilateral_sigma_r = 0.3,                  # Kernel width for embedding space distance
    bilateral_t       = 3,                    # Number of diffusion steps
    # Gaussian filtering parameters
    gaussian_sigma    = 1.0,                  # Standard deviation for gaussian filter
    gaussian_k        = None,                 # Neighbor count for gaussian smoothing (None -> knn_k)
    gaussian_t        = 1,                    # Number of gaussian smoothing steps
    gaussian_operator = None,                # 'chunked'|'sparse'|'auto' or bool for gaussian operator
    gaussian_operator_max_mb = None,         # Max MB allowed for sparse operator in 'auto' mode
    gaussian_sparse_sort = None,             # Sort sparse indices for gaussian operator (bool/None)
    # Guided filtering parameters
    guided_k          = None,                # Neighbor count for guided filtering (None -> knn_k)
    guided_eps        = 1e-3,                # Regularization for guided filter
    guided_sigma      = None,                # Spatial sigma for guided filter (None -> uniform)
    guided_t          = 1,                   # Number of guided filtering steps
    # Heat kernel parameters
    heat_time         = 1.0,                 # Diffusion time for heat kernel
    heat_k            = None,                # Neighbor count for heat kernel (None -> gaussian_k/knn_k)
    heat_sigma        = None,                # Spatial sigma for heat kernel (None -> gaussian_sigma)
    heat_method       = None,                # 'krylov'|'poisson'|'auto'
    heat_tol          = 1e-4,                # Poisson-series tail tolerance (fraction of mass)
    heat_max_k        = None,                # Max series steps for heat_method='poisson'
    heat_min_k        = 0,                   # Min series steps for heat_method='poisson'
    heat_sparse_sort  = None,                # Sort sparse indices for heat operator (bool/None)
    # Rescaling behavior
    rescale_mode      = 'final',           # 'per_step' (backward compatible), 'final', or 'none'
    n_jobs = None,
    # Parallelization backend
    backend = 'processes',                    # 'processes' (default) or 'threads'
    mp_start_method = None,                   # None=auto; else one of {'fork','spawn','forkserver'}
    implementation: str = "auto",             # 'auto'|'vectorized'|'legacy'
):
    """
    Sequentially apply one or more smoothing methods to an embedding and store
    the final smoothed embedding in adata.obsm[output]. Neighbor queries are
    precomputed once for speed and memory efficiency, and only per-dimension
    vectors are sent to worker processes to avoid copying the full matrix.

    Smoothing methods:
    - 'knn': K-Nearest Neighbors weighted averaging based on spatial distance.
    - 'graph': Graph diffusion based on a spatially weighted graph.
    - 'bilateral': Edge-preserving smoothing using spatial and embedding distances.
    - 'gaussian': Gaussian smoothing using spatial Gaussian weights (KNN-limited).
    - 'guided': Guided filter on a spatial neighbor graph (edge-preserving).
    - 'heat': Heat-kernel smoothing on a spatial neighbor graph.

    Args:
        adata (anndata.AnnData): AnnData with spatial coordinates in adata.uns['tiles']
            and embeddings in adata.obsm.
        methods (list[str]): Smoothing methods to apply in sequence.
        embedding (str): Key in adata.obsm for the raw embedding matrix (n x d).
        embedding_dims (list[int] | None): Indices of embedding dimensions to process
            (None = all dimensions).
        output (str): Key in adata.obsm where the smoothed embedding is stored.
        obs_mask (str|array-like|None): If provided, smooth only the selected
            observations (e.g. `obs_mask="in_tissue"`). Unselected rows are
            copied through from the input embedding.
        knn_k (int): Neighbor count for KNN/gaussian smoothing.
        knn_sigma (float): Spatial kernel width for KNN smoothing.
        knn_chunk (int): Process KNN/gaussian in chunks to cap memory.
        graph_k (int): Neighbor count for graph construction.
        graph_t (int): Number of graph diffusion steps.
        graph_tol (float|None): If set, stop early when max abs change <= graph_tol.
            This can speed up large t at a small accuracy cost.
        graph_check_every (int): Check convergence every N steps when graph_tol is set.
        approx_mode (str|None): If 'grid', run all requested methods on a coarse
            grid and map results back to cells (approximate).
        approx_target_n (int): Target number of grid bins for approx_mode='grid'.
        approx_bin (float|None): Grid bin size in coordinate units; overrides target.
        approx_max_bins (int|None): Hard cap on bin count for approx_mode='grid'.
        graph_explicit (str|bool|None): 'explicit' uses symmetric operator (1 matmul/step),
            'implicit' uses W + Wᵀ (2 matmul/step), 'auto' chooses by memory; bool maps
            to explicit/implicit. None uses env or default.
        graph_explicit_max_mb (float|None): Auto-mode memory cap for explicit operator (MB).
        use_float32 (bool|None): If True, use float32 for graph weights/degree; if None,
            fallback to SPIX_SMOOTH_FLOAT32.
        fast_float32 (bool): If True, use float32 for neighbor-weight math (KNN/gaussian/bilateral)
            and accumulation for higher speed with small numeric drift.
        bilateral_k (int): Neighbor count for bilateral filtering.
        bilateral_sigma_r (float): Embedding kernel width for bilateral filtering.
        bilateral_t (int): Number of bilateral diffusion steps.
        gaussian_sigma (float): Spatial sigma for gaussian smoothing.
        gaussian_k (int|None): Neighbor count for gaussian smoothing (None -> knn_k).
        gaussian_t (int): Number of gaussian smoothing steps.
        gaussian_operator (str|bool|None): 'sparse' uses a CSR operator (1 matmul/step),
            'chunked' recomputes weights per step, 'auto' chooses by memory; bool maps
            to sparse/chunked. None uses env or default.
        gaussian_operator_max_mb (float|None): Auto-mode memory cap for sparse operator (MB).
        gaussian_sparse_sort (bool|None): If True, sort sparse indices (may help or hurt);
            None uses env or default.
        guided_k (int|None): Neighbor count for guided filtering (None -> knn_k).
        guided_eps (float): Guided filter regularization (epsilon).
        guided_sigma (float|None): Spatial sigma for guided filter weights (None -> uniform).
        guided_t (int): Number of guided filtering steps.
        heat_time (float): Diffusion time for heat kernel.
        heat_k (int|None): Neighbor count for heat kernel (None -> gaussian_k/knn_k).
        heat_sigma (float|None): Spatial sigma for heat kernel (None -> gaussian_sigma).
        heat_method (str|None): 'krylov' (expm_multiply), 'poisson' (series),
            or 'auto' (choose based on size); None uses env or default.
        heat_tol (float): Poisson-series tail tolerance (smaller = more accurate).
        heat_max_k (int|None): Cap for Poisson-series steps; None uses a heuristic.
        heat_min_k (int): Minimum Poisson-series steps to take.
        heat_sparse_sort (bool|None): Sort sparse indices for heat operator;
            None uses env or default.
        rescale_mode (str): 'per_step' (default, backward compatible), 'final' (only
            rescale after all methods), or 'none' (no rescaling).
        n_jobs (int|None): Total worker count used for neighbor queries and smoothing.

    Returns:
        anndata.AnnData: With smoothed embedding at adata.obsm[output].
    """
    global _SMOOTH_PRECOMPUTED
    logging.info("Starting embedding smoothing process.")
    if isinstance(methods, (str, bytes)):
        methods = [methods]
    methods = [str(m).strip().lower() for m in methods]
    logging.info(f"Input embedding key: '{embedding}'")
    logging.info(f"Output embedding key: '{output}'")
    logging.info(f"Smoothing methods to apply (in order): {methods}")
    if use_float32 is None:
        use_float32 = str(os.environ.get("SPIX_SMOOTH_FLOAT32", "0")).strip().lower() in {"1", "true", "yes"}
    else:
        use_float32 = bool(use_float32)
    if fast_float32:
        use_float32 = True
    if use_float32:
        logging.info("Smoothing precision: float32")
    if fast_float32:
        logging.info("Fast float32 math enabled for neighbor weights.")
    if graph_tol is not None:
        graph_tol = float(graph_tol)
        if graph_tol <= 0:
            graph_tol = None
    graph_check_every = int(max(1, graph_check_every))
    if graph_tol is not None:
        logging.info("Graph early-stop enabled: tol=%g, check_every=%d", graph_tol, graph_check_every)
    approx_mode_val = None
    if approx_mode is not None:
        if isinstance(approx_mode, bool):
            approx_mode_val = "grid" if approx_mode else None
        else:
            approx_mode_val = str(approx_mode).strip().lower()
            if approx_mode_val in {"none", "off", "false", "0"}:
                approx_mode_val = None
    if approx_mode_val is not None and approx_mode_val not in {"grid"}:
        raise ValueError("approx_mode must be one of {None,'grid','none'}.")
    if approx_max_bins is None:
        try:
            approx_max_bins = int(
                os.environ.get("SPIX_SMOOTH_APPROX_MAX_BINS", "500000")
            )
        except Exception:
            approx_max_bins = 500000
    else:
        approx_max_bins = int(approx_max_bins)
    if approx_max_bins is not None and approx_max_bins <= 0:
        approx_max_bins = None
    # 1) load embedding first (to get n)
    try:
        raw_embedding = np.asarray(adata.obsm[embedding])
        # Avoid an unconditional full-matrix copy; only convert if needed.
        if raw_embedding.dtype != np.float32:
            raw_embedding = raw_embedding.astype(np.float32, copy=False)
    except KeyError:
        logging.error(f"Embedding key '{embedding}' not found in adata.obsm.")
        raise

    n, total_dim = raw_embedding.shape
    # Resolve spatial coordinates aligned to obs
    coords = None
    try:
        if 'spatial' in adata.obsm:
            coords_candidate = np.asarray(adata.obsm['spatial'], dtype='float32')
            if coords_candidate.shape[0] == n:
                coords = coords_candidate[:, :2]
    except Exception:
        coords = None

    if coords is None:
        try:
            tiles_df = adata.uns['tiles']
            tiles_origin = tiles_df[tiles_df['origin'] == 1].copy()
            tiles_origin['barcode'] = tiles_origin['barcode'].astype(str)
            obs_idx = pd.Index(adata.obs_names.astype(str))
            tiles_aligned = tiles_origin.set_index('barcode').reindex(obs_idx)
            if tiles_aligned[['x', 'y']].isna().any().any():
                missing = int(tiles_aligned[['x', 'y']].isna().any(axis=1).sum())
                raise ValueError(
                    f"Missing coordinates for {missing} observations when aligning tiles to obs."
                )
            coords = tiles_aligned[['x', 'y']].to_numpy(dtype='float32')
        except KeyError as e:
            logging.error(f"Spatial coordinates not found in adata.obsm['spatial'] or adata.uns['tiles']: {e}")
            raise
    logging.info(f"Loaded coordinates for {coords.shape[0]} observations.")
    # Optional: smooth only a subset of observations (e.g. in_tissue) to avoid
    # bleeding signal into background regions (common in VisiumHD).
    full_mask = None
    raw_embedding_full = None
    coords_full = None
    if obs_mask is not None:
        if isinstance(obs_mask, (str, bytes)):
            key = str(obs_mask)
            if key not in adata.obs:
                raise KeyError(f"obs_mask key '{key}' not found in adata.obs.")
            full_mask = np.asarray(adata.obs[key]).astype(bool, copy=False)
        else:
            full_mask = np.asarray(obs_mask).astype(bool, copy=False)
        if full_mask.shape[0] != n:
            raise ValueError("obs_mask must have length n_obs.")
        if not bool(full_mask.any()):
            raise ValueError("obs_mask selects 0 observations.")
        if bool(full_mask.all()):
            full_mask = None
        else:
            raw_embedding_full = raw_embedding
            coords_full = coords
            raw_embedding = raw_embedding[full_mask]
            coords = coords[full_mask]
            n, total_dim = raw_embedding.shape
    if embedding_dims is None:
        dims_to_use = list(range(total_dim))
    else:
        dims_to_use = embedding_dims
    
    logging.info(f"Embedding shape: ({n}, {total_dim}), processing dims: {dims_to_use}")
    if n_jobs is None or int(n_jobs) == -1:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1 or -1.")
    neighbor_workers = n_jobs

    # 3) Precompute neighbors/graphs once in the parent (memory + speed)
    precomputed = {}

    need_graph = 'graph' in methods
    need_knn = 'knn' in methods
    need_gaussian = 'gaussian' in methods
    need_guided = 'guided' in methods
    need_heat = 'heat' in methods
    need_tree = need_graph or need_knn or need_gaussian or need_guided or need_heat or ('bilateral' in methods)

    if need_gaussian:
        gaussian_t = int(gaussian_t)
        if gaussian_t < 1:
            raise ValueError("gaussian_t must be >= 1.")
        if float(gaussian_sigma) <= 0:
            raise ValueError("gaussian_sigma must be > 0.")
    else:
        gaussian_t = int(gaussian_t)

    if need_guided:
        guided_t = int(guided_t)
        if guided_t < 1:
            raise ValueError("guided_t must be >= 1.")
        if guided_sigma is not None and float(guided_sigma) <= 0:
            raise ValueError("guided_sigma must be > 0 when provided.")
    else:
        guided_t = int(guided_t)

    if need_heat:
        heat_time = float(heat_time)
        if heat_time < 0:
            raise ValueError("heat_time must be >= 0.")
        heat_sigma_eff = gaussian_sigma if heat_sigma is None else heat_sigma
        if float(heat_sigma_eff) <= 0:
            raise ValueError("heat_sigma must be > 0.")
    else:
        heat_time = float(heat_time)
        heat_sigma_eff = gaussian_sigma if heat_sigma is None else heat_sigma
    if heat_method is None:
        heat_method = str(os.environ.get("SPIX_SMOOTH_HEAT_METHOD", "krylov")).strip().lower()
    else:
        heat_method = str(heat_method).strip().lower()
    if heat_method in {"expm", "expm_multiply"}:
        heat_method = "krylov"
    elif heat_method in {"series", "taylor"}:
        heat_method = "poisson"
    elif heat_method in {"auto"}:
        # Favor the Poisson-series approximation for large problems.
        heat_method = "poisson" if int(n) >= 200_000 else "krylov"
    if heat_method not in {"krylov", "poisson"}:
        raise ValueError("heat_method must be one of {'krylov','poisson','auto'}.")
    heat_tol = float(heat_tol)
    if heat_tol < 0:
        raise ValueError("heat_tol must be >= 0.")
    heat_min_k = int(max(0, heat_min_k))
    heat_max_k = None if heat_max_k is None else int(max(0, heat_max_k))
    if heat_max_k is not None and heat_max_k < heat_min_k:
        heat_max_k = heat_min_k
    if heat_sparse_sort is None:
        heat_sparse_sort = str(
            os.environ.get("SPIX_SMOOTH_HEAT_SPARSE_SORT", "0")
        ).strip().lower() in {"1", "true", "yes"}
    else:
        heat_sparse_sort = bool(heat_sparse_sort)

    k_knn = int(max(1, min(int(knn_k), n))) if need_knn else 0
    if need_gaussian:
        gaussian_k_eff = knn_k if gaussian_k is None else gaussian_k
        k_gauss = int(max(1, min(int(gaussian_k_eff), n)))
    else:
        k_gauss = 0
    if need_guided:
        guided_k_eff = knn_k if guided_k is None else guided_k
        k_guided = int(max(1, min(int(guided_k_eff), n)))
    else:
        k_guided = 0
    if need_heat:
        if heat_k is None:
            heat_k_eff = gaussian_k if gaussian_k is not None else knn_k
        else:
            heat_k_eff = heat_k
        k_heat = int(max(1, min(int(heat_k_eff), n)))
    else:
        k_heat = 0

    if gaussian_operator is None:
        gaussian_mode = str(os.environ.get("SPIX_SMOOTH_GAUSSIAN_OPERATOR", "chunked")).strip().lower()
    elif isinstance(gaussian_operator, bool):
        gaussian_mode = "sparse" if gaussian_operator else "chunked"
    else:
        gaussian_mode = str(gaussian_operator).strip().lower()
    if gaussian_mode in {"weights", "cache", "cached"}:
        gaussian_mode = "chunked"
    if gaussian_operator_max_mb is None:
        try:
            gaussian_operator_max_mb = float(
                os.environ.get("SPIX_SMOOTH_GAUSSIAN_OPERATOR_MAX_MB", "1024")
            )
        except Exception:
            gaussian_operator_max_mb = 1024.0
    else:
        gaussian_operator_max_mb = float(gaussian_operator_max_mb)
    if gaussian_sparse_sort is None:
        gaussian_sparse_sort = str(
            os.environ.get("SPIX_SMOOTH_GAUSSIAN_SPARSE_SORT", "0")
        ).strip().lower() in {"1", "true", "yes"}
    else:
        gaussian_sparse_sort = bool(gaussian_sparse_sort)

    if approx_mode_val is not None:
        logging.info("Approximate smoothing enabled: mode=%s", approx_mode_val)
        approx_info = _precompute_approx_grid(
            coords,
            graph_k=graph_k,
            graph_t=graph_t,
            graph_explicit=graph_explicit,
            graph_explicit_max_mb=graph_explicit_max_mb,
            use_float32=use_float32,
            target_n=approx_target_n,
            bin_size=approx_bin,
            max_bins=approx_max_bins,
            build_graph=('graph' in methods),
            neighbor_workers=neighbor_workers,
        )
        precomputed_approx = {}
        if 'graph' in methods:
            g = approx_info.get("graph")
            if g is None:
                raise RuntimeError("Approximate graph operator missing.")
            precomputed_approx["graph"] = g
        centers = approx_info.get("centers")
        if centers is None:
            raise RuntimeError("Approximate bin centers missing.")
        n_bins = int(approx_info.get("n_bins", centers.shape[0]))

        need_knn = 'knn' in methods
        need_gaussian = 'gaussian' in methods
        need_guided = 'guided' in methods
        need_heat = 'heat' in methods
        need_bilateral = 'bilateral' in methods

        k_knn = int(max(1, min(int(knn_k), n_bins))) if need_knn else 0
        if need_gaussian:
            gaussian_k_eff = knn_k if gaussian_k is None else gaussian_k
            k_gauss = int(max(1, min(int(gaussian_k_eff), n_bins)))
        else:
            k_gauss = 0
        if need_guided:
            guided_k_eff = knn_k if guided_k is None else guided_k
            k_guided = int(max(1, min(int(guided_k_eff), n_bins)))
        else:
            k_guided = 0
        if need_heat:
            if heat_k is None:
                heat_k_eff = gaussian_k if gaussian_k is not None else knn_k
            else:
                heat_k_eff = heat_k
            k_heat = int(max(1, min(int(heat_k_eff), n_bins)))
        else:
            k_heat = 0
        if need_bilateral:
            k_bi = int(max(1, min(int(bilateral_k) + 1, n_bins)))
        else:
            k_bi = 0

        if need_knn or need_gaussian or need_guided or need_heat or need_bilateral:
            tree_bins = cKDTree(centers, balanced_tree=True, compact_nodes=True)
            k_all = int(max(k_knn, k_gauss, k_guided, k_heat, k_bi))
            d_all, idx_all = tree_bins.query(centers, k=k_all, workers=neighbor_workers)
            if k_all == 1:
                d_all = d_all[:, None]
                idx_all = idx_all[:, None]
            if need_knn:
                precomputed_approx['knn'] = {
                    'dists': d_all[:, :k_knn].astype('float32'),
                    'idx': idx_all[:, :k_knn].astype('int32'),
                }
            if need_gaussian:
                if need_knn and k_gauss == k_knn:
                    precomputed_approx['gaussian'] = precomputed_approx['knn']
                else:
                    precomputed_approx['gaussian'] = {
                        'dists': d_all[:, :k_gauss].astype('float32'),
                        'idx': idx_all[:, :k_gauss].astype('int32'),
                    }
            if need_guided:
                guided_entry = {
                    'idx': idx_all[:, :k_guided].astype('int32'),
                }
                if guided_sigma is not None:
                    guided_entry['dists'] = d_all[:, :k_guided].astype('float32')
                precomputed_approx['guided'] = guided_entry
            if need_heat:
                precomputed_approx['heat'] = {
                    'dists': d_all[:, :k_heat].astype('float32'),
                    'idx': idx_all[:, :k_heat].astype('int32'),
                }
            if need_bilateral:
                d_bi = d_all[:, :k_bi].astype('float32')
                idx_bi = idx_all[:, :k_bi].astype('int32')
                if k_bi > 1:
                    sigma_s = (np.median(d_bi[:, 1:], axis=1).astype('float32') + 1e-9).astype('float32')
                else:
                    sigma_s = np.full(d_bi.shape[0], 1.0, dtype='float32')
                precomputed_approx['bilateral'] = {
                    'dists': d_bi,
                    'idx': idx_bi,
                    'sigma_s': sigma_s,
                }
            try:
                del d_all, idx_all
            except Exception:
                pass
            gc.collect()

        if need_gaussian and gaussian_mode not in {"chunked", "0", "false", "no"}:
            use_sparse = False
            est_mb = None
            if gaussian_mode in {"sparse", "1", "true", "yes"}:
                use_sparse = True
            else:
                if gaussian_t > 1:
                    nnz = int(n_bins) * int(k_gauss)
                    data_bytes = np.dtype(np.float32 if fast_float32 else np.float64).itemsize
                    use_int32 = nnz <= np.iinfo(np.int32).max and n_bins <= np.iinfo(np.int32).max
                    index_bytes = np.dtype(np.int32 if use_int32 else np.int64).itemsize
                    est_bytes = nnz * (data_bytes + index_bytes) + (int(n_bins) + 1) * index_bytes
                    est_mb = est_bytes / (1024.0 ** 2)
                    use_sparse = est_mb <= float(gaussian_operator_max_mb)
            if use_sparse:
                gauss_data = precomputed_approx.get("gaussian")
                if gauss_data is None:
                    raise RuntimeError("Gaussian neighbors not precomputed but 'gaussian' requested.")
                Wg = _build_row_stochastic_csr(
                    gauss_data["dists"],
                    gauss_data["idx"],
                    sigma=float(gaussian_sigma),
                    fast_float32=fast_float32,
                    sort_indices=gaussian_sparse_sort,
                )
                gauss_data["W"] = Wg
                if gauss_data is not precomputed_approx.get("knn"):
                    gauss_data.pop("dists", None)
                    gauss_data.pop("idx", None)
                if est_mb is None:
                    nnz = int(Wg.nnz)
                    data_bytes = np.dtype(Wg.dtype).itemsize
                    index_bytes = np.dtype(Wg.indices.dtype).itemsize
                    est_bytes = nnz * (data_bytes + index_bytes) + (int(n_bins) + 1) * index_bytes
                    est_mb = est_bytes / (1024.0 ** 2)
                logging.info(
                    "Gaussian operator (approx): sparse (1 matmul/step), est %.1f MB.",
                    est_mb,
                )

        if need_heat and heat_time > 0:
            heat_data = precomputed_approx.get("heat")
            if heat_data is None:
                raise RuntimeError("Heat neighbors not precomputed but 'heat' requested.")
            W_heat = None
            if need_gaussian and k_heat == k_gauss and float(heat_sigma_eff) == float(gaussian_sigma):
                gauss_data = precomputed_approx.get("gaussian")
                if gauss_data is not None:
                    W_heat = gauss_data.get("W")
            if W_heat is None:
                W_heat = _build_row_stochastic_csr(
                    heat_data["dists"],
                    heat_data["idx"],
                    sigma=float(heat_sigma_eff),
                    fast_float32=fast_float32,
                    sort_indices=heat_sparse_sort,
                )
            heat_data["W"] = W_heat
            heat_data.pop("dists", None)
            heat_data.pop("idx", None)
            logging.info("Heat operator ready (approx).")

        sum_dtype = np.float32 if fast_float32 else np.float64
        E_bins = _aggregate_by_bins(
            raw_embedding[:, dims_to_use],
            approx_info["bin_id"],
            approx_info["counts"],
            sum_dtype,
        )

        impl = (implementation or "auto").lower()
        if impl not in {"auto", "vectorized", "legacy"}:
            raise ValueError("implementation must be one of {'auto','vectorized','legacy'}.")
        can_vectorize = (
            rescale_mode in {"final", "none"}
            and ("bilateral" not in methods)
            and all(m in {"graph", "gaussian", "knn", "guided", "heat"} for m in methods)
        )
        if impl == "vectorized" and not can_vectorize:
            logging.warning(
                "Vectorized engine not compatible with methods=%s; falling back to legacy.",
                methods,
            )
            impl = "legacy"
        if impl == "vectorized" or (impl == "auto" and can_vectorize):
            logging.info("Using vectorized smoothing engine (approx).")
            E_bins_smoothed = _smooth_matrix_vectorized(
                E_bins,
                methods=methods,
                precomputed=precomputed_approx,
                knn_sigma=knn_sigma,
                knn_chunk=knn_chunk,
                graph_t=graph_t,
                graph_tol=graph_tol,
                graph_check_every=graph_check_every,
                gaussian_sigma=gaussian_sigma,
                gaussian_t=gaussian_t,
                guided_eps=guided_eps,
                guided_sigma=guided_sigma,
                guided_t=guided_t,
                heat_time=heat_time,
                heat_method=heat_method,
                heat_tol=heat_tol,
                heat_min_k=heat_min_k,
                heat_max_k=heat_max_k,
                rescale_mode=rescale_mode,
                n_workers=max(1, min(n_jobs, len(dims_to_use))),
                fast_float32=fast_float32,
            )
        else:
            logging.info("Using legacy smoothing engine (approx).")
            max_workers = max(1, min(n_jobs, len(dims_to_use)))
            results = [None] * len(dims_to_use)
            if backend == 'threads':
                func = partial(
                    _process_dimension,
                    methods=methods,
                    precomputed=precomputed_approx,
                    knn_sigma=knn_sigma,
                    knn_chunk=knn_chunk,
                    bilateral_sigma_r=bilateral_sigma_r,
                    bilateral_t=bilateral_t,
                    graph_t=graph_t,
                    graph_tol=graph_tol,
                    graph_check_every=graph_check_every,
                    gaussian_t=gaussian_t,
                    gaussian_sigma=gaussian_sigma,
                    guided_eps=guided_eps,
                    guided_sigma=guided_sigma,
                    guided_t=guided_t,
                    heat_time=heat_time,
                    heat_method=heat_method,
                    heat_tol=heat_tol,
                    heat_min_k=heat_min_k,
                    heat_max_k=heat_max_k,
                    rescale_mode=rescale_mode,
                    fast_float32=fast_float32,
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(func, E_bins[:, idx].astype('float32')): idx
                        for idx in range(len(dims_to_use))
                    }
                    for future in tqdm(as_completed(future_to_idx),
                                       total=len(future_to_idx),
                                       desc="Smoothing dimensions (approx)"):
                        idx = future_to_idx[future]
                        results[idx] = future.result()
            else:
                if mp_start_method is None:
                    if platform.system() == 'Linux':
                        start = 'fork'
                    else:
                        start = 'spawn'
                else:
                    start = mp_start_method
                logging.info(f"Process start method (approx): {start}")
                mp_context = multiprocessing.get_context(start)
                _SMOOTH_PRECOMPUTED = precomputed_approx
                init_arg = None if start == 'fork' else precomputed_approx
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=mp_context,
                    initializer=_init_smoothing_worker,
                    initargs=(init_arg,)
                ) as executor:
                    future_to_idx = {
                        executor.submit(
                            _process_dimension,
                            E_bins[:, idx].astype('float32'),
                            methods,
                            None,
                            knn_sigma,
                            knn_chunk,
                            bilateral_sigma_r,
                            bilateral_t,
                            graph_t,
                            graph_tol,
                            graph_check_every,
                            gaussian_t,
                            gaussian_sigma,
                            guided_eps,
                            guided_sigma,
                            guided_t,
                            heat_time,
                            heat_method,
                            heat_tol,
                            heat_min_k,
                            heat_max_k,
                            rescale_mode,
                            fast_float32,
                        ): idx
                        for idx in range(len(dims_to_use))
                    }
                    for future in tqdm(as_completed(future_to_idx),
                                       total=len(future_to_idx),
                                       desc="Smoothing dimensions (approx)"):
                        idx = future_to_idx[future]
                        results[idx] = future.result()
            E_bins_smoothed = np.stack(results, axis=1).astype('float32')
            if rescale_mode == 'final':
                mn = E_bins_smoothed.min(axis=0, keepdims=True)
                mx = E_bins_smoothed.max(axis=0, keepdims=True)
                denom = (mx - mn)
                denom[denom < 1e-12] = 1e-12
                E_bins_smoothed = (E_bins_smoothed - mn) / denom

        E_smoothed = E_bins_smoothed[approx_info["bin_id"]]
        if full_mask is not None:
            out_full = raw_embedding_full[:, dims_to_use].astype("float32", copy=True)
            out_full[full_mask] = E_smoothed.astype("float32", copy=False)
            adata.obsm[output] = out_full
        else:
            adata.obsm[output] = E_smoothed.astype('float32', copy=False)
        logging.info(f"Finished. Stored smoothed embedding → adata.obsm['{output}']")
        return adata

    tree_spatial = None
    if need_tree:
        tree_spatial = cKDTree(coords, balanced_tree=True, compact_nodes=True)

    # If both graph + (knn/gaussian) are requested, a single KDTree query can
    # be shared. However, for huge VisiumHD datasets the float64/int64 arrays
    # returned by SciPy can dominate peak RSS. Default to separate queries for
    # large n to keep peak memory lower (can be overridden by env var).
    dists_all = None
    neigh_all = None
    if need_graph and (need_knn or need_gaussian or need_guided or need_heat):
        share_mode = str(os.environ.get("SPIX_SMOOTH_SHARED_NEIGHBORS", "auto")).strip().lower()
        try:
            max_n_shared = int(os.environ.get("SPIX_SMOOTH_SHARED_NEIGHBORS_MAX_N", "2000000"))
        except Exception:
            max_n_shared = 2000000
        use_shared = share_mode in {"1", "true", "yes"} or (share_mode == "auto" and int(n) <= int(max_n_shared))
        if use_shared:
            k_all = int(max(1, min(max(graph_k + 1, k_knn, k_gauss, k_guided, k_heat), n)))
            logging.info(f"Precomputing shared neighbors (k={k_all}) for graph+knn/gaussian...")
            dists_all, neigh_all = tree_spatial.query(coords, k=k_all, workers=neighbor_workers)

    if need_graph:
        logging.info("Precomputing spatial graph for diffusion...")
        k_graph = int(max(1, min(graph_k + 1, n)))
        if dists_all is not None and neigh_all is not None and dists_all.shape[1] >= k_graph:
            dists_g, neigh_g = dists_all[:, :k_graph], neigh_all[:, :k_graph]
        else:
            dists_g, neigh_g = tree_spatial.query(coords, k=k_graph, workers=neighbor_workers)
        if k_graph > 1:
            dists_g = dists_g[:, 1:]
            neigh_g = neigh_g[:, 1:]
            gk = dists_g.shape[1]
            sigma_loc = np.median(dists_g, axis=1) + 1e-9

            # Build CSR directly from (data, indices, indptr) to avoid allocating
            # an explicit `rows` array of length n*gk.
            nnz = int(n) * int(gk)
            use_int32 = nnz <= np.iinfo(np.int32).max and n <= np.iinfo(np.int32).max
            index_dtype = np.int32 if use_int32 else np.int64
            indptr = (np.arange(n + 1, dtype=index_dtype) * gk)
            indices = neigh_g.reshape(-1).astype(index_dtype, copy=False)
            if use_float32:
                d32 = dists_g.astype(np.float32, copy=False)
                s32 = sigma_loc.astype(np.float32, copy=False)
                data = np.exp(-(d32 ** 2) / (2.0 * (s32[:, None] ** 2))).reshape(-1).astype(np.float32, copy=False)
            else:
                data = np.exp(-(dists_g ** 2) / (2.0 * (sigma_loc[:, None] ** 2))).reshape(-1)
            Wg = csr_matrix((data, indices, indptr), shape=(n, n))
        else:
            Wg = csr_matrix((n, n))
        # Memory-optimized graph diffusion:
        # Legacy behavior builds a symmetric graph S = 0.5*(W + Wᵀ) then row-normalizes
        # it into P = D⁻¹ S. Constructing S and P can double/triple peak memory for
        # large nnz. Instead, store W and apply the operator implicitly:
        #   P·X = D⁻¹ * 0.5*(W·X + Wᵀ·X)
        # which is mathematically equivalent (up to floating-point associativity).
        row_sum = Wg.sum(1).A1
        col_sum = Wg.sum(0).A1
        deg = 0.5 * (row_sum + col_sum) + 1e-12
        invdeg = 1.0 / deg
        if use_float32 and invdeg.dtype != np.float32:
            invdeg = invdeg.astype(np.float32, copy=False)
        # Cache transpose once to avoid per-dimension/block transpose overhead.
        Wt = None
        if int(graph_t) > 0:
            Wt = Wg.transpose(copy=False)
        precomputed["graph"] = {
            "W": Wg,
            "Wt": Wt,
            "invdeg": invdeg,
        }
        # Optional: precompute explicit symmetric operator for faster diffusion
        # at the cost of extra memory. Enable via SPIX_SMOOTH_GRAPH_EXPLICIT.
        if graph_explicit is None:
            graph_mode = str(os.environ.get("SPIX_SMOOTH_GRAPH_EXPLICIT", "implicit")).strip().lower()
        elif isinstance(graph_explicit, bool):
            graph_mode = "explicit" if graph_explicit else "implicit"
        else:
            graph_mode = str(graph_explicit).strip().lower()
        if graph_explicit_max_mb is None:
            try:
                graph_explicit_max_mb = float(
                    os.environ.get("SPIX_SMOOTH_GRAPH_EXPLICIT_MAX_MB", "1024")
                )
            except Exception:
                graph_explicit_max_mb = 1024.0
        else:
            graph_explicit_max_mb = float(graph_explicit_max_mb)
        use_explicit = False
        est_mb = None
        if int(graph_t) > 0:
            if graph_mode in {"1", "true", "yes", "explicit"}:
                use_explicit = True
            elif graph_mode in {"0", "false", "no", "implicit"}:
                use_explicit = False
            else:
                nnz = int(Wg.nnz)
                data_bytes = int(np.dtype(Wg.dtype).itemsize)
                index_bytes = int(np.dtype(Wg.indices.dtype).itemsize)
                est_bytes = (2 * nnz) * (data_bytes + index_bytes) + (int(n) + 1) * index_bytes
                est_mb = est_bytes / (1024.0 ** 2)
                use_explicit = est_mb <= float(graph_explicit_max_mb)
        if use_explicit:
            if Wt is None:
                Wt = Wg.transpose(copy=False)
            S = (Wg + Wt).tocsr()
            S.data *= 0.5
            precomputed["graph"]["S"] = S
            precomputed["graph"]["W"] = None
            precomputed["graph"]["Wt"] = None
            Wg = None
            Wt = None
            if est_mb is None:
                nnz = int(S.nnz)
                data_bytes = int(np.dtype(S.dtype).itemsize)
                index_bytes = int(np.dtype(S.indices.dtype).itemsize)
                est_bytes = nnz * (data_bytes + index_bytes) + (int(n) + 1) * index_bytes
                est_mb = est_bytes / (1024.0 ** 2)
            logging.info(
                "Graph operator: explicit symmetric (1 matmul/step), est %.1f MB.",
                est_mb,
            )
        # Proactively release large temporary arrays from the neighbor query / weight construction
        # to reduce peak RSS before subsequent KNN precompute and smoothing steps.
        try:
            del dists_g, neigh_g, sigma_loc
        except Exception:
            pass
        try:
            del row_sum, col_sum, deg
        except Exception:
            pass
        gc.collect()
        logging.info("Spatial graph ready.")

    if need_knn or need_gaussian or need_guided or need_heat:
        logging.info(
            "Precomputing neighbors for knn/gaussian/guided/heat (k_knn=%d, k_gaussian=%d, k_guided=%d, k_heat=%d)...",
            k_knn,
            k_gauss,
            k_guided,
            k_heat,
        )
        k_ng = int(max(k_knn, k_gauss, k_guided, k_heat))
        if dists_all is not None and neigh_all is not None and dists_all.shape[1] >= k_ng:
            d_ng, idx_ng = dists_all[:, :k_ng], neigh_all[:, :k_ng]
        else:
            d_ng, idx_ng = tree_spatial.query(coords, k=k_ng, workers=neighbor_workers)
        if need_knn:
            precomputed['knn'] = {
                'dists': d_ng[:, :k_knn].astype('float32'),
                'idx': idx_ng[:, :k_knn].astype('int32'),
            }
        if need_gaussian:
            if need_knn and k_gauss == k_knn:
                precomputed['gaussian'] = precomputed['knn']
            else:
                precomputed['gaussian'] = {
                    'dists': d_ng[:, :k_gauss].astype('float32'),
                    'idx': idx_ng[:, :k_gauss].astype('int32'),
                }
        if need_guided:
            guided_entry = {
                'idx': idx_ng[:, :k_guided].astype('int32'),
            }
            if guided_sigma is not None:
                guided_entry['dists'] = d_ng[:, :k_guided].astype('float32')
            precomputed['guided'] = guided_entry
        if need_heat:
            precomputed['heat'] = {
                'dists': d_ng[:, :k_heat].astype('float32'),
                'idx': idx_ng[:, :k_heat].astype('int32'),
            }
        # Release float64/int64 query outputs (can be several GB for VisiumHD).
        try:
            del d_ng, idx_ng
        except Exception:
            pass
        gc.collect()

    if need_gaussian and gaussian_mode not in {"chunked", "0", "false", "no"}:
        use_sparse = False
        est_mb = None
        if gaussian_mode in {"sparse", "1", "true", "yes"}:
            use_sparse = True
        else:
            if gaussian_t > 1:
                nnz = int(n) * int(k_gauss)
                data_bytes = np.dtype(np.float32 if fast_float32 else np.float64).itemsize
                use_int32 = nnz <= np.iinfo(np.int32).max and n <= np.iinfo(np.int32).max
                index_bytes = np.dtype(np.int32 if use_int32 else np.int64).itemsize
                est_bytes = nnz * (data_bytes + index_bytes) + (int(n) + 1) * index_bytes
                est_mb = est_bytes / (1024.0 ** 2)
                use_sparse = est_mb <= float(gaussian_operator_max_mb)
        if use_sparse:
            gauss_data = precomputed.get("gaussian")
            if gauss_data is None:
                raise RuntimeError("Gaussian neighbors not precomputed but 'gaussian' requested.")
            Wg = _build_row_stochastic_csr(
                gauss_data["dists"],
                gauss_data["idx"],
                sigma=float(gaussian_sigma),
                fast_float32=fast_float32,
                sort_indices=gaussian_sparse_sort,
            )
            gauss_data["W"] = Wg
            if gauss_data is not precomputed.get("knn"):
                gauss_data.pop("dists", None)
                gauss_data.pop("idx", None)
            if est_mb is None:
                nnz = int(Wg.nnz)
                data_bytes = np.dtype(Wg.dtype).itemsize
                index_bytes = np.dtype(Wg.indices.dtype).itemsize
                est_bytes = nnz * (data_bytes + index_bytes) + (int(n) + 1) * index_bytes
                est_mb = est_bytes / (1024.0 ** 2)
            logging.info(
                "Gaussian operator: sparse (1 matmul/step), est %.1f MB.",
                est_mb,
            )

    if need_heat and heat_time > 0:
        heat_data = precomputed.get("heat")
        if heat_data is None:
            raise RuntimeError("Heat neighbors not precomputed but 'heat' requested.")
        W_heat = None
        if need_gaussian and k_heat == k_gauss and float(heat_sigma_eff) == float(gaussian_sigma):
            gauss_data = precomputed.get("gaussian")
            if gauss_data is not None:
                W_heat = gauss_data.get("W")
        if W_heat is None:
            W_heat = _build_row_stochastic_csr(
                heat_data["dists"],
                heat_data["idx"],
                sigma=float(heat_sigma_eff),
                fast_float32=fast_float32,
                sort_indices=heat_sparse_sort,
            )
        heat_data["W"] = W_heat
        heat_data.pop("dists", None)
        heat_data.pop("idx", None)
        logging.info("Heat operator ready.")

    # Release the shared neighbor arrays (can be huge for VisiumHD)
    dists_all = None
    neigh_all = None

    if 'bilateral' in methods:
        logging.info("Precomputing neighbors for bilateral filter...")
        k_bi = int(max(1, min(bilateral_k + 1, n)))
        d_bi, idx_bi = tree_spatial.query(coords, k=k_bi, workers=neighbor_workers)
        # keep self as first neighbor (consistent with previous behavior)
        precomputed['bilateral'] = {
            'dists': d_bi.astype('float32'),
            'idx': idx_bi.astype('int32'),
            'sigma_s': (np.median(d_bi[:, 1:], axis=1).astype('float32') + 1e-9).astype('float32')
        }

    # 4) decide number of processes
    max_workers = max(1, min(n_jobs, len(dims_to_use)))
    logging.info(f"Backend: {backend}; workers: {max_workers} (requested={n_jobs})")

    # 5) Fast vectorized engine (huge win for graph/gaussian/knn)
    impl = (implementation or "auto").lower()
    if impl not in {"auto", "vectorized", "legacy"}:
        raise ValueError("implementation must be one of {'auto','vectorized','legacy'}.")
    can_vectorize = (
        rescale_mode in {"final", "none"}  # per_step is kept on legacy path for strict compatibility
        and ("bilateral" not in methods)   # bilateral would require huge (n,k,d) intermediates
        and all(m in {"graph", "gaussian", "knn", "guided", "heat"} for m in methods)
    )
    if impl == "vectorized" and not can_vectorize:
        logging.warning(
            "Vectorized engine not compatible with methods=%s; falling back to legacy.",
            methods,
        )
        impl = "legacy"
    if impl == "vectorized" or (impl == "auto" and can_vectorize):
        logging.info("Using vectorized smoothing engine.")
        E_smoothed = _smooth_matrix_vectorized(
            raw_embedding[:, dims_to_use],
            methods=methods,
            precomputed=precomputed,
            knn_sigma=knn_sigma,
            knn_chunk=knn_chunk,
            graph_t=graph_t,
            graph_tol=graph_tol,
            graph_check_every=graph_check_every,
            gaussian_sigma=gaussian_sigma,
            gaussian_t=gaussian_t,
            guided_eps=guided_eps,
            guided_sigma=guided_sigma,
            guided_t=guided_t,
            heat_time=heat_time,
            heat_method=heat_method,
            heat_tol=heat_tol,
            heat_min_k=heat_min_k,
            heat_max_k=heat_max_k,
            rescale_mode=rescale_mode,
            n_workers=max_workers,
            fast_float32=fast_float32,
        )
        if full_mask is not None:
            out_full = raw_embedding_full[:, dims_to_use].astype("float32", copy=True)
            out_full[full_mask] = E_smoothed.astype("float32", copy=False)
            adata.obsm[output] = out_full
        else:
            adata.obsm[output] = E_smoothed
        logging.info(f"Finished. Stored smoothed embedding → adata.obsm['{output}']")
        return adata
    if impl == "legacy" or (impl == "auto" and not can_vectorize):
        logging.info("Using legacy smoothing engine.")


    # Prepare the dimension-processing function (pass only the column vector, not the full matrix)
    func = None  # will be defined per-backend below

    # Parallel execution with real-time progress bar per dimension
    # Build tasks as (vector, name) to avoid pickling the full matrix per worker
    results = [None] * len(dims_to_use)

    if backend == 'threads':
        # Threading shares memory; safe to pass precomputed directly
        func = partial(
            _process_dimension,
            methods=methods,
            precomputed=precomputed,
            knn_sigma=knn_sigma,
            knn_chunk=knn_chunk,
            bilateral_sigma_r=bilateral_sigma_r,
            bilateral_t=bilateral_t,
            graph_t=graph_t,
            graph_tol=graph_tol,
            graph_check_every=graph_check_every,
            gaussian_sigma=gaussian_sigma,
            gaussian_t=gaussian_t,
            guided_eps=guided_eps,
            guided_sigma=guided_sigma,
            guided_t=guided_t,
            heat_time=heat_time,
            heat_method=heat_method,
            heat_tol=heat_tol,
            heat_min_k=heat_min_k,
            heat_max_k=heat_max_k,
            rescale_mode=rescale_mode,
            fast_float32=fast_float32,
        )
        logging.info("Launching ThreadPoolExecutor for smoothing tasks.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(func, raw_embedding[:, dim_idx].astype('float32')): idx
                for idx, dim_idx in enumerate(dims_to_use)
            }
            for future in tqdm(as_completed(future_to_idx),
                               total=len(future_to_idx),
                               desc="Smoothing dimensions"):
                idx = future_to_idx[future]
                results[idx] = future.result()
    else:
        # Processes: avoid sending large 'precomputed' in every task.
        # Use a per-worker global initialized once; where possible, prefer 'fork' to inherit memory.
        # Determine start method
        if mp_start_method is None:
            if platform.system() == 'Linux':
                start = 'fork'
            else:
                start = 'spawn'
        else:
            start = mp_start_method
        logging.info(f"Process start method: {start}")

        mp_context = multiprocessing.get_context(start)

        # Set module-level global for forked children; for spawn, pass via initializer
        _SMOOTH_PRECOMPUTED = precomputed

        init_arg = None if start == 'fork' else precomputed

        logging.info("Launching ProcessPoolExecutor for smoothing tasks.")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_init_smoothing_worker,
            initargs=(init_arg,)
        ) as executor:
            # Submit tasks with precomputed=None to use per-worker global
            future_to_idx = {
                executor.submit(
                    _process_dimension,
                    raw_embedding[:, dim_idx].astype('float32'),
                    methods,
                    None,  # use global precomputed
                    knn_sigma,
                    knn_chunk,
                    bilateral_sigma_r,
                    bilateral_t,
                    graph_t,
                    graph_tol,
                    graph_check_every,
                    gaussian_t,
                    gaussian_sigma,
                    guided_eps,
                    guided_sigma,
                    guided_t,
                    heat_time,
                    heat_method,
                    heat_tol,
                    heat_min_k,
                    heat_max_k,
                    rescale_mode,
                    fast_float32,
                ): idx
                for idx, dim_idx in enumerate(dims_to_use)
            }
            for future in tqdm(as_completed(future_to_idx),
                               total=len(future_to_idx),
                               desc="Smoothing dimensions"):
                idx = future_to_idx[future]
                results[idx] = future.result()

    # 6) stack and save
    # Optional final rescaling across dimensions if requested
    E_smoothed = np.stack(results, axis=1).astype('float32')
    if rescale_mode == 'final':
        mn = E_smoothed.min(axis=0, keepdims=True)
        mx = E_smoothed.max(axis=0, keepdims=True)
        denom = (mx - mn)
        denom[denom < 1e-12] = 1e-12
        E_smoothed = (E_smoothed - mn) / denom
    if full_mask is not None:
        out_full = raw_embedding_full[:, dims_to_use].astype("float32", copy=True)
        out_full[full_mask] = E_smoothed.astype("float32", copy=False)
        adata.obsm[output] = out_full
    else:
        adata.obsm[output] = E_smoothed
    logging.info(f"Finished. Stored smoothed embedding → adata.obsm['{output}']")
    return adata


def _smooth_matrix_vectorized(
    E_in: np.ndarray,
    *,
    methods,
    precomputed,
    knn_sigma: float,
    knn_chunk: int,
    graph_t: int,
    graph_tol,
    graph_check_every: int,
    gaussian_sigma: float,
    gaussian_t: int,
    guided_eps: float,
    guided_sigma,
    guided_t: int,
    heat_time: float,
    heat_method: str,
    heat_tol: float,
    heat_min_k: int,
    heat_max_k,
    rescale_mode: str,
    n_workers: int = 1,
    fast_float32: bool = False,
) -> np.ndarray:
    """
    Blocked vectorized smoothing for methods independent across channels
    ('graph', 'gaussian', 'knn').

    Key goals:
    - Keep results consistent with the legacy per-dimension implementation.
    - Reduce Python overhead by processing multiple dimensions per task.
    - Control peak memory by processing dimension blocks of bounded size.

    Matches legacy output dtype/rescaling:
    - internal ops run in float32/float64 as induced by SciPy/fast_float32
    - for rescale_mode='final', min-max scaling is done in float32
    """
    methods = list(methods)
    E_src = np.asarray(E_in, dtype=np.float32)
    n_obs, n_dim = E_src.shape
    n_workers = int(max(1, n_workers))
    graph_t_int = int(graph_t)
    graph_tol_val = None if graph_tol is None else float(graph_tol)
    if graph_tol_val is not None and graph_tol_val <= 0:
        graph_tol_val = None
    graph_check_every_int = int(max(1, graph_check_every))
    gaussian_t_int = int(gaussian_t)
    guided_t_int = int(guided_t)
    heat_time_val = float(heat_time)
    heat_method_val = str(heat_method).strip().lower()
    heat_tol_val = float(heat_tol)
    heat_min_k_int = int(max(0, heat_min_k))
    heat_max_k_val = None if heat_max_k is None else int(max(0, heat_max_k))
    diff_chunk_rows = int(max(1, knn_chunk)) if knn_chunk is not None else 10000

    # Control peak memory by limiting float64 block size. Users can tune via env vars.
    try:
        # Default is sized to allow processing many dims in one sparse×dense call
        # for common n_dim=30 workflows, while still capping peak memory.
        target_block_mb = float(os.environ.get("SPIX_SMOOTH_BLOCK_MB", "512"))
    except Exception:
        target_block_mb = 512.0
    try:
        block_max = int(os.environ.get("SPIX_SMOOTH_BLOCK_MAX", "64"))
    except Exception:
        block_max = 64

    # float64 bytes per element = 8
    target_bytes = max(1.0, target_block_mb) * 1024.0 * 1024.0
    block_size_mem = int(target_bytes // max(1, n_obs) // 8)

    # Choose a block size that:
    # - caps peak memory (block_size_mem)
    # - avoids too many tiny blocks (Python overhead)
    # - still enables parallelism across blocks when beneficial
    block_size_mem = max(1, int(block_size_mem))
    try:
        # Minimum dimensions per block when splitting work across blocks.
        # Default=1 maximizes CPU utilization for the common n_dim=30 workflow.
        block_min = int(os.environ.get("SPIX_SMOOTH_BLOCK_MIN", "1"))
    except Exception:
        block_min = 1
    block_min = max(1, min(int(block_min), int(n_dim)))
    n_blocks_cap = int(max(1, np.ceil(n_dim / block_min)))
    n_blocks = int(min(n_workers, n_blocks_cap))
    block_size_parallel = int(np.ceil(n_dim / max(1, n_blocks)))
    block_size = min(block_size_mem, block_max, max(1, block_size_parallel))
    block_size = max(1, min(n_dim, int(block_size)))

    out = np.empty((n_obs, n_dim), dtype=np.float32)

    def _rescale_inplace_float64(X: np.ndarray) -> np.ndarray:
        mn = X.min(axis=0, keepdims=True)
        mx = X.max(axis=0, keepdims=True)
        denom = mx - mn
        denom[denom < 1e-12] = 1e-12
        return (X - mn) / denom

    def _rescale_inplace_float32(X: np.ndarray) -> np.ndarray:
        mn = X.min(axis=0, keepdims=True)
        mx = X.max(axis=0, keepdims=True)
        denom = mx - mn
        denom[denom < 1e-12] = 1e-12
        return (X - mn) / denom

    if "graph" in methods:
        g = precomputed.get("graph")
        if g is not None:
            W = g.get("W")
            if W is not None and g.get("Wt") is None and graph_t_int > 0:
                g["Wt"] = W.transpose(copy=False)

    def _process_block(col_start: int, col_end: int) -> None:
        # Ensure contiguous block to reduce internal copies during sparse matmul.
        E_blk = E_src[:, col_start:col_end]
        if not E_blk.flags["C_CONTIGUOUS"]:
            E_blk = np.ascontiguousarray(E_blk)
        E_work: np.ndarray = E_blk

        for method in methods:
            if method == "graph":
                g = precomputed.get("graph")
                if g is not None:
                    E_work = _apply_graph_operator(
                        g,
                        E_work,
                        graph_t=graph_t_int,
                        graph_tol=graph_tol_val,
                        graph_check_every=graph_check_every_int,
                        diff_chunk_rows=diff_chunk_rows,
                    )
                else:
                    Pg = precomputed.get("Pg")
                    if Pg is None:
                        raise RuntimeError("Graph transition matrix not precomputed but 'graph' requested.")
                    tmp = E_work
                    for _ in range(graph_t_int):
                        tmp = Pg.dot(tmp)
                    E_work = tmp
            elif method == "knn":
                knn_data = precomputed.get("knn")
                if knn_data is None:
                    raise RuntimeError("KNN neighbors not precomputed but 'knn' requested.")
                E_work = _apply_weighted_neighbors_chunked(
                    E_work,
                    idx=knn_data["idx"],
                    dists=knn_data["dists"],
                    sigma=float(knn_sigma),
                    chunk_rows=int(max(1, knn_chunk)),
                    fast_float32=fast_float32,
                )
            elif method == "gaussian":
                gauss_data = precomputed.get("gaussian")
                if gauss_data is None:
                    gauss_data = precomputed.get("knn")
                if gauss_data is None:
                    raise RuntimeError("Gaussian neighbors not precomputed but 'gaussian' requested.")
                Wg = gauss_data.get("W")
                if Wg is not None:
                    tmp = E_work
                    for _ in range(gaussian_t_int):
                        tmp = Wg.dot(tmp)
                    E_work = tmp
                else:
                    for _ in range(gaussian_t_int):
                        E_work = _apply_weighted_neighbors_chunked(
                            E_work,
                            idx=gauss_data["idx"],
                            dists=gauss_data["dists"],
                                sigma=float(gaussian_sigma),
                                chunk_rows=int(max(1, knn_chunk)),
                                fast_float32=fast_float32,
                            )
            elif method == "guided":
                gdata = precomputed.get("guided")
                if gdata is None:
                    raise RuntimeError("Guided neighbors not precomputed but 'guided' requested.")
                dists_g = gdata.get("dists") if guided_sigma is not None else None
                E_work = _guided_filter_chunked(
                    E_work,
                    idx=gdata["idx"],
                    dists=dists_g,
                    sigma=guided_sigma,
                    eps=float(guided_eps),
                    steps=guided_t_int,
                    chunk_rows=int(max(1, knn_chunk)),
                    fast_float32=fast_float32,
                )
            elif method == "heat":
                hdata = precomputed.get("heat")
                if hdata is None:
                    raise RuntimeError("Heat operator not precomputed but 'heat' requested.")
                if heat_time_val > 0:
                    W = hdata.get("W")
                    if W is None:
                        raise RuntimeError("Heat operator missing matrix in precomputed['heat'].")
                    if heat_method_val == "poisson":
                        E_work = _apply_heat_poisson(
                            W,
                            E_work,
                            time=heat_time_val,
                            tol=heat_tol_val,
                            min_k=heat_min_k_int,
                            max_k=heat_max_k_val,
                            fast_float32=fast_float32,
                        )
                    else:
                        n_rows = W.shape[0]
                        Wt = hdata.get("Wt")
                        if Wt is None:
                            Wt = W.transpose(copy=False)
                            hdata["Wt"] = Wt
                        matvec = lambda x, W=W: x - W.dot(x)
                        rmatvec = lambda x, Wt=Wt: x - Wt.dot(x)
                        matmat = lambda X, W=W: X - W.dot(X)
                        rmatmat = lambda X, Wt=Wt: X - Wt.dot(X)
                        L_op = LinearOperator((n_rows, n_rows), matvec, rmatvec, matmat, W.dtype, rmatmat)
                        E_work = expm_multiply((-heat_time_val) * L_op, E_work)
            else:
                raise ValueError(f"Unknown method '{method}'.")

            if rescale_mode == "per_step":
                E_work = _rescale_inplace_float64(np.asarray(E_work, dtype=np.float64))

        E_out = np.asarray(E_work, dtype=np.float32)
        if rescale_mode == "final":
            E_out = _rescale_inplace_float32(E_out)
        out[:, col_start:col_end] = E_out

    blocks = [(s, min(s + block_size, n_dim)) for s in range(0, n_dim, block_size)]
    try:
        logging.info(
            f"Vectorized smoothing blocks: n_dim={n_dim}, block_size={block_size}, blocks={len(blocks)}, workers={n_workers}"
        )
    except Exception:
        pass
    if n_workers <= 1 or len(blocks) <= 1:
        for s, e in blocks:
            _process_block(s, e)
        return out

    max_workers = int(min(n_workers, len(blocks)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_block, s, e) for s, e in blocks]
        for fut in futures:
            fut.result()
    return out


def _max_abs_diff_chunked(a: np.ndarray, b: np.ndarray, chunk_rows: int) -> float:
    """Return max(abs(a - b)) using row chunks to cap temporary memory."""
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    n = a.shape[0]
    chunk_rows = int(max(1, chunk_rows))
    max_val = 0.0
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        diff = np.abs(a[start:end] - b[start:end])
        local_max = float(diff.max())
        if local_max > max_val:
            max_val = local_max
    return max_val


def _aggregate_by_bins(
    E: np.ndarray,
    bin_id: np.ndarray,
    counts: np.ndarray,
    sum_dtype,
) -> np.ndarray:
    """Aggregate per-cell values into bins using bincount (sum / counts)."""
    n_bins = int(counts.shape[0])
    out = np.empty((n_bins, E.shape[1]), dtype=sum_dtype)
    for j in range(E.shape[1]):
        out[:, j] = np.bincount(bin_id, weights=E[:, j], minlength=n_bins)
    denom = counts.astype(sum_dtype, copy=True)
    denom[denom < 1e-12] = 1e-12
    out /= denom[:, None]
    return out


def _apply_graph_operator(
    g: dict,
    X: np.ndarray,
    *,
    graph_t: int,
    graph_tol,
    graph_check_every: int,
    diff_chunk_rows: int,
) -> np.ndarray:
    """Apply graph diffusion operator for graph_t steps with optional early stop."""
    t = int(graph_t)
    if t <= 0:
        return X
    invdeg = g.get("invdeg")
    if invdeg is None:
        raise RuntimeError("Graph operator missing fields in precomputed['graph'].")
    graph_tol_val = None if graph_tol is None else float(graph_tol)
    if graph_tol_val is not None and graph_tol_val <= 0:
        graph_tol_val = None
    graph_check_every_int = int(max(1, graph_check_every))
    diff_chunk_rows = int(max(1, diff_chunk_rows))

    S = g.get("S")
    W = g.get("W")
    tmp = X
    if S is not None:
        for step in range(t):
            y = S.dot(tmp)
            y *= invdeg[:, None]
            if graph_tol_val is not None and (step + 1) % graph_check_every_int == 0:
                if _max_abs_diff_chunked(y, tmp, diff_chunk_rows) <= graph_tol_val:
                    tmp = y
                    break
            tmp = y
    else:
        if W is None:
            raise RuntimeError("Graph operator missing matrix in precomputed['graph'].")
        Wt = g.get("Wt")
        if Wt is None:
            Wt = W.transpose(copy=False)
            g["Wt"] = Wt
        for step in range(t):
            y = W.dot(tmp)
            y += Wt.dot(tmp)
            y *= 0.5
            y *= invdeg[:, None]
            if graph_tol_val is not None and (step + 1) % graph_check_every_int == 0:
                if _max_abs_diff_chunked(y, tmp, diff_chunk_rows) <= graph_tol_val:
                    tmp = y
                    break
            tmp = y
    return tmp


def _precompute_approx_grid(
    coords: np.ndarray,
    *,
    graph_k: int,
    graph_t: int,
    graph_explicit,
    graph_explicit_max_mb,
    use_float32: bool,
    target_n,
    bin_size,
    max_bins,
    build_graph: bool = True,
    neighbor_workers: int,
) -> dict:
    """Build a coarse grid graph and mapping for approximate diffusion."""
    coords = np.asarray(coords, dtype=np.float32)
    n = int(coords.shape[0])
    if n == 0:
        raise ValueError("coords must be non-empty for approximation.")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    area = float(span[0] * span[1])

    if bin_size is None:
        try:
            target_n = int(target_n) if target_n is not None else 0
        except Exception:
            target_n = 0
        if target_n <= 0:
            target_n = int(min(200_000, max(10_000, n // 50)))
        bin_size = float(np.sqrt(area / max(1.0, float(target_n))))
    else:
        bin_size = float(bin_size)
    if bin_size <= 0:
        raise ValueError("approx_bin must be > 0.")
    if max_bins is not None:
        est_bins = area / max(1e-12, (bin_size * bin_size))
        if est_bins > float(max_bins):
            new_bin = float(np.sqrt(area / max(1.0, float(max_bins))))
            logging.warning(
                "Approx bin_size %.3g would create ~%.1f bins; capping to %.3g.",
                float(bin_size),
                float(est_bins),
                float(new_bin),
            )
            bin_size = new_bin

    bx = np.floor((coords[:, 0] - mins[0]) / bin_size).astype(np.int64)
    by = np.floor((coords[:, 1] - mins[1]) / bin_size).astype(np.int64)
    if bx.size == 0 or by.size == 0:
        raise ValueError("Empty bin arrays computed for approximation.")
    if bx.min() < 0:
        bx = np.maximum(bx, 0)
    if by.min() < 0:
        by = np.maximum(by, 0)
    max_bx = int(bx.max())
    max_by = int(by.max())
    use_key = max_bx <= np.iinfo(np.int32).max and max_by <= np.iinfo(np.int32).max
    if use_key:
        bx32 = bx.astype(np.uint64, copy=False)
        by32 = by.astype(np.uint64, copy=False)
        key = (bx32 << np.uint64(32)) | by32
        uniq, inv = np.unique(key, return_inverse=True)
        bx_u = (uniq >> np.uint64(32)).astype(np.int64, copy=False)
        by_u = (uniq & np.uint64(0xFFFFFFFF)).astype(np.int64, copy=False)
        n_bins = int(uniq.shape[0])
        del key, bx32, by32, uniq
    else:
        pairs = np.stack([bx, by], axis=1)
        uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
        bx_u = uniq_pairs[:, 0]
        by_u = uniq_pairs[:, 1]
        n_bins = int(uniq_pairs.shape[0])
        del pairs, uniq_pairs
    del bx, by
    gc.collect()

    if n_bins <= 0:
        raise ValueError("No bins created for approximation.")
    if n_bins <= np.iinfo(np.int32).max:
        inv = inv.astype(np.int32, copy=False)
    counts = np.bincount(inv, minlength=n_bins).astype(np.float32)
    centers = np.column_stack(
        (
            mins[0] + (bx_u.astype(np.float32) + 0.5) * bin_size,
            mins[1] + (by_u.astype(np.float32) + 0.5) * bin_size,
        )
    ).astype(np.float32)
    if n_bins == 1:
        graph = None
        if build_graph:
            invdeg = np.ones(1, dtype=np.float32 if use_float32 else np.float64)
            graph = {"W": csr_matrix((1, 1)), "Wt": None, "invdeg": invdeg}
        logging.info(
            "Approx grid: bins=%d (%.1fx reduction), bin_size=%.3g.",
            n_bins,
            float(n) / float(n_bins),
            float(bin_size),
        )
        return {
            "mode": "grid",
            "bin_id": inv,
            "counts": counts,
            "graph": graph,
            "centers": centers,
            "bin_size": bin_size,
            "n_bins": n_bins,
        }

    if not build_graph:
        logging.info(
            "Approx grid: bins=%d (%.1fx reduction), bin_size=%.3g.",
            n_bins,
            float(n) / float(n_bins),
            float(bin_size),
        )
        return {
            "mode": "grid",
            "bin_id": inv,
            "counts": counts,
            "graph": None,
            "centers": centers,
            "bin_size": bin_size,
            "n_bins": n_bins,
        }

    tree = cKDTree(centers, balanced_tree=True, compact_nodes=True)
    k_graph = int(max(1, min(int(graph_k) + 1, n_bins)))
    dists_g, neigh_g = tree.query(centers, k=k_graph, workers=int(neighbor_workers))
    if k_graph > 1:
        dists_g = dists_g[:, 1:]
        neigh_g = neigh_g[:, 1:]
        gk = dists_g.shape[1]
        sigma_loc = np.median(dists_g, axis=1) + 1e-9

        nnz = int(n_bins) * int(gk)
        use_int32 = nnz <= np.iinfo(np.int32).max and n_bins <= np.iinfo(np.int32).max
        index_dtype = np.int32 if use_int32 else np.int64
        indptr = (np.arange(n_bins + 1, dtype=index_dtype) * gk)
        indices = neigh_g.reshape(-1).astype(index_dtype, copy=False)
        if use_float32:
            d32 = dists_g.astype(np.float32, copy=False)
            s32 = sigma_loc.astype(np.float32, copy=False)
            data = np.exp(-(d32 ** 2) / (2.0 * (s32[:, None] ** 2))).reshape(-1).astype(np.float32, copy=False)
        else:
            data = np.exp(-(dists_g ** 2) / (2.0 * (sigma_loc[:, None] ** 2))).reshape(-1)
        Wg = csr_matrix((data, indices, indptr), shape=(n_bins, n_bins))
    else:
        Wg = csr_matrix((n_bins, n_bins))

    row_sum = Wg.sum(1).A1
    col_sum = Wg.sum(0).A1
    deg = 0.5 * (row_sum + col_sum) + 1e-12
    invdeg = 1.0 / deg
    if use_float32 and invdeg.dtype != np.float32:
        invdeg = invdeg.astype(np.float32, copy=False)
    Wt = None
    if int(graph_t) > 0:
        Wt = Wg.transpose(copy=False)
    graph = {"W": Wg, "Wt": Wt, "invdeg": invdeg}

    if graph_explicit is None:
        graph_mode = str(os.environ.get("SPIX_SMOOTH_GRAPH_EXPLICIT", "implicit")).strip().lower()
    elif isinstance(graph_explicit, bool):
        graph_mode = "explicit" if graph_explicit else "implicit"
    else:
        graph_mode = str(graph_explicit).strip().lower()
    if graph_explicit_max_mb is None:
        try:
            graph_explicit_max_mb = float(
                os.environ.get("SPIX_SMOOTH_GRAPH_EXPLICIT_MAX_MB", "1024")
            )
        except Exception:
            graph_explicit_max_mb = 1024.0
    else:
        graph_explicit_max_mb = float(graph_explicit_max_mb)

    use_explicit = False
    est_mb = None
    if int(graph_t) > 0:
        if graph_mode in {"1", "true", "yes", "explicit"}:
            use_explicit = True
        elif graph_mode in {"0", "false", "no", "implicit"}:
            use_explicit = False
        else:
            nnz = int(Wg.nnz)
            data_bytes = int(np.dtype(Wg.dtype).itemsize)
            index_bytes = int(np.dtype(Wg.indices.dtype).itemsize)
            est_bytes = (2 * nnz) * (data_bytes + index_bytes) + (int(n_bins) + 1) * index_bytes
            est_mb = est_bytes / (1024.0 ** 2)
            use_explicit = est_mb <= float(graph_explicit_max_mb)
    if use_explicit:
        if Wt is None:
            Wt = Wg.transpose(copy=False)
        S = (Wg + Wt).tocsr()
        S.data *= 0.5
        graph["S"] = S
        graph["W"] = None
        graph["Wt"] = None
        if est_mb is None:
            nnz = int(S.nnz)
            data_bytes = int(np.dtype(S.dtype).itemsize)
            index_bytes = int(np.dtype(S.indices.dtype).itemsize)
            est_bytes = nnz * (data_bytes + index_bytes) + (int(n_bins) + 1) * index_bytes
            est_mb = est_bytes / (1024.0 ** 2)
        logging.info(
            "Approx grid graph operator: explicit symmetric (1 matmul/step), est %.1f MB.",
            est_mb,
        )

    logging.info(
        "Approx grid: bins=%d (%.1fx reduction), bin_size=%.3g.",
        n_bins,
        float(n) / float(n_bins),
        float(bin_size),
    )
    return {
        "mode": "grid",
        "bin_id": inv,
        "counts": counts,
        "graph": graph,
        "centers": centers,
        "bin_size": bin_size,
        "n_bins": n_bins,
    }


def _apply_weighted_neighbors_chunked(
    E: np.ndarray,
    *,
    idx: np.ndarray,
    dists: np.ndarray,
    sigma: float,
    chunk_rows: int,
    fast_float32: bool = False,
) -> np.ndarray:
    """Apply Gaussian distance weights using precomputed neighbors (chunked rows).

    This mirrors the legacy per-dimension implementation but operates on a
    (n, b) block of dimensions at once to reduce overhead while keeping results
    consistent per dimension. fast_float32 trades small numeric drift for speed.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    idx = np.asarray(idx)
    dists = np.asarray(dists)
    X = np.asarray(E)
    if X.ndim != 2:
        raise ValueError("E must be a 2D array.")

    n = X.shape[0]
    chunk_rows = int(max(1, chunk_rows))
    sum_dtype = np.float32 if fast_float32 else np.float64
    out = np.empty((n, X.shape[1]), dtype=sum_dtype)
    if fast_float32:
        sigma_val = np.float32(sigma)
        denom = np.float32(2.0) * sigma_val * sigma_val
    else:
        sigma_val = float(sigma)
        denom = 2.0 * sigma_val * sigma_val

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        idx_chunk = idx[start:end]
        # Match legacy numeric behavior: distances are float32, and the square is
        # computed in float32 before promotion during division.
        d_chunk = dists[start:end].astype(np.float32, copy=False)
        w = np.exp(-(d_chunk ** 2) / denom)
        w_sum = w.sum(axis=1, keepdims=True, dtype=sum_dtype)
        w_sum[w_sum < 1e-12] = 1e-12
        w /= w_sum
        # Apply weights per-dimension to preserve legacy results while avoiding
        # recomputing weights for each dimension.
        for j in range(X.shape[1]):
            neigh = X[:, j][idx_chunk]
            out[start:end, j] = (w * neigh).sum(axis=1, dtype=sum_dtype)
    # Legacy uses float64; fast_float32 switches to float32 intermediates.
    return out


def _apply_heat_poisson(
    W: csr_matrix,
    X: np.ndarray,
    *,
    time: float,
    tol: float,
    min_k: int,
    max_k,
    fast_float32: bool = False,
) -> np.ndarray:
    """Approximate exp(-t(I-W))X via a Poisson-weighted series of W^k X."""
    if time <= 0:
        return np.asarray(X)
    X0 = np.asarray(X)
    if X0.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if tol < 0:
        raise ValueError("tol must be >= 0.")

    out_dtype = np.float32 if fast_float32 else np.float64
    Xk = X0.astype(out_dtype, copy=False)
    weight = np.exp(-time)
    weight = np.float32(weight) if fast_float32 else float(weight)
    out = Xk * weight
    sum_w = float(weight)

    if max_k is None:
        max_k = int(np.ceil(float(time) + 8.0 * np.sqrt(float(time) + 1.0)))
    max_k = int(max(min_k, max_k))

    for k in range(1, max_k + 1):
        Xk = W.dot(Xk)
        weight *= float(time) / float(k)
        out += (np.float32(weight) if fast_float32 else float(weight)) * Xk
        sum_w += float(weight)
        if k >= int(min_k) and (1.0 - sum_w) <= tol:
            break
    return out


def _apply_uniform_neighbors_chunked(
    E: np.ndarray,
    *,
    idx: np.ndarray,
    chunk_rows: int,
    fast_float32: bool = False,
) -> np.ndarray:
    """Apply uniform neighbor averaging (chunked rows)."""
    idx = np.asarray(idx)
    X = np.asarray(E)
    if X.ndim != 2:
        raise ValueError("E must be a 2D array.")
    if idx.ndim != 2:
        raise ValueError("idx must be a 2D array.")

    n = X.shape[0]
    k = int(idx.shape[1])
    if k < 1:
        raise ValueError("idx must have at least 1 neighbor.")
    chunk_rows = int(max(1, chunk_rows))
    sum_dtype = np.float32 if fast_float32 else np.float64
    out = np.empty((n, X.shape[1]), dtype=sum_dtype)
    inv_k = np.float32(1.0 / float(k)) if fast_float32 else 1.0 / float(k)

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        idx_chunk = idx[start:end]
        neigh = X[idx_chunk]
        out[start:end] = neigh.sum(axis=1, dtype=sum_dtype) * inv_k
    return out


def _guided_filter_chunked(
    E: np.ndarray,
    *,
    idx: np.ndarray,
    dists,
    sigma,
    eps: float,
    steps: int,
    chunk_rows: int,
    fast_float32: bool = False,
) -> np.ndarray:
    """Guided filter on a neighbor graph (self-guided by default)."""
    if steps < 1:
        return np.asarray(E)
    E_work = np.asarray(E)
    eps_val = np.float32(eps) if fast_float32 else float(eps)
    use_uniform = sigma is None

    for _ in range(int(steps)):
        if use_uniform:
            mean_I = _apply_uniform_neighbors_chunked(
                E_work, idx=idx, chunk_rows=chunk_rows, fast_float32=fast_float32
            )
            mean_I2 = _apply_uniform_neighbors_chunked(
                E_work * E_work, idx=idx, chunk_rows=chunk_rows, fast_float32=fast_float32
            )
        else:
            if dists is None:
                raise RuntimeError("guided_sigma provided but guided distances are missing.")
            mean_I = _apply_weighted_neighbors_chunked(
                E_work,
                idx=idx,
                dists=dists,
                sigma=float(sigma),
                chunk_rows=chunk_rows,
                fast_float32=fast_float32,
            )
            mean_I2 = _apply_weighted_neighbors_chunked(
                E_work * E_work,
                idx=idx,
                dists=dists,
                sigma=float(sigma),
                chunk_rows=chunk_rows,
                fast_float32=fast_float32,
            )
        var_I = mean_I2 - mean_I * mean_I
        a = var_I / (var_I + eps_val)
        b = mean_I - a * mean_I
        if use_uniform:
            mean_a = _apply_uniform_neighbors_chunked(
                a, idx=idx, chunk_rows=chunk_rows, fast_float32=fast_float32
            )
            mean_b = _apply_uniform_neighbors_chunked(
                b, idx=idx, chunk_rows=chunk_rows, fast_float32=fast_float32
            )
        else:
            mean_a = _apply_weighted_neighbors_chunked(
                a,
                idx=idx,
                dists=dists,
                sigma=float(sigma),
                chunk_rows=chunk_rows,
                fast_float32=fast_float32,
            )
            mean_b = _apply_weighted_neighbors_chunked(
                b,
                idx=idx,
                dists=dists,
                sigma=float(sigma),
                chunk_rows=chunk_rows,
                fast_float32=fast_float32,
            )
        E_work = mean_a * E_work + mean_b
    return E_work


def _build_row_stochastic_csr(
    dists: np.ndarray,
    idx: np.ndarray,
    *,
    sigma: float,
    fast_float32: bool,
    sort_indices: bool = False,
) -> csr_matrix:
    """Build a row-stochastic CSR operator from neighbor distances."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    d = np.asarray(dists)
    indices_src = np.asarray(idx)
    if d.ndim != 2:
        raise ValueError("dists must be a 2D array.")
    n, k = d.shape
    nnz = int(n) * int(k)
    use_int32 = nnz <= np.iinfo(np.int32).max and n <= np.iinfo(np.int32).max
    index_dtype = np.int32 if use_int32 else np.int64
    indptr = (np.arange(n + 1, dtype=index_dtype) * k)
    indices = indices_src.reshape(-1).astype(index_dtype, copy=False)

    base_dtype = np.float32
    w = d.astype(base_dtype, copy=True)
    if fast_float32:
        sigma_val = np.float32(sigma)
        denom = np.float32(2.0) * sigma_val * sigma_val
        sum_dtype = np.float32
    else:
        sigma_val = float(sigma)
        denom = 2.0 * sigma_val * sigma_val
        sum_dtype = np.float64
    w *= w
    w /= denom
    w *= -1.0
    np.exp(w, out=w)
    w_sum = w.sum(axis=1, keepdims=True, dtype=sum_dtype)
    w_sum[w_sum < 1e-12] = 1e-12
    w /= w_sum
    if fast_float32:
        data = w.reshape(-1).astype(np.float32, copy=False)
    else:
        data = w.reshape(-1).astype(np.float64, copy=False)
    Wg = csr_matrix((data, indices, indptr), shape=(n, n))
    if sort_indices:
        # Sorting indices can speed up repeated sparse×dense matmul in some cases.
        Wg.sum_duplicates()
        Wg.sort_indices()
    return Wg


def _init_smoothing_worker(precomputed_or_none):
    """Initializer to set per-process global with heavy precomputed data."""
    global _SMOOTH_PRECOMPUTED
    if precomputed_or_none is not None:
        _SMOOTH_PRECOMPUTED = precomputed_or_none
    try:
        logging.info(f"Smoothing worker ready. PID={os.getpid()}")
    except Exception:
        pass


def _build_graph_precomputed_from_neighbors(
    dists: np.ndarray,
    idx: np.ndarray,
    *,
    graph_t: int,
    graph_explicit="implicit",
    graph_explicit_max_mb=None,
    use_float32: bool = True,
):
    dists = np.asarray(dists)
    idx = np.asarray(idx)
    n_bins = int(dists.shape[0])
    if n_bins <= 0:
        raise ValueError("Graph precompute requires at least one row.")
    if dists.ndim != 2 or idx.ndim != 2 or dists.shape != idx.shape:
        raise ValueError("dists/idx must be 2D arrays with matching shape.")
    if n_bins == 1 or dists.shape[1] == 0:
        invdeg = np.ones(max(1, n_bins), dtype=np.float32 if use_float32 else np.float64)
        return {"W": csr_matrix((n_bins, n_bins)), "Wt": None, "invdeg": invdeg}

    sigma_loc = np.median(dists, axis=1) + 1e-9
    gk = int(dists.shape[1])
    nnz = int(n_bins) * int(gk)
    use_int32 = nnz <= np.iinfo(np.int32).max and n_bins <= np.iinfo(np.int32).max
    index_dtype = np.int32 if use_int32 else np.int64
    indptr = (np.arange(n_bins + 1, dtype=index_dtype) * gk)
    indices = idx.reshape(-1).astype(index_dtype, copy=False)
    if use_float32:
        d32 = dists.astype(np.float32, copy=False)
        s32 = sigma_loc.astype(np.float32, copy=False)
        data = np.exp(-(d32 ** 2) / (2.0 * (s32[:, None] ** 2))).reshape(-1).astype(np.float32, copy=False)
    else:
        data = np.exp(-(dists ** 2) / (2.0 * (sigma_loc[:, None] ** 2))).reshape(-1)
    Wg = csr_matrix((data, indices, indptr), shape=(n_bins, n_bins))

    row_sum = Wg.sum(1).A1
    col_sum = Wg.sum(0).A1
    deg = 0.5 * (row_sum + col_sum) + 1e-12
    invdeg = 1.0 / deg
    if use_float32 and invdeg.dtype != np.float32:
        invdeg = invdeg.astype(np.float32, copy=False)
    Wt = None
    if int(graph_t) > 0:
        Wt = Wg.transpose(copy=False)
    graph = {"W": Wg, "Wt": Wt, "invdeg": invdeg}

    if graph_explicit is None:
        graph_mode = str(os.environ.get("SPIX_SMOOTH_GRAPH_EXPLICIT", "implicit")).strip().lower()
    elif isinstance(graph_explicit, bool):
        graph_mode = "explicit" if graph_explicit else "implicit"
    else:
        graph_mode = str(graph_explicit).strip().lower()
    if graph_explicit_max_mb is None:
        try:
            graph_explicit_max_mb = float(os.environ.get("SPIX_SMOOTH_GRAPH_EXPLICIT_MAX_MB", "1024"))
        except Exception:
            graph_explicit_max_mb = 1024.0
    else:
        graph_explicit_max_mb = float(graph_explicit_max_mb)

    use_explicit = False
    if int(graph_t) > 0:
        if graph_mode in {"1", "true", "yes", "explicit"}:
            use_explicit = True
        elif graph_mode in {"0", "false", "no", "implicit"}:
            use_explicit = False
        else:
            nnz = int(Wg.nnz)
            data_bytes = int(np.dtype(Wg.dtype).itemsize)
            index_bytes = int(np.dtype(Wg.indices.dtype).itemsize)
            est_bytes = (2 * nnz) * (data_bytes + index_bytes) + (int(n_bins) + 1) * index_bytes
            est_mb = est_bytes / (1024.0 ** 2)
            use_explicit = est_mb <= float(graph_explicit_max_mb)
    if use_explicit:
        if Wt is None:
            Wt = Wg.transpose(copy=False)
        S = (Wg + Wt).tocsr()
        S.data *= 0.5
        graph["S"] = S
        graph["W"] = None
        graph["Wt"] = None
    return graph


def _resolve_smoothing_tuning_inputs(
    adata,
    *,
    embedding: str,
    embedding_dims,
    obs_mask,
    coords,
    n_dims_eval: int,
    random_state: int,
):
    raw_embedding = np.asarray(adata.obsm[embedding])
    if raw_embedding.dtype != np.float32:
        raw_embedding = raw_embedding.astype(np.float32, copy=False)
    n_obs, n_dim_total = raw_embedding.shape

    if coords is None:
        coords_local = None
        try:
            if "spatial" in adata.obsm:
                coords_candidate = np.asarray(adata.obsm["spatial"], dtype=np.float32)
                if coords_candidate.shape[0] == n_obs:
                    coords_local = coords_candidate[:, :2]
        except Exception:
            coords_local = None
        if coords_local is None:
            try:
                tiles_df = adata.uns["tiles"]
                tiles_origin = tiles_df[tiles_df["origin"] == 1].copy()
                tiles_origin["barcode"] = tiles_origin["barcode"].astype(str)
                obs_idx = pd.Index(adata.obs_names.astype(str))
                tiles_aligned = tiles_origin.set_index("barcode").reindex(obs_idx)
                if tiles_aligned[["x", "y"]].isna().any().any():
                    missing = int(tiles_aligned[["x", "y"]].isna().any(axis=1).sum())
                    raise ValueError(f"Missing coordinates for {missing} observations when aligning tiles to obs.")
                coords_local = tiles_aligned[["x", "y"]].to_numpy(dtype=np.float32)
            except Exception as e:
                raise ValueError(
                    "Could not resolve coordinates from adata.obsm['spatial'] or adata.uns['tiles']."
                ) from e
        coords = coords_local
    else:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[0] != n_obs or coords.shape[1] < 2:
            raise ValueError("coords must be shaped (n_obs, 2+).")
        coords = coords[:, :2]

    if obs_mask is not None:
        if isinstance(obs_mask, (str, bytes)):
            key = str(obs_mask)
            if key not in adata.obs:
                raise KeyError(f"obs_mask key '{key}' not found in adata.obs.")
            mask = np.asarray(adata.obs[key]).astype(bool, copy=False)
        else:
            mask = np.asarray(obs_mask).astype(bool, copy=False)
        if mask.shape[0] != n_obs:
            raise ValueError("obs_mask must have length n_obs.")
        if not bool(mask.any()):
            raise ValueError("obs_mask selects 0 observations.")
        if not bool(mask.all()):
            raw_embedding = raw_embedding[mask]
            coords = coords[mask]
            n_obs = int(raw_embedding.shape[0])

    if embedding_dims is None:
        dims_pool = np.arange(n_dim_total, dtype=np.int32)
    else:
        dims_pool = np.asarray(list(embedding_dims), dtype=np.int32)
        if dims_pool.size == 0:
            raise ValueError("embedding_dims must be non-empty when provided.")
        if dims_pool.min() < 0 or dims_pool.max() >= n_dim_total:
            raise ValueError("embedding_dims contains out-of-range indices.")

    n_dims_eval = int(max(1, n_dims_eval))
    rng = np.random.default_rng(int(random_state))
    if dims_pool.size <= n_dims_eval:
        dims_eval = [int(x) for x in dims_pool.tolist()]
    else:
        dims_eval = sorted(int(x) for x in rng.choice(dims_pool, size=n_dims_eval, replace=False))
    return raw_embedding, coords, dims_eval


def _make_smoothing_candidates(
    *,
    methods,
    knn_k,
    knn_sigma,
    graph_k,
    graph_t,
    bilateral_k,
    bilateral_sigma_r,
    bilateral_t,
    gaussian_k,
    gaussian_sigma,
    gaussian_t,
    guided_k,
    guided_eps,
    guided_sigma,
    guided_t,
    heat_k,
    heat_sigma,
    heat_time,
    knn_k_grid=None,
    knn_sigma_grid=None,
    graph_k_grid=None,
    graph_t_grid=None,
    bilateral_k_grid=None,
    bilateral_sigma_r_grid=None,
    bilateral_t_grid=None,
    gaussian_k_grid=None,
    gaussian_sigma_grid=None,
    gaussian_t_grid=None,
    guided_k_grid=None,
    guided_eps_grid=None,
    guided_sigma_grid=None,
    guided_t_grid=None,
    heat_k_grid=None,
    heat_sigma_grid=None,
    heat_time_grid=None,
):
    methods = [str(m).strip().lower() for m in methods]
    grid_map = {
        "knn_k": [int(knn_k)] if knn_k_grid is None else sorted(set(int(max(1, x)) for x in knn_k_grid)),
        "knn_sigma": [float(knn_sigma)] if knn_sigma_grid is None else sorted(set(float(x) for x in knn_sigma_grid)),
        "graph_k": [int(graph_k)] if graph_k_grid is None else sorted(set(int(max(1, x)) for x in graph_k_grid)),
        "graph_t": [int(graph_t)] if graph_t_grid is None else sorted(set(int(max(1, x)) for x in graph_t_grid)),
        "bilateral_k": [int(bilateral_k)] if bilateral_k_grid is None else sorted(set(int(max(1, x)) for x in bilateral_k_grid)),
        "bilateral_sigma_r": [float(bilateral_sigma_r)] if bilateral_sigma_r_grid is None else sorted(set(float(x) for x in bilateral_sigma_r_grid)),
        "bilateral_t": [int(bilateral_t)] if bilateral_t_grid is None else sorted(set(int(max(1, x)) for x in bilateral_t_grid)),
        "gaussian_k": [int(knn_k if gaussian_k is None else gaussian_k)] if gaussian_k_grid is None else sorted(set(int(max(1, x)) for x in gaussian_k_grid)),
        "gaussian_sigma": [float(gaussian_sigma)] if gaussian_sigma_grid is None else sorted(set(float(x) for x in gaussian_sigma_grid)),
        "gaussian_t": [int(gaussian_t)] if gaussian_t_grid is None else sorted(set(int(max(1, x)) for x in gaussian_t_grid)),
        "guided_k": [int(knn_k if guided_k is None else guided_k)] if guided_k_grid is None else sorted(set(int(max(1, x)) for x in guided_k_grid)),
        "guided_eps": [float(guided_eps)] if guided_eps_grid is None else sorted(set(float(x) for x in guided_eps_grid)),
        "guided_sigma": [guided_sigma] if guided_sigma_grid is None else list(guided_sigma_grid),
        "guided_t": [int(guided_t)] if guided_t_grid is None else sorted(set(int(max(1, x)) for x in guided_t_grid)),
        "heat_k": [int((gaussian_k if gaussian_k is not None else knn_k) if heat_k is None else heat_k)] if heat_k_grid is None else sorted(set(int(max(1, x)) for x in heat_k_grid)),
        "heat_sigma": [float((gaussian_sigma if heat_sigma is None else heat_sigma))] if heat_sigma_grid is None else sorted(set(float(x) for x in heat_sigma_grid)),
        "heat_time": [float(heat_time)] if heat_time_grid is None else sorted(set(float(x) for x in heat_time_grid)),
    }
    active_keys = []
    if "knn" in methods:
        active_keys += ["knn_k", "knn_sigma"]
    if "graph" in methods:
        active_keys += ["graph_k", "graph_t"]
    if "bilateral" in methods:
        active_keys += ["bilateral_k", "bilateral_sigma_r", "bilateral_t"]
    if "gaussian" in methods:
        active_keys += ["gaussian_k", "gaussian_sigma", "gaussian_t"]
    if "guided" in methods:
        active_keys += ["guided_k", "guided_eps", "guided_sigma", "guided_t"]
    if "heat" in methods:
        active_keys += ["heat_k", "heat_sigma", "heat_time"]
    active_keys = list(dict.fromkeys(active_keys))
    candidates = []
    for values in product(*[grid_map[k] for k in active_keys]):
        row = dict(zip(active_keys, values))
        for key, vals in grid_map.items():
            row.setdefault(key, vals[0])
        candidates.append(row)
    return candidates


def _ratio_preserve_score(x: float, eps: float = 1e-12) -> float:
    x = float(max(eps, x))
    return float(np.exp(-abs(np.log(x))))


def _stabilized_edge_ratios(
    d0: np.ndarray,
    d1: np.ndarray,
    *,
    q_low: float,
    q_high: float,
):
    d0 = np.asarray(d0, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    thr_lo = float(np.quantile(d0, q_low))
    thr_hi = float(np.quantile(d0, q_high))
    low_mask = d0 <= thr_lo
    high_mask = d0 >= thr_hi
    d0_low = float(d0[low_mask].mean()) if bool(low_mask.any()) else float(d0.mean())
    d1_low = float(d1[low_mask].mean()) if bool(low_mask.any()) else float(d1.mean())
    d0_hi = float(d0[high_mask].mean()) if bool(high_mask.any()) else float(d0.mean())
    d1_hi = float(d1[high_mask].mean()) if bool(high_mask.any()) else float(d1.mean())
    d0_all = float(d0.mean())
    d1_all = float(d1.mean())
    eps = 1e-12
    # Very-flat edges make raw ratios explode and break selection for strong
    # smoothers such as graph diffusion. Use robust denominator floors so we
    # compare against a meaningful local scale instead of near-zero noise.
    low_floor = max(eps, thr_lo, 0.1 * d0_all)
    high_floor = max(eps, thr_hi, 0.25 * d0_all)
    all_floor = max(eps, 0.25 * d0_all)
    r_low = d1_low / max(low_floor, d0_low)
    r_hi = d1_hi / max(high_floor, d0_hi)
    r_all = d1_all / max(all_floor, d0_all)
    return {
        "ratio_low": float(r_low),
        "ratio_high": float(r_hi),
        "ratio_all": float(r_all),
    }


def _finalize_smoothing_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["smooth_gain"] = np.clip(1.0 - df["ratio_low"].astype(float), 0.0, None)
    df["edge_preserve_ratio"] = df["ratio_high"].astype(float)
    df["global_preserve_ratio"] = df["ratio_all"].astype(float)
    df["edge_preserve"] = df["edge_preserve_ratio"].map(_ratio_preserve_score)
    df["global_preserve"] = df["global_preserve_ratio"].map(_ratio_preserve_score)
    df["tradeoff_product"] = (
        df["smooth_gain"].astype(float)
        * df["edge_preserve"].astype(float)
        * df["global_preserve"].astype(float)
    )
    return df


def _select_best_smoothing_candidate(
    df: pd.DataFrame,
    *,
    selection_mode: str,
    select_gain_fraction_val,
    min_edge_val,
    prefer_smooth_gain_within_keep: bool = False,
):
    if selection_mode == "score":
        best_id = int(df.sort_values(["score"], ascending=False, kind="mergesort").iloc[0]["candidate_id"])
    elif selection_mode == "product":
        best_id = int(df.sort_values(["tradeoff_product"], ascending=False, kind="mergesort").iloc[0]["candidate_id"])
    elif selection_mode == "max_gain":
        keep = np.ones(df.shape[0], dtype=bool)
        if min_edge_val is not None:
            keep = keep & (df["edge_preserve"] >= min_edge_val)
        kept = df[keep]
        if kept.shape[0] == 0:
            best_id = int(df.sort_values(["tradeoff_product"], ascending=False, kind="mergesort").iloc[0]["candidate_id"])
        else:
            best_id = int(
                kept.sort_values(
                    ["smooth_gain", "edge_preserve", "global_preserve", "score"],
                    ascending=[False, False, False, False],
                    kind="mergesort",
                ).iloc[0]["candidate_id"]
            )
    else:
        max_gain = float(df["smooth_gain"].max())
        gain_thr = float(select_gain_fraction_val) * max_gain
        keep = df["smooth_gain"] >= gain_thr
        if min_edge_val is not None:
            keep = keep & (df["edge_preserve"] >= min_edge_val)
        kept = df[keep]
        if kept.shape[0] == 0:
            best_id = int(df.sort_values(["tradeoff_product"], ascending=False, kind="mergesort").iloc[0]["candidate_id"])
        else:
            sort_cols = ["edge_preserve", "global_preserve", "smooth_gain", "score"]
            if prefer_smooth_gain_within_keep:
                sort_cols = ["tradeoff_product", "smooth_gain", "edge_preserve", "global_preserve"]
            best_id = int(
                kept.sort_values(
                    sort_cols,
                    ascending=[False, False, False, False],
                    kind="mergesort",
                ).iloc[0]["candidate_id"]
            )
    return best_id


def _build_smoothing_recommendation(best_row: dict, methods: list[str]) -> dict:
    methods = [str(m).strip().lower() for m in methods]
    keys = []
    if "knn" in methods:
        keys += ["knn_k", "knn_sigma"]
    if "graph" in methods:
        keys += ["graph_k", "graph_t"]
    if "bilateral" in methods:
        keys += ["bilateral_k", "bilateral_sigma_r", "bilateral_t"]
    if "gaussian" in methods:
        keys += ["gaussian_k", "gaussian_sigma", "gaussian_t"]
    if "guided" in methods:
        keys += ["guided_k", "guided_eps", "guided_sigma", "guided_t"]
    if "heat" in methods:
        keys += ["heat_k", "heat_sigma", "heat_time"]
    params = {k: best_row[k] for k in keys if k in best_row and pd.notna(best_row[k])}
    return {
        "methods": list(methods),
        "candidate_id": int(best_row["candidate_id"]),
        "params": params,
        "summary": ", ".join(f"{k}={v}" for k, v in params.items()),
        "reason": (
            f"Selected candidate {int(best_row['candidate_id'])} with "
            f"smooth_gain={float(best_row.get('smooth_gain', float('nan'))):.4g}, "
            f"edge_preserve={float(best_row.get('edge_preserve', float('nan'))):.4g}, "
            f"global_preserve={float(best_row.get('global_preserve', float('nan'))):.4g}, "
            f"score={float(best_row.get('score', float('nan'))):.4g}."
        ),
    }


def _normalize_smoothing_approx_mode(approx_mode):
    if approx_mode is None:
        return None
    if isinstance(approx_mode, bool):
        return "grid" if approx_mode else None
    mode = str(approx_mode).strip().lower()
    if mode in {"", "none", "off", "false", "0", "full", "exact"}:
        return None
    if mode in {"approx", "grid", "coarse"}:
        return "grid"
    raise ValueError("approx_mode must be one of {None,'grid','approx','full','exact',True,False}.")


def evaluate_smoothing_sweep(
    adata,
    *,
    embedding: str = "X_embedding",
    methods=("bilateral",),
    embedding_dims=None,
    obs_mask=None,
    coords=None,
    approx_mode: str | bool | None = "grid",
    approx_target_n: int = 20_000,
    approx_bin=None,
    approx_max_bins=None,
    knn_k: int = 35,
    knn_sigma: float = 2.0,
    knn_chunk: int = 10_000,
    graph_k: int = 30,
    graph_t: int = 2,
    graph_tol=None,
    graph_check_every: int = 1,
    graph_explicit="implicit",
    graph_explicit_max_mb=None,
    bilateral_k: int = 30,
    bilateral_sigma_r: float = 0.3,
    bilateral_t: int = 3,
    gaussian_sigma: float = 1.0,
    gaussian_k=None,
    gaussian_t: int = 1,
    guided_k=None,
    guided_eps: float = 1e-3,
    guided_sigma=None,
    guided_t: int = 1,
    heat_time: float = 1.0,
    heat_k=None,
    heat_sigma=None,
    heat_method: str = "krylov",
    heat_tol: float = 1e-4,
    heat_min_k: int = 0,
    heat_max_k=None,
    knn_k_grid=None,
    knn_sigma_grid=None,
    graph_k_grid=None,
    graph_t_grid=None,
    bilateral_k_grid=None,
    bilateral_sigma_r_grid=None,
    bilateral_t_grid=None,
    gaussian_k_grid=None,
    gaussian_sigma_grid=None,
    gaussian_t_grid=None,
    guided_k_grid=None,
    guided_eps_grid=None,
    guided_sigma_grid=None,
    guided_t_grid=None,
    heat_k_grid=None,
    heat_sigma_grid=None,
    heat_time_grid=None,
    n_dims_eval: int = 3,
    random_state: int = 0,
    score_sample_edges: int = 200_000,
    score_q_low: float = 0.2,
    score_q_high: float = 0.8,
    selection: str = "gain_then_edge",
    select_gain_fraction=None,
    select_min_edge_preserve=None,
    score_w_edge: float = 1.0,
    score_w_all: float = 0.2,
    rescale_mode: str = "final",
    use_float32=None,
    fast_float32: bool = True,
    n_jobs=None,
    candidate_jobs=None,
    candidate_backend: str = "threads",
    verbose: bool = True,
):
    log = logging.getLogger(__name__)
    if not verbose:
        prev_level = log.level
        log.setLevel(logging.WARNING)
    try:
        if isinstance(methods, (str, bytes)):
            methods = [methods]
        methods = [str(m).strip().lower() for m in methods]
        valid_methods = {"knn", "graph", "bilateral", "gaussian", "guided", "heat"}
        if not methods or any(m not in valid_methods for m in methods):
            raise ValueError(f"methods must be a non-empty subset of {sorted(valid_methods)}.")
        if score_q_low <= 0 or score_q_high >= 1 or score_q_low >= score_q_high:
            raise ValueError("score_q_low/high must satisfy 0 < low < high < 1.")

        selection_mode = str(selection).strip().lower()
        if selection_mode not in {"gain_then_edge", "product", "score", "max_gain"}:
            raise ValueError("selection must be one of {'gain_then_edge','product','score','max_gain'}.")
        bilateral_only = set(methods) == {"bilateral"}
        aggressive_bilateral_default = bool(
            bilateral_only
            and selection_mode == "gain_then_edge"
            and select_gain_fraction is None
            and select_min_edge_preserve is None
        )
        select_gain_fraction_val = None
        if select_gain_fraction is not None:
            select_gain_fraction_val = float(select_gain_fraction)
            if not np.isfinite(select_gain_fraction_val) or not (0 < select_gain_fraction_val <= 1.0):
                raise ValueError("select_gain_fraction must be in (0,1].")
        min_edge_val = None if select_min_edge_preserve is None else float(select_min_edge_preserve)
        if min_edge_val is not None and (not np.isfinite(min_edge_val) or not (0 < min_edge_val <= 1.0)):
            raise ValueError("select_min_edge_preserve must be in (0,1].")

        raw_embedding, coords_arr, dims_eval = _resolve_smoothing_tuning_inputs(
            adata,
            embedding=embedding,
            embedding_dims=embedding_dims,
            obs_mask=obs_mask,
            coords=coords,
            n_dims_eval=n_dims_eval,
            random_state=random_state,
        )
        n_obs = int(raw_embedding.shape[0])

        if n_jobs is None or int(n_jobs) == -1:
            n_jobs = multiprocessing.cpu_count()
        n_jobs = int(n_jobs)
        neighbor_workers = max(1, n_jobs)

        if use_float32 is None:
            use_float32 = str(os.environ.get("SPIX_SMOOTH_FLOAT32", "0")).strip().lower() in {"1", "true", "yes"}
        else:
            use_float32 = bool(use_float32)
        if fast_float32:
            use_float32 = True

        approx_mode_val = _normalize_smoothing_approx_mode(approx_mode)

        if approx_max_bins is None:
            try:
                approx_max_bins = int(os.environ.get("SPIX_SMOOTH_APPROX_MAX_BINS", "500000"))
            except Exception:
                approx_max_bins = 500000

        if approx_mode_val == "grid":
            approx_info = _precompute_approx_grid(
                coords_arr,
                graph_k=int(graph_k),
                graph_t=0,
                graph_explicit=graph_explicit,
                graph_explicit_max_mb=graph_explicit_max_mb,
                use_float32=use_float32,
                target_n=int(approx_target_n),
                bin_size=approx_bin,
                max_bins=approx_max_bins,
                build_graph=False,
                neighbor_workers=neighbor_workers,
            )
            bin_id = approx_info["bin_id"]
            counts = approx_info["counts"]
            centers = approx_info["centers"]
            n_bins = int(approx_info["n_bins"])

            E_eval = raw_embedding[:, dims_eval]
            sum_dtype = np.float32 if fast_float32 else np.float64
            E_bins = _aggregate_by_bins(E_eval, bin_id, counts, sum_dtype=sum_dtype).astype(np.float32, copy=False)
            approx_meta = {
                "mode": "grid",
                "n_bins": int(n_bins),
                "bin_size": float(approx_info.get("bin_size")),
            }
            if verbose:
                log.info(
                    "Smoothing sweep approx mode: grid | n_obs=%d -> n_bins=%d | bin_size=%.6g",
                    int(n_obs),
                    int(n_bins),
                    float(approx_info.get("bin_size")),
                )
        else:
            centers = np.asarray(coords_arr[:, :2], dtype=np.float32, copy=False)
            n_bins = int(centers.shape[0])
            E_bins = np.asarray(raw_embedding[:, dims_eval], dtype=np.float32, copy=False)
            approx_meta = {
                "mode": "full",
                "n_bins": int(n_bins),
                "bin_size": None,
            }
            if verbose:
                log.info(
                    "Smoothing sweep approx mode: full | n_obs=%d | no coarse binning",
                    int(n_obs),
                )
        tuning_rescale_mode = str(rescale_mode).strip().lower()
        if tuning_rescale_mode in {"final", "per_step"}:
            mn0 = E_bins.min(axis=0, keepdims=True)
            mx0 = E_bins.max(axis=0, keepdims=True)
            denom0 = mx0 - mn0
            denom0[denom0 < 1e-12] = 1e-12
            E_bins = (E_bins - mn0) / denom0

        candidates = _make_smoothing_candidates(
            methods=methods,
            knn_k=knn_k,
            knn_sigma=knn_sigma,
            graph_k=graph_k,
            graph_t=graph_t,
            bilateral_k=bilateral_k,
            bilateral_sigma_r=bilateral_sigma_r,
            bilateral_t=bilateral_t,
            gaussian_k=gaussian_k,
            gaussian_sigma=gaussian_sigma,
            gaussian_t=gaussian_t,
            guided_k=guided_k,
            guided_eps=guided_eps,
            guided_sigma=guided_sigma,
            guided_t=guided_t,
            heat_k=heat_k,
            heat_sigma=heat_sigma,
            heat_time=heat_time,
            knn_k_grid=knn_k_grid,
            knn_sigma_grid=knn_sigma_grid,
            graph_k_grid=graph_k_grid,
            graph_t_grid=graph_t_grid,
            bilateral_k_grid=bilateral_k_grid,
            bilateral_sigma_r_grid=bilateral_sigma_r_grid,
            bilateral_t_grid=bilateral_t_grid,
            gaussian_k_grid=gaussian_k_grid,
            gaussian_sigma_grid=gaussian_sigma_grid,
            gaussian_t_grid=gaussian_t_grid,
            guided_k_grid=guided_k_grid,
            guided_eps_grid=guided_eps_grid,
            guided_sigma_grid=guided_sigma_grid,
            guided_t_grid=guided_t_grid,
            heat_k_grid=heat_k_grid,
            heat_sigma_grid=heat_sigma_grid,
            heat_time_grid=heat_time_grid,
        )
        if len(candidates) == 0:
            raise RuntimeError("No smoothing candidates generated.")

        k_needed = []
        if "knn" in methods:
            k_needed.append(max(int(c["knn_k"]) for c in candidates))
        if "graph" in methods:
            k_needed.append(max(int(c["graph_k"]) for c in candidates) + 1)
        if "bilateral" in methods:
            k_needed.append(max(int(c["bilateral_k"]) for c in candidates) + 1)
        if "gaussian" in methods:
            k_needed.append(max(int(c["gaussian_k"]) for c in candidates))
        if "guided" in methods:
            k_needed.append(max(int(c["guided_k"]) for c in candidates))
        if "heat" in methods:
            k_needed.append(max(int(c["heat_k"]) for c in candidates))
        k_query = int(min(max(1, max(k_needed, default=1)), n_bins))
        tree_bins = cKDTree(centers, balanced_tree=True, compact_nodes=True)
        d_all, idx_all = tree_bins.query(centers, k=k_query, workers=neighbor_workers)
        if k_query == 1:
            d_all = d_all[:, None]
            idx_all = idx_all[:, None]
        d_all = d_all.astype(np.float32, copy=False)
        idx_all = idx_all.astype(np.int32, copy=False)

        if idx_all.shape[1] >= 1:
            is_self = (idx_all[:, 0] == np.arange(n_bins, dtype=idx_all.dtype))
            includes_self = bool(is_self.mean() >= 0.9)
        else:
            includes_self = False
        start_pos = 1 if includes_self and idx_all.shape[1] > 1 else 0

        rng = np.random.default_rng(int(random_state))
        n_edges = int(max(1, score_sample_edges))
        src = rng.integers(0, n_bins, size=n_edges, dtype=np.int32)
        u = rng.random(size=n_edges).astype(np.float32, copy=False)

        if selection_mode == "gain_then_edge" and select_gain_fraction_val is None:
            # Bilateral-only tuning tends to prefer very small sigma_r because
            # edge preservation dominates early. Apply only a mild default bias
            # toward stronger smoothers while keeping edge preservation central.
            select_gain_fraction_val = 0.85 if aggressive_bilateral_default else 0.8

        def _score_candidate(E0_bins: np.ndarray, E1_bins: np.ndarray, *, k_eff: int) -> dict:
            k_eff = int(max(1, min(k_eff, idx_all.shape[1] - (0 if not includes_self else 0))))
            span = int(max(1, min(k_eff + (1 if includes_self else 0), idx_all.shape[1]) - start_pos))
            pos = start_pos + (u * span).astype(np.int32)
            dst = idx_all[src, pos]
            scores = []
            ratio_low_all = []
            ratio_high_all = []
            ratio_all_all = []
            for j in range(E0_bins.shape[1]):
                d0 = np.abs(E0_bins[src, j] - E0_bins[dst, j])
                d1 = np.abs(E1_bins[src, j] - E1_bins[dst, j])
                ratios = _stabilized_edge_ratios(
                    d0,
                    d1,
                    q_low=score_q_low,
                    q_high=score_q_high,
                )
                r_low = ratios["ratio_low"]
                r_hi = ratios["ratio_high"]
                r_all = ratios["ratio_all"]
                eps = 1e-12
                score = (1.0 - r_low) - float(score_w_edge) * abs(float(np.log(max(eps, r_hi)))) - float(score_w_all) * abs(float(np.log(max(eps, r_all))))
                scores.append(float(score))
                ratio_low_all.append(float(r_low))
                ratio_high_all.append(float(r_hi))
                ratio_all_all.append(float(r_all))
            return {
                "score": float(np.mean(scores)) if scores else float("nan"),
                "ratio_low": float(np.mean(ratio_low_all)) if ratio_low_all else float("nan"),
                "ratio_high": float(np.mean(ratio_high_all)) if ratio_high_all else float("nan"),
                "ratio_all": float(np.mean(ratio_all_all)) if ratio_all_all else float("nan"),
            }

        if candidate_jobs is None:
            candidate_jobs = n_jobs
        candidate_jobs = int(max(1, min(int(candidate_jobs), len(candidates))))
        candidate_backend = str(candidate_backend).strip().lower()
        if candidate_backend not in {"threads", "processes"}:
            raise ValueError("candidate_backend must be one of {'threads','processes'}.")
        if candidate_backend == "processes":
            if verbose:
                log.warning(
                    "evaluate_smoothing_sweep: process-based candidate parallelism is not enabled for this path; falling back to threads."
                )
            candidate_backend = "threads"

        def _evaluate_one(item):
            cand_id, cand = item
            precomputed = {}
            if "knn" in methods:
                k = int(max(1, min(int(cand["knn_k"]), idx_all.shape[1])))
                precomputed["knn"] = {
                    "dists": d_all[:, :k].astype(np.float32, copy=False),
                    "idx": idx_all[:, :k].astype(np.int32, copy=False),
                }
            if "graph" in methods:
                k = int(max(1, min(int(cand["graph_k"]) + 1, idx_all.shape[1])))
                d = d_all[:, 1:k].astype(np.float32, copy=False) if includes_self and k > 1 else d_all[:, :max(1, k - start_pos)].astype(np.float32, copy=False)
                ix = idx_all[:, 1:k].astype(np.int32, copy=False) if includes_self and k > 1 else idx_all[:, :max(1, k - start_pos)].astype(np.int32, copy=False)
                precomputed["graph"] = _build_graph_precomputed_from_neighbors(
                    d,
                    ix,
                    graph_t=int(cand["graph_t"]),
                    graph_explicit=graph_explicit,
                    graph_explicit_max_mb=graph_explicit_max_mb,
                    use_float32=use_float32,
                )
            if "bilateral" in methods:
                k = int(max(1, min(int(cand["bilateral_k"]) + 1, idx_all.shape[1])))
                d = d_all[:, :k].astype(np.float32, copy=False)
                ix = idx_all[:, :k].astype(np.int32, copy=False)
                if k > 1:
                    sigma_s = (np.median(d[:, start_pos:], axis=1).astype(np.float32) + 1e-9).astype(np.float32)
                else:
                    sigma_s = np.full(d.shape[0], 1.0, dtype=np.float32)
                precomputed["bilateral"] = {"dists": d, "idx": ix, "sigma_s": sigma_s}
            if "gaussian" in methods:
                k = int(max(1, min(int(cand["gaussian_k"]), idx_all.shape[1])))
                precomputed["gaussian"] = {
                    "dists": d_all[:, :k].astype(np.float32, copy=False),
                    "idx": idx_all[:, :k].astype(np.int32, copy=False),
                }
            if "guided" in methods:
                k = int(max(1, min(int(cand["guided_k"]), idx_all.shape[1])))
                entry = {"idx": idx_all[:, :k].astype(np.int32, copy=False)}
                if cand["guided_sigma"] is not None:
                    entry["dists"] = d_all[:, :k].astype(np.float32, copy=False)
                precomputed["guided"] = entry
            if "heat" in methods:
                k = int(max(1, min(int(cand["heat_k"]), idx_all.shape[1])))
                heat_entry = {
                    "W": _build_row_stochastic_csr(
                        d_all[:, :k].astype(np.float32, copy=False),
                        idx_all[:, :k].astype(np.int32, copy=False),
                        sigma=float(cand["heat_sigma"]),
                        fast_float32=fast_float32,
                        sort_indices=False,
                    )
                }
                precomputed["heat"] = heat_entry

            E1_cols = []
            for j in range(E_bins.shape[1]):
                out_j = _process_dimension(
                    E_bins[:, j].astype(np.float32, copy=False),
                    methods,
                    precomputed,
                    float(cand["knn_sigma"]),
                    int(knn_chunk),
                    float(cand["bilateral_sigma_r"]),
                    int(cand["bilateral_t"]),
                    int(cand["graph_t"]),
                    graph_tol,
                    int(graph_check_every),
                    int(cand["gaussian_t"]),
                    float(cand["gaussian_sigma"]),
                    float(cand["guided_eps"]),
                    cand["guided_sigma"],
                    int(cand["guided_t"]),
                    float(cand["heat_time"]),
                    str(heat_method),
                    float(heat_tol),
                    int(heat_min_k),
                    heat_max_k,
                    "none" if tuning_rescale_mode == "final" else rescale_mode,
                    fast_float32,
                ).astype(np.float32, copy=False)
                E1_cols.append(out_j)
            E1_bins = np.stack(E1_cols, axis=1).astype(np.float32, copy=False)

            score = _score_candidate(E_bins, E1_bins, k_eff=max(1, int(cand.get("graph_k", cand.get("bilateral_k", cand.get("gaussian_k", cand.get("knn_k", 1)))))))
            return {"candidate_id": int(cand_id), **cand, **score}

        candidate_items = list(enumerate(candidates))
        if verbose:
            log.info(
                "Smoothing sweep candidate evaluation: candidates=%d | backend=%s | jobs=%d",
                len(candidate_items),
                candidate_backend,
                candidate_jobs,
            )
        results = []
        if candidate_jobs <= 1 or len(candidate_items) <= 1:
            for item in candidate_items:
                results.append(_evaluate_one(item))
        elif candidate_backend == "threads":
            with ThreadPoolExecutor(max_workers=candidate_jobs) as executor:
                future_to_item = {executor.submit(_evaluate_one, item): item for item in candidate_items}
                for future in as_completed(future_to_item):
                    results.append(future.result())

        df = _finalize_smoothing_scores(pd.DataFrame(results))
        best_id = _select_best_smoothing_candidate(
            df,
            selection_mode=selection_mode,
            select_gain_fraction_val=select_gain_fraction_val,
            min_edge_val=min_edge_val,
            prefer_smooth_gain_within_keep=aggressive_bilateral_default,
        )

        df["is_best"] = df["candidate_id"] == int(best_id)
        df = df.sort_values(["is_best", "tradeoff_product"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
        best_row = df.iloc[0].to_dict()

        best = {
            "methods": list(methods),
            "embedding": embedding,
            "rescale_mode": str(rescale_mode),
        }
        active_param_keys = list(_build_smoothing_recommendation(best_row, methods)["params"].keys())
        for key in active_param_keys:
            if key in best_row:
                best[key] = best_row[key]
        if embedding_dims is not None:
            best["embedding_dims"] = list(embedding_dims)
        if isinstance(obs_mask, (str, bytes)):
            best["obs_mask"] = str(obs_mask)

        return {
            "best": best,
            "candidates": df,
            "dims_eval": dims_eval,
            "approx": approx_meta,
            "selection_result": {
                "selected_candidate_id": int(best_id),
                "selected": best_row,
                "selection_mode": selection_mode,
                "select_gain_fraction": select_gain_fraction_val,
                "select_min_edge_preserve": min_edge_val,
                "aggressive_bilateral_default": bool(aggressive_bilateral_default),
            },
            "recommendation": _build_smoothing_recommendation(best_row, methods),
        }
    finally:
        if not verbose:
            log.setLevel(prev_level)


def plot_smoothing_sweep(
    selection_result,
    *,
    param_x: str | None = None,
    show: bool = True,
    figsize=(12, 8),
):
    if not isinstance(selection_result, dict) or "candidates" not in selection_result:
        raise ValueError("selection_result must be a dict returned by evaluate_smoothing_sweep or tune_smooth_image_params.")
    df = selection_result["candidates"]
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        raise ValueError("selection_result['candidates'] must be a non-empty DataFrame.")

    candidate_cols = [
        "knn_k", "knn_sigma", "graph_k", "graph_t",
        "bilateral_k", "bilateral_sigma_r", "bilateral_t",
        "gaussian_k", "gaussian_sigma", "gaussian_t",
        "guided_k", "guided_eps", "guided_sigma", "guided_t",
        "heat_k", "heat_sigma", "heat_time",
    ]
    if param_x is None:
        varying = [c for c in candidate_cols if c in df.columns and pd.Series(df[c]).nunique(dropna=False) > 1]
        if not varying:
            raise ValueError("No varying smoothing parameter found to plot on x-axis.")
        param_x = varying[0]
    if param_x not in df.columns:
        raise ValueError(f"param_x '{param_x}' not present in candidates DataFrame.")

    candidate_cols = [
        "knn_k", "knn_sigma", "graph_k", "graph_t",
        "bilateral_k", "bilateral_sigma_r", "bilateral_t",
        "gaussian_k", "gaussian_sigma", "gaussian_t",
        "guided_k", "guided_eps", "guided_sigma", "guided_t",
        "heat_k", "heat_sigma", "heat_time",
    ]
    varying = [c for c in candidate_cols if c in df.columns and pd.Series(df[c]).nunique(dropna=False) > 1]
    multi_param = len(varying) > 1
    plot_df = df.sort_values(param_x, kind="mergesort").reset_index(drop=True)
    selected_mask = plot_df.get("is_best", pd.Series(False, index=plot_df.index)).astype(bool)
    selection_meta = selection_result.get("selection_result", {})
    recommendation = selection_result.get("recommendation", {})
    metrics = [
        ("score", "score"),
        ("smooth_gain", "smooth_gain"),
        ("edge_preserve", "edge_preserve"),
        ("tradeoff_product", "tradeoff_product"),
        ("ratio_low", "ratio_low"),
        ("ratio_high", "ratio_high"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    x = plot_df[param_x].to_numpy()
    for ax, (col, title) in zip(axes, metrics):
        if col not in plot_df.columns:
            ax.axis("off")
            continue
        y = plot_df[col].to_numpy(dtype=float)
        if multi_param:
            ax.scatter(x, y, s=24, alpha=0.7)
            agg = (
                plot_df[[param_x, col]]
                .groupby(param_x, as_index=False)
                .median()
                .sort_values(param_x, kind="mergesort")
            )
            ax.plot(
                agg[param_x].to_numpy(dtype=float),
                agg[col].to_numpy(dtype=float),
                linewidth=1.5,
                color="black",
                alpha=0.8,
            )
        else:
            ax.plot(x, y, marker="o", linewidth=1.5)
        if np.any(selected_mask):
            ax.scatter(
                plot_df.loc[selected_mask, param_x].to_numpy(dtype=float),
                plot_df.loc[selected_mask, col].to_numpy(dtype=float),
                c="black",
                s=40,
                zorder=5,
            )
        ax.set_title(title)
        ax.set_xlabel(param_x)
    methods = selection_result.get("best", {}).get("methods", None)
    fig.suptitle(
        (
            f"Smoothing Sweep | methods={methods} | aggregated over {', '.join(v for v in varying if v != param_x)}"
            if multi_param and methods is not None
            else (f"Smoothing Sweep | methods={methods}" if methods is not None else "Smoothing Sweep")
        ),
        fontsize=14,
    )
    if recommendation:
        summary = recommendation.get("summary", "")
        sel_mode = selection_meta.get("selection_mode", None)
        gain_frac = selection_meta.get("select_gain_fraction", None)
        msg = f"selected: {summary}" if summary else f"selected candidate={recommendation.get('candidate_id')}"
        if sel_mode == "gain_then_edge":
            if gain_frac is not None:
                msg += f"\nrule: edge-preserve best among gain >= {float(gain_frac):.2f} x max_gain"
            else:
                msg += "\nrule: gain_then_edge"
        elif sel_mode:
            msg += f"\nrule: {sel_mode}"
        fig.text(
            0.99,
            0.01,
            msg,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
    fig.tight_layout()
    if show:
        plt.show()
    return {"fig": fig, "axes": axes, "param_x": param_x}


def plot_smoothing_grid(
    selection_result,
    *,
    x: str,
    y: str,
    facet: str | None = None,
    value: str = "tradeoff_product",
    cmap: str = "viridis",
    show: bool = True,
    figsize=None,
):
    if not isinstance(selection_result, dict) or "candidates" not in selection_result:
        raise ValueError("selection_result must be a dict returned by evaluate_smoothing_sweep or tune_smooth_image_params.")
    df = selection_result["candidates"]
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        raise ValueError("selection_result['candidates'] must be a non-empty DataFrame.")
    for col in [x, y, value]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in candidates DataFrame.")
    if facet is not None and facet not in df.columns:
        raise ValueError(f"Facet column '{facet}' not found in candidates DataFrame.")

    if facet is None:
        facet_values = [None]
    else:
        facet_values = list(pd.unique(df[facet]))
    n_panels = len(facet_values)
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (4.5 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    selected = selection_result.get("selection_result", {}).get("selected", None)
    im_ref = None
    for i, facet_val in enumerate(facet_values):
        ax = axes_flat[i]
        sub = df if facet is None else df[df[facet] == facet_val]
        pivot = pd.pivot_table(
            sub,
            index=y,
            columns=x,
            values=value,
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap=cmap)
        im_ref = im
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([str(v) for v in pivot.index])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        title = f"{facet}={facet_val}" if facet is not None else value
        ax.set_title(title)

        if selected is not None:
            sel_ok = True
            if facet is not None:
                sel_ok = selected.get(facet) == facet_val
            if sel_ok and selected.get(x) in pivot.columns and selected.get(y) in pivot.index:
                x_idx = list(pivot.columns).index(selected.get(x))
                y_idx = list(pivot.index).index(selected.get(y))
                ax.scatter([x_idx], [y_idx], s=120, facecolors="none", edgecolors="white", linewidths=2.0)
                ax.scatter([x_idx], [y_idx], s=40, c="black", zorder=5)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")
    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes_flat[:n_panels], shrink=0.9, label=value)
    rec = selection_result.get("recommendation", None)
    if rec is not None:
        fig.suptitle(f"Smoothing Grid | best={rec['params']}", fontsize=13)
    else:
        fig.suptitle("Smoothing Grid", fontsize=13)
    fig.tight_layout()
    if show:
        plt.show()
    return {"fig": fig, "axes": axes, "x": x, "y": y, "facet": facet, "value": value}


def fast_select_smoothing(
    adata,
    *,
    embedding: str = "X_embedding",
    methods=("bilateral",),
    embedding_dims=None,
    obs_mask=None,
    coords=None,
    approx_mode: str | bool | None = "grid",
    approx_target_n: int = 20_000,
    approx_bin=None,
    approx_max_bins=None,
    n_dims_eval: int = 3,
    random_state: int = 0,
    score_sample_edges: int = 100_000,
    selection: str = "gain_then_edge",
    n_jobs=None,
    candidate_jobs=None,
    candidate_backend: str = "threads",
    verbose: bool = True,
    **kwargs,
):
    return evaluate_smoothing_sweep(
        adata,
        embedding=embedding,
        methods=methods,
        embedding_dims=embedding_dims,
        obs_mask=obs_mask,
        coords=coords,
        approx_mode=approx_mode,
        approx_target_n=approx_target_n,
        approx_bin=approx_bin,
        approx_max_bins=approx_max_bins,
        n_dims_eval=n_dims_eval,
        random_state=random_state,
        score_sample_edges=score_sample_edges,
        selection=selection,
        n_jobs=n_jobs,
        candidate_jobs=candidate_jobs,
        candidate_backend=candidate_backend,
        verbose=verbose,
        **kwargs,
    )


def _process_dimension(
    E_vec,
    methods,
    precomputed,
    knn_sigma,
    knn_chunk,
    bilateral_sigma_r,
    bilateral_t,
    graph_t,
    graph_tol,
    graph_check_every,
    gaussian_t,
    gaussian_sigma,
    guided_eps,
    guided_sigma,
    guided_t,
    heat_time,
    heat_method,
    heat_tol,
    heat_min_k,
    heat_max_k,
    rescale_mode,
    fast_float32: bool = False,
):
    """
    Process a single embedding dimension: apply sequential smoothing
    methods (including optional Gaussian filtering) and min-max rescale.
    Returns the smoothed 1D array.
    """
    # Resolve precomputed: from arg or per-process global
    if precomputed is None:
        global _SMOOTH_PRECOMPUTED
        precomputed = _SMOOTH_PRECOMPUTED
    if precomputed is None:
        raise RuntimeError("Precomputed data not initialized in worker.")

    graph_t_int = int(graph_t)
    graph_tol_val = None if graph_tol is None else float(graph_tol)
    if graph_tol_val is not None and graph_tol_val <= 0:
        graph_tol_val = None
    graph_check_every_int = int(max(1, graph_check_every))
    gaussian_t_int = int(gaussian_t)
    guided_t_int = int(guided_t)
    heat_time_val = float(heat_time)
    heat_method_val = str(heat_method).strip().lower()
    heat_tol_val = float(heat_tol)
    heat_min_k_int = int(max(0, heat_min_k))
    heat_max_k_val = None if heat_max_k is None else int(max(0, heat_max_k))
    knn_chunk_int = int(max(1, knn_chunk))
    if fast_float32:
        knn_sigma_val = np.float32(knn_sigma)
        gauss_sigma_val = np.float32(gaussian_sigma)
        two = np.float32(2.0)
        knn_denom = two * knn_sigma_val * knn_sigma_val
        gauss_denom = two * gauss_sigma_val * gauss_sigma_val
    else:
        knn_sigma_val = float(knn_sigma)
        gauss_sigma_val = float(gaussian_sigma)
        knn_denom = 2.0 * knn_sigma_val * knn_sigma_val
        gauss_denom = 2.0 * gauss_sigma_val * gauss_sigma_val

    # E as (n, 1) for broadcasting-friendly shapes
    E = E_vec.reshape(-1, 1).astype('float32', copy=True)

    for method in methods:
        if method == 'knn':
            knn_data = precomputed.get('knn')
            if knn_data is None:
                raise RuntimeError("KNN neighbors not precomputed but 'knn' requested.")
            dists = knn_data['dists']
            idx = knn_data['idx']
            new_E = np.empty_like(E)
            n = E.shape[0]
            for start in range(0, n, knn_chunk_int):
                end = min(start + knn_chunk_int, n)
                d = dists[start:end]
                w = np.exp(-(d ** 2) / knn_denom)
                w_sum = w.sum(axis=1, keepdims=True, dtype=np.float32 if fast_float32 else np.float64)
                w_sum[w_sum < 1e-12] = 1e-12
                w /= w_sum
                new_E[start:end] = (w[:, :, None] * E[idx[start:end]]).sum(
                    axis=1, dtype=np.float32 if fast_float32 else np.float64
                )
            E = new_E

        elif method == 'graph':
            g = precomputed.get("graph")
            if g is not None:
                E = _apply_graph_operator(
                    g,
                    E,
                    graph_t=graph_t_int,
                    graph_tol=graph_tol_val,
                    graph_check_every=graph_check_every_int,
                    diff_chunk_rows=knn_chunk_int,
                )
            else:
                Pg = precomputed['Pg']
                tmp = E
                for _ in range(graph_t_int):
                    tmp = Pg.dot(tmp)
                E = tmp

        elif method == 'bilateral':
            bi = precomputed.get('bilateral')
            if bi is None:
                raise RuntimeError("Bilateral neighbors not precomputed but 'bilateral' requested.")

            dists_b = bi['dists']
            idx_b = bi['idx']
            sigma_s = bi['sigma_s']

            # Chunked bilateral smoothing to cap peak memory.
            # This preserves per-row math but avoids allocating (n, k, 1) tensors.
            try:
                chunk_rows = int(os.environ.get("SPIX_BILATERAL_CHUNK", knn_chunk_int))
            except Exception:
                chunk_rows = knn_chunk_int
            chunk_rows = max(1, int(chunk_rows))
            if fast_float32:
                sigma_r2 = np.float32(bilateral_sigma_r) ** 2
                two = np.float32(2.0)
            else:
                sigma_r2 = float(bilateral_sigma_r) ** 2
                two = 2.0
            # Work on a 1D view to reduce temporary array sizes.
            E_prev = E[:, 0].astype(np.float32, copy=False)
            n_rows = E_prev.shape[0]
            for _ in range(int(bilateral_t)):
                new_vec = np.empty_like(E_prev)
                for start in range(0, n_rows, chunk_rows):
                    end = min(start + chunk_rows, n_rows)
                    idx_chunk = idx_b[start:end]
                    # Neighbor embedding values (chunk, k)
                    neigh = E_prev[idx_chunk]
                    # Spatial weights (chunk, k)
                    d_chunk = dists_b[start:end].astype(np.float32, copy=False)
                    sigma_chunk = sigma_s[start:end].astype(np.float32, copy=False)
                    ws = np.exp(-(d_chunk ** 2) / (two * (sigma_chunk[:, None] ** 2)))
                    # Range weights (chunk, k)
                    center = E_prev[start:end].astype(np.float32, copy=False)
                    ed = center[:, None] - neigh
                    wr = np.exp(-(ed ** 2) / (two * sigma_r2))
                    w = ws * wr
                    w_sum = w.sum(axis=1, keepdims=True, dtype=np.float32 if fast_float32 else np.float64)
                    w_sum[w_sum < 1e-12] = 1e-12
                    w /= w_sum
                    new_vec[start:end] = (w * neigh).sum(
                        axis=1, dtype=np.float32 if fast_float32 else np.float64
                    )
                E_prev = new_vec
            E = E_prev[:, None]

        elif method == 'gaussian':
            gauss_data = precomputed.get('gaussian')
            if gauss_data is None:
                gauss_data = precomputed.get('knn')
            if gauss_data is None:
                raise RuntimeError("Gaussian neighbors not precomputed but 'gaussian' requested.")
            Wg = gauss_data.get('W')
            if Wg is not None:
                tmp = E
                for _ in range(gaussian_t_int):
                    tmp = Wg.dot(tmp)
                E = tmp
            else:
                dists = gauss_data['dists']
                idx = gauss_data['idx']
                w = np.exp(-(dists ** 2) / gauss_denom)
                w_sum = w.sum(axis=1, keepdims=True, dtype=np.float32 if fast_float32 else np.float64)
                w_sum[w_sum < 1e-12] = 1e-12
                w /= w_sum
                for _ in range(gaussian_t_int):
                    E = (w[:, :, None] * E[idx]).sum(
                        axis=1, dtype=np.float32 if fast_float32 else np.float64
                    )

        elif method == 'guided':
            gdata = precomputed.get('guided')
            if gdata is None:
                raise RuntimeError("Guided neighbors not precomputed but 'guided' requested.")
            dists_g = gdata.get("dists") if guided_sigma is not None else None
            E = _guided_filter_chunked(
                E,
                idx=gdata["idx"],
                dists=dists_g,
                sigma=guided_sigma,
                eps=float(guided_eps),
                steps=guided_t_int,
                chunk_rows=knn_chunk_int,
                fast_float32=fast_float32,
            )

        elif method == 'heat':
            hdata = precomputed.get('heat')
            if hdata is None:
                raise RuntimeError("Heat operator not precomputed but 'heat' requested.")
            if heat_time_val > 0:
                W = hdata.get("W")
                if W is None:
                    raise RuntimeError("Heat operator missing matrix in precomputed['heat'].")
                if heat_method_val == "poisson":
                    E = _apply_heat_poisson(
                        W,
                        E,
                        time=heat_time_val,
                        tol=heat_tol_val,
                        min_k=heat_min_k_int,
                        max_k=heat_max_k_val,
                        fast_float32=fast_float32,
                    )
                else:
                    n_rows = W.shape[0]
                    Wt = hdata.get("Wt")
                    if Wt is None:
                        Wt = W.transpose(copy=False)
                        hdata["Wt"] = Wt
                    matvec = lambda x, W=W: x - W.dot(x)
                    rmatvec = lambda x, Wt=Wt: x - Wt.dot(x)
                    matmat = lambda X, W=W: X - W.dot(X)
                    rmatmat = lambda X, Wt=Wt: X - Wt.dot(X)
                    L_op = LinearOperator((n_rows, n_rows), matvec, rmatvec, matmat, W.dtype, rmatmat)
                    E = expm_multiply((-heat_time_val) * L_op, E)

        else:
            raise ValueError(f"Unknown method '{method}'.")

        # rescale if requested per step (backward compatible)
        if rescale_mode == 'per_step':
            mn, mx = E.min(axis=0), E.max(axis=0)
            denom = mx - mn
            denom[denom < 1e-12] = 1e-12
            E = (E - mn) / denom

    return E.ravel()


def tune_smooth_image_params(
    adata,
    *,
    embedding: str = "X_embedding",
    methods=("bilateral",),
    embedding_dims=None,
    obs_mask=None,
    coords=None,
    # Approximation (grid) controls for fast tuning on large datasets
    approx_target_n: int = 20_000,
    approx_bin=None,
    approx_max_bins=None,
    # Neighbor + diffusion defaults (match smooth_image defaults)
    graph_k: int = 30,
    graph_t: int = 2,
    graph_tol=None,
    graph_check_every: int = 1,
    graph_explicit="implicit",
    graph_explicit_max_mb=None,
    knn_sigma: float = 2.0,
    knn_chunk: int = 10_000,
    # Bilateral grid search
    bilateral_k: int = 30,
    bilateral_k_grid=None,
    bilateral_sigma_r_grid=None,
    bilateral_t_grid=None,
    # Scoring controls
    n_dims_eval: int = 3,
    random_state: int = 0,
    score_sample_edges: int = 200_000,
    score_q_low: float = 0.2,
    score_q_high: float = 0.8,
    # Selection controls (no manual 'strength' needed)
    selection: str = "gain_then_edge",
    select_gain_fraction=None,
    select_min_edge_preserve=None,
    # Legacy scoring weights (kept for diagnostics only)
    score_w_edge: float = 1.0,
    score_w_all: float = 0.2,
    # Optional preference knob (only applied if explicitly provided)
    strength=None,
    # Numeric behavior
    rescale_mode: str = "final",
    use_float32=None,
    fast_float32: bool = True,
    n_jobs=None,
    verbose: bool = True,
):
    """
    Automatically suggest dataset-specific `smooth_image(...)` parameters.

    This tuner is designed for very large datasets (e.g. VisiumHD) and evaluates
    candidates on an approximate coarse grid (bins) to keep runtime reasonable.

    Currently focuses on tuning the bilateral filter parameters:
      - `bilateral_sigma_r`
      - `bilateral_t`
      - optionally `bilateral_k`

    Returns:
        dict with:
          - 'best' (dict): recommended keyword args for `smooth_image(...)`
          - 'candidates' (pd.DataFrame): per-candidate scores/diagnostics
          - 'dims_eval' (list[int]): embedding dimensions used for tuning
          - 'approx' (dict): coarse-grid info (bin_size, n_bins)
    """
    log = logging.getLogger(__name__)
    if not verbose:
        prev_level = log.level
        log.setLevel(logging.WARNING)
    try:
        strength_val = None
        if strength is not None:
            strength_val = float(strength)
            if not np.isfinite(strength_val) or strength_val <= 0:
                raise ValueError("strength must be a finite positive float.")

        if isinstance(methods, (str, bytes)):
            methods = [methods]
        methods = [str(m).strip().lower() for m in methods]
        if "bilateral" not in methods:
            raise ValueError("tune_smooth_image_params currently requires methods to include 'bilateral'.")
        if score_q_low <= 0 or score_q_high >= 1 or score_q_low >= score_q_high:
            raise ValueError("score_q_low/high must satisfy 0 < low < high < 1.")

        selection_mode = str(selection).strip().lower()
        if selection_mode not in {"gain_then_edge", "product", "score", "max_gain"}:
            raise ValueError("selection must be one of {'gain_then_edge','product','score','max_gain'}.")
        select_gain_fraction_val = None
        if select_gain_fraction is not None:
            select_gain_fraction_val = float(select_gain_fraction)
            if not np.isfinite(select_gain_fraction_val) or not (0 < select_gain_fraction_val <= 1.0):
                raise ValueError("select_gain_fraction must be in (0,1].")
        min_edge_val = None if select_min_edge_preserve is None else float(select_min_edge_preserve)
        if min_edge_val is not None and (not np.isfinite(min_edge_val) or not (0 < min_edge_val <= 1.0)):
            raise ValueError("select_min_edge_preserve must be in (0,1].")

        # Load embedding (minimal copy).
        raw_embedding = np.asarray(adata.obsm[embedding])
        if raw_embedding.dtype != np.float32:
            raw_embedding = raw_embedding.astype(np.float32, copy=False)
        n_obs, n_dim_total = raw_embedding.shape

        # Resolve coords aligned to obs.
        if coords is None:
            coords_local = None
            try:
                if "spatial" in adata.obsm:
                    coords_candidate = np.asarray(adata.obsm["spatial"], dtype=np.float32)
                    if coords_candidate.shape[0] == n_obs:
                        coords_local = coords_candidate[:, :2]
            except Exception:
                coords_local = None
            if coords_local is None:
                try:
                    tiles_df = adata.uns["tiles"]
                    tiles_origin = tiles_df[tiles_df["origin"] == 1].copy()
                    tiles_origin["barcode"] = tiles_origin["barcode"].astype(str)
                    obs_idx = pd.Index(adata.obs_names.astype(str))
                    tiles_aligned = tiles_origin.set_index("barcode").reindex(obs_idx)
                    if tiles_aligned[["x", "y"]].isna().any().any():
                        missing = int(tiles_aligned[["x", "y"]].isna().any(axis=1).sum())
                        raise ValueError(
                            f"Missing coordinates for {missing} observations when aligning tiles to obs."
                        )
                    coords_local = tiles_aligned[["x", "y"]].to_numpy(dtype=np.float32)
                except Exception as e:
                    raise ValueError(
                        "Could not resolve coordinates from adata.obsm['spatial'] or adata.uns['tiles']."
                    ) from e
            coords = coords_local
        else:
            coords = np.asarray(coords, dtype=np.float32)
            if coords.ndim != 2 or coords.shape[0] != n_obs or coords.shape[1] < 2:
                raise ValueError("coords must be shaped (n_obs, 2+).")
            coords = coords[:, :2]

        # Optional: tune on a subset (e.g. in_tissue only). This often improves
        # results on VisiumHD where background tiles dominate neighborhood stats.
        if obs_mask is not None:
            if isinstance(obs_mask, (str, bytes)):
                key = str(obs_mask)
                if key not in adata.obs:
                    raise KeyError(f"obs_mask key '{key}' not found in adata.obs.")
                mask = np.asarray(adata.obs[key]).astype(bool, copy=False)
            else:
                mask = np.asarray(obs_mask).astype(bool, copy=False)
            if mask.shape[0] != n_obs:
                raise ValueError("obs_mask must have length n_obs.")
            if not bool(mask.any()):
                raise ValueError("obs_mask selects 0 observations.")
            if not bool(mask.all()):
                raw_embedding = raw_embedding[mask]
                coords = coords[mask]
                n_obs = int(raw_embedding.shape[0])

        if embedding_dims is None:
            dims_pool = np.arange(n_dim_total, dtype=np.int32)
        else:
            dims_pool = np.asarray(list(embedding_dims), dtype=np.int32)
            if dims_pool.size == 0:
                raise ValueError("embedding_dims must be non-empty when provided.")
            if dims_pool.min() < 0 or dims_pool.max() >= n_dim_total:
                raise ValueError("embedding_dims contains out-of-range indices.")

        n_dims_eval = int(max(1, n_dims_eval))
        rng = np.random.default_rng(int(random_state))
        if dims_pool.size <= n_dims_eval:
            dims_eval = [int(x) for x in dims_pool.tolist()]
        else:
            dims_eval = sorted(int(x) for x in rng.choice(dims_pool, size=n_dims_eval, replace=False))

        if n_jobs is None or int(n_jobs) == -1:
            n_jobs = multiprocessing.cpu_count()
        n_jobs = int(n_jobs)
        neighbor_workers = max(1, n_jobs)

        if use_float32 is None:
            use_float32 = str(os.environ.get("SPIX_SMOOTH_FLOAT32", "0")).strip().lower() in {"1", "true", "yes"}
        else:
            use_float32 = bool(use_float32)
        if fast_float32:
            use_float32 = True

        if approx_max_bins is None:
            try:
                approx_max_bins = int(os.environ.get("SPIX_SMOOTH_APPROX_MAX_BINS", "500000"))
            except Exception:
                approx_max_bins = 500000

        # Precompute approximate grid (bins + centers).
        approx_info = _precompute_approx_grid(
            coords,
            graph_k=int(graph_k),
            graph_t=int(graph_t),
            graph_explicit=graph_explicit,
            graph_explicit_max_mb=graph_explicit_max_mb,
            use_float32=use_float32,
            target_n=int(approx_target_n),
            bin_size=approx_bin,
            max_bins=approx_max_bins,
            build_graph=("graph" in methods),
            neighbor_workers=neighbor_workers,
        )
        bin_id = approx_info["bin_id"]
        counts = approx_info["counts"]
        centers = approx_info["centers"]
        n_bins = int(approx_info["n_bins"])

        # Aggregate into bins for evaluation dims only.
        E_eval = raw_embedding[:, dims_eval]
        sum_dtype = np.float32 if fast_float32 else np.float64
        E_bins = _aggregate_by_bins(E_eval, bin_id, counts, sum_dtype=sum_dtype).astype(np.float32, copy=False)
        # For tuning, compare candidates in the same scale as the final output.
        # `smooth_image(rescale_mode='final')` min-max rescales *after* all methods,
        # so we normalize the baseline as well to make ratios meaningful.
        tuning_rescale_mode = str(rescale_mode).strip().lower()
        if tuning_rescale_mode in {"final", "per_step"}:
            mn0 = E_bins.min(axis=0, keepdims=True)
            mx0 = E_bins.max(axis=0, keepdims=True)
            denom0 = mx0 - mn0
            denom0[denom0 < 1e-12] = 1e-12
            E_bins = (E_bins - mn0) / denom0

        # Build neighbor queries on bin centers once (use max-k and slice later).
        need_bilateral = True
        k_grid = bilateral_k_grid
        if k_grid is None:
            k_grid = [int(bilateral_k)]
        else:
            k_grid = [int(k) for k in k_grid]
        k_grid = [int(max(1, k)) for k in k_grid]
        k_grid = sorted(set(k_grid))
        k_max = int(max(k_grid))
        k_query = int(min(k_max + 1, n_bins))  # +1 for self when possible

        tree_bins = cKDTree(centers, balanced_tree=True, compact_nodes=True)
        d_all, idx_all = tree_bins.query(centers, k=k_query, workers=neighbor_workers)
        if k_query == 1:
            d_all = d_all[:, None]
            idx_all = idx_all[:, None]
        d_all = d_all.astype(np.float32, copy=False)
        idx_all = idx_all.astype(np.int32, copy=False)

        # Determine whether the first neighbor is "self" (common for KDTree with k+1).
        if idx_all.shape[1] >= 1:
            is_self = (idx_all[:, 0] == np.arange(n_bins, dtype=idx_all.dtype))
            includes_self = bool(is_self.mean() >= 0.9)
        else:
            includes_self = False
        start_pos = 1 if includes_self and idx_all.shape[1] > 1 else 0

        # Pre-generate a deterministic edge sample (src indices + uniform neighbor selector).
        n_edges = int(max(1, score_sample_edges))
        src = rng.integers(0, n_bins, size=n_edges, dtype=np.int32)
        u = rng.random(size=n_edges).astype(np.float32, copy=False)

        # Derive a data-adaptive default sigma_r grid from median neighbor diffs.
        edge_sep = None
        if bilateral_sigma_r_grid is None:
            k_for_sigma = int(max(2, min(k_query, 16)))  # small-k is enough for stats
            span = int(max(1, k_for_sigma - start_pos))
            pos_sigma = start_pos + (u * span).astype(np.int32)
            dst_sigma = idx_all[src, pos_sigma]
            medians = []
            seps = []
            for j in range(E_bins.shape[1]):
                d0 = np.abs(E_bins[src, j] - E_bins[dst_sigma, j])
                medians.append(float(np.quantile(d0, 0.5)))
                ql = float(np.quantile(d0, score_q_low))
                qh = float(np.quantile(d0, score_q_high))
                seps.append(qh / max(1e-12, ql))
            sigma_base = float(np.median(medians)) if len(medians) else 0.1
            sigma_base = float(np.clip(sigma_base, 1e-3, 2.0))
            edge_sep = float(np.median(seps)) if len(seps) else None
            # Higher sigma_r => weaker range penalty => stronger smoothing.
            # Use a wide default grid so users don't have to hand-tune "strength".
            mult = np.asarray(
                [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0],
                dtype=np.float32,
            )
            grid = (sigma_base * mult).tolist()
            # Keep within typical [0,1] embedding range, but allow a bit above.
            bilateral_sigma_r_grid = sorted(set(float(np.clip(x, 1e-3, 2.0)) for x in grid))
        else:
            bilateral_sigma_r_grid = sorted(set(float(x) for x in bilateral_sigma_r_grid))

        if bilateral_t_grid is None:
            # More steps => stronger smoothing. Use a wide default window.
            bilateral_t_grid = [1, 2, 3, 4, 5, 6, 8, 10, 12]
        bilateral_t_grid = sorted(set(int(max(1, t)) for t in bilateral_t_grid))

        if selection_mode == "gain_then_edge" and select_gain_fraction_val is None:
            # Auto tradeoff: if edges are strongly separated from flats (large edge_sep),
            # allow more aggressive smoothing; otherwise be more conservative.
            if edge_sep is None or not np.isfinite(edge_sep) or edge_sep <= 0:
                select_gain_fraction_val = 0.8
            else:
                select_gain_fraction_val = float(np.clip(0.65 + 0.08 * np.log(edge_sep), 0.6, 0.9))
            if verbose:
                log.info("tune_smooth_image_params: auto select_gain_fraction=%.3g (edge_sep=%.3g).", select_gain_fraction_val, float(edge_sep) if edge_sep is not None else float("nan"))
        elif selection_mode == "gain_then_edge":
            # Default when user does not request auto.
            if select_gain_fraction_val is None:
                select_gain_fraction_val = 0.8

        # Scoring helper: ratios on low/high-diff edges.
        def _score_candidate(E0_bins: np.ndarray, E1_bins: np.ndarray, *, k_eff: int) -> dict:
            k_eff = int(max(1, min(k_eff, idx_all.shape[1] - (0 if not includes_self else 0))))
            span = int(max(1, min(k_eff + (1 if includes_self else 0), idx_all.shape[1]) - start_pos))
            pos = start_pos + (u * span).astype(np.int32)
            dst = idx_all[src, pos]

            scores = []
            ratio_low_all = []
            ratio_high_all = []
            ratio_all_all = []
            w_edge_eff = float(score_w_edge)
            w_all_eff = float(score_w_all)
            for j in range(E0_bins.shape[1]):
                d0 = np.abs(E0_bins[src, j] - E0_bins[dst, j])
                d1 = np.abs(E1_bins[src, j] - E1_bins[dst, j])
                ratios = _stabilized_edge_ratios(
                    d0,
                    d1,
                    q_low=score_q_low,
                    q_high=score_q_high,
                )
                r_low = ratios["ratio_low"]
                r_hi = ratios["ratio_high"]
                r_all = ratios["ratio_all"]
                eps = 1e-12
                score = (1.0 - r_low) - w_edge_eff * abs(float(np.log(max(eps, r_hi)))) - w_all_eff * abs(
                    float(np.log(max(eps, r_all)))
                )
                scores.append(float(score))
                ratio_low_all.append(float(r_low))
                ratio_high_all.append(float(r_hi))
                ratio_all_all.append(float(r_all))

            return {
                "score": float(np.mean(scores)) if len(scores) else float("nan"),
                "ratio_low": float(np.mean(ratio_low_all)) if len(ratio_low_all) else float("nan"),
                "ratio_high": float(np.mean(ratio_high_all)) if len(ratio_high_all) else float("nan"),
                "ratio_all": float(np.mean(ratio_all_all)) if len(ratio_all_all) else float("nan"),
            }

        # Base precomputed dictionary for _process_dimension (bilateral only; graph already inside approx_info if needed).
        precomputed = {}
        if approx_info.get("graph") is not None:
            precomputed["graph"] = approx_info["graph"]

        results = []
        for k_eff, sigma_r, t_eff in product(k_grid, bilateral_sigma_r_grid, bilateral_t_grid):
            k_eff = int(max(1, min(int(k_eff), k_max)))
            # Slice neighbors for bilateral (keep self if present).
            k_slice = int(min(k_eff + (1 if includes_self else 0), idx_all.shape[1]))
            d_bi = d_all[:, :k_slice]
            idx_bi = idx_all[:, :k_slice]
            if k_slice - start_pos > 1:
                sigma_s = (np.median(d_bi[:, start_pos:], axis=1).astype(np.float32) + 1e-9).astype(np.float32)
            else:
                sigma_s = np.full(d_bi.shape[0], 1.0, dtype=np.float32)
            precomputed["bilateral"] = {"dists": d_bi, "idx": idx_bi, "sigma_s": sigma_s}

            # Apply smoothing on bins for each eval dim (sequential, to avoid process overhead).
            E1_cols = []
            for j in range(E_bins.shape[1]):
                out_j = _process_dimension(
                    E_bins[:, j].astype(np.float32, copy=False),
                    methods,
                    precomputed,
                    knn_sigma,
                    knn_chunk,
                    float(sigma_r),
                    int(t_eff),
                    int(graph_t),
                    graph_tol,
                    int(graph_check_every),
                    1,  # gaussian_t (unused unless 'gaussian' in methods)
                    1.0,  # gaussian_sigma (unused unless 'gaussian' in methods)
                    1e-3,  # guided_eps (unused unless 'guided' in methods)
                    None,  # guided_sigma (unused unless 'guided' in methods)
                    1,  # guided_t (unused unless 'guided' in methods)
                    0.0,  # heat_time (unused unless 'heat' in methods)
                    "krylov",  # heat_method (unused unless 'heat' in methods)
                    1e-4,  # heat_tol (unused unless 'heat' in methods)
                    0,  # heat_min_k (unused)
                    None,  # heat_max_k (unused)
                    "none" if tuning_rescale_mode == "final" else rescale_mode,
                    fast_float32,
                ).astype(np.float32, copy=False)
                E1_cols.append(out_j)
            E1_bins = np.stack(E1_cols, axis=1).astype(np.float32, copy=False)

            score = _score_candidate(E_bins, E1_bins, k_eff=k_eff)
            cand_id = len(results)
            results.append(
                {
                    "candidate_id": int(cand_id),
                    "bilateral_k": int(k_eff),
                    "bilateral_sigma_r": float(sigma_r),
                    "bilateral_t": int(t_eff),
                    **score,
                }
            )

        df = pd.DataFrame(results)
        if df.shape[0] == 0:
            raise RuntimeError("No candidates evaluated.")
        df = _finalize_smoothing_scores(df)
        best_id = _select_best_smoothing_candidate(
            df,
            selection_mode=selection_mode,
            select_gain_fraction_val=select_gain_fraction_val,
            min_edge_val=min_edge_val,
        )

        df["is_best"] = df["candidate_id"] == int(best_id)
        # Present best first, then by balanced tradeoff.
        df = df.sort_values(["is_best", "tradeoff_product"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

        best_row = df.iloc[0].to_dict()
        best = {
            "methods": list(methods),
            "embedding": embedding,
            "bilateral_k": int(best_row["bilateral_k"]),
            "bilateral_sigma_r": float(best_row["bilateral_sigma_r"]),
            "bilateral_t": int(best_row["bilateral_t"]),
            "rescale_mode": str(rescale_mode),
        }
        # Preserve user intent: if they passed embedding_dims, keep it; otherwise let
        # smooth_image default to "all dims".
        if embedding_dims is not None:
            best["embedding_dims"] = list(embedding_dims)
        if isinstance(obs_mask, (str, bytes)):
            best["obs_mask"] = str(obs_mask)

        if verbose:
            log.info(
                "tune_smooth_image_params: best k=%d, sigma_r=%.4g, t=%d | gain=%.3g edge=%.3g | selection=%s on %d bins, dims=%s",
                best["bilateral_k"],
                best["bilateral_sigma_r"],
                best["bilateral_t"],
                float(best_row.get("smooth_gain", float("nan"))),
                float(best_row.get("edge_preserve", float("nan"))),
                selection_mode,
                n_bins,
                dims_eval,
            )

        return {
            "best": best,
            "candidates": df,
            "dims_eval": dims_eval,
            "approx": {
                "n_bins": int(n_bins),
                "bin_size": float(approx_info.get("bin_size")),
            },
            "recommendation": _build_smoothing_recommendation(best_row, methods),
        }
    finally:
        if not verbose:
            log.setLevel(prev_level)
