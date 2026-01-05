import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply, LinearOperator
import pandas as pd
from tqdm import tqdm
import logging
import multiprocessing
import platform
import os
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_SMOOTH_PRECOMPUTED = None  # set per-process via initializer (or inherited via fork)


def smooth_image(
    adata,
    methods           = ['bilateral'],        # list of methods in order: 'knn', 'graph', 'bilateral', 'gaussian', 'guided', 'heat'
    embedding         = 'X_embedding',        # key for raw embeddings in adata.obsm
    embedding_dims    = None,                 # list of embedding dimensions to use (None for all)
    output            = 'X_embedding_smooth', # key for smoothed embeddings in adata.obsm
    # K-NN smoothing parameters
    knn_k             = 35,                   # Number of neighbors for KNN/gaussian
    knn_sigma         = 2.0,                  # Kernel width for KNN spatial distance
    knn_chunk         = 10_000,               # Chunk size for KNN/gaussian weighted averaging (memory control)
    # Graph-diffusion smoothing parameters
    graph_k           = 30,                   # Number of neighbors for graph construction
    graph_t           = 2,                    # Number of diffusion steps
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
    implementation: str = "vectorized",             # 'auto'|'vectorized'|'legacy'
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
        knn_k (int): Neighbor count for KNN/gaussian smoothing.
        knn_sigma (float): Spatial kernel width for KNN smoothing.
        knn_chunk (int): Process KNN/gaussian in chunks to cap memory.
        graph_k (int): Neighbor count for graph construction.
        graph_t (int): Number of graph diffusion steps.
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
    logging.info("Starting embedding smoothing process.")
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
    tree_spatial = cKDTree(coords, balanced_tree=True, compact_nodes=True)

    need_graph = 'graph' in methods
    need_knn = 'knn' in methods
    need_gaussian = 'gaussian' in methods
    need_guided = 'guided' in methods
    need_heat = 'heat' in methods

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
    if impl == "vectorized" or (impl == "auto" and can_vectorize):
        logging.info("Using vectorized smoothing engine.")
        E_smoothed = _smooth_matrix_vectorized(
            raw_embedding[:, dims_to_use],
            methods=methods,
            precomputed=precomputed,
            knn_sigma=knn_sigma,
            knn_chunk=knn_chunk,
            graph_t=graph_t,
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
        adata.obsm[output] = E_smoothed
        logging.info(f"Finished. Stored smoothed embedding → adata.obsm['{output}']")
        return adata


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
        global _SMOOTH_PRECOMPUTED
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
    gaussian_t_int = int(gaussian_t)
    guided_t_int = int(guided_t)
    heat_time_val = float(heat_time)
    heat_method_val = str(heat_method).strip().lower()
    heat_tol_val = float(heat_tol)
    heat_min_k_int = int(max(0, heat_min_k))
    heat_max_k_val = None if heat_max_k is None else int(max(0, heat_max_k))

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
                    S = g.get("S")
                    W = g.get("W")
                    invdeg = g.get("invdeg")
                    if invdeg is None:
                        raise RuntimeError("Graph operator missing fields in precomputed['graph'].")
                    t = graph_t_int
                    if t > 0:
                        tmp = E_work
                        if S is not None:
                            for _ in range(t):
                                y = S.dot(tmp)
                                y *= invdeg[:, None]
                                tmp = y
                        else:
                            if W is None:
                                raise RuntimeError("Graph operator missing matrix in precomputed['graph'].")
                            Wt = g.get("Wt")
                            if Wt is None:
                                Wt = W.transpose(copy=False)
                            for _ in range(t):
                                y = W.dot(tmp)
                                y += Wt.dot(tmp)
                                y *= 0.5
                                y *= invdeg[:, None]
                                tmp = y
                        E_work = tmp
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


def _process_dimension(
    E_vec,
    methods,
    precomputed,
    knn_sigma,
    knn_chunk,
    bilateral_sigma_r,
    bilateral_t,
    graph_t,
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
                S = g.get("S")
                W = g.get("W")
                invdeg = g.get("invdeg")
                if invdeg is None:
                    raise RuntimeError("Graph operator missing fields in precomputed['graph'].")
                if graph_t_int > 0:
                    tmp = E
                    if S is not None:
                        for _ in range(graph_t_int):
                            y = S.dot(tmp)
                            y *= invdeg[:, None]
                            tmp = y
                    else:
                        if W is None:
                            raise RuntimeError("Graph operator missing matrix in precomputed['graph'].")
                        Wt = g.get("Wt")
                        if Wt is None:
                            Wt = W.transpose(copy=False)
                            g["Wt"] = Wt
                        for _ in range(graph_t_int):
                            y = W.dot(tmp)
                            y += Wt.dot(tmp)
                            y *= 0.5
                            y *= invdeg[:, None]
                            tmp = y
                    E = tmp
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
