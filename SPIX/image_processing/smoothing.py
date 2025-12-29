import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
import pandas as pd
from tqdm import tqdm
import logging
import multiprocessing
import platform
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_SMOOTH_PRECOMPUTED = None  # set per-process via initializer (or inherited via fork)


def smooth_image(
    adata,
    methods           = ['bilateral'],        # list of methods in order: 'knn', 'graph', 'bilateral', 'gaussian'
    embedding         = 'X_embedding',        # key for raw embeddings in adata.obsm
    embedding_dims    = None,                 # list of embedding dimensions to use (None for all)
    output            = 'X_embedding_smooth', # key for smoothed embeddings in adata.obsm
    # K-NN smoothing parameters
    knn_k             = 35,                   # Number of neighbors for KNN/gaussian
    knn_sigma         = 2.0,                  # Kernel width for KNN spatial distance
    knn_chunk         = 10_000,               # Chunk size for KNN/gaussian weighted averaging (memory control)
    knn_n_jobs        = -1,                   # Number of jobs for KNN neighbor search (-1 means all)
    # Graph-diffusion smoothing parameters
    graph_k           = 30,                   # Number of neighbors for graph construction
    graph_t           = 2,                    # Number of diffusion steps
    graph_n_jobs      = -1,                   # Number of jobs for graph neighbor search (-1 means all)
    # Bilateral filtering parameters
    bilateral_k       = 30,                   # Number of neighbors for bilateral filtering
    bilateral_sigma_r = 0.3,                  # Kernel width for embedding space distance
    bilateral_t       = 3,                    # Number of diffusion steps
    bilateral_n_jobs  = -1,                   # Number of jobs for bilateral neighbor search (-1 means all)
    # Gaussian filtering parameter
    gaussian_sigma    = 1.0,                  # Standard deviation for gaussian filter
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
        knn_n_jobs (int): Parallel workers for KNN neighbor query (-1 = all cores).
        graph_k (int): Neighbor count for graph construction.
        graph_t (int): Number of graph diffusion steps.
        graph_n_jobs (int): Parallel workers for graph neighbor query.
        bilateral_k (int): Neighbor count for bilateral filtering.
        bilateral_sigma_r (float): Embedding kernel width for bilateral filtering.
        bilateral_t (int): Number of bilateral diffusion steps.
        bilateral_n_jobs (int): Parallel workers for bilateral neighbor query.
        gaussian_sigma (float): Spatial sigma for gaussian smoothing.
        rescale_mode (str): 'per_step' (default, backward compatible), 'final' (only
            rescale after all methods), or 'none' (no rescaling).

    Returns:
        anndata.AnnData: With smoothed embedding at adata.obsm[output].
    """
    logging.info("Starting embedding smoothing process.")
    logging.info(f"Input embedding key: '{embedding}'")
    logging.info(f"Output embedding key: '{output}'")
    logging.info(f"Smoothing methods to apply (in order): {methods}")
    use_float32 = str(os.environ.get("SPIX_SMOOTH_FLOAT32", "0")).strip().lower() in {"1", "true", "yes"}
    if use_float32:
        logging.info("Smoothing precision: float32 (SPIX_SMOOTH_FLOAT32=1)")
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

    # 3) Precompute neighbors/graphs once in the parent (memory + speed)
    precomputed = {}
    tree_spatial = cKDTree(coords, balanced_tree=True, compact_nodes=True)

    need_graph = 'graph' in methods
    need_knn = ('knn' in methods) or ('gaussian' in methods)

    def _merge_workers(a: int, b: int) -> int:
        # SciPy cKDTree: workers=-1 uses all cores
        if a == -1 or b == -1:
            return -1
        return int(max(a, b))

    # If both graph + (knn/gaussian) are requested, do a single KDTree query and
    # slice it for both to avoid duplicate work and peak memory.
    dists_all = None
    neigh_all = None
    if need_graph and need_knn:
        k_all = int(max(1, min(max(graph_k + 1, knn_k), n)))
        logging.info(f"Precomputing shared neighbors (k={k_all}) for graph+knn/gaussian...")
        dists_all, neigh_all = tree_spatial.query(
            coords, k=k_all, workers=_merge_workers(graph_n_jobs, knn_n_jobs)
        )

    if need_graph:
        logging.info("Precomputing spatial graph for diffusion...")
        k_graph = int(max(1, min(graph_k + 1, n)))
        if dists_all is not None and neigh_all is not None and dists_all.shape[1] >= k_graph:
            dists_g, neigh_g = dists_all[:, :k_graph], neigh_all[:, :k_graph]
        else:
            dists_g, neigh_g = tree_spatial.query(coords, k=k_graph, workers=graph_n_jobs)
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
        Wg = 0.5 * (Wg + Wg.T)
        invdeg = 1.0 / (Wg.sum(1).A1 + 1e-12)
        # Row-normalize without constructing an explicit diagonal matrix.
        Pg = Wg.multiply(invdeg[:, None])
        # SciPy may return COO from sparse.multiply; ensure CSR for fast dot.
        if not hasattr(Pg, "indptr"):
            Pg = Pg.tocsr()
        if use_float32 and Pg.dtype != np.float32:
            Pg = Pg.astype(np.float32, copy=False)
        precomputed['Pg'] = Pg
        logging.info("Spatial graph ready.")

    if need_knn:
        logging.info("Precomputing KNN neighbors for knn/gaussian...")
        k_knn = int(max(1, min(knn_k, n)))
        if dists_all is not None and neigh_all is not None and dists_all.shape[1] >= k_knn:
            d_knn, idx_knn = dists_all[:, :k_knn], neigh_all[:, :k_knn]
        else:
            d_knn, idx_knn = tree_spatial.query(coords, k=k_knn, workers=knn_n_jobs)
        precomputed['knn'] = {'dists': d_knn.astype('float32'), 'idx': idx_knn.astype('int32')}

    # Release the shared neighbor arrays (can be huge for VisiumHD)
    dists_all = None
    neigh_all = None

    if 'bilateral' in methods:
        logging.info("Precomputing neighbors for bilateral filter...")
        k_bi = int(max(1, min(bilateral_k + 1, n)))
        d_bi, idx_bi = tree_spatial.query(coords, k=k_bi, workers=bilateral_n_jobs)
        # keep self as first neighbor (consistent with previous behavior)
        precomputed['bilateral'] = {
            'dists': d_bi.astype('float32'),
            'idx': idx_bi.astype('int32'),
            'sigma_s': (np.median(d_bi[:, 1:], axis=1).astype('float32') + 1e-9).astype('float32')
        }

    # 4) decide number of processes
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    max_workers = max(1, min(n_jobs, len(dims_to_use)))
    logging.info(f"Backend: {backend}; workers: {max_workers} (requested={n_jobs})")

    # 5) Fast vectorized engine (huge win for graph/gaussian/knn)
    impl = (implementation or "auto").lower()
    if impl not in {"auto", "vectorized", "legacy"}:
        raise ValueError("implementation must be one of {'auto','vectorized','legacy'}.")
    can_vectorize = (
        rescale_mode in {"final", "none"}  # per_step is kept on legacy path for strict compatibility
        and ("bilateral" not in methods)   # bilateral would require huge (n,k,d) intermediates
        and all(m in {"graph", "gaussian", "knn"} for m in methods)
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
            rescale_mode=rescale_mode,
            n_workers=max_workers,
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
            rescale_mode=rescale_mode
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
                    gaussian_sigma,
                    rescale_mode,
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
    rescale_mode: str,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Blocked vectorized smoothing for methods independent across channels
    ('graph', 'gaussian', 'knn').

    Key goals:
    - Keep results consistent with the legacy per-dimension implementation.
    - Reduce Python overhead by processing multiple dimensions per task.
    - Control peak memory by processing dimension blocks of bounded size.

    Matches legacy output dtype/rescaling:
    - internal ops run in float32/float64 as induced by SciPy
    - for rescale_mode='final', min-max scaling is done in float32
    """
    methods = list(methods)
    E_src = np.asarray(E_in, dtype=np.float32)
    n_obs, n_dim = E_src.shape
    n_workers = int(max(1, n_workers))

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

    def _process_block(col_start: int, col_end: int) -> None:
        # Work on a view of the source embedding.
        E_blk = E_src[:, col_start:col_end].astype(np.float32, copy=False)
        E_work: np.ndarray = E_blk

        for method in methods:
            if method == "graph":
                Pg = precomputed.get("Pg")
                if Pg is None:
                    raise RuntimeError("Graph transition matrix not precomputed but 'graph' requested.")
                tmp = E_work
                for _ in range(int(graph_t)):
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
                )
            elif method == "gaussian":
                knn_data = precomputed.get("knn")
                if knn_data is None:
                    raise RuntimeError("KNN neighbors not precomputed but 'gaussian' requested.")
                E_work = _apply_weighted_neighbors_chunked(
                    E_work,
                    idx=knn_data["idx"],
                    dists=knn_data["dists"],
                    sigma=float(gaussian_sigma),
                    chunk_rows=int(max(1, knn_chunk)),
                )
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
) -> np.ndarray:
    """Apply Gaussian distance weights using precomputed neighbors (chunked rows).

    This mirrors the legacy per-dimension implementation but operates on a
    (n, b) block of dimensions at once to reduce overhead while keeping results
    consistent per dimension.
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
    out = np.empty((n, X.shape[1]), dtype=np.float64)

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        idx_chunk = idx[start:end]
        # Match legacy numeric behavior: distances are float32, and the square is
        # computed in float32 before promotion during division.
        d_chunk = dists[start:end].astype(np.float32, copy=False)
        w = np.exp(-(d_chunk ** 2) / (2.0 * (sigma ** 2)))
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum[w_sum < 1e-12] = 1e-12
        w /= w_sum
        # Apply weights per-dimension to preserve legacy results while avoiding
        # recomputing weights for each dimension.
        for j in range(X.shape[1]):
            neigh = X[:, j][idx_chunk]
            out[start:end, j] = (w * neigh).sum(axis=1)
    # Match legacy: weighted neighbor smoothing outputs float64 intermediates.
    return out

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
    gaussian_sigma,
    rescale_mode,
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
            for start in range(0, n, max(1, knn_chunk)):
                end = min(start + max(1, knn_chunk), n)
                d = dists[start:end]
                w = np.exp(-(d ** 2) / (2 * knn_sigma ** 2))
                w_sum = w.sum(axis=1, keepdims=True)
                w_sum[w_sum < 1e-12] = 1e-12
                w /= w_sum
                new_E[start:end] = (w[:, :, None] * E[idx[start:end]]).sum(axis=1)
            E = new_E

        elif method == 'graph':
            Pg = precomputed['Pg']
            new_E = E.copy()
            for _ in range(graph_t):
                new_E = Pg.dot(new_E)
            E = new_E

        elif method == 'bilateral':
            bi = precomputed.get('bilateral')
            if bi is None:
                raise RuntimeError("Bilateral neighbors not precomputed but 'bilateral' requested.")

            dists_b = bi['dists']
            idx_b = bi['idx']
            sigma_s = bi['sigma_s']

            new_E = E.copy()
            for _ in range(bilateral_t):
                # E_neighbors: (n, k+1, 1)
                E_neighbors = new_E[idx_b]
                # Spatial weights: (n, k+1)
                ws = np.exp(-(dists_b ** 2) / (2 * (sigma_s[:, None] ** 2)))
                # Range weights based on current embedding values
                ed = np.abs(new_E[:, None, :] - E_neighbors)
                wr = np.exp(-(ed[:, :, 0] ** 2) / (2 * bilateral_sigma_r ** 2))
                w = ws * wr
                w_sum = w.sum(axis=1, keepdims=True)
                w_sum[w_sum < 1e-12] = 1e-12
                w_norm = w / w_sum
                new_E = (w_norm[:, :, None] * E_neighbors).sum(axis=1)
            E = new_E

        elif method == 'gaussian':
            knn_data = precomputed.get('knn')
            if knn_data is None:
                raise RuntimeError("KNN neighbors not precomputed but 'gaussian' requested.")
            dists = knn_data['dists']
            idx = knn_data['idx']
            w = np.exp(-(dists ** 2) / (2 * gaussian_sigma ** 2))
            w_sum = w.sum(axis=1, keepdims=True)
            w_sum[w_sum < 1e-12] = 1e-12
            w /= w_sum
            E = (w[:, :, None] * E[idx]).sum(axis=1)

        else:
            raise ValueError(f"Unknown method '{method}'.")

        # rescale if requested per step (backward compatible)
        if rescale_mode == 'per_step':
            mn, mx = E.min(axis=0), E.max(axis=0)
            denom = mx - mn
            denom[denom < 1e-12] = 1e-12
            E = (E - mn) / denom

    return E.ravel()
