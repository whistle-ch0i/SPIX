import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from tqdm import tqdm
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def smooth_image(
    adata,
    methods           = ['bilateral'],        # list of methods in order: 'knn', 'graph', 'bilateral', 'gaussian'
    embedding         = 'X_embedding',  # key for raw embeddings in adata.obsm
    embedding_dims    = None,         # list of embedding dimensions to use (None for all)
    output            = 'X_embedding_smooth', # key for smoothed embeddings in adata.obsm
    # K-NN smoothing parameters
    knn_k             = 35,           # Number of neighbors for KNN
    knn_sigma         = 2.0,          # Kernel width for KNN spatial distance
    knn_chunk         = 10_000,       # Chunk size for KNN neighbor search (to save memory)
    knn_n_jobs        = -1,           # Number of jobs for KNN neighbor search (-1 means all)
    # Graph-diffusion smoothing parameters
    graph_k           = 30,           # Number of neighbors for graph construction
    graph_t           = 2,            # Number of diffusion steps
    graph_n_jobs      = -1,           # Number of jobs for graph neighbor search (-1 means all)
    # Bilateral filtering parameters
    bilateral_k       = 30,           # Number of neighbors for bilateral filtering
    bilateral_sigma_r = 0.3,          # Kernel width for embedding space distance
    bilateral_t       = 3,            # Number of diffusion steps
    bilateral_n_jobs  = -1,            # Number of jobs for bilateral neighbor search (-1 means all)
    # Gaussian filtering parameter
    gaussian_sigma    = 1.0,          # Standard deviation for gaussian filter
    n_jobs = None
):
    """
    Sequentially apply one or more smoothing methods to an embedding and store
    the final smoothed embedding in adata.obsm[output].

    Smoothing methods:
    - 'knn': K-Nearest Neighbors weighted averaging based on spatial distance.
    - 'graph': Graph diffusion based on spatial distance-weighted graph.
    - 'bilateral': Graph diffusion based on both spatial and embedding distance-weighted graph (bilateral filter).
    - 'gaussian': Gaussian smoothing using spatial Gaussian weights.

    Args:
        adata (anndata.AnnData): An AnnData object containing spatial coordinates
                                 in adata.uns['tiles'] and embeddings in adata.obsm.
        methods (list): A list of smoothing methods to apply in sequence.
                        Supported methods: 'knn', 'graph', 'bilateral', 'gaussian'.
        embedding (str): The key in adata.obsm where the raw embedding is stored.
        embedding_dims (list or None): A list of integer indices specifying which
                                       dimensions of the embedding to use. If None, uses all dimensions.
        output (str): The key in adata.obsm where the smoothed embedding will be stored.
        knn_k (int): Number of neighbors for KNN smoothing.
        knn_sigma (float): Spatial kernel width for KNN smoothing.
        knn_chunk (int): Chunk size for KNN neighbor queries.
        knn_n_jobs (int): Number of jobs for KNN neighbor queries.
        graph_k (int): Number of neighbors for graph construction (graph diffusion).
        graph_t (int): Number of diffusion steps (graph diffusion).
        graph_n_jobs (int): Number of jobs for graph neighbor queries.
        bilateral_k (int): Number of neighbors for bilateral filtering.
        bilateral_sigma_r (float): Embedding kernel width for bilateral filtering.
        bilateral_t (int): Number of diffusion steps (bilateral filtering).
        bilateral_n_jobs (int): Number of jobs for bilateral neighbor queries.
        gaussian_sigma (float): Spatial scale (sigma) for Gaussian smoothing.

    Returns:
        anndata.AnnData: The AnnData object with the smoothed embedding added
                         to adata.obsm[output].
    """
    logging.info("Starting embedding smoothing process.")
    logging.info(f"Input embedding key: '{embedding}'")
    logging.info(f"Output embedding key: '{output}'")
    logging.info(f"Smoothing methods to apply (in order): {methods}")
    try:
        tiles  = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
        coords = tiles[['x', 'y']].values.astype('float32')
        logging.info(f"Loaded coordinates for {coords.shape[0]} tiles.")
    except KeyError as e:
        logging.error(f"Missing key in adata.uns: {e}")
        raise

    # 2) load embedding
    try:
        raw_embedding = adata.obsm[embedding].astype('float32')
    except KeyError:
        logging.error(f"Embedding key '{embedding}' not found in adata.obsm.")
        raise
        
    n, total_dim = raw_embedding.shape
    if embedding_dims is None:
        dims_to_use = list(range(total_dim))
    else:
        dims_to_use = embedding_dims
    
    logging.info(f"Embedding shape: ({n}, {total_dim}), processing dims: {dims_to_use}")

    # 3) precompute spatial graph if needed
    precomputed_graph = {}
    if 'graph' in methods:
        logging.info("Precomputing spatial graph for diffusion...")
        tree_spatial = cKDTree(coords, balanced_tree=True, compact_nodes=True)
        dists_g, neigh_g = tree_spatial.query(coords, k=graph_k + 1, workers=graph_n_jobs)
        dists_g = dists_g[:, 1:]; neigh_g = neigh_g[:, 1:]
        sigma_loc = np.median(dists_g, axis=1) + 1e-9
        rows, cols, data = [], [], []
        for i in tqdm(range(n), desc=f'Graph W (k={graph_k})'):
            wij = np.exp(-(dists_g[i]**2) / (2 * sigma_loc[i]**2))
            rows.append(np.full(graph_k, i, dtype=int))
            cols.append(neigh_g[i])
            data.append(wij)
        rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
        Wg = csr_matrix((data, (rows, cols)), shape=(n, n))
        Wg = 0.5 * (Wg + Wg.T)
        invdeg = 1.0 / (Wg.sum(1).A1 + 1e-12)
        precomputed_graph['Pg'] = diags(invdeg).dot(Wg)
        logging.info("Spatial graph ready.")

    # 4) decide number of processes
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    logging.info(f"Using {n_jobs} processes for smoothing.")


    # Prepare the dimension-processing function
    func = partial(
        _process_dimension,
        coords=coords,
        raw_embedding=raw_embedding,
        methods=methods,
        knn_k=knn_k, knn_sigma=knn_sigma, knn_chunk=knn_chunk, knn_n_jobs=knn_n_jobs,
        precomputed_graph=precomputed_graph,
        bilateral_k=bilateral_k, bilateral_sigma_r=bilateral_sigma_r,
        bilateral_t=bilateral_t, bilateral_n_jobs=bilateral_n_jobs,
        graph_t=graph_t,
        gaussian_sigma=gaussian_sigma
    )

    # Parallel execution with real-time progress bar per dimension
    results = [None] * len(dims_to_use)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_idx = {
            executor.submit(func, dim_idx): idx
            for idx, dim_idx in enumerate(dims_to_use)
        }
        for future in tqdm(as_completed(future_to_idx),
                           total=len(future_to_idx),
                           desc="Smoothing dimensions"):
            idx = future_to_idx[future]
            results[idx] = future.result()

    # 6) stack and save
    E_smoothed = np.stack(results, axis=1).astype('float32')
    adata.obsm[output] = E_smoothed
    logging.info(f"Finished. Stored smoothed embedding → adata.obsm['{output}']")
    return adata

def _process_dimension(
    dim_idx,
    coords,
    raw_embedding,
    methods,
    knn_k, knn_sigma, knn_chunk, knn_n_jobs,
    precomputed_graph,
    bilateral_k, bilateral_sigma_r, bilateral_t, bilateral_n_jobs,
    graph_t,
    gaussian_sigma
):
    """
    Process a single embedding dimension: apply sequential smoothing
    methods (including optional Gaussian filtering) and min-max rescale.
    Returns the smoothed 1D array.
    """
    E = raw_embedding[:, [dim_idx]].copy()
    tree_spatial = cKDTree(coords, balanced_tree=True, compact_nodes=True)

    for method in methods:
        if method == 'knn':
            new_E = np.empty_like(E)
            for start in range(0, E.shape[0], knn_chunk):
                end = min(start + knn_chunk, E.shape[0])
                dists, idx = tree_spatial.query(
                    coords[start:end], k=knn_k, workers=knn_n_jobs
                )
                w = np.exp(-(dists ** 2) / (2 * knn_sigma ** 2))
                w_sum = w.sum(axis=1, keepdims=True)
                w_sum[w_sum < 1e-12] = 1e-12
                w /= w_sum
                new_E[start:end] = (w[:, :, None] * E[idx]).sum(axis=1)
            E = new_E

        elif method == 'graph':
            Pg = precomputed_graph['Pg']
            new_E = E.copy()
            for _ in range(graph_t):
                new_E = Pg.dot(new_E)
            E = new_E

        elif method == 'bilateral':
            dists_b, idx_b = tree_spatial.query(
                coords, k=bilateral_k + 1, workers=bilateral_n_jobs
            )
            sigma_s = np.median(dists_b[:, 1:], axis=1) + 1e-9
            rows, cols, vals = [], [], []
            for i in range(E.shape[0]):
                sd = dists_b[i]
                neigh = idx_b[i]
                ed = np.abs(E[i] - E[neigh]).ravel()
                w = np.exp(
                    - sd ** 2 / (2 * sigma_s[i] ** 2)
                    - ed ** 2 / (2 * bilateral_sigma_r ** 2)
                )
                rows.append(np.full(bilateral_k + 1, i, dtype=int))
                cols.append(neigh)
                vals.append(w)
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            vals = np.concatenate(vals)
            Wb = csr_matrix((vals, (rows, cols)), shape=(E.shape[0], E.shape[0]))
            Wb = 0.5 * (Wb + Wb.T)
            invdeg = 1.0 / (Wb.sum(1).A1 + 1e-12)
            Pb = diags(invdeg).dot(Wb)
            new_E = E.copy()
            for _ in range(bilateral_t):
                new_E = Pb.dot(new_E)
            E = new_E

        elif method == 'gaussian':
            dists, idx = tree_spatial.query(coords, k=knn_k, workers=knn_n_jobs)
            w = np.exp(-(dists ** 2) / (2 * gaussian_sigma ** 2))
            w_sum = w.sum(axis=1, keepdims=True)
            w_sum[w_sum < 1e-12] = 1e-12
            w /= w_sum
            E = (w[:, :, None] * E[idx]).sum(axis=1)

        else:
            raise ValueError(f"Unknown method '{method}'.")

        # min-max scaling
        mn, mx = E.min(axis=0), E.max(axis=0)
        denom = mx - mn
        denom[denom < 1e-12] = 1e-12
        E = (E - mn) / denom

    return E.ravel()
