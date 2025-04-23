import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from tqdm import tqdm
import logging

# Logging setup (can be configured to desired level and format)
# By default, it prints INFO level messages.
# Other levels include WARNING, ERROR, CRITICAL, etc.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def smooth_image(
    adata,
    methods           = ['bilateral'],        # list of methods in order: 'knn', 'graph', 'bilateral'
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
    bilateral_n_jobs  = -1            # Number of jobs for bilateral neighbor search (-1 means all)
):
    """
    Sequentially apply one or more smoothing methods to an embedding and store
    the final smoothed embedding in adata.obsm[output].

    Smoothing methods:
    - 'knn': K-Nearest Neighbors weighted averaging based on spatial distance.
    - 'graph': Graph diffusion based on spatial distance-weighted graph.
    - 'bilateral': Graph diffusion based on both spatial and embedding distance-weighted graph (bilateral filter).

    Args:
        adata (anndata.AnnData): An AnnData object containing spatial coordinates
                                 in adata.uns['tiles'] and embeddings in adata.obsm.
        methods (list): A list of smoothing methods to apply in sequence.
                        Supported methods: 'knn', 'graph', 'bilateral'.
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

    Returns:
        anndata.AnnData: The AnnData object with the smoothed embedding added
                         to adata.obsm[output].
    """
    logging.info("Starting embedding smoothing process.")
    logging.info(f"Input embedding key: '{embedding}'")
    logging.info(f"Output embedding key: '{output}'")
    logging.info(f"Smoothing methods to apply (in order): {methods}")

    # 1) Data Preparation: Extract only tiles with origin == 1 and load coordinates and embedding.
    #    adata.uns['tiles'] contains information for all tiles; filter by the 'origin' column.
    #    Tiles with origin == 1 are the original tiles used for analysis.
    try:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
        coords = tiles[['x','y']].values.astype('float32')
        logging.info(f"Extracted coordinates for {len(coords)} tiles with origin=1.")
    except KeyError as e:
        logging.error(f"Missing expected key in adata.uns: {e}")
        raise
    except Exception as e:
        logging.error(f"Error extracting coordinates: {e}")
        raise

    # 2) Embedding Initialization: Use specified embedding dimensions or all dimensions.
    #    `E` will store the embedding data used in the current smoothing step.
    try:
        raw_embedding = adata.obsm[embedding]
        if embedding_dims is not None:
            # Use only specified dimensions
            E = raw_embedding[:, embedding_dims].astype('float32')
            logging.info(f"Using specified embedding dimensions: {embedding_dims}")
        else:
             # Use all dimensions
             E = raw_embedding.astype('float32')
             logging.info("Using all embedding dimensions.")

        n, dim = E.shape
        logging.info(f"Initial embedding shape: ({n}, {dim})")
        if n == 0 or dim == 0:
             logging.error("Embedding data is empty.")
             raise ValueError("Embedding data is empty.")

    except KeyError as e:
        logging.error(f"Input embedding key '{embedding}' not found in adata.obsm.")
        raise
    except Exception as e:
        logging.error(f"Error loading initial embedding: {e}")
        raise


    # Apply each requested smoothing method in sequence
    for method in methods:
        logging.info(f"--- Applying smoothing method: '{method}' ---")

        if method == 'knn':
            # KNN smoothing: Weighted averaging based on spatial proximity
            logging.info(f"KNN parameters: k={knn_k}, sigma={knn_sigma}, chunk={knn_chunk}, n_jobs={knn_n_jobs}")
            # Build KD-tree for spatial coordinates (for fast nearest neighbor search)
            tree = cKDTree(coords, balanced_tree=True, compact_nodes=True)
            logging.info("Built cKDTree for spatial coordinates.")

            new_E = np.empty_like(E) # Initialize array to store smoothing results

            # Process data in chunks to reduce memory usage
            logging.info(f"Processing in chunks of size {knn_chunk}...")
            for start in tqdm(range(0, n, knn_chunk), desc=f'KNN smooth (k={knn_k}, σ={knn_sigma})'):
                end = min(start + knn_chunk, n)
                # Query k nearest neighbors for the current chunk
                dists, idx = tree.query(coords[start:end], k=knn_k, workers=knn_n_jobs)
                # Calculate weights based on spatial distance (Gaussian kernel)
                w = np.exp(- (dists**2) / (2 * knn_sigma**2))
                # Normalize weights (so they sum to 1)
                w_sum = w.sum(axis=1, keepdims=True)
                # Prevent division by zero by replacing very small sums
                w_sum[w_sum < 1e-12] = 1e-12
                w /= w_sum

                # Calculate weighted average: The new embedding for each point is the weighted average of neighbor embeddings
                # w[:,:,None] is shaped (chunk_size, k, 1) for broadcasting
                # E[idx] is shaped (chunk_size, k, embedding_dim)
                # (w[:,:,None] * E[idx]) is (chunk_size, k, embedding_dim)
                # .sum(axis=1) sums along the k dimension, resulting in (chunk_size, embedding_dim)
                new_E[start:end] = (w[:,:,None] * E[idx]).sum(axis=1)

            E = new_E # Update current embedding with smoothed result
            logging.info("KNN smoothing completed.")

        elif method == 'graph':
            # Graph Diffusion: Diffuse embedding information over a spatial graph
            logging.info(f"Graph diffusion parameters: k={graph_k}, t={graph_t}, n_jobs={graph_n_jobs}")
            # Build KD-tree for spatial coordinates
            tree = cKDTree(coords, balanced_tree=True)
            # Query k+1 neighbors including self, then drop self (distance 0)
            dists, neigh = tree.query(coords, k=graph_k+1, workers=graph_n_jobs)
            dists = dists[:,1:]; neigh = neigh[:,1:] # Drop self
            logging.info(f"Queried {graph_k} spatial neighbors for graph construction.")

            # Calculate local scale (sigma_loc) per point
            # Typically uses the median distance to neighbors
            sigma_loc = np.median(dists, axis=1) + 1e-9 # Prevent zeros
            logging.info("Calculated local spatial scale (sigma_loc).")

            # Build spatial distance-based weight graph (W)
            logging.info("Building spatial affinity matrix W...")
            rows, cols, data = [], [], []
            for i in tqdm(range(n), desc=f'Graph W (k={graph_k}, σ_loc)'):
                si = sigma_loc[i]
                dij = dists[i] # Distance to neighbors
                # Calculate weights using Gaussian kernel
                wij = np.exp(- (dij**2) / (2 * si*si))
                rows.append(np.full(graph_k, i, dtype=int))
                cols.append(neigh[i]) # Indices of neighbors
                data.append(wij)

            # Convert to sparse matrix format
            rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
            W = csr_matrix((data, (rows, cols)), shape=(n,n))
            logging.info(f"Built sparse affinity matrix W with {W.nnz} non-zero elements.")

            # Symmetrize graph (remove directionality)
            W = 0.5*(W + W.T)
            logging.info("Symmetrized W.")

            # Calculate transition matrix (P): P = D^-1 * W (row-normalized)
            invdeg = 1.0/(W.sum(1).A1 + 1e-12) # Calculate inverse degree (prevent zeros)
            P = diags(invdeg).dot(W)
            logging.info("Computed transition matrix P (row-normalized W).")

            # Apply graph diffusion
            logging.info(f"Performing graph diffusion for {graph_t} steps...")
            new_E = E.copy() # Diffusion is performed by repeatedly multiplying the current embedding by P
            for step in range(graph_t):
                 # new_E = P * new_E (matrix multiplication)
                new_E = P.dot(new_E)
                logging.debug(f"Graph diffusion step {step+1}/{graph_t} completed.") # Detailed logging
            E = new_E # Update current embedding with diffused result
            logging.info("Graph diffusion completed.")

        elif method == 'bilateral':
            # Bilateral Filtering: Weighted graph diffusion considering both spatial and embedding distance
            logging.info(f"Bilateral parameters: k={bilateral_k}, sigma_r={bilateral_sigma_r}, t={bilateral_t}, n_jobs={bilateral_n_jobs}")
            # Build KD-tree for spatial coordinates
            tree = cKDTree(coords, balanced_tree=True)
            # Query k+1 neighbors including self
            dists, idx = tree.query(coords, k=bilateral_k+1, workers=bilateral_n_jobs)
            # dists: (n, k+1) - spatial distance from each point to its neighbors
            # idx:   (n, k+1) - indices of the k+1 neighbors for each point
            logging.info(f"Queried {bilateral_k+1} spatial neighbors for bilateral graph construction.")

            # Calculate spatial distance scale (sigma_s) (using median distance to neighbors excluding self)
            sigma_s = np.median(dists[:,1:], axis=1) + 1e-9 # Prevent zeros
            logging.info("Calculated spatial scale (sigma_s).")

            # Build weight graph (W) considering both spatial and embedding distance
            logging.info("Building bilateral affinity matrix W...")
            rows, cols, vals = [], [], []
            for i in tqdm(range(n), desc=f'Bilateral W (k={bilateral_k}, σ_r={bilateral_sigma_r})'):
                si = sigma_s[i]       # Spatial scale for the current point
                neigh = idx[i]        # Indices of neighbors for the current point (including self)
                sd = dists[i]         # Spatial distance from current point to neighbors (including self)

                # Calculate distance between the current point's embedding E[i] and neighbor embeddings E[neigh]
                # E[neigh] is shape (k+1, dim)
                # E[i] is shape (dim,). E[i] - E[neigh] broadcasts to (k+1, dim)
                # np.linalg.norm(..., axis=1) calculates the L2 norm for each row (per neighbor)
                ed = np.linalg.norm(E[i] - E[neigh], axis=1) # Embedding space distance

                # Calculate weights: exp(-(spatial_dist^2)/(2*sigma_s^2) - (embedding_dist^2)/(2*sigma_r^2))
                w  = np.exp(- sd**2/(2*si*si) - ed**2/(2*bilateral_sigma_r**2))

                rows.append(np.full(bilateral_k+1, i, dtype=int))
                cols.append(neigh)
                vals.append(w)

            # Convert to sparse matrix format
            rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
            W = csr_matrix((vals, (rows, cols)), shape=(n,n))
            logging.info(f"Built sparse bilateral affinity matrix W with {W.nnz} non-zero elements.")

            # Symmetrize W
            W = 0.5*(W + W.T)
            logging.info("Symmetrized W.")

            # Calculate transition matrix (P)
            invdeg = 1.0/(W.sum(1).A1 + 1e-12) # Calculate inverse degree (prevent zeros)
            P = diags(invdeg).dot(W)
            logging.info("Computed transition matrix P (row-normalized W).")

            # Apply diffusion
            logging.info(f"Performing bilateral diffusion for {bilateral_t} steps...")
            new_E = E.copy()
            for step in range(bilateral_t):
                new_E = P.dot(new_E)
                logging.debug(f"Bilateral diffusion step {step+1}/{bilateral_t} completed.") # Detailed logging
            E = new_E # Update current embedding with diffused result
            logging.info("Bilateral diffusion completed.")

        else:
            logging.error(f"Unknown smoothing method specified: '{method}'")
            raise ValueError(f"Unknown method '{method}'. Supported methods are 'knn', 'graph', 'bilateral'.")

        # Apply min-max scaling after each smoothing step
        # Prevents embedding values from diverging or going out of a certain range
        logging.info("Applying min-max scaling to current embedding E...")
        mn, mx = E.min(axis=0), E.max(axis=0)
        # Add small value to denominator to prevent division by zero
        denominator = (mx - mn)
        denominator[denominator < 1e-12] = 1e-12 # Replace with small value to prevent division by zero
        E = (E - mn) / denominator
        # Check if values are between 0 and 1 after scaling (for debugging)
        # logging.debug(f"Min after scaling: {E.min()}, Max after scaling: {E.max()}")
        logging.info("Min-max scaling completed.")


    # Store the final smoothed embedding result in adata.obsm
    adata.obsm[output] = E
    logging.info(f"Final smoothed embedding stored in adata.obsm['{output}'] with shape {E.shape}.")

    logging.info("Embedding smoothing process finished successfully.")
    return adata
