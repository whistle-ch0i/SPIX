import numpy as np
import scanpy as sc

try:
    import rapids_singlecell  # optional GPU accel
    _HAS_RAPIDS = True
except Exception:  # ModuleNotFoundError or any import-time issue
    rapids_singlecell = None
    _HAS_RAPIDS = False

def leiden_slic_segmentation(
    embeddings,
    spatial_coords,
    resolution=1.0,
    compactness=1.0,
    scaling=0.3,
    n_neighbors=15,
    random_state=42
):
    """
    Perform Leiden clustering on embeddings scaled with spatial coordinates and compactness.
    
    Parameters
    ----------
    embeddings : np.ndarray
        The embedding matrix.
    spatial_coords : np.ndarray
        The spatial coordinates associated with the embeddings.
    resolution : float, optional (default=1.0)
        Resolution parameter for clustering.
    compactness : float, optional (default=1.0)
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float, optional (default=0.3)
        Scaling factor for spatial coordinates.
    n_neighbors : int, optional (default=15)
        Number of neighbors for constructing the nearest neighbor graph.
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Cluster labels for each embedding.
    """
    # Compute scaling factors
    sc_spat = np.max([np.max(spatial_coords[:, 0]), np.max(spatial_coords[:, 1])]) * scaling
    sc_col = np.max(np.std(embeddings, axis=0))
    
    # Compute ratio
    ratio = (sc_spat / sc_col) / compactness
    
    # Scale embeddings
    embeddings_scaled = embeddings * ratio
    
    # Combine embeddings and spatial coordinates
    combined_data = np.concatenate([embeddings_scaled, spatial_coords], axis=1)
    
    # Create AnnData object
    temp_adata = sc.AnnData(X=combined_data)
    # Nearest neighbors (RAPIDS if available, else CPU/scanpy)
    if _HAS_RAPIDS:
        print('RAPIDS: computing neighbors for Leiden')
        rapids_singlecell.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
    else:
        print('CPU: computing neighbors for Leiden')
        sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')

    # Leiden clustering (use scanpy to keep behavior consistent across backends)
    sc.tl.leiden(
        temp_adata,
        flavor='igraph',
        n_iterations=2,
        resolution=resolution,
        random_state=random_state
    )
    clusters = temp_adata.obs['leiden'].astype(int).values

    return clusters, combined_data
