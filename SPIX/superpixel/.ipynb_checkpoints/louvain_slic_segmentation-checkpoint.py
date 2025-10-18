import numpy as np
import scanpy as sc
import rapids_singlecell

def louvain_slic_segmentation(
    embeddings,
    spatial_coords,
    resolution=1.0,
    compactness=1.0,
    scaling=0.3,
    n_neighbors=15,
    random_state=42
):
    """
    Perform Louvain clustering on embeddings scaled with spatial coordinates and compactness.
    
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
    # sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
    # sc.tl.louvain(temp_adata, resolution=resolution, random_state=random_state)
    print('rapids Louvain')
    rapids_singlecell.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
    rapids_singlecell.tl.louvain(temp_adata,resolution=resolution,n_iterations=100000)
    
    clusters = temp_adata.obs['louvain'].astype(int).values

    return clusters, combined_data