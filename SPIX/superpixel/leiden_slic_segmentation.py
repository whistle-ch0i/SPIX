import numpy as np
import scanpy as sc

def leiden_slic_segmentation(
    embeddings,
    spatial_coords,
    resolution=1.0,
    compactness=1.0,
    scaling=0.3,
    n_neighbors=15,
    random_state=42,
    use_gpu=False
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
    use_gpu : bool, optional (default=False)
        If ``True``, use ``rapids-singlecell`` to construct the neighbor graph.
        Falls back to the CPU implementation if the package is missing.

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

    if use_gpu:
        try:
            import rapids_singlecell
            print('Using rapids-singlecell for neighbor graph')
            rapids_singlecell.pp.neighbors(
                temp_adata, n_neighbors=n_neighbors, use_rep='X'
            )
        except ImportError:
            print(
                "GPU option selected but 'rapids-singlecell' is not installed.\n"
                "Install it with `pip install 'SPIX[gpu]'` or `pip install rapids-singlecell` and proper cuml version.\n"
                "Falling back to CPU implementation."
            )
            sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
    else:
        sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')

    sc.tl.leiden(
        temp_adata,
        flavor='igraph',
        n_iterations=2,
        resolution=resolution,
        random_state=random_state,
    )
    clusters = temp_adata.obs['leiden'].astype(int).values

    return clusters, combined_data
