import scanpy as sc
import pandas as pd

def louvain_segmentation(embeddings, resolution=1.0, n_neighbors=15, random_state=42):
    """
    Perform Louvain clustering on embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        The embedding matrix.
    resolution : float, optional (default=1.0)
        Resolution parameter for clustering.
    n_neighbors : int, optional (default=15)
        Number of neighbors for constructing the nearest neighbor graph.
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Cluster labels for each embedding.
    """
    # Create a temporary AnnData object
    temp_adata = sc.AnnData(X=embeddings)
    sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.louvain(temp_adata, resolution=resolution, random_state=random_state)
    clusters = temp_adata.obs['louvain'].astype(int).values
    return clusters