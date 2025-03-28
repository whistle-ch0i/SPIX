import numpy as np
from typing import Union, Optional
from sklearn.cluster import KMeans
import multiprocessing
from joblib import parallel_backend
from ..utils.utils import select_initial_indices

def slic_segmentation(
    embeddings: np.ndarray,
    spatial_coords: np.ndarray,
    n_segments: int = 100,
    compactness: float = 1.0,
    scaling: float = 0.3,
    index_selection: Optional[Union[str, list]] = None,
    max_iter: int = 1000,
    verbose: bool = True
) -> np.ndarray:
    """
    Perform SLIC-like segmentation using K-Means clustering with spatial and feature data.
    
    Parameters
    ----------
    embeddings : np.ndarray
        The embedding matrix.
    spatial_coords : np.ndarray
        The spatial coordinates associated with the embeddings.
    n_segments : int, optional (default=100)
        The number of segments (clusters) to form.
    compactness : float, optional (default=1.0)
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float, optional (default=0.3)
        Scaling factor for spatial coordinates.
    index_selection : str, optional (default='bubble')
        Method for selecting initial cluster centers ('bubble', 'random', 'hex').
        As an alternative, you can also provide an array of initial indices.
    max_iter : int, optional (default=1000)
        Maximum number of iterations for convergence in initial cluster selection.
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    
    Returns
    -------
    np.ndarray
        Cluster labels for each embedding.
    """
    if verbose:
        print("Starting SLIC segmentation...")

    # Compute scaling factors
    sc_spat = np.max([np.max(spatial_coords[:, 0]), np.max(spatial_coords[:, 1])]) * scaling
    sc_col = np.max(np.std(embeddings, axis=0))
    ratio = (sc_spat / sc_col) / compactness

    # Scale embeddings
    embeddings_scaled = embeddings * ratio

    # Combine embeddings and spatial coordinates
    combined_data = np.concatenate([embeddings_scaled, spatial_coords], axis=1)

    if isinstance(index_selection, str):
        indices = select_initial_indices(
            spatial_coords,
            n_centers=n_segments,
            method=index_selection,
            max_iter=max_iter,
            verbose=verbose
        )
        
        initial_centers = combined_data[indices]
    else :
        initial_centers = combined_data[index_selection]
        n_segments = len(index_selection)

    if verbose:
        print("Running K-Means clustering...")

    # Run K-Means clustering
    kmeans = KMeans(
        n_clusters=n_segments,
        init=initial_centers,
        n_init=1,
        max_iter=max_iter,
        verbose=0,
        random_state=42
    )
    
    
    num_cores = multiprocessing.cpu_count()
    num_jobs = min(16, num_cores)  # Adjust based on system capacity
    with parallel_backend("threading", n_jobs=num_jobs):
        kmeans.fit(combined_data)
    # kmeans.fit(combined_data)
    clusters = kmeans.labels_
    if verbose:
        print("SLIC segmentation completed.")

    return clusters, combined_data