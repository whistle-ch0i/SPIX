import numpy as np
from sklearn.cluster import KMeans

def kmeans_segmentation(embeddings, resolution=1.0, random_state=42):
    """
    Perform K-Means clustering on embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        The embedding matrix.
    resolution : float, optional (default=1.0)
        Number of clusters to form.
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Cluster labels for each embedding.
    """
    n_clusters = int(resolution)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(embeddings)
    return clusters