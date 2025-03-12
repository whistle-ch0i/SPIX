import numpy as np
from minisom import MiniSom

def som_segmentation(embeddings, resolution=10):
    """
    Perform Self-Organizing Map (SOM) clustering.
    
    Parameters
    ----------
    embeddings : np.ndarray
        The embedding matrix.
    resolution : int, optional (default=10)
        Grid size for the SOM (e.g., 10x10 grid).
    
    Returns
    -------
    np.ndarray
        Cluster labels for each embedding.
    """
    som_grid_size = int(np.sqrt(resolution))
    som = MiniSom(som_grid_size, som_grid_size, embeddings.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(embeddings)
    som.train_random(embeddings, 100)
    clusters = np.zeros(embeddings.shape[0], dtype=int)
    for i, x in enumerate(embeddings):
        winner = som.winner(x)
        clusters[i] = winner[0] * som_grid_size + winner[1]
    return clusters