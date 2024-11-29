import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import scanpy as sc
import pandas as pd
from anndata import AnnData
import logging
from shapely.geometry import MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize
from shapely.validation import explain_validity
from scipy.spatial import ConvexHull, Delaunay
from minisom import MiniSom
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import warnings
import networkx as nx

from .utils import bubble_stack, random_sampling, hex_grid, create_pseudo_centroids, is_collinear

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def segment_image(
    adata: AnnData,
    dimensions: list = list(range(30)),
    embedding: str = 'X_embedding',
    method: str = 'slic',
    resolution: float = 10.0,
    compactness: float = 1.0,
    scaling: float = 0.3,
    n_neighbors: int = 15,
    random_state: int = 42,
    Segment: str = 'Segment',
    index_selection: str = 'random',
    max_iter: int = 1000,
    origin: bool = True,
    verbose: bool = True
):
    """
    Segment embeddings to find initial territories.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    dimensions : list of int, optional (default=list(range(30)))
        List of embedding dimensions to use for segmentation.
    embedding : str, optional (default='X_embedding')
        Key in adata.obsm where the embeddings are stored.
    method : str, optional (default='slic')
        Segmentation method: 'kmeans', 'louvain', 'leiden', 'slic', 'som', 'leiden_slic', 'louvain_slic'.
    resolution : float, optional (default=10.0)
        Resolution parameter for clustering methods.
    compactness : float, optional (default=1.0)
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float, optional (default=0.3)
        Scaling factor for spatial coordinates.
    n_neighbors : int, optional (default=15)
        Number of neighbors for constructing the nearest neighbor graph.
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    Segment : str, optional (default='Segment')
        Name of the segment column to store in adata.obs.
    index_selection : str, optional (default='random')
        Method for selecting initial cluster centers ('bubble', 'random', 'hex').
    max_iter : int, optional (default=1000)
        Maximum number of iterations for convergence in initial cluster selection.
    origin : bool, optional (default=True)
        Whether to use only origin tiles.
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    
    Returns
    -------
    None
        Updates the AnnData object with segmentation information.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    if origin:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    else:
        tiles = adata.uns['tiles']

    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors['barcode'] = adata.obs.index

    # Merge tiles with embeddings based on 'barcode'
    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="right").dropna().reset_index(drop=True)

    # Extract spatial coordinates and embeddings
    spatial_coords = coordinates_df.loc[:, ["x", "y"]].values
    embeddings = coordinates_df.drop(columns=["barcode", "x", "y", "origin"]).values  # Convert to NumPy array
    embeddings_df = coordinates_df.drop(columns=['barcode',"x", "y", "origin"])
    embeddings_df.index = coordinates_df['barcode']
    barcode = coordinates_df['barcode']
    clusters_df, pseudo_centroids, combined_data = segment_image_inner(embeddings,
                                                                                embeddings_df,
                                                                                barcode,
                                                                               spatial_coords,
                                                                                dimensions,
                                                                                method,
                                                                                resolution,
                                                                                compactness,
                                                                                scaling,
                                                                                n_neighbors,
                                                                                random_state,
                                                                                Segment,
                                                                                index_selection,
                                                                                max_iter,
                                                                                verbose
                                                                               )
    
    adata.obsm['X_embedding_segment'] = pseudo_centroids
    adata.obsm['X_embedding_scaled_for_segment'] = combined_data
    # adata.uns['inertia'] = inertia
    
    # Store the clusters in adata.obs
    adata.obs[Segment] = pd.Categorical(clusters_df['Segment'])


def segment_image_inner(
    embeddings,
    embeddings_df,
    barcode,
    spatial_coords,
    dimensions=list(range(30)),
    method='slic',
    resolution=10.0,
    compactness=1.0,
    scaling=0.3,
    n_neighbors=15,
    random_state=42,
    Segment='Segment',
    index_selection='random',
    max_iter=1000,
    verbose=True
):
    """
    Segment embeddings to find initial territories.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings.
    dimensions : list of int
        List of dimensions to use for segmentation.
    embedding : str
        Key in adata.obsm where the embeddings are stored.
    method : str
        Segmentation method: 'kmeans', 'louvain', 'leiden', 'slic', 'som', 'leiden_slic', 'louvain_slic'.
    resolution : float or int
        Resolution parameter for clustering methods.
    compactness : float
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float
        Scaling factor for spatial coordinates.
    n_neighbors : int
        Number of neighbors for constructing the nearest neighbor graph.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Whether to display progress messages.
    """
    if verbose:
        print(f"Starting image segmentation using method '{method}'...")


    if method == 'kmeans':
        clusters = kmeans_segmentation(embeddings, resolution, random_state)
    elif method == 'louvain':
        clusters = louvain_segmentation(embeddings, resolution, n_neighbors, random_state)
    elif method == 'leiden':
        clusters = leiden_segmentation(embeddings, resolution, n_neighbors, random_state)
    elif method == 'slic':
        clusters, combined_data = slic_segmentation(
            embeddings,
            spatial_coords,
            n_segments=int(resolution),
            compactness=compactness,
            scaling=scaling,
            index_selection=index_selection,
            max_iter=max_iter,
            verbose=verbose
        )
    elif method == 'som':
        clusters = som_segmentation(embeddings, resolution)
    elif method == 'leiden_slic':
        clusters, combined_data = leiden_slic_segmentation(embeddings, spatial_coords, resolution, compactness, scaling, n_neighbors, random_state)
    elif method == 'louvain_slic':
        clusters, combined_data = louvain_slic_segmentation(embeddings, spatial_coords, resolution, compactness, scaling, n_neighbors, random_state)
    else:
        raise ValueError(f"Unknown segmentation method '{method}'")


    # Step 2: Create Clusters DataFrame
    clusters_df = pd.DataFrame({
        'barcode': barcode,
        'Segment': clusters
    })

    # Step 3: Create Pseudo-Centroids
    pseudo_centroids = create_pseudo_centroids(
        embeddings_df,
        clusters_df,
        dimensions
    )
    
    if verbose:
        print("Image segmentation completed.")

    return clusters_df, pseudo_centroids, combined_data



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


def leiden_segmentation(embeddings, resolution=1.0, n_neighbors=15, random_state=42):
    """
    Perform Leiden clustering on embeddings.
    
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
    sc.tl.leiden(temp_adata, flavor='igraph', n_iterations=2, resolution=resolution, random_state=random_state)
    clusters = temp_adata.obs['leiden'].astype(int).values
    return clusters


def slic_segmentation(
    embeddings: np.ndarray,
    spatial_coords: np.ndarray,
    n_segments: int = 100,
    compactness: float = 1.0,
    scaling: float = 0.3,
    index_selection: str = 'bubble',
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

    # Select initial cluster centers
    indices = select_initial_indices(
        spatial_coords,
        n_centers=n_segments,
        method=index_selection,
        max_iter=max_iter,
        verbose=verbose
    )

    # Initialize KMeans with initial centers
    initial_centers = combined_data[indices]

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
    
    from joblib import parallel_backend
    
    num_cores = multiprocessing.cpu_count()
    num_jobs = min(16, num_cores)  # Adjust based on system capacity
    with parallel_backend("threading", n_jobs=num_jobs):
        kmeans.fit(combined_data)
    # kmeans.fit(combined_data)
    clusters = kmeans.labels_
    if verbose:
        print("SLIC segmentation completed.")

    return clusters, combined_data


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
    sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.louvain(temp_adata, resolution=resolution, random_state=random_state)
    clusters = temp_adata.obs['louvain'].astype(int).values

    return clusters, combined_data


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
    sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.leiden(temp_adata, flavor='igraph', n_iterations=2, resolution=resolution, random_state=random_state)
    clusters = temp_adata.obs['leiden'].astype(int).values

    return clusters, combined_data
