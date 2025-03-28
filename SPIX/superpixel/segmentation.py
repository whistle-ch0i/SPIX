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
import multiprocessing

from ..utils.utils import bubble_stack, random_sampling, hex_grid, create_pseudo_centroids, is_collinear, select_initial_indices

from .kmeans_segmentation import kmeans_segmentation
from .louvain_segmentation import louvain_segmentation
from .leiden_segmentation import leiden_segmentation
from .slic_segmentation import slic_segmentation
from .som_segmentation import som_segmentation
from .louvain_slic_segmentation import louvain_slic_segmentation
from .leiden_slic_segmentation import leiden_slic_segmentation

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
    return adata


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
