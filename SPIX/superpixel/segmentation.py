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
    verbose: bool = True,
    library_id: str = None  # Specify library_id column name in adata.obs
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
    library_id : str, optional (default=None)
        Column name in adata.obs to group by. If specified, segmentation is performed separately for each group.
    
    Returns
    -------
    None
        Updates the AnnData object with segmentation information.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Retrieve tiles from adata.uns
    if origin:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    else:
        tiles = adata.uns['tiles']

    # Create a DataFrame from the selected embedding dimensions and add barcode from adata.obs index
    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors['barcode'] = adata.obs.index

    # Merge tiles with embeddings based on 'barcode'
    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="right").dropna().reset_index(drop=True)

    # If library_id is specified, merge the column from adata.obs into coordinates_df
    if library_id is not None:
        if library_id not in adata.obs.columns:
            raise ValueError(f"Library ID column '{library_id}' not found in adata.obs.")
        lib_info = adata.obs[[library_id]].copy()
        lib_info['barcode'] = lib_info.index
        coordinates_df = pd.merge(coordinates_df, lib_info, on="barcode", how="left")

    # If library_id is specified, perform segmentation per library group
    if library_id is not None:
        clusters_list = []
        pseudo_centroids_list = []
        combined_data_list = []
        # Group by the specified library_id column
        # Pass observed=False to silence FutureWarning (or observed=True to adopt future behavior)
        for lib, group in coordinates_df.groupby(library_id, observed=False):
            if verbose:
                print(f"Segmenting library_id: {lib} ...")
            # Extract spatial coordinates
            spatial_coords_group = group[['x', 'y']].values
            # Drop columns that are not part of the embedding
            drop_cols = ["barcode", "x", "y", "origin", library_id]
            drop_cols = [col for col in drop_cols if col in group.columns]
            embeddings_group = group.drop(columns=drop_cols).values
            embeddings_df_group = group.drop(columns=drop_cols)
            embeddings_df_group.index = group['barcode']
            barcode_group = group['barcode']
            
            clusters_df_group, pseudo_centroids_group, combined_data_group = segment_image_inner(
                embeddings_group,
                embeddings_df_group,
                barcode_group,
                spatial_coords_group,
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
            # Prefix segment labels with the library_id for uniqueness
            clusters_df_group['Segment'] = clusters_df_group['Segment'].apply(lambda x: f"{lib}_{x}")
            clusters_list.append(clusters_df_group)
            # Add library_id column to pseudo_centroids
            if isinstance(pseudo_centroids_group, pd.DataFrame):
                pseudo_centroids_group[library_id] = lib
            else:
                pseudo_centroids_group = pd.DataFrame(pseudo_centroids_group)
                pseudo_centroids_group[library_id] = lib
            pseudo_centroids_list.append(pseudo_centroids_group)
            # Add library_id column to combined_data if available
            if combined_data_group is not None:
                if isinstance(combined_data_group, pd.DataFrame):
                    combined_data_group[library_id] = lib
                else:
                    combined_data_group = pd.DataFrame(combined_data_group)
                    combined_data_group[library_id] = lib
            combined_data_list.append(combined_data_group)
        
        clusters_df = pd.concat(clusters_list)
        pseudo_centroids = pd.concat(pseudo_centroids_list)
        pseudo_centroids = pseudo_centroids.iloc[:,:-1]
        combined_data = pd.concat(combined_data_list) if any(x is not None for x in combined_data_list) else None
        combined_data.index = pseudo_centroids.index
        combined_data = combined_data.iloc[:,:-1]
        clusters_df.index = clusters_df['barcode']
        combined_data = combined_data.reindex(adata.obs.index)
        clusters_df = clusters_df.reindex(adata.obs.index)
        pseudo_centroids = pseudo_centroids.reindex(adata.obs.index)
        
    else:
        # No library_id specified, process all data together
        spatial_coords = coordinates_df.loc[:, ["x", "y"]].values
        embeddings = coordinates_df.drop(columns=["barcode", "x", "y", "origin"]).values  # Convert to NumPy array
        embeddings_df = coordinates_df.drop(columns=["barcode", "x", "y", "origin"])
        embeddings_df.index = coordinates_df['barcode']
        barcode = coordinates_df['barcode']
        
        clusters_df, pseudo_centroids, combined_data = segment_image_inner(
            embeddings,
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
    
    # Convert combined_data to numpy array if it is a DataFrame to avoid index alignment issues
    if combined_data is not None and isinstance(combined_data, pd.DataFrame):
        combined_data = combined_data.values

    # Store the pseudo-centroids and scaled embeddings in adata.obsm
    adata.obsm['X_embedding_segment'] = pseudo_centroids
    adata.obsm['X_embedding_scaled_for_segment'] = combined_data
    
    # Store the clusters in adata.obs as a categorical variable
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
    embeddings : numpy.ndarray
        Array of embedding values.
    embeddings_df : pd.DataFrame
        DataFrame of embedding values with barcode as index.
    barcode : pd.Series or list
        Barcode identifiers.
    spatial_coords : numpy.ndarray
        Spatial coordinates (x, y) for each point.
    dimensions : list of int, optional
        List of dimensions to use for segmentation.
    method : str, optional
        Segmentation method: 'kmeans', 'louvain', 'leiden', 'slic', 'som', 'leiden_slic', 'louvain_slic'.
    resolution : float or int, optional
        Resolution parameter for clustering methods.
    compactness : float
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float
        Scaling factor for spatial coordinates.
    n_neighbors : int
        Number of neighbors for constructing the nearest neighbor graph.
    random_state : int
        Random seed for reproducibility.
    Segment : str
        Name of the segment column.
    index_selection : str, optional
        Method for selecting initial cluster centers.
    max_iter : int, optional
        Maximum number of iterations for convergence.
    verbose : bool, optional
        Whether to display progress messages.
    
    Returns
    -------
    clusters_df : pd.DataFrame
        DataFrame with barcode and corresponding segment labels.
    pseudo_centroids : pd.DataFrame or numpy.ndarray
        Pseudo-centroids calculated from the embeddings.
    combined_data : pd.DataFrame or numpy.ndarray or None
        Combined data used for segmentation (if applicable).
    """
    if verbose:
        print(f"Starting image segmentation using method '{method}'...")

    # Initialize combined_data as None in case the segmentation method does not return it
    combined_data = None

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
        clusters, combined_data = leiden_slic_segmentation(
            embeddings,
            spatial_coords,
            resolution,
            compactness,
            scaling,
            n_neighbors,
            random_state
        )
    elif method == 'louvain_slic':
        clusters, combined_data = louvain_slic_segmentation(
            embeddings,
            spatial_coords,
            resolution,
            compactness,
            scaling,
            n_neighbors,
            random_state
        )
    else:
        raise ValueError(f"Unknown segmentation method '{method}'")
    
    # Create clusters DataFrame with barcode and segment labels
    clusters_df = pd.DataFrame({
        'barcode': barcode,
        'Segment': clusters
    })

    # Create pseudo-centroids using the provided utility function
    pseudo_centroids = create_pseudo_centroids(
        embeddings_df,
        clusters_df,
        dimensions
    )
    
    if verbose:
        print("Image segmentation completed.")

    return clusters_df, pseudo_centroids, combined_data
