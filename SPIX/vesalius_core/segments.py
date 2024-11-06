
### image segmentation -slic,louvain slic, leiden slic- ###
from sklearn.cluster import KMeans
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import slic
from skimage import graph as skimage_graph
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import gray2rgb
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import warnings
import pandas as pd

def segment_image(
    adata,
    dimensions=list(range(30)),
    embedding='X_embedding',
    method='slic',
    resolution=10.0,
    compactness=1.0,
    scaling=0.3,
    n_neighbors=15,
    random_state=42,
    Segment='Segment',
    index_selection='random',
    max_iter=1000,
    origin=True,
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
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    if verbose:
        print(f"Starting image segmentation using method '{method}'...")

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
    
    # embeddings = adata.obsm[embedding][:, dimensions]
    # spatial_coords = adata.obsm['spatial']

    if method == 'kmeans':
        clusters = kmeans_segmentation(embeddings, resolution, random_state)
    elif method == 'louvain':
        clusters = louvain_segmentation(embeddings, resolution, n_neighbors, random_state)
    elif method == 'leiden':
        clusters = leiden_segmentation(embeddings, resolution, n_neighbors, random_state)
    elif method == 'slic':
        clusters = slic_segmentation(
            adata,
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
        clusters = leiden_slic_segmentation(embeddings, spatial_coords, resolution, compactness, scaling, n_neighbors, random_state)
    elif method == 'louvain_slic':
        clusters = louvain_slic_segmentation(embeddings, spatial_coords, resolution, compactness, scaling, n_neighbors, random_state)
    else:
        raise ValueError(f"Unknown segmentation method '{method}'")


    # Step 2: Create Clusters DataFrame
    clusters_df = pd.DataFrame({
        'barcode': coordinates_df['barcode'],
        'Segment': clusters
    })

    # Step 3: Create Pseudo-Centroids
    pseudo_centroids = create_pseudo_centroids(
        embeddings_df,
        clusters_df,
        dimensions
    )
    
    adata.obsm['X_embedding_segment'] = pseudo_centroids
    
    # Store the clusters in adata.obs
    adata.obs[Segment] = pd.Categorical(clusters)

    if verbose:
        print("Image segmentation completed.")


def louvain_segmentation(embeddings, resolution=1.0, n_neighbors = 1.5, random_state =42):
    """
    Perform Louvain clustering on embeddings.
    """
    import scanpy as sc

    # Create a temporary AnnData object
    temp_adata = sc.AnnData(X=embeddings)
    sc.pp.neighbors(temp_adata, n_neighbors = n_neighbors, use_rep='X')
    sc.tl.louvain(temp_adata, resolution=resolution, random_state = random_state)
    clusters = temp_adata.obs['louvain'].astype(int).values
    return clusters

def leiden_segmentation(embeddings, resolution=1.0, n_neighbors = 1.5, random_state =42):
    """
    Perform Leiden clustering on embeddings.
    """
    import scanpy as sc

    # Create a temporary AnnData object
    temp_adata = sc.AnnData(X=embeddings)
    sc.pp.neighbors(temp_adata, n_neighbors = n_neighbors, use_rep='X')
    sc.tl.leiden(temp_adata,flavor='igraph',n_iterations=2, resolution=resolution, random_state = random_state)
    clusters = temp_adata.obs['leiden'].astype(int).values
    return clusters
