
import pandas as pd
import scanpy as sc
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import slic
from skimage import graph as skimage_graph
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import gray2rgb
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData


from spix.spix_utils import select_initial_indices, create_pseudo_centroids


def spix_image(
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
        Segmentation method: 'slic', 'leiden_slic', 'louvain_slic'.
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

    if method == 'slic':
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
        


def slic_segmentation(
    embeddings,
    spatial_coords,
    n_segments=100,
    compactness=1.0,
    scaling=0.3,
    index_selection='bubble',
    max_iter=1000,
    verbose=True
):
    """
    Perform SLIC-like segmentation using K-Means clustering with spatial and feature data.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings.
    embeddings : np.ndarray
        The embeddings to use for segmentation.
    spatial_coords : np.ndarray
        The spatial coordinates associated with the embeddings.
    n_segments : int
        The number of segments (clusters) to form.
    compactness : float
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float
        Scaling factor for spatial coordinates.
    index_selection : str
        Method for selecting initial cluster centers ('bubble', 'random', 'hex').
    max_iter : int
        Maximum number of iterations for the K-Means algorithm.
    verbose : bool
        Whether to display progress messages.

    Returns
    -------
    clusters : np.ndarray
        Cluster labels for each point.
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
    kmeans.fit(combined_data)
    clusters = kmeans.labels_

    if verbose:
        print("SLIC segmentation completed.")

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
    return clusters

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
    sc.tl.leiden(temp_adata, flavor='igraph',n_iterations=2, resolution=resolution, random_state=random_state)
    clusters = temp_adata.obs['leiden'].astype(int).values

    return clusters






def hot_spot_image(adata:AnnData,
    gene_set : list = None,
    method = 'sum'):
    if gene_set is None:
        raise ValueError('Please provide gene set to select hotspots')
    match method:
        case 'sum':
            indices = sum_gene_set(adata, gene_set)
        case 'mean':
            indices = average_gene_set(adata, gene_set)
        case 'quantile':
            indices = qunatile_gene_Set(adata, gene_set)
        case 'pca':
            indices = var_gene_set(adata, gene_set)
        case 'encode':
            indices = encode_gene_set(adata, gene_set)
        case _:
            raise ValueError('Unknonw hotspot method')