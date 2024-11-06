
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
        


def slic_segmentation(
    adata,
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

def select_initial_indices(
    coordinates,
    n_centers=100,
    method='bubble',
    max_iter=500,
    verbose=True
):
    """
    Select initial indices for cluster centers using various methods.

    Parameters
    ----------
    coordinates : np.ndarray
        Spatial coordinates of the data points.
    n_centers : int
        Number of cluster centers to select.
    method : str
        Method for selecting initial centers ('bubble', 'random', 'hex').
    max_iter : int
        Maximum number of iterations for convergence (used in 'bubble' method).
    verbose : bool
        Whether to display progress messages.

    Returns
    -------
    indices : list of int
        Indices of the selected initial centers.
    """
    if method == 'bubble':
        indices = bubble_stack(coordinates, n_centers=n_centers, max_iter=max_iter, verbose=verbose)
    elif method == 'random':
        indices = random_sampling(coordinates, n_centers=n_centers)
    elif method == 'hex':
        indices = hex_grid(coordinates, n_centers=n_centers)
    else:
        raise ValueError(f"Unknown index selection method '{method}'")
    return indices

def random_sampling(coordinates, n_centers=100):
    """
    Randomly sample initial indices.

    Parameters
    ----------
    coordinates : np.ndarray
        Spatial coordinates of the data points.
    n_centers : int
        Number of indices to sample.

    Returns
    -------
    indices : np.ndarray
        Randomly selected indices.
    """
    indices = np.random.choice(len(coordinates), size=n_centers, replace=False)
    return indices

def bubble_stack(coordinates, n_centers=100, max_iter=500, verbose=True):
    """
    Select initial indices using the 'bubble' method.

    Parameters
    ----------
    coordinates : np.ndarray
        Spatial coordinates of the data points.
    n_centers : int
        Desired number of cluster centers.
    max_iter : int
        Maximum number of iterations for convergence.
    verbose : bool
        Whether to display progress messages.

    Returns
    -------
    background_grid : list of int
        Indices of the selected initial centers.
    """
    from sklearn.neighbors import NearestNeighbors

    # Initialize radius
    nbrs = NearestNeighbors(n_neighbors=max(2, int(len(coordinates) * 0.2))).fit(coordinates)
    distances, indices_nn = nbrs.kneighbors(coordinates)
    radius = np.max(distances)

    convergence = False
    iter_count = 0

    while not convergence and iter_count < max_iter:
        if verbose:
            print(f"Iteration {iter_count+1}: Adjusting radius to achieve desired number of centers...")

        background_grid = []
        active = set(range(len(coordinates)))
        nn_indices = indices_nn
        nn_distances = distances

        while active:
            random_start = np.random.choice(list(active))
            background_grid.append(random_start)

            # Find neighbors within radius
            neighbors = nn_indices[random_start][nn_distances[random_start] <= radius]
            # Remove these from active
            active -= set(neighbors)

        if len(background_grid) == n_centers:
            convergence = True
        elif len(background_grid) < n_centers:
            radius *= 0.75  # Decrease radius
        else:
            radius *= 1.25  # Increase radius

        iter_count += 1

    if iter_count == max_iter and not convergence:
        warnings.warn("Max iterations reached without convergence, returning approximation.")

    if len(background_grid) > n_centers:
        background_grid = background_grid[:n_centers]

    return background_grid

def hex_grid(coordinates, n_centers=100):
    """
    Select initial indices using a hexagonal grid.

    Parameters
    ----------
    coordinates : np.ndarray
        Spatial coordinates of the data points.
    n_centers : int
        Number of cluster centers to select.

    Returns
    -------
    unique_indices : np.ndarray
        Indices of the data points closest to the grid points.
    """
    x_min, x_max = np.min(coordinates[:,0]), np.max(coordinates[:,0])
    y_min, y_max = np.min(coordinates[:,1]), np.max(coordinates[:,1])

    grid_size = int(np.sqrt(n_centers))
    x_coords = np.linspace(x_min, x_max, grid_size)
    y_coords = np.linspace(y_min, y_max, grid_size)

    shift = (x_coords[1] - x_coords[0]) / 2
    c_x = []
    c_y = []
    for i, y in enumerate(y_coords):
        if i % 2 == 0:
            c_x.extend(x_coords)
        else:
            c_x.extend(x_coords + shift)
        c_y.extend([y] * grid_size)

    grid_points = np.vstack([c_x, c_y]).T

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(coordinates)
    distances, indices = nbrs.kneighbors(grid_points)

    unique_indices = np.unique(indices.flatten())
    return unique_indices[:n_centers]


def som_segmentation(embeddings, resolution=10):
    """
    Perform Self-Organizing Map (SOM) clustering.
    """
    som_grid_size = int(np.sqrt(resolution))
    som = MiniSom(som_grid_size, som_grid_size, embeddings.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(embeddings)
    som.train_random(embeddings, 100)
    win_map = som.win_map(embeddings)
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


import pandas as pd

def create_pseudo_centroids(df, clusters, dimensions):
    """
    Creates pseudo centroids by replacing the values ​​of the specified dimensions by the mean of the cluster.
    
    Parameters:
    - df (pd.DataFrame): Original dataframe. The index must match barcode.
    - clusters (pd.DataFrame): Dataframe containing cluster information. Must contain 'Segment' and 'barcode' columns.
    - dimensions (list of str): List of dimension names for which to compute the mean.
    
    Returns:
    - pd.DataFrame: Modified dataframe.
    """
    # Modify by copying the original data.
    active = df
    
    # Merge to compute means by cluster.
    merged = clusters.merge(active[dimensions], left_on='barcode', right_index=True)
    
    # Compute means by the specified dimensions by 'Segment'.
    means = merged.groupby('Segment')[dimensions].transform('mean')
    
    # Match the index based on 'barcode' and assign the mean value to the specified dimension
    active.loc[clusters['barcode'], dimensions] = means.values
    
    return active


