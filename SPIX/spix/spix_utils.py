import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors


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

    
    nbrs = NearestNeighbors(n_neighbors=1).fit(coordinates)
    distances, indices = nbrs.kneighbors(grid_points)

    unique_indices = np.unique(indices.flatten())
    return unique_indices[:n_centers]



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


