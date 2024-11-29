# SPIX/utils.py

import numpy as np
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from anndata import AnnData
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Polygon, LineString, MultiPolygon, GeometryCollection
from shapely.ops import polygonize, unary_union
from shapely.validation import explain_validity
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import logging
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import alphashape
import scipy
import cv2
import gc
from scipy.spatial import ConvexHull, Delaunay, KDTree
from scipy.ndimage import gaussian_filter
from skimage import img_as_float, exposure
from typing import Tuple, List, Optional
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def filter_grid_function(coordinates: pd.DataFrame, filter_grid: float) -> pd.DataFrame:
    """
    Filter out grid coordinates based on the specified grid filter.

    Parameters
    ----------
    coordinates : pd.DataFrame
        DataFrame containing spatial coordinates with columns [0, 1].
    filter_grid : float
        Grid filtering threshold to remove outlier beads.

    Returns
    -------
    pd.DataFrame
        Filtered coordinates.
    """
    # Round coordinates to create grid identifiers
    grid_x = np.round(coordinates[0] * filter_grid)
    grid_y = np.round(coordinates[1] * filter_grid)
    
    # Create unique grid identifiers
    grid_coord = np.char.add(grid_x.astype(str), "_")
    grid_coord = np.char.add(grid_coord, grid_y.astype(str))

    # Count occurrences in each grid
    grid_counts = Counter(grid_coord)
    counts = np.array(list(grid_counts.values()))
    cutoff = np.quantile(counts, 0.01)

    # Identify grids with counts below the cutoff
    low_count_grids = {k for k, v in grid_counts.items() if v <= cutoff}

    # Create mask for coordinates to keep
    mask = np.array([gc not in low_count_grids for gc in grid_coord])
    filtered_coordinates = coordinates[mask]

    return filtered_coordinates


def reduce_tensor_resolution(coordinates: pd.DataFrame, tensor_resolution: float = 1) -> pd.DataFrame:
    """
    Reduce tensor resolution of spatial coordinates.

    Parameters
    ----------
    coordinates : pd.DataFrame
        DataFrame containing spatial coordinates with columns [0, 1].
    tensor_resolution : float, optional (default=1)
        Resolution factor to reduce the tensor.

    Returns
    -------
    pd.DataFrame
        Reduced resolution coordinates with columns ['x', 'y'].
    """
    # Scale coordinates
    coordinates = coordinates.copy()
    coordinates[0] = np.round(coordinates[0] * tensor_resolution) + 1
    coordinates[1] = np.round(coordinates[1] * tensor_resolution) + 1

    # Group by reduced coordinates and compute mean
    coords_df = coordinates.rename(columns={0: 'x', 1: 'y'})
    coords_df['coords'] = coords_df['x'].astype(str) + '_' + coords_df['y'].astype(str)
    coords_df['original_index'] = coords_df.index
    
    coords_grouped = coords_df.groupby('coords').mean(numeric_only=True).reset_index()
    coords_grouped['original_index'] = coords_df.groupby('coords')['original_index'].first().values
    coords_grouped.set_index('original_index', inplace=True)
    filtered_coordinates = coords_grouped[['x', 'y']]

    return filtered_coordinates


def filter_tiles(vor, coordinates: pd.DataFrame, filter_threshold: float) -> Tuple[List[List[int]], np.ndarray, List]:
    """
    Filter Voronoi regions based on area threshold.

    Parameters
    ----------
    vor : Voronoi
        The Voronoi tessellation object.
    coordinates : pd.DataFrame
        DataFrame containing spatial coordinates with columns ['x', 'y'].
    filter_threshold : float
        Threshold to filter tiles based on area (quantile, e.g., 0.995).

    Returns
    -------
    Tuple[List[List[int]], np.ndarray, List]
        Filtered regions, filtered coordinates, and their indices.
    """
    point_region = vor.point_region
    regions = vor.regions
    vertices = vor.vertices

    areas = []
    valid_points = []
    valid_regions = []
    index = []
    # Calculate areas of Voronoi regions
    for idx, region_index in tqdm(enumerate(point_region), total=len(point_region), desc="Calculating Voronoi region areas"):
        region = regions[region_index]
        if -1 in region or len(region) == 0:
            continue  # Skip regions with infinite vertices or empty regions
        polygon = Polygon(vertices[region])
        area = polygon.area
        areas.append(area)
        valid_points.append(coordinates.iloc[idx].values)
        valid_regions.append(region)
        index.append(coordinates.index[idx])
    areas = np.array(areas)
    max_area = np.quantile(areas, filter_threshold)

    filtered_regions = []
    filtered_points = []
    filtered_index = []
    # Filter regions based on area
    for area, region, point, idx in tqdm(zip(areas, valid_regions, valid_points, index), total=len(areas), desc="Filtering tiles by area"):
        if area <= max_area:
            filtered_regions.append(region)
            filtered_points.append(point)
            filtered_index.append(idx)

    return filtered_regions, np.array(filtered_points), filtered_index


def rasterise(filtered_regions: List[List[int]], filtered_points: np.ndarray, index: List, vor, chunksize: int = 1000, n_jobs: int = None) -> pd.DataFrame:
    """
    Rasterize Voronoi regions into grid tiles.

    Parameters
    ----------
    filtered_regions : List[List[int]]
        List of regions (vertex indices) to rasterize.
    filtered_points : np.ndarray
        Array of spatial coordinates corresponding to the regions.
    index : List
        List of indices for each region.
    vor : Voronoi
        The Voronoi tessellation object.
    chunksize : int, optional (default=1000)
        Number of tiles per chunk for parallel processing.
    n_jobs : int, optional (default=None)
        Number of parallel jobs to run. If None, uses all available CPUs.

    Returns
    -------
    pd.DataFrame
        DataFrame containing rasterized tiles with x, y coordinates, barcode, and origin flags.
    """
    if n_jobs is None:
        n_jobs = os.cpu_count()
    elif n_jobs < 0:
        n_jobs = max(1, os.cpu_count() + n_jobs)
    elif n_jobs == 0:
        n_jobs = 1

    iterable = zip(filtered_regions, filtered_points, index, itertools.repeat(vor))
    desc = "Rasterising tiles"

    total = len(filtered_regions)

    tiles_list = _process_in_parallel_map(
        rasterise_single_tile,
        iterable,
        desc,
        n_jobs,
        chunksize,
        total=total
    )

    if tiles_list:
        tiles = pd.concat(tiles_list, ignore_index=True)
    else:
        tiles = pd.DataFrame(columns=['x', 'y', 'barcode', 'origin'])

    return tiles


def _process_in_parallel_map(function, iterable, desc, n_jobs, chunksize: int = 1000, total: int = None) -> List[pd.DataFrame]:
    """
    Helper function to process tasks in parallel with a progress bar using map.

    Parameters
    ----------
    function : callable
        The function to apply to each item in the iterable.
    iterable : iterable
        An iterable of tasks to process.
    desc : str
        Description for the progress bar.
    n_jobs : int
        Number of parallel jobs to run.
    chunksize : int, optional (default=1000)
        Number of tasks per chunk for parallel processing.
    total : int, optional (default=None)
        Total number of tasks for the progress bar.

    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames resulting from the function.
    """
    results = []
    if n_jobs == 1:
        # Serial processing with progress bar
        for item in tqdm(iterable, desc=desc, total=total):
            results.append(function(item))
    else:
        # Parallel processing with progress bar using map
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in tqdm(executor.map(function, iterable, chunksize=chunksize), desc=desc, total=total):
                results.append(result)
    return results


def rasterise_single_tile(args: Tuple[List[int], np.ndarray, int, Voronoi]) -> pd.DataFrame:
    """
    Rasterize a single Voronoi region into grid tiles.

    Parameters
    ----------
    args : Tuple[List[int], np.ndarray, int, Voronoi]
        Tuple containing:
        - region: List[int] - Indices of the vertices forming the region.
        - point: np.ndarray - Coordinates of the generating point.
        - index: int - Index of the region.
        - vor: Voronoi - The Voronoi tessellation object.

    Returns
    -------
    pd.DataFrame
        DataFrame containing rasterized tile with x, y coordinates, barcode, and origin flag.
    """
    try:
        region, point, index, vor = args
        vertices = vor.vertices[region]
        polygon = Polygon(vertices)

        # Compute bounding box of the polygon
        minx, miny, maxx, maxy = polygon.bounds

        # Generate grid points within the bounding box
        x_range = np.arange(int(np.floor(minx)), int(np.ceil(maxx)) + 1)
        y_range = np.arange(int(np.floor(miny)), int(np.ceil(maxy)) + 1)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

        # Vectorized point-in-polygon test using shapely
        mask = contains(polygon, grid_points[:, 0], grid_points[:, 1])
        pixels = grid_points[mask]

        if len(pixels) == 0:
            # Return an empty DataFrame with required columns
            return pd.DataFrame(columns=['x', 'y', 'barcode', 'origin'])

        # Calculate origin flags using vectorized operations
        origin_x = int(np.round(point[0]))
        origin_y = int(np.round(point[1]))
        origins = ((pixels[:, 0] == origin_x) & (pixels[:, 1] == origin_y)).astype(int)

        # Create DataFrame for the rasterized tile
        tile_df = pd.DataFrame({
            'x': pixels[:, 0],
            'y': pixels[:, 1],
            'barcode': str(index),
            'origin': origins
        })

        # If the origin pixel is not present, add it manually
        if not tile_df['origin'].any():
            # Create a DataFrame for the origin pixel
            origin_df = pd.DataFrame({
                'x': [origin_x],
                'y': [origin_y],
                'barcode': [str(index)],
                'origin': [1]
            })

            # Concatenate the origin pixel to the existing tile_df
            tile_df = pd.concat([tile_df, origin_df], ignore_index=True)

        return tile_df

    except Exception as e:
        # Log the error with relevant information
        logging.error(f"Error processing tile with point {args[1]}: {e}")
        # Return an empty DataFrame with required columns to maintain consistency
        return pd.DataFrame(columns=['x', 'y', 'barcode', 'origin'])


from shapely.vectorized import contains as shapely_contains
def contains(polygon: Polygon, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized point-in-polygon test using shapely.

    Parameters
    ----------
    polygon : Polygon
        The polygon to test against.
    x : np.ndarray
        Array of x-coordinates.
    y : np.ndarray
        Array of y-coordinates.

    Returns
    -------
    np.ndarray
        Boolean array indicating if points are inside the polygon.
    """
    try:
        return shapely_contains(polygon, x, y)
    except Exception as e:
        logging.error(f"Error in contains function: {e}")
        return np.zeros_like(x, dtype=bool)


def calculate_pca_loadings(args: Tuple[int, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Calculate PCA loadings for a single cell.

    Parameters
    ----------
    args : Tuple[int, np.ndarray, np.ndarray]
        Tuple containing:
        - i: int - Index of the cell.
        - counts: np.ndarray - Counts matrix.
        - loadings: np.ndarray - PCA loadings matrix.

    Returns
    -------
    np.ndarray
        The PCA loadings for the cell.
    """
    i, counts, loadings = args
    if scipy.sparse.issparse(counts):
        expressed_genes = counts[i].toarray().flatten() > 0
    else:
        expressed_genes = counts[i] > 0
    cell_loadings = np.abs(loadings[expressed_genes, :])
    cell_embedding = cell_loadings.sum(axis=0)
    return cell_embedding


def run_lsi(adata: AnnData, n_components: int = 30, remove_first: bool = True) -> np.ndarray:
    """
    Run Latent Semantic Indexing (LSI) on the data.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    n_components : int, optional (default=30)
        Number of components for LSI.
    remove_first : bool, optional (default=True)
        Whether to remove the first component.

    Returns
    -------
    np.ndarray
        The LSI embeddings.
    """
    counts = adata.X
    if scipy.sparse.issparse(counts):
        counts = counts.tocsc()
        tf = counts.multiply(1 / counts.sum(axis=1))
    else:
        tf = counts / counts.sum(axis=1)[:, None]
    idf = np.log(1 + counts.shape[0] / (1 + (counts > 0).sum(axis=0)))
    if scipy.sparse.issparse(counts):
        tfidf = tf.multiply(idf)
    else:
        tfidf = tf * idf

    U, Sigma, VT = randomized_svd(tfidf, n_components=n_components + 1)

    if remove_first:
        U = U[:, 1:]
        Sigma = Sigma[1:]
        VT = VT[1:, :]

    embeddings = np.dot(U, np.diag(Sigma))
    embeddings = MinMaxScaler().fit_transform(embeddings)

    return embeddings


def rebalance_colors(coordinates, dimensions, method="minmax"):
    """
    Rebalance color values based on the given method.

    Parameters
    ----------
    coordinates : pd.DataFrame
        DataFrame containing coordinate data.
    dimensions : list
        List of dimensions to consider (1D or 3D).
    method : str
        Method for rebalancing ('minmax' is default).

    Returns
    -------
    pd.DataFrame
        DataFrame with rebalanced colors.
    """
    if len(dimensions) == 3:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        colors = coordinates.iloc[:, [4, 5, 6]].values

        if method == "minmax":
            colors = np.apply_along_axis(min_max, 0, colors)
        else:
            colors = np.clip(colors, 0, 1)

        template = pd.concat([template, pd.DataFrame(colors, columns=["R", "G", "B"])], axis=1)

    else:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        colors = coordinates.iloc[:, 4].values

        if method == "minmax":
            colors = min_max(colors)
        else:
            colors = np.clip(colors, 0, 1)

        template = pd.concat([template, pd.Series(colors, name="Grey")], axis=1)

    return template


def brighten_colors(colors, factor=1.3):
    """
    Brightens an array of colors by the specified factor.

    Parameters
    ----------
    colors : np.ndarray
        Array of colors with shape (N, 3).
    factor : float
        Brightening factor.

    Returns
    -------
    np.ndarray
        Array of brightened colors.
    """
    return np.clip(colors * factor, 0, 1)

def is_collinear(points, tol=1e-8):
    """
    Check if a set of points are exactly collinear within a tolerance.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2).
    tol : float
        Tolerance for collinearity.

    Returns
    -------
    bool
        True if all points are collinear, False otherwise.
    """
    if len(points) < 3:
        return True
    p0, p1 = points[0], points[1]
    for p in points[2:]:
        area = 0.5 * np.abs((p1[0] - p0[0]) * (p[1] - p0[1]) - (p[0] - p0[0]) * (p1[1] - p0[1]))
        if area > tol:
            return False
    return True


def is_almost_collinear(points, tol=1e-5):
    """
    Check if a set of points are nearly collinear within a tolerance.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2).
    tol : float
        Tolerance for determining "almost" collinear.

    Returns
    -------
    bool
        True if points are almost collinear, False otherwise.
    """
    if len(points) < 3:
        return True
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        y_pred = slope * x + intercept
        residuals = y - y_pred
        return np.max(np.abs(residuals)) < tol
    except np.linalg.LinAlgError:
        return False
        
        
def smooth_polygon(polygon, buffer_dist=0.01):
    """
    Smooth a polygon using buffering.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to be smoothed.
    buffer_dist : float
        Buffer distance, adjust based on data scale.

    Returns
    -------
    shapely.geometry.Polygon
        Smoothed polygon.
    """
    try:
        smoothed = polygon.buffer(buffer_dist).buffer(-buffer_dist)
        if not smoothed.is_valid:
            logging.warning("Smoothed polygon is invalid. Attempting to fix with buffer(0).")
            smoothed = smoothed.buffer(0)
        return smoothed
    except Exception as e:
        logging.error(f"Failed to smooth polygon: {e}")
        return polygon  # Return original if smoothing fails


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (n, 2) representing the coordinates of the points.
    alpha : float
        Alpha value to influence the concaveness of the border. Smaller
        numbers don't fall inward as much as larger numbers. Too large,
        and you lose everything!

    Returns
    -------
    Polygon or MultiPolygon or None
        The resulting alpha shape polygon or None if not enough points.
    """
    if len(points) < 4:
        # Not enough points to form a polygon
        return None

    try:
        tri = Delaunay(points)
    except Exception as e:
        logging.error(f"Delaunay triangulation failed: {e}")
        return None

    triangles = points[tri.simplices]
    a = np.linalg.norm(triangles[:,0] - triangles[:,1], axis=1)
    b = np.linalg.norm(triangles[:,1] - triangles[:,2], axis=1)
    c = np.linalg.norm(triangles[:,2] - triangles[:,0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circum_r = (a * b * c) / (4.0 * area)
    filtered = circum_r < 1.0 / alpha
    triangles = tri.simplices[filtered]

    if len(triangles) == 0:
        logging.warning("No triangles found with the given alpha. Consider increasing alpha.")
        return None

    edges = set()
    edge_points = []
    for tri in triangles:
        edges.update([
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ])

    for edge in edges:
        edge_points.append(points[list(edge)])

    polygons = list(polygonize(edge_points))
    if not polygons:
        return None
    return unary_union(polygons)

def _check_spatial_data(uns, library_id):
    """
    Helper function to retrieve spatial data from AnnData's uns.

    Parameters
    ----------
    uns : dict
        The `uns` attribute from an AnnData object.
    library_id : str or None
        The library ID to retrieve. If None, it attempts to select the appropriate library.

    Returns
    -------
    tuple
        (library_id, spatial_data)
    """
    spatial_mapping = uns.get('spatial', {})
    if library_id is None:
        if len(spatial_mapping) > 1:
            raise ValueError(
                "Multiple libraries found in adata.uns['spatial']. Please specify library_id."
            )
        elif len(spatial_mapping) == 1:
            library_id = list(spatial_mapping.keys())[0]
        else:
            library_id = None
    spatial_data = spatial_mapping.get(library_id, None)
    return library_id, spatial_data

def _check_img(spatial_data, img, img_key, bw=False):
    """
    Helper function to retrieve and process the background image.

    Parameters
    ----------
    spatial_data : dict or None
        The spatial data dictionary from AnnData's uns.
    img : np.ndarray or None
        The image array provided by the user.
    img_key : str or None
        The key to select the image from spatial data.
    bw : bool
        Whether to convert the image to grayscale.

    Returns
    -------
    tuple
        (img, img_key)
    """
    if img is None and spatial_data is not None:
        if img_key is None:
            img_key = next((k for k in ['hires', 'lowres'] if k in spatial_data['images']), None)
            if img_key is None:
                raise ValueError("No image found in spatial data.")
        img = spatial_data['images'][img_key]
    if bw and img is not None:
        # Convert to grayscale using luminosity method
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img, img_key

def _check_scale_factor(spatial_data, img_key, scaling_factor):
    """
    Helper function to determine the scaling factor.

    Parameters
    ----------
    spatial_data : dict or None
        The spatial data dictionary from AnnData's uns.
    img_key : str or None
        The key to select the image from spatial data.
    scaling_factor : float or None
        The scaling factor provided by the user.

    Returns
    -------
    float
        The determined scaling factor.
    """
    if scaling_factor is not None:
        return scaling_factor
    elif spatial_data is not None and img_key is not None:
        return spatial_data['scalefactors'][f'tissue_{img_key}_scalef']
    else:
        return 1.0

def balance_simplest(data: np.ndarray, sleft: float = 1.0, sright: float = 1.0) -> np.ndarray:
    """
    Balance the histogram by saturating a certain percentage on both sides.

    Parameters
    ----------
    data : np.ndarray
        1D array of data values.
    sleft : float, optional (default=1.0)
        Percentage of data to saturate on the lower end.
    sright : float, optional (default=1.0)
        Percentage of data to saturate on the upper end.

    Returns
    -------
    np.ndarray
        Balanced and normalized data.
    """
    lower = np.percentile(data, sleft)
    upper = np.percentile(data, 100 - sright)
    balanced = np.clip(data, lower, upper)
    return min_max(balanced)


def min_max(values):
    """
    Apply min-max normalization to a numpy array.

    Parameters
    ----------
    values : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    return (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)


def equalize_piecewise(data: np.ndarray, N: int = 1, smax: float = 1.0) -> np.ndarray:
    """
    Perform piecewise equalization by dividing data into N segments and equalizing each.

    Parameters
    ----------
    data : np.ndarray
        1D array of data values.
    N : int, optional (default=1)
        Number of segments to divide the data into.
    smax : float, optional (default=1.0)
        Maximum saturation limit per segment.

    Returns
    -------
    np.ndarray
        Piecewise equalized data.
    """
    if N < 1:
        raise ValueError("Number of segments N must be at least 1.")
    bins = np.linspace(np.min(data), np.max(data), N + 1)
    equalized = np.empty_like(data)
    for i in range(N):
        mask = (data >= bins[i]) & (data < bins[i + 1])
        segment = data[mask]
        if len(segment) > 0:
            equalized[mask] = balance_simplest(segment, sleft=0.0, sright=0.0)
        else:
            equalized[mask] = 0
    # Handle the last bin
    mask = data == bins[-1]
    equalized[mask] = 1.0
    return equalized


def spe_equalization(data: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
    """
    Perform SPE (Specific method, e.g., background correction) equalization.

    Parameters
    ----------
    data : np.ndarray
        1D array of data values.
    lambda_ : float, optional (default=0.1)
        Strength of background correction.

    Returns
    -------
    np.ndarray
        SPE equalized data.
    """
    # Placeholder implementation; replace with actual SPE equalization logic
    data_mean = np.mean(data)
    data_std = np.std(data)
    normalized = (data - data_mean) / (data_std + 1e-10)
    equalized = normalized / (1 + lambda_ * np.abs(normalized))
    return min_max(equalized)


def equalize_dp(data: np.ndarray, down: float = 10.0, up: float = 100.0) -> np.ndarray:
    """
    Perform DP (Dual-Pixel?) equalization by clipping data between down and up thresholds and normalizing.

    Parameters
    ----------
    data : np.ndarray
        1D array of data values.
    down : float, optional (default=10.0)
        Lower threshold for clipping.
    up : float, optional (default=100.0)
        Upper threshold for clipping.

    Returns
    -------
    np.ndarray
        DP equalized data.
    """
    clipped = np.clip(data, down, up)
    return min_max(clipped)


def equalize_adp(data: np.ndarray) -> np.ndarray:
    """
    Perform ADP (Adaptive Histogram Equalization) on the data.

    Parameters
    ----------
    data : np.ndarray
        1D array of data values.

    Returns
    -------
    np.ndarray
        ADP equalized data.
    """
    # Reshape data for skimage's adaptive equalization
    data_reshaped = data.reshape(1, -1)
    equalized = exposure.equalize_adapthist(data_reshaped, clip_limit=0.03)
    return equalized.flatten()


def ecdf_eq(data: np.ndarray) -> np.ndarray:
    """
    Perform ECDF-based histogram equalization.

    Parameters
    ----------
    data : np.ndarray
        1D array of data values.

    Returns
    -------
    np.ndarray
        ECDF equalized data.
    """
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return np.interp(data, sorted_data, cdf)
    
def plot_boundaries_on_ax(ax: plt.Axes, coordinates_df: pd.DataFrame, boundary_method: str, alpha: float,
                          boundary_color: str, boundary_linewidth: float, alpha_shape_alpha: float,
                          aspect_ratio_threshold: float, jitter: float, tol_collinear: float,
                          tol_almost_collinear: float):
    """
    Plot segment boundaries on the given Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the boundaries.
    coordinates_df : pandas.DataFrame
        DataFrame containing the coordinates and segment information.
    boundary_method : str
        Method to compute boundaries ('convex_hull', 'alpha_shape').
    alpha : float
        Transparency for the boundary polygons.
    boundary_color : str
        Color for the boundaries.
    boundary_linewidth : float
        Line width for the boundaries.
    alpha_shape_alpha : float
        Alpha value for alpha shapes (only relevant if using 'alpha_shape').
    aspect_ratio_threshold : float
        Threshold to determine if a segment is thin and long based on aspect ratio.
    jitter : float
        Amount of jitter to add to points to avoid collinearity.
    tol_collinear : float
        Tolerance for exact collinearity.
    tol_almost_collinear : float
        Tolerance for near collinearity.
    """
    # Group by segments
    segments = coordinates_df['Segment'].unique()
    for seg in segments:
        seg_tiles = coordinates_df[coordinates_df['Segment'] == seg]
        if len(seg_tiles) < 2:
            # Not enough points to form a boundary
            logging.info(f"Segment {seg} has less than 2 points. Skipping boundary plot.")
            continue

        points = seg_tiles[['x', 'y']].values

        # Calculate aspect ratio
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        aspect_ratio = max(x_range, y_range) / (min(x_range, y_range) + 1e-10)  # Avoid division by zero

        if aspect_ratio > aspect_ratio_threshold:
            # Treat as thin and long segment
            if len(seg_tiles) >= 2:
                # Sort points by x for consistent line
                sorted_indices = np.argsort(points[:, 0])
                sorted_points = points[sorted_indices]
                line = LineString(sorted_points)
                # Buffer the line to create a polygon with small thickness
                buffer_size = min(x_range, y_range) * 0.05  # Buffer size proportional to smaller range
                buffered_line = line.buffer(buffer_size)
                # Plot the buffered line
                if not buffered_line.is_empty:
                    if isinstance(buffered_line, Polygon):
                        polygons = [buffered_line]
                    elif isinstance(buffered_line, MultiPolygon):
                        polygons = list(buffered_line)
                    else:
                        logging.warning(f"Buffered line for segment {seg} is neither Polygon nor MultiPolygon.")
                        continue
                    for poly in polygons:
                        patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                                fill=False, edgecolor=boundary_color, 
                                                linewidth=boundary_linewidth, alpha=alpha)
                        ax.add_patch(patch)
            continue

        # Use ConvexHull or alpha_shape for regular segments
        if boundary_method == 'convex_hull':
            if len(seg_tiles) < 3:
                logging.warning(f"Segment {seg} has less than 3 points. ConvexHull cannot be computed.")
                continue
            try:
                # Check for unique points
                unique_points = np.unique(points, axis=0)
                if unique_points.shape[0] < 3:
                    logging.warning(f"Segment {seg} has less than 3 unique points. ConvexHull cannot be computed.")
                    continue
                # Check for collinearity
                if is_collinear(unique_points, tol_collinear):
                    logging.warning(f"Segment {seg} points are exactly collinear. Plotting LineString.")
                    line = LineString(unique_points)
                    if not line.is_empty:
                        x, y = line.xy
                        ax.plot(x, y, color=boundary_color, linewidth=boundary_linewidth, alpha=alpha)
                    continue
                elif is_almost_collinear(unique_points, tol_almost_collinear):
                    logging.info(f"Segment {seg} points are nearly collinear. Adding jitter.")
                    unique_points += np.random.uniform(-jitter, jitter, unique_points.shape)
                hull = ConvexHull(unique_points)
                hull_points = unique_points[hull.vertices]
                polygon = Polygon(hull_points)
            except Exception as e:
                logging.error(f"ConvexHull failed for segment {seg}: {e}")
                continue
        elif boundary_method == 'alpha_shape':
            polygon = alpha_shape_func(points, alpha=alpha_shape_alpha)
            if polygon is None:
                logging.warning(f"Alpha shape failed for segment {seg}.")
                continue
        else:
            raise ValueError(f"Unknown boundary method '{boundary_method}'")

        if boundary_method == 'convex_hull':
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # Attempt to fix invalid polygons

            if polygon.is_empty:
                logging.warning(f"Polygon is empty for segment {seg}.")
                continue

            # Create a patch from the polygon
            if isinstance(polygon, Polygon):
                polygons = [polygon]
            elif isinstance(polygon, MultiPolygon):
                polygons = list(polygon)
            else:
                logging.warning(f"Polygon for segment {seg} is neither Polygon nor MultiPolygon.")
                continue

            for poly in polygons:
                patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                        fill=False, edgecolor=boundary_color, 
                                        linewidth=boundary_linewidth, alpha=alpha)
                ax.add_patch(patch)
        elif boundary_method == 'alpha_shape':
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # Attempt to fix invalid polygons

            if polygon.is_empty:
                logging.warning(f"Polygon is empty for segment {seg}.")
                continue

            # Create a patch from the polygon
            if isinstance(polygon, Polygon):
                polygons = [polygon]
            elif isinstance(polygon, MultiPolygon):
                polygons = list(polygon)
            else:
                logging.warning(f"Polygon for segment {seg} is neither Polygon nor MultiPolygon.")
                continue

            for poly in polygons:
                patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                        fill=False, edgecolor=boundary_color, 
                                        linewidth=boundary_linewidth, alpha=alpha)
                ax.add_patch(patch)
                
def process_single_dimension(args: Tuple[int, np.ndarray, np.ndarray, str, float]) -> np.ndarray:
    """
    Process a single dimension with smoothing.

    Parameters
    ----------
    args : Tuple[int, np.ndarray, np.ndarray, str, float]
        Tuple containing:
        - dim: int - The dimension index.
        - embedding_dim: np.ndarray - The embedding for the dimension.
        - tiles_shifted: np.ndarray - Shifted tiles data.
        - method: str - Smoothing method ('iso', 'gaussian', 'median', etc.).
        - sigma: float - Smoothing parameter.

    Returns
    -------
    np.ndarray
        The smoothed embedding.
    """
    try:
        dim, embedding_dim, tiles_shifted, method, sigma = args
    except ValueError as ve:
        logging.error(f"Error unpacking arguments: {ve}. Args received: {args}")
        return np.array([])  # Return an empty array or handle as needed

    try:
        logging.info(f"Processing dimension {dim} with method '{method}' and sigma {sigma}")
        if method in ['iso', 'gaussian']:
            smoothed = gaussian_filter(embedding_dim, sigma=sigma)
        elif method == 'median':
            smoothed = median_filter(embedding_dim, size=3)
        else:
            logging.warning(f"Unknown smoothing method '{method}'. Skipping smoothing for dimension {dim}.")
            smoothed = embedding_dim  # No smoothing applied
        return smoothed
    except Exception as e:
        logging.error(f"Error processing dimension {dim}: {e}")
        return np.array([])  # Return an empty array or handle as needed

def process_single_dimension_fill_nan(args: Tuple[int, np.ndarray, pd.Index, pd.DataFrame, List[str], int, float, int, float, bool, bool, str, int, int, bool]) -> Tuple[int, np.ndarray]:
    """
    Process a single dimension by filling NaN values.

    Parameters
    ----------
    args : Tuple containing:
        - dim (int): Dimension index.
        - embeddings_dim (np.ndarray): Embedding values for the dimension.
        - adata_obs_names (pd.Index): Observation names from AnnData.
        - tiles_shifted (pd.DataFrame): Shifted tiles DataFrame with 'x', 'y', 'barcode'.
        - method (List[str]): List of methods for processing.
        - iter (int): Number of iterations.
        - sigma (float): Sigma value for Gaussian smoothing.
        - box (int): Box size for median blur.
        - threshold (float): Threshold value.
        - neuman (bool): Neumann boundary condition.
        - na_rm (bool): Remove NAs.
        - across_levels (str): Method for combining across levels.
        - image_height (int): Height of the image.
        - image_width (int): Width of the image.
        - verbose (bool): Verbosity flag.

    Returns
    -------
    Tuple[int, np.ndarray]:
        - dim (int): Dimension index.
        - image_filled (np.ndarray): Image with NaNs filled.
    """
    try:
        (
            dim,
            embeddings_dim,
            adata_obs_names,
            tiles_shifted,
            method,
            iter,
            sigma,
            box,
            threshold,
            neuman,
            na_rm,
            across_levels,
            image_height,
            image_width,
            verbose
        ) = args
    except ValueError as ve:
        logging.error(f"Error unpacking arguments: {ve}. Args received: {args}")
        return (-1, np.array([]))  # Return invalid dim and empty array

    try:
        if verbose:
            logging.info(f"Filling NaNs for dimension {dim}...")

        # Map barcodes to embedding values
        barcode_to_embed = dict(zip(adata_obs_names, embeddings_dim))

        # Assign embedding values to image pixels
        embed_array = tiles_shifted['barcode'].map(barcode_to_embed).values

        # Check for any missing embeddings
        if np.isnan(embed_array).any():
            missing_barcodes = tiles_shifted['barcode'][np.isnan(embed_array)].unique()
            logging.error(f"Missing embedding values for barcodes: {missing_barcodes}")
            # Handle missing embeddings as per your requirements
            embed_array = np.nan_to_num(embed_array, nan=0.0)

        # Initialize image with NaNs
        image = np.full((image_height, image_width), np.nan, dtype=np.float32)

        # **Convert x and y to integers before indexing**
        y_indices = tiles_shifted['y'].astype(int).values
        x_indices = tiles_shifted['x'].astype(int).values

        image[y_indices, x_indices] = embed_array

        # Fill NaNs using OpenCV's inpainting
        image_filled = fill_nan_with_opencv_inpaint(image)

        return (dim, image_filled)

    except Exception as e:
        logging.error(f"Error processing dimension {dim}: {e}")
        return (dim, np.array([]))  # Return dim and empty array

def process_single_dimension_smooth(args: Tuple[int, pd.DataFrame, List[str], int, float, int, float, bool, str, bool]) -> Tuple[int, pd.DataFrame, np.ndarray]:
    """
    Process a single dimension by applying smoothing.

    Parameters
    ----------
    args : Tuple containing:
        - dim (int): Dimension index.
        - tiles_shifted (pd.DataFrame): Shifted tiles DataFrame with 'x', 'y', 'barcode'.
        - method (List[str]): List of smoothing methods.
        - iter (int): Number of iterations.
        - sigma (float): Sigma value for Gaussian smoothing.
        - box (int): Box size for median blur.
        - threshold (float): Threshold value.
        - neuman (bool): Neumann boundary condition.
        - across_levels (str): Method for combining across levels.
        - verbose (bool): Verbosity flag.

    Returns
    -------
    Tuple[int, pd.DataFrame, np.ndarray]:
        - dim (int): Dimension index.
        - smoothed_barcode_grouped (pd.DataFrame): DataFrame with smoothed embeddings per barcode.
        - combined_image (np.ndarray): Combined smoothed image.
    """
    try:
        (
            dim,
            tiles_shifted,
            method,
            iter,
            sigma,
            box,
            threshold,
            neuman,
            across_levels,
            verbose,
            image_filled
        ) = args
    except ValueError as ve:
        logging.error(f"Error unpacking arguments: {ve}. Args received: {args}")
        return (-1, pd.DataFrame(), np.array([]))  # Return invalid dim, empty DataFrame, and empty array

    try:
        if verbose:
            logging.info(f"Smoothing for dimension {dim}...")

        # Initialize a list to store smoothed images for each iteration
        smoothed_images = []

        # Assign embedding values to image pixels
        y_indices = tiles_shifted['y'].astype(int).values
        x_indices = tiles_shifted['x'].astype(int).values

        # Apply smoothing iterations
        for i in range(iter):
            if verbose:
                logging.info(f"Iteration {i+1}/{iter} for dimension {dim}.")

            # Apply each smoothing method
            for m in method:
                if m == 'median':
                    # Median blur requires odd kernel size
                    kernel_size = int(box)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    if kernel_size < 1:
                        raise ValueError("Box size must be positive for median blur.")
                    image_filled = median_filter(image_filled, size=kernel_size)
                elif m == 'iso':
                    # Gaussian blur
                    image_filled = gaussian_filter(image_filled, sigma=sigma, mode='reflect' if neuman else 'constant')
                elif m == 'box':
                    # Box blur using average filter
                    kernel_size = int(box)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    if kernel_size < 1:
                        raise ValueError("Box size must be positive for box blur.")
                    image_filled = cv2.blur(image_filled, (kernel_size, kernel_size))
                else:
                    raise ValueError(f"Smoothing method '{m}' is not recognized.")

            # Apply threshold if specified
            if threshold > 0:
                image_filled = np.where(image_filled >= threshold, image_filled, 0)

            # Append to smoothed_images list
            smoothed_images.append(image_filled)

        # Combine smoothed images across iterations
        if len(smoothed_images) > 1:
            if across_levels == 'min':
                combined_image = np.minimum.reduce(smoothed_images)
            elif across_levels == 'mean':
                combined_image = np.mean(smoothed_images, axis=0)
            elif across_levels == 'max':
                combined_image = np.maximum.reduce(smoothed_images)
            else:
                raise ValueError(f"Across_levels method '{across_levels}' is not recognized.")
        else:
            combined_image = smoothed_images[0]

        # Reconstruct the mapping from (y_int, x_int) to embedding values
        smoothed_tile_values = combined_image[y_indices, x_indices]

        # Assign the smoothed values to the corresponding barcodes
        smoothed_barcode_grouped = pd.DataFrame({
            'barcode': tiles_shifted['barcode'],
            'smoothed_embed': smoothed_tile_values
        }).groupby('barcode', as_index=False)['smoothed_embed'].mean()

        return (dim, smoothed_barcode_grouped, combined_image)

    except Exception as e:
        logging.error(f"Error smoothing dimension {dim}: {e}")
        return (dim, pd.DataFrame(), np.array([]))  # Return dim, empty DataFrame, and empty array

def fill_nan_with_opencv_inpaint(image: np.ndarray, method: str = 'telea', inpaint_radius: int = 3) -> np.ndarray:
    """
    Fill NaN values in the image using OpenCV's inpainting.

    Parameters:
    - image: 2D numpy array with NaNs representing missing values.
    - method: Inpainting method ('telea' or 'ns').
    - inpaint_radius: Radius of circular neighborhood of each point inpainted.

    Returns:
    - image_filled: 2D numpy array with NaNs filled.
    """
    # Ensure the input is a float32 array for memory efficiency
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Create a boolean mask where NaNs are True
    mask = np.isnan(image)

    # If there are no NaNs, return the original image to save computation
    if not np.any(mask):
        return image

    # Convert the mask to uint8 format required by OpenCV (255 for NaNs, 0 otherwise)
    mask_uint8 = mask.astype(np.uint8) * 255

    # Replace NaNs with zero in place to avoid creating a copy
    image[mask] = 0

    # Find the minimum and maximum values in the image (excluding NaNs)
    image_min = image.min()
    image_max = image.max()

    # Handle the edge case where all non-NaN values are the same
    if image_max == image_min:
        # If image_max equals image_min, normalization is not needed
        image_normalized = np.zeros_like(image, dtype=np.uint8)
    else:
        # Normalize the image to the range [0, 255] using OpenCV's efficient normalize function
        image_normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image_normalized = image_normalized.astype(np.uint8)
    del image
    gc.collect()

    # Select the appropriate inpainting method
    if method == 'telea':
        cv_method = cv2.INPAINT_TELEA
    elif method == 'ns':
        cv_method = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unknown inpainting method '{method}'. Choose 'telea' or 'ns'.")

    # Perform inpainting using OpenCV's optimized function
    image_inpainted = cv2.inpaint(image_normalized, mask_uint8, inpaint_radius, cv_method)
    del image_normalized
    gc.collect()

    # Convert the inpainted image back to the original scale
    if image_max == image_min:
        # If no normalization was done, simply add the min value back
        image_filled = image_inpainted.astype(np.float32) + image_min
    else:
        # Scale the inpainted image back to the original range
        scale = (image_max - image_min) / 255.0
        image_filled = image_inpainted.astype(np.float32) * scale + image_min
    del image_inpainted
    gc.collect()

    return image_filled
    
    
def bubble_stack(data: np.ndarray, stack_size: int = 5, axis: int = 0) -> np.ndarray:
    """
    Create a bubble stack by aggregating data across a specified axis.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    stack_size : int, optional (default=5)
        Number of consecutive slices to stack.
    axis : int, optional (default=0)
        Axis along which to stack the data.

    Returns
    -------
    np.ndarray
        Bubble stacked data array.
    """
    if stack_size < 1:
        raise ValueError("stack_size must be at least 1.")
    if axis < 0 or axis >= data.ndim:
        raise ValueError(f"axis must be between 0 and {data.ndim - 1}.")

    # Initialize list to hold stacked data
    stacked = []
    for i in range(data.shape[axis] - stack_size + 1):
        slices = [data.take(i + j, axis=axis) for j in range(stack_size)]
        stacked.append(np.stack(slices, axis=-1))
    return np.stack(stacked, axis=axis)
    
    
def create_pseudo_centroids(df, clusters, dimensions):
    """
    Creates pseudo centroids by replacing the values of the specified dimensions by the mean of the cluster.
    
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
    
def hex_grid(x_min: float, x_max: float, y_min: float, y_max: float, spacing: float) -> np.ndarray:
    """
    Generate a hexagonal grid within specified bounds.

    Parameters
    ----------
    x_min : float
        Minimum x-coordinate.
    x_max : float
        Maximum x-coordinate.
    y_min : float
        Minimum y-coordinate.
    y_max : float
        Maximum y-coordinate.
    spacing : float
        Spacing between hexagon centers.

    Returns
    -------
    np.ndarray
        Array of hexagon center coordinates.
    """
    dx = spacing * 3/2
    dy = spacing * np.sqrt(3)
    points = []
    y = y_min
    row = 0
    while y <= y_max:
        x_offset = spacing * 0.75 if row % 2 else 0
        x = x_min + x_offset
        while x <= x_max:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
    return np.array(points)

def random_sampling(data: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """
    Randomly sample a subset of the data.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    sample_size : int, optional (default=1000)
        Number of samples to draw.

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame.
    """
    return data.sample(n=sample_size, random_state=42)

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
