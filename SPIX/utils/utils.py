# SPIX/utils.py

import os
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
from shapely.vectorized import contains as shapely_contains
from shapely.geometry import Polygon
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import randomized_svd
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

_GLOBAL_VOR: Voronoi = None
_GLOBAL_COORDS_VALUES: np.ndarray = None
_GLOBAL_COORDS_INDEX: List = None
_GLOBAL_BOUNDARY: Optional[Polygon] = None
# Rasterisation sampling globals (configured by rasterise initializer)
_GLOBAL_RASTER_STRIDE: int = 1
_GLOBAL_RASTER_SAMPLE_FRAC: Optional[float] = None
_GLOBAL_RASTER_MAX_PER_TILE: Optional[int] = None
_GLOBAL_RASTER_RNG: Optional[np.random.Generator] = None
_GLOBAL_RASTER_RESTORE_EMPTY: bool = True

def _init_filter_worker(vor: Voronoi, coords_values: np.ndarray, coords_index: List):
    """
    Worker initializer: store shared Voronoi and coordinate data in globals.
    """
    global _GLOBAL_VOR, _GLOBAL_COORDS_VALUES, _GLOBAL_COORDS_INDEX
    _GLOBAL_VOR = vor
    _GLOBAL_COORDS_VALUES = coords_values
    _GLOBAL_COORDS_INDEX = coords_index

def _compute_region_area(args: Tuple[int, int]):
    """
    Compute area of a single Voronoi region and return
    (area, region_vertices, point, barcode). Returns None on invalid region.
    """
    idx, region_index = args
    vor = _GLOBAL_VOR
    regions = vor.regions
    vertices = vor.vertices

    # invalid region
    if region_index < 0 or region_index >= len(regions):
        return None
    region = regions[region_index]
    if -1 in region or len(region) == 0:
        return None

    # compute polygon area
    poly = Polygon(vertices[region])
    area = poly.area

    # grab original point & its barcode/index
    point = _GLOBAL_COORDS_VALUES[idx]
    barcode = _GLOBAL_COORDS_INDEX[idx]

    return area, region, point, barcode
    
def _init_raster_worker(
    vor: Voronoi,
    boundary: Optional[Polygon] = None,
    raster_stride: int = 1,
    raster_sample_frac: Optional[float] = None,
    raster_max_per_tile: Optional[int] = None,
    random_seed: Optional[int] = None,
    restore_zero_pixel_tiles: bool = True,
):
    """
    Worker initializer: store the shared Voronoi object in a global variable.
    This avoids pickling it for every single tile.
    """
    global _GLOBAL_VOR, _GLOBAL_BOUNDARY
    global _GLOBAL_RASTER_STRIDE, _GLOBAL_RASTER_SAMPLE_FRAC, _GLOBAL_RASTER_MAX_PER_TILE, _GLOBAL_RASTER_RNG
    global _GLOBAL_RASTER_RESTORE_EMPTY
    _GLOBAL_VOR = vor
    _GLOBAL_BOUNDARY = boundary
    _GLOBAL_RASTER_RESTORE_EMPTY = bool(restore_zero_pixel_tiles)
    # Configure sampling controls
    try:
        _GLOBAL_RASTER_STRIDE = int(raster_stride) if raster_stride and raster_stride > 0 else 1
    except Exception:
        _GLOBAL_RASTER_STRIDE = 1
    _GLOBAL_RASTER_SAMPLE_FRAC = None
    if raster_sample_frac is not None:
        try:
            # clamp to (0,1]
            f = float(raster_sample_frac)
            if f > 0 and f <= 1:
                _GLOBAL_RASTER_SAMPLE_FRAC = f if f < 1 else None
        except Exception:
            _GLOBAL_RASTER_SAMPLE_FRAC = None
    _GLOBAL_RASTER_MAX_PER_TILE = None
    if raster_max_per_tile is not None:
        try:
            m = int(raster_max_per_tile)
            if m > 0:
                _GLOBAL_RASTER_MAX_PER_TILE = m
        except Exception:
            _GLOBAL_RASTER_MAX_PER_TILE = None
    # RNG for subsampling
    seed = None if random_seed is None else int(random_seed)
    try:
        _GLOBAL_RASTER_RNG = np.random.default_rng(None if seed is None else (seed + os.getpid()))
    except Exception:
        _GLOBAL_RASTER_RNG = np.random.default_rng()

def _rasterise_chunk(
    chunk_args: list[tuple[list[int], np.ndarray, int]],
) -> pd.DataFrame:
    """
    Process a batch of tiles in one go.
    This reduces the number of pickles between main <-> worker.
    """
    x_parts = []
    y_parts = []
    barcode_parts = []
    origin_parts = []
    for region, point, idx in chunk_args:
        barcode = str(idx)
        # exactly the same per‐tile logic as before
        try:
            verts = _GLOBAL_VOR.vertices[region]
            poly = Polygon(verts)
            if _GLOBAL_BOUNDARY is not None:
                poly = poly.intersection(_GLOBAL_BOUNDARY)
                if poly.is_empty:
                    continue

            minx, miny, maxx, maxy = poly.bounds
            # Step grid by stride for coarser sampling
            sx = max(1, int(_GLOBAL_RASTER_STRIDE))
            sy = sx
            xs = np.arange(int(np.floor(minx)), int(np.ceil(maxx)) + 1, sx)
            ys = np.arange(int(np.floor(miny)), int(np.ceil(maxy)) + 1, sy)
            gx, gy = np.meshgrid(xs, ys)
            pts = np.vstack([gx.ravel(), gy.ravel()]).T

            mask = shapely_contains(poly, pts[:, 0], pts[:, 1])
            pixels = pts[mask]
            ox, oy = int(round(point[0])), int(round(point[1]))
            if pixels.size == 0:
                # Optionally keep only the origin pixel
                if _GLOBAL_RASTER_RESTORE_EMPTY:
                    x_parts.append(np.array([ox], dtype=np.int64))
                    y_parts.append(np.array([oy], dtype=np.int64))
                    barcode_parts.append(np.array([barcode], dtype=object))
                    origin_parts.append(np.array([1], dtype=np.int8))
                continue
            # Optional subsampling within the tile
            if _GLOBAL_RASTER_SAMPLE_FRAC is not None and pixels.shape[0] > 0:
                try:
                    rng = _GLOBAL_RASTER_RNG or np.random.default_rng()
                    keep_n = max(1, int(np.floor(_GLOBAL_RASTER_SAMPLE_FRAC * pixels.shape[0])))
                    if keep_n < pixels.shape[0]:
                        sample_idx = rng.choice(pixels.shape[0], size=keep_n, replace=False)
                        pixels = pixels[sample_idx]
                except Exception:
                    pass

            if _GLOBAL_RASTER_MAX_PER_TILE is not None and pixels.shape[0] > _GLOBAL_RASTER_MAX_PER_TILE:
                try:
                    rng = _GLOBAL_RASTER_RNG or np.random.default_rng()
                    sample_idx = rng.choice(
                        pixels.shape[0], size=_GLOBAL_RASTER_MAX_PER_TILE, replace=False
                    )
                    pixels = pixels[sample_idx]
                except Exception:
                    pixels = pixels[:_GLOBAL_RASTER_MAX_PER_TILE]

            origin_flags = ((pixels[:, 0] == ox) & (pixels[:, 1] == oy)).astype(np.int8, copy=False)
            if not np.any(origin_flags):
                pixels = np.vstack([pixels, np.array([[ox, oy]], dtype=pixels.dtype)])
                origin_flags = np.concatenate([origin_flags, np.array([1], dtype=np.int8)])

            x_parts.append(pixels[:, 0])
            y_parts.append(pixels[:, 1])
            barcode_parts.append(np.full(pixels.shape[0], barcode, dtype=object))
            origin_parts.append(origin_flags)
        except Exception as e:
            logging.error(f"Error in tile {barcode}: {e}")

    if x_parts:
        return pd.DataFrame(
            {
                "x": np.concatenate(x_parts),
                "y": np.concatenate(y_parts),
                "barcode": np.concatenate(barcode_parts),
                "origin": np.concatenate(origin_parts),
            }
        )
    else:
        # return empty with correct columns
        return pd.DataFrame(columns=["x", "y", "barcode", "origin"])

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


def reduce_tensor_resolution(
    coordinates: pd.DataFrame, tensor_resolution: float = 1
) -> pd.DataFrame:
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
    # coordinates = coordinates.copy()
    coordinates = coordinates.copy()
    if 0 not in coordinates.columns or 1 not in coordinates.columns:
        cols = list(coordinates.columns[:2])
        coordinates.rename(columns={cols[0]: 0, cols[1]: 1}, inplace=True)
        
    coordinates[0] = np.round(coordinates[0] * tensor_resolution) + 1
    coordinates[1] = np.round(coordinates[1] * tensor_resolution) + 1

    # Group by reduced coordinates and compute mean
    coords_df = coordinates.rename(columns={0: "x", 1: "y"})
    coords_df["coords"] = coords_df["x"].astype(str) + "_" + coords_df["y"].astype(str)
    coords_df["original_index"] = coords_df.index

    coords_grouped = coords_df.groupby("coords").mean(numeric_only=True).reset_index()
    coords_grouped["original_index"] = (
        coords_df.groupby("coords")["original_index"].first().values
    )
    coords_grouped.set_index("original_index", inplace=True)
    filtered_coordinates = coords_grouped[["x", "y"]]

    return filtered_coordinates


def filter_tiles(
    vor: Voronoi,
    coordinates: pd.DataFrame,
    filter_threshold: float,
    n_jobs: int = None,
    chunksize: int = None,
    verbose: bool = True,
) -> Tuple[List[List[int]], np.ndarray, List]:
    """
    Parallel filtering of Voronoi regions by area threshold.

    Parameters
    ----------
    vor : Voronoi
        Voronoi tessellation object.
    coordinates : pd.DataFrame
        Spatial coordinates with columns ['x','y'] and index=barcodes.
    filter_threshold : float
        Quantile threshold (e.g. 0.995) to cut large regions.
    n_jobs : int, optional
        Number of processes (None → os.cpu_count()).
    chunksize : int, optional
        Tasks per chunk for executor.map (None → total//(n_jobs*4)).
    verbose : bool, optional
        Print progress logs.

    Returns
    -------
    filtered_regions : List[List[int]]
    filtered_points : np.ndarray  # shape=(M,2)
    filtered_index : List        # length=M
    """
    # ── determine n_jobs ────────────────────────────────────────────────
    if n_jobs is None:
        n_jobs = os.cpu_count()
    elif n_jobs < 0:
        n_jobs = max(1, os.cpu_count() + n_jobs)
    elif n_jobs == 0:
        n_jobs = 1

    total = len(vor.point_region)

    # ── determine chunksize ─────────────────────────────────────────────
    if chunksize is None:
        chunksize = max(1, total // (n_jobs * 4))

    if verbose:
        logging.info(f"filter_tiles: n_jobs={n_jobs}, chunksize={chunksize}")

    # ── prepare data for workers ───────────────────────────────────────
    coords_values = coordinates.values
    coords_index = coordinates.index.tolist()
    tasks = list(enumerate(vor.point_region))

    # ── parallel area computation ──────────────────────────────────────
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_init_filter_worker,
        initargs=(vor, coords_values, coords_index),
    ) as executor:
        results = list(
            tqdm(
                executor.map(_compute_region_area, tasks, chunksize=chunksize),
                total=total,
                desc="Calculating Voronoi region areas",
            )
        )

    # ── collect valid results ──────────────────────────────────────────
    valid = [r for r in results if r is not None]
    if not valid:
        return [], np.array([]), []

    areas, regions, points, indices = zip(*valid)
    areas = np.array(areas)

    # ── compute threshold and filter ───────────────────────────────────
    max_area = np.quantile(areas, filter_threshold)
    mask = areas <= max_area

    filtered_regions = [reg for reg, m in zip(regions, mask) if m]
    filtered_points = np.array([pt for pt, m in zip(points, mask) if m])
    filtered_index = [idx for idx, m in zip(indices, mask) if m]

    if verbose:
        logging.info(
            f"filter_tiles: kept {len(filtered_regions)}/{total} regions (≤{filter_threshold*100:.1f}th pctile)."
        )

    return filtered_regions, filtered_points, filtered_index


def rasterise(
    filtered_regions: list[list[int]],
    filtered_points: np.ndarray,
    index: list,
    vor: Voronoi,
    boundary: Optional[Polygon] = None,
    chunksize: int = None,
    n_jobs: int = None,
    *,
    raster_stride: int = 1,
    raster_sample_frac: Optional[float] = None,
    raster_max_pixels_per_tile: Optional[int] = None,
    raster_random_seed: Optional[int] = None,
    restore_zero_pixel_tiles: bool = True,
) -> pd.DataFrame:
    """
    Rasterize Voronoi regions into pixels, in parallel by batch.

    Parameters
    ----------
    filtered_regions : list of list of int
        Vertex indices of each region.
    filtered_points : np.ndarray
        Nx2 array of original point coords.
    index : list
        Region identifiers.
    vor : Voronoi
        Shared tessellation.
    boundary : Polygon or None
        Optional polygon to clip each tile. If None, tiles are not clipped.
    chunksize : int, optional
        Number of tiles per worker‐job. None -> auto = total/(n_jobs*4).
    n_jobs : int or None
        Number of processes. None -> all CPUs.

    Returns
    -------
    pd.DataFrame
        All pixels from all tiles.
    """
    # ―――― determine n_jobs ――――
    if n_jobs is None:
        n_jobs = os.cpu_count()
    elif n_jobs < 0:
        n_jobs = max(1, os.cpu_count() + n_jobs)
    elif n_jobs == 0:
        n_jobs = 1

    total = len(filtered_regions)
    # ―――― determine chunksize ――――
    if chunksize is None:
        chunksize = max(1, total // (n_jobs * 4))

    logging.info(
        f"Rasterise: n_jobs={n_jobs}, tiles={total}, chunksize={chunksize}, stride={raster_stride}, sample_frac={raster_sample_frac}, max_per_tile={raster_max_pixels_per_tile}"
    )

    total_batches = len(range(0, total, chunksize))

    def _iter_batches():
        for start in range(0, total, chunksize):
            stop = start + chunksize
            yield list(zip(filtered_regions[start:stop], filtered_points[start:stop], index[start:stop]))

    results = []
    desc = "Rasterising tiles"

    if n_jobs == 1:
        # serial fallback
        _init_raster_worker(
            vor,
            boundary,
            raster_stride=raster_stride,
            raster_sample_frac=raster_sample_frac,
            raster_max_per_tile=raster_max_pixels_per_tile,
            random_seed=raster_random_seed,
            restore_zero_pixel_tiles=restore_zero_pixel_tiles,
        )
        for batch in tqdm(_iter_batches(), total=total_batches, desc=desc):
            results.append(_rasterise_chunk(batch))
    else:
        # parallel with one init per worker
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_raster_worker,
            initargs=(
                vor,
                boundary,
                raster_stride,
                raster_sample_frac,
                raster_max_pixels_per_tile,
                raster_random_seed,
                restore_zero_pixel_tiles,
            ),
        ) as exe:
            for df in tqdm(
                exe.map(_rasterise_chunk, _iter_batches()), total=total_batches, desc=desc
            ):
                results.append(df)

    # concat all batch‐DFs
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["x", "y", "barcode", "origin"])


def _process_in_parallel_map(
    function, iterable, desc, n_jobs, chunksize: int = 1000, total: int = None
) -> List[pd.DataFrame]:
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
            for result in tqdm(
                executor.map(function, iterable, chunksize=chunksize),
                desc=desc,
                total=total,
            ):
                results.append(result)
    return results

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
        # Prefer CSR for row operations; SVD accepts CSR/CSC
        counts = counts.tocsr()
        row_sums = np.asarray(counts.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        inv_row = 1.0 / row_sums
        tf = counts.multiply(inv_row[:, None])
        df = (counts > 0).astype(np.int32).sum(axis=0).A1
        idf = np.log(1 + counts.shape[0] / (1 + df))
        tfidf = tf.multiply(idf).tocsr()
    else:
        tf = counts / counts.sum(axis=1, keepdims=True)
        df = (counts > 0).sum(axis=0)
        idf = np.log(1 + counts.shape[0] / (1 + df))
        tfidf = tf * idf

    U, Sigma, VT = randomized_svd(tfidf, n_components=n_components + 1)

    if remove_first:
        U = U[:, 1:]
        Sigma = Sigma[1:]
        VT = VT[1:, :]

    embeddings = np.dot(U, np.diag(Sigma))
    embeddings = MinMaxScaler().fit_transform(embeddings)

    return embeddings


def rebalance_colors(coordinates, dimensions, method="minmax", vmin=None, vmax=None):
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
    vmin, vmax : float or array-like, optional
        If provided, apply a fixed min/max scaling instead of computing min/max
        from the passed-in (possibly cropped) coordinates. For 3D, these can be
        length-3 arrays for per-channel scaling.

    Returns
    -------
    pd.DataFrame
        DataFrame with rebalanced colors.
    """
    def _fixed_scale(arr: np.ndarray, vmin_, vmax_) -> np.ndarray:
        arr = arr.astype(float, copy=False)
        arr_stats = arr.copy()
        arr_stats[~np.isfinite(arr_stats)] = np.nan
        if vmin_ is None:
            vmin_ = np.nanmin(arr_stats, axis=0)
        if vmax_ is None:
            vmax_ = np.nanmax(arr_stats, axis=0)
        vmin_arr = np.asarray(vmin_, dtype=float)
        vmax_arr = np.asarray(vmax_, dtype=float)
        denom = (vmax_arr - vmin_arr) + 1e-10
        scaled = (arr - vmin_arr) / denom
        return np.clip(scaled, 0.0, 1.0)

    if len(dimensions) == 3:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        if all(col in coordinates.columns for col in ["dim0", "dim1", "dim2"]):
            colors = coordinates.loc[:, ["dim0", "dim1", "dim2"]].to_numpy(dtype=float, copy=False)
        else:
            colors = coordinates.iloc[:, [4, 5, 6]].values

        if (vmin is not None) or (vmax is not None):
            colors = _fixed_scale(colors, vmin, vmax)
        elif method == "minmax":
            colors = np.apply_along_axis(min_max, 0, colors)
        else:
            colors = np.clip(colors, 0, 1)

        template = pd.concat(
            [template, pd.DataFrame(colors, columns=["R", "G", "B"], index=template.index)],
            axis=1,
        )

    else:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        if "dim0" in coordinates.columns:
            colors = coordinates.loc[:, "dim0"].to_numpy(dtype=float, copy=False)
        else:
            colors = coordinates.iloc[:, 4].values

        if (vmin is not None) or (vmax is not None):
            colors = _fixed_scale(np.asarray(colors), vmin, vmax).astype(float)
        elif method == "minmax":
            colors = min_max(colors)
        else:
            colors = np.clip(colors, 0, 1)

        template = pd.concat(
            [template, pd.Series(colors, name="Grey", index=template.index)],
            axis=1,
        )

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
    arr = np.asarray(values, dtype=float)
    arr_stats = arr.copy()
    arr_stats[~np.isfinite(arr_stats)] = np.nan
    vmin = np.nanmin(arr_stats)
    vmax = np.nanmax(arr_stats)
    return (arr - vmin) / (vmax - vmin + 1e-10)


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

import numpy as np
import pandas as pd


# Untested!!!! Will need to check this function first
# Pushing to main so you can see it.
def block_permutation_test(
    adata,
    gene,
    cell_type_col='cell_type',
    block_col='block',
    n_permutations=1000,
    stat_func=None,
    random_state=0
):
    """
    Test for cell-type-specific expression of a gene using block-aware permutation tests.

    Parameters
    ----------
    adata : AnnData
        Object containing .X (expression) and .obs with cell type and block columns.
    gene : str
        Gene name to test.
    cell_type_col : str
        Column name in adata.obs for cell-type labels.
    block_col : str
        Column name in adata.obs for block labels.
    n_permutations : int
        Number of block-permutations to run.
    stat_func : callable, optional
        Custom statistic function f(expr, is_type) -> float.
        Default: mean(expr[is_type]) - mean(expr[~is_type]).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ['cell_type', 'observed', 'pval', 'zscore']
    """
    rng = np.random.default_rng(random_state)

    expr = adata[:, gene].X
    if hasattr(expr, "toarray"):  # handle sparse matrix
        expr = expr.toarray().flatten()
    else:
        expr = np.asarray(expr).flatten()

    cell_types = adata.obs[cell_type_col].values
    blocks = adata.obs[block_col].values
    unique_types = np.unique(cell_types)
    unique_blocks = np.unique(blocks)

    # default statistic: mean difference
    if stat_func is None:
        def stat_func(expr, mask):
            return np.mean(expr[mask]) - np.mean(expr[~mask])

    results = []

    for ct in unique_types:
        mask = (cell_types == ct)
        observed = stat_func(expr, mask)

        perm_stats = np.zeros(n_permutations)
        for i in range(n_permutations):
            perm_labels = cell_types.copy()
            # shuffle labels within each block
            for b in unique_blocks:
                idx = np.where(blocks == b)[0]
                perm_labels[idx] = rng.permutation(perm_labels[idx])
            mask_perm = (perm_labels == ct)
            perm_stats[i] = stat_func(expr, mask_perm)

        # two-sided p-value
        pval = np.mean(np.abs(perm_stats) >= np.abs(observed))
        zscore = (observed - np.mean(perm_stats)) / np.std(perm_stats)
        results.append((ct, observed, pval, zscore))

    return pd.DataFrame(results, columns=['cell_type', 'observed', 'pval', 'zscore'])


def detect_spatial_coordinates(
    adata: AnnData,
    *,
    prefer_obs_um: bool = True,
) -> tuple[np.ndarray, str]:
    """
    Resolve 2D spatial coordinates from an AnnData object.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates stored in ``.obs`` or ``.obsm``.
    prefer_obs_um
        When True, prioritize micron-like coordinates from ``.obs`` before
        normalized coordinates in ``.obsm['spatial']``.

    Returns
    -------
    (coords, source)
        ``coords`` is an ``(n_obs, 2)`` float array and ``source`` describes
        the chosen field.
    """
    obs_candidates = [
        ("Centroid_X", "Centroid_Y"),
        ("center_x", "center_y"),
        ("x_centroid", "y_centroid"),
        ("x", "y"),
        ("X", "Y"),
    ]
    obsm_candidates = ["spatial", "X_spatial", "spatial3d"]

    def _from_obs() -> tuple[np.ndarray, str] | None:
        for x_col, y_col in obs_candidates:
            if x_col in adata.obs.columns and y_col in adata.obs.columns:
                arr = adata.obs[[x_col, y_col]].to_numpy(dtype=float, copy=True)
                if arr.ndim == 2 and arr.shape[1] == 2 and np.isfinite(arr).any():
                    return arr, f"obs:{x_col},{y_col}"
        return None

    def _from_obsm() -> tuple[np.ndarray, str] | None:
        for key in obsm_candidates:
            if key not in adata.obsm:
                continue
            arr = np.asarray(adata.obsm[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2 and np.isfinite(arr).any():
                return np.asarray(arr[:, :2], dtype=float), f"obsm:{key}"
        return None

    ordered = (_from_obs, _from_obsm) if prefer_obs_um else (_from_obsm, _from_obs)
    for resolver in ordered:
        out = resolver()
        if out is not None:
            coords, source = out
            valid = np.isfinite(coords).all(axis=1)
            if not np.any(valid):
                raise ValueError("Resolved spatial coordinates but none are finite.")
            return coords[valid], source

    raise ValueError("Could not locate 2D spatial coordinates in AnnData.")


def estimate_effective_pitch_um(
    adata: Optional[AnnData] = None,
    *,
    coords: Optional[np.ndarray] = None,
    support_method: str = "occupancy_grid",
    grid_size_um: Optional[float] = None,
    alpha: Optional[float] = None,
    sample_size: int = 50000,
    prefer_obs_um: bool = True,
    return_details: bool = False,
) -> float | dict:
    """
    Estimate an effective pitch for cell-based spatial data such as MERFISH/Xenium.

    The effective pitch is defined as ``sqrt(A_support / N)``, where ``A_support``
    is an estimated tissue support area in square microns and ``N`` is the number
    of spatial observations. This is useful when observations are cells rather
    than regular lattice spots.

    Parameters
    ----------
    adata
        AnnData containing spatial coordinates. Required when ``coords`` is not
        provided.
    coords
        Optional explicit ``(n_obs, 2)`` coordinate array in microns.
    support_method
        One of ``'occupancy_grid'``, ``'convex_hull'``, ``'alpha_shape'``,
        or ``'bbox'``.
    grid_size_um
        Grid size for ``support_method='occupancy_grid'``. If omitted, the median
        nearest-neighbor distance is used.
    alpha
        Alpha parameter for ``support_method='alpha_shape'``. When omitted, a
        heuristic ``1 / median_nn`` is used.
    sample_size
        Maximum number of points used for nearest-neighbor estimation.
    prefer_obs_um
        Passed to :func:`detect_spatial_coordinates` when ``coords`` is not given.
    return_details
        When True, return a dictionary with intermediate estimates.

    Returns
    -------
    float or dict
        Effective pitch in microns, or a detailed result dictionary.
    """
    if coords is None:
        if adata is None:
            raise ValueError("Provide either `adata` or `coords`.")
        coords, coord_source = detect_spatial_coordinates(adata, prefer_obs_um=prefer_obs_um)
    else:
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("`coords` must be a 2D array with at least two columns.")
        coords = np.asarray(coords[:, :2], dtype=float)
        coords = coords[np.isfinite(coords).all(axis=1)]
        coord_source = "coords"

    n_obs = int(coords.shape[0])
    if n_obs < 2:
        raise ValueError("Need at least two spatial observations to estimate effective pitch.")

    coords_nn = coords
    if coords_nn.shape[0] > int(sample_size):
        rng = np.random.default_rng(42)
        idx = rng.choice(coords_nn.shape[0], size=int(sample_size), replace=False)
        coords_nn = coords_nn[idx]

    tree = KDTree(coords_nn)
    dists, _ = tree.query(coords_nn, k=2)
    nn = np.asarray(dists[:, 1], dtype=float)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        raise ValueError("Could not compute a positive nearest-neighbor distance.")

    median_nn_um = float(np.median(nn))
    mean_nn_um = float(np.mean(nn))

    method = str(support_method).lower()
    support_area_um2 = None
    support_meta = {}

    if method == "occupancy_grid":
        cell_um = float(grid_size_um) if grid_size_um is not None else float(median_nn_um)
        cell_um = max(cell_um, 1e-9)
        gx = np.floor(coords[:, 0] / cell_um).astype(np.int64)
        gy = np.floor(coords[:, 1] / cell_um).astype(np.int64)
        occupied = pd.DataFrame({"gx": gx, "gy": gy}).drop_duplicates().shape[0]
        support_area_um2 = float(occupied) * float(cell_um ** 2)
        support_meta = {"grid_size_um": cell_um, "occupied_bins": int(occupied)}
    elif method == "convex_hull":
        hull = MultiPoint(coords).convex_hull
        support_area_um2 = float(hull.area)
    elif method == "alpha_shape":
        alpha_eff = float(alpha) if alpha is not None else float(1.0 / max(median_nn_um, 1e-9))
        poly = alphashape.alphashape(coords, alpha_eff)
        if poly is None or getattr(poly, "is_empty", False):
            raise ValueError("Alpha-shape support area estimation failed.")
        support_area_um2 = float(poly.area)
        support_meta = {"alpha": alpha_eff}
    elif method == "bbox":
        xmin, ymin = np.min(coords, axis=0)
        xmax, ymax = np.max(coords, axis=0)
        support_area_um2 = float(max(xmax - xmin, 0.0) * max(ymax - ymin, 0.0))
    else:
        raise ValueError(
            "Unknown support_method. Use one of: "
            "'occupancy_grid', 'convex_hull', 'alpha_shape', 'bbox'."
        )

    if not np.isfinite(support_area_um2) or support_area_um2 <= 0:
        raise ValueError("Estimated support area is non-positive.")

    effective_pitch_um = float(np.sqrt(support_area_um2 / float(n_obs)))
    density_cells_per_um2 = float(n_obs / support_area_um2)

    details = {
        "effective_pitch_um": effective_pitch_um,
        "support_area_um2": float(support_area_um2),
        "density_cells_per_um2": density_cells_per_um2,
        "n_obs": int(n_obs),
        "median_nn_um": median_nn_um,
        "mean_nn_um": mean_nn_um,
        "support_method": method,
        "coord_source": coord_source,
        **support_meta,
    }

    return details if return_details else effective_pitch_um


def estimate_median_nn_pitch_um(
    adata: Optional[AnnData] = None,
    *,
    coords: Optional[np.ndarray] = None,
    sample_size: int = 50000,
    prefer_obs_um: bool = True,
    return_details: bool = False,
) -> float | dict:
    """
    Estimate pitch from the median nearest-neighbor distance.

    This is most appropriate for grid-like spatial data such as VisiumHD or
    Stereo-seq binned coordinates.
    """
    if coords is None:
        if adata is None:
            raise ValueError("Provide either `adata` or `coords`.")
        coords, coord_source = detect_spatial_coordinates(adata, prefer_obs_um=prefer_obs_um)
    else:
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("`coords` must be a 2D array with at least two columns.")
        coords = np.asarray(coords[:, :2], dtype=float)
        coords = coords[np.isfinite(coords).all(axis=1)]
        coord_source = "coords"

    if coords.shape[0] < 2:
        raise ValueError("Need at least two spatial observations to estimate pitch.")

    coords_nn = coords
    if coords_nn.shape[0] > int(sample_size):
        rng = np.random.default_rng(42)
        idx = rng.choice(coords_nn.shape[0], size=int(sample_size), replace=False)
        coords_nn = coords_nn[idx]

    tree = KDTree(coords_nn)
    dists, _ = tree.query(coords_nn, k=2)
    nn = np.asarray(dists[:, 1], dtype=float)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        raise ValueError("Could not compute a positive nearest-neighbor distance.")

    median_nn_um = float(np.median(nn))
    mean_nn_um = float(np.mean(nn))
    nn_cv = float(np.std(nn) / max(mean_nn_um, 1e-12))

    details = {
        "pitch_um": median_nn_um,
        "median_nn_um": median_nn_um,
        "mean_nn_um": mean_nn_um,
        "nn_cv": nn_cv,
        "n_obs": int(coords.shape[0]),
        "coord_source": coord_source,
    }
    return details if return_details else median_nn_um


def infer_spatial_sampling_kind(
    adata: Optional[AnnData] = None,
    *,
    coords: Optional[np.ndarray] = None,
    sample_size: int = 50000,
    prefer_obs_um: bool = True,
    return_details: bool = False,
) -> str | dict:
    """
    Infer whether a spatial dataset is more grid-like or cell-like.

    Heuristics:
    - centroid-style obs columns strongly indicate cell-based data
    - array row/col style columns strongly indicate grid-like data
    - otherwise fall back to nearest-neighbor regularity
    """
    obs_columns = set()
    if adata is not None:
        obs_columns = set(map(str, adata.obs.columns))

    centroid_pairs = {
        ("Centroid_X", "Centroid_Y"),
        ("center_x", "center_y"),
        ("x_centroid", "y_centroid"),
    }
    grid_markers = {
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
        "in_tissue",
    }

    for x_col, y_col in centroid_pairs:
        if x_col in obs_columns and y_col in obs_columns:
            details = {"kind": "cell", "reason": f"obs centroid columns: {x_col},{y_col}"}
            return details if return_details else "cell"

    if any(marker in obs_columns for marker in grid_markers):
        hit = sorted(marker for marker in grid_markers if marker in obs_columns)
        details = {"kind": "grid", "reason": f"grid marker columns: {','.join(hit)}"}
        return details if return_details else "grid"

    nn_details = estimate_median_nn_pitch_um(
        adata,
        coords=coords,
        sample_size=sample_size,
        prefer_obs_um=prefer_obs_um,
        return_details=True,
    )
    nn_cv = float(nn_details["nn_cv"])
    kind = "grid" if nn_cv <= 0.25 else "cell"
    details = {
        "kind": kind,
        "reason": f"nearest-neighbor regularity heuristic (nn_cv={nn_cv:.4f})",
        **nn_details,
    }
    return details if return_details else kind


def resolve_pitch_um(
    adata: Optional[AnnData] = None,
    *,
    coords: Optional[np.ndarray] = None,
    platform: Optional[str] = "auto",
    grid_pitch_um: Optional[float] = None,
    support_method: str = "occupancy_grid",
    grid_size_um: Optional[float] = None,
    alpha: Optional[float] = None,
    sample_size: int = 50000,
    prefer_obs_um: bool = True,
    return_details: bool = False,
) -> float | dict:
    """
    Convenience wrapper to resolve a usable ``pitch_um`` for cell-based platforms.

    Recommended usage
    -----------------
    - MERFISH / Xenium / CosMx: ``platform='merfish'`` / ``'xenium'`` / ``'cosmx'``
    - Unsure: ``platform='auto'``

    Rules
    -----
    - cell-like data -> effective pitch from support area
    - grid-like platforms are intentionally not handled here; pass explicit pitch
      for VisiumHD / Stereo-seq instead of using this helper.
    """
    platform_key = "auto" if platform is None else str(platform).strip().lower()
    aliases = {
        "merfish": "merfish",
        "merscope": "merfish",
        "xenium": "xenium",
        "cosmx": "cosmx",
        "cell": "cell",
        "auto": "auto",
    }
    platform_key = aliases.get(platform_key, platform_key)

    cell_platforms = {"merfish", "xenium", "cosmx", "cell"}
    unsupported_grid_platforms = {"visiumhd", "stereo-seq", "grid", "visium hd", "stereo seq", "stereoseq"}

    if platform_key in unsupported_grid_platforms:
        raise ValueError(
            "resolve_pitch_um() is now cell-platform-only. "
            "For VisiumHD / Stereo-seq, pass the known assay pitch explicitly."
        )
    elif platform_key in cell_platforms:
        kind = "cell"
        kind_reason = f"platform={platform_key}"
    elif platform_key == "auto":
        inferred = infer_spatial_sampling_kind(
            adata,
            coords=coords,
            sample_size=sample_size,
            prefer_obs_um=prefer_obs_um,
            return_details=True,
        )
        kind = str(inferred["kind"])
        kind_reason = str(inferred["reason"])
        if kind != "cell":
            raise ValueError(
                "resolve_pitch_um(auto) inferred grid-like sampling. "
                "For grid platforms such as VisiumHD / Stereo-seq, use the known assay pitch directly."
            )
    else:
        raise ValueError(
            "Unknown platform. Use one of: "
            "'auto', 'merfish', 'xenium', 'cosmx', 'cell'."
        )

    eff_details = estimate_effective_pitch_um(
        adata,
        coords=coords,
        support_method=support_method,
        grid_size_um=grid_size_um,
        alpha=alpha,
        sample_size=sample_size,
        prefer_obs_um=prefer_obs_um,
        return_details=True,
    )
    details = {
        "pitch_um": float(eff_details["effective_pitch_um"]),
        "kind": kind,
        "strategy": "effective_pitch",
        "platform": platform_key,
        "reason": kind_reason,
        **eff_details,
    }
    return details if return_details else float(eff_details["effective_pitch_um"])


def resolve_cell_target_segment_um(
    adata: Optional[AnnData] = None,
    *,
    coords: Optional[np.ndarray] = None,
    target_cells_per_segment: float,
    platform: Optional[str] = "auto",
    support_method: str = "occupancy_grid",
    grid_size_um: Optional[float] = None,
    alpha: Optional[float] = None,
    sample_size: int = 50000,
    prefer_obs_um: bool = True,
    return_details: bool = False,
) -> float | dict:
    """
    Convert a desired number of cells per segment into an approximate target size in microns.
    """
    target_cells = float(target_cells_per_segment)
    if not np.isfinite(target_cells) or target_cells <= 0:
        raise ValueError("target_cells_per_segment must be positive.")

    pitch_details = resolve_pitch_um(
        adata,
        coords=coords,
        platform=platform,
        support_method=support_method,
        grid_size_um=grid_size_um,
        alpha=alpha,
        sample_size=sample_size,
        prefer_obs_um=prefer_obs_um,
        return_details=True,
    )
    pitch_um = float(pitch_details["pitch_um"])
    target_segment_um = float(np.sqrt(target_cells) * pitch_um)

    details = {
        "target_segment_um": target_segment_um,
        "target_cells_per_segment": target_cells,
        **pitch_details,
    }
    return details if return_details else target_segment_um


def select_compactness_cells(
    adata: AnnData,
    *,
    target_cells_per_segment: Optional[float] = None,
    target_segment_um: Optional[float] = None,
    pitch_um: Optional[float] = None,
    platform: Optional[str] = "auto",
    support_method: str = "occupancy_grid",
    grid_size_um: Optional[float] = None,
    alpha: Optional[float] = None,
    lock_reference_target: bool = True,
    dimensions: Optional[list[int]] = None,
    search_dimensions: Optional[list[int]] = None,
    embedding: str = "X_embedding",
    origin: bool = True,
    compactness_candidates: Optional[list[float]] = None,
    compactness_search_jobs: int = 4,
    compactness_search_downsample_factor: float = 1.0,
    compactness_scale_power: float = 0.3,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Cell-platform convenience wrapper around ``SPIX.sp.fast_select_compactness``.

    Differences from the raw SPIX API:
    - can take ``target_cells_per_segment`` directly
    - resolves ``pitch_um`` automatically for Xenium/MERFISH-style data
    - by default locks ``search_target_segment_um`` to ``target_segment_um`` to
      avoid the auto-reference behavior that can collapse many scales to one
      reference target
    """
    import SPIX

    if (target_cells_per_segment is None) == (target_segment_um is None):
        raise ValueError("Provide exactly one of `target_cells_per_segment` or `target_segment_um`.")

    dims = list(range(30)) if dimensions is None else list(dimensions)

    pitch_details = None
    if pitch_um is None:
        pitch_details = resolve_pitch_um(
            adata,
            platform=platform,
            support_method=support_method,
            grid_size_um=grid_size_um,
            alpha=alpha,
            return_details=True,
        )
        pitch_um = float(pitch_details["pitch_um"])
    else:
        pitch_um = float(pitch_um)

    target_details = None
    if target_segment_um is None:
        target_details = resolve_cell_target_segment_um(
            adata,
            target_cells_per_segment=float(target_cells_per_segment),
            platform=platform,
            support_method=support_method,
            grid_size_um=grid_size_um,
            alpha=alpha,
            return_details=True,
        )
        target_segment_um = float(target_details["target_segment_um"])
    else:
        target_segment_um = float(target_segment_um)

    result = SPIX.sp.fast_select_compactness(
        adata,
        dimensions=dims,
        search_dimensions=search_dimensions,
        embedding=embedding,
        target_segment_um=float(target_segment_um),
        pitch_um=float(pitch_um),
        origin=origin,
        compactness_candidates=compactness_candidates,
        compactness_search_jobs=int(compactness_search_jobs),
        compactness_search_downsample_factor=float(compactness_search_downsample_factor),
        search_target_segment_um=float(target_segment_um) if lock_reference_target else None,
        compactness_scale_power=float(compactness_scale_power),
        verbose=verbose,
        **kwargs,
    )

    result = dict(result)
    result["pitch_um"] = float(pitch_um)
    result["target_segment_um"] = float(target_segment_um)
    result["target_cells_per_segment"] = None if target_cells_per_segment is None else float(target_cells_per_segment)
    result["lock_reference_target"] = bool(lock_reference_target)
    if pitch_details is not None:
        result["pitch_details"] = pitch_details
    if target_details is not None:
        result["target_details"] = target_details
    return result


def _cell_compactness_candidate_frame(selection_result: dict) -> pd.DataFrame:
    import SPIX

    frame = SPIX.sp.compactness_search_frame(selection_result)
    if frame.empty:
        return frame

    stage = frame["search_stage"].astype(str)
    refine_mask = stage.str.contains("refine", case=False, na=False)
    if bool(refine_mask.any()):
        frame = frame.loc[refine_mask].copy()
    else:
        coarse_mask = stage.str.contains("coarse", case=False, na=False)
        if bool(coarse_mask.any()):
            frame = frame.loc[coarse_mask].copy()
        else:
            frame = frame.copy()

    for col in (
        "compactness",
        "score",
        "fidelity_cost",
        "regularity_cost",
        "tradeoff_gain",
        "mean_log_aspect",
        "size_cv",
        "neighbor_disagreement",
    ):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame.sort_values("compactness", kind="mergesort").reset_index(drop=True)


def reselect_compactness_for_cells(
    selection_result: dict,
    *,
    score_relax: float = 0.15,
    fidelity_relax: float = 0.15,
    regularity_relax: float = 0.35,
    compactness_cap: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """
    Re-select a compactness value from an existing sweep using a cell-based rule.

    Logic:
    - keep candidates whose original score is near-best
    - keep candidates whose fidelity cost is near-best
    - allow some regularity degradation to avoid over-penalizing sparse cell centroids
    - among the remaining candidates, choose the smallest compactness

    This is intended for Xenium/MERFISH/CosMx-style cell centroids where the
    default tradeoff often drifts toward overly compact, rounded segments.
    """
    frame = _cell_compactness_candidate_frame(selection_result)
    if frame.empty:
        result = dict(selection_result)
        result["cell_reselection"] = {
            "applied": False,
            "reason": "empty_candidate_frame",
        }
        return result

    work = frame.copy()
    work["eligible"] = True

    def _slack(values: np.ndarray, frac: float, floor: float) -> float:
        finite = values[np.isfinite(values)]
        if finite.size <= 1:
            return float(floor)
        span = float(np.nanmax(finite) - np.nanmin(finite))
        return float(max(floor, float(frac) * span))

    score_vals = work["score"].to_numpy(dtype=float)
    if np.isfinite(score_vals).any():
        score_min = float(np.nanmin(score_vals))
        score_slack = _slack(score_vals, score_relax, 0.02)
        work["eligible"] &= work["score"] <= (score_min + score_slack)
    else:
        score_min = None
        score_slack = None

    fidelity_vals = work["fidelity_cost"].to_numpy(dtype=float) if "fidelity_cost" in work else np.array([], dtype=float)
    if fidelity_vals.size and np.isfinite(fidelity_vals).any():
        fidelity_min = float(np.nanmin(fidelity_vals))
        fidelity_slack = _slack(fidelity_vals, fidelity_relax, 0.02)
        work["eligible"] &= work["fidelity_cost"] <= (fidelity_min + fidelity_slack)
    else:
        fidelity_min = None
        fidelity_slack = None

    regularity_vals = work["regularity_cost"].to_numpy(dtype=float) if "regularity_cost" in work else np.array([], dtype=float)
    if regularity_vals.size and np.isfinite(regularity_vals).any():
        regularity_min = float(np.nanmin(regularity_vals))
        regularity_slack = _slack(regularity_vals, regularity_relax, 0.05)
        work["eligible"] &= work["regularity_cost"] <= (regularity_min + regularity_slack)
    else:
        regularity_min = None
        regularity_slack = None

    if compactness_cap is not None and np.isfinite(float(compactness_cap)):
        capped = work["compactness"] <= float(compactness_cap)
        if bool((work["eligible"] & capped).any()):
            work["eligible"] &= capped

    eligible = work.loc[work["eligible"]].copy()
    if eligible.empty:
        eligible = work.copy()

    chosen = eligible.sort_values(["compactness", "score"], kind="mergesort").iloc[0]
    selected_compactness = float(chosen["compactness"])
    selected_stage = str(chosen.get("search_stage", ""))

    result = dict(selection_result)
    result["base_selected_compactness"] = float(selection_result.get("selected_compactness", selected_compactness))
    result["selected_compactness"] = selected_compactness
    if "search_selected_compactness" in result:
        result["search_selected_compactness"] = selected_compactness
    for stage_key in ("coarse_search", "refine_search"):
        stage_detail = result.get(stage_key, None)
        if not isinstance(stage_detail, dict):
            continue
        if str(stage_detail.get("mode", stage_key)) == selected_stage:
            updated_stage = dict(stage_detail)
            updated_stage["base_selected_compactness"] = float(stage_detail.get("selected_compactness", selected_compactness))
            updated_stage["selected_compactness"] = float(selected_compactness)
            result[stage_key] = updated_stage
    result["cell_reselection"] = {
        "applied": True,
        "strategy": "prefer_smallest_near_best",
        "score_relax": float(score_relax),
        "fidelity_relax": float(fidelity_relax),
        "regularity_relax": float(regularity_relax),
        "compactness_cap": None if compactness_cap is None else float(compactness_cap),
        "candidate_count": int(len(work)),
        "eligible_count": int(len(eligible)),
        "selected_compactness": float(selected_compactness),
        "selected_stage": selected_stage,
        "base_selected_compactness": float(selection_result.get("selected_compactness", selected_compactness)),
        "selected_candidate": {
            "compactness": float(chosen["compactness"]),
            "score": None if not np.isfinite(float(chosen.get("score", np.nan))) else float(chosen["score"]),
            "fidelity_cost": None if not np.isfinite(float(chosen.get("fidelity_cost", np.nan))) else float(chosen["fidelity_cost"]),
            "regularity_cost": None if not np.isfinite(float(chosen.get("regularity_cost", np.nan))) else float(chosen["regularity_cost"]),
            "tradeoff_gain": None if not np.isfinite(float(chosen.get("tradeoff_gain", np.nan))) else float(chosen["tradeoff_gain"]),
            "mean_log_aspect": None if not np.isfinite(float(chosen.get("mean_log_aspect", np.nan))) else float(chosen["mean_log_aspect"]),
            "size_cv": None if not np.isfinite(float(chosen.get("size_cv", np.nan))) else float(chosen["size_cv"]),
            "neighbor_disagreement": None if not np.isfinite(float(chosen.get("neighbor_disagreement", np.nan))) else float(chosen["neighbor_disagreement"]),
        },
        "thresholds": {
            "score_min": score_min,
            "score_slack": score_slack,
            "fidelity_min": fidelity_min,
            "fidelity_slack": fidelity_slack,
            "regularity_min": regularity_min,
            "regularity_slack": regularity_slack,
        },
    }
    if verbose:
        base = float(selection_result.get("selected_compactness", selected_compactness))
        print(
            "[cell-compactness] reselection: "
            f"base={base:.6g} -> selected={selected_compactness:.6g} | "
            f"eligible={int(len(eligible))}/{int(len(work))}"
        )
    return result


def select_compactness_cells_robust(
    adata: AnnData,
    *,
    target_cells_per_segment: Optional[float] = None,
    target_segment_um: Optional[float] = None,
    pitch_um: Optional[float] = None,
    platform: Optional[str] = "auto",
    support_method: str = "occupancy_grid",
    grid_size_um: Optional[float] = None,
    alpha: Optional[float] = None,
    dimensions: Optional[list[int]] = None,
    search_dimensions: Optional[list[int]] = None,
    embedding: str = "X_embedding",
    origin: bool = True,
    compactness_candidates: Optional[list[float]] = None,
    compactness_search_jobs: int = 4,
    compactness_search_downsample_factor: float = 1.0,
    score_relax: float = 0.15,
    fidelity_relax: float = 0.15,
    regularity_relax: float = 0.35,
    compactness_cap: Optional[float] = None,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Exact compactness selection for cell-based platforms with a lower-compactness
    re-selection pass.

    This wrapper is intended for Xenium/MERFISH/CosMx where the default search
    often over-rewards geometric regularity after rasterization.
    """
    import SPIX

    if (target_cells_per_segment is None) == (target_segment_um is None):
        raise ValueError("Provide exactly one of `target_cells_per_segment` or `target_segment_um`.")

    dims = list(range(30)) if dimensions is None else list(dimensions)

    pitch_details = None
    if pitch_um is None:
        pitch_details = resolve_pitch_um(
            adata,
            platform=platform,
            support_method=support_method,
            grid_size_um=grid_size_um,
            alpha=alpha,
            return_details=True,
        )
        pitch_um = float(pitch_details["pitch_um"])
    else:
        pitch_um = float(pitch_um)

    target_details = None
    if target_segment_um is None:
        target_details = resolve_cell_target_segment_um(
            adata,
            target_cells_per_segment=float(target_cells_per_segment),
            platform=platform,
            support_method=support_method,
            grid_size_um=grid_size_um,
            alpha=alpha,
            return_details=True,
        )
        target_segment_um = float(target_details["target_segment_um"])
    else:
        target_segment_um = float(target_segment_um)

    base_result = SPIX.sp.select_compactness(
        adata,
        dimensions=dims,
        search_dimensions=search_dimensions,
        embedding=embedding,
        target_segment_um=float(target_segment_um),
        pitch_um=float(pitch_um),
        origin=origin,
        compactness_candidates=compactness_candidates,
        compactness_search_jobs=int(compactness_search_jobs),
        compactness_search_downsample_factor=float(compactness_search_downsample_factor),
        verbose=verbose,
        **kwargs,
    )

    result = reselect_compactness_for_cells(
        base_result,
        score_relax=float(score_relax),
        fidelity_relax=float(fidelity_relax),
        regularity_relax=float(regularity_relax),
        compactness_cap=compactness_cap,
        verbose=verbose,
    )
    result["pitch_um"] = float(pitch_um)
    result["target_segment_um"] = float(target_segment_um)
    result["target_cells_per_segment"] = None if target_cells_per_segment is None else float(target_cells_per_segment)
    if pitch_details is not None:
        result["pitch_details"] = pitch_details
    if target_details is not None:
        result["target_details"] = target_details
    return result


def reselect_compactness_for_stereo_seq(
    selection_result: dict,
    *,
    score_relax: float = 0.12,
    fidelity_relax: float = 0.20,
    regularity_relax: float = 0.15,
    compactness_floor: Optional[float] = None,
    compactness_cap: Optional[float] = None,
    prefer_higher_compactness: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Re-select a compactness value from an existing sweep using a Stereo-seq rule.

    Logic:
    - keep candidates whose original score is near-best
    - keep candidates whose fidelity cost is near-best
    - keep candidates whose regularity cost is near-best
    - if possible, enforce a monotone lower bound from the previous scale
    - among the remaining candidates, prefer the most regular candidate and
      break ties toward larger compactness

    This is intended for dense spot lattices where the default selector can
    drift toward overly low compactness at coarse scales.
    """
    frame = _cell_compactness_candidate_frame(selection_result)
    if frame.empty:
        result = dict(selection_result)
        result["stereo_seq_reselection"] = {
            "applied": False,
            "reason": "empty_candidate_frame",
        }
        return result

    work = frame.copy()
    work["eligible"] = True

    def _slack(values: np.ndarray, frac: float, floor: float) -> float:
        finite = values[np.isfinite(values)]
        if finite.size <= 1:
            return float(floor)
        span = float(np.nanmax(finite) - np.nanmin(finite))
        return float(max(floor, float(frac) * span))

    score_vals = work["score"].to_numpy(dtype=float)
    if np.isfinite(score_vals).any():
        score_min = float(np.nanmin(score_vals))
        score_slack = _slack(score_vals, score_relax, 0.02)
        work["eligible"] &= work["score"] <= (score_min + score_slack)
    else:
        score_min = None
        score_slack = None

    fidelity_vals = (
        work["fidelity_cost"].to_numpy(dtype=float)
        if "fidelity_cost" in work
        else np.array([], dtype=float)
    )
    if fidelity_vals.size and np.isfinite(fidelity_vals).any():
        fidelity_min = float(np.nanmin(fidelity_vals))
        fidelity_slack = _slack(fidelity_vals, fidelity_relax, 0.03)
        work["eligible"] &= work["fidelity_cost"] <= (fidelity_min + fidelity_slack)
    else:
        fidelity_min = None
        fidelity_slack = None

    regularity_vals = (
        work["regularity_cost"].to_numpy(dtype=float)
        if "regularity_cost" in work
        else np.array([], dtype=float)
    )
    if regularity_vals.size and np.isfinite(regularity_vals).any():
        regularity_min = float(np.nanmin(regularity_vals))
        regularity_slack = _slack(regularity_vals, regularity_relax, 0.03)
        work["eligible"] &= work["regularity_cost"] <= (regularity_min + regularity_slack)
    else:
        regularity_min = None
        regularity_slack = None

    floor_applied = False
    if compactness_floor is not None and np.isfinite(float(compactness_floor)):
        floor_value = float(compactness_floor)
        above_floor = work["compactness"] >= (floor_value - 1e-8)
        if bool((work["eligible"] & above_floor).any()):
            work["eligible"] &= above_floor
            floor_applied = True

    cap_applied = False
    if compactness_cap is not None and np.isfinite(float(compactness_cap)):
        below_cap = work["compactness"] <= float(compactness_cap)
        if bool((work["eligible"] & below_cap).any()):
            work["eligible"] &= below_cap
            cap_applied = True

    eligible = work.loc[work["eligible"]].copy()
    if eligible.empty:
        eligible = work.copy()

    chosen = eligible.sort_values(
        ["score", "fidelity_cost", "regularity_cost", "compactness"],
        ascending=[True, True, True, not bool(prefer_higher_compactness)],
        kind="mergesort",
    ).iloc[0]

    selected_search_compactness = float(chosen["compactness"])
    selected_stage = str(chosen.get("search_stage", ""))
    observed_segment_count = None
    if "n_segments" in chosen and np.isfinite(float(chosen["n_segments"])):
        observed_segment_count = int(round(float(chosen["n_segments"])))

    result = dict(selection_result)
    base_selected = float(selection_result.get("selected_compactness", selected_search_compactness))
    result["base_selected_compactness"] = base_selected
    result["selected_observed_segment_count"] = observed_segment_count

    scaled_from_reference = bool(result.get("scaled_from_search_target_um", False))
    if scaled_from_reference and ("search_target_segment_um" in result) and ("target_segment_um" in result):
        import SPIX

        search_target_um = float(result["search_target_segment_um"])
        target_um = float(result["target_segment_um"])
        scale_power = float(result.get("compactness_scale_power", 0.3))
        selected_compactness = float(
            SPIX.sp.scale_compactness(
                selected_search_compactness,
                from_um=search_target_um,
                to_um=target_um,
                power=scale_power,
            )
        )
        result["search_selected_compactness"] = selected_search_compactness
        result["selected_compactness"] = selected_compactness
    else:
        result["selected_compactness"] = selected_search_compactness
        if "search_selected_compactness" in result:
            result["search_selected_compactness"] = selected_search_compactness

    for stage_key in ("coarse_search", "refine_search"):
        stage_detail = result.get(stage_key, None)
        if not isinstance(stage_detail, dict):
            continue
        if str(stage_detail.get("mode", stage_key)) == selected_stage:
            updated_stage = dict(stage_detail)
            updated_stage["base_selected_compactness"] = float(
                stage_detail.get("selected_compactness", selected_search_compactness)
            )
            updated_stage["selected_compactness"] = selected_search_compactness
            result[stage_key] = updated_stage

    result["stereo_seq_reselection"] = {
        "applied": True,
        "strategy": "prefer_regular_near_best_with_monotone_floor",
        "score_relax": float(score_relax),
        "fidelity_relax": float(fidelity_relax),
        "regularity_relax": float(regularity_relax),
        "compactness_floor": None if compactness_floor is None else float(compactness_floor),
        "compactness_cap": None if compactness_cap is None else float(compactness_cap),
        "floor_applied": bool(floor_applied),
        "cap_applied": bool(cap_applied),
        "prefer_higher_compactness": bool(prefer_higher_compactness),
        "candidate_count": int(len(work)),
        "eligible_count": int(len(eligible)),
        "selected_compactness": float(result["selected_compactness"]),
        "selected_search_compactness": float(selected_search_compactness),
        "selected_stage": selected_stage,
        "base_selected_compactness": float(base_selected),
        "selected_candidate": {
            "compactness": float(chosen["compactness"]),
            "score": None if not np.isfinite(float(chosen.get("score", np.nan))) else float(chosen["score"]),
            "fidelity_cost": None if not np.isfinite(float(chosen.get("fidelity_cost", np.nan))) else float(chosen["fidelity_cost"]),
            "regularity_cost": None if not np.isfinite(float(chosen.get("regularity_cost", np.nan))) else float(chosen["regularity_cost"]),
            "tradeoff_gain": None if not np.isfinite(float(chosen.get("tradeoff_gain", np.nan))) else float(chosen["tradeoff_gain"]),
            "mean_log_aspect": None if not np.isfinite(float(chosen.get("mean_log_aspect", np.nan))) else float(chosen["mean_log_aspect"]),
            "size_cv": None if not np.isfinite(float(chosen.get("size_cv", np.nan))) else float(chosen["size_cv"]),
            "neighbor_disagreement": None if not np.isfinite(float(chosen.get("neighbor_disagreement", np.nan))) else float(chosen["neighbor_disagreement"]),
            "n_segments": observed_segment_count,
        },
        "thresholds": {
            "score_min": score_min,
            "score_slack": score_slack,
            "fidelity_min": fidelity_min,
            "fidelity_slack": fidelity_slack,
            "regularity_min": regularity_min,
            "regularity_slack": regularity_slack,
        },
    }
    if verbose:
        print(
            "[stereo-seq-compactness] reselection: "
            f"base={base_selected:.6g} -> selected={float(result['selected_compactness']):.6g} "
            f"(search={selected_search_compactness:.6g}) | "
            f"eligible={int(len(eligible))}/{int(len(work))}"
        )
    return result


def select_compactness_stereo_seq_robust(
    adata: AnnData,
    *,
    target_segment_um: float,
    pitch_um: float = 2.0,
    dimensions: Optional[list[int]] = None,
    search_dimensions: Optional[list[int]] = None,
    embedding: str = "X_embedding",
    origin: bool = True,
    compactness_candidates: Optional[list[float]] = None,
    compactness_search_jobs: int = 4,
    compactness_search_downsample_factor: float = 2.0,
    lock_reference_target: bool = True,
    search_target_segment_um: Optional[float] = None,
    score_relax: float = 0.12,
    fidelity_relax: float = 0.20,
    regularity_relax: float = 0.15,
    compactness_floor: Optional[float] = None,
    compactness_cap: Optional[float] = None,
    prefer_higher_compactness: bool = True,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Exact compactness selection for Stereo-seq-style dense spot lattices with a
    robust reselection pass that prefers regular coarse segments.
    """
    import SPIX

    dims = list(range(30)) if dimensions is None else list(dimensions)
    target_um = float(target_segment_um)
    search_target = (
        float(target_um)
        if lock_reference_target and search_target_segment_um is None
        else (
            None
            if search_target_segment_um is None
            else float(search_target_segment_um)
        )
    )

    base_result = SPIX.sp.fast_select_compactness(
        adata,
        dimensions=dims,
        search_dimensions=search_dimensions,
        embedding=embedding,
        target_segment_um=target_um,
        pitch_um=float(pitch_um),
        origin=origin,
        compactness_candidates=compactness_candidates,
        compactness_search_jobs=int(compactness_search_jobs),
        compactness_search_downsample_factor=float(compactness_search_downsample_factor),
        search_target_segment_um=search_target,
        verbose=verbose,
        **kwargs,
    )

    result = reselect_compactness_for_stereo_seq(
        base_result,
        score_relax=float(score_relax),
        fidelity_relax=float(fidelity_relax),
        regularity_relax=float(regularity_relax),
        compactness_floor=compactness_floor,
        compactness_cap=compactness_cap,
        prefer_higher_compactness=bool(prefer_higher_compactness),
        verbose=verbose,
    )
    result["pitch_um"] = float(pitch_um)
    result["target_segment_um"] = float(target_um)
    result["lock_reference_target"] = bool(lock_reference_target)
    result["search_target_segment_um"] = (
        float(search_target)
        if search_target is not None
        else result.get("search_target_segment_um", None)
    )
    return result


def select_compactness_stereo_seq_multiscale_robust(
    adata: AnnData,
    *,
    target_segment_ums: List[float],
    pitch_um: float = 2.0,
    dimensions: Optional[list[int]] = None,
    search_dimensions: Optional[list[int]] = None,
    embedding: str = "X_embedding",
    origin: bool = True,
    compactness_candidates: Optional[list[float]] = None,
    compactness_search_jobs: int = 4,
    compactness_search_downsample_factor: float = 2.0,
    lock_reference_target: bool = True,
    monotone_non_decreasing: bool = True,
    score_relax: float = 0.12,
    fidelity_relax: float = 0.20,
    regularity_relax: float = 0.15,
    compactness_cap: Optional[float] = None,
    prefer_higher_compactness: bool = True,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Run Stereo-seq robust compactness selection across multiple target sizes.

    The previous selected compactness is optionally used as a lower bound for
    the next scale so coarse resolutions do not collapse to a smaller value.
    """
    targets = [float(x) for x in target_segment_ums]
    if len(targets) == 0:
        raise ValueError("`target_segment_ums` must contain at least one target size.")

    ordered_targets = sorted(targets)
    results_by_target = {}
    summary_rows = []
    previous_selected = None

    for target_um in ordered_targets:
        floor_value = float(previous_selected) if (monotone_non_decreasing and previous_selected is not None) else None
        result = select_compactness_stereo_seq_robust(
            adata,
            target_segment_um=float(target_um),
            pitch_um=float(pitch_um),
            dimensions=dimensions,
            search_dimensions=search_dimensions,
            embedding=embedding,
            origin=origin,
            compactness_candidates=compactness_candidates,
            compactness_search_jobs=int(compactness_search_jobs),
            compactness_search_downsample_factor=float(compactness_search_downsample_factor),
            lock_reference_target=bool(lock_reference_target),
            score_relax=float(score_relax),
            fidelity_relax=float(fidelity_relax),
            regularity_relax=float(regularity_relax),
            compactness_floor=floor_value,
            compactness_cap=compactness_cap,
            prefer_higher_compactness=bool(prefer_higher_compactness),
            verbose=verbose,
            **kwargs,
        )
        reselection = result.get("stereo_seq_reselection", {})
        search_selected = float(result.get("search_selected_compactness", result["selected_compactness"]))
        previous_selected = search_selected
        results_by_target[float(target_um)] = result
        summary_rows.append(
            {
                "target_segment_um": float(target_um),
                "search_target_segment_um": result.get("search_target_segment_um", None),
                "base_selected_compactness": result.get("base_selected_compactness", None),
                "selected_compactness": result.get("selected_compactness", None),
                "selected_search_compactness": search_selected,
                "compactness_floor": reselection.get("compactness_floor", floor_value),
                "floor_applied": reselection.get("floor_applied", False),
                "eligible_count": reselection.get("eligible_count", None),
                "candidate_count": reselection.get("candidate_count", None),
                "requested_n_segments": result.get("requested_n_segments", None),
                "search_n_segments": result.get("search_n_segments", None),
                "observed_n_segments": result.get("selected_observed_segment_count", None),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return {
        "targets_um": ordered_targets,
        "results_by_target_um": results_by_target,
        "summary": summary,
        "pitch_um": float(pitch_um),
        "lock_reference_target": bool(lock_reference_target),
        "monotone_non_decreasing": bool(monotone_non_decreasing),
        "score_relax": float(score_relax),
        "fidelity_relax": float(fidelity_relax),
        "regularity_relax": float(regularity_relax),
    }
