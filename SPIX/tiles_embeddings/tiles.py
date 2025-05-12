import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import logging
import itertools
from scipy.spatial import Voronoi
import os

from ..utils.utils import filter_grid_function, reduce_tensor_resolution, filter_tiles, rasterise

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_tiles(
    adata: AnnData,
    tensor_resolution: float = 1,
    filter_grid: float = 0.01,
    filter_threshold: float = 0.995,
    verbose: bool = True,
    chunksize: int = 1000,
    n_jobs: int = None
) -> AnnData:
    """
    Generate spatial tiles for the given AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    tensor_resolution : float, optional (default=1)
        Resolution parameter for tensor reduction.
    filter_grid : float, optional (default=0.01)
        Grid filtering threshold to remove outlier beads.
    filter_threshold : float, optional (default=0.995)
        Threshold to filter tiles based on area.
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    chunksize : int, optional (default=1000)
        Number of tiles per chunk for parallel processing.
    n_jobs : int, optional (default=None)
        Number of parallel jobs to run.
    
    Returns
    -------
    AnnData
        The updated AnnData object with generated tiles.
    """
    # Check if tiles are already generated
    if 'tiles_generated' in adata.uns and adata.uns['tiles_generated']:
        if verbose:
            logging.info("Tiles have already been generated. Skipping tile generation.")
        return adata

    if verbose:
        logging.info("Starting generate_tiles...")

    # Copy original spatial coordinates
    coordinates = pd.DataFrame(adata.obsm['spatial'])
    coordinates.index = adata.obs.index.copy()
    if verbose:
        logging.info(f"Original coordinates: {coordinates.shape}")

    # Apply grid filtering if specified
    if 0 < filter_grid < 1:
        if verbose:
            logging.info("Filtering outlier beads...")
        coordinates = filter_grid_function(coordinates, filter_grid)
        if verbose:
            logging.info(f"Coordinates after filtering: {coordinates.shape}")

    # Reduce tensor resolution if specified
    if 0 < tensor_resolution < 1:
        if verbose:
            logging.info("Reducing tensor resolution...")
        coordinates = reduce_tensor_resolution(coordinates, tensor_resolution)
        if verbose:
            logging.info(f"Coordinates after resolution reduction: {coordinates.shape}")

    # Perform Voronoi tessellation
    if verbose:
        logging.info("Performing Voronoi tessellation...")
    vor = Voronoi(coordinates)
    if verbose:
        logging.info("Voronoi tessellation completed.")

    # Filter tiles based on area
    if verbose:
        logging.info("Filtering tiles...")
    filtered_regions, filtered_coordinates, index = filter_tiles(
        vor,
        coordinates,
        filter_threshold,
        n_jobs=n_jobs,
        chunksize=chunksize,
        verbose=verbose
    )
    if verbose:
        logging.info(f"Filtered regions: {len(filtered_regions)}, Filtered coordinates: {filtered_coordinates.shape}")

    # Rasterize tiles with parallel processing
    if verbose:
        logging.info("Rasterising tiles...")
    tiles = rasterise(filtered_regions, filtered_coordinates, index, vor, chunksize, n_jobs=n_jobs)
    if verbose:
        logging.info(f"Rasterisation completed. Number of tiles: {len(tiles)}")

    # Store tiles in AnnData object
    adata.uns['tiles'] = tiles
    adata.uns['tiles_generated'] = True
    if verbose:
        logging.info("Tiles have been stored in adata.uns['tiles'].")

    # Subset adata based on filtered tiles
    filtered_barcodes = tiles['barcode'].unique()
    initial_obs = adata.n_obs
    adata = adata[filtered_barcodes, :].copy()
    final_obs = adata.n_obs

    if verbose:
        logging.info(f"adata has been subset from {initial_obs} to {final_obs} observations based on filtered tiles.")

    if verbose:
        logging.info("generate_tiles completed.")

    return adata
