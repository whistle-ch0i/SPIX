### generate embeddings and tiles ###
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import scipy
from shapely.geometry import Polygon
from shapely.vectorized import contains  # Ensure Shapely version >= 2.0
from collections import Counter
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_embeddings(
    adata,
    dim_reduction='PCA',
    normalization='log_norm',
    use_counts='raw',
    dimensions=30,
    tensor_resolution=1,
    filter_grid=0.01,
    filter_threshold=0.995,
    nfeatures=2000,
    features=None,
    min_cutoff='q5',
    remove_lsi_1=True,
    n_jobs=None,
    chunksize=5000,
    verbose=True
):
    """
    Generate embeddings for the given AnnData object using specified parameters.

    Parameters:
    - adata: AnnData object containing the data.
    - dim_reduction: Method for dimensionality reduction ('PCA', 'UMAP', etc.).
    - normalization: Normalization method ('log_norm', 'SCT', 'TFIDF', 'none').
    - use_counts: Which counts to use ('raw' or layer name).
    - dimensions: Number of dimensions for reduction.
    - tensor_resolution: Resolution parameter for tensor.
    - filter_grid: Grid filtering parameter.
    - filter_threshold: Threshold for filtering tiles.
    - nfeatures: Number of features to select.
    - features: Specific features to use.
    - min_cutoff: Minimum cutoff for feature selection.
    - remove_lsi_1: Whether to remove the first LSI component.
    - n_jobs: Number of parallel jobs.
    - verbose: Whether to print verbose messages.

    Returns:
    - adata: AnnData object with embeddings added.
    """
    if verbose:
        print("Starting generate_embeddings...")

    # Generate tiles if not already generated
    if 'tiles_generated' not in adata.uns or not adata.uns['tiles_generated']:
        if verbose:
            print("Tiles not found. Generating tiles...")
        adata = generate_tiles(
            adata,
            tensor_resolution=tensor_resolution,
            filter_grid=filter_grid,
            filter_threshold=filter_threshold,
            verbose=verbose,
            chunksize=chunksize,
            n_jobs=n_jobs
        )

    # Process counts with specified normalization
    if verbose:
        print("Processing counts...")
    adata_proc = process_counts(
        adata,
        method=normalization,
        use_counts=use_counts,
        nfeatures=nfeatures,
        min_cutoff=min_cutoff,
        verbose=verbose
    )

    adata.layers['log_norm'] = adata_proc.X.copy()
    # Embed latent space using specified dimensionality reduction
    if verbose:
        print("Embedding latent space...")
    embeds = embed_latent_space(
        adata_proc,
        dim_reduction=dim_reduction,
        dimensions=dimensions,
        features=features,
        remove_lsi_1=remove_lsi_1,
        verbose=verbose,
        n_jobs=n_jobs
    )

    # Store embeddings in AnnData object
    adata.obsm['X_embedding'] = embeds
    adata.uns['embedding_method'] = dim_reduction
    adata.uns['tensor_resolution'] = tensor_resolution
    if verbose:
        print("generate_embeddings completed.")

    return adata

def generate_tiles(
    adata,
    tensor_resolution=1,
    filter_grid=0.01,
    filter_threshold=0.995,
    verbose=True,
    chunksize=1000,
    n_jobs=None
):
    """
    Generate spatial tiles for the given AnnData object.

    Parameters:
    - adata: AnnData object containing spatial data.
    - tensor_resolution: Resolution parameter for tensor.
    - filter_grid: Grid filtering parameter.
    - filter_threshold: Threshold for filtering tiles.
    - verbose: Whether to print verbose messages.
    - n_jobs: Number of parallel jobs.

    Returns:
    - adata: AnnData object with tiles generated.
    """
    # Check if tiles are already generated
    if 'tiles_generated' in adata.uns and adata.uns['tiles_generated']:
        if verbose:
            print("Tiles have already been generated. Skipping tile generation.")
        return adata

    if verbose:
        print("Starting generate_tiles...")

    # Copy original spatial coordinates
    coordinates = pd.DataFrame(adata.obsm['spatial'])
    coordinates.index = adata.obs.index.copy()
    if verbose:
        print(f"Original coordinates: {coordinates.shape}")

    # Apply grid filtering if specified
    if 0 < filter_grid < 1:
        if verbose:
            print("Filtering outlier beads...")
        coordinates = filter_grid_function(coordinates, filter_grid)
        if verbose:
            print(f"Coordinates after filtering: {coordinates.shape}")

    # Reduce tensor resolution if specified
    if 0 < tensor_resolution < 1:
        if verbose:
            print("Reducing tensor resolution...")
        coordinates = reduce_tensor_resolution(coordinates, tensor_resolution)
        if verbose:
            print(f"Coordinates after resolution reduction: {coordinates.shape}")

    # Perform Voronoi tessellation
    if verbose:
        print("Performing Voronoi tessellation...")
    from scipy.spatial import Voronoi
    vor = Voronoi(coordinates)
    if verbose:
        print("Voronoi tessellation completed.")

    # Filter tiles based on area
    if verbose:
        print("Filtering tiles...")
    filtered_regions, filtered_coordinates, index = filter_tiles(vor, coordinates, filter_threshold)
    if verbose:
        print(f"Filtered regions: {len(filtered_regions)}, Filtered coordinates: {filtered_coordinates.shape}")

    # Rasterize tiles with parallel processing
    if verbose:
        print("Rasterising tiles...")
    tiles = rasterise(filtered_regions, filtered_coordinates, index, vor,chunksize, n_jobs=n_jobs)
    if verbose:
        print(f"Rasterisation completed. Number of tiles: {len(tiles)}")

    # Store tiles in AnnData object
    adata.uns['tiles'] = tiles
    adata.uns['tiles_generated'] = True
    if verbose:
        print("Tiles have been stored in adata.uns['tiles'].")
    filtered_barcodes = tiles['barcode'].unique()
    initial_obs = adata.n_obs
    adata = adata[filtered_barcodes, :].copy()
    final_obs = adata.n_obs

    if verbose:
        print(f"adata has been subset from {initial_obs} to {final_obs} observations based on filtered tiles.")

    if verbose:
        print("generate_tiles completed.")

    return adata

def filter_grid_function(coordinates, filter_grid):
    """
    Filter out grid coordinates based on the specified grid filter.

    Parameters:
    - coordinates: Array of spatial coordinates.
    - filter_grid: Grid filtering parameter.

    Returns:
    - filtered_coordinates: Filtered array of spatial coordinates.
    """
    grid_x = np.round(coordinates[0] * filter_grid)
    grid_y = np.round(coordinates[1] * filter_grid)
    
    # Use numpy.char.add for string concatenation
    grid_coord = np.char.add(grid_x.astype(str), "_")
    grid_coord = np.char.add(grid_coord, grid_y.astype(str))

    # Count occurrences in each grid cell
    grid_counts = Counter(grid_coord)
    counts = np.array(list(grid_counts.values()))
    threshold = np.quantile(counts, 0.01)

    # Identify low-count grid cells
    low_count_grids = {k for k, v in grid_counts.items() if v <= threshold}

    # Create mask to filter out low-count grid cells
    mask = np.array([gc not in low_count_grids for gc in grid_coord])
    filtered_coordinates = coordinates[mask]

    return filtered_coordinates

def reduce_tensor_resolution(coordinates, tensor_resolution=1):
    """
    Reduce tensor resolution of spatial coordinates.

    Parameters:
    - coordinates: Array of spatial coordinates.
    - tensor_resolution: Resolution parameter for tensor.

    Returns:
    - filtered_coordinates: Reduced resolution coordinates.
    """
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

def filter_tiles(vor, coordinates, filter_threshold):
    """
    Filter Voronoi regions based on area threshold.

    Parameters:
    - vor: Voronoi object.
    - coordinates: Array of spatial coordinates.
    - filter_threshold: Threshold for filtering tiles based on area.

    Returns:
    - filtered_regions: List of filtered Voronoi region indices.
    - filtered_coordinates: Array of filtered spatial coordinates.
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
        valid_points.append(coordinates.iloc[idx])
        valid_regions.append(region)
        index.append(coordinates.index[idx])
    areas = np.array(areas)
    max_area = np.quantile(areas, filter_threshold)

    filtered_regions = []
    filtered_points = []
    filtered_index = []
    # Filter regions based on area
    for area, region, point, index in tqdm(zip(areas, valid_regions, valid_points, index), total=len(areas), desc="Filtering tiles by area"):
        if area <= max_area:
            filtered_regions.append(region)
            filtered_points.append(point)
            filtered_index.append(index)

    return filtered_regions, np.array(filtered_points), filtered_index

def rasterise(filtered_regions, filtered_points, index, vor, chunksize=1000, n_jobs=None):
    """
    Rasterize Voronoi regions into grid tiles.

    Parameters:
    - filtered_regions: List of filtered Voronoi region indices.
    - filtered_points: Array of filtered spatial coordinates.
    - vor: Voronoi object.
    - n_jobs: Number of parallel jobs.

    Returns:
    - tiles: DataFrame containing rasterized tiles.
    """
    # Set default number of jobs if not specified
    if n_jobs is None:
        n_jobs = os.cpu_count()
    elif n_jobs < 0:
        n_jobs = n_jobs
    elif n_jobs == 0:
        n_jobs = 1

    # Create iterable of arguments for rasterization
    iterable = zip(filtered_regions, filtered_points, index, itertools.repeat(vor))
    desc = "Rasterising tiles"

    # Calculate total number of regions
    total = len(filtered_regions)

    # Process rasterization in parallel or serially using the optimized helper function
    tiles_list = _process_in_parallel_map(
        rasterise_single_tile,
        iterable,
        desc,
        n_jobs,
        chunksize,
        total=total
    )

    # Concatenate all tile DataFrames
    if tiles_list:
        tiles = pd.concat(tiles_list, ignore_index=True)
    else:
        tiles = pd.DataFrame(columns=['x', 'y', 'barcode', 'origin_x', 'origin_y', 'origin'])

    return tiles

def _process_in_parallel_map(function, iterable, desc, n_jobs, chunksize=1000, total=None):
    """
    Helper function to process tasks in parallel with a progress bar using map.
    If n_jobs == 1, processes serially.

    Parameters:
    - function: Function to apply to each item.
    - iterable: Iterable of items to process.
    - desc: Description for the progress bar.
    - n_jobs: Number of parallel jobs.
    - chunksize: Number of tasks to submit in each batch.
    - total: Total number of items in the iterable.

    Returns:
    - results: List of results after applying the function.
    """
    results = []
    if n_jobs == 1:
        # Serial processing with progress bar
        for item in tqdm(iterable, desc=desc, total=total):
            results.append(function(item))
    else:
        # Parallel processing with progress bar using map
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Using map with chunksize to optimize task submission
            for result in tqdm(executor.map(function, iterable, chunksize=chunksize), desc=desc, total=total):
                results.append(result)
    return results

def rasterise_single_tile(args):
    """
    Rasterize a single Voronoi region into grid tiles.

    Parameters:
    - args: Tuple containing (region, point, index, vor).

    Returns:
    - tile_df: DataFrame containing rasterized tile information.
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

        # Vectorized point-in-polygon test using shapely.vectorized.contains
        mask = contains(polygon, grid_points[:, 0], grid_points[:, 1])
        pixels = grid_points[mask]

        if len(pixels) == 0:
            # If no pixels are inside the polygon, return an empty DataFrame
            return pd.DataFrame()

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
            # logging.warning(f"Origin pixel not found for barcode {index}. Adding manually.")
            
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
        return pd.DataFrame()  # Return an empty DataFrame or handle as needed



def process_counts(
    adata,
    method='log_norm',
    use_counts='raw',
    nfeatures=2000,
    min_cutoff='q5',
    verbose=True
):
    """
    Process count data with specified normalization and feature selection.

    Parameters:
    - adata: AnnData object containing count data.
    - method: Normalization method ('log_norm', 'SCT', 'TFIDF', 'none').
    - use_counts: Which counts to use ('raw' or layer name).
    - nfeatures: Number of highly variable genes to select.
    - min_cutoff: Minimum cutoff for feature selection.
    - verbose: Whether to print verbose messages.

    Returns:
    - adata_proc: Processed AnnData object.
    """
    if verbose:
        print(f"Processing counts with method: {method}")

    # Select counts based on 'use_counts' parameter
    if use_counts == 'raw':
        counts = adata.raw.X if adata.raw is not None else adata.X.copy()
        adata_proc = adata.copy()
    else:
        counts = adata.layers[use_counts]
        adata_proc = AnnData(counts)
        adata_proc.var_names = adata.var_names.copy()
        adata_proc.obs_names = adata.obs_names.copy()

    # Apply normalization and feature selection
    if method == 'log_norm':
        sc.pp.normalize_total(adata_proc, target_sum=1e4)
        sc.pp.log1p(adata_proc)
        sc.pp.highly_variable_genes(adata_proc, n_top_genes=nfeatures)
        # adata_proc = adata_proc[:, adata_proc.var['highly_variable']]
    elif method == 'SCT':
        raise NotImplementedError("SCTransform normalization is not implemented in this code.")
    elif method == 'TFIDF':
        # Calculate Term Frequency (TF)
        tf = counts / counts.sum(axis=1)
        # Calculate Inverse Document Frequency (IDF)
        idf = np.log(1 + counts.shape[0] / (1 + (counts > 0).sum(axis=0)))
        counts_tfidf = tf.multiply(idf)
        adata_proc.X = counts_tfidf

        # Feature selection based on variance
        if min_cutoff.startswith('q'):
            quantile = float(min_cutoff[1:]) / 100
            if scipy.sparse.issparse(counts_tfidf):
                counts_tfidf_dense = counts_tfidf.toarray()
            else:
                counts_tfidf_dense = counts_tfidf
            variances = np.var(counts_tfidf_dense, axis=0)
            cutoff = np.quantile(variances, quantile)
            selected_features = variances >= cutoff
            adata_proc = adata_proc[:, selected_features]
        else:
            raise ValueError("Invalid min_cutoff format.")
    elif method == 'none':
        pass
    else:
        raise ValueError(f"Normalization method '{method}' is not recognized.")

    if verbose:
        print("Counts processing completed.")

    return adata_proc

def embed_latent_space(
    adata_proc,
    dim_reduction='PCA',
    dimensions=30,
    features=None,
    remove_lsi_1=True,
    verbose=True,
    n_jobs=None
):
    """
    Embed the processed data into a latent space using specified dimensionality reduction.

    Parameters:
    - adata_proc: Processed AnnData object.
    - dim_reduction: Method for dimensionality reduction ('PCA', 'UMAP', etc.).
    - dimensions: Number of dimensions for reduction.
    - features: Specific features to use.
    - remove_lsi_1: Whether to remove the first LSI component.
    - verbose: Whether to print verbose messages.
    - n_jobs: Number of parallel jobs.

    Returns:
    - embeds: Numpy array of embedded coordinates.
    """
    if verbose:
        print(f"Embedding latent space using {dim_reduction}...")

    # Subset to specific features if provided
    if features is not None:
        adata_proc = adata_proc[:, features]

    embeds = None

    if dim_reduction == 'PCA':
        sc.tl.pca(adata_proc, n_comps=dimensions,)
        embeds = adata_proc.obsm['X_pca']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'PCA_L':
        sc.tl.pca(adata_proc, n_comps=dimensions)
        loadings = adata_proc.varm['PCs']
        counts = adata_proc.X

        # Create iterable of arguments for PCA loadings calculation
        iterable = zip(range(counts.shape[0]), itertools.repeat(counts), itertools.repeat(loadings))
        desc = "Calculating PCA loadings"

        # Process PCA loadings in parallel or serially using the optimized helper function
        embeds_list = _process_in_parallel_map(
            calculate_pca_loadings,
            iterable,
            desc,
            n_jobs,
            chunksize=1000,
            total=counts.shape[0]
        )
        embeds = np.array(embeds_list)
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'UMAP':
        sc.pp.neighbors(adata_proc, n_pcs=dimensions)
        sc.tl.umap(adata_proc, n_components=3)
        embeds = adata_proc.obsm['X_umap']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'LSI':
        embeds = run_lsi(adata_proc, n_components=dimensions, remove_first=remove_lsi_1)
    elif dim_reduction == 'LSI_UMAP':
        lsi_embeds = run_lsi(adata_proc, n_components=dimensions, remove_first=remove_lsi_1)
        adata_proc.obsm['X_lsi'] = lsi_embeds
        sc.pp.neighbors(adata_proc, use_rep='X_lsi')
        sc.tl.umap(adata_proc, n_components=3)
        embeds = adata_proc.obsm['X_umap']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'NMF':
        nmf_model = NMF(n_components=dimensions, init='random', random_state=0)
        W = nmf_model.fit_transform(adata_proc.X)
        embeds = MinMaxScaler().fit_transform(W)
    else:
        raise ValueError(f"Dimensionality reduction method '{dim_reduction}' is not recognized.")

    if verbose:
        print("Latent space embedding completed.")

    return embeds

def calculate_pca_loadings(args):
    """
    Calculate PCA loadings for a single cell.

    Parameters:
    - args: Tuple containing (cell index, counts matrix, loadings matrix).

    Returns:
    - cell_embedding: Numpy array of PCA loadings for the cell.
    """
    i, counts, loadings = args
    if scipy.sparse.issparse(counts):
        expressed_genes = counts[i].toarray().flatten() > 0
    else:
        expressed_genes = counts[i] > 0
    cell_loadings = np.abs(loadings[expressed_genes, :])
    cell_embedding = cell_loadings.sum(axis=0)
    return cell_embedding

def run_lsi(adata, n_components=30, remove_first=True):
    """
    Run Latent Semantic Indexing (LSI) on the data.

    Parameters:
    - adata: AnnData object containing count data.
    - n_components: Number of LSI components.
    - remove_first: Whether to remove the first LSI component.

    Returns:
    - embeddings: Numpy array of LSI embeddings.
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

    from sklearn.utils.extmath import randomized_svd
    U, Sigma, VT = randomized_svd(tfidf, n_components=n_components + 1)

    if remove_first:
        U = U[:, 1:]
        Sigma = Sigma[1:]
        VT = VT[1:, :]

    embeddings = np.dot(U, np.diag(Sigma))
    embeddings = MinMaxScaler().fit_transform(embeddings)

    return embeddings



### image plot ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import matplotlib.patches as patches
from shapely.geometry import MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import ConvexHull
import logging

def image_plot(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding_segment',  # Use the segmented embedding
    figsize=(10, 10),
    point_size=None,  # Set default to None
    scaling_factor=10,
    origin=True,
    plot_boundaries=True,  # Toggle boundary plotting
    boundary_method='convex_hull',  # 'convex_hull', 'alpha_shape', 'oriented_bbox', 'buffered_line'
    alpha=0.3,  # Transparency for polygons
    boundary_color='black',  # Color for boundaries
    boundary_linewidth=1.0,  # Line width for boundaries
    alpha_shape_alpha=0.1,  # Alpha for alpha shapes
    aspect_ratio_threshold=3.0  # Threshold to determine if segment is thin and long
):
    """
    Visualize embeddings as an image with optional segment boundaries.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings and spatial coordinates.
    dimensions : list of int
        List of dimensions to use for visualization (1 or 3 dimensions).
    embedding : str
        Key in adata.obsm where the embeddings are stored.
    figsize : tuple of float
        Figure size in inches (width, height).
    point_size : float or None
        Size of the points in the scatter plot. If None, it will be automatically determined based on figsize.
    scaling_factor : float
        Scaling factor for point size calculation.
    origin : bool
        Whether to use only origin tiles.
    plot_boundaries : bool
        Whether to plot segment boundaries.
    boundary_method : str
        Method to compute boundaries ('convex_hull', 'alpha_shape', 'oriented_bbox', 'buffered_line').
    alpha : float
        Transparency for the boundary polygons.
    boundary_color : str
        Color for the boundary lines.
    boundary_linewidth : float
        Line width for the boundary lines.
    alpha_shape_alpha : float
        Alpha value for alpha shapes (only relevant if using 'alpha_shape').
    aspect_ratio_threshold : float
        Threshold to determine if a segment is considered thin and long based on aspect ratio.
    """
    if len(dimensions) not in [1, 3]:
        raise ValueError("Only 1 or 3 dimensions can be used for visualization.")
    
    # Extract embeddings
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Extract spatial coordinates
    if origin:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    else:
        tiles = adata.uns['tiles']
    
    # Ensure 'Segment' exists in adata.obs
    # if 'Segment' not in adata.obs.columns:
    #     raise ValueError("Segment information not found in adata.obs['Segment']. Please run the segmentation first.")
    
    # Prepare tile colors and merge with coordinates
    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors['barcode'] = adata.obs.index
    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="right").dropna().reset_index(drop=True)

    if 'Segment' in adata.obs.columns:
        # Map 'Segment' based on 'barcode'
        coordinates_df['Segment'] = coordinates_df['barcode'].map(adata.obs['Segment'])
        # Check for any missing mappings
        if coordinates_df['Segment'].isnull().any():
            missing = coordinates_df['Segment'].isnull().sum()
            logging.warning(f"{missing} barcodes in coordinates_df do not have corresponding 'Segment' labels in adata.obs.")
            # Optionally, you can drop these rows or handle them differently
            coordinates_df = coordinates_df.dropna(subset=['Segment'])
    
    # Normalize embeddings between 0 and 1
    coordinates = rebalance_colors(coordinates_df, dimensions)
    
    if len(dimensions) == 3:
        # Normalize the RGB values between 0 and 1
        cols = [
            f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            for r, g, b in zip(coordinates['R'], coordinates['G'], coordinates['B'])
        ]
    else:
        # For grayscale, replicate the grey value across RGB channels
        cols = (coordinates['Grey'].values[:, np.newaxis] * 255).astype(int)
        cols = np.repeat(cols, 3, axis=1) / 255.0  # Normalize to [0,1]
        cols = cols.tolist()
    
    # Automatically determine point_size if not provided
    if point_size is None:
        figure_area = figsize[0] * figsize[1]  # in square inches
        point_size = scaling_factor * (figure_area / 100)  # Adjust denominator as needed

    if 'Segment' in adata.obs.columns:
        # Determine plot boundaries based on spatial coordinates
        x_min, x_max = coordinates['x'].min(), coordinates['x'].max()
        y_min, y_max = coordinates['y'].min(), coordinates['y'].max()
        plot_boundary_polygon = Polygon([
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_max),
            (x_max, y_min)
        ])
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    sc_kwargs = dict(
        x=coordinates['x'],
        y=coordinates['y'],
        s=point_size,
        c=cols,
        marker='s',
        linewidths=0
    )
    scatter = ax.scatter(**sc_kwargs)
    ax.set_aspect('equal')
    ax.axis('off')
    title = f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
    ax.set_title(title, fontsize=figsize[0] * 1.5)
    if 'Segment' in adata.obs.columns:
        if plot_boundaries:
            # Group by segments
            segments = coordinates_df['Segment'].unique()
            for seg in segments:
                seg_tiles = coordinates_df[coordinates_df['Segment'] == seg]
                if len(seg_tiles) < 2:
                    # Not enough points to form a boundary
                    continue
                
                points = seg_tiles[['x', 'y']].values
    
                # Calculate aspect ratio
                x_range = points[:, 0].max() - points[:, 0].min()
                y_range = points[:, 1].max() - points[:, 1].min()
                aspect_ratio = max(x_range, y_range) / (min(x_range, y_range) + 1e-10)  # Avoid division by zero
    
                if aspect_ratio > aspect_ratio_threshold:
                    # Treat as thin and long segment
                    if len(seg_tiles) >= 2:
                        # Sort points by x or y for consistent line
                        sorted_indices = np.argsort(points[:, 0])
                        sorted_points = points[sorted_indices]
                        line = LineString(sorted_points)
                        # Buffer the line to create a polygon with small thickness
                        buffered_line = line.buffer(y_range * 0.05)  # Buffer size proportional to y_range
                        # Clip the buffered polygon to plot boundaries
                        clipped_polygon = buffered_line.intersection(plot_boundary_polygon)
                        if not clipped_polygon.is_empty:
                            if isinstance(clipped_polygon, Polygon):
                                polygons = [clipped_polygon]
                            else:
                                # Handle MultiPolygon
                                polygons = list(clipped_polygon)
                            for poly in polygons:
                                patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                                        fill=False, edgecolor=boundary_color, 
                                                        linewidth=boundary_linewidth, alpha=alpha)
                                ax.add_patch(patch)
                else:
                    # Use ConvexHull or alpha_shape for regular segments
                    if boundary_method == 'convex_hull':
                        try:
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]
                            polygon = Polygon(hull_points)
                        except Exception as e:
                            logging.error(f"ConvexHull failed for segment {seg}: {e}")
                            # Attempt to plot a line instead
                            if len(points) >= 2:
                                sorted_indices = points[:, 0].argsort()
                                sorted_points = points[sorted_indices]
                                line = LineString(sorted_points)
                                patch = patches.Polygon(list(line.coords), closed=False,
                                                        edgecolor=boundary_color,
                                                        linewidth=boundary_linewidth,
                                                        alpha=alpha)
                                ax.add_patch(patch)
                            continue
                    elif boundary_method == 'alpha_shape':
                        polygon = alpha_shape(points, alpha=alpha_shape_alpha)
                        if polygon is None:
                            logging.warning(f"Alpha shape failed for segment {seg}.")
                            continue
                    else:
                        raise ValueError(f"Unknown boundary method '{boundary_method}'")
        
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)  # Attempt to fix invalid polygons
        
                    if polygon.is_empty:
                        logging.warning(f"Polygon is empty for segment {seg}.")
                        continue
        
                    # Clip the polygon to the plot boundary to prevent it from going outside
                    clipped_polygon = polygon.intersection(plot_boundary_polygon)
                    if clipped_polygon.is_empty:
                        logging.warning(f"Clipped polygon is empty for segment {seg}.")
                        continue
        
                    # Create a patch from the clipped polygon
                    if isinstance(clipped_polygon, Polygon):
                        polygons = [clipped_polygon]
                    else:
                        # Handle MultiPolygon
                        polygons = list(clipped_polygon)
        
                    for poly in polygons:
                        patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                                fill=False, edgecolor=boundary_color, 
                                                linewidth=boundary_linewidth, alpha=alpha)
                        ax.add_patch(patch)
        
        # Set plot limits to the plot boundary
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    # Display the figure
    plt.show()
    # Do not return the figure to prevent automatic display

def min_max(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))

def rebalance_colors(coordinates, dimensions, method="minmax"):
    if len(dimensions) == 3:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        colors = coordinates.iloc[:, [4, 5, 6]].values
        
        if method == "minmax":
            colors = np.apply_along_axis(min_max, 0, colors)
        else:
            colors[colors < 0] = 0
            colors[colors > 1] = 1
        
        template = pd.concat([template, pd.DataFrame(colors, columns=["R", "G", "B"])], axis=1)
    
    else:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        colors = coordinates.iloc[:, 4].values
        
        if method == "minmax":
            colors = min_max(colors)
        else:
            colors[colors < 0] = 0
            colors[colors > 1] = 1
        
        template = pd.concat([template, pd.Series(colors, name="Grey")], axis=1)
    
    return template


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.
    
    Parameters
    ----------
    points : np.ndarray
        An array of shape (n, 2) representing the coordinates of the points.
    alpha : float
        Alpha value to influence the gooeyness of the border. Smaller
        numbers don't fall inward as much as larger numbers. Too large,
        and you lose everything!
    
    Returns
    -------
    Polygon or MultiPolygon
        The resulting alpha shape polygon.
    """
    if len(points) < 4:
        # Not enough points to form a polygon
        return None

    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError("scipy is required for alpha_shape computation.")

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

    m = MultiPoint([tuple(p) for p in points])
    polygons = list(polygonize(edge_points))
    if not polygons:
        return None
    return unary_union(polygons)

### image plot with Spatial image ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import ConvexHull
import logging

def image_plot_with_spatial_image(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding_segment',
    figsize=(10, 10),
    point_size=None,
    point_scaling_factor=None,
    scaling_factor=None,
    origin=True,
    plot_boundaries=True,
    boundary_method='convex_hull',
    alpha=0.3,
    alpha_point=1,
    boundary_color='black',
    boundary_linewidth=1.0,
    alpha_shape_alpha=0.1,
    aspect_ratio_threshold=3.0,
    img=None,
    img_key=None,
    library_id=None,
    crop=True,
    alpha_img=1.0,
    bw=False,
):
    """
    Visualize embeddings as an image with optional segment boundaries and background image overlay.
    """
    if len(dimensions) not in [1, 3]:
        raise ValueError("Only 1 or 3 dimensions can be used for visualization.")
    
    # Extract embeddings
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Extract image data if available
    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    
    # Retrieve tensor_resolution from adata.uns
    tensor_resolution = adata.uns.get('tensor_resolution', 1.0)
    
    # Adjust scaling_factor based on tensor_resolution
    if scaling_factor is None:
        scaling_factor = _check_scale_factor(spatial_data, img_key, scaling_factor)
    scaling_factor_adjusted = scaling_factor / tensor_resolution
    
    # Extract spatial coordinates
    if origin:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    else:
        tiles = adata.uns['tiles']
    
    # Prepare tile colors and merge with coordinates
    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors['barcode'] = adata.obs.index
    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="inner").dropna().reset_index(drop=True)

    # Apply adjusted scaling factor to coordinates
    coordinates_df['x'] *= scaling_factor_adjusted
    coordinates_df['y'] *= scaling_factor_adjusted

    if 'Segment' in adata.obs.columns:
        # Map 'Segment' based on 'barcode'
        coordinates_df['Segment'] = coordinates_df['barcode'].map(adata.obs['Segment'])
        # Drop rows with missing Segment
        coordinates_df = coordinates_df.dropna(subset=['Segment'])
    
    # Normalize embeddings between 0 and 1
    coordinates = rebalance_colors(coordinates_df, dimensions)
    
    if len(dimensions) == 3:
        # Normalize the RGB values between 0 and 1
        cols = [
            f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            for r, g, b in zip(coordinates['R'], coordinates['G'], coordinates['B'])
        ]
    else:
        # For grayscale, replicate the grey value across RGB channels
        grey_values = coordinates['Grey'].values
        cols = [
            f'#{int(g*255):02x}{int(g*255):02x}{int(g*255):02x}'
            for g in grey_values
        ]
    
    # Automatically determine point_size if not provided
    if point_size is None:
        figure_area = figsize[0] * figsize[1]  # in square inches
        point_size = point_scaling_factor * (figure_area / 100)  # Adjust denominator as needed

    # Determine plot boundaries based on spatial coordinates
    x_min, x_max = coordinates['x'].min(), coordinates['x'].max()
    y_min, y_max = coordinates['y'].min(), coordinates['y'].max()

    # Adjust order if necessary
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)

    if img is not None:
        y_max_img, x_max_img = img.shape[:2]
        extent = [0, x_max_img, y_max_img, 0]  # Left, Right, Bottom, Top

        # Display the image
        ax.imshow(img, extent=extent, alpha=alpha_img)

        if crop:
            # Set axis limits to the extent of the spots
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
        else:
            # Show the full image
            ax.set_xlim(0, x_max_img)
            ax.set_ylim(y_max_img, 0)
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

    # Plot the spots
    sc_kwargs = dict(
        x=coordinates['x'],
        y=coordinates['y'],
        s=point_size,
        c=cols,
        alpha = alpha_point,
        marker='o',
        linewidths=0,
    )
    scatter = ax.scatter(**sc_kwargs)
    
    ax.set_aspect('equal')
    ax.axis('off')
    title = f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
    ax.set_title(title, fontsize=figsize[0] * 1.5)

    if 'Segment' in adata.obs.columns and plot_boundaries:
        # Plot segment boundaries
        plot_boundaries_on_ax(ax, coordinates_df, boundary_method, alpha, boundary_color,
                              boundary_linewidth, alpha_shape_alpha, aspect_ratio_threshold)
    
    # Display the figure
    plt.show()
    # Do not return the figure to prevent automatic display


def min_max(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))

def rebalance_colors(coordinates, dimensions, method="minmax"):
    if len(dimensions) == 3:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        colors = coordinates.iloc[:, [4, 5, 6]].values
        
        if method == "minmax":
            colors = np.apply_along_axis(min_max, 0, colors)
        else:
            colors[colors < 0] = 0
            colors[colors > 1] = 1
        
        template = pd.concat([template, pd.DataFrame(colors, columns=["R", "G", "B"])], axis=1)
    
    else:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        colors = coordinates.iloc[:, 4].values
        
        if method == "minmax":
            colors = min_max(colors)
        else:
            colors[colors < 0] = 0
            colors[colors > 1] = 1
        
        template = pd.concat([template, pd.Series(colors, name="Grey")], axis=1)
    
    return template

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.
    """
    if len(points) < 4:
        # Not enough points to form a polygon
        return None

    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError("scipy is required for alpha_shape computation.")

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

def plot_boundaries_on_ax(ax, coordinates_df, boundary_method, alpha, boundary_color,
                          boundary_linewidth, alpha_shape_alpha, aspect_ratio_threshold):
    # Group by segments
    segments = coordinates_df['Segment'].unique()
    for seg in segments:
        seg_tiles = coordinates_df[coordinates_df['Segment'] == seg]
        if len(seg_tiles) < 2:
            # Not enough points to form a boundary
            continue
        
        points = seg_tiles[['x', 'y']].values

        # Calculate aspect ratio
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        aspect_ratio = max(x_range, y_range) / (min(x_range, y_range) + 1e-10)  # Avoid division by zero

        if aspect_ratio > aspect_ratio_threshold:
            # Treat as thin and long segment
            if len(seg_tiles) >= 2:
                # Sort points by x or y for consistent line
                sorted_indices = np.argsort(points[:, 0])
                sorted_points = points[sorted_indices]
                line = LineString(sorted_points)
                # Buffer the line to create a polygon with small thickness
                buffered_line = line.buffer(y_range * 0.05)  # Buffer size proportional to y_range
                if not buffered_line.is_empty:
                    patch = patches.Polygon(list(buffered_line.exterior.coords), closed=True, 
                                            fill=False, edgecolor=boundary_color, 
                                            linewidth=boundary_linewidth, alpha=alpha)
                    ax.add_patch(patch)
        else:
            # Use ConvexHull or alpha_shape for regular segments
            if boundary_method == 'convex_hull':
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    polygon = Polygon(hull_points)
                except Exception as e:
                    logging.error(f"ConvexHull failed for segment {seg}: {e}")
                    continue
            elif boundary_method == 'alpha_shape':
                polygon = alpha_shape(points, alpha=alpha_shape_alpha)
                if polygon is None:
                    logging.warning(f"Alpha shape failed for segment {seg}.")
                    continue
            else:
                raise ValueError(f"Unknown boundary method '{boundary_method}'")

            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # Attempt to fix invalid polygons

            if polygon.is_empty:
                logging.warning(f"Polygon is empty for segment {seg}.")
                continue

            # Create a patch from the polygon
            if isinstance(polygon, Polygon):
                polygons = [polygon]
            else:
                # Handle MultiPolygon
                polygons = list(polygon)

            for poly in polygons:
                patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                        fill=False, edgecolor=boundary_color, 
                                        linewidth=boundary_linewidth, alpha=alpha)
                ax.add_patch(patch)

def _check_spatial_data(uns, library_id):
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
    if img is None and spatial_data is not None:
        if img_key is None:
            img_key = next((k for k in ['hires', 'lowres'] if k in spatial_data['images']), None)
            if img_key is None:
                raise ValueError("No image found in spatial data.")
        img = spatial_data['images'][img_key]
    if bw and img is not None:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img, img_key

def _check_scale_factor(spatial_data, img_key, scaling_factor):
    if scaling_factor is not None:
        return scaling_factor
    elif spatial_data is not None and img_key is not None:
        return spatial_data['scalefactors'][f'tissue_{img_key}_scalef']
    else:
        return 1.0


### smooth using tiles ###
import numpy as np
import pandas as pd
import scanpy as sc
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import logging
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def smooth_image(
    adata,
    dimensions=[0, 1, 2],
    embedding='last',
    method=['iso'],
    iter=1,
    sigma=1,
    box=20,
    threshold=0,
    neuman=True,
    gaussian=True,
    na_rm=False,
    across_levels='min',
    output_embedding='X_embedding_smooth',
    n_jobs=1,
    verbose=True,
    multi_method='threading', # loky, multiprocessing, threading
    max_image_size=10000 
):
    """
    Apply iterative smoothing to embeddings in the AnnData object and store the results in a new embedding.
    """
    if verbose:
        logging.info("Starting smooth_image...")

    # Determine which embedding to use
    if embedding == 'last':
        embedding_key = list(adata.obsm.keys())[-1]
    else:
        embedding_key = embedding
        if embedding_key not in adata.obsm:
            raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm.")

    embeddings = adata.obsm[embedding_key]
    if verbose:
        logging.info(f"Using embedding '{embedding_key}' with shape {embeddings.shape}.")

    # Access tiles
    if 'tiles' not in adata.uns:
        raise ValueError("Tiles not found in adata.uns['tiles'].")

    tiles = adata.uns['tiles'][['x','y','barcode']]
    if not isinstance(tiles, pd.DataFrame):
        raise ValueError("Tiles should be a pandas DataFrame.")

    required_columns = {'x', 'y', 'barcode'}
    if not required_columns.issubset(tiles.columns):
        raise ValueError(f"Tiles DataFrame must contain columns: {required_columns}")

    adata_obs_names = adata.obs_names

    # Precompute shifts and image size
    min_x = tiles['x'].min()
    min_y = tiles['y'].min()
    shift_x = -min_x
    shift_y = -min_y

    if verbose:
        logging.info(f"Shifting coordinates by ({shift_x}, {shift_y}) to start from 0.")

    # Shift coordinates to start from 0
    tiles_shifted = tiles.copy()
    tiles_shifted['x'] += shift_x
    tiles_shifted['y'] += shift_y
    del tiles

    max_x = tiles_shifted['x'].max()
    max_y = tiles_shifted['y'].max()
    image_width = int(np.ceil(max_x)) + 1
    image_height = int(np.ceil(max_y)) + 1

    if verbose:
        logging.info(f"Image size: width={image_width}, height={image_height}")

    if image_width > max_image_size or image_height > max_image_size:
        logging.warning(f"Image size ({image_width}x{image_height}) exceeds the maximum allowed size of {max_image_size}x{max_image_size}. Consider downscaling or processing in smaller tiles.")

    # Parallel processing of dimensions with progress bar
    if verbose:
        logging.info("Starting parallel processing of dimensions...")

    # Precompute the list of intermediate keys
    # intermediate_key_list = [f"na_filled_image_dim_{dim}" for dim in dimensions]
    images_dict_key = 'na_filled_images_dict'
    
    # Check if all intermediate keys are present in adata.uns
    # all_keys_in_uns = all(key in adata.uns for key in intermediate_key_list)
    
    if images_dict_key not in adata.uns:
        # Prepare arguments for parallel processing
        args_list = []
        for dim in dimensions:
            if dim >= embeddings.shape[1]:
                raise ValueError(f"Dimension {dim} is out of bounds for embeddings with shape {embeddings.shape}.")
    
            embeddings_dim = embeddings[:, dim]
    
            args = (
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
            )
    
            args_list.append(args)
            
        with tqdm_joblib(tqdm(total=len(args_list), desc="Processing dimensions")):
            results = Parallel(n_jobs=n_jobs, backend=multi_method)(
                delayed(process_single_dimension)(args) for args in args_list
            )
        na_filled_images_dict = {}
        for result in results:
            dim, image_filled = result
            # Assign image_filled to adata.uns
            na_filled_images_dict[dim] = image_filled
        del results
        adata.uns['na_filled_images_dict'] = na_filled_images_dict
        args_list = []
        # Prepare arguments for parallel processing
        for dim in dimensions:
            if dim >= embeddings.shape[1]:
                raise ValueError(f"Dimension {dim} is out of bounds for embeddings with shape {embeddings.shape}.")
            args = (
                dim,
                tiles_shifted,
                method,
                iter,
                sigma,
                box,
                threshold,
                neuman,
                across_levels,
                verbose
            )
    
            args_list.append(args)
        with tqdm_joblib(tqdm(total=len(args_list), desc="Processing dimensions")):
            results = Parallel(n_jobs=n_jobs, backend=multi_method)(
                delayed(process_single_dimension_smooth)(args) for args in args_list
            )        
    else:
        args_list = []
        # Prepare arguments for parallel processing
        for dim in dimensions:
            if dim >= embeddings.shape[1]:
                raise ValueError(f"Dimension {dim} is out of bounds for embeddings with shape {embeddings.shape}.")
            args = (
                dim,
                tiles_shifted,
                method,
                iter,
                sigma,
                box,
                threshold,
                neuman,
                across_levels,
                verbose
            )
    
            args_list.append(args)
        with tqdm_joblib(tqdm(total=len(args_list), desc="Processing dimensions")):
            results = Parallel(n_jobs=n_jobs, backend=multi_method)(
                delayed(process_single_dimension_rerun)(args) for args in args_list
            )        

    # Initialize a DataFrame to hold all smoothed embeddings
    smoothed_df = pd.DataFrame(index=adata.obs_names)
    
    smoothed_images_dict = {}
    
    # Iterate through results and assign to smoothed_df and adata.uns
    for result in results:
        dim, smoothed_barcode_grouped, smoothed_image = result
        
        smoothed_images_dict[dim] = smoothed_image
        
        # Ensure the order matches adata.obs_names
        
        smoothed_barcode_grouped = smoothed_barcode_grouped.set_index('barcode').loc[adata.obs_names].reset_index()

        # Assign to the DataFrame
        smoothed_df.loc[:, f"dim_{dim}"] = smoothed_barcode_grouped['smoothed_embed'].values

    adata.uns['smoothed_images_dict'] = smoothed_images_dict
    del results
    
    # Perform Min-Max Scaling using scikit-learn's MinMaxScaler
    if verbose:
        logging.info("Applying Min-Max scaling to the smoothed embeddings.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    smoothed_scaled = scaler.fit_transform(smoothed_df)

    # Assign the scaled smoothed embeddings to the new embedding key
    adata.obsm[output_embedding] = smoothed_scaled

    if verbose:
        logging.info(f"Smoothed embeddings stored in adata.obsm['{output_embedding}'].")

    if verbose:
        logging.info("Smoothing completed and embeddings updated in AnnData object.")

    return adata

def process_single_dimension(args):
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

    # Minimize logging inside parallel processes
    if verbose:
        logging.info(f"Processing dimension {dim}...")

    # Convert to numpy arrays for faster operations
    embed_values = embeddings_dim.astype(np.float32)
    barcodes = np.array(adata_obs_names)

    # Map barcodes to embedding values
    barcode_to_embed = dict(zip(barcodes, embed_values))

    # Assign embedding values to tiles
    embed_array = tiles_shifted['barcode'].map(barcode_to_embed).values

    # Check for any missing embeddings
    if np.isnan(embed_array).any():
        missing_barcodes = tiles_shifted['barcode'][np.isnan(embed_array)].unique()
        raise ValueError(f"Missing embedding values for barcodes: {missing_barcodes}")

    # Initialize image with NaNs
    image = np.full((image_height, image_width), np.nan, dtype=np.float32)

    # Assign embedding values to image pixels
    y_indices = tiles_shifted['y'].values
    x_indices = tiles_shifted['x'].values
    image[y_indices, x_indices] = embed_array

    # Handle multiple tiles per (x, y) by averaging embedding values
    # Since we used map above, ensure that overlapping tiles are averaged
    # Alternatively, use groupby beforehand if necessary

    # Fill NaNs using interpolation
    if verbose:
        logging.info(f"Filling NaN values in image for dimension {dim} using interpolation.")
    image_filled = fill_nan_with_opencv_inpaint(image)  # You can choose 'linear' or 'cubic' if desired
    
    return (dim, image_filled)
    
def process_single_dimension_smooth(args):
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
        verbose
    ) = args

    # Minimize logging inside parallel processes
    if verbose:
        logging.info(f"Smoothing Processing dimension {dim}...")

    # if verbose:
    #     logging.info(f"Skipping NaN fill section since na-filled image already exists for dim {dim}.")

    # Retrieve the filled image from adata.uns
    # image = adata.uns[f"na_filled_image_dim_{dim}"]
    image = adata.uns['na_filled_images_dict'][dim]
    
    # Initialize a list to store smoothed images for each iteration
    smoothed_images = []

    # Assign embedding values to image pixels
    y_indices = tiles_shifted['y'].values
    x_indices = tiles_shifted['x'].values

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
                image = cv2.medianBlur(image, kernel_size)
            elif m == 'iso':
                # Gaussian blur
                image = gaussian_filter(image, sigma=sigma, mode='reflect' if neuman else 'constant')
            elif m == 'box':
                # Box blur using average filter
                kernel_size = int(box)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                if kernel_size < 1:
                    raise ValueError("Box size must be positive for box blur.")
                image = cv2.blur(image, (kernel_size, kernel_size))
            else:
                raise ValueError(f"Smoothing method '{m}' is not recognized.")

        # Apply threshold if specified
        if threshold > 0:
            image = np.where(image >= threshold, image, 0)
        logging.info(f"Smoothing end {dim}.")
        # Append to smoothed_images list
        smoothed_images.append(image)
        logging.info(f"copy end {dim}.")
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
    logging.info(f"mapping end {dim}.")
    # Assign the smoothed values to the corresponding barcodes
    smoothed_barcode_grouped = pd.DataFrame({
        'barcode': tiles_shifted['barcode'],
        'smoothed_embed': smoothed_tile_values
    })
    logging.info(f"making end {dim}.")
    # Group by barcode and average the smoothed embeddings (in case of multiple tiles per barcode)
    smoothed_barcode_grouped = smoothed_barcode_grouped.groupby('barcode')['smoothed_embed'].mean().reset_index()
    logging.info(f"final end {dim}.")
    return (dim, smoothed_barcode_grouped, combined_image)


def process_single_dimension_rerun(args):
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
        verbose
    ) = args

    # Minimize logging inside parallel processes
    if verbose:
        logging.info(f"Processing dimension {dim}...")

    if verbose:
        logging.info(f"Skipping NaN fill section since na-filled image already exists for dim {dim}.")

    # Retrieve the filled image from adata.uns
    image = adata.uns['na_filled_images_dict'][dim]

    # Initialize a list to store smoothed images for each iteration
    smoothed_images = []

    # Assign embedding values to image pixels
    y_indices = tiles_shifted['y'].values
    x_indices = tiles_shifted['x'].values

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
                image = cv2.medianBlur(image, kernel_size)
            elif m == 'iso':
                # Gaussian blur
                image = gaussian_filter(image, sigma=sigma, mode='reflect' if neuman else 'constant')
            elif m == 'box':
                # Box blur using average filter
                kernel_size = int(box)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                if kernel_size < 1:
                    raise ValueError("Box size must be positive for box blur.")
                image = cv2.blur(image, (kernel_size, kernel_size))
            else:
                raise ValueError(f"Smoothing method '{m}' is not recognized.")

        # Apply threshold if specified
        if threshold > 0:
            image = np.where(image >= threshold, image, 0)
        logging.info(f"Smooth end {dim}.")
        # Append to smoothed_images list
        smoothed_images.append(image)
        # logging.info(f"copy end {dim}.")
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
    logging.info(f"mapping end {dim}.")
    # Assign the smoothed values to the corresponding barcodes
    # smoothed_barcode_grouped = pd.DataFrame({
    #     'barcode': tiles_shifted['barcode'],
    #     'smoothed_embed': smoothed_tile_values
    # })
    logging.info(f"making end {dim}.")
    # Group by barcode and average the smoothed embeddings (in case of multiple tiles per barcode)
    # smoothed_barcode_grouped = smoothed_barcode_grouped.groupby('barcode')['smoothed_embed'].mean().reset_index()
    smoothed_barcode_grouped = (
        pd.DataFrame({
            'barcode': tiles_shifted['barcode'],
            'smoothed_embed': smoothed_tile_values
        })
        .groupby('barcode', as_index=False)['smoothed_embed'].mean()
    )
    logging.info(f"final end {dim}.")
    return (dim, smoothed_barcode_grouped, combined_image)



def fill_nan_with_opencv_inpaint(image, method='telea', inpaint_radius=3):
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

    # Select the appropriate inpainting method
    if method == 'telea':
        cv_method = cv2.INPAINT_TELEA
    elif method == 'ns':
        cv_method = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unknown inpainting method '{method}'. Choose 'telea' or 'ns'.")

    # Perform inpainting using OpenCV's optimized function
    image_inpainted = cv2.inpaint(image_normalized, mask_uint8, inpaint_radius, cv_method)

    # Convert the inpainted image back to the original scale
    if image_max == image_min:
        # If no normalization was done, simply add the min value back
        image_filled = image_inpainted.astype(np.float32) + image_min
    else:
        # Scale the inpainted image back to the original range
        scale = (image_max - image_min) / 255.0
        image_filled = image_inpainted.astype(np.float32) * scale + image_min

    return image_filled

### equalize using tiles ###
import numpy as np
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from anndata import AnnData
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor
from typing import List
import warnings
from sklearn.preprocessing import MinMaxScaler

def equalize_image(
    adata: AnnData,
    dimensions: List[int] = [0, 1, 2],
    method: str = 'BalanceSimplest',
    N: int = 1,
    smax: float = 1.0,
    sleft: float = 1.0,
    sright: float = 1.0,
    lambda_: float = 0.1,
    up: float = 100.0,
    down: float = 10.0,
    verbose: bool = True,
    n_jobs: int = 1  
) -> AnnData:
    """
    Equalize histogram of the smoothed images stored in adata.uns['smoothed_images_dict'].
    """
    if 'smoothed_images_dict' not in adata.uns:
        raise ValueError("smoothed_images_dict not found in adata.uns. Please run smooth_image first.")

    if 'tiles' not in adata.uns:
        raise ValueError("Tiles not found in adata.uns['tiles'].")

    # Retrieve and prepare the shifted tile coordinates
    tiles_shifted = adata.uns['tiles'][['x','y','barcode']].copy()
    min_x = tiles_shifted['x'].min()
    min_y = tiles_shifted['y'].min()
    shift_x = -min_x
    shift_y = -min_y
    tiles_shifted['x'] += shift_x
    tiles_shifted['y'] += shift_y
    x_indices = tiles_shifted['x'].astype(int).values
    y_indices = tiles_shifted['y'].astype(int).values

    # Prepare a DataFrame to store equalized embeddings
    equalized_df = pd.DataFrame(index=adata.obs_names)

    def process_dimension(dim):
        if verbose:
            print(f"Equalizing dimension {dim} using method '{method}'")
        # Retrieve the smoothed image for this dimension
        if dim not in adata.uns['smoothed_images_dict']:
            raise ValueError(f"Dimension {dim} not found in smoothed_images_dict.")
        image = adata.uns['smoothed_images_dict'][dim]
        # Apply the equalization method to the image
        data_eq = apply_equalization(image, method=method, N=N, smax=smax, sleft=sleft, sright=sright, lambda_=lambda_, up=up, down=down)
        # Extract the equalized values at the positions corresponding to observations
        equalized_values = data_eq[y_indices, x_indices]
        # Map to barcodes
        equalized_barcode_grouped = pd.DataFrame({
            'barcode': tiles_shifted['barcode'],
            'equalized_embed': equalized_values
        })
        # Group by barcode in case of duplicates
        equalized_barcode_grouped = equalized_barcode_grouped.groupby('barcode', as_index=False)['equalized_embed'].mean()
        # Ensure the order matches adata.obs_names
        equalized_barcode_grouped = equalized_barcode_grouped.set_index('barcode').loc[adata.obs_names]
        return equalized_barcode_grouped['equalized_embed'].values

    # Now process all dimensions
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_dimension, dimensions))
    else:
        results = [process_dimension(dim) for dim in dimensions]

    # Assign results to equalized_df
    for i, dim in enumerate(dimensions):
        equalized_df[f'dim_{dim}'] = results[i]

    # Optionally, you can perform scaling if necessary
    # For example, MinMax scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    equalized_scaled = scaler.fit_transform(equalized_df)

    # Store the equalized embeddings in adata.obsm
    adata.obsm['X_embedding_equalize'] = equalized_scaled

    # Log the parameters used
    if verbose:
        print("Logging changes to AnnData.uns['equalize_image_log']")
    adata.uns['equalize_image_log'] = {
        'method': method,
        'parameters': {
            'N': N,
            'smax': smax,
            'sleft': sleft,
            'sright': sright,
            'lambda_': lambda_,
            'up': up,
            'down': down
        },
        'dimensions': dimensions
    }

    if verbose:
        print("Histogram equalization completed and embeddings updated in AnnData object.")

    return adata

def apply_equalization(image, method='BalanceSimplest', N=1, smax=1.0, sleft=1.0, sright=1.0, lambda_=0.1, up=100.0, down=10.0):
    """
    Apply the specified equalization method to the image.
    """
    # Flatten the image to a 1D array for methods that require it
    data = image.flatten()

    if method == 'BalanceSimplest':
        data_eq = balance_simplest(data, sleft=sleft, sright=sright)
    elif method == 'EqualizePiecewise':
        data_eq = equalize_piecewise(data, N=N, smax=smax)
    elif method == 'SPE':
        data_eq = spe_equalization(data, lambda_=lambda_)
    elif method == 'EqualizeDP':
        data_eq = equalize_dp(data, down=down, up=up)
    elif method == 'EqualizeADP':
        data_eq = equalize_adp(data)
    elif method == 'ECDF':
        data_eq = ecdf_eq(data)
    elif method == 'histogram':
        data_eq = equalize_hist(data)
    elif method == 'adaptive':
        data_eq = equalize_adapthist(image)  # For images, adaptive method can work directly
        data_eq = data_eq.flatten()
    else:
        raise ValueError(f"Unknown equalization method '{method}'")

    # Reshape back to image shape
    data_eq_image = data_eq.reshape(image.shape)
    return data_eq_image

# Equalization methods (same as before)
def balance_simplest(data: np.ndarray, sleft: float = 1.0, sright: float = 1.0, range_limits: tuple = (0, 1)) -> np.ndarray:
    p_left = np.percentile(data, sleft)
    p_right = np.percentile(data, 100 - sright)
    data_eq = rescale_intensity(data, in_range=(p_left, p_right), out_range=range_limits)
    return data_eq

def equalize_piecewise(data: np.ndarray, N: int = 1, smax: float = 1.0) -> np.ndarray:
    data_eq = np.copy(data)
    quantiles = np.linspace(0, 100, N + 1)
    for i in range(N):
        lower = np.percentile(data, quantiles[i])
        upper = np.percentile(data, quantiles[i + 1])
        mask = (data >= lower) & (data < upper)
        if np.any(mask):
            segment = rescale_intensity(data[mask], in_range=(lower, upper), out_range=(0, smax))
            data_eq[mask] = segment
    return data_eq

def spe_equalization(data: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
    data_smoothed = gaussian_filter(data, sigma=lambda_)
    data_eq = rescale_intensity(data_smoothed, out_range=(0, 1))
    return data_eq

def equalize_dp(data: np.ndarray, down: float = 10.0, up: float = 100.0) -> np.ndarray:
    p_down = np.percentile(data, down)
    p_up = np.percentile(data, up)
    data_clipped = np.clip(data, p_down, p_up)
    data_eq = rescale_intensity(data_clipped, out_range=(0, 1))
    return data_eq

def equalize_adp(data: np.ndarray) -> np.ndarray:
    data_eq = equalize_adapthist(data.reshape(1, -1), clip_limit=0.03).flatten()
    return data_eq

def ecdf_eq(data: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(data)
    ecdf = np.arange(1, len(data) + 1) / len(data)
    data_eq = np.zeros_like(data)
    data_eq[sorted_idx] = ecdf
    return data_eq

### equalize using embedding -faster but not using tiles- ###
import numpy as np
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from anndata import AnnData
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor
from typing import List
import warnings



def equalize_image(
    adata: AnnData,
    dimensions: List[int] = [0, 1, 2],
    embedding: str = 'X_embedding',
    method: str = 'BalanceSimplest',
    N: int = 1,
    smax: float = 1.0,
    sleft: float = 1.0,
    sright: float = 1.0,
    lambda_: float = 0.1,
    up: float = 100.0,
    down: float = 10.0,
    verbose: bool = True,
    n_jobs: int = 1  
) -> AnnData:
    """
    Equalize histogram of embeddings.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings.
    dimensions : list of int
        List of dimensions to use for equalization (0-based indexing).
    embedding : str
        Key in adata.obsm where the embeddings are stored.
    method : str
        Equalization method: 'BalanceSimplest', 'EqualizePiecewise', 'SPE', 
        'EqualizeDP', 'EqualizeADP', 'ECDF', 'histogram', 'adaptive'.
    N : int, optional
        Number of segments for EqualizePiecewise (default is 1).
    smax : float, optional
        Upper limit for contrast stretching in EqualizePiecewise.
    sleft : float, optional
        Percentage of pixels to saturate on the left side for BalanceSimplest (default is 1.0).
    sright : float, optional
        Percentage of pixels to saturate on the right side for BalanceSimplest (default is 1.0).
    lambda_ : float, optional
        Strength of background correction for SPE (default is 0.1).
    up : float, optional
        Upper color value threshold for EqualizeDP (default is 100.0).
    down : float, optional
        Lower color value threshold for EqualizeDP (default is 10.0).
    verbose : bool, optional
        Whether to display progress messages (default is True).
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).
    
    Returns
    -------
    AnnData
        The updated AnnData object with equalized embeddings.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")
    
    if verbose:
        print("Starting equalization...")
    
    embeddings = adata.obsm[embedding][:, dimensions].copy()
    
    def process_dimension(i, dim):
        if verbose:
            print(f"Equalizing dimension {dim} using method '{method}'")
        data = embeddings[:, i]
        
        if method == 'BalanceSimplest':
            return balance_simplest(data, sleft=sleft, sright=sright)
        elif method == 'EqualizePiecewise':
            return equalize_piecewise(data, N=N, smax=smax)
        elif method == 'SPE':
            return spe_equalization(data, lambda_=lambda_)
        elif method == 'EqualizeDP':
            return equalize_dp(data, down=down, up=up)
        elif method == 'EqualizeADP':
            return equalize_adp(data)
        elif method == 'ECDF':
            return ecdf_eq(data)
        elif method == 'histogram':
            return equalize_hist(data)
        elif method == 'adaptive':
            return equalize_adapthist(data.reshape(1, -1), clip_limit=0.03).flatten()
        else:
            raise ValueError(f"Unknown equalization method '{method}'")
    
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_dimension, range(len(dimensions)), dimensions))
    else:
        results = [process_dimension(i, dim) for i, dim in enumerate(dimensions)]
    
    for i, result in enumerate(results):
        embeddings[:, i] = result
    # Update the embeddings in adata
    if 'X_embedding_equalize' not in adata.obsm:
        adata.obsm['X_embedding_equalize'] = np.full((adata.n_obs, len(dimensions)), np.nan)
    # Update the embeddings in adata
    adata.obsm['X_embedding_equalize'][:, dimensions] = embeddings
    
    # 로그 기록
    if verbose:
        print("Logging changes to AnnData.uns['equalize_image_log']")
    adata.uns['equalize_image_log'] = {
        'method': method,
        'parameters': {
            'N': N,
            'smax': smax,
            'sleft': sleft,
            'sright': sright,
            'lambda_': lambda_,
            'up': up,
            'down': down
        },
        'dimensions': dimensions,
        'embedding': embedding
    }
    
    if verbose:
        print("Histogram equalization completed.")
    
    return adata


# Placeholder implementations for methods not available in skimage
def balance_simplest(data: np.ndarray, sleft: float = 1.0, sright: float = 1.0, range_limits: tuple = (0, 1)) -> np.ndarray:
    """
    BalanceSimplest equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    sleft : float
        Percentage of pixels to saturate on the left.
    sright : float
        Percentage of pixels to saturate on the right.
    range_limits : tuple
        Output range after rescaling.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    p_left = np.percentile(data, sleft)
    p_right = np.percentile(data, 100 - sright)
    data_eq = rescale_intensity(data, in_range=(p_left, p_right), out_range=range_limits)
    return data_eq

def equalize_piecewise(data: np.ndarray, N: int = 1, smax: float = 1.0) -> np.ndarray:
    """
    EqualizePiecewise equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    N : int
        Number of segments to divide the data into.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    data_eq = np.copy(data)
    quantiles = np.linspace(0, 100, N + 1)
    for i in range(N):
        lower = np.percentile(data, quantiles[i])
        upper = np.percentile(data, quantiles[i + 1])
        mask = (data >= lower) & (data < upper)
        if np.any(mask):
            segment = rescale_intensity(data[mask], in_range=(lower, upper), out_range=(0, 1))
            data_eq[mask] = np.minimum(segment, smax)  # smax 적용
    return data_eq

def spe_equalization(data: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
    """
    SPE (Screened Poisson Equation) equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    lambda_ : float
        Strength parameter for background correction.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    # Placeholder implementation: apply Gaussian smoothing
    data_smoothed = gaussian_filter(data, sigma=lambda_)
    data_eq = rescale_intensity(data_smoothed, out_range=(0, 1))
    return data_eq

# from scipy.sparse import diags
# from scipy.sparse.linalg import spsolve

# def spe_equalization(data: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
#     data_2d = data.reshape(1, -1)
#     n = data_2d.shape[1]
#     diagonals = [diags([1 + 2*lambda_], [0], shape=(n, n))]
#     A = diags([-lambda_, -lambda_], [-1, 1], shape=(n, n))
#     A = A + diagonals[0]
#     b = data_2d.flatten()
#     data_eq = spsolve(A, b)
#     return np.clip(data_eq, 0, 1)




def equalize_dp(data: np.ndarray, down: float = 10.0, up: float = 100.0) -> np.ndarray:
    """
    EqualizeDP (Dynamic Programming) equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    down : float
        Lower percentile threshold.
    up : float
        Upper percentile threshold.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    p_down = np.percentile(data, down)
    p_up = np.percentile(data, up)
    data_clipped = np.clip(data, p_down, p_up)
    data_eq = rescale_intensity(data_clipped, out_range=(0, 1))
    return data_eq

def equalize_adp(data: np.ndarray) -> np.ndarray:
    """
    EqualizeADP (Adaptive Dynamic Programming) equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    # Placeholder implementation: use adaptive histogram equalization
    data_eq = equalize_adapthist(data.reshape(1, -1), clip_limit=0.03).flatten()
    return data_eq

def ecdf_eq(data: np.ndarray) -> np.ndarray:
    """
    Equalize data using empirical cumulative distribution function (ECDF).
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    sorted_idx = np.argsort(data)
    ecdf = np.arange(1, len(data) + 1) / len(data)
    data_eq = np.zeros_like(data)
    data_eq[sorted_idx] = ecdf
    return data_eq


### regulization using embedding -still need update- ####
import numpy as np
from skimage.restoration import denoise_tv_chambolle




### regularise image -still need modified- ###
def regularise_image(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding',
    weight=0.1,
    n_iter_max=200,
    grid_size=None,
    verbose=True
):
    """
    Denoise embeddings via total variation regularization.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    if verbose:
        print("Starting image regularization...")

    embeddings = adata.obsm[embedding][:, dimensions].copy()

    # Get spatial coordinates
    spatial_coords = adata.obsm['spatial']
    x = spatial_coords[:, 0]
    y = spatial_coords[:, 1]

    # Create a grid mapping
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    if grid_size is None:
        # Calculate grid size based on data density
        x_range = x_max - x_min
        y_range = y_max - y_min
        grid_size = max(x_range, y_range) / 100  # Adjust 100 as needed

    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    # Map coordinates to grid indices
    x_idx = np.digitize(x, bins=x_bins) - 1
    y_idx = np.digitize(y, bins=y_bins) - 1

    # Initialize grid arrays for each dimension
    grid_shape = (len(y_bins), len(x_bins))
    embedding_grids = [np.full(grid_shape, np.nan) for _ in dimensions]

    # Place embedding values into the grids
    for i, dim in enumerate(dimensions):
        embedding_grid = embedding_grids[i]
        embedding_values = embeddings[:, i]
        embedding_grid[y_idx, x_idx] = embedding_values

    # Apply regularization to each grid
    for i in range(len(dimensions)):
        embedding_grid = embedding_grids[i]
        embedding_grid = regularise(embedding_grid, weight=weight, n_iter_max=n_iter_max)
        embedding_grids[i] = embedding_grid

    # Map the regularized grid values back to embeddings
    for i in range(len(dimensions)):
        embedding_grid = embedding_grids[i]
        embeddings[:, i] = embedding_grid[y_idx, x_idx]

    # Update the embeddings in adata
    adata.obsm[embedding][:, dimensions] = embeddings

    if verbose:
        print("Image regularization completed.")

def regularise(embedding_grid, weight=0.1, n_iter_max=200):
    """
    Denoise data using total variation regularization.
    """
    # Handle NaNs
    nan_mask = np.isnan(embedding_grid)
    if np.all(nan_mask):
        return embedding_grid  # Return as is if all values are NaN

    # Replace NaNs with mean of existing values
    mean_value = np.nanmean(embedding_grid)
    embedding_grid_filled = np.copy(embedding_grid)
    embedding_grid_filled[nan_mask] = mean_value

    # Apply total variation denoising
    # Adjust parameter based on scikit-image version
    from skimage import __version__ as skimage_version
    if skimage_version >= '0.19':
        # Use max_num_iter for scikit-image >= 0.19
        embedding_grid_denoised = denoise_tv_chambolle(
            embedding_grid_filled,
            weight=weight,
            max_num_iter=n_iter_max
        )
    else:
        # Use n_iter_max for scikit-image < 0.19
        embedding_grid_denoised = denoise_tv_chambolle(
            embedding_grid_filled,
            weight=weight,
            n_iter_max=n_iter_max
        )

    # Restore NaNs
    embedding_grid_denoised[nan_mask] = np.nan

    return embedding_grid_denoised
from sklearn.neighbors import NearestNeighbors

def regularise_image_knn(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding',
    weight=0.1,
    n_neighbors=5,
    n_iter=1,
    verbose=True
):
    """
    Denoise embeddings via KNN total variation regularization.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings.
    dimensions : list of int
        List of dimensions to use for denoising.
    embedding : str
        Key in adata.obsm where the embeddings are stored.
    weight : float
        Denoising weight parameter.
    n_neighbors : int
        Number of neighbors to use in KNN.
    n_iter : int
        Number of iterations.
    verbose : bool
        Whether to display progress messages.
    
    Returns
    -------
    None
        The function updates the embeddings in adata.obsm[embedding].
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    if verbose:
        print("Starting KNN image regularization...")

    embeddings = adata.obsm[embedding][:, dimensions].copy()
    spatial_coords = adata.obsm['spatial']

    # Build KNN graph
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)

    for iteration in range(n_iter):
        if verbose:
            print(f"Iteration {iteration + 1}/{n_iter}")
        embeddings_new = np.copy(embeddings)
        for i in range(embeddings.shape[0]):
            neighbor_indices = indices[i]
            neighbor_embeddings = embeddings[neighbor_indices]
            # Apply total variation denoising to the neighbors
            denoised_embedding = denoise_tv_chambolle(neighbor_embeddings, weight=weight)
            embeddings_new[i] = denoised_embedding[0]  # Update the current point
        embeddings = embeddings_new

    # Update the embeddings in adata
    adata.obsm[embedding][:, dimensions] = embeddings

    if verbose:
        print("KNN image regularization completed.")


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


def create_pseudo_centroids(adata, clusters, dimensions):
    active = adata
    
    for d in dimensions:
        for clust in clusters['Segment'].unique():
            loc = clusters.loc[clusters['Segment'] == clust, 'barcode'].map(lambda x: active.index.get_loc(x))
            active.iloc[loc, d] = active.iloc[loc, d].mean()
    
    return active


