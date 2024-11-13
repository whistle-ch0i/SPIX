### generate embeddings and tiles ###
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import itertools
import os
import logging
from anndata import AnnData
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon
from shapely.vectorized import contains
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
