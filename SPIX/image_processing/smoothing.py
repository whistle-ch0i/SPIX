import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import logging
from sklearn.preprocessing import MinMaxScaler
import cv2
import gc
from typing import List

from ..utils.utils import (
    process_single_dimension_fill_nan,
    process_single_dimension_smooth
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def smooth_image(
    adata: AnnData,
    dimensions: List[int] = [0, 1, 2],
    embedding: str = 'last',
    method: List[str] = ['iso'],
    iter: int = 1,
    sigma: float = 1.0,
    box: int = 20,
    threshold: float = 0.0,
    neuman: bool = True,
    gaussian: bool = True,
    na_rm: bool = False,
    across_levels: str = 'min',
    output_embedding: str = 'X_embedding_smooth',
    n_jobs: int = 1,
    verbose: bool = True,
    multi_method: str = 'threading',  # 'loky', 'multiprocessing', 'threading'
    max_image_size: int = 10000 
) -> AnnData:
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

    tiles = adata.uns['tiles'][['x', 'y', 'barcode']]
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
    gc.collect()
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

    # Initialize the dictionary to store filled images
    if 'na_filled_images_dict' not in adata.uns:
        args_list_fill_nan = []
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
            args_list_fill_nan.append(args)

        with tqdm_joblib(tqdm(total=len(args_list_fill_nan), desc="Filling NaNs")):
            results_fill_nan = Parallel(n_jobs=n_jobs, backend=multi_method)(
                delayed(process_single_dimension_fill_nan)(args) for args in args_list_fill_nan
            )

        na_filled_images_dict = {}
        for dim, image_filled in results_fill_nan:
            if dim == -1:
                continue  # Skip invalid results
            na_filled_images_dict[dim] = image_filled

        adata.uns['na_filled_images_dict'] = na_filled_images_dict
        del results_fill_nan
        gc.collect()

    # Now proceed to smoothing
    args_list_smooth = []
    for dim in dimensions:
        if dim >= embeddings.shape[1]:
            raise ValueError(f"Dimension {dim} is out of bounds for embeddings with shape {embeddings.shape}.")
        image_filled = adata.uns['na_filled_images_dict'][dim]
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
            verbose,
            image_filled  # **Pass the filled image as part of the arguments**
        )
        args_list_smooth.append(args)

    with tqdm_joblib(tqdm(total=len(args_list_smooth), desc="Smoothing dimensions")):
        results_smooth = Parallel(n_jobs=n_jobs, backend=multi_method)(
            delayed(process_single_dimension_smooth)(args) for args in args_list_smooth
        )

    # Initialize a DataFrame to hold all smoothed embeddings
    smoothed_df = pd.DataFrame(index=adata.obs_names)

    smoothed_images_dict = {}

    # Iterate through results and assign to smoothed_df and adata.uns
    for result in results_smooth:
        dim, smoothed_barcode_grouped, combined_image = result

        if dim == -1:
            continue  # Skip invalid results

        smoothed_images_dict[dim] = combined_image

        # Ensure the order matches adata.obs_names
        try:
            smoothed_barcode_grouped = smoothed_barcode_grouped.set_index('barcode').loc[adata.obs_names].reset_index()
        except KeyError as ke:
            logging.error(f"KeyError for dimension {dim}: {ke}")
            continue  # Skip this dimension if barcode mapping fails

        # Assign to the DataFrame
        smoothed_df.loc[:, f"dim_{dim}"] = smoothed_barcode_grouped['smoothed_embed'].values

    adata.uns['smoothed_images_dict'] = smoothed_images_dict
    del results_smooth
    gc.collect()

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
