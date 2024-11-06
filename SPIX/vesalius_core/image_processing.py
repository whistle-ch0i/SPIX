

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
