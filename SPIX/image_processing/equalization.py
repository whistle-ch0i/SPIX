import numpy as np
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from anndata import AnnData
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor
from typing import List
import logging
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import warnings

from ..utils.utils import min_max, balance_simplest, equalize_piecewise, spe_equalization, equalize_dp, equalize_adp, ecdf_eq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        The annotated data matrix.
    dimensions : list of int, optional (default=[0, 1, 2])
        List of embedding dimensions to equalize.
    embedding : str, optional (default='X_embedding')
        Key in adata.obsm where the embeddings are stored.
    method : str, optional (default='BalanceSimplest')
        Equalization method: 'BalanceSimplest', 'EqualizePiecewise', 'SPE', 
        'EqualizeDP', 'EqualizeADP', 'ECDF', 'histogram', 'adaptive'.
    N : int, optional (default=1)
        Number of segments for EqualizePiecewise.
    smax : float, optional (default=1.0)
        Upper limit for contrast stretching in EqualizePiecewise.
    sleft : float, optional (default=1.0)
        Percentage of pixels to saturate on the left side for BalanceSimplest.
    sright : float, optional (default=1.0)
        Percentage of pixels to saturate on the right side for BalanceSimplest.
    lambda_ : float, optional (default=0.1)
        Strength of background correction for SPE.
    up : float, optional (default=100.0)
        Upper color value threshold for EqualizeDP.
    down : float, optional (default=10.0)
        Lower color value threshold for EqualizeDP.
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    n_jobs : int, optional (default=1)
        Number of parallel jobs to run.
    
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
    
    # Log changes
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
