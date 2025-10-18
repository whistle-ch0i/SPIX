import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
import warnings
import gc
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import List
import squidpy as sq
import NaiveDE
#import SpatialDE
from scipy.sparse import csr_matrix, issparse

# Import necessary functions from other modules
from ..superpixel.segmentation import segment_image_inner
from ..utils.utils import (
    is_collinear,
    is_almost_collinear,
    brighten_colors,
    rebalance_colors,
    smooth_polygon,
    alpha_shape,
    plot_boundaries_on_ax,
    _check_spatial_data,
    _check_img,
    _check_scale_factor
)

# Configure logging for the analysis module
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SPIX.analysis')


def segment_image_parameter(args):
    """
    Wrapper function to process image segmentation and subsequent analysis.

    Args:
        args (tuple): A tuple containing all necessary parameters.

    Returns:
        dict: A dictionary with results for the given parameter combination.
    """
    (
        adata_obsm_coords,
        original_moranI,
        adata_index,
        adata_var_index,
        adata_X,
        embeddings,
        embeddings_df,
        barcode,
        spatial_coords,
        dimensions, 
        embedding, 
        method, 
        resolution, 
        compactness, 
        scaling, 
        n_neighbors, 
        random_state, 
        segment_col, 
        index_selection, 
        max_iter, 
    ) = args

    try:
        embeddings_df = embeddings_df.copy()

        # Perform segmentation
        clusters_df, pseudo_centroids, combined_data = segment_image_inner(
            embeddings,
            embeddings_df,
            barcode,
            spatial_coords,
            dimensions,
            method,
            resolution,
            compactness,
            scaling,
            n_neighbors,
            random_state,
            segment_col,
            index_selection,
            max_iter,
            False,
            figsize=(10, 10),
            imshow_tile_size=None,
            imshow_scale_factor=1.0,
            enforce_connectivity=False,
            pixel_shape="square",
            show_image=False,
        )

        # Calculate the number of unique segments
        segments = clusters_df['Segment']
        segments.index = clusters_df['barcode']
        segment_count = segments.nunique()
        segments = segments.astype("category")
        segments = segments.cat.rename_categories(lambda x: str(x))

        # Export Segmented Embedding data
        X = combined_data

        # Calculate Metrics
        ch_score = calinski_harabasz_score(X, segments)
        db_score = davies_bouldin_score(X, segments)

        # Pseudo-Bulk Aggregation: Group by 'Segment' and compute mean expression and spatial coordinates
        pseudo_bulk_coords = pd.DataFrame(adata_obsm_coords, index=adata_index)
        pseudo_bulk_coords = pseudo_bulk_coords.groupby(segments, observed=True).mean()

        if issparse(adata_X):
            # For sparse matrix
            unique_segments = segments.cat.categories
            segment_indices = {seg: np.where(segments == seg)[0] for seg in unique_segments}
            aggregated_data = []
            for seg in unique_segments:
                idx = segment_indices[seg]
                # mean_expr = adata_X[idx].sum(axis=0)
                mean_expr = adata_X[idx].mean(axis=0)
                if isinstance(mean_expr, np.matrix):
                    mean_expr = np.array(mean_expr).flatten()
                else:
                    mean_expr = mean_expr.A1  # Convert to 1D array
                aggregated_data.append(mean_expr)
            # Convert to dense matrix, then back to sparse
            X_bulk = csr_matrix(np.vstack(aggregated_data).astype('float32'))
            del aggregated_data
            gc.collect()
        else:
            # For dense matrix
            # X_bulk = pd.DataFrame(adata_X, index=adata_index).groupby(segments, observed=True).sum().values.astype('float32')
            X_bulk = pd.DataFrame(adata_X, index=adata_index).groupby(segments, observed=True).mean().values.astype('float32')
            X_bulk = csr_matrix(X_bulk)

        # Prepare `var` DataFrame (variable annotations remain the same)
        var_bulk = pd.DataFrame(index=adata_var_index)

        # Create the new AnnData object with aggregated data
        new_adata = sc.AnnData(
            X=X_bulk,
            var=var_bulk
        )

        # Add spatial coordinates to the new AnnData object
        new_adata.obsm['spatial'] = pseudo_bulk_coords.values

        # Preprocessing steps
        sc.pp.filter_genes(new_adata, min_cells=1)
        sc.pp.normalize_total(new_adata, inplace=True)
        sc.pp.log1p(new_adata)


        sq.gr.spatial_neighbors(new_adata)
        sq.gr.spatial_autocorr(new_adata, mode="moran", genes=new_adata.var_names)

        superpixel_moranI = new_adata.uns["moranI"][new_adata.uns["moranI"]['I'] > 0.1].copy()

        sc.pp.highly_variable_genes(new_adata, inplace=True)
        sc.tl.pca(new_adata)

        # Dynamically set n_neighbors based on the number of observations
        current_n_obs = new_adata.n_obs
        adjusted_n_neighbors = min(n_neighbors, current_n_obs - 1) if current_n_obs > 1 else 1
        sc.pp.neighbors(new_adata, n_neighbors=adjusted_n_neighbors, use_rep='X_pca')

        # Leiden Clustering
        sc.tl.leiden(new_adata, flavor="igraph", n_iterations=2, resolution=1)
        leiden_cluster_count = new_adata.obs['leiden'].nunique()

        # Clean up to free memory
        del adata_X, adata_obsm_coords, pseudo_bulk_coords, X_bulk, var_bulk, new_adata
        gc.collect()

        Superpixel_vs_original = superpixel_moranI[
            superpixel_moranI.index.isin(np.setdiff1d(superpixel_moranI.index, original_moranI.index))
        ]['I'].mean()

        original_vs_Superpixel = original_moranI[
            original_moranI.index.isin(np.setdiff1d(original_moranI.index, superpixel_moranI.index))
        ]['I'].mean()

        # Superpixel unique scores
        Superpixel_unique = superpixel_moranI[
            superpixel_moranI.index.isin(np.setdiff1d(superpixel_moranI.index, original_moranI.index))
        ]
        Superpixel_nunique = Superpixel_unique[Superpixel_unique['I'] > 0.5].index.nunique()
        Superpixel_unique_score_mean = Superpixel_unique[Superpixel_unique['I'] > 0.5]['I'].mean()
        Superpixel_unique_score_max = Superpixel_unique['I'].max()

        # Original unique scores
        original_unique = original_moranI[
            original_moranI.index.isin(np.setdiff1d(original_moranI.index, superpixel_moranI.index))
        ]
        original_nunique = original_unique[original_unique['I'] > 0.5].index.nunique()


        return {
            'resolution': resolution,
            'compactness': compactness,
            'Calinski-Harabasz': ch_score,
            'Davies-Bouldin': db_score,
            'segment_count': segment_count,
            'leiden_cluster_count': leiden_cluster_count,
            'Superpixel_nunique': Superpixel_nunique,
            'original_nunique': original_nunique,
            'Superpixel_unique_score_max': Superpixel_unique_score_max,
            'Superpixel_unique_score_mean': Superpixel_unique_score_mean,
            'Superpixel_vs_original': Superpixel_vs_original,
            'original_vs_Superpixel': original_vs_Superpixel
        }

    except Exception as e:
        # In case of error, return the parameters and the error message
        return {
            'resolution': resolution,
            'compactness': compactness,
            'error': str(e)
        }



def parameter_selection(
    adata,
    original_moranI,
    method = 'leiden_slic',
    resolutions: List[int] = [1, 5, 10, 20, 30, 100,200, 300,500,700,1000, 2000],
    compactness_values: List[int] = [1, 3, 5, 10, 20, 30, 40, 50, 100,200,500,1000],
    n_jobs: int = 1,
    scaling: float = 0.3,
    n_neighbors: int = 15,
    random_state: int = 42,
    verbose: bool = False
):
    """
    Main function to perform segmentation analysis over parameter combinations.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial and expression data.
    resolutions : list of int, optional
        List of resolution values to sweep over. Default is [1, 5, 10, 20, 30, 100, 300, 1000, 2000].
    compactness_values : list of int, optional
        List of compactness values to sweep over. Default is [1, 3, 5, 10, 20, 30, 40, 50, 100,200,500,1000].
    n_jobs : int, optional
        Number of parallel jobs to run. Default is the number of CPU cores available.
    scaling : float, optional
        Scaling factor for segmentation. Default is 0.3.
    n_neighbors : int, optional
        Number of neighbors for segmentation. Default is 15.
    random_state : int, optional
        Random state for reproducibility. Default is 42.
    verbose : bool, optional
        If True, display informational messages. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing results for each parameter combination.
    """
    # Create a local logger
    logger = logging.getLogger('SPIX.analysis.main')
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    # Avoid duplicate handlers if the function is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Ensure indices are strings to prevent implicit conversions
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    # Define parameter ranges if not provided
    if resolutions is None:
        resolutions = [1, 5, 10, 20, 30, 100, 300, 1000, 2000]
    if compactness_values is None:
        compactness_values = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]

    # Create all combinations of parameters
    parameter_combinations = product(compactness_values, resolutions)

    # Extract tiles with origin == 1
    tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]

    # Prepare embeddings and merge with tiles based on 'barcode'
    tile_colors = pd.DataFrame(np.array(adata.obsm['X_embedding_equalize'])[:, list(range(30))])
    tile_colors['barcode'] = adata.obs.index

    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="right").dropna().reset_index(drop=True)

    # Extract spatial coordinates and embeddings
    spatial_coords = coordinates_df.loc[:, ["x", "y"]].values
    embeddings = coordinates_df.drop(columns=["barcode", "x", "y", "origin"]).values  # Convert to NumPy array
    embeddings_df = coordinates_df.drop(columns=['barcode', "x", "y", "origin"])
    embeddings_df.index = coordinates_df['barcode']
    barcode = coordinates_df['barcode']
    adata_obsm_coords = adata.obsm['spatial']
    adata_index = adata.obs_names
    adata_var_index = adata.var.index
    adata_X = adata.X

    del tiles, tile_colors, coordinates_df
    gc.collect()

    # Prepare arguments for each task as a generator to save memory
    tasks = (
        (
            adata_obsm_coords,
            original_moranI,
            adata_index,
            adata_var_index,
            adata_X,
            embeddings,
            embeddings_df,
            barcode,
            spatial_coords,
            list(range(30)), 
            'X_embedding_equalize',  # embedding
            method,           # method
            resolution,
            compactness,
            scaling,
            n_neighbors,
            random_state,
            'Segment',               # segment_col
            'random',                # index_selection
            1000,                    # max_iter
        )
        for compactness, resolution in parameter_combinations
    )

    # Execute tasks in parallel
    parallel_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(segment_image_parameter)(task) for task in tasks
    )

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(parallel_results)

    # Display the results
    #print(results_df)

    # Plotting
    # Ensure that matplotlib and seaborn are imported
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plot Segment Count and Leiden Cluster Count
    plt.figure(figsize=(16, 8))

    # Subplot for Segment Count
    plt.subplot(2, 2, 1)
    sns.lineplot(data=results_df, x='resolution', y='segment_count', hue='compactness', marker='o')
    plt.title('Segment Count vs Resolution')
    plt.xlabel('Resolution')
    plt.ylabel('Number of Segments')
    plt.legend(title='Compactness', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Subplot for Leiden Cluster Count
    plt.subplot(2, 2, 2)
    sns.lineplot(data=results_df, x='resolution', y='leiden_cluster_count', hue='compactness', marker='o')
    plt.title('Leiden Cluster Count vs Resolution')
    plt.xlabel('Resolution')
    plt.ylabel('Number of Leiden Clusters')
    plt.legend(title='Compactness', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot Calinski-Harabasz and Davies-Bouldin Scores
    # Calinski-Harabasz
    plt.subplot(2, 2, 3)
    sns.lineplot(data=results_df, x='compactness', y='Calinski-Harabasz', hue='resolution', marker='o')
    plt.title('Calinski-Harabasz Score vs Compactness')
    plt.xlabel('Compactness')
    plt.ylabel('Calinski-Harabasz Score')
    plt.legend(title='Resolution', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Davies-Bouldin
    plt.subplot(2, 2, 4)
    sns.lineplot(data=results_df, x='compactness', y='Davies-Bouldin', hue='resolution', marker='o')
    plt.title('Davies-Bouldin Score vs Compactness')
    plt.xlabel('Compactness')
    plt.ylabel('Davies-Bouldin Score')
    plt.legend(title='Resolution', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return results_df
