import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from scipy.sparse import issparse, csr_matrix, vstack
import gc
from typing import Optional, Tuple, Dict, Any
import logging

# Set up logging 
# This checks if handlers are already configured to avoid adding duplicates
if not logging.getLogger('').handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get a logger instance for this module/script
_logger = logging.getLogger(__name__)

# Ensure scanpy.external is available for harmony
try:
    import scanpy.external.pp as sce
    _logger.info("scanpy.external.pp imported successfully.")
except ImportError:
    sce = None
    _logger.warning("scanpy.external not found. Harmony integration will not be available.")


def perform_pseudo_bulk_analysis(
    adata: sc.AnnData,
    segment_key: str = 'Segment',
    batch_key: Optional[str] = None,
    min_genes_in_segment: int = 1,
    normalize_total: bool = True,
    log_transform: bool = True,
    moranI_threshold: float = 0.1,
    mode: str = "moran",
    sq_neighbors_kwargs: Optional[Dict[str, Any]] = None,
    sq_autocorr_kwargs: Optional[Dict[str, Any]] = None,
    add_bulked_layer_to_original_adata: bool = False,
    highly_variable: bool = True,
    perform_pca: bool = True,
    sc_neighbors_params: Optional[Dict[str, Any]] = None
) -> Tuple[sc.AnnData, pd.DataFrame]:
    """
    Perform Pseudo-Bulk Aggregation (by sum), preprocess data,
    calculate Moran's I on the aggregated data, and optionally
    add aggregated data back to the original object. Includes optional
    batch correction (HVG, Harmony) on the pseudo-bulk data.
    Uses standard Python logging for output messages.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to analyze (Spatial transcriptome data with segments).
        Must contain `'spatial'` coordinates in `adata.obsm`.
    segment_key : str, optional (default='Segment')
        The key in `adata.obs` to use for pseudo-bulk aggregation.
        Each unique value in this column will become a pseudo-bulk observation.
    batch_key : str or None, optional (default=None)
        The key in `adata.obs` to use for batch information. If provided (not None)
        and present in `adata.obs`, this column will be added to `new_adata.obs`
         and used as the `batch_key` for batch-aware HVG calculation and the `key`
         for Harmony integration (if `perform_pca` is True). Set to None to disable
        batch correction steps.
    min_genes_in_segment : int, optional (default=1)
        Minimum number of segments a gene must be detected in (count > 0)
        in the pseudo-bulk sum matrix before normalization) to be retained
        in the pseudo-bulk AnnData object (`new_adata`).
    normalize_total : bool, optional (default=True)
        Whether to normalize the total counts (sum to 1e4) in the pseudo-bulk data
        (`new_adata.X`) after aggregation and gene filtering.
    log_transform : bool, optional (default=True)
        Whether to log-transform the pseudo-bulk data (`new_adata.X`)
        after normalization.
    moranI_threshold : float, optional (default=0.1)
        Threshold for Moran's I values. Genes with Moran's I > this value
        in the pseudo-bulk analysis will be returned.
    mode : str, optional (default="moran")
        Mode for `squidpy.gr.spatial_autocorr` ('moran', 'geary').
    sq_neighbors_kwargs : dict, optional
        Additional arguments to pass to `squidpy.gr.spatial_neighbors`
        when computing the spatial graph on the pseudo-bulk data (`new_adata`).
    sq_autocorr_kwargs : dict, optional
        Additional arguments to pass to `squidpy.gr.spatial_autocorr`.
    add_bulked_layer_to_original_adata : bool, optional (default=False)
        If True, calculate the sum expression for each gene within each segment
        and add this segment-level sum expression back to the *original* `adata`
        object as a layer named `'bulked'`. Each cell in the original `adata`
        will get the sum expression of the segment it belongs to.
        This step is performed *after* aggregation but *before* normalization/log1p
        of the pseudo-bulk data, using the sum values.
    highly_variable : bool, optional (default=True)
        Whether to calculate Highly Variable Genes on the pseudo-bulk data (`new_adata`).
        Uses `new_adata.X` (normalized/log1p if applied). If `batch_key` is provided,
        uses batch-aware HVG calculation if supported by scanpy version.
    perform_pca : bool, optional (default=True)
        Whether to perform PCA (and optional Harmony integration using `batch_key`
        if provided) on the pseudo-bulk data (`new_adata`). Uses HVGs if computed
        and available.
    sc_neighbors_params : dict, optional
        Additional arguments to pass to `scanpy.pp.neighbors` when computing
        neighbors on the PCA space of the pseudo-bulk data (`new_adata`).

    Returns
    -------
    new_adata : AnnData
        A new AnnData object representing the pseudo-bulk data,
        with aggregated expression, spatial coordinates, spatial graph,
        Moran's I results, and optionally HVGs/PCA/neighbors computed.
    superpixel_moranI : pandas.DataFrame
        A DataFrame containing genes with a Moran's I value greater than
        `moranI_threshold`, derived from `new_adata.uns['moranI']`.
        Returns an empty DataFrame if spatial autocorrelation fails or no
        genes meet the threshold.

    Raises
    -------
    ValueError
        If `segment_key` is not found in `adata.obs`.
        If `adata.obsm` does not contain `'spatial'`.
        If `batch_key` is provided but not found in `adata.obs`.
    RuntimeError
        If required spatial neighbors graph is missing for autocorrelation.
    """

    # Set default arguments if not provided
    if sq_neighbors_kwargs is None:
        sq_neighbors_kwargs = {}
    if sq_autocorr_kwargs is None:
        sq_autocorr_kwargs = {}
    if sc_neighbors_params is None:
        sc_neighbors_params = {}

    # --- Input Validation ---
    if segment_key not in adata.obs:
        _logger.error(f"'{segment_key}' key is not present in adata.obs.")
        raise ValueError(f"'{segment_key}' key is not present in adata.obs.")
    if 'spatial' not in adata.obsm:
         _logger.error("'spatial' coordinates not found in adata.obsm.")
         raise ValueError("'spatial' coordinates not found in adata.obsm.")
    if batch_key is not None and batch_key not in adata.obs.columns:
         _logger.error(f"'{batch_key}' key specified for batch correction is not present in adata.obs.")
         raise ValueError(f"'{batch_key}' key specified for batch correction is not present in adata.obs.")
    if perform_pca and batch_key is not None and sce is None:
         _logger.warning("Harmony integration requested via batch_key but scanpy.external (needed for harmony) is not available.")


    _logger.info("--- Starting Pseudo-Bulk Aggregation and Analysis ---")
    _logger.info(f"Segmenting based on key: '{segment_key}'")
    if batch_key:
        _logger.info(f"Batch key specified for correction: '{batch_key}'")

    # Prepare data for aggregation
    adata_X = adata.X
    adata_obsm_coords = adata.obsm['spatial']
    # Convert segment key to string category for robustness
    segments = adata.obs[segment_key].astype("category").cat.rename_categories(lambda x: str(x))
    # Handle potential NaN segments (groupby might drop them, ensure consistency)
    # It's safer to drop rows with NaN segments *before* aggregation if they exist
    # For now, assume segment_key has no NaNs or groupby handles them reasonably.
    unique_segments = segments.cat.categories # This gets categories present in the data (observed=True implicitly uses these)
    _logger.info(f"Found {len(unique_segments)} unique segments.")

    # --- Pseudo-Bulk Aggregation ---

    # Aggregate spatial coordinates
    _logger.info("Aggregating spatial coordinates (mean)...")
    pseudo_bulk_coords_df = pd.DataFrame(adata_obsm_coords, index=adata.obs_names)
    # observed=True ensures that only categories present in the data are used as group keys
    pseudo_bulk_coords = pseudo_bulk_coords_df.groupby(segments, observed=True).mean()
    _logger.info("Spatial coordinate aggregation complete.")
    del pseudo_bulk_coords_df
    gc.collect() # Clean up temporary DataFrame

    # Aggregate expression data (sum)
    _logger.info("Aggregating expression data (sum)...")
    # Use a more memory-efficient approach for sparse matrices
    if issparse(adata_X):
        _logger.info("Input data is sparse. Using sparse-aware aggregation.")
        # Create a mapping from original cell index to the row index in the resulting bulk matrix
        # This uses pandas category codes which align with the groupby order when observed=True
        segment_list_ordered = list(pseudo_bulk_coords.index) # Ensure order matches aggregated coords
        # Create categorical series *again* to ensure codes match the ordered unique_segments from coords groupby
        segments_cat_ordered = pd.Categorical(segments, categories=segment_list_ordered)
        original_to_bulked_row_indices = segments_cat_ordered.codes

        # Filter out any cells whose segment wasn't in the final aggregated segments (e.g. due to NaNs if not handled upstream)
        valid_cell_mask = original_to_bulked_row_indices != -1
        row_indices_map = np.arange(adata_X.shape[0])[valid_cell_mask]
        col_indices_map = original_to_bulked_row_indices[valid_cell_mask]

        num_cells_orig = adata_X.shape[0] # Original number of cells
        num_bulked_segments = len(segment_list_ordered) # Number of resulting pseudo-bulk samples
        _logger.info(f"Aggregating data from {num_cells_orig} cells into {num_bulked_segments} segments.")

        # Create an indicator matrix: rows=original cells (valid), cols=bulked segments
        data_map = np.ones(len(row_indices_map))
        indicator_matrix = csr_matrix((data_map, (row_indices_map, col_indices_map)), shape=(num_cells_orig, num_bulked_segments))

        # For dot product (indicator_matrix.T @ adata_X), if adata_X is CSR, indicator should be CSC
        # We need X_bulk_sum[j, k] = sum of adata_X[i, k] for all i belonging to segment j
        # This is achieved by indicator_matrix.T @ adata_X
        # indicator_matrix is (N_cells x N_segments), X is (N_cells x N_genes)
        # (N_segments x N_cells) @ (N_cells x N_genes) = (N_segments x N_genes)
        indicator_matrix_csc = indicator_matrix.tocsc()
        X_bulk_sum = indicator_matrix_csc.T @ adata_X # Result is (bulked_segments x genes) sparse matrix

        # # Calculate mean by dividing by cell counts per segment
        # # Get counts per segment based on the original_to_bulked_row_indices for valid cells
        # counts_per_segment = np.bincount(original_to_bulked_row_indices[valid_cell_mask], minlength=num_bulked_segments)
        # counts_col_vector = counts_per_segment.reshape(-1, 1)

        # # Avoid division by zero for segments with 0 cells (shouldn't happen with observed=True and valid_mask but safer)
        # safe_counts = np.maximum(counts_col_vector, 1)

        # # Element-wise division using sparse matrix multiplication (identity matrix with inverse counts)
        # # Result = Diag(1/counts) @ X_bulk_sum
        # inv_counts_diag = csr_matrix((1.0 / safe_counts.flatten(), (np.arange(num_bulked_segments), np.arange(num_bulked_segments))), shape=(num_bulked_segments, num_bulked_segments))
        # X_bulk_mean = inv_counts_diag @ X_bulk_sum # Result is (bulked_segments x genes) sparse matrix

        X_bulk = csr_matrix(X_bulk_sum) # Ensure final format is CSR

        del indicator_matrix, indicator_matrix_csc, X_bulk_sum, original_to_bulked_row_indices, segments_cat_ordered, valid_cell_mask
        gc.collect()

    else:
        # For dense matrix, use pandas groupby is efficient enough
        _logger.info("Input data is dense. Using pandas groupby aggregation.")
        X_bulk_df = pd.DataFrame(adata_X, index=adata.obs_names).groupby(segments, observed=True).mean()
        # Ensure order matches pseudo_bulk_coords index
        X_bulk_df = X_bulk_df.loc[pseudo_bulk_coords.index]
        X_bulk = csr_matrix(X_bulk_df.values.astype('float32')) # Convert to sparse for consistency later
        del X_bulk_df
        gc.collect()

    _logger.info("Expression data aggregation complete.")

    # --- Create New AnnData Object for Pseudo-Bulk ---
    _logger.info("Creating new AnnData object for pseudo-bulk data...")
    new_adata = sc.AnnData(
        X=X_bulk,
        obs=pd.DataFrame(index=pseudo_bulk_coords.index), # Start with empty obs, index is the segment names
        var=pd.DataFrame(index=adata.var.index) # Keep original var index
    )
    # Add spatial coordinates to obsm
    new_adata.obsm['spatial'] = pseudo_bulk_coords.values
    _logger.info("New AnnData object initialized with spatial coordinates.")
    del pseudo_bulk_coords
    gc.collect()

    # Add 'counts' layer (the raw aggregated means before norm/log)
    # This is useful to keep the original mean values
    new_adata.layers['counts'] = new_adata.X.copy()
    _logger.info("Aggregated mean counts added to 'counts' layer.")

    # Add batch_key from original adata.obs if provided and present
    perform_batch_correction = False
    if batch_key and batch_key in adata.obs.columns:
        _logger.info(f"Attempting to add '{batch_key}' to new_adata.obs...")
        try:
            # Map segments to batch_key using the first occurrence in that segment
            # Use the processed segments and the index of the bulked matrix (new_adata.obs_names)
            # to ensure mapping is correct and handles potential missing segments consistently
            segment_mapping_df = adata.obs[[segment_key, batch_key]].copy()
            # Ensure segment_key column matches the processed segments used for aggregation
            segment_mapping_df['_processed_segment'] = segments

            batch_ids = (
                segment_mapping_df
                .groupby('_processed_segment', observed=True)[batch_key]
                .first() # Get the first batch ID for each segment
            )

            # Assign to new_adata.obs ensuring index alignment with new_adata.obs_names
            # Use .reindex() to align batch_ids to the final segments present in new_adata
            new_adata.obs[batch_key] = batch_ids.reindex(new_adata.obs_names).values

            if len(new_adata.obs[batch_key].dropna().unique()) > 1:
                 perform_batch_correction = True
                 _logger.info(f"'{batch_key}' added to new_adata.obs. Multiple batches detected. Batch correction steps enabled.")
            else:
                 _logger.warning(f"'{batch_key}' added to new_adata.obs, but only one unique value found. Skipping batch correction steps.")

            del segment_mapping_df, batch_ids
            gc.collect()
        except Exception as e:
             _logger.warning(f"Failed to map and add '{batch_key}' to new_adata.obs. Error: {e}")
             batch_key = None # Set parameter to None to prevent later use
             perform_batch_correction = False # Ensure flag is False

    else:
        _logger.info("No batch key specified or key not found in original adata.obs. Skipping batch correction steps.")


    # --- Optional: Add bulked expression back to original adata ---
    # This step adds the segment-level mean expression back to the original cell data
    if add_bulked_layer_to_original_adata:
        _logger.info("Adding bulked expression layer to original adata...")
        try:
            # Get the aggregated mean expression matrix (the initial new_adata.X before filtering/norm)
            bulked_mean_X = new_adata.layers['counts'] # Use the 'counts' layer which holds the mean

            # Create a mapping from original cell index to the row index in bulked_mean_X
            # Uses the processed segments and the index of the bulked matrix (new_adata.obs_names)
            original_segments_cat = pd.Categorical(segments, categories=new_adata.obs_names)
            bulked_row_indices = original_segments_cat.codes # Maps cell index to new_adata row index (or -1 if category missing)

            # Build the new layer for the original adata
            num_cells_orig = adata.n_obs

            # Use sparse-aware indexing if bulked_mean_X is sparse
            if issparse(bulked_mean_X):
                 _logger.info("Bulked data is sparse. Using sparse-aware method to add layer to original adata.")
                 # Create an index array for the rows we need to select from bulked_mean_X
                 valid_bulked_indices = bulked_row_indices[bulked_row_indices != -1]
                 # Create an index array for the rows in the original adata where data should be placed
                 original_cell_indices_to_fill = np.arange(num_cells_orig)[bulked_row_indices != -1]

                 # Create a selection matrix: Rows = original cells (filtered), Cols = bulked rows (filtered)
                 # Entry (i, j) is 1 if original cell i maps to bulked segment j
                 select_matrix = csr_matrix((np.ones(len(valid_bulked_indices)), (original_cell_indices_to_fill, valid_bulked_indices)),
                                           shape=(num_cells_orig, new_adata.n_obs))
                 # The operation is (cells x bulked_segments) @ (bulked_segments x genes) -> (cells x genes)
                 # If bulked_mean_X is CSR, select_matrix should be CSC
                 select_matrix_csc = select_matrix.tocsc()
                 bulked_layer_matrix = select_matrix_csc @ bulked_mean_X

                 # Add to original adata layers
                 adata.layers['bulked'] = bulked_layer_matrix # Should already be CSR

                 del select_matrix, select_matrix_csc, bulked_layer_matrix, valid_bulked_indices, original_cell_indices_to_fill
                 gc.collect()

            else:
                # For dense bulked_mean_X (less likely after refactoring sparse)
                _logger.info("Bulked data is dense. Using dense indexing.")
                # Handle potential -1 codes by creating a temporary dense matrix
                bulked_layer_data = np.zeros((num_cells_orig, new_adata.n_vars), dtype=bulked_mean_X.dtype) # Initialize with zeros
                valid_mask = bulked_row_indices != -1
                bulked_layer_data[valid_mask, :] = bulked_mean_X[bulked_row_indices[valid_mask], :] # Simple indexing

                # Add to original adata layers
                adata.layers['bulked'] = bulked_layer_data

                del bulked_layer_data, valid_mask
                gc.collect()

            _logger.info("'bulked' layer added to original adata.")

        except Exception as e:
            _logger.warning(f"Failed to add 'bulked' layer to original adata. Error: {e}")
            # Continue with the rest of the analysis

    # --- Preprocessing Steps on new_adata ---
    _logger.info("--- Preprocessing Pseudo-Bulk Data ---")
    _logger.info(f"Filtering genes: Keeping genes detected in at least {min_genes_in_segment} segments.")
    initial_genes = new_adata.n_vars
    # min_cells in scanpy.pp.filter_genes corresponds to min_segments in our new_adata
    sc.pp.filter_genes(new_adata, min_cells=min_genes_in_segment)
    _logger.info(f"Gene filtering complete. Retained {new_adata.n_vars} genes (removed {initial_genes - new_adata.n_vars}).")


    if normalize_total:
        _logger.info("Normalizing total counts to 1e4...")
        # Normalization is done on the current new_adata.X, which is the mean counts after filtering
        sc.pp.normalize_total(new_adata, target_sum=1e4)
        _logger.info("Normalization complete.")

    if log_transform:
        _logger.info("Log transforming data (log1p)...")
        sc.pp.log1p(new_adata)
        _logger.info("Log transformation complete.")

    # --- Compute Spatial Neighbors and Autocorrelation on new_adata ---
    _logger.info("--- Computing Spatial Neighbors and Autocorrelation ---")
    _logger.info("Computing spatial neighbors on pseudo-bulk data using 'spatial' coordinates...")
    # Requires 'spatial' in new_adata.obsm
    try:
         # Use 'generic' coord_type as these are aggregated coords, not strictly histology
         sq.gr.spatial_neighbors(new_adata, coord_type='generic', **sq_neighbors_kwargs)
         _logger.info("Spatial neighbors computation complete.")
    except Exception as e:
         _logger.error(f"Error computing spatial neighbors: {e}")
         # Cannot proceed with spatial autocorrelation without neighbors graph
         _logger.warning("Skipping spatial autocorrelation calculation.")
         # Return the processed new_adata but no Moran's I results
         return new_adata, pd.DataFrame()


    # Calculate spatial autocorrelation
    _logger.info(f"Calculating spatial autocorrelation ({mode})...")
    # Requires the spatial graph computed by sq.gr.spatial_neighbors
    if 'spatial_neighbors' not in new_adata.uns:
         # This should have been caught by the previous try-except, but double check
         _logger.error("Spatial neighbors graph not found in new_adata.uns. Cannot compute spatial autocorrelation.")
         return new_adata, pd.DataFrame()

    try:
        # Use genes=new_adata.var_names to compute for all current genes
        sq.gr.spatial_autocorr(new_adata, mode=mode, genes=new_adata.var_names, **sq_autocorr_kwargs)
        _logger.info("Spatial autocorrelation calculation complete.")
    except Exception as e:
         _logger.error(f"Error calculating spatial autocorrelation: {e}")
         # The result 'moranI' might not be in new_adata.uns
         pass # Continue, but check for results later


    # Filter spatial autocorrelation results
    superpixel_moranI = pd.DataFrame() # Initialize as empty
    # The key depends on the mode, e.g., 'moranI' or 'gearyC'
    results_key = f"{mode}C" if mode == "geary" else f"{mode}I"

    if results_key in new_adata.uns and not new_adata.uns[results_key].empty:
        results_df = new_adata.uns[results_key]
        # Filter based on the index name relevant for the mode ('I' for Moran, 'C' for Geary)
        index_col = 'I' if mode == 'moran' else 'C'
        if index_col in results_df.columns:
             superpixel_moranI = results_df[results_df[index_col] > moranI_threshold].copy()
             _logger.info(f"Number of genes with {mode}'s {index_col} > {moranI_threshold}: {superpixel_moranI.shape[0]}")
        else:
             _logger.warning(f"Expected index column '{index_col}' not found in '{results_key}' results.")

    else:
        _logger.warning(f"Spatial autocorrelation results ('{results_key}') not found in new_adata.uns or are empty.")


    # --- Additional Pseudo-Bulk Preprocessing (HVG, PCA, Neighbors) ---
    _logger.info("--- Additional Pseudo-Bulk Preprocessing (HVG, PCA) ---")
    # These steps use new_adata.X which is normalized/log1p data


    if highly_variable:
        _logger.info("Calculating Highly Variable Genes...")
        if perform_batch_correction and batch_key in new_adata.obs.columns:
            _logger.info(f"Using '{batch_key}' for batch-aware HVG calculation.")
            try:
                sc.pp.highly_variable_genes(new_adata, batch_key=batch_key, inplace=True)
                _logger.info(f"Found {new_adata.var['highly_variable'].sum()} highly variable genes.")
            except TypeError: # Older scanpy versions might not support batch_key in HVG
                 _logger.warning("Batch-aware HVG failed (possibly due to old scanpy version). Calculating HVGs without batch key.")
                 sc.pp.highly_variable_genes(new_adata, inplace=True)
            except Exception as e:
                 _logger.warning(f"Batch-aware HVG failed: {e}. Calculating HVGs without batch key.")
                 sc.pp.highly_variable_genes(new_adata, inplace=True)
        else:
            sc.pp.highly_variable_genes(new_adata, inplace=True)
            _logger.info(f"Found {new_adata.var['highly_variable'].sum()} highly variable genes.")
        _logger.info("Highly Variable Genes calculation complete.")
    else:
        _logger.info("Skipping Highly Variable Genes calculation.")

    if perform_pca:
        _logger.info("Performing PCA...")
        # Scanpy's pca function can automatically use HVGs if `use_highly_variable=True`
        # and they have been computed and marked in new_adata.var
        use_hvg = highly_variable and 'highly_variable' in new_adata.var.columns and new_adata.var['highly_variable'].sum() > 0
        if use_hvg:
             _logger.info("Using highly variable genes for PCA.")
        else:
             _logger.warning("No highly variable genes computed or found. PCA will use all genes.")

        try:
             sc.tl.pca(new_adata, use_highly_variable=use_hvg)
             _logger.info("PCA complete.")

             # Perform Harmony integration if batch_key is available and multiple batches exist
             if perform_batch_correction:
                 if sce is not None and hasattr(sce, 'harmony_integrate'):
                     _logger.info(f"Attempting Harmony integration using '{batch_key}'...")
                     try:
                        sce.harmony_integrate(new_adata, key=batch_key)
                        # Replace main X_pca with Harmony output if successful
                        new_adata.obsm['X_pca'] = new_adata.obsm['X_pca_harmony'].copy()
                        _logger.info("Harmony integration complete. X_pca replaced with X_pca_harmony.")
                     except Exception as e:
                        _logger.warning(f"Harmony integration failed: {e}")
                        _logger.info("Using original X_pca.")
                 else:
                      _logger.warning("Harmony integration requested but scanpy.external.harmony_integrate is not available.")
                      _logger.info("Using original X_pca.")
             else:
                _logger.info("Skipping Harmony integration.")


             # Compute neighbors on the PCA space
             _logger.info("Computing neighbors on PCA space...")
             # Use the potentially Harmony-integrated X_pca
             sc.pp.neighbors(new_adata, use_rep='X_pca', **sc_neighbors_params)
             _logger.info("Neighbors computation complete.")

        except Exception as e:
             _logger.error(f"Error during PCA or Neighbors computation: {e}")
             _logger.warning("Skipping PCA and Neighbors.")
    else:
        _logger.info("Skipping PCA.")


    _logger.info("--- Pseudo-Bulk Analysis Complete ---")
    # Return moranI_df even if empty
    return new_adata, superpixel_moranI
