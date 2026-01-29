import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from scipy.sparse import issparse, csr_matrix, vstack
import gc
from typing import Optional, Tuple, Dict, Any, List
import logging
from pandas.api.types import CategoricalDtype, is_numeric_dtype

# Set up logging 
# This checks if handlers are already configured to avoid adding duplicates
if not logging.getLogger('').handlers:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Get a logger instance for this module/script
_logger = logging.getLogger(__name__)

# Ensure scanpy.external is available for harmony
try:
    import scanpy.external.pp as sce
    _logger.debug("scanpy.external.pp imported successfully.")
except ImportError:
    sce = None
    _logger.warning("scanpy.external not found. Harmony integration will not be available.")


def perform_pseudo_bulk_analysis(
    adata: sc.AnnData,
    segment_key: str = 'Segment',
    segment_codes: Optional[np.ndarray] = None,
    expr_agg: str = "sum",         # "sum" or "mean"
    batch_key: Optional[str] = None,
    min_genes_in_segment: int = 1,
    normalize_total: bool = True,
    log_transform: bool = True,
    moranI_threshold: float = 0.1,
    mode: str = "moran",
    compute_spatial_autocorr: bool = True,
    sq_neighbors_kwargs: Optional[Dict[str, Any]] = None,
    sq_autocorr_kwargs: Optional[Dict[str, Any]] = None,
    add_bulked_layer_to_original_adata: bool = False,
    highly_variable: bool = True,
    perform_pca: bool = True,
    sc_neighbors_params: Optional[Dict[str, Any]] = None,
    segment_graph_strategy: str = "collapsed",
    segment_coords_strategy: str = "centroid",
    # Collapsed-graph performance/shape controls
    collapse_method: str = "bincount",              # "bincount" (fast) | "sparse_mm" (old)
    collapse_topk_per_segment: Optional[int] = None, # keep top-k neighbors per segment (None=all)
    collapse_make_binary: bool = False,              # binarize edges after collapse
    collapse_row_normalize: bool = False,            # row-normalize weights after (or binary) collapse
    collapse_compute_distances: bool = False,        # compute mean distances for segment pairs (slower)
    # Graph-based coordinate (layout) options
    segment_coords_graph_align: bool = True,         # align graph layout to centroid coords via Procrustes
    # Segment-level feature aggregation
    aggregate_tiles: bool = False,
    tile_numeric_columns: Optional[List[str]] = None,
    aggregate_obsm_keys: Optional[List[str]] = None,
    obsm_aggregate_suffix: Optional[str] = None,
    segment_obs_columns: Optional[List[str]] = None,
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
    expr_agg : str, optional (default='sum')
        Aggregation method for expression data. One of 'sum' or 'mean'.
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
    compute_spatial_autocorr : bool, optional (default=True)
        Whether to compute spatial autocorrelation on the pseudo-bulk data.
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
    segment_graph_strategy : {"centroid", "collapsed"}, optional (default="centroid")
        Strategy for building the segment-level spatial graph used by Squidpy:
        - "centroid": compute neighbors on segment centroids stored in `new_adata.obsm['spatial']`.
        - "collapsed": collapse the original spot-level spatial neighbor graph to
          segment-level using `M.T @ A @ M` (A: spot adjacency; M: membership),
          preserving true segment adjacency/boundaries. Falls back to "centroid"
          if the spot-level graph is unavailable and cannot be computed.
    segment_coords_strategy : {"centroid", "medoid", "boundary_weighted", "graph_spectral"}, optional (default="centroid")
        How to compute `new_adata.obsm['spatial']` (segment coordinates):
        - "centroid": mean of member spot coordinates (current behavior).
        - "medoid": pick the member spot closest to the centroid.
        - "boundary_weighted": weighted mean of member spot coordinates using
          each spot's cross-segment edge count (from the spot graph) as weights;
          falls back to centroid if no boundary edges.
        - "graph_spectral": 2D spectral embedding (Laplacian eigenmaps) of the
          segment graph; if `segment_coords_graph_align=True`, align to centroid
          coordinates with orthogonal Procrustes for stable orientation/scale.
    collapse_method : {"bincount", "sparse_mm"}, optional (default="bincount")
        How to collapse the spot graph to segments when `segment_graph_strategy='collapsed'`.
        - "bincount": fast vectorized accumulation on the COO edge list (memory‑friendly, default).
        - "sparse_mm": previous implementation using sparse matrix products (M.T @ A @ M).
    collapse_topk_per_segment : int or None, optional (default=None)
        If provided, keep only the top-k neighbors (by edge weight) per segment after collapse.
    collapse_make_binary : bool, optional (default=False)
        If True, convert collapsed weights to 0/1.
    collapse_row_normalize : bool, optional (default=False)
        If True, row‑normalize the collapsed adjacency (each row sums to 1 for nonzero rows).
    collapse_compute_distances : bool, optional (default=False)
        If True and spot distances exist, compute mean inter-segment distances.
        This can be slower on large graphs; keep False for performance.
    segment_coords_graph_align : bool, optional (default=True)
        When using `segment_coords_strategy='graph_spectral'`, align the graph
        layout to the original centroid coordinates via orthogonal Procrustes
        (rigid + scale) for a more interpretable overlay.
    aggregate_tiles : bool, optional (default=False)
        If True, compute mean values of tile-level numeric columns per segment and
        append them to `new_adata.obs` (prefixed with ``tile_mean_``). Requires
        `adata.uns['tiles']` with a ``'barcode'`` column for mapping.
    tile_numeric_columns : list of str or None, optional (default=None)
        Columns from `adata.uns['tiles']` to aggregate when `aggregate_tiles=True`.
        If None, all numeric columns (excluding ``'origin'``) are averaged.
    aggregate_obsm_keys : list of str or None, optional (default=None)
        Keys in `adata.obsm` whose values should be averaged per segment and stored
        on the pseudo-bulk object. Useful for propagating embeddings such as
        `'X_embedding'` or `'X_embedding_equalize'`.
    obsm_aggregate_suffix : str or None, optional (default=None)
        Optional suffix appended to aggregated embedding keys when they are stored
        in `new_adata.obsm`. If None, the original key name is reused.
    segment_obs_columns : list of str or None, optional (default=None)
        Column names from `adata.obs` to summarize per segment and append to
        `new_adata.obs`. Categorical columns are aggregated by majority vote
        (ties broken alphabetically), whereas numeric columns are averaged.
        When None, all columns whose names start with `'Segment'` except
        `segment_key` are included.

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
        If `segment_key` is not found in `adata.obs` (and `segment_codes` is None).
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
    # Validate segment graph strategy
    if segment_graph_strategy not in ("centroid", "collapsed"):
        _logger.error(f"Invalid segment_graph_strategy '{segment_graph_strategy}'. Must be 'centroid' or 'collapsed'.")
        raise ValueError(
            f"Invalid segment_graph_strategy '{segment_graph_strategy}'. Must be 'centroid' or 'collapsed'."
        )
    # Validate segment coords strategy
    if segment_coords_strategy not in ("centroid", "medoid", "boundary_weighted", "graph_spectral"):
        _logger.error(
            f"Invalid segment_coords_strategy '{segment_coords_strategy}'. Must be 'centroid', 'medoid', 'boundary_weighted', or 'graph_spectral'."
        )
        raise ValueError(
            f"Invalid segment_coords_strategy '{segment_coords_strategy}'. Must be 'centroid', 'medoid', 'boundary_weighted', or 'graph_spectral'."
        )
    if collapse_method not in ("bincount", "sparse_mm"):
        _logger.error(
            f"Invalid collapse_method '{collapse_method}'. Must be 'bincount' or 'sparse_mm'."
        )
        raise ValueError(
            f"Invalid collapse_method '{collapse_method}'. Must be 'bincount' or 'sparse_mm'."
        )

# --- Input Validation ---
    if expr_agg not in ("sum", "mean"):
        _logger.error(f"Invalid expr_agg '{expr_agg}'. Must be 'sum' or 'mean'.")
        raise ValueError(f"Invalid expr_agg '{expr_agg}'. Must be 'sum' or 'mean'.")
    if segment_codes is None:
        if segment_key not in adata.obs:
            _logger.error(f"'{segment_key}' key is not present in adata.obs.")
            raise ValueError(f"'{segment_key}' key is not present in adata.obs.")
    else:
        segment_codes = np.asarray(segment_codes)
        if segment_codes.ndim != 1:
            raise ValueError("segment_codes must be a 1D array of length adata.n_obs")
        if segment_codes.shape[0] != adata.n_obs:
            raise ValueError(
                f"segment_codes length ({segment_codes.shape[0]}) != adata.n_obs ({adata.n_obs})"
            )
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
    _logger.info(f"Segment key: '{segment_key}', expr_agg: '{expr_agg}'")
    if batch_key:
        _logger.info(f"Batch key specified for correction: '{batch_key}'")

    # Prepare data for aggregation
    adata_X = adata.X
    adata_obsm_coords = adata.obsm['spatial']
    # Prepare segments as a categorical Series.
    # Important: downstream (graph collapse, squidpy) is most reliable when segment labels are strings,
    # so we normalize categories to string labels even if codes are numeric.
    if segment_codes is None:
        segments = adata.obs[segment_key].astype("category")
        segments = segments.cat.rename_categories(lambda x: str(x))
    else:
        max_code = int(segment_codes.max()) if segment_codes.size else -1
        if max_code < 0:
            raise ValueError("segment_codes has no valid (>=0) segment ids")
        categories = np.arange(max_code + 1, dtype=np.int64).astype(str)
        segments = pd.Series(
            pd.Categorical.from_codes(segment_codes.astype(np.int32, copy=False), categories=categories),
            index=adata.obs_names,
            name=segment_key,
        )
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

    # Aggregate expression data
    _logger.info(f"Aggregating expression data ({expr_agg})...")
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

        if expr_agg == "mean":
            _logger.info("Calculating mean expression per segment...")
            # Calculate mean by dividing by cell counts per segment
            # Get counts per segment based on the original_to_bulked_row_indices for valid cells
            counts_per_segment = np.bincount(original_to_bulked_row_indices[valid_cell_mask], minlength=num_bulked_segments)
            counts_col_vector = counts_per_segment.reshape(-1, 1)

            # Avoid division by zero for segments with 0 cells (shouldn't happen with observed=True and valid_mask but safer)
            safe_counts = np.maximum(counts_col_vector, 1)

            # Element-wise division using sparse matrix multiplication (identity matrix with inverse counts)
            # Result = Diag(1/counts) @ X_bulk_sum
            inv_counts_diag = csr_matrix((1.0 / safe_counts.flatten(), (np.arange(num_bulked_segments), np.arange(num_bulked_segments))), shape=(num_bulked_segments, num_bulked_segments))
            X_bulk = inv_counts_diag @ X_bulk_sum

        else:
            _logger.info("Calculating sum expression per segment...")
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

    # Create a minimal tiles table so plotting utilities expecting adata.uns['tiles'] work
    try:
        spatial_df = pd.DataFrame(
            new_adata.obsm['spatial'],
            index=new_adata.obs_names,
        )
        # Ensure at least x/y column names; fallback to generic if higher dimensional
        coord_cols = ['x', 'y'] + [f'z{i}' for i in range(spatial_df.shape[1] - 2)]
        spatial_df.columns = coord_cols[: spatial_df.shape[1]]
        tiles_stub = spatial_df.reset_index().rename(columns={'index': 'barcode'})
        tiles_stub['barcode'] = tiles_stub['barcode'].astype(str)
        if 'origin' not in tiles_stub.columns:
            tiles_stub['origin'] = 1
        new_adata.uns['tiles'] = tiles_stub
    except Exception as exc:
        _logger.warning(f"Failed to initialize segment-level tiles table for plotting: {exc}")

    # Add 'counts' layer (the raw aggregated means before norm/log)
    # This is useful to keep the original mean values
    new_adata.layers['counts'] = new_adata.X.copy()
    _logger.info("Aggregated mean counts added to 'counts' layer.")

    # Optionally aggregate additional obs columns (e.g., other Segment labels) to segment-level summaries
    if segment_obs_columns is None:
        segment_obs_columns = [
            col for col in adata.obs.columns
            if col != segment_key and isinstance(col, str) and col.startswith("Segment")
        ]
    if segment_obs_columns:
        _logger.info(f"Aggregating supplemental obs columns per segment: {segment_obs_columns}")
        available_cols = []
        missing_cols = []
        for col in segment_obs_columns:
            if col in adata.obs.columns:
                available_cols.append(col)
            else:
                missing_cols.append(col)

        if missing_cols:
            _logger.warning(f"Requested obs columns not found and skipped: {missing_cols}")

        if available_cols:
            obs_subset = adata.obs[available_cols].copy()
            obs_subset["_segment_tmp"] = segments.values
            grouped = obs_subset.groupby("_segment_tmp", observed=True)

            numeric_cols = [col for col in available_cols if is_numeric_dtype(obs_subset[col])]
            if numeric_cols:
                try:
                    numeric_means = (
                        grouped[numeric_cols]
                        .mean()
                        .reindex(new_adata.obs_names)
                        .astype(np.float32)
                    )
                    new_adata.obs = new_adata.obs.join(numeric_means)
                except Exception as exc:
                    _logger.warning(f"Failed to aggregate numeric columns {numeric_cols}: {exc}")

            cat_cols = [col for col in available_cols if col not in numeric_cols]
            if cat_cols:
                cat_results = {}
                for col in cat_cols:
                    series = obs_subset[col]
                    try:
                        df_counts = (
                            pd.DataFrame(
                                {
                                    "_segment_tmp": obs_subset["_segment_tmp"],
                                    "value": series,
                                }
                            )
                            .dropna(subset=["value"])
                            .groupby(["_segment_tmp", "value"], observed=True)
                            .size()
                            .unstack("value", fill_value=0)
                        )
                        if df_counts.empty:
                            majority = pd.Series(np.nan, index=new_adata.obs_names)
                        else:
                            df_counts = df_counts.reindex(
                                sorted(df_counts.columns, key=lambda x: str(x)),
                                axis=1,
                            )
                            arr = df_counts.to_numpy()
                            max_counts = arr.max(axis=1)
                            mask = arr == max_counts[:, None]
                            first_idx = mask.argmax(axis=1)
                            winners = df_counts.columns.to_numpy()[first_idx]
                            majority = pd.Series(winners, index=df_counts.index)
                            majority = majority.reindex(new_adata.obs_names)
                        cat_results[col] = majority
                    except Exception as exc:
                        _logger.warning(f"Failed to aggregate column '{col}' by segment: {exc}")

                if cat_results:
                    cat_df = pd.DataFrame(cat_results)
                    new_adata.obs = new_adata.obs.join(cat_df)
                    for col, series in cat_results.items():
                        source = obs_subset[col]
                        if isinstance(source.dtype, CategoricalDtype):
                            new_adata.obs[col] = pd.Categorical(
                                new_adata.obs[col],
                                categories=list(source.cat.categories),
                            )

    # Optionally aggregate additional embeddings from obsm onto the pseudo-bulk object
    if aggregate_obsm_keys:
        _logger.info(f"Aggregating additional obsm keys per segment: {aggregate_obsm_keys}")
        for key in aggregate_obsm_keys:
            if key not in adata.obsm:
                _logger.warning(f"Requested obsm key '{key}' not found in adata.obsm; skipping.")
                continue
            emb_data = adata.obsm[key]
            if isinstance(emb_data, pd.DataFrame):
                emb_df = emb_data.copy()
            else:
                emb_array = np.asarray(emb_data)
                if emb_array.shape[0] != adata.n_obs:
                    _logger.warning(
                        f"obsm key '{key}' has shape {emb_array.shape}; expected first dimension to equal n_obs ({adata.n_obs}). Skipping aggregation."
                    )
                    continue
                emb_df = pd.DataFrame(emb_array, index=adata.obs_names)

            # Ensure only numeric columns are used for aggregation
            numeric_cols = emb_df.select_dtypes(include=[np.number]).columns
            if emb_df.shape[1] != len(numeric_cols):
                if len(numeric_cols) == 0:
                    _logger.warning(f"obsm key '{key}' does not contain numeric data; skipping aggregation.")
                    continue
                emb_df = emb_df[numeric_cols]
                _logger.debug(f"Subset obsm '{key}' to numeric columns for aggregation: {list(numeric_cols)}")

            emb_means = (
                emb_df.groupby(segments, observed=True)
                .mean()
                .reindex(new_adata.obs_names)
                .astype(np.float32)
            )
            key_out = key if obsm_aggregate_suffix is None else f"{key}{obsm_aggregate_suffix}"
            new_adata.obsm[key_out] = emb_means
            _logger.debug(f"Stored aggregated obsm key '{key_out}' with shape {emb_means.shape}.")

    # Optionally aggregate tile-level information to segment summaries
    if aggregate_tiles:
        if 'tiles' not in adata.uns:
            _logger.warning("aggregate_tiles=True but 'tiles' not found in adata.uns; skipping tile aggregation.")
        else:
            tiles_df = adata.uns['tiles']
            if not isinstance(tiles_df, pd.DataFrame):
                _logger.warning("adata.uns['tiles'] is not a pandas DataFrame; skipping tile aggregation.")
            elif 'barcode' not in tiles_df.columns:
                _logger.warning("adata.uns['tiles'] lacks 'barcode' column; cannot map tiles to segments.")
            else:
                tiles_df = tiles_df.copy()
                tiles_df['barcode'] = tiles_df['barcode'].astype(str)
                segment_map = pd.DataFrame({
                    'barcode': adata.obs_names.astype(str),
                    '_segment_tmp_key': segments.astype(str).values,
                })
                tiles_with_segments = tiles_df.merge(segment_map, on='barcode', how='inner')
                if tiles_with_segments.empty:
                    _logger.warning("No tiles matched to segments during aggregation; skipping.")
                else:
                    if tile_numeric_columns is None:
                        numeric_cols = tiles_with_segments.select_dtypes(include=[np.number]).columns.tolist()
                        # Remove helper or identifier columns
                        if '_segment_tmp_key' in numeric_cols:
                            numeric_cols.remove('_segment_tmp_key')
                        # averaging origin flag is often not meaningful; drop unless explicitly requested
                        if 'origin' in numeric_cols and tile_numeric_columns is None:
                            numeric_cols.remove('origin')
                    else:
                        numeric_cols = [col for col in tile_numeric_columns if col in tiles_with_segments.columns]
                        missing_cols = sorted(set(tile_numeric_columns) - set(numeric_cols))
                        if missing_cols:
                            _logger.warning(
                                f"Requested tile columns {missing_cols} not found in tiles DataFrame; they were skipped."
                            )

                    if not numeric_cols:
                        _logger.warning("No numeric tile columns available for aggregation after filtering; skipping.")
                    else:
                        aggregated_numeric = (
                            tiles_with_segments.groupby('_segment_tmp_key', observed=True)[numeric_cols]
                            .mean()
                            .reindex(new_adata.obs_names.astype(str))
                            .astype(np.float32)
                        )
                        aggregated_numeric.index = new_adata.obs_names  # ensure identical index object
                        tile_means_pref = aggregated_numeric.add_prefix('tile_mean_')
                        new_adata.obs = new_adata.obs.join(tile_means_pref)
                        _logger.info(
                            f"Aggregated tile columns per segment and added {tile_means_pref.shape[1]} columns to new_adata.obs."
                        )
                        # Assemble a segment-level tiles table using aggregated means
                        tiles_table = aggregated_numeric.copy()
                        # Ensure canonical coordinate columns exist; fallback to spatial centroids
                        if not {'x', 'y'}.issubset(tiles_table.columns):
                            spatial_cols = pd.DataFrame(
                                new_adata.obsm['spatial'],
                                index=new_adata.obs_names,
                            )
                            coord_cols = ['x', 'y'] + [f'z{i}' for i in range(spatial_cols.shape[1] - 2)]
                            spatial_cols.columns = coord_cols[: spatial_cols.shape[1]]
                            for col in spatial_cols.columns:
                                if col not in tiles_table.columns:
                                    tiles_table[col] = spatial_cols[col].astype(np.float32)
                        tiles_table['origin'] = 1
                        tiles_table = tiles_table.reset_index().rename(columns={'index': 'barcode'})
                        tiles_table['barcode'] = tiles_table['barcode'].astype(str)
                        new_adata.uns['tiles'] = tiles_table

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

    neighbors_built = False
    if segment_graph_strategy == "collapsed":
        _logger.info("Building segment graph by collapsing spot-level neighbor graph (M.T @ A @ M)...")
        try:
            # Ensure original spot-level spatial neighbors exist; if not, compute them.
            if 'spatial_connectivities' not in adata.obsp:
                _logger.info("Original spot-level neighbors not found. Computing with squidpy (coord_type='generic').")
                try:
                    sq.gr.spatial_neighbors(adata, coord_type='generic', **sq_neighbors_kwargs)
                except Exception as e:
                    _logger.error(f"Failed to compute original spot-level neighbors for collapse: {e}")
                    _logger.warning("Falling back to centroid-based neighbors for segments.")
                    # Fallback to centroid strategy
                    segment_graph_strategy = "centroid"

            if segment_graph_strategy == "collapsed" and 'spatial_connectivities' in adata.obsp:
                A = adata.obsp['spatial_connectivities']
                if not issparse(A):
                    A = csr_matrix(A)

                # Categories aligned to new_adata.obs_names
                seg_codes = pd.Categorical(segments, categories=new_adata.obs_names).codes
                n_cells = adata.n_obs
                n_segs = new_adata.n_obs

                if collapse_method == "sparse_mm":
                    # Previous method using sparse matrix products
                    valid_mask = seg_codes != -1
                    M = csr_matrix((np.ones(valid_mask.sum(), dtype=np.float32),
                                    (np.where(valid_mask)[0], seg_codes[valid_mask])),
                                   shape=(n_cells, n_segs))
                    AM = A @ M
                    B = (M.T @ AM).tocsr()
                    B.setdiag(0)
                    B.eliminate_zeros()
                    # Distances mean via counts
                    if 'spatial_distances' in adata.obsp:
                        D = adata.obsp['spatial_distances']
                        if not issparse(D):
                            D = csr_matrix(D)
                        DM = D @ M
                        B_dist_sum = (M.T @ DM).tocsr()
                        A_bin = A.copy().tocsr(); A_bin.data = np.ones_like(A_bin.data)
                        B_edge_counts = (M.T @ (A_bin @ M)).tocsr()
                        # Compute mean where count>0
                        from scipy.sparse import coo_matrix
                        C_coo = B_edge_counts.tocoo()
                        S_coo = B_dist_sum.tocoo()
                        mean_rows, mean_cols, mean_vals = [], [], []
                        counts_map = {(i, j): v for i, j, v in zip(C_coo.row, C_coo.col, C_coo.data)}
                        for i, j, v in zip(S_coo.row, S_coo.col, S_coo.data):
                            c = counts_map.get((i, j), 0.0)
                            if c > 0 and i != j:
                                mean_rows.append(i); mean_cols.append(j); mean_vals.append(v / c)
                        if mean_vals:
                            from scipy.sparse import coo_matrix as _coo
                            new_adata.obsp['spatial_distances'] = _coo((np.array(mean_vals, dtype=np.float32),
                                                                        (np.array(mean_rows), np.array(mean_cols))),
                                                                       shape=(n_segs, n_segs)).tocsr()
                else:
                    # Fast collapse via COO builder + sum_duplicates in SciPy (avoids Python sorting)
                    A_coo = A.tocoo()
                    row = A_coo.row; col = A_coo.col; w = A_coo.data.astype(np.float32, copy=False)
                    # Use only upper-triangular edges to avoid double counting
                    mask = row < col
                    row = row[mask]; col = col[mask]; w = w[mask]
                    s_row = seg_codes[row]; s_col = seg_codes[col]
                    # Drop edges mapping to invalid or same-segment pairs
                    valid = (s_row >= 0) & (s_col >= 0) & (s_row != s_col)
                    if not np.any(valid):
                        from scipy.sparse import csr_matrix as _csr
                        B = _csr((n_segs, n_segs), dtype=np.float32)
                    else:
                        s_row = s_row[valid]; s_col = s_col[valid]; w = w[valid]
                        from scipy.sparse import coo_matrix
                        B_dir = coo_matrix((w, (s_row, s_col)), shape=(n_segs, n_segs)).tocsr()
                        # Symmetrize
                        B = (B_dir + B_dir.T)
                        B.setdiag(0); B.eliminate_zeros()
                    # Optional: compute mean distances (slower)
                    if collapse_compute_distances and ('spatial_distances' in adata.obsp):
                        D = adata.obsp['spatial_distances']
                        if issparse(D):
                            D = D.tocoo()
                            drow = D.row; dcol = D.col; dv = D.data.astype(np.float32, copy=False)
                            dmask = drow < dcol
                            drow = drow[dmask]; dcol = dcol[dmask]; dv = dv[dmask]
                            sr = seg_codes[drow]; segc = seg_codes[dcol]
                            vmask = (sr >= 0) & (segc >= 0) & (sr != segc)
                            if np.any(vmask):
                                sr = sr[vmask]; segc = segc[vmask]; dv = dv[vmask]
                                from scipy.sparse import coo_matrix as _coo
                                D_dir = _coo((dv, (sr, segc)), shape=(n_segs, n_segs)).tocsr()
                                D_sum = (D_dir + D_dir.T)
                                # counts per segment pair from adjacency (binary)
                                B_bin = B.copy(); B_bin.data = np.ones_like(B_bin.data)
                                # Compute mean distances by dividing where counts>0
                                # Align data arrays via conversion to COO and mapping indices
                                Dc = D_sum.tocoo(); Bc = B_bin.tocoo()
                                keyD = (Dc.row.astype(np.int64) * n_segs + Dc.col.astype(np.int64))
                                keyB = (Bc.row.astype(np.int64) * n_segs + Bc.col.astype(np.int64))
                                order = np.argsort(keyB)
                                keyB_sorted = keyB[order]; cnt_sorted = Bc.data[order]
                                from bisect import bisect_left
                                k2p = {int(k): int(i) for i, k in enumerate(keyB_sorted.tolist())}
                                m_rows, m_cols, m_vals = [], [], []
                                for r, c, val, kd in zip(Dc.row, Dc.col, Dc.data, keyD):
                                    pos = k2p.get(int(kd))
                                    if pos is None:
                                        continue
                                    cnt = float(cnt_sorted[pos])
                                    if cnt > 0 and r != c:
                                        m_rows.append(int(r)); m_cols.append(int(c)); m_vals.append(float(val) / cnt)
                                if m_vals:
                                    new_adata.obsp['spatial_distances'] = _coo((np.array(m_vals, dtype=np.float32),
                                                                                (np.array(m_rows), np.array(m_cols))),
                                                                               shape=(n_segs, n_segs)).tocsr()

                # Post-process collapsed graph B (common path)
                if 'B' not in locals():
                    # If using the fast path, B is defined; in sparse_mm path it is defined as well
                    pass

                if collapse_make_binary:
                    B.data = np.ones_like(B.data, dtype=np.float32)

                if collapse_topk_per_segment is not None and collapse_topk_per_segment > 0:
                    # Keep only top-k per row
                    B = B.tolil()
                    k = collapse_topk_per_segment
                    for i in range(B.shape[0]):
                        row_data = np.asarray(B.data[i])
                        if row_data.size > k:
                            idx = np.argpartition(row_data, -k)[: row_data.size - k]
                            # zero out the smaller ones by removing them
                            cols_i = np.asarray(B.rows[i])
                            # Build mask to keep top-k
                            keep_mask = np.ones(row_data.size, dtype=bool)
                            keep_mask[idx] = False
                            B.rows[i] = cols_i[keep_mask].tolist()
                            B.data[i] = row_data[keep_mask].tolist()
                    B = B.tocsr()

                if collapse_row_normalize:
                    row_sums = np.array(B.sum(axis=1)).reshape(-1)
                    nz = row_sums > 0
                    inv = np.zeros_like(row_sums, dtype=np.float32)
                    inv[nz] = (1.0 / row_sums[nz]).astype(np.float32)
                    Dinv = csr_matrix((inv, (np.arange(B.shape[0]), np.arange(B.shape[0]))), shape=B.shape)
                    B = Dinv @ B

                new_adata.obsp['spatial_connectivities'] = B.tocsr()
                new_adata.uns['spatial_neighbors'] = {
                    'params': {'mode': f'segments_collapsed_{collapse_method}'},
                    'connectivities_key': 'spatial_connectivities',
                    'distances_key': 'spatial_distances' if 'spatial_distances' in new_adata.obsp else None,
                }
                neighbors_built = True
                _logger.info("Segment-level neighbors built via graph collapse (optimized).")

                # Optionally update segment coordinates in new_adata.obsm['spatial']
                try:
                    if segment_coords_strategy != "centroid":
                        coords_cells = np.asarray(adata.obsm['spatial'])
                        # Build membership codes aligned with new_adata.obs_names
                        seg_codes = pd.Categorical(segments, categories=new_adata.obs_names).codes
                        n_cells = adata.n_obs
                        n_segs = new_adata.n_obs

                        new_coords = new_adata.obsm['spatial'].copy()

                        if segment_coords_strategy == "medoid":
                            # Approximate medoid: pick the spot nearest to the centroid per segment
                            for s in range(n_segs):
                                idxs = np.where(seg_codes == s)[0]
                                if idxs.size == 0:
                                    continue
                                cent = coords_cells[idxs].mean(axis=0)
                                d2 = ((coords_cells[idxs] - cent) ** 2).sum(axis=1)
                                pick = idxs[np.argmin(d2)]
                                new_coords[s, :] = coords_cells[pick, :]

                        elif segment_coords_strategy == "boundary_weighted":
                            # Compute cross-segment boundary degree for each cell
                            # A: (cells x cells), M: (cells x segs)
                            A = A.tocsr()
                            M = csr_matrix((np.ones(np.count_nonzero(seg_codes != -1), dtype=np.float32),
                                            (np.where(seg_codes != -1)[0], seg_codes[seg_codes != -1])),
                                           shape=(n_cells, n_segs))
                            AM = A @ M  # (cells x segs)
                            deg_all = np.asarray(A.sum(axis=1)).ravel()
                            # internal degree per cell: ((A @ M) * M).sum(axis=1)
                            internal = np.asarray((AM.multiply(M)).sum(axis=1)).ravel()
                            boundary_deg = deg_all - internal

                            # Weighted centroid per segment; fallback to centroid if no boundary weights
                            for s in range(n_segs):
                                idxs = np.where(seg_codes == s)[0]
                                if idxs.size == 0:
                                    continue
                                w = boundary_deg[idxs]
                                sw = w.sum()
                                if sw > 0:
                                    new_coords[s, :] = (coords_cells[idxs] * w[:, None]).sum(axis=0) / sw
                                # else keep previous centroid

                        elif segment_coords_strategy == "graph_spectral":
                            # 2D Spectral embedding using normalized adjacency S = D^(-1/2) W D^(-1/2)
                            # Compute top eigenvectors (largest) of S to avoid slow shift-invert for Lsym.
                            try:
                                from scipy.sparse import diags
                                from scipy.sparse.linalg import eigsh
                                W = new_adata.obsp.get('spatial_connectivities', None)
                                if W is None:
                                    raise RuntimeError("Collapsed graph not found to compute spectral layout.")
                                W = W.tocsr()
                                # Ensure symmetry and non-negativity
                                W = W.maximum(W.T)
                                if W.nnz == 0:
                                    raise RuntimeError("Collapsed graph is empty.")
                                n_s = W.shape[0]
                                d = np.array(W.sum(axis=1)).ravel()
                                nz = d > 0
                                d_inv_sqrt = np.zeros_like(d, dtype=np.float64)
                                d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
                                D_inv_sqrt = diags(d_inv_sqrt)
                                # Normalized adjacency (symmetric)
                                S_mat = (D_inv_sqrt @ W @ D_inv_sqrt).tocsr()
                                # Decide solver strategy
                                k_need = 3 if n_s > 3 else max(2, min(n_s - 1, 2))
                                if n_s <= 1500:
                                    # Dense solver is often faster at small scale
                                    S_dense = S_mat.toarray()
                                    evals, evecs = np.linalg.eigh(S_dense)
                                    order = np.argsort(evals)  # ascending
                                    # Take the top eigenvectors (largest eigenvalues), drop the trivial first largest if possible
                                    if n_s >= 3:
                                        pick = order[-3:]
                                        evecs = evecs[:, pick]
                                        # Last column corresponds to largest eigenvalue (≈1), drop it
                                        Y = evecs[:, :2]
                                    else:
                                        Y = evecs[:, order[-2:]]
                                else:
                                    # Iterative eigensolver on S for largest magnitude eigenpairs (fast, no factorization)
                                    evals, evecs = eigsh(S_mat, k=k_need, which='LM', tol=1e-3)
                                    order = np.argsort(evals)  # ascending
                                    if evecs.shape[1] >= 3:
                                        # Top-2 non-trivial: take last 3 and drop the last
                                        idx = order[-3:]
                                        U = evecs[:, idx]
                                        Y = U[:, :2]
                                    else:
                                        Y = evecs[:, order[-2:]]
                                # Standardize axes
                                Y = Y / (np.std(Y, axis=0, ddof=1) + 1e-8)
                                if segment_coords_graph_align and new_coords.shape[1] == 2:
                                    # Align Y to current coords (centroids) with orthogonal Procrustes
                                    Xref = new_coords.copy()
                                    Xc = Xref - Xref.mean(axis=0, keepdims=True)
                                    Yc = Y - Y.mean(axis=0, keepdims=True)
                                    try:
                                        from scipy.linalg import orthogonal_procrustes
                                        R, s = orthogonal_procrustes(Yc, Xc)
                                        Y = (Yc @ R) * s + Xref.mean(axis=0, keepdims=True)
                                    except Exception:
                                        # Fallback: scale to match std and translate
                                        scale = (np.std(Xc) / (np.std(Yc) + 1e-8))
                                        Y = Yc * scale + Xref.mean(axis=0, keepdims=True)
                                new_coords = np.asarray(Y, dtype=np.float64)
                                _logger.info("Computed 'graph_spectral' coordinates (fast solver).")
                            except Exception as e:
                                _logger.warning(f"Failed to compute 'graph_spectral' coords: {e}")

                        new_adata.obsm['spatial'] = new_coords
                        _logger.info(f"Segment coordinates updated using '{segment_coords_strategy}'.")
                except Exception as e:
                    _logger.warning(f"Failed to update segment coordinates using '{segment_coords_strategy}': {e}")

        except Exception as e:
            _logger.error(f"Error while collapsing graph to segments: {e}")
            _logger.warning("Falling back to centroid-based neighbors for segments.")
            segment_graph_strategy = "centroid"

    if segment_graph_strategy == "centroid" and not neighbors_built:
        _logger.info("Computing spatial neighbors on pseudo-bulk centroids (coord_type='generic')...")
        try:
            sq.gr.spatial_neighbors(new_adata, coord_type='generic', **sq_neighbors_kwargs)
            neighbors_built = True
            _logger.info("Spatial neighbors computation complete (centroid-based).")
        except Exception as e:
            _logger.error(f"Error computing centroid-based spatial neighbors: {e}")
            if compute_spatial_autocorr:
                _logger.warning("Skipping spatial autocorrelation calculation.")
                return new_adata, pd.DataFrame()
            _logger.warning("Continuing without spatial autocorrelation.")


    superpixel_moranI = pd.DataFrame()
    if compute_spatial_autocorr:
        _logger.info(f"Calculating spatial autocorrelation ({mode})...")
        if ('spatial_connectivities' not in new_adata.obsp) and ('spatial_neighbors' not in new_adata.uns):
            _logger.error(
                "No spatial graph found in new_adata (obsp['spatial_connectivities'] or uns['spatial_neighbors'])."
            )
            return new_adata, pd.DataFrame()

        try:
            sq.gr.spatial_autocorr(new_adata, mode=mode, genes=new_adata.var_names, **sq_autocorr_kwargs)
            _logger.info("Spatial autocorrelation calculation complete.")
        except Exception as e:
            _logger.error(f"Error calculating spatial autocorrelation: {e}")

        results_key = f"{mode}C" if mode == "geary" else f"{mode}I"
        if results_key in new_adata.uns and not new_adata.uns[results_key].empty:
            results_df = new_adata.uns[results_key]
            index_col = 'I' if mode == 'moran' else 'C'
            if index_col in results_df.columns:
                superpixel_moranI = results_df[results_df[index_col] > moranI_threshold].copy()
                _logger.info(
                    f"Number of genes with {mode}'s {index_col} > {moranI_threshold}: {superpixel_moranI.shape[0]}"
                )
            else:
                _logger.warning(f"Expected index column '{index_col}' not found in '{results_key}' results.")
        else:
            _logger.warning(f"Spatial autocorrelation results ('{results_key}') not found in new_adata.uns or are empty.")
    else:
        _logger.info("Skipping spatial autocorrelation.")


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
