import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from scipy.sparse import issparse, csr_matrix
import gc

def perform_pseudo_bulk_analysis(
    adata,
    segment_key: str = 'Segment',
    min_cells: int = 1,
    normalize_total: bool = True,
    log_transform: bool = True,
    moranI_threshold: float = 0.1,
    mode: str = "moran",
    neighbors_kwargs: dict = None,
    autocorr_kwargs: dict = None,
    aggregate_expression: bool = False,
    add_bulk_layer: bool = False,
    highly_variable: bool = True,
    perform_pca: bool = True,
    neighbors_params: dict = {'n_neighbors': 15}
) -> sc.AnnData:
    """
    Perform Pseudo-Bulk Aggregation, preprocess data, and calculate Moran's I.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to analyze.
    segment_key : str, optional (default='Segment')
        The key in adata.obs to use for pseudo-bulk aggregation.
    min_cells : int, optional (default=1)
        Minimum number of cells required for filtering genes.
    normalize_total : bool, optional (default=True)
        Whether to normalize the total counts.
    log_transform : bool, optional (default=True)
        Whether to log-transform the data.
    moranI_threshold : float, optional (default=0.1)
        Threshold for Moran's I values.
    mode : str, optional (default="moran")
        Mode for Moran's I calculation.
    neighbors_kwargs : dict, optional
        Additional arguments to pass to squidpy.gr.spatial_neighbors.
    autocorr_kwargs : dict, optional
        Additional arguments to pass to squidpy.gr.spatial_autocorr.
    aggregate_expression : bool, optional (default=False)
        Whether to enable aggregation of expression data (commented-out code).
    add_bulk_layer : bool, optional (default=False)
        Whether to add bulked expression to layers.
    highly_variable : bool, optional (default=True)
        Whether to calculate Highly Variable Genes.
    perform_pca : bool, optional (default=True)
        Whether to perform PCA.
    neighbors_params : dict, optional
        Additional arguments to pass to scanpy.pp.neighbors.
    
    Returns
    -------
    new_adata : AnnData
        A new AnnData object with pseudo-bulk aggregation applied.
    superpixel_moranI : pandas.DataFrame
        A DataFrame of genes with Moran's I values above moranI_threshold.
    """

    # Set default arguments if not provided
    if neighbors_kwargs is None:
        neighbors_kwargs = {}
    if autocorr_kwargs is None:
        autocorr_kwargs = {}
    if neighbors_params is None:
        neighbors_params = {}

    # Validate that segment_key exists in adata.obs
    if segment_key not in adata.obs:
        raise ValueError(f"'{segment_key}' key is not present in adata.obs.")

    # Pseudo-Bulk Aggregation
    print("Starting Pseudo-Bulk Aggregation...")
    
    adata_var_index = adata.var.index
    adata_X = adata.X
    adata_obsm_coords = adata.obsm['spatial']
    adata_index = adata.obs_names
    segments = adata.obs[segment_key]

    # Aggregate spatial coordinates
    pseudo_bulk_coords = pd.DataFrame(adata_obsm_coords, index=adata_index)
    pseudo_bulk_coords = pseudo_bulk_coords.groupby(segments, observed=True).mean()
    print("Spatial coordinate aggregation complete.")

    # Process segments
    segments = adata.obs[segment_key].astype("category")
    segment_count = segments.nunique()
    segments = segments.cat.rename_categories(lambda x: str(x))
    print(f"Aggregated into {segment_count} segments.")

    # Aggregate expression data
    print("Aggregating expression data...")
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
    else:
        # For dense matrix
        # X_bulk = pd.DataFrame(adata_X, index=adata_index).groupby(segments, observed=True).sum().values.astype('float32')
        X_bulk = pd.DataFrame(adata_X, index=adata_index).groupby(segments, observed=True).mean().values.astype('float32')
        X_bulk = csr_matrix(X_bulk)
    print("Expression data aggregation complete.")

    del aggregated_data
    gc.collect()

    # Prepare var DataFrame
    var_bulk = pd.DataFrame(index=adata_var_index)

    # Create new AnnData object
    new_adata = sc.AnnData(
        X=X_bulk,
        var=var_bulk
    )
    # Set the obs index to be the aggregated segments
    new_adata.obs_names = pseudo_bulk_coords.index

    # Add spatial coordinates to obsm
    new_adata.obsm['spatial'] = pseudo_bulk_coords.values

    # Add 'counts' layer
    new_adata.layers['counts'] = new_adata.X.copy()
    print("New AnnData object creation complete.")

    # Add library_id from original adata.obs if present
    if 'library_id' in adata.obs.columns:
        # Group by segment and take the first library_id for each segment.
        library_ids = adata.obs.groupby(adata.obs[segment_key], observed=True)['library_id'].first()
        # Ensure the index of library_ids matches the new_adata.obs_names
        new_adata.obs['library_id'] = library_ids.loc[new_adata.obs_names]
        print("library_id added to new_adata.obs.")

    # Optional: Aggregate expression data and add to layers
    if aggregate_expression or add_bulk_layer:
        print("Performing optional expression data aggregation...")
        if aggregate_expression:
            # Enable commented-out expression aggregation code
            bulked_expression_df = new_adata.to_df().copy()
            bulked_expression_df['Segment'] = bulked_expression_df.index
            segments_df = adata.obs[segment_key].astype(str).to_frame()
            segments_df.index = adata.obs.index
            result = segments_df.join(bulked_expression_df, on='Segment', rsuffix='_bulk')
            result = result.drop(columns=['Segment', 'Segment_bulk'])
            print("Expression data aggregation and merging complete.")

            del bulked_expression_df
            gc.collect()

            if add_bulk_layer:
                # Add to 'bulked' layer
                adata.layers['bulked'] = csr_matrix(result)
                print("'bulked' layer addition complete.")

            del result
            gc.collect()

    # Preprocessing steps
    print("Starting preprocessing...")
    sc.pp.filter_genes(new_adata, min_cells=min_cells)
    print(f"Gene filtering complete: Retained genes present in at least {min_cells} segments.")

    if normalize_total:
        sc.pp.normalize_total(new_adata, target_sum=1e4)
        print("Total count normalization complete.")

    if log_transform:
        sc.pp.log1p(new_adata)
        print("Log transformation complete.")

    # Compute spatial neighbors
    sq.gr.spatial_neighbors(new_adata, **neighbors_kwargs)
    print("Spatial neighbors computation complete.")

    # Calculate Moran's I
    sq.gr.spatial_autocorr(new_adata, mode=mode, genes=new_adata.var_names, **autocorr_kwargs)
    print("Moran's I calculation complete.")

    # Filter Moran's I results
    if "moranI" not in new_adata.uns:
        raise ValueError("Moran's I results not found in new_adata.uns['moranI'].")

    superpixel_moranI = new_adata.uns["moranI"][new_adata.uns["moranI"]['I'] > moranI_threshold].copy()
    print(f"Number of genes with Moran's I >= {moranI_threshold}: {superpixel_moranI.shape[0]}")

    # Additional preprocessing: Highly Variable Genes and PCA
    if highly_variable:
        if 'library_id' in new_adata.obs.columns:
            sc.pp.highly_variable_genes(new_adata, batch_key='library_id', inplace=True)
        else:
            sc.pp.highly_variable_genes(new_adata, inplace=True)
        print("Highly Variable Genes calculation complete.")

    if perform_pca:
        sc.tl.pca(new_adata)
        if 'library_id' in new_adata.obs.columns:
            sc.external.pp.harmony_integrate(new_adata, key='library_id')
            new_adata.obsm['X_pca'] = new_adata.obsm['X_pca_harmony'].copy()
        print("PCA complete.")

        sc.pp.neighbors(new_adata, use_rep='X_pca', **neighbors_kwargs)
        print("Neighbors computation complete.")

    print("Pseudo-Bulk analysis complete.")
    return new_adata, superpixel_moranI
