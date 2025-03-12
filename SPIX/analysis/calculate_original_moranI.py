import scanpy as sc
import squidpy as sq

def calculate_original_moranI(
    adata,
    min_cells: int = 1,
    normalize_total: bool = True,
    log_transform: bool = True,
    moranI_threshold: float = 0.1,
    mode: str = "moran",
    neighbors_kwargs: dict = None,
    autocorr_kwargs: dict = None
) -> sc.AnnData:
    """
    Preprocess gene data and calculate Moran's I spatial autocorrelation to
    return genes with Moran's I values above a specified threshold.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to analyze.
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
        Additional arguments to pass to `squidpy.gr.spatial_neighbors`.
    autocorr_kwargs : dict, optional
        Additional arguments to pass to `squidpy.gr.spatial_autocorr`.

    Returns
    -------
    original_moranI : pandas.DataFrame
        A DataFrame containing genes with Moran's I values greater than or equal to `moranI_threshold`.
    """

    # Set default arguments if not provided
    if neighbors_kwargs is None:
        neighbors_kwargs = {}
    if autocorr_kwargs is None:
        autocorr_kwargs = {}

    # Filter genes based on the minimum number of cells
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Gene filtering complete: Retained genes present in at least {min_cells} cells.")

    # Normalize total counts
    if normalize_total:
        sc.pp.normalize_total(adata)
        print("Total count normalization complete.")

    # Log-transform the data
    if log_transform:
        sc.pp.log1p(adata)
        print("Log transformation complete.")

    # Compute spatial neighbors
    sq.gr.spatial_neighbors(adata, **neighbors_kwargs)
    print("Spatial neighbors computation complete.")

    # Calculate Moran's I
    sq.gr.spatial_autocorr(adata, mode=mode, genes=adata.var_names, **autocorr_kwargs)
    print("Moran's I calculation complete.")

    # Check if Moran's I results exist in `adata.uns`
    if "moranI" not in adata.uns:
        raise ValueError("Moran's I results not found in adata.uns['moranI'].")

    # Filter Moran's I results based on the threshold
    moranI_df = adata.uns["moranI"]
    filtered_moranI = moranI_df[moranI_df['I'] > moranI_threshold].copy()
    print(f"Number of genes with Moran's I >= {moranI_threshold}: {filtered_moranI.shape[0]}")

    return filtered_moranI