import numpy as np
import scanpy as sc
import pandas as pd
from anndata import AnnData
from sklearn.cluster import KMeans
import logging
import time
import re

from ..utils.utils import create_pseudo_centroids

from .kmeans_segmentation import kmeans_segmentation
from .louvain_segmentation import louvain_segmentation
from .leiden_segmentation import leiden_segmentation
from .slic_segmentation import slic_segmentation
from .som_segmentation import som_segmentation
from .louvain_slic_segmentation import louvain_slic_segmentation
from .leiden_slic_segmentation import leiden_slic_segmentation
from .image_plot_slic_segmentation import image_plot_slic_segmentation, slic_segmentation_from_cached_image


def _aggregate_labels_majority(
    barcodes: pd.Series | list,
    labels: np.ndarray,
    origin_flags: np.ndarray | pd.Series | list | None = None,
) -> pd.DataFrame:
    """Aggregate tile-level labels to one label per barcode by majority vote.

    Tie-breaking rules:
    - Prefer the label with the most occurrences among tiles with origin==1
    - If still tied or origin_flags absent, pick the smallest numeric label for determinism

    Returns a DataFrame with columns ['barcode', 'Segment'] with unique barcodes.
    """
    df = pd.DataFrame({
        "barcode": pd.Series(barcodes, dtype=str).values,
        "Segment": np.asarray(labels),
    })
    if origin_flags is not None:
        df["_origin"] = np.asarray(origin_flags).astype(int)
    else:
        df["_origin"] = 0

    out_rows = []
    for bc, sub in df.groupby("barcode", sort=False):
        counts = sub["Segment"].value_counts()
        max_count = counts.max()
        candidates = counts[counts == max_count].index.to_list()
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            # Prefer the label that appears most often among origin==1 tiles
            sub_origin = sub[sub["_origin"] == 1]
            if len(sub_origin) > 0:
                origin_counts = sub_origin["Segment"].value_counts()
                best_origin = None
                best_origin_count = -1
                for c in candidates:
                    cnt = int(origin_counts.get(c, 0))
                    if cnt > best_origin_count:
                        best_origin = c
                        best_origin_count = cnt
                tie_after_origin = [c for c in candidates if int(origin_counts.get(c, 0)) == best_origin_count]
                if len(tie_after_origin) == 1:
                    chosen = tie_after_origin[0]
                else:
                    # Final deterministic fallback
                    chosen = sorted(tie_after_origin)[0]
            else:
                chosen = sorted(candidates)[0]
        out_rows.append((bc, chosen))
    return pd.DataFrame(out_rows, columns=["barcode", "Segment"])


def _aggregate_matrix_by_barcode(
    values: np.ndarray | pd.DataFrame,
    barcodes: pd.Series | list,
    unique_order: pd.Series | list | np.ndarray,
) -> np.ndarray:
    """Aggregate a row-wise matrix by barcode using mean, aligned to unique_order.

    values shape must match length of barcodes.
    Returns numpy array with len(unique_order) rows.
    """
    if values is None:
        return None
    if isinstance(values, pd.DataFrame):
        val_df = values.copy()
    else:
        val_df = pd.DataFrame(values)
    val_df["barcode"] = pd.Series(barcodes, dtype=str).values
    agg = val_df.groupby("barcode", observed=True).mean()
    agg = agg.reindex(pd.Series(unique_order, dtype=str).values)
    return agg.drop(columns=[c for c in agg.columns if c == "barcode"]).values


def _sanitize_segment_name(name: str) -> str:
    """Return a filesystem-safe suffix for storing obsm keys."""
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name))
    sanitized = sanitized.strip("_")
    if not sanitized:
        return "Segment"
    return sanitized


def _pseudo_centroids_from_segment_labels(
    embeddings: np.ndarray, segment_labels: np.ndarray
) -> np.ndarray:
    """Compute per-observation pseudo-centroids (cluster means).

    Parameters
    ----------
    embeddings
        (n_obs, n_dims) numeric array.
    segment_labels
        (n_obs,) array-like with labels; missing values allowed (NaN/None).

    Returns
    -------
    np.ndarray
        (n_obs, n_dims) float32 array where each row is the mean embedding of
        that observation's segment (NaN rows remain NaN).
    """
    labels = np.asarray(segment_labels, dtype=object)
    mask = pd.notna(labels)
    if not mask.any():
        return np.full((embeddings.shape[0], embeddings.shape[1]), np.nan, dtype=np.float32)

    codes, _ = pd.factorize(labels[mask], sort=False)
    E = np.asarray(embeddings[mask], dtype=np.float64)
    k = int(codes.max()) + 1
    sums = np.zeros((k, E.shape[1]), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    np.add.at(sums, codes, E)
    np.add.at(counts, codes, 1)
    means = sums / counts[:, None]

    out = np.full((embeddings.shape[0], embeddings.shape[1]), np.nan, dtype=np.float32)
    out[mask] = means[codes].astype(np.float32, copy=False)
    return out


def n_segments_from_size(
    tiles_df: pd.DataFrame, target_um: float = 100.0, pitch_um: float = 2.0
) -> int:
    """Calculate SLIC ``n_segments`` for a desired segment size.

    Parameters
    ----------
    tiles_df : pd.DataFrame
        DataFrame with one row per spot (e.g. ``adata.uns['tiles']``).
    target_um : float, optional (default=100.0)
        Desired approximate segment width in micrometers.
    pitch_um : float, optional (default=2.0)
        Distance between neighbouring spots in micrometers.

    Returns
    -------
    int
        Number of segments to use in SLIC to achieve the desired size.
    """

    n_spots = len(tiles_df)
    spots_per_segment = (target_um / pitch_um) ** 2
    n_segments = max(1, int(round(n_spots / spots_per_segment)))
    return n_segments


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# def segment_image(
#     adata: AnnData,
#     dimensions: list = list(range(30)),
#     embedding: str = 'X_embedding',
#     method: str = 'slic',
#     resolution: float = 10.0,
#     compactness: float = 1.0,
#     scaling: float = 0.3,
#     n_neighbors: int = 15,
#     random_state: int = 42,
#     Segment: str = 'Segment',
#     index_selection: str = 'random',
#     max_iter: int = 1000,
#     origin: bool = True,
#     verbose: bool = True,
#     library_id: str = None
# ):
#     """
#     Segment embeddings to find initial territories, with fully vectorized coordinate lookup.
#     """
#     if embedding not in adata.obsm:
#         raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

#     # --- 1) Extract embedding and barcode arrays ---
#     emb_all = adata.obsm[embedding][:, dimensions].astype(np.float32)
#     barcodes = adata.obs_names.to_numpy()  # numpy array of strings

#     # --- 2) Prepare tile arrays and vectorized mask ---
#     tiles = adata.uns['tiles']
#     if origin:
#         tiles = tiles[tiles['origin'] == 1].reset_index(drop=True)
#     tile_bcs = tiles['barcode'].to_numpy()
#     tile_x = tiles['x'].to_numpy(dtype=np.float32)
#     tile_y = tiles['y'].to_numpy(dtype=np.float32)

#     # Map barcode -> tile array index
#     tile_idx = {bc: i for i, bc in enumerate(tile_bcs)}

#     # Build boolean mask of barcodes present in tiles
#     sel_mask = np.fromiter((bc in tile_idx for bc in barcodes), dtype=bool, count=barcodes.shape[0])
#     idx_sel = np.nonzero(sel_mask)[0]

#     # --- 3) Subset embeddings & compute coords without pandas loops ---
#     emb_sel = emb_all[sel_mask]               # (N_sel, D)
#     sel_barcodes = barcodes[sel_mask]
#     tile_inds = np.fromiter((tile_idx[bc] for bc in sel_barcodes), dtype=np.int64, count=sel_barcodes.shape[0])
#     coords = np.column_stack((tile_x[tile_inds], tile_y[tile_inds]))  # (N_sel, 2)

#     # --- 4) If per-library segmentation, split in parallel ---
#     if library_id is not None:
#         if library_id not in adata.obs.columns:
#             raise ValueError(f"Library ID column '{library_id}' not found in adata.obs.")
#         libs = adata.obs[library_id].to_numpy()[sel_mask]
#         unique_libs = np.unique(libs)

#         def _process_lib(lib):
#             sub = (libs == lib)
#             labels, pc, comb = segment_image_inner(
#                 emb_sel[sub], coords[sub],
#                 method, resolution, compactness,
#                 scaling, n_neighbors, random_state,
#                 index_selection, max_iter, verbose
#             )
#             # prefix for uniqueness
#             labels = np.array([f"{lib}_{lbl}" for lbl in labels], dtype=object)
#             return sub, labels, pc, comb

#         results = Parallel(n_jobs=-1)(
#             delayed(_process_lib)(lib) for lib in unique_libs
#         )

#         full_labels = np.empty(sel_mask.sum(), dtype=object)
#         full_pc = np.zeros((sel_mask.sum(), emb_sel.shape[1]), dtype=np.float32)
#         full_comb = None
#         if any(r[3] is not None for r in results):
#             full_comb = np.zeros((sel_mask.sum(), emb_sel.shape[1] + 2), dtype=np.float32)

#         for sub, labels, pc, comb in results:
#             full_labels[sub] = labels
#             full_pc[sub] = pc
#             if comb is not None:
#                 full_comb[sub] = comb

#     else:
#         # single batch
#         full_labels, full_pc, full_comb = segment_image_inner(
#             emb_sel, coords,
#             method, resolution, compactness,
#             scaling, n_neighbors, random_state,
#             index_selection, max_iter, verbose
#         )

#     # --- 5) Write results back into AnnData ---
#     # Segment labels
#     seg_col = np.empty(adata.n_obs, dtype=object)
#     seg_col[idx_sel] = full_labels
#     adata.obs[Segment] = pd.Categorical(seg_col)

#     # Pseudo-centroids
#     pc_array = np.zeros((adata.n_obs, full_pc.shape[1]), dtype=np.float32)
#     pc_array[idx_sel] = full_pc
#     adata.obsm['X_embedding_segment'] = pc_array

#     # Combined data
#     if full_comb is not None:
#         comb_array = np.zeros((adata.n_obs, full_comb.shape[1]), dtype=np.float32)
#         comb_array[idx_sel] = full_comb
#         adata.obsm['X_embedding_scaled_for_segment'] = comb_array


# def segment_image_inner(
#     embeddings: np.ndarray,
#     spatial_coords: np.ndarray,
#     method: str,
#     resolution: float,
#     compactness: float,
#     scaling: float,
#     n_neighbors: int,
#     random_state: int,
#     index_selection: str,
#     max_iter: int,
#     verbose: bool
# ):
#     """
#     Core segmentation logic returning:
#       - cluster labels (1d np.array)
#       - pseudo-centroids (2d np.ndarray)
#       - combined data (2d np.ndarray or None)
#     """
#     if verbose:
#         logging.info(f"Starting segmentation using '{method}'")

#     combined = None
#     if method == 'kmeans':
#         clusters = kmeans_segmentation(embeddings, resolution, random_state)
#     elif method == 'louvain':
#         clusters = louvain_segmentation(embeddings, resolution, n_neighbors, random_state)
#     elif method == 'leiden':
#         clusters = leiden_segmentation(embeddings, resolution, n_neighbors, random_state)
#     elif method == 'slic':
#         clusters, combined = slic_segmentation(
#             embeddings, spatial_coords,
#             n_segments=int(resolution),
#             compactness=compactness,
#             scaling=scaling,
#             index_selection=index_selection,
#             max_iter=max_iter,
#             verbose=verbose
#         )
#     elif method == 'leiden_slic':
#         clusters, combined = leiden_slic_segmentation(
#             embeddings, spatial_coords,
#             resolution, compactness, scaling, n_neighbors, random_state
#         )
#     elif method == 'louvain_slic':
#         clusters, combined = louvain_slic_segmentation(
#             embeddings, spatial_coords,
#             resolution, compactness, scaling, n_neighbors, random_state
#         )
#     else:
#         raise ValueError(f"Unknown segmentation method '{method}'")

#     # compute pseudo-centroids in pure numpy
#     pseudo = _compute_pseudo_centroids_np(embeddings, clusters)

#     if verbose:
#         logging.info("Segmentation completed.")
#     return clusters, pseudo, combined

# def _compute_pseudo_centroids_np(embeddings: np.ndarray, clusters: np.ndarray) -> np.ndarray:
#     """
#     Compute per-cluster means in pure numpy:
#      1) unique + inverse mapping
#      2) sum & count via np.add.at
#      3) divide to get means
#      4) broadcast back to each point
#     """
#     uniq, inv = np.unique(clusters, return_inverse=True)
#     K, D = uniq.shape[0], embeddings.shape[1]

#     sums = np.zeros((K, D), dtype=np.float64)
#     counts = np.zeros(K, dtype=np.int64)
#     np.add.at(sums, inv, embeddings.astype(np.float64))
#     np.add.at(counts, inv, 1)
#     means = (sums / counts[:, None]).astype(np.float32)

#     return means[inv]


def segment_image(
    adata: AnnData,
    dimensions: list = list(range(30)),
    embedding: str = "X_embedding",
    method: str = "slic",
    resolution: float = 10.0,
    compactness: float = 1.0,
    scaling: float = 0.3,
    n_neighbors: int = 15,
    random_state: int = 42,
    Segment: str = "Segment",
    index_selection: str = "random",
    max_iter: int = 1000,
    origin: bool = True,
    verbose: bool = True,
    library_id: str = None,  # Specify library_id column name in adata.obs
    figsize: tuple = (10, 10),
    fig_dpi: int | None = None,
    imshow_tile_size: float | None = None,
    imshow_scale_factor: float = 1.0,
    enforce_connectivity: bool = False,
    pixel_shape: str = "square",
    show_image: bool = False,
    target_segment_um: float | None = None,
    pitch_um: float = 2.0,
    n_jobs: int | None = None,
    use_cached_image: bool = False,
    image_cache_key: str = "image_plot_slic",
    cache_fast_path: bool = True,
):
    """
    Segment embeddings to find initial territories.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    dimensions : list of int, optional (default=list(range(30)))
        List of embedding dimensions to use for segmentation.
    embedding : str, optional (default='X_embedding')
        Key in adata.obsm where the embeddings are stored.
    method : str, optional (default='slic')
        Segmentation method: 'kmeans', 'louvain', 'leiden', 'slic', 'som',
        'leiden_slic', 'louvain_slic', 'image_plot_slic'.
    resolution : float, optional (default=10.0)
        Resolution parameter for clustering methods.
    compactness : float, optional (default=1.0)
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float, optional (default=0.3)
        Scaling factor for spatial coordinates.
    n_neighbors : int, optional (default=15)
        Number of neighbors for constructing the nearest neighbor graph.
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    Segment : str, optional (default='Segment')
        Name of the segment column to store in adata.obs.
    index_selection : str, optional (default='random')
        Method for selecting initial cluster centers ('bubble', 'random', 'hex').
    max_iter : int, optional (default=1000)
        Maximum number of iterations for convergence in initial cluster selection.
    origin : bool, optional (default=True)
        Whether to use only origin tiles.
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    library_id : str, optional (default=None)
        Column name in adata.obs to group by. If specified, segmentation is performed separately for each group.
    figsize : tuple, optional (default=(10, 10))
        Figure size used when creating the intermediate image for ``image_plot_slic``.
    fig_dpi : int or None, optional (default=None)
        DPI used when constructing the internal image for ``image_plot_slic``.
    imshow_tile_size : float or None, optional
        Tile size used for estimating pixel scaling in ``image_plot_slic``.
    imshow_scale_factor : float, optional (default=1.0)
        Additional scaling factor for ``image_plot_slic``.
    enforce_connectivity : bool, optional (default=False)
        Whether to enforce connectivity in ``image_plot_slic``.
    pixel_shape : str, optional (default ``"square"``)
        Shape used for rasterizing spots in ``image_plot_slic``.
    show_image : bool, optional (default ``False``)
        If ``True`` display the intermediate images produced by
        ``image_plot_slic_segmentation``.
    n_jobs : int or None, optional (default=None)
        Number of worker threads for ``image_plot_slic`` rasterization. When
        ``None``, defers to ``SPIX_IMAGE_PLOT_SLIC_N_JOBS`` env var or 1.
    use_cached_image : bool, optional (default=False)
        If True and ``method='image_plot_slic'``, use a precomputed raster image
        stored in ``adata.uns[image_cache_key]`` instead of regenerating.
    image_cache_key : str, optional (default='image_plot_slic')
        Key in ``adata.uns`` where the cached image dict is stored.
    target_segment_um : float or None, optional (default=None)
        Desired segment size in micrometers. If provided, ``resolution``
        is ignored and the number of segments is calculated automatically.
    pitch_um : float, optional (default=2.0)
        Distance between neighbouring spots in micrometers used when
        ``target_segment_um`` is specified.

    Returns
    -------
    None
        Updates the AnnData object with segmentation information.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Retrieve tiles from adata.uns
    tiles_all = adata.uns["tiles"]
    if origin:
        tiles = tiles_all[tiles_all.get("origin", 1) == 1]
    else:
        tiles = tiles_all

    # For target-based segment estimation, always base on origin==1 spots
    tiles_for_n = tiles_all
    try:
        if "origin" in tiles_all.columns and (tiles_all["origin"] == 1).any():
            tiles_for_n = tiles_all[tiles_all["origin"] == 1]
        # ensure one row per barcode
        if "barcode" in tiles_for_n.columns:
            tiles_for_n = tiles_for_n.drop_duplicates("barcode")
    except Exception:
        pass

    if target_segment_um is not None:
        resolution = n_segments_from_size(
            tiles_for_n, target_um=target_segment_um, pitch_um=pitch_um
        )
        if verbose:
            print(f"→ SLIC  n_segments = {resolution}")

    # Fast path: cached image SLIC without building coordinates_df/embeddings_df
    if (
        method == "image_plot_slic"
        and use_cached_image
        and library_id is None
        and cache_fast_path
        and image_cache_key in adata.uns
    ):
        cache = adata.uns[image_cache_key]
        cached_emb = cache.get("embedding_key", None)
        cached_dims = cache.get("dimensions", None)
        if (cached_emb is None or cached_emb == embedding) and (
            cached_dims is None or list(cached_dims) == list(dimensions)
        ):
            if show_image:
                cache_figsize = cache.get("figsize", None)
                cache_figdpi = cache.get("fig_dpi", None)
                print(
                    f"[info] use_cached_image=True: showing with cached figsize={cache_figsize}, fig_dpi={cache_figdpi}"
                )

            t0 = time.perf_counter()
            labels_tile, _ = slic_segmentation_from_cached_image(
                cache,
                n_segments=int(resolution),
                compactness=compactness,
                enforce_connectivity=enforce_connectivity,
                show_image=show_image,
                verbose=verbose,
                figsize=None,
                fig_dpi=None,
            )
            t1 = time.perf_counter()

            seg_vals = np.full(adata.n_obs, np.nan, dtype=object)
            tile_obs_indices = cache.get("tile_obs_indices", None)
            if tile_obs_indices is not None:
                idx = np.asarray(tile_obs_indices, dtype=np.int64)
                tile_labels = np.asarray(labels_tile)
                if idx.ndim != 1 or idx.size != tile_labels.size:
                    raise ValueError(
                        f"Cached image '{image_cache_key}' has tile_obs_indices of length {idx.size}, "
                        f"but got {tile_labels.size} tile labels."
                    )
                # Fast path: one tile per obs (unique indices)
                if np.unique(idx).size == idx.size:
                    seg_vals[idx] = tile_labels
                else:
                    # Fallback: if duplicates exist, use barcode aggregation if available.
                    cache_barcodes = pd.Series(cache.get("barcodes") or [], dtype=str)
                    if cache_barcodes.empty:
                        raise ValueError(
                            f"Cached image '{image_cache_key}' has duplicate tile_obs_indices but no barcodes to aggregate."
                        )
                    clusters_df = _aggregate_labels_majority(
                        cache_barcodes,
                        tile_labels,
                        origin_flags=cache.get("origin_flags", None),
                    )
                    tile_barcodes = clusters_df["barcode"].astype(str).to_numpy()
                    tile_labels2 = clusters_df["Segment"].to_numpy()
                    obs_barcodes = pd.Index(adata.obs.index.astype(str))
                    obs_pos = obs_barcodes.get_indexer(pd.Index(tile_barcodes))
                    ok = obs_pos >= 0
                    seg_vals[obs_pos[ok]] = tile_labels2[ok]
            else:
                # Backward-compatible mapping by barcode list
                cache_barcodes = pd.Series(cache.get("barcodes") or [], dtype=str)
                if cache_barcodes.empty:
                    raise ValueError(f"Cached image '{image_cache_key}' missing barcodes.")

                if cache_barcodes.duplicated().any():
                    clusters_df = _aggregate_labels_majority(
                        cache_barcodes,
                        labels_tile,
                        origin_flags=cache.get("origin_flags", None),
                    )
                    tile_barcodes = clusters_df["barcode"].astype(str).to_numpy()
                    tile_labels = clusters_df["Segment"].to_numpy()
                else:
                    tile_barcodes = cache_barcodes.to_numpy()
                    tile_labels = np.asarray(labels_tile)

                obs_barcodes = pd.Index(adata.obs.index.astype(str))
                obs_pos = obs_barcodes.get_indexer(pd.Index(tile_barcodes))
                ok = obs_pos >= 0
                seg_vals[obs_pos[ok]] = tile_labels[ok]
            adata.obs[Segment] = pd.Categorical(seg_vals)

            # Compute pseudo-centroids in obs order
            E_full = np.asarray(adata.obsm[embedding])
            if E_full.shape[1] == len(dimensions):
                E_use = E_full
            else:
                E_use = E_full[:, dimensions]
            pseudo_centroids = _pseudo_centroids_from_segment_labels(E_use, seg_vals)
            segment_suffix = _sanitize_segment_name(Segment)
            adata.obsm["X_embedding_segment"] = pseudo_centroids
            adata.obsm[f"X_embedding_{segment_suffix}"] = pseudo_centroids

            # No combined_data for image_plot_slic
            for key in ("X_embedding_scaled_for_segment", f"X_embedding_scaled_for_{segment_suffix}"):
                if key in adata.obsm:
                    del adata.obsm[key]

            if verbose:
                print(f"[perf] cached SLIC (fast path): {t1 - t0:.3f}s")
            return

    # Create a DataFrame from the selected embedding dimensions and add barcode from adata.obs index
    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors["barcode"] = adata.obs.index
    # Base per-barcode embeddings for centroid computation (unique index)
    base_embeddings_df = tile_colors.drop(columns=["barcode"]).copy()
    base_embeddings_df.index = tile_colors["barcode"].astype(str)

    # Merge tiles with embeddings based on 'barcode'
    coordinates_df = (
        pd.merge(tiles, tile_colors, on="barcode", how="right")
        .dropna()
        .reset_index(drop=True)
    )

    # If library_id is specified, merge the column from adata.obs into coordinates_df
    if library_id is not None:
        if library_id not in adata.obs.columns:
            raise ValueError(
                f"Library ID column '{library_id}' not found in adata.obs."
            )
        lib_info = adata.obs[[library_id]].copy()
        lib_info["barcode"] = lib_info.index
        coordinates_df = pd.merge(coordinates_df, lib_info, on="barcode", how="left")

    # If library_id is specified, perform segmentation per library group
    if library_id is not None:
        clusters_list = []
        pseudo_centroids_list = []
        combined_data_list = []
        # Group by the specified library_id column
        # Pass observed=False to silence FutureWarning (or observed=True to adopt future behavior)
        for lib, group in coordinates_df.groupby(library_id, observed=False):
            if verbose:
                print(f"Segmenting library_id: {lib} ...")
            # Extract spatial coordinates
            spatial_coords_group = group[["x", "y"]].values
            # Drop columns that are not part of the embedding
            drop_cols = ["barcode", "x", "y", "origin", library_id]
            drop_cols = [col for col in drop_cols if col in group.columns]
            embeddings_group = group.drop(columns=drop_cols).values
            embeddings_df_group = group.drop(columns=drop_cols)
            embeddings_df_group.index = group["barcode"]
            barcode_group = group["barcode"]

            if method == "image_plot_slic":
                if use_cached_image:
                    print("[warning] use_cached_image=True ignored when library_id is set; regenerating per-library image.")
                labels, _, _ = image_plot_slic_segmentation(
                    embeddings_group,
                    spatial_coords_group,
                    n_segments=int(resolution),
                    compactness=compactness,
                    figsize=figsize,
                    fig_dpi=fig_dpi,
                    imshow_tile_size=imshow_tile_size,
                    imshow_scale_factor=imshow_scale_factor,
                    enforce_connectivity=enforce_connectivity,
                    pixel_shape=pixel_shape,
                    show_image=show_image,
                    verbose=verbose,
                    n_jobs=n_jobs,
                )
                # Aggregate tile labels -> per-barcode labels (use origin flags for tie-break)
                clusters_df_group = _aggregate_labels_majority(
                    barcode_group,
                    labels,
                    origin_flags=group.get("origin", pd.Series(index=group.index, data=0)).values,
                )
                # Use per-barcode embeddings to compute pseudo-centroids
                emb_base_group = (
                    base_embeddings_df.loc[clusters_df_group["barcode"].astype(str)]
                    .copy()
                )
                pseudo_centroids_group = create_pseudo_centroids(
                    emb_base_group,
                    clusters_df_group,
                    emb_base_group.columns,
                )
                combined_data_group = None
            else:
                clusters_df_group, pseudo_centroids_group, combined_data_group = (
                    segment_image_inner(
                        embeddings_group,
                        embeddings_df_group,
                        barcode_group,
                        spatial_coords_group,
                        dimensions,
                        method,
                        resolution,
                        compactness,
                        scaling,
                        n_neighbors,
                        random_state,
                        Segment,
                        index_selection,
                        max_iter,
                        verbose,
                        figsize,
                        imshow_tile_size,
                        imshow_scale_factor,
                        enforce_connectivity,
                        pixel_shape,
                        show_image,
                        fig_dpi=fig_dpi,
                    )
                )
                # If duplicates (origin=False), aggregate to one label per barcode and recompute centroids
                if clusters_df_group["barcode"].duplicated().any():
                    clusters_df_group = _aggregate_labels_majority(
                        clusters_df_group["barcode"],
                        clusters_df_group["Segment"].values,
                        origin_flags=group.get("origin", pd.Series(index=group.index, data=0)).values,
                    )
                    emb_base_group = (
                        base_embeddings_df.loc[clusters_df_group["barcode"].astype(str)]
                        .copy()
                    )
                    pseudo_centroids_group = create_pseudo_centroids(
                        emb_base_group,
                        clusters_df_group,
                        emb_base_group.columns,
                    )
                    # Aggregate combined_data to per-barcode mean if available
                    if combined_data_group is not None:
                        combined_data_group = _aggregate_matrix_by_barcode(
                            combined_data_group,
                            barcode_group,
                            clusters_df_group["barcode"],
                        )
            # Prefix segment labels with the library_id for uniqueness
            clusters_df_group["Segment"] = clusters_df_group["Segment"].apply(
                lambda x: f"{lib}_{x}"
            )
            clusters_list.append(clusters_df_group)
            # Add library_id column to pseudo_centroids
            if isinstance(pseudo_centroids_group, pd.DataFrame):
                pseudo_centroids_group[library_id] = lib
            else:
                pseudo_centroids_group = pd.DataFrame(pseudo_centroids_group)
                pseudo_centroids_group[library_id] = lib
            pseudo_centroids_list.append(pseudo_centroids_group)
            # Add library_id column to combined_data if available
            if combined_data_group is not None:
                if isinstance(combined_data_group, pd.DataFrame):
                    combined_data_group[library_id] = lib
                else:
                    combined_data_group = pd.DataFrame(combined_data_group)
                    combined_data_group[library_id] = lib
            combined_data_list.append(combined_data_group)

        clusters_df = pd.concat(clusters_list)
        pseudo_centroids = pd.concat(pseudo_centroids_list)
        pseudo_centroids = pseudo_centroids.iloc[:, :-1]
        combined_data = (
            pd.concat(combined_data_list)
            if any(x is not None for x in combined_data_list)
            else None
        )
        combined_data.index = pseudo_centroids.index
        combined_data = combined_data.iloc[:, :-1]
        clusters_df.index = clusters_df["barcode"]
        combined_data = combined_data.reindex(adata.obs.index)
        clusters_df = clusters_df.reindex(adata.obs.index)
        pseudo_centroids = pseudo_centroids.reindex(adata.obs.index)

    else:
        # No library_id specified, process all data together
        spatial_coords = coordinates_df.loc[:, ["x", "y"]].values
        embeddings = coordinates_df.drop(
            columns=["barcode", "x", "y", "origin"]
        ).values  # Convert to NumPy array
        embeddings_df = coordinates_df.drop(columns=["barcode", "x", "y", "origin"])
        embeddings_df.index = coordinates_df["barcode"]
        barcode = coordinates_df["barcode"]

        if method == "image_plot_slic":
            if use_cached_image and image_cache_key in adata.uns:
                cache = adata.uns[image_cache_key]
                cached_emb = cache.get("embedding_key", None)
                cached_dims = cache.get("dimensions", None)
                try:
                    if (cached_emb is not None and cached_emb != embedding) or (
                        cached_dims is not None and list(cached_dims) != list(dimensions)
                    ):
                        print("[warning] Cached image embedding/dimensions differ; regenerating image.")
                        raise KeyError

                    # When using cached image, always display with cached figsize/dpi
                    # (ignore provided figsize/fig_dpi for visualization consistency)
                    if show_image:
                        cache_figsize = cache.get("figsize", None)
                        cache_figdpi = cache.get("fig_dpi", None)
                        if figsize is not None or fig_dpi is not None:
                            print(
                                f"[info] use_cached_image=True: showing with cached figsize={cache_figsize}, fig_dpi={cache_figdpi} (ignoring provided figsize/fig_dpi)"
                            )
                        else:
                            print(
                                f"[info] use_cached_image=True: showing with cached figsize={cache_figsize}, fig_dpi={cache_figdpi}"
                            )

                    t0 = time.perf_counter()
                    labels_tile, _ = slic_segmentation_from_cached_image(
                        cache,
                        n_segments=int(resolution),
                        compactness=compactness,
                        enforce_connectivity=enforce_connectivity,
                        show_image=show_image,
                        verbose=verbose,
                        figsize=None,
                        fig_dpi=None,
                    )
                    t1 = time.perf_counter()
                    cache_barcodes = pd.Series(cache.get("barcodes", []), dtype=str)
                    if not cache_barcodes.duplicated().any():
                        clusters_df = pd.DataFrame({
                            "barcode": cache_barcodes.values,
                            "Segment": labels_tile,
                        })
                    else:
                        clusters_df = _aggregate_labels_majority(
                            cache_barcodes,
                            labels_tile,
                            origin_flags=cache.get("origin_flags", None),
                        )
                    t2 = time.perf_counter()
                    emb_base = base_embeddings_df.loc[clusters_df["barcode"].astype(str)].copy()
                    pseudo_centroids = create_pseudo_centroids(
                        emb_base,
                        clusters_df,
                        emb_base.columns,
                    )
                    t3 = time.perf_counter()
                    if verbose:
                        print(f"[perf] cached SLIC: {t1 - t0:.3f}s | aggregate: {t2 - t1:.3f}s | pseudo: {t3 - t2:.3f}s")
                    combined_data = None
                except Exception:
                    # Fallback to regeneration
                    t0 = time.perf_counter()
                    labels, _, _ = image_plot_slic_segmentation(
                        embeddings,
                        spatial_coords,
                        n_segments=int(resolution),
                        compactness=compactness,
                        figsize=figsize,
                        fig_dpi=fig_dpi,
                        imshow_tile_size=imshow_tile_size,
                        imshow_scale_factor=imshow_scale_factor,
                        enforce_connectivity=enforce_connectivity,
                        pixel_shape=pixel_shape,
                        show_image=show_image,
                        verbose=verbose,
                        n_jobs=n_jobs,
                    )
                    t1 = time.perf_counter()
                    clusters_df = _aggregate_labels_majority(
                        barcode,
                        labels,
                        origin_flags=coordinates_df.get("origin", pd.Series(index=coordinates_df.index, data=0)).values,
                    )
                    t2 = time.perf_counter()
                    emb_base = base_embeddings_df.loc[clusters_df["barcode"].astype(str)].copy()
                    pseudo_centroids = create_pseudo_centroids(
                        emb_base,
                        clusters_df,
                        emb_base.columns,
                    )
                    t3 = time.perf_counter()
                    if verbose:
                        print(f"[perf] regen SLIC: {t1 - t0:.3f}s | aggregate: {t2 - t1:.3f}s | pseudo: {t3 - t2:.3f}s")
                    combined_data = None
            else:
                t0 = time.perf_counter()
                labels, _, _ = image_plot_slic_segmentation(
                    embeddings,
                    spatial_coords,
                    n_segments=int(resolution),
                    compactness=compactness,
                    figsize=figsize,
                    fig_dpi=fig_dpi,
                    imshow_tile_size=imshow_tile_size,
                    imshow_scale_factor=imshow_scale_factor,
                    enforce_connectivity=enforce_connectivity,
                    pixel_shape=pixel_shape,
                    show_image=show_image,
                    verbose=verbose,
                    n_jobs=n_jobs,
                )
                t1 = time.perf_counter()
                if not barcode.duplicated().any():
                    clusters_df = pd.DataFrame({
                        "barcode": barcode.astype(str).values,
                        "Segment": labels,
                    })
                else:
                    clusters_df = _aggregate_labels_majority(
                        barcode,
                        labels,
                        origin_flags=coordinates_df.get("origin", pd.Series(index=coordinates_df.index, data=0)).values,
                    )
                t2 = time.perf_counter()
                emb_base = base_embeddings_df.loc[clusters_df["barcode"].astype(str)].copy()
                pseudo_centroids = create_pseudo_centroids(
                    emb_base,
                    clusters_df,
                    emb_base.columns,
                )
                t3 = time.perf_counter()
                if verbose:
                    print(f"[perf] direct SLIC: {t1 - t0:.3f}s | aggregate: {t2 - t1:.3f}s | pseudo: {t3 - t2:.3f}s")
                combined_data = None
        else:
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
                Segment,
                index_selection,
                max_iter,
                verbose,
                figsize,
                imshow_tile_size,
                imshow_scale_factor,
                enforce_connectivity,
                pixel_shape,
                show_image,
                fig_dpi=fig_dpi,
            )
            # If duplicates (origin=False), aggregate to one label per barcode and recompute centroids
            if clusters_df["barcode"].duplicated().any():
                clusters_df = _aggregate_labels_majority(
                    barcode,
                    clusters_df["Segment"].values,
                    origin_flags=coordinates_df.get("origin", pd.Series(index=coordinates_df.index, data=0)).values,
                )
                emb_base = base_embeddings_df.loc[clusters_df["barcode"].astype(str)].copy()
                pseudo_centroids = create_pseudo_centroids(
                    emb_base,
                    clusters_df,
                    emb_base.columns,
                )
                # Aggregate combined_data to per-barcode mean if available
                if combined_data is not None:
                    combined_data = _aggregate_matrix_by_barcode(
                        combined_data,
                        barcode,
                        clusters_df["barcode"],
                    )

    # Ensure alignment to adata.obs
    seg_map = clusters_df.set_index("barcode")["Segment"].astype(object)
    adata.obs[Segment] = pd.Categorical(
        pd.Series(adata.obs.index.astype(str)).map(seg_map).values
    )
    # Pseudo-centroids to numpy aligned to adata order
    if isinstance(pseudo_centroids, pd.DataFrame):
        pseudo_centroids = pseudo_centroids.reindex(adata.obs.index.astype(str)).values
    segment_suffix = _sanitize_segment_name(Segment)
    base_obsm_key = "X_embedding_segment"
    prefixed_obsm_key = f"X_embedding_{segment_suffix}"
    adata.obsm[base_obsm_key] = pseudo_centroids
    adata.obsm[prefixed_obsm_key] = pseudo_centroids
    # Combined data (if available) — align to adata order
    if combined_data is not None:
        if isinstance(combined_data, pd.DataFrame):
            cd_df = combined_data
        else:
            cd_df = pd.DataFrame(combined_data, index=clusters_df["barcode"].astype(str))
        cd_df = cd_df.reindex(adata.obs.index.astype(str))
        scaled_base_key = "X_embedding_scaled_for_segment"
        scaled_prefixed_key = f"X_embedding_scaled_for_{segment_suffix}"
        adata.obsm[scaled_base_key] = cd_df.values
        adata.obsm[scaled_prefixed_key] = cd_df.values
    else:
        for key in ("X_embedding_scaled_for_segment", f"X_embedding_scaled_for_{segment_suffix}"):
            if key in adata.obsm:
                del adata.obsm[key]


def segment_image_inner(
    embeddings,
    embeddings_df,
    barcode,
    spatial_coords,
    dimensions=list(range(30)),
    method="slic",
    resolution=10.0,
    compactness=1.0,
    scaling=0.3,
    n_neighbors=15,
    random_state=42,
    Segment="Segment",
    index_selection="random",
    max_iter=1000,
    verbose=True,
    figsize=(10, 10),
    imshow_tile_size: float | None = None,
    imshow_scale_factor: float = 1.0,
    enforce_connectivity: bool = False,
    pixel_shape: str = "square",
    show_image: bool = False,
    fig_dpi: int | None = None,
    n_jobs: int | None = None,
):
    """
    Segment embeddings to find initial territories.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Array of embedding values.
    embeddings_df : pd.DataFrame
        DataFrame of embedding values with barcode as index.
    barcode : pd.Series or list
        Barcode identifiers.
    spatial_coords : numpy.ndarray
        Spatial coordinates (x, y) for each point.
    dimensions : list of int, optional
        List of dimensions to use for segmentation.
    method : str, optional
        Segmentation method: 'kmeans', 'louvain', 'leiden', 'slic', 'som',
        'leiden_slic', 'louvain_slic', 'image_plot_slic'.
    resolution : float or int, optional
        Resolution parameter for clustering methods.
    compactness : float
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float
        Scaling factor for spatial coordinates.
    n_neighbors : int
        Number of neighbors for constructing the nearest neighbor graph.
    random_state : int
        Random seed for reproducibility.
    Segment : str
        Name of the segment column.
    index_selection : str, optional
        Method for selecting initial cluster centers.
    max_iter : int, optional
        Maximum number of iterations for convergence.
    verbose : bool, optional
        Whether to display progress messages.
    figsize : tuple, optional
        Figure size used when creating the intermediate image for ``image_plot_slic``.
    imshow_tile_size : float or None, optional
        Tile size used for estimating pixel scaling in ``image_plot_slic``.
    imshow_scale_factor : float, optional
        Additional scaling factor for ``image_plot_slic``.
    enforce_connectivity : bool, optional
        Whether to enforce connectivity in ``image_plot_slic``.
    pixel_shape : str, optional
        Shape used when rendering the temporary image in ``image_plot_slic``.
        ``"circle"`` mimics Visium style spots while ``"square"`` draws
        traditional tiles.
    show_image : bool, optional
        If ``True`` display the constructed image and label image during
        ``image_plot_slic`` segmentation.
    n_jobs : int or None, optional
        Number of worker threads for ``image_plot_slic`` rasterization. When
        ``None``, defers to ``SPIX_IMAGE_PLOT_SLIC_N_JOBS`` env var or 1.
    target_segment_um : float or None, optional
        Desired segment size in micrometers. Overrides ``resolution``.
    pitch_um : float, optional
        Spot spacing in micrometers when computing ``target_segment_um``.

    Returns
    -------
    clusters_df : pd.DataFrame
        DataFrame with barcode and corresponding segment labels.
    pseudo_centroids : pd.DataFrame or numpy.ndarray
        Pseudo-centroids calculated from the embeddings.
    combined_data : pd.DataFrame or numpy.ndarray or None
        Combined data used for segmentation (if applicable).
    """
    if verbose:
        print(f"Starting image segmentation using method '{method}'...")

    # Initialize combined_data as None in case the segmentation method does not return it
    combined_data = None

    if method == "kmeans":
        clusters = kmeans_segmentation(embeddings, resolution, random_state)
    elif method == "louvain":
        clusters = louvain_segmentation(
            embeddings, resolution, n_neighbors, random_state
        )
    elif method == "leiden":
        clusters = leiden_segmentation(
            embeddings, resolution, n_neighbors, random_state
        )
    elif method == "slic":
        clusters, combined_data = slic_segmentation(
            embeddings,
            spatial_coords,
            n_segments=int(resolution),
            compactness=compactness,
            scaling=scaling,
            index_selection=index_selection,
            max_iter=max_iter,
            verbose=verbose,
        )
    elif method == "image_plot_slic":
        clusters, _, _ = image_plot_slic_segmentation(
            embeddings,
            spatial_coords,
            n_segments=int(resolution),
            compactness=compactness,
            figsize=figsize,
            fig_dpi=fig_dpi,
            imshow_tile_size=imshow_tile_size,
            imshow_scale_factor=imshow_scale_factor,
            enforce_connectivity=enforce_connectivity,
            pixel_shape=pixel_shape,
            show_image=show_image,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        combined_data = None
    elif method == "som":
        clusters = som_segmentation(embeddings, resolution)
    elif method == "leiden_slic":
        clusters, combined_data = leiden_slic_segmentation(
            embeddings,
            spatial_coords,
            resolution,
            compactness,
            scaling,
            n_neighbors,
            random_state,
        )
    elif method == "louvain_slic":
        clusters, combined_data = louvain_slic_segmentation(
            embeddings,
            spatial_coords,
            resolution,
            compactness,
            scaling,
            n_neighbors,
            random_state,
        )
    else:
        raise ValueError(f"Unknown segmentation method '{method}'")

    # Create clusters DataFrame with barcode and segment labels
    clusters_df = pd.DataFrame({"barcode": pd.Series(barcode, dtype=str), "Segment": clusters})

    # If multiple rows per barcode, aggregate to a single label per barcode
    if clusters_df["barcode"].duplicated().any():
        clusters_df = _aggregate_labels_majority(clusters_df["barcode"], clusters_df["Segment"].values)
        # Use per-barcode base embeddings (mean across duplicates)
        emb_base = embeddings_df.groupby(embeddings_df.index.astype(str)).mean()
        emb_base = emb_base.loc[clusters_df["barcode"].astype(str)]
        pseudo_centroids = create_pseudo_centroids(emb_base, clusters_df, emb_base.columns)
    else:
        # Unique barcodes already: compute directly
        pseudo_centroids = create_pseudo_centroids(embeddings_df, clusters_df, dimensions)

    if verbose:
        print("Image segmentation completed.")

    return clusters_df, pseudo_centroids, combined_data
