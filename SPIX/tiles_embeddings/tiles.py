import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from shapely.geometry import Polygon, MultiPoint
import logging
import time
from scipy.spatial import Voronoi, cKDTree
from typing import Optional

from ..utils.utils import (
    filter_grid_function,
    reduce_tensor_resolution,
    filter_tiles,
    rasterise,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_tiles(
    adata: AnnData,
    tensor_resolution: float = 1,
    filter_grid: float = 1,
    filter_threshold: float = 0.995,
    verbose: bool = True,
    chunksize: int = 1000,
    force: bool = False,
    n_jobs: int = None,
    use_coords_as_tiles: bool = False,
    coords_max_gap_factor: Optional[float] = 10.0,
    coords_compaction: str = 'cap',
    coords_tight_step: float = 1.0,
    coords_rescale_to_nn: Optional[bool] = None,
    coords_filter: bool = False,
    coords_filter_k: int = 5,
    coords_filter_mode: str = 'knn',
    # Rasterisation sampling controls (Voronoi mode)
    raster_stride: int = 1,
    raster_sample_frac: Optional[float] = None,
    raster_max_pixels_per_tile: Optional[int] = None,
    raster_random_seed: Optional[int] = None,
    # Restoration toggles
    restore_zero_pixel_tiles: bool = True,
    restore_voronoi_dropped: bool = True,
) -> AnnData:
    """
    Generate spatial tiles for the given AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    tensor_resolution : float, optional (default=1)
        Resolution parameter for tensor reduction.
    filter_grid : float, optional (default=0.01)
        Grid filtering threshold to remove outlier beads.
    filter_threshold : float, optional (default=0.995)
        Threshold to filter tiles based on area.
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    chunksize : int, optional (default=1000)
        Number of tiles per chunk for parallel processing.
    n_jobs : int, optional (default=None)
        Number of parallel jobs to run.
    use_coords_as_tiles : bool, optional (default=False)
        If True, skip Voronoi/rasterise completely and build ``tiles`` directly
        from coordinates as one row per barcode with ``origin=1``. If
        ``tensor_resolution`` != 1, the coordinates are reduced using
        ``reduce_tensor_resolution`` before writing. Additionally, coordinates
        are compacted to avoid excessively large gaps between tiles (see
        ``coords_max_gap_factor``).
    coords_max_gap_factor : float or None, optional (default=10.0)
        When ``use_coords_as_tiles`` is True, limit one-dimensional gaps along
        X and Y to at most this factor times the median gap on that axis,
        preserving relative order while shrinking large empty spaces. Set to
        ``None`` or ``<= 0`` to disable compaction.
    coords_compaction : {'cap','tight'}, optional (default='cap')
        Compaction strategy when ``use_coords_as_tiles`` is True.
        - 'cap': cap large axis gaps to ``coords_max_gap_factor`` times median gap.
        - 'tight': map each unique X and Y to consecutive positions (rank-based),
          eliminating all empty columns/rows and packing tiles tightly while
          preserving axis ordering.
    coords_tight_step : float, optional (default=1.0)
        Spacing step between consecutive ranks when ``coords_compaction='tight'``.
    coords_rescale_to_nn : bool or None, optional (default=None)
        After compaction, rescale XY so the median 1-nearest-neighbor distance
        matches the original. If None, defaults to True for 'cap' and False for
        'tight'.
    coords_filter : bool, optional (default=False)
        When ``use_coords_as_tiles`` is True, optionally filter isolated points
        using a kNN-distance density heuristic (analog of Voronoi area filter).
        Points with mean distance to k nearest neighbors larger than the
        ``filter_threshold`` quantile are removed.
    coords_filter_k : int, optional (default=5)
        Number of neighbors for the kNN density filter in coords-as-tiles mode.
    coords_filter_mode : {'knn','voronoi'}, optional (default='knn')
        Filtering strategy in coords-as-tiles mode. 'knn' uses mean kNN distance
        with a quantile cutoff; 'voronoi' reproduces the original Voronoi
        area-based filtering to match previous behavior as closely as possible.
    raster_stride : int, optional (default=1)
        In Voronoi+rasterise mode, sample the interior grid every ``stride``
        pixels along X and Y to reduce the number of generated tile pixels
        while preserving spatial structure. ``1`` keeps all pixels.
    raster_sample_frac : float or None, optional (default=None)
        Additional random subsampling fraction per tile in (0,1]. Applied after
        ``raster_stride``. ``None`` disables.
    raster_max_pixels_per_tile : int or None, optional (default=None)
        Cap the maximum number of pixels kept per tile. Applied after stride and
        fractional subsampling. ``None`` disables.
    raster_random_seed : int or None, optional (default=None)
        Random seed for reproducible subsampling.
    restore_zero_pixel_tiles : bool, optional (default=True)
        If True, spots that pass Voronoi filtering but yield zero pixels after
        rasterisation are restored as origin-only tiles.
    restore_voronoi_dropped : bool, optional (default=True)
        If True, spots dropped by Voronoi area filtering (invalid/infinite regions)
        are restored as origin-only tiles. When False, they remain filtered.
    
    Returns
    -------
    AnnData
        The updated AnnData object with generated tiles.
        A removal report is stored in ``adata.uns['tiles_removal_report']`` with
        counts and barcode lists per step (grid, tensor_resolution, coords_filter
        or voronoi_area, rasterise) and the mode used.
    """
    # Check if tiles are already generated
    tiles_exist = adata.uns.get('tiles_generated', False)
    if tiles_exist and not force:
        if verbose:
            logging.info("Tiles have already been generated. Skipping tile generation.")
        return adata
    elif tiles_exist and force:
        if verbose:
            logging.info("Force regeneration enabled: regenerating tiles...")

    if verbose:
        logging.info("Starting generate_tiles...")

    # Copy original spatial coordinates
    coordinates_orig = pd.DataFrame(adata.obsm['spatial']).copy()
    coordinates_orig.index = adata.obs.index.copy()
    coordinates = coordinates_orig.copy()
    removal_report = {
        'mode': 'coords_as_tiles' if use_coords_as_tiles else 'voronoi',
        'filter_mode': None,
        'counts': {},
        'removed': {
            'grid': [],
            'tensor_resolution': [],
            'coords_filter': [],
            'voronoi_area': [],
            'rasterise': [],
        },
    }
    if verbose:
        logging.info(f"Original coordinates: {coordinates.shape}")

    # Apply grid filtering if specified
    if 0 < filter_grid < 1:
        if verbose:
            logging.info("Filtering outlier beads...")
        before_idx = coordinates.index.copy()
        coordinates = filter_grid_function(coordinates, filter_grid)
        removed_grid = list(set(before_idx) - set(coordinates.index))
        removal_report['removed']['grid'] = sorted(map(str, removed_grid))
        removal_report['counts']['grid'] = len(removed_grid)
        if verbose:
            logging.info(f"Coordinates after filtering: {coordinates.shape}")

    # Reduce tensor resolution if specified
    if tensor_resolution != 1 and tensor_resolution > 0:
        if verbose:
            logging.info("Reducing tensor resolution...")
        before_idx = coordinates.index.copy()
        coordinates = reduce_tensor_resolution(coordinates, tensor_resolution)
        removed_tensor = list(set(before_idx) - set(coordinates.index))
        removal_report['removed']['tensor_resolution'] = sorted(map(str, removed_tensor))
        removal_report['counts']['tensor_resolution'] = len(removed_tensor)
        if verbose:
            logging.info(f"Coordinates after resolution reduction: {coordinates.shape}")

    # Fast path: directly use coordinates as tiles and skip Voronoi/rasterise
    if use_coords_as_tiles:
        if verbose:
            logging.info("Skipping Voronoi/rasterise; using coordinates as tiles with optional compaction.")

        # Use current coordinates (after optional filtering/reduction)
        # Avoid an extra full DataFrame copy (coordinates is already a copy).
        coords_used = coordinates
        if 0 in coords_used.columns and 1 in coords_used.columns:
            coords_used = coords_used.rename(columns={0: 'x', 1: 'y'})
        else:
            cols = list(coords_used.columns[:2])
            coords_used = coords_used.rename(columns={cols[0]: 'x', cols[1]: 'y'})

        # Optional: filter isolated points in coords-as-tiles mode
        if coords_filter and len(coords_used) > 1:
            mode = (coords_filter_mode or 'knn').lower()
            removal_report['filter_mode'] = mode
            coords_used_before_filter = coords_used.copy()
            if mode == 'voronoi':
                try:
                    # Reproduce original Voronoi area-based filtering to match behavior
                    vor_tmp = Voronoi(coords_used[['x', 'y']].values)
                    fr, fp, findex = filter_tiles(
                        vor_tmp,
                        coords_used,
                        filter_threshold,
                        n_jobs=n_jobs,
                        chunksize=chunksize,
                        verbose=verbose,
                    )
                    if len(findex) > 0:
                        coords_used = coords_used.loc[findex]
                        removed_cf = list(set(coords_used_before_filter.index) - set(coords_used.index))
                        removal_report['removed']['coords_filter'] = sorted(map(str, removed_cf))
                        removal_report['counts']['coords_filter'] = len(removed_cf)
                        if verbose:
                            logging.info(
                                f"Coords-as-tiles Voronoi filter kept {len(findex)}/{len(vor_tmp.point_region)} (q={filter_threshold})."
                            )
                except Exception as e:
                    logging.warning(f"Coords-as-tiles Voronoi filtering failed; falling back to kNN: {e}")
                    mode = 'knn'
            if mode == 'knn':
                try:
                    coords_vals = coords_used[['x', 'y']].values.astype(float)
                    k = int(max(1, min(coords_filter_k, len(coords_vals) - 1)))
                    tree = cKDTree(coords_vals)
                    dists, _ = tree.query(coords_vals, k=k + 1)
                    # exclude self at column 0
                    if dists.ndim == 1:
                        mean_knn = dists
                    else:
                        mean_knn = dists[:, 1:].mean(axis=1)
                    cutoff = np.quantile(mean_knn, filter_threshold)
                    mask = mean_knn <= cutoff
                    kept = mask.sum()
                    if kept > 0:
                        coords_used = coords_used.loc[mask]
                        removed_cf = list(coords_used_before_filter.index[~mask])
                        removal_report['removed']['coords_filter'] = sorted(map(str, removed_cf))
                        removal_report['counts']['coords_filter'] = len(removed_cf)
                        if verbose:
                            logging.info(
                                f"Coords-as-tiles kNN filter kept {kept}/{len(mask)} (k={k}, q={filter_threshold})."
                            )
                    else:
                        if verbose:
                            logging.warning("Coords-as-tiles kNN filter would drop all points; skipping filter.")
                except Exception as e:
                    logging.warning(f"Coords-as-tiles kNN filtering failed; skipping: {e}")

        # Optionally compact coordinates to avoid overly large gaps between tiles
        def _limit_axis_gaps(vals: np.ndarray, cap_factor: float) -> np.ndarray:
            """
            Monotonic gap-limiting transform for a 1D coordinate array.
            Preserves order. Large gaps are capped to (cap_factor * median_gap).
            """
            if len(vals) <= 2 or not np.isfinite(vals).all():
                return vals.copy()
            # Sort unique values to compute gaps
            uniq = np.unique(vals)
            if uniq.size < 2:
                return vals.copy()
            diffs = np.diff(uniq)
            # Guard against zero/near-zero median
            med = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else None
            if med is None or med == 0:
                # Fallback: use rank transform spacing of 1
                ranks = pd.Series(vals).rank(method='average').to_numpy()
                ranks -= ranks.min()
                return ranks
            cap = cap_factor * med
            # Cap large gaps
            capped_diffs = np.minimum(diffs, cap)
            # Reconstruct a monotonic mapping from old -> new positions
            new_positions = np.concatenate([[0.0], np.cumsum(capped_diffs)])
            # Vectorized map back to original values (much faster than Python dict loop).
            idx = np.searchsorted(uniq, vals)
            return new_positions[idx]

        def _rank_compact(vals: np.ndarray, step: float) -> np.ndarray:
            """
            Rank-based compaction: map unique sorted values to 0, step, 2*step, ...
            Eliminates empty columns/rows while preserving order.
            """
            if len(vals) == 0 or not np.isfinite(vals).all():
                return vals.copy()
            uniq = np.unique(vals)
            # Ensure deterministic order
            uniq_sorted = np.sort(uniq)
            idx = np.searchsorted(uniq_sorted, vals)
            return idx.astype(np.float64, copy=False) * step

        # Avoid deep copy for large datasets; we'll overwrite columns as needed.
        tiles = coords_used.loc[:, ['x', 'y']].copy(deep=False)

        # Decide rescaling policy up front so we can skip expensive kNN work.
        rescale_flag = coords_rescale_to_nn
        if rescale_flag is None:
            rescale_flag = (coords_compaction != 'tight')

        # Compute original median 1-NN distance (for optional rescaling)
        med_nn_orig = None
        if rescale_flag:
            try:
                t_nn0 = time.perf_counter()
                # Keep float64 for strict backward-compatible distances/scales.
                xy_orig = tiles[['x', 'y']].to_numpy(dtype=np.float64, copy=False)
                kdt_orig = cKDTree(xy_orig)
                workers = -1 if (n_jobs in (None, -1)) else int(n_jobs)
                try:
                    dists_orig, _ = kdt_orig.query(xy_orig, k=2, workers=workers)
                except TypeError:
                    dists_orig, _ = kdt_orig.query(xy_orig, k=2)
                med_nn_orig = np.median(dists_orig[:, 1]) if dists_orig.shape[1] > 1 else None
                if verbose:
                    logging.info(f"Computed original median 1-NN distance in {time.perf_counter() - t_nn0:.2f}s")
            except Exception:
                med_nn_orig = None

        if coords_compaction == 'tight':
            try:
                tiles['x'] = _rank_compact(tiles['x'].to_numpy(dtype=float), float(coords_tight_step))
                tiles['y'] = _rank_compact(tiles['y'].to_numpy(dtype=float), float(coords_tight_step))
                if verbose:
                    logging.info("Coords-as-tiles tight packing applied (rank-based).")
            except Exception as e:
                logging.warning(f"Tight packing failed; skipping: {e}")
        elif coords_max_gap_factor is not None and coords_max_gap_factor > 0:
            try:
                t_compact = time.perf_counter()
                x_new = _limit_axis_gaps(tiles['x'].to_numpy(dtype=float), float(coords_max_gap_factor))
                y_new = _limit_axis_gaps(tiles['y'].to_numpy(dtype=float), float(coords_max_gap_factor))
                tiles['x'] = x_new
                tiles['y'] = y_new
                if verbose:
                    logging.info(
                        f"Coords-as-tiles compaction applied (gap cap factor={coords_max_gap_factor})."
                    )
                    logging.info(f"Coords-as-tiles compaction took {time.perf_counter() - t_compact:.2f}s")
            except Exception as e:
                logging.warning(f"Coordinate compaction skipped due to error: {e}")

        # Optionally rescale to match original median 1-NN distance
        if rescale_flag and med_nn_orig is not None and med_nn_orig > 0:
            try:
                t_nn1 = time.perf_counter()
                # Keep float64 for strict backward-compatible distances/scales.
                xy_new = tiles[['x', 'y']].to_numpy(dtype=np.float64, copy=False)
                kdt_new = cKDTree(xy_new)
                workers = -1 if (n_jobs in (None, -1)) else int(n_jobs)
                try:
                    dists_new, _ = kdt_new.query(xy_new, k=2, workers=workers)
                except TypeError:
                    dists_new, _ = kdt_new.query(xy_new, k=2)
                med_nn_new = np.median(dists_new[:, 1]) if dists_new.shape[1] > 1 else None
                if med_nn_new is not None and med_nn_new > 0:
                    scale = med_nn_orig / med_nn_new
                    # Faster than 2-col DataFrame multiplication for large n.
                    tiles['x'] = tiles['x'].to_numpy(dtype=np.float64, copy=False) * float(scale)
                    tiles['y'] = tiles['y'].to_numpy(dtype=np.float64, copy=False) * float(scale)
                    if verbose:
                        logging.info("Post-compaction rescale applied to match median 1-NN distance.")
                        logging.info(f"Computed new median 1-NN + rescaled in {time.perf_counter() - t_nn1:.2f}s")
            except Exception as e:
                logging.warning(f"Rescale after compaction skipped: {e}")

        tiles['barcode'] = tiles.index.astype(str)
        tiles['origin'] = 1

        adata.uns["tiles"] = tiles.reset_index(drop=True)
        adata.uns["tiles_generated"] = True
        # Save removal report
        adata.uns["tiles_removal_report"] = removal_report
        if verbose:
            logging.info("Tiles have been stored in adata.uns['tiles'] (coords mode).")
        # Subset adata based on tiles (robust to dtype mismatches)
        filtered_barcodes = tiles["barcode"].unique()
        initial_obs = adata.n_obs
        try:
            # Compare as strings to avoid label/positional ambiguity when obs_names are ints
            obs_names_str = adata.obs_names.astype(str)
            keep_mask = obs_names_str.isin(pd.Index(filtered_barcodes).astype(str))
            adata = adata[keep_mask, :].copy()
        except Exception:
            # Fallback to label-based selection; may raise if types differ
            adata = adata[filtered_barcodes, :].copy()
        final_obs = adata.n_obs
        if verbose:
            logging.info(
                f"adata has been subset from {initial_obs} to {final_obs} observations based on tiles (coords mode)."
            )
        if verbose:
            logging.info("generate_tiles completed (coords mode).")
        return adata

    # Perform Voronoi tessellation
    if verbose:
        logging.info("Performing Voronoi tessellation...")
    vor = Voronoi(coordinates)
    if verbose:
        logging.info("Voronoi tessellation completed.")

    # Filter tiles based on area
    if verbose:
        logging.info("Filtering tiles...")
    filtered_regions, filtered_coordinates, index = filter_tiles(
        vor,
        coordinates,
        filter_threshold,
        n_jobs=n_jobs,
        chunksize=chunksize,
        verbose=verbose,
    )
    # Removed by Voronoi area filtering
    removed_voronoi = list(set(coordinates.index) - set(index))
    removal_report['removed']['voronoi_area'] = sorted(map(str, removed_voronoi))
    removal_report['counts']['voronoi_area'] = len(removed_voronoi)
    removal_report['filter_mode'] = 'voronoi'
    if verbose:
        logging.info(
            f"Filtered regions: {len(filtered_regions)}, Filtered coordinates: {filtered_coordinates.shape}"
        )

    # Compute boundary polygon for clipping based on filtered points
    boundary_polygon = MultiPoint(filtered_coordinates).convex_hull

    # Rasterize tiles with parallel processing
    if verbose:
        logging.info("Rasterising tiles...")
    tiles = rasterise(
        filtered_regions,
        filtered_coordinates,
        index,
        vor,
        boundary_polygon,
        chunksize,
        n_jobs=n_jobs,
        raster_stride=raster_stride,
        raster_sample_frac=raster_sample_frac,
        raster_max_pixels_per_tile=raster_max_pixels_per_tile,
        raster_random_seed=raster_random_seed,
        restore_zero_pixel_tiles=restore_zero_pixel_tiles,
    )
    if verbose:
        logging.info(f"Rasterisation completed. Number of tiles: {len(tiles)}")

    # Fallbacks: optionally restore missing spots as origin-only tiles
    try:
        # Build comparable string indices
        coords_idx_str = pd.Index(index).astype(str)
        rasterised_barcodes = set(tiles["barcode"].astype(str).unique()) if len(tiles) else set()
        if restore_zero_pixel_tiles:
            missing_after_raster = list(set(coords_idx_str) - rasterised_barcodes)
            if missing_after_raster:
                # Map from barcode -> original coordinate from the pre-Voronoi coordinate table
                coords_used = coordinates.copy()
                if 0 in coords_used.columns and 1 in coords_used.columns:
                    coords_used = coords_used.rename(columns={0: 'x', 1: 'y'})
                else:
                    cols = list(coords_used.columns[:2])
                    coords_used = coords_used.rename(columns={cols[0]: 'x', cols[1]: 'y'})
                # Filter to those missing barcodes (indices may be non-string -> cast to str for matching)
                coords_used_index_str = coords_used.index.astype(str)
                keep_mask = coords_used_index_str.isin(pd.Index(missing_after_raster))
                to_add = coords_used.loc[keep_mask, ["x", "y"]].copy()
                if len(to_add):
                    to_add["x"] = to_add["x"].round().astype(int)
                    to_add["y"] = to_add["y"].round().astype(int)
                    to_add["barcode"] = to_add.index.astype(str)
                    to_add["origin"] = 1
                    tiles = pd.concat([tiles, to_add.reset_index(drop=True)], ignore_index=True)
                    if verbose:
                        logging.info(f"Added {len(to_add)} origin-only tiles to preserve missing spots.")

        if restore_voronoi_dropped:
            removed_voronoi_all = list(set(coordinates.index.astype(str)) - set(pd.Index(index).astype(str)))
            if removed_voronoi_all:
                coords_used2 = coordinates.copy()
                if 0 in coords_used2.columns and 1 in coords_used2.columns:
                    coords_used2 = coords_used2.rename(columns={0: 'x', 1: 'y'})
                else:
                    cols2 = list(coords_used2.columns[:2])
                    coords_used2 = coords_used2.rename(columns={cols2[0]: 'x', cols2[1]: 'y'})
                idx_str2 = coords_used2.index.astype(str)
                keep_mask2 = idx_str2.isin(pd.Index(removed_voronoi_all))
                to_add2 = coords_used2.loc[keep_mask2, ["x", "y"]].copy()
                if len(to_add2):
                    to_add2["x"] = to_add2["x"].round().astype(int)
                    to_add2["y"] = to_add2["y"].round().astype(int)
                    to_add2["barcode"] = to_add2.index.astype(str)
                    to_add2["origin"] = 1
                    tiles = pd.concat([tiles, to_add2.reset_index(drop=True)], ignore_index=True)
                    if verbose:
                        logging.info(f"Added {len(to_add2)} origin-only tiles for Voronoi-dropped spots.")
    except Exception as e:
        logging.warning(f"Failed to add origin-only tiles fallback: {e}")

    # Store tiles and metadata in AnnData object
    adata.uns["tiles"] = tiles
    adata.uns["tiles_generated"] = True
    # Some filtered indices may have produced no pixels after clipping → report
    rasterised_barcodes = set(tiles["barcode"].astype(str).unique()) if len(tiles) else set()
    removed_after_raster = list(set(map(str, index)) - set(map(str, rasterised_barcodes)))
    removal_report['removed']['rasterise'] = sorted(removed_after_raster)
    removal_report['counts']['rasterise'] = len(removed_after_raster)
    # Save removal report
    adata.uns["tiles_removal_report"] = removal_report
    if verbose:
        logging.info("Tiles have been stored in adata.uns['tiles'].")

    # Subset adata based on filtered tiles (robust to dtype mismatches)
    filtered_barcodes = tiles["barcode"].unique()
    initial_obs = adata.n_obs
    try:
        # Compare as strings to avoid label/positional ambiguity when obs_names are ints
        obs_names_str = adata.obs_names.astype(str)
        keep_mask = obs_names_str.isin(pd.Index(filtered_barcodes).astype(str))
        adata = adata[keep_mask, :].copy()
    except Exception:
        # Fallback to label-based selection; may raise if types differ
        adata = adata[filtered_barcodes, :].copy()
    final_obs = adata.n_obs

    if verbose:
        logging.info(
            f"adata has been subset from {initial_obs} to {final_obs} observations based on filtered tiles."
        )

    if verbose:
        logging.info("generate_tiles completed.")

    return adata
