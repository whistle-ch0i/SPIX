import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from shapely.geometry import Polygon, MultiPoint
import logging
import time
from scipy.spatial import Voronoi, cKDTree
from scipy import ndimage as ndi
from typing import Optional

from ..utils.utils import (
    filter_grid_function,
    reduce_tensor_resolution,
    filter_tiles,
    rasterise,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _infer_regular_grid_pitch(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    n_jobs: int = None,
    sample_size: int = 200_000,
) -> float:
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    pts = np.column_stack((x_vals[finite], y_vals[finite]))
    if pts.shape[0] < 2:
        return 1.0
    if pts.shape[0] > int(sample_size):
        rng = np.random.default_rng(42)
        sel = rng.choice(pts.shape[0], size=int(sample_size), replace=False)
        pts = pts[sel]
    tree = cKDTree(pts)
    workers = -1 if (n_jobs in (None, -1)) else int(n_jobs)
    try:
        dists, _ = tree.query(pts, k=2, workers=workers)
    except TypeError:
        dists, _ = tree.query(pts, k=2)
    if np.ndim(dists) == 1:
        nn = np.asarray(dists, dtype=float)
    else:
        nn = np.asarray(dists[:, 1], dtype=float)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        return 1.0
    q = float(np.quantile(nn, 0.20))
    small = nn[nn <= q]
    pitch = float(np.median(small)) if small.size else float(np.median(nn))
    if not np.isfinite(pitch) or pitch <= 0:
        pitch = float(np.median(nn))
    if not np.isfinite(pitch) or pitch <= 0:
        return 1.0
    return pitch


def _infer_axis_origin(vals: np.ndarray, pitch: float) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(pitch) or pitch <= 0:
        return 0.0
    phase = np.mod(vals, pitch)
    if phase.size == 0:
        return float(np.min(vals))
    candidates = [float(np.median(phase))]
    candidates.append(candidates[0] - float(pitch))

    def _score(origin: float) -> float:
        idx = np.rint((vals - origin) / pitch)
        resid = np.abs(vals - (origin + idx * pitch))
        return float(np.median(resid))

    best = min(candidates, key=_score)
    return float(best)


def _binary_disk(radius: int) -> np.ndarray:
    radius = int(max(0, radius))
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return ((xx * xx + yy * yy) <= (radius * radius)).astype(bool)


def _prepare_inferred_grid_state(
    coordinates: pd.DataFrame,
    verbose: bool = True,
    n_jobs: int = None,
    inferred_grid_pitch: Optional[float] = None,
    inferred_grid_neighbor_radius: int = 1,
    inferred_grid_min_neighbors: int = 3,
    inferred_grid_closing_radius: int = 0,
    inferred_grid_fill_holes: bool = False,
) -> dict:
    coords_used = coordinates.copy(deep=False)
    if 0 in coords_used.columns and 1 in coords_used.columns:
        coords_used = coords_used.rename(columns={0: 'x', 1: 'y'})
    else:
        cols = list(coords_used.columns[:2])
        coords_used = coords_used.rename(columns={cols[0]: 'x', cols[1]: 'y'})
    coords_used = coords_used.loc[:, ['x', 'y']].copy(deep=False)

    x_vals = coords_used['x'].to_numpy(dtype=float, copy=False)
    y_vals = coords_used['y'].to_numpy(dtype=float, copy=False)
    barcode_vals = coords_used.index.astype(str).to_numpy(copy=False)

    if inferred_grid_pitch is None:
        pitch0 = _infer_regular_grid_pitch(x_vals, y_vals, n_jobs=n_jobs)
        pitch_x = float(pitch0)
        pitch_y = float(pitch0)
    else:
        pitch_x = float(inferred_grid_pitch)
        pitch_y = float(inferred_grid_pitch)

    origin_x = _infer_axis_origin(x_vals, pitch_x)
    origin_y = _infer_axis_origin(y_vals, pitch_y)

    gx_all = np.rint((x_vals - origin_x) / pitch_x).astype(np.int64, copy=False)
    gy_all = np.rint((y_vals - origin_y) / pitch_y).astype(np.int64, copy=False)
    gx_min = int(gx_all.min())
    gy_min = int(gy_all.min())
    gx_all = gx_all - gx_min
    gy_all = gy_all - gy_min
    origin_x = float(origin_x + gx_min * pitch_x)
    origin_y = float(origin_y + gy_min * pitch_y)

    grid_keys = pd.Index(gx_all.astype(str) + '_' + gy_all.astype(str))
    keep_mask = ~grid_keys.duplicated(keep='first')
    dropped_duplicates = int((~keep_mask).sum())
    if dropped_duplicates and verbose:
        logging.info(
            f"Inferred-grid mode collapsed {dropped_duplicates} duplicated barcodes sharing the same inferred grid cell for support-mask construction."
        )

    gx_occ = gx_all[keep_mask]
    gy_occ = gy_all[keep_mask]

    grid_w = int(gx_occ.max()) + 1
    grid_h = int(gy_occ.max()) + 1
    total_cells = int(grid_h) * int(grid_w)
    max_dense_cells = 200_000_000
    if total_cells > max_dense_cells:
        raise MemoryError(
            f"Inferred-grid mask would require {grid_h}x{grid_w}={total_cells:,} cells. "
            f"Specify inferred_grid_pitch explicitly or increase tensor_resolution; "
            f"current inferred pitch was ({pitch_x:.6g}, {pitch_y:.6g})."
        )

    observed_mask = np.zeros((grid_h, grid_w), dtype=bool)
    observed_mask[gy_occ, gx_occ] = True
    observed_cells = int(observed_mask.sum())
    observed_barcodes = int(barcode_vals.shape[0])

    support_radius = int(max(0, int(inferred_grid_neighbor_radius)))
    min_neighbors = int(max(0, int(inferred_grid_min_neighbors)))
    filled_mask = observed_mask.copy()
    if support_radius > 0:
        support_kernel = _binary_disk(support_radius).astype(np.int16)
        support = ndi.convolve(observed_mask.astype(np.int16), support_kernel, mode='constant', cval=0)
        filled_mask = support >= max(1, min_neighbors)
        filled_mask |= observed_mask

    if inferred_grid_closing_radius and int(inferred_grid_closing_radius) > 0:
        radius = int(inferred_grid_closing_radius)
        filled_mask = ndi.binary_closing(filled_mask, structure=_binary_disk(radius))
    if inferred_grid_fill_holes:
        filled_mask = ndi.binary_fill_holes(filled_mask)
    filled_mask |= observed_mask

    obs_x_centers_all = origin_x + gx_all.astype(np.float64, copy=False) * float(pitch_x)
    obs_y_centers_all = origin_y + gy_all.astype(np.float64, copy=False) * float(pitch_y)
    disp = np.sqrt((x_vals - obs_x_centers_all) ** 2 + (y_vals - obs_y_centers_all) ** 2)

    return {
        'x_all': x_vals,
        'y_all': y_vals,
        'barcode_all': barcode_vals,
        'gx_all': gx_all,
        'gy_all': gy_all,
        'origin_x': float(origin_x),
        'origin_y': float(origin_y),
        'pitch_x': float(pitch_x),
        'pitch_y': float(pitch_y),
        'grid_shape': (int(grid_h), int(grid_w)),
        'observed_mask': observed_mask,
        'filled_mask': filled_mask,
        'observed_cells': int(observed_cells),
        'observed_barcodes': int(observed_barcodes),
        'dropped_duplicate_cells': int(dropped_duplicates),
        'neighbor_radius': int(support_radius),
        'min_neighbors': int(min_neighbors),
        'closing_radius': int(max(0, int(inferred_grid_closing_radius))),
        'fill_holes': bool(inferred_grid_fill_holes),
        'snap_disp_median': float(np.median(disp)) if disp.size else 0.0,
        'snap_disp_p95': float(np.quantile(disp, 0.95)) if disp.size else 0.0,
        'snap_disp_max': float(np.max(disp)) if disp.size else 0.0,
    }


def _build_inferred_grid_tiles(
    coordinates: pd.DataFrame,
    verbose: bool = True,
    n_jobs: int = None,
    inferred_grid_pitch: Optional[float] = None,
    inferred_grid_neighbor_radius: int = 1,
    inferred_grid_min_neighbors: int = 3,
    inferred_grid_closing_radius: int = 0,
    inferred_grid_fill_holes: bool = False,
) -> tuple[pd.DataFrame, dict]:
    state = _prepare_inferred_grid_state(
        coordinates,
        verbose=verbose,
        n_jobs=n_jobs,
        inferred_grid_pitch=inferred_grid_pitch,
        inferred_grid_neighbor_radius=inferred_grid_neighbor_radius,
        inferred_grid_min_neighbors=inferred_grid_min_neighbors,
        inferred_grid_closing_radius=inferred_grid_closing_radius,
        inferred_grid_fill_holes=inferred_grid_fill_holes,
    )

    x_all = state['x_all']
    y_all = state['y_all']
    barcode_all = state['barcode_all']
    observed_mask = state['observed_mask']
    filled_mask = state['filled_mask']
    origin_x = state['origin_x']
    origin_y = state['origin_y']
    pitch_x = state['pitch_x']
    pitch_y = state['pitch_y']
    observed_barcodes = state['observed_barcodes']
    observed_cells = state['observed_cells']
    filled_cells = int(filled_mask.sum())
    synthetic_cells = int(max(0, filled_cells - observed_cells))

    observed_tiles = pd.DataFrame({
        'x': x_all.astype(np.float64, copy=False),
        'y': y_all.astype(np.float64, copy=False),
        'barcode': barcode_all.astype(str, copy=False),
        'origin': np.ones(observed_barcodes, dtype=np.int8),
    })

    if synthetic_cells > 0:
        syn_gy, syn_gx = np.nonzero(filled_mask & ~observed_mask)
        syn_x = origin_x + syn_gx.astype(np.float64, copy=False) * float(pitch_x)
        syn_y = origin_y + syn_gy.astype(np.float64, copy=False) * float(pitch_y)
        obs_pts = np.column_stack([x_all, y_all])
        syn_pts = np.column_stack([syn_x, syn_y])
        tree = cKDTree(obs_pts)
        workers = -1 if (n_jobs in (None, -1)) else int(n_jobs)
        try:
            _, nn_idx = tree.query(syn_pts, k=1, workers=workers)
        except TypeError:
            _, nn_idx = tree.query(syn_pts, k=1)
        syn_barcodes = barcode_all[np.asarray(nn_idx, dtype=np.int64)]
        synthetic_tiles = pd.DataFrame({
            'x': syn_x,
            'y': syn_y,
            'barcode': syn_barcodes.astype(str, copy=False),
            'origin': np.zeros(synthetic_cells, dtype=np.int8),
        })
        tiles = pd.concat([observed_tiles, synthetic_tiles], ignore_index=True)
    else:
        tiles = observed_tiles.reset_index(drop=True)

    info = dict(state)
    info.update({
        'mode': 'direct',
        'filled_cells': int(filled_cells),
        'synthetic_cells': int(synthetic_cells),
    })
    return tiles, info


def _build_inferred_boundary_voronoi_tiles(
    coordinates: pd.DataFrame,
    verbose: bool = True,
    n_jobs: int = None,
    inferred_grid_pitch: Optional[float] = None,
    inferred_grid_neighbor_radius: int = 1,
    inferred_grid_min_neighbors: int = 3,
    inferred_grid_closing_radius: int = 0,
    inferred_grid_fill_holes: bool = False,
) -> tuple[pd.DataFrame, dict]:
    state = _prepare_inferred_grid_state(
        coordinates,
        verbose=verbose,
        n_jobs=n_jobs,
        inferred_grid_pitch=inferred_grid_pitch,
        inferred_grid_neighbor_radius=inferred_grid_neighbor_radius,
        inferred_grid_min_neighbors=inferred_grid_min_neighbors,
        inferred_grid_closing_radius=inferred_grid_closing_radius,
        inferred_grid_fill_holes=inferred_grid_fill_holes,
    )

    x_all = state['x_all']
    y_all = state['y_all']
    barcode_all = state['barcode_all']
    observed_mask = state['observed_mask']
    filled_mask = state['filled_mask']
    origin_x = state['origin_x']
    origin_y = state['origin_y']
    pitch_x = state['pitch_x']
    pitch_y = state['pitch_y']
    observed_barcodes = state['observed_barcodes']

    support_mask = ndi.binary_fill_holes(filled_mask)
    support_mask |= filled_mask
    support_cells = int(support_mask.sum())
    empty_support_cells = int(max(0, support_cells - int(observed_mask.sum())))

    support_gy, support_gx = np.nonzero(support_mask)
    support_x = origin_x + support_gx.astype(np.float64, copy=False) * float(pitch_x)
    support_y = origin_y + support_gy.astype(np.float64, copy=False) * float(pitch_y)

    obs_pts = np.column_stack([x_all, y_all])
    support_pts = np.column_stack([support_x, support_y])
    tree = cKDTree(obs_pts)
    workers = -1 if (n_jobs in (None, -1)) else int(n_jobs)
    try:
        _, nn_idx = tree.query(support_pts, k=1, workers=workers)
    except TypeError:
        _, nn_idx = tree.query(support_pts, k=1)
    support_barcodes = barcode_all[np.asarray(nn_idx, dtype=np.int64)]

    observed_tiles = pd.DataFrame({
        'x': x_all.astype(np.float64, copy=False),
        'y': y_all.astype(np.float64, copy=False),
        'barcode': barcode_all.astype(str, copy=False),
        'origin': np.ones(observed_barcodes, dtype=np.int8),
    })
    support_tiles = pd.DataFrame({
        'x': support_x,
        'y': support_y,
        'barcode': support_barcodes.astype(str, copy=False),
        'origin': np.zeros(support_cells, dtype=np.int8),
    })
    tiles = pd.concat([observed_tiles, support_tiles], ignore_index=True)

    info = dict(state)
    info.update({
        'mode': 'boundary_voronoi',
        'filled_cells': int(filled_mask.sum()),
        'support_cells': int(support_cells),
        'synthetic_cells': int(support_cells),
        'generated_cells': int(support_cells),
        'empty_support_cells': int(empty_support_cells),
        'boundary_fill_holes_forced': True,
    })
    return tiles, info


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
    use_inferred_grid_tiles: bool = False,
    inferred_grid_mode: str = 'direct',
    inferred_grid_pitch: Optional[float] = None,
    inferred_grid_neighbor_radius: int = 1,
    inferred_grid_min_neighbors: int = 3,
    inferred_grid_closing_radius: int = 0,
    inferred_grid_fill_holes: bool = False,
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
    use_inferred_grid_tiles : bool, optional (default=False)
        If True, infer a post-QC regular lattice from the surviving coordinates
        and use one of the inferred-grid tile builders controlled by
        ``inferred_grid_mode``.
    inferred_grid_mode : {'direct','boundary_voronoi'}, optional (default='direct')
        Inferred-grid tiling mode.
        - 'direct': keep observed spots at their original coordinates and add
          only synthetic support cells near observed occupancy.
        - 'boundary_voronoi': infer a support boundary on the post-QC lattice,
          fill its interior, and assign every support cell to the nearest
          surviving barcode (discrete Voronoi on the inferred support).
    inferred_grid_pitch : float or None, optional (default=None)
        Override the inferred lattice pitch. When None, X/Y pitches are inferred
        separately from the surviving coordinates.
    inferred_grid_neighbor_radius : int, optional (default=1)
        Radius of the local support neighborhood used to propose inferred grid
        cells near observed ones.
    inferred_grid_min_neighbors : int, optional (default=3)
        Minimum number of observed neighbors within the support neighborhood for
        an empty inferred grid cell to be synthesized.
    inferred_grid_closing_radius : int, optional (default=0)
        Optional additional binary-closing radius on the inferred occupancy grid.
        Keep at 0 for the most conservative behavior.
    inferred_grid_fill_holes : bool, optional (default=False)
        If True, fill enclosed holes in the inferred occupancy grid before
        nearest-spot assignment. This is more aggressive and more artificial.
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
    inferred_grid_mode_norm = str(inferred_grid_mode or 'direct').lower()
    removal_report = {
        'mode': ((f'inferred_grid_{inferred_grid_mode_norm}') if use_inferred_grid_tiles else ('coords_as_tiles' if use_coords_as_tiles else 'voronoi')),
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

    if use_coords_as_tiles and use_inferred_grid_tiles:
        raise ValueError("use_coords_as_tiles and use_inferred_grid_tiles cannot both be True.")
    if use_inferred_grid_tiles and inferred_grid_mode_norm not in {'direct', 'boundary_voronoi'}:
        raise ValueError("inferred_grid_mode must be one of {'direct', 'boundary_voronoi'}.")

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

        # Only attempt NN-preserving rescale when compaction is actually enabled.
        will_attempt_compaction = (
            coords_compaction == 'tight'
            or (coords_max_gap_factor is not None and coords_max_gap_factor > 0)
        )

        # Compute original median 1-NN distance only when needed for rescaling.
        med_nn_orig = None
        if rescale_flag and will_attempt_compaction:
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

        compaction_applied = False
        if coords_compaction == 'tight':
            try:
                tiles['x'] = _rank_compact(tiles['x'].to_numpy(dtype=float), float(coords_tight_step))
                tiles['y'] = _rank_compact(tiles['y'].to_numpy(dtype=float), float(coords_tight_step))
                compaction_applied = True
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
                compaction_applied = True
                if verbose:
                    logging.info(
                        f"Coords-as-tiles compaction applied (gap cap factor={coords_max_gap_factor})."
                    )
                    logging.info(f"Coords-as-tiles compaction took {time.perf_counter() - t_compact:.2f}s")
            except Exception as e:
                logging.warning(f"Coordinate compaction skipped due to error: {e}")

        # Optionally rescale to match original median 1-NN distance
        if rescale_flag and not compaction_applied and verbose:
            logging.info("Skipping NN-distance rescale because no compaction was applied.")

        if rescale_flag and compaction_applied and med_nn_orig is not None and med_nn_orig > 0:
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
            if keep_mask.size == initial_obs and bool(np.all(keep_mask)):
                # Nothing was filtered; avoid an expensive full AnnData copy.
                if verbose:
                    logging.info("No observations were filtered in coords mode; skipping adata subsetting copy.")
            else:
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

    if use_inferred_grid_tiles:
        if inferred_grid_mode_norm == 'boundary_voronoi':
            if verbose:
                logging.info("Skipping Voronoi/rasterise; inferring a post-QC support boundary and filling it by nearest-barcode Voronoi assignment.")
        else:
            if verbose:
                logging.info("Skipping Voronoi/rasterise; inferring a post-QC lattice from surviving coordinates.")
        removal_report['filter_mode'] = f'inferred_grid:{inferred_grid_mode_norm}'
        if inferred_grid_mode_norm == 'boundary_voronoi':
            tiles, inferred_info = _build_inferred_boundary_voronoi_tiles(
                coordinates,
                verbose=verbose,
                n_jobs=n_jobs,
                inferred_grid_pitch=inferred_grid_pitch,
                inferred_grid_neighbor_radius=inferred_grid_neighbor_radius,
                inferred_grid_min_neighbors=inferred_grid_min_neighbors,
                inferred_grid_closing_radius=inferred_grid_closing_radius,
                inferred_grid_fill_holes=inferred_grid_fill_holes,
            )
        else:
            tiles, inferred_info = _build_inferred_grid_tiles(
                coordinates,
                verbose=verbose,
                n_jobs=n_jobs,
                inferred_grid_pitch=inferred_grid_pitch,
                inferred_grid_neighbor_radius=inferred_grid_neighbor_radius,
                inferred_grid_min_neighbors=inferred_grid_min_neighbors,
                inferred_grid_closing_radius=inferred_grid_closing_radius,
                inferred_grid_fill_holes=inferred_grid_fill_holes,
            )
        adata.uns["tiles"] = tiles.reset_index(drop=True)
        adata.uns["tiles_generated"] = True
        adata.uns["tiles_inferred_grid_info"] = inferred_info
        adata.uns["tiles_removal_report"] = removal_report
        if verbose:
            if inferred_grid_mode_norm == 'boundary_voronoi':
                logging.info(
                    "Inferred-boundary Voronoi tiles: pitch=(%.4f, %.4f), grid=%dx%d, barcodes=%d, occupied=%d, support=%d, generated=%d",
                    inferred_info['pitch_x'],
                    inferred_info['pitch_y'],
                    inferred_info['grid_shape'][0],
                    inferred_info['grid_shape'][1],
                    inferred_info['observed_barcodes'],
                    inferred_info['observed_cells'],
                    inferred_info['support_cells'],
                    inferred_info['generated_cells'],
                )
            else:
                logging.info(
                    "Inferred-grid tiles: pitch=(%.4f, %.4f), grid=%dx%d, barcodes=%d, occupied=%d, filled=%d, synthetic=%d",
                    inferred_info['pitch_x'],
                    inferred_info['pitch_y'],
                    inferred_info['grid_shape'][0],
                    inferred_info['grid_shape'][1],
                    inferred_info['observed_barcodes'],
                    inferred_info['observed_cells'],
                    inferred_info['filled_cells'],
                    inferred_info['synthetic_cells'],
                )
            logging.info(
                "Inferred-grid snap displacement: median=%.4g, p95=%.4g, max=%.4g",
                inferred_info['snap_disp_median'],
                inferred_info['snap_disp_p95'],
                inferred_info['snap_disp_max'],
            )
            logging.info("Tiles have been stored in adata.uns['tiles'] (%s mode).", inferred_grid_mode_norm)

        filtered_barcodes = pd.Index(coordinates.index.astype(str))
        initial_obs = adata.n_obs
        try:
            obs_names_str = adata.obs_names.astype(str)
            keep_mask = obs_names_str.isin(filtered_barcodes)
            if keep_mask.size == initial_obs and bool(np.all(keep_mask)):
                if verbose:
                    logging.info("No observations were filtered in inferred-grid mode; skipping adata subsetting copy.")
            else:
                adata = adata[keep_mask, :].copy()
        except Exception:
            adata = adata[filtered_barcodes, :].copy()
        final_obs = adata.n_obs
        if verbose:
            logging.info(
                f"adata has been subset from {initial_obs} to {final_obs} observations based on tiles (inferred-grid mode)."
            )
            logging.info("generate_tiles completed (inferred-grid mode).")
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
        coords_idx_str = pd.Index(index).astype(str)
        rasterised_barcodes = set(tiles["barcode"].astype(str).unique()) if len(tiles) else set()
        all_coords_idx_str = pd.Index(coordinates.index).astype(str)
        coords_xy = None
        if restore_zero_pixel_tiles or restore_voronoi_dropped:
            coords_xy = coordinates.copy(deep=False)
            if 0 in coords_xy.columns and 1 in coords_xy.columns:
                coords_xy = coords_xy.rename(columns={0: 'x', 1: 'y'})
            else:
                cols = list(coords_xy.columns[:2])
                coords_xy = coords_xy.rename(columns={cols[0]: 'x', cols[1]: 'y'})
            coords_xy = coords_xy.loc[:, ["x", "y"]].copy(deep=False)
            coords_xy.index = all_coords_idx_str

        if restore_zero_pixel_tiles:
            missing_after_raster = coords_idx_str.difference(pd.Index(rasterised_barcodes))
            if len(missing_after_raster):
                to_add = coords_xy.loc[coords_xy.index.intersection(missing_after_raster)].copy()
                if len(to_add):
                    to_add["x"] = to_add["x"].round().astype(int)
                    to_add["y"] = to_add["y"].round().astype(int)
                    to_add["barcode"] = to_add.index.astype(str)
                    to_add["origin"] = 1
                    tiles = pd.concat([tiles, to_add.reset_index(drop=True)], ignore_index=True)
                    if verbose:
                        logging.info(f"Added {len(to_add)} origin-only tiles to preserve missing spots.")

        if restore_voronoi_dropped:
            removed_voronoi_all = all_coords_idx_str.difference(coords_idx_str)
            if len(removed_voronoi_all):
                to_add2 = coords_xy.loc[coords_xy.index.intersection(removed_voronoi_all)].copy()
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
        if keep_mask.size == initial_obs and bool(np.all(keep_mask)):
            if verbose:
                logging.info("No observations were filtered after Voronoi/rasterise; skipping adata subsetting copy.")
        else:
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
