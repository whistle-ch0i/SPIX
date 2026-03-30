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

_INFERRED_GRID_TARGET_OBSERVED_RATIO = 0.99
_INFERRED_GRID_MIN_PITCH_SCALE = 0.65
_INFERRED_GRID_SHRINK_FACTOR = 0.90
_INFERRED_GRID_BINARY_STEPS = 4
_INFERRED_GRID_MAX_DENSE_CELLS = 200_000_000


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


def _measure_inferred_grid_assignment(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    pitch_x: float,
    pitch_y: float,
    return_keep_mask: bool = False,
    return_disp: bool = False,
) -> dict:
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

    grid_w = int(gx_all.max()) + 1
    grid_h = int(gy_all.max()) + 1
    total_cells = int(grid_h) * int(grid_w)
    grid_ids = gy_all.astype(np.int64, copy=False) * np.int64(grid_w) + gx_all.astype(np.int64, copy=False)

    if return_keep_mask:
        unique_ids, first_idx = np.unique(grid_ids, return_index=True)
        keep_mask = np.zeros(grid_ids.size, dtype=bool)
        keep_mask[np.asarray(first_idx, dtype=np.int64)] = True
        observed_cells = int(unique_ids.size)
    else:
        keep_mask = None
        observed_cells = int(np.unique(grid_ids).size)
    dropped_duplicates = int(grid_ids.size - observed_cells)

    result = {
        'origin_x': float(origin_x),
        'origin_y': float(origin_y),
        'grid_shape': (int(grid_h), int(grid_w)),
        'total_cells': int(total_cells),
        'observed_cells': int(observed_cells),
        'dropped_duplicates': int(dropped_duplicates),
        'observed_ratio': float(observed_cells / max(1, grid_ids.size)),
        'keep_mask': keep_mask,
    }
    if return_keep_mask or return_disp:
        result['gx_all'] = gx_all
        result['gy_all'] = gy_all
    if return_disp:
        obs_x_centers_all = origin_x + gx_all.astype(np.float64, copy=False) * float(pitch_x)
        obs_y_centers_all = origin_y + gy_all.astype(np.float64, copy=False) * float(pitch_y)
        disp = np.sqrt((x_vals - obs_x_centers_all) ** 2 + (y_vals - obs_y_centers_all) ** 2)
        result['disp'] = disp
    return result


def _auto_tune_inferred_grid_pitch(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    pitch_x: float,
    pitch_y: float,
    target_observed_ratio: float = _INFERRED_GRID_TARGET_OBSERVED_RATIO,
    min_scale: float = _INFERRED_GRID_MIN_PITCH_SCALE,
    shrink_factor: float = _INFERRED_GRID_SHRINK_FACTOR,
    binary_steps: int = _INFERRED_GRID_BINARY_STEPS,
    max_dense_cells: int = _INFERRED_GRID_MAX_DENSE_CELLS,
) -> tuple[float, float, dict]:
    cache: dict[tuple[float, float], dict] = {}
    candidate_count = 0

    def _eval(px: float, py: float) -> dict:
        nonlocal candidate_count
        key = (float(px), float(py))
        if key not in cache:
            cache[key] = _measure_inferred_grid_assignment(
                x_vals,
                y_vals,
                pitch_x=float(px),
                pitch_y=float(py),
                return_keep_mask=False,
                return_disp=False,
            )
            candidate_count += 1
        return cache[key]

    base_metrics = _eval(pitch_x, pitch_y)
    selected_pitch_x = float(pitch_x)
    selected_pitch_y = float(pitch_y)
    selected_metrics = base_metrics
    status = 'retained'

    if base_metrics['observed_ratio'] < float(target_observed_ratio):
        floor_pitch_x = float(pitch_x) * float(min_scale)
        floor_pitch_y = float(pitch_y) * float(min_scale)
        safe_pitch_x = None
        safe_pitch_y = None
        safe_metrics = None
        unsafe_pitch_x = float(pitch_x)
        unsafe_pitch_y = float(pitch_y)
        unsafe_metrics = base_metrics
        best_pitch_x = float(pitch_x)
        best_pitch_y = float(pitch_y)
        best_metrics = base_metrics
        curr_pitch_x = float(pitch_x)
        curr_pitch_y = float(pitch_y)
        dense_limit_hit = False

        while (curr_pitch_x > floor_pitch_x) or (curr_pitch_y > floor_pitch_y):
            next_pitch_x = max(floor_pitch_x, curr_pitch_x * float(shrink_factor))
            next_pitch_y = max(floor_pitch_y, curr_pitch_y * float(shrink_factor))
            if np.isclose(next_pitch_x, curr_pitch_x) and np.isclose(next_pitch_y, curr_pitch_y):
                break
            metrics = _eval(next_pitch_x, next_pitch_y)
            if metrics['total_cells'] > int(max_dense_cells):
                dense_limit_hit = True
                break
            if (
                metrics['observed_ratio'] > best_metrics['observed_ratio']
                or (
                    np.isclose(metrics['observed_ratio'], best_metrics['observed_ratio'])
                    and next_pitch_x > best_pitch_x
                )
            ):
                best_pitch_x = float(next_pitch_x)
                best_pitch_y = float(next_pitch_y)
                best_metrics = metrics
            if metrics['observed_ratio'] >= float(target_observed_ratio):
                safe_pitch_x = float(next_pitch_x)
                safe_pitch_y = float(next_pitch_y)
                safe_metrics = metrics
                break
            unsafe_pitch_x = float(next_pitch_x)
            unsafe_pitch_y = float(next_pitch_y)
            unsafe_metrics = metrics
            curr_pitch_x = float(next_pitch_x)
            curr_pitch_y = float(next_pitch_y)

        if safe_metrics is not None:
            selected_pitch_x = safe_pitch_x
            selected_pitch_y = safe_pitch_y
            selected_metrics = safe_metrics
            status = 'adjusted'
            lo_x = float(safe_pitch_x)
            lo_y = float(safe_pitch_y)
            hi_x = float(unsafe_pitch_x)
            hi_y = float(unsafe_pitch_y)
            for _ in range(int(max(0, binary_steps))):
                if (
                    np.isclose(lo_x, hi_x)
                    or np.isclose(lo_y, hi_y)
                    or ((hi_x - lo_x) / max(lo_x, 1e-12) < 0.01)
                ):
                    break
                mid_x = 0.5 * (lo_x + hi_x)
                mid_y = 0.5 * (lo_y + hi_y)
                metrics = _eval(mid_x, mid_y)
                if metrics['total_cells'] > int(max_dense_cells):
                    hi_x = float(mid_x)
                    hi_y = float(mid_y)
                    continue
                if metrics['observed_ratio'] >= float(target_observed_ratio):
                    lo_x = float(mid_x)
                    lo_y = float(mid_y)
                    selected_pitch_x = float(mid_x)
                    selected_pitch_y = float(mid_y)
                    selected_metrics = metrics
                else:
                    hi_x = float(mid_x)
                    hi_y = float(mid_y)
        else:
            selected_pitch_x = best_pitch_x
            selected_pitch_y = best_pitch_y
            selected_metrics = best_metrics
            if dense_limit_hit:
                status = 'dense_limit'
            elif (selected_pitch_x < float(pitch_x)) or (selected_pitch_y < float(pitch_y)):
                status = 'min_scale_reached'

    return selected_pitch_x, selected_pitch_y, {
        'status': str(status),
        'applied': bool(
            (not np.isclose(selected_pitch_x, float(pitch_x)))
            or (not np.isclose(selected_pitch_y, float(pitch_y)))
        ),
        'target_observed_ratio': float(target_observed_ratio),
        'min_pitch_scale': float(min_scale),
        'shrink_factor': float(shrink_factor),
        'binary_steps': int(max(0, binary_steps)),
        'candidate_count': int(candidate_count),
        'initial_pitch_x': float(pitch_x),
        'initial_pitch_y': float(pitch_y),
        'selected_pitch_x': float(selected_pitch_x),
        'selected_pitch_y': float(selected_pitch_y),
        'initial_observed_ratio': float(base_metrics['observed_ratio']),
        'selected_observed_ratio': float(selected_metrics['observed_ratio']),
        'initial_observed_cells': int(base_metrics['observed_cells']),
        'selected_observed_cells': int(selected_metrics['observed_cells']),
        'initial_dropped_duplicates': int(base_metrics['dropped_duplicates']),
        'selected_dropped_duplicates': int(selected_metrics['dropped_duplicates']),
        'initial_total_cells': int(base_metrics['total_cells']),
        'selected_total_cells': int(selected_metrics['total_cells']),
    }


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

    pitch_tuning = None
    if inferred_grid_pitch is None:
        pitch0 = _infer_regular_grid_pitch(x_vals, y_vals, n_jobs=n_jobs)
        pitch_x = float(pitch0)
        pitch_y = float(pitch0)
        pitch_x, pitch_y, pitch_tuning = _auto_tune_inferred_grid_pitch(
            x_vals,
            y_vals,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
        )
    else:
        pitch_x = float(inferred_grid_pitch)
        pitch_y = float(inferred_grid_pitch)

    assignment = _measure_inferred_grid_assignment(
        x_vals,
        y_vals,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        return_keep_mask=True,
        return_disp=True,
    )
    origin_x = float(assignment['origin_x'])
    origin_y = float(assignment['origin_y'])
    gx_all = assignment['gx_all']
    gy_all = assignment['gy_all']
    keep_mask = assignment['keep_mask']
    dropped_duplicates = int(assignment['dropped_duplicates'])
    if dropped_duplicates and verbose:
        logging.info(
            f"Inferred-grid mode collapsed {dropped_duplicates} duplicated barcodes sharing the same inferred grid cell for support-mask construction."
        )

    gx_occ = gx_all[keep_mask]
    gy_occ = gy_all[keep_mask]

    grid_h, grid_w = assignment['grid_shape']
    total_cells = int(assignment['total_cells'])
    if total_cells > _INFERRED_GRID_MAX_DENSE_CELLS:
        raise MemoryError(
            f"Inferred-grid mask would require {grid_h}x{grid_w}={total_cells:,} cells. "
            f"Specify inferred_grid_pitch explicitly or increase tensor_resolution; "
            f"current inferred pitch was ({pitch_x:.6g}, {pitch_y:.6g})."
        )

    observed_mask = np.zeros((grid_h, grid_w), dtype=bool)
    observed_mask[gy_occ, gx_occ] = True
    observed_cells = int(observed_mask.sum())
    observed_barcodes = int(barcode_vals.shape[0])
    observed_ratio = float(observed_cells / max(1, observed_barcodes))

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

    disp = np.asarray(assignment['disp'], dtype=float)

    if pitch_tuning is not None and verbose and bool(pitch_tuning.get('applied', False)):
        logging.info(
            "Auto-tuned inferred-grid pitch from (%.4f, %.4f) to (%.4f, %.4f); observed ratio %.4f -> %.4f.",
            pitch_tuning['initial_pitch_x'],
            pitch_tuning['initial_pitch_y'],
            pitch_tuning['selected_pitch_x'],
            pitch_tuning['selected_pitch_y'],
            pitch_tuning['initial_observed_ratio'],
            pitch_tuning['selected_observed_ratio'],
        )

    state = {
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
        'observed_ratio': float(observed_ratio),
        'dropped_duplicate_cells': int(dropped_duplicates),
        'neighbor_radius': int(support_radius),
        'min_neighbors': int(min_neighbors),
        'closing_radius': int(max(0, int(inferred_grid_closing_radius))),
        'fill_holes': bool(inferred_grid_fill_holes),
        'snap_disp_median': float(np.median(disp)) if disp.size else 0.0,
        'snap_disp_p95': float(np.quantile(disp, 0.95)) if disp.size else 0.0,
        'snap_disp_max': float(np.max(disp)) if disp.size else 0.0,
    }
    if pitch_tuning is not None:
        state['pitch_tuning'] = pitch_tuning
    return state


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
    observed_grid_cells = int(state['observed_cells'])

    support_mask = ndi.binary_fill_holes(filled_mask)
    support_mask |= filled_mask
    support_cells = int(support_mask.sum())
    empty_support_mask = support_mask & ~observed_mask
    empty_support_cells = int(empty_support_mask.sum())

    empty_support_gy, empty_support_gx = np.nonzero(empty_support_mask)
    synthetic_support_x = origin_x + empty_support_gx.astype(np.float64, copy=False) * float(pitch_x)
    synthetic_support_y = origin_y + empty_support_gy.astype(np.float64, copy=False) * float(pitch_y)

    if empty_support_cells > 0:
        obs_pts = np.column_stack([x_all, y_all])
        support_pts = np.column_stack([synthetic_support_x, synthetic_support_y])
        tree = cKDTree(obs_pts)
        workers = -1 if (n_jobs in (None, -1)) else int(n_jobs)
        try:
            _, nn_idx = tree.query(support_pts, k=1, workers=workers)
        except TypeError:
            _, nn_idx = tree.query(support_pts, k=1)
        synthetic_support_barcodes = barcode_all[np.asarray(nn_idx, dtype=np.int64)]
    else:
        synthetic_support_barcodes = np.empty(0, dtype=barcode_all.dtype)

    observed_tiles = pd.DataFrame({
        'x': x_all.astype(np.float64, copy=False),
        'y': y_all.astype(np.float64, copy=False),
        'barcode': barcode_all.astype(str, copy=False),
        'origin': np.ones(observed_barcodes, dtype=np.int8),
    })
    observed_support_tiles = pd.DataFrame({
        'x': x_all.astype(np.float64, copy=False),
        'y': y_all.astype(np.float64, copy=False),
        'barcode': barcode_all.astype(str, copy=False),
        'origin': np.zeros(observed_barcodes, dtype=np.int8),
    })
    support_tiles = pd.DataFrame({
        'x': synthetic_support_x,
        'y': synthetic_support_y,
        'barcode': synthetic_support_barcodes.astype(str, copy=False),
        'origin': np.zeros(empty_support_cells, dtype=np.int8),
    })
    tiles = pd.concat([observed_tiles, observed_support_tiles, support_tiles], ignore_index=True)
    support_tile_rows = int(observed_barcodes + empty_support_cells)

    info = dict(state)
    info.update({
        'mode': 'boundary_voronoi',
        'observed_grid_cells': int(observed_grid_cells),
        'observed_grid_ratio': float(observed_grid_cells / max(1, observed_barcodes)),
        'observed_cells': int(observed_barcodes),
        'observed_ratio': 1.0,
        'filled_cells': int(filled_mask.sum()),
        'support_cells': int(support_cells),
        'support_tile_rows': int(support_tile_rows),
        'synthetic_cells': int(empty_support_cells),
        'generated_cells': int(support_tile_rows),
        'empty_support_cells': int(empty_support_cells),
        'support_represented_barcodes': int(observed_barcodes),
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
    coords_max_gap_factor: Optional[float] = None,
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
        ``reduce_tensor_resolution`` before writing. Optional coordinate
        compaction can be enabled via ``coords_max_gap_factor`` /
        ``coords_compaction``.
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
    coords_max_gap_factor : float or None, optional (default=None)
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
                    "Inferred-boundary Voronoi tiles: pitch=(%.4f, %.4f), grid=%dx%d, barcodes=%d, preserved=%d, grid-occupied=%d, support=%d, synthetic=%d",
                    inferred_info['pitch_x'],
                    inferred_info['pitch_y'],
                    inferred_info['grid_shape'][0],
                    inferred_info['grid_shape'][1],
                    inferred_info['observed_barcodes'],
                    inferred_info['observed_cells'],
                    inferred_info.get('observed_grid_cells', inferred_info['observed_cells']),
                    inferred_info['support_cells'],
                    inferred_info['synthetic_cells'],
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
