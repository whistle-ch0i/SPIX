import numpy as np
import scanpy as sc
import pandas as pd
import hashlib
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from anndata import AnnData
from sklearn.cluster import KMeans
import logging
import time
import re
from typing import Any

from ..utils.utils import create_pseudo_centroids

from .kmeans_segmentation import kmeans_segmentation
from .louvain_segmentation import louvain_segmentation
from .leiden_segmentation import leiden_segmentation
from .slic_segmentation import slic_segmentation
from .som_segmentation import som_segmentation
from .louvain_slic_segmentation import louvain_slic_segmentation
from .leiden_slic_segmentation import leiden_slic_segmentation
from .image_plot_slic_segmentation import (
    _resolve_cached_preview_size,
    image_plot_slic_segmentation,
    slic_segmentation_from_cached_image,
)
from .compactness_mode import (
    DEFAULT_COMPACTNESS_SEARCH_MULTIPLIERS,
    evaluate_candidate_metrics_from_labels,
    finalize_compactness_rows,
    infer_compactness_range as infer_compactness_range_from_arrays,
    prepare_compactness_metric_context,
    resolve_compactness,
    schedule_compactness,
    search_compactness,
)
from ..image_processing.image_cache import cache_embedding_image


_CACHED_IMAGE_SEARCH_STATE: dict[str, Any] | None = None


def _set_cached_image_search_state(state: dict[str, Any] | None) -> None:
    global _CACHED_IMAGE_SEARCH_STATE
    _CACHED_IMAGE_SEARCH_STATE = state


def _cached_image_search_worker(candidate: float) -> dict[str, float]:
    state = _CACHED_IMAGE_SEARCH_STATE
    if state is None:
        raise RuntimeError("Cached-image compactness worker state is not initialized.")
    labels = slic_segmentation_from_cached_image(
        state["cache"],
        n_segments=int(state["n_segments"]),
        compactness=float(candidate),
        enforce_connectivity=bool(state["enforce_connectivity"]),
        show_image=False,
        verbose=False,
        figsize=None,
        fig_dpi=None,
        mask_background=bool(state["mask_background"]),
    )[0]
    metrics = evaluate_candidate_metrics_from_labels(
        labels=labels,
        metric_context=state["metric_context"],
        compactness_search_dims=int(state["compactness_search_dims"]),
    )
    return {"compactness": float(candidate), **metrics}


def _summarize_cache_overlap(cache: dict[str, Any]) -> dict[str, Any]:
    img = np.asarray(cache.get("img"))
    if img.ndim != 3:
        raise ValueError("Cached object does not contain a 3D image array.")
    support_mask = np.asarray(cache.get("support_mask"), dtype=bool)
    cx = np.asarray(cache.get("cx"), dtype=np.int32).ravel()
    cy = np.asarray(cache.get("cy"), dtype=np.int32).ravel()
    n_tiles = int(min(cx.size, cy.size))
    if n_tiles <= 0:
        raise ValueError("Cached image does not contain cx/cy tile centers.")
    flat = cy.astype(np.int64, copy=False) * np.int64(max(1, img.shape[1])) + cx.astype(np.int64, copy=False)
    unique, counts = np.unique(flat, return_counts=True)
    unique_centers = int(unique.size)
    collided_tiles = int(n_tiles - unique_centers)
    support_pixels = int(np.count_nonzero(support_mask))
    canvas_pixels = int(support_mask.size)
    return {
        "shape": tuple(int(x) for x in img.shape),
        "n_tiles": int(n_tiles),
        "unique_centers": int(unique_centers),
        "collided_tiles": int(collided_tiles),
        "collision_fraction": float(collided_tiles / max(1, n_tiles)),
        "max_stack": int(counts.max()) if counts.size else 1,
        "support_pixels": int(support_pixels),
        "canvas_pixels": int(canvas_pixels),
        "support_fraction": float(support_pixels / max(1, canvas_pixels)),
        "soft_rasterization": bool(cache.get("soft_rasterization", False)),
        "tile_px": int(cache.get("tile_px", 1)),
        "scale": float(cache.get("scale", 1.0)),
        "pixel_perfect": bool(cache.get("pixel_perfect", False)),
        "runtime_fill_from_boundary": bool(cache.get("runtime_fill_from_boundary", False)),
        "runtime_fill_holes": bool(cache.get("runtime_fill_holes", False)),
    }


def _print_compactness_cache_summary(cache: dict[str, Any]) -> None:
    stats = _summarize_cache_overlap(cache)
    shape = stats["shape"]
    print(
        "[perf] compactness raster cache: "
        f"shape={shape[1]}x{shape[0]}x{shape[2]} | "
        f"pixel_perfect={int(stats['pixel_perfect'])} | "
        f"soft={int(stats['soft_rasterization'])} | "
        f"tile_px={int(stats['tile_px'])} | "
        f"fillb={int(stats['runtime_fill_from_boundary'])} | "
        f"fillh={int(stats['runtime_fill_holes'])} | "
        f"scale={float(stats['scale']):.6f}"
    )
    print(
        "[perf] compactness raster overlap: "
        f"n_tiles={int(stats['n_tiles'])} | "
        f"unique_centers={int(stats['unique_centers'])} | "
        f"collided_tiles={int(stats['collided_tiles'])} | "
        f"collision_fraction={float(stats['collision_fraction']):.6f} | "
        f"max_stack={int(stats['max_stack'])} | "
        f"support_pixels={int(stats['support_pixels'])} | "
        f"support_fraction={float(stats['support_fraction']):.6f}"
    )


def _resolve_requested_coordinate_mode(
    adata: AnnData,
    *,
    coordinate_mode: str,
    origin: bool,
    array_row_key: str,
    array_col_key: str,
) -> str:
    coord_mode = str(coordinate_mode or "spatial").lower()
    if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
        coord_mode = "spatial"
    if coord_mode == "auto":
        if (
            bool(origin)
            and (str(array_row_key) in adata.obs.columns)
            and (str(array_col_key) in adata.obs.columns)
            and int(adata.n_obs) >= 200_000
        ):
            coord_mode = "visiumhd"
        else:
            coord_mode = "spatial"
    return coord_mode


def _cache_matches_segmentation_raster_config(
    adata: AnnData,
    cache: dict[str, Any],
    *,
    coordinate_mode: str,
    origin: bool,
    array_row_key: str,
    array_col_key: str,
    pixel_perfect: bool,
    pixel_shape: str,
    runtime_fill_from_boundary: bool,
    runtime_fill_closing_radius: int,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool,
    soft_rasterization: bool,
    resolve_center_collisions: bool,
    center_collision_radius: int,
) -> tuple[bool, list[str]]:
    requested_coord_mode = _resolve_requested_coordinate_mode(
        adata,
        coordinate_mode=str(coordinate_mode),
        origin=bool(origin),
        array_row_key=str(array_row_key),
        array_col_key=str(array_col_key),
    )
    params = cache.get("image_plot_params", {}) or {}
    mismatches: list[str] = []

    def _compare(name: str, cached_value: Any, requested_value: Any) -> None:
        if isinstance(requested_value, float):
            if cached_value is None or not np.isclose(float(cached_value), float(requested_value)):
                mismatches.append(name)
            return
        if cached_value is None or cached_value != requested_value:
            mismatches.append(name)

    _compare("coordinate_mode", cache.get("coordinate_mode", None), str(requested_coord_mode))
    _compare("array_row_key", cache.get("array_row_key", None), str(array_row_key))
    _compare("array_col_key", cache.get("array_col_key", None), str(array_col_key))
    _compare("pixel_perfect", cache.get("pixel_perfect", None), bool(pixel_perfect))
    _compare("pixel_shape", cache.get("pixel_shape", None), str(pixel_shape))
    _compare("downsample_factor", cache.get("downsample_factor", None), 1.0)
    _compare(
        "runtime_fill_from_boundary",
        params.get("runtime_fill_from_boundary", None),
        bool(runtime_fill_from_boundary),
    )
    _compare(
        "runtime_fill_closing_radius",
        params.get("runtime_fill_closing_radius", None),
        int(max(0, int(runtime_fill_closing_radius))),
    )
    _compare(
        "runtime_fill_external_radius",
        params.get("runtime_fill_external_radius", None),
        int(max(0, int(runtime_fill_external_radius))),
    )
    _compare(
        "runtime_fill_holes",
        params.get("runtime_fill_holes", None),
        bool(runtime_fill_holes),
    )
    _compare("soft_rasterization", cache.get("soft_rasterization", None), bool(soft_rasterization))
    _compare(
        "resolve_center_collisions",
        params.get("resolve_center_collisions", None),
        bool(resolve_center_collisions),
    )
    _compare(
        "center_collision_radius",
        params.get("center_collision_radius", None),
        int(center_collision_radius),
    )
    return (len(mismatches) == 0), mismatches


def _resolve_compactness_search_raster_options(
    adata: AnnData,
    *,
    coordinate_mode: str,
    origin: bool,
    array_row_key: str,
    array_col_key: str,
    pixel_perfect: bool,
    pixel_shape: str,
    soft_rasterization: bool,
    resolve_center_collisions: bool,
    center_collision_radius: int,
    compactness_search_downsample_factor: float,
    compactness_search_raster_mode: str,
    verbose: bool,
) -> dict[str, Any]:
    requested_coord_mode = _resolve_requested_coordinate_mode(
        adata,
        coordinate_mode=str(coordinate_mode),
        origin=bool(origin),
        array_row_key=str(array_row_key),
        array_col_key=str(array_col_key),
    )
    mode = str(compactness_search_raster_mode or "auto").lower()
    effective_pixel_perfect = bool(pixel_perfect)
    effective_pixel_shape = str(pixel_shape)
    effective_soft = bool(soft_rasterization)
    effective_resolve = bool(resolve_center_collisions)
    reason = "requested"

    auto_dense_origin_soft = bool(
        bool(origin)
        and str(requested_coord_mode) == "spatial"
        and float(compactness_search_downsample_factor) > 1.0
    )
    if mode in {"auto", "dense_origin_soft"} and (mode == "dense_origin_soft" or auto_dense_origin_soft):
        effective_pixel_perfect = True
        effective_pixel_shape = "square"
        effective_soft = True
        effective_resolve = False
        reason = "dense_origin_soft"
    elif mode in {"legacy", "hard"}:
        reason = "legacy"

    result = {
        "requested_coordinate_mode": str(requested_coord_mode),
        "mode": mode,
        "reason": reason,
        "pixel_perfect": bool(effective_pixel_perfect),
        "pixel_shape": str(effective_pixel_shape),
        "soft_rasterization": bool(effective_soft),
        "resolve_center_collisions": bool(effective_resolve),
        "center_collision_radius": int(center_collision_radius),
        "downsample_factor": float(compactness_search_downsample_factor),
    }
    if verbose and reason == "dense_origin_soft":
        print(
            "[perf] compactness search raster mode: "
            "auto -> dense_origin_soft | "
            f"coord={str(requested_coord_mode)} | "
            f"origin={int(bool(origin))} | "
            f"ds={float(compactness_search_downsample_factor):.4g}"
        )
    return result


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
        out_rows.append((bc, _choose_majority_label(sub)))
    return pd.DataFrame(out_rows, columns=["barcode", "Segment"])


def _choose_majority_label(sub: pd.DataFrame):
    """Choose one label with deterministic tie-breaks for a grouped tile subset."""
    counts = sub["Segment"].value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index.to_list()
    if len(candidates) == 1:
        return candidates[0]

    # Prefer the label that appears most often among origin==1 tiles.
    sub_origin = sub[sub["_origin"] == 1]
    if len(sub_origin) > 0:
        origin_counts = sub_origin["Segment"].value_counts()
        best_origin_count = max(int(origin_counts.get(c, 0)) for c in candidates)
        tie_after_origin = [c for c in candidates if int(origin_counts.get(c, 0)) == best_origin_count]
        if len(tie_after_origin) == 1:
            return tie_after_origin[0]
        return sorted(tie_after_origin)[0]

    # Final deterministic fallback.
    return sorted(candidates)[0]


def _aggregate_labels_majority_by_obs_index(
    obs_indices: np.ndarray | pd.Series | list,
    labels: np.ndarray,
    origin_flags: np.ndarray | pd.Series | list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate tile-level labels to one label per observation index.

    Fast path for numeric labels using NumPy sorting/run-length encoding. Falls
    back to the older pandas implementation for non-numeric labels.
    """
    obs_idx = np.asarray(obs_indices, dtype=np.int64)
    lab = np.asarray(labels)
    if obs_idx.ndim != 1 or lab.ndim != 1 or obs_idx.size != lab.size:
        raise ValueError("obs_indices and labels must be 1D arrays of the same length")
    if obs_idx.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=lab.dtype)

    if origin_flags is None:
        origin = np.zeros(obs_idx.size, dtype=np.int8)
    else:
        origin = np.asarray(origin_flags).astype(np.int8, copy=False)
        if origin.ndim != 1 or origin.size != obs_idx.size:
            raise ValueError("origin_flags must be None or a 1D array matching obs_indices length")

    if not np.issubdtype(lab.dtype, np.number):
        df = pd.DataFrame({
            "obs_index": obs_idx,
            "Segment": lab,
            "_origin": origin.astype(int, copy=False),
        })
        out_idx: list[int] = []
        out_labels: list = []
        for obs_i, sub in df.groupby("obs_index", sort=False):
            out_idx.append(int(obs_i))
            out_labels.append(_choose_majority_label(sub))
        return np.asarray(out_idx, dtype=np.int64), np.asarray(out_labels, dtype=object)

    lab_int = np.asarray(lab, dtype=np.int64)
    perm = np.lexsort((lab_int, obs_idx))
    obs_s = obs_idx[perm]
    lab_s = lab_int[perm]
    org_s = origin[perm].astype(np.int64, copy=False)

    pair_start = np.empty(obs_s.size, dtype=bool)
    pair_start[0] = True
    pair_start[1:] = (obs_s[1:] != obs_s[:-1]) | (lab_s[1:] != lab_s[:-1])
    starts = np.flatnonzero(pair_start)
    pair_obs = obs_s[starts]
    pair_lab = lab_s[starts]
    pair_counts = np.diff(np.append(starts, obs_s.size)).astype(np.int64, copy=False)
    pair_origin = np.add.reduceat(org_s, starts).astype(np.int64, copy=False)

    # Sort by obs asc, count desc, origin-count desc, label asc; first row per obs wins.
    choose_perm = np.lexsort((pair_lab, -pair_origin, -pair_counts, pair_obs))
    best_obs = pair_obs[choose_perm]
    best_lab = pair_lab[choose_perm]
    keep = np.empty(best_obs.size, dtype=bool)
    keep[0] = True
    keep[1:] = best_obs[1:] != best_obs[:-1]
    return best_obs[keep].astype(np.int64, copy=False), best_lab[keep]


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


def _compactness_mode_header(
    *,
    compactness: float | None,
    compactness_mode: str,
    target_segment_um: float | None,
    compactness_base_size_um: float | None,
    compactness_scale_power: float,
    compactness_min: float | None,
    compactness_max: float | None,
    compactness_search_multipliers,
    compactness_search_dims: int,
    compactness_search_jobs: int,
):
    return {
        "compactness": None if compactness is None else float(compactness),
        "compactness_mode": str(compactness_mode),
        "target_segment_um": None if target_segment_um is None else float(target_segment_um),
        "compactness_base_size_um": None if compactness_base_size_um is None else float(compactness_base_size_um),
        "compactness_scale_power": float(compactness_scale_power),
        "compactness_min": None if compactness_min is None else float(compactness_min),
        "compactness_max": None if compactness_max is None else float(compactness_max),
        "compactness_search_multipliers": [float(x) for x in compactness_search_multipliers],
        "compactness_search_dims": int(compactness_search_dims),
        "compactness_search_jobs": int(compactness_search_jobs),
    }


def _store_segment_pixel_boundary_payload(
    adata: AnnData,
    *,
    segment_name: str,
    label_img: np.ndarray | None,
    raster_meta: dict[str, Any] | None,
    origin: bool,
    coordinate_mode: str,
    array_row_key: str,
    array_col_key: str,
) -> None:
    if label_img is None or raster_meta is None:
        return
    segment_suffix = _sanitize_segment_name(segment_name)
    adata.uns[f"{segment_suffix}_pixel_boundary"] = {
        "segment_key": str(segment_name),
        "origin": bool(origin),
        "coordinate_mode": str(coordinate_mode),
        "array_row_key": str(array_row_key),
        "array_col_key": str(array_col_key),
        "label_img": np.asarray(label_img, dtype=np.int32, copy=True),
        **dict(raster_meta),
    }


def fixed_grid_segmentation(
    spatial_coords: np.ndarray,
    side_um: float,
    origin_x: float | None = None,
    origin_y: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Assign each coordinate to an exact square grid cell.

    This is the exact analytic grid used in the figure4 helper scripts:
    labels are defined by ``floor((x - x0) / side_um)`` and
    ``floor((y - y0) / side_um)`` with ``x0/y0`` taken from either the
    provided origin or the minimum observed coordinates.
    """
    coords = np.asarray(spatial_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("spatial_coords must be a 2D array with at least two columns.")

    side_um = float(max(side_um, 1e-9))
    labels = np.full(coords.shape[0], np.nan, dtype=object)
    valid = np.isfinite(coords[:, :2]).all(axis=1)
    if not np.any(valid):
        return labels, {
            "grid_mode": "fixed_grid",
            "grid_side_um": side_um,
            "grid_origin_x": float(origin_x) if origin_x is not None else np.nan,
            "grid_origin_y": float(origin_y) if origin_y is not None else np.nan,
            "grid_nx": 0,
            "grid_ny": 0,
            "grid_total_cells": 0,
            "grid_occupied_segments": 0,
        }

    xy = coords[valid, :2]
    x = xy[:, 0]
    y = xy[:, 1]
    x0 = float(np.nanmin(x) if origin_x is None else origin_x)
    y0 = float(np.nanmin(y) if origin_y is None else origin_y)
    gx = np.floor((x - x0) / side_um).astype(np.int64)
    gy = np.floor((y - y0) / side_um).astype(np.int64)
    labels_valid = gx.astype(str) + "_" + gy.astype(str)
    labels[valid] = labels_valid

    x1 = float(np.nanmax(x))
    y1 = float(np.nanmax(y))
    nx = max(int(np.floor((x1 - x0) / side_um)) + 1, 1)
    ny = max(int(np.floor((y1 - y0) / side_um)) + 1, 1)
    meta = {
        "grid_mode": "fixed_grid",
        "grid_side_um": side_um,
        "grid_origin_x": x0,
        "grid_origin_y": y0,
        "grid_nx": nx,
        "grid_ny": ny,
        "grid_total_cells": int(nx * ny),
        "grid_occupied_segments": int(pd.Index(labels_valid).nunique()),
    }
    return labels, meta


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


def _compactness_search_cache_key(
    *,
    base_key: str,
    embedding: str,
    dimensions: list[int],
    origin: bool,
    coordinate_mode: str,
    array_row_key: str,
    array_col_key: str,
    pixel_perfect: bool,
    pixel_shape: str,
    soft_rasterization: bool,
    resolve_center_collisions: bool,
    center_collision_radius: int,
    runtime_fill_from_boundary: bool,
    runtime_fill_closing_radius: int,
    runtime_fill_external_radius: int,
    runtime_fill_holes: bool,
    downsample_factor: float,
) -> str:
    payload = "|".join(
        [
            str(base_key),
            str(embedding),
            ",".join(str(int(x)) for x in dimensions),
            f"origin={int(bool(origin))}",
            f"coord={coordinate_mode}",
            f"row={array_row_key}",
            f"col={array_col_key}",
            f"pp={int(bool(pixel_perfect))}",
            f"pixel={pixel_shape}",
            f"soft={int(bool(soft_rasterization))}",
            f"collide={int(bool(resolve_center_collisions))}",
            f"cr={int(center_collision_radius)}",
            f"fillb={int(bool(runtime_fill_from_boundary))}",
            f"fillr={int(runtime_fill_closing_radius)}",
            f"fille={int(runtime_fill_external_radius)}",
            f"fillh={int(bool(runtime_fill_holes))}",
            f"ds={float(downsample_factor):.6g}",
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{base_key}__compactness_search__{digest}"


def _search_compactness_cached_image_process(
    *,
    candidates: list[float],
    cache: dict[str, Any],
    n_segments: int,
    embeddings: np.ndarray,
    spatial_coords: np.ndarray,
    compactness_search_dims: int,
    compactness_min: float | None,
    compactness_max: float | None,
    scheduled: float | None,
    mode: str,
    target_segment_um: float | None,
    compactness_search_jobs: int,
    compactness_metric_sample_size: int | None,
    enforce_connectivity: bool,
    mask_background: bool,
    verbose: bool,
) -> tuple[float, dict[str, Any]]:
    clipped = sorted(
        {
            float(max(float(compactness_min), x) if compactness_min is not None else x)
            for x in [
                float(min(float(compactness_max), c) if compactness_max is not None else c)
                for c in candidates
                if np.isfinite(float(c)) and float(c) > 0
            ]
        }
    )
    if not clipped:
        raise ValueError("No valid compactness candidates remain after clipping.")
    t0 = time.perf_counter()
    metric_context = prepare_compactness_metric_context(
        spatial_coords=np.asarray(spatial_coords, dtype=float),
        embeddings=np.asarray(embeddings, dtype=float),
        compactness_metric_sample_size=compactness_metric_sample_size,
    )
    t1 = time.perf_counter()
    if verbose:
        print(
            f"[perf] {mode} process candidate prep: "
            f"sample+knn={t1 - t0:.3f}s | candidates={len(clipped)} | jobs={max(1, int(compactness_search_jobs))}"
        )
    state = {
        "cache": cache,
        "n_segments": int(n_segments),
        "metric_context": metric_context,
        "compactness_search_dims": int(compactness_search_dims),
        "enforce_connectivity": bool(enforce_connectivity),
        "mask_background": bool(mask_background),
    }
    rows = []
    _set_cached_image_search_state(state)
    try:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=max(1, min(int(compactness_search_jobs), len(clipped))),
            mp_context=ctx,
        ) as executor:
            if verbose:
                for candidate in clipped:
                    print(f"[perf] {mode} process candidate submit: compactness={float(candidate):.6g}")
            rows = list(executor.map(_cached_image_search_worker, clipped))
    finally:
        _set_cached_image_search_state(None)
    return finalize_compactness_rows(
        rows,
        scheduled=scheduled,
        mode=str(mode),
        target_segment_um=target_segment_um,
        compactness_search_jobs=int(compactness_search_jobs),
    )


def _effective_compactness_search_jobs(
    *,
    requested_jobs: int,
    n_candidates: int,
    n_segments: int,
    backend: str,
    verbose: bool,
) -> int:
    jobs = max(1, min(int(requested_jobs), int(n_candidates)))
    if str(backend) == "process":
        if int(n_segments) >= 5000:
            cap = 2
        elif int(n_segments) >= 1500:
            cap = 4
        elif int(n_segments) >= 500:
            cap = 6
        else:
            cap = 8
        effective = max(1, min(jobs, cap))
        if verbose and effective != jobs:
            print(
                f"[perf] compactness search jobs capped for process backend: "
                f"requested={jobs} -> effective={effective} | n_segments={int(n_segments)}"
            )
        return int(effective)
    return int(jobs)


def _pseudo_centroids_from_clusters_df(
    adata: AnnData,
    clusters_df: pd.DataFrame,
    *,
    embedding: str,
    dimensions: list[int],
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    """Fast pseudo-centroids in adata.obs order from (barcode -> Segment) mapping.

    This replaces the slow pandas merge/groupby/transform path in `create_pseudo_centroids`
    for very large datasets.
    """
    if "barcode" not in clusters_df.columns or "Segment" not in clusters_df.columns:
        raise ValueError("clusters_df must have columns: 'barcode', 'Segment'")

    obs_barcodes = pd.Index(adata.obs.index.astype(str))
    tile_barcodes = pd.Index(clusters_df["barcode"].astype(str))
    obs_pos = obs_barcodes.get_indexer(tile_barcodes)
    ok = obs_pos >= 0

    E_full = np.asarray(adata.obsm[embedding])
    if E_full.ndim != 2 or E_full.shape[0] != adata.n_obs:
        raise ValueError(f"Embedding '{embedding}' must be a 2D array with n_obs rows.")
    E_use = E_full if E_full.shape[1] == len(dimensions) else E_full[:, dimensions]
    E_use = np.asarray(E_use, dtype=np.float32)
    d = int(E_use.shape[1])

    if not np.any(ok):
        return np.full((adata.n_obs, d), np.nan, dtype=np.float32)

    pos = obs_pos[ok].astype(np.int64, copy=False)
    labels = clusters_df.loc[ok, "Segment"].to_numpy()
    codes, _uniques = pd.factorize(labels, sort=False)
    codes = codes.astype(np.int64, copy=False)
    k = int(codes.max(initial=-1)) + 1
    if k <= 0:
        return np.full((adata.n_obs, d), np.nan, dtype=np.float32)

    counts = np.bincount(codes, minlength=k).astype(np.int64, copy=False)
    sums = np.zeros((k, d), dtype=np.float32)

    if chunk_size < 1:
        chunk_size = 1_000_000
    n = int(pos.size)
    for i in range(0, n, int(chunk_size)):
        j = min(i + int(chunk_size), n)
        p = pos[i:j]
        c = codes[i:j]
        np.add.at(sums, c, E_use[p])

    denom = counts.astype(np.float32)
    denom[denom < 1] = np.nan
    means = (sums / denom[:, None]).astype(np.float32, copy=False)

    out = np.full((adata.n_obs, d), np.nan, dtype=np.float32)
    out[pos] = means[codes]
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


def scale_compactness(
    compactness: float,
    *,
    from_um: float,
    to_um: float,
    power: float = 0.3,
    compactness_min: float | None = None,
    compactness_max: float | None = None,
) -> float:
    """Scale a reference compactness from one target segment size to another."""
    value = float(compactness)
    base = float(from_um)
    target = float(to_um)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("compactness must be a finite positive number.")
    if not np.isfinite(base) or base <= 0:
        raise ValueError("from_um must be a finite positive number.")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("to_um must be a finite positive number.")
    scaled = float(value * (target / base) ** float(power))
    if compactness_min is not None:
        scaled = max(float(compactness_min), scaled)
    if compactness_max is not None:
        scaled = min(float(compactness_max), scaled)
    return float(scaled)


def resolve_fast_compactness_reference_target_um(
    adata: AnnData,
    *,
    target_segment_um: float,
    pitch_um: float = 2.0,
    origin: bool = True,
    compactness_search_downsample_factor: float = 2.0,
) -> float:
    """Resolve the data-aware reference target used by ``fast_select_compactness``.

    This returns the target size that fast compactness search will actually use
    before optional scaling back to ``target_segment_um``.
    """
    try:
        tiles_all = adata.uns["tiles"]
        tiles_for_n = tiles_all
        if "origin" in tiles_all.columns and bool(origin) and (tiles_all["origin"] == 1).any():
            tiles_for_n = tiles_all[tiles_all["origin"] == 1]
        if "barcode" in tiles_for_n.columns:
            tiles_for_n = tiles_for_n.drop_duplicates("barcode")
        n_tiles_for_reference = int(len(tiles_for_n))
    except Exception:
        n_tiles_for_reference = None
    resolved = _auto_fast_compactness_search_target_um(
        float(target_segment_um),
        n_tiles=n_tiles_for_reference,
        pitch_um=float(pitch_um),
        compactness_search_downsample_factor=float(compactness_search_downsample_factor),
        min_reference_segment_side_px=10.0,
        max_stable_search_n_segments=15000,
    )
    if resolved is None:
        raise ValueError("Could not resolve a valid reference target segment size.")
    return float(resolved)


def _auto_fast_compactness_search_target_um(
    target_segment_um: float | None,
    *,
    n_tiles: int | None = None,
    pitch_um: float = 2.0,
    compactness_search_downsample_factor: float = 2.0,
    min_reference_segment_side_px: float = 10.0,
    max_stable_search_n_segments: int = 15000,
) -> float | None:
    if target_segment_um is None:
        return None
    target = float(target_segment_um)
    if not np.isfinite(target) or target <= 0:
        return None
    pitch = float(pitch_um)
    downsample = float(compactness_search_downsample_factor)
    min_side_px = float(min_reference_segment_side_px)
    n_tiles_eff = None if n_tiles is None else int(max(1, n_tiles))
    max_search_n = int(max(1, max_stable_search_n_segments))
    if (
        not np.isfinite(pitch) or pitch <= 0
        or not np.isfinite(downsample) or downsample <= 0
        or not np.isfinite(min_side_px) or min_side_px <= 0
    ):
        return target
    base = pitch * downsample * min_side_px
    if n_tiles_eff is not None and n_tiles_eff > 0:
        base_by_search_n = pitch * np.sqrt(float(n_tiles_eff) / float(max_search_n))
        if np.isfinite(base_by_search_n) and base_by_search_n > 0:
            base = max(float(base), float(base_by_search_n))
    nice_targets = np.asarray([16.0, 20.0, 24.0, 30.0, 40.0, 50.0, 64.0, 80.0, 100.0, 128.0, 160.0, 200.0], dtype=float)
    ge = nice_targets[nice_targets >= float(base)]
    if ge.size > 0:
        base = float(ge[0])
    if target < base:
        return float(base)
    return target


def _resolve_segmentation_tiles(
    adata: AnnData,
    *,
    origin: bool,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tiles_all = adata.uns["tiles"]
    inferred_info = adata.uns.get("tiles_inferred_grid_info", {})
    inferred_mode = str(inferred_info.get("mode", "")).lower()
    if origin:
        tiles = tiles_all[tiles_all.get("origin", 1) == 1]
    else:
        if inferred_mode == "boundary_voronoi" and ("origin" in tiles_all.columns):
            tiles = tiles_all[tiles_all["origin"] == 0]
            if verbose:
                print("[info] Using inferred boundary support-grid tiles only for origin=False segmentation.")
        else:
            tiles = tiles_all
    if "barcode" in tiles.columns:
        tiles = tiles.copy()
        tiles["barcode"] = tiles["barcode"].astype(str)
        if tiles["barcode"].duplicated().any():
            origin_vals = pd.to_numeric(tiles.get("origin", 1), errors="coerce").fillna(1)
            has_non_origin = (origin_vals != 1).any()
            keep_dups = (not origin) and has_non_origin
            if keep_dups:
                if verbose:
                    print(
                        "[info] adata.uns['tiles'] has duplicated barcodes (expected for origin=False raster tiles); keeping duplicates."
                    )
            else:
                if verbose:
                    print(
                        "[warning] adata.uns['tiles'] has duplicated barcodes; keeping first occurrence for segmentation."
                    )
                tiles = tiles.drop_duplicates("barcode", keep="first")

    tiles_for_n = tiles_all
    try:
        if "origin" in tiles_all.columns and (tiles_all["origin"] == 1).any():
            tiles_for_n = tiles_all[tiles_all["origin"] == 1]
        if "barcode" in tiles_for_n.columns:
            tiles_for_n = tiles_for_n.drop_duplicates("barcode")
    except Exception:
        pass
    return tiles, tiles_for_n


def _prepare_image_plot_coordinate_inputs(
    adata: AnnData,
    *,
    origin: bool,
    library_id: str | None,
    coordinate_mode: str,
    array_row_key: str,
    array_col_key: str,
    target_segment_um: float | None,
    pitch_um: float,
    resolution: float,
    verbose: bool,
) -> dict[str, Any]:
    if library_id is not None:
        raise NotImplementedError("select_compactness currently supports library_id=None only.")

    t0 = time.perf_counter()
    use_direct_origin_spatial = False
    spatial_direct = None
    if origin:
        try:
            spatial_direct = np.asarray(adata.obsm["spatial"])
        except Exception:
            spatial_direct = None
        if (
            spatial_direct is not None
            and spatial_direct.ndim == 2
            and spatial_direct.shape[0] == adata.n_obs
            and spatial_direct.shape[1] >= 2
        ):
            use_direct_origin_spatial = True
    t1 = time.perf_counter()

    if use_direct_origin_spatial:
        tiles = pd.DataFrame()
        tiles_for_n = pd.DataFrame(index=np.arange(int(adata.n_obs)))
        resolved_resolution = float(resolution)
        if target_segment_um is not None:
            spots_per_segment = max((float(target_segment_um) / float(pitch_um)) ** 2, 1.0)
            resolved_resolution = float(max(1, int(round(float(adata.n_obs) / spots_per_segment))))
    else:
        tiles, tiles_for_n = _resolve_segmentation_tiles(adata, origin=origin, verbose=verbose)
        resolved_resolution = float(resolution)
        if target_segment_um is not None:
            resolved_resolution = float(
                n_segments_from_size(tiles_for_n, target_um=target_segment_um, pitch_um=pitch_um)
            )
    t2 = time.perf_counter()

    if use_direct_origin_spatial:
        obs_index_arr = np.arange(adata.n_obs, dtype=np.int64)
        barcode_arr = adata.obs.index.astype(str).to_numpy()
        x = np.asarray(spatial_direct[:, 0], dtype=float)
        y = np.asarray(spatial_direct[:, 1], dtype=float)
        origin_flags = np.ones(adata.n_obs, dtype=np.int8)
        valid = np.isfinite(x) & np.isfinite(y)
        coordinates_df = pd.DataFrame(
            {
                "barcode": barcode_arr[valid],
                "obs_index": obs_index_arr[valid],
                "x": x[valid],
                "y": y[valid],
                "origin": origin_flags[valid],
            }
        )
    else:
        if "barcode" not in tiles.columns:
            raise ValueError("adata.uns['tiles'] must contain a 'barcode' column.")
        if ("x" not in tiles.columns) or ("y" not in tiles.columns):
            raise ValueError("adata.uns['tiles'] must contain 'x' and 'y' columns.")
        obs_index = pd.Index(adata.obs.index.astype(str))
        tile_barcodes = tiles["barcode"].astype(str).to_numpy()
        tile_obs_indices = obs_index.get_indexer(tile_barcodes)
        x = pd.to_numeric(tiles["x"], errors="coerce").to_numpy(dtype=float, copy=False)
        y = pd.to_numeric(tiles["y"], errors="coerce").to_numpy(dtype=float, copy=False)
        origin_flags = pd.to_numeric(tiles.get("origin", 1), errors="coerce").fillna(1).to_numpy()
        valid = (tile_obs_indices >= 0) & np.isfinite(x) & np.isfinite(y)
        coordinates_df = pd.DataFrame(
            {
                "barcode": tile_barcodes[valid],
                "obs_index": tile_obs_indices[valid].astype(np.int64, copy=False),
                "x": x[valid],
                "y": y[valid],
                "origin": origin_flags[valid],
            }
        )
    t3 = time.perf_counter()

    coord_mode = str(coordinate_mode or "spatial").lower()
    if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
        coord_mode = "spatial"
    if coord_mode == "auto":
        if origin and (array_row_key in adata.obs.columns) and (array_col_key in adata.obs.columns) and int(adata.n_obs) >= 200_000:
            coord_mode = "visiumhd"
        else:
            coord_mode = "spatial"
    if coord_mode in {"array", "visium", "visiumhd"}:
        if (array_row_key not in adata.obs.columns) or (array_col_key not in adata.obs.columns):
            raise ValueError(
                f"coordinate_mode='array' requires adata.obs['{array_row_key}'] and adata.obs['{array_col_key}']."
            )
        col = pd.to_numeric(coordinates_df["barcode"].map(adata.obs[array_col_key]), errors="coerce")
        row = pd.to_numeric(coordinates_df["barcode"].map(adata.obs[array_row_key]), errors="coerce")
        if coord_mode == "visium":
            row_i = row.astype("Int64")
            parity = (row_i % 2).astype(float)
            coordinates_df["x"] = col.astype(float) + 0.5 * parity
            coordinates_df["y"] = row.astype(float) * (np.sqrt(3.0) / 2.0)
        else:
            coordinates_df["x"] = col.astype(float)
            coordinates_df["y"] = row.astype(float)
        coordinates_df = coordinates_df.dropna(subset=["x", "y"]).reset_index(drop=True)
    t4 = time.perf_counter()

    spatial_coords = coordinates_df.loc[:, ["x", "y"]].values
    t5 = time.perf_counter()
    if verbose:
        print(
            "[perf] prepare coordinate inputs detail: "
            f"detect_spatial={t1 - t0:.3f}s | resolve_tiles={t2 - t1:.3f}s | "
            f"build_coords_df={t3 - t2:.3f}s | coord_transform={t4 - t3:.3f}s | "
            f"to_numpy={t5 - t4:.3f}s"
        )
    return {
        "tiles": tiles,
        "tiles_for_n": tiles_for_n,
        "coordinates_df": coordinates_df,
        "spatial_coords": spatial_coords,
        "resolution": float(resolved_resolution),
    }


def _prepare_image_plot_compactness_inputs(
    adata: AnnData,
    *,
    dimensions: list[int],
    embedding: str,
    origin: bool,
    library_id: str | None,
    coordinate_mode: str,
    array_row_key: str,
    array_col_key: str,
    target_segment_um: float | None,
    pitch_um: float,
    resolution: float,
    verbose: bool,
) -> dict[str, Any]:
    prepared = _prepare_image_plot_coordinate_inputs(
        adata,
        origin=origin,
        library_id=library_id,
        coordinate_mode=coordinate_mode,
        array_row_key=array_row_key,
        array_col_key=array_col_key,
        target_segment_um=target_segment_um,
        pitch_um=pitch_um,
        resolution=resolution,
        verbose=verbose,
    )
    E_full = np.asarray(adata.obsm[embedding])
    if E_full.ndim != 2 or E_full.shape[0] != adata.n_obs:
        raise ValueError(f"Embedding '{embedding}' must be a 2D array with n_obs rows.")
    E_use = E_full if E_full.shape[1] == len(dimensions) else E_full[:, dimensions]
    obs_idx = prepared["coordinates_df"]["obs_index"].to_numpy(dtype=np.int64, copy=False)
    embeddings = np.asarray(E_use[obs_idx], dtype=float)
    emb_valid = np.isfinite(embeddings).all(axis=1)
    if not np.all(emb_valid):
        coordinates_df = prepared["coordinates_df"].iloc[np.flatnonzero(emb_valid)].reset_index(drop=True)
        spatial_coords = coordinates_df.loc[:, ["x", "y"]].to_numpy(dtype=float, copy=False)
        embeddings = embeddings[emb_valid]
    else:
        coordinates_df = prepared["coordinates_df"]
        spatial_coords = prepared["spatial_coords"]
    return {
        **prepared,
        "coordinates_df": coordinates_df,
        "spatial_coords": spatial_coords,
        "embeddings": embeddings,
    }


def infer_compactness_range(
    adata: AnnData,
    *,
    dimensions: list[int] = list(range(30)),
    embedding: str = "X_embedding",
    resolution: float = 10.0,
    target_segment_um: float | None = None,
    pitch_um: float = 2.0,
    origin: bool = True,
    library_id: str | None = None,
    coordinate_mode: str = "spatial",
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    compactness_min: float | None = None,
    compactness_max: float | None = None,
    compactness_search_dims: int = 12,
    verbose: bool = True,
) -> dict[str, Any]:
    t_prepare0 = time.perf_counter()
    prepared = _prepare_image_plot_compactness_inputs(
        adata,
        dimensions=list(dimensions),
        embedding=embedding,
        origin=origin,
        library_id=library_id,
        coordinate_mode=coordinate_mode,
        array_row_key=array_row_key,
        array_col_key=array_col_key,
        target_segment_um=target_segment_um,
        pitch_um=float(pitch_um),
        resolution=float(resolution),
        verbose=verbose,
    )
    t_prepare1 = time.perf_counter()
    if verbose:
        print(f"[perf] compactness prepare inputs: {t_prepare1 - t_prepare0:.3f}s")
    t_range0 = time.perf_counter()
    range_info = infer_compactness_range_from_arrays(
        embeddings=np.asarray(prepared["embeddings"], dtype=float),
        spatial_coords=np.asarray(prepared["spatial_coords"], dtype=float),
        target_segment_um=None if target_segment_um is None else float(target_segment_um),
        resolution=float(prepared["resolution"]),
        pitch_um=float(pitch_um),
        compactness_min=compactness_min,
        compactness_max=compactness_max,
        embedding_dims=int(compactness_search_dims),
    )
    return {
        **range_info,
        "resolution": float(prepared["resolution"]),
        "n_points": int(np.asarray(prepared["spatial_coords"]).shape[0]),
    }


def select_compactness(
    adata: AnnData,
    *,
    dimensions: list[int] = list(range(30)),
    search_dimensions: list[int] | None = None,
    embedding: str = "X_embedding",
    resolution: float = 10.0,
    target_segment_um: float | None = None,
    pitch_um: float = 2.0,
    origin: bool = True,
    library_id: str | None = None,
    coordinate_mode: str = "spatial",
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    figsize: tuple | None = None,
    fig_dpi: int | None = None,
    imshow_tile_size: float | None = None,
    imshow_scale_factor: float = 1.0,
    pixel_perfect: bool = False,
    enforce_connectivity: bool = False,
    pixel_shape: str = "square",
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool = True,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool = False,
    center_collision_radius: int = 2,
    mask_background: bool = False,
    compactness_min: float | None = None,
    compactness_max: float | None = None,
    compactness_candidates: list[float] | None = None,
    coarse_steps: int = 5,
    refine_steps: int = 5,
    compactness_search_dims: int = 12,
    compactness_search_jobs: int = 1,
    compactness_metric_sample_size: int | None = 12000,
    compactness_search_downsample_factor: float = 1.0,
    compactness_search_raster_mode: str = "auto",
    compactness_search_resolution_cap: int | None = None,
    use_cached_image: bool = True,
    image_cache_key: str = "image_plot_slic",
    verbose: bool = True,
) -> dict[str, Any]:
    effective_dimensions = list(dimensions if search_dimensions is None else search_dimensions)
    search_raster_options = _resolve_compactness_search_raster_options(
        adata,
        coordinate_mode=str(coordinate_mode),
        origin=bool(origin),
        array_row_key=str(array_row_key),
        array_col_key=str(array_col_key),
        pixel_perfect=bool(pixel_perfect),
        pixel_shape=str(pixel_shape),
        soft_rasterization=bool(soft_rasterization),
        resolve_center_collisions=bool(resolve_center_collisions),
        center_collision_radius=int(center_collision_radius),
        compactness_search_downsample_factor=float(compactness_search_downsample_factor),
        compactness_search_raster_mode=str(compactness_search_raster_mode),
        verbose=bool(verbose),
    )
    effective_pixel_perfect = bool(search_raster_options["pixel_perfect"])
    effective_pixel_shape = str(search_raster_options["pixel_shape"])
    effective_soft_rasterization = bool(search_raster_options["soft_rasterization"])
    effective_resolve_center_collisions = bool(search_raster_options["resolve_center_collisions"])
    t_prepare0 = time.perf_counter()
    prepared = _prepare_image_plot_compactness_inputs(
        adata,
        dimensions=effective_dimensions,
        embedding=embedding,
        origin=origin,
        library_id=library_id,
        coordinate_mode=coordinate_mode,
        array_row_key=array_row_key,
        array_col_key=array_col_key,
        target_segment_um=target_segment_um,
        pitch_um=float(pitch_um),
        resolution=float(resolution),
        verbose=verbose,
    )
    t_prepare1 = time.perf_counter()
    if verbose:
        print(f"[perf] compactness prepare inputs: {t_prepare1 - t_prepare0:.3f}s")
    requested_resolution = float(prepared["resolution"])
    search_resolution = float(requested_resolution)
    if compactness_search_resolution_cap is not None:
        search_resolution = float(
            min(
                int(round(requested_resolution)),
                max(1, int(compactness_search_resolution_cap)),
            )
        )
        if verbose and int(round(search_resolution)) != int(round(requested_resolution)):
            print(
                f"[perf] compactness search resolution capped: "
                f"requested_n={int(round(requested_resolution))} -> search_n={int(round(search_resolution))}"
            )
    t_range0 = time.perf_counter()
    range_info = infer_compactness_range_from_arrays(
        embeddings=np.asarray(prepared["embeddings"], dtype=float),
        spatial_coords=np.asarray(prepared["spatial_coords"], dtype=float),
        target_segment_um=None if target_segment_um is None else float(target_segment_um),
        resolution=float(search_resolution),
        pitch_um=float(pitch_um),
        compactness_min=compactness_min,
        compactness_max=compactness_max,
        embedding_dims=int(compactness_search_dims),
    )
    t_range1 = time.perf_counter()
    if verbose:
        print(f"[perf] compactness infer range: {t_range1 - t_range0:.3f}s")

    transient_cache_key = None
    cache = None
    reusable_cache_key = image_cache_key
    if use_cached_image:
        reusable_cache_key = _compactness_search_cache_key(
                base_key=str(image_cache_key),
                embedding=str(embedding),
                dimensions=list(effective_dimensions),
                origin=bool(origin),
                coordinate_mode=str(coordinate_mode),
                array_row_key=str(array_row_key),
                array_col_key=str(array_col_key),
                pixel_perfect=bool(effective_pixel_perfect),
                pixel_shape=str(effective_pixel_shape),
                soft_rasterization=bool(effective_soft_rasterization),
                resolve_center_collisions=bool(effective_resolve_center_collisions),
                center_collision_radius=int(center_collision_radius),
                runtime_fill_from_boundary=bool(runtime_fill_from_boundary),
                runtime_fill_closing_radius=int(runtime_fill_closing_radius),
            runtime_fill_external_radius=int(runtime_fill_external_radius),
            runtime_fill_holes=bool(runtime_fill_holes),
            downsample_factor=float(compactness_search_downsample_factor),
        )
    if use_cached_image and reusable_cache_key in adata.uns:
        candidate_cache = adata.uns[reusable_cache_key]
        cached_emb = candidate_cache.get("embedding_key", None)
        cached_dims = candidate_cache.get("dimensions", None)
        cached_origin = candidate_cache.get("origin", None)
        if (
            (cached_emb is None or cached_emb == embedding)
            and (cached_dims is None or list(cached_dims) == effective_dimensions)
            and (cached_origin is None or bool(cached_origin) == bool(origin))
            and (
                candidate_cache.get("downsample_factor", 1.0) is None
                or np.isclose(float(candidate_cache.get("downsample_factor", 1.0)), float(compactness_search_downsample_factor))
            )
        ):
            cache = candidate_cache
            if verbose:
                print("[perf] reusing raster cache for compactness selection.")
                _print_compactness_cache_summary(cache)
        elif verbose:
            print("[warning] Existing cached image does not match embedding/dimensions/origin; rebuilding cache for compactness selection.")
    if cache is None:
        transient_cache_key = None if use_cached_image else "__spix_select_compactness_cache"
        cache_key = reusable_cache_key if use_cached_image else transient_cache_key
        t_cache0 = time.perf_counter()
        cache = cache_embedding_image(
            adata,
            embedding=embedding,
            dimensions=effective_dimensions,
            origin=origin,
            key=cache_key,
            coordinate_mode=coordinate_mode,
            array_row_key=array_row_key,
            array_col_key=array_col_key,
            figsize=figsize,
            fig_dpi=fig_dpi,
            imshow_tile_size=imshow_tile_size,
            imshow_scale_factor=imshow_scale_factor,
            pixel_perfect=effective_pixel_perfect,
            pixel_shape=effective_pixel_shape,
            verbose=False,
            show=False,
            runtime_fill_from_boundary=runtime_fill_from_boundary,
            runtime_fill_closing_radius=runtime_fill_closing_radius,
            runtime_fill_external_radius=runtime_fill_external_radius,
            runtime_fill_holes=runtime_fill_holes,
            soft_rasterization=effective_soft_rasterization,
            resolve_center_collisions=effective_resolve_center_collisions,
            center_collision_radius=center_collision_radius,
            downsample_factor=float(compactness_search_downsample_factor),
        )
        t_cache1 = time.perf_counter()
        if verbose:
            print(f"[perf] built raster cache for compactness selection: {t_cache1 - t_cache0:.3f}s")
            _print_compactness_cache_summary(cache)

    try:
        use_process_search = bool(
            cache is not None
            and int(compactness_search_jobs) > 1
            and os.name != "nt"
        )
        if compactness_candidates is None:
            lower = float(compactness_min) if compactness_min is not None else float(range_info["lower"])
            upper = float(compactness_max) if compactness_max is not None else float(range_info["upper"])
            if upper < lower:
                raise ValueError("compactness_max must be >= compactness_min.")
            coarse_candidates = np.geomspace(lower, upper, num=max(int(coarse_steps), 2)).tolist()
        else:
            coarse_candidates = [float(x) for x in compactness_candidates]
        t_coarse0 = time.perf_counter()
        coarse_jobs = _effective_compactness_search_jobs(
            requested_jobs=int(compactness_search_jobs),
            n_candidates=len(coarse_candidates),
            n_segments=int(search_resolution),
            backend="process" if use_process_search else "thread",
            verbose=verbose,
        )
        if use_process_search:
            if verbose:
                print("[perf] compactness search backend: process")
            coarse_best, coarse_detail = _search_compactness_cached_image_process(
                candidates=coarse_candidates,
                cache=cache,
                n_segments=int(search_resolution),
                embeddings=np.asarray(prepared["embeddings"], dtype=float),
                spatial_coords=np.asarray(prepared["spatial_coords"], dtype=float),
                compactness_search_dims=int(compactness_search_dims),
                compactness_min=compactness_min,
                compactness_max=compactness_max,
                scheduled=None,
                mode="coarse_search",
                target_segment_um=target_segment_um,
                compactness_search_jobs=int(coarse_jobs),
                compactness_metric_sample_size=compactness_metric_sample_size,
                enforce_connectivity=bool(enforce_connectivity),
                mask_background=bool(mask_background),
                verbose=verbose,
            )
        else:
            if verbose:
                print("[perf] compactness search backend: thread")
            coarse_best, coarse_detail = search_compactness(
                candidates=coarse_candidates,
                embeddings=np.asarray(prepared["embeddings"], dtype=float),
                spatial_coords=np.asarray(prepared["spatial_coords"], dtype=float),
                segmenter=lambda candidate: slic_segmentation_from_cached_image(
                    cache,
                    n_segments=int(search_resolution),
                    compactness=float(candidate),
                    enforce_connectivity=enforce_connectivity,
                    show_image=False,
                    verbose=False,
                    figsize=None,
                    fig_dpi=None,
                    mask_background=mask_background,
                )[0],
                compactness_search_dims=int(compactness_search_dims),
                compactness_min=compactness_min,
                compactness_max=compactness_max,
                mode="coarse_search",
                target_segment_um=target_segment_um,
                compactness_search_jobs=int(coarse_jobs),
                compactness_metric_sample_size=compactness_metric_sample_size,
                verbose=verbose,
            )
        t_coarse1 = time.perf_counter()
        if verbose:
            print(f"[perf] compactness coarse search: {t_coarse1 - t_coarse0:.3f}s")
        if compactness_candidates is None and int(refine_steps) > 0:
            refine_lower = max(lower, float(coarse_best) / 1.7)
            refine_upper = min(upper, float(coarse_best) * 1.7)
            refine_candidates = np.geomspace(
                refine_lower,
                refine_upper,
                num=max(int(refine_steps), 3),
            ).tolist()
            t_refine0 = time.perf_counter()
            refine_jobs = _effective_compactness_search_jobs(
                requested_jobs=int(compactness_search_jobs),
                n_candidates=len(refine_candidates),
                n_segments=int(search_resolution),
                backend="process" if use_process_search else "thread",
                verbose=verbose,
            )
            if use_process_search:
                refine_best, refine_detail = _search_compactness_cached_image_process(
                    candidates=refine_candidates,
                    cache=cache,
                    n_segments=int(search_resolution),
                    embeddings=np.asarray(prepared["embeddings"], dtype=float),
                    spatial_coords=np.asarray(prepared["spatial_coords"], dtype=float),
                    compactness_search_dims=int(compactness_search_dims),
                    compactness_min=compactness_min,
                    compactness_max=compactness_max,
                    scheduled=float(coarse_best),
                    mode="refine_search",
                    target_segment_um=target_segment_um,
                    compactness_search_jobs=int(refine_jobs),
                    compactness_metric_sample_size=compactness_metric_sample_size,
                    enforce_connectivity=bool(enforce_connectivity),
                    mask_background=bool(mask_background),
                    verbose=verbose,
                )
            else:
                refine_best, refine_detail = search_compactness(
                    candidates=refine_candidates,
                    embeddings=np.asarray(prepared["embeddings"], dtype=float),
                    spatial_coords=np.asarray(prepared["spatial_coords"], dtype=float),
                    segmenter=lambda candidate: slic_segmentation_from_cached_image(
                        cache,
                        n_segments=int(search_resolution),
                        compactness=float(candidate),
                        enforce_connectivity=enforce_connectivity,
                        show_image=False,
                        verbose=False,
                        figsize=None,
                        fig_dpi=None,
                        mask_background=mask_background,
                    )[0],
                    compactness_search_dims=int(compactness_search_dims),
                    compactness_min=compactness_min,
                    compactness_max=compactness_max,
                    scheduled=float(coarse_best),
                    mode="refine_search",
                    target_segment_um=target_segment_um,
                    compactness_search_jobs=int(refine_jobs),
                    compactness_metric_sample_size=compactness_metric_sample_size,
                    verbose=verbose,
                )
            t_refine1 = time.perf_counter()
            if verbose:
                print(f"[perf] compactness refine search: {t_refine1 - t_refine0:.3f}s")
        else:
            refine_best = float(coarse_best)
            refine_detail = None
        chosen_detail = refine_detail if refine_detail is not None else coarse_detail
        selected_rows = [
            row for row in chosen_detail["candidates"]
            if float(row["compactness"]) == float(refine_best)
        ]
        observed_segment_count = (
            int(selected_rows[0]["n_segments"])
            if selected_rows
            else None
        )
        result = {
            "selected_compactness": float(refine_best),
            "resolution": float(requested_resolution),
            "requested_n_segments": int(round(float(requested_resolution))),
            "search_resolution_used": float(search_resolution),
            "search_n_segments": int(round(float(search_resolution))),
            "selected_observed_segment_count": observed_segment_count,
            "search_dimensions": effective_dimensions,
            "compactness_search_downsample_factor": float(compactness_search_downsample_factor),
            "compactness_metric_sample_size": None if compactness_metric_sample_size is None else int(compactness_metric_sample_size),
            "compactness_search_raster_options": search_raster_options,
            "mask_background": bool(mask_background),
            "range": range_info,
            "coarse_search": coarse_detail,
            "refine_search": refine_detail,
        }
        return result
    finally:
        if transient_cache_key is not None:
            adata.uns.pop(transient_cache_key, None)


def fast_select_compactness(
    adata: AnnData,
    *,
    dimensions: list[int] = list(range(30)),
    search_dimensions: list[int] | None = None,
    embedding: str = "X_embedding",
    resolution: float = 10.0,
    target_segment_um: float | None = None,
    pitch_um: float = 2.0,
    origin: bool = True,
    mask_background: bool = False,
    library_id: str | None = None,
    coordinate_mode: str = "spatial",
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    soft_rasterization: bool = False,
    compactness_min: float | None = None,
    compactness_max: float | None = None,
    compactness_candidates: list[float] | None = None,
    coarse_steps: int = 6,
    compactness_search_jobs: int = 4,
    compactness_metric_sample_size: int | None = 12000,
    compactness_search_downsample_factor: float = 2.0,
    compactness_search_raster_mode: str = "auto",
    compactness_search_resolution_cap: int | None = None,
    search_target_segment_um: float | None = None,
    compactness_scale_power: float = 0.3,
    verbose: bool = True,
    **kwargs,
) -> dict[str, Any]:
    effective_search_dimensions = (
        list(search_dimensions)
        if search_dimensions is not None
        else list(dimensions[: min(len(dimensions), 8)])
    )
    if search_target_segment_um is None:
        try:
            _tiles_all = adata.uns["tiles"]
            _tiles_for_n = _tiles_all
            if "origin" in _tiles_all.columns and bool(origin) and (_tiles_all["origin"] == 1).any():
                _tiles_for_n = _tiles_all[_tiles_all["origin"] == 1]
            if "barcode" in _tiles_for_n.columns:
                _tiles_for_n = _tiles_for_n.drop_duplicates("barcode")
            n_tiles_for_reference = int(len(_tiles_for_n))
        except Exception:
            n_tiles_for_reference = None
        selection_target_segment_um = _auto_fast_compactness_search_target_um(
            target_segment_um,
            n_tiles=n_tiles_for_reference,
            pitch_um=float(pitch_um),
            compactness_search_downsample_factor=float(compactness_search_downsample_factor),
            min_reference_segment_side_px=10.0,
            max_stable_search_n_segments=15000,
        )
        auto_reference_target = bool(
            selection_target_segment_um is not None
            and target_segment_um is not None
            and not np.isclose(float(selection_target_segment_um), float(target_segment_um))
        )
    else:
        selection_target_segment_um = float(search_target_segment_um)
        auto_reference_target = False
    result = select_compactness(
        adata,
        dimensions=dimensions,
        search_dimensions=effective_search_dimensions,
        embedding=embedding,
        resolution=resolution,
        target_segment_um=selection_target_segment_um if selection_target_segment_um is not None else target_segment_um,
        pitch_um=pitch_um,
        origin=origin,
        mask_background=mask_background,
        library_id=library_id,
        coordinate_mode=coordinate_mode,
        array_row_key=array_row_key,
        array_col_key=array_col_key,
        soft_rasterization=soft_rasterization,
        compactness_min=compactness_min,
        compactness_max=compactness_max,
        compactness_candidates=compactness_candidates,
        coarse_steps=coarse_steps,
        refine_steps=0,
        compactness_search_jobs=compactness_search_jobs,
        compactness_metric_sample_size=compactness_metric_sample_size,
        compactness_search_downsample_factor=compactness_search_downsample_factor,
        compactness_search_raster_mode=compactness_search_raster_mode,
        compactness_search_resolution_cap=compactness_search_resolution_cap,
        verbose=verbose,
        **kwargs,
    )
    if (
        selection_target_segment_um is not None
        and target_segment_um is not None
        and np.isfinite(float(selection_target_segment_um))
        and np.isfinite(float(target_segment_um))
        and float(selection_target_segment_um) > 0
        and not np.isclose(float(selection_target_segment_um), float(target_segment_um))
    ):
        search_selected = float(result["selected_compactness"])
        scaled_selected = schedule_compactness(
            compactness=float(search_selected),
            compactness_mode="scaled",
            target_segment_um=float(target_segment_um),
            compactness_base_size_um=float(selection_target_segment_um),
            compactness_scale_power=float(compactness_scale_power),
            compactness_min=compactness_min,
            compactness_max=compactness_max,
        )
        result = dict(result)
        result["search_target_segment_um"] = float(selection_target_segment_um)
        result["search_selected_compactness"] = float(search_selected)
        result["selected_compactness"] = float(scaled_selected)
        result["target_segment_um"] = float(target_segment_um)
        result["compactness_scale_power"] = float(compactness_scale_power)
        result["scaled_from_search_target_um"] = True
        result["auto_reference_target"] = bool(auto_reference_target)
        if verbose:
            if auto_reference_target:
                print(
                    "[perf] auto-scaled compactness from reference target: "
                    f"search_target_um={float(selection_target_segment_um):.4g} | "
                    f"search_selected={float(search_selected):.6g} | "
                    f"target_um={float(target_segment_um):.4g} | "
                    f"scaled_selected={float(scaled_selected):.6g} | "
                    f"power={float(compactness_scale_power):.4g}"
                )
            else:
                print(
                    "[perf] scaled compactness from reference target: "
                    f"search_target_um={float(selection_target_segment_um):.4g} | "
                    f"search_selected={float(search_selected):.6g} | "
                    f"target_um={float(target_segment_um):.4g} | "
                    f"scaled_selected={float(scaled_selected):.6g} | "
                    f"power={float(compactness_scale_power):.4g}"
                )
    return result


def compactness_search_frame(selection_result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_compactness = selection_result.get(
        "search_selected_compactness",
        selection_result.get("selected_compactness", None),
    )
    for stage in ("coarse_search", "refine_search"):
        detail = selection_result.get(stage, None)
        if not detail:
            continue
        stage_name = str(detail.get("mode", stage))
        stage_selected = detail.get("selected_compactness", None)
        for row in detail.get("candidates", []):
            compactness = float(row["compactness"])
            rows.append(
                {
                    "search_stage": stage_name,
                    "compactness": compactness,
                    "score": row.get("score", None),
                    "compactness_penalty": row.get("compactness_penalty", None),
                    "tradeoff_gain": row.get("tradeoff_gain", None),
                    "fidelity_cost": row.get("fidelity_cost", None),
                    "fidelity_preservation": row.get("fidelity_preservation", None),
                    "regularity_cost": row.get("regularity_cost", None),
                    "regularity_gain": row.get("regularity_gain", None),
                    "embedding_var": row.get("embedding_var", None),
                    "mean_log_aspect": row.get("mean_log_aspect", None),
                    "size_cv": row.get("size_cv", None),
                    "neighbor_disagreement": row.get("neighbor_disagreement", None),
                    "n_segments": row.get("n_segments", None),
                    "is_stage_selected": (
                        stage_selected is not None and np.isclose(compactness, float(stage_selected))
                    ),
                    "is_selected": (
                        selected_compactness is not None and np.isclose(compactness, float(selected_compactness))
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["search_stage", "compactness"], kind="mergesort").reset_index(drop=True)


def evaluate_compactness_sweep(
    adata: AnnData,
    *,
    dimensions: list[int] = list(range(30)),
    search_dimensions: list[int] | None = None,
    embedding: str = "X_embedding",
    resolution: float = 10.0,
    target_segment_um: float | None = None,
    pitch_um: float = 2.0,
    origin: bool = True,
    library_id: str | None = None,
    coordinate_mode: str = "spatial",
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    figsize: tuple | None = None,
    fig_dpi: int | None = None,
    imshow_tile_size: float | None = None,
    imshow_scale_factor: float = 1.0,
    pixel_perfect: bool = False,
    enforce_connectivity: bool = False,
    pixel_shape: str = "square",
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool = True,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool = False,
    center_collision_radius: int = 2,
    mask_background: bool = False,
    compactness_min: float | None = None,
    compactness_max: float | None = None,
    compactness_candidates: list[float] | None = None,
    coarse_steps: int = 5,
    refine_steps: int = 5,
    compactness_search_dims: int = 12,
    compactness_search_jobs: int = 1,
    compactness_metric_sample_size: int | None = 12000,
    compactness_search_downsample_factor: float = 1.0,
    compactness_search_raster_mode: str = "auto",
    use_cached_image: bool = True,
    image_cache_key: str = "image_plot_slic",
    verbose: bool = True,
) -> pd.DataFrame:
    selection_result = select_compactness(
        adata,
        dimensions=dimensions,
        search_dimensions=search_dimensions,
        embedding=embedding,
        resolution=resolution,
        target_segment_um=target_segment_um,
        pitch_um=pitch_um,
        origin=origin,
        mask_background=mask_background,
        library_id=library_id,
        coordinate_mode=coordinate_mode,
        array_row_key=array_row_key,
        array_col_key=array_col_key,
        figsize=figsize,
        fig_dpi=fig_dpi,
        imshow_tile_size=imshow_tile_size,
        imshow_scale_factor=imshow_scale_factor,
        pixel_perfect=pixel_perfect,
        enforce_connectivity=enforce_connectivity,
        pixel_shape=pixel_shape,
        runtime_fill_from_boundary=runtime_fill_from_boundary,
        runtime_fill_closing_radius=runtime_fill_closing_radius,
        runtime_fill_external_radius=runtime_fill_external_radius,
        runtime_fill_holes=runtime_fill_holes,
        soft_rasterization=soft_rasterization,
        resolve_center_collisions=resolve_center_collisions,
        center_collision_radius=center_collision_radius,
        compactness_min=compactness_min,
        compactness_max=compactness_max,
        compactness_candidates=compactness_candidates,
        coarse_steps=coarse_steps,
        refine_steps=refine_steps,
        compactness_search_dims=compactness_search_dims,
        compactness_search_jobs=compactness_search_jobs,
        compactness_metric_sample_size=compactness_metric_sample_size,
        compactness_search_downsample_factor=compactness_search_downsample_factor,
        compactness_search_raster_mode=compactness_search_raster_mode,
        use_cached_image=use_cached_image,
        image_cache_key=image_cache_key,
        verbose=verbose,
    )
    return compactness_search_frame(selection_result)


def plot_compactness_sweep(
    adata: AnnData | None = None,
    *,
    selection_result: dict[str, Any] | None = None,
    dimensions: list[int] = list(range(30)),
    search_dimensions: list[int] | None = None,
    embedding: str = "X_embedding",
    resolution: float = 10.0,
    target_segment_um: float | None = None,
    pitch_um: float = 2.0,
    origin: bool = True,
    library_id: str | None = None,
    coordinate_mode: str = "spatial",
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    figsize: tuple[float, float] = (12, 10),
    fig_dpi: int | None = None,
    imshow_tile_size: float | None = None,
    imshow_scale_factor: float = 1.0,
    pixel_perfect: bool = False,
    enforce_connectivity: bool = False,
    pixel_shape: str = "square",
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool = True,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool = False,
    center_collision_radius: int = 2,
    mask_background: bool = False,
    compactness_min: float | None = None,
    compactness_max: float | None = None,
    compactness_candidates: list[float] | None = None,
    coarse_steps: int = 5,
    refine_steps: int = 5,
    compactness_search_dims: int = 12,
    compactness_search_jobs: int = 1,
    compactness_metric_sample_size: int | None = None,
    compactness_search_downsample_factor: float = 1.0,
    compactness_search_raster_mode: str = "auto",
    use_cached_image: bool = True,
    image_cache_key: str = "image_plot_slic",
    metrics: list[str] | None = None,
    log_x: bool = False,
    show: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    if selection_result is None:
        if adata is None:
            raise ValueError("Either adata or selection_result must be provided.")
        selection_result = select_compactness(
            adata,
            dimensions=dimensions,
            search_dimensions=search_dimensions,
            embedding=embedding,
            resolution=resolution,
            target_segment_um=target_segment_um,
            pitch_um=pitch_um,
            origin=origin,
            mask_background=mask_background,
            library_id=library_id,
            coordinate_mode=coordinate_mode,
            array_row_key=array_row_key,
            array_col_key=array_col_key,
            figsize=None,
            fig_dpi=fig_dpi,
            imshow_tile_size=imshow_tile_size,
            imshow_scale_factor=imshow_scale_factor,
            pixel_perfect=pixel_perfect,
            enforce_connectivity=enforce_connectivity,
            pixel_shape=pixel_shape,
            runtime_fill_from_boundary=runtime_fill_from_boundary,
            runtime_fill_closing_radius=runtime_fill_closing_radius,
            runtime_fill_external_radius=runtime_fill_external_radius,
            runtime_fill_holes=runtime_fill_holes,
            soft_rasterization=soft_rasterization,
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=center_collision_radius,
            compactness_min=compactness_min,
            compactness_max=compactness_max,
            compactness_candidates=compactness_candidates,
            coarse_steps=coarse_steps,
            refine_steps=refine_steps,
            compactness_search_dims=compactness_search_dims,
            compactness_search_jobs=compactness_search_jobs,
            compactness_metric_sample_size=compactness_metric_sample_size,
            compactness_search_downsample_factor=compactness_search_downsample_factor,
            compactness_search_raster_mode=compactness_search_raster_mode,
            use_cached_image=use_cached_image,
            image_cache_key=image_cache_key,
            verbose=verbose,
        )
    frame = compactness_search_frame(selection_result)
    if frame.empty:
        raise ValueError("No compactness sweep candidates are available to plot.")

    import matplotlib.pyplot as plt

    metric_list = metrics or [
        "score",
        "tradeoff_gain",
        "fidelity_preservation",
        "regularity_gain",
        "compactness_penalty",
        "embedding_var",
        "size_cv",
        "mean_log_aspect",
        "neighbor_disagreement",
        "n_segments",
    ]
    metric_list = [metric for metric in metric_list if metric in frame.columns]
    if not metric_list:
        raise ValueError("No valid metrics available to plot.")

    n_panels = len(metric_list)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=fig_dpi, squeeze=False)
    axes_flat = axes.ravel()
    selected_compactness = float(
        selection_result.get(
            "search_selected_compactness",
            selection_result["selected_compactness"],
        )
    )
    final_selected_compactness = float(selection_result["selected_compactness"])
    stage_styles = {
        "coarse_search": {"marker": "o", "linestyle": "-", "linewidth": 1.5},
        "refine_search": {"marker": "s", "linestyle": "--", "linewidth": 1.5},
    }

    for idx, metric in enumerate(metric_list):
        ax = axes_flat[idx]
        for stage_name, stage_df in frame.groupby("search_stage", sort=False):
            stage_df = stage_df.sort_values("compactness", kind="mergesort")
            style = stage_styles.get(stage_name, {"marker": "o", "linestyle": "-", "linewidth": 1.2})
            ax.plot(
                stage_df["compactness"].to_numpy(dtype=float),
                stage_df[metric].to_numpy(dtype=float),
                label=stage_name,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=0.9,
            )
            chosen = stage_df[stage_df["is_stage_selected"]]
            if not chosen.empty:
                ax.scatter(
                    chosen["compactness"].to_numpy(dtype=float),
                    chosen[metric].to_numpy(dtype=float),
                    s=55,
                    color="black",
                    zorder=4,
                )
        ax.axvline(selected_compactness, color="crimson", linestyle=":", linewidth=1.5)
        if bool(log_x):
            ax.set_xscale("log")
        ax.set_xlabel("compactness")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.25, linewidth=0.6)
        if idx == 0:
            ax.legend(frameon=False)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].axis("off")

    title = (
        f"Compactness Sweep | selected={selected_compactness:.4g} | "
        f"requested_n={int(selection_result['requested_n_segments'])}"
    )
    if "search_selected_compactness" in selection_result and not np.isclose(selected_compactness, final_selected_compactness):
        title = (
            f"Compactness Sweep | search_selected={selected_compactness:.4g} | "
            f"final_selected={final_selected_compactness:.4g} | "
            f"requested_n={int(selection_result['requested_n_segments'])}"
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    if show:
        plt.show()
    return {
        "selection": selection_result,
        "frame": frame,
        "figure": fig,
        "axes": axes,
    }


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
    compactness: float | None = None,
    scaling: float = 0.3,
    n_neighbors: int = 15,
    random_state: int = 42,
    Segment: str = "Segment",
    index_selection: str = "random",
    max_iter: int = 1000,
    origin: bool = True,
    verbose: bool = True,
    library_id: str = None,  # Specify library_id column name in adata.obs
    figsize: tuple | None = None,
    fig_dpi: int | None = None,
    imshow_tile_size: float | None = None,
    imshow_scale_factor: float = 1.0,
    pixel_perfect: bool = False,
    enforce_connectivity: bool = False,
    pixel_shape: str = "square",
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool = True,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool = False,
    center_collision_radius: int = 2,
    mask_background: bool = False,
    show_image: bool = False,
    brighten_continuous: bool = False,
    continuous_gamma: float = 0.8,
    target_segment_um: float | None = None,
    pitch_um: float = 2.0,
    n_jobs: int | None = None,
    use_cached_image: bool = False,
    image_cache_key: str = "image_plot_slic",
    cache_fast_path: bool = True,
    compute_pseudo_centroids: bool = True,
    coordinate_mode: str = "spatial",  # 'spatial'|'array'|'visium'|'visiumhd'|'auto'
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    grid_origin_x: float | None = None,
    grid_origin_y: float | None = None,
    compactness_mode: str = "fixed",
    compactness_base_size_um: float | None = None,
    compactness_scale_power: float = 1.0,
    compactness_min: float | None = None,
    compactness_max: float | None = None,
    compactness_search_multipliers = DEFAULT_COMPACTNESS_SEARCH_MULTIPLIERS,
    compactness_search_dims: int = 12,
    compactness_search_jobs: int = 1,
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
        'leiden_slic', 'louvain_slic', 'image_plot_slic', 'fixed_grid'.
    resolution : float, optional (default=10.0)
        Resolution parameter for clustering methods.
    compactness : float or None, optional (default=None)
        Compactness parameter influencing the importance of spatial proximity in
        clustering. When omitted for ``method='image_plot_slic'``, automatic
        compactness search is used.
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
    figsize : tuple or None, optional (default=None)
        Figure size used when creating the intermediate image for ``image_plot_slic``.
    fig_dpi : int or None, optional (default=None)
        DPI used when constructing the internal image for ``image_plot_slic``.
    imshow_tile_size : float or None, optional
        Tile size used for estimating pixel scaling in ``image_plot_slic``.
    imshow_scale_factor : float, optional (default=1.0)
        Additional scaling factor for ``image_plot_slic``.
    pixel_perfect : bool, optional (default ``True``)
        Pixel-perfect rasterization (integer pitch + center-based tiles) used when
        generating the temporary image. When ``use_cached_image=True`` and a cached
        image is available, this flag does not affect segmentation (the cache is used
        as-is).
    enforce_connectivity : bool, optional (default=False)
        Whether to enforce connectivity in ``image_plot_slic``.
    pixel_shape : str, optional (default ``"square"``)
        Shape used for rasterizing spots in ``image_plot_slic``.
    runtime_fill_from_boundary : bool, optional (default ``False``)
        If True, keep the original spot tiles but morphologically fill the
        raster support before ``image_plot_slic`` segmentation.
    runtime_fill_closing_radius : int, optional (default ``1``)
        Pixel-radius used for binary closing of the observed raster support
        when ``runtime_fill_from_boundary=True``.
    runtime_fill_external_radius : int, optional (default ``0``)
        Additional pixel-radius used to slightly expand the recovered support
        outward after internal filling when ``runtime_fill_from_boundary=True``.
    runtime_fill_holes : bool, optional (default ``True``)
        Whether to fill enclosed holes in the raster support when
        ``runtime_fill_from_boundary=True``.
    soft_rasterization : bool, optional (default ``False``)
        If True, use bilinear splatting and bilinear label readback for
        ``image_plot_slic`` instead of hard one-pixel assignment. This is
        intended for dense ``origin=True`` rasterization where many spots snap
        to the same pixel.
    resolve_center_collisions : bool, optional (default ``False``)
        If True, locally reassign duplicate raster centers onto nearby empty
        pixels before ``image_plot_slic`` runs. This is mainly useful for dense
        ``origin=True`` workflows where many spots quantize to the same pixel.
    center_collision_radius : int, optional (default ``2``)
        Search radius in raster pixels used when
        ``resolve_center_collisions=True``.
    show_image : bool, optional (default ``False``)
        If ``True`` display the intermediate images produced by
        ``image_plot_slic_segmentation``.
    brighten_continuous : bool, optional (default ``False``)
        If ``True``, brighten the RGB preview shown when ``show_image=True`` for
        ``method='image_plot_slic'``. This affects preview only, not segmentation.
    continuous_gamma : float, optional (default ``0.8``)
        Gamma used when ``brighten_continuous=True`` for the preview image.
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
        For ``method='fixed_grid'``, this is interpreted as the exact square
        grid side length in micrometers.
    pitch_um : float, optional (default=2.0)
        Distance between neighbouring spots in micrometers used when
        ``target_segment_um`` is specified.
    compactness_mode : {'fixed', 'scaled', 'adaptive_search', 'auto_search'}, optional
        Strategy used to interpret ``compactness`` for ``method='image_plot_slic'``.
    compactness_base_size_um : float or None, optional
        Reference size in micrometers used by ``compactness_mode='scaled'`` or
        ``'adaptive_search'``.
    compactness_scale_power : float, optional (default=1.0)
        Exponent used when scaling compactness by size.
    compactness_min / compactness_max : float or None, optional
        Optional lower/upper bounds applied after compactness scheduling.
    compactness_search_multipliers : sequence of float, optional
        Candidate multipliers around the scheduled compactness used when
        ``compactness_mode='adaptive_search'``.
    compactness_search_dims : int, optional (default=12)
        Number of embedding dimensions used when scoring adaptive compactness
        candidates.
    compactness_search_jobs : int, optional (default=1)
        Number of parallel worker threads used when evaluating compactness
        candidates for ``adaptive_search`` or ``auto_search``.
    coordinate_mode : str, optional (default='spatial')
        Coordinate system used for the internal rasterization in ``image_plot_slic``.
        Use ``'array'`` for Visium/VisiumHD to rasterize on the exact grid
        (uses ``adata.obs[array_row_key/array_col_key]``).
    array_row_key / array_col_key : str
        Column names in ``adata.obs`` used when ``coordinate_mode='array'``.
    grid_origin_x / grid_origin_y : float or None, optional
        Optional explicit grid origin for ``method='fixed_grid'``.
        When omitted, the minimum observed x/y coordinates are used.

    Returns
    -------
    None
        Updates the AnnData object with segmentation information.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Retrieve tiles from adata.uns
    tiles_all = adata.uns["tiles"]
    inferred_info = adata.uns.get("tiles_inferred_grid_info", {})
    inferred_mode = str(inferred_info.get("mode", "")).lower()
    if origin:
        tiles = tiles_all[tiles_all.get("origin", 1) == 1]
    else:
        if inferred_mode == "boundary_voronoi" and ("origin" in tiles_all.columns):
            tiles = tiles_all[tiles_all["origin"] == 0]
            if verbose:
                print("[info] Using inferred boundary support-grid tiles only for origin=False segmentation.")
        else:
            tiles = tiles_all
    if "barcode" in tiles.columns:
        tiles = tiles.copy()
        tiles["barcode"] = tiles["barcode"].astype(str)
        if tiles["barcode"].duplicated().any():
            origin_vals = pd.to_numeric(tiles.get("origin", 1), errors="coerce").fillna(1)
            has_non_origin = (origin_vals != 1).any()
            keep_dups = (not origin) and has_non_origin
            if keep_dups:
                if verbose:
                    print(
                        "[info] adata.uns['tiles'] has duplicated barcodes (expected for origin=False raster tiles); keeping duplicates."
                    )
            else:
                if verbose:
                    print(
                        "[warning] adata.uns['tiles'] has duplicated barcodes; keeping first occurrence for segmentation."
                    )
                tiles = tiles.drop_duplicates("barcode", keep="first")

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

    if target_segment_um is not None and method != "fixed_grid":
        resolution = n_segments_from_size(
            tiles_for_n, target_um=target_segment_um, pitch_um=pitch_um
        )
        if verbose:
            print(f"→ SLIC  n_segments = {resolution}")
    elif target_segment_um is not None and method == "fixed_grid":
        resolution = float(target_segment_um)
        if verbose:
            print(f"→ Fixed grid side_um = {resolution}")

    resolved_compactness_mode = str(compactness_mode or "fixed")
    requested_compactness = None if compactness is None else float(compactness)
    if (
        method == "image_plot_slic"
        and requested_compactness is None
        and resolved_compactness_mode == "fixed"
    ):
        resolved_compactness_mode = "auto_search"
        if verbose:
            print("[info] compactness was omitted; using compactness_mode='auto_search'.")
    base_compactness = 1.0 if requested_compactness is None else float(requested_compactness)

    if method == "fixed_grid" and verbose:
        print(
            "[info] fixed_grid assigns exact square cells from coordinates only; "
            "embedding/dimensions are used only for pseudo-centroids. "
            "pitch_um, compactness, enforce_connectivity, show_image, and use_cached_image are ignored."
        )

    if method != "image_plot_slic" and resolved_compactness_mode != "fixed" and verbose:
        print(
            f"[info] compactness_mode='{resolved_compactness_mode}' is only applied to method='image_plot_slic'; "
            f"using raw compactness for method='{method}'."
        )

    compactness_metadata = None

    # Fast path: cached image SLIC without building coordinates_df/embeddings_df
    if (
        method == "image_plot_slic"
        and use_cached_image
        and library_id is None
        and cache_fast_path
        and image_cache_key in adata.uns
        and resolved_compactness_mode not in {"adaptive_search", "auto_search"}
    ):
        cache = adata.uns[image_cache_key]
        cached_emb = cache.get("embedding_key", None)
        cached_dims = cache.get("dimensions", None)
        cached_origin = cache.get("origin", None)
        cache_matches_config, cache_mismatches = _cache_matches_segmentation_raster_config(
            adata,
            cache,
            coordinate_mode=coordinate_mode,
            origin=origin,
            array_row_key=array_row_key,
            array_col_key=array_col_key,
            pixel_perfect=pixel_perfect,
            pixel_shape=pixel_shape,
            runtime_fill_from_boundary=runtime_fill_from_boundary,
            runtime_fill_closing_radius=runtime_fill_closing_radius,
            runtime_fill_external_radius=runtime_fill_external_radius,
            runtime_fill_holes=runtime_fill_holes,
            soft_rasterization=soft_rasterization,
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=center_collision_radius,
        )
        if (cached_emb is None or cached_emb == embedding) and (
            cached_dims is None or list(cached_dims) == list(dimensions)
        ) and (
            cached_origin is None or bool(cached_origin) == bool(origin)
        ) and cache_matches_config:
            if show_image:
                preview_figsize, preview_figdpi = _resolve_cached_preview_size(
                    cache,
                    figsize=figsize,
                    fig_dpi=fig_dpi,
                    img_shape=np.asarray(cache.get("img")).shape,
                )
                cache_pp = cache.get("pixel_perfect", None)
                if verbose:
                    print(
                        f"[info] use_cached_image=True: preview figsize={preview_figsize}, fig_dpi={preview_figdpi}, pixel_perfect={cache_pp}"
                    )

            compactness_value, compactness_detail = resolve_compactness(
                compactness=requested_compactness,
                compactness_mode=resolved_compactness_mode,
                target_segment_um=float(target_segment_um) if target_segment_um is not None else None,
                compactness_base_size_um=compactness_base_size_um,
                compactness_scale_power=float(compactness_scale_power),
                compactness_min=compactness_min,
                compactness_max=compactness_max,
                compactness_search_multipliers=compactness_search_multipliers,
                compactness_search_dims=int(compactness_search_dims),
                compactness_search_jobs=int(compactness_search_jobs),
                embeddings=None,
                spatial_coords=np.zeros((int(adata.n_obs), 2), dtype=float),
                segmenter=None,
            )

            t0 = time.perf_counter()
            labels_tile, label_img, raster_meta = slic_segmentation_from_cached_image(
                cache,
                n_segments=int(resolution),
                compactness=compactness_value,
                enforce_connectivity=enforce_connectivity,
                show_image=show_image,
                verbose=verbose,
                figsize=figsize,
                fig_dpi=fig_dpi,
                brighten_continuous=brighten_continuous,
                continuous_gamma=continuous_gamma,
                mask_background=mask_background,
                return_raster_meta=True,
            )
            t1 = time.perf_counter()

            # Build per-observation segment codes without allocating an object array.
            # Use -1 for missing values; `pd.Categorical.from_codes` treats -1 as NaN.
            n_obs = int(adata.n_obs)
            seg_codes = np.full(n_obs, -1, dtype=np.int32)
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
                    seg_codes[idx] = tile_labels.astype(np.int32, copy=False)
                else:
                    agg_idx, agg_labels = _aggregate_labels_majority_by_obs_index(
                        idx,
                        tile_labels,
                        origin_flags=cache.get("origin_flags", None),
                    )
                    seg_codes[agg_idx] = np.asarray(agg_labels, dtype=np.int32)
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
                seg_codes[obs_pos[ok]] = np.asarray(tile_labels[ok], dtype=np.int32)

            # Categories are integer labels. Some SLIC implementations can return
            # labels that are not exactly 0..n_segments-1, so derive the upper
            # bound from observed labels to keep codes valid.
            max_code = int(seg_codes.max(initial=-1))
            n_segments = int(max_code + 1) if max_code >= 0 else 0
            categories = np.arange(n_segments, dtype=np.int32)
            adata.obs[Segment] = pd.Categorical.from_codes(seg_codes, categories=categories)
            _store_segment_pixel_boundary_payload(
                adata,
                segment_name=Segment,
                label_img=label_img,
                raster_meta=raster_meta,
                origin=origin,
                coordinate_mode=coordinate_mode,
                array_row_key=array_row_key,
                array_col_key=array_col_key,
            )

            if not compute_pseudo_centroids:
                if verbose:
                    print(f"[perf] cached SLIC (fast path): {t1 - t0:.3f}s")
                return

            # Compute pseudo-centroids in obs order
            E_full = np.asarray(adata.obsm[embedding])
            if E_full.shape[1] == len(dimensions):
                E_use = E_full
            else:
                E_use = E_full[:, dimensions]
            # Convert codes -> per-observation pseudo-centroid vectors.
            mask = seg_codes >= 0
            if not mask.any():
                pseudo_centroids = np.full((n_obs, E_use.shape[1]), np.nan, dtype=np.float32)
            else:
                E = np.asarray(E_use[mask], dtype=np.float64)
                codes = seg_codes[mask].astype(np.int64, copy=False)
                sums = np.zeros((n_segments, E.shape[1]), dtype=np.float64)
                counts = np.bincount(codes, minlength=n_segments).astype(np.float64)
                for j in range(E.shape[1]):
                    np.add.at(sums[:, j], codes, E[:, j])
                denom = counts.copy()
                denom[denom < 1.0] = np.nan
                means = (sums / denom[:, None]).astype(np.float32)
                pseudo_centroids = np.full((n_obs, E_use.shape[1]), np.nan, dtype=np.float32)
                pseudo_centroids[mask] = means[codes]
            segment_suffix = _sanitize_segment_name(Segment)
            adata.obsm["X_embedding_segment"] = pseudo_centroids
            adata.obsm[f"X_embedding_{segment_suffix}"] = pseudo_centroids
            compactness_metadata = {
                **_compactness_mode_header(
                    compactness=requested_compactness,
                    compactness_mode=resolved_compactness_mode,
                    target_segment_um=float(target_segment_um) if target_segment_um is not None else None,
                    compactness_base_size_um=compactness_base_size_um,
                    compactness_scale_power=float(compactness_scale_power),
                    compactness_min=compactness_min,
                    compactness_max=compactness_max,
                    compactness_search_multipliers=compactness_search_multipliers,
                    compactness_search_dims=int(compactness_search_dims),
                    compactness_search_jobs=int(compactness_search_jobs),
                ),
                "selection": compactness_detail,
            }
            adata.uns[f"{segment_suffix}_compactness"] = compactness_metadata

            # No combined_data for image_plot_slic
            for key in ("X_embedding_scaled_for_segment", f"X_embedding_scaled_for_{segment_suffix}"):
                if key in adata.obsm:
                    del adata.obsm[key]

            if verbose:
                print(f"[perf] cached SLIC (fast path): {t1 - t0:.3f}s")
            return
        elif verbose and (
            (cached_emb is None or cached_emb == embedding)
            and (cached_dims is None or list(cached_dims) == list(dimensions))
            and (cached_origin is None or bool(cached_origin) == bool(origin))
            and (not cache_matches_config)
        ):
            print(
                "[warning] Cached image raster settings differ from requested segmentation settings; "
                f"regenerating image. mismatches={cache_mismatches}"
            )
        elif cached_origin is not None and bool(cached_origin) != bool(origin) and verbose:
            print(
                f"[warning] Cached image origin={bool(cached_origin)} differs from requested origin={bool(origin)}; regenerating image."
            )

    # Fast path: exact fixed-grid labels do not require merge-heavy DataFrame construction.
    if method == "fixed_grid" and library_id is None:
        coord_mode = str(coordinate_mode or "spatial").lower()
        if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
            coord_mode = "spatial"
        if coord_mode == "auto":
            if origin and (array_row_key in adata.obs.columns) and (array_col_key in adata.obs.columns) and int(adata.n_obs) >= 200_000:
                coord_mode = "visiumhd"
            else:
                coord_mode = "spatial"

        tile_barcodes = pd.Series(tiles.get("barcode", []), dtype=str).to_numpy()
        if tile_barcodes.size == 0:
            raise ValueError("adata.uns['tiles'] must contain a non-empty 'barcode' column for fixed_grid segmentation.")

        obs_barcodes = pd.Index(adata.obs.index.astype(str))
        tile_obs_idx = obs_barcodes.get_indexer(pd.Index(tile_barcodes))
        obs_ok = tile_obs_idx >= 0

        x = np.full(tile_barcodes.size, np.nan, dtype=float)
        y = np.full(tile_barcodes.size, np.nan, dtype=float)
        if coord_mode in {"array", "visium", "visiumhd"}:
            if (array_row_key not in adata.obs.columns) or (array_col_key not in adata.obs.columns):
                raise ValueError(
                    f"coordinate_mode='array' requires adata.obs['{array_row_key}'] and adata.obs['{array_col_key}']."
                )
            obs_col = pd.to_numeric(adata.obs[array_col_key], errors="coerce").to_numpy(dtype=float, copy=False)
            obs_row = pd.to_numeric(adata.obs[array_row_key], errors="coerce").to_numpy(dtype=float, copy=False)
            valid_obs_idx = tile_obs_idx[obs_ok]
            if coord_mode == "visium":
                row_vals = obs_row[valid_obs_idx]
                col_vals = obs_col[valid_obs_idx]
                parity = np.mod(row_vals, 2.0)
                x[obs_ok] = col_vals + 0.5 * parity
                y[obs_ok] = row_vals * (np.sqrt(3.0) / 2.0)
            else:
                x[obs_ok] = obs_col[valid_obs_idx]
                y[obs_ok] = obs_row[valid_obs_idx]
        else:
            if ("x" not in tiles.columns) or ("y" not in tiles.columns):
                raise ValueError("adata.uns['tiles'] must contain numeric 'x' and 'y' columns for fixed_grid segmentation.")
            x = pd.to_numeric(tiles["x"], errors="coerce").to_numpy(dtype=float, copy=False)
            y = pd.to_numeric(tiles["y"], errors="coerce").to_numpy(dtype=float, copy=False)

        E_full = np.asarray(adata.obsm[embedding])
        if E_full.ndim != 2 or E_full.shape[0] != adata.n_obs:
            raise ValueError(f"Embedding '{embedding}' must be a 2D array with n_obs rows.")
        E_use = E_full if E_full.shape[1] == len(dimensions) else E_full[:, dimensions]

        emb_ok = np.zeros(tile_barcodes.size, dtype=bool)
        if np.any(obs_ok):
            emb_ok[obs_ok] = np.isfinite(np.asarray(E_use[tile_obs_idx[obs_ok]], dtype=np.float32)).all(axis=1)

        tile_valid = obs_ok & np.isfinite(x) & np.isfinite(y) & emb_ok
        coords_valid = np.column_stack([x[tile_valid], y[tile_valid]])

        t0 = time.perf_counter()
        tile_labels, grid_meta = fixed_grid_segmentation(
            coords_valid,
            side_um=float(resolution),
            origin_x=grid_origin_x,
            origin_y=grid_origin_y,
        )
        t1 = time.perf_counter()

        valid_obs_idx = tile_obs_idx[tile_valid].astype(np.int64, copy=False)
        origin_flags = np.asarray(
            tiles.get("origin", pd.Series(index=tiles.index, data=0)),
            dtype=np.int8,
        )[tile_valid]

        seg_full = np.full(adata.n_obs, np.nan, dtype=object)
        if valid_obs_idx.size:
            if np.unique(valid_obs_idx).size == valid_obs_idx.size:
                seg_full[valid_obs_idx] = np.asarray(tile_labels, dtype=object)
            else:
                label_codes, uniques = pd.factorize(np.asarray(tile_labels, dtype=object), sort=False)
                agg_idx, agg_codes = _aggregate_labels_majority_by_obs_index(
                    valid_obs_idx,
                    label_codes.astype(np.int64, copy=False),
                    origin_flags=origin_flags,
                )
                seg_full[agg_idx] = uniques[np.asarray(agg_codes, dtype=np.int64)]
        t2 = time.perf_counter()

        adata.obs[Segment] = pd.Categorical(seg_full)

        segment_suffix = _sanitize_segment_name(Segment)
        if compute_pseudo_centroids:
            pseudo_centroids = _pseudo_centroids_from_segment_labels(
                np.asarray(E_use, dtype=np.float32),
                seg_full,
            )
            adata.obsm["X_embedding_segment"] = pseudo_centroids
            adata.obsm[f"X_embedding_{segment_suffix}"] = pseudo_centroids
        t3 = time.perf_counter()

        for key in ("X_embedding_scaled_for_segment", f"X_embedding_scaled_for_{segment_suffix}"):
            if key in adata.obsm:
                del adata.obsm[key]

        fixed_grid_uns = dict(grid_meta)
        fixed_grid_uns["origin"] = bool(origin)
        adata.uns[f"{segment_suffix}_fixed_grid"] = fixed_grid_uns

        if verbose:
            print(
                f"[perf] fixed_grid fast path: assign {t1 - t0:.3f}s | "
                f"aggregate {t2 - t1:.3f}s | pseudo {t3 - t2:.3f}s"
            )
        return

    # Create a DataFrame from the selected embedding dimensions and add barcode from adata.obs index
    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors["barcode"] = adata.obs.index.astype(str)
    tile_colors["obs_index"] = np.arange(adata.n_obs, dtype=np.int64)
    obs_barcodes_all = adata.obs.index.astype(str).to_numpy()
    # Base per-barcode embeddings for centroid computation (unique index)
    base_embeddings_df = tile_colors.drop(columns=["barcode", "obs_index"]).copy()
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

    coord_mode = str(coordinate_mode or "spatial").lower()
    if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
        coord_mode = "spatial"
    if coord_mode == "auto":
        if origin and (array_row_key in adata.obs.columns) and (array_col_key in adata.obs.columns) and int(adata.n_obs) >= 200_000:
            coord_mode = "visiumhd"
        else:
            coord_mode = "spatial"
    if coord_mode in {"array", "visium", "visiumhd"}:
        if (array_row_key not in adata.obs.columns) or (array_col_key not in adata.obs.columns):
            raise ValueError(
                f"coordinate_mode='array' requires adata.obs['{array_row_key}'] and adata.obs['{array_col_key}']."
            )
        col = pd.to_numeric(coordinates_df["barcode"].map(adata.obs[array_col_key]), errors="coerce")
        row = pd.to_numeric(coordinates_df["barcode"].map(adata.obs[array_row_key]), errors="coerce")
        if coord_mode == "visium":
            row_i = row.astype("Int64")
            parity = (row_i % 2).astype(float)
            coordinates_df["x"] = col.astype(float) + 0.5 * parity
            coordinates_df["y"] = row.astype(float) * (np.sqrt(3.0) / 2.0)
        else:
            coordinates_df["x"] = col.astype(float)
            coordinates_df["y"] = row.astype(float)
        coordinates_df = coordinates_df.dropna(subset=["x", "y"])

    # If library_id is specified, perform segmentation per library group
    if library_id is not None:
        clusters_list = []
        pseudo_centroids_list = []
        combined_data_list = []
        compactness_details_by_library = {}
        # Group by the specified library_id column
        # Pass observed=False to silence FutureWarning (or observed=True to adopt future behavior)
        for lib, group in coordinates_df.groupby(library_id, observed=False):
            if verbose:
                print(f"Segmenting library_id: {lib} ...")
            # Extract spatial coordinates
            spatial_coords_group = group[["x", "y"]].values
            # Drop columns that are not part of the embedding
            drop_cols = ["barcode", "obs_index", "x", "y", "origin", library_id]
            drop_cols = [col for col in drop_cols if col in group.columns]
            embeddings_group = group.drop(columns=drop_cols).values
            embeddings_df_group = group.drop(columns=drop_cols)
            embeddings_df_group.index = group["barcode"]
            barcode_group = group["barcode"]

            if method == "image_plot_slic":
                if use_cached_image and verbose:
                    print(
                        "[warning] use_cached_image=True ignored when library_id is set; regenerating per-library image."
                    )
                compactness_value, compactness_detail = resolve_compactness(
                    compactness=requested_compactness,
                    compactness_mode=resolved_compactness_mode,
                    target_segment_um=float(target_segment_um) if target_segment_um is not None else None,
                    compactness_base_size_um=compactness_base_size_um,
                    compactness_scale_power=float(compactness_scale_power),
                    compactness_min=compactness_min,
                    compactness_max=compactness_max,
                    compactness_search_multipliers=compactness_search_multipliers,
                    compactness_search_dims=int(compactness_search_dims),
                    compactness_search_jobs=int(compactness_search_jobs),
                    embeddings=np.asarray(embeddings_group, dtype=float),
                    spatial_coords=np.asarray(spatial_coords_group, dtype=float),
                    segmenter=lambda candidate: image_plot_slic_segmentation(
                        embeddings_group,
                        spatial_coords_group,
                        n_segments=int(resolution),
                        compactness=float(candidate),
                        n_points_eff=int(spatial_coords_group.shape[0]),
                        figsize=figsize,
                        fig_dpi=fig_dpi,
                        imshow_tile_size=imshow_tile_size,
                        imshow_scale_factor=imshow_scale_factor,
                        pixel_perfect=pixel_perfect,
                        enforce_connectivity=enforce_connectivity,
                        pixel_shape=pixel_shape,
                        runtime_fill_from_boundary=runtime_fill_from_boundary,
                        runtime_fill_closing_radius=runtime_fill_closing_radius,
                        runtime_fill_external_radius=runtime_fill_external_radius,
                        runtime_fill_holes=runtime_fill_holes,
                        soft_rasterization=soft_rasterization,
                        resolve_center_collisions=resolve_center_collisions,
                        center_collision_radius=center_collision_radius,
                        show_image=False,
                        verbose=False,
                        n_jobs=n_jobs,
                        mask_background=mask_background,
                    )[0],
                )
                labels, _, _ = image_plot_slic_segmentation(
                    embeddings_group,
                    spatial_coords_group,
                    n_segments=int(resolution),
                    compactness=compactness_value,
                    n_points_eff=int(spatial_coords_group.shape[0]),
                    figsize=figsize,
                    fig_dpi=fig_dpi,
                    imshow_tile_size=imshow_tile_size,
                    imshow_scale_factor=imshow_scale_factor,
                    pixel_perfect=pixel_perfect,
                    enforce_connectivity=enforce_connectivity,
                    pixel_shape=pixel_shape,
                    runtime_fill_from_boundary=runtime_fill_from_boundary,
                    runtime_fill_closing_radius=runtime_fill_closing_radius,
                    runtime_fill_external_radius=runtime_fill_external_radius,
                    runtime_fill_holes=runtime_fill_holes,
                    soft_rasterization=soft_rasterization,
                    resolve_center_collisions=resolve_center_collisions,
                    center_collision_radius=center_collision_radius,
                    show_image=show_image,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    mask_background=mask_background,
                )
                compactness_details_by_library[str(lib)] = compactness_detail
                # Aggregate tile labels -> per-observation labels (use origin flags for tie-break)
                group_obs_idx = group["obs_index"].to_numpy(dtype=np.int64, copy=False)
                if np.unique(group_obs_idx).size == group_obs_idx.size:
                    clusters_df_group = pd.DataFrame({
                        "barcode": obs_barcodes_all[group_obs_idx],
                        "Segment": np.asarray(labels),
                    })
                else:
                    agg_idx, agg_labels = _aggregate_labels_majority_by_obs_index(
                        group_obs_idx,
                        labels,
                        origin_flags=group.get("origin", pd.Series(index=group.index, data=0)).values,
                    )
                    clusters_df_group = pd.DataFrame({
                        "barcode": obs_barcodes_all[agg_idx],
                        "Segment": agg_labels,
                    })
                # Use per-barcode embeddings to compute pseudo-centroids
                emb_base_group = base_embeddings_df.loc[clusters_df_group["barcode"].astype(str)].copy()
                pseudo_centroids_group = create_pseudo_centroids(
                    emb_base_group,
                    clusters_df_group,
                    emb_base_group.columns,
                )
                combined_data_group = None
            else:
                clusters_df_group, pseudo_centroids_group, combined_data_group = segment_image_inner(
                    embeddings_group,
                    embeddings_df_group,
                    barcode_group,
                    spatial_coords_group,
                    dimensions,
                    method,
                    resolution,
                    base_compactness,
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
                    pixel_perfect,
                    enforce_connectivity,
                    pixel_shape,
                    show_image,
                    fig_dpi=fig_dpi,
                    grid_origin_x=grid_origin_x,
                    grid_origin_y=grid_origin_y,
                )
                # If duplicates (origin=False), aggregate to one label per barcode and recompute centroids
                if clusters_df_group["barcode"].duplicated().any():
                    clusters_df_group = _aggregate_labels_majority(
                        clusters_df_group["barcode"],
                        clusters_df_group["Segment"].values,
                        origin_flags=group.get("origin", pd.Series(index=group.index, data=0)).values,
                    )
                    emb_base_group = base_embeddings_df.loc[clusters_df_group["barcode"].astype(str)].copy()
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
        if method == "image_plot_slic":
            segment_suffix = _sanitize_segment_name(Segment)
            compactness_metadata = {
                **_compactness_mode_header(
                    compactness=requested_compactness,
                    compactness_mode=resolved_compactness_mode,
                    target_segment_um=float(target_segment_um) if target_segment_um is not None else None,
                    compactness_base_size_um=compactness_base_size_um,
                    compactness_scale_power=float(compactness_scale_power),
                    compactness_min=compactness_min,
                    compactness_max=compactness_max,
                    compactness_search_multipliers=compactness_search_multipliers,
                    compactness_search_dims=int(compactness_search_dims),
                    compactness_search_jobs=int(compactness_search_jobs),
                ),
                "by_library": compactness_details_by_library,
            }

    else:
        # No library_id specified, process all data together
        spatial_coords = coordinates_df.loc[:, ["x", "y"]].values
        embeddings = coordinates_df.drop(
            columns=["barcode", "obs_index", "x", "y", "origin"]
        ).values  # Convert to NumPy array
        embeddings_df = coordinates_df.drop(columns=["barcode", "obs_index", "x", "y", "origin"])
        embeddings_df.index = coordinates_df["barcode"]
        barcode = coordinates_df["barcode"]

        if method == "image_plot_slic":
            compactness_detail = None
            transient_cache_key = None
            cache = None
            if (
                not use_cached_image
                and resolved_compactness_mode in {"adaptive_search", "auto_search"}
            ):
                segment_suffix = _sanitize_segment_name(Segment)
                transient_cache_key = f"__spix_tmp_cache_{segment_suffix}"
                cache = cache_embedding_image(
                    adata,
                    embedding=embedding,
                    dimensions=list(dimensions),
                    origin=origin,
                    key=transient_cache_key,
                    coordinate_mode=coordinate_mode,
                    array_row_key=array_row_key,
                    array_col_key=array_col_key,
                    figsize=figsize,
                    fig_dpi=fig_dpi,
                    imshow_tile_size=imshow_tile_size,
                    imshow_scale_factor=imshow_scale_factor,
                    pixel_perfect=pixel_perfect,
                    pixel_shape=pixel_shape,
                    verbose=False,
                    show=False,
                    runtime_fill_from_boundary=runtime_fill_from_boundary,
                    runtime_fill_closing_radius=runtime_fill_closing_radius,
                    runtime_fill_external_radius=runtime_fill_external_radius,
                    runtime_fill_holes=runtime_fill_holes,
                    soft_rasterization=soft_rasterization,
                    resolve_center_collisions=resolve_center_collisions,
                    center_collision_radius=center_collision_radius,
                )
                if verbose:
                    print("[perf] built transient raster cache for compactness search.")
            elif use_cached_image and image_cache_key in adata.uns:
                cache = adata.uns[image_cache_key]

            if cache is not None:
                cached_emb = cache.get("embedding_key", None)
                cached_dims = cache.get("dimensions", None)
                cached_origin = cache.get("origin", None)
                cache_matches_config, cache_mismatches = _cache_matches_segmentation_raster_config(
                    adata,
                    cache,
                    coordinate_mode=coordinate_mode,
                    origin=origin,
                    array_row_key=array_row_key,
                    array_col_key=array_col_key,
                    pixel_perfect=pixel_perfect,
                    pixel_shape=pixel_shape,
                    runtime_fill_from_boundary=runtime_fill_from_boundary,
                    runtime_fill_closing_radius=runtime_fill_closing_radius,
                    runtime_fill_external_radius=runtime_fill_external_radius,
                    runtime_fill_holes=runtime_fill_holes,
                    soft_rasterization=soft_rasterization,
                    resolve_center_collisions=resolve_center_collisions,
                    center_collision_radius=center_collision_radius,
                )
                try:
                    if (cached_emb is not None and cached_emb != embedding) or (
                        cached_dims is not None and list(cached_dims) != list(dimensions)
                    ):
                        if verbose:
                            print("[warning] Cached image embedding/dimensions differ; regenerating image.")
                        raise KeyError
                    if (cached_origin is not None) and (bool(cached_origin) != bool(origin)):
                        if verbose:
                            print(
                                f"[warning] Cached image origin={bool(cached_origin)} differs from requested origin={bool(origin)}; regenerating image."
                            )
                        raise KeyError
                    if not cache_matches_config:
                        if verbose:
                            print(
                                "[warning] Cached image raster settings differ from requested segmentation settings; "
                                f"regenerating image. mismatches={cache_mismatches}"
                            )
                        raise KeyError

                    if show_image:
                        preview_figsize, preview_figdpi = _resolve_cached_preview_size(
                            cache,
                            figsize=figsize,
                            fig_dpi=fig_dpi,
                            img_shape=np.asarray(cache.get("img")).shape,
                        )
                        cache_pp = cache.get("pixel_perfect", None)
                        if verbose:
                            print(
                                f"[info] use_cached_image=True: preview figsize={preview_figsize}, fig_dpi={preview_figdpi}, pixel_perfect={cache_pp}"
                            )

                    compactness_value, compactness_detail = resolve_compactness(
                        compactness=requested_compactness,
                        compactness_mode=resolved_compactness_mode,
                        target_segment_um=float(target_segment_um) if target_segment_um is not None else None,
                        compactness_base_size_um=compactness_base_size_um,
                        compactness_scale_power=float(compactness_scale_power),
                        compactness_min=compactness_min,
                        compactness_max=compactness_max,
                        compactness_search_multipliers=compactness_search_multipliers,
                        compactness_search_dims=int(compactness_search_dims),
                        compactness_search_jobs=int(compactness_search_jobs),
                        embeddings=np.asarray(embeddings, dtype=float),
                        spatial_coords=np.asarray(spatial_coords, dtype=float),
                        segmenter=lambda candidate: slic_segmentation_from_cached_image(
                            cache,
                            n_segments=int(resolution),
                            compactness=float(candidate),
                            enforce_connectivity=enforce_connectivity,
                            show_image=False,
                            verbose=False,
                            figsize=None,
                            fig_dpi=None,
                            mask_background=mask_background,
                        )[0],
                    )

                    t0 = time.perf_counter()
                    labels_tile, label_img, raster_meta = slic_segmentation_from_cached_image(
                        cache,
                        n_segments=int(resolution),
                        compactness=compactness_value,
                        enforce_connectivity=enforce_connectivity,
                        show_image=show_image,
                        verbose=verbose,
                        figsize=figsize,
                        fig_dpi=fig_dpi,
                        brighten_continuous=brighten_continuous,
                        continuous_gamma=continuous_gamma,
                        mask_background=mask_background,
                        return_raster_meta=True,
                    )
                    t1 = time.perf_counter()
                    tile_obs_indices = cache.get("tile_obs_indices", None)
                    if tile_obs_indices is not None:
                        idx = np.asarray(tile_obs_indices, dtype=np.int64)
                        tile_labels = np.asarray(labels_tile)
                        if idx.ndim != 1 or idx.size != tile_labels.size:
                            raise ValueError(
                                f"Cached image '{transient_cache_key or image_cache_key}' has tile_obs_indices of length {idx.size}, "
                                f"but got {tile_labels.size} tile labels."
                            )
                        obs_barcodes = adata.obs.index.astype(str).to_numpy()
                        if np.unique(idx).size == idx.size:
                            clusters_df = pd.DataFrame({
                                "barcode": obs_barcodes[idx],
                                "Segment": tile_labels,
                            })
                        else:
                            agg_idx, agg_labels = _aggregate_labels_majority_by_obs_index(
                                idx,
                                tile_labels,
                                origin_flags=cache.get("origin_flags", None),
                            )
                            clusters_df = pd.DataFrame({
                                "barcode": obs_barcodes[agg_idx],
                                "Segment": agg_labels,
                            })
                    else:
                        cache_barcodes = pd.Series(cache.get("barcodes", []), dtype=str)
                        if cache_barcodes.empty:
                            raise ValueError(
                                f"Cached image '{transient_cache_key or image_cache_key}' missing both tile_obs_indices and barcodes."
                            )
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
                    pseudo_centroids = _pseudo_centroids_from_clusters_df(
                        adata,
                        clusters_df,
                        embedding=embedding,
                        dimensions=list(dimensions),
                    )
                    t3 = time.perf_counter()
                    if verbose:
                        print(f"[perf] cached SLIC: {t1 - t0:.3f}s | aggregate: {t2 - t1:.3f}s | pseudo: {t3 - t2:.3f}s")
                    _store_segment_pixel_boundary_payload(
                        adata,
                        segment_name=Segment,
                        label_img=label_img,
                        raster_meta=raster_meta,
                        origin=origin,
                        coordinate_mode=coordinate_mode,
                        array_row_key=array_row_key,
                        array_col_key=array_col_key,
                    )
                    combined_data = None
                except Exception:
                    # Fallback to regeneration
                    compactness_value, compactness_detail = resolve_compactness(
                        compactness=requested_compactness,
                        compactness_mode=resolved_compactness_mode,
                        target_segment_um=float(target_segment_um) if target_segment_um is not None else None,
                        compactness_base_size_um=compactness_base_size_um,
                        compactness_scale_power=float(compactness_scale_power),
                        compactness_min=compactness_min,
                        compactness_max=compactness_max,
                        compactness_search_multipliers=compactness_search_multipliers,
                        compactness_search_dims=int(compactness_search_dims),
                        compactness_search_jobs=int(compactness_search_jobs),
                        embeddings=np.asarray(embeddings, dtype=float),
                        spatial_coords=np.asarray(spatial_coords, dtype=float),
                        segmenter=lambda candidate: image_plot_slic_segmentation(
                            embeddings,
                            spatial_coords,
                            n_segments=int(resolution),
                            compactness=float(candidate),
                            n_points_eff=int(spatial_coords.shape[0]),
                            figsize=figsize,
                            fig_dpi=fig_dpi,
                            imshow_tile_size=imshow_tile_size,
                            imshow_scale_factor=imshow_scale_factor,
                            pixel_perfect=pixel_perfect,
                            enforce_connectivity=enforce_connectivity,
                            pixel_shape=pixel_shape,
                            runtime_fill_from_boundary=runtime_fill_from_boundary,
                            runtime_fill_closing_radius=runtime_fill_closing_radius,
                            runtime_fill_external_radius=runtime_fill_external_radius,
                            runtime_fill_holes=runtime_fill_holes,
                            soft_rasterization=soft_rasterization,
                            resolve_center_collisions=resolve_center_collisions,
                            center_collision_radius=center_collision_radius,
                            show_image=False,
                            verbose=False,
                            n_jobs=n_jobs,
                            mask_background=mask_background,
                        )[0],
                    )
                    t0 = time.perf_counter()
                    labels, label_img, _, raster_meta = image_plot_slic_segmentation(
                        embeddings,
                        spatial_coords,
                        n_segments=int(resolution),
                        compactness=compactness_value,
                        n_points_eff=int(spatial_coords.shape[0]),
                        figsize=figsize,
                        fig_dpi=fig_dpi,
                        imshow_tile_size=imshow_tile_size,
                        imshow_scale_factor=imshow_scale_factor,
                        pixel_perfect=pixel_perfect,
                        enforce_connectivity=enforce_connectivity,
                        pixel_shape=pixel_shape,
                        runtime_fill_from_boundary=runtime_fill_from_boundary,
                        runtime_fill_closing_radius=runtime_fill_closing_radius,
                        runtime_fill_external_radius=runtime_fill_external_radius,
                        runtime_fill_holes=runtime_fill_holes,
                        soft_rasterization=soft_rasterization,
                        resolve_center_collisions=resolve_center_collisions,
                        center_collision_radius=center_collision_radius,
                        show_image=show_image,
                        verbose=verbose,
                        n_jobs=n_jobs,
                        brighten_continuous=brighten_continuous,
                        continuous_gamma=continuous_gamma,
                        mask_background=mask_background,
                        return_raster_meta=True,
                    )
                    t1 = time.perf_counter()
                    tile_obs_idx = coordinates_df["obs_index"].to_numpy(dtype=np.int64, copy=False)
                    if np.unique(tile_obs_idx).size == tile_obs_idx.size:
                        clusters_df = pd.DataFrame({
                            "barcode": obs_barcodes_all[tile_obs_idx],
                            "Segment": np.asarray(labels),
                        })
                    else:
                        agg_idx, agg_labels = _aggregate_labels_majority_by_obs_index(
                            tile_obs_idx,
                            labels,
                            origin_flags=coordinates_df.get("origin", pd.Series(index=coordinates_df.index, data=0)).values,
                        )
                        clusters_df = pd.DataFrame({
                            "barcode": obs_barcodes_all[agg_idx],
                            "Segment": agg_labels,
                        })
                    t2 = time.perf_counter()
                    pseudo_centroids = _pseudo_centroids_from_clusters_df(
                        adata,
                        clusters_df,
                        embedding=embedding,
                        dimensions=list(dimensions),
                    )
                    t3 = time.perf_counter()
                    if verbose:
                        print(f"[perf] regen SLIC: {t1 - t0:.3f}s | aggregate: {t2 - t1:.3f}s | pseudo: {t3 - t2:.3f}s")
                    _store_segment_pixel_boundary_payload(
                        adata,
                        segment_name=Segment,
                        label_img=label_img,
                        raster_meta=raster_meta,
                        origin=origin,
                        coordinate_mode=coordinate_mode,
                        array_row_key=array_row_key,
                        array_col_key=array_col_key,
                    )
                    combined_data = None
                finally:
                    if transient_cache_key is not None:
                        adata.uns.pop(transient_cache_key, None)
            else:
                compactness_value, compactness_detail = resolve_compactness(
                    compactness=requested_compactness,
                    compactness_mode=resolved_compactness_mode,
                    target_segment_um=float(target_segment_um) if target_segment_um is not None else None,
                    compactness_base_size_um=compactness_base_size_um,
                    compactness_scale_power=float(compactness_scale_power),
                    compactness_min=compactness_min,
                    compactness_max=compactness_max,
                    compactness_search_multipliers=compactness_search_multipliers,
                    compactness_search_dims=int(compactness_search_dims),
                    compactness_search_jobs=int(compactness_search_jobs),
                    embeddings=np.asarray(embeddings, dtype=float),
                    spatial_coords=np.asarray(spatial_coords, dtype=float),
                    segmenter=lambda candidate: image_plot_slic_segmentation(
                        embeddings,
                        spatial_coords,
                        n_segments=int(resolution),
                        compactness=float(candidate),
                        n_points_eff=int(spatial_coords.shape[0]),
                        figsize=figsize,
                        fig_dpi=fig_dpi,
                        imshow_tile_size=imshow_tile_size,
                        imshow_scale_factor=imshow_scale_factor,
                        pixel_perfect=pixel_perfect,
                        enforce_connectivity=enforce_connectivity,
                        pixel_shape=pixel_shape,
                        runtime_fill_from_boundary=runtime_fill_from_boundary,
                        runtime_fill_closing_radius=runtime_fill_closing_radius,
                        runtime_fill_external_radius=runtime_fill_external_radius,
                        runtime_fill_holes=runtime_fill_holes,
                        soft_rasterization=soft_rasterization,
                        resolve_center_collisions=resolve_center_collisions,
                        center_collision_radius=center_collision_radius,
                        show_image=False,
                        verbose=False,
                        n_jobs=n_jobs,
                        mask_background=mask_background,
                    )[0],
                )
                t0 = time.perf_counter()
                labels, label_img, _, raster_meta = image_plot_slic_segmentation(
                    embeddings,
                    spatial_coords,
                    n_segments=int(resolution),
                    compactness=compactness_value,
                    n_points_eff=int(spatial_coords.shape[0]),
                    figsize=figsize,
                    fig_dpi=fig_dpi,
                    imshow_tile_size=imshow_tile_size,
                    imshow_scale_factor=imshow_scale_factor,
                    pixel_perfect=pixel_perfect,
                    enforce_connectivity=enforce_connectivity,
                    pixel_shape=pixel_shape,
                    runtime_fill_from_boundary=runtime_fill_from_boundary,
                    runtime_fill_closing_radius=runtime_fill_closing_radius,
                    runtime_fill_external_radius=runtime_fill_external_radius,
                    runtime_fill_holes=runtime_fill_holes,
                    soft_rasterization=soft_rasterization,
                    resolve_center_collisions=resolve_center_collisions,
                    center_collision_radius=center_collision_radius,
                    show_image=show_image,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    brighten_continuous=brighten_continuous,
                    continuous_gamma=continuous_gamma,
                    mask_background=mask_background,
                    return_raster_meta=True,
                )
                t1 = time.perf_counter()
                tile_obs_idx = coordinates_df["obs_index"].to_numpy(dtype=np.int64, copy=False)
                if np.unique(tile_obs_idx).size == tile_obs_idx.size:
                    clusters_df = pd.DataFrame({
                        "barcode": obs_barcodes_all[tile_obs_idx],
                        "Segment": np.asarray(labels),
                    })
                else:
                    agg_idx, agg_labels = _aggregate_labels_majority_by_obs_index(
                        tile_obs_idx,
                        labels,
                        origin_flags=coordinates_df.get("origin", pd.Series(index=coordinates_df.index, data=0)).values,
                    )
                    clusters_df = pd.DataFrame({
                        "barcode": obs_barcodes_all[agg_idx],
                        "Segment": agg_labels,
                    })
                t2 = time.perf_counter()
                pseudo_centroids = _pseudo_centroids_from_clusters_df(
                    adata,
                    clusters_df,
                    embedding=embedding,
                    dimensions=list(dimensions),
                )
                t3 = time.perf_counter()
                if verbose:
                    print(f"[perf] direct SLIC: {t1 - t0:.3f}s | aggregate: {t2 - t1:.3f}s | pseudo: {t3 - t2:.3f}s")
                _store_segment_pixel_boundary_payload(
                    adata,
                    segment_name=Segment,
                    label_img=label_img,
                    raster_meta=raster_meta,
                    origin=origin,
                    coordinate_mode=coordinate_mode,
                    array_row_key=array_row_key,
                    array_col_key=array_col_key,
                )
                combined_data = None
            compactness_metadata = {
                **_compactness_mode_header(
                    compactness=requested_compactness,
                    compactness_mode=resolved_compactness_mode,
                    target_segment_um=float(target_segment_um) if target_segment_um is not None else None,
                    compactness_base_size_um=compactness_base_size_um,
                    compactness_scale_power=float(compactness_scale_power),
                    compactness_min=compactness_min,
                    compactness_max=compactness_max,
                    compactness_search_multipliers=compactness_search_multipliers,
                    compactness_search_dims=int(compactness_search_dims),
                    compactness_search_jobs=int(compactness_search_jobs),
                ),
                "selection": compactness_detail,
            }
        else:
            clusters_df, pseudo_centroids, combined_data = segment_image_inner(
                embeddings,
                embeddings_df,
                barcode,
                spatial_coords,
                dimensions,
                method,
                resolution,
                base_compactness,
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
                pixel_perfect,
                enforce_connectivity,
                pixel_shape,
                runtime_fill_from_boundary=runtime_fill_from_boundary,
                runtime_fill_closing_radius=runtime_fill_closing_radius,
                runtime_fill_holes=runtime_fill_holes,
                soft_rasterization=soft_rasterization,
                resolve_center_collisions=resolve_center_collisions,
                center_collision_radius=center_collision_radius,
                show_image=show_image,
                fig_dpi=fig_dpi,
                grid_origin_x=grid_origin_x,
                grid_origin_y=grid_origin_y,
            )
            # If duplicates (origin=False), aggregate to one label per barcode and recompute centroids
            if clusters_df["barcode"].duplicated().any():
                clusters_df = _aggregate_labels_majority(
                    barcode,
                    clusters_df["Segment"].values,
                    origin_flags=coordinates_df.get("origin", pd.Series(index=coordinates_df.index, data=0)).values,
                )
                pseudo_centroids = _pseudo_centroids_from_clusters_df(
                    adata,
                    clusters_df,
                    embedding=embedding,
                    dimensions=list(dimensions),
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
    if hasattr(adata.obs[Segment], "cat"):
        adata.obs[Segment] = adata.obs[Segment].cat.remove_unused_categories()
    if verbose:
        observed_n_segments = int(pd.Series(adata.obs[Segment]).dropna().nunique())
        source_n_segments = int(pd.Series(clusters_df["Segment"]).dropna().nunique())
        try:
            requested_n_segments = int(round(float(resolution))) if str(method) == "image_plot_slic" else None
        except Exception:
            requested_n_segments = None
        if requested_n_segments is not None:
            print(
                f"→ Result  observed_n_segments = {observed_n_segments} "
                f"(requested={requested_n_segments}, source={source_n_segments})"
            )
        else:
            print(
                f"→ Result  observed_n_segments = {observed_n_segments} "
                f"(source={source_n_segments})"
            )
    # Pseudo-centroids to numpy aligned to adata order
    if isinstance(pseudo_centroids, pd.DataFrame):
        pseudo_centroids = pseudo_centroids.reindex(adata.obs.index.astype(str)).values
    segment_suffix = _sanitize_segment_name(Segment)
    base_obsm_key = "X_embedding_segment"
    prefixed_obsm_key = f"X_embedding_{segment_suffix}"
    adata.obsm[base_obsm_key] = pseudo_centroids
    adata.obsm[prefixed_obsm_key] = pseudo_centroids
    if compactness_metadata is not None:
        adata.uns[f"{segment_suffix}_compactness"] = compactness_metadata
    if method == "fixed_grid":
        resolved_grid_origin_x = None
        resolved_grid_origin_y = None
        if len(coordinates_df):
            if grid_origin_x is None:
                resolved_grid_origin_x = float(pd.to_numeric(coordinates_df["x"], errors="coerce").min())
            else:
                resolved_grid_origin_x = float(grid_origin_x)
            if grid_origin_y is None:
                resolved_grid_origin_y = float(pd.to_numeric(coordinates_df["y"], errors="coerce").min())
            else:
                resolved_grid_origin_y = float(grid_origin_y)
        adata.uns[f"{segment_suffix}_fixed_grid"] = {
            "grid_mode": "fixed_grid",
            "grid_side_um": float(resolution),
            "grid_origin_x": resolved_grid_origin_x,
            "grid_origin_y": resolved_grid_origin_y,
            "origin": bool(origin),
        }
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
    figsize=None,
    imshow_tile_size: float | None = None,
    imshow_scale_factor: float = 1.0,
    pixel_perfect: bool = False,
    enforce_connectivity: bool = False,
    pixel_shape: str = "square",
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool = True,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool = False,
    center_collision_radius: int = 2,
    show_image: bool = False,
    fig_dpi: int | None = None,
    n_jobs: int | None = None,
    grid_origin_x: float | None = None,
    grid_origin_y: float | None = None,
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
        'leiden_slic', 'louvain_slic', 'image_plot_slic', 'fixed_grid'.
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
    figsize : tuple or None, optional
        Figure size used when creating the intermediate image for ``image_plot_slic``.
    imshow_tile_size : float or None, optional
        Tile size used for estimating pixel scaling in ``image_plot_slic``.
    imshow_scale_factor : float, optional
        Additional scaling factor for ``image_plot_slic``.
    pixel_perfect : bool, optional (default ``True``)
        If True, use pixel-perfect rasterization when constructing the internal
        image for ``image_plot_slic``.
    enforce_connectivity : bool, optional
        Whether to enforce connectivity in ``image_plot_slic``.
    pixel_shape : str, optional
        Shape used when rendering the temporary image in ``image_plot_slic``.
        ``"circle"`` mimics Visium style spots while ``"square"`` draws
        traditional tiles.
    runtime_fill_from_boundary : bool, optional
        If True, keep the original spot tiles but morphologically fill the
        raster support before ``image_plot_slic`` segmentation.
    runtime_fill_closing_radius : int, optional
        Pixel-radius used for binary closing of the observed raster support
        when ``runtime_fill_from_boundary=True``.
    runtime_fill_external_radius : int, optional
        Additional pixel-radius used to slightly expand the recovered support
        outward after internal filling when ``runtime_fill_from_boundary=True``.
    runtime_fill_holes : bool, optional
        Whether to fill enclosed holes in the raster support when
        ``runtime_fill_from_boundary=True``.
    soft_rasterization : bool, optional
        If True, use bilinear splatting and bilinear label readback for
        ``image_plot_slic``.
    resolve_center_collisions : bool, optional
        If True, locally reassign duplicate raster centers onto nearby empty
        pixels before ``image_plot_slic`` runs.
    center_collision_radius : int, optional
        Search radius in raster pixels used when
        ``resolve_center_collisions=True``.
    show_image : bool, optional
        If ``True`` display the constructed image and label image during
        ``image_plot_slic`` segmentation.
    n_jobs : int or None, optional
        Number of worker threads for ``image_plot_slic`` rasterization. When
        ``None``, defers to ``SPIX_IMAGE_PLOT_SLIC_N_JOBS`` env var or 1.
    grid_origin_x / grid_origin_y : float or None, optional
        Optional explicit grid origin for ``method='fixed_grid'``.
        When omitted, the minimum observed x/y coordinates are used.

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
            pixel_perfect=pixel_perfect,
            enforce_connectivity=enforce_connectivity,
            pixel_shape=pixel_shape,
            runtime_fill_from_boundary=runtime_fill_from_boundary,
            runtime_fill_closing_radius=runtime_fill_closing_radius,
            runtime_fill_external_radius=runtime_fill_external_radius,
            runtime_fill_holes=runtime_fill_holes,
            soft_rasterization=soft_rasterization,
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=center_collision_radius,
            show_image=show_image,
            verbose=verbose,
            n_jobs=n_jobs,
            mask_background=mask_background,
        )
        combined_data = None
    elif method == "fixed_grid":
        clusters, _ = fixed_grid_segmentation(
            spatial_coords,
            side_um=float(resolution),
            origin_x=grid_origin_x,
            origin_y=grid_origin_y,
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
        if verbose:
            print("Image segmentation completed.")

    return clusters_df, pseudo_centroids, combined_data
