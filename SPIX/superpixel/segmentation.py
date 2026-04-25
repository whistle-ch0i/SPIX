import gc
import numpy as np
import scanpy as sc
import pandas as pd
import hashlib
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from anndata import AnnData
from sklearn.cluster import KMeans
import logging
import time
import re
from typing import Any
from skimage.measure import label as _connected_components_label

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
    _clip_candidates,
    evaluate_candidate_metrics_from_labels,
    finalize_compactness_rows,
    infer_compactness_range as infer_compactness_range_from_arrays,
    prepare_compactness_metric_context,
    resolve_compactness,
    schedule_compactness,
    search_compactness,
)
from ..image_processing.image_cache import (
    DEFAULT_CACHE_STORAGE,
    cache_embedding_image,
    get_cached_image,
    get_cached_image_shape,
    release_cached_image,
)
from ..visualization import raster as _raster


_CACHED_IMAGE_SEARCH_STATE: dict[str, Any] | None = None


def _normalize_gap_collapse_axis_safe(axis) -> str | None:
    normalize_fn = getattr(_raster, "normalize_gap_collapse_axis", None)
    if callable(normalize_fn):
        return normalize_fn(axis)
    if axis is None:
        return None
    text = str(axis).strip().lower()
    if text in {"", "none", "off", "false", "no", "0"}:
        return None
    if text in {"y", "row", "rows", "vertical"}:
        return "y"
    if text in {"x", "col", "cols", "column", "columns", "horizontal"}:
        return "x"
    if text in {"both", "xy", "yx"}:
        return "both"
    return None


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
        sample_take=state.get("metric_take", None),
    )[0]
    metrics = evaluate_candidate_metrics_from_labels(
        labels=labels,
        metric_context=state["metric_context_eval"],
        compactness_search_dims=int(state["compactness_search_dims"]),
    )
    return {"compactness": float(candidate), **metrics}


def _summarize_cache_overlap(cache: dict[str, Any]) -> dict[str, Any]:
    img_shape = get_cached_image_shape(cache)
    support_mask = np.asarray(cache.get("support_mask"), dtype=bool)
    plot_params = cache.get("image_plot_params", {}) or {}
    cx = np.asarray(cache.get("cx"), dtype=np.int32).ravel()
    cy = np.asarray(cache.get("cy"), dtype=np.int32).ravel()
    n_tiles = int(min(cx.size, cy.size))
    if n_tiles <= 0:
        raise ValueError("Cached image does not contain cx/cy tile centers.")
    flat = cy.astype(np.int64, copy=False) * np.int64(max(1, img_shape[1])) + cx.astype(np.int64, copy=False)
    unique, counts = np.unique(flat, return_counts=True)
    unique_centers = int(unique.size)
    collided_tiles = int(n_tiles - unique_centers)
    support_pixels = int(np.count_nonzero(support_mask))
    canvas_pixels = int(support_mask.size)
    return {
        "shape": tuple(int(x) for x in img_shape),
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
        "runtime_fill_from_boundary": bool(plot_params.get("runtime_fill_from_boundary", False)),
        "runtime_fill_holes": bool(plot_params.get("runtime_fill_holes", False)),
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
        _raster.normalize_collision_mode(resolve_center_collisions),
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
    resolve_center_collisions,
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
    effective_resolve = resolve_center_collisions
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


def _component_counts_by_segment(
    label_img: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(label_img, dtype=np.int32)
    valid = arr >= 0
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != arr.shape:
            raise ValueError("support_mask must match label_img shape.")
        valid &= support
    if not np.any(valid):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    work = np.where(valid, arr, -1)
    cc = _connected_components_label(work, connectivity=1, background=-1)
    cc_mask = cc > 0
    if not np.any(cc_mask):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    comp_ids = np.asarray(cc[cc_mask], dtype=np.int64)
    comp_labels = np.asarray(work[cc_mask], dtype=np.int32)
    order = np.argsort(comp_ids, kind="mergesort")
    comp_ids = comp_ids[order]
    comp_labels = comp_labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(comp_ids)) + 1]
    per_component_labels = comp_labels[starts]
    seg_labels, comp_counts = np.unique(per_component_labels, return_counts=True)
    return np.asarray(seg_labels, dtype=np.int32), np.asarray(comp_counts, dtype=np.int32)


def summarize_segment_integrity(
    adata: AnnData,
    *,
    segment_key: str = "Segment",
    image_cache_key: str = "image_plot_slic",
    target_segment_um: float | None = None,
    pitch_um: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Summarize whether an existing segmentation behaves cleanly on the raster.

    When the stored pixel-boundary payload is available (created by
    ``segment_image(..., method='image_plot_slic')``), this reports:
    - observed segment size distribution in ``adata.obs[segment_key]``
    - cached raster overlap/collision stats
    - how many unsupported/background raster pixels were assigned a label
    - how often a single segment breaks into multiple connected components
      within the observed support mask
    """
    if segment_key not in adata.obs.columns:
        raise KeyError(f"No segmentation column adata.obs['{segment_key}'].")

    seg_series = pd.Series(adata.obs[segment_key])
    seg_nonnull = seg_series.dropna()
    counts = seg_nonnull.value_counts(dropna=True)
    result: dict[str, Any] = {
        "segment_key": str(segment_key),
        "n_obs_total": int(adata.n_obs),
        "n_obs_segmented": int(seg_nonnull.size),
        "n_segments_observed": int(counts.size),
        "segment_size_mean": float(counts.mean()) if not counts.empty else np.nan,
        "segment_size_median": float(counts.median()) if not counts.empty else np.nan,
        "segment_size_min": int(counts.min()) if not counts.empty else 0,
        "segment_size_p05": float(counts.quantile(0.05)) if not counts.empty else np.nan,
        "segment_size_p95": float(counts.quantile(0.95)) if not counts.empty else np.nan,
        "segment_size_max": int(counts.max()) if not counts.empty else 0,
        "segment_size_cv": (
            float(counts.std(ddof=0) / max(float(counts.mean()), 1e-12))
            if not counts.empty else np.nan
        ),
    }

    if target_segment_um is not None and pitch_um is not None:
        expected_spots_per_segment = max((float(target_segment_um) / float(pitch_um)) ** 2, 1.0)
        result["target_segment_um"] = float(target_segment_um)
        result["pitch_um"] = float(pitch_um)
        result["expected_spots_per_segment"] = float(expected_spots_per_segment)
        if seg_nonnull.size > 0:
            result["expected_n_segments"] = int(round(float(seg_nonnull.size) / float(expected_spots_per_segment)))
            result["mean_vs_expected_size_ratio"] = float(result["segment_size_mean"] / float(expected_spots_per_segment))
            result["median_vs_expected_size_ratio"] = float(result["segment_size_median"] / float(expected_spots_per_segment))

    if image_cache_key in adata.uns:
        try:
            result["cache_overlap"] = _summarize_cache_overlap(adata.uns[image_cache_key])
        except Exception as exc:
            result["cache_overlap_error"] = str(exc)

    segment_suffix = _sanitize_segment_name(segment_key)
    payload = adata.uns.get(f"{segment_suffix}_pixel_boundary", None)
    if isinstance(payload, dict) and ("label_img" in payload):
        label_img = np.asarray(payload["label_img"], dtype=np.int32)
        support_mask_raw = payload.get("support_mask", None)
        support_mask = None if support_mask_raw is None else np.asarray(support_mask_raw, dtype=bool)
        raster_diag: dict[str, Any] = {
            "label_img_shape": tuple(int(x) for x in label_img.shape),
            "has_support_mask": bool(support_mask is not None),
        }
        if support_mask is not None and support_mask.shape == label_img.shape:
            support_valid = support_mask & (label_img >= 0)
            background_valid = (~support_mask) & (label_img >= 0)
            support_labels = np.asarray(label_img[support_valid], dtype=np.int32)
            background_labels = np.asarray(label_img[background_valid], dtype=np.int32)
            seg_labels_support, comp_counts_support = _component_counts_by_segment(
                label_img,
                support_mask=support_mask,
            )
            raster_diag.update(
                {
                    "support_pixels": int(np.count_nonzero(support_mask)),
                    "background_pixels": int(np.count_nonzero(~support_mask)),
                    "support_labeled_pixels": int(np.count_nonzero(support_valid)),
                    "background_labeled_pixels": int(np.count_nonzero(background_valid)),
                    "background_labeled_fraction_of_canvas": float(np.count_nonzero(background_valid) / max(1, label_img.size)),
                    "background_labeled_fraction_of_background": float(np.count_nonzero(background_valid) / max(1, np.count_nonzero(~support_mask))),
                    "n_labels_in_support_raster": int(np.unique(support_labels).size) if support_labels.size else 0,
                    "n_labels_touching_background": int(np.unique(background_labels).size) if background_labels.size else 0,
                    "labels_touching_background_fraction": (
                        float(np.unique(background_labels).size / max(1, np.unique(support_labels).size))
                        if background_labels.size else 0.0
                    ),
                    "split_labels_within_support": int(np.count_nonzero(comp_counts_support > 1)),
                    "split_labels_fraction_within_support": float(np.count_nonzero(comp_counts_support > 1) / max(1, seg_labels_support.size)),
                    "max_components_per_label_within_support": int(comp_counts_support.max()) if comp_counts_support.size else 0,
                    "median_components_per_label_within_support": float(np.median(comp_counts_support)) if comp_counts_support.size else 0.0,
                }
            )
        result["raster_diagnostics"] = raster_diag
    else:
        result["raster_diagnostics"] = None

    if verbose:
        print(
            f"[segment_integrity] key={segment_key} | "
            f"n_segments={result['n_segments_observed']} | "
            f"mean_size={result['segment_size_mean']:.3f} | "
            f"median_size={result['segment_size_median']:.3f} | "
            f"size_cv={result['segment_size_cv']:.3f}"
        )
        cache_overlap = result.get("cache_overlap", None)
        if isinstance(cache_overlap, dict):
            print(
                "[segment_integrity] cache overlap: "
                f"collision_fraction={float(cache_overlap['collision_fraction']):.6f} | "
                f"support_fraction={float(cache_overlap['support_fraction']):.6f} | "
                f"tile_px={int(cache_overlap['tile_px'])}"
            )
        raster_diag = result.get("raster_diagnostics", None)
        if isinstance(raster_diag, dict):
            print(
                "[segment_integrity] raster: "
                f"bg_labeled={int(raster_diag.get('background_labeled_pixels', 0))} | "
                f"bg_frac={float(raster_diag.get('background_labeled_fraction_of_background', 0.0)):.6f} | "
                f"split_labels={int(raster_diag.get('split_labels_within_support', 0))} | "
                f"split_frac={float(raster_diag.get('split_labels_fraction_within_support', 0.0)):.6f}"
            )
    return result


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
            f"collide={_raster.normalize_collision_mode(resolve_center_collisions)}",
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
    cache_storage_kind = str(cache.get("cache_storage", "memory"))
    cache_preload = (
        cache.get("img", None) is None
        and cache.get("img_path", None) is not None
        and cache_storage_kind not in {"memory", "float32_memmap"}
    )
    if cache_preload:
        t_preload0 = time.perf_counter()
        get_cached_image(cache, cache_in_memory=True)
        t_preload1 = time.perf_counter()
        if verbose:
            print(
                f"[perf] {mode} process cache preload: "
                f"{t_preload1 - t_preload0:.3f}s | storage={cache_storage_kind}"
            )
    metric_take = metric_context.get("metric_take", None)
    metric_context_eval = metric_context
    if metric_take is not None:
        metric_context_eval = dict(metric_context)
        metric_context_eval["metric_take"] = None
    if verbose:
        print(
            f"[perf] {mode} process candidate prep: "
            f"sample+knn={t1 - t0:.3f}s | candidates={len(clipped)} | jobs={max(1, int(compactness_search_jobs))}"
        )
    state = {
        "cache": cache,
        "n_segments": int(n_segments),
        "metric_context_eval": metric_context_eval,
        "metric_take": None if metric_take is None else np.asarray(metric_take, dtype=np.int64),
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


def _normalize_compactness_search_mode(compactness_search_mode: str | None) -> str:
    mode = str(compactness_search_mode or "grid").strip().lower()
    aliases = {
        "grid": "grid",
        "default": "grid",
        "adaptive": "adaptive_multifidelity",
        "adaptive_multifidelity": "adaptive_multifidelity",
        "multifidelity": "adaptive_multifidelity",
    }
    if mode not in aliases:
        raise ValueError("compactness_search_mode must be one of {'grid','adaptive_multifidelity'}.")
    return aliases[mode]


def _resolve_adaptive_multifidelity_stage_downsamples(
    final_downsample: float,
    stage_downsample_factors: tuple[float, ...] | list[float] | None,
) -> list[float]:
    if stage_downsample_factors is not None:
        raw = [float(x) for x in stage_downsample_factors if np.isfinite(float(x)) and float(x) > 0]
    else:
        final = float(max(1.0, float(final_downsample)))
        raw = []
        if final < 4.0:
            raw.append(4.0)
        if final < 2.0:
            raw.append(2.0)
        raw.append(final)
    ordered: list[float] = []
    seen: set[float] = set()
    for value in sorted(raw, reverse=True):
        key = round(float(value), 6)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(float(max(1.0, value)))
    return ordered or [float(max(1.0, float(final_downsample)))]


def _resolve_adaptive_multifidelity_stage_sample_sizes(
    final_sample_size: int | None,
    n_stages: int,
    stage_sample_sizes: tuple[int | None, ...] | list[int | None] | None,
) -> list[int | None]:
    if stage_sample_sizes is not None:
        resolved: list[int | None] = []
        for value in list(stage_sample_sizes)[: max(1, int(n_stages))]:
            if value is None:
                resolved.append(None)
            else:
                resolved.append(int(max(1, int(value))))
        if not resolved:
            resolved = [None if final_sample_size is None else int(max(1, int(final_sample_size)))]
        while len(resolved) < int(n_stages):
            resolved.append(resolved[-1])
        return resolved

    if final_sample_size is None:
        defaults = [4000, 8000, None]
        if n_stages <= len(defaults):
            return defaults[-int(n_stages):]
        return [defaults[0]] * (int(n_stages) - len(defaults)) + defaults

    final = int(max(1, int(final_sample_size)))
    if int(n_stages) <= 1:
        return [final]
    out: list[int | None] = []
    for idx in range(int(n_stages)):
        frac = float(idx + 1) / float(int(n_stages))
        out.append(int(max(1000, round(final * frac))))
    out[-1] = final
    return out


def _refine_adaptive_multifidelity_candidates(
    *,
    candidates: list[float],
    detail: dict[str, Any],
    compactness_min: float | None,
    compactness_max: float | None,
    top_k: int,
    target_count: int | None = None,
) -> list[float]:
    pool = sorted(
        {
            float(x)
            for x in candidates
            if np.isfinite(float(x)) and float(x) > 0
        }
    )
    if len(pool) <= 1:
        return pool

    scores = {
        round(float(row["compactness"]), 12): float(row.get("score", np.inf))
        for row in detail.get("candidates", [])
    }
    best = float(detail.get("selected_compactness", pool[len(pool) // 2]))
    ranked = sorted(
        pool,
        key=lambda value: (
            scores.get(round(float(value), 12), np.inf),
            abs(np.log(float(value) / max(best, 1e-12))),
        ),
    )
    winners = sorted(ranked[: max(1, int(top_k))])
    pool_arr = np.asarray(pool, dtype=float)
    neighbor_idx: set[int] = set()
    for value in winners:
        idx = int(np.argmin(np.abs(pool_arr - float(value))))
        for delta in (-1, 0, 1):
            j = idx + delta
            if 0 <= j < len(pool):
                neighbor_idx.add(int(j))
    selected = [float(pool[j]) for j in sorted(neighbor_idx)]
    low = float(min(selected) if selected else best)
    high = float(max(selected) if selected else best)
    if not np.isfinite(low) or low <= 0:
        low = float(best) / 1.5
    if not np.isfinite(high) or high <= low:
        high = float(best) * 1.5
    dense_n = int(max(5, target_count if target_count is not None else (max(2 * int(top_k), 7))))
    dense = np.geomspace(low / 1.35, high * 1.35, num=dense_n).tolist()
    refined = sorted(
        {
            float(x)
            for x in _clip_candidates(
                list(selected) + list(winners) + list(dense) + [float(best)],
                compactness_min=compactness_min,
                compactness_max=compactness_max,
            )
        }
    )
    return refined or winners or pool


def _run_compactness_search_with_cache(
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
    use_process_search: bool,
) -> tuple[float, dict[str, Any]]:
    effective_jobs = _effective_compactness_search_jobs(
        requested_jobs=int(compactness_search_jobs),
        n_candidates=len(candidates),
        n_segments=int(n_segments),
        backend="process" if use_process_search else "thread",
        verbose=verbose,
    )
    if use_process_search:
        if verbose:
            print(f"[perf] {mode}: compactness search backend: process")
        return _search_compactness_cached_image_process(
            candidates=candidates,
            cache=cache,
            n_segments=int(n_segments),
            embeddings=np.asarray(embeddings, dtype=float),
            spatial_coords=np.asarray(spatial_coords, dtype=float),
            compactness_search_dims=int(compactness_search_dims),
            compactness_min=compactness_min,
            compactness_max=compactness_max,
            scheduled=scheduled,
            mode=str(mode),
            target_segment_um=target_segment_um,
            compactness_search_jobs=int(effective_jobs),
            compactness_metric_sample_size=compactness_metric_sample_size,
            enforce_connectivity=bool(enforce_connectivity),
            mask_background=bool(mask_background),
            verbose=verbose,
        )
    if verbose:
        print(f"[perf] {mode}: compactness search backend: thread")
    return search_compactness(
        candidates=candidates,
        embeddings=np.asarray(embeddings, dtype=float),
        spatial_coords=np.asarray(spatial_coords, dtype=float),
        segmenter=lambda candidate: slic_segmentation_from_cached_image(
            cache,
            n_segments=int(n_segments),
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
        scheduled=scheduled,
        mode=str(mode),
        target_segment_um=target_segment_um,
        compactness_search_jobs=int(effective_jobs),
        compactness_metric_sample_size=compactness_metric_sample_size,
        verbose=verbose,
    )


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


def _cached_reference_tile_count(
    adata: AnnData,
    *,
    origin: bool,
    verbose: bool,
) -> int | None:
    cache_key = "__spix_fast_compactness_reference_counts__"
    origin_key = "origin_true" if bool(origin) else "origin_false"
    uns = getattr(adata, "uns", None)
    cached_counts = uns.get(cache_key, None) if uns is not None else None
    if isinstance(cached_counts, dict) and origin_key in cached_counts:
        cached_value = cached_counts.get(origin_key, None)
        if cached_value is not None:
            if verbose:
                print(
                    f"[perf] fast compactness reference count cache hit: "
                    f"origin={int(bool(origin))} | n_tiles={int(cached_value)}"
                )
            return int(cached_value)

    t_ref0 = time.perf_counter()
    try:
        tiles_all = adata.uns["tiles"]
        if bool(origin) and ("spatial" in adata.obsm):
            spatial_direct = np.asarray(adata.obsm["spatial"])
            if (
                spatial_direct.ndim == 2
                and spatial_direct.shape[0] == adata.n_obs
                and spatial_direct.shape[1] >= 2
            ):
                direct_n = int(adata.n_obs)
                if verbose:
                    print(
                        f"[perf] fast compactness reference count direct: "
                        f"origin=1 | n_tiles={direct_n} | source=adata.obsm['spatial'] | "
                        f"time={time.perf_counter() - t_ref0:.3f}s"
                    )
                if uns is not None:
                    if not isinstance(cached_counts, dict):
                        cached_counts = {}
                    cached_counts[origin_key] = int(direct_n)
                    uns[cache_key] = cached_counts
                return direct_n

        tiles_for_n = tiles_all
        if "origin" in tiles_all.columns and bool(origin) and (tiles_all["origin"] == 1).any():
            tiles_for_n = tiles_all[tiles_all["origin"] == 1]
        if "barcode" in tiles_for_n.columns:
            tiles_for_n = tiles_for_n.drop_duplicates("barcode")
        n_tiles = int(len(tiles_for_n))
        if verbose:
            print(
                f"[perf] fast compactness reference count computed: "
                f"origin={int(bool(origin))} | n_tiles={n_tiles} | "
                f"time={time.perf_counter() - t_ref0:.3f}s"
            )
        if uns is not None:
            if not isinstance(cached_counts, dict):
                cached_counts = {}
            cached_counts[origin_key] = int(n_tiles)
            uns[cache_key] = cached_counts
        return n_tiles
    except Exception:
        if verbose:
            print(
                f"[perf] fast compactness reference count unavailable: "
                f"origin={int(bool(origin))} | time={time.perf_counter() - t_ref0:.3f}s"
            )
        return None


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
    gap_collapse_axis: str | None = None,
    gap_collapse_max_run_px: int = 1,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool | str = "auto",
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
    compactness_search_mode: str = "grid",
    compactness_search_raster_mode: str = "auto",
    compactness_search_resolution_cap: int | None = None,
    adaptive_multifidelity_top_k: int = 4,
    adaptive_multifidelity_stage_downsample_factors: tuple[float, ...] | None = None,
    adaptive_multifidelity_stage_sample_sizes: tuple[int | None, ...] | None = None,
    use_cached_image: bool = True,
    image_cache_key: str = "image_plot_slic",
    cache_storage: str = DEFAULT_CACHE_STORAGE,
    verbose: bool = True,
) -> dict[str, Any]:
    compactness_search_mode_norm = _normalize_compactness_search_mode(compactness_search_mode)
    effective_dimensions = list(dimensions if search_dimensions is None else search_dimensions)
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

    transient_cache_keys: list[str] = []
    stage_cache_state: dict[float, tuple[dict[str, Any], dict[str, Any]]] = {}

    def _get_stage_cache(stage_downsample_factor: float) -> tuple[dict[str, Any], dict[str, Any]]:
        stage_key = round(float(stage_downsample_factor), 6)
        cached_state = stage_cache_state.get(stage_key, None)
        if cached_state is not None:
            return cached_state

        stage_raster_options = _resolve_compactness_search_raster_options(
            adata,
            coordinate_mode=str(coordinate_mode),
            origin=bool(origin),
            array_row_key=str(array_row_key),
            array_col_key=str(array_col_key),
            pixel_perfect=bool(pixel_perfect),
            pixel_shape=str(pixel_shape),
            soft_rasterization=bool(soft_rasterization),
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=int(center_collision_radius),
            compactness_search_downsample_factor=float(stage_downsample_factor),
            compactness_search_raster_mode=str(compactness_search_raster_mode),
            verbose=bool(verbose),
        )
        stage_pixel_perfect = bool(stage_raster_options["pixel_perfect"])
        stage_pixel_shape = str(stage_raster_options["pixel_shape"])
        stage_soft_rasterization = bool(stage_raster_options["soft_rasterization"])
        stage_resolve_center_collisions = bool(stage_raster_options["resolve_center_collisions"])

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
                pixel_perfect=bool(stage_pixel_perfect),
                pixel_shape=str(stage_pixel_shape),
                soft_rasterization=bool(stage_soft_rasterization),
                resolve_center_collisions=bool(stage_resolve_center_collisions),
                center_collision_radius=int(center_collision_radius),
                runtime_fill_from_boundary=bool(runtime_fill_from_boundary),
                runtime_fill_closing_radius=int(runtime_fill_closing_radius),
                runtime_fill_external_radius=int(runtime_fill_external_radius),
                runtime_fill_holes=bool(runtime_fill_holes),
                downsample_factor=float(stage_downsample_factor),
            )
            if reusable_cache_key in adata.uns:
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
                        or np.isclose(float(candidate_cache.get("downsample_factor", 1.0)), float(stage_downsample_factor))
                    )
                ):
                    cache = candidate_cache
                    if verbose:
                        print(
                            "[perf] reusing raster cache for compactness selection: "
                            f"ds={float(stage_downsample_factor):.4g}"
                        )
                        _print_compactness_cache_summary(cache)
                elif verbose:
                    print(
                        "[warning] Existing cached image does not match compactness stage settings; "
                        f"rebuilding cache for ds={float(stage_downsample_factor):.4g}."
                    )

        if cache is None:
            transient_cache_key = None if use_cached_image else f"__spix_select_compactness_cache__ds_{stage_key:g}"
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
                pixel_perfect=stage_pixel_perfect,
                pixel_shape=stage_pixel_shape,
                verbose=False,
                show=False,
                runtime_fill_from_boundary=runtime_fill_from_boundary,
                runtime_fill_closing_radius=runtime_fill_closing_radius,
                runtime_fill_external_radius=runtime_fill_external_radius,
                runtime_fill_holes=runtime_fill_holes,
                soft_rasterization=stage_soft_rasterization,
                resolve_center_collisions=stage_resolve_center_collisions,
                center_collision_radius=center_collision_radius,
                downsample_factor=float(stage_downsample_factor),
                memmap_scope="transient" if transient_cache_key is not None else "persistent",
                memmap_cleanup="on_exit" if transient_cache_key is not None else "none",
                cache_storage="float32_memmap" if transient_cache_key is not None else cache_storage,
            )
            t_cache1 = time.perf_counter()
            if transient_cache_key is not None:
                transient_cache_keys.append(transient_cache_key)
            if verbose:
                print(
                    "[perf] built raster cache for compactness selection: "
                    f"ds={float(stage_downsample_factor):.4g} | {t_cache1 - t_cache0:.3f}s"
                )
                _print_compactness_cache_summary(cache)

        out = (cache, stage_raster_options)
        stage_cache_state[stage_key] = out
        return out

    try:
        use_process_search = bool(
            int(compactness_search_jobs) > 1
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
        final_search_raster_options = _resolve_compactness_search_raster_options(
            adata,
            coordinate_mode=str(coordinate_mode),
            origin=bool(origin),
            array_row_key=str(array_row_key),
            array_col_key=str(array_col_key),
            pixel_perfect=bool(pixel_perfect),
            pixel_shape=str(pixel_shape),
            soft_rasterization=bool(soft_rasterization),
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=int(center_collision_radius),
            compactness_search_downsample_factor=float(compactness_search_downsample_factor),
            compactness_search_raster_mode=str(compactness_search_raster_mode),
            verbose=False,
        )

        adaptive_stage_details: list[dict[str, Any]] = []
        if compactness_search_mode_norm == "adaptive_multifidelity":
            stage_downsamples = _resolve_adaptive_multifidelity_stage_downsamples(
                float(compactness_search_downsample_factor),
                adaptive_multifidelity_stage_downsample_factors,
            )
            stage_sample_sizes = _resolve_adaptive_multifidelity_stage_sample_sizes(
                compactness_metric_sample_size,
                len(stage_downsamples),
                adaptive_multifidelity_stage_sample_sizes,
            )
            current_candidates = sorted(
                {
                    float(x)
                    for x in coarse_candidates
                    if np.isfinite(float(x)) and float(x) > 0
                }
            )
            if verbose:
                print(
                    "[perf] compactness adaptive multifidelity: "
                    f"stages={len(stage_downsamples)} | downsample={stage_downsamples} | "
                    f"samples={stage_sample_sizes}"
                )
            stage_best = None
            chosen_detail = None
            for stage_idx, (stage_downsample, stage_sample_size) in enumerate(
                zip(stage_downsamples, stage_sample_sizes),
                start=1,
            ):
                cache, stage_raster_options = _get_stage_cache(float(stage_downsample))
                stage_mode = f"adaptive_multifidelity_stage_{stage_idx}"
                t_stage0 = time.perf_counter()
                stage_best, stage_detail = _run_compactness_search_with_cache(
                    candidates=current_candidates,
                    cache=cache,
                    n_segments=int(search_resolution),
                    embeddings=np.asarray(prepared["embeddings"], dtype=float),
                    spatial_coords=np.asarray(prepared["spatial_coords"], dtype=float),
                    compactness_search_dims=int(compactness_search_dims),
                    compactness_min=compactness_min,
                    compactness_max=compactness_max,
                    scheduled=None if stage_best is None else float(stage_best),
                    mode=stage_mode,
                    target_segment_um=target_segment_um,
                    compactness_search_jobs=int(compactness_search_jobs),
                    compactness_metric_sample_size=stage_sample_size,
                    enforce_connectivity=bool(enforce_connectivity),
                    mask_background=bool(mask_background),
                    verbose=verbose,
                    use_process_search=use_process_search,
                )
                t_stage1 = time.perf_counter()
                stage_detail = dict(stage_detail)
                stage_detail["stage_index"] = int(stage_idx)
                stage_detail["fidelity_downsample_factor"] = float(stage_downsample)
                stage_detail["fidelity_metric_sample_size"] = (
                    None if stage_sample_size is None else int(stage_sample_size)
                )
                stage_detail["input_candidates"] = [float(x) for x in current_candidates]
                stage_detail["search_raster_options"] = dict(stage_raster_options)
                adaptive_stage_details.append(stage_detail)
                chosen_detail = stage_detail
                final_search_raster_options = dict(stage_raster_options)
                if verbose:
                    print(
                        f"[perf] compactness adaptive stage {stage_idx}: "
                        f"candidates={len(current_candidates)} | ds={float(stage_downsample):.4g} | "
                        f"time={t_stage1 - t_stage0:.3f}s | selected={float(stage_best):.6g}"
                    )
                if stage_idx < len(stage_downsamples):
                    current_candidates = _refine_adaptive_multifidelity_candidates(
                        candidates=current_candidates,
                        detail=stage_detail,
                        compactness_min=compactness_min,
                        compactness_max=compactness_max,
                        top_k=int(adaptive_multifidelity_top_k),
                        target_count=max(int(adaptive_multifidelity_top_k) * 2, 7),
                    )
            coarse_best = float(stage_best)
            coarse_detail = adaptive_stage_details[0]
            refine_detail = adaptive_stage_details[-1] if len(adaptive_stage_details) > 1 else None
            refine_best = float(stage_best)
        else:
            cache, final_search_raster_options = _get_stage_cache(float(compactness_search_downsample_factor))
            t_coarse0 = time.perf_counter()
            coarse_best, coarse_detail = _run_compactness_search_with_cache(
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
                compactness_search_jobs=int(compactness_search_jobs),
                compactness_metric_sample_size=compactness_metric_sample_size,
                enforce_connectivity=bool(enforce_connectivity),
                mask_background=bool(mask_background),
                verbose=verbose,
                use_process_search=use_process_search,
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
                refine_best, refine_detail = _run_compactness_search_with_cache(
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
                    compactness_search_jobs=int(compactness_search_jobs),
                    compactness_metric_sample_size=compactness_metric_sample_size,
                    enforce_connectivity=bool(enforce_connectivity),
                    mask_background=bool(mask_background),
                    verbose=verbose,
                    use_process_search=use_process_search,
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
            "compactness_search_mode": str(compactness_search_mode_norm),
            "compactness_metric_sample_size": None if compactness_metric_sample_size is None else int(compactness_metric_sample_size),
            "compactness_search_raster_options": final_search_raster_options,
            "mask_background": bool(mask_background),
            "range": range_info,
            "coarse_search": coarse_detail,
            "refine_search": refine_detail,
        }
        if adaptive_stage_details:
            result["adaptive_multifidelity_search"] = adaptive_stage_details
        return result
    finally:
        for transient_cache_key in transient_cache_keys:
            release_cached_image(adata, transient_cache_key, remove_memmap_file=True)


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
    compactness_search_mode: str = "grid",
    compactness_search_raster_mode: str = "auto",
    compactness_search_resolution_cap: int | None = None,
    adaptive_multifidelity_top_k: int = 4,
    adaptive_multifidelity_stage_downsample_factors: tuple[float, ...] | None = None,
    adaptive_multifidelity_stage_sample_sizes: tuple[int | None, ...] | None = None,
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
    t_fast0 = time.perf_counter()
    if search_target_segment_um is None:
        n_tiles_for_reference = _cached_reference_tile_count(
            adata,
            origin=bool(origin),
            verbose=bool(verbose),
        )
        selection_target_segment_um = _auto_fast_compactness_search_target_um(
            target_segment_um,
            n_tiles=n_tiles_for_reference,
            pitch_um=float(pitch_um),
            compactness_search_downsample_factor=float(compactness_search_downsample_factor),
            min_reference_segment_side_px=10.0,
            max_stable_search_n_segments=15000,
        )
        if verbose:
            print(
                f"[perf] fast compactness auto reference target: "
                f"requested_target_um={None if target_segment_um is None else float(target_segment_um):.6g} | "
                f"search_target_um={None if selection_target_segment_um is None else float(selection_target_segment_um):.6g} | "
                f"n_tiles_for_reference={None if n_tiles_for_reference is None else int(n_tiles_for_reference)} | "
                f"time={time.perf_counter() - t_fast0:.3f}s"
            )
        auto_reference_target = bool(
            selection_target_segment_um is not None
            and target_segment_um is not None
            and not np.isclose(float(selection_target_segment_um), float(target_segment_um))
        )
    else:
        selection_target_segment_um = float(search_target_segment_um)
        auto_reference_target = False
        if verbose:
            print(
                f"[perf] fast compactness reference target override: "
                f"requested_target_um={None if target_segment_um is None else float(target_segment_um):.6g} | "
                f"search_target_um={float(selection_target_segment_um):.6g} | "
                f"time={time.perf_counter() - t_fast0:.3f}s"
            )
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
        compactness_search_mode=compactness_search_mode,
        compactness_search_raster_mode=compactness_search_raster_mode,
        compactness_search_resolution_cap=compactness_search_resolution_cap,
        adaptive_multifidelity_top_k=adaptive_multifidelity_top_k,
        adaptive_multifidelity_stage_downsample_factors=adaptive_multifidelity_stage_downsample_factors,
        adaptive_multifidelity_stage_sample_sizes=adaptive_multifidelity_stage_sample_sizes,
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
    gap_collapse_axis: str | None = None,
    gap_collapse_max_run_px: int = 1,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool | str = "auto",
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
    cache_storage: str = DEFAULT_CACHE_STORAGE,
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
        cache_storage=cache_storage,
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
    gap_collapse_axis: str | None = None,
    gap_collapse_max_run_px: int = 1,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool | str = "auto",
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
    cache_storage: str = DEFAULT_CACHE_STORAGE,
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
            cache_storage=cache_storage,
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


def _format_precompute_scale_value(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:g}".replace("+", "")


def _precompute_scale_id(resolution: float, compactness: float | None = None) -> str:
    res_text = _format_precompute_scale_value(float(resolution))
    if compactness is None:
        return f"r{res_text}"
    comp_text = _format_precompute_scale_value(float(compactness))
    return f"r{res_text}_c{comp_text}"


def _normalize_precompute_float_list(
    values: float | list[float] | tuple[float, ...] | np.ndarray | None,
    *,
    name: str,
) -> list[float]:
    if values is None:
        return []
    if np.isscalar(values):
        return [float(values)]
    try:
        out = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be a scalar or iterable of numeric values.") from exc
    if len(out) == 0:
        return []
    return out


def _normalize_precompute_storage_mode(storage: str) -> str:
    mode = str(storage or "uns").strip().lower()
    aliases = {
        "memory": "uns",
        "mem": "uns",
        "adata.uns": "uns",
        "uns": "uns",
        "file": "disk",
        "files": "disk",
        "disk": "disk",
        "both": "both",
    }
    if mode not in aliases:
        raise ValueError("storage must be one of {'uns', 'disk', 'both'}.")
    return aliases[mode]


def _compress_observed_labels_for_export(
    labels_tile: np.ndarray,
    label_img: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    labels_tile_arr = np.asarray(labels_tile, dtype=np.int32)
    valid = labels_tile_arr >= 0

    if not np.any(valid):
        if label_img is None:
            return labels_tile_arr, None, 0
        label_img_out = np.full_like(np.asarray(label_img, dtype=np.int32), -1)
        return labels_tile_arr, label_img_out, 0

    observed = np.unique(labels_tile_arr[valid]).astype(np.int32)
    max_label = int(max(int(labels_tile_arr.max()), int(observed.max())))
    remap = np.full(max_label + 1, -1, dtype=np.int32)
    remap[observed] = np.arange(observed.size, dtype=np.int32)

    labels_tile_out = np.where(valid, remap[labels_tile_arr], -1).astype(np.int32, copy=False)

    label_img_out = None
    if label_img is not None:
        label_img_arr = np.asarray(label_img, dtype=np.int32)
        label_img_out = np.full_like(label_img_arr, -1)
        img_valid = (label_img_arr >= 0) & (label_img_arr <= max_label)
        label_img_out[img_valid] = remap[label_img_arr[img_valid]]

    return labels_tile_out, label_img_out, int(observed.size)


def _native_identity_available(cache: dict[str, Any]) -> bool:
    tile_obs_indices = cache.get("tile_obs_indices", None)
    if tile_obs_indices is not None:
        idx = np.asarray(tile_obs_indices, dtype=np.int64)
        return bool(idx.ndim == 1 and idx.size > 0 and np.unique(idx).size == idx.size)
    cache_barcodes = pd.Series(cache.get("barcodes") or [], dtype=str)
    return bool((not cache_barcodes.empty) and (not cache_barcodes.duplicated().any()))


def _identity_segmentation_from_cache(
    cache: dict[str, Any],
    *,
    n_tiles: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels_tile = np.arange(int(n_tiles), dtype=np.int32)
    support_mask = cache.get("support_mask", None)
    if support_mask is not None:
        label_shape = np.asarray(support_mask, dtype=bool).shape
    else:
        label_shape = get_cached_image_shape(cache)[:2]
    label_img = np.full(label_shape, -1, dtype=np.int32)

    cx = np.asarray(cache.get("cx", []), dtype=np.int64).ravel()
    cy = np.asarray(cache.get("cy", []), dtype=np.int64).ravel()
    n_centers = int(min(cx.size, cy.size, labels_tile.size))
    if n_centers > 0:
        label_img[cy[:n_centers], cx[:n_centers]] = labels_tile[:n_centers]

    return labels_tile, label_img


def _cached_tile_labels_to_obs_codes(
    adata: AnnData,
    cache: dict[str, Any],
    labels_tile: np.ndarray,
    *,
    image_cache_key: str,
) -> np.ndarray:
    tile_labels = np.asarray(labels_tile, dtype=np.int32)
    n_obs = int(adata.n_obs)
    seg_codes = np.full(n_obs, -1, dtype=np.int32)

    tile_obs_indices = cache.get("tile_obs_indices", None)
    if tile_obs_indices is not None:
        idx = np.asarray(tile_obs_indices, dtype=np.int64)
        if idx.ndim != 1 or idx.size != tile_labels.size:
            raise ValueError(
                f"Cached image '{image_cache_key}' has tile_obs_indices of length {idx.size}, "
                f"but got {tile_labels.size} tile labels."
            )
        valid = (idx >= 0) & (idx < n_obs)
        idx = idx[valid]
        valid_labels = tile_labels[valid]
        if idx.size == 0:
            return seg_codes
        if np.unique(idx).size == idx.size:
            seg_codes[idx] = valid_labels
            return seg_codes
        origin_flags = cache.get("origin_flags", None)
        origin_valid = None
        if origin_flags is not None:
            origin_arr = np.asarray(origin_flags)
            if origin_arr.ndim == 1 and origin_arr.size == tile_labels.size:
                origin_valid = origin_arr[valid]
        agg_idx, agg_labels = _aggregate_labels_majority_by_obs_index(
            idx,
            valid_labels,
            origin_flags=origin_valid,
        )
        seg_codes[agg_idx] = np.asarray(agg_labels, dtype=np.int32)
        return seg_codes

    cache_barcodes = pd.Series(cache.get("barcodes") or [], dtype=str)
    if cache_barcodes.empty:
        raise ValueError(
            f"Cached image '{image_cache_key}' missing both tile_obs_indices and barcodes."
        )
    if cache_barcodes.size != tile_labels.size:
        raise ValueError(
            f"Cached image '{image_cache_key}' has {cache_barcodes.size} barcodes, "
            f"but got {tile_labels.size} tile labels."
        )

    if cache_barcodes.duplicated().any():
        clusters_df = _aggregate_labels_majority(
            cache_barcodes,
            tile_labels,
            origin_flags=cache.get("origin_flags", None),
        )
        tile_barcodes = clusters_df["barcode"].astype(str).to_numpy()
        tile_labels_use = clusters_df["Segment"].to_numpy(dtype=np.int32)
    else:
        tile_barcodes = cache_barcodes.to_numpy()
        tile_labels_use = tile_labels

    obs_barcodes = pd.Index(adata.obs.index.astype(str))
    obs_pos = obs_barcodes.get_indexer(pd.Index(tile_barcodes))
    ok = obs_pos >= 0
    seg_codes[obs_pos[ok]] = np.asarray(tile_labels_use[ok], dtype=np.int32)
    return seg_codes


def precompute_multiscale_segments(
    adata: AnnData,
    *,
    resolutions: list[float] | tuple[float, ...] | np.ndarray | float | None = None,
    manual_pairs: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
    compactnesses: list[float] | None = None,
    compactness_candidates: list[float] | None = None,
    extra_compactnesses: list[float] | None = None,
    dimensions: list[int] = list(range(30)),
    embedding: str = "X_embedding",
    segment_method: str = "image_plot_slic",
    image_cache_key: str = "image_plot_slic",
    storage: str = "disk",
    uns_key: str = "multiscale_segments",
    out_dir: str | None = "./multiscale_segments",
    filename_prefix: str = "segments__",
    index_filename: str = "segments_index.csv",
    pitch_um: float = 2.0,
    origin: bool = True,
    use_cached_image: bool = True,
    enforce_connectivity: bool = False,
    max_workers: int = 8,
    compactness_search_jobs: int = 16,
    compactness_search_downsample_factor: float = 2.0,
    compactness_scale_power: float = 0.3,
    native_identity: bool = True,
    native_tol: float = 1e-8,
    save_compressed: bool = False,
    cache_kwargs: dict[str, Any] | None = None,
    fast_select_kwargs: dict[str, Any] | None = None,
    slic_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Precompute cached-image SLIC segmentations over many settings.

    Two execution modes are supported:

    1. Manual compactness grid mode:
       pass ``compactnesses=[...]`` to evaluate every ``resolution x compactness`` pair.
    2. Auto compactness mode:
       omit ``compactnesses`` and pass ``compactness_candidates=[...]``. Compactness is
       selected once per reference target using ``fast_select_compactness`` and then
       scaled per resolution. Optionally pass ``extra_compactnesses=[...]`` to also
       export fixed-compactness reference segmentations for every resolution in the
       same run (for example GRID-like controls such as ``1e7``).

    Returns
    -------
    pd.DataFrame
        Index table for the computed scales. By default this follows the legacy
        workflow and exports ``segments__*.npz`` plus the index CSV to disk. If
        ``storage`` includes ``"uns"``, the full bundle is also stored in
        ``adata.uns[uns_key]``.
    """
    if str(segment_method) != "image_plot_slic":
        raise ValueError("precompute_multiscale_segments currently supports only segment_method='image_plot_slic'.")
    if not bool(use_cached_image):
        raise ValueError("precompute_multiscale_segments currently requires use_cached_image=True.")
    storage_mode = _normalize_precompute_storage_mode(storage)
    resolutions_use = _normalize_precompute_float_list(resolutions, name="resolutions")
    compactnesses_use = _normalize_precompute_float_list(compactnesses, name="compactnesses")
    compactness_candidates_use = _normalize_precompute_float_list(
        compactness_candidates,
        name="compactness_candidates",
    )
    extra_compactnesses_use = _normalize_precompute_float_list(
        extra_compactnesses,
        name="extra_compactnesses",
    )

    if manual_pairs is not None and compactnesses is not None:
        raise ValueError("Use either manual_pairs or compactnesses, not both.")

    manual_pairs_use: list[tuple[float, float]] = []
    if manual_pairs is not None:
        for pair in manual_pairs:
            if pair is None or len(pair) != 2:
                raise ValueError("manual_pairs must contain (resolution, compactness) pairs.")
            manual_pairs_use.append((float(pair[0]), float(pair[1])))

    manual_mode = bool(manual_pairs_use) or (compactnesses is not None)
    if manual_mode and (not manual_pairs_use) and len(compactnesses_use) == 0:
        raise ValueError("compactnesses must be non-empty when provided.")
    if (not manual_mode) and len(resolutions_use) == 0:
        raise ValueError("resolutions must contain at least one value.")
    if manual_pairs_use and len(resolutions_use) > 0:
        raise ValueError("When manual_pairs is provided, omit resolutions.")
    if (not manual_mode) and len(compactness_candidates_use) == 0:
        raise ValueError(
            "Provide compactnesses for manual mode or compactness_candidates for auto mode."
        )
    if manual_mode and (not manual_pairs_use) and len(resolutions_use) == 0:
        raise ValueError("resolutions must contain at least one value for manual grid mode.")
    if manual_mode and len(extra_compactnesses_use) > 0:
        raise ValueError("extra_compactnesses is only supported in auto compactness mode.")

    cache_kwargs_use = dict(cache_kwargs or {})
    fast_select_kwargs_use = dict(fast_select_kwargs or {})
    slic_kwargs_use = dict(slic_kwargs or {})

    if storage_mode in {"disk", "both"}:
        if out_dir is None:
            raise ValueError("out_dir must be provided when storage includes 'disk'.")
        os.makedirs(out_dir, exist_ok=True)

    coordinate_mode = str(cache_kwargs_use.get("coordinate_mode", "spatial"))
    array_row_key = str(cache_kwargs_use.get("array_row_key", "array_row"))
    array_col_key = str(cache_kwargs_use.get("array_col_key", "array_col"))
    pixel_perfect = bool(cache_kwargs_use.get("pixel_perfect", False))
    pixel_shape = str(cache_kwargs_use.get("pixel_shape", "square"))
    runtime_fill_from_boundary = bool(cache_kwargs_use.get("runtime_fill_from_boundary", False))
    runtime_fill_closing_radius = int(max(0, int(cache_kwargs_use.get("runtime_fill_closing_radius", 1))))
    runtime_fill_external_radius = int(max(0, int(cache_kwargs_use.get("runtime_fill_external_radius", 0))))
    runtime_fill_holes = bool(cache_kwargs_use.get("runtime_fill_holes", True))
    soft_rasterization = bool(cache_kwargs_use.get("soft_rasterization", False))
    resolve_center_collisions = cache_kwargs_use.get("resolve_center_collisions", "auto")
    center_collision_radius = int(cache_kwargs_use.get("center_collision_radius", 2))

    need_cache = (image_cache_key not in adata.uns)
    if not need_cache:
        cache = adata.uns[image_cache_key]
        need_cache = (
            cache.get("embedding_key") != embedding
            or list(cache.get("dimensions", [])) != list(dimensions)
            or (
                cache.get("origin", None) is not None
                and bool(cache.get("origin")) != bool(origin)
            )
        )
        if not need_cache:
            cache_matches_config, _ = _cache_matches_segmentation_raster_config(
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
            need_cache = not bool(cache_matches_config)

    if need_cache:
        if verbose:
            print("[cache] building cached image...", flush=True)
        cache_build_kwargs = dict(cache_kwargs_use)
        cache_build_kwargs.setdefault("origin", bool(origin))
        cache_build_kwargs.setdefault("verbose", False)
        cache_build_kwargs.setdefault("show", False)
        cache_build_kwargs.setdefault("store", "auto")
        cache_build_kwargs.setdefault("store_barcodes", "auto")
        cache_embedding_image(
            adata,
            embedding=embedding,
            dimensions=list(dimensions),
            key=image_cache_key,
            **cache_build_kwargs,
        )

    cache = adata.uns[image_cache_key]
    n_obs = int(adata.n_obs)

    tiles_all = adata.uns["tiles"]
    tiles_for_n = tiles_all
    if "origin" in tiles_all.columns and bool(origin) and (tiles_all["origin"] == 1).any():
        tiles_for_n = tiles_all[tiles_all["origin"] == 1]
    if "barcode" in tiles_for_n.columns:
        tiles_for_n = tiles_for_n.drop_duplicates("barcode")

    save_fn = np.savez_compressed if bool(save_compressed) else np.savez
    native_available = bool(native_identity) and _native_identity_available(cache)

    reference_compactness: dict[float, dict[str, Any]] = {}
    if not manual_mode:
        reference_groups: dict[float, list[float]] = {}
        for res in resolutions_use:
            res_value = float(res)
            if native_available and np.isclose(res_value, float(pitch_um), atol=float(native_tol)):
                continue
            ref_um = float(
                resolve_fast_compactness_reference_target_um(
                    adata,
                    target_segment_um=res_value,
                    pitch_um=float(pitch_um),
                    origin=bool(origin),
                    compactness_search_downsample_factor=float(compactness_search_downsample_factor),
                )
            )
            reference_groups.setdefault(ref_um, []).append(res_value)

        if verbose:
            print(f"[compactness] reference groups: {reference_groups}", flush=True)

        for ref_um in sorted(reference_groups):
            fast_params = dict(fast_select_kwargs_use)
            fast_params.update(
                {
                    "dimensions": list(dimensions),
                    "embedding": embedding,
                    "target_segment_um": float(ref_um),
                    "pitch_um": float(pitch_um),
                    "origin": bool(origin),
                }
            )
            fast_params.setdefault("compactness_candidates", list(compactness_candidates_use))
            fast_params.setdefault("compactness_search_jobs", int(compactness_search_jobs))
            fast_params.setdefault(
                "compactness_search_downsample_factor",
                float(compactness_search_downsample_factor),
            )
            fast_params.setdefault("verbose", bool(verbose))
            sel_ref = fast_select_compactness(adata, **fast_params)
            reference_compactness[float(ref_um)] = {
                "search_target_um": float(sel_ref.get("search_target_segment_um", ref_um)),
                "search_selected_compactness": float(
                    sel_ref.get("search_selected_compactness", sel_ref["selected_compactness"])
                ),
                "selected_compactness": float(sel_ref["selected_compactness"]),
                "selection_result": sel_ref,
            }
            if verbose:
                print(
                    f"[compactness] ref_um={ref_um:.1f}  "
                    f"search_target_um={reference_compactness[float(ref_um)]['search_target_um']:.1f}  "
                    f"search_selected={reference_compactness[float(ref_um)]['search_selected_compactness']:.6g}",
                    flush=True,
                )

    def _pack_record(
        *,
        sid: str,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        payload = dict(meta)
        payload.setdefault("cache_key", str(image_cache_key))
        payload.setdefault("embedding_key", str(embedding))
        record = {
            "scale_id": sid,
            **{
                key: value
                for key, value in payload.items()
                if key not in {"seg_codes", "label_img"}
            },
        }
        return {
            "scale_id": sid,
            "record": record,
            "payload": payload,
        }

    def _segment_manual(resolution: float, compactness: float) -> dict[str, Any]:
        t0 = time.perf_counter()
        sid = _precompute_scale_id(resolution, compactness=compactness)
        requested_n_segments = int(
            n_segments_from_size(
                tiles_for_n,
                target_um=float(resolution),
                pitch_um=float(pitch_um),
            )
        )

        slic_params = dict(slic_kwargs_use)
        slic_params.update(
            {
                "n_segments": int(requested_n_segments),
                "compactness": float(compactness),
                "enforce_connectivity": bool(enforce_connectivity),
                "show_image": False,
                "verbose": False,
                "figsize": None,
                "fig_dpi": None,
            }
        )
        labels_tile, label_img = slic_segmentation_from_cached_image(cache, **slic_params)
        labels_tile, label_img, observed_tile_n_segments = _compress_observed_labels_for_export(
            labels_tile,
            label_img=np.asarray(label_img, dtype=np.int32),
        )
        seg_codes = _cached_tile_labels_to_obs_codes(
            adata,
            cache,
            labels_tile,
            image_cache_key=image_cache_key,
        )
        observed_obs_n_segments = int(np.unique(seg_codes[seg_codes >= 0]).size)
        dt = time.perf_counter() - t0
        return _pack_record(
            sid=sid,
            meta={
                "mode": "manual_grid",
                "resolution": float(resolution),
                "compactness": float(compactness),
                "segment_method": str(segment_method),
                "pitch_um": float(pitch_um),
                "reference_target_um": np.nan,
                "search_target_um": np.nan,
                "reference_compactness": np.nan,
                "scaled_compactness": float(compactness),
                "requested_n_segments": int(requested_n_segments),
                "observed_tile_n_segments": int(observed_tile_n_segments),
                "observed_obs_n_segments": int(observed_obs_n_segments),
                "max_label": int(labels_tile.max(initial=-1)),
                "native_identity": False,
                "origin": bool(origin),
                "use_cached_image": bool(use_cached_image),
                "seconds": float(dt),
                "seg_codes": seg_codes,
                "label_img": label_img,
            },
        )

    def _segment_extra_reference(resolution: float, compactness: float) -> dict[str, Any]:
        completed = _segment_manual(float(resolution), float(compactness))
        payload = dict(completed["payload"])
        payload["mode"] = "extra_reference"
        payload["reference_target_um"] = np.nan
        payload["search_target_um"] = np.nan
        payload["reference_compactness"] = float(compactness)
        payload["scaled_compactness"] = float(compactness)

        record = dict(completed["record"])
        record.update(
            {
                "mode": "extra_reference",
                "reference_target_um": np.nan,
                "search_target_um": np.nan,
                "reference_compactness": float(compactness),
                "scaled_compactness": float(compactness),
            }
        )
        return {
            "scale_id": str(completed["scale_id"]),
            "record": record,
            "payload": payload,
        }

    def _segment_auto(resolution: float) -> dict[str, Any]:
        t0 = time.perf_counter()
        sid = _precompute_scale_id(resolution)

        if native_available and np.isclose(float(resolution), float(pitch_um), atol=float(native_tol)):
            n_tiles = None
            tile_obs_indices = cache.get("tile_obs_indices", None)
            if tile_obs_indices is not None:
                n_tiles = int(np.asarray(tile_obs_indices).size)
            else:
                n_tiles = int(len(cache.get("barcodes", [])))
            labels_tile, label_img = _identity_segmentation_from_cache(cache, n_tiles=int(n_tiles))
            seg_codes = _cached_tile_labels_to_obs_codes(
                adata,
                cache,
                labels_tile,
                image_cache_key=image_cache_key,
            )
            observed_tile_n_segments = int(labels_tile.shape[0])
            observed_obs_n_segments = int(np.unique(seg_codes[seg_codes >= 0]).size)
            dt = time.perf_counter() - t0
            return _pack_record(
                sid=sid,
                meta={
                    "mode": "native_identity",
                    "resolution": float(resolution),
                    "compactness": np.nan,
                    "segment_method": str(segment_method),
                    "pitch_um": float(pitch_um),
                    "reference_target_um": float(resolution),
                    "search_target_um": float(resolution),
                    "reference_compactness": np.nan,
                    "scaled_compactness": np.nan,
                    "requested_n_segments": int(observed_tile_n_segments),
                    "observed_tile_n_segments": int(observed_tile_n_segments),
                    "observed_obs_n_segments": int(observed_obs_n_segments),
                    "max_label": int(labels_tile.max(initial=-1)),
                    "native_identity": True,
                    "origin": bool(origin),
                    "use_cached_image": bool(use_cached_image),
                    "seconds": float(dt),
                    "seg_codes": seg_codes,
                    "label_img": label_img,
                },
            )

        requested_n_segments = int(
            n_segments_from_size(
                tiles_for_n,
                target_um=float(resolution),
                pitch_um=float(pitch_um),
            )
        )
        ref_um = float(
            resolve_fast_compactness_reference_target_um(
                adata,
                target_segment_um=float(resolution),
                pitch_um=float(pitch_um),
                origin=bool(origin),
                compactness_search_downsample_factor=float(compactness_search_downsample_factor),
            )
        )
        ref_info = reference_compactness[ref_um]
        if np.isclose(float(resolution), ref_um):
            scaled_compactness = float(ref_info["selected_compactness"])
        else:
            scaled_compactness = float(
                scale_compactness(
                    ref_info["search_selected_compactness"],
                    from_um=float(ref_info["search_target_um"]),
                    to_um=float(resolution),
                    power=float(compactness_scale_power),
                )
            )

        slic_params = dict(slic_kwargs_use)
        slic_params.update(
            {
                "n_segments": int(requested_n_segments),
                "compactness": float(scaled_compactness),
                "enforce_connectivity": bool(enforce_connectivity),
                "show_image": False,
                "verbose": False,
                "figsize": None,
                "fig_dpi": None,
            }
        )
        labels_tile, label_img = slic_segmentation_from_cached_image(cache, **slic_params)
        labels_tile, label_img, observed_tile_n_segments = _compress_observed_labels_for_export(
            labels_tile,
            label_img=np.asarray(label_img, dtype=np.int32),
        )
        seg_codes = _cached_tile_labels_to_obs_codes(
            adata,
            cache,
            labels_tile,
            image_cache_key=image_cache_key,
        )
        observed_obs_n_segments = int(np.unique(seg_codes[seg_codes >= 0]).size)
        dt = time.perf_counter() - t0
        return _pack_record(
            sid=sid,
            meta={
                "mode": "auto_reference",
                "resolution": float(resolution),
                "compactness": float(scaled_compactness),
                "segment_method": str(segment_method),
                "pitch_um": float(pitch_um),
                "reference_target_um": float(ref_um),
                "search_target_um": float(ref_info["search_target_um"]),
                "reference_compactness": float(ref_info["search_selected_compactness"]),
                "scaled_compactness": float(scaled_compactness),
                "requested_n_segments": int(requested_n_segments),
                "observed_tile_n_segments": int(observed_tile_n_segments),
                "observed_obs_n_segments": int(observed_obs_n_segments),
                "max_label": int(labels_tile.max(initial=-1)),
                "native_identity": False,
                "origin": bool(origin),
                "use_cached_image": bool(use_cached_image),
                "seconds": float(dt),
                "seg_codes": seg_codes,
                "label_img": label_img,
            },
        )

    if manual_mode:
        if manual_pairs_use:
            param_grid = list(manual_pairs_use)
            mode_label = "manual pairs"
        else:
            param_grid = [(float(r), float(c)) for r in resolutions_use for c in compactnesses_use]
            mode_label = "manual grid"
        if verbose:
            print(
                f"▶ segment precompute: {len(param_grid)} {mode_label} settings (threads={int(max_workers)})",
                flush=True,
            )
    else:
        param_grid = [float(r) for r in resolutions_use]
        if verbose:
            print(
                f"▶ segment precompute: {len(param_grid)} auto scales"
                f"{' + ' + str(len(param_grid) * len(extra_compactnesses_use)) + ' fixed references' if len(extra_compactnesses_use) > 0 else ''}"
                f" (threads={int(max_workers)})",
                flush=True,
            )

    results: list[dict[str, Any]] = []
    uns_records: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        future_map = {}
        if manual_mode:
            for res_value, comp_value in param_grid:
                future = executor.submit(_segment_manual, res_value, comp_value)
                future_map[future] = (res_value, comp_value)
        else:
            for res_value in param_grid:
                future = executor.submit(_segment_auto, res_value)
                future_map[future] = res_value
            for res_value in param_grid:
                for comp_value in extra_compactnesses_use:
                    future = executor.submit(_segment_extra_reference, res_value, comp_value)
                    future_map[future] = (res_value, comp_value)

        for future in as_completed(future_map):
            completed = future.result()
            sid = str(completed["scale_id"])
            record = dict(completed["record"])
            payload = dict(completed["payload"])

            if storage_mode in {"uns", "both"}:
                uns_records[sid] = payload

            if storage_mode in {"disk", "both"}:
                path = os.path.join(str(out_dir), f"{filename_prefix}{sid}.npz")
                save_fn(path, **payload)
                record["path"] = path
            else:
                record["path"] = None

            results.append(record)
            if verbose:
                if record["mode"] == "manual_grid":
                    print(
                        f"[done] {record['scale_id']}  "
                        f"compactness={record['compactness']:.6g}  "
                        f"requested={record['requested_n_segments']}  "
                        f"observed_tile={record['observed_tile_n_segments']}  "
                        f"observed_obs={record['observed_obs_n_segments']}  "
                        f"{record['seconds']:.2f}s",
                        flush=True,
                    )
                else:
                    if record["mode"] == "extra_reference":
                        print(
                            f"[done] {record['scale_id']}  "
                            f"fixed_compactness={record['compactness']:.6g}  "
                            f"requested={record['requested_n_segments']}  "
                            f"observed_tile={record['observed_tile_n_segments']}  "
                            f"observed_obs={record['observed_obs_n_segments']}  "
                            f"{record['seconds']:.2f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"[done] {record['scale_id']}  "
                            f"ref_um={record['reference_target_um']:.1f}  "
                            f"compactness={record.get('scaled_compactness', np.nan)}  "
                            f"requested={record['requested_n_segments']}  "
                            f"observed_tile={record['observed_tile_n_segments']}  "
                            f"observed_obs={record['observed_obs_n_segments']}  "
                            f"{record['seconds']:.2f}s",
                            flush=True,
                        )
            gc.collect()

    sort_cols = ["resolution", "compactness"] if ("compactness" in pd.DataFrame(results).columns) else ["resolution"]
    index_df = pd.DataFrame(results).sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if storage_mode in {"disk", "both"}:
        index_csv = os.path.join(str(out_dir), str(index_filename))
        index_df.to_csv(index_csv, index=False)
        if verbose:
            print(f"saved: {index_csv}", flush=True)
    if storage_mode in {"uns", "both"}:
        adata.uns[str(uns_key)] = {
            "index": index_df.copy(),
            "records": uns_records,
            "config": {
                "resolutions": list(resolutions_use),
                "manual_pairs": list(manual_pairs_use),
                "compactnesses": list(compactnesses_use),
                "compactness_candidates": list(compactness_candidates_use),
                "extra_compactnesses": list(extra_compactnesses_use),
                "dimensions": list(dimensions),
                "embedding": str(embedding),
                "segment_method": str(segment_method),
                "image_cache_key": str(image_cache_key),
                "pitch_um": float(pitch_um),
                "origin": bool(origin),
                "storage": str(storage_mode),
                "filename_prefix": str(filename_prefix),
                "index_filename": str(index_filename),
            },
        }
        if verbose:
            print(f"stored: adata.uns['{uns_key}']", flush=True)
    return index_df


def export_precomputed_multiscale_segments(
    source: AnnData | dict[str, Any],
    *,
    uns_key: str = "multiscale_segments",
    out_dir: str = "./multiscale_segments",
    filename_prefix: str | None = None,
    index_filename: str | None = None,
    save_compressed: bool = False,
) -> pd.DataFrame:
    """Export a precomputed multiscale segmentation bundle from memory to disk."""
    if isinstance(source, AnnData):
        if str(uns_key) not in source.uns:
            raise KeyError(f"adata.uns['{uns_key}'] not found.")
        bundle = source.uns[str(uns_key)]
    else:
        bundle = source

    if not isinstance(bundle, dict):
        raise ValueError("source must be an AnnData object or a multiscale segmentation bundle dict.")
    if "index" not in bundle or "records" not in bundle:
        raise KeyError("bundle must contain 'index' and 'records'.")

    bundle_index = bundle["index"]
    bundle_records = bundle["records"]
    config = bundle.get("config", {}) or {}
    if not isinstance(bundle_index, pd.DataFrame):
        bundle_index = pd.DataFrame(bundle_index)
    if not isinstance(bundle_records, dict):
        raise ValueError("bundle['records'] must be a dict keyed by scale_id.")

    filename_prefix_use = str(
        config.get("filename_prefix", "segments__") if filename_prefix is None else filename_prefix
    )
    index_filename_use = str(
        config.get("index_filename", "segments_index.csv") if index_filename is None else index_filename
    )
    save_fn = np.savez_compressed if bool(save_compressed) else np.savez

    os.makedirs(out_dir, exist_ok=True)

    exported_index = bundle_index.copy()
    exported_paths: list[str] = []
    for _, row in exported_index.iterrows():
        sid = str(row["scale_id"])
        if sid not in bundle_records:
            raise KeyError(f"scale_id '{sid}' missing from bundle['records'].")
        payload = bundle_records[sid]
        path = os.path.join(out_dir, f"{filename_prefix_use}{sid}.npz")
        save_fn(path, **payload)
        exported_paths.append(path)

    exported_index["path"] = exported_paths
    index_csv = os.path.join(out_dir, index_filename_use)
    exported_index.to_csv(index_csv, index=False)
    return exported_index


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
    gap_collapse_axis: str | None = None,
    gap_collapse_max_run_px: int = 1,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool | str = "auto",
    center_collision_radius: int = 2,
    mask_background: bool = False,
    masked_slic_mode: str = "strict",
    ignore_cache_param_mismatches: bool = True,
    show_image: bool = False,
    brighten_continuous: bool = False,
    continuous_gamma: float = 0.8,
    target_segment_um: float | None = None,
    pitch_um: float = 2.0,
    n_jobs: int | None = None,
    use_cached_image: bool = False,
    image_cache_key: str = "image_plot_slic",
    cache_storage: str = DEFAULT_CACHE_STORAGE,
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
    gap_collapse_axis : {"x", "y", "both"} or None, optional (default ``None``)
        Collapse short fully-empty raster row/column gaps before
        ``image_plot_slic`` runs.
    gap_collapse_max_run_px : int, optional (default ``1``)
        Maximum empty row/column run length that will be removed when
        ``gap_collapse_axis`` is enabled.
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
    mask_background : bool, optional (default ``False``)
        When ``True``, restrict ``image_plot_slic`` to the observed/fill-completed
        raster support instead of running across the full canvas.
    masked_slic_mode : {"strict", "fast_fill"}, optional (default ``"strict"``)
        Execution mode used when ``mask_background=True`` for
        ``method='image_plot_slic'``. ``"strict"`` runs SLIC with an explicit
        mask for maximum fidelity but can be slow on large masks. ``"fast_fill"``
        fills background pixels from the nearest supported location, runs
        unmasked SLIC, and then discards background labels; this is usually much
        faster while preserving masked behavior approximately.
    ignore_cache_param_mismatches : bool, optional (default ``True``)
        When ``use_cached_image=True``, reuse the cached raster even if raster
        generation settings (fill, pixel snapping, soft rasterization, etc.)
        differ from the current request. A warning is still emitted so the
        effective segmentation input is explicit. Embedding/dimension/origin
        mismatches remain strict and still trigger regeneration.
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
    gap_collapse_axis = _normalize_gap_collapse_axis_safe(gap_collapse_axis)
    gap_collapse_active = bool(
        method == "image_plot_slic"
        and gap_collapse_axis is not None
        and int(max(0, int(gap_collapse_max_run_px))) > 0
    )
    if gap_collapse_active and use_cached_image:
        if verbose:
            print(
                "[warning] gap_collapse is not yet compatible with use_cached_image=True; "
                "regenerating the raster directly."
            )
        use_cached_image = False

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
        cache_raster_reusable = bool(cache_matches_config) or bool(ignore_cache_param_mismatches)
        if (cached_emb is None or cached_emb == embedding) and (
            cached_dims is None or list(cached_dims) == list(dimensions)
        ) and (
            cached_origin is None or bool(cached_origin) == bool(origin)
        ) and cache_raster_reusable:
            if verbose and (not cache_matches_config):
                print(
                    "[warning] Cached image raster settings differ from requested segmentation settings; "
                    f"reusing cache anyway. mismatches={cache_mismatches}"
                )
            if show_image:
                preview_figsize, preview_figdpi = _resolve_cached_preview_size(
                    cache,
                    figsize=figsize,
                    fig_dpi=fig_dpi,
                    img_shape=get_cached_image_shape(cache),
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
                masked_slic_mode=masked_slic_mode,
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
            and (not ignore_cache_param_mismatches)
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
                        gap_collapse_axis=gap_collapse_axis,
                        gap_collapse_max_run_px=gap_collapse_max_run_px,
                        soft_rasterization=soft_rasterization,
                        resolve_center_collisions=resolve_center_collisions,
                        center_collision_radius=center_collision_radius,
                        show_image=False,
                        verbose=False,
                        n_jobs=n_jobs,
                        mask_background=mask_background,
                        masked_slic_mode=masked_slic_mode,
                        persistent_store=getattr(adata, "uns", None),
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
                    gap_collapse_axis=gap_collapse_axis,
                    gap_collapse_max_run_px=gap_collapse_max_run_px,
                    soft_rasterization=soft_rasterization,
                    resolve_center_collisions=resolve_center_collisions,
                    center_collision_radius=center_collision_radius,
                    show_image=show_image,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    mask_background=mask_background,
                    masked_slic_mode=masked_slic_mode,
                    persistent_store=getattr(adata, "uns", None),
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
                    memmap_scope="transient",
                    memmap_cleanup="on_exit",
                    cache_storage="float32_memmap",
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
                        if bool(ignore_cache_param_mismatches):
                            if verbose:
                                print(
                                    "[warning] Cached image raster settings differ from requested segmentation settings; "
                                    f"reusing cache anyway. mismatches={cache_mismatches}"
                                )
                        else:
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
                            img_shape=get_cached_image_shape(cache),
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
                            masked_slic_mode=masked_slic_mode,
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
                        masked_slic_mode=masked_slic_mode,
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
                            gap_collapse_axis=gap_collapse_axis,
                            gap_collapse_max_run_px=gap_collapse_max_run_px,
                            soft_rasterization=soft_rasterization,
                            resolve_center_collisions=resolve_center_collisions,
                            center_collision_radius=center_collision_radius,
                            show_image=False,
                            verbose=False,
                            n_jobs=n_jobs,
                            mask_background=mask_background,
                            masked_slic_mode=masked_slic_mode,
                            persistent_store=getattr(adata, "uns", None),
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
                        gap_collapse_axis=gap_collapse_axis,
                        gap_collapse_max_run_px=gap_collapse_max_run_px,
                        soft_rasterization=soft_rasterization,
                        resolve_center_collisions=resolve_center_collisions,
                        center_collision_radius=center_collision_radius,
                        show_image=show_image,
                        verbose=verbose,
                        n_jobs=n_jobs,
                        brighten_continuous=brighten_continuous,
                        continuous_gamma=continuous_gamma,
                        mask_background=mask_background,
                        masked_slic_mode=masked_slic_mode,
                        persistent_store=getattr(adata, "uns", None),
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
                        release_cached_image(adata, transient_cache_key, remove_memmap_file=True)
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
                        gap_collapse_axis=gap_collapse_axis,
                        gap_collapse_max_run_px=gap_collapse_max_run_px,
                        soft_rasterization=soft_rasterization,
                        resolve_center_collisions=resolve_center_collisions,
                        center_collision_radius=center_collision_radius,
                        show_image=False,
                        verbose=False,
                        n_jobs=n_jobs,
                        mask_background=mask_background,
                        masked_slic_mode=masked_slic_mode,
                        persistent_store=getattr(adata, "uns", None),
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
                    gap_collapse_axis=gap_collapse_axis,
                    gap_collapse_max_run_px=gap_collapse_max_run_px,
                    soft_rasterization=soft_rasterization,
                    resolve_center_collisions=resolve_center_collisions,
                    center_collision_radius=center_collision_radius,
                    show_image=show_image,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    brighten_continuous=brighten_continuous,
                    continuous_gamma=continuous_gamma,
                    mask_background=mask_background,
                    masked_slic_mode=masked_slic_mode,
                    persistent_store=getattr(adata, "uns", None),
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
    gap_collapse_axis: str | None = None,
    gap_collapse_max_run_px: int = 1,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool | str = "auto",
    center_collision_radius: int = 2,
    mask_background: bool = False,
    masked_slic_mode: str = "strict",
    show_image: bool = False,
    fig_dpi: int | None = None,
    n_jobs: int | None = None,
    persistent_store=None,
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
    gap_collapse_axis / gap_collapse_max_run_px : optional
        Optional empty-gap collapsing controls passed through to
        ``image_plot_slic``.
    soft_rasterization : bool, optional
        If True, use bilinear splatting and bilinear label readback for
        ``image_plot_slic``.
    resolve_center_collisions : bool or {"auto"}, optional
        ``True`` always reassigns duplicate raster centers onto nearby empty
        pixels before ``image_plot_slic`` runs. ``"auto"`` only does this for
        dense one-pixel rasters with substantial collisions.
    center_collision_radius : int, optional
        Search radius in raster pixels used when
        ``resolve_center_collisions=True``.
    show_image : bool, optional
        If ``True`` display the constructed image and label image during
        ``image_plot_slic`` segmentation.
    n_jobs : int or None, optional
        Number of worker threads for ``image_plot_slic`` rasterization. When
        ``None``, defers to ``SPIX_IMAGE_PLOT_SLIC_N_JOBS`` env var or 1.
    persistent_store : mapping-like or None, optional
        Optional cache store passed to ``image_plot_slic`` helpers.
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
            gap_collapse_axis=gap_collapse_axis,
            gap_collapse_max_run_px=gap_collapse_max_run_px,
            soft_rasterization=soft_rasterization,
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=center_collision_radius,
            show_image=show_image,
            verbose=verbose,
            n_jobs=n_jobs,
            mask_background=mask_background,
            masked_slic_mode=masked_slic_mode,
            persistent_store=persistent_store,
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
