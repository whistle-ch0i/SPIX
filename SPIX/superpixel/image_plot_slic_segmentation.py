import numpy as np
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import slic
from skimage.transform import resize

from SPIX.visualization import raster as _raster
from SPIX.image_processing.image_cache import get_cached_image


def _normalize_gap_collapse_axis_safe(axis) -> Optional[str]:
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


def _apply_brighten_continuous_rgb(rgb: np.ndarray, continuous_gamma: float) -> np.ndarray:
    """Apply image_plot-style brightening/gamma to RGB arrays."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.shape[-1] != 3:
        return arr
    max_per_pixel = arr.max(axis=-1, keepdims=True)
    scale = np.where(max_per_pixel > 0, 1.0 / max_per_pixel, 0.0)
    arr = np.clip(arr * scale, 0.0, 1.0)
    try:
        gamma = float(continuous_gamma)
        if gamma > 0 and gamma != 1.0:
            arr = np.clip(np.power(arr, gamma), 0.0, 1.0)
    except Exception:
        pass
    return arr


def _resolve_cached_preview_size(
    cache: dict,
    *,
    figsize=None,
    fig_dpi=None,
    img_shape: Optional[tuple[int, ...]] = None,
):
    """Resolve preview figure size for cached-image displays.

    If the caller does not provide `figsize`/`fig_dpi` and the cache does not
    carry explicit preview settings, fall back to the cached raster size using
    `raster_dpi` so notebook output reflects the original raster dimensions
    rather than matplotlib's small default figure.
    """
    resolved_figsize = figsize if figsize is not None else cache.get("figsize", None)
    resolved_dpi = fig_dpi if fig_dpi is not None else cache.get("fig_dpi", None)
    if resolved_dpi is None:
        resolved_dpi = cache.get("raster_dpi", cache.get("effective_dpi", 100))
    if resolved_figsize is None and img_shape is not None:
        try:
            h = int(img_shape[0])
            w = int(img_shape[1])
            dpi_use = float(resolved_dpi) if resolved_dpi is not None else 100.0
            if h > 0 and w > 0 and dpi_use > 0:
                resolved_figsize = (float(w) / dpi_use, float(h) / dpi_use)
        except Exception:
            resolved_figsize = None
    return resolved_figsize, resolved_dpi


def _downsample_support_mask(mask: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape == (int(new_h), int(new_w)):
        return mask_arr
    resized = resize(
        mask_arr.astype(np.float32),
        (int(new_h), int(new_w)),
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    return (resized >= 0.5).astype(bool, copy=False)


def _count_observed_segments(labels: np.ndarray) -> int:
    arr = np.asarray(labels, dtype=np.int64)
    arr = arr[arr >= 0]
    return int(np.unique(arr).size)


def _estimate_internal_n_segments(
    *,
    requested_n_segments: int,
    support_mask: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
) -> tuple[int, dict]:
    target = max(1, int(requested_n_segments))
    mask = np.asarray(support_mask, dtype=bool)
    canvas_pixels = int(mask.size)
    support_pixels = int(np.count_nonzero(mask))
    sx = np.asarray(sample_x, dtype=np.int32).ravel()
    sy = np.asarray(sample_y, dtype=np.int32).ravel()
    n_tiles = int(min(sx.size, sy.size))
    if canvas_pixels <= 0 or support_pixels <= 0 or n_tiles <= 0:
        return target, {
            "requested_n_segments": int(target),
            "estimated_n_segments": int(target),
            "support_fraction": None,
            "unique_center_fraction": None,
        }
    flat = sy.astype(np.int64, copy=False) * np.int64(max(1, mask.shape[1])) + sx.astype(np.int64, copy=False)
    unique_centers = int(np.unique(flat).size)
    support_fraction = float(support_pixels) / float(canvas_pixels)
    unique_center_fraction = float(unique_centers) / float(n_tiles)
    # One-shot estimate is intentionally conservative: observed segment counts
    # are usually a bit lower than plain support-area scaling suggests because
    # some superpixels spill across support gaps/boundaries and disappear at the
    # sampled tile centers. Bias the visible fraction slightly downward so the
    # internal requested count is biased upward without introducing a re-run.
    safety_margin = float(np.clip(float(os.environ.get("SPIX_SEGMENT_TARGET_MARGIN", "0.98")), 0.80, 1.0))
    observed_fraction = float(np.clip(support_fraction * unique_center_fraction * safety_margin, 0.05, 1.0))
    est = int(np.rint(float(target) / float(observed_fraction)))
    est = max(int(target), int(est))
    est = min(int(est), int(canvas_pixels))
    return int(est), {
        "requested_n_segments": int(target),
        "estimated_n_segments": int(est),
        "support_fraction": float(support_fraction),
        "unique_center_fraction": float(unique_center_fraction),
        "safety_margin": float(safety_margin),
    }


def _run_fast_masked_slic(
    *,
    img: np.ndarray,
    support_mask: np.ndarray,
    requested_n_segments: int,
    compactness: float,
    enforce_connectivity: bool,
    sample_labels_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(support_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("support_mask must be a 2D boolean array.")
    invalid = ~mask
    if not np.any(invalid):
        filled_img = np.asarray(img, dtype=np.float32, copy=False)
    else:
        filled_img = np.asarray(img, dtype=np.float32).copy()
        _dist, nearest = distance_transform_edt(invalid, return_indices=True)
        nearest_y = np.asarray(nearest[0], dtype=np.intp)
        nearest_x = np.asarray(nearest[1], dtype=np.intp)
        filled_img[invalid] = filled_img[nearest_y[invalid], nearest_x[invalid]]
    label_img = slic(
        filled_img,
        n_segments=int(requested_n_segments),
        compactness=float(compactness),
        sigma=0,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=bool(enforce_connectivity),
    )
    label_img = np.asarray(label_img, dtype=np.int32)
    label_img[~mask] = -1
    labels = np.asarray(sample_labels_fn(label_img), dtype=np.int32)
    return labels, label_img, filled_img


def _enforce_requested_observed_count(
    labels: np.ndarray,
    *,
    requested_n_segments: int,
    spatial_coords: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    valid = arr >= 0
    if not np.any(valid):
        return arr
    target = max(1, int(requested_n_segments))
    coords = np.asarray(spatial_coords, dtype=np.float32)[valid]
    emb = np.asarray(embeddings, dtype=np.float32)[valid]
    uniq, inv = np.unique(arr[valid], return_inverse=True)
    current = int(uniq.size)
    if current == target:
        return arr

    coord_sd = np.nanstd(coords, axis=0, keepdims=True)
    coord_sd[~np.isfinite(coord_sd) | (coord_sd <= 1e-6)] = 1.0
    coords_n = coords / coord_sd
    emb_use = emb[:, : min(emb.shape[1], 8)] if emb.ndim == 2 else emb[:, None]
    emb_sd = np.nanstd(emb_use, axis=0, keepdims=True)
    emb_sd[~np.isfinite(emb_sd) | (emb_sd <= 1e-6)] = 1.0
    emb_n = emb_use / emb_sd
    feats = np.concatenate([coords_n, emb_n], axis=1)

    def _recompute(curr_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = np.bincount(curr_inv)
        sums = np.zeros((counts.size, feats.shape[1]), dtype=np.float64)
        np.add.at(sums, curr_inv, feats.astype(np.float64))
        centers = sums / np.maximum(counts[:, None], 1.0)
        return counts, centers

    curr_inv = inv.astype(np.int32, copy=False)
    counts, centers = _recompute(curr_inv)

    while counts.size > target:
        src = int(np.argmin(counts))
        if counts.size <= 1:
            break
        other = np.arange(counts.size) != src
        dst_idx = np.flatnonzero(other)
        d = np.sum((centers[dst_idx] - centers[src]) ** 2, axis=1)
        dst = int(dst_idx[int(np.argmin(d))])
        curr_inv[curr_inv == src] = dst
        _, curr_inv = np.unique(curr_inv, return_inverse=True)
        curr_inv = curr_inv.astype(np.int32, copy=False)
        counts, centers = _recompute(curr_inv)

    next_label = int(counts.size)
    while counts.size < target:
        src = int(np.argmax(counts))
        members = np.flatnonzero(curr_inv == src)
        if members.size < 2:
            break
        local = feats[members]
        center = local.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(local - center, full_matrices=False)
        axis = vh[0] if vh.ndim == 2 and vh.shape[0] > 0 else np.zeros(local.shape[1], dtype=np.float32)
        proj = (local - center) @ axis
        right = proj > float(np.median(proj))
        if not np.any(right) or np.all(right):
            order = np.argsort(proj)
            right = np.zeros(members.size, dtype=bool)
            right[order[members.size // 2 :]] = True
        if not np.any(right) or np.all(right):
            break
        curr_inv[members[right]] = int(next_label)
        next_label += 1
        _, curr_inv = np.unique(curr_inv, return_inverse=True)
        curr_inv = curr_inv.astype(np.int32, copy=False)
        counts, centers = _recompute(curr_inv)

    arr[valid] = curr_inv.astype(np.int32, copy=False)
    return arr


def _run_masked_slic_with_target(
    *,
    img: np.ndarray,
    support_mask: np.ndarray | None,
    requested_n_segments: int,
    compactness: float,
    enforce_connectivity: bool,
    sample_labels_fn,
    max_segment_tuning_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    requested = max(1, int(requested_n_segments))
    trial_n = requested
    tried: set[int] = set()
    best_labels = None
    best_label_img = None
    best_error = None

    for _ in range(max(1, int(max_segment_tuning_steps))):
        if trial_n in tried:
            break
        tried.add(trial_n)
        label_img = slic(
            img,
            n_segments=trial_n,
            compactness=float(compactness),
            sigma=0,
            start_label=0,
            mask=support_mask,
            channel_axis=-1,
            enforce_connectivity=bool(enforce_connectivity),
        )
        label_img = np.asarray(label_img, dtype=np.int32)
        if support_mask is not None and support_mask.shape == label_img.shape:
            label_img[~support_mask] = -1
        labels = np.asarray(sample_labels_fn(label_img), dtype=np.int32)
        observed = _count_observed_segments(labels)
        error = abs(int(observed) - requested)
        if best_error is None or error < best_error:
            best_error = error
            best_labels = labels
            best_label_img = label_img
        if observed == requested:
            return labels, label_img

        if observed <= 0:
            next_n = trial_n * 2
        else:
            next_n = int(round(float(trial_n) * float(requested) / float(observed)))
            if observed < requested:
                next_n = max(next_n, trial_n + 1)
            else:
                next_n = min(next_n, trial_n - 1)
        if next_n <= 0:
            break
        if next_n in tried:
            direction = 1 if observed < requested else -1
            next_n = trial_n + direction * max(1, error)
        trial_n = max(1, int(next_n))

    if best_labels is None or best_label_img is None:
        raise RuntimeError("Failed to generate SLIC labels.")
    return best_labels, best_label_img


def image_plot_slic_segmentation(
    embeddings: np.ndarray,
    spatial_coords: np.ndarray,
    n_segments: int = 100,
    compactness: float = 10.0,
    n_points_eff: Optional[int] = None,
    figsize: tuple | None = None,
    fig_dpi: Optional[int] = None,
    imshow_tile_size: Optional[float] = None,
    imshow_scale_factor: float = 1.0,
    imshow_tile_size_mode: str = "auto",
    imshow_tile_size_quantile: Optional[float] = None,
    imshow_tile_size_rounding: str = "floor",
    imshow_tile_size_shrink: float = 0.98,
    pixel_perfect: bool = False,
    enforce_connectivity: bool = False,
    downsample_factor: float = 1.0,
    chunk_size: int = 50000,
    *,
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
    show_image: bool = False,
    verbose: bool = True,
    n_jobs: Optional[int] = None,
    brighten_continuous: bool = False,
    continuous_gamma: float = 0.8,
    mask_background: bool = False,
    masked_slic_mode: str = "strict",
    match_requested_segments: bool = False,
    max_segment_tuning_steps: int = 4,
    return_raster_meta: bool = False,
    persistent_store=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """CPU-based SLIC segmentation on embeddings with chunked image creation.

    Parameters
    ----------
    pixel_shape : str, optional (default ``"square"``)
        Shape used when rasterizing the points. ``"circle"`` mimics Visium style
        spots while ``"square"`` retains the previous behaviour.
    fig_dpi : int or None, optional (default ``None``)
        DPI to use for the internal buffer sizing and optional figures.
        When ``None``, uses 100 DPI for buffer sizing.
    show_image : bool, optional (default ``False``)
        If ``True`` display the intermediate constructed image and the
        segmentation label image.
    n_jobs : int or None, optional (default ``None``)
        Reserved for potential parallel rasterization; currently unused (no-op).
    runtime_fill_from_boundary : bool, optional (default ``False``)
        If ``True``, preserve the original spot centers but morphologically fill
        empty raster pixels inside the observed support before running SLIC.
        This avoids coordinate snapping and does not require a user-provided pitch.
    runtime_fill_closing_radius : int, optional (default ``1``)
        Pixel-radius used for binary closing of the raster occupancy mask when
        ``runtime_fill_from_boundary=True``.
    runtime_fill_holes : bool, optional (default ``True``)
        Whether to fill enclosed holes in the raster occupancy mask when
        ``runtime_fill_from_boundary=True``.
    gap_collapse_axis : {"x", "y", "both"} or None, optional (default ``None``)
        Collapse short fully-empty raster row/column gaps before SLIC.
    gap_collapse_max_run_px : int, optional (default ``1``)
        Maximum empty run length that will be removed when
        ``gap_collapse_axis`` is enabled.
    soft_rasterization : bool, optional (default ``False``)
        If ``True``, rasterize each spot with bilinear splatting and read SLIC
        labels back with bilinear voting instead of hard one-pixel assignment.
        Currently supported only for ``pixel_perfect=True``, ``tile_px==1`` and
        ``pixel_shape='square'``.
    resolve_center_collisions : bool or {"auto"}, optional (default ``"auto"``)
        ``True`` always reassigns duplicate raster centers onto nearby free
        pixels before painting/fill/label lookup. ``"auto"`` applies the same
        step only for dense one-pixel rasters with substantial collisions.
    center_collision_radius : int, optional (default ``2``)
        Search radius in raster pixels used when
        ``resolve_center_collisions=True``.
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if spatial_coords.shape[0] != embeddings.shape[0]:
        raise ValueError("Number of coordinates must match embeddings.")

    num_tiles = embeddings.shape[0]
    dims = embeddings.shape[1]

    x_min, x_max = spatial_coords[:, 0].min(), spatial_coords[:, 0].max()
    y_min, y_max = spatial_coords[:, 1].min(), spatial_coords[:, 1].max()
    log = logger if verbose else None
    raster_canvas = _raster.resolve_raster_canvas(
        pts=spatial_coords,
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        figsize=figsize,
        fig_dpi=fig_dpi,
        imshow_tile_size=imshow_tile_size,
        imshow_scale_factor=imshow_scale_factor,
        imshow_tile_size_mode=imshow_tile_size_mode,
        imshow_tile_size_quantile=imshow_tile_size_quantile,
        imshow_tile_size_rounding=imshow_tile_size_rounding,
        imshow_tile_size_shrink=imshow_tile_size_shrink,
        pixel_perfect=bool(pixel_perfect),
        n_points=int(n_points_eff) if n_points_eff is not None else int(spatial_coords.shape[0]),
        logger=log,
        context="image_plot_slic_segmentation",
    )
    figsize = raster_canvas["figsize"]
    fig_dpi = raster_canvas["fig_dpi"]
    dpi = int(raster_canvas["dpi_in"])
    s_data_units = float(raster_canvas["s_data_units"])
    x_min_eff = float(raster_canvas["x_min_eff"])
    x_max_eff = float(raster_canvas["x_max_eff"])
    y_min_eff = float(raster_canvas["y_min_eff"])
    y_max_eff = float(raster_canvas["y_max_eff"])
    scale = float(raster_canvas["scale"])
    s = int(raster_canvas["tile_px"])
    w = int(raster_canvas["w"])
    h = int(raster_canvas["h"])

    scaler = MinMaxScaler()
    cols = scaler.fit_transform(embeddings.astype(np.float32))

    dx, dy = _raster.tile_pixel_offsets(int(s), pixel_shape=pixel_shape)

    soft_enabled = bool(soft_rasterization)
    if soft_enabled and (not bool(pixel_perfect) or int(s) != 1 or str(pixel_shape).lower() != "square"):
        if log is not None:
            log.warning(
                "soft_rasterization requires pixel_perfect=True, tile_px==1, and pixel_shape='square'; falling back to hard rasterization."
            )
        soft_enabled = False
    if soft_enabled:
        cx_seed, cy_seed, soft_fx_seed, soft_fy_seed = _raster.scale_points_to_canvas_soft(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=True,
            w=int(w),
            h=int(h),
        )
    else:
        soft_fx_seed = None
        soft_fy_seed = None
        cx_seed, cy_seed = _raster.scale_points_to_canvas(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=bool(pixel_perfect),
            w=int(w),
            h=int(h),
        )
    w_output = int(w)
    h_output = int(h)
    cx_seed_output = np.asarray(cx_seed, dtype=np.int32, copy=True)
    cy_seed_output = np.asarray(cy_seed, dtype=np.int32, copy=True)
    collision_info = None
    apply_collision_resolve, collision_stats, _ = _raster.should_resolve_center_collisions(
        resolve_center_collisions,
        cx=cx_seed,
        cy=cy_seed,
        w=int(w),
        h=int(h),
        tile_px=int(s),
        pixel_perfect=bool(pixel_perfect),
        soft_enabled=bool(soft_enabled),
        logger=log,
        context="image_plot_slic_segmentation",
    )
    if apply_collision_resolve:
        collision_radius_eff = _raster.resolve_collision_search_radius(
            resolve_center_collisions,
            base_radius=int(center_collision_radius),
            collision_stats=collision_stats,
            logger=log,
            context="image_plot_slic_segmentation",
        )
        cx_seed, cy_seed, collision_info = _raster.resolve_center_collisions(
            cx_seed,
            cy_seed,
            w=int(w),
            h=int(h),
            max_radius=int(collision_radius_eff),
            logger=log,
            context="image_plot_slic_segmentation",
        )
    elif collision_stats is not None:
        collision_info = {
            "applied": False,
            "enabled": _raster.normalize_collision_mode(resolve_center_collisions) != "never",
            "search_radius": int(max(0, center_collision_radius)),
            "n_tiles": int(collision_stats["n_tiles"]),
            "n_unique_centers_before": int(collision_stats["n_unique_centers"]),
            "n_unique_centers_after": int(collision_stats["n_unique_centers"]),
            "collision_fraction_before": float(collision_stats["collision_fraction"]),
            "collision_fraction_after": float(collision_stats["collision_fraction"]),
            "n_collided_tiles_before": int(collision_stats["n_collided_tiles"]),
            "n_collided_tiles_after": int(collision_stats["n_collided_tiles"]),
            "n_reassigned_tiles": 0,
            "max_stack_before": int(collision_stats["max_stack"]),
            "max_stack_after": int(collision_stats["max_stack"]),
            "unresolved_tiles": int(collision_stats["n_collided_tiles"]),
        }

    gap_collapse_axis_norm = _normalize_gap_collapse_axis_safe(gap_collapse_axis)
    gap_collapse_active = bool(gap_collapse_axis_norm is not None and int(max(0, int(gap_collapse_max_run_px))) > 0)
    gap_collapse_info = {
        "applied": False,
        "axis": gap_collapse_axis_norm,
        "max_gap_run_px": int(max(0, int(gap_collapse_max_run_px))),
        "rows_removed": 0,
        "cols_removed": 0,
        "row_runs_removed": 0,
        "col_runs_removed": 0,
        "w_before": int(w),
        "h_before": int(h),
        "w_after": int(w),
        "h_after": int(h),
    }
    if gap_collapse_active and soft_enabled:
        if log is not None:
            log.warning(
                "gap_collapse is currently incompatible with soft_rasterization; falling back to hard rasterization."
            )
        soft_enabled = False
        soft_fx_seed = None
        soft_fy_seed = None
    if gap_collapse_active:
        cx_seed, cy_seed, w, h, gap_collapse_info = _raster.collapse_empty_center_gaps(
            cx=cx_seed,
            cy=cy_seed,
            w=int(w),
            h=int(h),
            axis=gap_collapse_axis_norm,
            max_gap_run_px=int(gap_collapse_max_run_px),
            logger=log,
            context="image_plot_slic_segmentation",
        )
    if log is not None:
        log.info("Creating image with chunking (chunk size: %d)...", int(chunk_size))
    img = np.ones((h, w, dims), dtype=np.float32)
    paint_mask = np.zeros((h, w), dtype=bool)

    if soft_enabled:
        _raster.rasterize_soft_bilinear(
            img=img,
            occupancy_mask=paint_mask,
            seed_values=cols,
            cx=cx_seed,
            cy=cy_seed,
            fx=soft_fx_seed,
            fy=soft_fy_seed,
            chunk_size=int(chunk_size),
        )
    else:
        for i in range(0, num_tiles, chunk_size):
            chunk_end = min(i + chunk_size, num_tiles)
            chunk_indices = np.arange(i, chunk_end)
            chunk_cx = cx_seed[chunk_indices]
            chunk_cy = cy_seed[chunk_indices]
            chunk_all_x = np.clip((chunk_cx[:, None] + dx[None, :]).ravel(), 0, w - 1)
            chunk_all_y = np.clip((chunk_cy[:, None] + dy[None, :]).ravel(), 0, h - 1)
            chunk_tile_indices_map = np.repeat(chunk_indices, len(dx))
            img[chunk_all_y, chunk_all_x] = cols[chunk_tile_indices_map]
            paint_mask[chunk_all_y, chunk_all_x] = True

    if log is not None:
        log.info("Image creation complete.")

    fill_active = bool(runtime_fill_from_boundary) or bool(runtime_fill_holes)
    fill_radius = int(runtime_fill_closing_radius) if bool(runtime_fill_from_boundary) else 0
    external_radius = int(runtime_fill_external_radius) if bool(runtime_fill_from_boundary) else 0
    if fill_active:
        _raster.fill_raster_from_boundary(
            img=img,
            occupancy_mask=paint_mask,
            seed_x=cx_seed,
            seed_y=cy_seed,
            seed_values=cols,
            closing_radius=int(fill_radius),
            external_radius=int(external_radius),
            fill_holes=bool(runtime_fill_holes),
            use_observed_pixels=bool(int(s) == 1),
            persistent_store=persistent_store,
            logger=log,
            context="image_plot_slic_segmentation",
        )
    support_mask = np.asarray(paint_mask, dtype=bool)

    if downsample_factor > 1.0:
        if log is not None:
            log.info("Downsampling image by a factor of %.3f...", float(downsample_factor))
        new_h, new_w = int(h / downsample_factor), int(w / downsample_factor)
        img = resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
        support_mask = _downsample_support_mask(support_mask, new_h, new_w)
        scale /= downsample_factor
        w, h = new_w, new_h
        if soft_enabled:
            cx, cy, soft_fx, soft_fy = _raster.scale_points_to_canvas_soft(
                spatial_coords,
                x_min_eff=float(x_min_eff),
                y_min_eff=float(y_min_eff),
                scale=float(scale),
                pixel_perfect=True,
                w=int(w),
                h=int(h),
            )
        elif gap_collapse_active or (collision_info is not None and bool(collision_info.get("applied", False))):
            factor = int(max(1, round(float(downsample_factor))))
            cx = np.clip((cx_seed // factor).astype(np.int32, copy=False), 0, max(0, w - 1))
            cy = np.clip((cy_seed // factor).astype(np.int32, copy=False), 0, max(0, h - 1))
        else:
            cx, cy = _raster.scale_points_to_canvas(
                spatial_coords,
                x_min_eff=float(x_min_eff),
                y_min_eff=float(y_min_eff),
                scale=float(scale),
                pixel_perfect=bool(pixel_perfect),
                w=int(w),
                h=int(h),
            )
        if log is not None:
            log.info("Downsampling complete.")
            if collision_info is not None:
                log.info(
                    "image_plot_slic_segmentation: center collision fraction %.4f -> %.4f.",
                    float(collision_info["collision_fraction_before"]),
                    float(collision_info["collision_fraction_after"]),
                )
    else:
        cx, cy = cx_seed, cy_seed
        soft_fx, soft_fy = soft_fx_seed, soft_fy_seed

    slic_img = img
    slic_mask = support_mask if bool(mask_background) else None
    slic_cx = np.asarray(cx, dtype=np.int32)
    slic_cy = np.asarray(cy, dtype=np.int32)
    if log is not None:
        log.info("Running CPU segmentation backend...")
    if soft_enabled:
        sample_labels_fn = lambda arr: _raster.sample_labels_soft_bilinear(
            arr,
            slic_cx,
            slic_cy,
            soft_fx,
            soft_fy,
            chunk_size=max(100_000, int(chunk_size)),
        ).astype(int, copy=False)
    else:
        sample_labels_fn = lambda arr: arr[slic_cy, slic_cx].astype(int)
    masked_mode = str(masked_slic_mode or "strict").lower()
    if masked_mode not in {"strict", "fast_fill"}:
        masked_mode = "strict"
    if bool(match_requested_segments) or (slic_mask is not None and masked_mode == "strict"):
        label_img = slic(
            slic_img,
            n_segments=int(n_segments),
            compactness=float(compactness),
            sigma=0,
            start_label=0,
            mask=slic_mask,
            channel_axis=-1,
            enforce_connectivity=bool(enforce_connectivity),
        )
        label_img = np.asarray(label_img, dtype=np.int32)
        if slic_mask is not None and slic_mask.shape == label_img.shape:
            label_img[~slic_mask] = -1
        raw_labels = np.asarray(sample_labels_fn(label_img), dtype=np.int32)
        display_source_img = slic_img
        labels = _enforce_requested_observed_count(
            raw_labels,
            requested_n_segments=int(n_segments),
            spatial_coords=np.asarray(spatial_coords, dtype=np.float32),
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )
        if log is not None:
            log.info(
                "Adjusted observed segment count to %d (raw=%d, adjusted=%d).",
                int(n_segments),
                int(_count_observed_segments(np.asarray(raw_labels, dtype=np.int32))),
                int(_count_observed_segments(labels)),
            )
    else:
        if slic_mask is not None:
            if log is not None:
                log.info("Running fast masked SLIC via nearest-support fill...")
            labels, label_img, display_source_img = _run_fast_masked_slic(
                img=slic_img,
                support_mask=slic_mask,
                requested_n_segments=int(n_segments),
                compactness=float(compactness),
                enforce_connectivity=bool(enforce_connectivity),
                sample_labels_fn=sample_labels_fn,
            )
        else:
            run_n_segments, est_info = _estimate_internal_n_segments(
                requested_n_segments=int(n_segments),
                support_mask=support_mask,
                sample_x=slic_cx,
                sample_y=slic_cy,
            )
            if log is not None and int(run_n_segments) != int(n_segments):
                log.info(
                    "Estimated internal n_segments for unmasked SLIC: requested=%d -> internal=%d | support_fraction=%.4f | unique_center_fraction=%.4f",
                    int(n_segments),
                    int(run_n_segments),
                    float(est_info["support_fraction"]),
                    float(est_info["unique_center_fraction"]),
                )
            label_img = slic(
                slic_img,
                n_segments=int(run_n_segments),
                compactness=compactness,
                sigma=0,
                start_label=0,
                mask=None,
                channel_axis=-1,
                enforce_connectivity=enforce_connectivity,
            )
            label_img = np.asarray(label_img, dtype=np.int32)
            labels = np.asarray(sample_labels_fn(label_img), dtype=np.int32)
            display_source_img = slic_img
    if log is not None:
        log.info("CPU SLIC complete.")
    display_label_img = label_img
    support_mask_output = np.asarray(support_mask, dtype=bool, copy=True)
    if gap_collapse_active:
        label_img_output = np.full((int(h_output), int(w_output)), -1, dtype=np.int32)
        support_mask_output = np.zeros((int(h_output), int(w_output)), dtype=bool)
        tile_value_count = int(len(dx))
        for i in range(0, num_tiles, chunk_size):
            chunk_end = min(i + chunk_size, num_tiles)
            chunk_indices = np.arange(i, chunk_end)
            out_x = np.clip((cx_seed_output[chunk_indices, None] + dx[None, :]).ravel(), 0, max(0, int(w_output) - 1))
            out_y = np.clip((cy_seed_output[chunk_indices, None] + dy[None, :]).ravel(), 0, max(0, int(h_output) - 1))
            tile_labels = np.repeat(np.asarray(labels[chunk_indices], dtype=np.int32), tile_value_count)
            label_img_output[out_y, out_x] = tile_labels
            support_mask_output[out_y, out_x] = True
        label_img = label_img_output
    else:
        label_img = np.asarray(label_img, dtype=np.int32)
    if show_image:
        import matplotlib.pyplot as plt

        # ``matplotlib`` only supports 1, 3 or 4 channel images. Reduce or
        # expand the embedding image so it is displayable.
        display_img = np.asarray(display_source_img)
        if display_source_img.shape[2] > 3:
            display_img = display_img[:, :, :3]
        elif display_source_img.shape[2] < 3:
            gray = display_img.mean(axis=2, keepdims=True)
            display_img = np.repeat(gray, 3, axis=2)
        if bool(brighten_continuous):
            display_img = _apply_brighten_continuous_rgb(display_img, float(continuous_gamma))

        plt.figure(figsize=figsize, dpi=fig_dpi if fig_dpi is not None else None)
        plt.imshow(display_img, origin="lower", interpolation="nearest")
        plt.title("Embeddings Image")
        plt.axis("off")
        plt.show()

        from matplotlib.colors import ListedColormap

        valid_labels = display_label_img[display_label_img >= 0]
        max_label = int(valid_labels.max()) if valid_labels.size > 0 else 0
        rng = np.random.default_rng(0)
        rand_cmap = ListedColormap(rng.random((max_label + 1, 3)))
        display_labels = np.ma.masked_where(display_label_img < 0, display_label_img)

        plt.figure(figsize=figsize, dpi=fig_dpi if fig_dpi is not None else None)
        plt.imshow(display_labels, origin="lower", cmap="nipy_spectral", interpolation="nearest")
        plt.imshow(display_labels, origin="lower", cmap=rand_cmap, vmin=0, vmax=max_label, interpolation="nearest")
        
        plt.title("SLIC Segmentation")
        plt.axis("off")
        plt.show()

    raster_meta = {
        "x_min_eff": float(x_min_eff),
        "x_max_eff": float(x_max_eff),
        "y_min_eff": float(y_min_eff),
        "y_max_eff": float(y_max_eff),
        "w": int(label_img.shape[1]),
        "h": int(label_img.shape[0]),
        "support_mask": np.asarray(support_mask_output, dtype=bool, copy=True),
        "mask_background": bool(mask_background),
        "runtime_fill_from_boundary": bool(runtime_fill_from_boundary),
        "runtime_fill_closing_radius": int(runtime_fill_closing_radius),
        "runtime_fill_external_radius": int(runtime_fill_external_radius),
        "runtime_fill_holes": bool(runtime_fill_holes),
        "resolve_center_collisions": _raster.normalize_collision_mode(resolve_center_collisions),
        "center_collision_radius": int(center_collision_radius),
        "pixel_shape": str(pixel_shape),
        "pixel_perfect": bool(pixel_perfect),
        "gap_collapse_axis": gap_collapse_axis_norm,
        "gap_collapse_max_run_px": int(max(0, int(gap_collapse_max_run_px))),
        "gap_collapse_info": dict(gap_collapse_info),
    }
    if return_raster_meta:
        return labels, label_img, display_source_img, raster_meta
    return labels, label_img, display_source_img


def slic_segmentation_from_cached_image(
    cache: dict,
    *,
    n_segments: int = 100,
    compactness: float = 10.0,
    enforce_connectivity: bool = False,
    show_image: bool = False,
    verbose: bool = True,
    figsize: Optional[tuple] = None,
    fig_dpi: Optional[int] = None,
    brighten_continuous: bool | None = None,
    continuous_gamma: float | None = None,
    mask_background: bool = False,
    masked_slic_mode: str = "strict",
    match_requested_segments: bool = False,
    max_segment_tuning_steps: int = 4,
    return_raster_meta: bool = False,
    sample_take: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """Run SLIC on a precomputed cached image dict.

    Expects a dict as produced by ``image_processing.image_cache.cache_embedding_image``.
    Returns tile-level labels (aligned to the cached tile order) and the label image.
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    t_load0 = time.perf_counter()
    img = np.asarray(get_cached_image(cache))
    if img.ndim != 3:
        raise ValueError("Cached image must have shape (H, W, C).")

    support_mask = cache.get("support_mask", None)
    support_mask_arr = None if support_mask is None else np.asarray(support_mask, dtype=bool)
    t_load1 = time.perf_counter()
    if verbose:
        logger.info(
            "Loaded cached image and support mask in %.3fs (img_shape=%s, has_mask=%s).",
            float(t_load1 - t_load0),
            tuple(int(x) for x in img.shape),
            bool(support_mask_arr is not None),
        )

    cx = np.asarray(cache.get("cx"), dtype=int)
    cy = np.asarray(cache.get("cy"), dtype=int)
    if cx.size == 0 or cy.size == 0:
        raise ValueError("Cached image missing per-tile sampling indices 'cx','cy'.")
    if bool(cache.get("soft_rasterization", False)):
        soft_fx = cache.get("soft_fx", None)
        soft_fy = cache.get("soft_fy", None)
        if soft_fx is None or soft_fy is None:
            raise ValueError("Cached soft-raster image is missing 'soft_fx'/'soft_fy' metadata.")
        sample_labels_fn = lambda arr: _raster.sample_labels_soft_bilinear(
            arr,
            cx,
            cy,
            np.asarray(soft_fx, dtype=np.float32),
            np.asarray(soft_fy, dtype=np.float32),
        ).astype(int, copy=False)
    else:
        sample_labels_fn = lambda arr: arr[cy, cx].astype(int)

    slic_mask = support_mask_arr if bool(mask_background) else None
    slic_img = img
    slic_cx = np.asarray(cx, dtype=np.int32)
    slic_cy = np.asarray(cy, dtype=np.int32)
    sample_idx = None if sample_take is None else np.asarray(sample_take, dtype=np.int64).ravel()
    if sample_idx is None:
        sample_cx = slic_cx
        sample_cy = slic_cy
    else:
        sample_cx = slic_cx[sample_idx]
        sample_cy = slic_cy[sample_idx]
    if bool(cache.get("soft_rasterization", False)):
        soft_fx_arr = np.asarray(soft_fx, dtype=np.float32)
        soft_fy_arr = np.asarray(soft_fy, dtype=np.float32)
        sample_soft_fx = soft_fx_arr if sample_idx is None else soft_fx_arr[sample_idx]
        sample_soft_fy = soft_fy_arr if sample_idx is None else soft_fy_arr[sample_idx]
    masked_mode = str(masked_slic_mode or "strict").lower()
    if masked_mode not in {"strict", "fast_fill"}:
        masked_mode = "strict"
    if bool(match_requested_segments) or (slic_mask is not None and masked_mode == "strict"):
        if verbose and slic_mask is not None:
            logger.info(
                "Running cached masked SLIC in strict mode (mask loaded from cache; support_pixels=%d).",
                int(np.count_nonzero(slic_mask)),
            )
        if bool(cache.get("soft_rasterization", False)):
            sample_labels_fn = lambda arr: _raster.sample_labels_soft_bilinear(
                arr,
                sample_cx,
                sample_cy,
                sample_soft_fx,
                sample_soft_fy,
            ).astype(int, copy=False)
        else:
            sample_labels_fn = lambda arr: arr[sample_cy, sample_cx].astype(int)
        if bool(cache.get("soft_rasterization", False)):
            sample_labels_fn = lambda arr: _raster.sample_labels_soft_bilinear(
                arr,
                sample_cx,
                sample_cy,
                sample_soft_fx,
                sample_soft_fy,
            ).astype(int, copy=False)
        else:
            sample_labels_fn = lambda arr: arr[sample_cy, sample_cx].astype(int)
        t_slic0 = time.perf_counter()
        label_img = slic(
            slic_img,
            n_segments=int(n_segments),
            compactness=float(compactness),
            sigma=0,
            start_label=0,
            mask=slic_mask,
            channel_axis=-1,
            enforce_connectivity=bool(enforce_connectivity),
        )
        t_slic1 = time.perf_counter()
        label_img = np.asarray(label_img, dtype=np.int32)
        if slic_mask is not None and slic_mask.shape == label_img.shape:
            label_img[~slic_mask] = -1
        raw_labels = np.asarray(sample_labels_fn(label_img), dtype=np.int32)
        display_source_img = slic_img
        point_features = np.asarray(slic_img[sample_cy, sample_cx], dtype=np.float32)
        t_adjust0 = time.perf_counter()
        labels = _enforce_requested_observed_count(
            raw_labels,
            requested_n_segments=int(n_segments),
            spatial_coords=np.column_stack([sample_cx, sample_cy]).astype(np.float32, copy=False),
            embeddings=point_features,
        )
        t_adjust1 = time.perf_counter()
        if verbose:
            logger.info(
                "Adjusted observed cached segment count to %d (raw=%d, adjusted=%d).",
                int(n_segments),
                int(_count_observed_segments(np.asarray(raw_labels, dtype=np.int32))),
                int(_count_observed_segments(labels)),
            )
            logger.info(
                "Cached strict masked SLIC timings: slic=%.3fs | adjust=%.3fs.",
                float(t_slic1 - t_slic0),
                float(t_adjust1 - t_adjust0),
            )
    else:
        if bool(cache.get("soft_rasterization", False)):
            sample_labels_fn = lambda arr: _raster.sample_labels_soft_bilinear(
                arr,
                sample_cx,
                sample_cy,
                sample_soft_fx,
                sample_soft_fy,
            ).astype(int, copy=False)
        else:
            sample_labels_fn = lambda arr: arr[sample_cy, sample_cx].astype(int)
        if slic_mask is not None:
            if verbose:
                logger.info(
                    "Running fast masked cached SLIC via nearest-support fill (mask loaded from cache; support_pixels=%d).",
                    int(np.count_nonzero(slic_mask)),
                )
            t_fast0 = time.perf_counter()
            labels, label_img, display_source_img = _run_fast_masked_slic(
                img=slic_img,
                support_mask=slic_mask,
                requested_n_segments=int(n_segments),
                compactness=float(compactness),
                enforce_connectivity=bool(enforce_connectivity),
                sample_labels_fn=sample_labels_fn,
            )
            t_fast1 = time.perf_counter()
            if verbose:
                logger.info("Cached fast masked SLIC timing: total=%.3fs.", float(t_fast1 - t_fast0))
        else:
            run_n_segments, est_info = _estimate_internal_n_segments(
                requested_n_segments=int(n_segments),
                support_mask=support_mask_arr if support_mask_arr is not None else np.ones(slic_img.shape[:2], dtype=bool),
                sample_x=slic_cx,
                sample_y=slic_cy,
            )
            if verbose and int(run_n_segments) != int(n_segments):
                logger.info(
                    "Estimated internal cached n_segments for unmasked SLIC: requested=%d -> internal=%d | support_fraction=%.4f | unique_center_fraction=%.4f",
                    int(n_segments),
                    int(run_n_segments),
                    float(est_info["support_fraction"]),
                    float(est_info["unique_center_fraction"]),
                )
            t_slic0 = time.perf_counter()
            label_img = slic(
                slic_img,
                n_segments=int(run_n_segments),
                compactness=float(compactness),
                sigma=0,
                start_label=0,
                mask=None,
                channel_axis=-1,
                enforce_connectivity=bool(enforce_connectivity),
            )
            t_slic1 = time.perf_counter()
            label_img = np.asarray(label_img, dtype=np.int32)
            labels = np.asarray(sample_labels_fn(label_img), dtype=np.int32)
            display_source_img = slic_img
            if verbose:
                logger.info("Cached unmasked SLIC timing: slic=%.3fs.", float(t_slic1 - t_slic0))

    if show_image:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        display_img = np.asarray(display_source_img)
        if display_source_img.shape[2] > 3:
            display_img = display_img[:, :, :3]
        elif display_source_img.shape[2] < 3:
            gray = display_img.mean(axis=2, keepdims=True)
            display_img = np.repeat(gray, 3, axis=2)

        # Prefer provided figsize/fig_dpi; else use cached values only when the
        # user explicitly set them during cache creation.
        _figsize, _dpi = _resolve_cached_preview_size(
            cache,
            figsize=figsize,
            fig_dpi=fig_dpi,
            img_shape=display_img.shape,
        )
        plot_params = cache.get("image_plot_params", {}) or {}
        apply_brighten = (
            bool(brighten_continuous)
            if brighten_continuous is not None
            else bool(plot_params.get("brighten_continuous", False))
        )
        gamma = (
            float(continuous_gamma)
            if continuous_gamma is not None
            else float(plot_params.get("continuous_gamma", 0.8))
        )
        cache_channels = int(cache.get("channels", img.shape[2]))
        already_applied = bool(plot_params.get("brighten_applied", False)) and cache_channels == 3
        if apply_brighten and not already_applied:
            display_img = _apply_brighten_continuous_rgb(display_img, gamma)
        plt.figure(figsize=_figsize, dpi=_dpi)
        plt.imshow(display_img, origin="lower")
        plt.title("Cached Embeddings Image")
        plt.axis("off")
        plt.show()

        valid_labels = label_img[label_img >= 0]
        max_label = int(valid_labels.max()) if valid_labels.size > 0 else 0
        rng = np.random.default_rng(0)
        rand_cmap = ListedColormap(rng.random((max_label + 1, 3)))
        display_labels = np.ma.masked_where(label_img < 0, label_img)
        plt.figure(figsize=_figsize, dpi=_dpi)
        plt.imshow(display_labels, origin="lower", cmap=rand_cmap, vmin=0, vmax=max_label)
        plt.title("SLIC Segmentation (cached)")
        plt.axis("off")
        plt.show()

    raster_meta = {
        "x_min_eff": float(cache.get("x_min_eff", cache.get("x_min", 0.0))),
        "x_max_eff": float(cache.get("x_max_eff", cache.get("x_max", float(img.shape[1])))),
        "y_min_eff": float(cache.get("y_min_eff", cache.get("y_min", 0.0))),
        "y_max_eff": float(cache.get("y_max_eff", cache.get("y_max", float(img.shape[0])))),
        "w": int(img.shape[1]),
        "h": int(img.shape[0]),
        "support_mask": None if support_mask_arr is None else np.asarray(support_mask_arr, dtype=bool, copy=True),
        "mask_background": bool(mask_background),
        "runtime_fill_from_boundary": bool((cache.get("image_plot_params", {}) or {}).get("runtime_fill_from_boundary", False)),
        "runtime_fill_closing_radius": int((cache.get("image_plot_params", {}) or {}).get("runtime_fill_closing_radius", 0)),
        "runtime_fill_external_radius": int((cache.get("image_plot_params", {}) or {}).get("runtime_fill_external_radius", 0)),
        "runtime_fill_holes": bool((cache.get("image_plot_params", {}) or {}).get("runtime_fill_holes", False)),
        "resolve_center_collisions": bool((cache.get("image_plot_params", {}) or {}).get("resolve_center_collisions", False)),
        "center_collision_radius": int((cache.get("image_plot_params", {}) or {}).get("center_collision_radius", 0)),
        "pixel_shape": str(cache.get("pixel_shape", "square")),
        "pixel_perfect": bool(cache.get("pixel_perfect", False)),
    }
    if return_raster_meta:
        return labels, label_img, raster_meta
    return labels, label_img
