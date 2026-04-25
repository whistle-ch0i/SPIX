# -*- coding: utf-8 -*-
# Full code with alpha-driven draw priority so less transparent (more opaque) spots render on top.

import os
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from shapely.geometry import (
    MultiPoint,
    Polygon,
    LineString,
    MultiPolygon,
    GeometryCollection,
    MultiLineString,
)
from shapely.validation import explain_validity
from scipy.spatial import KDTree
import alphashape
from skimage.measure import find_contours
from scipy.ndimage import (
    gaussian_filter,
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    label,
)

import logging
from time import perf_counter

from SPIX.visualization import raster as _raster

# Import helper functions from utils.py (assumed available)
from SPIX.utils.utils import (
    brighten_colors,
    rebalance_colors,
    is_collinear,
    smooth_polygon,
    _check_spatial_data,
    _check_img,
    _check_scale_factor,
)

# Configure logging to display warnings and errors
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _normalize_gap_collapse_axis_safe(axis):
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


def _infer_regular_grid_step(values, *, q: float = 0.05, max_samples: int = 200_000):
    """Infer a 1D lattice step from integer-like coordinates (raster-expanded tiles)."""
    vals = np.asarray(values, dtype=float)
    if vals.size < 2:
        return None
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None
    if vals.size > int(max_samples):
        try:
            idx = np.random.choice(vals.size, int(max_samples), replace=False)
            vals = vals[idx]
        except Exception:
            vals = vals[: int(max_samples)]

    try:
        frac = np.abs(vals - np.rint(vals))
        if float(np.nanmax(frac)) > 1e-3:
            return None
    except Exception:
        return None

    vals_i = np.rint(vals).astype(np.int64, copy=False)
    u = np.unique(vals_i)
    if u.size < 2:
        return None
    u.sort()
    d = np.diff(u)
    d = d[d > 0]
    if d.size == 0:
        return None
    try:
        qq = float(q)
        if not (0.0 < qq <= 1.0):
            qq = 0.05
    except Exception:
        qq = 0.05
    step = int(np.quantile(d, qq))
    if step <= 0:
        step = int(np.min(d))
    return int(step) if step > 0 else None


def _coords_to_grid_indices(x, y, *, step: int):
    """Map integer-like x/y onto a (0..w-1, 0..h-1) grid with given step."""
    x_i = np.rint(np.asarray(x, dtype=float)).astype(np.int64, copy=False)
    y_i = np.rint(np.asarray(y, dtype=float)).astype(np.int64, copy=False)
    x0 = int(x_i.min())
    y0 = int(y_i.min())
    s = int(max(1, step))
    xi = np.floor((x_i - x0) / s + 0.5).astype(np.int64, copy=False)
    yi = np.floor((y_i - y0) / s + 0.5).astype(np.int64, copy=False)
    xi -= int(xi.min(initial=0))
    yi -= int(yi.min(initial=0))
    w0 = int(xi.max(initial=0)) + 1
    h0 = int(yi.max(initial=0)) + 1
    return xi, yi, w0, h0, x0, y0


def _apply_axis_crop(ax, *, xlim=None, ylim=None):
    if xlim is not None:
        x = np.asarray(xlim, dtype=float).reshape(-1)
        if x.size >= 2 and np.isfinite(x[:2]).all():
            cur = ax.get_xlim()
            descending = bool(cur[1] < cur[0])
            x0, x1 = float(x[0]), float(x[1])
            ax.set_xlim(x1, x0) if descending else ax.set_xlim(x0, x1)
    if ylim is not None:
        y = np.asarray(ylim, dtype=float).reshape(-1)
        if y.size >= 2 and np.isfinite(y[:2]).all():
            cur = ax.get_ylim()
            descending = bool(cur[1] < cur[0])
            y0, y1 = float(y[0]), float(y[1])
            ax.set_ylim(y1, y0) if descending else ax.set_ylim(y0, y1)


def _freeze_plot_cache_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return tuple(_freeze_plot_cache_value(v) for v in value.tolist())
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze_plot_cache_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(
            sorted((str(k), _freeze_plot_cache_value(v)) for k, v in value.items())
        )
    return repr(value)


def _sanitize_plot_cache_component(value, *, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "")).strip("_")
    return text or str(fallback)


def _default_plot_image_cache_key(
    *,
    embedding,
    dimensions,
    origin,
    color_by=None,
    alpha_by=None,
    plot_boundaries: bool = False,
) -> str:
    emb_tag = _sanitize_plot_cache_component(embedding, fallback="embedding")
    dims_tag = "_".join(str(int(d)) for d in np.asarray(dimensions, dtype=int).tolist())
    mode_tag = "origin" if bool(origin) else "tiles"
    parts = ["__spix_plot_cache", emb_tag, mode_tag, f"dims_{dims_tag}"]
    if color_by is not None:
        parts.append(f"color_{_sanitize_plot_cache_component(color_by, fallback='obs')}")
    if alpha_by is not None:
        parts.append(f"alpha_{_sanitize_plot_cache_component(alpha_by, fallback='alpha')}")
    if plot_boundaries:
        parts.append("boundaries")
    return "__".join(parts)


def _build_plot_cache_request(**cache_kwargs):
    return {
        "embedding": str(cache_kwargs.get("embedding", "")),
        "dimensions": tuple(
            int(d) for d in np.asarray(cache_kwargs.get("dimensions", []), dtype=int).tolist()
        ),
        "origin": bool(cache_kwargs.get("origin", True)),
        "coordinate_mode": str(cache_kwargs.get("coordinate_mode", "spatial")),
        "array_row_key": str(cache_kwargs.get("array_row_key", "array_row")),
        "array_col_key": str(cache_kwargs.get("array_col_key", "array_col")),
        "color_by": cache_kwargs.get("color_by", None),
        "palette": _freeze_plot_cache_value(cache_kwargs.get("palette", None)),
        "alpha_by": cache_kwargs.get("alpha_by", None),
        "alpha_range": _freeze_plot_cache_value(cache_kwargs.get("alpha_range", None)),
        "alpha_clip": _freeze_plot_cache_value(cache_kwargs.get("alpha_clip", None)),
        "alpha_invert": bool(cache_kwargs.get("alpha_invert", False)),
        "prioritize_high_values": bool(cache_kwargs.get("prioritize_high_values", False)),
        "overlap_priority": cache_kwargs.get("overlap_priority", None),
        "segment_color_by_major": bool(cache_kwargs.get("segment_color_by_major", False)),
        "use_imshow": bool(cache_kwargs.get("use_imshow", True)),
        "brighten_continuous": bool(cache_kwargs.get("brighten_continuous", False)),
        "continuous_gamma": float(cache_kwargs.get("continuous_gamma", 0.8)),
        "plot_boundaries": bool(cache_kwargs.get("plot_boundaries", False)),
        "boundary_method": str(cache_kwargs.get("boundary_method", "pixel")),
        "boundary_color": cache_kwargs.get("boundary_color", "black"),
        "boundary_linewidth": float(cache_kwargs.get("boundary_linewidth", 1.0)),
        "boundary_style": str(cache_kwargs.get("boundary_style", "solid")),
        "boundary_alpha": float(cache_kwargs.get("boundary_alpha", 0.7)),
        "pixel_smoothing_sigma": float(cache_kwargs.get("pixel_smoothing_sigma", 0.0)),
        "highlight_segments": _freeze_plot_cache_value(cache_kwargs.get("highlight_segments", None)),
        "highlight_boundary_color": cache_kwargs.get("highlight_boundary_color", None),
        "highlight_linewidth": cache_kwargs.get("highlight_linewidth", None),
        "highlight_brighten_factor": float(cache_kwargs.get("highlight_brighten_factor", 1.2)),
        "dim_other_segments": float(cache_kwargs.get("dim_other_segments", 0.3)),
        "dim_to_grey": bool(cache_kwargs.get("dim_to_grey", True)),
        "segment_key": cache_kwargs.get("segment_key", "Segment"),
        "imshow_tile_size": cache_kwargs.get("imshow_tile_size", None),
        "imshow_scale_factor": float(cache_kwargs.get("imshow_scale_factor", 1.0)),
        "imshow_tile_size_mode": str(cache_kwargs.get("imshow_tile_size_mode", "auto")),
        "imshow_tile_size_quantile": cache_kwargs.get("imshow_tile_size_quantile", None),
        "imshow_tile_size_rounding": str(cache_kwargs.get("imshow_tile_size_rounding", "floor")),
        "imshow_tile_size_shrink": float(cache_kwargs.get("imshow_tile_size_shrink", 0.98)),
        "pixel_perfect": bool(cache_kwargs.get("pixel_perfect", False)),
        "pixel_shape": str(cache_kwargs.get("pixel_shape", "square")),
        "runtime_fill_from_boundary": bool(cache_kwargs.get("runtime_fill_from_boundary", False)),
        "runtime_fill_closing_radius": int(cache_kwargs.get("runtime_fill_closing_radius", 1)),
        "runtime_fill_external_radius": int(cache_kwargs.get("runtime_fill_external_radius", 0)),
        "runtime_fill_holes": bool(cache_kwargs.get("runtime_fill_holes", True)),
        "gap_collapse_axis": _freeze_plot_cache_value(cache_kwargs.get("gap_collapse_axis", None)),
        "gap_collapse_max_run_px": int(cache_kwargs.get("gap_collapse_max_run_px", 1)),
        "soft_rasterization": bool(cache_kwargs.get("soft_rasterization", False)),
        "resolve_center_collisions": _freeze_plot_cache_value(cache_kwargs.get("resolve_center_collisions", "auto")),
        "center_collision_radius": int(cache_kwargs.get("center_collision_radius", 2)),
    }


def _dims_subset_of_cache(request_dims, cache_dims) -> bool:
    req = tuple(int(d) for d in request_dims)
    have = tuple(int(d) for d in cache_dims)
    have_set = set(have)
    return all(int(d) in have_set for d in req)


def _cache_matches_plot_request(cache, request) -> bool:
    if not isinstance(cache, dict):
        return False

    cache_dims = tuple(int(d) for d in cache.get("dimensions", []) or [])
    if not cache_dims or not _dims_subset_of_cache(request["dimensions"], cache_dims):
        return False
    if str(cache.get("embedding_key", "")) != str(request["embedding"]):
        return False
    if bool(cache.get("origin", True)) != bool(request["origin"]):
        return False
    if str(cache.get("coordinate_mode", "spatial")) != str(request["coordinate_mode"]):
        return False
    if str(cache.get("array_row_key", "array_row")) != str(request["array_row_key"]):
        return False
    if str(cache.get("array_col_key", "array_col")) != str(request["array_col_key"]):
        return False

    plot_params = cache.get("image_plot_params", {}) or {}
    keys_to_compare = [
        "color_by",
        "palette",
        "alpha_by",
        "alpha_range",
        "alpha_clip",
        "alpha_invert",
        "prioritize_high_values",
        "overlap_priority",
        "segment_color_by_major",
        "use_imshow",
        "brighten_continuous",
        "continuous_gamma",
        "plot_boundaries",
        "boundary_method",
        "boundary_color",
        "boundary_linewidth",
        "boundary_style",
        "boundary_alpha",
        "pixel_smoothing_sigma",
        "highlight_segments",
        "highlight_boundary_color",
        "highlight_linewidth",
        "highlight_brighten_factor",
        "dim_other_segments",
        "dim_to_grey",
        "segment_key",
        "runtime_fill_from_boundary",
        "runtime_fill_closing_radius",
        "runtime_fill_external_radius",
        "runtime_fill_holes",
        "gap_collapse_axis",
        "gap_collapse_max_run_px",
        "soft_rasterization",
        "resolve_center_collisions",
        "center_collision_radius",
    ]
    for key in keys_to_compare:
        cache_value = plot_params.get(key, None)
        if key == "segment_key" and cache_value is None:
            cache_value = cache.get("segment_key_requested", cache.get("segment_key", None))
        if _freeze_plot_cache_value(cache_value) != _freeze_plot_cache_value(request.get(key, None)):
            return False
    return True


def _find_compatible_plot_cache(adata, request):
    for key, value in getattr(adata, "uns", {}).items():
        if _cache_matches_plot_request(value, request):
            return str(key), value
    return None, None


def _prepare_plot_image_cache(
    adata,
    *,
    cache_image: bool,
    image_cache_key,
    refresh_image_cache: bool,
    verbose: bool,
    **cache_kwargs,
):
    request = _build_plot_cache_request(**cache_kwargs)

    if (not cache_image) and (image_cache_key is None):
        found_key, found_cache = _find_compatible_plot_cache(adata, request)
        if found_cache is not None:
            if verbose:
                print(f"[plot cache] Auto-reusing compatible cached image from adata.uns['{found_key}']")
            return found_cache
        return None

    from SPIX.image_processing.image_cache import cache_embedding_image

    key = str(
        image_cache_key
        or _default_plot_image_cache_key(
            embedding=cache_kwargs.get("embedding", "X_embedding"),
            dimensions=cache_kwargs.get("dimensions", [0, 1, 2]),
            origin=cache_kwargs.get("origin", True),
            color_by=cache_kwargs.get("color_by", None),
            alpha_by=cache_kwargs.get("alpha_by", None),
            plot_boundaries=cache_kwargs.get("plot_boundaries", False),
        )
    )

    signature = {"version": 2, **request}

    existing = adata.uns.get(key, None)
    if (
        isinstance(existing, dict)
        and (not refresh_image_cache)
        and (
            existing.get("plot_cache_signature", None) == signature
            or _cache_matches_plot_request(existing, request)
        )
    ):
        if verbose:
            print(f"[plot cache] Reusing cached image under adata.uns['{key}']")
        return existing

    cache = cache_embedding_image(
        adata,
        embedding=cache_kwargs.get("embedding", "X_embedding"),
        dimensions=list(cache_kwargs.get("dimensions", [0, 1, 2])),
        origin=bool(cache_kwargs.get("origin", True)),
        key=key,
        coordinate_mode=cache_kwargs.get("coordinate_mode", "spatial"),
        array_row_key=cache_kwargs.get("array_row_key", "array_row"),
        array_col_key=cache_kwargs.get("array_col_key", "array_col"),
        figsize=None,
        fig_dpi=None,
        imshow_tile_size=cache_kwargs.get("imshow_tile_size", None),
        imshow_scale_factor=float(cache_kwargs.get("imshow_scale_factor", 1.0)),
        imshow_tile_size_mode=cache_kwargs.get("imshow_tile_size_mode", "auto"),
        imshow_tile_size_quantile=cache_kwargs.get("imshow_tile_size_quantile", None),
        imshow_tile_size_rounding=cache_kwargs.get("imshow_tile_size_rounding", "floor"),
        imshow_tile_size_shrink=float(cache_kwargs.get("imshow_tile_size_shrink", 0.98)),
        pixel_perfect=bool(cache_kwargs.get("pixel_perfect", False)),
        pixel_shape=cache_kwargs.get("pixel_shape", "square"),
        chunk_size=int(cache_kwargs.get("chunk_size", 50000)),
        verbose=bool(verbose),
        show=False,
        color_by=cache_kwargs.get("color_by", None),
        palette=cache_kwargs.get("palette", "tab20"),
        alpha_by=cache_kwargs.get("alpha_by", None),
        alpha_range=cache_kwargs.get("alpha_range", (0.1, 1.0)),
        alpha_clip=cache_kwargs.get("alpha_clip", None),
        alpha_invert=bool(cache_kwargs.get("alpha_invert", False)),
        prioritize_high_values=bool(cache_kwargs.get("prioritize_high_values", False)),
        overlap_priority=cache_kwargs.get("overlap_priority", None),
        segment_color_by_major=bool(cache_kwargs.get("segment_color_by_major", False)),
        title=cache_kwargs.get("title", None),
        use_imshow=bool(cache_kwargs.get("use_imshow", True)),
        brighten_continuous=bool(cache_kwargs.get("brighten_continuous", False)),
        continuous_gamma=float(cache_kwargs.get("continuous_gamma", 0.8)),
        plot_boundaries=bool(cache_kwargs.get("plot_boundaries", False)),
        boundary_method=cache_kwargs.get("boundary_method", "pixel"),
        boundary_color=cache_kwargs.get("boundary_color", "black"),
        boundary_linewidth=float(cache_kwargs.get("boundary_linewidth", 1.0)),
        boundary_style=cache_kwargs.get("boundary_style", "solid"),
        boundary_alpha=float(cache_kwargs.get("boundary_alpha", 0.7)),
        pixel_smoothing_sigma=float(cache_kwargs.get("pixel_smoothing_sigma", 0.0)),
        highlight_segments=cache_kwargs.get("highlight_segments", None),
        highlight_boundary_color=cache_kwargs.get("highlight_boundary_color", None),
        highlight_linewidth=cache_kwargs.get("highlight_linewidth", None),
        highlight_brighten_factor=float(cache_kwargs.get("highlight_brighten_factor", 1.2)),
        dim_other_segments=float(cache_kwargs.get("dim_other_segments", 0.3)),
        dim_to_grey=bool(cache_kwargs.get("dim_to_grey", True)),
        segment_key=cache_kwargs.get("segment_key", "Segment"),
        runtime_fill_from_boundary=bool(cache_kwargs.get("runtime_fill_from_boundary", False)),
        runtime_fill_closing_radius=int(cache_kwargs.get("runtime_fill_closing_radius", 1)),
        runtime_fill_external_radius=int(cache_kwargs.get("runtime_fill_external_radius", 0)),
        runtime_fill_holes=bool(cache_kwargs.get("runtime_fill_holes", True)),
        gap_collapse_axis=cache_kwargs.get("gap_collapse_axis", None),
        gap_collapse_max_run_px=int(cache_kwargs.get("gap_collapse_max_run_px", 1)),
        soft_rasterization=bool(cache_kwargs.get("soft_rasterization", False)),
        resolve_center_collisions=cache_kwargs.get("resolve_center_collisions", "auto"),
        center_collision_radius=int(cache_kwargs.get("center_collision_radius", 2)),
    )
    try:
        cache["plot_cache_signature"] = signature
        adata.uns[key] = cache
    except Exception:
        pass
    if verbose:
        print(f"[plot cache] Cached image under adata.uns['{key}']")
    return cache


def _cached_preview_matches_dimensions(cache, requested_dimensions) -> bool:
    if "img_preview" not in cache:
        return False
    cache_dims = tuple(int(d) for d in cache.get("dimensions", []) or [])
    req = tuple(int(d) for d in requested_dimensions)
    if not cache_dims:
        return False
    if len(cache_dims) >= 3:
        return tuple(cache_dims[:3]) == req
    return cache_dims == req


def _cached_display_image(cache, *, requested_dimensions, prefer_preview: bool = True):
    from SPIX.image_processing.image_cache import get_cached_image_channels

    use_preview = bool(
        prefer_preview and _cached_preview_matches_dimensions(cache, requested_dimensions)
    )
    cache_dims = tuple(int(d) for d in cache.get("dimensions", []) or [])
    req = tuple(int(d) for d in requested_dimensions)
    if not cache_dims:
        cache_dims = tuple(range(int(cache.get("channels", 0))))
    channel_idx = None
    if req:
        dim_to_idx = {int(dim): i for i, dim in enumerate(cache_dims)}
        if not all(int(dim) in dim_to_idx for dim in req):
            raise ValueError("Requested display dimensions are not present in cached image.")
        channel_idx = [int(dim_to_idx[int(dim)]) for dim in req]

    img = (
        np.asarray(cache.get("img_preview"))
        if use_preview
        else np.asarray(get_cached_image_channels(cache, channels=channel_idx))
    )
    if img.ndim != 3:
        raise ValueError("Cached image must be 3D for display.")
    if img.shape[2] >= 3:
        return np.asarray(img[:, :, :3], dtype=np.float32)
    if img.shape[2] == 2:
        return np.stack(
            [img[:, :, 0], img[:, :, 1], np.zeros_like(img[:, :, 0])],
            axis=2,
        ).astype(np.float32)
    return np.repeat(np.asarray(img, dtype=np.float32), 3, axis=2)


def _render_plot_cache_fast(
    *,
    cache,
    requested_dimensions,
    title,
    figsize,
    fig_dpi,
    xlim=None,
    ylim=None,
    show_axes: bool = False,
    background_img=None,
    background_extent=None,
    background_origin=None,
    background_alpha: float = 1.0,
):
    display_img = _cached_display_image(
        cache,
        requested_dimensions=requested_dimensions,
        prefer_preview=bool(cache.get("image_plot_params", {}).get("plot_boundaries", False)),
    )
    extent = [
        float(cache["x_min_eff"]),
        float(cache["x_max_eff"]),
        float(cache["y_min_eff"]),
        float(cache["y_max_eff"]),
    ]

    if figsize is None:
        figsize = (10, 10)
    if fig_dpi is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)

    if background_img is not None:
        kwargs = {"extent": background_extent, "alpha": float(background_alpha)}
        if background_origin is not None:
            kwargs["origin"] = background_origin
        ax.imshow(background_img, **kwargs)

    ax.imshow(
        display_img,
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    _apply_axis_crop(ax, xlim=xlim, ylim=ylim)
    ax.set_aspect("equal")
    if not show_axes:
        ax.axis("off")
    ax.set_title(title, fontsize=figsize[0] * 1.5)
    plt.show()
    return fig, ax


def _can_render_plot_cache_fast(
    *,
    cache,
    requested_dimensions,
    show_colorbar: bool,
    show_legend: bool,
    segment_show_pie: bool = False,
    fill_boundaries: bool = False,
    plot_boundaries: bool = False,
    boundary_method: str = "pixel",
    use_imshow: bool = True,
):
    if cache is None or (not use_imshow):
        return False
    if show_colorbar or show_legend or segment_show_pie or fill_boundaries:
        return False
    if plot_boundaries and str(boundary_method) != "pixel":
        return False
    if not _dims_subset_of_cache(
        tuple(int(d) for d in requested_dimensions),
        tuple(int(d) for d in cache.get("dimensions", []) or []),
    ):
        return False
    plot_params = cache.get("image_plot_params", {}) or {}
    if plot_boundaries and ("img_preview" not in cache) and bool(plot_params.get("plot_boundaries", False)):
        return False
    if plot_boundaries and (not _cached_preview_matches_dimensions(cache, requested_dimensions)):
        return False
    return True


def _string_index_no_copy(values):
    idx = pd.Index(values, copy=False)
    inferred = getattr(idx, "inferred_type", None)
    if inferred in {"string", "unicode"}:
        return idx
    try:
        return idx.astype(str)
    except Exception:
        return pd.Index(np.asarray(idx, dtype=str))


def _string_array_no_copy(values):
    arr = values.to_numpy(copy=False) if hasattr(values, "to_numpy") else np.asarray(values)
    dtype = getattr(arr, "dtype", None)
    if dtype is not None and (dtype.kind in {"U", "S"} or dtype == object):
        return arr
    return np.asarray(arr, dtype=str)


def _float_array_fast(values):
    if hasattr(values, "dtype") and is_numeric_dtype(values.dtype):
        return values.to_numpy(dtype=float, copy=False)
    arr = values.to_numpy(copy=False) if hasattr(values, "to_numpy") else np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float, copy=False)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=False)


def _rebalance_color_array(values, *, method="minmax", vmin=None, vmax=None):
    arr = np.asarray(values, dtype=float)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
        squeeze = True

    arr_stats = arr.copy()
    arr_stats[~np.isfinite(arr_stats)] = np.nan

    if (vmin is not None) or (vmax is not None):
        if vmin is None:
            vmin_arr = np.nanmin(arr_stats, axis=0)
        else:
            vmin_arr = np.asarray(vmin, dtype=float)
        if vmax is None:
            vmax_arr = np.nanmax(arr_stats, axis=0)
        else:
            vmax_arr = np.asarray(vmax, dtype=float)
        scaled = (arr - vmin_arr) / ((vmax_arr - vmin_arr) + 1e-10)
    elif method == "minmax":
        vmin_arr = np.nanmin(arr_stats, axis=0)
        vmax_arr = np.nanmax(arr_stats, axis=0)
        scaled = (arr - vmin_arr) / ((vmax_arr - vmin_arr) + 1e-10)
    else:
        scaled = arr

    scaled = np.clip(scaled, 0.0, 1.0)
    if squeeze:
        return scaled[:, 0]
    return scaled


def _grouped_obs_indexer(tile_barcodes, obs_index):
    arr = np.asarray(tile_barcodes)
    n = int(arr.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int64), 0
    change = np.empty(n, dtype=bool)
    change[0] = True
    if n > 1:
        change[1:] = arr[1:] != arr[:-1]
    starts = np.flatnonzero(change)
    grouped_barcodes = arr[starts]
    grouped_index = pd.Index(grouped_barcodes)
    if grouped_index.has_duplicates:
        pos = obs_index.get_indexer(arr)
        return pos.astype(np.int64, copy=False), int(pd.Index(arr).nunique())
    grouped_pos = obs_index.get_indexer(grouped_barcodes)
    lengths = np.diff(np.append(starts, n))
    pos = np.repeat(grouped_pos, lengths)
    return pos.astype(np.int64, copy=False), int(grouped_barcodes.shape[0])


def _compute_alpha_from_values(values, alpha_range=(0.1, 1.0), clip=None, invert=False):
    """Map raw numeric values -> alphas in [alpha_range[0], alpha_range[1]].
    - clip: (vmin, vmax) or None -> None uses [1, 99] percentiles.
    - invert: if True, high values become low alphas (reversed mapping).
    """
    vals = np.asarray(values, dtype=float)
    if clip is None:
        vmin, vmax = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
    else:
        vmin, vmax = clip
        if vmin is None:
            vmin = np.nanmin(vals)
        if vmax is None:
            vmax = np.nanmax(vals)
    if vmax <= vmin:
        vmax = vmin + 1e-9

    t = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    if invert:
        t = 1.0 - t

    a0, a1 = float(alpha_range[0]), float(alpha_range[1])
    a = a0 + t * (a1 - a0)
    return a.astype(np.float32), (vmin, vmax)


def _estimate_tile_size_units(
    pts,
    imshow_tile_size,
    imshow_scale_factor,
    imshow_tile_size_mode,
    imshow_tile_size_quantile,
    imshow_tile_size_shrink,
    logger=None,
):
    return _raster.estimate_tile_size_units(
        np.asarray(pts),
        imshow_tile_size,
        imshow_scale_factor,
        imshow_tile_size_mode,
        imshow_tile_size_quantile,
        imshow_tile_size_shrink,
        logger=logger,
    )


def _round_tile_size_px(s_data_units, scale, rounding):
    mode = str(rounding or "ceil").lower()
    s_float = float(s_data_units) * float(scale)
    if mode == "floor":
        s = int(np.floor(s_float))
    elif mode == "round":
        s = int(np.round(s_float))
    else:
        s = int(np.ceil(s_float))
    return max(1, s)


def _resolve_figsize_dpi_for_tiles(
    *,
    figsize,
    fig_dpi,
    x_range: float,
    y_range: float,
    s_data_units: float,
    pixel_perfect: bool,
    n_points: int | None = None,
    logger=None,
):
    return _raster.resolve_figsize_dpi_for_tiles(
        figsize=figsize,
        fig_dpi=fig_dpi,
        x_range=float(x_range),
        y_range=float(y_range),
        s_data_units=float(s_data_units),
        pixel_perfect=bool(pixel_perfect),
        n_points=(None if n_points is None else int(n_points)),
        logger=logger,
    )


def _cap_tile_size_by_density(
    *,
    s_data_units: float,
    x_range: float,
    y_range: float,
    n_points: int,
    logger=None,
    context: str = "",
) -> float:
    return _raster.cap_tile_size_by_density(
        s_data_units=float(s_data_units),
        x_range=float(x_range),
        y_range=float(y_range),
        n_points=int(n_points),
        logger=logger,
        context=str(context or ""),
    )


def _cap_tile_size_by_render_extent(
    *,
    s_data_units: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pixel_perfect: bool,
    n_points: int,
    logger=None,
    context: str = "",
) -> float:
    return _raster.cap_tile_size_by_render_extent(
        s_data_units=float(s_data_units),
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        pixel_perfect=bool(pixel_perfect),
        n_points=int(n_points),
        logger=logger,
        context=str(context or ""),
    )


def _build_subpixel_boundary_mask(
    labels: np.ndarray,
    *,
    include_background_interfaces: bool,
) -> np.ndarray:
    labels = np.asarray(labels)
    h, w = labels.shape
    boundary_mask = np.zeros((max(1, 2 * h - 1), max(1, 2 * w - 1)), dtype=bool)

    if h >= 1 and w >= 2:
        left = labels[:, :-1]
        right = labels[:, 1:]
        horiz = left != right
        if include_background_interfaces:
            horiz &= (left >= 0) | (right >= 0)
        else:
            horiz &= (left >= 0) & (right >= 0)
        boundary_mask[0::2, 1::2] = horiz
    else:
        horiz = np.zeros((h, max(0, w - 1)), dtype=bool)

    if h >= 2 and w >= 1:
        top = labels[:-1, :]
        bottom = labels[1:, :]
        vert = top != bottom
        if include_background_interfaces:
            vert &= (top >= 0) | (bottom >= 0)
        else:
            vert &= (top >= 0) & (bottom >= 0)
        boundary_mask[1::2, 0::2] = vert
    else:
        vert = np.zeros((max(0, h - 1), w), dtype=bool)

    if h >= 2 and w >= 2:
        junction = np.zeros((h - 1, w - 1), dtype=bool)
        if horiz.size:
            junction |= horiz[:-1, :]
            junction |= horiz[1:, :]
        if vert.size:
            junction |= vert[:, :-1]
            junction |= vert[:, 1:]
        boundary_mask[1::2, 1::2] = junction

    return boundary_mask


def _regularize_support_mask(
    support_mask: np.ndarray,
    *,
    fill_holes: bool = True,
    closing_radius: int = 1,
    min_component_size: int = 64,
) -> np.ndarray:
    support = np.asarray(support_mask, dtype=bool)
    if fill_holes:
        support = binary_fill_holes(support)
    close_radius = int(max(0, closing_radius))
    if close_radius > 0:
        structure = np.ones((2 * close_radius + 1, 2 * close_radius + 1), dtype=bool)
        support = binary_closing(support, structure=structure)
        support = binary_fill_holes(support)
    min_size = int(max(0, min_component_size))
    if min_size > 0 and np.any(support):
        comp_labels, n_comp = label(support)
        if n_comp > 0:
            counts = np.bincount(comp_labels.ravel())
            keep = counts >= min_size
            if keep.size:
                keep[0] = False
                support = keep[comp_labels]
    return support


def _prepare_outline_labels(
    labels: np.ndarray,
    *,
    fill_holes: bool = True,
    closing_radius: int = 1,
    min_component_size: int = 64,
):
    labels_arr = np.asarray(labels, dtype=int)
    valid_mask = labels_arr >= 0
    support = _regularize_support_mask(
        valid_mask,
        fill_holes=fill_holes,
        closing_radius=closing_radius,
        min_component_size=min_component_size,
    )
    if not np.any(support):
        return np.full_like(labels_arr, -1), support

    filled = labels_arr.copy()
    add_mask = support & ~valid_mask
    if np.any(add_mask) and np.any(valid_mask):
        _, nearest_idx = distance_transform_edt(~valid_mask, return_indices=True)
        filled[add_mask] = labels_arr[
            nearest_idx[0][add_mask],
            nearest_idx[1][add_mask],
        ]
    filled = np.where(support, filled, -1)
    return filled, support


def _build_support_outline_mask(
    labels: np.ndarray,
    *,
    fill_holes: bool = True,
    closing_radius: int = 1,
    min_component_size: int = 64,
) -> np.ndarray:
    support = _regularize_support_mask(
        np.asarray(labels) >= 0,
        fill_holes=fill_holes,
        closing_radius=closing_radius,
        min_component_size=min_component_size,
    )
    if not np.any(support):
        return np.zeros((max(1, 2 * labels.shape[0] - 1), max(1, 2 * labels.shape[1] - 1)), dtype=bool)
    support_labels = np.where(support, 1, -1)
    return _build_subpixel_boundary_mask(
        support_labels,
        include_background_interfaces=True,
    )


def _render_boundary_mask(
    ax,
    boundary_mask,
    extent,
    boundary_color,
    boundary_linewidth,
    alpha,
    linestyle,
    smooth_sigma=0,
    *,
    render: str = "contour",
    thickness_px: int = 1,
    antialiased: bool = True,
):
    x_min, x_max, y_min, y_max = extent
    render_mode = str(render or "contour").lower()
    if render_mode not in {"contour", "raster"}:
        render_mode = "contour"

    if render_mode == "raster":
        thick = int(max(1, thickness_px))
        mask = np.asarray(boundary_mask, dtype=bool)
        if thick > 1:
            mask = binary_dilation(mask, iterations=int(thick - 1))

        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        col = mcolors.to_rgba(boundary_color, alpha=float(alpha))
        rgba[mask] = col
        ax.imshow(
            rgba,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            interpolation="nearest",
        )
        return

    mask_float = np.asarray(boundary_mask, dtype=float)
    if smooth_sigma > 0:
        mask_float = gaussian_filter(mask_float, smooth_sigma)

    contours = find_contours(mask_float, 0.5)
    if not contours:
        return

    mh, mw = boundary_mask.shape
    segments = [
        np.column_stack(
            (
                contour[:, 1] / max(1, mw) * (x_max - x_min) + x_min,
                contour[:, 0] / max(1, mh) * (y_max - y_min) + y_min,
            )
        )
        for contour in contours
    ]
    lc = LineCollection(
        segments,
        colors=boundary_color,
        linewidths=boundary_linewidth,
        linestyles=linestyle,
        alpha=alpha,
        antialiaseds=bool(antialiased),
    )
    ax.add_collection(lc)


def _plot_boundaries_from_label_image(
    ax,
    label_img,
    extent,
    boundary_color,
    boundary_linewidth,
    alpha,
    linestyle,
    smooth_sigma=0,
    *,
    render: str = "contour",  # 'contour'|'raster'
    thickness_px: int = 1,    # used when render='raster'
    antialiased: bool = True, # used when render='contour'
    include_background: bool = False,
):
    """Plot boundaries extracted from a label image.

    ``include_background`` supports three modes:
    - ``True``: draw segment-segment and segment-background interfaces.
    - ``False``: draw only segment-segment interfaces.
    - ``\"outline\"``: draw segment-segment interfaces plus the outer support
      outline, without internal background/support interfaces.
    """
    labels = np.asarray(label_img)
    include_mode = include_background
    draw_support_outline = False
    if isinstance(include_mode, str):
        mode_norm = include_mode.strip().lower()
        if mode_norm in {"outline", "outer", "support", "external"}:
            include_mode = False
            draw_support_outline = True
        elif mode_norm in {"true", "all", "full", "background"}:
            include_mode = True
        elif mode_norm in {"false", "none", "segment_only", "segments"}:
            include_mode = False
        else:
            include_mode = bool(include_background)
    else:
        include_mode = bool(include_background)

    labels_for_base = labels
    if draw_support_outline:
        labels_for_base, _ = _prepare_outline_labels(
            labels,
            fill_holes=True,
            closing_radius=1,
            min_component_size=64,
        )

    base_mask = _build_subpixel_boundary_mask(
        labels_for_base,
        include_background_interfaces=include_mode,
    )
    _render_boundary_mask(
        ax,
        base_mask,
        extent,
        boundary_color,
        boundary_linewidth,
        alpha,
        linestyle,
        smooth_sigma,
        render=render,
        thickness_px=thickness_px,
        antialiased=antialiased,
    )

    if draw_support_outline:
        support_outline_mask = _build_support_outline_mask(
            labels,
            fill_holes=True,
            closing_radius=1,
            min_component_size=64,
        )
        _render_boundary_mask(
            ax,
            support_outline_mask,
            extent,
            boundary_color,
            boundary_linewidth,
            alpha,
            linestyle,
            smooth_sigma,
            render=render,
            thickness_px=thickness_px,
            antialiased=antialiased,
        )


def _sanitize_segment_key_for_uns(name: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name))
    sanitized = sanitized.strip("_")
    return sanitized or "Segment"


def _resolve_stored_pixel_boundary_payload(
    payload,
    *,
    default_extent=None,
    extent_scale: float = 1.0,
):
    """Return masked label image plus plotting extent from a stored payload."""
    if not isinstance(payload, dict):
        return None, default_extent

    label_img = np.asarray(payload.get("label_img"), dtype=int)
    stored_support_mask = payload.get("support_mask", None)
    if stored_support_mask is not None:
        stored_support_mask = np.asarray(stored_support_mask, dtype=bool)
        if stored_support_mask.shape == label_img.shape:
            label_img = np.where(stored_support_mask, label_img, -1)

    if default_extent is None:
        default_extent = [0.0, float(label_img.shape[1]), 0.0, float(label_img.shape[0])]

    scale = float(extent_scale)
    extent = [
        float(payload.get("x_min_eff", default_extent[0])) * scale,
        float(payload.get("x_max_eff", default_extent[1])) * scale,
        float(payload.get("y_min_eff", default_extent[2])) * scale,
        float(payload.get("y_max_eff", default_extent[3])) * scale,
    ]
    return label_img, extent


def _resolve_plot_raster_state(
    pts,
    *,
    x_min_eff,
    x_max_eff,
    y_min_eff,
    y_max_eff,
    scale,
    s,
    w,
    h,
    pixel_perfect: bool,
    pixel_shape: str,
    coord_mode: str = "spatial",
    gap_collapse_axis: str | None = None,
    gap_collapse_max_run_px: int = 1,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool | str = "auto",
    center_collision_radius: int = 2,
    logger=None,
    verbose: bool = False,
    context: str = "image_plot",
):
    """Resolve raster centers plus optional soft/collision behavior for plot-like renderers."""
    pts_arr = np.asarray(pts, dtype=float)
    soft_enabled = bool(soft_rasterization)
    if soft_enabled and (str(coord_mode).lower() == "visiumhd" and bool(pixel_perfect)):
        if verbose and logger is not None:
            logger.warning("soft_rasterization is ignored for pixel-perfect VisiumHD grid rendering.")
        soft_enabled = False
    if soft_enabled and (not bool(pixel_perfect) or int(s) != 1 or str(pixel_shape).lower() != "square"):
        if verbose and logger is not None:
            logger.warning(
                "soft_rasterization requires pixel_perfect=True, tile_px==1, and pixel_shape='square'; falling back to hard rasterization."
            )
        soft_enabled = False

    if soft_enabled:
        cx, cy, soft_fx, soft_fy = _raster.scale_points_to_canvas_soft(
            pts_arr,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=True,
            w=int(w),
            h=int(h),
        )
    else:
        soft_fx = None
        soft_fy = None
        cx, cy = _raster.scale_points_to_canvas(
            pts_arr,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=bool(pixel_perfect),
            w=int(w),
            h=int(h),
        )

    w_eff = int(w)
    h_eff = int(h)
    s_eff = int(s)
    x_min_eff_out = float(x_min_eff)
    x_max_eff_out = float(x_max_eff)
    y_min_eff_out = float(y_min_eff)
    y_max_eff_out = float(y_max_eff)

    if str(coord_mode).lower() == "visiumhd" and bool(pixel_perfect):
        x_i = np.floor(pts_arr[:, 0] + 0.5).astype(np.int64, copy=False)
        y_i = np.floor(pts_arr[:, 1] + 0.5).astype(np.int64, copy=False)
        x0, x1 = int(x_i.min()), int(x_i.max())
        y0, y1 = int(y_i.min()), int(y_i.max())
        w_eff = max(1, int(x1 - x0 + 1))
        h_eff = max(1, int(y1 - y0 + 1))
        cx = (x_i - x0).astype(np.int32, copy=False)
        cy = (y_i - y0).astype(np.int32, copy=False)
        s_eff = 1
        x_min_eff_out, x_max_eff_out = float(x0) - 0.5, float(x1) + 0.5
        y_min_eff_out, y_max_eff_out = float(y0) - 0.5, float(y1) + 0.5

    collision_info = None
    apply_collision_resolve, collision_stats, _ = _raster.should_resolve_center_collisions(
        resolve_center_collisions,
        cx=cx,
        cy=cy,
        w=int(w_eff),
        h=int(h_eff),
        tile_px=int(s_eff),
        pixel_perfect=bool(pixel_perfect),
        soft_enabled=bool(soft_enabled),
        logger=logger if verbose else None,
        context=context,
    )
    if apply_collision_resolve:
        collision_radius_eff = _raster.resolve_collision_search_radius(
            resolve_center_collisions,
            base_radius=int(center_collision_radius),
            collision_stats=collision_stats,
            logger=logger if verbose else None,
            context=context,
        )
        cx, cy, collision_info = _raster.resolve_center_collisions(
            cx,
            cy,
            w=int(w_eff),
            h=int(h_eff),
            max_radius=int(collision_radius_eff),
            logger=logger if verbose else None,
            context=context,
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
    gap_collapse_active = bool(
        gap_collapse_axis_norm is not None and int(max(0, int(gap_collapse_max_run_px))) > 0
    )
    if gap_collapse_active and soft_enabled:
        if verbose and logger is not None:
            logger.warning(
                "gap_collapse is currently incompatible with soft_rasterization; falling back to hard rasterization."
            )
        soft_enabled = False
        soft_fx = None
        soft_fy = None
    if gap_collapse_active:
        cx, cy, w_eff, h_eff, _ = _raster.collapse_empty_center_gaps(
            cx=cx,
            cy=cy,
            w=int(w_eff),
            h=int(h_eff),
            axis=gap_collapse_axis_norm,
            max_gap_run_px=int(gap_collapse_max_run_px),
            logger=logger if verbose else None,
            context=context,
        )

    return {
        "cx": np.asarray(cx, dtype=np.int32),
        "cy": np.asarray(cy, dtype=np.int32),
        "soft_fx": None if soft_fx is None else np.asarray(soft_fx, dtype=np.float32),
        "soft_fy": None if soft_fy is None else np.asarray(soft_fy, dtype=np.float32),
        "soft_enabled": bool(soft_enabled),
        "collision_info": collision_info,
        "w": int(w_eff),
        "h": int(h_eff),
        "s": int(s_eff),
        "x_min_eff": float(x_min_eff_out),
        "x_max_eff": float(x_max_eff_out),
        "y_min_eff": float(y_min_eff_out),
        "y_max_eff": float(y_max_eff_out),
    }


def _rasterize_rgba_and_label_buffer(
    *,
    cols,
    cx,
    cy,
    w: int,
    h: int,
    s: int,
    pixel_shape: str,
    alpha_arr=None,
    alpha_point: float = 1.0,
    vals=None,
    prioritize_high_values: bool = False,
    soft_enabled: bool = False,
    soft_fx=None,
    soft_fy=None,
    label_codes=None,
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool = True,
    chunk_size: int = 50000,
    persistent_store=None,
    logger=None,
    verbose: bool = False,
    context: str = "image_plot",
):
    """Rasterize RGBA tiles plus optional label image with shared soft/fill behavior."""
    cols_arr = np.asarray(cols, dtype=np.float32)
    cx_arr = np.asarray(cx, dtype=np.int32)
    cy_arr = np.asarray(cy, dtype=np.int32)
    if cols_arr.ndim != 2 or cols_arr.shape[0] != cx_arr.size:
        raise ValueError("cols must have shape (N, 3) aligned to raster centers.")

    img = np.zeros((int(h), int(w), 4), dtype=np.float32)
    paint_mask = np.zeros((int(h), int(w)), dtype=bool)
    label_img = None if label_codes is None else np.full((int(h), int(w)), -1, dtype=int)
    label_codes_arr = None if label_codes is None else np.asarray(label_codes, dtype=np.int32)
    alpha_chunk = None if alpha_arr is None else np.asarray(alpha_arr, dtype=np.float32)
    value_buf = None

    ox, oy = _raster.tile_pixel_offsets(int(s), pixel_shape=pixel_shape)
    pixels_per_tile = int(ox.size)
    target_pixels = int(os.environ.get("SPIX_PLOT_RASTER_TARGET_PIXELS", "10000000"))
    chunk_n = int(max(1, min(int(chunk_size), int(max(1, target_pixels // max(1, pixels_per_tile))))))

    if soft_enabled:
        value_buf = np.full((int(h), int(w)), -np.inf, dtype=np.float32) if label_img is not None else None
        if prioritize_high_values and verbose and logger is not None:
            logger.warning("soft_rasterization does not support prioritize_high_values; using weighted blending instead.")
        seed_rgba = np.empty((cols_arr.shape[0], 4), dtype=np.float32)
        seed_rgba[:, :3] = cols_arr
        if alpha_chunk is not None:
            seed_rgba[:, 3] = alpha_chunk
        else:
            seed_rgba[:, 3] = float(alpha_point)
        _raster.rasterize_soft_bilinear(
            img=img,
            occupancy_mask=paint_mask,
            seed_values=seed_rgba,
            cx=cx_arr,
            cy=cy_arr,
            fx=np.asarray(soft_fx, dtype=np.float32),
            fy=np.asarray(soft_fy, dtype=np.float32),
            chunk_size=int(chunk_n),
            background_value=None,
        )
        if label_img is not None:
            for i in range(0, cx_arr.size, chunk_n):
                end = min(i + chunk_n, cx_arr.size)
                x0, y0, x1, y1, w00, w10, w01, w11 = _raster.bilinear_support_from_centers(
                    cx_arr[i:end],
                    cy_arr[i:end],
                    np.asarray(soft_fx[i:end], dtype=np.float32),
                    np.asarray(soft_fy[i:end], dtype=np.float32),
                    w=int(w),
                    h=int(h),
                )
                seg_chunk = label_codes_arr[i:end]
                for px, py, pw in ((x0, y0, w00), (x1, y0, w10), (x0, y1, w01), (x1, y1, w11)):
                    mask_upd = pw > value_buf[py, px]
                    if np.any(mask_upd):
                        label_img[py[mask_upd], px[mask_upd]] = seg_chunk[mask_upd]
                        value_buf[py[mask_upd], px[mask_upd]] = pw[mask_upd]
    else:
        vals_arr = None if vals is None else np.asarray(vals, dtype=np.float32)
        if prioritize_high_values:
            value_buf = np.full((int(h), int(w)), -np.inf, dtype=np.float32)
        for i in range(0, cx_arr.size, chunk_n):
            end = min(i + chunk_n, cx_arr.size)
            chunk_cx = cx_arr[i:end]
            chunk_cy = cy_arr[i:end]
            chunk_cols = cols_arr[i:end]
            chunk_alpha = None if alpha_chunk is None else alpha_chunk[i:end]
            chunk_vals = None if vals_arr is None else vals_arr[i:end]

            if not prioritize_high_values:
                if pixels_per_tile == 1:
                    px = np.clip(chunk_cx, 0, int(w) - 1)
                    py = np.clip(chunk_cy, 0, int(h) - 1)
                    img[py, px, :3] = chunk_cols
                    if chunk_alpha is not None:
                        img[py, px, 3] = chunk_alpha
                    else:
                        img[py, px, 3] = float(alpha_point)
                    paint_mask[py, px] = True
                    if label_img is not None:
                        label_img[py, px] = label_codes_arr[i:end]
                else:
                    all_x = np.clip((chunk_cx[:, None] + ox[None, :]).ravel(), 0, int(w) - 1)
                    all_y = np.clip((chunk_cy[:, None] + oy[None, :]).ravel(), 0, int(h) - 1)
                    cols_rep = np.repeat(chunk_cols, pixels_per_tile, axis=0)
                    img[all_y, all_x, :3] = cols_rep
                    if chunk_alpha is not None:
                        img[all_y, all_x, 3] = np.repeat(chunk_alpha, pixels_per_tile, axis=0)
                    else:
                        img[all_y, all_x, 3] = float(alpha_point)
                    paint_mask[all_y, all_x] = True
                    if label_img is not None:
                        label_img[all_y, all_x] = np.repeat(label_codes_arr[i:end], pixels_per_tile, axis=0)
            else:
                if chunk_vals is None:
                    chunk_vals = np.arange(i, end, dtype=np.float32)
                for dx, dy in zip(ox, oy):
                    px = np.clip(chunk_cx + dx, 0, int(w) - 1)
                    py = np.clip(chunk_cy + dy, 0, int(h) - 1)
                    mask_upd = chunk_vals > value_buf[py, px]
                    if np.any(mask_upd):
                        img[py[mask_upd], px[mask_upd], :3] = chunk_cols[mask_upd]
                        img[py[mask_upd], px[mask_upd], 3] = (
                            chunk_alpha[mask_upd] if chunk_alpha is not None else float(alpha_point)
                        )
                        value_buf[py[mask_upd], px[mask_upd]] = chunk_vals[mask_upd]
                        paint_mask[py[mask_upd], px[mask_upd]] = True
                        if label_img is not None:
                            label_img[py[mask_upd], px[mask_upd]] = label_codes_arr[i:end][mask_upd]

    label_paint_mask = paint_mask.copy() if label_img is not None else None
    fill_active = bool(runtime_fill_from_boundary) or bool(runtime_fill_holes)
    fill_radius = int(runtime_fill_closing_radius) if bool(runtime_fill_from_boundary) else 0
    external_radius = int(runtime_fill_external_radius) if bool(runtime_fill_from_boundary) else 0
    if fill_active:
        seed_rgba = np.empty((cols_arr.shape[0], 4), dtype=np.float32)
        seed_rgba[:, :3] = cols_arr
        if alpha_chunk is not None:
            seed_rgba[:, 3] = alpha_chunk
        else:
            seed_rgba[:, 3] = float(alpha_point)
        _raster.fill_raster_from_boundary(
            img=img,
            occupancy_mask=paint_mask,
            seed_x=cx_arr,
            seed_y=cy_arr,
            seed_values=seed_rgba,
            closing_radius=int(fill_radius),
            external_radius=int(external_radius),
            fill_holes=bool(runtime_fill_holes),
            use_observed_pixels=bool(int(s) == 1),
            persistent_store=persistent_store,
            logger=logger if verbose else None,
            context=context,
        )
    if label_img is not None:
        # Keep boundary extraction anchored to observed rasterized labels only.
        # Filling label_img from the morphological support causes synthetic labels
        # to spill into blank/background regions, producing contour artifacts.
        label_img[~label_paint_mask] = -1

    return img, label_img, paint_mask


def image_plot(
    adata,
    dimensions=[0, 1, 2],
    embedding="X_embedding_segment",  # Use segmented embedding
    figsize=None,
    fig_dpi=None,
    point_size=None,  # Default is None
    scaling_factor=2,
    imshow_tile_size=None,
    imshow_scale_factor=1.0,
    imshow_tile_size_mode="auto",
    imshow_tile_size_quantile=None,
    imshow_tile_size_rounding="floor",
    imshow_tile_size_shrink=0.98,
    pixel_perfect: bool = False,
    origin=True,
    plot_boundaries=False,  # Toggle for plotting boundaries
    boundary_method="pixel",  # Options: 'convex_hull', 'alphashape', 'oriented_bbox', 'concave_hull', 'pixel'
    alpha=1.0,  # Transparency of boundaries (not per-spot alpha)
    boundary_color="black",  # Default boundary color (if using a fixed color)
    boundary_linewidth=1.0,  # Line width of boundaries
    aspect_ratio_threshold=3.0,  # Threshold to determine if a segment is elongated
    jitter=1e-6,  # Jitter value to avoid collinearity
    boundary_style="solid",  # Boundary line style
    fill_boundaries=False,  # Toggle for filling the inside of boundaries
    fill_color="blue",  # Fill color
    fill_alpha=0.1,  # Fill transparency
    brighten_factor=1.3,  # Factor for brightening colors
    fixed_boundary_color=None,  # Fixed boundary color
    highlight_segments=None,  # Segment labels to highlight
    highlight_linewidth=None,  # Line width for highlighted boundaries
    highlight_boundary_color=None,  # Color used for highlighted boundaries
    highlight_brighten_factor=1.5,  # Brighten factor for highlighted segments
    dim_other_segments=0.3,  # Dimming factor for non-highlighted segments
    dim_to_grey=True,  # Convert dimmed segments to grey
    cmap=None,  # Colormap for 1D embeddings (if not categorical)
    # Normalization controls for consistent colors across crops
    rebalance_method: str = "minmax",  # 'minmax' (default) or 'clip'
    rebalance_vmin=None,              # scalar (1D) or length-3 (3D) for fixed scaling
    rebalance_vmax=None,              # scalar (1D) or length-3 (3D) for fixed scaling
    color_vmin=None,                  # fixed scaling for numeric color_by
    color_vmax=None,                  # fixed scaling for numeric color_by
    color_percentiles=(1, 99),        # used when color_vmin/vmax are None
    pixel_smoothing_sigma=0,
    # Pixel-boundary rendering controls (boundary_method='pixel' only)
    pixel_boundary_render: str = "contour",  # 'contour'|'raster'
    pixel_boundary_thickness_px: int = 1,    # used when pixel_boundary_render='raster'
    pixel_boundary_antialiased: bool = True, # used when pixel_boundary_render='contour'
    pixel_boundary_include_background: str | bool = "outline",
    use_imshow=True,  # When True, render points as an image for speed
    pixel_shape="square",  # 'square' (default) or 'circle'
    chunk_size=50000,  # Control chunking
    prioritize_high_values=False,  # Draw high-value pixels last (depth buffer / order)
    verbose=False,  # Control verbosity
    title=None,  # Optional plot title
    show_colorbar=False,  # Show color scale bar for 1D embeddings
    # Categorical coloring from adata.obs (e.g., celltype)
    color_by=None,                # Column in adata.obs to color by (categorical)
    palette="tab20",              # Matplotlib palette name or colormap
    show_legend=False,            # Show legend for categories
    legend_loc="best",            # Legend location
    legend_ncol=2,                # Legend columns
    legend_outside=True,          # Place legend outside axes by default
    legend_bbox_to_anchor=(0.83, 0.5),  # Anchor (right center by default)
    legend_fontsize=None,         # Legend font size (None keeps default)
    # Categorical draw priority when tiles overlap
    overlap_priority='few_front',        # None|'few_front'|'many_front'
    # Segment-level composition overlays
    segment_annotate_by=None,     # obs key for per-segment composition (e.g., 'celltype')
    segment_annotate_mode="pie",  # 'pie'
    segment_min_tiles=10,         # skip small segments
    segment_top_n=3,              # top-N categories for pies
    segment_pie_scale=3.0,        # radius multiplier vs. tile size (data units)
    segment_label_fmt="{label} ({pct:.0f}%)",  # Deprecated (no effect)
    segment_label_color="white",               # Deprecated (no effect)
    segment_label_bbox=True,                   # Deprecated (no effect)
    # Control segment pie overlay visibility and styling
    segment_show_pie=False,
    segment_pie_edgecolor="black",
    segment_pie_linewidth=0.8,
    # Segment-major coloring
    segment_color_by_major=False,  # If True, color all tiles in a segment by the segment-major of `color_by`
    segment_key="Segment",         # obs column storing segment assignments
    # Per-spot alpha controls
    alpha_by=None,               # None|'embedding'|<obs column name>; per-spot alpha source
    alpha_range=(0.1, 1.0),      # min/max alpha for mapping to RGBA
    alpha_clip=None,             # (vmin, vmax) or None -> auto percentiles
    alpha_invert=False,          # invert mapping
    # Brightness / mixing controls for continuous (3D) embeddings
    brighten_continuous=False,   # If True, brighten 3D continuous colors
    continuous_gamma=0.8,        # Gamma < 1 brightens mid-tones
    # Visium/VisiumHD grid rendering
    coordinate_mode: str = "spatial",  # 'spatial'|'array'|'visium'|'visiumhd'|'auto'
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    # Raster-expanded rendering controls (origin=False)
    gapless: bool = False,  # If True, render raster-expanded tiles as a packed grid (no holes, faster)
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool = True,
    gap_collapse_axis: str | None = None,
    gap_collapse_max_run_px: int = 1,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool | str = "auto",
    center_collision_radius: int = 2,
    cache_image: bool = False,
    image_cache_key: str | None = None,
    refresh_image_cache: bool = False,
    show_axes: bool = False,
    xlim=None,
    ylim=None,
):
    """Visualize embeddings as a raster image with optional boundaries and per-spot alpha.

    The color can be categorical via `color_by`, while the transparency can be driven by
    `alpha_by='embedding'` (1D embedding) or an obs column of numeric values.

    Parameters
    ----------
    pixel_perfect : bool, optional (default ``True``)
        If True, snap the internal raster scale so the estimated tile pitch becomes an
        integer number of pixels, and use rounded pixel centers. This makes tiles look
        "packed" (no visible seams/gaps) across different ``figsize``/DPI settings.
    resolve_center_collisions : bool or {"auto"}, optional (default ``"auto"``)
        ``True`` always reassigns duplicate raster centers onto nearby empty
        pixels before painting/filling. ``"auto"`` enables the same behavior
        only for dense one-pixel rasters with substantial center collisions.
    soft_rasterization : bool, optional (default ``False``)
        If True, use bilinear splatting for the image raster instead of hard
        one-pixel assignment. Currently supported only for
        ``pixel_perfect=True``, ``tile_px==1``, ``pixel_shape='square'``, and
        the standard imshow raster path.
    gap_collapse_axis / gap_collapse_max_run_px : optional
        Optionally collapse short fully-empty raster row/column gaps before
        rendering.
    """
    # Create a local logger
    logger = logging.getLogger("image_plot")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False

    # Avoid duplicate handlers if the function is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    t_all = perf_counter()

    if len(dimensions) not in [1, 3]:
        raise ValueError("Visualization requires 1D or 3D embeddings.")

    plot_cache = _prepare_plot_image_cache(
        adata,
        cache_image=cache_image,
        image_cache_key=image_cache_key,
        refresh_image_cache=refresh_image_cache,
        verbose=bool(verbose),
        embedding=embedding,
        dimensions=dimensions,
        origin=origin,
        coordinate_mode=coordinate_mode,
        array_row_key=array_row_key,
        array_col_key=array_col_key,
        color_by=color_by,
        palette=palette,
        alpha_by=alpha_by,
        alpha_range=alpha_range,
        alpha_clip=alpha_clip,
        alpha_invert=alpha_invert,
        prioritize_high_values=prioritize_high_values,
        overlap_priority=overlap_priority,
        segment_color_by_major=segment_color_by_major,
        title=title,
        use_imshow=use_imshow,
        brighten_continuous=brighten_continuous,
        continuous_gamma=continuous_gamma,
        plot_boundaries=plot_boundaries,
        boundary_method=boundary_method,
        boundary_color=boundary_color,
        boundary_linewidth=boundary_linewidth,
        boundary_style=boundary_style,
        boundary_alpha=alpha,
        pixel_smoothing_sigma=pixel_smoothing_sigma,
        highlight_segments=highlight_segments,
        highlight_boundary_color=highlight_boundary_color,
        highlight_linewidth=highlight_linewidth,
        highlight_brighten_factor=highlight_brighten_factor,
        dim_other_segments=dim_other_segments,
        dim_to_grey=dim_to_grey,
        segment_key=segment_key,
        imshow_tile_size=imshow_tile_size,
        imshow_scale_factor=imshow_scale_factor,
        imshow_tile_size_mode=imshow_tile_size_mode,
        imshow_tile_size_quantile=imshow_tile_size_quantile,
        imshow_tile_size_rounding=imshow_tile_size_rounding,
        imshow_tile_size_shrink=imshow_tile_size_shrink,
        pixel_perfect=pixel_perfect,
        pixel_shape=pixel_shape,
        chunk_size=chunk_size,
        runtime_fill_from_boundary=runtime_fill_from_boundary,
        runtime_fill_closing_radius=runtime_fill_closing_radius,
        runtime_fill_external_radius=runtime_fill_external_radius,
        runtime_fill_holes=runtime_fill_holes,
        gap_collapse_axis=gap_collapse_axis,
        gap_collapse_max_run_px=gap_collapse_max_run_px,
        soft_rasterization=soft_rasterization,
        resolve_center_collisions=resolve_center_collisions,
        center_collision_radius=center_collision_radius,
    )

    if _can_render_plot_cache_fast(
        cache=plot_cache,
        requested_dimensions=dimensions,
        show_colorbar=bool(show_colorbar),
        show_legend=bool(show_legend),
        segment_show_pie=bool(segment_show_pie),
        fill_boundaries=bool(fill_boundaries),
        plot_boundaries=bool(plot_boundaries),
        boundary_method=str(boundary_method),
        use_imshow=bool(use_imshow),
    ):
        title_text = (
            f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
            if title is None
            else title
        )
        if verbose:
            logger.info("image_plot: rendering directly from cached image.")
        _render_plot_cache_fast(
            cache=plot_cache,
            requested_dimensions=dimensions,
            title=title_text,
            figsize=figsize,
            fig_dpi=fig_dpi,
            xlim=xlim,
            ylim=ylim,
            show_axes=bool(show_axes),
        )
        return

    # Extract embedding
    if embedding not in adata.obsm:
        raise ValueError(f"'{embedding}' embedding does not exist in adata.obsm.")

    # Extract spatial coordinates.
    use_direct_origin_spatial = False
    tiles = None
    origin_vals = None
    inferred_mode = ""
    if origin:
        spatial_direct = None
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
            tile_barcodes = _string_array_no_copy(adata.obs.index)
            x_vals = np.asarray(spatial_direct[:, 0], dtype=float)
            y_vals = np.asarray(spatial_direct[:, 1], dtype=float)
            if verbose:
                logger.info("Using adata.obsm['spatial'] directly for origin=True plotting.")
        else:
            tiles_all = adata.uns["tiles"]
            tile_cols = [c for c in ("x", "y", "barcode", "origin") if c in tiles_all.columns]
            if "origin" in tiles_all.columns:
                origin_mask = _float_array_fast(tiles_all["origin"]) == 1
                tiles = tiles_all.loc[origin_mask, tile_cols]
            else:
                tiles = tiles_all.loc[:, tile_cols]
    else:
        tiles_all = adata.uns["tiles"]
        tile_cols = [c for c in ("x", "y", "barcode", "origin") if c in tiles_all.columns]
        tiles = tiles_all.loc[:, tile_cols]
        inferred_info = adata.uns.get("tiles_inferred_grid_info", {})
        inferred_mode = str(inferred_info.get("mode", "")).lower()
        if inferred_mode == "boundary_voronoi" and ("origin" in tiles.columns):
            support_mask = _float_array_fast(tiles["origin"]) == 0
            if bool(np.any(support_mask)):
                tiles = tiles.loc[support_mask, tile_cols]
                if verbose:
                    logger.info("Using inferred boundary support-grid tiles only for origin=False plotting.")

    if not use_direct_origin_spatial:
        if "barcode" not in tiles.columns:
            raise ValueError("adata.uns['tiles'] must contain a 'barcode' column.")
        if "x" not in tiles.columns or "y" not in tiles.columns:
            raise ValueError("adata.uns['tiles'] must contain 'x' and 'y' columns.")
        tile_barcodes = _string_array_no_copy(tiles["barcode"])
        x_vals = _float_array_fast(tiles["x"])
        y_vals = _float_array_fast(tiles["y"])
        if "origin" in tiles.columns:
            origin_vals = _float_array_fast(tiles["origin"])

    # Prepare tile colors and align to tile rows (avoid large pandas merge for origin=False).
    t0 = perf_counter()
    obs_index = _string_index_no_copy(adata.obs.index)

    grouped_barcode_count = None
    if use_direct_origin_spatial:
        pos = np.arange(int(adata.n_obs), dtype=np.int64)
    else:
        if origin:
            barcode_index = pd.Index(tile_barcodes)
            if barcode_index.has_duplicates:
                if verbose:
                    logger.warning(
                        "adata.uns['tiles'] contains duplicated barcodes; keeping first occurrence for plotting."
                    )
                keep_mask = ~barcode_index.duplicated(keep="first")
                tiles = tiles.loc[keep_mask]
                tile_barcodes = tile_barcodes[keep_mask]
                x_vals = x_vals[keep_mask]
                y_vals = y_vals[keep_mask]
                if origin_vals is not None:
                    origin_vals = origin_vals[keep_mask]
            pos = obs_index.get_indexer(tile_barcodes)
        else:
            has_non_origin = False
            if origin_vals is not None:
                try:
                    has_non_origin = np.isfinite(origin_vals).any() and (origin_vals != 1).any()
                except Exception:
                    has_non_origin = False
            if verbose and has_non_origin:
                logger.info(
                    "adata.uns['tiles'] contains duplicated barcodes (expected for origin=False raster tiles); keeping duplicates."
                )
            if has_non_origin:
                pos, grouped_barcode_count = _grouped_obs_indexer(tile_barcodes, obs_index)
            else:
                pos = obs_index.get_indexer(tile_barcodes)

    valid = (pos >= 0) & np.isfinite(x_vals) & np.isfinite(y_vals)
    if not np.any(valid):
        raise ValueError("No tile barcodes match adata.obs index.")
    embedding_full = np.asarray(adata.obsm[embedding])
    embedding_dims = embedding_full[:, dimensions]
    dim_names = ["dim0", "dim1", "dim2"] if len(dimensions) == 3 else ["dim0"]
    coord_mode = str(coordinate_mode or "spatial").lower()
    if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
        coord_mode = "spatial"
    if coord_mode == "auto":
        if origin and (array_row_key in adata.obs.columns) and (array_col_key in adata.obs.columns) and int(adata.n_obs) >= 200_000:
            coord_mode = "visiumhd"
        else:
            coord_mode = "spatial"
    fast_continuous_rgb = bool(
        use_imshow
        and (color_by is None)
        and (alpha_by is None)
        and (len(dimensions) == 3)
        and (not plot_boundaries)
        and (not fill_boundaries)
        and (not segment_show_pie)
        and (not segment_color_by_major)
        and (not highlight_segments)
        and (not prioritize_high_values)
    )
    if fast_continuous_rgb:
        pos_valid = pos[valid].astype(np.int64, copy=False)
        x_plot = np.asarray(x_vals[valid], dtype=float)
        y_plot = np.asarray(y_vals[valid], dtype=float)
        cols = _rebalance_color_array(
            embedding_dims[pos_valid],
            method=rebalance_method,
            vmin=rebalance_vmin,
            vmax=rebalance_vmax,
        )
        origin_plot = origin_vals[valid] if origin_vals is not None else None

        if coord_mode in {"array", "visium", "visiumhd"}:
            if (array_row_key not in adata.obs.columns) or (array_col_key not in adata.obs.columns):
                raise ValueError(
                    f"coordinate_mode='array' requires adata.obs['{array_row_key}'] and adata.obs['{array_col_key}']."
                )
            col_all = _float_array_fast(adata.obs[array_col_key])
            row_all = _float_array_fast(adata.obs[array_row_key])
            col = col_all[pos_valid]
            row = row_all[pos_valid]
            coord_valid = np.isfinite(col) & np.isfinite(row)
            if coord_mode == "visium":
                row_use = row[coord_valid].astype(float, copy=False)
                parity = (np.rint(row_use).astype(np.int64, copy=False) % 2).astype(float)
                x_plot = col[coord_valid].astype(float, copy=False) + 0.5 * parity
                y_plot = row_use * (np.sqrt(3.0) / 2.0)
            else:
                x_plot = col[coord_valid].astype(float, copy=False)
                y_plot = row[coord_valid].astype(float, copy=False)
            cols = cols[coord_valid]
            pos_valid = pos_valid[coord_valid]
            if origin_plot is not None:
                origin_plot = origin_plot[coord_valid]

        coord_valid = np.isfinite(x_plot) & np.isfinite(y_plot) & np.isfinite(cols).all(axis=1)
        if not np.any(coord_valid):
            raise ValueError("No finite coordinates/colors available for plotting.")
        x_plot = x_plot[coord_valid]
        y_plot = y_plot[coord_valid]
        cols = cols[coord_valid]
        if origin_plot is not None:
            origin_plot = origin_plot[coord_valid]

        if brighten_continuous:
            max_per_pixel = cols.max(axis=1, keepdims=True)
            scale_cols = np.where(max_per_pixel > 0, 1.0 / max_per_pixel, 0.0)
            cols = np.clip(cols * scale_cols, 0.0, 1.0)
            try:
                gamma = float(continuous_gamma)
                if gamma > 0 and gamma != 1.0:
                    cols = np.clip(np.power(cols, gamma), 0.0, 1.0)
            except Exception:
                pass

        x_min, x_max = float(np.min(x_plot)), float(np.max(x_plot))
        y_min, y_max = float(np.min(y_plot)), float(np.max(y_plot))
        pts = np.column_stack((x_plot, y_plot))
        raster_canvas_for_imshow = None
        n_points_eff = int(pts.shape[0])
        imshow_tile_size_eff = imshow_tile_size
        if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
            imshow_tile_size_eff = 1.0
        raster_canvas_for_imshow = _raster.resolve_raster_canvas(
            pts=pts,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            figsize=figsize,
            fig_dpi=fig_dpi,
            imshow_tile_size=imshow_tile_size_eff,
            imshow_scale_factor=imshow_scale_factor,
            imshow_tile_size_mode=imshow_tile_size_mode,
            imshow_tile_size_quantile=imshow_tile_size_quantile,
            imshow_tile_size_rounding=imshow_tile_size_rounding,
            imshow_tile_size_shrink=imshow_tile_size_shrink,
            pixel_perfect=bool(pixel_perfect),
            n_points=int(n_points_eff),
            logger=logger,
            context="image_plot(auto-size)",
            persistent_store=getattr(adata, "uns", None),
        )
        s_data_units_for_imshow = float(raster_canvas_for_imshow["s_data_units"])
        figsize = raster_canvas_for_imshow["figsize"]
        fig_dpi = raster_canvas_for_imshow["fig_dpi"]

        if figsize is None:
            figsize = (10, 10)
        if fig_dpi is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)
        if verbose:
            logger.info(
                "Fast RGB path rows=%d; n_eff=%d; final figsize=%s, fig_dpi=%s",
                int(pts.shape[0]),
                int(n_points_eff),
                tuple(figsize),
                fig.get_dpi(),
            )

        if gapless and (not origin) and (origin_plot is not None):
            try:
                has_non_origin = np.isfinite(origin_plot).any() and (origin_plot != 1).any()
            except Exception:
                has_non_origin = False
            if has_non_origin:
                step_x = _infer_regular_grid_step(x_plot)
                step_y = _infer_regular_grid_step(y_plot)
                step = None
                if step_x is not None and step_y is not None:
                    step = int(max(1, min(int(step_x), int(step_y))))
                elif step_x is not None:
                    step = int(max(1, int(step_x)))
                elif step_y is not None:
                    step = int(max(1, int(step_y)))
                if step is not None:
                    t_grid = perf_counter()
                    xi0, yi0, w0, h0, x0_int, y0_int = _coords_to_grid_indices(x_plot, y_plot, step=step)
                    max_raster_px = _raster.resolve_max_raster_pixels(
                        env_key="SPIX_PLOT_MAX_RASTER_PIXELS",
                        n_points=len(x_plot),
                    )
                    total0 = int(w0) * int(h0)
                    ds = int(max(1, int(np.ceil(np.sqrt(float(total0) / float(max_raster_px))))))
                    xi = (xi0 // ds).astype(np.int64, copy=False)
                    yi = (yi0 // ds).astype(np.int64, copy=False)
                    w = int(xi.max(initial=0)) + 1
                    h = int(yi.max(initial=0)) + 1
                    img = np.zeros((h, w, 4), dtype=np.float32)
                    img[yi, xi, :3] = cols
                    img[yi, xi, 3] = 1.0
                    step_eff = float(step) * float(ds)
                    x_min_eff = float(x0_int) - 0.5 * step_eff
                    x_max_eff = float(x0_int) + (float(w) - 0.5) * step_eff
                    y_min_eff = float(y0_int) - 0.5 * step_eff
                    y_max_eff = float(y0_int) + (float(h) - 0.5) * step_eff
                    if verbose:
                        logger.info(
                            "gapless grid render: step=%d, ds=%d, buffer=(%d,%d) from (%d,%d) in %.3fs",
                            int(step),
                            int(ds),
                            int(w),
                            int(h),
                            int(w0),
                            int(h0),
                            perf_counter() - t_grid,
                        )
                    ax.imshow(
                        img,
                        origin="lower",
                        extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
                        interpolation="nearest",
                    )
                    ax.set_xlim(x_min_eff, x_max_eff)
                    ax.set_ylim(y_min_eff, y_max_eff)
                    _apply_axis_crop(ax, xlim=xlim, ylim=ylim)
                    ax.set_aspect("equal")
                    if not show_axes:
                        ax.axis("off")
                    title_text = (
                        f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
                        if title is None
                        else title
                    )
                    ax.set_title(title_text, fontsize=figsize[0] * 1.5)
                    if verbose:
                        logger.info("image_plot total time: %.3fs", perf_counter() - t_all)
                    plt.show()
                    return

        raster_canvas = raster_canvas_for_imshow
        s_data_units = float(raster_canvas["s_data_units"])
        x_min_eff = float(raster_canvas["x_min_eff"])
        x_max_eff = float(raster_canvas["x_max_eff"])
        y_min_eff = float(raster_canvas["y_min_eff"])
        y_max_eff = float(raster_canvas["y_max_eff"])
        scale = float(raster_canvas["scale"])
        s = int(raster_canvas["tile_px"])
        w = int(raster_canvas["w"])
        h = int(raster_canvas["h"])
        if verbose and not (coord_mode == "visiumhd" and pixel_perfect):
            logger.info("Rendering to an image buffer of size (w, h): (%d, %d)", int(w), int(h))
        soft_enabled = bool(soft_rasterization)
        if soft_enabled and (coord_mode == "visiumhd" and pixel_perfect):
            if verbose:
                logger.warning("soft_rasterization is ignored for pixel-perfect VisiumHD grid rendering.")
            soft_enabled = False
        if soft_enabled and (not bool(pixel_perfect) or int(s) != 1 or str(pixel_shape).lower() != "square"):
            if verbose:
                logger.warning(
                    "soft_rasterization requires pixel_perfect=True, tile_px==1, and pixel_shape='square'; falling back to hard rasterization."
                )
            soft_enabled = False
        if soft_enabled:
            cx, cy, soft_fx, soft_fy = _raster.scale_points_to_canvas_soft(
                pts,
                x_min_eff=float(x_min_eff),
                y_min_eff=float(y_min_eff),
                scale=float(scale),
                pixel_perfect=True,
                w=int(w),
                h=int(h),
            )
        else:
            soft_fx = None
            soft_fy = None
            cx, cy = _raster.scale_points_to_canvas(
                pts,
                x_min_eff=float(x_min_eff),
                y_min_eff=float(y_min_eff),
                scale=float(scale),
                pixel_perfect=bool(pixel_perfect),
            )
        if coord_mode == "visiumhd" and pixel_perfect:
            x_i = np.floor(x_plot + 0.5).astype(np.int64, copy=False)
            y_i = np.floor(y_plot + 0.5).astype(np.int64, copy=False)
            x0, x1 = int(x_i.min()), int(x_i.max())
            y0, y1 = int(y_i.min()), int(y_i.max())
            w = max(1, int(x1 - x0 + 1))
            h = max(1, int(y1 - y0 + 1))
            cx = (x_i - x0).astype(np.int32, copy=False)
            cy = (y_i - y0).astype(np.int32, copy=False)
            if verbose:
                logger.info("VisiumHD grid raster: buffer (w, h)=(%d, %d)", int(w), int(h))
            s = 1
            x_min_eff, x_max_eff = float(x0) - 0.5, float(x1) + 0.5
            y_min_eff, y_max_eff = float(y0) - 0.5, float(y1) + 0.5
        if verbose:
            logger.info("Scaled tile size (in pixels): %d", int(s))
        collision_info = None
        apply_collision_resolve, collision_stats, _ = _raster.should_resolve_center_collisions(
            resolve_center_collisions,
            cx=cx,
            cy=cy,
            w=int(w),
            h=int(h),
            tile_px=int(s),
            pixel_perfect=bool(pixel_perfect),
            soft_enabled=bool(soft_enabled),
            logger=logger if verbose else None,
            context="image_plot",
        )
        if apply_collision_resolve:
            collision_radius_eff = _raster.resolve_collision_search_radius(
                resolve_center_collisions,
                base_radius=int(center_collision_radius),
                collision_stats=collision_stats,
                logger=logger if verbose else None,
                context="image_plot",
            )
            cx, cy, collision_info = _raster.resolve_center_collisions(
                cx,
                cy,
                w=int(w),
                h=int(h),
                max_radius=int(collision_radius_eff),
                logger=logger if verbose else None,
                context="image_plot",
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
        gap_collapse_active = bool(
            gap_collapse_axis_norm is not None and int(max(0, int(gap_collapse_max_run_px))) > 0
        )
        if gap_collapse_active and soft_enabled:
            if verbose:
                logger.warning(
                    "gap_collapse is currently incompatible with soft_rasterization; falling back to hard rasterization."
                )
            soft_enabled = False
            soft_fx = None
            soft_fy = None
        if gap_collapse_active:
            cx, cy, w, h, _ = _raster.collapse_empty_center_gaps(
                cx=cx,
                cy=cy,
                w=int(w),
                h=int(h),
                axis=gap_collapse_axis_norm,
                max_gap_run_px=int(gap_collapse_max_run_px),
                logger=logger if verbose else None,
                context="image_plot",
            )

        img = np.zeros((h, w, 4), dtype=np.float32)
        paint_mask = np.zeros((h, w), dtype=bool)
        ox, oy = _raster.tile_pixel_offsets(int(s), pixel_shape=pixel_shape)
        pixels_per_tile = int(ox.size)
        target_pixels = int(os.environ.get("SPIX_PLOT_RASTER_TARGET_PIXELS", "10000000"))
        chunk_n = int(max(1, min(int(chunk_size), int(max(1, target_pixels // max(1, pixels_per_tile))))))
        if soft_enabled:
            seed_rgba = np.empty((cols.shape[0], 4), dtype=np.float32)
            seed_rgba[:, :3] = cols.astype(np.float32, copy=False)
            seed_rgba[:, 3] = 1.0
            _raster.rasterize_soft_bilinear(
                img=img,
                occupancy_mask=paint_mask,
                seed_values=seed_rgba,
                cx=cx,
                cy=cy,
                fx=soft_fx,
                fy=soft_fy,
                chunk_size=int(chunk_n),
                background_value=None,
            )
        else:
            for i in range(0, len(cx), chunk_n):
                end = min(i + chunk_n, len(cx))
                all_x = np.clip((cx[i:end, None] + ox[None, :]).ravel(), 0, w - 1)
                all_y = np.clip((cy[i:end, None] + oy[None, :]).ravel(), 0, h - 1)
                cols_rep = np.repeat(cols[i:end], pixels_per_tile, axis=0)
                img[all_y, all_x, :3] = cols_rep
                img[all_y, all_x, 3] = 1.0
                paint_mask[all_y, all_x] = True

        fill_active = bool(runtime_fill_from_boundary) or bool(runtime_fill_holes)
        fill_radius = int(runtime_fill_closing_radius) if bool(runtime_fill_from_boundary) else 0
        external_radius = int(runtime_fill_external_radius) if bool(runtime_fill_from_boundary) else 0
        if fill_active:
            seed_rgba = np.empty((cols.shape[0], 4), dtype=np.float32)
            seed_rgba[:, :3] = cols.astype(np.float32, copy=False)
            seed_rgba[:, 3] = 1.0
            _raster.fill_raster_from_boundary(
                img=img,
                occupancy_mask=paint_mask,
                seed_x=cx,
                seed_y=cy,
                seed_values=seed_rgba,
                closing_radius=int(fill_radius),
                external_radius=int(external_radius),
                fill_holes=bool(runtime_fill_holes),
                use_observed_pixels=bool(int(s) == 1),
                persistent_store=getattr(adata, "uns", None),
                logger=logger if verbose else None,
                context="image_plot",
            )
        if verbose and collision_info is not None:
            logger.info(
                "image_plot: center collision fraction %.4f -> %.4f.",
                float(collision_info["collision_fraction_before"]),
                float(collision_info["collision_fraction_after"]),
            )

        ax.imshow(
            img,
            origin="lower",
            extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
            interpolation="nearest",
        )
        _apply_axis_crop(ax, xlim=xlim, ylim=ylim)
        ax.set_aspect("equal")
        if not show_axes:
            ax.axis("off")
        title_text = (
            f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
            if title is None
            else title
        )
        ax.set_title(title_text, fontsize=figsize[0] * 1.5)
        if verbose:
            logger.info("image_plot total time: %.3fs", perf_counter() - t_all)
        plt.show()
        return
    coord_dict = {
        "x": x_vals[valid],
        "y": y_vals[valid],
        "barcode": tile_barcodes[valid],
    }
    if origin_vals is not None:
        coord_dict["origin"] = origin_vals[valid]
    elif use_direct_origin_spatial:
        coord_dict["origin"] = np.ones(int(np.count_nonzero(valid)), dtype=np.float32)
    coordinates_df = pd.DataFrame(coord_dict)
    coordinates_df.loc[:, dim_names] = embedding_dims[pos[valid]]
    coordinates_df = coordinates_df.reset_index(drop=True)
    if verbose:
        logger.info(
            "Prepared coordinates_df rows=%d in %.3fs",
            int(coordinates_df.shape[0]),
            perf_counter() - t0,
        )

    if coord_mode in {"array", "visium", "visiumhd"}:
        if (array_row_key not in adata.obs.columns) or (array_col_key not in adata.obs.columns):
            raise ValueError(
                f"coordinate_mode='array' requires adata.obs['{array_row_key}'] and adata.obs['{array_col_key}']."
            )
        coordinates_df[array_col_key] = coordinates_df["barcode"].map(adata.obs[array_col_key])
        coordinates_df[array_row_key] = coordinates_df["barcode"].map(adata.obs[array_row_key])
        col = pd.to_numeric(coordinates_df[array_col_key], errors="coerce")
        row = pd.to_numeric(coordinates_df[array_row_key], errors="coerce")
        if coord_mode == "visium":
            # Visium-style hex grid in a rectangular coordinate system.
            # x: stagger columns by row parity; y: compress rows by sqrt(3)/2.
            # This makes nearest-neighbor distances uniform and improves packing.
            row_i = row.astype("Int64")
            parity = (row_i % 2).astype(float)
            coordinates_df["x"] = col.astype(float) + 0.5 * parity
            coordinates_df["y"] = row.astype(float) * (np.sqrt(3.0) / 2.0)
        else:
            coordinates_df["x"] = col.astype(float)
            coordinates_df["y"] = row.astype(float)
        coordinates_df = coordinates_df.dropna(subset=["x", "y"])

    seg_col = None
    has_segments = False
    need_segments = bool(
        plot_boundaries or segment_show_pie or segment_color_by_major or bool(highlight_segments)
    )
    if need_segments and segment_key is not None:
        candidate = segment_key
        if candidate in adata.obs.columns:
            seg_col = candidate
            has_segments = True
        elif candidate != "Segment" and "Segment" in adata.obs.columns:
            logger.warning(
                f"Segment key '{candidate}' not found; falling back to 'Segment' column."
            )
            segment_key = "Segment"
            seg_col = "Segment"
            has_segments = True
        else:
            logger.warning(
                f"Segment key '{candidate}' not found in adata.obs; disabling segment-specific annotations."
            )
    if has_segments and seg_col is not None:
        coordinates_df[seg_col] = coordinates_df["barcode"].map(adata.obs[seg_col])
        if coordinates_df[seg_col].isnull().any():
            missing = coordinates_df[seg_col].isnull().sum()
            logger.warning(
                f"'{seg_col}' label for {missing} barcodes is missing. These rows will be dropped."
            )
            coordinates_df = coordinates_df.dropna(subset=[seg_col])
        coordinates_df["Segment"] = coordinates_df[seg_col]
    else:
        coordinates_df = coordinates_df.drop(columns=["Segment"], errors="ignore")
        has_segments = False
        seg_col = None

    segment_available = has_segments and "Segment" in coordinates_df.columns

    coordinates_df["x"] = pd.to_numeric(coordinates_df["x"], errors="coerce")
    coordinates_df["y"] = pd.to_numeric(coordinates_df["y"], errors="coerce")

    coord_xy = coordinates_df[["x", "y"]].to_numpy(dtype=float, copy=False)
    finite_coord_mask = np.isfinite(coord_xy).all(axis=1)
    if not np.all(finite_coord_mask):
        dropped = int(finite_coord_mask.size - np.count_nonzero(finite_coord_mask))
        logger.warning(
            "Found %d non-finite values in 'x' or 'y' after conversion. Dropping these rows.",
            dropped,
        )
        coordinates_df = coordinates_df.loc[finite_coord_mask].reset_index(drop=True)

    # Values for depth buffering (initial default)
    if len(dimensions) == 1:
        vals = coordinates_df[dim_names[0]].to_numpy()
    else:
        vals = coordinates_df[dim_names].sum(axis=1).to_numpy()

    # Capture original range for colorbar before normalization
    if len(dimensions) == 1:
        raw_vals = coordinates_df[dim_names[0]].values
    else:
        raw_vals = None
    color_norm_vmin = None
    color_norm_vmax = None

    # --- Per-spot alpha mapping (optional) ---
    alpha_arr = None
    if alpha_by is not None:
        if alpha_by == "embedding":
            if len(dimensions) != 1:
                raise ValueError("alpha_by='embedding' requires len(dimensions) == 1.")
            alpha_source = coordinates_df[dim_names[0]].values
        else:
            if alpha_by not in adata.obs.columns:
                raise ValueError(f"alpha_by '{alpha_by}' not found in adata.obs.")
            alpha_source = coordinates_df["barcode"].map(adata.obs[alpha_by]).astype(float).values

        alpha_arr, _ = _compute_alpha_from_values(
            alpha_source, alpha_range=alpha_range, clip=alpha_clip, invert=alpha_invert
        )

    # NEW: If per-spot alpha exists, use it as the depth metric by default.
    # When 'prioritize_high_values=True', higher alpha will be drawn last (in front).
    if alpha_arr is not None:
        vals = alpha_arr.astype(float)

    # Choose coloring strategy: continuous from embedding or categorical from obs
    use_categorical = False
    if color_by is not None:
        # Auto-wire segment annotation settings when pies are shown
        if segment_color_by_major and segment_show_pie:
            if segment_annotate_by is None:
                segment_annotate_by = color_by
            segment_annotate_mode = "pie"

        if color_by not in adata.obs.columns:
            raise ValueError(
                f"'{color_by}' not found in adata.obs. Available: {list(adata.obs.columns)}"
            )
        coordinates_df[color_by] = coordinates_df["barcode"].map(adata.obs[color_by])
        if coordinates_df[color_by].isnull().any():
            missing = coordinates_df[color_by].isnull().sum()
            logger.warning(
                f"'{color_by}' for {missing} barcodes is missing. Dropping these rows."
            )
            coordinates_df = coordinates_df.dropna(subset=[color_by])
        if is_numeric_dtype(coordinates_df[color_by]):
            v = pd.to_numeric(coordinates_df[color_by], errors="coerce").to_numpy(dtype=float)
            raw_vals = v
            finite = np.isfinite(v)
            if finite.any():
                if (color_vmin is not None) or (color_vmax is not None):
                    vmin = float(np.nanmin(v[finite]) if color_vmin is None else color_vmin)
                    vmax = float(np.nanmax(v[finite]) if color_vmax is None else color_vmax)
                else:
                    try:
                        p_lo, p_hi = color_percentiles
                    except Exception:
                        p_lo, p_hi = 1, 99
                    vmin = float(np.nanpercentile(v[finite], p_lo))
                    vmax = float(np.nanpercentile(v[finite], p_hi))
            else:
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-9
            color_norm_vmin, color_norm_vmax = vmin, vmax
            t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
            cm = plt.get_cmap(cmap or "viridis")
            cols = cm(t)[:, :3]
            coordinates = coordinates_df.copy()
            coordinates["R"] = cols[:, 0]
            coordinates["G"] = cols[:, 1]
            coordinates["B"] = cols[:, 2]
            coordinates["Grey"] = np.dot(cols, [0.299, 0.587, 0.114])
            vals = v.astype(float)
            prioritize_high_values = True
            cat_priority_vals = None
            use_categorical = False
            show_legend = False
        else:
            cats = pd.Categorical(coordinates_df[color_by])
            categories = cats.categories
            n_cat = len(categories)

            # Resolve palette to a set of category colors (Nx3 RGB)
            category_lut = None
            try:
                if isinstance(palette, (list, tuple, np.ndarray)):
                    palette_list = list(palette)
                    if len(palette_list) == 0:
                        raise ValueError("Provided palette list is empty.")
                    category_lut = np.array(
                        [mcolors.to_rgb(palette_list[i % len(palette_list)]) for i in range(n_cat)]
                    )
                elif isinstance(palette, str) and palette.startswith("sc.pl.palettes."):
                    try:
                        import scanpy as sc  # type: ignore

                        attr = palette.split(".")[-1]
                        pal_list = getattr(sc.pl.palettes, attr)
                        pal_list = list(pal_list)
                        category_lut = np.array(
                            [mcolors.to_rgb(pal_list[i % len(pal_list)]) for i in range(n_cat)]
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load Scanpy palette '{palette}': {e}. Falling back to matplotlib."
                        )
                if category_lut is None:
                    # Fallback to Matplotlib colormap
                    cmap_cat = plt.get_cmap(palette, n_cat) if isinstance(palette, str) else palette
                    category_lut = np.array([cmap_cat(i)[:3] for i in range(n_cat)])
            except Exception as e:
                logger.warning(f"Palette resolution failed ({e}); falling back to 'tab20'.")
                cmap_cat = plt.get_cmap("tab20", n_cat)
                category_lut = np.array([cmap_cat(i)[:3] for i in range(n_cat)])

            codes = cats.codes
            cols = category_lut[codes]
            coordinates = coordinates_df.copy()

            # Segment-major coloring
            if segment_color_by_major and segment_available:
                try:
                    seg_major = coordinates_df.groupby("Segment")[color_by].agg(lambda s: s.value_counts().idxmax())
                    major_labels = coordinates_df["Segment"].map(seg_major)
                    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
                    major_idx = major_labels.map(cat_to_idx).fillna(0).astype(int)
                    cols = category_lut[major_idx.values]
                except Exception as e:
                    logger.warning(
                        f"Failed to compute segment-major colors; using per-tile colors. Reason: {e}"
                    )

            # Attach RGB for later boundary coloring
            coordinates["R"] = cols[:, 0]
            coordinates["G"] = cols[:, 1]
            coordinates["B"] = cols[:, 2]
            coordinates["Grey"] = np.dot(cols, [0.299, 0.587, 0.114])

            # Category-based overlap priority
            do_cat_priority = overlap_priority in {"few_front", "many_front"}
            if do_cat_priority:
                cat_counts = pd.Series(cats).value_counts().reindex(categories, fill_value=0).values
                row_counts = cat_counts[codes]
                if overlap_priority == "few_front":
                    max_c = row_counts.max() if len(row_counts) > 0 else 0
                    vals = (max_c - row_counts).astype(float)
                else:
                    vals = row_counts.astype(float)
                prioritize_high_values = True
                cat_priority_vals = vals.copy()
            else:
                # CHANGED: Keep alpha-based vals if available; otherwise fall back to neutral/category code.
                if alpha_arr is not None:
                    vals = alpha_arr.astype(float)
                else:
                    vals = np.zeros_like(codes, dtype=float) if not prioritize_high_values else codes.astype(float)
                cat_priority_vals = None

            use_categorical = True
    else:
        # Continuous coloring path
        coordinates = rebalance_colors(
            coordinates_df, dimensions, method=rebalance_method, vmin=rebalance_vmin, vmax=rebalance_vmax
        )
        if len(dimensions) == 1:
            if (rebalance_vmin is not None) or (rebalance_vmax is not None):
                vmin0 = raw_vals.min() if raw_vals is not None else 0.0
                vmax0 = raw_vals.max() if raw_vals is not None else 1.0
                try:
                    vmin0 = float(np.asarray(rebalance_vmin).reshape(-1)[0]) if rebalance_vmin is not None else float(vmin0)
                    vmax0 = float(np.asarray(rebalance_vmax).reshape(-1)[0]) if rebalance_vmax is not None else float(vmax0)
                except Exception:
                    pass
                color_norm_vmin, color_norm_vmax = vmin0, vmax0
            elif raw_vals is not None and len(raw_vals) > 0:
                color_norm_vmin, color_norm_vmax = float(np.nanmin(raw_vals)), float(np.nanmax(raw_vals))
        if len(dimensions) == 3:
            cols = coordinates[["R", "G", "B"]].values

            # Optional brightening / "light mixing" for continuous 3D embeddings.
            # This keeps existing behavior by default and only adjusts when explicitly requested.
            if brighten_continuous:
                # Per-pixel normalization so the strongest channel reaches 1.0,
                # mimicking additive light mixing where strong multi-channel signals look brighter.
                max_per_pixel = cols.max(axis=1, keepdims=True)
                # Avoid division by zero; pixels with all zeros remain black.
                scale = np.where(max_per_pixel > 0, 1.0 / max_per_pixel, 0.0)
                cols = np.clip(cols * scale, 0.0, 1.0)

                # Optional gamma brightening of mid-tones (continuous_gamma < 1 => brighter).
                try:
                    gamma = float(continuous_gamma)
                    if gamma > 0 and gamma != 1.0:
                        cols = np.clip(np.power(cols, gamma), 0.0, 1.0)
                except Exception:
                    # If gamma is invalid, fall back to normalized colors without gamma change.
                    pass
        else:
            grey_vals = coordinates["Grey"].values
            if cmap is None:
                cols = np.repeat(grey_vals[:, np.newaxis], 3, axis=1)
            else:
                cm = plt.get_cmap(cmap)
                cols = cm(grey_vals)[:, :3]

    x_min, x_max = coordinates["x"].min(), coordinates["x"].max()
    y_min, y_max = coordinates["y"].min(), coordinates["y"].max()
    plot_boundary_polygon = Polygon(
        [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
    )

    # Auto-adjust figsize/dpi (when either is None) so the canvas has enough pixels
    # to represent the full tile grid without forced downsampling.
    user_figsize_given = figsize is not None
    user_dpi_given = fig_dpi is not None
    auto_sized = (not user_figsize_given) or (not user_dpi_given)
    pts_for_imshow = None
    s_data_units_for_imshow = None
    raster_canvas_for_imshow = None

    if use_imshow:
        # Use the full raster tile count for origin=False auto canvas sizing.
        n_points_eff = int(coordinates.shape[0])

        pts = coordinates[["x", "y"]].values
        pts_for_imshow = pts
        imshow_tile_size_eff = imshow_tile_size
        if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
            # VisiumHD uses a dense square lattice in array_row/array_col space.
            # Force pitch=1 (in array units) unless the user overrides it.
            imshow_tile_size_eff = 1.0
        raster_canvas_for_imshow = _raster.resolve_raster_canvas(
            pts=pts,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            figsize=figsize,
            fig_dpi=fig_dpi,
            imshow_tile_size=imshow_tile_size_eff,
            imshow_scale_factor=imshow_scale_factor,
            imshow_tile_size_mode=imshow_tile_size_mode,
            imshow_tile_size_quantile=imshow_tile_size_quantile,
            imshow_tile_size_rounding=imshow_tile_size_rounding,
            imshow_tile_size_shrink=imshow_tile_size_shrink,
            pixel_perfect=bool(pixel_perfect),
            n_points=int(n_points_eff),
            logger=logger,
            context="image_plot(auto-size)",
            persistent_store=getattr(adata, "uns", None),
        )
        s_data_units_for_imshow = float(raster_canvas_for_imshow["s_data_units"])
        figsize = raster_canvas_for_imshow["figsize"]
        fig_dpi = raster_canvas_for_imshow["fig_dpi"]

    if figsize is None:
        figsize = (10, 10)

    if fig_dpi is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)
    if verbose:
        logger.info(
            "Auto-size uses n_eff=%d (rows=%d); final figsize=%s, fig_dpi=%s",
            int(n_points_eff) if "n_points_eff" in locals() else int(coordinates.shape[0]),
            int(coordinates.shape[0]),
            tuple(figsize),
            fig.get_dpi(),
        )
    linestyle_options = {"solid": "-", "dashed": "--", "dotted": ":"}
    linestyle = linestyle_options.get(boundary_style, "-")

    color_for_pixel = (
        fixed_boundary_color if fixed_boundary_color is not None else boundary_color
    )
    stored_pixel_boundary = None
    if segment_available and plot_boundaries and boundary_method == "pixel":
        seg_uns_key = f"{_sanitize_segment_key_for_uns(seg_col or segment_key)}_pixel_boundary"
        candidate_payload = adata.uns.get(seg_uns_key, None)
        if isinstance(candidate_payload, dict):
            try:
                if bool(candidate_payload.get("origin", origin)) == bool(origin):
                    stored_pixel_boundary = candidate_payload
                    if verbose:
                        logger.info("image_plot: using stored pixel boundary payload from adata.uns['%s'].", seg_uns_key)
            except Exception:
                stored_pixel_boundary = None

    if segment_available:
        # Normalize highlight_segments input
        if highlight_segments is not None:
            if isinstance(highlight_segments, str):
                highlight_segments = {highlight_segments}
            else:
                highlight_segments = set(highlight_segments)
        else:
            highlight_segments = set()

        highlight_mask = coordinates_df["Segment"].isin(highlight_segments)
        cols = cols.copy()
        if highlight_segments:
            if dim_to_grey and len(dimensions) == 3:
                grey_values = np.dot(cols[~highlight_mask], [0.299, 0.587, 0.114])
                grey_colors = np.repeat(grey_values[:, np.newaxis], 3, axis=1)
                cols[~highlight_mask] = grey_colors * dim_other_segments
            else:
                cols[~highlight_mask] = cols[~highlight_mask] * dim_other_segments
            if highlight_brighten_factor != 1:
                cols[highlight_mask] = brighten_colors(
                    cols[highlight_mask], factor=highlight_brighten_factor
                )

        if highlight_linewidth is None:
            highlight_linewidth = boundary_linewidth * 2

    if use_imshow:
        # Optional gapless packed-grid rendering for raster-expanded tiles (origin=False).
        # This avoids visible "holes" when raster_stride > 1 and is much faster because
        # it paints one pixel per sampled coordinate (with optional downsampling).
        if gapless and (not origin) and (not plot_boundaries) and ("origin" in coordinates.columns):
            try:
                origin_vals_eff = pd.to_numeric(coordinates["origin"], errors="coerce").fillna(1)
                has_non_origin = (origin_vals_eff != 1).any()
            except Exception:
                has_non_origin = False
            if has_non_origin and ("x" in coordinates.columns) and ("y" in coordinates.columns):
                x = coordinates["x"].to_numpy(dtype=float, copy=False)
                y = coordinates["y"].to_numpy(dtype=float, copy=False)
                step_x = _infer_regular_grid_step(x)
                step_y = _infer_regular_grid_step(y)
                step = None
                if step_x is not None and step_y is not None:
                    step = int(max(1, min(int(step_x), int(step_y))))
                elif step_x is not None:
                    step = int(max(1, int(step_x)))
                elif step_y is not None:
                    step = int(max(1, int(step_y)))

                if step is not None:
                    t_grid = perf_counter()
                    xi0, yi0, w0, h0, x0_int, y0_int = _coords_to_grid_indices(x, y, step=step)

                    max_raster_px = _raster.resolve_max_raster_pixels(
                        env_key="SPIX_PLOT_MAX_RASTER_PIXELS",
                        n_points=len(x),
                    )
                    total0 = int(w0) * int(h0)
                    ds = int(max(1, int(np.ceil(np.sqrt(float(total0) / float(max_raster_px))))))
                    xi = (xi0 // ds).astype(np.int64, copy=False)
                    yi = (yi0 // ds).astype(np.int64, copy=False)
                    w = int(xi.max(initial=0)) + 1
                    h = int(yi.max(initial=0)) + 1

                    img = np.zeros((h, w, 4), dtype=np.float32)
                    img[yi, xi, :3] = cols
                    if alpha_arr is not None:
                        img[yi, xi, 3] = np.asarray(alpha_arr, dtype=np.float32)
                    else:
                        img[yi, xi, 3] = 1.0

                    step_eff = float(step) * float(ds)
                    x_min_eff = float(x0_int) - 0.5 * step_eff
                    x_max_eff = float(x0_int) + (float(w) - 0.5) * step_eff
                    y_min_eff = float(y0_int) - 0.5 * step_eff
                    y_max_eff = float(y0_int) + (float(h) - 0.5) * step_eff

                    if verbose:
                        logger.info(
                            "gapless grid render: step=%d, ds=%d, buffer=(%d,%d) from (%d,%d) in %.3fs",
                            int(step),
                            int(ds),
                            int(w),
                            int(h),
                            int(w0),
                            int(h0),
                            perf_counter() - t_grid,
                        )
                    ax.imshow(
                        img,
                        origin="lower",
                        extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
                        interpolation="nearest",
                    )
                    ax.set_xlim(x_min_eff, x_max_eff)
                    ax.set_ylim(y_min_eff, y_max_eff)
                    _apply_axis_crop(ax, xlim=xlim, ylim=ylim)
                    if verbose:
                        logger.info("image_plot total time: %.3fs", perf_counter() - t_all)
                    plt.show()
                    return

        # --- Stable IMShow rendering with RGBA for per-spot alpha ---
        # 1) Estimate tile size in data units
        pts = pts_for_imshow if pts_for_imshow is not None else coordinates[["x", "y"]].values
        imshow_tile_size_eff = imshow_tile_size
        if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
            imshow_tile_size_eff = 1.0
        raster_canvas = raster_canvas_for_imshow
        if raster_canvas is None:
            raster_context = "image_plot(pixel-boundary)" if (plot_boundaries and boundary_method == "pixel") else "image_plot"
            raster_context = (
                "image_plot_with_spatial_image(pixel-boundary)"
                if (plot_boundaries and boundary_method == "pixel")
                else "image_plot_with_spatial_image"
            )
            raster_canvas = _raster.resolve_raster_canvas(
                pts=pts,
                x_min=float(x_min),
                x_max=float(x_max),
                y_min=float(y_min),
                y_max=float(y_max),
                figsize=figsize,
                fig_dpi=fig.get_dpi(),
                imshow_tile_size=imshow_tile_size_eff,
                imshow_scale_factor=imshow_scale_factor,
                imshow_tile_size_mode=imshow_tile_size_mode,
                imshow_tile_size_quantile=imshow_tile_size_quantile,
                imshow_tile_size_rounding=imshow_tile_size_rounding,
                imshow_tile_size_shrink=imshow_tile_size_shrink,
                pixel_perfect=bool(pixel_perfect),
                n_points=int(n_points_eff),
                logger=logger,
                context=raster_context,
                persistent_store=getattr(adata, "uns", None),
            )
        s_data_units = float(raster_canvas["s_data_units"])
        x_min_eff = float(raster_canvas["x_min_eff"])
        x_max_eff = float(raster_canvas["x_max_eff"])
        y_min_eff = float(raster_canvas["y_min_eff"])
        y_max_eff = float(raster_canvas["y_max_eff"])
        scale = float(raster_canvas["scale"])
        s = int(raster_canvas["tile_px"])
        w = int(raster_canvas["w"])
        h = int(raster_canvas["h"])
        if verbose and not (coord_mode == "visiumhd" and pixel_perfect):
            logger.info("Rendering to an image buffer of size (w, h): (%d, %d)", int(w), int(h))

        # 3) Scale coordinates and tile size to buffer
        # Avoid banker's rounding ties (e.g., *.5) on regular grids when pixel_perfect=True.
        soft_enabled = bool(soft_rasterization)
        if soft_enabled and (coord_mode == "visiumhd" and pixel_perfect):
            if verbose:
                logger.warning("soft_rasterization is ignored for pixel-perfect VisiumHD grid rendering.")
            soft_enabled = False
        if soft_enabled and (not bool(pixel_perfect) or int(s) != 1 or str(pixel_shape).lower() != "square"):
            if verbose:
                logger.warning(
                    "soft_rasterization requires pixel_perfect=True, tile_px==1, and pixel_shape='square'; falling back to hard rasterization."
                )
            soft_enabled = False
        if soft_enabled:
            cx, cy, soft_fx, soft_fy = _raster.scale_points_to_canvas_soft(
                coordinates[["x", "y"]].to_numpy(dtype=float, copy=False),
                x_min_eff=float(x_min_eff),
                y_min_eff=float(y_min_eff),
                scale=float(scale),
                pixel_perfect=True,
                w=int(w),
                h=int(h),
            )
        else:
            soft_fx = None
            soft_fy = None
            cx, cy = _raster.scale_points_to_canvas(
                coordinates[["x", "y"]].to_numpy(dtype=float, copy=False),
                x_min_eff=float(x_min_eff),
                y_min_eff=float(y_min_eff),
                scale=float(scale),
                pixel_perfect=bool(pixel_perfect),
            )
        if coord_mode == "visiumhd" and pixel_perfect:
            # VisiumHD: rasterize directly on the (array_row/array_col) grid.
            # This removes any dependence on fig pixel scaling and eliminates overlap/gap artifacts.
            x_vals = np.asarray(coordinates["x"].to_numpy(), dtype=float)
            y_vals = np.asarray(coordinates["y"].to_numpy(), dtype=float)
            x_i = np.floor(x_vals + 0.5).astype(np.int64, copy=False)
            y_i = np.floor(y_vals + 0.5).astype(np.int64, copy=False)
            x0, x1 = int(x_i.min()), int(x_i.max())
            y0, y1 = int(y_i.min()), int(y_i.max())
            w = max(1, int(x1 - x0 + 1))
            h = max(1, int(y1 - y0 + 1))
            cx = (x_i - x0).astype(np.int32, copy=False)
            cy = (y_i - y0).astype(np.int32, copy=False)
            if verbose:
                logger.info("VisiumHD grid raster: buffer (w, h)=(%d, %d)", int(w), int(h))
                flat = cx.astype(np.int64) + cy.astype(np.int64) * int(w)
                n_unique = int(np.unique(flat).size)
                if n_unique != int(len(flat)):
                    logger.warning(
                        "VisiumHD grid centers collide: %d/%d unique (duplicates=%d).",
                        n_unique,
                        int(len(flat)),
                        int(len(flat) - n_unique),
                    )
            s = 1
            scale = 1.0
            x_min_eff, x_max_eff = float(x0) - 0.5, float(x1) + 0.5
            y_min_eff, y_max_eff = float(y0) - 0.5, float(y1) + 0.5
        if verbose:
            logger.info("Scaled tile size (in pixels): %d", int(s))
        collision_info = None
        apply_collision_resolve, collision_stats, _ = _raster.should_resolve_center_collisions(
            resolve_center_collisions,
            cx=cx,
            cy=cy,
            w=int(w),
            h=int(h),
            tile_px=int(s),
            pixel_perfect=bool(pixel_perfect),
            soft_enabled=bool(soft_enabled),
            logger=logger if verbose else None,
            context="image_plot",
        )
        if apply_collision_resolve:
            collision_radius_eff = _raster.resolve_collision_search_radius(
                resolve_center_collisions,
                base_radius=int(center_collision_radius),
                collision_stats=collision_stats,
                logger=logger if verbose else None,
                context="image_plot",
            )
            cx, cy, collision_info = _raster.resolve_center_collisions(
                cx,
                cy,
                w=int(w),
                h=int(h),
                max_radius=int(collision_radius_eff),
                logger=logger if verbose else None,
                context="image_plot",
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
        gap_collapse_active = bool(
            gap_collapse_axis_norm is not None and int(max(0, int(gap_collapse_max_run_px))) > 0
        )
        if gap_collapse_active and soft_enabled:
            if verbose:
                logger.warning(
                    "gap_collapse is currently incompatible with soft_rasterization; falling back to hard rasterization."
                )
            soft_enabled = False
            soft_fx = None
            soft_fy = None
        if gap_collapse_active:
            cx, cy, w, h, _ = _raster.collapse_empty_center_gaps(
                cx=cx,
                cy=cy,
                w=int(w),
                h=int(h),
                axis=gap_collapse_axis_norm,
                max_gap_run_px=int(gap_collapse_max_run_px),
                logger=logger if verbose else None,
                context="image_plot",
            )

        # 4) RGBA buffer + depth buffer
        img = np.zeros((h, w, 4), dtype=np.float32)  # start fully transparent
        paint_mask = np.zeros((h, w), dtype=bool)
        value_buf = np.full((h, w), -np.inf, dtype=float)
        use_stored_boundary_labels = stored_pixel_boundary is not None
        label_img = None
        if use_stored_boundary_labels:
            label_img, _ = _resolve_stored_pixel_boundary_payload(
                stored_pixel_boundary,
                default_extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
            )
        elif plot_boundaries and boundary_method == "pixel" and segment_available:
            label_img = np.full((h, w), -1, dtype=int)
            seg_codes = pd.Categorical(coordinates_df["Segment"]).codes

        # Precompute offsets for circle/square tiles (relative to the tile center).
        ox, oy = _raster.tile_pixel_offsets(int(s), pixel_shape=pixel_shape)
        pixels_per_tile = int(ox.size)

        # Keep intermediate index arrays bounded to reduce RAM spikes and improve speed.
        target_pixels = int(os.environ.get("SPIX_PLOT_RASTER_TARGET_PIXELS", "10000000"))
        chunk_n = int(max(1, min(int(chunk_size), int(max(1, target_pixels // max(1, pixels_per_tile))))))

        if soft_enabled:
            if prioritize_high_values and verbose:
                logger.warning("soft_rasterization does not support prioritize_high_values; using weighted blending instead.")
            seed_rgba = np.empty((cols.shape[0], 4), dtype=np.float32)
            seed_rgba[:, :3] = cols.astype(np.float32, copy=False)
            if alpha_arr is not None:
                seed_rgba[:, 3] = np.asarray(alpha_arr, dtype=np.float32)
            else:
                seed_rgba[:, 3] = 1.0
            _raster.rasterize_soft_bilinear(
                img=img,
                occupancy_mask=paint_mask,
                seed_values=seed_rgba,
                cx=cx,
                cy=cy,
                fx=soft_fx,
                fy=soft_fy,
                chunk_size=int(chunk_n),
                background_value=None,
            )
            if label_img is not None and (not use_stored_boundary_labels):
                for i in range(0, len(cx), chunk_n):
                    end = min(i + chunk_n, len(cx))
                    x0, y0, x1, y1, w00, w10, w01, w11 = _raster.bilinear_support_from_centers(
                        cx[i:end],
                        cy[i:end],
                        soft_fx[i:end],
                        soft_fy[i:end],
                        w=int(w),
                        h=int(h),
                    )
                    seg_chunk = seg_codes[i:end]
                    for px, py, pw in ((x0, y0, w00), (x1, y0, w10), (x0, y1, w01), (x1, y1, w11)):
                        mask_upd = pw > value_buf[py, px]
                        if np.any(mask_upd):
                            label_img[py[mask_upd], px[mask_upd]] = seg_chunk[mask_upd]
                            value_buf[py[mask_upd], px[mask_upd]] = pw[mask_upd]
        else:
            for i in range(0, len(cx), chunk_n):
                end = min(i + chunk_n, len(cx))
                chunk_cx, chunk_cy = cx[i:end], cy[i:end]
                chunk_cols = cols[i:end]
                chunk_vals = vals[i:end]  # depth metric; with alpha_arr set, this is alpha
                chunk_alpha = alpha_arr[i:end] if alpha_arr is not None else None

                if not prioritize_high_values:
                    # Fast path: paint all pixels in one vectorized pass (order matters only for overlaps).
                    all_x = np.clip((chunk_cx[:, None] + ox[None, :]).ravel(), 0, w - 1)
                    all_y = np.clip((chunk_cy[:, None] + oy[None, :]).ravel(), 0, h - 1)
                    cols_rep = np.repeat(chunk_cols, pixels_per_tile, axis=0)
                    img[all_y, all_x, :3] = cols_rep
                    if chunk_alpha is not None:
                        img[all_y, all_x, 3] = np.repeat(chunk_alpha, pixels_per_tile, axis=0)
                    else:
                        img[all_y, all_x, 3] = 1.0
                    paint_mask[all_y, all_x] = True
                    if label_img is not None and (not use_stored_boundary_labels):
                        label_img[all_y, all_x] = np.repeat(seg_codes[i:end], pixels_per_tile, axis=0)
                else:
                    # Slow path: depth buffer / draw priority.
                    for dx, dy in zip(ox, oy):
                        px = np.clip(chunk_cx + dx, 0, w - 1)
                        py = np.clip(chunk_cy + dy, 0, h - 1)
                        mask_upd = chunk_vals > value_buf[py, px]
                        if np.any(mask_upd):
                            img[py[mask_upd], px[mask_upd], :3] = chunk_cols[mask_upd]
                            img[py[mask_upd], px[mask_upd], 3] = (
                                chunk_alpha[mask_upd] if chunk_alpha is not None else 1.0
                            )
                            value_buf[py[mask_upd], px[mask_upd]] = chunk_vals[mask_upd]
                            paint_mask[py[mask_upd], px[mask_upd]] = True
                            if label_img is not None and (not use_stored_boundary_labels):
                                label_img[py[mask_upd], px[mask_upd]] = seg_codes[i:end][mask_upd]

        label_paint_mask = paint_mask.copy() if (label_img is not None and (not use_stored_boundary_labels)) else None
        fill_active = bool(runtime_fill_from_boundary) or bool(runtime_fill_holes)
        fill_radius = int(runtime_fill_closing_radius) if bool(runtime_fill_from_boundary) else 0
        external_radius = int(runtime_fill_external_radius) if bool(runtime_fill_from_boundary) else 0
        if fill_active:
            seed_rgba = np.empty((cols.shape[0], 4), dtype=np.float32)
            seed_rgba[:, :3] = cols.astype(np.float32, copy=False)
            if alpha_arr is not None:
                seed_rgba[:, 3] = np.asarray(alpha_arr, dtype=np.float32)
            else:
                seed_rgba[:, 3] = 1.0
            _raster.fill_raster_from_boundary(
                img=img,
                occupancy_mask=paint_mask,
                seed_x=cx,
                seed_y=cy,
                seed_values=seed_rgba,
                closing_radius=int(fill_radius),
                external_radius=int(external_radius),
                fill_holes=bool(runtime_fill_holes),
                use_observed_pixels=bool(int(s) == 1),
                persistent_store=getattr(adata, "uns", None),
                logger=logger if verbose else None,
                context="image_plot",
            )
        if label_img is not None and (not use_stored_boundary_labels):
            label_img[~label_paint_mask] = -1
        if verbose and collision_info is not None:
            logger.info(
                "image_plot: center collision fraction %.4f -> %.4f.",
                float(collision_info["collision_fraction_before"]),
                float(collision_info["collision_fraction_after"]),
            )

        ax.imshow(
            img,
            origin="lower",
            extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
            interpolation="nearest",
        )
        if verbose:
            logger.info("image_plot total time: %.3fs", perf_counter() - t_all)

        # Categorical legend (imshow path)
        if use_categorical and show_legend:
            categories = pd.Categorical(coordinates_df[color_by]).categories
            legend_colors = [category_lut[i] for i in range(len(categories))]
            handles = [patches.Patch(color=legend_colors[i], label=str(cat)) for i, cat in enumerate(categories)]
            if legend_outside:
                fig.subplots_adjust(right=0.82)
                loc_final = 'center left' if legend_loc == 'best' else legend_loc
                if legend_fontsize is None:
                    n_items = len(categories)
                    rows = int(np.ceil(n_items / max(1, legend_ncol)))
                    size_factor = (figsize[0] * figsize[1]) ** 0.5
                    base = 12 + (size_factor - 9) * 0.5
                    if rows > 8: base -= 2
                    elif rows > 5: base -= 1
                    auto_fs = int(np.clip(round(base), 10, 28))
                else:
                    auto_fs = legend_fontsize
                fig.legend(handles=handles, loc=loc_final, bbox_to_anchor=legend_bbox_to_anchor,
                           ncol=legend_ncol, frameon=False, fontsize=auto_fs)
            else:
                if legend_fontsize is None:
                    n_items = len(categories)
                    rows = int(np.ceil(n_items / max(1, legend_ncol)))
                    size_factor = (figsize[0] * figsize[1]) ** 0.5
                    base = 12 + (size_factor - 9) * 0.5
                    if rows > 8: base -= 2
                    elif rows > 5: base -= 1
                    auto_fs = int(np.clip(round(base), 10, 28))
                else:
                    auto_fs = legend_fontsize
                ax.legend(handles=handles, loc=legend_loc, ncol=legend_ncol, frameon=False, fontsize=auto_fs)

        # --- Segment pies (imshow path) ---
        if segment_show_pie and (segment_annotate_by is not None) and segment_available:
            seg_ser = coordinates_df["Segment"].astype("category")
            comp_ser = coordinates_df["barcode"].map(adata.obs[segment_annotate_by])
            comp_ser = pd.Categorical(comp_ser)
            seg_centroids = coordinates.groupby(seg_ser).agg({"x": "median", "y": "median"})
            comp_df = pd.crosstab(seg_ser, comp_ser, dropna=False)
            seg_sizes = comp_df.sum(axis=1)
            valid_segs = seg_sizes[seg_sizes >= max(1, segment_min_tiles)].index
            comp_df = comp_df.loc[valid_segs]
            seg_centroids = seg_centroids.loc[valid_segs]

            comp_cats = comp_df.columns
            comp_n = len(comp_cats)
            try:
                if (color_by is not None) and (segment_annotate_by == color_by):
                    cat_to_color = {str(cat): category_lut[i] for i, cat in enumerate(categories)}
                    comp_lut = np.array([cat_to_color.get(str(cat), (0.7, 0.7, 0.7)) for cat in comp_cats])
                else:
                    if isinstance(palette, (list, tuple, np.ndarray)):
                        pal = list(palette)
                        comp_lut = np.array([mcolors.to_rgb(pal[i % len(pal)]) for i in range(comp_n)])
                    elif isinstance(palette, str) and palette.startswith("sc.pl.palettes."):
                        try:
                            import scanpy as sc
                            attr = palette.split(".")[-1]
                            pal_list = list(getattr(sc.pl.palettes, attr))
                            comp_lut = np.array([mcolors.to_rgb(pal_list[i % len(pal_list)]) for i in range(comp_n)])
                        except Exception:
                            cmc = plt.get_cmap("tab20", comp_n)
                            comp_lut = np.array([cmc(i)[:3] for i in range(comp_n)])
                    else:
                        cmc = plt.get_cmap(palette if isinstance(palette, str) else "tab20", comp_n)
                        comp_lut = np.array([cmc(i)[:3] for i in range(comp_n)])
            except Exception:
                cmc = plt.get_cmap("tab20", comp_n)
                comp_lut = np.array([cmc(i)[:3] for i in range(comp_n)])

            pts = coordinates[["x", "y"]].values
            if len(pts) > 1:
                sample_size = min(len(pts), 100000)
                sample_indices = np.random.choice(len(pts), sample_size, replace=False)
                tree_sz = KDTree(pts[sample_indices])
                d_sz, _ = tree_sz.query(pts[sample_indices], k=2)
                nn = d_sz[:, 1][d_sz[:, 1] > 0]
                s_est = np.median(nn) if len(nn) > 0 else 1.0
            else:
                s_est = 1.0
            pie_r = max(0.5, s_est * float(segment_pie_scale))

            for seg, centroid in seg_centroids.iterrows():
                counts = comp_df.loc[seg]
                total = counts.sum()
                if total <= 0:
                    continue
                sorted_items = counts.sort_values(ascending=False)
                top = sorted_items.iloc[: max(1, int(segment_top_n))]
                rest = sorted_items.iloc[max(1, int(segment_top_n)) :].sum()
                labels = list(top.index)
                vals_local = list(top.values.astype(float))
                colors_local = [comp_lut[list(comp_cats).index(lbl)] if lbl in comp_cats else (0.7, 0.7, 0.7) for lbl in labels]
                if rest > 0:
                    labels.append("Other"); vals_local.append(float(rest)); colors_local.append((0.8, 0.8, 0.8))
                val_sum = sum(vals_local)
                angle = 0.0
                for v, c in zip(vals_local, colors_local):
                    theta = 360.0 * (v / val_sum) if val_sum > 0 else 0
                    if theta <= 0: continue
                    wdg = patches.Wedge((centroid["x"], centroid["y"]), pie_r, angle, angle + theta,
                                        facecolor=c, edgecolor=segment_pie_edgecolor,
                                        linewidth=segment_pie_linewidth, alpha=0.9)
                    ax.add_patch(wdg)
                    angle += theta

        if label_img is not None:
            if stored_pixel_boundary is not None:
                _, boundary_extent = _resolve_stored_pixel_boundary_payload(
                    stored_pixel_boundary,
                    default_extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
                )
            else:
                boundary_extent = [x_min_eff, x_max_eff, y_min_eff, y_max_eff]
            _plot_boundaries_from_label_image(
                ax,
                label_img,
                boundary_extent,
                color_for_pixel,
                boundary_linewidth,
                alpha,
                linestyle,
                pixel_smoothing_sigma,
                render=pixel_boundary_render,
                thickness_px=pixel_boundary_thickness_px,
                antialiased=pixel_boundary_antialiased,
                include_background=pixel_boundary_include_background,
            )
            if highlight_segments:
                categories = pd.Categorical(coordinates_df["Segment"]).categories
                highlight_codes = [
                    categories.get_loc(seg)
                    for seg in highlight_segments
                    if seg in categories
                ]
                if highlight_codes:
                    mask = np.isin(label_img, highlight_codes)
                    if np.any(mask):
                        highlight_label_img = np.where(mask, label_img, -1)
                        if highlight_boundary_color is not None:
                            highlight_col = highlight_boundary_color
                        else:
                            highlight_col = color_for_pixel
                            if highlight_brighten_factor != 1:
                                base_col = np.array([mcolors.to_rgb(color_for_pixel)])
                                highlight_col = brighten_colors(
                                    base_col, factor=highlight_brighten_factor
                                )[0]
                        _plot_boundaries_from_label_image(
                            ax,
                            highlight_label_img,
                            boundary_extent,
                            highlight_col,
                            highlight_linewidth,
                            alpha,
                            linestyle,
                            pixel_smoothing_sigma,
                            render=pixel_boundary_render,
                            thickness_px=pixel_boundary_thickness_px,
                            antialiased=pixel_boundary_antialiased,
                            include_background=pixel_boundary_include_background,
                        )

    else:
        # ---Automatic point size calculation for scatter plots ---
        if point_size is None:
            num_points = len(coordinates)
            base_size = (figsize[0] * figsize[1] * 100**2) / (num_points + 1)
            point_size = (scaling_factor / 5.0) * base_size
            point_size = np.clip(point_size, 0.01, 200)

        # Alpha-driven draw order for scatter:
        # If alpha_arr exists and prioritize_high_values=True (and not using categorical overlap ordering),
        # draw transparent points first, opaque points last (on top).
        if alpha_arr is not None and prioritize_high_values and overlap_priority not in {"few_front", "many_front"}:
            order_idx = np.argsort(alpha_arr)  # ascending: low alpha first
            coordinates = coordinates.iloc[order_idx]
            cols = cols[order_idx]
            alpha_arr = alpha_arr[order_idx]

        # Category-based overlap priority (scatter)
        if color_by is not None and overlap_priority in {"few_front", "many_front"}:
            if overlap_priority == "few_front":
                order_idx = np.argsort(cat_priority_vals)[::-1]
            else:
                order_idx = np.argsort(cat_priority_vals)
            coordinates = coordinates.iloc[order_idx]
            cols = cols[order_idx]
            if alpha_arr is not None:
                alpha_arr = alpha_arr[order_idx]

        # Build RGBA for per-point alpha (do not pass `alpha=` concurrently)
        if alpha_arr is not None:
            cols_rgba = np.concatenate([cols, alpha_arr[:, None]], axis=1)
        else:
            cols_rgba = np.concatenate([cols, np.ones((cols.shape[0], 1), dtype=cols.dtype)], axis=1)

        ax.scatter(
            x=coordinates["x"],
            y=coordinates["y"],
            s=point_size,
            c=cols_rgba,
            marker="o" if pixel_shape == "circle" else "s",
            linewidths=0,
        )

        # Categorical legend
        if color_by is not None and show_legend:
            categories = pd.Categorical(coordinates_df[color_by]).categories
            legend_colors = [category_lut[i] for i in range(len(categories))]
            handles = [patches.Patch(color=legend_colors[i], label=str(cat)) for i, cat in enumerate(categories)]
            if legend_outside:
                fig.subplots_adjust(right=0.82)
                loc_final = 'center left' if legend_loc == 'best' else legend_loc
                if legend_fontsize is None:
                    n_items = len(categories)
                    rows = int(np.ceil(n_items / max(1, legend_ncol)))
                    size_factor = (figsize[0] * figsize[1]) ** 0.5
                    base = 12 + (size_factor - 9) * 0.5
                    if rows > 8: base -= 2
                    elif rows > 5: base -= 1
                    auto_fs = int(np.clip(round(base), 10, 28))
                else:
                    auto_fs = legend_fontsize
                fig.legend(handles=handles, loc=loc_final, bbox_to_anchor=legend_bbox_to_anchor,
                           ncol=legend_ncol, frameon=False, fontsize=auto_fs)
            else:
                if legend_fontsize is None:
                    n_items = len(categories)
                    rows = int(np.ceil(n_items / max(1, legend_ncol)))
                    size_factor = (figsize[0] * figsize[1]) ** 0.5
                    base = 12 + (size_factor - 9) * 0.5
                    if rows > 8: base -= 2
                    elif rows > 5: base -= 1
                    auto_fs = int(np.clip(round(base), 10, 28))
                else:
                    auto_fs = legend_fontsize
                ax.legend(handles=handles, loc=legend_loc, ncol=legend_ncol, frameon=False, fontsize=auto_fs)

        # Pixel-style boundaries from scatter points
        if plot_boundaries and boundary_method == "pixel" and segment_available:
            fig_dpi = fig.get_dpi()
            w_pixels = int(np.ceil(figsize[0] * fig_dpi))
            h_pixels = int(np.ceil(figsize[1] * fig_dpi))

            x_range = x_max - x_min if x_max > x_min else 1
            y_range = y_max - y_min if y_max > y_min else 1

            scale = (
                min(w_pixels / x_range, h_pixels / y_range)
                if x_range > 0 and y_range > 0
                else 1
            )

            w = int(np.ceil(x_range * scale))
            h = int(np.ceil(y_range * scale))
            w, h = max(w, 1), max(h, 1)

            xi = ((coordinates["x"] - x_min) * scale).astype(int).values
            yi = ((coordinates["y"] - y_min) * scale).astype(int).values

            pts = coordinates[["x", "y"]].values
            s_data_units = _estimate_tile_size_units(
                pts,
                imshow_tile_size,
                imshow_scale_factor,
                imshow_tile_size_mode,
                imshow_tile_size_quantile,
                imshow_tile_size_shrink,
                logger=None,
            )
            s = _round_tile_size_px(s_data_units, scale, imshow_tile_size_rounding)

            label_img = np.full((h, w), -1, dtype=int)
            seg_codes = pd.Categorical(coordinates_df["Segment"]).codes

            if pixel_shape == "circle":
                yy, xx = np.ogrid[:s, :s]
                center = (s - 1) / 2
                mask = (xx - center) ** 2 + (yy - center) ** 2 <= (s / 2) ** 2
                circle_y_offsets, circle_x_offsets = np.nonzero(mask)
            else:
                tile_x_offsets, tile_y_offsets = np.meshgrid(np.arange(s), np.arange(s))

            for i in range(0, len(xi), chunk_size):
                end = min(i + chunk_size, len(xi))
                chunk_xi, chunk_yi = xi[i:end], yi[i:end]

                if pixel_shape == "circle":
                    for dx, dy in zip(circle_x_offsets, circle_y_offsets):
                        cx = np.clip(chunk_xi + dx, 0, w - 1)
                        cy = np.clip(chunk_yi + dy, 0, h - 1)
                        label_img[cy, cx] = seg_codes[i:end]
                else:
                    chunk_all_x = chunk_xi[:, None, None] + tile_x_offsets
                    chunk_all_y = chunk_yi[:, None, None] + tile_y_offsets

                    chunk_all_x = np.clip(chunk_all_x, 0, w - 1)
                    chunk_all_y = np.clip(chunk_all_y, 0, h - 1)

                    chunk_labels = seg_codes[i:end]
                    label_img[chunk_all_y, chunk_all_x] = chunk_labels[:, None, None]

            _plot_boundaries_from_label_image(
                ax,
                label_img,
                [x_min, x_max, y_min, y_max],
                color_for_pixel,
                boundary_linewidth,
                alpha,
                linestyle,
                pixel_smoothing_sigma,
                render=pixel_boundary_render,
                thickness_px=pixel_boundary_thickness_px,
                antialiased=pixel_boundary_antialiased,
                include_background=pixel_boundary_include_background,
            )

            if highlight_segments:
                categories = pd.Categorical(coordinates_df["Segment"]).categories
                highlight_codes = [
                    categories.get_loc(seg)
                    for seg in highlight_segments
                    if seg in categories
                ]
                if highlight_codes:
                    mask = np.isin(label_img, highlight_codes)
                    if np.any(mask):
                        highlight_label_img = np.where(mask, label_img, -1)
                        if highlight_boundary_color is not None:
                            highlight_col = highlight_boundary_color
                        else:
                            highlight_col = color_for_pixel
                            if highlight_brighten_factor != 1:
                                base_col = np.array([mcolors.to_rgb(color_for_pixel)])
                                highlight_col = brighten_colors(
                                    base_col, factor=highlight_brighten_factor
                                )[0]
                        _plot_boundaries_from_label_image(
                            ax,
                            highlight_label_img,
                            [x_min, x_max, y_min, y_max],
                            highlight_col,
                            highlight_linewidth,
                            alpha,
                            linestyle,
                            pixel_smoothing_sigma,
                            render=pixel_boundary_render,
                            thickness_px=pixel_boundary_thickness_px,
                            antialiased=pixel_boundary_antialiased,
                            include_background=pixel_boundary_include_background,
                        )

    ax.set_aspect("equal")
    if not show_axes:
        ax.axis("off")
    title_text = (
        f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
        if title is None
        else title
    )
    ax.set_title(title_text, fontsize=figsize[0] * 1.5)
    if show_colorbar and len(dimensions) == 1 and not use_categorical:
        cm = plt.get_cmap(cmap if cmap is not None else "Greys")
        if (color_norm_vmin is not None) and (color_norm_vmax is not None):
            vmin, vmax = float(color_norm_vmin), float(color_norm_vmax)
        else:
            vmin = raw_vals.min() if raw_vals is not None else grey_vals.min()
            vmax = raw_vals.max() if raw_vals is not None else grey_vals.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # Legend for segment pies when coloring by embeddings (not categories)
    if (
        segment_show_pie
        and show_legend
        and (segment_annotate_by is not None)
        and (not use_categorical)
        and segment_available
    ):
        try:
            comp_ser = coordinates_df["barcode"].map(adata.obs[segment_annotate_by])
            comp_ser = pd.Categorical(comp_ser)
            comp_cats = comp_ser.categories
            comp_n = len(comp_cats)

            try:
                if isinstance(palette, (list, tuple, np.ndarray)):
                    pal = list(palette)
                    if not pal:
                        raise ValueError("Empty palette list")
                    comp_lut = np.array([
                        mcolors.to_rgb(pal[i % len(pal)]) for i in range(comp_n)
                    ])
                elif isinstance(palette, str) and palette.startswith("sc.pl.palettes."):
                    try:
                        import scanpy as sc  # type: ignore
                        attr = palette.split(".")[-1]
                        pal_list = list(getattr(sc.pl.palettes, attr))
                        comp_lut = np.array([
                            mcolors.to_rgb(pal_list[i % len(pal_list)])
                            for i in range(comp_n)
                        ])
                    except Exception:
                        cmc = plt.get_cmap("tab20", comp_n)
                        comp_lut = np.array([cmc(i)[:3] for i in range(comp_n)])
                else:
                    cmc = plt.get_cmap(palette if isinstance(palette, str) else "tab20", comp_n)
                    comp_lut = np.array([cmc(i)[:3] for i in range(comp_n)])
            except Exception:
                cmc = plt.get_cmap("tab20", comp_n)
                comp_lut = np.array([cmc(i)[:3] for i in range(comp_n)])

            handles = [
                patches.Patch(color=comp_lut[i], label=str(cat))
                for i, cat in enumerate(comp_cats)
            ]

            if legend_outside:
                fig.subplots_adjust(right=0.82)
                loc_final = "center left" if legend_loc == "best" else legend_loc
                if legend_fontsize is None:
                    n_items = len(comp_cats)
                    rows = int(np.ceil(n_items / max(1, legend_ncol)))
                    size_factor = (figsize[0] * figsize[1]) ** 0.5
                    base = 12 + (size_factor - 9) * 0.5
                    if rows > 8:
                        base -= 2
                    elif rows > 5:
                        base -= 1
                    auto_fs = int(np.clip(round(base), 10, 28))
                else:
                    auto_fs = legend_fontsize
                fig.legend(
                    handles=handles,
                    loc=loc_final,
                    bbox_to_anchor=legend_bbox_to_anchor,
                    ncol=legend_ncol,
                    frameon=False,
                    fontsize=auto_fs,
                )
            else:
                if legend_fontsize is None:
                    n_items = len(comp_cats)
                    rows = int(np.ceil(n_items / max(1, legend_ncol)))
                    size_factor = (figsize[0] * figsize[1]) ** 0.5
                    base = 12 + (size_factor - 9) * 0.5
                    if rows > 8:
                        base -= 2
                    elif rows > 5:
                        base -= 1
                    auto_fs = int(np.clip(round(base), 10, 28))
                else:
                    auto_fs = legend_fontsize
                ax.legend(
                    handles=handles,
                    loc=legend_loc,
                    ncol=legend_ncol,
                    frameon=False,
                    fontsize=auto_fs,
                )
        except Exception as e:
            logger.warning(f"Failed to generate pie legend: {e}")

    # Non-pixel boundary methods
    linestyle_options = {"solid": "-", "dashed": "--", "dotted": ":"}
    linestyle = linestyle_options.get(boundary_style, "-")

    if segment_available and plot_boundaries and boundary_method != "pixel":
        tree = KDTree(coordinates[["x", "y"]].values)
        segments = coordinates_df["Segment"].unique()

        for seg in segments:
            is_highlight = seg in highlight_segments
            seg_tiles = coordinates_df[coordinates_df["Segment"] == seg]
            if len(seg_tiles) < 3:
                if verbose:
                    logger.info(
                        "Segment %s has fewer than 3 points. Skipping boundary plotting.",
                        str(seg),
                    )
                continue

            points = seg_tiles[["x", "y"]].values

            unique_points = np.unique(points, axis=0)
            if unique_points.shape[0] < 3 or is_collinear(unique_points):
                logger.warning(
                    f"Segment {seg} points are collinear or insufficient. Using LineString as boundary."
                )
                line = LineString(unique_points)
                clipped_line = line.intersection(plot_boundary_polygon)
                if clipped_line.is_empty:
                    continue

                line_points = []
                if isinstance(clipped_line, (LineString)):
                    x, y = clipped_line.xy
                    line_points = np.column_stack((x, y))
                elif isinstance(clipped_line, (MultiLineString, GeometryCollection)):
                    for geom in clipped_line.geoms:
                        if isinstance(geom, LineString):
                            x, y = geom.xy
                            line_points.extend(np.column_stack((x, y)))
                    line_points = np.array(line_points)
                else:
                    logger.warning(
                        f"Unsupported geometry type after clipping: {type(clipped_line)}"
                    )
                    continue

                _, idx = tree.query(line_points)

                if fixed_boundary_color is not None:
                    line_colors = np.array([fixed_boundary_color] * len(line_points))
                else:
                    if len(dimensions) == 3:
                        line_colors = coordinates.iloc[idx][["R", "G", "B"]].values
                    else:
                        grey = coordinates.iloc[idx]["Grey"].values
                        if cmap is None:
                            line_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        else:
                            cm = plt.get_cmap(cmap)
                            line_colors = cm(grey)[:, :3]
                    line_colors = brighten_colors(
                        np.clip(line_colors, 0, 1), factor=brighten_factor
                    )
                    if is_highlight:
                        if highlight_boundary_color is not None:
                            line_colors = np.array(
                                [mcolors.to_rgb(highlight_boundary_color)]
                                * len(line_points)
                            )
                        elif highlight_brighten_factor != 1:
                            line_colors = brighten_colors(
                                line_colors, factor=highlight_brighten_factor
                            )
                    elif highlight_segments:
                        if dim_to_grey and len(dimensions) == 3:
                            grey = np.dot(line_colors, [0.299, 0.587, 0.114])
                            line_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        line_colors *= dim_other_segments

                segments_lines = np.array(
                    [line_points[i : i + 2] for i in range(len(line_points) - 1)]
                )
                current_lw = highlight_linewidth if is_highlight else boundary_linewidth
                lc = LineCollection(
                    segments_lines,
                    colors=line_colors[:-1],
                    linewidth=current_lw,
                    alpha=alpha,
                    linestyle=linestyle,
                )
                ax.add_collection(lc)

                continue

            try:
                if boundary_method == "alphashape":
                    avg_dist = np.mean(
                        np.linalg.norm(points - points.mean(axis=0), axis=1)
                    )
                    alpha_value = (1.0 / avg_dist) * 1.5
                    concave_hull = alphashape.alphashape(points, alpha=alpha_value)
                    if concave_hull.is_empty or concave_hull.geom_type not in [
                        "Polygon",
                        "MultiPolygon",
                    ]:
                        logger.warning(
                            f"Alpha shape for Segment {seg} is empty or not Polygon/MultiPolygon. Using Convex Hull as fallback."
                        )
                        concave_hull = MultiPoint(points).convex_hull
                elif boundary_method == "convex_hull":
                    concave_hull = MultiPoint(points).convex_hull
                elif boundary_method == "oriented_bbox":
                    concave_hull = MultiPoint(points).minimum_rotated_rectangle
                elif boundary_method == "concave_hull":
                    avg_dist = np.mean(
                        np.linalg.norm(points - points.mean(axis=0), axis=1)
                    )
                    alpha_value = (1.0 / avg_dist) * 1.0
                    concave_hull = alphashape.alphashape(points, alpha=alpha_value)
                    if concave_hull.is_empty or concave_hull.geom_type not in [
                        "Polygon",
                        "MultiPolygon",
                    ]:
                        logger.warning(
                            f"Concave hull for Segment {seg} is empty or not Polygon/MultiPolygon. Using Convex Hull as fallback."
                        )
                        concave_hull = MultiPoint(points).convex_hull
                else:
                    raise ValueError(f"Unknown boundary method '{boundary_method}'")

                if not concave_hull.is_valid:
                    logger.warning(
                        f"Boundary for Segment {seg} is invalid: {explain_validity(concave_hull)}. Attempting to fix with buffer(0)."
                    )
                    concave_hull = concave_hull.buffer(0)
                    if not concave_hull.is_valid:
                        logger.error(
                            f"Unable to fix boundary for Segment {seg}. Skipping this segment."
                        )
                        continue

                concave_hull = smooth_polygon(concave_hull, buffer_dist=0.01)

            except Exception as e:
                logger.error(f"Failed to calculate boundary for Segment {seg}: {e}")
                continue

            try:
                clipped_polygon = concave_hull.intersection(plot_boundary_polygon)
                if clipped_polygon.is_empty:
                    logger.warning(f"Clipped boundary for Segment {seg} is empty.")
                    continue
            except Exception as e:
                logger.error(f"Failed to clip boundary for Segment {seg}: {e}")
                continue

            polygons = []
            if isinstance(clipped_polygon, (Polygon, LineString)):
                polygons = [clipped_polygon]
            elif isinstance(clipped_polygon, MultiPolygon):
                polygons = list(clipped_polygon.geoms)
            elif isinstance(clipped_polygon, GeometryCollection):
                for geom in clipped_polygon.geoms:
                    if isinstance(geom, (Polygon, LineString)):
                        polygons.append(geom)
            else:
                logger.warning(
                    f"Unsupported geometry type in clipped polygon for Segment {seg}: {type(clipped_polygon)}"
                )
                continue

            for poly in polygons:
                if isinstance(poly, Polygon):
                    x, y = poly.exterior.xy
                elif isinstance(poly, LineString):
                    x, y = poly.xy
                else:
                    logger.warning(
                        f"Unsupported geometry type in polygon for Segment {seg}: {type(poly)}"
                    )
                    continue

                boundary_points = np.column_stack((x, y))
                _, idx = tree.query(boundary_points)

                if fixed_boundary_color is not None:
                    boundary_colors = np.array(
                        [fixed_boundary_color] * len(boundary_points)
                    )
                else:
                    if len(dimensions) == 3:
                        boundary_colors = coordinates.iloc[idx][["R", "G", "B"]].values
                    else:
                        grey = coordinates.iloc[idx]["Grey"].values
                        if cmap is None:
                            boundary_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        else:
                            cm = plt.get_cmap(cmap)
                            boundary_colors = cm(grey)[:, :3]

                    boundary_colors = brighten_colors(
                        np.clip(boundary_colors, 0, 1), factor=brighten_factor
                    )
                    if is_highlight:
                        if highlight_boundary_color is not None:
                            boundary_colors = np.array(
                                [mcolors.to_rgb(highlight_boundary_color)]
                                * len(boundary_points)
                            )
                        elif highlight_brighten_factor != 1:
                            boundary_colors = brighten_colors(
                                boundary_colors, factor=highlight_brighten_factor
                            )
                    elif highlight_segments:
                        if dim_to_grey and len(dimensions) == 3:
                            grey = np.dot(boundary_colors, [0.299, 0.587, 0.114])
                            boundary_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        boundary_colors *= dim_other_segments

                segments_lines = np.array(
                    [
                        boundary_points[i : i + 2]
                        for i in range(len(boundary_points) - 1)
                    ]
                )

                current_lw = highlight_linewidth if is_highlight else boundary_linewidth
                lc = LineCollection(
                    segments_lines,
                    colors=boundary_colors[:-1],
                    linewidth=current_lw,
                    alpha=alpha,
                    linestyle=linestyle,
                )
                ax.add_collection(lc)

                if fill_boundaries and isinstance(poly, Polygon):
                    patch = patches.Polygon(
                        np.array(poly.exterior.coords),
                        closed=True,
                        facecolor=fill_color,
                        edgecolor=None,
                        alpha=fill_alpha,
                    )
                    ax.add_patch(patch)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    _apply_axis_crop(ax, xlim=xlim, ylim=ylim)
    plt.show()
    # Do not return plot object to avoid auto display


def image_plot_with_spatial_image(
    adata,
    dimensions=[0, 1, 2],
    embedding="X_embedding_segment",
    figsize=None,
    point_size=None,
    fig_dpi=None,
    scaling_factor=2,
    img_scaling_factor=None,
    imshow_tile_size=None,
    imshow_scale_factor=1.0,
    imshow_tile_size_mode="auto",
    imshow_tile_size_quantile=None,
    imshow_tile_size_rounding="floor",
    imshow_tile_size_shrink=0.98,
    pixel_perfect: bool = False,
    origin=True,
    plot_boundaries=False,
    boundary_method="pixel",
    alpha=1.0,            # boundary alpha (not per-spot)
    alpha_point=1.0,      # default per-spot alpha when alpha_by is None (scatter/imshow path)
    boundary_color="black",
    boundary_linewidth=1.0,
    alpha_shape_alpha=0.1,
    aspect_ratio_threshold=3.0,
    img=None,
    img_key=None,
    library_id=None,
    crop=True,
    xlim=None,
    ylim=None,
    display_orientation: str = "cartesian",  # 'cartesian' (default) or 'image'
    alpha_img=1.0,
    bw=False,
    jitter=1e-6,
    tol_collinear=1e-8,
    tol_almost_collinear=1e-5,
    boundary_style="solid",
    fill_boundaries=False,
    fill_color="blue",
    fill_alpha=0.1,
    brighten_factor=1.3,
    fixed_boundary_color=None,
    highlight_segments=None,
    highlight_linewidth=None,
    highlight_boundary_color=None,
    highlight_brighten_factor=1.5,
    dim_other_segments=0.3,
    dim_to_grey=True,
    cmap=None,
    # Normalization controls for consistent colors across crops
    rebalance_method: str = "minmax",  # 'minmax' (default) or 'clip'
    rebalance_vmin=None,              # scalar (1D) or length-3 (3D) for fixed scaling
    rebalance_vmax=None,              # scalar (1D) or length-3 (3D) for fixed scaling
    color_vmin=None,                  # fixed scaling for numeric color_by
    color_vmax=None,                  # fixed scaling for numeric color_by
    color_percentiles=(1, 99),        # used when color_vmin/vmax are None
    pixel_smoothing_sigma=0,
    # Pixel-boundary rendering controls (boundary_method='pixel' only)
    pixel_boundary_render: str = "contour",  # 'contour'|'raster'
    pixel_boundary_thickness_px: int = 1,    # used when pixel_boundary_render='raster'
    pixel_boundary_antialiased: bool = True, # used when pixel_boundary_render='contour'
    pixel_boundary_include_background: str | bool = "outline",
    use_imshow=True,
    pixel_shape="square",
    chunk_size=50000,
    prioritize_high_values=False,
    verbose=False,
    title=None,
    show_colorbar=False,
    # Categorical coloring from adata.obs
    color_by=None,
    palette="tab20",
    show_legend=False,
    legend_loc="best",
    legend_ncol=2,
    legend_outside=True,
    legend_bbox_to_anchor=(0.83, 0.5),
    legend_fontsize=None,
    # Overlap draw priority for categories
    overlap_priority='few_front',
    # Segment pies and major coloring
    segment_annotate_by=None,
    segment_annotate_mode="pie",
    segment_min_tiles=10,
    segment_top_n=3,
    segment_pie_scale=3.0,
    segment_label_fmt="{label} ({pct:.0f}%)",  # deprecated
    segment_label_color="white",               # deprecated
    segment_label_bbox=True,                   # deprecated
    segment_show_pie=False,
    segment_pie_edgecolor="black",
    segment_pie_linewidth=0.8,
    segment_color_by_major=False,
    segment_key="Segment",
    # Per-spot alpha controls
    alpha_by=None,               # None|'embedding'|<obs column name>
    alpha_range=(0.1, 1.0),
    alpha_clip=None,
    alpha_invert=False,
    # Brightness / mixing controls for continuous (3D) embeddings
    brighten_continuous=False,   # If True, brighten 3D continuous colors
    continuous_gamma=0.8,        # Gamma < 1 brightens mid-tones
    # Visium/VisiumHD grid rendering
    coordinate_mode: str = "spatial",  # 'spatial'|'array'|'visium'|'visiumhd'|'auto'
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    # Raster-expanded rendering controls (origin=False)
    gapless: bool = False,  # If True, render raster-expanded tiles as a packed grid (no holes, faster)
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool = True,
    gap_collapse_axis: str | None = None,
    gap_collapse_max_run_px: int = 1,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool | str = "auto",
    center_collision_radius: int = 2,
    cache_image: bool = False,
    image_cache_key: str | None = None,
    refresh_image_cache: bool = False,
    show_axes: bool = False,
):
    """
    Visualize embeddings as an image with optional segment boundaries and background image overlay.

    Supports per-spot alpha via `alpha_by='embedding'` (1D) or numeric obs column.
    """
    # Create a local logger
    logger = logging.getLogger("image_plot_with_spatial_image")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False

    # Avoid duplicate handlers if the function is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    t_all = perf_counter()

    if len(dimensions) not in [1, 3]:
        raise ValueError("Only 1 or 3 dimensions can be used for visualization.")

    plot_cache = _prepare_plot_image_cache(
        adata,
        cache_image=cache_image,
        image_cache_key=image_cache_key,
        refresh_image_cache=refresh_image_cache,
        verbose=bool(verbose),
        embedding=embedding,
        dimensions=dimensions,
        origin=origin,
        coordinate_mode=coordinate_mode,
        array_row_key=array_row_key,
        array_col_key=array_col_key,
        color_by=color_by,
        palette=palette,
        alpha_by=alpha_by,
        alpha_range=alpha_range,
        alpha_clip=alpha_clip,
        alpha_invert=alpha_invert,
        prioritize_high_values=prioritize_high_values,
        overlap_priority=overlap_priority,
        segment_color_by_major=segment_color_by_major,
        title=title,
        use_imshow=use_imshow,
        brighten_continuous=brighten_continuous,
        continuous_gamma=continuous_gamma,
        plot_boundaries=plot_boundaries,
        boundary_method=boundary_method,
        boundary_color=boundary_color,
        boundary_linewidth=boundary_linewidth,
        boundary_style=boundary_style,
        boundary_alpha=alpha,
        pixel_smoothing_sigma=pixel_smoothing_sigma,
        highlight_segments=highlight_segments,
        highlight_boundary_color=highlight_boundary_color,
        highlight_linewidth=highlight_linewidth,
        highlight_brighten_factor=highlight_brighten_factor,
        dim_other_segments=dim_other_segments,
        dim_to_grey=dim_to_grey,
        segment_key=segment_key,
        imshow_tile_size=imshow_tile_size,
        imshow_scale_factor=imshow_scale_factor,
        imshow_tile_size_mode=imshow_tile_size_mode,
        imshow_tile_size_quantile=imshow_tile_size_quantile,
        imshow_tile_size_rounding=imshow_tile_size_rounding,
        imshow_tile_size_shrink=imshow_tile_size_shrink,
        pixel_perfect=pixel_perfect,
        pixel_shape=pixel_shape,
        chunk_size=chunk_size,
        runtime_fill_from_boundary=runtime_fill_from_boundary,
        runtime_fill_closing_radius=runtime_fill_closing_radius,
        runtime_fill_external_radius=runtime_fill_external_radius,
        runtime_fill_holes=runtime_fill_holes,
        gap_collapse_axis=gap_collapse_axis,
        gap_collapse_max_run_px=gap_collapse_max_run_px,
        soft_rasterization=soft_rasterization,
        resolve_center_collisions=resolve_center_collisions,
        center_collision_radius=center_collision_radius,
    )

    # Extract embeddings
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Extract image data if available
    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)

    # Retrieve tensor_resolution from adata.uns
    tensor_resolution = adata.uns.get("tensor_resolution", 1.0)

    # Adjust img_scaling_factor based on tensor_resolution
    if img_scaling_factor is None:
        img_scaling_factor = _check_scale_factor(
            spatial_data, img_key, img_scaling_factor
        )
    img_scaling_factor_adjusted = (
        img_scaling_factor / tensor_resolution if img_scaling_factor else 1.0
    )
    img_origin_x = 0.0
    img_origin_y = 0.0
    if spatial_data is not None and img_key is not None:
        try:
            image_origins_um = spatial_data.get("metadata", {}).get("image_origins_um", {})
            img_origin_um = image_origins_um.get(img_key)
            if img_origin_um is not None and len(img_origin_um) >= 2:
                img_origin_x = float(img_origin_um[0]) * float(img_scaling_factor_adjusted)
                img_origin_y = float(img_origin_um[1]) * float(img_scaling_factor_adjusted)
        except Exception:
            img_origin_x = 0.0
            img_origin_y = 0.0

    if _can_render_plot_cache_fast(
        cache=plot_cache,
        requested_dimensions=dimensions,
        show_colorbar=bool(show_colorbar),
        show_legend=bool(show_legend),
        segment_show_pie=bool(segment_show_pie),
        fill_boundaries=bool(fill_boundaries),
        plot_boundaries=bool(plot_boundaries),
        boundary_method=str(boundary_method),
        use_imshow=bool(use_imshow),
    ):
        title_text = (
            f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
            if title is None
            else title
        )
        background_extent = None
        background_origin = None
        orientation = str(display_orientation or "image").lower()
        if orientation not in {"image", "cartesian"}:
            orientation = "image"
        if img is not None:
            y_max_img, x_max_img = img.shape[:2]
            if orientation == "cartesian":
                background_extent = [0, x_max_img, 0, y_max_img]
                background_origin = "lower"
            else:
                background_extent = [0, x_max_img, y_max_img, 0]
                background_origin = None
        if verbose:
            logger.info("image_plot_with_spatial_image: rendering directly from cached image.")
        _render_plot_cache_fast(
            cache=plot_cache,
            requested_dimensions=dimensions,
            title=title_text,
            figsize=figsize,
            fig_dpi=fig_dpi,
            xlim=xlim,
            ylim=ylim,
            show_axes=bool(show_axes),
            background_img=img,
            background_extent=background_extent,
            background_origin=background_origin,
            background_alpha=alpha_img,
        )
        return

    # Extract spatial coordinates
    use_direct_origin_spatial = False
    tiles = None
    origin_vals = None
    if origin:
        spatial_direct = None
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
            tile_barcodes = _string_array_no_copy(adata.obs.index)
            x_vals = np.asarray(spatial_direct[:, 0], dtype=float)
            y_vals = np.asarray(spatial_direct[:, 1], dtype=float)
            if verbose:
                logger.info("Using adata.obsm['spatial'] directly for origin=True plotting.")
        else:
            tiles_all = adata.uns["tiles"]
            tile_cols = [c for c in ("x", "y", "barcode", "origin") if c in tiles_all.columns]
            if "origin" in tiles_all.columns:
                origin_mask = pd.to_numeric(tiles_all["origin"], errors="coerce").fillna(0).to_numpy() == 1
                tiles = tiles_all.loc[origin_mask, tile_cols]
            else:
                tiles = tiles_all.loc[:, tile_cols]
    else:
        tiles_all = adata.uns["tiles"]
        tile_cols = [c for c in ("x", "y", "barcode", "origin") if c in tiles_all.columns]
        tiles = tiles_all.loc[:, tile_cols]
        inferred_info = adata.uns.get("tiles_inferred_grid_info", {})
        inferred_mode = str(inferred_info.get("mode", "")).lower()
        if inferred_mode == "boundary_voronoi" and ("origin" in tiles.columns):
            support_mask = _float_array_fast(tiles["origin"]) == 0
            if bool(np.any(support_mask)):
                tiles = tiles.loc[support_mask, tile_cols]
                if verbose:
                    logger.info("Using inferred boundary support-grid tiles only for origin=False plotting.")

    if not use_direct_origin_spatial:
        if "barcode" not in tiles.columns:
            raise ValueError("adata.uns['tiles'] must contain a 'barcode' column.")
        if "x" not in tiles.columns or "y" not in tiles.columns:
            raise ValueError("adata.uns['tiles'] must contain 'x' and 'y' columns.")
        tile_barcodes = _string_array_no_copy(tiles["barcode"])
        x_vals = _float_array_fast(tiles["x"])
        y_vals = _float_array_fast(tiles["y"])
        if "origin" in tiles.columns:
            origin_vals = _float_array_fast(tiles["origin"])

    # Prepare tile colors and align to tile rows (avoid large pandas merge for origin=False).
    t0 = perf_counter()
    obs_index = _string_index_no_copy(adata.obs.index)

    grouped_barcode_count = None
    if use_direct_origin_spatial:
        pos = np.arange(int(adata.n_obs), dtype=np.int64)
    else:
        if origin:
            barcode_index = pd.Index(tile_barcodes)
            if barcode_index.has_duplicates:
                if verbose:
                    logger.warning(
                        "adata.uns['tiles'] contains duplicated barcodes; keeping first occurrence for plotting."
                    )
                keep_mask = ~barcode_index.duplicated(keep="first")
                tiles = tiles.loc[keep_mask]
                tile_barcodes = tile_barcodes[keep_mask]
                x_vals = x_vals[keep_mask]
                y_vals = y_vals[keep_mask]
                if origin_vals is not None:
                    origin_vals = origin_vals[keep_mask]
            pos = obs_index.get_indexer(tile_barcodes)
        else:
            has_non_origin = False
            if origin_vals is not None:
                try:
                    has_non_origin = np.isfinite(origin_vals).any() and (origin_vals != 1).any()
                except Exception:
                    has_non_origin = False
            if verbose and has_non_origin:
                logger.info(
                    "adata.uns['tiles'] contains duplicated barcodes (expected for origin=False raster tiles); keeping duplicates."
                )
            if has_non_origin:
                pos, grouped_barcode_count = _grouped_obs_indexer(tile_barcodes, obs_index)
            else:
                pos = obs_index.get_indexer(tile_barcodes)

    valid = (pos >= 0) & np.isfinite(x_vals) & np.isfinite(y_vals)
    if not np.any(valid):
        raise ValueError("No tile barcodes match adata.obs index.")
    embedding_full = np.asarray(adata.obsm[embedding])
    embedding_dims = embedding_full[:, dimensions]
    dim_names = ["dim0", "dim1", "dim2"] if len(dimensions) == 3 else ["dim0"]
    coord_dict = {
        "x": x_vals[valid],
        "y": y_vals[valid],
        "barcode": tile_barcodes[valid],
    }
    if origin_vals is not None:
        coord_dict["origin"] = origin_vals[valid]
    elif use_direct_origin_spatial:
        coord_dict["origin"] = np.ones(int(np.count_nonzero(valid)), dtype=np.float32)
    coordinates_df = pd.DataFrame(coord_dict)
    coordinates_df.loc[:, dim_names] = embedding_dims[pos[valid]]
    coordinates_df = coordinates_df.reset_index(drop=True)
    if verbose:
        logger.info(
            "Prepared coordinates_df rows=%d in %.3fs",
            int(coordinates_df.shape[0]),
            perf_counter() - t0,
        )

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
        coordinates_df[array_col_key] = coordinates_df["barcode"].map(adata.obs[array_col_key])
        coordinates_df[array_row_key] = coordinates_df["barcode"].map(adata.obs[array_row_key])
        col = pd.to_numeric(coordinates_df[array_col_key], errors="coerce")
        row = pd.to_numeric(coordinates_df[array_row_key], errors="coerce")
        if coord_mode == "visium":
            row_i = row.astype("Int64")
            parity = (row_i % 2).astype(float)
            coordinates_df["x"] = col.astype(float) + 0.5 * parity
            coordinates_df["y"] = row.astype(float) * (np.sqrt(3.0) / 2.0)
        else:
            coordinates_df["x"] = col.astype(float)
            coordinates_df["y"] = row.astype(float)
        coordinates_df = coordinates_df.dropna(subset=["x", "y"])

    seg_col = None
    has_segments = False
    need_segments = bool(
        plot_boundaries or segment_show_pie or segment_color_by_major or bool(highlight_segments)
    )
    if need_segments and segment_key is not None:
        candidate = segment_key
        if candidate in adata.obs.columns:
            seg_col = candidate
            has_segments = True
        elif candidate != "Segment" and "Segment" in adata.obs.columns:
            logger.warning(
                f"Segment key '{candidate}' not found; falling back to 'Segment' column."
            )
            seg_col = "Segment"
            has_segments = True
        else:
            logger.warning(
                f"Segment key '{candidate}' not found in adata.obs; disabling segment-specific annotations."
            )
    if has_segments and seg_col is not None:
        coordinates_df[seg_col] = coordinates_df["barcode"].map(adata.obs[seg_col])
        if coordinates_df[seg_col].isnull().any():
            missing = coordinates_df[seg_col].isnull().sum()
            logger.warning(
                f"'{seg_col}' label for {missing} barcodes is missing. These rows will be dropped."
            )
            coordinates_df = coordinates_df.dropna(subset=[seg_col])
        coordinates_df["Segment"] = coordinates_df[seg_col]
    else:
        coordinates_df = coordinates_df.drop(columns=["Segment"], errors="ignore")
        has_segments = False
        seg_col = None

    segment_available = has_segments and "Segment" in coordinates_df.columns

    coordinates_df["x"] = pd.to_numeric(coordinates_df["x"], errors="coerce")
    coordinates_df["y"] = pd.to_numeric(coordinates_df["y"], errors="coerce")

    coord_xy = coordinates_df[["x", "y"]].to_numpy(dtype=float, copy=False)
    finite_coord_mask = np.isfinite(coord_xy).all(axis=1)
    if not np.all(finite_coord_mask):
        dropped = int(finite_coord_mask.size - np.count_nonzero(finite_coord_mask))
        logger.warning(
            "Found %d non-finite values in 'x' or 'y' after conversion. Dropping these rows.",
            dropped,
        )
        coordinates_df = coordinates_df.loc[finite_coord_mask].reset_index(drop=True)

    # Apply adjusted scaling factor to coordinates
    coordinates_df["x"] *= img_scaling_factor_adjusted
    coordinates_df["y"] *= img_scaling_factor_adjusted

    # Depth metric default
    if len(dimensions) == 1:
        vals = coordinates_df[dim_names[0]].to_numpy()
    else:
        vals = coordinates_df[dim_names].sum(axis=1).to_numpy()

    # Capture original range for colorbar before normalization
    if len(dimensions) == 1:
        raw_vals = coordinates_df[dim_names[0]].values
    else:
        raw_vals = None
    color_norm_vmin = None
    color_norm_vmax = None

    # --- Per-spot alpha mapping (optional) ---
    alpha_arr = None
    if alpha_by is not None:
        if alpha_by == "embedding":
            if len(dimensions) != 1:
                raise ValueError("alpha_by='embedding' requires len(dimensions) == 1.")
            alpha_source = coordinates_df[dim_names[0]].values
        else:
            if alpha_by not in adata.obs.columns:
                raise ValueError(f"alpha_by '{alpha_by}' not found in adata.obs.")
            alpha_source = coordinates_df["barcode"].map(adata.obs[alpha_by]).astype(float).values

        alpha_arr, _ = _compute_alpha_from_values(
            alpha_source, alpha_range=alpha_range, clip=alpha_clip, invert=alpha_invert
        )

    # NEW: Use per-spot alpha as the depth metric if present
    if alpha_arr is not None:
        vals = alpha_arr.astype(float)

    use_categorical = False
    category_lut = None
    categories = None
    cat_priority_vals = None
    if color_by is not None:
        if color_by not in adata.obs.columns:
            raise ValueError(f"'{color_by}' not found in adata.obs. Available: {list(adata.obs.columns)}")
        coordinates_df[color_by] = coordinates_df["barcode"].map(adata.obs[color_by])
        if coordinates_df[color_by].isnull().any():
            coordinates_df = coordinates_df.dropna(subset=[color_by])
        if is_numeric_dtype(coordinates_df[color_by]):
            v = pd.to_numeric(coordinates_df[color_by], errors="coerce").to_numpy(dtype=float)
            raw_vals = v
            finite = np.isfinite(v)
            if finite.any():
                if (color_vmin is not None) or (color_vmax is not None):
                    vmin = float(np.nanmin(v[finite]) if color_vmin is None else color_vmin)
                    vmax = float(np.nanmax(v[finite]) if color_vmax is None else color_vmax)
                else:
                    try:
                        p_lo, p_hi = color_percentiles
                    except Exception:
                        p_lo, p_hi = 1, 99
                    vmin = float(np.nanpercentile(v[finite], p_lo))
                    vmax = float(np.nanpercentile(v[finite], p_hi))
            else:
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-9
            color_norm_vmin, color_norm_vmax = vmin, vmax
            t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
            cm = plt.get_cmap(cmap or "viridis")
            cols = cm(t)[:, :3]
            coordinates = coordinates_df.copy()
            coordinates["R"], coordinates["G"], coordinates["B"] = cols[:, 0], cols[:, 1], cols[:, 2]
            coordinates["Grey"] = np.dot(cols, [0.299, 0.587, 0.114])
            vals = v.astype(float)
            prioritize_high_values = True
            cat_priority_vals = None
            use_categorical = False
            show_legend = False
        else:
            cats = pd.Categorical(coordinates_df[color_by])
            categories = cats.categories
            n_cat = len(categories)
            try:
                if isinstance(palette, (list, tuple, np.ndarray)):
                    pal = list(palette)
                    if not pal:
                        raise ValueError("Empty palette list")
                    category_lut = np.array([mcolors.to_rgb(pal[i % len(pal)]) for i in range(n_cat)])
                elif isinstance(palette, str) and palette.startswith("sc.pl.palettes."):
                    try:
                        import scanpy as sc

                        attr = palette.split(".")[-1]
                        pal_list = list(getattr(sc.pl.palettes, attr))
                        category_lut = np.array([mcolors.to_rgb(pal_list[i % len(pal_list)]) for i in range(n_cat)])
                    except Exception as e:
                        logger.warning(f"Failed to load Scanpy palette '{palette}': {e}; fallback to Matplotlib")
                if category_lut is None:
                    cmap_cat = plt.get_cmap(palette, n_cat) if isinstance(palette, str) else palette
                    category_lut = np.array([cmap_cat(i)[:3] for i in range(n_cat)])
            except Exception:
                cmap_cat = plt.get_cmap("tab20", n_cat)
                category_lut = np.array([cmap_cat(i)[:3] for i in range(n_cat)])

            codes = cats.codes
            cols = category_lut[codes]
            # Segment-major unification
            if segment_color_by_major and segment_available:
                try:
                    seg_major = coordinates_df.groupby("Segment")[color_by].agg(lambda s: s.value_counts().idxmax())
                    major_labels = coordinates_df["Segment"].map(seg_major)
                    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
                    major_idx = major_labels.map(cat_to_idx).fillna(0).astype(int)
                    cols = category_lut[major_idx.values]
                except Exception as e:
                    logger.warning(f"Failed to apply segment_color_by_major: {e}")

            # Build coordinates DF for later boundary coloring
            coordinates = coordinates_df.copy()
            coordinates["R"], coordinates["G"], coordinates["B"] = cols[:, 0], cols[:, 1], cols[:, 2]
            coordinates["Grey"] = np.dot(cols, [0.299, 0.587, 0.114])

            # Overlap priority
            do_cat_priority = overlap_priority in {"few_front", "many_front"}
            if do_cat_priority:
                cat_counts = pd.Series(cats).value_counts().reindex(categories, fill_value=0).values
                row_counts = cat_counts[codes]
                if overlap_priority == "few_front":
                    max_c = row_counts.max() if len(row_counts) else 0
                    vals = (max_c - row_counts).astype(float)
                else:
                    vals = row_counts.astype(float)
                prioritize_high_values = True
                cat_priority_vals = vals.copy()
            else:
                # CHANGED: if alpha_arr exists, keep it as depth; else neutral/category code
                if alpha_arr is not None:
                    vals = alpha_arr.astype(float)
                else:
                    vals = np.zeros_like(codes, dtype=float) if not prioritize_high_values else codes.astype(float)
                cat_priority_vals = None

            use_categorical = True
    else:
        coordinates = rebalance_colors(
            coordinates_df, dimensions, method=rebalance_method, vmin=rebalance_vmin, vmax=rebalance_vmax
        )
        if len(dimensions) == 1:
            if (rebalance_vmin is not None) or (rebalance_vmax is not None):
                vmin0 = raw_vals.min() if raw_vals is not None else 0.0
                vmax0 = raw_vals.max() if raw_vals is not None else 1.0
                try:
                    vmin0 = float(np.asarray(rebalance_vmin).reshape(-1)[0]) if rebalance_vmin is not None else float(vmin0)
                    vmax0 = float(np.asarray(rebalance_vmax).reshape(-1)[0]) if rebalance_vmax is not None else float(vmax0)
                except Exception:
                    pass
                color_norm_vmin, color_norm_vmax = vmin0, vmax0
            elif raw_vals is not None and len(raw_vals) > 0:
                color_norm_vmin, color_norm_vmax = float(np.nanmin(raw_vals)), float(np.nanmax(raw_vals))
        if len(dimensions) == 3:
            cols = coordinates[["R", "G", "B"]].values

            # Match image_plot() behavior for continuous 3D embeddings.
            if brighten_continuous:
                max_per_pixel = cols.max(axis=1, keepdims=True)
                scale = np.where(max_per_pixel > 0, 1.0 / max_per_pixel, 0.0)
                cols = np.clip(cols * scale, 0.0, 1.0)
                try:
                    gamma = float(continuous_gamma)
                    if gamma > 0 and gamma != 1.0:
                        cols = np.clip(np.power(cols, gamma), 0.0, 1.0)
                except Exception:
                    pass
        else:
            grey_vals = coordinates["Grey"].values
            if cmap is None:
                cols = np.repeat(grey_vals[:, np.newaxis], 3, axis=1)
            else:
                cm = plt.get_cmap(cmap)
                cols = cm(grey_vals)[:, :3]

    x_min, x_max = coordinates["x"].min(), coordinates["x"].max()
    y_min, y_max = coordinates["y"].min(), coordinates["y"].max()
    plot_boundary_polygon = Polygon(
        [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
    )

    user_figsize_given = figsize is not None
    user_dpi_given = fig_dpi is not None
    auto_sized = (not user_figsize_given) or (not user_dpi_given)
    pts_for_imshow = None
    s_data_units_for_imshow = None
    raster_canvas_for_imshow = None

    if use_imshow:
        # Use the full raster tile count for origin=False auto canvas sizing.
        n_points_eff = int(coordinates.shape[0])

        pts = coordinates[["x", "y"]].values
        pts_for_imshow = pts
        imshow_tile_size_eff = imshow_tile_size
        if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
            imshow_tile_size_eff = 1.0
        raster_canvas_for_imshow = _raster.resolve_raster_canvas(
            pts=pts,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            figsize=figsize,
            fig_dpi=fig_dpi,
            imshow_tile_size=imshow_tile_size_eff,
            imshow_scale_factor=imshow_scale_factor,
            imshow_tile_size_mode=imshow_tile_size_mode,
            imshow_tile_size_quantile=imshow_tile_size_quantile,
            imshow_tile_size_rounding=imshow_tile_size_rounding,
            imshow_tile_size_shrink=imshow_tile_size_shrink,
            pixel_perfect=bool(pixel_perfect),
            n_points=int(n_points_eff),
            logger=logger,
            context="image_plot_with_spatial_image(auto-size)",
            persistent_store=getattr(adata, "uns", None),
        )
        s_data_units_for_imshow = float(raster_canvas_for_imshow["s_data_units"])
        figsize = raster_canvas_for_imshow["figsize"]
        fig_dpi = raster_canvas_for_imshow["fig_dpi"]

    if figsize is None:
        figsize = (10, 10)

    if fig_dpi is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)
    if verbose:
        logger.info(
            "Auto-size uses n_eff=%d (rows=%d); final figsize=%s, fig_dpi=%s",
            int(n_points_eff) if "n_points_eff" in locals() else int(coordinates.shape[0]),
            int(coordinates.shape[0]),
            tuple(figsize),
            fig.get_dpi(),
        )
    linestyle_options = {"solid": "-", "dashed": "--", "dotted": ":"}
    linestyle = linestyle_options.get(boundary_style, "-")
    color_for_pixel = (
        fixed_boundary_color if fixed_boundary_color is not None else boundary_color
    )
    if segment_available:
        # Normalize highlight_segments input
        if highlight_segments is not None:
            if isinstance(highlight_segments, str):
                highlight_segments = {highlight_segments}
            else:
                highlight_segments = set(highlight_segments)
        else:
            highlight_segments = set()

        highlight_mask = coordinates_df["Segment"].isin(highlight_segments)
        cols = cols.copy()
        if highlight_segments:
            if dim_to_grey and len(dimensions) == 3:
                grey_values = np.dot(cols[~highlight_mask], [0.299, 0.587, 0.114])
                grey_colors = np.repeat(grey_values[:, np.newaxis], 3, axis=1)
                cols[~highlight_mask] = grey_colors * dim_other_segments
            else:
                cols[~highlight_mask] = cols[~highlight_mask] * dim_other_segments
            if highlight_brighten_factor != 1:
                cols[highlight_mask] = brighten_colors(
                    cols[highlight_mask], factor=highlight_brighten_factor
                )

        if highlight_linewidth is None:
            highlight_linewidth = boundary_linewidth * 2

    stored_pixel_boundary = None
    if segment_available and plot_boundaries and boundary_method == "pixel":
        seg_uns_key = f"{_sanitize_segment_key_for_uns(seg_col or segment_key)}_pixel_boundary"
        candidate_payload = adata.uns.get(seg_uns_key, None)
        if isinstance(candidate_payload, dict):
            try:
                if bool(candidate_payload.get("origin", origin)) == bool(origin):
                    stored_pixel_boundary = candidate_payload
                    if verbose:
                        logger.info(
                            "image_plot_with_spatial_image: using stored pixel boundary payload from adata.uns['%s'].",
                            seg_uns_key,
                        )
            except Exception:
                stored_pixel_boundary = None

    orientation = str(display_orientation or "image").lower()
    if orientation not in {"image", "cartesian"}:
        orientation = "image"

    if img is not None:
        y_max_img, x_max_img = img.shape[:2]
        img_x_min = float(img_origin_x)
        img_x_max = float(img_origin_x) + float(x_max_img)
        img_y_min = float(img_origin_y)
        img_y_max = float(img_origin_y) + float(y_max_img)
        if orientation == "cartesian":
            extent = [img_x_min, img_x_max, img_y_min, img_y_max]
            ax.imshow(img, extent=extent, origin="lower", alpha=alpha_img)
        else:
            extent = [img_x_min, img_x_max, img_y_max, img_y_min]  # Left, Right, Bottom, Top
            ax.imshow(img, extent=extent, alpha=alpha_img)

        if crop:
            x_range_spots = x_max - x_min
            y_range_spots = y_max - y_min
            crop_padding_factor = 0.05
            x_padding = x_range_spots * crop_padding_factor
            y_padding = y_range_spots * crop_padding_factor
            view_x_min = x_min - x_padding
            view_x_max = x_max + x_padding
            view_y_min_data = y_min - y_padding
            view_y_max_data = y_max + y_padding
            final_x_min = max(img_x_min, view_x_min)
            final_x_max = min(img_x_max, view_x_max)
            final_y_bottom_limit = min(img_y_max, view_y_max_data)
            final_y_top_limit = max(img_y_min, view_y_min_data)
            ax.set_xlim(final_x_min, final_x_max)
            if orientation == "cartesian":
                ax.set_ylim(final_y_top_limit, final_y_bottom_limit)
            else:
                ax.set_ylim(final_y_bottom_limit, final_y_top_limit)
        else:
            ax.set_xlim(img_x_min, img_x_max)
            if orientation == "cartesian":
                ax.set_ylim(img_y_min, img_y_max)
            else:
                ax.set_ylim(img_y_max, img_y_min)
    else:
        ax.set_xlim(x_min, x_max)
        if orientation == "cartesian":
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(y_max, y_min)

    _apply_axis_crop(ax, xlim=xlim, ylim=ylim)

    if use_imshow:
        # Optional gapless packed-grid rendering for raster-expanded tiles (origin=False).
        if gapless and (not origin) and (not plot_boundaries) and ("origin" in coordinates.columns):
            try:
                origin_vals_eff = pd.to_numeric(coordinates["origin"], errors="coerce").fillna(1)
                has_non_origin = (origin_vals_eff != 1).any()
            except Exception:
                has_non_origin = False
            if has_non_origin and ("x" in coordinates.columns) and ("y" in coordinates.columns):
                x = coordinates["x"].to_numpy(dtype=float, copy=False)
                y = coordinates["y"].to_numpy(dtype=float, copy=False)
                step_x = _infer_regular_grid_step(x)
                step_y = _infer_regular_grid_step(y)
                step = None
                if step_x is not None and step_y is not None:
                    step = int(max(1, min(int(step_x), int(step_y))))
                elif step_x is not None:
                    step = int(max(1, int(step_x)))
                elif step_y is not None:
                    step = int(max(1, int(step_y)))

                if step is not None:
                    t_grid = perf_counter()
                    xi0, yi0, w0, h0, x0_int, y0_int = _coords_to_grid_indices(x, y, step=step)

                    max_raster_px = _raster.resolve_max_raster_pixels(
                        env_key="SPIX_PLOT_MAX_RASTER_PIXELS",
                        n_points=len(x),
                    )
                    total0 = int(w0) * int(h0)
                    ds = int(max(1, int(np.ceil(np.sqrt(float(total0) / float(max_raster_px))))))
                    xi = (xi0 // ds).astype(np.int64, copy=False)
                    yi = (yi0 // ds).astype(np.int64, copy=False)
                    w = int(xi.max(initial=0)) + 1
                    h = int(yi.max(initial=0)) + 1

                    img_rgba = np.zeros((h, w, 4), dtype=np.float32)
                    img_rgba[yi, xi, :3] = cols
                    if alpha_arr is not None:
                        img_rgba[yi, xi, 3] = np.asarray(alpha_arr, dtype=np.float32)
                    else:
                        img_rgba[yi, xi, 3] = 1.0

                    step_eff = float(step) * float(ds)
                    x_min_eff = float(x0_int) - 0.5 * step_eff
                    x_max_eff = float(x0_int) + (float(w) - 0.5) * step_eff
                    y_min_eff = float(y0_int) - 0.5 * step_eff
                    y_max_eff = float(y0_int) + (float(h) - 0.5) * step_eff

                    if verbose:
                        logger.info(
                            "gapless grid render: step=%d, ds=%d, buffer=(%d,%d) from (%d,%d) in %.3fs",
                            int(step),
                            int(ds),
                            int(w),
                            int(h),
                            int(w0),
                            int(h0),
                            perf_counter() - t_grid,
                        )
                    ax.imshow(
                        img_rgba,
                        origin="lower",
                        extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
                        interpolation="nearest",
                    )
                    ax.set_xlim(x_min_eff, x_max_eff)
                    ax.set_ylim(y_min_eff, y_max_eff)
                    _apply_axis_crop(ax, xlim=xlim, ylim=ylim)
                    if verbose:
                        logger.info("image_plot_with_spatial_image total time: %.3fs", perf_counter() - t_all)
                    plt.show()
                    return

        # --- Stable IMShow rendering with RGBA for per-spot alpha ---
        pts = pts_for_imshow if pts_for_imshow is not None else coordinates[["x", "y"]].values
        imshow_tile_size_eff = imshow_tile_size
        if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
            imshow_tile_size_eff = 1.0
        raster_canvas = raster_canvas_for_imshow
        if raster_canvas is None:
            raster_canvas = _raster.resolve_raster_canvas(
                pts=pts,
                x_min=float(x_min),
                x_max=float(x_max),
                y_min=float(y_min),
                y_max=float(y_max),
                figsize=figsize,
                fig_dpi=fig.get_dpi(),
                imshow_tile_size=imshow_tile_size_eff,
                imshow_scale_factor=imshow_scale_factor,
                imshow_tile_size_mode=imshow_tile_size_mode,
                imshow_tile_size_quantile=imshow_tile_size_quantile,
                imshow_tile_size_rounding=imshow_tile_size_rounding,
                imshow_tile_size_shrink=imshow_tile_size_shrink,
                pixel_perfect=bool(pixel_perfect),
                n_points=int(n_points_eff),
                logger=logger,
                context=raster_context,
                persistent_store=getattr(adata, "uns", None),
            )
        s_data_units = float(raster_canvas["s_data_units"])
        x_min_eff = float(raster_canvas["x_min_eff"])
        x_max_eff = float(raster_canvas["x_max_eff"])
        y_min_eff = float(raster_canvas["y_min_eff"])
        y_max_eff = float(raster_canvas["y_max_eff"])
        scale = float(raster_canvas["scale"])
        s = int(raster_canvas["tile_px"])
        w = int(raster_canvas["w"])
        h = int(raster_canvas["h"])
        if verbose:
            logger.info("Rendering to an image buffer of size (w, h): (%d, %d)", int(w), int(h))

        raster_state = _resolve_plot_raster_state(
            coordinates[["x", "y"]].to_numpy(dtype=float, copy=False),
            x_min_eff=float(x_min_eff),
            x_max_eff=float(x_max_eff),
            y_min_eff=float(y_min_eff),
            y_max_eff=float(y_max_eff),
            scale=float(scale),
            s=int(s),
            w=int(w),
            h=int(h),
            pixel_perfect=bool(pixel_perfect),
            pixel_shape=pixel_shape,
            coord_mode=coord_mode,
            gap_collapse_axis=gap_collapse_axis,
            gap_collapse_max_run_px=gap_collapse_max_run_px,
            soft_rasterization=soft_rasterization,
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=center_collision_radius,
            logger=logger,
            verbose=verbose,
            context="image_plot_with_spatial_image",
        )
        x_min_eff = float(raster_state["x_min_eff"])
        x_max_eff = float(raster_state["x_max_eff"])
        y_min_eff = float(raster_state["y_min_eff"])
        y_max_eff = float(raster_state["y_max_eff"])
        s = int(raster_state["s"])
        w = int(raster_state["w"])
        h = int(raster_state["h"])
        if verbose:
            logger.info("Scaled tile size (in pixels): %d", int(s))

        seg_codes = None
        use_stored_boundary_labels = stored_pixel_boundary is not None
        if (not use_stored_boundary_labels) and plot_boundaries and boundary_method == "pixel" and segment_available:
            seg_source = coordinates["Segment"] if "Segment" in coordinates.columns else coordinates_df["Segment"]
            seg_codes = pd.Categorical(seg_source).codes

        img_rgba, label_img, _ = _rasterize_rgba_and_label_buffer(
            cols=cols,
            cx=raster_state["cx"],
            cy=raster_state["cy"],
            w=int(w),
            h=int(h),
            s=int(s),
            pixel_shape=pixel_shape,
            alpha_arr=alpha_arr,
            alpha_point=float(alpha_point),
            vals=vals,
            prioritize_high_values=bool(prioritize_high_values),
            soft_enabled=bool(raster_state["soft_enabled"]),
            soft_fx=raster_state["soft_fx"],
            soft_fy=raster_state["soft_fy"],
            label_codes=seg_codes,
            runtime_fill_from_boundary=bool(runtime_fill_from_boundary),
            runtime_fill_closing_radius=int(runtime_fill_closing_radius),
            runtime_fill_external_radius=int(runtime_fill_external_radius),
            runtime_fill_holes=bool(runtime_fill_holes),
            chunk_size=int(chunk_size),
            persistent_store=getattr(adata, "uns", None),
            logger=logger,
            verbose=verbose,
            context="image_plot_with_spatial_image",
        )
        if use_stored_boundary_labels:
            label_img, _ = _resolve_stored_pixel_boundary_payload(
                stored_pixel_boundary,
                default_extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
                extent_scale=float(img_scaling_factor_adjusted),
            )
        if verbose and raster_state["collision_info"] is not None:
            logger.info(
                "image_plot_with_spatial_image: center collision fraction %.4f -> %.4f.",
                float(raster_state["collision_info"]["collision_fraction_before"]),
                float(raster_state["collision_info"]["collision_fraction_after"]),
            )

        ax.imshow(
            img_rgba,
            origin="lower",
            extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
            interpolation="nearest",
        )
        if verbose:
            logger.info("image_plot_with_spatial_image total time: %.3fs", perf_counter() - t_all)
        if label_img is not None:
            if stored_pixel_boundary is not None:
                _, boundary_extent = _resolve_stored_pixel_boundary_payload(
                    stored_pixel_boundary,
                    default_extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
                    extent_scale=float(img_scaling_factor_adjusted),
                )
            else:
                boundary_extent = [x_min_eff, x_max_eff, y_min_eff, y_max_eff]
            _plot_boundaries_from_label_image(
                ax,
                label_img,
                boundary_extent,
                color_for_pixel,
                boundary_linewidth,
                alpha,
                linestyle,
                pixel_smoothing_sigma,
                render=pixel_boundary_render,
                thickness_px=pixel_boundary_thickness_px,
                antialiased=pixel_boundary_antialiased,
                include_background=pixel_boundary_include_background,
            )

    else:
        # ---Automatic point size calculation for scatter plots ---
        if point_size is None:
            num_points = len(coordinates)
            base_size = (figsize[0] * figsize[1] * 100**2) / (num_points + 1)
            point_size = (scaling_factor / 5.0) * base_size
            point_size = np.clip(point_size, 0.01, 200)

        # Alpha-driven draw order for scatter (transparent first, opaque last/on top)
        if alpha_arr is not None and prioritize_high_values and overlap_priority not in {"few_front", "many_front"}:
            order_idx = np.argsort(alpha_arr)
            coordinates = coordinates.iloc[order_idx]
            cols = cols[order_idx]
            alpha_arr = alpha_arr[order_idx]

        # Category overlap priority ordering for scatter
        if color_by is not None and overlap_priority in {"few_front", "many_front"} and cat_priority_vals is not None:
            if overlap_priority == "few_front":
                order_idx = np.argsort(cat_priority_vals)[::-1]
            else:
                order_idx = np.argsort(cat_priority_vals)
            coordinates = coordinates.iloc[order_idx]
            cols = cols[order_idx]
            if alpha_arr is not None:
                alpha_arr = alpha_arr[order_idx]

        # Build RGBA colors for per-point alpha.
        if alpha_arr is not None:
            cols_rgba = np.concatenate([cols, alpha_arr[:, None]], axis=1)
            scatter_alpha = None  # RGBA carries alpha
        else:
            cols_rgba = np.concatenate([cols, np.full((cols.shape[0], 1), alpha_point, dtype=cols.dtype)], axis=1)
            scatter_alpha = None

        ax.scatter(
            x=coordinates["x"],
            y=coordinates["y"],
            s=point_size,
            c=cols_rgba,
            marker="o" if pixel_shape == "circle" else "s",
            linewidths=0,
        )
        # Legend for categorical coloring (scatter path)
        if color_by is not None and show_legend:
            categories = pd.Categorical(coordinates_df[color_by]).categories
            legend_colors = [category_lut[i] for i in range(len(categories))]
            handles = [patches.Patch(color=legend_colors[i], label=str(cat)) for i, cat in enumerate(categories)]
            if legend_outside:
                fig.subplots_adjust(right=0.82)
                loc_final = 'center left' if legend_loc == 'best' else legend_loc
                if legend_fontsize is None:
                    n_items = len(categories)
                    rows = int(np.ceil(n_items / max(1, legend_ncol)))
                    size_factor = (figsize[0] * figsize[1]) ** 0.5
                    base = 12 + (size_factor - 9) * 0.5
                    if rows > 8: base -= 2
                    elif rows > 5: base -= 1
                    auto_fs = int(np.clip(round(base), 10, 28))
                else:
                    auto_fs = legend_fontsize
                fig.legend(handles=handles, loc=loc_final, bbox_to_anchor=legend_bbox_to_anchor,
                           ncol=legend_ncol, frameon=False, fontsize=auto_fs)
            else:
                if legend_fontsize is None:
                    n_items = len(categories)
                    rows = int(np.ceil(n_items / max(1, legend_ncol)))
                    size_factor = (figsize[0] * figsize[1]) ** 0.5
                    base = 12 + (size_factor - 9) * 0.5
                    if rows > 8: base -= 2
                    elif rows > 5: base -= 1
                    auto_fs = int(np.clip(round(base), 10, 28))
                else:
                    auto_fs = legend_fontsize
                ax.legend(handles=handles, loc=legend_loc, ncol=legend_ncol, frameon=False, fontsize=auto_fs)

        # Pixel boundary extraction for scatter
        if plot_boundaries and boundary_method == "pixel" and segment_available:
            if stored_pixel_boundary is not None:
                label_img, boundary_extent = _resolve_stored_pixel_boundary_payload(
                    stored_pixel_boundary,
                    default_extent=[x_min, x_max, y_min, y_max],
                    extent_scale=float(img_scaling_factor_adjusted),
                )
                _plot_boundaries_from_label_image(
                    ax,
                    label_img,
                    boundary_extent,
                    color_for_pixel,
                    boundary_linewidth,
                    alpha,
                    linestyle,
                    pixel_smoothing_sigma,
                    render=pixel_boundary_render,
                    thickness_px=pixel_boundary_thickness_px,
                    antialiased=pixel_boundary_antialiased,
                    include_background=pixel_boundary_include_background,
                )
            else:
                pts = coordinates[["x", "y"]].values
                raster_canvas_scatter = _raster.resolve_raster_canvas(
                    pts=pts,
                    x_min=float(x_min),
                    x_max=float(x_max),
                    y_min=float(y_min),
                    y_max=float(y_max),
                    figsize=figsize,
                    fig_dpi=fig.get_dpi(),
                    imshow_tile_size=imshow_tile_size,
                    imshow_scale_factor=imshow_scale_factor,
                    imshow_tile_size_mode=imshow_tile_size_mode,
                    imshow_tile_size_quantile=imshow_tile_size_quantile,
                    imshow_tile_size_rounding=imshow_tile_size_rounding,
                    imshow_tile_size_shrink=imshow_tile_size_shrink,
                    pixel_perfect=bool(pixel_perfect),
                    n_points=int(len(pts)),
                    logger=logger if verbose else None,
                    context="image_plot_with_spatial_image(scatter-boundaries)",
                    persistent_store=getattr(adata, "uns", None),
                )
                raster_state = _resolve_plot_raster_state(
                    pts,
                    x_min_eff=float(raster_canvas_scatter["x_min_eff"]),
                    x_max_eff=float(raster_canvas_scatter["x_max_eff"]),
                    y_min_eff=float(raster_canvas_scatter["y_min_eff"]),
                    y_max_eff=float(raster_canvas_scatter["y_max_eff"]),
                    scale=float(raster_canvas_scatter["scale"]),
                    s=int(raster_canvas_scatter["tile_px"]),
                    w=int(raster_canvas_scatter["w"]),
                    h=int(raster_canvas_scatter["h"]),
                    pixel_perfect=bool(pixel_perfect),
                    pixel_shape=pixel_shape,
                    coord_mode=coord_mode,
                    gap_collapse_axis=gap_collapse_axis,
                    gap_collapse_max_run_px=gap_collapse_max_run_px,
                    soft_rasterization=soft_rasterization,
                    resolve_center_collisions=resolve_center_collisions,
                    center_collision_radius=center_collision_radius,
                    logger=logger,
                    verbose=verbose,
                    context="image_plot_with_spatial_image(scatter-boundaries)",
                )
                seg_source = coordinates["Segment"] if "Segment" in coordinates.columns else coordinates_df["Segment"]
                seg_codes = pd.Categorical(seg_source).codes
                _, label_img, _ = _rasterize_rgba_and_label_buffer(
                    cols=np.zeros((len(pts), 3), dtype=np.float32),
                    cx=raster_state["cx"],
                    cy=raster_state["cy"],
                    w=int(raster_state["w"]),
                    h=int(raster_state["h"]),
                    s=int(raster_state["s"]),
                    pixel_shape=pixel_shape,
                    alpha_arr=None,
                    alpha_point=1.0,
                    vals=vals,
                    prioritize_high_values=bool(prioritize_high_values),
                    soft_enabled=bool(raster_state["soft_enabled"]),
                    soft_fx=raster_state["soft_fx"],
                    soft_fy=raster_state["soft_fy"],
                    label_codes=seg_codes,
                    runtime_fill_from_boundary=bool(runtime_fill_from_boundary),
                    runtime_fill_closing_radius=int(runtime_fill_closing_radius),
                    runtime_fill_external_radius=int(runtime_fill_external_radius),
                    runtime_fill_holes=bool(runtime_fill_holes),
                    chunk_size=int(chunk_size),
                    persistent_store=getattr(adata, "uns", None),
                    logger=logger,
                    verbose=verbose,
                    context="image_plot_with_spatial_image(scatter-boundaries)",
                )

                _plot_boundaries_from_label_image(
                    ax,
                    label_img,
                    [
                        float(raster_state["x_min_eff"]),
                        float(raster_state["x_max_eff"]),
                        float(raster_state["y_min_eff"]),
                        float(raster_state["y_max_eff"]),
                    ],
                    color_for_pixel,
                    boundary_linewidth,
                    alpha,
                    linestyle,
                    pixel_smoothing_sigma,
                    render=pixel_boundary_render,
                    thickness_px=pixel_boundary_thickness_px,
                    antialiased=pixel_boundary_antialiased,
                    include_background=pixel_boundary_include_background,
                )

    # Non-pixel boundary methods (parity with `image_plot`)
    if segment_available and plot_boundaries and boundary_method != "pixel":
        tree = KDTree(coordinates[["x", "y"]].values)
        segments = coordinates_df["Segment"].unique()

        for seg in segments:
            is_highlight = seg in highlight_segments
            seg_tiles = coordinates_df[coordinates_df["Segment"] == seg]
            if len(seg_tiles) < 3:
                if verbose:
                    logger.info(
                        "Segment %s has fewer than 3 points. Skipping boundary plotting.",
                        str(seg),
                    )
                continue

            points = seg_tiles[["x", "y"]].values

            unique_points = np.unique(points, axis=0)
            if unique_points.shape[0] < 3 or is_collinear(unique_points):
                logger.warning(
                    f"Segment {seg} points are collinear or insufficient. Using LineString as boundary."
                )
                line = LineString(unique_points)
                clipped_line = line.intersection(plot_boundary_polygon)
                if clipped_line.is_empty:
                    continue

                line_points = []
                if isinstance(clipped_line, (LineString)):
                    x, y = clipped_line.xy
                    line_points = np.column_stack((x, y))
                elif isinstance(clipped_line, (MultiLineString, GeometryCollection)):
                    for geom in clipped_line.geoms:
                        if isinstance(geom, LineString):
                            x, y = geom.xy
                            line_points.extend(np.column_stack((x, y)))
                    line_points = np.array(line_points)
                else:
                    logger.warning(
                        f"Unsupported geometry type after clipping: {type(clipped_line)}"
                    )
                    continue

                _, idx = tree.query(line_points)

                if fixed_boundary_color is not None:
                    line_colors = np.array([fixed_boundary_color] * len(line_points))
                else:
                    if len(dimensions) == 3:
                        line_colors = coordinates.iloc[idx][["R", "G", "B"]].values
                    else:
                        grey = coordinates.iloc[idx]["Grey"].values
                        if cmap is None:
                            line_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        else:
                            cm = plt.get_cmap(cmap)
                            line_colors = cm(grey)[:, :3]
                    line_colors = brighten_colors(
                        np.clip(line_colors, 0, 1), factor=brighten_factor
                    )
                    if is_highlight:
                        if highlight_boundary_color is not None:
                            line_colors = np.array(
                                [mcolors.to_rgb(highlight_boundary_color)] * len(line_points)
                            )
                        elif highlight_brighten_factor != 1:
                            line_colors = brighten_colors(
                                line_colors, factor=highlight_brighten_factor
                            )
                    elif highlight_segments:
                        if dim_to_grey and len(dimensions) == 3:
                            grey = np.dot(line_colors, [0.299, 0.587, 0.114])
                            line_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        line_colors *= dim_other_segments

                line_segments = np.array(
                    [line_points[i : i + 2] for i in range(len(line_points) - 1)]
                )
                current_lw = highlight_linewidth if is_highlight else boundary_linewidth
                lc = LineCollection(
                    line_segments,
                    colors=line_colors[:-1],
                    linewidth=current_lw,
                    alpha=alpha,
                    linestyle=linestyle,
                )
                ax.add_collection(lc)
                continue

            poly = None
            try:
                if boundary_method == "alphashape":
                    alpha_shape = alphashape.alphashape(unique_points, alpha_shape_alpha)
                    poly = alpha_shape
                elif boundary_method == "convex_hull":
                    poly = MultiPoint(unique_points).convex_hull
                elif boundary_method == "oriented_bbox":
                    poly = MultiPoint(unique_points).minimum_rotated_rectangle
                elif boundary_method == "concave_hull":
                    poly = alphashape.alphashape(unique_points, alpha_shape_alpha)
                    poly = smooth_polygon(poly)
                else:
                    raise ValueError(f"Unknown boundary method '{boundary_method}'")
            except Exception as e:
                logger.warning(f"Failed to compute boundary for Segment {seg}: {e}")
                continue

            try:
                clipped_polygon = poly.intersection(plot_boundary_polygon)
                if clipped_polygon.is_empty:
                    if verbose:
                        logger.warning(f"Clipped boundary for Segment {seg} is empty.")
                    continue
            except Exception as e:
                logger.error(f"Failed to clip boundary for Segment {seg}: {e}")
                continue

            polygons = []
            if isinstance(clipped_polygon, (Polygon, LineString)):
                polygons = [clipped_polygon]
            elif isinstance(clipped_polygon, MultiPolygon):
                polygons = list(clipped_polygon.geoms)
            elif isinstance(clipped_polygon, GeometryCollection):
                for geom in clipped_polygon.geoms:
                    if isinstance(geom, (Polygon, LineString)):
                        polygons.append(geom)
            else:
                logger.warning(
                    f"Unsupported geometry type in clipped polygon for Segment {seg}: {type(clipped_polygon)}"
                )
                continue

            for poly in polygons:
                if isinstance(poly, Polygon):
                    x, y = poly.exterior.xy
                elif isinstance(poly, LineString):
                    x, y = poly.xy
                else:
                    logger.warning(
                        f"Unsupported geometry type in polygon for Segment {seg}: {type(poly)}"
                    )
                    continue

                boundary_points = np.column_stack((x, y))
                _, idx = tree.query(boundary_points)

                if fixed_boundary_color is not None:
                    boundary_colors = np.array([fixed_boundary_color] * len(boundary_points))
                else:
                    if len(dimensions) == 3:
                        boundary_colors = coordinates.iloc[idx][["R", "G", "B"]].values
                    else:
                        grey = coordinates.iloc[idx]["Grey"].values
                        if cmap is None:
                            boundary_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        else:
                            cm = plt.get_cmap(cmap)
                            boundary_colors = cm(grey)[:, :3]

                    boundary_colors = brighten_colors(
                        np.clip(boundary_colors, 0, 1), factor=brighten_factor
                    )
                    if is_highlight:
                        if highlight_boundary_color is not None:
                            boundary_colors = np.array(
                                [mcolors.to_rgb(highlight_boundary_color)] * len(boundary_points)
                            )
                        elif highlight_brighten_factor != 1:
                            boundary_colors = brighten_colors(
                                boundary_colors, factor=highlight_brighten_factor
                            )
                    elif highlight_segments:
                        if dim_to_grey and len(dimensions) == 3:
                            grey = np.dot(boundary_colors, [0.299, 0.587, 0.114])
                            boundary_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        boundary_colors *= dim_other_segments

                segments_lines = np.array(
                    [boundary_points[i : i + 2] for i in range(len(boundary_points) - 1)]
                )

                current_lw = highlight_linewidth if is_highlight else boundary_linewidth
                lc = LineCollection(
                    segments_lines,
                    colors=boundary_colors[:-1],
                    linewidth=current_lw,
                    alpha=alpha,
                    linestyle=linestyle,
                )
                ax.add_collection(lc)

                if fill_boundaries and isinstance(poly, Polygon):
                    patch = patches.Polygon(
                        np.array(poly.exterior.coords),
                        closed=True,
                        facecolor=fill_color,
                        edgecolor=None,
                        alpha=fill_alpha,
                    )
                    ax.add_patch(patch)

    if title is None:
        title_text = f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
    else:
        title_text = title
    if not show_axes:
        ax.axis("off")
    ax.set_title(title_text, fontsize=figsize[0] * 1.5)
    if show_colorbar and len(dimensions) == 1 and not use_categorical:
        cm = plt.get_cmap(cmap if cmap is not None else "Greys")
        if (color_norm_vmin is not None) and (color_norm_vmax is not None):
            vmin, vmax = float(color_norm_vmin), float(color_norm_vmax)
        else:
            vmin = raw_vals.min() if raw_vals is not None else grey_vals.min()
            vmax = raw_vals.max() if raw_vals is not None else grey_vals.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    plt.show()
    # Do not return plot object to avoid auto display
    coord_mode = str(coordinate_mode or "spatial").lower()
    if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
        coord_mode = "spatial"
    if coord_mode == "auto":
        if origin and (array_row_key in adata.obs.columns) and (array_col_key in adata.obs.columns) and int(adata.n_obs) >= 200_000:
            coord_mode = "visiumhd"
        else:
            coord_mode = "spatial"
