import hashlib
import os
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import KDTree


_DEFAULT_MAX_RASTER_PIXELS = 25_000_000
_POINT_SIGNATURE_SAMPLES = 128
_TILE_SIZE_CACHE_MAX = 16
_RASTER_CANVAS_CACHE_MAX = 16
_PERSISTENT_RASTER_CANVAS_CACHE_MAX = 64
_PERSISTENT_RASTER_CANVAS_BUCKET_KEY = "_spix_raster_canvas_cache_v1"
_RASTER_CANVAS_CACHE_VERSION = "20260417_persist_v1"
_FILL_ASSIGNMENT_CACHE_MAX = 16
_PERSISTENT_FILL_CACHE_MAX = 32
_PERSISTENT_FILL_CACHE_BUCKET_KEY = "_spix_raster_fill_cache_v1"
_RASTER_FILL_CACHE_VERSION = "20260417_fill_v1"
_TILE_SIZE_CACHE: "OrderedDict[tuple, float]" = OrderedDict()
_RASTER_CANVAS_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_FILL_ASSIGNMENT_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()


def _lru_cache_get(store: "OrderedDict[tuple, object]", key: tuple):
    value = store.get(key, None)
    if value is not None:
        store.move_to_end(key)
    return value


def _lru_cache_put(store: "OrderedDict[tuple, object]", key: tuple, value, *, max_items: int) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > int(max_items):
        store.popitem(last=False)


def _persistent_canvas_cache_digest(cache_key: tuple) -> str:
    payload = repr(cache_key).encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()[:24]


def _persistent_canvas_cache_get(store, cache_key: tuple):
    if not isinstance(store, dict):
        return None
    bucket = store.get(_PERSISTENT_RASTER_CANVAS_BUCKET_KEY, None)
    if not isinstance(bucket, dict):
        return None
    digest = _persistent_canvas_cache_digest(cache_key)
    cached = bucket.get(digest, None)
    if not isinstance(cached, dict):
        return None
    # Touch for simple insertion-ordered LRU behavior.
    try:
        bucket.pop(digest, None)
        bucket[digest] = dict(cached)
    except Exception:
        pass
    return dict(cached)


def _persistent_canvas_cache_put(store, cache_key: tuple, value: dict) -> None:
    if not isinstance(store, dict):
        return
    digest = _persistent_canvas_cache_digest(cache_key)
    bucket = store.get(_PERSISTENT_RASTER_CANVAS_BUCKET_KEY, None)
    if not isinstance(bucket, dict):
        bucket = {}
        store[_PERSISTENT_RASTER_CANVAS_BUCKET_KEY] = bucket
    bucket[digest] = dict(value)
    while len(bucket) > int(_PERSISTENT_RASTER_CANVAS_CACHE_MAX):
        try:
            oldest_key = next(iter(bucket))
        except StopIteration:
            break
        bucket.pop(oldest_key, None)


def _persistent_fill_cache_get(store, cache_key: tuple):
    if not isinstance(store, dict):
        return None
    bucket = store.get(_PERSISTENT_FILL_CACHE_BUCKET_KEY, None)
    if not isinstance(bucket, dict):
        return None
    digest = _persistent_canvas_cache_digest(cache_key)
    cached = bucket.get(digest, None)
    if not isinstance(cached, dict):
        return None
    try:
        bucket.pop(digest, None)
        bucket[digest] = dict(cached)
    except Exception:
        pass
    return dict(cached)


def _persistent_fill_cache_put(store, cache_key: tuple, value: dict) -> None:
    if not isinstance(store, dict):
        return
    digest = _persistent_canvas_cache_digest(cache_key)
    bucket = store.get(_PERSISTENT_FILL_CACHE_BUCKET_KEY, None)
    if not isinstance(bucket, dict):
        bucket = {}
        store[_PERSISTENT_FILL_CACHE_BUCKET_KEY] = bucket
    bucket[digest] = dict(value)
    while len(bucket) > int(_PERSISTENT_FILL_CACHE_MAX):
        try:
            oldest_key = next(iter(bucket))
        except StopIteration:
            break
        bucket.pop(oldest_key, None)


def _points_signature(points: np.ndarray, *, max_samples: int = _POINT_SIGNATURE_SAMPLES) -> tuple:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return ("invalid", int(pts.size))
    pts2 = pts[:, :2]
    n = int(pts2.shape[0])
    if n <= 0:
        return ("empty", 0)
    sample_n = min(n, int(max_samples))
    if sample_n == n:
        sample = np.ascontiguousarray(pts2, dtype=np.float32)
    else:
        idx = np.linspace(0, n - 1, num=sample_n, dtype=np.int64)
        sample = np.ascontiguousarray(pts2[idx], dtype=np.float32)
    hasher = hashlib.sha1()
    hasher.update(np.asarray([n], dtype=np.int64).tobytes())
    hasher.update(np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0).tobytes())
    return ("pts", n, hasher.hexdigest())


def _centers_signature(seed_x: np.ndarray, seed_y: np.ndarray, *, max_samples: int = _POINT_SIGNATURE_SAMPLES) -> tuple:
    sx = np.asarray(seed_x, dtype=np.int32).ravel()
    sy = np.asarray(seed_y, dtype=np.int32).ravel()
    n = int(min(sx.size, sy.size))
    if n <= 0:
        return ("centers", 0, "empty")
    sample_n = min(n, int(max_samples))
    if sample_n == n:
        sx_s = np.ascontiguousarray(sx[:n], dtype=np.int32)
        sy_s = np.ascontiguousarray(sy[:n], dtype=np.int32)
    else:
        idx = np.linspace(0, n - 1, num=sample_n, dtype=np.int64)
        sx_s = np.ascontiguousarray(sx[idx], dtype=np.int32)
        sy_s = np.ascontiguousarray(sy[idx], dtype=np.int32)
    hasher = hashlib.sha1()
    hasher.update(np.asarray([n], dtype=np.int64).tobytes())
    hasher.update(sx_s.tobytes())
    hasher.update(sy_s.tobytes())
    hasher.update(np.asarray([int(np.nanmin(sx[:n])), int(np.nanmax(sx[:n])), int(np.nanmin(sy[:n])), int(np.nanmax(sy[:n]))], dtype=np.int32).tobytes())
    return ("centers", n, hasher.hexdigest())


def _occupancy_signature(mask: np.ndarray) -> tuple:
    occ = np.asarray(mask, dtype=bool)
    if occ.ndim != 2:
        return ("occ_invalid", tuple(int(x) for x in occ.shape), int(occ.size))
    flat = np.ascontiguousarray(occ.reshape(-1), dtype=np.uint8)
    packed = np.packbits(flat, bitorder="little")
    hasher = hashlib.sha1()
    hasher.update(np.asarray(occ.shape, dtype=np.int64).tobytes())
    hasher.update(np.asarray([int(flat.sum())], dtype=np.int64).tobytes())
    hasher.update(packed.tobytes())
    return ("occ", int(occ.shape[0]), int(occ.shape[1]), hasher.hexdigest())


def resolve_max_raster_pixels(
    *,
    env_key: str = "SPIX_PLOT_MAX_RASTER_PIXELS",
    fallback_env_key: Optional[str] = None,
    n_points: Optional[int] = None,
    support_fraction: Optional[float] = None,
    default: int = _DEFAULT_MAX_RASTER_PIXELS,
) -> int:
    """Resolve a raster pixel cap, auto-raising it for large point counts.

    Default behavior keeps the historical 25M cap, but if the raw number of
    points exceeds that cap we allow the cap to grow to at least ``n_points``.
    """
    raw = os.environ.get(env_key, None)
    if raw is None and fallback_env_key is not None:
        raw = os.environ.get(fallback_env_key, None)
    try:
        base = int(raw) if raw is not None else int(default)
    except Exception:
        base = int(default)
    base = max(1, int(base))
    if n_points is not None:
        try:
            base = max(base, int(n_points))
        except Exception:
            pass
    if support_fraction is not None and n_points is not None:
        try:
            sf = float(support_fraction)
            if np.isfinite(sf) and sf > 0:
                margin = float(os.environ.get("SPIX_PLOT_SUPPORT_FRACTION_CAP_MARGIN", "1.30"))
                margin = max(1.0, float(margin))
                needed = float(n_points) / float(max(sf, 1e-6))
                base = max(base, int(np.ceil(float(needed) * float(margin))))
        except Exception:
            pass
    return int(base)


def _tile_sample_rng() -> np.random.Generator:
    """Return a local RNG for tile-size estimation.

    By default this is deterministic so cached raster geometry is stable across
    kernel restarts. Set ``SPIX_PLOT_TILE_SAMPLE_SEED=random`` to opt back into
    non-deterministic sampling, or provide an integer seed to control it.
    """
    seed_raw = os.environ.get("SPIX_PLOT_TILE_SAMPLE_SEED", "0")
    if str(seed_raw).strip().lower() in {"random", "none"}:
        return np.random.default_rng()
    try:
        seed = int(seed_raw)
    except Exception:
        seed = 0
    return np.random.default_rng(seed)


def round_tile_size_px(s_data_units: float, scale: float, rounding: Optional[str]) -> int:
    mode = str(rounding or "ceil").lower()
    s_float = float(s_data_units) * float(scale)
    if mode == "floor":
        s = int(np.floor(s_float))
    elif mode == "round":
        s = int(np.round(s_float))
    else:
        s = int(np.ceil(s_float))
    return max(1, int(s))


def estimate_tile_size_units(
    pts: np.ndarray,
    imshow_tile_size: Optional[float],
    imshow_scale_factor: float,
    imshow_tile_size_mode: Optional[str],
    imshow_tile_size_quantile: Optional[float],
    imshow_tile_size_shrink: float,
    logger=None,
) -> float:
    """Estimate tile pitch in data units (image_plot-compatible)."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("pts must be a 2D array with at least two columns.")
    pts = pts[:, :2]
    finite_mask = np.isfinite(pts).all(axis=1)
    if not np.all(finite_mask):
        if logger is not None:
            logger.warning(
                "Dropping %d non-finite coordinate rows before tile-size estimation.",
                int(np.size(finite_mask) - np.count_nonzero(finite_mask)),
            )
        pts = pts[finite_mask]

    s_data_units = 1.0
    mode = str(imshow_tile_size_mode or "nn").lower()
    if mode not in {"nn", "axis", "auto"}:
        mode = "nn"
    if imshow_tile_size_quantile is None:
        q = 0.2 if mode == "auto" else 0.5
    else:
        try:
            q = float(imshow_tile_size_quantile)
            if not (0.0 < q <= 1.0):
                q = 0.5
        except Exception:
            q = 0.5

    cache_key = (
        _points_signature(pts),
        None if imshow_tile_size is None else float(imshow_tile_size),
        float(imshow_scale_factor),
        str(mode),
        float(q),
        float(imshow_tile_size_shrink),
    )
    cached = _lru_cache_get(_TILE_SIZE_CACHE, cache_key)
    if cached is not None:
        return float(cached)

    if imshow_tile_size is None:
        if len(pts) > 1:
            sample_size = min(len(pts), 100000)
            sample_indices = _tile_sample_rng().choice(len(pts), sample_size, replace=False)
            pts_s = pts[sample_indices]
            k = min(7, len(pts_s)) if mode in {"axis", "auto"} else 2
            tree = KDTree(pts_s)
            d, idx = tree.query(pts_s, k=k)

            nn = d[:, 1] if d.ndim > 1 and d.shape[1] > 1 else d
            nn = nn[nn > 0]
            nn_est = float(np.quantile(nn, q)) if nn.size else None

            axis_est = None
            if mode in {"axis", "auto"} and k >= 2:
                neighbors = pts_s[idx[:, 1:]]
                deltas = neighbors - pts_s[:, None, :]
                abs_dx = np.abs(deltas[..., 0]).ravel()
                abs_dy = np.abs(deltas[..., 1]).ravel()
                abs_dx = abs_dx[abs_dx > 0]
                abs_dy = abs_dy[abs_dy > 0]
                if abs_dx.size and abs_dy.size:
                    axis_est = float(min(np.quantile(abs_dx, q), np.quantile(abs_dy, q)))

            if mode == "nn":
                s_data_units = nn_est if nn_est is not None else s_data_units
            elif mode == "axis":
                s_data_units = axis_est if axis_est is not None else (nn_est or s_data_units)
            else:
                if axis_est is not None and nn_est is not None:
                    if axis_est >= 0.7 * nn_est:
                        s_data_units = axis_est
                    else:
                        s_data_units = nn_est
                else:
                    s_data_units = axis_est if axis_est is not None else (nn_est or s_data_units)

            if logger is not None:
                logger.info("Auto-calculated tile size (data units): %.2f", float(s_data_units))
        elif logger is not None:
            logger.warning(
                "Tile-size estimation received fewer than two finite points; using fallback size %.2f.",
                float(s_data_units),
            )
    else:
        s_data_units = float(imshow_tile_size)

    s_data_units *= float(imshow_scale_factor)
    s_data_units *= float(imshow_tile_size_shrink)
    # Keep very small but valid pitches for normalized/continuous coordinate systems
    # (for example Squidpy MERFISH `obsm["spatial"]`). A hard 0.1 floor turns those
    # into oversized display tiles and collapses the raster canvas.
    finite_pts = pts[np.isfinite(pts).all(axis=1)]
    if finite_pts.shape[0] >= 2:
        x_span = float(np.ptp(finite_pts[:, 0]))
        y_span = float(np.ptp(finite_pts[:, 1]))
        span = max(x_span, y_span, 1.0)
    else:
        span = 1.0
    min_pitch = max(np.finfo(float).eps, span * 1e-9)
    s_data_units = max(float(min_pitch), float(s_data_units))
    out = float(s_data_units)
    _lru_cache_put(_TILE_SIZE_CACHE, cache_key, out, max_items=_TILE_SIZE_CACHE_MAX)
    return out


def cap_tile_size_by_density(
    *,
    s_data_units: float,
    x_range: float,
    y_range: float,
    n_points: int,
    logger=None,
    context: str = "",
) -> float:
    """Reduce an overestimated tile pitch based on point density."""
    if n_points <= 0:
        return float(s_data_units)
    xr = float(x_range) if x_range and x_range > 0 else 1.0
    yr = float(y_range) if y_range and y_range > 0 else 1.0
    area = xr * yr
    if area <= 0:
        return float(s_data_units)

    factor = float(os.environ.get("SPIX_PLOT_DENSITY_PITCH_FACTOR", "1.0"))
    factor = float(np.clip(factor, 0.5, 2.0))
    s_density = float(np.sqrt(area / float(n_points)))
    s_cap = max(0.1, s_density * factor)
    if float(s_data_units) > s_cap:
        if logger is not None:
            msg = f"{context}: " if context else ""
            logger.warning(
                "%sAuto tile-size %.2f looks too large for %d points; capping to %.2f (density-based).",
                msg,
                float(s_data_units),
                int(n_points),
                float(s_cap),
            )
        return float(s_cap)
    return float(s_data_units)


def cap_tile_size_by_render_extent(
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
    x0, x1, y0, y1 = effective_extent(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        s_data_units=float(s_data_units),
        pixel_perfect=bool(pixel_perfect),
    )
    return cap_tile_size_by_density(
        s_data_units=float(s_data_units),
        x_range=float(max(1.0, x1 - x0)),
        y_range=float(max(1.0, y1 - y0)),
        n_points=int(n_points),
        logger=logger,
        context=str(context or ""),
    )


def resolve_figsize_dpi_for_tiles(
    *,
    figsize,
    fig_dpi,
    x_range: float,
    y_range: float,
    s_data_units: float,
    pixel_perfect: bool,
    n_points: int | None = None,
    logger=None,
) -> tuple:
    """Auto-pick (figsize, dpi) so the raster canvas can represent all tiles."""
    if s_data_units <= 0:
        return figsize, fig_dpi

    # Default to 1.0 so auto-size stays minimal; users can increase this env var
    # to reduce overlaps (at the cost of a larger raster).
    tiles_factor = float(os.environ.get("SPIX_PLOT_MIN_PIXELS_PER_TILE", "1.0"))
    tiles_factor = max(1.0, float(tiles_factor))

    xp = float(x_range) if x_range and x_range > 0 else 1.0
    yp = float(y_range) if y_range and y_range > 0 else 1.0

    pitch_px: Optional[int] = None
    if n_points is not None and int(n_points) > 0:
        area = max(1e-9, xp * yp)
        req_total_px = float(n_points) * float(tiles_factor)
        scale_needed = float(np.sqrt(req_total_px / area))
        if pixel_perfect:
            # Choose an integer pixel pitch up-front so later "scale snap" does not
            # shrink the final raster below the requested pixel budget.
            pitch_float = float(s_data_units) * float(scale_needed)
            # Use round (not ceil) to avoid jumping 1→2 when slightly above 1,
            # keeping auto-size minimal while still matching the later snap logic.
            pitch_px = max(1, int(np.round(pitch_float)))
            scale_needed = float(pitch_px) / float(s_data_units)
        req_w_px = int(np.ceil(xp * scale_needed))
        req_h_px = int(np.ceil(yp * scale_needed))
        req_w_px = max(1, req_w_px)
        req_h_px = max(1, req_h_px)
        if pixel_perfect and pitch_px is not None:
            cap_total = float(req_w_px) * float(req_h_px)
            # If the integer pitch snap forces the pixel grid below the requested
            # pixel budget, overlaps are unavoidable. Warn once so users can
            # choose a larger canvas or different pitch settings.
            if cap_total + 0.5 < float(req_total_px) and logger is not None:
                logger.warning(
                    "Pixel-perfect snap yields %dx%d=%.1fM px for ~%d tiles (%.2f px/tile); overlaps likely.",
                    int(req_w_px),
                    int(req_h_px),
                    cap_total / 1e6,
                    int(n_points),
                    cap_total / float(max(1, int(n_points))),
                )
    else:
        # Fallback: approximate by tile pitch (data units) when n_points is unknown.
        nx = max(1, int(np.ceil(float(x_range) / float(s_data_units))))
        ny = max(1, int(np.ceil(float(y_range) / float(s_data_units))))
        req_w_px = int(nx)
        req_h_px = int(ny)

    max_raster_px = resolve_max_raster_pixels(
        env_key="SPIX_PLOT_MAX_RASTER_PIXELS",
        n_points=n_points,
    )
    req_total = int(req_w_px) * int(req_h_px)
    if req_total > max_raster_px:
        if logger is not None:
            logger.warning(
                "Canvas needed for all tiles is huge (%dx%d=%.1fM px). "
                "Capping to SPIX_PLOT_MAX_RASTER_PIXELS=%d; expect overlaps.",
                req_w_px,
                req_h_px,
                req_total / 1e6,
                max_raster_px,
            )
        scale_cap = float(np.sqrt(float(max_raster_px) / float(max(1, req_total))))
        if pixel_perfect and pitch_px is not None:
            # Reduce integer pitch until within cap.
            new_pitch = max(1, int(np.floor(float(pitch_px) * float(scale_cap))))
            if new_pitch < pitch_px:
                pitch_px = new_pitch
                scale_needed = float(pitch_px) / float(s_data_units)
                req_w_px = max(1, int(np.ceil(xp * scale_needed)))
                req_h_px = max(1, int(np.ceil(yp * scale_needed)))
                # Guard against ceil pushing us slightly above cap.
                for _ in range(5):
                    if int(req_w_px) * int(req_h_px) <= max_raster_px or pitch_px <= 1:
                        break
                    pitch_px = max(1, int(pitch_px) - 1)
                    scale_needed = float(pitch_px) / float(s_data_units)
                    req_w_px = max(1, int(np.ceil(xp * scale_needed)))
                    req_h_px = max(1, int(np.ceil(yp * scale_needed)))
        else:
            req_w_px = max(1, int(np.floor(req_w_px * scale_cap)))
            req_h_px = max(1, int(np.floor(req_h_px * scale_cap)))

    if figsize is not None and fig_dpi is not None:
        cur_w_px = int(np.ceil(float(figsize[0]) * float(fig_dpi)))
        cur_h_px = int(np.ceil(float(figsize[1]) * float(fig_dpi)))
        if (cur_w_px < req_w_px) or (cur_h_px < req_h_px):
            if logger is not None:
                logger.warning(
                    "figsize/dpi too small for tile grid: have %dx%d px, need ~%dx%d px "
                    "(set figsize=None or fig_dpi=None to auto-adjust).",
                    cur_w_px,
                    cur_h_px,
                    req_w_px,
                    req_h_px,
                )
        return figsize, fig_dpi

    max_inches = float(os.environ.get("SPIX_PLOT_MAX_INCHES", "30"))
    max_inches = max(1.0, max_inches)
    min_dpi = int(os.environ.get("SPIX_PLOT_MIN_DPI", "100"))
    max_dpi = int(os.environ.get("SPIX_PLOT_MAX_DPI", "400"))
    min_dpi = max(50, min_dpi)
    max_dpi = max(min_dpi, max_dpi)

    if figsize is None and fig_dpi is None:
        dpi = int(np.ceil(float(max(req_w_px, req_h_px)) / float(max_inches)))
        dpi = int(np.clip(dpi, min_dpi, max_dpi))
        figsize = (float(req_w_px) / float(dpi), float(req_h_px) / float(dpi))
        fig_dpi = dpi
        if logger is not None:
            logger.info(
                "Auto-size target pixels=%dx%d (%.1fM) for n_tiles=%d (min_pixels_per_tile=%.2f).",
                int(req_w_px),
                int(req_h_px),
                (float(req_w_px) * float(req_h_px)) / 1e6,
                int(n_points) if n_points is not None else -1,
                float(tiles_factor),
            )
        return figsize, fig_dpi

    if figsize is None and fig_dpi is not None:
        dpi = int(fig_dpi)
        dpi = max(1, dpi)
        figsize = (float(req_w_px) / float(dpi), float(req_h_px) / float(dpi))
        return figsize, fig_dpi

    if figsize is not None and fig_dpi is None:
        w_in, h_in = float(figsize[0]), float(figsize[1])
        w_in = max(1e-6, w_in)
        h_in = max(1e-6, h_in)
        dpi_needed = int(np.ceil(max(float(req_w_px) / w_in, float(req_h_px) / h_in)))
        dpi_needed = int(np.clip(dpi_needed, min_dpi, max_dpi))
        return figsize, dpi_needed

    return figsize, fig_dpi


def maybe_boost_canvas_for_unique_centers(
    *,
    pts: np.ndarray,
    x_min_eff: float,
    y_min_eff: float,
    x_range: float,
    y_range: float,
    scale: float,
    w: int,
    h: int,
    pixel_perfect: bool,
    tile_px: int,
    logger=None,
    context: str = "",
) -> tuple[float, int, int, dict]:
    if int(tile_px) != 1:
        return float(scale), int(w), int(h), {
            "applied": False,
            "unique_before": None,
            "unique_after": None,
            "boost": 1.0,
            "forced_search": False,
            "max_raster_px_initial": None,
            "max_raster_px_final": None,
        }
    pts_arr = np.asarray(pts, dtype=float)
    n_pts = int(pts_arr.shape[0])
    if n_pts <= 1:
        return float(scale), int(w), int(h), {
            "applied": False,
            "unique_before": n_pts,
            "unique_after": n_pts,
            "boost": 1.0,
            "forced_search": False,
            "max_raster_px_initial": None,
            "max_raster_px_final": None,
        }
    if str(os.environ.get("SPIX_PLOT_AUTO_BOOST_UNIQUE_CENTERS", "1")).strip().lower() in {"0", "false", "no"}:
        return float(scale), int(w), int(h), {
            "applied": False,
            "unique_before": None,
            "unique_after": None,
            "boost": 1.0,
            "forced_search": False,
            "max_raster_px_initial": None,
            "max_raster_px_final": None,
        }

    default_target_unique_fraction = "1.0"
    default_max_boost = "4.0"
    default_force_max_boost = "64.0"
    target_unique_fraction = float(
        np.clip(
            float(
                os.environ.get("SPIX_PLOT_TARGET_UNIQUE_CENTER_FRACTION", default_target_unique_fraction)
            ),
            0.75,
            1.0,
        )
    )
    max_boost = float(
        np.clip(
            float(
                os.environ.get("SPIX_PLOT_MAX_CENTER_BOOST", default_max_boost)
            ),
            1.0,
            8.0,
        )
    )
    force_target_unique = str(
        os.environ.get("SPIX_PLOT_FORCE_TARGET_UNIQUE_CENTERS", "1")
    ).strip().lower() not in {"0", "false", "no"}
    force_max_boost = float(
        np.clip(
            float(
                os.environ.get("SPIX_PLOT_FORCE_MAX_CENTER_BOOST", default_force_max_boost)
            ),
            float(max_boost),
            256.0,
        )
    )

    cx0, cy0 = scale_points_to_canvas(
        pts_arr,
        x_min_eff=float(x_min_eff),
        y_min_eff=float(y_min_eff),
        scale=float(scale),
        pixel_perfect=bool(pixel_perfect),
        w=int(w),
        h=int(h),
    )
    unique_before = int(np.unique(cy0.astype(np.int64) * np.int64(max(1, int(w))) + cx0.astype(np.int64)).size)
    target_unique = int(np.ceil(float(n_pts) * float(target_unique_fraction)))
    if unique_before >= target_unique:
        return float(scale), int(w), int(h), {
            "applied": False,
            "unique_before": unique_before,
            "unique_after": unique_before,
            "boost": 1.0,
            "forced_search": False,
            "max_raster_px_initial": None,
            "max_raster_px_final": None,
        }

    support_fraction_est = float(unique_before) / float(max(1, int(w) * int(h)))
    max_raster_px = resolve_max_raster_pixels(
        env_key="SPIX_SEGMENT_MAX_RASTER_PIXELS",
        fallback_env_key="SPIX_PLOT_MAX_RASTER_PIXELS",
        n_points=n_pts,
        support_fraction=support_fraction_est,
    )
    initial_max_raster_px = int(max_raster_px)
    base_scale = float(scale)
    best_scale = float(scale)
    best_w = int(w)
    best_h = int(h)
    best_unique = int(unique_before)
    missing_unique = max(0, int(target_unique) - int(unique_before))
    success_scale: Optional[float] = None
    success_w: Optional[int] = None
    success_h: Optional[int] = None
    success_unique: Optional[int] = None
    evaluated_canvas_cache: dict[tuple[int, int], tuple[float, int, int, int]] = {}
    forced_search_applied = False

    def _accept_success(
        cand_scale: float,
        cand_scale_eval: float,
        cand_w: int,
        cand_h: int,
        cand_unique: int,
    ) -> None:
        nonlocal success_scale, success_w, success_h, success_unique
        if int(cand_unique) < int(target_unique):
            return
        cand_area = int(cand_w) * int(cand_h)
        if success_scale is None:
            success_scale = float(cand_scale_eval)
            success_w = int(cand_w)
            success_h = int(cand_h)
            success_unique = int(cand_unique)
            return
        prev_area = int(success_w) * int(success_h)
        if (
            cand_area < prev_area
            or (cand_area == prev_area and float(cand_scale_eval) < float(success_scale))
        ):
            success_scale = float(cand_scale_eval)
            success_w = int(cand_w)
            success_h = int(cand_h)
            success_unique = int(cand_unique)

    def _evaluate_scale(cand_scale: float) -> tuple[float, int, int, int] | None:
        cand_scale = float(cand_scale)
        cand_w = max(1, int(np.ceil(float(x_range) * float(cand_scale))))
        cand_h = max(1, int(np.ceil(float(y_range) * float(cand_scale))))
        if int(cand_w) * int(cand_h) > max_raster_px:
            return None
        cached_eval = evaluated_canvas_cache.get((int(cand_w), int(cand_h)), None)
        if cached_eval is not None:
            return cached_eval
        # For a fixed integer canvas (w, h), many nearby scales map to the same
        # canvas. Use the midpoint of that feasible interval so evaluation is
        # stable and does not accidentally penalize a valid canvas by choosing an
        # edge-case scale within the same ceil bucket.
        scale_lo_x = float(max(0, int(cand_w) - 1)) / float(max(float(x_range), 1e-12))
        scale_hi_x = float(int(cand_w)) / float(max(float(x_range), 1e-12))
        scale_lo_y = float(max(0, int(cand_h) - 1)) / float(max(float(y_range), 1e-12))
        scale_hi_y = float(int(cand_h)) / float(max(float(y_range), 1e-12))
        cand_scale_eval_lo = max(scale_lo_x, scale_lo_y)
        cand_scale_eval_hi = min(scale_hi_x, scale_hi_y)
        if cand_scale_eval_hi > cand_scale_eval_lo + 1e-12:
            cand_scale_eval = 0.5 * (float(cand_scale_eval_lo) + float(cand_scale_eval_hi))
        else:
            cand_scale_eval = float(cand_scale)
        cxc, cyc = scale_points_to_canvas(
            pts_arr,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(cand_scale_eval),
            pixel_perfect=bool(pixel_perfect),
            w=int(cand_w),
            h=int(cand_h),
        )
        cand_unique = int(
            np.unique(cyc.astype(np.int64) * np.int64(max(1, int(cand_w))) + cxc.astype(np.int64)).size
        )
        result = (float(cand_scale_eval), int(cand_w), int(cand_h), int(cand_unique))
        evaluated_canvas_cache[(int(cand_w), int(cand_h))] = result
        return result

    max_scale = float(scale) * float(max_boost)
    scale_cap = float(np.sqrt(float(max_raster_px) / float(max(1e-9, float(x_range) * float(y_range)))))
    max_scale = min(float(max_scale), float(scale_cap))
    if max_scale > float(scale) + 1e-9:
        coarse_n = int(max(6, int(os.environ.get("SPIX_PLOT_CENTER_BOOST_COARSE_CANDIDATES", "12"))))
        refine_n = int(max(0, int(os.environ.get("SPIX_PLOT_CENTER_BOOST_REFINE_CANDIDATES", "6"))))
        refine_topk = int(max(1, int(os.environ.get("SPIX_PLOT_CENTER_BOOST_REFINE_TOPK", "3"))))
        local_collision_fraction_threshold = float(
            np.clip(
                float(os.environ.get("SPIX_PLOT_CENTER_BOOST_LOCAL_COLLISION_FRACTION", "0.002")),
                0.0,
                0.1,
            )
        )
        local_max_boost = float(
            np.clip(
                float(os.environ.get("SPIX_PLOT_CENTER_BOOST_LOCAL_MAX_BOOST", "1.2")),
                1.0,
                float(max_boost),
            )
        )
        local_n = int(max(6, int(os.environ.get("SPIX_PLOT_CENTER_BOOST_LOCAL_CANDIDATES", "14"))))
        missing_fraction = float(missing_unique) / float(max(1, n_pts))
        local_search_scale_cap = min(float(max_scale), float(scale) * float(local_max_boost))
        if (
            missing_unique > 0
            and missing_fraction <= local_collision_fraction_threshold
            and local_search_scale_cap > float(scale) + 1e-9
        ):
            candidate_scales = np.geomspace(float(scale) * 1.0005, float(local_search_scale_cap), num=local_n)
        else:
            candidate_scales = np.geomspace(float(scale) * 1.01, float(max_scale), num=coarse_n)
        candidate_scales = np.unique(np.asarray(candidate_scales, dtype=float))
        evaluated: list[tuple[float, int]] = []
        for cand_scale in candidate_scales:
            cand_scale = float(cand_scale)
            evaluated_result = _evaluate_scale(cand_scale)
            if evaluated_result is None:
                continue
            cand_scale_eval, cand_w, cand_h, cand_unique = evaluated_result
            evaluated.append((float(cand_scale), int(cand_unique)))
            if cand_unique > best_unique:
                best_scale = float(cand_scale_eval)
                best_w = int(cand_w)
                best_h = int(cand_h)
                best_unique = int(cand_unique)
            if cand_unique >= target_unique:
                _accept_success(cand_scale, cand_scale_eval, cand_w, cand_h, cand_unique)
        if refine_n > 0 and len(evaluated) >= 2:
            scored_idx = np.argsort(np.asarray([u for _, u in evaluated], dtype=np.int64))[::-1]
            interval_pairs: list[tuple[float, float]] = []
            for rank_idx in scored_idx[: max(1, refine_topk)]:
                idx = int(rank_idx)
                lo_scale = float(scale) * 1.01 if idx <= 0 else float(evaluated[idx - 1][0])
                hi_scale = float(max_scale) if idx >= (len(evaluated) - 1) else float(evaluated[idx + 1][0])
                if hi_scale > lo_scale + 1e-9:
                    interval_pairs.append((lo_scale, hi_scale))
            if success_scale is not None:
                success_candidates = [i for i, (_, uniq) in enumerate(evaluated) if int(uniq) >= int(target_unique)]
                if success_candidates:
                    first_success_idx = int(min(success_candidates))
                    lo_scale = float(scale) * 1.01 if first_success_idx <= 0 else float(evaluated[first_success_idx - 1][0])
                    hi_scale = float(evaluated[first_success_idx][0])
                    if hi_scale > lo_scale + 1e-9:
                        interval_pairs.append((lo_scale, hi_scale))

            seen_intervals: set[tuple[int, int]] = set()
            for lo_scale, hi_scale in interval_pairs:
                key = (int(round(lo_scale * 1e12)), int(round(hi_scale * 1e12)))
                if key in seen_intervals:
                    continue
                seen_intervals.add(key)
                refine_scales = np.linspace(float(lo_scale), float(hi_scale), num=refine_n + 2)[1:-1]
                for cand_scale in np.unique(np.asarray(refine_scales, dtype=float)):
                    evaluated_result = _evaluate_scale(float(cand_scale))
                    if evaluated_result is None:
                        continue
                    cand_scale_eval, cand_w, cand_h, cand_unique = evaluated_result
                    if cand_unique > best_unique:
                        best_scale = float(cand_scale_eval)
                        best_w = int(cand_w)
                        best_h = int(cand_h)
                        best_unique = int(cand_unique)
                    if cand_unique >= target_unique:
                        _accept_success(float(cand_scale), cand_scale_eval, cand_w, cand_h, cand_unique)
        if success_scale is None and missing_unique > 0 and missing_fraction <= local_collision_fraction_threshold:
            fallback_scales = np.geomspace(
                float(scale) * 1.01,
                float(max_scale),
                num=coarse_n,
            )
            for cand_scale in np.unique(np.asarray(fallback_scales, dtype=float)):
                evaluated_result = _evaluate_scale(float(cand_scale))
                if evaluated_result is None:
                    continue
                cand_scale_eval, cand_w, cand_h, cand_unique = evaluated_result
                evaluated.append((float(cand_scale), int(cand_unique)))
                if cand_unique > best_unique:
                    best_scale = float(cand_scale_eval)
                    best_w = int(cand_w)
                    best_h = int(cand_h)
                    best_unique = int(cand_unique)
                if cand_unique >= target_unique:
                    _accept_success(float(cand_scale), cand_scale_eval, cand_w, cand_h, cand_unique)
        if success_scale is not None and success_unique is not None and success_w is not None and success_h is not None:
            best_scale = float(success_scale)
            best_w = int(success_w)
            best_h = int(success_h)
            best_unique = int(success_unique)

    if force_target_unique and success_scale is None and best_unique < target_unique:
        forced_search_applied = True
        forced_scale_hi = float(scale) * float(force_max_boost)
        forced_scale_hi = max(float(max_scale), float(forced_scale_hi))
        force_candidates = int(
            max(8, int(os.environ.get("SPIX_PLOT_FORCE_CENTER_BOOST_CANDIDATES", "20")))
        )
        force_rounds = int(
            max(1, int(os.environ.get("SPIX_PLOT_FORCE_CENTER_BOOST_ROUNDS", "8")))
        )
        search_lo = float(max(float(scale) * 1.01, float(max_scale) * 1.0005))
        search_hi = float(max_scale)
        area_units = float(max(1e-9, float(x_range) * float(y_range)))

        if logger is not None:
            msg = f"{context}: " if context else ""
            logger.warning(
                "%starget unique centers not reached under current raster cap; forcing larger canvas search "
                "(target=%d achieved=%d initial_max_raster_px=%d force_max_boost=%.3f).",
                msg,
                int(target_unique),
                int(best_unique),
                int(initial_max_raster_px),
                float(force_max_boost),
            )

        for _ in range(force_rounds):
            if success_scale is not None or search_hi >= forced_scale_hi - 1e-9:
                break
            next_hi = min(
                float(forced_scale_hi),
                max(float(search_hi) * 1.5, float(scale) * 1.05),
            )
            if next_hi <= search_lo + 1e-9:
                break
            required_px = int(np.ceil(float(area_units) * float(next_hi) * float(next_hi)))
            if required_px > int(max_raster_px):
                max_raster_px = int(required_px)
            forced_scales = np.geomspace(float(search_lo), float(next_hi), num=force_candidates)
            for cand_scale in np.unique(np.asarray(forced_scales, dtype=float)):
                evaluated_result = _evaluate_scale(float(cand_scale))
                if evaluated_result is None:
                    continue
                cand_scale_eval, cand_w, cand_h, cand_unique = evaluated_result
                if cand_unique > best_unique:
                    best_scale = float(cand_scale_eval)
                    best_w = int(cand_w)
                    best_h = int(cand_h)
                    best_unique = int(cand_unique)
                if cand_unique >= target_unique:
                    _accept_success(float(cand_scale), cand_scale_eval, cand_w, cand_h, cand_unique)
            search_lo = max(float(next_hi) * 1.0005, float(scale) * 1.01)
            search_hi = float(next_hi)

        if success_scale is not None and success_unique is not None and success_w is not None and success_h is not None:
            best_scale = float(success_scale)
            best_w = int(success_w)
            best_h = int(success_h)
            best_unique = int(success_unique)

    applied = best_unique > unique_before
    if applied and logger is not None:
        msg = f"{context}: " if context else ""
        logger.info(
            "%sboosted raster canvas for unique centers: %dx%d -> %dx%d | unique centers %d -> %d",
            msg,
            int(w),
            int(h),
            int(best_w),
            int(best_h),
            int(unique_before),
            int(best_unique),
        )
        if forced_search_applied and int(max_raster_px) > int(initial_max_raster_px):
            logger.info(
                "%sforced unique-center canvas expansion raised raster cap: %d -> %d pixels.",
                msg,
                int(initial_max_raster_px),
                int(max_raster_px),
            )
        if best_unique < target_unique:
            logger.warning(
                "%starget unique centers not fully reached after canvas boost search: target=%d achieved=%d max_boost=%.3f max_raster_px=%d",
                msg,
                int(target_unique),
                int(best_unique),
                float(max_boost),
                int(max_raster_px),
            )
    return float(best_scale), int(best_w), int(best_h), {
        "applied": bool(applied),
        "unique_before": int(unique_before),
        "unique_after": int(best_unique),
        "boost": float(best_scale / max(base_scale, 1e-9)),
        "forced_search": bool(forced_search_applied),
        "max_raster_px_initial": int(initial_max_raster_px),
        "max_raster_px_final": int(max_raster_px),
    }


def resolve_raster_canvas(
    *,
    pts: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    figsize,
    fig_dpi,
    imshow_tile_size: Optional[float],
    imshow_scale_factor: float,
    imshow_tile_size_mode: Optional[str],
    imshow_tile_size_quantile: Optional[float],
    imshow_tile_size_rounding: Optional[str],
    imshow_tile_size_shrink: float,
    pixel_perfect: bool,
    n_points: int,
    logger=None,
    context: str = "",
    default_figsize: Tuple[float, float] = (10.0, 10.0),
    persistent_store=None,
) -> dict:
    user_figsize_given = figsize is not None
    user_dpi_given = fig_dpi is not None
    auto_sized = (not user_figsize_given) or (not user_dpi_given)
    cache_key = (
        _RASTER_CANVAS_CACHE_VERSION,
        _points_signature(pts),
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        None if figsize is None else tuple(float(v) for v in figsize),
        None if fig_dpi is None else int(fig_dpi),
        None if imshow_tile_size is None else float(imshow_tile_size),
        float(imshow_scale_factor),
        None if imshow_tile_size_mode is None else str(imshow_tile_size_mode),
        None if imshow_tile_size_quantile is None else float(imshow_tile_size_quantile),
        None if imshow_tile_size_rounding is None else str(imshow_tile_size_rounding),
        float(imshow_tile_size_shrink),
        bool(pixel_perfect),
        int(n_points),
        str(context or ""),
        tuple(float(v) for v in default_figsize),
    )
    cached = _lru_cache_get(_RASTER_CANVAS_CACHE, cache_key)
    if cached is not None:
        return dict(cached)
    persistent_cached = _persistent_canvas_cache_get(persistent_store, cache_key)
    if persistent_cached is not None:
        _lru_cache_put(_RASTER_CANVAS_CACHE, cache_key, dict(persistent_cached), max_items=_RASTER_CANVAS_CACHE_MAX)
        return dict(persistent_cached)
    shrink_eff = 1.0 if pixel_perfect else float(imshow_tile_size_shrink)
    context_str = str(context or "")
    is_pixel_boundary_context = "pixel-boundary" in context_str.lower()
    is_display_family = (
        (
            ("image_plot" in context_str)
            or ("overlay_segment_boundaries_on_display" in context_str)
        )
        and ("image_plot_slic_segmentation" not in context_str)
        and ("cache_embedding_image" not in context_str)
        and ("rebuild_cached_image_from_obsm" not in context_str)
    )
    display_scale_factor = float(imshow_scale_factor)
    scale_factor_for_estimate = 1.0 if is_display_family else float(imshow_scale_factor)

    s_data_units = estimate_tile_size_units(
        np.asarray(pts),
        imshow_tile_size,
        scale_factor_for_estimate,
        imshow_tile_size_mode,
        imshow_tile_size_quantile,
        shrink_eff,
        logger=logger,
    )
    # Respect explicit tile-size controls. The density cap is useful for auto
    # estimation, but if the caller explicitly specifies imshow_tile_size or a
    # non-default imshow_scale_factor they expect visible changes in tile size.
    if (imshow_tile_size is None) and (float(scale_factor_for_estimate) == 1.0):
        s_data_units = cap_tile_size_by_render_extent(
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
    elif logger is not None and context:
        logger.info(
            "%sexplicit imshow tile sizing enabled: s_data_units=%.3f (tile=%s scale_factor=%.3f)",
            f"{context}: ",
            float(s_data_units),
            "auto" if imshow_tile_size is None else f"{float(imshow_tile_size):.3f}",
            float(scale_factor_for_estimate),
        )

    x_min_eff, x_max_eff, y_min_eff, y_max_eff = effective_extent(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        s_data_units=float(s_data_units),
        pixel_perfect=bool(pixel_perfect),
    )
    x_range = (x_max_eff - x_min_eff) or 1.0
    y_range = (y_max_eff - y_min_eff) or 1.0

    # Raster canvas geometry should depend on the data/tile density, not on the
    # user's output figure settings. Figure figsize/dpi are applied later only
    # to the rendered display.
    canvas_figsize, canvas_fig_dpi = resolve_figsize_dpi_for_tiles(
        figsize=None,
        fig_dpi=None,
        x_range=float(x_range),
        y_range=float(y_range),
        s_data_units=float(s_data_units),
        pixel_perfect=bool(pixel_perfect),
        n_points=int(n_points),
        logger=logger,
    )
    if canvas_figsize is None:
        canvas_figsize = tuple(default_figsize)
    dpi_in = int(canvas_fig_dpi) if canvas_fig_dpi is not None else 100
    w_pixels = int(np.ceil(float(canvas_figsize[0]) * float(dpi_in)))
    h_pixels = int(np.ceil(float(canvas_figsize[1]) * float(dpi_in)))

    scale_raw = scale_raw_from_canvas(
        w_pixels=int(w_pixels),
        h_pixels=int(h_pixels),
        x_range=float(x_range),
        y_range=float(y_range),
        pixel_perfect=bool(pixel_perfect),
        auto_sized=True,
    )
    if pixel_perfect and s_data_units > 0:
        pitch_px = max(1, int(np.round(float(s_data_units) * float(scale_raw))))
        scale = float(pitch_px) / float(s_data_units)
        tile_px = int(pitch_px)
    else:
        scale = float(scale_raw)
        tile_px = round_tile_size_px(float(s_data_units), float(scale), imshow_tile_size_rounding)

    w = max(1, int(np.ceil(float(x_range) * float(scale))))
    h = max(1, int(np.ceil(float(y_range) * float(scale))))
    scale, w, h, canvas_boost = maybe_boost_canvas_for_unique_centers(
        pts=np.asarray(pts),
        x_min_eff=float(x_min_eff),
        y_min_eff=float(y_min_eff),
        x_range=float(x_range),
        y_range=float(y_range),
        scale=float(scale),
        w=int(w),
        h=int(h),
        pixel_perfect=bool(pixel_perfect),
        tile_px=int(tile_px),
        logger=logger,
        context=str(context or ""),
    )

    explicit_tile_control = (imshow_tile_size is not None)
    if is_display_family and (not is_pixel_boundary_context) and (not explicit_tile_control) and (not bool(pixel_perfect)) and int(tile_px) == 1:
        support_fraction_est = float(n_points) / float(max(1, int(w) * int(h)))
        target_support_fraction = float(
            np.clip(
                float(os.environ.get("SPIX_PLOT_DISPLAY_MIN_SUPPORT_FRACTION", "0.22")),
                0.05,
                0.9,
            )
        )
        if support_fraction_est > 0 and support_fraction_est < target_support_fraction:
            tile_px_target = int(
                np.ceil(np.sqrt(float(target_support_fraction) / float(max(support_fraction_est, 1e-9))))
            )
            tile_px_target = int(
                np.clip(
                    tile_px_target,
                    1,
                    int(max(1, int(os.environ.get("SPIX_PLOT_DISPLAY_MAX_TILE_PX", "4")))),
                )
            )
            if tile_px_target > int(tile_px):
                tile_px = int(tile_px_target)
                if logger is not None:
                    logger.info(
                        "%sauto-increased display tile size: tile_px %d -> %d (support_est=%.4f target=%.4f)",
                        f"{context}: " if context else "",
                        1,
                        int(tile_px),
                        float(support_fraction_est),
                        float(target_support_fraction),
                    )
    elif is_display_family and (not is_pixel_boundary_context) and explicit_tile_control and logger is not None and context:
        logger.info(
            "%sdisplay auto tile grow skipped due to explicit absolute tile size (tile=%s)",
            f"{context}: ",
            "auto" if imshow_tile_size is None else f"{float(imshow_tile_size):.3f}",
        )

    if is_display_family and (not is_pixel_boundary_context) and float(display_scale_factor) != 1.0:
        tile_px_before = int(tile_px)
        tile_px = max(1, int(np.ceil(float(tile_px) * float(display_scale_factor))))
        if logger is not None and context:
            logger.info(
                "%sdisplay tile scale applied post-auto: tile_px %d -> %d (scale_factor=%.3f)",
                f"{context}: ",
                int(tile_px_before),
                int(tile_px),
                float(display_scale_factor),
            )

    fig_dpi_resolved = int(fig_dpi) if fig_dpi is not None else int(dpi_in)
    if figsize is not None:
        figsize_resolved = tuple(figsize)
    else:
        dpi_use = max(1, int(fig_dpi_resolved))
        figsize_resolved = (float(w) / float(dpi_use), float(h) / float(dpi_use))

    out = {
        "figsize": tuple(figsize_resolved),
        "fig_dpi": fig_dpi_resolved,
        "dpi_in": int(dpi_in),
        "auto_sized": bool(auto_sized),
        "s_data_units": float(s_data_units),
        "x_min_eff": float(x_min_eff),
        "x_max_eff": float(x_max_eff),
        "y_min_eff": float(y_min_eff),
        "y_max_eff": float(y_max_eff),
        "x_range": float(x_range),
        "y_range": float(y_range),
        "scale": float(scale),
        "tile_px": int(tile_px),
        "w": int(w),
        "h": int(h),
        "canvas_boost": canvas_boost,
    }
    _lru_cache_put(_RASTER_CANVAS_CACHE, cache_key, dict(out), max_items=_RASTER_CANVAS_CACHE_MAX)
    _persistent_canvas_cache_put(persistent_store, cache_key, dict(out))
    return out


def tile_pixel_offsets(tile_px: int, pixel_shape: str = "square") -> Tuple[np.ndarray, np.ndarray]:
    s = max(1, int(tile_px))
    center_off = int(s // 2)
    if pixel_shape == "circle":
        yy, xx = np.ogrid[:s, :s]
        center = (s - 1) / 2
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= (s / 2) ** 2
        dy, dx = np.nonzero(mask)
        dx = dx.astype(np.int32) - center_off
        dy = dy.astype(np.int32) - center_off
        return dx, dy

    rng = np.arange(s, dtype=np.int32)
    gx, gy = np.meshgrid(rng, rng)
    dx = (gx - center_off).ravel()
    dy = (gy - center_off).ravel()
    return dx, dy


def normalize_gap_collapse_axis(axis) -> Optional[str]:
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


def _collapse_empty_runs_1d(counts: np.ndarray, *, max_gap_run_px: int) -> tuple[np.ndarray, int, int]:
    keep = np.ones(int(counts.size), dtype=bool)
    empty = np.asarray(counts, dtype=np.int64) <= 0
    removed = 0
    removed_runs = 0
    i = 0
    n = int(empty.size)
    max_gap = int(max(0, int(max_gap_run_px)))
    while i < n:
        if not bool(empty[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(empty[j]):
            j += 1
        run_len = int(j - i)
        if run_len <= max_gap:
            keep[i:j] = False
            removed += run_len
            removed_runs += 1
        i = j
    return keep, int(removed), int(removed_runs)


def collapse_empty_center_gaps(
    *,
    cx: np.ndarray,
    cy: np.ndarray,
    w: int,
    h: int,
    axis=None,
    max_gap_run_px: int = 1,
    logger=None,
    context: str = "",
) -> tuple[np.ndarray, np.ndarray, int, int, dict]:
    axis_norm = normalize_gap_collapse_axis(axis)
    info = {
        "applied": False,
        "axis": axis_norm,
        "max_gap_run_px": int(max(0, int(max_gap_run_px))),
        "rows_removed": 0,
        "cols_removed": 0,
        "row_runs_removed": 0,
        "col_runs_removed": 0,
        "w_before": int(w),
        "h_before": int(h),
        "w_after": int(w),
        "h_after": int(h),
    }
    cx_arr = np.asarray(cx, dtype=np.int32).ravel()
    cy_arr = np.asarray(cy, dtype=np.int32).ravel()
    if axis_norm is None or cx_arr.size == 0 or cy_arr.size == 0 or int(max_gap_run_px) <= 0:
        return cx_arr, cy_arr, int(w), int(h), info

    w_new = int(w)
    h_new = int(h)
    cx_new = cx_arr.copy()
    cy_new = cy_arr.copy()

    if axis_norm in {"y", "both"} and int(h_new) > 0:
        row_counts = np.bincount(np.clip(cy_new, 0, max(0, h_new - 1)), minlength=int(h_new))
        keep_rows, rows_removed, row_runs_removed = _collapse_empty_runs_1d(
            row_counts,
            max_gap_run_px=int(max_gap_run_px),
        )
        if int(rows_removed) > 0:
            row_map = np.cumsum(keep_rows, dtype=np.int64) - 1
            cy_new = row_map[np.clip(cy_new, 0, max(0, h_new - 1))].astype(np.int32, copy=False)
            h_new = int(keep_rows.sum())
            info["rows_removed"] = int(rows_removed)
            info["row_runs_removed"] = int(row_runs_removed)

    if axis_norm in {"x", "both"} and int(w_new) > 0:
        col_counts = np.bincount(np.clip(cx_new, 0, max(0, w_new - 1)), minlength=int(w_new))
        keep_cols, cols_removed, col_runs_removed = _collapse_empty_runs_1d(
            col_counts,
            max_gap_run_px=int(max_gap_run_px),
        )
        if int(cols_removed) > 0:
            col_map = np.cumsum(keep_cols, dtype=np.int64) - 1
            cx_new = col_map[np.clip(cx_new, 0, max(0, w_new - 1))].astype(np.int32, copy=False)
            w_new = int(keep_cols.sum())
            info["cols_removed"] = int(cols_removed)
            info["col_runs_removed"] = int(col_runs_removed)

    info["w_after"] = int(w_new)
    info["h_after"] = int(h_new)
    info["applied"] = bool(
        int(info["rows_removed"]) > 0 or int(info["cols_removed"]) > 0
    )
    if info["applied"] and logger is not None:
        msg = f"{context}: " if context else ""
        logger.info(
            "%sApplied gap collapse axis=%s max_gap_run_px=%d | rows_removed=%d cols_removed=%d | canvas %dx%d -> %dx%d",
            msg,
            str(axis_norm),
            int(max_gap_run_px),
            int(info["rows_removed"]),
            int(info["cols_removed"]),
            int(w),
            int(h),
            int(w_new),
            int(h_new),
        )
    return cx_new, cy_new, int(w_new), int(h_new), info


def _binary_disk(radius: int) -> np.ndarray:
    radius = int(max(0, radius))
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return ((xx * xx + yy * yy) <= (radius * radius)).astype(bool)


def _infer_regular_grid_step_1d(values: np.ndarray, *, q: float = 0.05, max_samples: int = 200_000) -> Optional[int]:
    vals = np.asarray(values, dtype=float).ravel()
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
    uniq = np.unique(vals_i)
    if uniq.size < 2:
        return None
    uniq.sort()
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    try:
        qq = float(q)
        if not (0.0 < qq <= 1.0):
            qq = 0.05
    except Exception:
        qq = 0.05
    step = int(np.quantile(diffs, qq))
    if step <= 0:
        step = int(np.min(diffs))
    return int(step) if step > 0 else None


def _infer_grid_support_mask(
    *,
    occupancy_mask: np.ndarray,
    seed_x: np.ndarray,
    seed_y: np.ndarray,
    closing_radius: int,
    fill_holes: bool,
) -> Optional[dict]:
    """Infer a dense support mask from lattice-like seed centers.

    This is more stable than applying morphology directly to expanded tile
    pixels, especially for sparse `origin=True` coordinates. We synthesize
    support on the center lattice, then dilate it back to an approximate tile
    footprint in pixel space.
    """
    occ = np.asarray(occupancy_mask, dtype=bool)
    sx = np.asarray(seed_x, dtype=float).ravel()
    sy = np.asarray(seed_y, dtype=float).ravel()
    valid = (
        np.isfinite(sx)
        & np.isfinite(sy)
        & (sx >= 0)
        & (sy >= 0)
        & (sx < occ.shape[1])
        & (sy < occ.shape[0])
    )
    if not np.any(valid):
        return None

    sx = np.rint(sx[valid]).astype(np.int64, copy=False)
    sy = np.rint(sy[valid]).astype(np.int64, copy=False)
    w_occ = int(occ.shape[1])
    flat_centers = np.unique(sy * np.int64(w_occ) + sx)
    n_centers = int(flat_centers.size)
    if n_centers < 4:
        return None

    center_x_all = (flat_centers % np.int64(w_occ)).astype(np.int64, copy=False)
    center_y_all = (flat_centers // np.int64(w_occ)).astype(np.int64, copy=False)

    step_x = _infer_regular_grid_step_1d(center_x_all)
    step_y = _infer_regular_grid_step_1d(center_y_all)
    if step_x is None and step_y is None:
        return None
    if step_x is None:
        step_x = step_y
    if step_y is None:
        step_y = step_x
    step_x = int(max(1, step_x))
    step_y = int(max(1, step_y))

    x0 = int(center_x_all.min())
    y0 = int(center_y_all.min())
    xi = np.floor((center_x_all - x0) / float(step_x) + 0.5).astype(np.int64, copy=False)
    yi = np.floor((center_y_all - y0) / float(step_y) + 0.5).astype(np.int64, copy=False)
    w0 = int(xi.max(initial=0)) + 1
    h0 = int(yi.max(initial=0)) + 1
    total_cells = int(w0) * int(h0)
    if total_cells <= 0 or total_cells > max(int(occ.size) * 4, 2_000_000):
        return None

    observed = np.zeros((h0, w0), dtype=bool)
    observed[yi, xi] = True
    observed_cells = int(observed.sum())
    if observed_cells <= 0:
        return None

    # Estimate the rendered tile footprint from occupied pixels per unique seed.
    tile_area_est = float(max(1.0, float(occ.sum()) / float(max(1, n_centers))))
    tile_radius = int(max(0, np.ceil(np.sqrt(tile_area_est) / 2.0) - 1))

    # Interpret closing_radius in raster pixels, but account for the existing
    # tile footprint when mapping that radius back to the center lattice.
    # Without this, `closing_radius=5` on ~7 px tiles collapses to grid_radius=1
    # and under-fills the obvious one-cell gaps users expect to close.
    step_eff = float(max(1, min(step_x, step_y)))
    effective_radius_px = float(max(0, closing_radius)) + float(max(0, tile_radius))
    grid_radius = int(max(1, np.ceil(effective_radius_px / step_eff)))
    support = observed.copy()
    if grid_radius > 0:
        counts = ndi.convolve(observed.astype(np.int16), _binary_disk(grid_radius).astype(np.int16), mode="constant", cval=0)
        min_support_neighbors = 2 if grid_radius <= 2 else 3
        support = counts >= int(min_support_neighbors)
        support |= observed
        close_radius = int(max(1, grid_radius))
        if close_radius > 0:
            support = ndi.binary_closing(support, structure=_binary_disk(close_radius))

    # `runtime_fill_from_boundary=True` should recover the interior support
    # enclosed by the observed boundary even when `fill_holes=False`.
    support = ndi.binary_fill_holes(support)
    support |= observed

    gy, gx = np.nonzero(support)
    center_x = x0 + gx.astype(np.int64, copy=False) * int(step_x)
    center_y = y0 + gy.astype(np.int64, copy=False) * int(step_y)
    inside = (
        (center_x >= 0)
        & (center_y >= 0)
        & (center_x < occ.shape[1])
        & (center_y < occ.shape[0])
    )
    center_x = center_x[inside]
    center_y = center_y[inside]
    if center_x.size == 0:
        return None

    center_mask = np.zeros_like(occ, dtype=bool)
    center_mask[center_y, center_x] = True

    if tile_radius > 0:
        support_mask = ndi.binary_dilation(center_mask, structure=_binary_disk(tile_radius))
    else:
        support_mask = center_mask

    if tile_radius > 0:
        support_mask = ndi.binary_closing(support_mask, structure=_binary_disk(tile_radius))

    if fill_holes:
        support_mask = ndi.binary_fill_holes(support_mask)
    support_mask |= occ

    support_pixels = int(support_mask.sum())
    if support_pixels <= int(occ.sum()):
        return None
    if support_pixels >= int(occ.size):
        return None

    return {
        "support_mask": support_mask,
        "mode": "grid_support",
        "grid_step_x": int(step_x),
        "grid_step_y": int(step_y),
        "grid_radius": int(grid_radius),
        "tile_radius": int(tile_radius),
        "support_centers": int(center_x.size),
    }


def fill_raster_from_boundary(
    *,
    img: np.ndarray,
    occupancy_mask: np.ndarray,
    seed_x: np.ndarray,
    seed_y: np.ndarray,
    seed_values: np.ndarray,
    closing_radius: int = 1,
    external_radius: int = 0,
    fill_holes: bool = True,
    use_observed_pixels: bool = False,
    persistent_store=None,
    logger=None,
    context: str = "",
) -> dict:
    """Morphologically fill raster support from observed tiles without resnapping points.

    The image is updated in-place by assigning each newly filled pixel the value
    of its nearest observed tile center in raster space. The occupancy/support
    mask is also expanded in-place so downstream masked segmentation sees the
    filled support instead of only the original seed pixels.
    """
    occ = np.asarray(occupancy_mask, dtype=bool)
    if occ.ndim != 2:
        raise ValueError("occupancy_mask must be a 2D boolean array.")
    if not np.any(occ):
        return {
            "applied": False,
            "filled_pixels": 0,
            "support_pixels": 0,
            "closing_radius": int(max(0, closing_radius)),
            "external_radius": int(max(0, external_radius)),
            "fill_holes": bool(fill_holes),
        }

    radius = int(max(0, closing_radius))
    external = int(max(0, external_radius))
    fill_cache_key = (
        _RASTER_FILL_CACHE_VERSION,
        _occupancy_signature(occ),
        _centers_signature(seed_x, seed_y),
        int(occ.shape[0]),
        int(occ.shape[1]),
        int(radius),
        int(external),
        int(bool(fill_holes)),
        int(bool(use_observed_pixels)),
    )
    fill_cache = _lru_cache_get(_FILL_ASSIGNMENT_CACHE, fill_cache_key)
    if fill_cache is None:
        fill_cache = _persistent_fill_cache_get(persistent_store, fill_cache_key)
        if fill_cache is not None:
            _lru_cache_put(_FILL_ASSIGNMENT_CACHE, fill_cache_key, dict(fill_cache), max_items=_FILL_ASSIGNMENT_CACHE_MAX)
    support_info = None
    support_mask = occ.copy()
    fill_cache_hit = False
    if isinstance(fill_cache, dict):
        cached_support_mask = fill_cache.get("support_mask", None)
        if cached_support_mask is not None:
            support_mask = np.asarray(cached_support_mask, dtype=bool)
            support_info = fill_cache.get("support_info", None)
            fill_cache_hit = True
    if not fill_cache_hit:
        if radius > 0:
            support_info = _infer_grid_support_mask(
                occupancy_mask=occ,
                seed_x=seed_x,
                seed_y=seed_y,
                closing_radius=radius,
                fill_holes=bool(fill_holes),
            )
            if support_info is not None:
                support_mask = np.asarray(support_info["support_mask"], dtype=bool)
            else:
                support_mask = ndi.binary_closing(support_mask, structure=_binary_disk(radius))
                support_mask = ndi.binary_fill_holes(support_mask)
        elif fill_holes:
            support_mask = ndi.binary_fill_holes(support_mask)
        support_mask |= occ
        if external > 0:
            support_mask = ndi.binary_dilation(support_mask, structure=_binary_disk(external))
            support_mask |= occ

    fill_mask = support_mask & ~occ
    filled_pixels = int(fill_mask.sum())
    assignment_cache_hit = False
    fill_dst_flat = None
    fill_src_flat = None
    if filled_pixels > 0:
        vals_valid = None
        if isinstance(fill_cache, dict):
            cached_dst = fill_cache.get("fill_dst_flat", None)
            cached_src = fill_cache.get("fill_src_flat", None)
            if cached_dst is not None and cached_src is not None:
                fill_dst_flat = np.asarray(cached_dst, dtype=np.int64).ravel()
                fill_src_flat = np.asarray(cached_src, dtype=np.int64).ravel()
                if fill_dst_flat.size == fill_src_flat.size == filled_pixels:
                    flat_img = img.reshape((-1, img.shape[-1]))
                    flat_img[fill_dst_flat] = flat_img[fill_src_flat]
                    assignment_cache_hit = True
                else:
                    fill_dst_flat = None
                    fill_src_flat = None
        if (not assignment_cache_hit) and bool(use_observed_pixels):
            sy_obs, sx_obs = np.nonzero(occ)
            if sy_obs.size > 0:
                seed_xy = np.empty((sx_obs.size, 2), dtype=np.float32)
                seed_xy[:, 0] = sx_obs.astype(np.float32, copy=False)
                seed_xy[:, 1] = sy_obs.astype(np.float32, copy=False)
                vals_valid = np.asarray(img[sy_obs, sx_obs])

        if (not assignment_cache_hit) and vals_valid is None:
            sx = np.asarray(seed_x, dtype=np.int32).ravel()
            sy = np.asarray(seed_y, dtype=np.int32).ravel()
            vals = np.asarray(seed_values)
            valid = (
                np.isfinite(sx)
                & np.isfinite(sy)
                & (sx >= 0)
                & (sy >= 0)
                & (sx < occ.shape[1])
                & (sy < occ.shape[0])
            )
            if np.any(valid):
                valid_idx = np.flatnonzero(valid).astype(np.int64, copy=False)
                seed_xy = np.empty((valid_idx.size, 2), dtype=np.float32)
                seed_xy[:, 0] = sx[valid_idx]
                seed_xy[:, 1] = sy[valid_idx]
                vals_valid = vals[valid_idx]

        if (not assignment_cache_hit) and vals_valid is not None:
            tree = KDTree(seed_xy)

            try:
                workers = int(os.environ.get("SPIX_PLOT_FILL_QUERY_WORKERS", "-1"))
            except Exception:
                workers = -1
            query_chunk = int(max(1, int(os.environ.get("SPIX_PLOT_FILL_QUERY_CHUNK", "2000000"))))
            row_block = int(max(1, int(os.environ.get("SPIX_PLOT_FILL_ROW_BLOCK", "512"))))
            cache_assignments = bool(use_observed_pixels)
            max_cache_pixels = int(max(0, int(os.environ.get("SPIX_PLOT_FILL_CACHE_MAX_PIXELS", "5000000"))))
            collect_assignment = cache_assignments and filled_pixels <= max_cache_pixels
            fill_dst_chunks: list[np.ndarray] = []
            fill_src_chunks: list[np.ndarray] = []

            for row_start in range(0, fill_mask.shape[0], row_block):
                row_end = min(fill_mask.shape[0], row_start + row_block)
                submask = fill_mask[row_start:row_end]
                fy_local, fx_local = np.nonzero(submask)
                if fy_local.size == 0:
                    continue
                fy_block = (fy_local + row_start).astype(np.int32, copy=False)
                fx_block = fx_local.astype(np.int32, copy=False)

                for start in range(0, fy_block.size, query_chunk):
                    end = min(fy_block.size, start + query_chunk)
                    q = np.empty((end - start, 2), dtype=np.float32)
                    q[:, 0] = fx_block[start:end]
                    q[:, 1] = fy_block[start:end]
                    _, nn_idx = tree.query(q, k=1, workers=int(workers))
                    nn_idx = np.asarray(nn_idx, dtype=np.int64)
                    img[fy_block[start:end], fx_block[start:end]] = vals_valid[nn_idx]
                    if collect_assignment:
                        dst_flat = fy_block[start:end].astype(np.int64, copy=False) * np.int64(occ.shape[1]) + fx_block[start:end].astype(np.int64, copy=False)
                        src_flat = sy_obs[nn_idx].astype(np.int64, copy=False) * np.int64(occ.shape[1]) + sx_obs[nn_idx].astype(np.int64, copy=False)
                        fill_dst_chunks.append(dst_flat)
                        fill_src_chunks.append(src_flat)
            if collect_assignment and fill_dst_chunks and fill_src_chunks:
                fill_dst_flat = np.concatenate(fill_dst_chunks).astype(np.int32, copy=False)
                fill_src_flat = np.concatenate(fill_src_chunks).astype(np.int32, copy=False)
        else:
            filled_pixels = 0

    if not fill_cache_hit:
        fill_cache_payload = {
            "support_mask": np.asarray(support_mask, dtype=bool, copy=True),
            "support_info": None if support_info is None else dict(support_info),
        }
        if filled_pixels > 0 and fill_dst_flat is not None and fill_src_flat is not None:
            fill_cache_payload["fill_dst_flat"] = np.asarray(fill_dst_flat, dtype=np.int32, copy=False)
            fill_cache_payload["fill_src_flat"] = np.asarray(fill_src_flat, dtype=np.int32, copy=False)
        _lru_cache_put(_FILL_ASSIGNMENT_CACHE, fill_cache_key, dict(fill_cache_payload), max_items=_FILL_ASSIGNMENT_CACHE_MAX)
        _persistent_fill_cache_put(persistent_store, fill_cache_key, dict(fill_cache_payload))

    # Propagate the filled support back to the caller's occupancy mask so
    # downstream mask-based logic uses the morphologically completed support.
    try:
        occupancy_mask[...] = support_mask
    except Exception:
        pass

    if logger is not None and filled_pixels > 0:
        msg = f"{context}: " if context else ""
        if fill_cache_hit:
            logger.info(
                "%sReused raster fill cache%s; filled %d pixels.",
                msg,
                " with assignment map" if assignment_cache_hit else "",
                int(filled_pixels),
            )
        if support_info is not None:
            logger.info(
                "%sApplied raster boundary fill with closing_radius=%d, fill_holes=%s using %s (step=%dx%d, grid_radius=%d, tile_radius=%d); filled %d pixels.",
                msg,
                int(radius),
                bool(fill_holes),
                str(support_info.get("mode", "grid_support")),
                int(support_info.get("grid_step_x", 1)),
                int(support_info.get("grid_step_y", 1)),
                int(support_info.get("grid_radius", 0)),
                int(support_info.get("tile_radius", 0)),
                int(filled_pixels),
            )
        else:
            logger.info(
                "%sApplied raster boundary fill with closing_radius=%d, fill_holes=%s; filled %d pixels.",
                msg,
                int(radius),
                bool(fill_holes),
                int(filled_pixels),
            )

    return {
        "applied": bool(filled_pixels > 0),
        "filled_pixels": int(filled_pixels),
        "support_pixels": int(support_mask.sum()),
        "closing_radius": int(radius),
        "external_radius": int(external),
        "fill_holes": bool(fill_holes),
        "mode": str((support_info or {}).get("mode", "pixel")),
        "cache_hit": bool(fill_cache_hit),
        "assignment_cache_hit": bool(assignment_cache_hit),
    }


def _collision_search_offsets(max_radius: int) -> list[tuple[int, int]]:
    radius = int(max(0, max_radius))
    if radius <= 0:
        return []
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            offsets.append((int(dx), int(dy)))
    offsets.sort(key=lambda xy: (xy[0] * xy[0] + xy[1] * xy[1], abs(xy[1]), abs(xy[0]), xy[1], xy[0]))
    return offsets


def normalize_collision_mode(resolve_center_collisions) -> str:
    if isinstance(resolve_center_collisions, str):
        mode = str(resolve_center_collisions).strip().lower()
        if mode in {"auto"}:
            return "auto"
        if mode in {"1", "true", "yes", "on", "always"}:
            return "always"
        if mode in {"0", "false", "no", "off", "none"}:
            return "never"
    return "always" if bool(resolve_center_collisions) else "never"


def summarize_center_collisions(
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    w: int,
    h: int,
) -> dict:
    cx0 = np.asarray(cx, dtype=np.int32).ravel()
    cy0 = np.asarray(cy, dtype=np.int32).ravel()
    n_tiles = int(cx0.size)
    if cy0.size != n_tiles:
        raise ValueError("cx and cy must have the same length.")
    if n_tiles == 0:
        return {
            "n_tiles": 0,
            "n_unique_centers": 0,
            "n_collided_tiles": 0,
            "collision_fraction": 0.0,
            "max_stack": 0,
            "canvas_pixels": int(max(1, w) * max(1, h)),
            "support_fraction": 0.0,
        }

    w_i = int(max(1, w))
    h_i = int(max(1, h))
    cx0 = np.clip(cx0, 0, max(0, w_i - 1))
    cy0 = np.clip(cy0, 0, max(0, h_i - 1))
    flat = cy0.astype(np.int64, copy=False) * np.int64(w_i) + cx0.astype(np.int64, copy=False)
    _, counts = np.unique(flat, return_counts=True)
    unique_centers = int(counts.size)
    collided_tiles = int(n_tiles - unique_centers)
    canvas_pixels = int(w_i * h_i)
    return {
        "n_tiles": int(n_tiles),
        "n_unique_centers": int(unique_centers),
        "n_collided_tiles": int(collided_tiles),
        "collision_fraction": float(collided_tiles / max(1, n_tiles)),
        "max_stack": int(counts.max()) if counts.size else 1,
        "canvas_pixels": int(canvas_pixels),
        "support_fraction": float(unique_centers / max(1, canvas_pixels)),
    }


def should_resolve_center_collisions(
    resolve_center_collisions,
    *,
    cx: np.ndarray,
    cy: np.ndarray,
    w: int,
    h: int,
    tile_px: int,
    pixel_perfect: bool,
    soft_enabled: bool,
    logger=None,
    context: str = "",
) -> tuple[bool, dict | None, str]:
    mode = normalize_collision_mode(resolve_center_collisions)
    if mode == "never":
        return False, None, mode
    if bool(soft_enabled):
        if logger is not None:
            logger.info("%s: ignoring resolve_center_collisions because soft_rasterization=True.", context)
        return False, None, mode

    collision_stats = summarize_center_collisions(cx, cy, w=int(w), h=int(h))
    if int(collision_stats["n_collided_tiles"]) <= 0:
        return False, collision_stats, mode
    if mode == "always":
        return True, collision_stats, mode

    min_collided = int(max(1, int(os.environ.get("SPIX_PLOT_AUTO_COLLISION_MIN_TILES", "100000"))))
    min_fraction = float(np.clip(float(os.environ.get("SPIX_PLOT_AUTO_COLLISION_MIN_FRACTION", "0.01")), 0.0, 1.0))
    max_tile_px = int(max(1, int(os.environ.get("SPIX_PLOT_AUTO_COLLISION_MAX_TILE_PX", "1"))))
    allow_pixel_perfect = str(os.environ.get("SPIX_PLOT_AUTO_COLLISION_ALLOW_PIXEL_PERFECT", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    should_apply = (
        int(tile_px) <= max_tile_px
        and (allow_pixel_perfect or (not bool(pixel_perfect)))
        and int(collision_stats["n_collided_tiles"]) >= min_collided
        and float(collision_stats["collision_fraction"]) >= min_fraction
    )
    if logger is not None and should_apply:
        logger.info(
            "%s: auto-enabling resolve_center_collisions (collided=%d, frac=%.4f, tile_px=%d, pixel_perfect=%s).",
            context,
            int(collision_stats["n_collided_tiles"]),
            float(collision_stats["collision_fraction"]),
            int(tile_px),
            bool(pixel_perfect),
        )
    return bool(should_apply), collision_stats, mode


def resolve_collision_search_radius(
    resolve_center_collisions,
    *,
    base_radius: int,
    collision_stats: Optional[dict],
    logger=None,
    context: str = "",
) -> int:
    radius = int(max(0, int(base_radius)))
    mode = normalize_collision_mode(resolve_center_collisions)
    if mode != "auto" or collision_stats is None:
        return int(radius)

    max_auto_radius = int(max(radius, int(os.environ.get("SPIX_PLOT_AUTO_COLLISION_MAX_RADIUS", "4"))))
    if max_auto_radius <= radius:
        return int(radius)

    collided = int(collision_stats.get("n_collided_tiles", 0))
    frac = float(collision_stats.get("collision_fraction", 0.0))
    severe_tiles = int(max(1, int(os.environ.get("SPIX_PLOT_AUTO_COLLISION_SEVERE_TILES", "10000000"))))
    moderate_tiles = int(max(1, int(os.environ.get("SPIX_PLOT_AUTO_COLLISION_MODERATE_TILES", "1000000"))))
    severe_frac = float(np.clip(float(os.environ.get("SPIX_PLOT_AUTO_COLLISION_SEVERE_FRACTION", "0.15")), 0.0, 1.0))
    moderate_frac = float(np.clip(float(os.environ.get("SPIX_PLOT_AUTO_COLLISION_MODERATE_FRACTION", "0.05")), 0.0, 1.0))

    target_radius = int(radius)
    if collided >= severe_tiles and frac >= severe_frac:
        target_radius = int(max(target_radius, min(max_auto_radius, 4)))
    elif collided >= moderate_tiles and frac >= moderate_frac:
        target_radius = int(max(target_radius, min(max_auto_radius, 3)))

    if logger is not None and target_radius > radius:
        logger.info(
            "%s: auto-increased center collision radius %d -> %d (collided=%d, frac=%.4f).",
            context,
            int(radius),
            int(target_radius),
            int(collided),
            float(frac),
        )
    return int(target_radius)


def resolve_center_collisions(
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    w: int,
    h: int,
    max_radius: int = 2,
    chunk_size: int = 2_000_000,
    logger=None,
    context: str = "",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Move duplicate raster centers onto nearby empty pixels.

    The first tile at each pixel center stays fixed. Remaining colliding tiles are
    reassigned to the nearest free pixel within ``max_radius`` in raster space.
    Any tiles that still cannot be placed remain at their original centers.
    """
    cx0 = np.asarray(cx, dtype=np.int32).ravel()
    cy0 = np.asarray(cy, dtype=np.int32).ravel()
    n_tiles = int(cx0.size)
    if cy0.size != n_tiles:
        raise ValueError("cx and cy must have the same length.")
    if n_tiles == 0:
        return cx0.copy(), cy0.copy(), {
            "applied": False,
            "enabled": True,
            "search_radius": int(max(0, max_radius)),
            "n_tiles": 0,
            "n_unique_centers_before": 0,
            "n_unique_centers_after": 0,
            "collision_fraction_before": 0.0,
            "collision_fraction_after": 0.0,
            "n_collided_tiles_before": 0,
            "n_collided_tiles_after": 0,
            "n_reassigned_tiles": 0,
            "max_stack_before": 0,
            "max_stack_after": 0,
            "unresolved_tiles": 0,
        }

    w_i = int(max(1, w))
    h_i = int(max(1, h))
    if w_i * h_i <= 0:
        raise ValueError("w and h must be positive.")
    cx0 = np.clip(cx0, 0, max(0, w_i - 1))
    cy0 = np.clip(cy0, 0, max(0, h_i - 1))

    flat0 = cy0.astype(np.int64, copy=False) * np.int64(w_i) + cx0.astype(np.int64, copy=False)
    order = np.argsort(flat0, kind="mergesort")
    flat_sorted = flat0[order]
    dup_sorted = np.zeros(n_tiles, dtype=bool)
    if n_tiles > 1:
        dup_sorted[1:] = flat_sorted[1:] == flat_sorted[:-1]
    if not np.any(dup_sorted):
        info = {
            "applied": False,
            "enabled": True,
            "search_radius": int(max(0, max_radius)),
            "n_tiles": int(n_tiles),
            "n_unique_centers_before": int(n_tiles),
            "n_unique_centers_after": int(n_tiles),
            "collision_fraction_before": 0.0,
            "collision_fraction_after": 0.0,
            "n_collided_tiles_before": 0,
            "n_collided_tiles_after": 0,
            "n_reassigned_tiles": 0,
            "max_stack_before": 1,
            "max_stack_after": 1,
            "unresolved_tiles": 0,
        }
        return cx0.copy(), cy0.copy(), info

    run_starts = np.concatenate(([0], np.flatnonzero(flat_sorted[1:] != flat_sorted[:-1]) + 1))
    counts = np.diff(np.concatenate((run_starts, [n_tiles])))
    max_stack_before = int(counts.max()) if counts.size else 1
    unique_before = int(run_starts.size)
    collided_before = int(n_tiles - unique_before)

    keep_mask = np.ones(n_tiles, dtype=bool)
    keep_mask[order[dup_sorted]] = False
    move_idx = np.flatnonzero(~keep_mask)
    if move_idx.size == 0:
        info = {
            "applied": False,
            "enabled": True,
            "search_radius": int(max(0, max_radius)),
            "n_tiles": int(n_tiles),
            "n_unique_centers_before": int(unique_before),
            "n_unique_centers_after": int(unique_before),
            "collision_fraction_before": float(collided_before / max(1, n_tiles)),
            "collision_fraction_after": float(collided_before / max(1, n_tiles)),
            "n_collided_tiles_before": int(collided_before),
            "n_collided_tiles_after": int(collided_before),
            "n_reassigned_tiles": 0,
            "max_stack_before": int(max_stack_before),
            "max_stack_after": int(max_stack_before),
            "unresolved_tiles": int(collided_before),
        }
        return cx0.copy(), cy0.copy(), info

    cx_new = cx0.copy()
    cy_new = cy0.copy()
    occupancy = np.zeros(int(h_i) * int(w_i), dtype=bool)
    occupancy[flat0[keep_mask]] = True
    move_cx = cx0[move_idx]
    move_cy = cy0[move_idx]
    pending = np.ones(move_idx.size, dtype=bool)
    offsets = _collision_search_offsets(int(max_radius))
    work_chunk = int(max(1, chunk_size))

    for dx, dy in offsets:
        if not np.any(pending):
            break
        active_pos = np.flatnonzero(pending)
        for start in range(0, active_pos.size, work_chunk):
            local_pos = active_pos[start : start + work_chunk]
            tx = move_cx[local_pos].astype(np.int64, copy=False) + np.int64(dx)
            ty = move_cy[local_pos].astype(np.int64, copy=False) + np.int64(dy)
            valid = (tx >= 0) & (ty >= 0) & (tx < w_i) & (ty < h_i)
            if not np.any(valid):
                continue
            cand_pos = local_pos[valid]
            cand_flat = ty[valid] * np.int64(w_i) + tx[valid]
            free = ~occupancy[cand_flat]
            if not np.any(free):
                continue
            cand_pos = cand_pos[free]
            cand_flat = cand_flat[free]
            cand_order = np.argsort(cand_flat, kind="mergesort")
            cand_flat_sorted = cand_flat[cand_order]
            first_hits = np.ones(cand_flat_sorted.size, dtype=bool)
            if cand_flat_sorted.size > 1:
                first_hits[1:] = cand_flat_sorted[1:] != cand_flat_sorted[:-1]
            chosen = cand_order[first_hits]
            chosen_pos = cand_pos[chosen]
            chosen_flat = cand_flat[chosen]
            chosen_idx = move_idx[chosen_pos]
            cx_new[chosen_idx] = (chosen_flat % np.int64(w_i)).astype(np.int32, copy=False)
            cy_new[chosen_idx] = (chosen_flat // np.int64(w_i)).astype(np.int32, copy=False)
            occupancy[chosen_flat] = True
            pending[chosen_pos] = False

    unresolved_tiles = int(np.count_nonzero(pending))
    flat_final = cy_new.astype(np.int64, copy=False) * np.int64(w_i) + cx_new.astype(np.int64, copy=False)
    _, counts_after = np.unique(flat_final, return_counts=True)
    unique_after_n = int(counts_after.size)
    collided_after = int(n_tiles - unique_after_n)
    max_stack_after = int(counts_after.max()) if counts_after.size else 1
    reassigned = int(move_idx.size - unresolved_tiles)

    info = {
        "applied": bool(reassigned > 0),
        "enabled": True,
        "search_radius": int(max(0, max_radius)),
        "n_tiles": int(n_tiles),
        "n_unique_centers_before": int(unique_before),
        "n_unique_centers_after": int(unique_after_n),
        "collision_fraction_before": float(collided_before / max(1, n_tiles)),
        "collision_fraction_after": float(collided_after / max(1, n_tiles)),
        "n_collided_tiles_before": int(collided_before),
        "n_collided_tiles_after": int(collided_after),
        "n_reassigned_tiles": int(reassigned),
        "max_stack_before": int(max_stack_before),
        "max_stack_after": int(max_stack_after),
        "unresolved_tiles": int(unresolved_tiles),
    }

    if reassigned > 0:
        moved_dx = cx_new[move_idx] - cx0[move_idx]
        moved_dy = cy_new[move_idx] - cy0[move_idx]
        moved_dist = np.hypot(moved_dx.astype(np.float32, copy=False), moved_dy.astype(np.float32, copy=False))
        moved_mask = moved_dist > 0
        if np.any(moved_mask):
            moved_dist = moved_dist[moved_mask]
            info["mean_displacement_px"] = float(np.mean(moved_dist))
            info["max_displacement_px"] = float(np.max(moved_dist))
        else:
            info["mean_displacement_px"] = 0.0
            info["max_displacement_px"] = 0.0
    else:
        info["mean_displacement_px"] = 0.0
        info["max_displacement_px"] = 0.0

    if logger is not None and collided_before > 0:
        msg = f"{context}: " if context else ""
        logger.info(
            "%sResolved raster center collisions within radius=%d; before=%d after=%d reassigned=%d unresolved=%d mean_disp=%.2fpx max_disp=%.2fpx.",
            msg,
            int(max(0, max_radius)),
            int(collided_before),
            int(collided_after),
            int(reassigned),
            int(unresolved_tiles),
            float(info["mean_displacement_px"]),
            float(info["max_displacement_px"]),
        )

    return cx_new, cy_new, info


def scale_points_to_canvas(
    points: np.ndarray,
    *,
    x_min_eff: float,
    y_min_eff: float,
    scale: float,
    pixel_perfect: bool,
    w: Optional[int] = None,
    h: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must be a 2D array with at least two columns.")
    if pixel_perfect:
        cx = np.floor((pts[:, 0] - float(x_min_eff)) * float(scale) + 0.5).astype(np.int32)
        cy = np.floor((pts[:, 1] - float(y_min_eff)) * float(scale) + 0.5).astype(np.int32)
    else:
        cx = np.rint((pts[:, 0] - float(x_min_eff)) * float(scale)).astype(np.int32)
        cy = np.rint((pts[:, 1] - float(y_min_eff)) * float(scale)).astype(np.int32)
    if w is not None:
        cx = np.clip(cx, 0, max(0, int(w) - 1))
    if h is not None:
        cy = np.clip(cy, 0, max(0, int(h) - 1))
    return cx, cy


def scale_points_to_canvas_soft(
    points: np.ndarray,
    *,
    x_min_eff: float,
    y_min_eff: float,
    scale: float,
    pixel_perfect: bool,
    w: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest-center indices plus bilinear fractions for soft rasterization."""
    if not bool(pixel_perfect):
        raise ValueError("soft rasterization currently requires pixel_perfect=True.")
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must be a 2D array with at least two columns.")
    w_i = int(max(1, w))
    h_i = int(max(1, h))
    ux = (pts[:, 0] - float(x_min_eff)) * float(scale)
    uy = (pts[:, 1] - float(y_min_eff)) * float(scale)
    ux = np.clip(ux, 0.0, float(max(0, w_i - 1)))
    uy = np.clip(uy, 0.0, float(max(0, h_i - 1)))
    fx = ux - np.floor(ux)
    fy = uy - np.floor(uy)
    cx = np.floor(ux + 0.5).astype(np.int32)
    cy = np.floor(uy + 0.5).astype(np.int32)
    return cx, cy, fx.astype(np.float32), fy.astype(np.float32)


def bilinear_support_from_centers(
    cx: np.ndarray,
    cy: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    *,
    w: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand nearest-center metadata into 4-neighbor bilinear support."""
    cx_i = np.asarray(cx, dtype=np.int32).ravel()
    cy_i = np.asarray(cy, dtype=np.int32).ravel()
    fx_f = np.asarray(fx, dtype=np.float32).ravel()
    fy_f = np.asarray(fy, dtype=np.float32).ravel()
    n = int(cx_i.size)
    if cy_i.size != n or fx_f.size != n or fy_f.size != n:
        raise ValueError("cx, cy, fx, fy must have the same length.")
    w_i = int(max(1, w))
    h_i = int(max(1, h))
    x0 = cx_i.astype(np.int64, copy=False) - (fx_f >= 0.5).astype(np.int64, copy=False)
    y0 = cy_i.astype(np.int64, copy=False) - (fy_f >= 0.5).astype(np.int64, copy=False)
    x0 = np.clip(x0, 0, max(0, w_i - 1)).astype(np.int32, copy=False)
    y0 = np.clip(y0, 0, max(0, h_i - 1)).astype(np.int32, copy=False)
    x1 = np.clip(x0.astype(np.int64, copy=False) + 1, 0, max(0, w_i - 1)).astype(np.int32, copy=False)
    y1 = np.clip(y0.astype(np.int64, copy=False) + 1, 0, max(0, h_i - 1)).astype(np.int32, copy=False)
    one_minus_fx = (1.0 - fx_f).astype(np.float32, copy=False)
    one_minus_fy = (1.0 - fy_f).astype(np.float32, copy=False)
    w00 = (one_minus_fx * one_minus_fy).astype(np.float32, copy=False)
    w10 = (fx_f * one_minus_fy).astype(np.float32, copy=False)
    w01 = (one_minus_fx * fy_f).astype(np.float32, copy=False)
    w11 = (fx_f * fy_f).astype(np.float32, copy=False)
    return x0, y0, x1, y1, w00, w10, w01, w11


def rasterize_soft_bilinear(
    *,
    img: np.ndarray,
    occupancy_mask: np.ndarray,
    seed_values: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    chunk_size: int = 50_000,
    background_value: Optional[float] = 1.0,
) -> None:
    """Bilinearly splat per-tile values into a raster image."""
    img_arr = np.asarray(img)
    if img_arr.ndim != 3:
        raise ValueError("img must have shape (H, W, C).")
    occ = np.asarray(occupancy_mask)
    if occ.shape != img_arr.shape[:2]:
        raise ValueError("occupancy_mask must match img spatial shape.")
    vals = np.asarray(seed_values, dtype=np.float32)
    if vals.ndim != 2 or vals.shape[0] != int(np.asarray(cx).size):
        raise ValueError("seed_values must have shape (N, C) aligned to cx/cy.")

    h_i, w_i = img_arr.shape[:2]
    img_arr.fill(0.0)
    weight_sum = np.zeros((h_i, w_i), dtype=np.float32)
    work_chunk = int(max(1, chunk_size))

    for start in range(0, vals.shape[0], work_chunk):
        end = min(start + work_chunk, vals.shape[0])
        x0, y0, x1, y1, w00, w10, w01, w11 = bilinear_support_from_centers(
            np.asarray(cx[start:end]),
            np.asarray(cy[start:end]),
            np.asarray(fx[start:end]),
            np.asarray(fy[start:end]),
            w=w_i,
            h=h_i,
        )
        chunk_vals = vals[start:end]
        for px, py, pw in ((x0, y0, w00), (x1, y0, w10), (x0, y1, w01), (x1, y1, w11)):
            np.add.at(img_arr, (py, px), chunk_vals * pw[:, None])
            np.add.at(weight_sum, (py, px), pw)

    touched = weight_sum > 0
    occ[...] = touched
    if np.any(touched):
        img_arr[touched] /= weight_sum[touched, None]
    if background_value is not None:
        img_arr[~touched] = float(background_value)


def sample_labels_soft_bilinear(
    label_img: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    *,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """Read per-tile labels by bilinear voting over the 4-neighbor support."""
    labels_img = np.asarray(label_img)
    if labels_img.ndim != 2:
        raise ValueError("label_img must be a 2D array.")
    n = int(np.asarray(cx).size)
    out = np.empty(n, dtype=np.int32)
    h_i, w_i = labels_img.shape
    work_chunk = int(max(1, chunk_size))

    for start in range(0, n, work_chunk):
        end = min(start + work_chunk, n)
        x0, y0, x1, y1, w00, w10, w01, w11 = bilinear_support_from_centers(
            np.asarray(cx[start:end]),
            np.asarray(cy[start:end]),
            np.asarray(fx[start:end]),
            np.asarray(fy[start:end]),
            w=w_i,
            h=h_i,
        )
        candidates = np.stack(
            [
                labels_img[y0, x0],
                labels_img[y0, x1],
                labels_img[y1, x0],
                labels_img[y1, x1],
            ],
            axis=1,
        ).astype(np.int32, copy=False)
        weights = np.stack([w00, w10, w01, w11], axis=1).astype(np.float32, copy=False)
        scores = np.empty_like(weights)
        for j in range(4):
            scores[:, j] = np.sum(weights * (candidates == candidates[:, [j]]), axis=1)
        best = np.argmax(scores, axis=1)
        out[start:end] = candidates[np.arange(end - start), best]

    return out


def effective_extent(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    s_data_units: float,
    pixel_perfect: bool,
) -> Tuple[float, float, float, float]:
    if pixel_perfect:
        return (
            float(x_min) - 0.5 * float(s_data_units),
            float(x_max) + 0.5 * float(s_data_units),
            float(y_min) - 0.5 * float(s_data_units),
            float(y_max) + 0.5 * float(s_data_units),
        )
    return float(x_min), float(x_max), float(y_min), float(y_max)


def scale_raw_from_canvas(
    *,
    w_pixels: int,
    h_pixels: int,
    x_range: float,
    y_range: float,
    pixel_perfect: bool,
    auto_sized: bool,
) -> float:
    if x_range <= 0 or y_range <= 0:
        return 1.0
    if pixel_perfect and auto_sized:
        return float(np.sqrt((float(w_pixels) * float(h_pixels)) / (float(x_range) * float(y_range))))
    return float(min(float(w_pixels) / float(x_range), float(h_pixels) / float(y_range)))
