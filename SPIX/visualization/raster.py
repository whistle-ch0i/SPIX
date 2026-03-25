import os
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


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
    else:
        s_data_units = float(imshow_tile_size)

    s_data_units *= float(imshow_scale_factor)
    s_data_units *= float(imshow_tile_size_shrink)
    s_data_units = max(0.1, float(s_data_units))
    return float(s_data_units)


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

    max_raster_px = int(os.environ.get("SPIX_PLOT_MAX_RASTER_PIXELS", "25000000"))
    max_raster_px = max(1, max_raster_px)
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
) -> dict:
    user_figsize_given = figsize is not None
    user_dpi_given = fig_dpi is not None
    auto_sized = (not user_figsize_given) or (not user_dpi_given)
    shrink_eff = 1.0 if pixel_perfect else float(imshow_tile_size_shrink)

    s_data_units = estimate_tile_size_units(
        np.asarray(pts),
        imshow_tile_size,
        imshow_scale_factor,
        imshow_tile_size_mode,
        imshow_tile_size_quantile,
        shrink_eff,
        logger=logger,
    )
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

    figsize_resolved, fig_dpi_resolved = resolve_figsize_dpi_for_tiles(
        figsize=figsize,
        fig_dpi=fig_dpi,
        x_range=float(x_range),
        y_range=float(y_range),
        s_data_units=float(s_data_units),
        pixel_perfect=bool(pixel_perfect),
        n_points=int(n_points),
        logger=logger,
    )
    if figsize_resolved is None:
        figsize_resolved = tuple(default_figsize)

    dpi_in = int(fig_dpi_resolved) if fig_dpi_resolved is not None else 100
    w_pixels = int(np.ceil(float(figsize_resolved[0]) * float(dpi_in)))
    h_pixels = int(np.ceil(float(figsize_resolved[1]) * float(dpi_in)))

    scale_raw = scale_raw_from_canvas(
        w_pixels=int(w_pixels),
        h_pixels=int(h_pixels),
        x_range=float(x_range),
        y_range=float(y_range),
        pixel_perfect=bool(pixel_perfect),
        auto_sized=bool(auto_sized),
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

    return {
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
    }


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
