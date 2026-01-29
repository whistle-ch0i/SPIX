import os
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


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
            sample_indices = np.random.choice(len(pts), sample_size, replace=False)
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
