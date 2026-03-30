import numpy as np
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
from skimage.segmentation import slic
from skimage.transform import resize

from SPIX.visualization import raster as _raster


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
    pixel_perfect: bool = True,
    enforce_connectivity: bool = False,
    downsample_factor: float = 1.0,
    chunk_size: int = 50000,
    *,
    pixel_shape: str = "square",
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_holes: bool = True,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool = False,
    center_collision_radius: int = 2,
    show_image: bool = False,
    verbose: bool = True,
    n_jobs: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    soft_rasterization : bool, optional (default ``False``)
        If ``True``, rasterize each spot with bilinear splatting and read SLIC
        labels back with bilinear voting instead of hard one-pixel assignment.
        Currently supported only for ``pixel_perfect=True``, ``tile_px==1`` and
        ``pixel_shape='square'``.
    resolve_center_collisions : bool, optional (default ``False``)
        If ``True``, locally reassign duplicate raster centers onto nearby free
        pixels before painting/fill/label lookup. Useful for dense origin-based
        rasterization when many spots quantize onto the same pixel.
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

    if log is not None:
        log.info("Creating image with chunking (chunk size: %d)...", int(chunk_size))
    img = np.ones((h, w, dims), dtype=np.float32)
    paint_mask = np.zeros((h, w), dtype=bool)

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
    collision_info = None
    if resolve_center_collisions and not soft_enabled:
        cx_seed, cy_seed, collision_info = _raster.resolve_center_collisions(
            cx_seed,
            cy_seed,
            w=int(w),
            h=int(h),
            max_radius=int(center_collision_radius),
            logger=log,
            context="image_plot_slic_segmentation",
        )
    elif resolve_center_collisions and soft_enabled and log is not None:
        log.info("image_plot_slic_segmentation: ignoring resolve_center_collisions because soft_rasterization=True.")

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

    if runtime_fill_from_boundary:
        _raster.fill_raster_from_boundary(
            img=img,
            occupancy_mask=paint_mask,
            seed_x=cx_seed,
            seed_y=cy_seed,
            seed_values=cols,
            closing_radius=int(runtime_fill_closing_radius),
            fill_holes=bool(runtime_fill_holes),
            logger=log,
            context="image_plot_slic_segmentation",
        )

    if downsample_factor > 1.0:
        if log is not None:
            log.info("Downsampling image by a factor of %.3f...", float(downsample_factor))
        new_h, new_w = int(h / downsample_factor), int(w / downsample_factor)
        img = resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
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
        elif resolve_center_collisions:
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

    if log is not None:
        log.info("Running SLIC on CPU...")
    label_img = slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        sigma=0,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=enforce_connectivity,
    )
    if log is not None:
        log.info("CPU SLIC complete.")

    if soft_enabled:
        labels = _raster.sample_labels_soft_bilinear(
            label_img,
            cx,
            cy,
            soft_fx,
            soft_fy,
            chunk_size=max(100_000, int(chunk_size)),
        ).astype(int, copy=False)
    else:
        labels = label_img[cy, cx].astype(int)
    if show_image:
        import matplotlib.pyplot as plt

        # ``matplotlib`` only supports 1, 3 or 4 channel images. Reduce or
        # expand the embedding image so it is displayable.
        display_img = img
        if img.shape[2] > 3:
            display_img = img[:, :, :3]
        elif img.shape[2] < 3:
            gray = img.mean(axis=2, keepdims=True)
            display_img = np.repeat(gray, 3, axis=2)

        plt.figure(figsize=figsize, dpi=fig_dpi if fig_dpi is not None else None)
        plt.imshow(display_img, origin="lower", interpolation="nearest")
        plt.title("Embeddings Image")
        plt.axis("off")
        plt.show()

        from matplotlib.colors import ListedColormap

        max_label = int(label_img.max())
        rng = np.random.default_rng(0)
        rand_cmap = ListedColormap(rng.random((max_label + 1, 3)))

        plt.figure(figsize=figsize, dpi=fig_dpi if fig_dpi is not None else None)
        plt.imshow(label_img, origin="lower", cmap="nipy_spectral", interpolation="nearest")
        plt.imshow(label_img, origin="lower", cmap=rand_cmap, vmin=0, vmax=max_label, interpolation="nearest")
        
        plt.title("SLIC Segmentation")
        plt.axis("off")
        plt.show()

    return labels, label_img, img


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
) -> tuple[np.ndarray, np.ndarray]:
    """Run SLIC on a precomputed cached image dict.

    Expects a dict as produced by ``image_processing.image_cache.cache_embedding_image``.
    Returns tile-level labels (aligned to the cached tile order) and the label image.
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    img = np.asarray(cache.get("img"))
    if img.ndim != 3:
        raise ValueError("Cached image must have shape (H, W, C).")

    label_img = slic(
        img,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=0,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=bool(enforce_connectivity),
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
        labels = _raster.sample_labels_soft_bilinear(
            label_img,
            cx,
            cy,
            np.asarray(soft_fx, dtype=np.float32),
            np.asarray(soft_fy, dtype=np.float32),
        ).astype(int, copy=False)
    else:
        labels = label_img[cy, cx].astype(int)

    if show_image:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        display_img = img
        if img.shape[2] > 3:
            display_img = img[:, :, :3]
        elif img.shape[2] < 3:
            gray = img.mean(axis=2, keepdims=True)
            display_img = np.repeat(gray, 3, axis=2)

        # Prefer provided figsize/fig_dpi; else fall back to cached
        _figsize = figsize if figsize is not None else cache.get("figsize", (8, 8))
        _dpi = fig_dpi if fig_dpi is not None else cache.get("fig_dpi", None)
        plot_params = cache.get("image_plot_params", {}) or {}
        apply_brighten = bool(plot_params.get("brighten_continuous", False))
        gamma = float(plot_params.get("continuous_gamma", 0.8))
        cache_channels = int(cache.get("channels", img.shape[2]))
        already_applied = bool(plot_params.get("brighten_applied", False)) and cache_channels == 3
        if apply_brighten and not already_applied:
            display_img = _apply_brighten_continuous_rgb(display_img, gamma)
        plt.figure(figsize=_figsize, dpi=_dpi)
        plt.imshow(display_img, origin="lower")
        plt.title("Cached Embeddings Image")
        plt.axis("off")
        plt.show()

        max_label = int(label_img.max())
        rng = np.random.default_rng(0)
        rand_cmap = ListedColormap(rng.random((max_label + 1, 3)))
        plt.figure(figsize=_figsize, dpi=_dpi)
        plt.imshow(label_img, origin="lower", cmap=rand_cmap, vmin=0, vmax=max_label)
        plt.title("SLIC Segmentation (cached)")
        plt.axis("off")
        plt.show()

    return labels, label_img
