import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
from skimage.segmentation import slic
from skimage.transform import resize


def image_plot_slic_segmentation(
    embeddings: np.ndarray,
    spatial_coords: np.ndarray,
    n_segments: int = 100,
    compactness: float = 10.0,
    figsize: tuple = (10, 10),
    fig_dpi: Optional[int] = None,
    imshow_tile_size: Optional[float] = None,
    imshow_scale_factor: float = 1.0,
    enforce_connectivity: bool = False,
    downsample_factor: float = 1.0,
    chunk_size: int = 50000,
    *,
    pixel_shape: str = "square",
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
    if imshow_tile_size is None:
        s_data_units = 1.0
        if len(spatial_coords) > 1:
            sample_size = min(len(spatial_coords), 100000)
            sample_idx = np.random.choice(len(spatial_coords), sample_size, replace=False)
            tree = KDTree(spatial_coords[sample_idx])
            d, _ = tree.query(spatial_coords[sample_idx], k=2)
            nn = d[:, 1][d[:, 1] > 0]
            if len(nn) > 0:
                s_data_units = np.median(nn)
    else:
        s_data_units = imshow_tile_size
    s_data_units = max(0.1, s_data_units * imshow_scale_factor)
    dpi = int(fig_dpi) if fig_dpi is not None else 100
    w_pixels, h_pixels = int(np.ceil(figsize[0] * dpi)), int(np.ceil(figsize[1] * dpi))
    x_range, y_range = (x_max - x_min) or 1.0, (y_max - y_min) or 1.0
    scale = min(w_pixels / x_range, h_pixels / y_range)
    w, h = max(1, int(np.ceil(x_range * scale))), max(1, int(np.ceil(y_range * scale)))
    s = max(1, int(np.ceil(s_data_units * scale)))

    scaler = MinMaxScaler()
    cols = scaler.fit_transform(embeddings.astype(np.float32))

    logger.info(f"Creating image with chunking (chunk size: {chunk_size})...")
    img = np.ones((h, w, dims), dtype=np.float32)

    if pixel_shape == "circle":
        yy, xx = np.ogrid[:s, :s]
        center = (s - 1) / 2
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= (s / 2) ** 2
        dy, dx = np.nonzero(mask)
    else:
        s_range = np.arange(s)
        gx, gy = np.meshgrid(s_range, s_range)
        dx, dy = gx.flatten(), gy.flatten()

    for i in range(0, num_tiles, chunk_size):
        chunk_end = min(i + chunk_size, num_tiles)
        chunk_indices = np.arange(i, chunk_end)
        chunk_xi = ((spatial_coords[chunk_indices, 0] - x_min) * scale).astype(int)
        chunk_yi = ((spatial_coords[chunk_indices, 1] - y_min) * scale).astype(int)

        chunk_all_x = np.clip((chunk_xi[:, np.newaxis] + dx).flatten(), 0, w - 1)
        chunk_all_y = np.clip((chunk_yi[:, np.newaxis] + dy).flatten(), 0, h - 1)
        chunk_tile_indices_map = np.repeat(chunk_indices, len(dx))
        img[chunk_all_y, chunk_all_x] = cols[chunk_tile_indices_map]

    logger.info("Image creation complete.")

    if downsample_factor > 1.0:
        logger.info(f"Downsampling image by a factor of {downsample_factor}...")
        new_h, new_w = int(h / downsample_factor), int(w / downsample_factor)
        img = resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
        scale /= downsample_factor
        w, h = new_w, new_h
        logger.info("Downsampling complete.")

    logger.info("Running SLIC on CPU...")
    label_img = slic(
        img.astype(np.float64),
        n_segments=n_segments,
        compactness=compactness,
        sigma=0,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=enforce_connectivity,
    )
    logger.info("CPU SLIC complete.")

    final_xi = ((spatial_coords[:, 0] - x_min) * scale).astype(int)
    final_yi = ((spatial_coords[:, 1] - y_min) * scale).astype(int)
    final_s = max(1, int(np.ceil(s_data_units * scale)))
    cx = np.clip(final_xi + final_s // 2, 0, w - 1)
    cy = np.clip(final_yi + final_s // 2, 0, h - 1)
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
        plt.imshow(display_img, origin="lower")
        plt.title("Embeddings Image")
        plt.axis("off")
        plt.show()

        from matplotlib.colors import ListedColormap

        max_label = int(label_img.max())
        rng = np.random.default_rng(0)
        rand_cmap = ListedColormap(rng.random((max_label + 1, 3)))

        plt.figure(figsize=figsize, dpi=fig_dpi if fig_dpi is not None else None)
        plt.imshow(label_img, origin="lower", cmap="nipy_spectral")
        plt.imshow(label_img, origin="lower", cmap=rand_cmap, vmin=0, vmax=max_label)
        
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
        img.astype(np.float64),
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
