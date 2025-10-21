import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
from typing import Optional, Dict, Any, List, Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter, binary_dilation

# no additional utils needed here after function removals


def cache_embedding_image(
    adata,
    *,
    embedding: str = "X_embedding",
    dimensions: List[int] = [0, 1, 2],
    origin: bool = True,
    key: str = "image_plot_slic",
    figsize: tuple = (10, 10),
    fig_dpi: Optional[int] = None,
    imshow_tile_size: Optional[float] = None,
    imshow_scale_factor: float = 1.0,
    pixel_shape: str = "square",
    chunk_size: int = 50000,
    downsample_factor: float = 1.0,
    verbose: bool = False,
    show: bool = False,
    show_channels: Optional[Sequence[int]] = None,
    show_cmap: Optional[str] = None,
    # Selected image_plot compatibility options (accepted + stored; some applied):
    color_by: Optional[str] = None,
    palette: Optional[str | Sequence] = None,
    alpha_by: Optional[str] = None,
    alpha_range: Tuple[float, float] = (0.1, 1.0),
    alpha_clip: Optional[Tuple[Optional[float], Optional[float]]] = None,
    alpha_invert: bool = False,
    prioritize_high_values: bool = False,
    overlap_priority: Optional[str] = None,
    segment_color_by_major: bool = False,
    title: Optional[str] = None,
    use_imshow: bool = True,
    # Boundary-related options for preview overlay
    plot_boundaries: bool = False,
    boundary_method: str = "pixel",
    boundary_color: str = "black",
    boundary_linewidth: float = 1.0,
    boundary_style: str = "solid",
    boundary_alpha: float = 0.7,
    pixel_smoothing_sigma: float = 0.0,
    highlight_segments: Optional[Sequence[str]] = None,
    highlight_boundary_color: Optional[str] = None,
    highlight_linewidth: Optional[float] = None,
    highlight_brighten_factor: float = 1.2,
    dim_other_segments: float = 0.3,
    dim_to_grey: bool = True,
    segment_key: Optional[str] = "Segment",
    **plot_kwargs,
) -> Dict[str, Any]:
    """Rasterize embeddings + spatial coords to an image and cache in ``adata.uns``.

    The cached dict includes the image, geometry metadata, and a tile→obs
    mapping that allows later algorithms (e.g., SLIC) to sample labels
    for each tile efficiently without re-rasterizing.

    Returns the cached dict and stores it under ``adata.uns[key]``.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Build coordinates_df in the same way as segmentation uses it
    tiles_df = adata.uns["tiles"]
    tiles = tiles_df[tiles_df["origin"] == 1] if origin else tiles_df

    emb = np.asarray(adata.obsm[embedding])[:, dimensions]
    dim_cols = [f"dim{i}" for i, _ in enumerate(dimensions)]
    tile_colors = pd.DataFrame(emb, columns=dim_cols)
    tile_colors["barcode"] = adata.obs.index.astype(str)
    coordinates_df = (
        pd.merge(tiles, tile_colors, on="barcode", how="right")
        .dropna()
        .reset_index(drop=True)
    )

    if len(coordinates_df) == 0:
        raise ValueError("No coordinates found after merging tiles with embeddings.")

    segment_series = None
    seg_col: Optional[str] = None
    segment_available = False
    if segment_key is not None:
        candidate = segment_key
        if candidate in adata.obs.columns:
            seg_col = candidate
            segment_available = True
        elif candidate != "Segment" and "Segment" in adata.obs.columns:
            if verbose:
                print(f"[cache_embedding_image] segment_key '{candidate}' not found; using 'Segment' column instead.")
            seg_col = "Segment"
            segment_available = True
        elif verbose:
            print(f"[cache_embedding_image] segment_key '{candidate}' not found in adata.obs; segment overlays disabled.")

    if segment_available and seg_col is not None:
        coordinates_df[seg_col] = coordinates_df["barcode"].map(adata.obs[seg_col])
        if coordinates_df[seg_col].isnull().any():
            missing = int(coordinates_df[seg_col].isnull().sum())
            if verbose:
                print(f"[cache_embedding_image] '{seg_col}' labels missing for {missing} barcodes. Dropping them.")
            coordinates_df = coordinates_df.dropna(subset=[seg_col])
        coordinates_df["Segment"] = coordinates_df[seg_col]
        segment_series = adata.obs[seg_col]
    else:
        coordinates_df = coordinates_df.drop(columns=["Segment"], errors="ignore")
        segment_available = False
        seg_col = None

    # Order mapping back to AnnData.obs positions
    obs_indexer = {bc: i for i, bc in enumerate(adata.obs.index.astype(str))}
    tile_obs_indices = coordinates_df["barcode"].astype(str).map(obs_indexer).to_numpy()

    spatial_coords = coordinates_df[["x", "y"]].to_numpy(dtype=float)
    embeddings = coordinates_df[dim_cols].to_numpy(dtype=float)

    # Geometry and pixel scaling
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
                s_data_units = float(np.median(nn))
    else:
        s_data_units = float(imshow_tile_size)
    s_data_units = max(0.1, s_data_units * float(imshow_scale_factor))

    dpi_in = int(fig_dpi) if fig_dpi is not None else 100
    w_pixels = int(np.ceil(figsize[0] * dpi_in))
    h_pixels = int(np.ceil(figsize[1] * dpi_in))
    x_range = (x_max - x_min) or 1.0
    y_range = (y_max - y_min) or 1.0
    scale = min(w_pixels / x_range, h_pixels / y_range)
    w = max(1, int(np.ceil(x_range * scale)))
    h = max(1, int(np.ceil(y_range * scale)))
    s = max(1, int(np.ceil(s_data_units * scale)))

    # Normalize embedding channels 0..1 per channel
    dims = embeddings.shape[1]
    scaler = MinMaxScaler()
    cols = scaler.fit_transform(embeddings.astype(np.float32))

    if verbose:
        print(f"Rasterizing → image size (h, w, C)=({h}, {w}, {dims}) | tile_px={s}")

    img = np.ones((h, w, dims), dtype=np.float32)

    # Precompute pixel offsets for square/circle tiles
    if pixel_shape == "circle":
        yy, xx = np.ogrid[:s, :s]
        center = (s - 1) / 2
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= (s / 2) ** 2
        dy, dx = np.nonzero(mask)
    else:
        rng = np.arange(s)
        gx, gy = np.meshgrid(rng, rng)
        dx, dy = gx.flatten(), gy.flatten()

    # Optional depth ordering similar to image_plot (alpha-driven)
    def _compute_alpha_from_values(values, arange=(0.1, 1.0), clip=None, invert=False):
        v = np.asarray(values, dtype=float)
        if clip is None:
            vmin, vmax = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
        else:
            vmin, vmax = clip
            if vmin is None:
                vmin = np.nanmin(v)
            if vmax is None:
                vmax = np.nanmax(v)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
        if invert:
            t = 1.0 - t
        a0, a1 = float(arange[0]), float(arange[1])
        a = a0 + t * (a1 - a0)
        return a.astype(np.float32)

    order_idx = np.arange(spatial_coords.shape[0])
    if alpha_by is not None:
        if alpha_by == "embedding":
            # Use the first selected dimension as alpha source
            alpha_source = embeddings[:, 0]
        else:
            if alpha_by not in adata.obs.columns:
                raise ValueError(f"alpha_by '{alpha_by}' not found in adata.obs.")
            # Build aligned values for present barcodes
            vals_map = adata.obs[alpha_by].astype(float)
            alpha_source = coordinates_df["barcode"].map(vals_map).to_numpy(dtype=float)
        alpha_arr = _compute_alpha_from_values(alpha_source, arange=alpha_range, clip=alpha_clip, invert=alpha_invert)
        if prioritize_high_values:
            # draw high alphas last (front)
            order_idx = np.argsort(alpha_arr)

    # Chunked rasterization in chosen order
    num_tiles = spatial_coords.shape[0]
    for i in range(0, num_tiles, max(1, chunk_size)):
        idx = order_idx[i : min(i + chunk_size, num_tiles)]
        xi = ((spatial_coords[idx, 0] - x_min) * scale).astype(int)
        yi = ((spatial_coords[idx, 1] - y_min) * scale).astype(int)
        all_x = np.clip((xi[:, None] + dx).ravel(), 0, w - 1)
        all_y = np.clip((yi[:, None] + dy).ravel(), 0, h - 1)
        tile_idx_map = np.repeat(idx, len(dx))
        img[all_y, all_x] = cols[tile_idx_map]

    # Optional: build label image for boundary overlay (preview only)
    label_img = None
    if plot_boundaries and segment_available and segment_series is not None:
        # Map per-tile Segment labels
        seg_codes = pd.Categorical(coordinates_df["barcode"].map(segment_series)).codes
        label_img = np.full((h, w), -1, dtype=int)
        for i in range(0, num_tiles, max(1, chunk_size)):
            idx = order_idx[i : min(i + chunk_size, num_tiles)]
            xi = ((spatial_coords[idx, 0] - x_min) * scale).astype(int)
            yi = ((spatial_coords[idx, 1] - y_min) * scale).astype(int)
            if pixel_shape == "circle":
                # Paint circle
                for j, t in enumerate(idx):
                    lx = np.clip(xi[j] + dx, 0, w - 1)
                    ly = np.clip(yi[j] + dy, 0, h - 1)
                    label_img[ly, lx] = seg_codes[t]
            else:
                all_x = np.clip((xi[:, None] + dx).ravel(), 0, w - 1)
                all_y = np.clip((yi[:, None] + dy).ravel(), 0, h - 1)
                label_img[all_y, all_x] = np.repeat(seg_codes[idx], len(dx))

    if downsample_factor and downsample_factor > 1.0:
        # Simple average pooling downsample to avoid extra deps
        factor = int(downsample_factor)
        new_h = max(1, h // factor)
        new_w = max(1, w // factor)
        pooled = img[: new_h * factor, : new_w * factor]
        pooled = pooled.reshape(new_h, factor, new_w, factor, dims).mean(axis=(1, 3))
        img = pooled.astype(np.float32)
        h, w = img.shape[:2]
        scale /= factor
        s = max(1, int(np.ceil(s_data_units * scale)))
        dpi_effective = float(dpi_in) / float(factor)
    else:
        dpi_effective = float(dpi_in)

    # If we built a label image, downsample it to match
    if label_img is not None and downsample_factor and downsample_factor > 1.0:
        factor = int(downsample_factor)
        new_h = max(1, label_img.shape[0] // factor)
        new_w = max(1, label_img.shape[1] // factor)
        # nearest pooling for labels
        label_img = label_img[: new_h * factor, : new_w * factor]
        label_img = label_img.reshape(new_h, factor, new_w, factor).max(axis=(1, 3))

    # Compute sampling centers per tile (pixel indices)
    xi_final = ((spatial_coords[:, 0] - x_min) * scale).astype(int)
    yi_final = ((spatial_coords[:, 1] - y_min) * scale).astype(int)
    cx = np.clip(xi_final + s // 2, 0, w - 1)
    cy = np.clip(yi_final + s // 2, 0, h - 1)

    cache = {
        "img": img,
        "h": int(h),
        "w": int(w),
        "total_pixels": int(h * w),
        "channels": int(dims),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "scale": float(scale),
        "s_data_units": float(s_data_units),
        "tile_px": int(s),
        "pixel_shape": str(pixel_shape),
        "embedding_key": str(embedding),
        "dimensions": list(dimensions),
        "origin": bool(origin),
        "figsize": tuple(figsize),
        "fig_dpi": None if fig_dpi is None else int(fig_dpi),
        "raster_dpi": int(dpi_in),
        "effective_dpi": float(dpi_effective),
        # per-tile mapping/meta
        "cx": cx.astype(np.int32),
        "cy": cy.astype(np.int32),
        "tile_obs_indices": tile_obs_indices.astype(np.int64),
        "barcodes": coordinates_df["barcode"].astype(str).to_list(),
        "origin_flags": coordinates_df.get("origin", pd.Series(index=coordinates_df.index, data=1)).astype(int).to_numpy(),
        "segment_key": seg_col,
        "segment_key_requested": segment_key,
        # provenance
        "created_by": "cache_embedding_image",
        "history": [],
        # image_plot compatibility: store provided params for traceability
        "image_plot_params": {
            "color_by": color_by,
            "palette": palette,
            "alpha_by": alpha_by,
            "alpha_range": tuple(alpha_range) if alpha_range is not None else None,
            "alpha_clip": alpha_clip,
            "alpha_invert": bool(alpha_invert),
            "prioritize_high_values": bool(prioritize_high_values),
            "overlap_priority": overlap_priority,
            "segment_color_by_major": bool(segment_color_by_major),
            "title": title,
            "use_imshow": bool(use_imshow),
            # boundary params
            "plot_boundaries": bool(plot_boundaries),
            "boundary_method": boundary_method,
            "boundary_color": boundary_color,
            "boundary_linewidth": float(boundary_linewidth),
            "boundary_style": boundary_style,
            "boundary_alpha": float(boundary_alpha),
            "pixel_smoothing_sigma": float(pixel_smoothing_sigma),
            "highlight_segments": list(highlight_segments) if highlight_segments is not None else None,
            "highlight_boundary_color": highlight_boundary_color,
            "highlight_linewidth": None if highlight_linewidth is None else float(highlight_linewidth),
            "highlight_brighten_factor": float(highlight_brighten_factor),
            "dim_other_segments": float(dim_other_segments),
            "dim_to_grey": bool(dim_to_grey),
            "segment_key": seg_col,
            "segment_key_requested": segment_key,
            **plot_kwargs,
        },
    }

    # Optional: build preview RGB with boundary overlay
    if plot_boundaries and label_img is not None and boundary_method == "pixel":
        # base RGB from img (1,2,3 channels)
        if img.shape[2] >= 3:
            base = img[:, :, :3].copy()
        elif img.shape[2] == 2:
            base = np.stack([img[:, :, 0], img[:, :, 1], np.zeros_like(img[:, :, 0])], axis=2)
        else:
            base = np.repeat(img, 3, axis=2)
        # boundary mask
        bmask = find_boundaries(label_img, mode="outer", background=-1, connectivity=2)
        if pixel_smoothing_sigma and pixel_smoothing_sigma > 0:
            sm = gaussian_filter(bmask.astype(np.float32), pixel_smoothing_sigma)
            bmask = sm >= 0.5
        # line width via binary dilation
        lw = max(1, int(round(boundary_linewidth)))
        if lw > 1:
            bmask = binary_dilation(bmask, iterations=lw - 1)
        # color setup
        base_col = np.array(mcolors.to_rgb(boundary_color), dtype=np.float32)
        if highlight_segments is not None and segment_available and segment_series is not None:
            cats = pd.Categorical(segment_series).categories
            codes = [cats.get_loc(seg) for seg in highlight_segments if seg in cats]
            if codes:
                hmask = np.isin(label_img, np.array(codes, dtype=int)) & bmask
            else:
                hmask = np.zeros_like(bmask)
        else:
            hmask = np.zeros_like(bmask)
        if highlight_boundary_color is not None:
            high_col = np.array(mcolors.to_rgb(highlight_boundary_color), dtype=np.float32)
        else:
            high_col = base_col
        # blend
        out = base.copy()
        a = float(boundary_alpha)
        for c in range(3):
            # base boundary
            out[..., c] = np.where(bmask & ~hmask, a * base_col[c] + (1 - a) * out[..., c], out[..., c])
            # highlight boundary overrides
            out[..., c] = np.where(hmask, a * high_col[c] + (1 - a) * out[..., c], out[..., c])
        cache["img_preview"] = np.clip(out, 0.0, 1.0).astype(np.float32)
        cache["label_img"] = label_img.astype(np.int32)

    adata.uns[key] = cache
    if show:
        try:
            show_cached_image(adata, key=key, channels=show_channels, cmap=show_cmap)
        except Exception:
            pass
    return cache




def rebuild_cached_image_from_obsm(
    adata,
    *,
    key: str = "image_plot_slic",
    embedding: Optional[str] = None,
    dimensions: Optional[Sequence[int]] = None,
    in_place: bool = True,
    out_key: Optional[str] = None,
    show: bool = False,
    show_channels: Optional[Sequence[int]] = None,
    show_cmap: Optional[str] = None,
    normalize_channels: bool = True,
):
    """Rebuild a cached image from an embedding in ``adata.obsm``.

    Uses cached geometry (extent, scale, tile size, centers) to keep pixel
    alignment identical. This is used after smoothing/equalization of embeddings.
    """
    if key not in adata.uns:
        raise KeyError(f"No cached image found at adata.uns['{key}']")
    cache = adata.uns[key]

    emb_key = embedding or cache.get("embedding_key", "X_embedding")
    dims = list(dimensions) if dimensions is not None else list(cache.get("dimensions", [0, 1, 2]))

    if emb_key not in adata.obsm:
        raise KeyError(f"Embedding '{emb_key}' not found in adata.obsm.")

    # geometry
    h = int(cache["h"])
    w = int(cache["w"])
    s = int(cache["tile_px"])
    pixel_shape = str(cache.get("pixel_shape", "square"))
    barcodes = list(cache.get("barcodes", []))
    if not barcodes:
        raise ValueError("Cached image missing 'barcodes' order.")
    cx = np.asarray(cache.get("cx"), dtype=int)
    cy = np.asarray(cache.get("cy"), dtype=int)
    if cx.size != len(barcodes) or cy.size != len(barcodes):
        raise ValueError("Cached centers length mismatch with barcodes.")

    # collect embeddings aligned to cached barcodes order
    obs_idx_map = {str(bc): i for i, bc in enumerate(adata.obs.index.astype(str))}
    rows = np.fromiter((obs_idx_map[str(bc)] for bc in barcodes), dtype=int)
    E = np.asarray(adata.obsm[emb_key])
    # If the embedding only contains the selected dims (e.g., smoothed/equalized output),
    # treat dims as relative indices [0..len(dims)-1]. Otherwise, use absolute dim indices.
    if E.shape[1] == len(dims):
        cols_idx = np.arange(E.shape[1])
    else:
        cols_idx = np.asarray(dims, dtype=int)
    emb = E[rows][:, cols_idx].astype(np.float32)

    # normalize per channel if requested
    if normalize_channels:
        scaler = MinMaxScaler()
        emb = scaler.fit_transform(emb)

    img = np.ones((h, w, emb.shape[1]), dtype=np.float32)

    # reconstruct top-left from centers and tile size
    xi = np.clip(cx - s // 2, 0, w - 1)
    yi = np.clip(cy - s // 2, 0, h - 1)
    if pixel_shape == "circle":
        yy, xx = np.ogrid[:s, :s]
        center = (s - 1) / 2
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= (s / 2) ** 2
        dy, dx = np.nonzero(mask)
    else:
        rng = np.arange(s)
        gx, gy = np.meshgrid(rng, rng)
        dx, dy = gx.flatten(), gy.flatten()

    n = len(barcodes)
    for i in range(n):
        all_x = np.clip(xi[i] + dx, 0, w - 1)
        all_y = np.clip(yi[i] + dy, 0, h - 1)
        img[all_y, all_x] = emb[i]

    out_cache = cache if in_place else dict(cache)
    out_cache["img"] = img
    out_cache["embedding_key"] = emb_key
    out_cache["dimensions"] = list(dims)
    out_cache["total_pixels"] = int(h * w)
    out_cache.setdefault("history", []).append({
        "op": "rebuild_image",
        "embedding": emb_key,
        "dims": list(dims),
    })

    if not in_place and out_key:
        adata.uns[out_key] = out_cache

    if show:
        try:
            show_cached_image(adata, key=out_key if (not in_place and out_key) else key, channels=show_channels, cmap=show_cmap)
        except Exception:
            pass
    return out_cache


def _prepare_display_image(img: np.ndarray, channels: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return an RGB image from an arbitrary C-channel array for display.

    - If channels is provided, select those channels (expects length 1–3).
    - If C > 3 and no channels given: use first 3 channels.
    - If C == 2: pad with zeros to make 3.
    - If C == 1: repeat to 3.
    Values are clipped to [0, 1] for imshow.
    """
    h, w, c = img.shape
    if channels is not None:
        sel = list(channels)
        if len(sel) == 1:
            rgb = np.repeat(img[:, :, sel[0:1]], 3, axis=2)
        elif len(sel) == 2:
            a = img[:, :, sel[0]]
            b = img[:, :, sel[1]]
            rgb = np.stack([a, b, np.zeros_like(a)], axis=2)
        else:
            rgb = img[:, :, sel[:3]]
    else:
        if c >= 3:
            rgb = img[:, :, :3]
        elif c == 2:
            rgb = np.stack([img[:, :, 0], img[:, :, 1], np.zeros_like(img[:, :, 0])], axis=2)
        else:
            rgb = np.repeat(img, 3, axis=2)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def _auto_title_fontsize_from_figsize(figsize: Tuple[float, float]) -> int:
    """Choose a title fontsize that scales with figure size.

    Uses sqrt(area) as a simple proxy. Clamped to a sensible range.
    """
    try:
        w, h = float(figsize[0]), float(figsize[1])
    except Exception:
        w, h = 8.0, 8.0
    size_factor = (max(w, 0.1) * max(h, 0.1)) ** 0.5
    base = 12.0
    fs = base + (size_factor - 8.0) * 1.2
    return int(np.clip(round(fs), 8, 32))


def list_cached_images(adata) -> List[str]:
    """Return keys in ``adata.uns`` that look like cached embedding images."""
    keys: List[str] = []
    if not hasattr(adata, "uns"):
        return keys
    for k, v in adata.uns.items():
        try:
            if isinstance(v, dict) and "img" in v and isinstance(v["img"], np.ndarray) and v["img"].ndim == 3:
                keys.append(k)
        except Exception:
            continue
    return keys


def show_cached_image(
    adata,
    key: str = "image_plot_slic",
    *,
    channels: Optional[Sequence[int]] = None,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    fig_dpi: Optional[int] = None,
    prefer_preview: bool = True,
):
    """Display a single cached image stored in ``adata.uns[key]``.

    - For multi-channel images, defaults to RGB display. You can select
      specific channels via ``channels=[i,j,k]``.
    - For 1-channel images, uses ``cmap`` if provided; otherwise converts to RGB.
    """
    if key not in adata.uns:
        raise KeyError(f"No cached image under adata.uns['{key}']")
    cache = adata.uns[key]
    # Prefer preview with boundaries if available
    if prefer_preview and ("img_preview" in cache):
        img = np.asarray(cache.get("img_preview"))
    else:
        img = np.asarray(cache.get("img"))
    if img.ndim != 3:
        raise ValueError("Cached object does not contain a 3D image array.")

    if figsize is None:
        figsize = cache.get("figsize", (8, 8))

    plt.figure(figsize=figsize, dpi=fig_dpi)
    if img.shape[2] == 1 and cmap is not None and channels is None:
        plt.imshow(img[:, :, 0], origin="lower", cmap=cmap)
    else:
        rgb = _prepare_display_image(img, channels=channels)
        plt.imshow(rgb, origin="lower")

    if title is None:
        emb = cache.get("embedding_key", "?")
        dims = cache.get("dimensions", [])
        # resolve used channels for display title
        if channels is not None and len(channels) > 0:
            used_ch = list(channels)
        else:
            c = img.shape[2]
            if c >= 3:
                used_ch = [0, 1, 2]
            elif c == 2:
                used_ch = [0, 1]
            else:
                used_ch = [0]
        raster_dpi = cache.get("raster_dpi", cache.get("effective_dpi", None))
        total_px = cache.get("total_pixels", img.shape[0] * img.shape[1])
        title = (
            f"{key} | {img.shape[1]}x{img.shape[0]} C={img.shape[2]} "
            f"| emb={emb} "
            f"| ch=[{','.join(map(str,used_ch))}] "
            f"| dpi={raster_dpi} | px={total_px}"
        )
    fs = _auto_title_fontsize_from_figsize(figsize)
    plt.title(title, fontsize=fs)
    plt.axis("off")
    plt.show()


def show_all_cached_images(
    adata,
    keys: Optional[Sequence[str]] = None,
    *,
    max_cols: int = 3,
    channels: Optional[Sequence[int]] = None,
    cmap: Optional[str] = None,
    tight_layout: bool = True,
    prefer_preview: bool = True,
):
    """Display all cached images (or a provided subset of keys) as a grid."""
    if keys is None:
        keys = list_cached_images(adata)
    keys = list(keys)
    if len(keys) == 0:
        print("No cached images found in adata.uns.")
        return

    n = len(keys)
    cols = max(1, min(max_cols, n))
    rows = int(np.ceil(n / cols))
    # auto figure size: 4x4 per tile
    fig_w = 4 * cols
    fig_h = 4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    for i, k in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        cache = adata.uns[k]
        if prefer_preview and ("img_preview" in cache):
            img = np.asarray(cache.get("img_preview"))
        else:
            img = np.asarray(cache.get("img"))
        if img.ndim != 3:
            ax.set_visible(False)
            continue
        if img.shape[2] == 1 and cmap is not None and channels is None:
            ax.imshow(img[:, :, 0], origin="lower", cmap=cmap)
        else:
            rgb = _prepare_display_image(img, channels=channels)
            ax.imshow(rgb, origin="lower")
        emb = cache.get("embedding_key", "?")
        dims = cache.get("dimensions", [])
        raster_dpi = cache.get("raster_dpi", cache.get("effective_dpi", None))
        total_px = cache.get("total_pixels", img.shape[0] * img.shape[1])
        tile_fs = _auto_title_fontsize_from_figsize((fig_w / cols, fig_h / rows))
        ax.set_title(
            f"{k}\n{img.shape[1]}x{img.shape[0]} C={img.shape[2]} | emb={emb} | dpi={raster_dpi} | px={total_px}",
            fontsize=tile_fs,
        )
        ax.axis("off")

    # Hide any extra axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].set_visible(False)

    if tight_layout:
        plt.tight_layout()
    plt.show()
