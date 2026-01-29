# -*- coding: utf-8 -*-
# Full code with alpha-driven draw priority so less transparent (more opaque) spots render on top.

import os
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
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter, binary_dilation

import logging

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
):
    """Plot boundaries extracted from a label image.

    Uses skimage.segmentation.find_boundaries to locate pixel
    transitions so curved and donut-shaped segments render correctly.

    Parameters
    ----------
    smooth_sigma : float, optional
        Standard deviation for Gaussian smoothing of the boundary mask.
        ``0`` disables smoothing.
    """
    x_min, x_max, y_min, y_max = extent
    h, w = label_img.shape

    boundary_mask = find_boundaries(
        label_img,
        mode="outer",
        background=-1,
        connectivity=2,
    )
    render_mode = str(render or "contour").lower()
    if render_mode not in {"contour", "raster"}:
        render_mode = "contour"

    if render_mode == "raster":
        thick = int(max(1, thickness_px))
        mask = boundary_mask
        if thick > 1:
            mask = binary_dilation(mask, iterations=int(thick - 1))

        rgba = np.zeros((h, w, 4), dtype=np.float32)
        col = mcolors.to_rgba(boundary_color, alpha=float(alpha))
        rgba[mask] = col
        ax.imshow(
            rgba,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            interpolation="nearest",
        )
        return

    mask_float = boundary_mask.astype(float)
    if smooth_sigma > 0:
        mask_float = gaussian_filter(mask_float, smooth_sigma)

    contours = find_contours(mask_float, 0.5)
    if not contours:
        return

    segments = [
        np.column_stack(
            (
                contour[:, 1] / w * (x_max - x_min) + x_min,
                contour[:, 0] / h * (y_max - y_min) + y_min,
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
    pixel_perfect: bool = True,
    origin=True,
    plot_boundaries=False,  # Toggle for plotting boundaries
    boundary_method="convex_hull",  # Options: 'convex_hull', 'alphashape', 'oriented_bbox', 'concave_hull', 'pixel'
    alpha=0.3,  # Transparency of boundaries (not per-spot alpha)
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
    pixel_smoothing_sigma=0,
    # Pixel-boundary rendering controls (boundary_method='pixel' only)
    pixel_boundary_render: str = "contour",  # 'contour'|'raster'
    pixel_boundary_thickness_px: int = 1,    # used when pixel_boundary_render='raster'
    pixel_boundary_antialiased: bool = True, # used when pixel_boundary_render='contour'
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

    if len(dimensions) not in [1, 3]:
        raise ValueError("Visualization requires 1D or 3D embeddings.")

    # Extract embedding
    if embedding not in adata.obsm:
        raise ValueError(f"'{embedding}' embedding does not exist in adata.obsm.")

    # Extract spatial coordinates
    if origin:
        tiles = adata.uns["tiles"][adata.uns["tiles"]["origin"] == 1]
    else:
        tiles = adata.uns["tiles"]
    # Guard against duplicated barcodes in adata.uns['tiles'] (can cause apparent overlaps).
    if "barcode" in tiles.columns:
        tiles = tiles.copy()
        tiles["barcode"] = tiles["barcode"].astype(str)
        if tiles["barcode"].duplicated().any():
            if verbose:
                logger.warning(
                    "adata.uns['tiles'] contains duplicated barcodes; keeping first occurrence for plotting."
                )
            tiles = tiles.drop_duplicates("barcode", keep="first")

    # Prepare tile colors and merge with coordinates
    embedding_dims = np.array(adata.obsm[embedding])[:, dimensions]
    dim_names = ["dim0", "dim1", "dim2"] if len(dimensions) == 3 else ["dim0"]
    tile_colors = pd.DataFrame(embedding_dims, columns=dim_names)
    tile_colors["barcode"] = adata.obs.index.astype(str)
    coordinates_df = (
        pd.merge(tiles, tile_colors, on="barcode", how="right")
        .dropna()
        .reset_index(drop=True)
    )

    coord_mode = str(coordinate_mode or "spatial").lower()
    if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
        coord_mode = "spatial"
    if coord_mode == "auto":
        # Only auto-switch for very large datasets where array grid is almost certainly
        # a VisiumHD-style dense lattice; otherwise leave as spatial to avoid distortion.
        if (array_row_key in adata.obs.columns) and (array_col_key in adata.obs.columns) and int(adata.n_obs) >= 200_000:
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
    if segment_key is not None:
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

    if coordinates_df[["x", "y"]].isnull().any().any():
        logger.warning(
            "Found NaN values in 'x' or 'y' after conversion. Dropping these rows."
        )
        coordinates_df = coordinates_df.dropna(subset=["x", "y"])

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
                vmin = float(np.nanpercentile(v[finite], 1))
                vmax = float(np.nanpercentile(v[finite], 99))
            else:
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-9
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
        coordinates = rebalance_colors(coordinates_df, dimensions)
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

    if use_imshow:
        pts = coordinates[["x", "y"]].values
        pts_for_imshow = pts
        imshow_tile_size_shrink_eff = 1.0 if pixel_perfect else float(imshow_tile_size_shrink)
        imshow_tile_size_eff = imshow_tile_size
        if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
            # VisiumHD uses a dense square lattice in array_row/array_col space.
            # Force pitch=1 (in array units) unless the user overrides it.
            imshow_tile_size_eff = 1.0
        s_data_units0 = _estimate_tile_size_units(
            pts,
            imshow_tile_size_eff,
            imshow_scale_factor,
            imshow_tile_size_mode,
            imshow_tile_size_quantile,
            imshow_tile_size_shrink_eff,
            logger=logger,
        )
        s_data_units0 = _cap_tile_size_by_density(
            s_data_units=float(s_data_units0),
            x_range=float(max(1.0, x_max - x_min)),
            y_range=float(max(1.0, y_max - y_min)),
            n_points=int(coordinates.shape[0]),
            logger=logger,
            context="image_plot(auto-size)",
        )
        s_data_units_for_imshow = float(s_data_units0)
        if pixel_perfect:
            x_min_eff0 = float(x_min) - 0.5 * float(s_data_units0)
            x_max_eff0 = float(x_max) + 0.5 * float(s_data_units0)
            y_min_eff0 = float(y_min) - 0.5 * float(s_data_units0)
            y_max_eff0 = float(y_max) + 0.5 * float(s_data_units0)
        else:
            x_min_eff0, x_max_eff0, y_min_eff0, y_max_eff0 = float(x_min), float(x_max), float(y_min), float(y_max)
        x_range_eff0 = x_max_eff0 - x_min_eff0 if x_max_eff0 > x_min_eff0 else 1.0
        y_range_eff0 = y_max_eff0 - y_min_eff0 if y_max_eff0 > y_min_eff0 else 1.0
        figsize, fig_dpi = _resolve_figsize_dpi_for_tiles(
            figsize=figsize,
            fig_dpi=fig_dpi,
            x_range=float(x_range_eff0),
            y_range=float(y_range_eff0),
            s_data_units=float(s_data_units0),
            pixel_perfect=bool(pixel_perfect),
            n_points=int(coordinates.shape[0]),
            logger=logger,
        )

    if figsize is None:
        figsize = (10, 10)

    if fig_dpi is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)
    if verbose:
        logger.info(
            "Auto-size uses n_tiles=%d; final figsize=%s, fig_dpi=%s",
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

    if use_imshow:
        # --- Stable IMShow rendering with RGBA for per-spot alpha ---
        # 1) Estimate tile size in data units
        pts = pts_for_imshow if pts_for_imshow is not None else coordinates[["x", "y"]].values
        imshow_tile_size_eff = imshow_tile_size
        if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
            imshow_tile_size_eff = 1.0
        s_data_units = (
            float(s_data_units_for_imshow)
            if s_data_units_for_imshow is not None
            else float(
                _cap_tile_size_by_density(
                    s_data_units=float(
                        _estimate_tile_size_units(
                            pts,
                            imshow_tile_size_eff,
                            imshow_scale_factor,
                            imshow_tile_size_mode,
                            imshow_tile_size_quantile,
                            1.0 if pixel_perfect else float(imshow_tile_size_shrink),
                            logger=logger,
                        )
                    ),
                    x_range=float(max(1.0, x_max - x_min)),
                    y_range=float(max(1.0, y_max - y_min)),
                    n_points=int(len(pts)),
                    logger=logger,
                    context="image_plot",
                )
            )
        )

        # For pixel-perfect rendering, treat coordinates as tile centers and pad by half a tile.
        # This avoids half-tile empty margins when zooming/cropping.
        if pixel_perfect:
            x_min_eff = float(x_min) - 0.5 * float(s_data_units)
            x_max_eff = float(x_max) + 0.5 * float(s_data_units)
            y_min_eff = float(y_min) - 0.5 * float(s_data_units)
            y_max_eff = float(y_max) + 0.5 * float(s_data_units)
        else:
            x_min_eff, x_max_eff, y_min_eff, y_max_eff = x_min, x_max, y_min, y_max

        # 2) Output buffer size
        fig_dpi = fig.get_dpi()
        w_pixels = int(np.ceil(figsize[0] * fig_dpi))
        h_pixels = int(np.ceil(figsize[1] * fig_dpi))

        x_range = x_max_eff - x_min_eff if x_max_eff > x_min_eff else 1
        y_range = y_max_eff - y_min_eff if y_max_eff > y_min_eff else 1

        scale_raw = _raster.scale_raw_from_canvas(
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
            s = int(pitch_px)
        else:
            scale = scale_raw
            s = _round_tile_size_px(s_data_units, scale, imshow_tile_size_rounding)

        w = int(np.ceil(x_range * scale))
        h = int(np.ceil(y_range * scale))
        w, h = max(w, 1), max(h, 1)
        if verbose and not (coord_mode == "visiumhd" and pixel_perfect):
            logger.info("Rendering to an image buffer of size (w, h): (%d, %d)", int(w), int(h))

        # 3) Scale coordinates and tile size to buffer
        if pixel_perfect:
            # Avoid banker's rounding ties (e.g., *.5) which can introduce 1px pitch jitter
            # and cause apparent overlaps/gaps on regular grids.
            cx = np.floor((coordinates["x"] - x_min_eff) * scale + 0.5).astype(np.int32).values
            cy = np.floor((coordinates["y"] - y_min_eff) * scale + 0.5).astype(np.int32).values
        else:
            cx = np.rint((coordinates["x"] - x_min_eff) * scale).astype(np.int32).values
            cy = np.rint((coordinates["y"] - y_min_eff) * scale).astype(np.int32).values
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

        # 4) RGBA buffer + depth buffer
        img = np.zeros((h, w, 4), dtype=np.float32)  # start fully transparent
        value_buf = np.full((h, w), -np.inf, dtype=float)
        label_img = None
        if plot_boundaries and boundary_method == "pixel" and segment_available:
            label_img = np.full((h, w), -1, dtype=int)
            seg_codes = pd.Categorical(coordinates_df["Segment"]).codes

        # Precompute offsets for circle/square tiles (relative to the tile center).
        center_off = int(s // 2)
        if pixel_shape == "circle":
            yy, xx = np.ogrid[:s, :s]
            center = (s - 1) / 2
            mask = (xx - center) ** 2 + (yy - center) ** 2 <= (s / 2) ** 2
            oy, ox = np.nonzero(mask)
            ox = ox.astype(np.int32) - center_off
            oy = oy.astype(np.int32) - center_off
            pixels_per_tile = int(ox.size)
        else:
            tx, ty = np.meshgrid(np.arange(s, dtype=np.int32), np.arange(s, dtype=np.int32))
            ox = (tx - center_off).ravel()
            oy = (ty - center_off).ravel()
            pixels_per_tile = int(s * s)

        # Keep intermediate index arrays bounded to reduce RAM spikes and improve speed.
        target_pixels = int(os.environ.get("SPIX_PLOT_RASTER_TARGET_PIXELS", "10000000"))
        chunk_n = int(max(1, min(int(chunk_size), int(max(1, target_pixels // max(1, pixels_per_tile))))))

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
                if label_img is not None:
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
                        if label_img is not None:
                            label_img[py[mask_upd], px[mask_upd]] = seg_codes[i:end][mask_upd]

        ax.imshow(
            img,
            origin="lower",
            extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
            interpolation="nearest",
        )

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
            _plot_boundaries_from_label_image(
                ax,
                label_img,
                [x_min_eff, x_max_eff, y_min_eff, y_max_eff],
                color_for_pixel,
                boundary_linewidth,
                alpha,
                linestyle,
                pixel_smoothing_sigma,
                render=pixel_boundary_render,
                thickness_px=pixel_boundary_thickness_px,
                antialiased=pixel_boundary_antialiased,
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
                            [x_min_eff, x_max_eff, y_min_eff, y_max_eff],
                            highlight_col,
                            highlight_linewidth,
                            alpha,
                            linestyle,
                            pixel_smoothing_sigma,
                            render=pixel_boundary_render,
                            thickness_px=pixel_boundary_thickness_px,
                            antialiased=pixel_boundary_antialiased,
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
                        )

    ax.set_aspect("equal")
    ax.axis("off")
    title_text = (
        f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
        if title is None
        else title
    )
    ax.set_title(title_text, fontsize=figsize[0] * 1.5)
    if show_colorbar and len(dimensions) == 1 and not use_categorical:
        cm = plt.get_cmap(cmap if cmap is not None else "Greys")
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
    pixel_perfect: bool = True,
    origin=True,
    plot_boundaries=False,
    boundary_method="convex_hull",
    alpha=0.3,            # boundary alpha (not per-spot)
    alpha_point=1.0,      # default per-spot alpha when alpha_by is None (scatter/imshow path)
    boundary_color="black",
    boundary_linewidth=1.0,
    alpha_shape_alpha=0.1,
    aspect_ratio_threshold=3.0,
    img=None,
    img_key=None,
    library_id=None,
    crop=True,
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
    pixel_smoothing_sigma=0,
    # Pixel-boundary rendering controls (boundary_method='pixel' only)
    pixel_boundary_render: str = "contour",  # 'contour'|'raster'
    pixel_boundary_thickness_px: int = 1,    # used when pixel_boundary_render='raster'
    pixel_boundary_antialiased: bool = True, # used when pixel_boundary_render='contour'
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
    # Visium/VisiumHD grid rendering
    coordinate_mode: str = "spatial",  # 'spatial'|'array'|'visium'|'visiumhd'|'auto'
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
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

    if len(dimensions) not in [1, 3]:
        raise ValueError("Only 1 or 3 dimensions can be used for visualization.")

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

    # Extract spatial coordinates
    if origin:
        tiles = adata.uns["tiles"][adata.uns["tiles"]["origin"] == 1]
    else:
        tiles = adata.uns["tiles"]
    # Guard against duplicated barcodes in adata.uns['tiles'] (can cause apparent overlaps).
    if "barcode" in tiles.columns:
        tiles = tiles.copy()
        tiles["barcode"] = tiles["barcode"].astype(str)
        if tiles["barcode"].duplicated().any():
            if verbose:
                logger.warning(
                    "adata.uns['tiles'] contains duplicated barcodes; keeping first occurrence for plotting."
                )
            tiles = tiles.drop_duplicates("barcode", keep="first")

    # Prepare tile colors and merge with coordinates
    embedding_dims = np.array(adata.obsm[embedding])[:, dimensions]
    dim_names = ["dim0", "dim1", "dim2"] if len(dimensions) == 3 else ["dim0"]
    tile_colors = pd.DataFrame(embedding_dims, columns=dim_names)
    tile_colors["barcode"] = adata.obs.index.astype(str)
    coordinates_df = (
        pd.merge(tiles, tile_colors, on="barcode", how="right")
        .dropna()
        .reset_index(drop=True)
    )

    coord_mode = str(coordinate_mode or "spatial").lower()
    if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
        coord_mode = "spatial"
    if coord_mode == "auto":
        if (array_row_key in adata.obs.columns) and (array_col_key in adata.obs.columns) and int(adata.n_obs) >= 200_000:
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
    if segment_key is not None:
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

    if coordinates_df[["x", "y"]].isnull().any().any():
        logger.warning(
            "Found NaN values in 'x' or 'y' after conversion. Dropping these rows."
        )
        coordinates_df = coordinates_df.dropna(subset=["x", "y"])

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
                vmin = float(np.nanpercentile(v[finite], 1))
                vmax = float(np.nanpercentile(v[finite], 99))
            else:
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-9
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
        coordinates = rebalance_colors(coordinates_df, dimensions)
        if len(dimensions) == 3:
            cols = coordinates[["R", "G", "B"]].values
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

    if use_imshow:
        pts = coordinates[["x", "y"]].values
        pts_for_imshow = pts
        imshow_tile_size_shrink_eff = 1.0 if pixel_perfect else float(imshow_tile_size_shrink)
        imshow_tile_size_eff = imshow_tile_size
        if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
            imshow_tile_size_eff = 1.0
        s_data_units0 = _estimate_tile_size_units(
            pts,
            imshow_tile_size_eff,
            imshow_scale_factor,
            imshow_tile_size_mode,
            imshow_tile_size_quantile,
            imshow_tile_size_shrink_eff,
            logger=logger,
        )
        s_data_units0 = _cap_tile_size_by_density(
            s_data_units=float(s_data_units0),
            x_range=float(max(1.0, x_max - x_min)),
            y_range=float(max(1.0, y_max - y_min)),
            n_points=int(coordinates.shape[0]),
            logger=logger,
            context="image_plot_with_spatial_image(auto-size)",
        )
        s_data_units_for_imshow = float(s_data_units0)
        if pixel_perfect:
            x_min_eff0 = float(x_min) - 0.5 * float(s_data_units0)
            x_max_eff0 = float(x_max) + 0.5 * float(s_data_units0)
            y_min_eff0 = float(y_min) - 0.5 * float(s_data_units0)
            y_max_eff0 = float(y_max) + 0.5 * float(s_data_units0)
        else:
            x_min_eff0, x_max_eff0, y_min_eff0, y_max_eff0 = float(x_min), float(x_max), float(y_min), float(y_max)
        x_range_eff0 = x_max_eff0 - x_min_eff0 if x_max_eff0 > x_min_eff0 else 1.0
        y_range_eff0 = y_max_eff0 - y_min_eff0 if y_max_eff0 > y_min_eff0 else 1.0
        figsize, fig_dpi = _resolve_figsize_dpi_for_tiles(
            figsize=figsize,
            fig_dpi=fig_dpi,
            x_range=float(x_range_eff0),
            y_range=float(y_range_eff0),
            s_data_units=float(s_data_units0),
            pixel_perfect=bool(pixel_perfect),
            n_points=int(coordinates.shape[0]),
            logger=logger,
        )

    if figsize is None:
        figsize = (10, 10)

    if fig_dpi is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)
    if verbose:
        logger.info(
            "Auto-size uses n_tiles=%d; final figsize=%s, fig_dpi=%s",
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

    if img is not None:
        y_max_img, x_max_img = img.shape[:2]
        extent = [0, x_max_img, y_max_img, 0]  # Left, Right, Bottom, Top
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
            final_x_min = max(0, view_x_min)
            final_x_max = min(x_max_img, view_x_max)
            final_y_bottom_limit = min(y_max_img, view_y_max_data)
            final_y_top_limit = max(0, view_y_min_data)
            ax.set_xlim(final_x_min, final_x_max)
            ax.set_ylim(final_y_bottom_limit, final_y_top_limit)
        else:
            ax.set_xlim(0, x_max_img)
            ax.set_ylim(y_max_img, 0)
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

    if use_imshow:
        # --- Stable IMShow rendering with RGBA for per-spot alpha ---
        pts = pts_for_imshow if pts_for_imshow is not None else coordinates[["x", "y"]].values
        imshow_tile_size_eff = imshow_tile_size
        if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
            imshow_tile_size_eff = 1.0
        s_data_units = (
            float(s_data_units_for_imshow)
            if s_data_units_for_imshow is not None
            else float(
                _cap_tile_size_by_density(
                    s_data_units=float(
                        _estimate_tile_size_units(
                            pts,
                            imshow_tile_size_eff,
                            imshow_scale_factor,
                            imshow_tile_size_mode,
                            imshow_tile_size_quantile,
                            1.0 if pixel_perfect else float(imshow_tile_size_shrink),
                            logger=logger,
                        )
                    ),
                    x_range=float(max(1.0, x_max - x_min)),
                    y_range=float(max(1.0, y_max - y_min)),
                    n_points=int(len(pts)),
                    logger=logger,
                    context="image_plot_with_spatial_image",
                )
            )
        )

        if pixel_perfect:
            x_min_eff = float(x_min) - 0.5 * float(s_data_units)
            x_max_eff = float(x_max) + 0.5 * float(s_data_units)
            y_min_eff = float(y_min) - 0.5 * float(s_data_units)
            y_max_eff = float(y_max) + 0.5 * float(s_data_units)
        else:
            x_min_eff, x_max_eff, y_min_eff, y_max_eff = x_min, x_max, y_min, y_max

        fig_dpi = fig.get_dpi()
        w_pixels = int(np.ceil(figsize[0] * fig_dpi))
        h_pixels = int(np.ceil(figsize[1] * fig_dpi))

        x_range = x_max_eff - x_min_eff if x_max_eff > x_min_eff else 1
        y_range = y_max_eff - y_min_eff if y_max_eff > y_min_eff else 1

        scale_raw = _raster.scale_raw_from_canvas(
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
            s = int(pitch_px)
        else:
            scale = scale_raw
            s = _round_tile_size_px(s_data_units, scale, imshow_tile_size_rounding)

        w = int(np.ceil(x_range * scale))
        h = int(np.ceil(y_range * scale))
        w, h = max(w, 1), max(h, 1)
        if verbose:
            logger.info("Rendering to an image buffer of size (w, h): (%d, %d)", int(w), int(h))

        if pixel_perfect:
            cx = np.floor((coordinates["x"] - x_min_eff) * scale + 0.5).astype(np.int32).values
            cy = np.floor((coordinates["y"] - y_min_eff) * scale + 0.5).astype(np.int32).values
        else:
            cx = np.rint((coordinates["x"] - x_min_eff) * scale).astype(np.int32).values
            cy = np.rint((coordinates["y"] - y_min_eff) * scale).astype(np.int32).values
        if verbose:
            logger.info("Scaled tile size (in pixels): %d", int(s))

        img_rgba = np.zeros((h, w, 4), dtype=np.float32)
        value_buf = np.full((h, w), -np.inf, dtype=float)
        label_img = None
        if plot_boundaries and boundary_method == "pixel" and segment_available:
            label_img = np.full((h, w), -1, dtype=int)
            seg_codes = pd.Categorical(coordinates_df["Segment"]).codes

        center_off = int(s // 2)
        if pixel_shape == "circle":
            yy, xx = np.ogrid[:s, :s]
            center = (s - 1) / 2
            mask = (xx - center) ** 2 + (yy - center) ** 2 <= (s / 2) ** 2
            oy, ox = np.nonzero(mask)
            ox = ox.astype(np.int32) - center_off
            oy = oy.astype(np.int32) - center_off
            pixels_per_tile = int(ox.size)
        else:
            tx, ty = np.meshgrid(np.arange(s, dtype=np.int32), np.arange(s, dtype=np.int32))
            ox = (tx - center_off).ravel()
            oy = (ty - center_off).ravel()
            pixels_per_tile = int(s * s)

        target_pixels = int(os.environ.get("SPIX_PLOT_RASTER_TARGET_PIXELS", "10000000"))
        chunk_n = int(max(1, min(int(chunk_size), int(max(1, target_pixels // max(1, pixels_per_tile))))))

        for i in range(0, len(cx), chunk_n):
            end = min(i + chunk_n, len(cx))
            chunk_cx, chunk_cy = cx[i:end], cy[i:end]
            chunk_cols = cols[i:end]
            chunk_vals = vals[i:end]  # depth metric; with alpha_arr set, this is alpha
            chunk_alpha = alpha_arr[i:end] if alpha_arr is not None else None

            if not prioritize_high_values:
                all_x = np.clip((chunk_cx[:, None] + ox[None, :]).ravel(), 0, w - 1)
                all_y = np.clip((chunk_cy[:, None] + oy[None, :]).ravel(), 0, h - 1)
                cols_rep = np.repeat(chunk_cols, pixels_per_tile, axis=0)
                img_rgba[all_y, all_x, :3] = cols_rep
                if chunk_alpha is not None:
                    img_rgba[all_y, all_x, 3] = np.repeat(chunk_alpha, pixels_per_tile, axis=0)
                else:
                    img_rgba[all_y, all_x, 3] = float(alpha_point)
                if label_img is not None:
                    label_img[all_y, all_x] = np.repeat(seg_codes[i:end], pixels_per_tile, axis=0)
            else:
                for dx, dy in zip(ox, oy):
                    px = np.clip(chunk_cx + dx, 0, w - 1)
                    py = np.clip(chunk_cy + dy, 0, h - 1)
                    mask_upd = chunk_vals > value_buf[py, px]
                    if np.any(mask_upd):
                        img_rgba[py[mask_upd], px[mask_upd], :3] = chunk_cols[mask_upd]
                        img_rgba[py[mask_upd], px[mask_upd], 3] = (
                            chunk_alpha[mask_upd] if chunk_alpha is not None else float(alpha_point)
                        )
                        value_buf[py[mask_upd], px[mask_upd]] = chunk_vals[mask_upd]
                        if label_img is not None:
                            label_img[py[mask_upd], px[mask_upd]] = seg_codes[i:end][mask_upd]

        ax.imshow(
            img_rgba,
            origin="lower",
            extent=[x_min_eff, x_max_eff, y_min_eff, y_max_eff],
            interpolation="nearest",
        )
        if label_img is not None:
            _plot_boundaries_from_label_image(
                ax,
                label_img,
                [x_min_eff, x_max_eff, y_min_eff, y_max_eff],
                color_for_pixel,
                boundary_linewidth,
                alpha,
                linestyle,
                pixel_smoothing_sigma,
                render=pixel_boundary_render,
                thickness_px=pixel_boundary_thickness_px,
                antialiased=pixel_boundary_antialiased,
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
    ax.set_title(title_text, fontsize=figsize[0] * 1.5)
    if show_colorbar and len(dimensions) == 1 and not use_categorical:
        cm = plt.get_cmap(cmap if cmap is not None else "Greys")
        vmin = raw_vals.min() if raw_vals is not None else grey_vals.min()
        vmax = raw_vals.max() if raw_vals is not None else grey_vals.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    plt.show()
    # Do not return plot object to avoid auto display
