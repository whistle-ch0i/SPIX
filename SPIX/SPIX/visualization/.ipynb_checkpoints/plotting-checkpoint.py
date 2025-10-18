import numpy as np
import pandas as pd
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
from scipy.ndimage import gaussian_filter

import logging

# Import helper functions from utils.py
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


def _plot_boundaries_from_label_image(
    ax,
    label_img,
    extent,
    boundary_color,
    boundary_linewidth,
    alpha,
    linestyle,
    smooth_sigma=0,
):
    """Plot boundaries extracted from a label image.

    Uses :func:`skimage.segmentation.find_boundaries` to locate pixel
    transitions so curved and donut-shaped segments render correctly.

    Parameters
    ----------
    smooth_sigma : float, optional
        Standard deviation for Gaussian smoothing of the boundary mask.
        ``0`` disables smoothing.
    """
    x_min, x_max, y_min, y_max = extent
    h, w = label_img.shape

    # Compute a mask of all boundary pixels. ``find_boundaries`` handles
    # diagonal connections and keeps boundaries around holes, which improves
    # results for donut-shaped or highly curved segments.
    boundary_mask = find_boundaries(
        label_img,
        mode="outer",
        background=-1,
        connectivity=2,
    )
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
    )
    ax.add_collection(lc)


def image_plot(
    adata,
    dimensions=[0, 1, 2],
    embedding="X_embedding_segment",  # Use segmented embedding
    figsize=(10, 10),
    point_size=None,  # Default is None
    scaling_factor=2,
    imshow_tile_size=None,
    imshow_scale_factor=1.0,
    origin=True,
    plot_boundaries=False,  # Toggle for plotting boundaries
    boundary_method="convex_hull",  # Options: 'convex_hull', 'alphashape', 'oriented_bbox', 'concave_hull'
    alpha=0.3,  # Transparency of polygons
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
    cmap=None,  # Colormap for 1D embeddings
    pixel_smoothing_sigma=0,
    use_imshow=True,  # When True, render points as an image for speed
    pixel_shape="circle",  # 'circle' (default) or 'square'
    chunk_size=50000,  # Control chunking
    prioritize_high_values=False,  # Draw high-value pixels last
    verbose=False,  # Control verbosity
    title=None,  # Optional plot title
    show_colorbar=False,  # Show color scale bar for 1D embeddings
):
    """Visualize embeddings as a raster image with optional boundaries.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial coordinates and embeddings.
    dimensions : list of int
        Embedding dimensions to visualize (either length 1 or 3).
    embedding : str
        Key in ``adata.obsm`` where the embeddings are stored.
    figsize : tuple of float
        Size of the matplotlib figure.
    point_size : float or None
        Point size for the scatter plot when ``use_imshow`` is ``False``.
    scaling_factor : float
        Scaling factor used when automatically determining ``point_size``.
    imshow_tile_size : float or None
        Approximate tile size in data units when rendering with ``imshow``.
    imshow_scale_factor : float
        Additional scaling applied to ``imshow_tile_size``.
    origin : bool
        If ``True``, plot only tiles marked as origin.
    plot_boundaries : bool
        Whether to overlay segment boundaries.
    boundary_method : str
        Boundary calculation method (e.g. ``'convex_hull'`` or ``'pixel'``).
    alpha : float
        Transparency for boundary lines.
    boundary_color : str
        Default color for boundaries when using ``'pixel'`` mode.
    boundary_linewidth : float
        Line width of boundary outlines.
    aspect_ratio_threshold : float
        Threshold to decide if a segment is considered elongated.
    jitter : float
        Amount of jitter added to avoid collinearity.
    boundary_style : str
        Line style for drawn boundaries.
    fill_boundaries : bool
        Fill inside polygons when drawing boundaries.
    fill_color : str
        Color used when ``fill_boundaries`` is ``True``.
    fill_alpha : float
        Transparency of filled polygons.
    brighten_factor : float
        Factor used to brighten boundary colors.
    fixed_boundary_color : str or None
        Use a fixed color for all boundaries instead of sampling from tiles.
    highlight_segments : list or str or None
        Segment label or list of labels to highlight.
    dim_other_segments : float
        Factor to grey and dim colors of non-highlighted segments when
        ``highlight_segments`` is provided. Must be between 0 and 1.
    dim_to_grey : bool
        If ``True``, convert dimmed segments to greyscale. If ``False``, keep
        their original hues while dimming.
    cmap : str or matplotlib colormap, optional
        Colormap used when ``len(dimensions) == 1``. If ``None`` (default), a
        greyscale representation is used.
    highlight_linewidth : float or None
        Line width for highlighted segment boundaries. Defaults to twice
        ``boundary_linewidth`` when ``None``.
    highlight_boundary_color : str or None
        Color for highlighted boundaries. If ``None``, uses the underlying
        pixel colors brightened by ``highlight_brighten_factor``.
    highlight_brighten_factor : float
        Additional brightening applied to highlighted segments. Values greater
        than 1 increase brightness.
    pixel_smoothing_sigma : float
        Gaussian smoothing applied to pixel boundaries when ``boundary_method``
        is ``'pixel'``.
    use_imshow : bool
        Render points as an image for speed when ``True``. The pixel boundary
        method can still be used when ``False``.
    pixel_shape : str
        Shape of pixels when ``use_imshow`` is ``True``. Options are ``'square'``
        or ``'circle'``.
    chunk_size : int
        Number of tiles processed per chunk in ``imshow`` mode.
    prioritize_high_values : bool
        If ``True``, use a depth buffer so that higher-valued pixels
        overwrite lower-valued ones when they overlap.
    verbose : bool
        If ``True``, emit informational log messages.
    title : str or None
        Custom plot title. Defaults to ``embedding`` and ``dimensions`` when ``None``.
    show_colorbar : bool
        Display a color scale bar when ``len(dimensions) == 1``.
    """
    # Create a local logger
    logger = logging.getLogger("image_plot")
    logger.setLevel(logging.INFO if verbose else logging.ERROR)
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

    # Prepare tile colors and merge with coordinates
    embedding_dims = np.array(adata.obsm[embedding])[:, dimensions]
    dim_names = ["dim0", "dim1", "dim2"] if len(dimensions) == 3 else ["dim0"]
    tile_colors = pd.DataFrame(embedding_dims, columns=dim_names)
    tile_colors["barcode"] = adata.obs.index
    coordinates_df = (
        pd.merge(tiles, tile_colors, on="barcode", how="right")
        .dropna()
        .reset_index(drop=True)
    )

    if "Segment" in adata.obs.columns:
        coordinates_df["Segment"] = coordinates_df["barcode"].map(adata.obs["Segment"])
        if coordinates_df["Segment"].isnull().any():
            missing = coordinates_df["Segment"].isnull().sum()
            logger.warning(
                f"'Segment' label for {missing} barcodes is missing. These rows will be dropped."
            )
            coordinates_df = coordinates_df.dropna(subset=["Segment"])

    coordinates_df["x"] = pd.to_numeric(coordinates_df["x"], errors="coerce")
    coordinates_df["y"] = pd.to_numeric(coordinates_df["y"], errors="coerce")

    if coordinates_df[["x", "y"]].isnull().any().any():
        logger.warning(
            "Found NaN values in 'x' or 'y' after conversion. Dropping these rows."
        )
        coordinates_df = coordinates_df.dropna(subset=["x", "y"])

    # Prepare values used for depth buffering. ``vals`` is a scalar per tile
    # derived from the provided embedding dimensions.
    if len(dimensions) == 1:
        vals = coordinates_df[dim_names[0]].to_numpy()
    else:
        vals = coordinates_df[dim_names].sum(axis=1).to_numpy()

    # Capture original range for colorbar before normalization
    if len(dimensions) == 1:
        raw_vals = coordinates_df[dim_names[0]].values
    else:
        raw_vals = None

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

    fig, ax = plt.subplots(figsize=figsize)
    linestyle_options = {"solid": "-", "dashed": "--", "dotted": ":"}
    linestyle = linestyle_options.get(boundary_style, "-")

    color_for_pixel = (
        fixed_boundary_color if fixed_boundary_color is not None else boundary_color
    )

    if "Segment" in adata.obs.columns:
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
        # --- STABLE IMShow RENDERING LOGIC ---
        # This logic scales data to a fixed-size buffer to prevent memory issues and artifacts.

        # 1. Calculate tile size in original data coordinates (if not provided)
        s_data_units = 1
        if imshow_tile_size is None:
            pts = coordinates[["x", "y"]].values
            if len(pts) > 1:
                sample_size = min(len(pts), 100000)
                sample_indices = np.random.choice(len(pts), sample_size, replace=False)
                tree = KDTree(pts[sample_indices])
                d, _ = tree.query(pts[sample_indices], k=2)
                nn = d[:, 1][d[:, 1] > 0]
                if len(nn) > 0:
                    s_data_units = np.median(nn)
                    logger.info(
                        f"Auto-calculated tile size (in data units): {s_data_units:.2f}"
                    )
        else:
            s_data_units = imshow_tile_size

        s_data_units *= imshow_scale_factor
        s_data_units = max(0.1, s_data_units)

        # 2. Determine output image buffer size
        dpi = fig.get_dpi()
        w_pixels = int(np.ceil(figsize[0] * dpi))
        h_pixels = int(np.ceil(figsize[1] * dpi))

        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        # Calculate scale to fit data into the figure dimensions while preserving aspect ratio
        scale = (
            min(w_pixels / x_range, h_pixels / y_range)
            if x_range > 0 and y_range > 0
            else 1
        )

        w = int(np.ceil(x_range * scale))
        h = int(np.ceil(y_range * scale))
        w, h = max(w, 1), max(h, 1)
        logger.info(f"Rendering to an image buffer of size (w, h): ({w}, {h})")

        # 3. Scale coordinates and tile size to fit the image buffer
        xi = ((coordinates["x"] - x_min) * scale).astype(int).values
        yi = ((coordinates["y"] - y_min) * scale).astype(int).values
        s = int(np.ceil(s_data_units * scale))
        s = max(1, s)
        logger.info(f"Scaled tile size (in pixels): {s}")

        # 4. Create image buffer and fill it
        img = np.ones((h, w, 3), dtype=np.float32)
        value_buf = np.full((h, w), -np.inf, dtype=float)
        label_img = None
        if (
            plot_boundaries
            and boundary_method == "pixel"
            and "Segment" in coordinates_df.columns
        ):
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
            chunk_cols = cols[i:end]
            chunk_vals = vals[i:end]

            if pixel_shape == "circle":
                for dx, dy in zip(circle_x_offsets, circle_y_offsets):
                    cx = np.clip(chunk_xi + dx, 0, w - 1)
                    cy = np.clip(chunk_yi + dy, 0, h - 1)
                    if prioritize_high_values:
                        mask = chunk_vals > value_buf[cy, cx]
                        if np.any(mask):
                            img[cy[mask], cx[mask]] = chunk_cols[mask]
                            value_buf[cy[mask], cx[mask]] = chunk_vals[mask]
                            if label_img is not None:
                                label_img[cy[mask], cx[mask]] = seg_codes[i:end][mask]
                    else:
                        img[cy, cx] = chunk_cols
                        if label_img is not None:
                            label_img[cy, cx] = seg_codes[i:end]
            else:
                chunk_all_x = chunk_xi[:, None, None] + tile_x_offsets
                chunk_all_y = chunk_yi[:, None, None] + tile_y_offsets

                chunk_all_x = np.clip(chunk_all_x, 0, w - 1)
                chunk_all_y = np.clip(chunk_all_y, 0, h - 1)
                for dx in range(s):
                    for dy in range(s):
                        cx = chunk_all_x[:, dy, dx]
                        cy = chunk_all_y[:, dy, dx]
                        if prioritize_high_values:
                            mask = chunk_vals > value_buf[cy, cx]
                            if np.any(mask):
                                img[cy[mask], cx[mask]] = chunk_cols[mask]
                                value_buf[cy[mask], cx[mask]] = chunk_vals[mask]
                                if label_img is not None:
                                    label_img[cy[mask], cx[mask]] = seg_codes[i:end][
                                        mask
                                    ]
                        else:
                            img[cy, cx] = chunk_cols
                            if label_img is not None:
                                label_img[cy, cx] = seg_codes[i:end]

        ax.imshow(
            img,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            interpolation="nearest",
        )
        if label_img is not None:
            _plot_boundaries_from_label_image(
                ax,
                label_img,
                [x_min, x_max, y_min, y_max],
                color_for_pixel,
                boundary_linewidth,
                alpha,
                linestyle,
                pixel_smoothing_sigma,
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
                        )

    else:
        # ---Automatic point size calculation for scatter plots ---
        if point_size is None:
            # Heuristic based on point density.
            num_points = len(coordinates)
            # Aims to make point size smaller as point count increases.
            # The formula is empirical and may need tuning.
            # Default scaling_factor=10, 72 dpi assumed for figure.
            base_size = (figsize[0] * figsize[1] * 100**2) / (num_points + 1)
            point_size = (scaling_factor / 5.0) * base_size
            point_size = np.clip(point_size, 0.01, 200)  # Clamp to reasonable values

        ax.scatter(
            x=coordinates["x"],
            y=coordinates["y"],
            s=point_size,
            c=cols,
            marker="o" if pixel_shape == "circle" else "s",
            linewidths=0,
        )

        if (
            plot_boundaries
            and boundary_method == "pixel"
            and "Segment" in coordinates_df.columns
        ):
            dpi = fig.get_dpi()
            w_pixels = int(np.ceil(figsize[0] * dpi))
            h_pixels = int(np.ceil(figsize[1] * dpi))

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

            s_data_units = 1
            if imshow_tile_size is None:
                pts = coordinates[["x", "y"]].values
                if len(pts) > 1:
                    sample_size = min(len(pts), 100000)
                    sample_indices = np.random.choice(
                        len(pts), sample_size, replace=False
                    )
                    tree_tmp = KDTree(pts[sample_indices])
                    d, _ = tree_tmp.query(pts[sample_indices], k=2)
                    nn = d[:, 1][d[:, 1] > 0]
                    if len(nn) > 0:
                        s_data_units = np.median(nn)
            else:
                s_data_units = imshow_tile_size

            s_data_units *= imshow_scale_factor
            s_data_units = max(0.1, s_data_units)

            s = int(np.ceil(s_data_units * scale))
            s = max(1, s)

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
                        )

    ax.set_aspect("equal")
    ax.axis("off")
    title_text = (
        f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
        if title is None
        else title
    )
    ax.set_title(title_text, fontsize=figsize[0] * 1.5)
    if show_colorbar and len(dimensions) == 1:
        cm = plt.get_cmap(cmap if cmap is not None else "Greys")
        vmin = raw_vals.min() if raw_vals is not None else grey_vals.min()
        vmax = raw_vals.max() if raw_vals is not None else grey_vals.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # Define linestyle based on boundary_style
    linestyle_options = {"solid": "-", "dashed": "--", "dotted": ":"}
    linestyle = linestyle_options.get(boundary_style, "-")

    if (
        "Segment" in adata.obs.columns
        and plot_boundaries
        and boundary_method != "pixel"
    ):
        # Build KDTree for efficient nearest-neighbor search
        tree = KDTree(coordinates[["x", "y"]].values)

        # Group by segments
        segments = coordinates_df["Segment"].unique()

        for seg in segments:
            is_highlight = seg in highlight_segments
            seg_tiles = coordinates_df[coordinates_df["Segment"] == seg]
            if len(seg_tiles) < 3:
                logger.info(
                    f"Segment {seg} has fewer than 3 points. Skipping boundary plotting."
                )
                continue

            points = seg_tiles[["x", "y"]].values

            # Check for collinearity
            unique_points = np.unique(points, axis=0)
            if unique_points.shape[0] < 3 or is_collinear(unique_points):
                logger.warning(
                    f"Segment {seg} points are collinear or insufficient. Using LineString as boundary."
                )
                line = LineString(unique_points)

                # Clip boundary to plot bounds
                clipped_line = line.intersection(plot_boundary_polygon)
                if clipped_line.is_empty:
                    continue

                # Prepare to collect points from the clipped geometry
                line_points = []

                # Handle different geometry types
                if isinstance(clipped_line, (LineString)):
                    x, y = clipped_line.xy
                    line_points = np.column_stack((x, y))
                elif isinstance(clipped_line, (MultiLineString, GeometryCollection)):
                    for geom in clipped_line.geoms:
                        if isinstance(geom, LineString):
                            x, y = geom.xy
                            # Append points from each LineString segment
                            line_points.extend(np.column_stack((x, y)))
                    line_points = np.array(line_points)
                else:
                    logger.warning(
                        f"Unsupported geometry type after clipping: {type(clipped_line)}"
                    )
                    continue

                # Now, find nearest neighbors for each boundary point
                _, idx = tree.query(line_points)

                # Determine boundary colors as before
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

                # Create segments for LineCollection
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

            # Calculate boundaries
            try:
                # Calculate dynamic alpha based on average distance between points
                if boundary_method == "alphashape":
                    avg_dist = np.mean(
                        np.linalg.norm(points - points.mean(axis=0), axis=1)
                    )
                    alpha_value = (1.0 / avg_dist) * 1.5  # Adjust as needed
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
                    # Use small alpha for more concave boundary
                    avg_dist = np.mean(
                        np.linalg.norm(points - points.mean(axis=0), axis=1)
                    )
                    alpha_value = (1.0 / avg_dist) * 1.0  # Adjust as needed
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

                # Validate and fix boundary
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

                # Smooth the boundary
                concave_hull = smooth_polygon(concave_hull, buffer_dist=0.01)

            except Exception as e:
                logger.error(f"Failed to calculate boundary for Segment {seg}: {e}")
                continue

            # Clip boundary to plot bounds
            try:
                clipped_polygon = concave_hull.intersection(plot_boundary_polygon)
                if clipped_polygon.is_empty:
                    logger.warning(f"Clipped boundary for Segment {seg} is empty.")
                    continue
            except Exception as e:
                logger.error(f"Failed to clip boundary for Segment {seg}: {e}")
                continue

            # Create patches from clipped polygons
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

            # Set boundary color for the current segment
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

                # Convert boundary coordinates to points
                boundary_points = np.column_stack((x, y))

                # Find nearest neighbors for each boundary point
                _, idx = tree.query(boundary_points)

                if fixed_boundary_color is not None:
                    # Use fixed color
                    boundary_colors = np.array(
                        [fixed_boundary_color] * len(boundary_points)
                    )
                else:
                    if len(dimensions) == 3:
                        # Extract RGB colors
                        boundary_colors = coordinates.iloc[idx][["R", "G", "B"]].values
                    else:
                        # Extract grayscale values and apply colormap if provided
                        grey = coordinates.iloc[idx]["Grey"].values
                        if cmap is None:
                            boundary_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        else:
                            cm = plt.get_cmap(cmap)
                            boundary_colors = cm(grey)[:, :3]

                    # Normalize and brighten colors
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

                # Create segments for LineCollection
                segments_lines = np.array(
                    [
                        boundary_points[i : i + 2]
                        for i in range(len(boundary_points) - 1)
                    ]
                )

                # Assign color to each segment (use color of the starting point)
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
                    # Filling the boundary with pixel-based colors is complex, so use a single color for filling.
                    patch = patches.Polygon(
                        np.array(poly.exterior.coords),
                        closed=True,
                        facecolor=fill_color,
                        edgecolor=None,
                        alpha=fill_alpha,
                    )
                    ax.add_patch(patch)

        # Set plot boundaries
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Show plot
    plt.show()
    # Do not return plot object to avoid auto display


def image_plot_with_spatial_image(
    adata,
    dimensions=[0, 1, 2],
    embedding="X_embedding_segment",
    figsize=(10, 10),
    point_size=None,
    scaling_factor=2,
    img_scaling_factor=None,
    imshow_tile_size=None,
    imshow_scale_factor=1.0,
    origin=True,
    plot_boundaries=False,
    boundary_method="convex_hull",
    alpha=0.3,
    alpha_point=1,
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
    cmap=None,  # Colormap for 1D embeddings
    pixel_smoothing_sigma=0,
    use_imshow=True,  # When True, render points as an image for speed
    pixel_shape="circle",  # 'circle' (default) or 'square'
    chunk_size=50000,  # Control chunking
    prioritize_high_values=False,  # Draw high-value pixels last
    verbose=False,  # Control verbosity
    title=None,  # Optional plot title
    show_colorbar=False,  # Show color scale bar for 1D embeddings
):
    """
    Visualize embeddings as an image with optional segment boundaries and background image overlay.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings and spatial coordinates.
    dimensions : list of int
        List of dimensions to use for visualization (1 or 3 dimensions).
    embedding : str
        Key in adata.obsm where the embeddings are stored.
    figsize : tuple of float
        Figure size in inches (width, height).
    point_size : float or None
        Size of the points in the scatter plot. If None, it will be automatically determined.
    scaling_factor : float
        Scaling factor for point size calculation if point_size is not provided.
    img_scaling_factor : float or None
        Additional scaling factor for spatial coordinates.
    origin : bool
        Whether to use only origin tiles.
    plot_boundaries : bool
        Whether to plot segment boundaries.
    boundary_method : str
        Method to compute boundaries ('convex_hull', 'alpha_shape', 'oriented_bbox', 'concave_hull', 'pixel').
    alpha : float
        Transparency for the boundary polygons.
    alpha_point : float
        Transparency for the scatter points.
    boundary_color : str
        Color for the boundaries.
    boundary_linewidth : float
        Line width for the boundaries.
    alpha_shape_alpha : float
        Alpha value for alpha shapes (only relevant if using 'alpha_shape').
    aspect_ratio_threshold : float
        Threshold to determine if a segment is considered thin and long based on aspect ratio.
    img : np.ndarray or None
        Background image to overlay.
    img_key : str or None
        Key to select the image from spatial data.
    library_id : str or None
        Library ID for spatial data.
    crop : bool
        Whether to crop the image to the extent of the spots.
    alpha_img : float
        Transparency for the background image.
    bw : bool
        Whether to convert the background image to grayscale.
    jitter : float
        Amount of jitter to add to points to avoid collinearity.
    tol_collinear : float
        Tolerance for exact collinearity.
    tol_almost_collinear : float
        Tolerance for near collinearity.
    boundary_style : str, optional
        Line style for the boundaries. Options are 'solid', 'dashed', 'dotted'.
    fill_boundaries : bool, optional
        Whether to fill the inside of the boundaries. Default is False.
    fill_color : str, optional
        Fill color for the boundaries. Default is 'blue'.
    fill_alpha : float, optional
        Fill transparency for the boundaries. Default is 0.1.
    brighten_factor : float, optional
        Factor for brightening colors. Default is 1.3.
    fixed_boundary_color : str, optional
        Fixed color for the boundaries. If None, colors are adjusted based on pixel colors.
    highlight_segments : list or str or None
        Segment label or list of labels to highlight.
    dim_other_segments : float
        Factor to grey and dim colors of non-highlighted segments when
        ``highlight_segments`` is provided. Must be between 0 and 1.
    dim_to_grey : bool
        If ``True``, convert dimmed segments to greyscale. If ``False``, keep
        their original hues while dimming.
    cmap : str or matplotlib colormap, optional
        Colormap used when ``len(dimensions) == 1``. If ``None`` (default), a
        greyscale representation is used.
    highlight_linewidth : float or None
        Line width for highlighted segment boundaries. Defaults to twice
        ``boundary_linewidth`` when ``None``.
    highlight_boundary_color : str or None
        Color for highlighted boundaries. If ``None``, uses the underlying
        pixel colors brightened by ``highlight_brighten_factor``.
    highlight_brighten_factor : float
        Additional brightening applied to highlighted segments. Values greater
        than 1 increase brightness.
    pixel_smoothing_sigma : float, optional
        Gaussian smoothing applied to pixel boundaries when ``boundary_method`` is ``'pixel'``.
    use_imshow : bool
        Render points as an image for speed when ``True``. The pixel boundary
        method can still be used when ``False``.
    pixel_shape : str
        Shape of pixels when ``use_imshow`` is ``True``. Options are ``'square'``
        or ``'circle'``.
    chunk_size : int
        Number of tiles processed per chunk in ``imshow`` mode.
    prioritize_high_values : bool
        If ``True``, use a depth buffer so that higher-valued pixels
        overwrite lower-valued ones when they overlap.
    verbose : bool, optional
        If True, display warnings and informational messages. Default is False.
    title : str or None
        Custom plot title. Defaults to ``embedding`` and ``dimensions`` when ``None``.
    show_colorbar : bool
        Display a color scale bar when ``len(dimensions) == 1``.
    """
    # Create a local logger
    logger = logging.getLogger("image_plot_with_spatial_image")
    logger.setLevel(logging.INFO if verbose else logging.ERROR)
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

    # Prepare tile colors and merge with coordinates
    embedding_dims = np.array(adata.obsm[embedding])[:, dimensions]
    dim_names = ["dim0", "dim1", "dim2"] if len(dimensions) == 3 else ["dim0"]
    tile_colors = pd.DataFrame(embedding_dims, columns=dim_names)
    tile_colors["barcode"] = adata.obs.index
    coordinates_df = (
        pd.merge(tiles, tile_colors, on="barcode", how="right")
        .dropna()
        .reset_index(drop=True)
    )

    if "Segment" in adata.obs.columns:
        coordinates_df["Segment"] = coordinates_df["barcode"].map(adata.obs["Segment"])
        if coordinates_df["Segment"].isnull().any():
            missing = coordinates_df["Segment"].isnull().sum()
            logger.warning(
                f"'Segment' label for {missing} barcodes is missing. These rows will be dropped."
            )
            coordinates_df = coordinates_df.dropna(subset=["Segment"])

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

    if len(dimensions) == 1:
        vals = coordinates_df[dim_names[0]].to_numpy()
    else:
        vals = coordinates_df[dim_names].sum(axis=1).to_numpy()

    # Capture original range for colorbar before normalization
    if len(dimensions) == 1:
        raw_vals = coordinates_df[dim_names[0]].values
    else:
        raw_vals = None

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

    fig, ax = plt.subplots(figsize=figsize)
    linestyle_options = {"solid": "-", "dashed": "--", "dotted": ":"}
    linestyle = linestyle_options.get(boundary_style, "-")
    color_for_pixel = (
        fixed_boundary_color if fixed_boundary_color is not None else boundary_color
    )
    if "Segment" in adata.obs.columns:
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

        # Display the image
        ax.imshow(img, extent=extent, alpha=alpha_img)

        if crop:
            # Calculate padding based on the range of spot coordinates
            x_range_spots = x_max - x_min
            y_range_spots = (
                y_max - y_min
            )  # y_max is larger, y_min is smaller (image coordinates often inverted)

            crop_padding_factor = 0.05

            x_padding = x_range_spots * crop_padding_factor
            y_padding = y_range_spots * crop_padding_factor

            # Calculate new view limits with padding
            view_x_min = x_min - x_padding
            view_x_max = x_max + x_padding
            view_y_min_data = (
                y_min - y_padding
            )  # This will be the "top" of the view in data coords
            view_y_max_data = (
                y_max + y_padding
            )  # This will be the "bottom" of the view in data coords

            # Ensure the padded view does not exceed image boundaries
            final_x_min = max(0, view_x_min)
            final_x_max = min(x_max_img, view_x_max)
            # For set_ylim(bottom, top) with inverted image y-axis:
            # final_y_bottom_limit corresponds to the larger y-coordinate value (view_y_max_data)
            # final_y_top_limit corresponds to the smaller y-coordinate value (view_y_min_data)
            final_y_bottom_limit = min(y_max_img, view_y_max_data)
            final_y_top_limit = max(0, view_y_min_data)

            ax.set_xlim(final_x_min, final_x_max)
            ax.set_ylim(
                final_y_bottom_limit, final_y_top_limit
            )  # (bottom, top for inverted axis)
        else:
            # Show the full image
            ax.set_xlim(0, x_max_img)
            ax.set_ylim(y_max_img, 0)
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

    if use_imshow:
        # --- STABLE IMShow RENDERING LOGIC ---
        # This logic scales data to a fixed-size buffer to prevent memory issues and artifacts.

        # 1. Calculate tile size in original data coordinates (if not provided)
        s_data_units = 1
        if imshow_tile_size is None:
            pts = coordinates[["x", "y"]].values
            if len(pts) > 1:
                sample_size = min(len(pts), 100000)
                sample_indices = np.random.choice(len(pts), sample_size, replace=False)
                tree = KDTree(pts[sample_indices])
                d, _ = tree.query(pts[sample_indices], k=2)
                nn = d[:, 1][d[:, 1] > 0]
                if len(nn) > 0:
                    s_data_units = np.median(nn)
                    logger.info(
                        f"Auto-calculated tile size (in data units): {s_data_units:.2f}"
                    )
        else:
            s_data_units = imshow_tile_size

        s_data_units *= imshow_scale_factor
        s_data_units = max(0.1, s_data_units)

        # 2. Determine output image buffer size
        dpi = fig.get_dpi()
        w_pixels = int(np.ceil(figsize[0] * dpi))
        h_pixels = int(np.ceil(figsize[1] * dpi))

        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        # Calculate scale to fit data into the figure dimensions while preserving aspect ratio
        scale = (
            min(w_pixels / x_range, h_pixels / y_range)
            if x_range > 0 and y_range > 0
            else 1
        )

        w = int(np.ceil(x_range * scale))
        h = int(np.ceil(y_range * scale))
        w, h = max(w, 1), max(h, 1)
        logger.info(f"Rendering to an image buffer of size (w, h): ({w}, {h})")

        # 3. Scale coordinates and tile size to fit the image buffer
        xi = ((coordinates["x"] - x_min) * scale).astype(int).values
        yi = ((coordinates["y"] - y_min) * scale).astype(int).values
        s = int(np.ceil(s_data_units * scale))
        s = max(1, s)
        logger.info(f"Scaled tile size (in pixels): {s}")

        # 4. Create image buffer and fill it
        img = np.zeros((h, w, 4), dtype=np.float32)
        value_buf = np.full((h, w), -np.inf, dtype=float)
        label_img = None
        if (
            plot_boundaries
            and boundary_method == "pixel"
            and "Segment" in coordinates_df.columns
        ):
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
            chunk_cols = cols[i:end]
            chunk_vals = vals[i:end]

            if pixel_shape == "circle":
                for dx, dy in zip(circle_x_offsets, circle_y_offsets):
                    cx = np.clip(chunk_xi + dx, 0, w - 1)
                    cy = np.clip(chunk_yi + dy, 0, h - 1)
                    if prioritize_high_values:
                        mask = chunk_vals > value_buf[cy, cx]
                        if np.any(mask):
                            img[cy[mask], cx[mask], :3] = chunk_cols[mask]
                            img[cy[mask], cx[mask], 3] = alpha_point
                            value_buf[cy[mask], cx[mask]] = chunk_vals[mask]
                            if label_img is not None:
                                label_img[cy[mask], cx[mask]] = seg_codes[i:end][mask]
                    else:
                        img[cy, cx, :3] = chunk_cols
                        img[cy, cx, 3] = alpha_point
                        if label_img is not None:
                            label_img[cy, cx] = seg_codes[i:end]
            else:
                chunk_all_x = chunk_xi[:, None, None] + tile_x_offsets
                chunk_all_y = chunk_yi[:, None, None] + tile_y_offsets

                chunk_all_x = np.clip(chunk_all_x, 0, w - 1)
                chunk_all_y = np.clip(chunk_all_y, 0, h - 1)
                for dx in range(s):
                    for dy in range(s):
                        cx = chunk_all_x[:, dy, dx]
                        cy = chunk_all_y[:, dy, dx]
                        if prioritize_high_values:
                            mask = chunk_vals > value_buf[cy, cx]
                            if np.any(mask):
                                img[cy[mask], cx[mask], :3] = chunk_cols[mask]
                                img[cy[mask], cx[mask], 3] = alpha_point
                                value_buf[cy[mask], cx[mask]] = chunk_vals[mask]
                                if label_img is not None:
                                    label_img[cy[mask], cx[mask]] = seg_codes[i:end][
                                        mask
                                    ]
                        else:
                            img[cy, cx, :3] = chunk_cols
                            img[cy, cx, 3] = alpha_point
                            if label_img is not None:
                                label_img[cy, cx] = seg_codes[i:end]

        ax.imshow(
            img,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            interpolation="nearest",
        )
        if label_img is not None:
            _plot_boundaries_from_label_image(
                ax,
                label_img,
                [x_min, x_max, y_min, y_max],
                color_for_pixel,
                boundary_linewidth,
                alpha,
                linestyle,
                pixel_smoothing_sigma,
            )

    else:
        # ---Automatic point size calculation for scatter plots ---
        if point_size is None:
            # Heuristic based on point density.
            num_points = len(coordinates)
            # Aims to make point size smaller as point count increases.
            # The formula is empirical and may need tuning.
            # Default scaling_factor=10, 72 dpi assumed for figure.
            base_size = (figsize[0] * figsize[1] * 100**2) / (num_points + 1)
            point_size = (scaling_factor / 5.0) * base_size
            point_size = np.clip(point_size, 0.01, 200)  # Clamp to reasonable values

        ax.scatter(
            x=coordinates["x"],
            y=coordinates["y"],
            s=point_size,
            c=cols,
            marker="o" if pixel_shape == "circle" else "s",
            linewidths=0,
            alpha=alpha_point,
        )

        if (
            plot_boundaries
            and boundary_method == "pixel"
            and "Segment" in coordinates_df.columns
        ):
            dpi = fig.get_dpi()
            w_pixels = int(np.ceil(figsize[0] * dpi))
            h_pixels = int(np.ceil(figsize[1] * dpi))

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

            s_data_units = 1
            if imshow_tile_size is None:
                pts = coordinates[["x", "y"]].values
                if len(pts) > 1:
                    sample_size = min(len(pts), 100000)
                    sample_indices = np.random.choice(
                        len(pts), sample_size, replace=False
                    )
                    tree_tmp = KDTree(pts[sample_indices])
                    d, _ = tree_tmp.query(pts[sample_indices], k=2)
                    nn = d[:, 1][d[:, 1] > 0]
                    if len(nn) > 0:
                        s_data_units = np.median(nn)
            else:
                s_data_units = imshow_tile_size

            s_data_units *= imshow_scale_factor
            s_data_units = max(0.1, s_data_units)

            s = int(np.ceil(s_data_units * scale))
            s = max(1, s)

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
                        )

    # Define linestyle based on boundary_style
    linestyle_options = {"solid": "-", "dashed": "--", "dotted": ":"}
    linestyle = linestyle_options.get(boundary_style, "-")

    if (
        "Segment" in adata.obs.columns
        and plot_boundaries
        and boundary_method != "pixel"
    ):
        # Build KDTree for efficient nearest-neighbor search
        tree = KDTree(coordinates[["x", "y"]].values)

        # Group by segments
        segments = coordinates_df["Segment"].unique()

        for seg in segments:
            is_highlight = seg in highlight_segments
            seg_tiles = coordinates_df[coordinates_df["Segment"] == seg]
            if len(seg_tiles) < 3:
                logger.info(
                    f"Segment {seg} has fewer than 3 points. Skipping boundary plotting."
                )
                continue

            points = seg_tiles[["x", "y"]].values

            # Check for collinearity or insufficient points for a polygon
            unique_points = np.unique(
                points, axis=0
            )  # `points` are from current segment `seg`

            # Condition for using LineString boundary:
            # - Fewer than 3 unique points (so cannot form a polygon).
            # - Or, points are collinear (cannot form a non-degenerate polygon).
            # Note: `len(seg_tiles)` is at least 3 here due to an earlier check,
            # so `unique_points.shape[0]` will be at least 1.
            if unique_points.shape[0] < 3 or is_collinear(
                unique_points
            ):  # `is_collinear` is an assumed helper function
                logger.warning(
                    f"Segment {seg} has {unique_points.shape[0]} unique point(s) or points are collinear. "
                    f"Attempting to draw LineString as boundary."
                )

                # LineString can be constructed from 1 point (0-length) or 2+ points.
                line = LineString(unique_points)

                # Clip the segment's line representation against the overall plot boundary
                clipped_geometry = line.intersection(plot_boundary_polygon)

                if clipped_geometry.is_empty:
                    logger.info(
                        f"Segment {seg}: Collinear line/points do not intersect plot boundary. No boundary drawn."
                    )
                    continue  # to next segment

                # Collect all LineString parts from the clipped geometry
                lines_to_draw = []
                if isinstance(clipped_geometry, LineString):
                    lines_to_draw.append(clipped_geometry)
                elif isinstance(clipped_geometry, MultiLineString):
                    # .geoms of MultiLineString should yield LineStrings
                    lines_to_draw.extend(
                        list(
                            geom
                            for geom in clipped_geometry.geoms
                            if isinstance(geom, LineString)
                        )
                    )
                elif isinstance(clipped_geometry, GeometryCollection):
                    for geom_part in clipped_geometry.geoms:
                        if isinstance(geom_part, LineString):
                            lines_to_draw.append(geom_part)
                # Other geometry types (e.g., Point, MultiPoint) resulting from intersection are not drawn as lines.

                if not lines_to_draw:
                    logger.info(
                        f"Segment {seg}: Collinear intersection with plot boundary "
                        f"resulted in non-LineString geometry (type: {type(clipped_geometry).__name__}) "
                        f"or no valid LineString parts. No boundary drawn."
                    )
                    continue  # to next segment

                # Process each LineString part to generate segments for LineCollection
                all_line_segments_for_lc = []
                all_colors_for_lc = []

                for line_part in lines_to_draw:
                    # Skip if the line part is essentially a point or empty
                    if (
                        line_part.is_empty or line_part.length < 1e-9
                    ):  # Using a small tolerance for float comparison
                        continue

                    # Extract coordinates from the LineString part; .xy is valid here
                    x_coords, y_coords = line_part.xy
                    boundary_line_points = np.column_stack((x_coords, y_coords))

                    if (
                        len(boundary_line_points) < 2
                    ):  # Need at least two points to form a line segment
                        continue

                    # Determine colors for these boundary points (based on nearest data points)
                    _, nearest_data_indices = tree.query(boundary_line_points)

                    point_colors_on_boundary: np.ndarray
                    if fixed_boundary_color is not None:
                        # Use fixed color. Matplotlib's LineCollection can handle color strings.
                        point_colors_on_boundary = np.array(
                            [fixed_boundary_color] * len(boundary_line_points)
                        )
                    else:
                        if len(dimensions) == 3:  # RGB case
                            point_colors_on_boundary = coordinates.iloc[
                                nearest_data_indices
                            ][["R", "G", "B"]].values
                        else:  # Grayscale case
                            grey_values = coordinates.iloc[nearest_data_indices][
                                "Grey"
                            ].values
                            point_colors_on_boundary = np.repeat(
                                grey_values[:, np.newaxis], 3, axis=1
                            )

                        # Normalize and brighten colors (brighten_colors is an assumed helper function)
                        point_colors_on_boundary = brighten_colors(
                            np.clip(point_colors_on_boundary, 0, 1),
                            factor=brighten_factor,
                        )

                    # Create individual line segments ((x1,y1),(x2,y2)) for LineCollection
                    segments_for_this_part = np.array(
                        [
                            boundary_line_points[i : i + 2]
                            for i in range(len(boundary_line_points) - 1)
                        ]
                    )
                    # Colors for LineCollection are per-segment (use color of the starting point of each segment)
                    colors_for_this_part = point_colors_on_boundary[:-1]

                    if len(segments_for_this_part) > 0:
                        all_line_segments_for_lc.append(segments_for_this_part)
                        all_colors_for_lc.append(colors_for_this_part)

                if not all_line_segments_for_lc:
                    logger.info(
                        f"Segment {seg}: No drawable line segments found after processing all parts. No boundary drawn."
                    )
                    continue  # to next segment

                # Concatenate all parts to form a single LineCollection for this segment's boundary
                final_lc_segments = np.concatenate(all_line_segments_for_lc)
                final_lc_colors = np.concatenate(all_colors_for_lc)

                if len(final_lc_segments) > 0:
                    lc = LineCollection(
                        final_lc_segments,
                        colors=final_lc_colors,
                        linewidth=boundary_linewidth,
                        alpha=alpha,
                        linestyle=linestyle,
                    )
                    ax.add_collection(lc)

                continue  # End of handling for collinear/insufficient points case; proceed to next segment

            # Calculate boundaries
            try:
                # Calculate dynamic alpha based on average distance between points
                if boundary_method == "alphashape":
                    avg_dist = np.mean(
                        np.linalg.norm(points - points.mean(axis=0), axis=1)
                    )
                    alpha_value = (1.0 / avg_dist) * 1.5  # Adjust as needed
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
                    # Use small alpha for more concave boundary
                    avg_dist = np.mean(
                        np.linalg.norm(points - points.mean(axis=0), axis=1)
                    )
                    alpha_value = (1.0 / avg_dist) * 1.0  # Adjust as needed
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

                # Validate and fix boundary
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

                # Smooth the boundary
                concave_hull = smooth_polygon(concave_hull, buffer_dist=0.01)

            except Exception as e:
                logger.error(f"Failed to calculate boundary for Segment {seg}: {e}")
                continue

            # Clip boundary to plot bounds
            try:
                clipped_polygon = concave_hull.intersection(plot_boundary_polygon)
                if clipped_polygon.is_empty:
                    logger.warning(f"Clipped boundary for Segment {seg} is empty.")
                    continue
            except Exception as e:
                logger.error(f"Failed to clip boundary for Segment {seg}: {e}")
                continue

            # Create patches from clipped polygons
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

            # Set boundary color for the current segment
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

                # Convert boundary coordinates to points
                boundary_points = np.column_stack((x, y))

                # Find nearest neighbors for each boundary point
                _, idx = tree.query(boundary_points)

                if fixed_boundary_color is not None:
                    # Use fixed color
                    boundary_colors = np.array(
                        [fixed_boundary_color] * len(boundary_points)
                    )
                else:
                    if len(dimensions) == 3:
                        # Extract RGB colors
                        boundary_colors = coordinates.iloc[idx][["R", "G", "B"]].values
                    else:
                        # Extract grayscale values and apply colormap if provided
                        grey = coordinates.iloc[idx]["Grey"].values
                        if cmap is None:
                            boundary_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)
                        else:
                            cm = plt.get_cmap(cmap)
                            boundary_colors = cm(grey)[:, :3]

                    # Normalize and brighten colors
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

                # Create segments for LineCollection
                segments_lines = np.array(
                    [
                        boundary_points[i : i + 2]
                        for i in range(len(boundary_points) - 1)
                    ]
                )

                # Assign color to each segment (use color of the starting point)
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
                    # Filling the boundary with pixel-based colors is complex, so use a single color for filling.
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
    if show_colorbar and len(dimensions) == 1:
        cm = plt.get_cmap(cmap if cmap is not None else "Greys")
        vmin = raw_vals.min() if raw_vals is not None else grey_vals.min()
        vmax = raw_vals.max() if raw_vals is not None else grey_vals.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    # Show plot
    plt.show()
    # Do not return plot object to avoid auto display