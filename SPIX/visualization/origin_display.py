from __future__ import annotations

import logging
from time import perf_counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.ndimage import binary_dilation, distance_transform_edt

from SPIX.utils.utils import _check_img, _check_scale_factor, _check_spatial_data, rebalance_colors
from SPIX.visualization import raster as _raster
from .plotting import (
    _apply_axis_crop,
    _compute_alpha_from_values,
    _float_array_fast,
    _plot_boundaries_from_label_image,
    _rasterize_rgba_and_label_buffer,
    _rebalance_color_array,
    _resolve_plot_raster_state,
    _string_array_no_copy,
    _string_index_no_copy,
)


def _resolve_display_coordinates(
    adata,
    *,
    origin: bool,
    coordinate_mode: str,
    array_row_key: str,
    array_col_key: str,
    img_scaling_factor_adjusted: float,
    logger: logging.Logger,
    verbose: bool,
) -> pd.DataFrame:
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
                logger.info("Using adata.obsm['spatial'] directly for display coordinates.")
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
                    logger.info("Using inferred boundary support-grid tiles for display coordinates.")

    if not use_direct_origin_spatial:
        if "barcode" not in tiles.columns or "x" not in tiles.columns or "y" not in tiles.columns:
            raise ValueError("adata.uns['tiles'] must contain 'barcode', 'x', and 'y' columns.")
        tile_barcodes = _string_array_no_copy(tiles["barcode"])
        x_vals = _float_array_fast(tiles["x"])
        y_vals = _float_array_fast(tiles["y"])
        if "origin" in tiles.columns:
            origin_vals = _float_array_fast(tiles["origin"])

    obs_index = _string_index_no_copy(adata.obs.index)
    if use_direct_origin_spatial:
        pos = np.arange(int(adata.n_obs), dtype=np.int64)
    else:
        barcode_index = pd.Index(tile_barcodes)
        if barcode_index.has_duplicates:
            keep_mask = ~barcode_index.duplicated(keep="first")
            tile_barcodes = tile_barcodes[keep_mask]
            x_vals = x_vals[keep_mask]
            y_vals = y_vals[keep_mask]
            if origin_vals is not None:
                origin_vals = origin_vals[keep_mask]
        pos = obs_index.get_indexer(tile_barcodes)

    valid = (pos >= 0) & np.isfinite(x_vals) & np.isfinite(y_vals)
    if not np.any(valid):
        raise ValueError("No display barcodes match adata.obs index.")

    coord_dict: dict[str, np.ndarray] = {
        "x": x_vals[valid],
        "y": y_vals[valid],
        "barcode": tile_barcodes[valid],
    }
    if origin_vals is not None:
        coord_dict["origin"] = origin_vals[valid]
    elif use_direct_origin_spatial:
        coord_dict["origin"] = np.ones(int(np.count_nonzero(valid)), dtype=np.float32)

    coordinates_df = pd.DataFrame(coord_dict)

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

    coordinates_df["x"] = pd.to_numeric(coordinates_df["x"], errors="coerce") * float(img_scaling_factor_adjusted)
    coordinates_df["y"] = pd.to_numeric(coordinates_df["y"], errors="coerce") * float(img_scaling_factor_adjusted)
    coordinates_df = coordinates_df.dropna(subset=["x", "y"]).reset_index(drop=True)
    return coordinates_df


def _resolve_display_colors(
    adata,
    coordinates_df: pd.DataFrame,
    *,
    dimensions,
    embedding: str,
    color_by,
    palette,
    cmap,
    rebalance_method,
    rebalance_vmin,
    rebalance_vmax,
    color_vmin,
    color_vmax,
    color_percentiles,
):
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    embedding_full = np.asarray(adata.obsm[embedding])
    embedding_dims = embedding_full[:, dimensions]
    pos = _string_index_no_copy(adata.obs.index).get_indexer(_string_array_no_copy(coordinates_df["barcode"]))
    if np.any(pos < 0):
        raise ValueError("Some display coordinates do not map back to adata.obs.")

    dim_names = ["dim0", "dim1", "dim2"] if len(dimensions) == 3 else ["dim0"]
    coord = coordinates_df.copy()
    coord.loc[:, dim_names] = embedding_dims[pos]

    raw_vals = None
    use_categorical = False
    category_lut = None
    categories = None

    if color_by is not None:
        if color_by not in adata.obs.columns:
            raise ValueError(f"'{color_by}' not found in adata.obs.")
        coord[color_by] = coord["barcode"].map(adata.obs[color_by])
        coord = coord.dropna(subset=[color_by]).reset_index(drop=True)
        if is_numeric_dtype(coord[color_by]):
            v = pd.to_numeric(coord[color_by], errors="coerce").to_numpy(dtype=float)
            raw_vals = v
            finite = np.isfinite(v)
            if finite.any():
                if (color_vmin is not None) or (color_vmax is not None):
                    vmin = float(np.nanmin(v[finite]) if color_vmin is None else color_vmin)
                    vmax = float(np.nanmax(v[finite]) if color_vmax is None else color_vmax)
                else:
                    p_lo, p_hi = color_percentiles
                    vmin = float(np.nanpercentile(v[finite], p_lo))
                    vmax = float(np.nanpercentile(v[finite], p_hi))
            else:
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-9
            cm = plt.get_cmap(cmap or "viridis")
            t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
            cols = cm(t)[:, :3]
        else:
            cats = pd.Categorical(coord[color_by])
            categories = cats.categories
            n_cat = len(categories)
            if isinstance(palette, (list, tuple, np.ndarray)):
                pal = list(palette)
                category_lut = np.array([mcolors.to_rgb(pal[i % len(pal)]) for i in range(n_cat)])
            else:
                cmap_cat = plt.get_cmap(palette, n_cat) if isinstance(palette, str) else palette
                category_lut = np.array([cmap_cat(i)[:3] for i in range(n_cat)])
            cols = category_lut[cats.codes]
            use_categorical = True
    else:
        coord_before = coord.copy()
        reb = rebalance_colors(coord.copy(), dimensions, method=rebalance_method, vmin=rebalance_vmin, vmax=rebalance_vmax)
        for col in coord_before.columns:
            if col not in reb.columns:
                reb[col] = coord_before[col].values
        if len(dimensions) == 3:
            cols = reb[["R", "G", "B"]].to_numpy(dtype=float)
        else:
            raw_vals = reb["Grey"].to_numpy(dtype=float)
            cm = plt.get_cmap(cmap if cmap is not None else "Greys")
            cols = cm(raw_vals)[:, :3]
        coord = reb

    coord["R"] = cols[:, 0]
    coord["G"] = cols[:, 1]
    coord["B"] = cols[:, 2]
    return coord.reset_index(drop=True), cols, raw_vals, use_categorical, category_lut, categories


def overlay_segment_boundaries_on_display(
    ax,
    coordinates_df: pd.DataFrame,
    *,
    segment_key: str,
    boundary_color: str = "black",
    boundary_linewidth: float = 1.0,
    boundary_alpha: float = 0.9,
    boundary_style: str = "solid",
    pixel_smoothing_sigma: float = 0.0,
    pixel_boundary_render: str = "contour",
    pixel_boundary_thickness_px: int = 1,
    pixel_boundary_antialiased: bool = True,
    figsize=(10, 10),
    fig_dpi: float = 100.0,
    imshow_tile_size=None,
    imshow_scale_factor: float = 1.0,
    imshow_tile_size_mode: str = "auto",
    imshow_tile_size_quantile=None,
    imshow_tile_size_rounding: str = "floor",
    imshow_tile_size_shrink: float = 0.98,
    pixel_perfect: bool = True,
    pixel_shape: str = "square",
    chunk_size: int = 50000,
    neighborhood_fill_px: int | None = None,
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_holes: bool = True,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool = False,
    center_collision_radius: int = 2,
    xlim=None,
    ylim=None,
):
    if segment_key not in coordinates_df.columns:
        raise ValueError(f"'{segment_key}' must be present in coordinates_df.")

    x_min = float(coordinates_df["x"].min())
    x_max = float(coordinates_df["x"].max())
    y_min = float(coordinates_df["y"].min())
    y_max = float(coordinates_df["y"].max())
    pts = coordinates_df[["x", "y"]].to_numpy(dtype=float, copy=False)
    raster_canvas = _raster.resolve_raster_canvas(
        pts=pts,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        figsize=figsize,
        fig_dpi=fig_dpi,
        imshow_tile_size=imshow_tile_size,
        imshow_scale_factor=imshow_scale_factor,
        imshow_tile_size_mode=imshow_tile_size_mode,
        imshow_tile_size_quantile=imshow_tile_size_quantile,
        imshow_tile_size_rounding=imshow_tile_size_rounding,
        imshow_tile_size_shrink=imshow_tile_size_shrink,
        pixel_perfect=bool(pixel_perfect),
        n_points=int(len(pts)),
        logger=None,
        context="overlay_segment_boundaries_on_display",
    )

    x_min_eff = float(raster_canvas["x_min_eff"])
    x_max_eff = float(raster_canvas["x_max_eff"])
    y_min_eff = float(raster_canvas["y_min_eff"])
    y_max_eff = float(raster_canvas["y_max_eff"])
    scale = float(raster_canvas["scale"])
    s = int(raster_canvas["tile_px"])
    w = int(raster_canvas["w"])
    h = int(raster_canvas["h"])

    raster_state = _resolve_plot_raster_state(
        pts,
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
        coord_mode="spatial",
        soft_rasterization=soft_rasterization,
        resolve_center_collisions=resolve_center_collisions,
        center_collision_radius=center_collision_radius,
        logger=None,
        verbose=False,
        context="overlay_segment_boundaries_on_display",
    )

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
        vals=None,
        prioritize_high_values=False,
        soft_enabled=bool(raster_state["soft_enabled"]),
        soft_fx=raster_state["soft_fx"],
        soft_fy=raster_state["soft_fy"],
        label_codes=pd.Categorical(coordinates_df[segment_key]).codes.astype(np.int32),
        runtime_fill_from_boundary=bool(runtime_fill_from_boundary),
        runtime_fill_closing_radius=int(runtime_fill_closing_radius),
        runtime_fill_holes=bool(runtime_fill_holes),
        chunk_size=int(chunk_size),
        logger=None,
        verbose=False,
        context="overlay_segment_boundaries_on_display",
    )

    occupied = label_img >= 0
    if neighborhood_fill_px is None:
        fill_px = 0 if runtime_fill_from_boundary else max(2, int(np.ceil(max(1, int(raster_state["s"])) * 0.75)))
    else:
        fill_px = int(max(0, neighborhood_fill_px))
    if (not runtime_fill_from_boundary) and fill_px > 0 and np.any(occupied):
        neighborhood = binary_dilation(occupied, iterations=fill_px)
        missing = neighborhood & (label_img < 0)
        if np.any(missing):
            _, nearest = distance_transform_edt(label_img < 0, return_indices=True)
            filled = label_img.copy()
            filled[missing] = label_img[nearest[0][missing], nearest[1][missing]]
            label_img = filled
        else:
            neighborhood = occupied
    else:
        neighborhood = occupied

    linestyle = {"solid": "-", "dashed": "--", "dotted": ":"}.get(boundary_style, "-")
    if np.any(neighborhood):
        boundary_only = label_img.copy()
        boundary_only[~neighborhood] = -1
        _plot_boundaries_from_label_image(
            ax,
            boundary_only,
            [
                float(raster_state["x_min_eff"]),
                float(raster_state["x_max_eff"]),
                float(raster_state["y_min_eff"]),
                float(raster_state["y_max_eff"]),
            ],
            boundary_color,
            boundary_linewidth,
            boundary_alpha,
            linestyle,
            pixel_smoothing_sigma,
            render=pixel_boundary_render,
            thickness_px=pixel_boundary_thickness_px,
            antialiased=pixel_boundary_antialiased,
        )
    _apply_axis_crop(ax, xlim=xlim, ylim=ylim)


def image_plot_with_spatial_image_display_boundaries(
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
    segment_key="Segment",
    boundary_alpha=0.9,
    boundary_color="black",
    boundary_linewidth=1.0,
    img=None,
    img_key=None,
    library_id=None,
    crop=True,
    xlim=None,
    ylim=None,
    display_orientation: str = "cartesian",  # 'cartesian' (default) or 'image'
    alpha_img=1.0,
    bw=False,
    boundary_style="solid",
    cmap=None,
    rebalance_method: str = "minmax",
    rebalance_vmin=None,
    rebalance_vmax=None,
    color_vmin=None,
    color_vmax=None,
    color_percentiles=(1, 99),
    pixel_smoothing_sigma=0,
    pixel_boundary_render: str = "contour",
    pixel_boundary_thickness_px: int = 1,
    pixel_boundary_antialiased: bool = True,
    use_imshow=True,
    pixel_shape="square",
    chunk_size=50000,
    prioritize_high_values=False,
    alpha_point=1.0,
    verbose=False,
    title=None,
    show_colorbar=False,
    color_by=None,
    palette="tab20",
    show_legend=False,
    legend_loc="best",
    legend_ncol=2,
    legend_outside=True,
    legend_bbox_to_anchor=(0.83, 0.5),
    legend_fontsize=None,
    show=True,
    alpha_by=None,
    alpha_range=(0.1, 1.0),
    alpha_clip=None,
    alpha_invert=False,
    coordinate_mode: str = "spatial",
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_holes: bool = True,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool = False,
    center_collision_radius: int = 2,
):
    """
    Plot display coordinates and overlay segment boundaries re-rasterized on the displayed tiles.

    This is intended for the common workflow where segment assignments were produced using
    `origin=False`, but the user wants the boundaries shown on the `origin=True` display tiles
    or H&E background. The function does not use origin-false support tiles for boundary geometry;
    instead, it re-rasterizes the segment labels stored in `adata.obs[segment_key]` on the chosen
    display coordinates.
    """
    logger = logging.getLogger("image_plot_with_spatial_image_display_boundaries")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    t_all = perf_counter()

    if segment_key not in adata.obs.columns:
        raise ValueError(f"'{segment_key}' not found in adata.obs.")
    if len(dimensions) not in [1, 3]:
        raise ValueError("Visualization requires 1D or 3D embeddings.")

    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    suppress_img = (img is False) or (img_key is False)
    if suppress_img:
        img = None
        img_key = None
    else:
        img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    tensor_resolution = adata.uns.get("tensor_resolution", 1.0)
    if img_scaling_factor is None:
        img_scaling_factor = _check_scale_factor(spatial_data, img_key, img_scaling_factor)
    img_scaling_factor_adjusted = img_scaling_factor / tensor_resolution if img_scaling_factor else 1.0

    coordinates_df = _resolve_display_coordinates(
        adata,
        origin=origin,
        coordinate_mode=coordinate_mode,
        array_row_key=array_row_key,
        array_col_key=array_col_key,
        img_scaling_factor_adjusted=img_scaling_factor_adjusted,
        logger=logger,
        verbose=verbose,
    )

    coordinates_df[segment_key] = coordinates_df["barcode"].map(adata.obs[segment_key])
    coordinates_df = coordinates_df.dropna(subset=[segment_key]).reset_index(drop=True)
    coordinates_df, cols, raw_vals, use_categorical, category_lut, categories = _resolve_display_colors(
        adata,
        coordinates_df,
        dimensions=dimensions,
        embedding=embedding,
        color_by=color_by,
        palette=palette,
        cmap=cmap,
        rebalance_method=rebalance_method,
        rebalance_vmin=rebalance_vmin,
        rebalance_vmax=rebalance_vmax,
        color_vmin=color_vmin,
        color_vmax=color_vmax,
        color_percentiles=color_percentiles,
    )

    alpha_arr = None
    if alpha_by is not None:
        if alpha_by == "embedding":
            if len(dimensions) != 1:
                raise ValueError("alpha_by='embedding' requires len(dimensions) == 1.")
            alpha_source = coordinates_df["dim0"].to_numpy(dtype=float)
        else:
            if alpha_by not in adata.obs.columns:
                raise ValueError(f"alpha_by '{alpha_by}' not found in adata.obs.")
            alpha_source = coordinates_df["barcode"].map(adata.obs[alpha_by]).astype(float).to_numpy()
        alpha_arr, _ = _compute_alpha_from_values(
            alpha_source, alpha_range=alpha_range, clip=alpha_clip, invert=alpha_invert
        )

    x_min = float(coordinates_df["x"].min())
    x_max = float(coordinates_df["x"].max())
    y_min = float(coordinates_df["y"].min())
    y_max = float(coordinates_df["y"].max())
    pts = coordinates_df[["x", "y"]].to_numpy(dtype=float, copy=False)

    if use_imshow:
        raster_canvas = _raster.resolve_raster_canvas(
            pts=pts,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            figsize=figsize,
            fig_dpi=fig_dpi,
            imshow_tile_size=imshow_tile_size,
            imshow_scale_factor=imshow_scale_factor,
            imshow_tile_size_mode=imshow_tile_size_mode,
            imshow_tile_size_quantile=imshow_tile_size_quantile,
            imshow_tile_size_rounding=imshow_tile_size_rounding,
            imshow_tile_size_shrink=imshow_tile_size_shrink,
            pixel_perfect=bool(pixel_perfect),
            n_points=int(len(pts)),
            logger=logger if verbose else None,
            context="image_plot_with_spatial_image_display_boundaries",
        )
        figsize = raster_canvas["figsize"]
        fig_dpi = raster_canvas["fig_dpi"]

    if figsize is None:
        figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)

    orientation = str(display_orientation or "image").lower()
    if orientation not in {"image", "cartesian"}:
        orientation = "image"

    if img is not None:
        y_max_img, x_max_img = img.shape[:2]
        if orientation == "cartesian":
            extent = [0, x_max_img, 0, y_max_img]
            ax.imshow(img, extent=extent, origin="lower", alpha=alpha_img)
        else:
            extent = [0, x_max_img, y_max_img, 0]
            ax.imshow(img, extent=extent, alpha=alpha_img)
        if crop:
            x_range_spots = x_max - x_min
            y_range_spots = y_max - y_min
            x_padding = x_range_spots * 0.05
            y_padding = y_range_spots * 0.05
            ax.set_xlim(max(0, x_min - x_padding), min(x_max_img, x_max + x_padding))
            if orientation == "cartesian":
                ax.set_ylim(max(0, y_min - y_padding), min(y_max_img, y_max + y_padding))
            else:
                ax.set_ylim(min(y_max_img, y_max + y_padding), max(0, y_min - y_padding))
        else:
            ax.set_xlim(0, x_max_img)
            if orientation == "cartesian":
                ax.set_ylim(0, y_max_img)
            else:
                ax.set_ylim(y_max_img, 0)
    else:
        ax.set_xlim(x_min, x_max)
        if orientation == "cartesian":
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(y_max, y_min)

    _apply_axis_crop(ax, xlim=xlim, ylim=ylim)

    if use_imshow:
        raster_state = _resolve_plot_raster_state(
            pts,
            x_min_eff=float(raster_canvas["x_min_eff"]),
            x_max_eff=float(raster_canvas["x_max_eff"]),
            y_min_eff=float(raster_canvas["y_min_eff"]),
            y_max_eff=float(raster_canvas["y_max_eff"]),
            scale=float(raster_canvas["scale"]),
            s=int(raster_canvas["tile_px"]),
            w=int(raster_canvas["w"]),
            h=int(raster_canvas["h"]),
            pixel_perfect=bool(pixel_perfect),
            pixel_shape=pixel_shape,
            coord_mode=coordinate_mode,
            soft_rasterization=soft_rasterization,
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=center_collision_radius,
            logger=logger,
            verbose=verbose,
            context="image_plot_with_spatial_image_display_boundaries",
        )
        draw_vals = alpha_arr.astype(float) if alpha_arr is not None else np.arange(len(pts), dtype=float)
        img_rgba, _, _ = _rasterize_rgba_and_label_buffer(
            cols=cols,
            cx=raster_state["cx"],
            cy=raster_state["cy"],
            w=int(raster_state["w"]),
            h=int(raster_state["h"]),
            s=int(raster_state["s"]),
            pixel_shape=pixel_shape,
            alpha_arr=alpha_arr,
            alpha_point=float(alpha_point),
            vals=draw_vals,
            prioritize_high_values=bool(prioritize_high_values),
            soft_enabled=bool(raster_state["soft_enabled"]),
            soft_fx=raster_state["soft_fx"],
            soft_fy=raster_state["soft_fy"],
            label_codes=None,
            runtime_fill_from_boundary=bool(runtime_fill_from_boundary),
            runtime_fill_closing_radius=int(runtime_fill_closing_radius),
            runtime_fill_holes=bool(runtime_fill_holes),
            chunk_size=int(chunk_size),
            logger=logger,
            verbose=verbose,
            context="image_plot_with_spatial_image_display_boundaries",
        )

        ax.imshow(
            img_rgba,
            origin="lower",
            extent=[
                float(raster_state["x_min_eff"]),
                float(raster_state["x_max_eff"]),
                float(raster_state["y_min_eff"]),
                float(raster_state["y_max_eff"]),
            ],
            interpolation="nearest",
        )
        overlay_segment_boundaries_on_display(
            ax,
            coordinates_df,
            segment_key=segment_key,
            boundary_color=boundary_color,
            boundary_linewidth=boundary_linewidth,
            boundary_alpha=boundary_alpha,
            boundary_style=boundary_style,
            pixel_smoothing_sigma=pixel_smoothing_sigma,
            pixel_boundary_render=pixel_boundary_render,
            pixel_boundary_thickness_px=pixel_boundary_thickness_px,
            pixel_boundary_antialiased=pixel_boundary_antialiased,
            figsize=figsize,
            fig_dpi=fig.get_dpi(),
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
            runtime_fill_holes=runtime_fill_holes,
            soft_rasterization=soft_rasterization,
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=center_collision_radius,
            xlim=xlim,
            ylim=ylim,
        )
    else:
        if point_size is None:
            num_points = len(coordinates_df)
            base_size = (figsize[0] * figsize[1] * 100**2) / (num_points + 1)
            point_size = (scaling_factor / 5.0) * base_size
            point_size = np.clip(point_size, 0.01, 200)
        cols_rgba = np.concatenate(
            [cols, (alpha_arr[:, None] if alpha_arr is not None else np.full((cols.shape[0], 1), alpha_point))],
            axis=1,
        )
        ax.scatter(
            x=coordinates_df["x"],
            y=coordinates_df["y"],
            s=point_size,
            c=cols_rgba,
            marker="o" if pixel_shape == "circle" else "s",
            linewidths=0,
        )
        overlay_segment_boundaries_on_display(
            ax,
            coordinates_df,
            segment_key=segment_key,
            boundary_color=boundary_color,
            boundary_linewidth=boundary_linewidth,
            boundary_alpha=boundary_alpha,
            boundary_style=boundary_style,
            pixel_smoothing_sigma=pixel_smoothing_sigma,
            pixel_boundary_render=pixel_boundary_render,
            pixel_boundary_thickness_px=pixel_boundary_thickness_px,
            pixel_boundary_antialiased=pixel_boundary_antialiased,
            figsize=figsize,
            fig_dpi=fig.get_dpi(),
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
            runtime_fill_holes=runtime_fill_holes,
            soft_rasterization=soft_rasterization,
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=center_collision_radius,
            xlim=xlim,
            ylim=ylim,
        )

    if use_categorical and show_legend and (categories is not None) and (category_lut is not None):
        handles = [patches.Patch(color=category_lut[i], label=str(cat)) for i, cat in enumerate(categories)]
        if legend_outside:
            fig.subplots_adjust(right=0.82)
            loc_final = "center left" if legend_loc == "best" else legend_loc
            fig.legend(
                handles=handles,
                loc=loc_final,
                bbox_to_anchor=legend_bbox_to_anchor,
                ncol=legend_ncol,
                frameon=False,
                fontsize=legend_fontsize,
            )
        else:
            ax.legend(handles=handles, loc=legend_loc, ncol=legend_ncol, frameon=False, fontsize=legend_fontsize)

    if show_colorbar and (len(dimensions) == 1) and (raw_vals is not None) and (not use_categorical):
        cm = plt.get_cmap(cmap if cmap is not None else "Greys")
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=float(np.nanmin(raw_vals)), vmax=float(np.nanmax(raw_vals))), cmap=cm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title if title is not None else f"{embedding} - display boundaries", fontsize=figsize[0] * 1.5)
    if verbose:
        logger.info("image_plot_with_spatial_image_display_boundaries total time: %.3fs", perf_counter() - t_all)
    if show:
        plt.show()
        return None
    return fig, ax
