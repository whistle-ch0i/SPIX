import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from shapely.geometry import MultiPoint, Polygon, LineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import explain_validity
from scipy.spatial import ConvexHull, KDTree
import alphashape
import logging

# Import helper functions from utils.py
from .utils import (
    brighten_colors,
    rebalance_colors,
    is_collinear,
    is_almost_collinear,
    smooth_polygon,
    alpha_shape,
    _check_spatial_data,
    _check_img,
    _check_scale_factor
)
# Configure logging to display warnings and errors
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def image_plot(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding_segment',  # Use segmented embedding
    figsize=(10, 10),
    point_size=None,  # Default is None
    scaling_factor=10,
    origin=True,
    plot_boundaries=True,  # Toggle for plotting boundaries
    boundary_method='alphashape',  # Options: 'convex_hull', 'alphashape', 'oriented_bbox', 'concave_hull'
    alpha=0.3,  # Transparency of polygons
    boundary_color='black',  # Default boundary color (if using a fixed color)
    boundary_linewidth=1.0,  # Line width of boundaries
    aspect_ratio_threshold=3.0,  # Threshold to determine if a segment is elongated
    jitter=1e-6,  # Jitter value to avoid collinearity
    boundary_style='solid',  # Boundary line style
    fill_boundaries=False,   # Toggle for filling the inside of boundaries
    fill_color='blue',       # Fill color
    fill_alpha=0.1,          # Fill transparency
    brighten_factor=1.3,     # Factor for brightening colors
    fixed_boundary_color=None, # New parameter: Fixed boundary color
    verbose=False            # New parameter: Control verbosity
):
    """
    Visualize embeddings as an image, optionally displaying segment boundaries.
    You can set a fixed color for the boundaries or, if unspecified, colors are adjusted based on pixel colors.

    Parameters
    ----------
    ... [existing parameters] ...
    verbose : bool, optional
        If True, display warnings and informational messages. Default is False.
    """
    # Create a local logger
    logger = logging.getLogger('image_plot')
    logger.setLevel(logging.INFO if verbose else logging.ERROR)

    # Avoid duplicate handlers if the function is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if len(dimensions) not in [1, 3]:
        raise ValueError("Visualization requires 1D or 3D embeddings.")

    # Extract embedding
    if embedding not in adata.obsm:
        raise ValueError(f"'{embedding}' embedding does not exist in adata.obsm.")

    # Extract spatial coordinates
    if origin:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    else:
        tiles = adata.uns['tiles']

    # Prepare tile colors and merge with coordinates
    embedding_dims = np.array(adata.obsm[embedding])[:, dimensions]
    if len(dimensions) == 3:
        dim_names = ['dim0', 'dim1', 'dim2']
    else:
        dim_names = ['dim0']

    tile_colors = pd.DataFrame(embedding_dims, columns=dim_names)
    tile_colors['barcode'] = adata.obs.index
    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="right").dropna().reset_index(drop=True)

    if 'Segment' in adata.obs.columns:
        # Map 'Segment' to 'barcode'
        coordinates_df['Segment'] = coordinates_df['barcode'].map(adata.obs['Segment'])
        # Check for unmapped values
        if coordinates_df['Segment'].isnull().any():
            missing = coordinates_df['Segment'].isnull().sum()
            logger.warning(f"'Segment' label for {missing} barcodes in coordinates_df is missing in adata.obs.")
            # Drop rows with missing 'Segment' values
            coordinates_df = coordinates_df.dropna(subset=['Segment'])
    # **Convert 'x' and 'y' to numeric to prevent dtype=object issues**
    coordinates_df['x'] = pd.to_numeric(coordinates_df['x'], errors='coerce')
    coordinates_df['y'] = pd.to_numeric(coordinates_df['y'], errors='coerce')

    # Drop rows with NaN in 'x' or 'y'
    if coordinates_df[['x', 'y']].isnull().any().any():
        logger.warning("Found NaN values in 'x' or 'y' after conversion. These will be dropped.")
        coordinates_df = coordinates_df.dropna(subset=['x', 'y'])

    # Ensure 'x' and 'y' are numeric
    if not pd.api.types.is_numeric_dtype(coordinates_df['x']) or not pd.api.types.is_numeric_dtype(coordinates_df['y']):
        raise TypeError("Columns 'x' and 'y' must be numeric after conversion.")
    # Normalize embeddings to values between 0 and 1
    coordinates = rebalance_colors(coordinates_df, dimensions)

    if len(dimensions) == 3:
        cols = coordinates[['R', 'G', 'B']].values
    else:
        grey_values = coordinates['Grey'].values
        cols = np.repeat(grey_values[:, np.newaxis], 3, axis=1)

    # Set point_size automatically if not provided
    if point_size is None:
        figure_area = figsize[0] * figsize[1]  # In square inches
        point_size = scaling_factor * (figure_area / 100)  # Adjust denominator as needed

    # Determine plot boundaries based on spatial coordinates
    x_min, x_max = coordinates['x'].min(), coordinates['x'].max()
    y_min, y_max = coordinates['y'].min(), coordinates['y'].max()
    plot_boundary_polygon = Polygon([
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (x_max, y_min)
    ])

    # Define linestyle based on boundary_style
    linestyle_options = {'solid': '-', 'dashed': '--', 'dotted': ':'}
    linestyle = linestyle_options.get(boundary_style, '-')

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    sc_kwargs = dict(
        x=coordinates['x'],
        y=coordinates['y'],
        s=point_size,
        c=cols,
        marker='s',
        linewidths=0
    )
    ax.scatter(**sc_kwargs)
    ax.set_aspect('equal')
    ax.axis('off')
    title = f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
    ax.set_title(title, fontsize=figsize[0] * 1.5)

    if 'Segment' in adata.obs.columns and plot_boundaries:
        # Build KDTree for efficient nearest-neighbor search
        tree = KDTree(coordinates[['x', 'y']].values)

        # Group by segments
        segments = coordinates_df['Segment'].unique()

        for seg in segments:
            seg_tiles = coordinates_df[coordinates_df['Segment'] == seg]
            if len(seg_tiles) < 3:
                logger.info(f"Segment {seg} has fewer than 3 points. Skipping boundary plotting.")
                continue

            points = seg_tiles[['x', 'y']].values

            # Check for collinearity
            unique_points = np.unique(points, axis=0)
            if unique_points.shape[0] < 3 or is_collinear(unique_points):
                logger.warning(f"Segment {seg} points are collinear or insufficient. Using LineString as boundary.")
                line = LineString(unique_points)
                # Clip boundary to plot bounds
                clipped_line = line.intersection(plot_boundary_polygon)
                if not clipped_line.is_empty:
                    x, y = clipped_line.xy
                    # Set color for each point of the boundary
                    line_points = np.column_stack((x, y))
                    _, idx = tree.query(line_points)
                    if fixed_boundary_color is not None:
                        # Use fixed color
                        line_colors = np.array([fixed_boundary_color] * len(line_points))
                    else:
                        if len(dimensions) == 3:
                            # Extract RGB colors
                            line_colors = coordinates.iloc[idx][['R', 'G', 'B']].values
                        else:
                            # Extract grayscale values and replicate to RGB
                            grey = coordinates.iloc[idx]['Grey'].values
                            line_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)

                        # Normalize and brighten colors
                        line_colors = brighten_colors(np.clip(line_colors, 0, 1), factor=brighten_factor)

                    # Create LineCollection
                    segments_lines = np.array([line_points[i:i+2] for i in range(len(line_points)-1)])
                    lc = LineCollection(segments_lines, colors=line_colors[:-1], linewidth=boundary_linewidth, alpha=alpha, linestyle=linestyle)
                    ax.add_collection(lc)
                continue

            # Calculate boundaries
            try:
                # Calculate dynamic alpha based on average distance between points
                if boundary_method == 'alphashape':
                    avg_dist = np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1))
                    alpha_value = (1.0 / avg_dist) * 1.5  # Adjust as needed
                    concave_hull = alphashape.alphashape(points, alpha=alpha_value)
                    if concave_hull.is_empty or concave_hull.geom_type not in ['Polygon', 'MultiPolygon']:
                        logger.warning(f"Alpha shape for Segment {seg} is empty or not Polygon/MultiPolygon. Using Convex Hull as fallback.")
                        concave_hull = MultiPoint(points).convex_hull
                elif boundary_method == 'convex_hull':
                    concave_hull = MultiPoint(points).convex_hull
                elif boundary_method == 'oriented_bbox':
                    concave_hull = MultiPoint(points).minimum_rotated_rectangle
                elif boundary_method == 'concave_hull':
                    # Use small alpha for more concave boundary
                    avg_dist = np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1))
                    alpha_value = (1.0 / avg_dist) * 1.0  # Adjust as needed
                    concave_hull = alphashape.alphashape(points, alpha=alpha_value)
                    if concave_hull.is_empty or concave_hull.geom_type not in ['Polygon', 'MultiPolygon']:
                        logger.warning(f"Concave hull for Segment {seg} is empty or not Polygon/MultiPolygon. Using Convex Hull as fallback.")
                        concave_hull = MultiPoint(points).convex_hull
                else:
                    raise ValueError(f"Unknown boundary method '{boundary_method}'")

                # Validate and fix boundary
                if not concave_hull.is_valid:
                    logger.warning(f"Boundary for Segment {seg} is invalid: {explain_validity(concave_hull)}. Attempting to fix with buffer(0).")
                    concave_hull = concave_hull.buffer(0)
                    if not concave_hull.is_valid:
                        logger.error(f"Unable to fix boundary for Segment {seg}. Skipping this segment.")
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
                logger.warning(f"Unsupported geometry type in clipped polygon for Segment {seg}: {type(clipped_polygon)}")
                continue

            # Set boundary color for the current segment
            for poly in polygons:
                if isinstance(poly, Polygon):
                    x, y = poly.exterior.xy
                elif isinstance(poly, LineString):
                    x, y = poly.xy
                else:
                    logger.warning(f"Unsupported geometry type in polygon for Segment {seg}: {type(poly)}")
                    continue

                # Convert boundary coordinates to points
                boundary_points = np.column_stack((x, y))

                # Find nearest neighbors for each boundary point
                _, idx = tree.query(boundary_points)

                if fixed_boundary_color is not None:
                    # Use fixed color
                    boundary_colors = np.array([fixed_boundary_color] * len(boundary_points))
                else:
                    if len(dimensions) == 3:
                        # Extract RGB colors
                        boundary_colors = coordinates.iloc[idx][['R', 'G', 'B']].values
                    else:
                        # Extract grayscale values and replicate to RGB
                        grey = coordinates.iloc[idx]['Grey'].values
                        boundary_colors = np.repeat(grey[:, np.newaxis], 3, axis=1)

                    # Normalize and brighten colors
                    boundary_colors = brighten_colors(np.clip(boundary_colors, 0, 1), factor=brighten_factor)

                # Create segments for LineCollection
                segments_lines = np.array([boundary_points[i:i+2] for i in range(len(boundary_points)-1)])

                # Assign color to each segment (use color of the starting point)
                lc = LineCollection(segments_lines, colors=boundary_colors[:-1], linewidth=boundary_linewidth, alpha=alpha, linestyle=linestyle)
                ax.add_collection(lc)

                if fill_boundaries and isinstance(poly, Polygon):
                    # Filling the boundary with pixel-based colors is complex, so use a single color for filling.
                    patch = patches.Polygon(np.array(poly.exterior.coords), closed=True, facecolor=fill_color, edgecolor=None, alpha=fill_alpha)
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
    embedding='X_embedding_segment',
    figsize=(10, 10),
    point_size=None,
    point_scaling_factor=10,  # Set a default scaling factor
    scaling_factor=None,
    origin=True,
    plot_boundaries=True,
    boundary_method='convex_hull',
    alpha=0.3,
    alpha_point=1,
    boundary_color='black',
    boundary_linewidth=1.0,
    alpha_shape_alpha=0.1,
    aspect_ratio_threshold=3.0,
    img=None,
    img_key=None,
    library_id=None,
    crop=True,
    alpha_img=1.0,
    bw=False,
    jitter=1e-6,  # Jitter value to avoid collinearity
    tol_collinear=1e-8,  # Tolerance for exact collinearity
    tol_almost_collinear=1e-5  # Tolerance for near collinearity
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
    point_scaling_factor : float
        Scaling factor for point size calculation if point_size is not provided.
    scaling_factor : float or None
        Additional scaling factor for spatial coordinates.
    origin : bool
        Whether to use only origin tiles.
    plot_boundaries : bool
        Whether to plot segment boundaries.
    boundary_method : str
        Method to compute boundaries ('convex_hull', 'alpha_shape').
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
    """
    if len(dimensions) not in [1, 3]:
        raise ValueError("Only 1 or 3 dimensions can be used for visualization.")
    
    # Extract embeddings
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Extract image data if available
    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    
    # Retrieve tensor_resolution from adata.uns
    tensor_resolution = adata.uns.get('tensor_resolution', 1.0)
    
    # Adjust scaling_factor based on tensor_resolution
    if scaling_factor is None:
        scaling_factor = _check_scale_factor(spatial_data, img_key, scaling_factor)
    scaling_factor_adjusted = scaling_factor / tensor_resolution if scaling_factor else 1.0
    
    # Extract spatial coordinates
    if origin:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    else:
        tiles = adata.uns['tiles']
    
    # Prepare tile colors and merge with coordinates
    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors['barcode'] = adata.obs.index
    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="inner").dropna().reset_index(drop=True)

    # Apply adjusted scaling factor to coordinates
    coordinates_df['x'] *= scaling_factor_adjusted
    coordinates_df['y'] *= scaling_factor_adjusted

    if 'Segment' in adata.obs.columns:
        # Map 'Segment' based on 'barcode'
        coordinates_df['Segment'] = coordinates_df['barcode'].map(adata.obs['Segment'])
        # Drop rows with missing Segment
        coordinates_df = coordinates_df.dropna(subset=['Segment'])

    # Normalize embeddings between 0 and 1
    coordinates = rebalance_colors(coordinates_df, dimensions)
    
    if len(dimensions) == 3:
        # Normalize the RGB values between 0 and 1
        cols = [
            f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            for r, g, b in zip(coordinates['R'], coordinates['G'], coordinates['B'])
        ]
    else:
        # For grayscale, replicate the grey value across RGB channels
        grey_values = coordinates['Grey'].values
        cols = [
            f'#{int(g*255):02x}{int(g*255):02x}{int(g*255):02x}'
            for g in grey_values
        ]
    
    # Automatically determine point_size if not provided
    if point_size is None:
        if point_scaling_factor is None:
            point_scaling_factor = 10  # Default scaling factor if not provided
        figure_area = figsize[0] * figsize[1]  # in square inches
        point_size = point_scaling_factor * (figure_area / 100)  # Adjust denominator as needed

    # Determine plot boundaries based on spatial coordinates
    x_min, x_max = coordinates['x'].min(), coordinates['x'].max()
    y_min, y_max = coordinates['y'].min(), coordinates['y'].max()

    # Adjust order if necessary
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)

    if img is not None:
        y_max_img, x_max_img = img.shape[:2]
        extent = [0, x_max_img, y_max_img, 0]  # Left, Right, Bottom, Top

        # Display the image
        ax.imshow(img, extent=extent, alpha=alpha_img)

        if crop:
            # Set axis limits to the extent of the spots
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
        else:
            # Show the full image
            ax.set_xlim(0, x_max_img)
            ax.set_ylim(y_max_img, 0)
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

    # Plot the spots
    sc_kwargs = dict(
        x=coordinates['x'],
        y=coordinates['y'],
        s=point_size,
        c=cols,
        alpha=alpha_point,
        marker='o',
        linewidths=0,
    )
    scatter = ax.scatter(**sc_kwargs)
    
    ax.set_aspect('equal')
    ax.axis('off')
    title = f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
    ax.set_title(title, fontsize=figsize[0] * 1.5)

    if 'Segment' in adata.obs.columns and plot_boundaries:
        # Plot segment boundaries
        plot_boundaries_on_ax(
            ax, 
            coordinates_df, 
            boundary_method, 
            alpha, 
            boundary_color,
            boundary_linewidth, 
            alpha_shape_alpha, 
            aspect_ratio_threshold,
            jitter,
            tol_collinear,
            tol_almost_collinear
        )
    
    # Display the figure
    plt.show()
    # Do not return the figure to prevent automatic display

def plot_boundaries_on_ax(ax, coordinates_df, boundary_method, alpha, boundary_color,
                          boundary_linewidth, alpha_shape_alpha, aspect_ratio_threshold,
                          jitter, tol_collinear, tol_almost_collinear):
    """
    Plot segment boundaries on the given Axes object.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the boundaries.
    coordinates_df : pandas.DataFrame
        DataFrame containing the coordinates and segment information.
    boundary_method : str
        Method to compute boundaries ('convex_hull', 'alpha_shape').
    alpha : float
        Transparency for the boundary polygons.
    boundary_color : str
        Color for the boundaries.
    boundary_linewidth : float
        Line width for the boundaries.
    alpha_shape_alpha : float
        Alpha value for alpha shapes.
    aspect_ratio_threshold : float
        Threshold to determine if a segment is thin and long.
    jitter : float
        Amount of jitter to add to points to avoid collinearity.
    tol_collinear : float
        Tolerance for exact collinearity.
    tol_almost_collinear : float
        Tolerance for near collinearity.
    """
    # Group by segments
    segments = coordinates_df['Segment'].unique()
    for seg in segments:
        seg_tiles = coordinates_df[coordinates_df['Segment'] == seg]
        if len(seg_tiles) < 2:
            # Not enough points to form a boundary
            logging.info(f"Segment {seg} has less than 2 points. Skipping boundary plot.")
            continue
        
        points = seg_tiles[['x', 'y']].values

        # Calculate aspect ratio
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        aspect_ratio = max(x_range, y_range) / (min(x_range, y_range) + 1e-10)  # Avoid division by zero

        if aspect_ratio > aspect_ratio_threshold:
            # Treat as thin and long segment
            if len(seg_tiles) >= 2:
                # Sort points by x or y for consistent line
                sorted_indices = np.argsort(points[:, 0])
                sorted_points = points[sorted_indices]
                line = LineString(sorted_points)
                # Buffer the line to create a polygon with small thickness
                buffer_size = min(x_range, y_range) * 0.05  # Buffer size proportional to smaller range
                buffered_line = line.buffer(buffer_size)
                # Plot the buffered line
                if not buffered_line.is_empty:
                    if isinstance(buffered_line, Polygon):
                        polygons = [buffered_line]
                    elif isinstance(buffered_line, MultiPolygon):
                        polygons = list(buffered_line)
                    else:
                        logging.warning(f"Buffered line for segment {seg} is neither Polygon nor MultiPolygon.")
                        continue
                    for poly in polygons:
                        patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                                fill=False, edgecolor=boundary_color, 
                                                linewidth=boundary_linewidth, alpha=alpha)
                        ax.add_patch(patch)
        else:
            # Use ConvexHull or alpha_shape for regular segments
            if boundary_method == 'convex_hull':
                if len(seg_tiles) < 3:
                    logging.warning(f"Segment {seg} has less than 3 points. ConvexHull cannot be computed.")
                    continue
                try:
                    # Check for unique points
                    unique_points = np.unique(points, axis=0)
                    if unique_points.shape[0] < 3:
                        logging.warning(f"Segment {seg} has less than 3 unique points. ConvexHull cannot be computed.")
                        continue
                    # Check for collinearity
                    if is_collinear(unique_points, tol_collinear):
                        logging.warning(f"Segment {seg} points are exactly collinear. Plotting LineString.")
                        line = LineString(unique_points)
                        clipped_line = line  # Assuming already within plot boundaries
                        if not clipped_line.is_empty:
                            if isinstance(clipped_line, LineString):
                                lines = [clipped_line]
                            elif isinstance(clipped_line, MultiLineString):
                                lines = list(clipped_line)
                            else:
                                logging.warning(f"Clipped line for segment {seg} is neither LineString nor MultiLineString.")
                                continue
                            for ln in lines:
                                x, y = ln.xy
                                ax.plot(x, y, color=boundary_color, linewidth=boundary_linewidth, alpha=alpha)
                        continue
                    elif is_almost_collinear(unique_points, tol_almost_collinear):
                        logging.info(f"Segment {seg} points are nearly collinear. Adding jitter.")
                        unique_points += np.random.uniform(-jitter, jitter, unique_points.shape)
                    hull = ConvexHull(unique_points)
                    hull_points = unique_points[hull.vertices]
                    polygon = Polygon(hull_points)
                except Exception as e:
                    logging.error(f"ConvexHull failed for segment {seg}: {e}")
                    continue
            elif boundary_method == 'alpha_shape':
                polygon = alpha_shape(points, alpha_shape_alpha)
                if polygon is None:
                    logging.warning(f"Alpha shape failed for segment {seg}.")
                    continue
            else:
                raise ValueError(f"Unknown boundary method '{boundary_method}'")

            if boundary_method == 'convex_hull':
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # Attempt to fix invalid polygons

                if polygon.is_empty:
                    logging.warning(f"Polygon is empty for segment {seg}.")
                    continue

                # Create a patch from the polygon
                if isinstance(polygon, Polygon):
                    polygons = [polygon]
                elif isinstance(polygon, MultiPolygon):
                    polygons = list(polygon)
                else:
                    logging.warning(f"Polygon for segment {seg} is neither Polygon nor MultiPolygon.")
                    continue

                for poly in polygons:
                    patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                            fill=False, edgecolor=boundary_color, 
                                            linewidth=boundary_linewidth, alpha=alpha)
                    ax.add_patch(patch)
            elif boundary_method == 'alpha_shape':
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # Attempt to fix invalid polygons

                if polygon.is_empty:
                    logging.warning(f"Polygon is empty for segment {seg}.")
                    continue

                # Create a patch from the polygon
                if isinstance(polygon, Polygon):
                    polygons = [polygon]
                elif isinstance(polygon, MultiPolygon):
                    polygons = list(polygon)
                else:
                    logging.warning(f"Polygon for segment {seg} is neither Polygon nor MultiPolygon.")
                    continue

                for poly in polygons:
                    patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                            fill=False, edgecolor=boundary_color, 
                                            linewidth=boundary_linewidth, alpha=alpha)
                    ax.add_patch(patch)
