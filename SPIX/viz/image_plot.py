### image plot ####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import matplotlib.patches as patches
from shapely.geometry import MultiPoint, Polygon, LineString,MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import ConvexHull, Delaunay
import logging



# Configure logging to display warnings and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def image_plot(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding_segment',  # Use the segmented embedding
    figsize=(10, 10),
    point_size=None,  # Set default to None
    scaling_factor=10,
    origin=True,
    plot_boundaries=True,  # Toggle boundary plotting
    boundary_method='convex_hull',  # 'convex_hull', 'alpha_shape', 'oriented_bbox', 'buffered_line'
    alpha=0.3,  # Transparency for polygons
    boundary_color='black',  # Color for boundaries
    boundary_linewidth=1.0,  # Line width for boundaries
    alpha_shape_alpha=0.1,  # Alpha for alpha shapes
    aspect_ratio_threshold=3.0,  # Threshold to determine if segment is thin and long
    jitter=1e-6  # Jitter value to avoid collinearity
):
    """
    Visualize embeddings as an image with optional segment boundaries.
    
    [Docstring remains unchanged]
    """
    if len(dimensions) not in [1, 3]:
        raise ValueError("Only 1 or 3 dimensions can be used for visualization.")
    
    # Extract embeddings
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Extract spatial coordinates
    if origin:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    else:
        tiles = adata.uns['tiles']
    
    # Prepare tile colors and merge with coordinates
    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors['barcode'] = adata.obs.index
    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="right").dropna().reset_index(drop=True)

    if 'Segment' in adata.obs.columns:
        # Map 'Segment' based on 'barcode'
        coordinates_df['Segment'] = coordinates_df['barcode'].map(adata.obs['Segment'])
        # Check for any missing mappings
        if coordinates_df['Segment'].isnull().any():
            missing = coordinates_df['Segment'].isnull().sum()
            logging.warning(f"{missing} barcodes in coordinates_df do not have corresponding 'Segment' labels in adata.obs.")
            # Drop these rows or handle them differently
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
        cols = (coordinates['Grey'].values[:, np.newaxis] * 255).astype(int)
        cols = np.repeat(cols, 3, axis=1) / 255.0  # Normalize to [0,1]
        cols = cols.tolist()
    
    # Automatically determine point_size if not provided
    if point_size is None:
        figure_area = figsize[0] * figsize[1]  # in square inches
        point_size = scaling_factor * (figure_area / 100)  # Adjust denominator as needed

    if 'Segment' in adata.obs.columns:
        # Determine plot boundaries based on spatial coordinates
        x_min, x_max = coordinates['x'].min(), coordinates['x'].max()
        y_min, y_max = coordinates['y'].min(), coordinates['y'].max()
        plot_boundary_polygon = Polygon([
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_max),
            (x_max, y_min)
        ])
    
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
    scatter = ax.scatter(**sc_kwargs)
    ax.set_aspect('equal')
    ax.axis('off')
    title = f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
    ax.set_title(title, fontsize=figsize[0] * 1.5)
    
    if 'Segment' in adata.obs.columns and plot_boundaries:
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
                    # Clip the buffered polygon to plot boundaries
                    clipped_polygon = buffered_line.intersection(plot_boundary_polygon)
                    if not clipped_polygon.is_empty:
                        if isinstance(clipped_polygon, Polygon):
                            polygons = [clipped_polygon]
                        else:
                            # Handle MultiPolygon
                            polygons = list(clipped_polygon)
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
                        # Optionally, plot a simple bounding box or skip
                        # Example: skip plotting boundary
                        continue
                    try:
                        # Check for unique points
                        unique_points = np.unique(points, axis=0)
                        if unique_points.shape[0] < 3:
                            logging.warning(f"Segment {seg} has less than 3 unique points. ConvexHull cannot be computed.")
                            continue
                        # Check for collinearity
                        if is_collinear(unique_points):
                            logging.warning(f"Segment {seg} points are collinear. Using LineString as boundary.")
                            line = LineString(unique_points)
                            clipped_line = line.intersection(plot_boundary_polygon)
                            if not clipped_line.is_empty:
                                if isinstance(clipped_line, LineString):
                                    lines = [clipped_line]
                                else:
                                    # Handle MultiLineString
                                    lines = list(clipped_line)
                                for ln in lines:
                                    x, y = ln.xy
                                    ax.plot(x, y, color=boundary_color, linewidth=boundary_linewidth, alpha=alpha)
                            continue
                        # Add jitter to avoid collinearity
                        if is_almost_collinear(unique_points):
                            logging.info(f"Segment {seg} points are nearly collinear. Adding jitter to compute ConvexHull.")
                            unique_points += np.random.uniform(-jitter, jitter, unique_points.shape)
                        hull = ConvexHull(unique_points)
                        hull_points = unique_points[hull.vertices]
                        polygon = Polygon(hull_points)
                    except Exception as e:
                        logging.error(f"ConvexHull failed for segment {seg}: {e}")
                        # Attempt alternative methods
                        continue
                elif boundary_method == 'alpha_shape':
                    polygon = alpha_shape(points, alpha=alpha_shape_alpha)
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

                    # Clip the polygon to the plot boundary to prevent it from going outside
                    clipped_polygon = polygon.intersection(plot_boundary_polygon)
                    if clipped_polygon.is_empty:
                        logging.warning(f"Clipped polygon is empty for segment {seg}.")
                        continue

                    # Create a patch from the clipped polygon
                    if isinstance(clipped_polygon, Polygon):
                        polygons = [clipped_polygon]
                    else:
                        # Handle MultiPolygon
                        polygons = list(clipped_polygon)

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

                    # Clip the polygon to the plot boundary to prevent it from going outside
                    clipped_polygon = polygon.intersection(plot_boundary_polygon)
                    if clipped_polygon.is_empty:
                        logging.warning(f"Clipped polygon is empty for segment {seg}.")
                        continue

                    # Create a patch from the clipped polygon
                    if isinstance(clipped_polygon, Polygon):
                        polygons = [clipped_polygon]
                    else:
                        # Handle MultiPolygon
                        polygons = list(clipped_polygon)

                    for poly in polygons:
                        patch = patches.Polygon(list(poly.exterior.coords), closed=True, 
                                                fill=False, edgecolor=boundary_color, 
                                                linewidth=boundary_linewidth, alpha=alpha)
                        ax.add_patch(patch)
        
        # Set plot limits to the plot boundary
        if 'Segment' in adata.obs.columns:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
    
    # Display the figure
    plt.show()
    # Do not return the figure to prevent automatic display
    


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

