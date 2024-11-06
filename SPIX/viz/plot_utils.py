import numpy as np
import pandas as pd
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import logging

def rebalance_colors(coordinates, dimensions, method="minmax"):
    if len(dimensions) == 3:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        colors = coordinates.iloc[:, [4, 5, 6]].values
        
        if method == "minmax":
            colors = np.apply_along_axis(min_max, 0, colors)
        else:
            colors[colors < 0] = 0
            colors[colors > 1] = 1
        
        template = pd.concat([template, pd.DataFrame(colors, columns=["R", "G", "B"])], axis=1)
    
    else:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]].copy()
        colors = coordinates.iloc[:, 4].values
        
        if method == "minmax":
            colors = min_max(colors)
        else:
            colors[colors < 0] = 0
            colors[colors > 1] = 1
        
        template = pd.concat([template, pd.Series(colors, name="Grey")], axis=1)
    
    return template


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.
    
    Parameters
    ----------
    points : np.ndarray
        An array of shape (n, 2) representing the coordinates of the points.
    alpha : float
        Alpha value to influence the concaveness of the border. Smaller
        numbers don't fall inward as much as larger numbers. Too large,
        and you lose everything!
    
    Returns
    -------
    Polygon or MultiPolygon or None
        The resulting alpha shape polygon or None if not enough points.
    """
    if len(points) < 4:
        # Not enough points to form a polygon
        return None

    try:
        tri = Delaunay(points)
    except Exception as e:
        logging.error(f"Delaunay triangulation failed: {e}")
        return None

    triangles = points[tri.simplices]
    a = np.linalg.norm(triangles[:,0] - triangles[:,1], axis=1)
    b = np.linalg.norm(triangles[:,1] - triangles[:,2], axis=1)
    c = np.linalg.norm(triangles[:,2] - triangles[:,0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circum_r = (a * b * c) / (4.0 * area)
    filtered = circum_r < 1.0 / alpha
    triangles = tri.simplices[filtered]

    if len(triangles) == 0:
        logging.warning("No triangles found with the given alpha. Consider increasing alpha.")
        return None

    edges = set()
    edge_points = []
    for tri in triangles:
        edges.update([
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ])

    for edge in edges:
        edge_points.append(points[list(edge)])

    polygons = list(polygonize(edge_points))
    if not polygons:
        return None
    return unary_union(polygons)