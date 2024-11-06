import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import MultiPoint, Polygon, LineString, MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import ConvexHull, Delaunay
import logging
def _check_spatial_data(uns, library_id):
    spatial_mapping = uns.get('spatial', {})
    if library_id is None:
        if len(spatial_mapping) > 1:
            raise ValueError(
                "Multiple libraries found in adata.uns['spatial']. Please specify library_id."
            )
        elif len(spatial_mapping) == 1:
            library_id = list(spatial_mapping.keys())[0]
        else:
            library_id = None
    spatial_data = spatial_mapping.get(library_id, None)
    return library_id, spatial_data


def _check_img(spatial_data, img, img_key, bw=False):
    if img is None and spatial_data is not None:
        if img_key is None:
            img_key = next((k for k in ['hires', 'lowres'] if k in spatial_data['images']), None)
            if img_key is None:
                raise ValueError("No image found in spatial data.")
        img = spatial_data['images'][img_key]
    if bw and img is not None:
        # Convert to grayscale using luminosity method
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img, img_key


def _check_scale_factor(spatial_data, img_key, scaling_factor):
    if scaling_factor is not None:
        return scaling_factor
    elif spatial_data is not None and img_key is not None:
        return spatial_data['scalefactors'][f'tissue_{img_key}_scalef']
    else:
        return 1.0


def is_collinear(points, tol=1e-8):
    """
    Check if a set of points are exactly collinear within a tolerance.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2).
    tol : float
        Tolerance for collinearity.
    
    Returns
    -------
    bool
        True if all points are collinear, False otherwise.
    """
    if len(points) < 3:
        return True
    p0, p1 = points[0], points[1]
    for p in points[2:]:
        area = 0.5 * np.abs((p1[0] - p0[0]) * (p[1] - p0[1]) - (p[0] - p0[0]) * (p1[1] - p0[1]))
        if area > tol:
            return False
    return True


def is_almost_collinear(points, tol=1e-5):
    """
    Check if a set of points are nearly collinear within a tolerance.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2).
    tol : float
        Tolerance for determining "almost" collinear.
    
    Returns
    -------
    bool
        True if points are almost collinear, False otherwise.
    """
    if len(points) < 3:
        return True
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        y_pred = slope * x + intercept
        residuals = y - y_pred
        return np.max(np.abs(residuals)) < tol
    except np.linalg.LinAlgError:
        return False
