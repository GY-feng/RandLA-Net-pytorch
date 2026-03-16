from __future__ import annotations

import numpy as np
from typing import Iterable


def points_in_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    """
    Ray casting algorithm (vectorized over points).
    Returns boolean mask of points inside polygon.
    """
    pts = points_xy
    poly = polygon_xy
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be (N,2)")
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon_xy must be (M,2)")

    x = pts[:, 0]
    y = pts[:, 1]

    inside = np.zeros(len(pts), dtype=bool)
    n = poly.shape[0]
    if n < 3:
        return inside

    xj, yj = poly[-1, 0], poly[-1, 1]
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= intersect
        xj, yj = xi, yi
    return inside


def points_in_polygons(points_xy: np.ndarray, polygons_xy: Iterable[np.ndarray]) -> np.ndarray:
    """
    Union of multiple polygons.
    """
    mask = np.zeros(points_xy.shape[0], dtype=bool)
    for poly in polygons_xy:
        mask |= points_in_polygon(points_xy, poly)
    return mask
