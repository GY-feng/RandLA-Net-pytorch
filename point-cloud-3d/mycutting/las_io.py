from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import laspy


def read_las(path: Path) -> laspy.LasData:
    return laspy.read(path)


def get_xyz(las: laspy.LasData) -> np.ndarray:
    return np.vstack((las.x, las.y, las.z)).T.astype(np.float64, copy=False)


def get_xy(las: laspy.LasData) -> np.ndarray:
    return np.vstack((las.x, las.y)).T.astype(np.float64, copy=False)


def write_las_subset(las: laspy.LasData, mask: np.ndarray, out_path: Path) -> laspy.LasData:
    header = las.header.copy()
    new_las = laspy.LasData(header)
    new_las.points = las.points[mask]
    if hasattr(las, "evlrs") and las.evlrs is not None:
        new_las.evlrs = las.evlrs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_las.write(out_path)
    return new_las
