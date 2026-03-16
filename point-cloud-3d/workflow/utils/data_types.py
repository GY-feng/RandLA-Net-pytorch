import numpy as np
import cupy as cp
from typing import Literal
import pyproj
from pointcloud import PointCloud as PC

def to_numpy(arr):
    """
    如果输入是 CuPy 数组，则将其转换为 NumPy 数组 (移动到 CPU)。
    如果输入已经是 NumPy 数组，则直接返回。
    """
    if isinstance(arr, cp.ndarray):
        return arr.get()
    return arr


def wrap_points_to_pc(x: np.ndarray, y: np.ndarray, z: np.ndarray, crs: pyproj.CRS, device: Literal['CPU', 'GPU'] = 'GPU') -> PC:
    
    pc = PC(type=device)
    
    valid_mask = ~np.isnan(z)
    
    x_flat = x[valid_mask].ravel()
    y_flat = y[valid_mask].ravel()
    z_flat = z[valid_mask].ravel()

    # setter 会根据 device 自动处理 numpy -> cupy 的转换
    pc.x = x_flat
    pc.y = y_flat
    pc.z = z_flat

    pc.crs = crs

    return pc