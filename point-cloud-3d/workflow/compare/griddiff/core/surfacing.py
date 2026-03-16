import numpy as np
from typing import Literal
from scipy.interpolate import RegularGridInterpolator, griddata, RBFInterpolator

def interpolate_surface(points: np.ndarray, x_range: tuple, y_range: tuple, method: Literal['nearest', 'linear', 'cubic'], interval: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将离散散点插值到规则的矩形网格上（Gridding via Interpolation）。
    
    Args:
        points: (N, 3) 的散点数据 [x, y, z]
        x_range, y_range: 目标网格的范围
        interval: 网格步长（分辨率）
        method: 插值算法类型: linear(双线性插值)、nearest(最近邻插值)、cubic(三次样条插值)
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    num_x = int(np.ceil((x_max - x_min) / interval)) + 1
    num_y = int(np.ceil((y_max - y_min) / interval)) + 1

    xi = np.linspace(x_min, x_min + (num_x - 1) * interval, num_x)
    yi = np.linspace(y_min, y_min + (num_y - 1) * interval, num_y)
    
    XI, YI = np.meshgrid(xi, yi)
    
    ZI = griddata((points[:, 0], points[:, 1]), points[:, 2], (XI, YI), method=method, fill_value=np.nan)  # 可能存在凸包外的 NaN，但是nearest不会有NaN

    return XI, YI, ZI