#-*- encoding:utf-8 -*-
import numpy as np
import cupy as cp
import sys
import math
from pathlib import Path
from typing import Tuple
from scipy.interpolate import SmoothBivariateSpline

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger

def get_neighborhood_points(center_key, cutter, radius=1):
    """
    获取以 center_key 为中心的 (2*radius+1) × (2*radius+1) 邻域内的所有点云
    
    参数
    ----
    center_key : tuple[int, int]
        当前网格的 key，例如 (1200, 850)
    radius : int, default 1
        扩展半径。radius=1 → 3×3 区域；radius=2 → 5×5 区域
    
    返回
    ----
    x, y, z : np.ndarray
        邻域内所有点的坐标（已拼接好，可直接用于拟合）
        如果邻域内一个点都没有，返回空数组
    """
    # 1. 计算所有需要查询的 key（包含中心 + 周围）
    cx, cy = center_key
    neighbor_keys = [
            (cx + dx, cy + dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
        ]
    
    # 2. 取出所有邻域网格的点索引，注意已经过滤了超出边界的网格以及空洞网格，但是性能不一定高
    idx_chunks = [cutter.grid_indices.get(k) for k in neighbor_keys if k in cutter.grid_indices]
    if not idx_chunks:
        return np.array([]), np.array([]), np.array([])

    all_idx = cp.concatenate([cp.asarray(chunk) for chunk in idx_chunks])
    pc = cutter.pc
    x = cp.asnumpy(pc.x[all_idx])
    y = cp.asnumpy(pc.y[all_idx])
    z = cp.asnumpy(pc.z[all_idx])
    return x, y, z


def generate_grid_coordinates(key, block_size, num_per_side=20):
    """
    生成均匀分布的网格坐标
    参数
    ----------
    key:
        中心网格xy
    num_per_side: int
        各采样点之间的xy间距，1米网格下默认是20个间隔，即400个点
    返回
    -------
    sampled_x, sampled_y : ndarray
        均匀分布的网格坐标
    """
    cx, cy = key
    x_min, y_min = cx * block_size, cy * block_size
    x_max, y_max = x_min + block_size, y_min + block_size
    num_per_side = math.ceil(num_per_side * block_size)

    sampled_x = np.linspace(x_min, x_max, num_per_side)
    sampled_y = np.linspace(y_min, y_max, num_per_side)
    sampled_x, sampled_y = np.meshgrid(sampled_x, sampled_y)

    return sampled_x, sampled_y  # bisplev 需要 1D


def fit_surface_with_spline(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sampled_x: np.ndarray | None = None,
    sampled_y: np.ndarray | None = None,
    kx: int = 3,
    ky: int = 3,
    s: float | None = None,   # 推荐值，兼顾严格插值和数值稳定
) -> Tuple[np.ndarray, bool]:
    """
    使用 SciPy 推荐的 SmoothBivariateSpline 进行 B-样条曲面拟合（z = f(x,y)）。

    参数
    ----------
    x, y, z : ndarray, shape (N,)
        原始散点坐标（点云）。
    sampled_x, sampled_y : ndarray, optional
        评估网格的一维坐标。
        如果提供两者，会返回网格化的 fitted_z (2D)。
        如果不提供，会在原始散点位置返回拟合值（1D）。
    kx, ky : int, default 3
        样条阶数，3=三次（推荐），1=线性。
    s : float | None, default 1e-9
        平滑因子：
            - s=0：强制严格插值（点少时可以，点多时容易警告）
            - s=1e-9：几乎严格插值（残差 ~1e-10），但没有警告、震荡
            - s=None：自动选择平滑程度（适合有噪声的数据）

    返回
    -------
    fitted_z : ndarray
        如果提供了 sampled_x/y：shape (len(sampled_y), len(sampled_x))
        否则：shape (N,)
    """
    success = True
    try:
        # ------------------- 1. 拟合样条 -------------------
        # 点太少时自动降阶，防止 "length must be at least (kx+1)*(ky+1)" 报错
        n_points = len(x)
        min_points = (kx + 1) * (ky + 1)
        if n_points < min_points:
            logger.warning(f"点数 {n_points} < {min_points}，请考虑舍弃对该网格的曲面拟合")

        spl = SmoothBivariateSpline(x, y, z, kx=kx, ky=ky, s=s)
        logger.info(f"SmoothBivariateSpline | 点数量={n_points} | RMS={np.sqrt(spl.get_residual()):.2f}")

        # ------------------- 2. 评估 -------------------
        if sampled_x is not None and sampled_y is not None:
            # 这里用 .ev() 是安全的向量化方式，一般不会爆内存
            fitted_z = spl.ev(sampled_x, sampled_y)
            return fitted_z, success
        else:
            # 只在原始散点上返回拟合值（用于验证残差等）
            fitted_z = spl(x, y)
            return np.array(fitted_z), success

    except Exception as e:
        success = False
        logger.error(f"样条曲面拟合失败：{e}")
        # 返回全 NaN，形状与期望输出一致
        if sampled_x is not None and sampled_y is not None:
            return np.full((len(sampled_y), len(sampled_x)), np.nan), success
        else:
            return np.full_like(z, np.nan), success