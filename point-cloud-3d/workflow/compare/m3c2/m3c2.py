import sys
import os
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud
import numpy as np
from typing import Optional, Union, Dict
from tqdm import tqdm
from scipy.spatial import cKDTree

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

def _is_gpu_array(a) -> bool:
    return _HAS_CUPY and isinstance(a, cp.ndarray)

def _as_numpy(a):
    if _is_gpu_array(a):
        return cp.asnumpy(a)
    return a

def _stack_xyz_numpy(pc) -> np.ndarray:
    # 不改变 PointCloud 的 status，仅把坐标取到 CPU 内存
    x = _as_numpy(pc.x); y = _as_numpy(pc.y); z = _as_numpy(pc.z)
    return np.vstack((x, y, z)).T

def _normals_numpy(normals: Optional[np.ndarray], core: 'PointCloud') -> np.ndarray:
    if normals is None:
        if core.normals is None:
            raise ValueError("未提供 normals，且 core_points.normals 为空")
        normals = core.normals
    return _as_numpy(normals)

def _projected_signed_distances(points_xyz: np.ndarray, center: np.ndarray, normal: np.ndarray) -> np.ndarray:
    vecs = points_xyz - center[None, :]
    return vecs @ normal

def _in_cylindrical_neighborhood(points_xyz: np.ndarray,
                                 center: np.ndarray,  # 圆柱体的中心，形状为 (3,)，表示三维空间中的一个点 (x, y, z)
                                 normal: np.ndarray,  # 圆柱体的轴方向，形状为 (3,)，表示圆柱体沿哪个方向延伸（通常是单位向量）
                                 radius: float,
                                 half_length: float,
                                 only_positive_dir: bool) -> np.ndarray:
    """
    一个点属于圆柱邻域，必须满足两个条件：
    轴向条件：点到中心点的向量在法向量方向上的投影（轴向距离）在指定范围内。
    径向条件：点到圆柱轴的垂直距离（径向距离）小于或等于半径。
    """
    vecs = points_xyz - center[None, :]  # 计算每个点到圆柱中心点的位移向量,用于后续分解轴向和径向分量
    axial = vecs @ normal  # 计算每个点到中心点的向量在法向量方向上的有符号投影距离（轴向距离）
    if only_positive_dir:
        axial_mask = (axial >= 0.0) & (axial <= half_length)
    else:
        axial_mask = (np.abs(axial) <= half_length)
    radial_vecs = vecs - np.outer(axial, normal)  # radial_vecs[i] = vecs[i] - (axial[i] * normal)
    radial_sq = np.einsum('ij,ij->i', radial_vecs, radial_vecs)
    radial_mask = radial_sq <= radius * radius
    return np.nonzero(axial_mask & radial_mask)[0]

def _compute_stats(distances: np.ndarray, use_median: bool):
    n = distances.size
    if n == 0:
        return np.nan, 0.0
    if n == 1:
        return float(distances[0]), 0.0
    if use_median:
        d_sorted = np.sort(distances)
        median = float(np.median(d_sorted))
        num = n
        num_pts_each_half = (num + 1) // 2
        offset_second_half = num // 2
        q1 = float(np.median(d_sorted[:num_pts_each_half]))
        q3 = float(np.median(d_sorted[offset_second_half:offset_second_half + num_pts_each_half]))
        return median, (q3 - q1)
    else:
        mean = float(np.mean(distances))
        std = float(np.std(distances, ddof=0))
        return mean, std

def _precision_map_sigma_at(neigh_xyz: np.ndarray,
                            neigh_indices: np.ndarray,
                            normal: np.ndarray,
                            sigma_x: np.ndarray,
                            sigma_y: np.ndarray,
                            sigma_z: np.ndarray,
                            scale: float) -> float:
    if neigh_indices.size == 0:
        return 0.0
    if neigh_indices.size == 1:
        idx = int(neigh_indices[0])
    else:
        G = neigh_xyz.mean(axis=0)
        diffs = neigh_xyz - G[None, :]
        idx_local = int(np.argmin(np.einsum('ij,ij->i', diffs, diffs)))
        idx = int(neigh_indices[idx_local])
    sx = float(sigma_x[idx]) * scale
    sy = float(sigma_y[idx]) * scale
    sz = float(sigma_z[idx]) * scale
    return float(np.sqrt((normal[0]*sx)**2 + (normal[1]*sy)**2 + (normal[2]*sz)**2))

def m3c2(
    cloud1: 'PointCloud',
    cloud2: 'PointCloud',
    core_points: Union['PointCloud', np.ndarray],
    normals: Optional[np.ndarray] = None,
    *,
    projection_radius: float,
    projection_half_depth: float,
    use_median: bool = False,
    progressive_search: bool = False,
    progressive_steps: int = 10,
    only_positive_dir: bool = False,
    min_points_for_stats: int = 3,
    confidence_level: float = 1.96,  # 默认 95% 置信度
    registration_rms: float = 0.0,
    precision_maps: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    project_output_on_cloud2: bool = False,
    return_device: str = "match"  # "cpu" | "gpu" | "match"
):
    """
    - cloud1, cloud2: PointCloud
    - core_points: PointCloud 或 np.ndarray(K,3)
    - normals: 可选 np.ndarray(K,3)。若不提供且 core_points 为 PointCloud, 则使用 core_points.normals
    - confidence_level: 90% 置信度: 1.645; 95% 置信度: 1.96;99% 置信度: 2.576;99.9% 置信度:3.291
    - precision_maps: 形如
        {
          'cloud1': {'sigma_x': arr1, 'sigma_y': arr2, 'sigma_z': arr3, 'scale': float},
          'cloud2': {...}
        }
      其中 sigma_* 长度需与各自点云点数一致（与 cloud1/cloud2 对齐）
    - return_device: 结果返回到 CPU/GPU。"match" 表示跟随 core_points（若为 PointCloud 则跟随 core_points.status）
    """
    # 解析 core 点与法向（numpy）
    if isinstance(core_points, np.ndarray):
        core_xyz_np = core_points
        normals_np = _as_numpy(normals)
        core_is_gpu = False
    else:
        core_xyz_np = _stack_xyz_numpy(core_points)
        normals_np = _normals_numpy(normals, core_points)
        core_is_gpu = (core_points.status == "GPU")

    # 解析两帧点云坐标（numpy）
    cloud1_xyz = _stack_xyz_numpy(cloud1)
    cloud2_xyz = _stack_xyz_numpy(cloud2)

    # 归一化法向
    normals_np = normals_np / np.linalg.norm(normals_np, axis=1, keepdims=True)

    ball_radius = float(np.sqrt(projection_radius**2 + projection_half_depth**2))
    tree1 = cKDTree(cloud1_xyz)
    tree2 = cKDTree(cloud2_xyz)

    K = core_xyz_np.shape[0]
    dist = np.full(K, np.nan, dtype=np.float64)
    lod  = np.full(K, np.nan, dtype=np.float64)
    significant = np.zeros(K, dtype=bool)
    std1 = np.zeros(K, dtype=np.float64)  # 初始化 cloud1 邻域点的标准差/四分位距
    std2 = np.zeros(K, dtype=np.float64)  # 初始化 cloud2 邻域点的标准差/四分位距
    n1 = np.zeros(K, dtype=np.int32)  # 初始化 cloud1 邻域点的数量
    n2 = np.zeros(K, dtype=np.int32)  # 初始化 cloud2 邻域点的数量
    projected_point = core_xyz_np.copy()  # 复制核心点坐标，用于存储投影点

    use_pm = precision_maps is not None

    for i in tqdm(range(K), desc="在核心点上计算鲁棒距离"):
        P = core_xyz_np[i]
        N = normals_np[i]

        # cloud1
        idx_ball_1 = tree1.query_ball_point(P, r=ball_radius)
        neigh1_all = cloud1_xyz[idx_ball_1] if len(idx_ball_1) else np.empty((0, 3))
        mean1 = np.nan
        disp1 = 0.0
        valid1 = False

        if progressive_search:
            prev_count = -1
            for step in range(1, progressive_steps + 1):
                curr_half = projection_half_depth * step / progressive_steps
                if neigh1_all.shape[0] == 0:
                    break
                idx_cyl_1 = _in_cylindrical_neighborhood(neigh1_all, P, N, projection_radius, curr_half, only_positive_dir)
                neigh1 = neigh1_all[idx_cyl_1]
                d1 = _projected_signed_distances(neigh1, P, N)
                if d1.size != prev_count and d1.size > 0:
                    if d1.size >= min_points_for_stats:
                        mean1, disp1 = _compute_stats(d1, use_median)
                        valid1 = True
                        # 早停条件：乘以 2.0 是为了近似覆盖数据的“显著范围”：
                        # 对于标准差，2 倍标准差大致覆盖正态分布的约 95% 数据（根据经验法则）
                        # 对于四分位距，2 倍四分位距提供一个保守的范围估计，覆盖数据的典型波动
                        # 判断当前半高是否已经足够大，覆盖了点云的显著分布，无需进一步扩大
                        if np.abs(mean1) + 2.0 * disp1 < curr_half:
                            break
                    prev_count = d1.size
        else:
            if neigh1_all.shape[0] > 0:
                idx_cyl_1 = _in_cylindrical_neighborhood(neigh1_all, P, N, projection_radius, projection_half_depth, only_positive_dir)
                neigh1 = neigh1_all[idx_cyl_1]
            else:
                neigh1 = np.empty((0, 3))
            d1 = _projected_signed_distances(neigh1, P, N)
            if d1.size > 0:
                mean1, disp1 = _compute_stats(d1, use_median)
                valid1 = True

        n1[i] = 0 if not valid1 else int(d1.size)
        std1[i] = float(disp1)

        # cloud2
        idx_ball_2 = tree2.query_ball_point(P, r=ball_radius)
        neigh2_all = cloud2_xyz[idx_ball_2] if len(idx_ball_2) else np.empty((0, 3))
        mean2 = np.nan
        disp2 = 0.0
        valid2 = False

        if progressive_search:
            prev_count = -1
            for step in range(1, progressive_steps + 1):
                curr_half = projection_half_depth * step / progressive_steps
                if neigh2_all.shape[0] == 0:
                    break
                idx_cyl_2 = _in_cylindrical_neighborhood(neigh2_all, P, N, projection_radius, curr_half, only_positive_dir)
                neigh2 = neigh2_all[idx_cyl_2]
                d2 = _projected_signed_distances(neigh2, P, N)
                if d2.size != prev_count and d2.size > 0:
                    if d2.size >= min_points_for_stats:
                        mean2, disp2 = _compute_stats(d2, use_median)
                        valid2 = True
                        if np.abs(mean2) + 2.0 * disp2 < curr_half:
                            break
                    prev_count = d2.size
        else:
            if neigh2_all.shape[0] > 0:
                idx_cyl_2 = _in_cylindrical_neighborhood(neigh2_all, P, N, projection_radius, projection_half_depth, only_positive_dir)
                neigh2 = neigh2_all[idx_cyl_2]
            else:
                neigh2 = np.empty((0, 3))
            d2 = _projected_signed_distances(neigh2, P, N)
            if d2.size > 0:
                mean2, disp2 = _compute_stats(d2, use_median)
                valid2 = True

        n2[i] = 0 if not valid2 else int(d2.size)

        # 精度图（若有，则用 PM 覆盖 std2；LOD 亦根据 PM 计算）
        if use_pm and valid2:
            pm2 = precision_maps.get('cloud2', None)
            if pm2 is not None:
                sigma_x2 = _as_numpy(pm2['sigma_x'])
                sigma_y2 = _as_numpy(pm2['sigma_y'])
                sigma_z2 = _as_numpy(pm2['sigma_z'])
                scale2 = float(pm2.get('scale', 1.0))
                disp2 = _precision_map_sigma_at(
                    neigh2, np.array(idx_ball_2)[idx_cyl_2] if len(idx_ball_2) else np.array([], dtype=int),
                    N, sigma_x2, sigma_y2, sigma_z2, scale2
                )
        std2[i] = float(disp2)

        if project_output_on_cloud2 and valid2 and not np.isnan(mean2):
            projected_point[i] = P + mean2 * N  # 核心点沿法向量方向移动 mean2 距离，得到新的投影点

        if valid1 and valid2 and not (np.isnan(mean1) or np.isnan(mean2)):
            dist[i] = float(mean2 - mean1)
            LODStdDev = np.nan
            if use_pm:
                pm1 = precision_maps.get('cloud1', None)
                if pm1 is not None and n1[i] > 0:
                    sigma_x1 = _as_numpy(pm1['sigma_x'])
                    sigma_y1 = _as_numpy(pm1['sigma_y'])
                    sigma_z1 = _as_numpy(pm1['sigma_z'])
                    scale1 = float(pm1.get('scale', 1.0))
                    disp1_pm = _precision_map_sigma_at(
                        neigh1, np.array(idx_ball_1)[idx_cyl_1] if len(idx_ball_1) else np.array([], dtype=int),
                        N, sigma_x1, sigma_y1, sigma_z1, scale1
                    )
                    LODStdDev = disp1_pm**2 + disp2**2
            else:
                if n1[i] >= min_points_for_stats and n2[i] >= min_points_for_stats:
                    LODStdDev = (std1[i]**2)/max(n1[i],1) + (std2[i]**2)/max(n2[i],1)
            if not np.isnan(LODStdDev):
                lod[i] = float(confidence_level * (np.sqrt(LODStdDev) + registration_rms))
                significant[i] = (dist[i] < -lod[i]) or (dist[i] > lod[i])

    # 返回设备选择
    if return_device == "match":
        return_device = "gpu" if core_is_gpu else "cpu"

    def _maybe_to_gpu(arr):
        if return_device == "gpu":
            if not _HAS_CUPY:
                raise RuntimeError("未安装 cupy，无法返回 GPU 结果")
            return cp.asarray(arr)
        return arr

    return {
        "dist": _maybe_to_gpu(dist),
        "lod": _maybe_to_gpu(lod),
        "significant": _maybe_to_gpu(significant.astype(np.bool_)),
        "std1": _maybe_to_gpu(std1),
        "std2": _maybe_to_gpu(std2),
        "n1": _maybe_to_gpu(n1),
        "n2": _maybe_to_gpu(n2),
        "projected_point": _maybe_to_gpu(projected_point),
    }