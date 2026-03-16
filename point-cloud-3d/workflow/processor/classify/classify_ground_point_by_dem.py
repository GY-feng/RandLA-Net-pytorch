import sys
from pathlib import Path

import rasterio
import cupy as cp
import numpy as np
import open3d as o3d
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC

def classify_ground_point_by_dem(
        pc: PC, 
        dem_path: str, 
        flann_multiplier: int = 1, 
        threshold: float = 0.1
) -> PC:
    """
    根据数字高程模型，筛选出地面点
    :param pc: 点云对象
    :param dem: DEM文件路径
    :param flann_multiplier: Z轴扩展搜索倍数（≥0）
    :param threshold: Z轴高度阈值（米）

    :return: 筛选出的地面点云对象
    """
    # 输入数据处    
    dem = rasterio.open(dem_path)
    dem_pc = PC()
    dem_pc.load_from_dem(dem_path)
    
    # 准备点云数据
    points = cp.vstack((pc.x, pc.y, pc.z)).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.get()))
    
    # 计算DEM栅格尺寸
    dem_block_width = (dem.bounds.right - dem.bounds.left) / dem.width
    dem_block_height = (dem.bounds.top - dem.bounds.bottom) / dem.height
    if not np.isclose(dem_block_width, dem_block_height, atol=1e-9):
        raise ValueError("DEM网格不是正方形")
    dem_block_size = dem_block_width
    base_radius = np.sqrt((dem_block_size / 2)**2 * 2)  # 对角线半径
    
    # 构建KDTree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    ground_indices_list = []  # 使用列表收集所有有效索引
    
    # 处理每个DEM点
    for dem_point in tqdm(cp.vstack((dem_pc.x, dem_pc.y, dem_pc.z)).T.get(), desc="正在遍历DEM点", unit='点'):
        x_dem, y_dem, z_dem = dem_point
        candidate_indices = set()
        
        # 基础搜索
        k, idx, _ = pcd_tree.search_radius_vector_3d([x_dem, y_dem, z_dem], base_radius)
        if k > 0:
            candidate_indices.update(idx)
        
        # 扩展搜索
        for i in range(1, flann_multiplier + 1):
            # 向上搜索
            k, idx, _ = pcd_tree.search_radius_vector_3d(
                [x_dem, y_dem, z_dem + i * base_radius], base_radius)
            if k > 0:
                candidate_indices.update(idx)
            # 向下搜索
            k, idx, _ = pcd_tree.search_radius_vector_3d(
                [x_dem, y_dem, z_dem - i * base_radius], base_radius)
            if k > 0:
                candidate_indices.update(idx)
        
        if not candidate_indices:
            continue
            
        # 转换为数组并过滤
        candidate_indices = cp.array(list(candidate_indices), dtype=cp.int32)
        candidates = points[candidate_indices]
        
        # 计算边界条件
        x_min, x_max = x_dem - dem_block_size/2, x_dem + dem_block_size/2
        y_min, y_max = y_dem - dem_block_size/2, y_dem + dem_block_size/2
        z_min, z_max = z_dem - threshold, z_dem + threshold
        
        mask = (
            (candidates[:, 0] >= x_min) & (candidates[:, 0] <= x_max) &
            (candidates[:, 1] >= y_min) & (candidates[:, 1] <= y_max) &
            (candidates[:, 2] >= z_min) & (candidates[:, 2] <= z_max)
        )
        
        valid_indices = candidate_indices[mask]
        if valid_indices.size > 0:
            ground_indices_list.append(valid_indices)
    
    # 合并并去重
    if ground_indices_list:
        ground_indices = cp.concatenate(ground_indices_list)
        ground_indices = cp.unique(ground_indices)
    else:
        ground_indices = cp.array([], dtype=np.int32)
    
    # 生成最终点云对象
    new_pc = pc[ground_indices]
    new_pc.classification = cp.full(len(ground_indices), 3, dtype=cp.uint8)  # 设置地面点分类
    return new_pc, ground_indices


    return new_pc
if __name__ == "__main__":
    import os
    pc = PC()
    pc.load_from_las("./data/AK0.las")
    pcA = classify_ground_point_by_dem(pc, "data/AK0_DEM.tif", flann_multiplier=2, threshold=0.1)
    from checker.check_point_cloud import check_point_cloud
    check_point_cloud(pcA)