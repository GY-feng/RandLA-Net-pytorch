import sys
from pathlib import Path
from typing import Union

import cupy as cp
import numpy as np
import open3d as o3d
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC
from processor.cutting.point_cloud_grid_cutting import PointCloudGridCutter
from filter.by_classification import filter_by_classification

def classify_ground_point_by_minz(
        pc: Union[PC, PointCloudGridCutter]
) -> PC:
    """
    :param pc: 点云对象
    :return: 筛选出的地面点云对象
    """    
    # 准备点云数据
    pc = filter_by_classification(pc, '2')  # 在地面点标记的基础上取筛选最小值
    cutter = PointCloudGridCutter(pc)
    cutter.cut(0.1, pc.crs, use_cache=False)
    # cutter.make_cache()
    
    assert cutter.cutted is True, "请先进行网格划分"

    exist_dims = cutter.pc.exist_dimensions
    new_pc_data = {dim: [] for dim in exist_dims}

    # 预计算非z维度
    non_z_dims = [dim for dim in exist_dims if dim != 'z']
    
    for key in tqdm(cutter.grid_indices.keys(), desc="筛选地面点"):
        block = cutter[key]
        new_pc_data['z'].append(cp.min(block.z))
        
        for dim in non_z_dims:
            new_pc_data[dim].append(cp.mean(getattr(block, dim)))
       
    # 生成最终点云对象
    new_pc = PC()
    for dim, values in new_pc_data.items():
        setattr(new_pc, dim, cp.array(values))
    
    # 复制元数据
    new_pc.offsets = cutter.pc.offsets
    new_pc.crs = cutter.pc.crs
    new_pc.scales = cutter.pc.scales
    new_pc.classification = cp.full(len(new_pc_data['z']), 4, dtype=cp.uint8)  # 设置地面点分类

    return new_pc

if __name__ == "__main__":
    import os
    from filter.by_classification import filter_by_classification
    pc = PC()
    pc.load_from_las("/home/CloudPointProcessing/点云实验20250708/DJI_202507081723_162_点云实验20250708重复扫描40m-90度-6cm/raw/las/LAS.las")

    pc = classify_ground_point_by_minz(pc)

    pc.export_to_las('/home/CloudPointProcessing/点云实验20250708/DJI_202507081723_162_点云实验20250708重复扫描40m-90度-6cm/raw/las_ground/by_minz.las')
    # from checker.check_point_cloud import check_point_cloud
    # check_point_cloud(pc)