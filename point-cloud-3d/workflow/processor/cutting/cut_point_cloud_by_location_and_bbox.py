import sys
import math
from pathlib import Path
from typing import Union

import cupy as cp
from pyproj import CRS, Transformer

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC
from utils import logger

def cut_point_cloud_by_location_and_bbox(
    pc: PC,
    center_x: float,
    center_y: float,
    coordinate: Union[str, CRS],
    width_m: float,
    height_m: float,
    rotation_deg: float = 0,
) -> PC:
    """
    从点云中裁切矩形区域内的点
    
    参数:
    - pc: PointCloud对象
    - center_x: 矩形中心X坐标（经度）
    - center_y: 矩形中心Y坐标（纬度）
    - coordinate: 矩形坐标系（EPSG:xxxx）
    - width_m: 矩形宽度（米）
    - height_m: 矩形高度（米）
    - rotation_deg: 矩形旋转角度（度，相对于正北方向顺时针）
    
    返回:
    - 包含矩形区域内点的新点云
    """

    # 检查坐标系
    if pc.crs is None:
        raise ValueError("点云对象没有坐标系")

    if isinstance(coordinate, str):
        try:
            bbox_crs = CRS.from_epsg(int(coordinate.split(':')[-1]))
        except ValueError:
            raise ValueError(f"无效的坐标系: {coordinate}")
    else:
        bbox_crs = coordinate
    
    # 坐标转换准备
    if bbox_crs != pc.crs:
        logger.warning(f"点云坐标系 ({pc.crs}) 与指定坐标系 ({bbox_crs}) 不匹配")
        logger.warning("尝试使用pyproj进行坐标系转换")
        transformer = Transformer.from_crs(bbox_crs, pc.crs, always_xy=True)
        center_x_proj, center_y_proj = transformer.transform(center_x, center_y)
    else:
        center_x_proj, center_y_proj = center_x, center_y
    
    # 获取点云坐标（使用GPU加速）
    points = cp.vstack((pc.x, pc.y)).T

    # 计算相对坐标
    dx = points[:, 0] - center_x_proj
    dy = points[:, 1] - center_y_proj
    # 处理旋转角度
    theta = math.radians(-rotation_deg)  # 转换为逆时针弧度
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    # 旋转坐标变换
    rotated_x = dx * cos_theta - dy * sin_theta
    rotated_y = dx * sin_theta + dy * cos_theta
    # 构建筛选条件
    half_width = width_m / 2
    half_height = height_m / 2
    mask = (rotated_x >= -half_width) & (rotated_x <= half_width) & \
           (rotated_y >= -half_height) & (rotated_y <= half_height)
    
    if not cp.any(mask):
        logger.warning("警告: 没有点在指定矩形区域内")
    
    # 创建新点云
    new_pc = pc[mask]
    
    return new_pc

# 测试代码
if __name__ == "__main__":
    from checker.check_cuda_status import check_cuda_status
    check_cuda_status()

    # 测试参数示例, , 
    test_params = {
        "center_lon": 113.93423726460573,
        "center_lat": 23.670950338371863,
        "epsg": "EPSG:4326", # WGS84经纬坐标系
        "width": 100,    # 米
        "height": 100,   # 米
        "rotation": 45  # 度
    }

    pc = PC()
    pc.load_from_las("./data/AK0.las")
    
    pcA = cut_point_cloud_by_location_and_bbox(
        pc,
        test_params["center_lon"],
        test_params["center_lat"],
        test_params["epsg"],
        test_params["width"],
        test_params["height"],
        test_params["rotation"]
    )

    from checker.check_point_cloud import check_point_cloud
    check_point_cloud(pcA)

else:
    from checker.check_cuda_status import check_cuda_status
    if not check_cuda_status(log=False, check_cv2=False, check_o3d=False):
        raise ImportError(f"CUDA依赖缺失, 无法导入{__name__}")