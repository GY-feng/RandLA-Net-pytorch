import sys
from typing import Union
from pathlib import Path

import cupy as cp
from pyproj import CRS, Transformer

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC
from utils.logger import logger



def cut_point_cloud_by_location_and_radius(
    pc: PC,
    center_x: float,
    center_y: float,
    coordinate: Union[str,CRS],
    radius: float
) -> PC: 
    """
    从点云中裁切圆形区域内的点

    参数:
    - pc: PointCloud对象
    - x: 圆心X坐标
    - y: 圆心Y坐标
    - coordinate: 圆心坐标系（EPSG:xxxx）
    - radius: 切割半径(米)
    
    返回:
    - 包含圆形区域内点的新LAS数据
    
    异常:
    - ValueError: 如果点云不是投影坐标系
    - TypeError: 如果输入参数类型不正确
    """
    
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
    
    # 获取点云坐标
    points = cp.vstack((pc.x, pc.y)).T
    
    # 计算点到圆心的距离
    distances = cp.sqrt((points[:, 0] - center_x_proj)**2 + (points[:, 1] - center_y_proj)**2)
    # 选择在半径内的点
    mask = distances <= radius
    
    if not cp.any(mask):
        logger.warning("警告: 没有点在指定半径内")
    
    # 创建新点云
    new_pc = PC()
    
    # 复制头文件信息
    new_pc.scales = pc.scales
    new_pc.offsets = pc.offsets
    new_pc.crs = pc.crs
    
    # 筛选点数据
    for dim in pc.exist_dimensions:
        setattr(new_pc, dim, getattr(pc, dim)[cp.asarray(mask)])
    
    return new_pc

if __name__ == "__main__":
    # 测试参数示例, , 
    test_params = {
        "center_lon": 113.93423726460573,
        "center_lat": 23.670950338371863,
        "epsg": "EPSG:4326", # WGS84经纬坐标系
        "radius": 50,    # 米
    }

    pc = PC()
    pc.load_from_las("./data/AK0.las")
    
    pcA = cut_point_cloud_by_location_and_radius(
        pc,
        test_params["center_lon"],
        test_params["center_lat"],
        test_params["epsg"],
        test_params["radius"]
    )

    from checker.check_point_cloud import check_point_cloud
    check_point_cloud(pcA)

else:
    from checker.check_cuda_status import check_cuda_status
    if not check_cuda_status(log=False, check_cv2=False, check_o3d=False):
        raise ImportError(f"CUDA依赖缺失, 无法导入{__name__}")