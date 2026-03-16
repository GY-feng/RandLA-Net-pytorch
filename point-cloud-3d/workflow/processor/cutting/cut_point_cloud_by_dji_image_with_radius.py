import math
import sys
import os
from typing import Union
from pathlib import Path
from pyproj import CRS, Transformer

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC
from utils.logger import logger
from utils.cmos_size import cmos_size
from processor.cutting.cut_point_cloud_by_location_and_radius import cut_point_cloud_by_location_and_radius
from checker.check_exif_data_with_exiftool import check_exif_data_with_exiftool

def cut_point_cloud_by_dji_image_with_radius(image_path: Union[str, os.PathLike], pc: PC, cmos, radius=-1, exiftool_path="./ExifTool/exiftool"):
    """
        根据半径截取点云

        参数：
            image_path: str, 图像路径
            pc: PointCloud, 点云对象
            cmos: str, 传感器型号
            radius: float, 截取半径，默认为-1，表示使用传感器分辨率作为半径
    """

    # 传感器参数（4/3英寸CMOS）

    sensor_width_mm, sensor_height_mm = cmos_size[cmos]
    logger.info(f"传感器大小：{sensor_width_mm}×{sensor_height_mm}mm")
    
    # 从EXIF提取元数据
    exif_data = check_exif_data_with_exiftool(image_path=image_path, exiftool_path=exiftool_path, is_print=False)
    
    # 图像分辨率（像素）
    
    image_width_px = int(exif_data['Composite:ImageSize'].split(" ")[0])
    image_height_px = int(exif_data['Composite:ImageSize'].split(" ")[1])
    logger.info(f"图像分辨率：{image_width_px}×{image_height_px}px")
    
    # 飞行参数
    relative_altitude_m = float(exif_data['XMP:RelativeAltitude'])
    focal_length_mm = float(exif_data['EXIF:FocalLength'])
    logger.info(f"相对高度：{relative_altitude_m}m, 焦距：{focal_length_mm}mm")
    
    # 计算像素大小（μm/像素）
    pixel_size_width_um = (sensor_width_mm * 1000) / image_width_px
    pixel_size_height_um = (sensor_height_mm * 1000) / image_height_px
    logger.info(f"像素大小（宽×高）: {pixel_size_width_um:.2f} μm × {pixel_size_height_um:.2f} μm")
    
    # 计算地面采样距离（GSD，cm/像素）
    gsd_width_cm = (relative_altitude_m * pixel_size_width_um / 1000) / focal_length_mm * 100  # 单位：cm
    gsd_height_cm = (relative_altitude_m * pixel_size_height_um / 1000) / focal_length_mm * 100
    logger.info(f"地面采样距离（宽×高）: {gsd_width_cm:.2f} cm/像素 × {gsd_height_cm:.2f} cm/像素")
    
    # 计算地面覆盖范围（m）
    ground_width_m = image_width_px * gsd_width_cm / 100  # GSD转回米
    ground_height_m = image_height_px * gsd_height_cm / 100
    logger.info(f"地面覆盖范围（宽×高）: {ground_width_m:.2f} m × {ground_height_m:.2f} m")
    
    # 计算矩形对角线的一半长度（m）
    if radius == -1:
        half_diagonal_m = math.sqrt(ground_width_m**2 + ground_height_m**2) / 2
    else:
        half_diagonal_m = radius
    
    # 输出结果（修正单位）
    logger.info(f"传感器像素大小（宽×高）: {pixel_size_width_um:.2f} × {pixel_size_height_um:.2f} μm")
    logger.info(f"地面采样距离（GSD）: {gsd_width_cm:.2f} cm/像素")
    logger.info(f"地面覆盖范围（宽×高）: {ground_width_m:.2f} m × {ground_height_m:.2f} m")
    logger.info(f"截取半径: {half_diagonal_m:.2f} m")

    lat = float(exif_data['EXIF:GPSLatitude'])
    lon = float(exif_data['EXIF:GPSLongitude'])
    logger.info(f"图片坐标：{lon, lat}")
    transformer = Transformer.from_crs(CRS.from_epsg(4326), pc.crs, always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)

    return cut_point_cloud_by_location_and_radius(pc, center_x, center_y, pc.crs, half_diagonal_m)


if __name__ == '__main__':
    las_path = "./data/AK0.las"
    img_path = "./data/AK0_A.JPG"

    pc = PC()
    pc.load_from_las(las_path)

    pcA = cut_point_cloud_by_dji_image_with_radius(img_path, pc, "L2", 30)

    from checker.check_point_cloud import check_point_cloud
    check_point_cloud(pcA)
    from processor.converter.point_cloud_rasterization import point_cloud_rasterization
    point_cloud_rasterization(
        pcA, 
        sampling_mode='max_z',
        sampling_rate=1000, 
        enable_fill=True, 
        enable_plt=True, 
        fill_algorithm='nearest', 
        save_path='./output/temp.png'
    )