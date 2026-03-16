import math
import sys
from pathlib import Path

from pyproj import CRS, Transformer

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC
from utils.logger import logger
from utils.cmos_size import cmos_size
from processor.cutting.cut_point_cloud_by_location_and_bbox import cut_point_cloud_by_location_and_bbox
from checker.check_exif_data_with_exiftool import check_exif_data_with_exiftool


def cut_point_cloud_by_dji_image_with_bbox(image_path: str, pc: PC, cmos, bbox=(2640, 1978, 2640, 1978, 0), is_reverse=True, exiftool_path = "./ExifTool/exiftool"):
    """
        根据半径截取点云
        参数：
            image_path: str, 图像路径
            las_path: str, 点云路径
            cmos: str, 传感器型号
            bbox: tuple, 裁切矩形的以像素为单位的参数(矩形中心X，矩形中心Y，矩形长，矩形高，矩形旋转角(顺时针为0到-90，逆时针为+90到0))
    """
    assert len(bbox) == 5, "bbox参数必须是5个元素"

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
    
    # 获取GPS坐标（优先使用XMP数据）
    lat = float(exif_data.get('XMP:GPSLatitude', exif_data['EXIF:GPSLatitude']))
    lon = float(exif_data.get('XMP:GPSLongitude', exif_data['EXIF:GPSLongitude']))
    logger.info(f"图片坐标：{lon, lat}")
    # 计算图像中心点
    image_center_x = image_width_px / 2
    image_center_y = image_height_px / 2
    
    # 解析bbox参数
    bbox_center_x, bbox_center_y, bbox_width_px, bbox_height_px, bbox_rotation = bbox
    
    # 计算像素偏移
    dx_pixel = bbox_center_x - image_center_x
    dy_pixel = bbox_center_y - image_center_y
    
    # 转换为米
    gsd_width_m = gsd_width_cm / 100
    gsd_height_m = gsd_height_cm / 100
    dx_east = dx_pixel * gsd_width_m
    dy_north = -dy_pixel * gsd_height_m  # 图像Y轴向下，取反
    
    # 获取云台偏航角（顺时针为正）
    yaw_deg = float(exif_data['XMP:GimbalYawDegree'])
    logger.info(f"云台偏航角：{yaw_deg}°")
    theta = math.radians(yaw_deg)

    # 旋转偏移量
    east_rotated = dx_east * math.cos(theta) + dy_north * math.sin(theta)
    north_rotated = -dx_east * math.sin(theta) + dy_north * math.cos(theta)
    
    # 转换到UTM坐标
    pc_crs = pc.crs
    wgs84_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84_crs, pc_crs, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    # 计算新中心点
    new_easting = easting + east_rotated
    new_northing = northing + north_rotated
    new_lon, new_lat =  new_easting, new_northing
    
    # 计算矩形实际尺寸
    width_m = bbox_width_px * gsd_width_m
    height_m = bbox_height_px * gsd_height_m
    
    # 总旋转角度（云台偏航 + bbox旋转）
    total_rotation = yaw_deg
    if is_reverse:
        total_rotation = -total_rotation
    total_rotation += bbox_rotation

    return cut_point_cloud_by_location_and_bbox(
        pc, new_lon, new_lat, pc.crs, width_m, height_m, total_rotation
    )

if __name__ == '__main__':
    las_path = "./data/AK0.las"
    img_path = "./data/AK0_A.JPG"
    # las_data = cut_point_cloud_by_dji_image_with_bbox(img_path, las_path, "L2", (2640, 1978, 200, 500, 0))

    pc = PC()
    pc.load_from_las(las_path)

    pcA = cut_point_cloud_by_dji_image_with_bbox(img_path, pc, "L2", (2640, 1978, 5280, 3956, 0))
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