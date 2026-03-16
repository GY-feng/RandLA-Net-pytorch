import sys
from pathlib import Path
import laspy
import cupy as cp

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.logger import logger

def check_las_info(las: laspy.LasData, block_size: float = 1):
    """打印LAS文件的详细头部和扩展信息"""
    logger.info("\n========== LAS文件信息 ==========")
    header = las.header
    logger.info(f"文件版本: {header.version}")
    logger.info(f"点数量: {header.point_count}")
    logger.info(f"点格式: {header.point_format.id}")
    
    # 高程单位检查
    logger.info("高程单位信息:")
    try:
        # 尝试获取垂直单位
        vertical_unit = "未知"
        if hasattr(header, 'vertical_units'):
            units = {0: "未指定", 1: "米", 2: "英尺", 3: "美国测量英尺", 4: "国际英尺"}
            vertical_unit = units.get(header.vertical_units, '未知')
        
        # 通过缩放因子辅助判断
        z_scale = header.z_scale
        if vertical_unit == "未知":
            if z_scale in (1.0, 0.1, 0.01, 0.001):
                vertical_unit = "可能为米"
            elif abs(z_scale - 0.3048) < 0.0001:
                vertical_unit = "可能为英尺"
        
        logger.info(f"  垂直单位: {vertical_unit}")
        logger.info(f"  Z值缩放因子: {z_scale}")
        logger.info(f"  Z值偏移量: {header.z_offset}")
        logger.info(f"  Z值范围: [{header.z_min:.3f}, {header.z_max:.3f}] (原始值)")
        logger.info(f"  Z值范围: [{(header.z_min * z_scale + header.z_offset):.3f}, "
                  f"{(header.z_max * z_scale + header.z_offset):.3f}] (缩放后)")
    except Exception as e:
        logger.warning(f"  高程单位解析失败: {str(e)}")
    
    # 坐标系检查
    logger.info("坐标系信息:")
    try:
        crs = header.parse_crs()
        if crs:
            logger.info(f"  EPSG: {crs.to_epsg()}")
            # 尝试从CRS获取垂直单位信息
            try:
                if crs.is_vertical:
                    logger.info(f"  垂直坐标系: {crs.vertical_crs.name}")
            except:
                pass
        else:
            logger.info("  未定义坐标系")
    except Exception as e:
        logger.warning(f"  坐标系解析失败: {str(e)}")
    
    # 扩展信息
    logger.info("\n扩展信息 (VLRs):")
    for vlr in header.vlrs:
        logger.info(f"  {vlr.description}:")
        logger.info(f"    记录ID: {vlr.record_id}")
        logger.info(f"    用户ID: {vlr.user_id}")
        logger.info(f"    数据长度: {vlr.record_data_bytes} 字节")
    
    # 反射率统计
    if hasattr(las, 'intensity'):
        intensity = las.intensity
        logger.info("\n反射率统计:")
        logger.info(f"  最小值: {cp.min(intensity)}")
        logger.info(f"  最大值: {cp.max(intensity)}")
        logger.info(f"  平均值: {cp.mean(intensity):.2f}")

if __name__ == "__main__":
    from checker.check_cuda_status import check_cuda_status
    if not check_cuda_status():
        raise RuntimeError("CUDA环境检查失败")

    las_file_path = "/home/kobayashi_bairuo/PointCloudBlockComparisonSystem/project/0303e182-ea1c-4278-874a-f99d05b369cf/raw/las/2025-04-02-15-35-07.las"
    las = laspy.read(las_file_path)
    check_las_info(las)

else:
    from checker.check_cuda_status import check_cuda_status
    if not check_cuda_status(log=False, check_cv2=False, check_o3d=False):
        raise ImportError(f"CUDA依赖缺失, 无法导入{__name__}")