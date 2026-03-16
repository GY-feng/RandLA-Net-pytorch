import sys
from pathlib import Path
import cupy as cp

sys.path.append(str(Path(__file__).parent.parent))

from pointcloud import PointCloud as PC
from utils.logger import logger

def filter_by_classification(pc: PC, class_code: str = '2') -> PC:
    """根据分类代码筛选点云"""
    codes = class_code.split(',')
    codes = [int(code) for code in codes]
    # 创建分类掩码
    mask = cp.isin(pc.classification, cp.asarray(codes))
    # 应用过滤
    filtered_pc = pc[mask]

    logger.info(f"DJI地面点标记: 原始点数 {pc.point_nums} -> 保留点数 {filtered_pc.point_nums}")
    return filtered_pc, cp.where(mask)[0]