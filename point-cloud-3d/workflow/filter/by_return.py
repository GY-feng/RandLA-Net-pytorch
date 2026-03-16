import sys
from pathlib import Path
import cupy as cp

sys.path.append(str(Path(__file__).parent.parent))

from pointcloud import PointCloud as PC
from utils.logger import logger

def filter_by_return(pc: PC, return_code: str = '1') -> PC:
    """根据回波数筛选点云"""
    codes = return_code.split(',')
    codes = [int(code) for code in codes]
    # 创建分类掩码
    mask = cp.isin(pc.return_number, cp.asarray(codes))
    # 应用过滤
    filtered_pc = pc[mask]

    logger.info(f"\n过滤结果: 原始点数 {pc.point_nums} -> 保留点数 {filtered_pc.point_nums}")
    return filtered_pc