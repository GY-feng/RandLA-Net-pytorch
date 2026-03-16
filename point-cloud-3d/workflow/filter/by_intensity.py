import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pointcloud import PointCloud as PC

def filter_by_intensity(pc: PC, min_intensity=0, max_intensity=65535) -> PC:
    """根据反射率范围过滤点云"""
    # 创建过滤掩码
    mask = (pc.intensity >= min_intensity) & (pc.intensity <= max_intensity)
    # 应用过滤
    filtered_pc = pc[mask]
    print(f"\n过滤结果: 原始点数 {pc.point_nums} -> 保留点数 {filtered_pc.point_nums}")
    return filtered_pc