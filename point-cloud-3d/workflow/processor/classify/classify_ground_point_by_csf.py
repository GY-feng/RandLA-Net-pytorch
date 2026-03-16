from code import interact
import sys
from pathlib import Path
import CSF
import cupy as cp
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC

def classify_ground_point_by_csf(
    pc: PC,
    *,
    bSloopSmooth: bool = False,
    time_step: float = 0.65,
    class_threshold: float = 0.5,
    cloth_resolution: float = 1.0,
    rigidness: int = 3,
    interation: int = 500
) -> PC:
    """
    使用 CSF (Cloth Simulation Filter) 算法提取地面点。

    参数
    ----------
    pc : PC
        待处理的点云对象。

    bSloopSmooth : bool, default=False
        是否在布料模拟前对坡度进行平滑，适合噪声较大的点云。
        
    time_step : float, default=0.65
        每步时间增量（米/迭代）。越大收敛快但易穿透点云；越小更稳定。

    class_threshold : float, default=0.5
        点到布料的垂直距离阈值（米）。小于该值的点被标记为地面。

    cloth_resolution : float, default=0.5
        布料网格单元大小（米）。越小越精细，计算量越大。
        推荐范围：
            - 平坦区域：0.5~1.0 m
            - 丘陵/植被密集：0.05~0.2 m

    rigidness : int, default=3
        布料刚性（1~5）。越大布料越硬，适合平坦地形；越小越软，适合起伏地形。

    interation : int, default=500
        布料模拟最大迭代次数。迭代不足会导致布料未完全下落。

    返回
    -------
    PC
        只包含地面点的子点云，``classification`` 统一设为 ``3``（cupy uint8）。

    """
    # ---------- 1. 数据准备 ----------
    on_GPU = pc.status == 'GPU'
    xp = cp if on_GPU else np

    points = xp.column_stack((pc.x, pc.y, pc.z))
    points_np = points.get() if on_GPU else points  # 统一转为 NumPy (CSF C++ 端要求)
    points_np = points_np.astype(np.float64, copy=False)

    # ---------- 2. CSF 参数 ----------
    csf = CSF.CSF()
    csf.params.bSloopSmooth      = bSloopSmooth
    csf.params.time_step         = time_step
    csf.params.class_threshold   = class_threshold
    csf.params.cloth_resolution  = cloth_resolution
    csf.params.rigidness         = rigidness
    csf.params.interation        = interation

    csf.setPointCloud(points_np)

    # ---------- 3. 过滤 ----------
    ground_idx = CSF.VecInt()  # 创建一个空的C++ std::vector<int>容器, 用来接收被分类为“地面点”的点在原始点云中的索引
    offground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, offground_idx, exportCloth=True)  # 这两个容器是输出参数，不是返回值（C++ 风格直接修改参数的引用）

    # ---------- 4. 构造返回点云 ----------
    # 索引必须是 CPU numpy 数组（不能用 xp.fromiter）
    ground_arr_np = np.fromiter(ground_idx, dtype=np.int64, count=len(ground_idx))
    ground_arr = cp.asarray(ground_arr_np) if on_GPU else ground_arr_np

    offground_arr_np = np.fromiter(offground_idx, dtype=np.int64, count=len(offground_idx))
    offground_arr = cp.asarray(offground_arr_np) if on_GPU else offground_arr_np

    ground_pc = pc[ground_arr]
    ground_pc.classification = xp.full(len(ground_arr), 1, dtype=xp.uint8)

    offground_pc = pc[offground_arr]
    offground_pc.classification = xp.full(len(offground_arr), 0, dtype=xp.uint8)

    return ground_pc, ground_arr, offground_pc, offground_arr

if __name__ == "__main__":
    data_root = Path('/home/gary/CloudPointProcessing/20260214学校操场实验/DJI_202602141438_011_20260214学校操场水平无垫高-重复-有植被')
    pc = PC()
    pc.load_from_las(data_root / 'raw' / 'las' / 'cloud_merged.las')
    ground_pc, ground_arr, offground_pc, offground_arr = classify_ground_point_by_csf(pc,
                                       bSloopSmooth=False,
                                       time_step=0.5,
                                       class_threshold=0.01,
                                       cloth_resolution=0.02,
                                       rigidness=1,
                                       interation=500)