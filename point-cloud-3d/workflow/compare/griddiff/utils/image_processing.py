#-*- encoding:utf-8 -*-
import numpy as np
import cupy as cp
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy import ndimage
from skimage.transform import resize
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from workflow.processor.converter.point_cloud_rasterization import point_cloud_rasterization
from utils.logger import logger

def compute_common_regions(cfg, lasB_cutter, common_keys):
    """快速拼接 LAS B 的重叠区域图像"""
    I_chunks = []  # 索引小块
    for k in tqdm(common_keys, desc='拼接common_lasB以可视化鸟瞰图'):
        idx = lasB_cutter.grid_indices.get(k, [])
        if len(idx)==0: continue
        I_chunks.append(idx)
    if not I_chunks:
        raise ValueError("未找到 common_lasB 有效点索引")
    I = np.concatenate(I_chunks)
    common_lasB = lasB_cutter.pc[I]

    return point_cloud_rasterization(
        pc=common_lasB,
        sampling_rate=cfg.rasterization.sampling_rate,
        sampling_mode=cfg.rasterization.sampling_mode,
        enable_fill=False,
        enable_plt=False,
        fill_algorithm='nearest',
        reverse=cfg.rasterization.reverse
    )


def erode_mask_outerlayer(mask, remove_ratio=0.20):
    """
        只从二值掩码的最外围边缘向内均匀剔除指定比例的白色像素，内部空洞完全保留不动。
        可有效去除点云稀疏、误差大的最外层边缘区
        同时保证中间因遮挡或扫描盲区形成的黑洞不被影响

        参数
        ----
        mask : np.ndarray
            二值掩码
            
        remove_ratio : float, default 0.20
            要从最外围剔除的白色像素比例，范围 (0.0, 1.0)。
            0.20 表示剔除约 20% 的最外层像素。

        返回
        ----
        np.ndarray
            与输入完全相同的 shape 和 dtype 的新掩码，
            已剔除最外围约 remove_ratio 比例的白色像素。
            内部所有空洞均 100% 保留。
    """
    mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        return mask.copy()
    
    total_valid = mask.sum()
    target_remove = int(total_valid * remove_ratio + 0.5)
    
    # 把二值图像中所有被前景（白色）完全包围的背景黑洞填成前景（白色）。
    filled = ndimage.binary_fill_holes(mask)
    
    # 计算每个像素到最外边界的欧式距离
    dist = ndimage.distance_transform_edt(filled)
    
    # 取出所有有效掩码的距离，并排序
    valid_element_distances = dist[mask]  # (N,) 有效掩码数量
    if len(valid_element_distances) <= target_remove:
        return np.zeros_like(mask, dtype=mask.dtype)
    
    # 找到几何距离阈值：第 target_remove 小的距离
    threshold = np.sort(valid_element_distances)[target_remove - 1]
    
    # 剔除所有距离小于等于的有效像素，即最外层
    to_remove = (dist <= threshold) & mask
    result = mask & (~to_remove)

    return result.astype(mask.dtype)


def apply_mask_to_rgba(image, mask):
    """将单通道掩码应用到 RGBA 图像上"""
    if isinstance(image, Image.Image):
        image = np.array(image)

    if isinstance(image, cp.ndarray):
        image = image.get()

    if isinstance(mask, cp.ndarray):
        mask = mask.get()

    image = image.copy()
    if image.shape[2] != 4:
        raise ValueError("Input image must be RGBA with 4 channels")

    # 调整掩码大小（如果维度不一致）
    if image.shape[:2] != mask.shape:
        logger.info("LAS图像和掩码维度不一致，需要对掩码进行resize")
        # order=0，表示使用最近邻插值，保持掩码值为 0 或 1，不产生中间值（如 0.3、0.7）
        mask = resize(mask, image.shape[:2], order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

    # 把掩码区域设置为完全透明（0），其余保持原值
    image[..., 3] = image[..., 3] * mask.astype(image.dtype)

    return image, mask


def geometric_downsample(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """
    按几何映射把 0/1 掩码缩到 (out_h, out_w)：
      - 小图每个像素对应大图中的一个实数边界矩形（完全等比分割，无取整切块）
      - 该矩形与哪些大图像素“有接触”，这些像素若出现 0 -> 小图像素置 0；否则置 1
    接触判定：只要有重叠面积或边界接触，都算接触
    """
    assert mask.ndim == 2
    H, W = mask.shape
    out_h, out_w = output_shape
    out = np.ones((out_h, out_w), dtype=mask.dtype)

    # —— 预处理：把 0/1 掩码转为“是否为 0”的 0/1，再做二维前缀和（带 1 行 1 列 padding，便于 O(1) 查询）
    zeros = (mask == 0).astype(np.int64)
    S = np.pad(zeros, ((1, 0), (1, 0)), mode='constant')  # shape: (H+1, W+1)
    S = S.cumsum(axis=0).cumsum(axis=1)

    # 预先计算低分辨率像素在高分辨率中的真实边界（浮点）
    r_edges_f = np.linspace(0.0, float(H), out_h + 1)
    c_edges_f = np.linspace(0.0, float(W), out_w + 1)

    # “接触即算”的整数行列范围：
    # 与区间 [r0,r1) 有接触的高分辨率行 r 满足： [r, r+1) 与 [r0, r1) 相交或接触
    # 等价于 r ∈ [floor(r0), ceil(r1)-1]；列同理
    r0_idx = np.floor(r_edges_f[:-1]).astype(int)
    r1_idx = np.ceil (r_edges_f[1: ]).astype(int) - 1   # 包含性右端点（行）
    c0_idx = np.floor(c_edges_f[:-1]).astype(int)
    c1_idx = np.ceil (c_edges_f[1: ]).astype(int) - 1   # 包含性右端点（列）

    # 夹紧到合法索引（极少数边界对齐时可能产生 -1 或 H/W）
    r0_idx = np.clip(r0_idx, 0, H-1); r1_idx = np.clip(r1_idx, 0, H-1)
    c0_idx = np.clip(c0_idx, 0, W-1); c1_idx = np.clip(c1_idx, 0, W-1)

    # 矩形和查询函数（使用带 padding 的前缀和 S，输入为“包含性”索引）
    def rect_sum_inclusive(r0, r1, c0, c1):
        # 把包含性索引转为 S 的半开区间 [r0, r1+1) × [c0, c1+1)
        r0p, r1p = r0, r1 + 1
        c0p, c1p = c0, c1 + 1
        return S[r1p, c1p] - S[r0p, c1p] - S[r1p, c0p] + S[r0p, c0p]

    # 逐块判定（O(out_h*out_w) 次 O(1) 查询）
    for i in range(out_h):
        rr0, rr1 = r0_idx[i], r1_idx[i]
        for j in range(out_w):
            cc0, cc1 = c0_idx[j], c1_idx[j]
            # 有 0 则置 0；否则保持 1
            out[i, j] = 0 if rect_sum_inclusive(rr0, rr1, cc0, cc1) > 0 else 1

    return out


def build_valid_mask(image):
    """
    构建掩码，排除空洞区域
    
    参数:
        image: np.ndarray 或 Image.Image，必须是 RGB 或 RGBA 图像
        
    返回:
        valid_mask: np.ndarray，uint8 类型，1 表示有效，0 表示掩掉
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if isinstance(image, cp.ndarray):
        image = image.get()

    # 分通道处理
    R, G, B, A = image[..., 0], image[..., 1], image[..., 2], image[..., 3]
    invalid_mask = A == 0

    # 合并掩码：排除掉黑色/阴影区域和无值区域 1是图像区域，0是空洞区域
    valid_mask = ~(invalid_mask)

    return valid_mask.astype(np.uint8)


def create_custom_diverging_cmap():
    """
    自定义 colormap：
    - 负值极值 → #08f9ff  (亮青色，代表最大负偏差)
    - 0        → #010082  (深蓝，代表无误差/基准)
    - 正值极值 → #ff0000  (纯红，代表最大正偏差)
    """
    # 定义关键点颜色（从负到正）
    colors = [
        "#08ffe6",   # 负值最大（亮青）
        "#010082",   # 0 值（深蓝）
        "#ee3f3f",   # 正值最大（纯红）
    ]
    
    cmap = LinearSegmentedColormap.from_list(
        name="DeepBlueZero",
        colors=colors,
        N=256  # 越高越平滑
    )

    try:
        # 方式1：官方推荐（Matplotlib 3.5+）
        plt.colormaps.register(cmap, force=True)
    except AttributeError:
        # 方式2：保底方案（少数旧环境）
        plt.cm.register_cmap(cmap=cmap)
    
    return cmap