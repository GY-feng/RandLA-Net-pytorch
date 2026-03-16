import sys
from pathlib import Path
import cupy as cp
import numpy as np
import open3d as o3d
import cv2
import laspy
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Literal, Union

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC
from utils.logger import logger

def point_cloud_rasterization(
    pc: PC,
    sampling_rate: float = 1000,
    sampling_mode: Literal['average', 'max_z'] = 'average',
    save_path: Optional[str] = None,
    background_color: tuple[int, int, int, int] = (0, 0, 0, 0),
    enable_fill: bool = False,  # 图片平滑算法开关
    enable_plt: bool = False,
    fill_algorithm: Literal['nearest', 'telea', 'morphology'] = 'nearest',  # 填充算法
    kernel_size: int = 3,  # 填充算法的核大小
    reverse: bool = True
) -> Image.Image:
    """CUDA加速的点云转图像转换器
    
    Args:
        pc: PointCloud对象
        sampling_rate: 采样率(>0)，值越大图像越清晰
        sampling_mode: 采样模式 ('average'或'max_z')
        save_path: 图片保存路径（None则显示）
        background_color: 背景色 (R,G,B,A)
    
    Returns:
        PIL.Image.Image: 生成的RGBA图像
    """
    
    # 获取点云数据
    x = pc.x
    y = pc.y
    z = pc.z
    
    # 处理颜色数据（LAS颜色通常是16bit，需要转换为8bit）- GPU操作
    if all(dim in pc.exist_dimensions for dim in ['red', 'green', 'blue']):
        if pc.red.dtype == np.uint16 and pc.green.dtype == np.uint16 and pc.blue.dtype == np.uint16:
            r = cp.asarray(pc.red >> 8).astype(cp.uint8)
            g = cp.asarray(pc.green >> 8).astype(cp.uint8)
            b = cp.asarray(pc.blue >> 8).astype(cp.uint8)
        elif pc.red.dtype == np.uint8 and pc.green.dtype == np.uint8 and pc.blue.dtype == np.uint8:
            r = cp.asarray(pc.red)
            g = cp.asarray(pc.green)
            b = cp.asarray(pc.blue)
        else:
            raise ValueError("颜色数据类型不支持，必须为uint16或uint8")

    else:
        # 如果没有颜色数据，使用高程值生成伪彩色 - GPU操作
        z_normalized = (pc.z - pc.z.min()) / (pc.z.max() - pc.z.min())
        r = cp.asarray((z_normalized * 255).astype(np.uint8))
        g = cp.asarray(((1 - z_normalized) * 255).astype(np.uint8))
        b = cp.zeros_like(r)
    
    # 计算边界
    min_x, max_x = cp.min(x), cp.max(x)
    min_y, max_y = cp.min(y), cp.max(y)
    x_range = cp.maximum(max_x - min_x, 1e-8)  # 避免除零
    y_range = cp.maximum(max_y - min_y, 1e-8)
    
    image_width = max(1, int(cp.asnumpy(x_range * sampling_rate / 100)))
    image_height = max(1, int(cp.asnumpy(y_range * sampling_rate / 100)))
    pixel_size_x = x_range / image_width
    pixel_size_y = y_range / image_height
    
    # logger.info(f"X轴范围:{x_range} Y轴范围:{y_range}")
    # logger.info(f"图像宽度:{image_width} 图像高度:{image_height}")
    # logger.info(f"X轴像素大小:{pixel_size_x} Y轴像素大小:{pixel_size_y}")
    # logger.info("")
    
    # 坐标到像素的映射 - GPU操作
    i = cp.clip(((x - min_x) / pixel_size_x).astype(cp.int32), 0, image_width-1)
    j = cp.clip(((y - min_y) / pixel_size_y).astype(cp.int32), 0, image_height-1)
    
    # 初始化GPU图像缓冲区
    rgba = cp.zeros((image_height, image_width, 4), dtype=cp.uint8)
    rgba[:] = cp.array(background_color, dtype=cp.uint8)
    
    if sampling_mode == 'average':
        # 平均颜色模式
        sum_r = cp.zeros((image_height, image_width), dtype=cp.float32)
        sum_g = cp.zeros_like(sum_r)
        sum_b = cp.zeros_like(sum_r)
        count = cp.zeros_like(sum_r, dtype=cp.int32)
        
        # 使用CUDA原子操作加速散射添加
        cp.add.at(sum_r, (j, i), r)
        cp.add.at(sum_g, (j, i), g)
        cp.add.at(sum_b, (j, i), b)
        cp.add.at(count, (j, i), 1)
        
        # 计算平均值
        mask = count > 0
        rgba[mask, 0] = (sum_r[mask] / count[mask]).astype(cp.uint8)
        rgba[mask, 1] = (sum_g[mask] / count[mask]).astype(cp.uint8)
        rgba[mask, 2] = (sum_b[mask] / count[mask]).astype(cp.uint8)
        rgba[mask, 3] = 255
    
    elif sampling_mode == 'max_z':
        # 最大Z值模式
        max_z = cp.full((image_height, image_width), -cp.inf)
        max_idx = cp.full((image_height, image_width), -1, dtype=cp.int32)
        
        # 并行寻找每个像素的最大Z索引
        z_gpu = cp.asarray(pc.z)
        linear_idx = cp.arange(len(z_gpu), dtype=cp.int32)
        cp.maximum.at(max_z, (j, i), z_gpu)
        cp.maximum.at(max_idx, (j, i), linear_idx)
        
        # 提取颜色 - GPU操作
        valid_mask = max_idx != -1
        valid_idx = max_idx[valid_mask]
        rgba[valid_mask, 0] = r[valid_idx]
        rgba[valid_mask, 1] = g[valid_idx]
        rgba[valid_mask, 2] = b[valid_idx]
        rgba[valid_mask, 3] = 255
    
    # 将GPU数据传回CPU
    rgba_cpu = cp.asnumpy(rgba)
    
    # 图像平滑处理
    if enable_fill:
        # 生成原始非空区域蒙版（Alpha>0的区域）
        original_mask = (rgba_cpu[:, :, 3] > 0).astype(np.uint8)
        
        # 对蒙版进行闭运算，消除空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        closed_mask = cv2.morphologyEx(original_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 只处理闭运算后的蒙版内区域（原图透明但被闭运算包含的区域）
        fill_mask = closed_mask - original_mask
        
        if np.any(fill_mask):  # 只有存在需要填充的区域时才处理
            src_rgb = rgba_cpu[:, :, :3].copy()
            
            if fill_algorithm in ['nearest', 'telea']:
                flags = cv2.INPAINT_NS if fill_algorithm == 'nearest' else cv2.INPAINT_TELEA
                radius = 3 if fill_algorithm == 'nearest' else 5
                filled_rgb = cv2.inpaint(src_rgb, fill_mask*255, radius, flags=flags)
            elif fill_algorithm == 'morphology':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                filled_rgb = cv2.morphologyEx(src_rgb, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 只修改闭运算蒙版内且原图透明的区域
            rgba_cpu = np.where(
                (fill_mask * closed_mask)[..., None],  # 添加通道维度用于广播
                np.dstack((filled_rgb, np.ones_like(fill_mask)*255)),  # 填充区域的Alpha设为255
                rgba_cpu
            )
    
    # 创建图像 - CPU操作
    img = Image.fromarray(rgba_cpu.astype(np.uint8), 'RGBA')

    # 如果reverse为True，则将图像上下翻转
    if reverse:
        img_np = np.array(img)  # 将PIL图像转换为numpy数组
        flipped_img_np = cv2.flip(img_np, 0)  # 使用OpenCV进行上下翻转（flipCode为0）
        img = Image.fromarray(flipped_img_np.astype(np.uint8), 'RGBA')  # 转换回PIL图像
    
    if save_path:
        img.save(save_path)
    if enable_plt:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    return img

if __name__ == "__main__":
    las_file = "/home/gary/CloudPointProcessing/L3L2/DJI_202510191519_255_点云实验20251019-40m-7ms-重复/raw/las/cloud_merged.las"
    
    pc = PC()
    pc.load_from_las(las_file)
    cloud_img = point_cloud_rasterization(
        pc=pc,
        sampling_rate=1500,
        sampling_mode='average',
        enable_fill=False,
        enable_plt=False,
        fill_algorithm='nearest',
        reverse=True)
    cloud_img.save('/home/gary/point-cloud-3d/output/可配对差分的点云图像/L3L2/DJI_202510191519_255_20251019-40m-7ms.png')