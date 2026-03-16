#-*- encoding:utf-8 -*-
import numpy as np
import cupy as cp
import sys
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from typing import Optional
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent.parent))

from workflow.compare.griddiff.utils.image_processing import create_custom_diverging_cmap

custom_cmap = create_custom_diverging_cmap()

def plot_statistic_heatmap(data, fig, axes, title, with_colorbar: bool, origin: str,
         cmap=None, x_unique=None, y_unique=None, underlay_rgb: Optional[np.ndarray] = None, refine=False):
    """在静态热力图上叠加RGB图像"""
    if isinstance(data, Image.Image):
        data = np.array(data)
    
    if isinstance(data, cp.ndarray):
        data = data.get()

    if underlay_rgb is not None:
        if isinstance(underlay_rgb, Image.Image):
            underlay_rgb = np.array(underlay_rgb)
        if isinstance(underlay_rgb, cp.ndarray):
            underlay_rgb = underlay_rgb.get()
    
    x_unique = np.arange(data.shape[1]) if x_unique is None else x_unique
    y_unique = np.arange(data.shape[0]) if y_unique is None else y_unique
    
    # x_unique是None的情况下，x_min等也是None，相当于是默认坐标范围
    x_min = min(x_unique)
    x_max = max(x_unique)
    y_min = min(y_unique)
    y_max = max(y_unique)

    if cmap:
        img = axes.imshow(data, cmap=cmap, origin=origin, extent=[x_min, x_max, y_min, y_max], aspect='equal')
    else:
        abs_max = np.abs(data).max()
        # 使用 2% 和 98% 分位数，避免单个极端值破坏整体色阶
        lower = np.percentile(data, 1)
        upper = np.percentile(data, 100)
        abs_max = max(abs(lower), abs(upper), abs_max * 0.1) # 至少保留一点范围

        vmin = -abs_max
        vmax = abs_max
        if refine:
            norm = TwoSlopeNorm(vmin=-0.55, vcenter=0, vmax=0.55)
        else:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        img = axes.imshow(data, cmap=custom_cmap, norm=norm, origin=origin, extent=[x_min, x_max, y_min, y_max], aspect='equal')
        # img.set_clim(vmin, vmax)  # 同步 clim，防止 colorbar 仍按数据范围缩放

    if underlay_rgb is not None:
        axes.imshow(underlay_rgb, origin='upper', extent=[x_min, x_max, y_min, y_max], alpha=0.4)

    axes.set_title(title)
    
    if with_colorbar:
        cbar = fig.colorbar(img, ax=axes, fraction=0.046, pad=0.04)


def plotly_heatmap(fig, heatmap, title, x_unique=None, y_unique=None, fit_flags=None):
    """使用plotly绘制可交互式热力图"""

    x_unique = np.asarray(x_unique)
    y_unique = np.asarray(y_unique)
    x_min, x_max = x_unique.min(), x_unique.max()
    y_min, y_max = y_unique.min(), y_unique.max()

    x_range = np.linspace(x_min, x_max, num=heatmap.shape[1])  # 生成 x 轴的线性范围
    y_range = np.linspace(y_min, y_max, num=heatmap.shape[0])  # 生成 y 轴的线性范围

    # 根据数据的最大值、2% 和 98% 分位数来设定颜色映射范围
    abs_max = np.abs(heatmap).max()
    lower = np.percentile(heatmap, 1)
    upper = np.percentile(heatmap, 100)
    abs_max = max(abs(lower), abs(upper), abs_max * 0.1)
    vmin = -abs_max
    vmax = abs_max
    
    # 将 matplotlib colormap 转换为 plotly 兼容的 colorscale 格式
    colorscale = [
        [i / 255.0, f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'] 
        for i, rgb in enumerate(custom_cmap(np.linspace(0, 1, 256)))  # 使用 256 个值来生成平滑的颜色渐变
    ]

    # 修改为左边缘坐标（网格左下角）
    # 假设均匀网格，步长 dx, dy
    dx = (x_max - x_min) / (heatmap.shape[1])      # 注意：除以列数，不是列数+1
    dy = (y_max - y_min) / (heatmap.shape[0])

    # 左边缘坐标（左下角）
    x_left = np.linspace(x_min, x_max - dx, num=heatmap.shape[1])   # 最后一个左边缘 = x_max - dx
    y_bottom = np.linspace(y_min, y_max - dy, num=heatmap.shape[0])

    fig.add_trace(go.Heatmap(
        x=x_left, y=y_bottom, z=heatmap,
        colorscale=colorscale, showscale=True, zsmooth=False,
        zmin=vmin, zmax=vmax,
        hovertemplate="X=%{x}<br>Y=%{y}<br>value=%{z}<extra></extra>"
    ))

    if fit_flags is not None:
        rows, cols = np.where(fit_flags == 1)
        fig.add_trace(go.Scatter(
            x=x_range[cols],
            y=y_range[rows],
            mode='markers',
            marker=dict(
                size=10,
                symbol='circle-open',
                line=dict(color='red', width=2),
            ),
            name='已拟合曲面',
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[x_min, x_max],
                    constrain="domain",
                    ticklen=10,
                    ),
        yaxis=dict(range=[y_min, y_max], scaleanchor="x", 
                    constrain="domain",
                    ticklen=10,
                    ),
        legend=dict(x=1, y=1.1)
    )
