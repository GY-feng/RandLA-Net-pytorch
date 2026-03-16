from __future__ import annotations
import numpy as np
import cupy as cp
import sys
import json
import warnings
from deprecated import deprecated
from typing import Optional, Tuple
from pathlib import Path
from functools import lru_cache
from decimal import Decimal

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import logger
from pointcloud import PointCloud as PC

try:
    from matplotlib.path import Path as _MPLPath  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from shapely.geometry import Polygon as _ShpPolygon, Point as _ShpPoint  # type: ignore
    from shapely.prepared import prep as _shp_prep  # type: ignore
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


def cut_point_cloud_by_polygon(
    pc: PC,
    polygon_xy: cp.ndarray,
    *,
    backend: str = "auto",          # "auto" | "cpuy"| | "shapely" | "mpl"
    include_boundary: bool = False,  # 边界是否算内部
) -> PC:
    """
    在 XY 平面按多边形裁剪点云，返回新的PC对象（与可选的原始索引）。
    """

    pts_xy = cp.column_stack([pc.x, pc.y])

    if pts_xy.ndim != 2 or pts_xy.shape[1] != 2:
        raise ValueError("pts_xy 应为 (N, 2) 数组")
    if polygon_xy.ndim != 2 or polygon_xy.shape[1] != 2:
        raise ValueError("polygon_xy 应为 (M, 2) 数组")

    # 如果第一个点和最后一个点是相同的（首尾闭合），就把最后一个点删掉，避免点在多边形内的判断等操作造成计算冗余
    if cp.allclose(polygon_xy[0], polygon_xy[-1]):
        polygon_xy = polygon_xy[:-1]

    # 后端选择
    chosen = backend
    if backend == "auto":
        if _HAS_SHAPELY:
            chosen = "shapely"
        elif _HAS_MPL:
            chosen = "mpl"
        else:
            chosen = "cupy"

    if chosen == "cupy":
        # cupy 射线法，速度最快
        # mask = _mask_by_cupy_ray(pts_xy, polygon_xy, include_boundary)
        mask = _mask_by_kernel_ray(pts_xy, polygon_xy, use_local_coords=True, include_boundary=True, origin='min', print_stats=True)
    elif chosen == "shapely":  # 精度最高
        mask = _mask_by_shapely(pts_xy, polygon_xy, include_boundary)
    elif chosen == 'mpl':
        mask = _mask_by_mpl(pts_xy, polygon_xy, include_boundary)

    else:
        logger.error(f'名为{backend}的后端绘图引擎不存在')
        raise ValueError(f'名为{backend}的后端绘图引擎不存在')
    
    new_pc = PC()
    
    new_pc.scales = pc.scales
    new_pc.offsets = pc.offsets
    new_pc.crs = pc.crs
    
    for dim in pc.exist_dimensions:
        setattr(new_pc, dim, getattr(pc, dim)[cp.asarray(mask)])

    return new_pc, mask


# -------------------- 工具函数区 --------------------
# @lru_cache(maxsize=None)
# def _get_kernel(device_id: int):
    
    _kernel_src  = r'''
    extern "C" __global__
    void pip_ray_kernel_stride(
        const double* __restrict__ ptsx,
        const double* __restrict__ ptsy,
        const int npts,
        const double* __restrict__ vx,
        const double* __restrict__ vy,
        const int m,
        const double xmin, const double xmax,
        const double ymin, const double ymax,
        const int include_boundary,
        unsigned char* __restrict__ outmask,
        unsigned char* __restrict__ dbg
    ){
        const int stride = gridDim.x * blockDim.x;
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        
        const double eps = 1e-8f;
        
        // bbox filter
        for (; i < npts; i += stride){
            unsigned char flag = 0;  // bit0: bbox pass, bit1: boundary hit,
                                 // bit2: condY seen, bit3: inside==1
            double x = ptsx[i];
            double y = ptsy[i];

            if (x < xmin || x > xmax || y < ymin || y > ymax){
                outmask[i] = 0;
                dbg[i] = flag;  // flag==0 means rejected by bbox
                continue;
            }
            flag |= 1;  // passed bbox

            if (include_boundary){
                for (int j = 0, k = m - 1; j < m; k = j++){
                    double xj = vx[j], yj = vy[j];
                    double xk = vx[k], yk = vy[k];
                    double vxj = xk - xj, vyj = yk - yj;
                    double wx  = x  - xj, wy  = y  - yj;
                    double cross = vxj * wy - vyj * wx;
                    if (fabs(cross) <= eps){
                        double dot = vxj * wx + vyj * wy;
                        double seglen2 = vxj * vxj + vyj * vyj;
                        if (dot >= -eps && dot <= seglen2 + eps){  
                            outmask[i] = 1;
                            flag |= 2;      // boundary hit
                            dbg[i] = flag;  // write debug state
                            goto NEXT_POINT;
                        }
                    }
                }
            }

            {
                int c = 0;
                for (int j = 0, k = m - 1; j < m; k = j++){
                    double xj = vx[j], yj = vy[j];
                    double xk = vx[k], yk = vy[k];
                    bool condY = ((yj <= y) && (y < yk)) || ((yk <= y) && (y < yj));
                    if (condY){
                        double dy = yk - yj;
                        double xints = xj + ((y - yj) * (xk - xj)) / dy;
                        if (x < xints) c = !c;
                    }
                }
                outmask[i] = (unsigned char)c;
                if (c) flag |= 8;  // inside
            }
            dbg[i] = flag;
            NEXT_POINT: ;
        }
    }
    ''';
    
    # 针对不同 GPU（设备）缓存各自的句柄
    with cp.cuda.Device(device_id):
        # CUDA源码编译成GPU可执行内核模块，得到CUDA内核句柄，调用时必须指定 block 和 grid 配置
        return cp.RawKernel(_kernel_src, 'pip_ray_kernel_stride')


# def _mask_by_kernel_ray(pts_xy: cp.ndarray, polygon_xy: cp.ndarray, include_boundary: bool=False) -> cp.ndarray:
    pts_xy = cp.asarray(pts_xy, dtype=cp.float64)
    polygon_xy = cp.asarray(polygon_xy, dtype=cp.float64)

    xmin = polygon_xy[:,0].min().item()
    xmax = polygon_xy[:,0].max().item()
    ymin = polygon_xy[:,1].min().item()
    ymax = polygon_xy[:,1].max().item()

    vx = polygon_xy[:,0].ravel()
    vy = polygon_xy[:,1].ravel()
    npts = pts_xy.shape[0]
    m = polygon_xy.shape[0]

    outmask = cp.empty(npts, dtype=cp.uint8)
    dbg = cp.zeros(npts, dtype=cp.uint8)  # <-- NEW
    
    dev_id = int(cp.cuda.Device().id)
    pip_ray_kernel_stride = _get_kernel(dev_id)
    
    # 线程配置：不用和 npts 同量级
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    sm_count = props['multiProcessorCount']  # SM数量
    logger.info(f"sm_count:{sm_count}")
    threads = 256  # 每个block的线程数
    blocks  = min(  # grid的大小
        sm_count * 32,  # 每个SM约32个block 的并发预算（经验值）
        65535  # x 维 block 上限（大多数设备）
    )
    
    ptsx = cp.ascontiguousarray(pts_xy[:, 0].astype(cp.float64, copy=False))
    ptsy = cp.ascontiguousarray(pts_xy[:, 1].astype(cp.float64, copy=False))
    
    pip_ray_kernel_stride(  # kernel(grid, block, args)
        (blocks,), (threads,),
        (  # 传递给CUDA内核函数的实参列表，.data.ptr拿到的是GPU数组的显存地址（指针），这样CUDA内核才能直接操作它
            ptsx.data.ptr, ptsy.data.ptr, npts,  # 注意不能直接传入非连续存储的切片数据，否则内核函数拿到的指针数据是不连续的，发生读取数据错位的情况
            vx.data.ptr, vy.data.ptr, m,
            cp.float64(xmin), cp.float64(xmax),  # 注意C内核函数中的形参float是32位的
            cp.float64(ymin), cp.float64(ymax),
            int(include_boundary),
            outmask.data.ptr,
            dbg.data.ptr
        )
    )
    logger.info(cp.unique(dbg, return_counts=True))
    return outmask.astype(cp.bool_)


_KERNEL_SRC = r'''

extern "C" __global__
void pip_ray_kernel_stride(
    const double* __restrict__ ptsx,
    const double* __restrict__ ptsy,
    const int npts,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const int m,
    const double xmin, const double xmax,
    const double ymin, const double ymax,
    const int include_boundary,
    unsigned char* __restrict__ mask,
    unsigned char* __restrict__ dbg)
{
    // bit0=bbox, bit1=boundary, bit2=condY, bit3=inside, bit4=NEAR_EDGE
    const int stride = gridDim.x * blockDim.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double dx_box = xmax - xmin;
    double dy_box = ymax - ymin;
    double scale = (dx_box > dy_box ? dx_box : dy_box);
    const double tol_base = (double)1e-12;
    const double tol = (scale > (double)1 ? tol_base * scale : tol_base);
    const double tol_cross = tol;
    const double tol_x = tol;
    const double tol_y = tol;

    for (; i < npts; i += stride){
        unsigned char flag = 0;
        double x = ptsx[i];
        double y = ptsy[i];

        if (x < xmin || x > xmax || y < ymin || y > ymax){
            mask[i] = 0;
            dbg[i] = flag;
            continue;
        }
        flag |= 1;

        if (include_boundary){
            for (int j = 0, k = m - 1; j < m; k = j++){
                double xj = vx[j], yj = vy[j];
                double xk = vx[k], yk = vy[k];
                double vxj = xk - xj, vyj = yk - yj;
                double wx  = x  - xj, wy  = y  - yj;

                double seglen2 = vxj*vxj + vyj*vyj;
                double seglen = sqrt(seglen2) + (double)1e-30;
                double cross = vxj * wy - vyj * wx;
                double cross_n = cross / seglen;

                if (fabs(cross_n) <= tol_cross){
                    double dot = vxj * wx + vyj * wy;
                    if (dot >= -tol_cross && dot <= seglen2 + tol_cross){
                        mask[i] = 1;
                        flag |= 2; // boundary
                        dbg[i] = flag;
                        goto NEXT_POINT;
                    }else{
                        flag |= 16;
                    }
                }
            }
        }

        {
            int c = 0;
            for (int j = 0, k = m - 1; j < m; k = j++){
                double xj = vx[j], yj = vy[j];
                double xk = vx[k], yk = vy[k];

                bool condY = ((yj <= y) && (y < yk)) || ((yk <= y) && (y < yj));
                if (condY){
                    flag |= 4;
                    double dy = yk - yj;

                    if (fabs(dy) < tol_y){
                        flag |= 16;
                        continue;
                    }
                    double xints = xj + ((y - yj) * (xk - xj)) / dy;
                    if (fabs(x - xints) <= tol_x) flag |= 16;

                    if (x < xints) c = !c;
                }
            }
            mask[i] = (unsigned char)c;
            if (c) flag |= 8; // inside
        }

        dbg[i] = flag;
        NEXT_POINT: ;
    }
}
'''

@lru_cache(maxsize=None)
def _get_kernel(device_id: int):
    with cp.cuda.Device(device_id):
        return cp.RawKernel(_KERNEL_SRC, 'pip_ray_kernel_stride', options=('-DUSE_DOUBLE=1',))


def _mask_by_kernel_ray(pts_xy, polygon_xy, use_local_coords, include_boundary, 
                        origin: Optional[str] = 'centroid', print_stats: bool = False):

    """
    Stage-1: GPU kernel in float64 (double).
    Stage-2: Use Shapely to refine ONLY near-edge points (dbg bit4 set).
    """
    pts_xy = cp.asarray(pts_xy)
    polygon_xy = cp.asarray(polygon_xy)

    # --- choose origin for local coords (shift only; no scaling)
    if use_local_coords:
        if origin == 'centroid':
            ox = float(cp.mean(polygon_xy[:,0]).item())
            oy = float(cp.mean(polygon_xy[:,1]).item())
        elif origin == 'min':
            ox = float(cp.min(polygon_xy[:,0]).item())
            oy = float(cp.min(polygon_xy[:,1]).item())
        else:
            logger.error("origin 只能是 'centroid' 或 'min'")
            raise ValueError("origin 只能是 'centroid' 或 'min'")

        pts_local = cp.empty_like(pts_xy, dtype=cp.float64)
        pts_local[:, 0] = (pts_xy[:, 0] - ox).astype(cp.float64, copy=False)
        pts_local[:, 1] = (pts_xy[:, 1] - oy).astype(cp.float64, copy=False)

        poly_local = cp.empty_like(polygon_xy, dtype=cp.float64)
        poly_local[:, 0] = (polygon_xy[:, 0] - ox).astype(cp.float64, copy=False)
        poly_local[:, 1] = (polygon_xy[:, 1] - oy).astype(cp.float64, copy=False)
    else:
        pts_local  = cp.asarray(pts_xy, dtype=cp.float64)
        poly_local = cp.asarray(polygon_xy, dtype=cp.float64)

    # --- bbox in local coords (float64 scalars)
    xmin = cp.float64(poly_local[:, 0].min().item())
    xmax = cp.float64(poly_local[:, 0].max().item())
    ymin = cp.float64(poly_local[:, 1].min().item())
    ymax = cp.float64(poly_local[:, 1].max().item())

    # --- device & launch config
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    sm_count = props['multiProcessorCount']
    threads = 256
    blocks = min(sm_count * 32, 65535)

    # --- Stage-1: GPU kernel
    dev_id = int(cp.cuda.Device().id)
    pip_ray_kernel_stride = _get_kernel(dev_id)

    npts = int(pts_local.shape[0])
    m = int(poly_local.shape[0])
    
    if m < 3:
        # degenerate polygon -> all False
        return cp.zeros(npts, dtype=cp.bool_)
    
    mask_u8 = cp.empty(npts, dtype=cp.uint8)
    dbg_u8 = cp.zeros(npts, dtype=cp.uint8)

    ptsx64 = cp.ascontiguousarray(pts_local[:,0])
    ptsy64 = cp.ascontiguousarray(pts_local[:,1])
    vx64 = cp.ascontiguousarray(poly_local[:,0])
    vy64 = cp.ascontiguousarray(poly_local[:,1])

    pip_ray_kernel_stride((blocks,), (threads,),
        (ptsx64.data.ptr, ptsy64.data.ptr, npts,
         vx64.data.ptr, vy64.data.ptr, m,
         xmin, xmax, ymin, ymax,
         int(include_boundary),
         mask_u8.data.ptr, dbg_u8.data.ptr))

    NEAR_EDGE = np.uint8(16)
    idx_uncertain = cp.where((dbg_u8 & NEAR_EDGE) != 0)[0]
    
    if print_stats:
        total = int(npts)
        n_unc = int(idx_uncertain.size)
        logger.info(f"[Stage1] total={total}, uncertain={n_unc} ({n_unc/total*100:.3f}%)")
        
    # 强制同步以便尽早暴露潜在的 kernel 错误
    cp.cuda.runtime.deviceSynchronize()
    
    # --- Stage-2: Shapely on near-edge only
    if idx_uncertain.size > 0:
        try:
            from shapely.geometry import Point, Polygon
            from shapely.prepared import prep

            # 用“原始坐标系”做精判，避免本地化容差不一致
            pts_np  = cp.asnumpy(pts_xy[idx_uncertain, :])
            poly_np = cp.asnumpy(polygon_xy)

            poly = Polygon(poly_np)
            P = prep(poly)

            if include_boundary:
                # covers: boundary included
                res = np.fromiter((P.covers(Point(x, y)) for x, y in pts_np),
                                  dtype=np.uint8, count=pts_np.shape[0])
            else:
                # contains: boundary excluded
                res = np.fromiter((P.contains(Point(x, y)) for x, y in pts_np),
                                  dtype=np.uint8, count=pts_np.shape[0])

            mask_u8[idx_uncertain] = cp.asarray(res, dtype=cp.uint8)
            # 可选：把 dbg 的 near-edge 位清除或保留为“曾回退”标记
            dbg_u8[idx_uncertain] = (dbg_u8[idx_uncertain] & (~NEAR_EDGE)) | np.uint8(0)

        except Exception as e:  # Shapely 不可用就保持 GPU 结果
            if print_stats:
                logger.info(f"[Shapely second-stage skipped] {e}")

    logger.info(cp.unique(dbg_u8, return_counts=True))

    return mask_u8.view(cp.bool_)


@deprecated(reason="请使用 _mask_by_kernel_ray 替代")
def _mask_by_cupy_ray(pts_xy: cp.ndarray, polygon_xy: cp.ndarray, include_boundary: bool = False) -> cp.ndarray:
    warnings.warn(
        "_mask_by_cupy_ray() 已弃用，请使用 _mask_by_kernel_ray() 替代",
        DeprecationWarning,
        stacklevel=2  # 指向用户调用处
    )
    # 外接矩形粗过滤，减轻计算量
    xmin, ymin = polygon_xy.min(axis=0)
    xmax, ymax = polygon_xy.max(axis=0)
    bbox_mask = (pts_xy[:, 0] >= xmin) & (pts_xy[:, 0] <= xmax) & \
                (pts_xy[:, 1] >= ymin) & (pts_xy[:, 1] <= ymax)
                        
    x, y = pts_xy[bbox_mask, 0], pts_xy[bbox_mask, 1]
    xj, yj = polygon_xy[:, 0], polygon_xy[:, 1]
    xk, yk = cp.roll(xj, -1), cp.roll(yj, -1)
    
    # 边界判定
    if include_boundary:
        X, Y = x[:, None], y[:, None]
        XJ, YJ = xj[None, :], yj[None, :]
        XK, YK = xk[None, :], yk[None, :]
        cross = (XK - XJ) * (Y - YJ) - (YK - YJ) * (X - XJ)
        eps = 1e-12
        on_line = cp.abs(cross) <= eps
        dot = (X - XJ) * (XK - XJ) + (Y - YJ) * (YK - YJ)
        seg_len2 = (XK - XJ) ** 2 + (YK - YJ) ** 2
        within = (dot >= -eps) & (dot <= seg_len2 + eps)
        on_edge = cp.any(on_line & within, axis=1)
    else:
        on_edge = cp.zeros(len(pts_xy), dtype=bool)

    # 射线法奇偶规则
    # (N, M):表示这N个点的水平[直线]是否穿过第M条边 (M个点构成M条边)
    cond_y = ((yj <= y[:, None]) & (yk > y[:, None])) | ((yk <= y[:, None]) & (yj > y[:, None]))  
    # with cp.errstate(divide='ignore', invalid='ignore'):
    """
    y[:, None] - yj[None, :]：每个点与多边形每个顶点的纵坐标差 [(N, 1) - (1, M)]
    xk[None, :] - xj[None, :]：多边形每条边的水平距离 [(1, M) - (1, M)]
    yk[None, :] - yj[None, :]：多边形每条边的垂直距离 [(1, M) - (1, M)]
    xints：(N, M) 是水平射线与多边形边的交点的x坐标（两点式，斜率相等）
    """
    xints = xj[None, :] + ( (y[:, None] - yj[None, :]) * (xk[None, :] - xj[None, :]) ) / (yk[None, :] - yj[None, :])
    xints = cp.where(cp.isnan(xints), cp.nan, xints)
    xints = cp.where(cp.isinf(xints), cp.inf, xints)
    # 判断点的水平射线是否与多边形的边发生了交点 (x[:, None] < xints)是只保留交点在点右侧的情况
    hits = cond_y & (x[:, None] < xints)
    # (N, ) 判断一个点是否在内部
    inside = (cp.count_nonzero(hits, axis=1) % 2 == 1)
    
    final_mask = bbox_mask.copy()
    final_mask[bbox_mask] = inside | on_edge
    
    return final_mask


def _mask_by_mpl(pts_xy: np.ndarray, polygon_xy: np.ndarray, include_boundary: bool) -> np.ndarray:
    """matplotlib.path.Path 实现"""
    if not _HAS_MPL:
        raise RuntimeError("需要 matplotlib，但未安装。")
    path = _MPLPath(polygon_xy, closed=True)
    if include_boundary:
        return path.contains_points(pts_xy, radius=1e-12)
    else:
        return path.contains_points(pts_xy)


def _mask_by_shapely(pts_xy: cp.ndarray, polygon_xy: cp.ndarray, include_boundary: bool) -> cp.ndarray:
    """shapely 实现，prepared geometry 加速，covers/contains 控制边界。"""
    if not _HAS_SHAPELY:
        raise RuntimeError("需要 shapely，但未安装。")
    polygon = _ShpPolygon(polygon_xy.get())
    P = _shp_prep(polygon)
    if include_boundary:
        it = (P.covers(_ShpPoint(x, y)) for x, y in pts_xy.get())
    else:
        it = (P.contains(_ShpPoint(x, y)) for x, y in pts_xy.get())
    return cp.fromiter(it, count=len(pts_xy), dtype=bool)


if __name__ == '__main__':
    las_path = '/home/CloudPointProcessing/惠清边坡测试/20250409163227_惠清打鼓岭AK1-175-450/raw/las/cloud_merged.las'
    json_file = '/home/point-cloud-3d/point-cloud-3d/output/惠清打鼓岭管辖区_投影坐标_decimal.json'

    pc = PC()
    pc.load_from_las(las_path)
    # pc.to_cpu()
    _get_kernel.cache_clear()

    slope_pc, mask = cut_point_cloud_by_polygon(pc, json_file, backend='cupy', include_boundary=True)

    from checker.check_point_cloud import check_point_cloud
    check_point_cloud(slope_pc)
    
    from processor.converter.point_cloud_rasterization import point_cloud_rasterization
    point_cloud_rasterization(
        slope_pc, 
        sampling_mode='max_z',
        sampling_rate=1000, 
        enable_fill=False, 
        enable_plt=False, 
        fill_algorithm='nearest', 
        save_path='./output/惠清管辖区_第一次巡飞_test.png'
    )
