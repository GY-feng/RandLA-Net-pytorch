#-*- encoding:utf-8 -*-
import cupy as cp
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from tqdm import tqdm
from collections import defaultdict
from .register_metric import register_metric, METRIC_REGISTRY
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger
from compare.griddiff.utils.numerical_processing import generate_grid_coordinates, get_neighborhood_points, fit_surface_with_spline


@dataclass 
class GridStatResult:
    '''
    stats的数据结构:
    {
        project_name: {
            data_type: {
                metric_name: cp.ndarray (K,)
            }
        }
    }
    diffs的数据结构:
    {
        data_type: {
            metric_name: cp.ndarray (K,)
        }
    }
    '''
    stats: Dict[str, Dict[str, Dict[str, Any]]] = None
    diffs: Dict[str, Dict[str, Any]] = None
    # fit_flags: cp.ndarray = None
    # fitted_surfaces: Dict[Tuple[int, int], Dict[str, Any]] = None

    def to_cpu(self):
        """
        将所有 cupy 数组转换为 numpy 数组，以便序列化或保存。
        保留数据层级结构不变。
        """
        def to_numpy(obj):
            """递归转换 cupy.ndarray → numpy.ndarray"""
            if isinstance(obj, cp.ndarray):
                return cp.asnumpy(obj)
            elif isinstance(obj, dict):
                return {k: to_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_numpy(v) for v in obj]
            else:
                return obj

        return GridStatResult(
            stats=to_numpy(self.stats),
            diffs=to_numpy(self.diffs),
            # fit_flags=to_numpy(self.fit_flags),
            # fitted_surfaces=to_numpy(self.fitted_surfaces)
        )

class GridStatComputer:
    def __init__(self, cfg, cloud_dict, common_keys):
        self.cfg = cfg
        self.cloud_dict = cloud_dict
        self.common_keys = common_keys
        self.key2gid = {k: i for i, k in enumerate(common_keys)}  # 注意key和gid的先后顺序一一对应
        self.K = len(common_keys)
        self.count_cache = None
        self.mean_z_cache = None
        self.fit_flags = None
        self.fitted_surfaces = defaultdict(dict)

        self.grid_stat_result = GridStatResult()

        # 在构造函数延迟注册各统计量计算函数，从而实现把函数与实例对象绑定
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_metric_name"):  # 只注册有标记的函数
                METRIC_REGISTRY[attr._metric_name] = attr

    def build_gid_and_coords(self, cutter):
        """
        该函数用于转换数据结构, 将 cutter 中按网格(grid)分组的点索引结构
        展平为可直接用于后续 GPU 矢量化分组归约的形式,
        即构建common_keys中点云的扁平化索引数组与对应的网格ID标签。

        参数
        ----------
        cutter : Cutter
            包含点云数据 (cutter.pc) 及网格划分结果 (cutter.grid_indices) 的对象,
            其中 cutter.grid_indices 是 {grid_key: point_index_list} 的映射。

        返回
        -------
        gid : cupy.ndarray of shape (N,)
            每个点对应的网格编号(整数标签),
            gid[i] 表示第 i 个点属于第 gid[i] 个网格。

        x, y, z : cupy.ndarray of shape (N,)
            对应 gid 的点的坐标值。三者与 gid 一一对齐,
            以便后续通过 cp.bincount() 等方法按网格聚合计算密度、高程、方差等统计量。
        
        说明
        -------
        本函数内部构造了两个关键的中间列表：

        - I_chunks: 
            存储每个网格中点的索引集合，
            例如: I_chunks = [[0,3,4], [10,11], [22,23,24]]
            在所有common_keys遍历完成后, 拼接为全局点索引数组。

        - G_chunks:
            存储与 I_chunks 对齐的网格ID标签, 每个 I_chunk 内的点索引都有相同的网格编号(从0开始),
            例如：G_chunks = [[0,0,0], [1,1], [2,2,2]],
            拼接后得到的 gid 数组与坐标向量 (x,y,z) 一一对应。

        这样，原本以字典分散存储的网格点信息被展平成统一的连续数组结构，
        提升后续利用 GPU 分组归约的执行效率与内存访问连续性。
        """
        I_chunks, G_chunks = [], []
        for k in tqdm(self.common_keys, desc="构建gid标签"):
            point_idxes = cutter.grid_indices.get(k, [])
            if len(point_idxes) == 0: 
                logger.error(f"网格 {k} 不存在点云，需检查代码")
                continue  # 理论上不会出现这个情况，因为通过PointCloudGridCutter得到的每个网格一定是有点的，没有点的区域不会被分配key
            I_chunks.append(cp.asarray(point_idxes, dtype=cp.int64))  # 这个修改点存疑：point_idxes如果是Python list，直接cp.concatenate在部分cupy版本/环境里可能隐式转换失败或产生非连续内存（影响 GPU 性能）。现在显式转为cupy数组，保证类型一致、内存连续、索引安全。
            G_chunks.append(cp.full((len(point_idxes),), self.key2gid[k], dtype=cp.int32))
        I = cp.concatenate(I_chunks)
        gid = cp.concatenate(G_chunks)
        return gid, cutter.pc.x[I], cutter.pc.y[I], cutter.pc.z[I]

    def _ensure_count_cache(self, gid):
        """私有 helper：确保 count_cache 存在（供所有统计量共享，避免重复计算）"""
        if not hasattr(self, 'count_cache') or self.count_cache is None:
            self.count_cache = cp.bincount(gid, minlength=self.K)
        return self.count_cache

    def _ensure_mean_cache(self, gid, z):
        """私有 helper：确保 mean_z_cache 存在（供高程平均值和高程标准差两个统计量共享，避免重复计算sum_z）"""
        if not hasattr(self, 'mean_z_cache') or self.mean_z_cache is None:
            count = self._ensure_count_cache(gid)
            sum_z = cp.bincount(gid, weights=z, minlength=self.K)
            denom = cp.maximum(count, 1)
            self.mean_z_cache = sum_z / denom
        return self.mean_z_cache

    @register_metric("密度")
    def grid_density(self, gid, x, y, z):
        """独立注册函数：只计算密度"""
        count = self._ensure_count_cache(gid)
        return {
            '密度': count.astype(cp.int32)
        }

    @register_metric("高程平均值")
    def grid_mean_z(self, gid, x, y, z):
        """独立注册函数：依赖count统计量，自动复用缓存"""
        mean_z = self._ensure_mean_cache(gid, z)
        return {
            '高程平均值': mean_z.astype(cp.float32)
        }

    @register_metric("高程标准差")
    def grid_std_z(self, gid, x, y, z):
        """独立注册函数：依赖count和mean，自动复用缓存，仅额外算一次sum_z2"""
        count = self._ensure_count_cache(gid)
        mean_z = self._ensure_mean_cache(gid, z)
        sum_z2 = cp.bincount(gid, weights=z * z, minlength=self.K)
        denom = cp.maximum(count, 1)
        var_z = cp.maximum(sum_z2 / denom - mean_z ** 2, 0.0)
        std_z = cp.sqrt(var_z)
        return {
            '高程标准差': std_z.astype(cp.float32)
        }

    @register_metric("高程分位数")
    def grid_quantiles(self, gid, x, y, z):
        """
        独立注册函数，暂时用不上
        按 gid 分组，计算每个网格的高程分位数。

        参数
        ----------
        gid : cupy.ndarray
            每个点所属网格的编号，形状为 (N,)。
        z : cupy.ndarray
            每个点的值，形状为 (N,)，是高度值或其他特征。
        count : cupy.ndarray
            每个网格包含的点数，形状为 (K,)。
        K : int
            网格总数。
        quantiles : list[float]
            要计算的分位数，范围通常是 [0, 1] 之间的浮点数，例如 [0.25, 0.5, 0.75]。

        返回
        -------
        cupy.ndarray
            形状为 (K, Q) 的数组，其中 K 是网格数量，K 是分位数的数量。
            每一行表示一个网格的各个分位数。
        """
        quantiles = self.cfg.grid_diff.quantiles  # 从配置中获取分位数
        Q = len(quantiles)
        if Q == 0:
            return {}  # 如果没有分位数要求，直接返回空字典
        
        # 如果没有缓存 count_cache，临时使用 gid 来计算每个网格的点数
        if not hasattr(self, 'count_cache') or self.count_cache is None:
            count = cp.bincount(gid, minlength=self.K)
        else:
            count = self.count_cache
 
        # 1) 将高程按 (gid, z) 多级排序(升序)，即先按照gid排序，gid相同的再按照z排
        order = cp.lexsort(cp.stack([z, gid], axis=0))
        z_sorted = z[order]
        
        # 2) 构造偏移表，记录每个网格内点的起止索引
        offsets = cp.empty(self.K + 1, dtype=cp.int64)
        offsets[0] = 0
        offsets[1:] = cp.cumsum(count, dtype=cp.int64)  # 计算累计和，得到每个网格的结束位置索引
        
        # 3) 计算组内 rank 索引
        qs = cp.asarray(quantiles)[None, :]    # 1×Q
        n  = count[:, None].astype(cp.int64)   # K×1
        rank_in_grid = cp.floor(qs * cp.maximum(n - 1, 0)).astype(cp.int64)  # K×Q 每个分位数在该网格内的排名
        idx_global = offsets[:-1, None] + rank_in_grid  # K×Q 每个分位数的全局索引，由每个网格的起始点索引加上本网格内的 rank 索引
        
        # 4) 取值（空网格填 NaN）
        valid = (n > 0)  # K
        out = cp.full((self.K, Q), cp.nan, dtype=z.dtype)
        mask = valid.repeat(Q, axis=1)  # K×Q
        out[mask] = z_sorted[idx_global[mask]] # 允许用K×Q维度的idx_global[mask]去索引N维的z_sorted
        
        # 5) 将计算结果转换为字典形式
        out_dict = {}
        for j, q in enumerate(quantiles):
            out_dict[f"{int(q * 100)}%分位数"] = out[:, j]

        return out_dict

    def group_reduction(self, gid, x, y, z):
        out = {}
        for name in self.cfg.grid_diff.metrics:
            func = METRIC_REGISTRY.get(name)
            if func is None:
                logger.warning(f'统计量函数 {name} 未在METRIC_REGISTRY中!')
                continue
            out.update(func(gid, x, y, z))  # 需保证不同指标使用不同名字，否则会覆盖之前的值
        return out

    # def surface_fit(self, diffs):

    #     # 写死了是地面点，只对地面点做拟合才有意义
    #     mean_z = diffs[self.cfg.wanted_ground]['高程平均值'].copy()
    #     std_z = diffs[self.cfg.wanted_ground]['高程标准差'].copy()

    #     ground_cutterA = self.cloud_dict[self.cfg.comp_pair[0]][f'{self.cfg.wanted_ground}_cutter']
    #     ground_cutterB = self.cloud_dict[self.cfg.comp_pair[1]][f'{self.cfg.wanted_ground}_cutter']

    #     self.fit_flags = cp.zeros(self.K, dtype=cp.bool_)
    #     self.fitted_surfaces = {}  # key: (i,j) -> dict(A=(x,y,z_fitted), B=...)

    #     try_fit = passed_fit = falied_fitA = falied_fitB = 0

    #     for i, key in tqdm(enumerate(self.common_keys), total=self.K, desc="拟合曲面", unit="网格"):

    #         if std_z[i] > self.cfg.grid_diff.fit_thres:  # 假设高程起伏差异过大的网格需要拟合曲面，也就是植被区域
    #             try_fit += 1
    #             point_idxesA = ground_cutterA.grid_indices.get(key)
    #             point_idxesB = ground_cutterB.grid_indices.get(key)
                
    #             # 点太少，无法拟合样条，回退到平均（样条至少需4点稳定）
    #             if min(len(point_idxesA), len(point_idxesB)) <= self.cfg.grid_diff.density_thres:
    #                 passed_fit += 1
    #                 continue
            
    #             gridA_x, gridA_y, gridA_z = get_neighborhood_points(key, ground_cutterA, radius=1)
    #             gridB_x, gridB_y, gridB_z = get_neighborhood_points(key, ground_cutterB, radius=1)
                
    #             sampled_x, sampled_y = generate_grid_coordinates(key, self.cfg.block_size)
    
    #             fitted_zA, success_A = fit_surface_with_spline(gridA_x, gridA_y, gridA_z, sampled_x, sampled_y)
    #             fitted_zB, success_B = fit_surface_with_spline(gridB_x, gridB_y, gridB_z, sampled_x, sampled_y)

    #             falied_fitA += not success_A
    #             falied_fitB += not success_B

    #             if success_A and success_B:
    #                 self.fit_flags[i] = True
    #                 self.fitted_surfaces[key] = {
    #                     'A': (sampled_x, sampled_y, fitted_zA),
    #                     'B': (sampled_x, sampled_y, fitted_zB)
    #                 }
    #                 transform = np.abs if self.cfg.grid_diff.abs_diff else (lambda v: v)
    #                 mean_z[i] = cp.float32(transform(np.mean(fitted_zA) - np.mean(fitted_zB)))

    #     diffs[self.cfg.wanted_ground]['高程平均值_部分拟合曲面'] =  mean_z

    #     logger.info(f"try_fit={try_fit}, passed_fit={passed_fit}, failed_fitA={falied_fitA}, failed_fitB={falied_fitB}")
    
    def compute(self):
        stats, diffs = defaultdict(dict), defaultdict(dict)
        dt = self.cfg.wanted_ground
        # 统计量函数 grid_density / grid_mean_z / grid_std_z 都依赖实例级缓存 self.count_cache 和 self.mean_z_cache
        # 如果多次调研实例函数（如group_reduction），那么后续计算点云B的统计量时由于gid不一致，会导致统计量计算异常
        # 所以每次调用group_reduction之前都要可能存在的清空缓存
        for project_name in self.cfg.comp_pair:
            cutter = self.cloud_dict[project_name][f'{dt}_cutter']
            # 清空缓存，确保每个 project 独立计算
            self.count_cache = None
            self.mean_z_cache = None

            gid, x, y, z = self.build_gid_and_coords(cutter)  # lasA和csf_ground B虽然common_keys都一样，但是每个网格里面的点数量不相等，所以gid和xyz等长度不一样
            stats[project_name][dt] = self.group_reduction(gid, x, y, z)

        transform = cp.abs if self.cfg.grid_diff.abs_diff else (lambda v: v)
        nameA, nameB = self.cfg.comp_pair[0], self.cfg.comp_pair[1]

        a, b = stats[nameA][dt], stats[nameB][dt]
        for metric in a.keys():
            diffs[dt][metric] = transform(b[metric] - a[metric])  # 最好用新训飞的点云减去旧点云

        # 根据需要拟合曲面
        # if self.cfg.grid_diff.spline_fit:
        #     self.surface_fit(diffs)

        self.grid_stat_result.stats = stats
        self.grid_stat_result.diffs = diffs
        self.grid_stat_result.fit_flags = self.fit_flags
        self.grid_stat_result.fitted_surfaces = self.fitted_surfaces

        return self.grid_stat_result.to_cpu()
