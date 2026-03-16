import sys
import os
from pathlib import Path
from typing import Union, Tuple
import cupy as cp
from pyproj import CRS, Transformer
from tqdm import tqdm 

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC
from utils.logger import logger

class PointCloudGridCutter:
    def __init__(self, pc: PC):
        self.pc = pc  # 存储原始点云
        self.grid_indices: dict = {}  # 存储网格ID和点云索引
        self.cutted = False
        self._cache = {}
        self.density = None
        self.point_dist = None
        
    def __getitem__(self, grid_id: Tuple[int, int]) -> PC:
        """
        获取指定栅格ID的点云块(低速)
        参数:
            grid_id: 栅格ID，格式为 (x, y)
        返回:
            PointCloud对象，包含对应栅格ID的点云数据
        """
        if not self.cutted:
            raise ValueError("未执行切割操作")

        if grid_id not in self.grid_indices:
            raise KeyError(f"栅格ID {grid_id} 不存在")
        if grid_id in self._cache.keys():
            return self._cache[grid_id]
        indices = self.grid_indices[grid_id]
        pc = self.pc[indices]
        self._cache[grid_id] = pc
        return pc

    def cut(self, block_size: float = 10, target_coordinate: Union[str, CRS] = 'EPSG:4547', use_cache=True, cache_dir=None):
        """
        使用GPU加速的优化点云栅格切割函数
        
        参数:
            pc: PointCloud点云对象
            block_size: 栅格边长(米)
            target_coordinate: 目标投影坐标系
        返回:
            字典，key为栅格索引ID，value为对应的点云块
        """
        if self.cutted:
            raise ValueError("已执行切割操作")
        
        logger.info("开始执行点云栅格切割")

        if use_cache:
            import time
            import pickle
            import hashlib
            start_time = time.time()
            self.pc.to_cpu()
            serialize = pickle.dumps({'pc': self.pc, 'block_size': block_size, 'target_coordinate': target_coordinate}, protocol=5)
            self.pc.to_gpu()
            hash = hashlib.sha256(serialize).hexdigest()
            cache_path = os.path.join(cache_dir, hash + '.pkl')
            if os.path.exists(cache_path) and os.path.isfile(cache_path):
                logger.info(f"使用缓存 {hash + '.pkl'}")
                with open(cache_path, 'rb') as f:
                    cache = pickle.load(f)
                self.grid_indices = cache['grid_indices']
                # self.pc = cache['pc']
                self.cutted = cache['cutted']
                logger.info(f"缓存加载耗时: {time.time() - start_time} 秒")
                return
            else:
                logger.warning(f"在{os.path.abspath(cache_dir)}下没有找到缓存，将在本次计算后缓存")
        
        if self.pc.crs is None:
            raise ValueError("点云对象没有坐标系")
        
        if isinstance(target_coordinate, str):
            target_coordinate = CRS.from_string(target_coordinate)
        # 初始检查
        if not target_coordinate.is_projected:
            raise ValueError(f"目标坐标系 {target_coordinate} 不是投影坐标系")

        if self.pc.crs != target_coordinate:
            logger.warning(f"点云坐标系 ({self.pc.crs}) 与指定坐标系 ({target_coordinate}) 不匹配")
            logger.warning("尝试使用pyproj进行坐标系转换")
            transformer = Transformer.from_crs(self.pc.crs, target_coordinate, always_xy=True)
            x_proj, y_proj = transformer.transform(self.pc.x.get(), self.pc.y.get())
        else:
            x_proj, y_proj = self.pc.x, self.pc.y
        
        # GPU数据传输
        xy_gpu = cp.column_stack((cp.asarray(x_proj), cp.asarray(y_proj)))
        
        # 网格分组计算
        logger.info(f"开始网格分组（块尺寸 {block_size}米）")
        grid_indices = cp.floor(xy_gpu / block_size).astype(cp.int32)
        keys = cp.vstack((grid_indices[:, 1], grid_indices[:, 0]))
        sorted_order = cp.lexsort(keys)  # 返回的是索引向量（长度 N），表示“原数组按这个顺序就排好序了”
        sorted_grid_indices = grid_indices[sorted_order]  # 得到按 (gx, gy) 排好序的网格号数组（形状仍是 (N,2)）。之后相同网格会连续出现，便于切段
        
        # 计算分块索引
        diff = cp.any(sorted_grid_indices[1:] != sorted_grid_indices[:-1], axis=1)  # 比较相邻两行是否网格号发生变化,得到长度 N-1 的布尔向量，True 表示“从这里开始进入了新网格”
        split_indices = cp.concatenate([
            cp.array([0]), 
            cp.where(diff)[0] + 1, 
            cp.array([len(grid_indices)])
        ])
        
        # 构建结果字典
        logger.info(f"正在处理 {len(split_indices)-1} 个分块")
        for i in tqdm(range(len(split_indices)-1), desc="生成分块索引", unit='块'):
            start, end = split_indices[i], split_indices[i+1]
            grid_id = tuple(sorted_grid_indices[start].get().tolist())
            self.grid_indices[grid_id] = sorted_order[start:end]

        # logger.info(f"正在区块重分布")
        # sorted_grid_indices = {}
        # sorted_pc = PC()
        # offset = 0
        # for grid_id, indices in tqdm(self.grid_indices.items(), desc="重分布分块索引", unit='块'):
        #     sorted_pc += self.pc[indices]
        #     sorted_grid_indices[grid_id] = slice(offset, offset + len(indices) - 1)
        #     offset += len(indices)

        # self.pc = sorted_pc
        # self.grid_indices = sorted_grid_indices

        self.cutted = True
        self.density = len(self.pc) / len(self.grid_indices) / (block_size * block_size)  # 点云密度，默认单位是每平米
        self.point_dist = 1 / cp.sqrt(self.density)  # 根据点云密度估算点间平均距离，默认单位是每平米；也可以构建KD树统计点的最近邻距离，但计算量大，不在这里设置
            
        if use_cache:
            logger.info(f"正在缓存 {hash + '.pkl'}")
            start_time = time.time()
            self.pc.to_cpu()
            serialize = pickle.dumps({
                'grid_indices': self.grid_indices,
                # 'pc': self.pc,
                'cutted': self.cutted
                })
            self.pc.to_gpu()
            with open(cache_path, 'wb') as f:
                f.write(serialize)
            logger.info(f"缓存写入耗时: {time.time() - start_time} 秒")
        logger.info(f"切割完成，共生成 {len(self.grid_indices)} 个分块")

    def make_cache(self):
        """
        生成视图缓存
        """
        if not self.cutted:
            raise ValueError("未执行切割操作")

        for grid_id, indices in tqdm(self.grid_indices.items(), desc="生成点云内存视图缓存", unit='块'):
            # 提取点数据
            pcA = PC()
            for dim in self.pc.exist_dimensions:
                setattr(pcA, dim, getattr(self.pc, dim)[indices])
            pcA.crs = self.pc.crs
            pcA.offsets = self.pc.offsets
            pcA.scales = self.pc.scales
            
            self._cache[grid_id] = pcA



# import pickle
# import hashlib
# import time
# import os
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# from tqdm import tqdm  # 进度条（可选）
# from checker.check_cuda_status import check_cuda_status
# import open3d as o3d


# def save_chunk(chunk_data, chunk_id, output_dir):
#     """保存单个分块到文件（每个进程独立执行）"""
#     chunk_path = os.path.join(output_dir, f"chunk_{chunk_id}.pkl")
#     with open(chunk_path, "wb") as f:
#         pickle.dump(chunk_data, f, protocol=5)
#     return chunk_path

# def load_chunk(chunk_path):
#     """加载单个分块（每个进程独立执行）"""
#     import cupy as cp
#     from cupy.cuda import runtime
#     with open(chunk_path, "rb") as f:
#         return pickle.load(f)

# def split_dict_parallel(data, chunk_size=1000, output_dir="./chunks"):
#     """多进程分割字典并保存到文件"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     keys = list(data.keys())
#     total_chunks = (len(keys) + chunk_size - 1) // chunk_size
#     chunk_tasks = []
    
#     # 准备分块任务（主进程执行）
#     for i in range(total_chunks):
#         start = i * chunk_size
#         end = start + chunk_size
#         chunk_keys = keys[start:end]
#         chunk_data = {k: data[k] for k in chunk_keys}
#         chunk_tasks.append((chunk_data, i, output_dir))
    
#     # 多进程保存
#     with ProcessPoolExecutor() as executor:
#         chunk_paths = list(tqdm(
#             executor.map(save_chunk, *zip(*chunk_tasks)),
#             total=total_chunks,
#             desc="Saving chunks (MP)"
#         ))
#     return chunk_paths

# def merge_chunks_parallel(chunk_paths):
#     """多进程加载分块并合并字典"""
#     merged_data = {}
    
#     # 多进程加载
#     with ProcessPoolExecutor() as executor:
#         chunks = list(tqdm(
#             executor.map(load_chunk, chunk_paths),
#             total=len(chunk_paths),
#             desc="Loading chunks (MP)"
#         ))
    
#     # 主进程合并结果（避免多进程写冲突）
#     for chunk in chunks:
#         merged_data.update(chunk)
#     return merged_data

# if __name__ == '__main__':
#     logger.info("开始执行点云栅格切割测试")
    
#     # 检查CUDA状态
#     if not check_cuda_status():
#         exit(1)
    
#     # 加载LAS文件
#     las_path = "./data/AK0.las"
#     pc = PC()
#     pc.load_from_las(las_path)  # 假设有 load_from_las 方法
#     block_size = 1
    
#     logger.info(f"测试参数: LAS文件={las_path}, 块大小={block_size}米")
#     cutter = PointCloudGridCutter(pc)
#     blocks = cutter.cut(block_size, pc.crs)
    
#     # 多进程分割保存
#     logger.info("多进程分割字典并保存分块...")
#     logger.info("将数据转移至CPU")
#     start_save = time.time()
#     # with ThreadPoolExecutor(max_workers=32) as executor:
#     #     # 提交所有任务到线程池
#     #     futures = [executor.submit(lambda x: x.to_cpu(), v) for v in blocks.values()]
        
#     #     # 等待所有任务完成（可选，with语句会自动等待）
#     #     for future in futures:
#     #         future.result()
#     cutter.pc.to_cpu()
#     logger.info("转移完成")
#     logger.info(f"转移耗时: {time.time() - start_save:.2f}s")
    
#     start_save = time.time()
#     chunk_paths = split_dict_parallel(blocks, chunk_size=1000, output_dir="./data/chunks")
#     logger.info(f"多进程保存耗时: {time.time() - start_save:.2f}s")
    
#     # 计算完整数据的哈希（主进程执行）
#     start_save = time.time()
#     serialize_all = pickle.dumps(blocks, protocol=5)
#     hash_all = hashlib.sha256(serialize_all).hexdigest()
#     logger.info(f"完整数据哈希: {hash_all}")
#     logger.info(f"计算哈希耗时: {time.time() - start_save:.2f}s")
    
#     # 多进程加载合并
#     logger.info("多进程加载分块并合并...")
#     start_load = time.time()
#     loaded_blocks = merge_chunks_parallel(chunk_paths)
#     logger.info(f"多进程加载耗时: {time.time() - start_load:.2f}s")
    
    
#     # 验证数据一致性
#     # loaded_hash = hashlib.sha256(pickle.dumps(loaded_blocks, protocol=5)).hexdigest()
#     # assert hash_all == loaded_hash, "数据哈希校验失败！"
#     # logger.info("数据校验通过，加载成功！")

    
#     logger.info("将数据转移至GPU")
#     start_save = time.time()
#     with ThreadPoolExecutor(max_workers=32) as executor:
#         # 提交所有任务到线程池
#         futures = [executor.submit(lambda x: x.to_gpu(), v) for v in blocks.values()]
        
#         # 等待所有任务完成（可选，with语句会自动等待）
#         for future in futures:
#             future.result()
#     logger.info("转移完成")
#     logger.info(f"转移耗时: {time.time() - start_save:.2f}s")





if __name__ == '__main__':
    logger.info("开始执行点云栅格切割测试")
    import pickle
    import hashlib
    import time
    from checker.check_cuda_status import check_cuda_status
    import open3d as o3d

    
    
    # 检查CUDA状态
    if not check_cuda_status():
        exit(1)
    
    # 加载LAS文件
    las_path = "./data/AK0.las"
    # las_path = "/home/CloudPointProcessing/点云实验20250708/82cbec02-2b54-4049-b849-e4470cc38fe4/raw/las/2025-07-08-16-23-06.las"
    pc = PC()
    pc.load_from_las(las_path)
    block_size = 0.5
    
    logger.info(f"测试参数: LAS文件={las_path}, 块大小={block_size}米")
    cutter = PointCloudGridCutter(pc)
    cutter.cut(block_size, pc.crs)
    
    start = time.time()
    cutter.pc.to_cpu()
    logger.info(f"转移至CPU耗时: {time.time()-start:.2f}s")

    start = time.time()
    serialize = pickle.dumps(cutter, protocol=5)
    hash = hashlib.sha256(serialize).hexdigest()
    logger.info(f"计算SHA256耗时: {time.time()-start:.2f}s")

    start = time.time()
    with open('./data/AK0_grid.dump', 'wb') as f:
        f.write(serialize)
    logger.info(f"缓存到文件耗时: {time.time()-start:.2f}s")

    logger.info(f"序列化后哈希={hash}")

    start = time.time()
    cutter: PointCloudGridCutter = pickle.loads(serialize)
    logger.info(f"加载缓存耗时: {time.time()-start:.2f}s")
    
    start = time.time()
    cutter.pc.to_gpu()
    logger.info(f"转移至GPU耗时: {time.time()-start:.2f}s")
    cutter.make_cache()

    # 打印结果统计
    logger.info(f"共切割成 {len(cutter.grid_indices)} 个块")
    show_each = False

    # common_grid_avg_height = {key: cp.mean(cutter[key].z) for key in tqdm(cutter.grid_indices.keys())}
    # common_grid_var_height = {key: cp.var(cutter[key].z) for key in tqdm(common_keys)}
    # common_grid_avg_intensity = {key: cp.mean(cutter[key].intensity) for key in tqdm(common_keys)}
    # common_grid_var_intensity = {key: cp.var(cutter[key].intensity) for key in tqdm(common_keys)}

    # common_grid_avg_red = {key: cp.mean(cutter[key].red) for key in tqdm(common_keys)}
    # common_grid_avg_blue = {key: cp.mean(cutter[key].blue) for key in tqdm(common_keys)}
    # common_grid_avg_green = {key: cp.mean(cutter[key].green) for key in tqdm(common_keys)}


    # exit(0)
    logger.info("cutter.grid_indices.keys():", cutter.grid_indices.keys())
    if show_each:
        for grid_id in cutter.grid_indices.keys():
            pc = cutter[grid_id]
            print(pc.status)
            logger.info(f"\n正在显示块 {grid_id} (包含 {pc.point_nums} 个点)")
            logger.info("按Q键关闭当前窗口继续查看下一个分块...")
            
            # 创建新窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f'点云分块 {grid_id}', width=800, height=600)
            render_option = vis.get_render_option()
            render_option.point_size = 1.0
            render_option.background_color = [0, 0, 0]
            render_option.light_on = False
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cp.vstack((pc.x, pc.y, pc.z)).T.get())
            pcd.colors = o3d.utility.Vector3dVector((cp.vstack((pc.red, pc.green, pc.blue)).T / 65535.0).get())

            # 添加点云和包围盒
            vis.add_geometry(pcd)
            
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)  # 红色边框
            vis.add_geometry(bbox)
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
        
        logger.info("\n所有分块已显示完毕！")
    else:
        # 创建一个可视化窗口来显示所有分块
        logger.info("\n正在显示所有分块在同一窗口...")
        logger.info("按Q键关闭窗口...")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='点云分块合集', width=1200, height=900)
        render_option = vis.get_render_option()
        render_option.point_size = 1.0
        render_option.background_color = [0, 0, 0]
        render_option.light_on = False
        
        # 为每个分块随机分配颜色并添加到可视化窗口
        for grid_id in tqdm(cutter.grid_indices.keys(), unit='块', desc="添加分块"):
            pc = cutter[grid_id]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cp.vstack((pc.x, pc.y, pc.z)).T.get())

            pcd.paint_uniform_color(cp.random.rand(3).get())
            vis.add_geometry(pcd)

            # 添加每个分块的包围盒
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)  # 红色边框
            vis.add_geometry(bbox)
        


        vis.add_geometry(pcd)
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
        logger.info("\n所有分块已显示完毕！")
else:
    from checker.check_cuda_status import check_cuda_status
    if not check_cuda_status(log=False, check_cv2=False, check_o3d=False):
        raise ImportError(f"CUDA依赖缺失, 无法导入{__name__}")