import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import cupy as cp
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(str(Path(__file__).parent.parent))

from pointcloud import PointCloud as PC
from utils.logger import logger

def compute_colormap(values: cp.ndarray, colors):
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
    values_gpu = cp.asarray(values, dtype=np.float32)
    color_table_gpu = cp.asarray(cmap(cp.linspace(0, 1, 256, dtype=np.float32).get())[:, :3])
    
    min_val = cp.min(values_gpu)
    max_val = cp.max(values_gpu)
    normalized = (values_gpu - min_val) / (max_val - min_val + 1e-10)
    indices = (normalized * 255).clip(0, 255).astype(cp.uint8)
    
    return cp.asnumpy(color_table_gpu[indices]).astype(np.float32)

def check_point_cloud(pc: PC):

    if len(pc) == 0:
        logger.error("错误: 输入点云为空")
        raise ValueError("输入点云为空")

    colors = {}
    key_mapping = {}

    if pc.red is not None and pc.green is not None and pc.blue is not None:
        rgb = cp.vstack((pc.red, pc.green, pc.blue)).T / 65535.0
        colors["RGB"] = rgb.get().astype(np.float32)
        key_mapping[ord("1")] = "RGB"

    if pc.intensity is not None:
        colors["Reflectance"] = compute_colormap(
            pc.intensity, 
            [(0,0,1), (0,1,1), (0,1,0), (1,1,0), (1,0,0)]
        )
        key_mapping[ord("2")] = "Reflectance"

    colors["Height"] = compute_colormap(pc.z, [(0,0,1), (1,0,0)])
    key_mapping[ord("3")] = "Height"

    if pc.return_number is not None:
        return_colors = cp.zeros((pc.point_nums, 3), dtype=np.float32)
        color_map = {
            1: [0,0,1], 2: [0,1,1], 3: [0,1,0], 
            4: [1,1,0], 5: [1,0.5,0], 6: [1,0,0]
        }
        for rn, color in color_map.items():
            mask = pc.return_number == rn
            return_colors[mask] = color
        colors["Return"] = return_colors.get()
        key_mapping[ord("4")] = "Return"

    if pc.classification is not None:
        class_color_map = {
            1: [0.5, 0.5, 0.5],  # 未分类 - 灰色
            2: [0.2, 0.8, 0.2],  # 地面 - 绿色
        }
        class_colors = cp.zeros((pc.point_nums, 3), dtype=np.float32)
        unique_classes = cp.unique(pc.classification)
        for cls in unique_classes:
            mask = pc.classification == cls
            color = class_color_map.get(int(cls), [1.0, 0.75, 0.8])  # 未定义类别显示粉色
            class_colors[mask] = color
        colors["Classification"] = class_colors.get()
        key_mapping[ord("5")] = "Classification"
    
    pcd = o3d.geometry.PointCloud()
    xyz = cp.vstack((pc.x, pc.y, pc.z)).T.astype(np.float64).get()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    try:
        pcd.colors = o3d.utility.Vector3dVector(colors["RGB"]) if "RGB" in colors else None
    except:
        pass

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1280, height=720)
    vis.add_geometry(pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = [0, 0, 0]
    render_option.light_on = False
    def make_switch_fn(view_name):
        def switch(vis):
            if view_name in colors:
                pcd.colors = o3d.utility.Vector3dVector(colors[view_name])
                vis.update_geometry(pcd)
                logger.info(f"切换到 {view_name} 视图")
        return switch
    
    for key, view in key_mapping.items():
        vis.register_key_callback(key, make_switch_fn(view))
    
    control_str = "按键控制: " + ",".join([f"{chr(key)}-{view}" for key, view in key_mapping.items()])
    logger.info(control_str)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    from checker.check_cuda_status import check_cuda_status
    from filter import filter_by_classification, filter_by_intensity, filter_by_return
    if not check_cuda_status():
        exit(1)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    las_file_path = "./data/AK0.las"
    pc = PC()
    pc.load_from_las(las_file_path)
    # 直接使用LAS文件路径作为输入
    # logger.info("\n示例1: 直接使用LAS文件路径")
    # check_point_cloud(pc)  # 直接传入文件路径
    
    # 分类过滤
    # class_code = input("输入要筛选的分类代码,以英文逗号隔开（默认2=地面点）: ") or '2'
    pcA = filter_by_classification(pc, '2')
    pcA = filter_by_return(pcA, '1')

    
    # # 反射率过滤
    # min_intensity = int(input("输入最小反射率(0-65535): ") or 0)
    # max_intensity = int(input("输入最大反射率(0-65535): ") or 65535)
    # pcA = filter_by_intensity(pcA, min_intensity, max_intensity)
    check_point_cloud(pcA)

else:
    from checker.check_cuda_status import check_cuda_status
    if not check_cuda_status(log=False):
        raise ImportError(f"CUDA依赖缺失, 无法导入{__name__}")