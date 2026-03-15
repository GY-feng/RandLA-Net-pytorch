import numpy as np
import laspy
import os
import sys
from pathlib import Path
from tqdm import tqdm
import math

'''
描述：

目标：
将无人机采集的高速公路 原始 LAS 点云数据 预处理成 适合深度学习（如 RandLA-Net / PointNet 系列）训练的 .npy 数据块。

具体完成的事情：
1，读取 .las 点云文件（坐标 + 强度 + 回波 + 颜色 + 语义标签）
2，对点云进行 滑动窗口切块（带重叠）

3，对每个块进行：
特征归一化，RGB，XYZ，反射值，回波数
局部坐标中心化
C++ 实现的体素网格下采样
4，按空间顺序划分 训练集 / 验证集
5，保存为深度学习可直接读取的 .npy 文件
'''
# --- 路径配置 ---
# 自动定位项目根目录 (假设当前脚本在 utils/ 下，如果不是请调整 .parent 数量)
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'utils'))

# 引入 C++ 下采样算子 (核心加速)  使用 C++ 实现的体素网格下采样
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

# --- 用户配置区域 ---
CFG = {
    'input_path': BASE_DIR / 'datasets/drone_highway/original_data', # 输入 .las 文件夹
    'output_path': BASE_DIR / 'datasets/drone_highway',              # 输出根目录
    'grid_size': 0.02,         # 2cm 体素下采样
    'block_size': 10,        # 空间切块大小 (米)
    'stride': 7,            # 步长 (米)，意味着有 10m 的重叠，（40m → 10m 重叠）
    'min_points': 10000,        # 若切块后点数少于此值，丢弃 (防止空块)
    'val_ratio': 5,            # 每 5 块取 1 块作为验证集 (空间离散划分)
    'features_dim': 9          # [x,y,z, r,g,b, i, ret_n, n_ret]
}

def normalize_features(las_data):
    """
    提取并归一化特征
    输出格式: [N, 6] -> [r, g, b, intensity, return_num, num_returns] (全为 0-1 float)
    """
    n_points = len(las_data)
    
    # 1. 颜色处理 (uint16 -> 0-1)
    # 并不是所有 las 都有颜色，增加健壮性检查
    try:
        r = np.asarray(las_data.red,   dtype=np.float32) / 255.0
        g = np.asarray(las_data.green, dtype=np.float32) / 255.0
        b = np.asarray(las_data.blue,  dtype=np.float32) / 255.0
    except AttributeError:
        print("警告: LAS文件没有颜色字段，填充为0")
        r = g = b = np.zeros(n_points, dtype=np.float32)

    # 2. 强度处理 (uint16 -> 0-1)
    # # 使用 99% 分位数截断，防止极高反光点导致整体偏暗
    # intensity = np.array(las_data.intensity, dtype=np.float32)
    # max_i = np.percentile(intensity, 99)
    # intensity = np.clip(intensity, 0, max_i) / (max_i + 1e-6)
    intensity=np.asarray(las_data.intensity, dtype=np.float32) / 65536.0

    # 3. 回波处理 (通常 return number 不会很大，直接归一化)
    # 假设最大回波次数不超过 7 (常规激光雷达)
    ret_n = np.array(las_data.return_number, dtype=np.float32) / 5.0
    n_ret = np.array(las_data.number_of_returns, dtype=np.float32) / 5.0

    # 堆叠特征
    features = np.vstack([r, g, b, intensity, ret_n, n_ret]).T
    return features.astype(np.float32)

def save_pointcloud_for_check(
    points,
    features,
    labels,
    save_path
):
    """
    保存为 PLY 文件，供 CloudCompare 可视化检查
    points:  [N, 3]
    features:[N, 6] -> r g b intensity ret_n n_ret (0–1)
    labels:  [N]
    """

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 颜色 (0–1)
    colors = features[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 将强度或标签存到 scalar field（CloudCompare 可读）
    intensity = features[:, 3]
    pcd.normals = o3d.utility.Vector3dVector(
        np.stack([intensity, intensity, intensity], axis=1)
    )

    o3d.io.write_point_cloud(str(save_path), pcd)

def process_and_save(points, features, labels, save_dir, file_stem, block_idx):
    """
    调用 C++ 算子进行下采样并保存
    """
    # 1. 局部中心化 (Local Centering)
    # 这对 PointNet 类网络至关重要，防止坐标数值过大
    xyz_min = np.amin(points, axis=0)
    points -= xyz_min

    # 2. C++ 网格下采样 (核心步骤)
    # 这一步会极快地把几万个点变成几千个代表性的点
    sub_points, sub_feats, sub_labels = cpp_subsampling.compute(
        points, 
        features=features, 
        classes=labels, 
        sampleDl=CFG['grid_size'], 
        verbose=0
    )

    # *************DEBUG 可视化*******************
    # save_pointcloud_for_check(
    #     sub_points,
    #     sub_feats,
    #     sub_labels,
    #     save_dir / f'debug_block_{block_idx}.ply'
    # )

    # 3. 保存
    # 拼接 [xyz, features, label] -> N * (3 + 6 + 1) = N * 10
    save_path = save_dir / f'{file_stem}_block_{block_idx:04d}.npy'
    
    # 注意：RandLA-Net data.py 期望的是所有数据拼在一起
    # sub_labels 需要 reshape 以便拼接
    output_data = np.hstack([sub_points, sub_feats, sub_labels.reshape(-1, 1)])
    np.save(save_path, output_data)


def main():
    # 准备目录
    for p in ['train', 'val', 'test']:
        (CFG['output_path'] / p).mkdir(parents=True, exist_ok=True)
    
    las_files = sorted(list(CFG['input_path'].glob('*.las')))
    print(f"找到 {len(las_files)} 个 LAS 文件。开始处理...")
    print(f"参数: Grid={CFG['grid_size']}m, Block={CFG['block_size']}m, Overlap={CFG['block_size']-CFG['stride']}m")

    global_block_counter = 0

    for las_file in tqdm(las_files):
        # --- 1. 读取 LAS (全量读取，因为你有 64G 内存) ---
        las = laspy.read(las_file)
        
        # 获取实际坐标 (Laspy 会自动应用 scale 和 offset)
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        
        # 获取特征
        feats = normalize_features(las)
        
        # 获取标签 (Classification)
        labels = np.array(las.classification, dtype=np.int32)
        
        # --- 2. 滑动窗口切块逻辑 ---
        # 计算边界
        x_min, y_min, _ = np.min(xyz, axis=0)
        x_max, y_max, _ = np.max(xyz, axis=0)
        
        # 生成网格锚点
        x_steps = int(np.ceil((x_max - x_min) / CFG['stride']))
        y_steps = int(np.ceil((y_max - y_min) / CFG['stride']))

        for i in range(x_steps):
            for j in range(y_steps):
                # 当前块的范围
                x_start = x_min + i * CFG['stride']
                x_end = x_start + CFG['block_size']
                y_start = y_min + j * CFG['stride']
                y_end = y_start + CFG['block_size']

                # 提取掩码 (Mask) - 这是一个简单的包围盒过滤
                mask = (xyz[:, 0] >= x_start) & (xyz[:, 0] < x_end) & \
                       (xyz[:, 1] >= y_start) & (xyz[:, 1] < y_end)
                
                block_points = xyz[mask]
                
                # --- 3. 过滤空块 ---
                if len(block_points) < CFG['min_points']:
                    continue
                
                block_feats = feats[mask]
                block_labels = labels[mask]
                
                # --- 4. 验证集分配策略 ---
                # 采用空间离散采样：每 5 个块中选 1 个做验证
                # 这样既保证了验证集覆盖了整个高速公路的不同路段，又不至于和训练集太像
                if global_block_counter % CFG['val_ratio'] == 0:
                    split_dir = CFG['output_path'] / 'val'
                else:
                    split_dir = CFG['output_path'] / 'train'
                
                # 处理并保存
                process_and_save(block_points, block_feats, block_labels, split_dir, las_file.stem, global_block_counter)
                
                global_block_counter += 1

    print("\n预处理完成！")
    print(f"数据已保存至: {CFG['output_path']}")
    print("下一步：请修改 tools.py 中的 Config，确保 num_points 和 d_in (输入维度) 匹配。")

if __name__ == '__main__':
    main()