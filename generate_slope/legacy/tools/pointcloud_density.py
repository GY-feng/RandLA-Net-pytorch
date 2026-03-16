import numpy as np
import open3d as o3d
from pathlib import Path
import laspy

# ================= 配置区 =================
INPUT_FILE = r"C:\Users\zjf\Desktop\tt\debug_check_las\4_hallway_6.las" # 点云文件路径
GRID_SIZE = 1.0                  # 网格大小 (米)
OUTPUT_FILE = r"C:\Users\zjf\Desktop\tt\density.txt"      # 输出 TXT 文件
# ========================================

def load_point_cloud(file_path):
    """
    支持多种格式加载点云，包括 .ply .pcd .txt .xyz .las
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext in [".ply", ".pcd"]:
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
    elif ext in [".txt", ".xyz", ".csv"]:
        points = np.loadtxt(file_path, delimiter=None)
        if points.shape[1] > 3:
            points = points[:, :3]
    elif ext == ".las":
        las = laspy.read(str(file_path))
        points = np.vstack((las.x, las.y, las.z)).T
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    return points

def compute_density(points, grid_size=1.0):
    """
    计算每个 grid_size x grid_size 网格的点数
    返回：
        density_dict {(ix, iy): 点数}
        min_coord, max_coord
    """
    min_x, min_y = points[:,0].min(), points[:,1].min()
    max_x, max_y = points[:,0].max(), points[:,1].max()

    ix = np.floor((points[:,0] - min_x) / grid_size).astype(int)
    iy = np.floor((points[:,1] - min_y) / grid_size).astype(int)

    density_dict = {}
    for x, y in zip(ix, iy):
        key = (x, y)
        density_dict[key] = density_dict.get(key, 0) + 1

    return density_dict, (min_x, min_y), (max_x, max_y)

def save_density_txt(density_dict, out_path):
    with open(out_path, "w") as f:
        for (x, y), count in density_dict.items():
            f.write(f"{x} {y} {count}\n")
    print(f"密度结果保存到 {out_path}")

def compute_global_density(points, min_coord, max_coord):
    """
    计算全局密度：总点数 / 总面积 (points/m^2)
    """
    total_points = points.shape[0]
    area = (max_coord[0] - min_coord[0]) * (max_coord[1] - min_coord[1])
    if area == 0:
        return 0
    return total_points / area

def main():
    points = load_point_cloud(INPUT_FILE)
    density_dict, min_coord, max_coord = compute_density(points, GRID_SIZE)
    
    # 输出基本信息
    print(f"点云大小: {points.shape[0]}")
    print(f"x范围: {min_coord[0]:.2f}-{max_coord[0]:.2f}, y范围: {min_coord[1]:.2f}-{max_coord[1]:.2f}")
    print(f"网格总数: {len(density_dict)}")
    
    # 保存每个网格的密度
    save_density_txt(density_dict, OUTPUT_FILE)
    
    # 计算全局密度
    global_density = compute_global_density(points, min_coord, max_coord)
    print(f"全局密度 (points/m²): {global_density:.2f}")

if __name__ == "__main__":
    main()