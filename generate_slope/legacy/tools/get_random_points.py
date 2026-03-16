import laspy
import numpy as np
import random

# ========== 配置区 ==========
las_file_path = r"C:\Users\zjf\Desktop\check_DJI_202510211456_297_云梧广州方向-K98-050-490-点云-bare2_multi_block_2212.las"  # 改成你的 las 文件路径
N = 10                                  # 随机抽取点的数量
# ===========================

# 读取 las 文件
las = laspy.read(las_file_path)

# 获取点数
num_points = len(las.x)
print(f"点云总点数: {num_points}")

# 随机抽取 N 个索引
if N > num_points:
    raise ValueError("N 大于点云总点数！")

random_indices = random.sample(range(num_points), N)

print("\n===== 随机抽取的点信息 =====")
for i, idx in enumerate(random_indices):
    print(f"\n点 {i+1}:")
    print(f"X: {las.x[idx]}")
    print(f"Y: {las.y[idx]}")
    print(f"Z: {las.z[idx]}")
    print(f"Intensity: {las.intensity[idx]}")
    print(f"Return number: {las.return_number[idx]}")
    print(f"Number of returns: {las.number_of_returns[idx]}")
    print(f"Classification: {las.classification[idx]}")
    print(f"Scan angle rank: {las.scan_angle_rank[idx]}")
    print(f"User data: {las.user_data[idx]}")
    print(f"Point source id: {las.point_source_id[idx]}")

# ===== 计算整体平均值 =====
mean_x = np.mean(las.x)
mean_y = np.mean(las.y)
mean_z = np.mean(las.z)

print("\n===== 整体点云平均坐标 =====")
print(f"Mean X: {mean_x}")
print(f"Mean Y: {mean_y}")
print(f"Mean Z: {mean_z}")
