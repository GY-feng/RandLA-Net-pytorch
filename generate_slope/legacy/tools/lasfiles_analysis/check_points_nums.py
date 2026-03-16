import os
import sys
from tkinter import filedialog, Tk

# ---------------- 修正 sys.path ----------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # 回到 PointCloudProject
core_path = os.path.join(project_root, "core")
if core_path not in sys.path:
    sys.path.insert(0, core_path)


##　忽略这里的报错
from las_classification_stats import compute_classification_stats



# ---------------- UI ----------------
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="请选择 LAS / LAZ 文件",
    filetypes=[("LAS files", "*.las *.laz")]
)

if not file_path or not os.path.exists(file_path):
    print("❌ 未选择文件或文件不存在")
else:
    # ---------------- 核心统计 ----------------
    stats = compute_classification_stats(file_path)

    total_points = stats["total_points"]
    class_stats = stats["class_stats"]

    # ---------------- 打印结果 ----------------
    print(f"文件: {file_path}")
    print("-" * 55)
    print(f"{'分类编号':<10} | {'点数计数':<15} | {'占比 (%)':<10}")
    print("-" * 55)

    for cls, info in class_stats.items():
        percentage = info["ratio"] * 100
        print(f"{cls:<14} | {info['count']:<15} | {percentage:>8.2f}%")

    print("-" * 55)
    print(f"总计点数: {total_points}")

# # 1. 弹出文件选择框
# root = Tk()
# root.withdraw()
# file_path = filedialog.askopenfilename(filetypes=[("LAS files", "*.las *.laz")])

# if file_path:
#     # 2. 读取 LAS 文件
#     las = laspy.read(file_path)
    
#     # 获取总点数
#     total_points = len(las.points)
    
#     # 3. 统计分类
#     classes, counts = np.unique(las.classification, return_counts=True)
    
#     print(f"文件: {file_path}")
#     print("-" * 55)
#     # 修改了表头，增加了百分比列
#     print(f"{'分类编号':<10} | {'点数计数':<15} | {'占比 (%)':<10}")
#     print("-" * 55)
    
#     for cls, count in zip(classes, counts):
#         # 计算百分比：(当前分类计数 / 总计) * 100
#         percentage = (count / total_points) * 100
#         # :.2f 表示保留两位小数
#         print(f"{cls:<14} | {count:<15} | {percentage:>8.2f}%")
        
#     print("-" * 55)
#     print(f"总计点数: {total_points}")
# else:
#     print("未选择任何文件")
# import laspy
# import numpy as np
# from tkinter import filedialog, Tk

# # 1. 弹出文件选择框
# root = Tk()
# root.withdraw()
# file_path = filedialog.askopenfilename(filetypes=[("LAS files", "*.las *.laz")])

# if file_path:
#     # 2. 使用 laspy 2.0 推荐的读取方式
#     las = laspy.read(file_path)
    
#     # 3. 统计分类
#     # 在 2.0 中，分类数据直接通过 .classification 访问
#     classes, counts = np.unique(las.classification, return_counts=True)
    
#     print(f"文件: {file_path}")
#     print("-" * 40)
#     print(f"{'分类编号':<10} | {'点数计数':<15}")
#     print("-" * 40)
    
#     for cls, count in zip(classes, counts):
#         print(f"{cls:<14} | {count:<15}")
        
#     print("-" * 40)
#     print(f"总计点数: {len(las.points)}")
# else:
#     print("未选择任何文件")