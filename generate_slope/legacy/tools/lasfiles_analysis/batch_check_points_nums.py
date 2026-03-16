import laspy
import numpy as np
import os
from tkinter import filedialog, Tk

# 1. 弹出【文件夹】选择框（核心修改点：选文件夹 而非 单个文件）
root = Tk()
root.withdraw()
# 选择目标文件夹，遍历该文件夹下所有las/laz（含所有子文件夹）
folder_path = filedialog.askdirectory(title="请选择存放LAS/LAZ文件的文件夹（会遍历所有子文件夹）")

# 2. 遍历逻辑 + 你的统计逻辑整合
if folder_path:
    # 定义要筛选的文件后缀
    target_suffix = [".las", ".laz"]
    # 统计找到的文件总数
    las_file_count = 0
    
    print(f"✅ 已选择文件夹：{folder_path}")
    print(f"🔍 正在遍历所有文件（含子文件夹），筛选 LAS/LAZ 文件...\n")
    
    # 递归遍历文件夹下所有文件（包括子文件夹）
    for root_dir, sub_dirs, files in os.walk(folder_path):
        for file_name in files:
            # 筛选后缀为 las/laz 的文件（不区分大小写）
            if any(file_name.lower().endswith(suffix) for suffix in target_suffix):
                las_file_count += 1
                file_full_path = os.path.join(root_dir, file_name)
                
                # ========== 以下是你原封不动的 LAS文件读取+统计逻辑 ==========
                las = laspy.read(file_full_path)
                # 获取总点数
                total_points = len(las.points)
                # 统计分类
                classes, counts = np.unique(las.classification, return_counts=True)

                print(f"📄 文件: {file_full_path}")
                print("-" * 55)
                print(f"{'分类编号':<10} | {'点数计数':<15} | {'占比 (%)':<10}")
                print("-" * 55)

                for cls, count in zip(classes, counts):
                    # 计算百分比：(当前分类计数 / 总计) * 100
                    percentage = (count / total_points) * 100
                    print(f"{cls:<14} | {count:<15} | {percentage:>8.2f}%")

                print("-" * 55)
                print(f"总计点数: {total_points}")
                print("="*80 + "\n") # 文件分隔线，区分不同文件的统计结果

    # 遍历完成后的汇总提示
    if las_file_count == 0:
        print("❌ 未在该文件夹及子文件夹中找到任何 .las / .laz 文件！")
    else:
        print(f"✅ 遍历完成！共处理 {las_file_count} 个 LAS/LAZ 文件")
else:
    print("❌ 未选择任何文件夹，程序退出")