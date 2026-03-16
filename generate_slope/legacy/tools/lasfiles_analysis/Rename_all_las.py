import os
import shutil
import tkinter as tk
from tkinter import filedialog

# 隐藏tk的默认窗口，只保留文件夹选择弹窗
root = tk.Tk()
root.withdraw()

# 弹窗选择目标根文件夹
root_folder = filedialog.askdirectory(title="请选择存放las文件的根文件夹（包含所有子文件夹）")

if root_folder:
    # 1. 定义新文件夹路径（在所选根文件夹下创建）
    new_save_folder = os.path.join(root_folder, "new_las_files")
    # 新建文件夹：如果已存在则不报错，不存在则创建
    os.makedirs(new_save_folder, exist_ok=True)
    
    # 2. 定义一个列表，存放遍历到的所有 .las 文件的完整路径
    all_las_files = []
    
    # 3. 递归遍历【根文件夹+所有子文件夹】，筛选所有.las文件
    print("正在遍历所有文件夹，查找las文件...")
    for root_dir, sub_dirs, files in os.walk(root_folder):
        for file_name in files:
            # 只筛选后缀为 .las 的文件（严格匹配，忽略大小写可选，当前是精准小写）
            if file_name.lower().endswith(".las"):
                file_full_path = os.path.join(root_dir, file_name)
                all_las_files.append(file_full_path)

    # 4. 批量重命名+复制到新文件夹
    if all_las_files:
        print(f"共找到 {len(all_las_files)} 个las文件，开始重命名并移动...")
        for file_num, old_file_path in enumerate(all_las_files, start=1):
            # 新文件名 1.las 2.las ...
            new_file_name = f"{file_num}_.las"
            new_file_path = os.path.join(new_save_folder, new_file_name)
            # 复制并重命名（原文件保留，不会删除）
            shutil.copy2(old_file_path, new_file_path)
        
        print(f"\n✅ 处理完成！所有文件已保存至：{new_save_folder}")
        print(f"✅ 重命名范围：1.las 至 {len(all_las_files)}.las")
    else:
        print("❌ 未在该文件夹及子文件夹中找到任何 .las 文件！")
else:
    print("❌ 取消了文件夹选择，程序退出")