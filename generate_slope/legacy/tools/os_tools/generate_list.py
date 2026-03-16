import os
import tkinter as tk
from tkinter import filedialog

# ================= 配置区 =================
TARGET_FILE_NAME = "folder_list.txt"
# 预期的子路径结构
SUB_PATH = os.path.join("lidars", "terra_las", "cloud_merged.las")
# ==========================================

def generate():
    root = tk.Tk()
    root.withdraw()

    # 1. 选择总文件夹
    print(">>> 请选择包含多个点云父文件夹的总目录...")
    base_dir = filedialog.askdirectory(title="选择总目录")
    
    if not base_dir:
        print("未选择目录，程序退出。")
        return

    # 2. 遍历直接下一级子文件夹
    print(f"\n>>> 正在扫描: {base_dir}")
    all_sub_folders = [os.path.join(base_dir, f) for f in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, f))]
    
    valid_paths = []
    
    for folder in all_sub_folders:
        las_path = os.path.join(folder, SUB_PATH)
        if os.path.exists(las_path):
            valid_paths.append(folder)
        else:
            # 预判不存在则 print 出来
            print(f"[-] 缺失文件，已跳过: {folder}")

    # 3. 写入 TXT 文件 (保存在脚本同级目录)
    output_txt = os.path.join(os.path.dirname(__file__), TARGET_FILE_NAME)
    with open(output_txt, "w", encoding="utf-8") as f:
        for path in valid_paths:
            f.write(path + "\n")
    
    print(f"\n成功! 共找到 {len(valid_paths)} 个合格文件夹。")
    print(f"清单已保存至: {output_txt}")
    print("你可以手动打开该文件进行删改，完成后运行 batch_run_cutting.py")

if __name__ == "__main__":
    generate()