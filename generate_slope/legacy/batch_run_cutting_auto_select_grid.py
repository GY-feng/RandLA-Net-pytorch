import os
import json
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from core.grid_cutter_auto import LasGridCutter

# ================= 配置区 =================
OUTPUT_ROOT = r"D:\Feng\cutting_PC"
M = 4  # Y轴方向块数
N = 3  # X轴方向块数
SUB_PATH = os.path.join("lidars", "terra_las", "cloud_merged.las")
# ==========================================

def batch_process():
    root = tk.Tk()
    root.withdraw()

    # 1. 手动选择 TXT 清单文件
    print(">>> 请选择包含路径清单的 TXT 文件...")
    txt_path = filedialog.askopenfilename(
        title="选择清单文件", 
        filetypes=[("Text files", "*.txt")]
    )
    
    if not txt_path:
        print("未选择清单，程序退出。")
        return

    # 2. 读取并清洗路径
    parent_folders = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip().replace('"', '')
            # 过滤：空行、注释行（#开头）、不存在的路径
            if not clean_line or clean_line.startswith("#"):
                continue
            if os.path.isdir(clean_line):
                parent_folders.append(clean_line)
            else:
                print(f"⚠️ 清单中的路径无效，已跳过: {clean_line}")

    if not parent_folders:
        print("清单中没有有效的文件夹路径，请检查 TXT。")
        return

    print(f"\n>>> 准备处理 {len(parent_folders)} 个任务...")
    
    success_count = 0
    failure_list = []

    # 3. 循环处理
    for folder in parent_folders:
        parent_name = os.path.basename(folder)
        las_input_path = os.path.join(folder, SUB_PATH)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_folder_name = f"{parent_name}_{N}x{M}_{timestamp}"
            target_dir = os.path.join(OUTPUT_ROOT, custom_folder_name)

            # 调用你的核心切割类(自动选择)
            cutter = LasGridCutter(las_input_path, OUTPUT_ROOT)
            cutter.output_dir = target_dir
            os.makedirs(target_dir, exist_ok=True)
            
            cutter.cut()
            success_count += 1
            print(f"✅ 完成: {parent_name}")
            
        except Exception as e:
            msg = f"{parent_name} 出错: {e}"
            print(f"❌ {msg}")
            failure_list.append(msg)

    # 4. 报告
    print("\n" + "="*50)
    print(f"处理完成！成功: {success_count} | 失败: {len(failure_list)}")
    if failure_list:
        print("\n失败清单:")
        for f in failure_list: print(f"  - {f}")
    print("="*50)

if __name__ == "__main__":
    batch_process()