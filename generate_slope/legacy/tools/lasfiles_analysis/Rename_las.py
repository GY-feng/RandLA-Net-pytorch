import os
import tkinter as tk
from tkinter import filedialog

# 隐藏tk的主窗口，只显示文件夹选择框
root = tk.Tk()
root.withdraw()

# 弹出文件夹选择对话框，手动选择要重命名的文件夹
folder_path = filedialog.askdirectory(title="请选择要批量重命名的文件夹")

# 判断是否选择了文件夹
if folder_path:
    # 获取文件夹内的所有文件(只取文件，不取子文件夹)，按默认顺序排序
    file_list = [f for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f))]
    
    # 开始批量重命名
    for index, file_name in enumerate(file_list, start=1):
        # 拼接原文件完整路径
        old_file = os.path.join(folder_path, file_name)
        # 拼接新文件完整路径 1.las、2.las...
        new_file = os.path.join(folder_path, f"{index}.las")
        # 执行重命名
        os.rename(old_file, new_file)
    
    print(f"✅ 重命名完成！共处理 {len(file_list)} 个文件，已全部改为 1.las, 2.las ...")
else:
    print("❌ 你取消了文件夹选择，程序退出")