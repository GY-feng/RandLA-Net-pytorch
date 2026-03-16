import os
import laspy
import numpy as np
import tkinter as tk
from tkinter import filedialog
'''

打开资源管理器选定一个父文件夹，分别给出父文件夹里面所有的las文件的点个数，种类
'''
def check_batch():
    root = tk.Tk()
    root.withdraw()
    # 选定父文件夹
    parent_dir = filedialog.askdirectory(title="选择父文件夹")
    
    if not parent_dir:
        return

    # 遍历该文件夹下所有 las 文件
    for file in os.listdir(parent_dir):
        if file.lower().endswith(".las"):
            path = os.path.join(parent_dir, file)
            try:
                las = laspy.read(path)
                points_count = len(las.points)
                unique_classes = np.unique(las.classification).tolist()
                
                print(f"File: {file}")
                print(f"Points: {points_count}")
                print(f"Classes: {unique_classes}")
                print("-" * 20)
            except Exception as e:
                print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    check_batch()