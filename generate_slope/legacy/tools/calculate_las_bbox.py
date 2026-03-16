# 计算代码的包围盒

import laspy
import tkinter as tk
from tkinter import filedialog
import os

def get_las_bounding_box():
    # 1. 初始化Tkinter并隐藏主窗口
    root = tk.Tk()
    root.withdraw()

    # 2. 打开资源管理器选择文件
    file_path = filedialog.askopenfilename(
        title="请选择一个LAS文件",
        filetypes=[("LAS files", "*.las"), ("LAZ files", "*.laz")]
    )

    if not file_path:
        print("未选择任何文件。")
        return

    print(f"正在处理文件: {os.path.basename(file_path)}...")

    try:
        # 3. 读取LAS文件 (使用laspy)
        # laspy 自动从文件头中读取 min/max 统计信息，效率极高
        with laspy.open(file_path) as fh:
            header = fh.header
            
            # 获取最大/最小坐标
            min_x, min_y, min_z = header.mins
            max_x, max_y, max_z = header.maxs

            # 4. 计算长、宽、高
            length = max_x - min_x
            width = max_y - min_y
            height = max_z - min_z

            # 5. 输出结果
            print("-" * 30)
            print(f"【包围盒信息】")
            print(f"X范围 (长度): {length:.3f}")
            print(f"Y范围 (宽度): {width:.3f}")
            print(f"Z范围 (高度): {height:.3f}")
            print("-" * 30)
            print(f"最小坐标: ({min_x:.3f}, {min_y:.3f}, {min_z:.3f})")
            print(f"最大坐标: ({max_x:.3f}, {max_y:.3f}, {max_z:.3f})")
            
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    get_las_bounding_box()
    # 防止控制台直接关闭
    input("\n按下回车键退出...")