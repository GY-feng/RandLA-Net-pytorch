import tkinter as tk
from tkinter import filedialog
import shutil
import os

def batch_copy_las():
    # 1. 初始化 tkinter 并隐藏主窗口
    root = tk.Tk()
    root.withdraw()

    # 2. 打开资源管理器选择 .las 文件
    file_path = filedialog.askopenfilename(
        title="请选择一个 .las 文件",
        filetypes=[("LAS files", "*.las"), ("All files", "*.*")]
    )

    if not file_path:
        print("未选择任何文件，程序退出。")
        return

    # 3. 获取文件所在的文件夹路径
    directory = os.path.dirname(file_path)
    
    print(f"开始复制文件到目录: {directory}")

    # 4. 循环复制 100 次
    for i in range(1, 101):
        # 构造新文件名：1.las, 2.las ... 100.las
        new_file_name = f"{i}.las"
        destination_path = os.path.join(directory, new_file_name)
        
        try:
            shutil.copy2(file_path, destination_path)
            print(f"已生成: {new_file_name}")
        except Exception as e:
            print(f"复制 {new_file_name} 时出错: {e}")

    print("\n任务完成！100个文件已全部生成。")

if __name__ == "__main__":
    batch_copy_las()