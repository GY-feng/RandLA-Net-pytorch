import numpy as np
import os
from collections import Counter
from tkinter import Tk, filedialog

def process_labels_individually():
    # 1. 初始化 Tkinter 并隐藏主窗口
    root = Tk()
    root.withdraw()

    # 2. 弹出文件夹选择对话框
    folder_path = filedialog.askdirectory(title="选择包含 .labels 文件的文件夹")

    if not folder_path:
        print("未选择文件夹，程序退出。")
        return

    # 3. 筛选文件夹中所有的 .labels 文件
    label_files = [f for f in os.listdir(folder_path) if f.endswith('.labels')]

    if not label_files:
        print(f"在路径 {folder_path} 下未找到任何 .labels 文件。")
        return

    print(f"开始处理，目标文件夹: {folder_path}")
    print("-" * 40)

    for file_name in label_files:
        file_path = os.path.join(folder_path, file_name)
        # 构造输出文件名：将 .labels 替换为 _stats.txt
        output_file_name = os.path.splitext(file_name)[0] + "_stats.txt"
        output_path = os.path.join(folder_path, output_file_name)

        try:
            # 加载标签数据
            labels = np.loadtxt(file_path, dtype=int)
            
            # 处理空文件情况
            if labels.size == 0:
                print(f"跳过空文件: {file_name}")
                continue

            # 统计当前文件
            counts = Counter(labels)
            total = len(labels)
            
            # 写入 TXT 文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"文件名称: {file_name}\n")
                f.write(f"总点数: {total}\n")
                f.write("-" * 30 + "\n")
                
                # 按类别编号升序排列
                for cls in sorted(counts.keys()):
                    num = counts[cls]
                    percentage = (num / total) * 100
                    f.write(f"Class {cls}: {num} points, {percentage:.2f}%\n")
            
            print(f"成功导出: {output_file_name}")

        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")

    print("-" * 40)
    print("所有文件处理完毕！")

if __name__ == "__main__":
    process_labels_individually()