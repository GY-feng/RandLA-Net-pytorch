import numpy as np
import laspy
import os
import random
from pathlib import Path

# --- 配置 ---
N = 18 # 随机抽取 5 个文件
INPUT_DIR = Path("datasets/drone_highway/K30")
OUTPUT_DIR = Path("debug_check_las")
OUTPUT_DIR.mkdir(exist_ok=True)

def check_and_convert():
    npy_files = list(INPUT_DIR.glob("*.npy"))
    if not npy_files:
        print(f"错误: 在 {INPUT_DIR} 没找到 .npy 文件")
        return

    selected_files = random.sample(npy_files, min(N, len(npy_files)))
    print(f"正在随机转换 {len(selected_files)} 个文件进行检查...")

    for npy_path in selected_files:
        # 加载数据 [N, 10] -> x, y, z, r, g, b, intensity, ret_n, n_ret, label
        data = np.load(npy_path)
        
        # 创建新的 LAS
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)

        # 1. 还原坐标 (保持归一化后的样子，以便检查范围)
        las.x = data[:, 0]
        las.y = data[:, 1]
        las.z = data[:, 2]

        # 2. 还原颜色 (0-1 -> 0-65535, laspy 颜色通常是 uint16)
        # 乘以 255 再左移 8 位 或者直接乘以 65535
        las.red = (data[:, 3] * 65535).astype(np.uint16)
        las.green = (data[:, 4] * 65535).astype(np.uint16)
        las.blue = (data[:, 5] * 65535).astype(np.uint16)

        # 3. 还原强度 (0-1 -> 0-65535)
        las.intensity = (data[:, 6] * 65535).astype(np.uint16)

        # 4. 标签
        las.classification = data[:, 9].astype(np.uint8)

        output_name = OUTPUT_DIR / f"check_{npy_path.stem}.las"
        las.write(output_name)
        print(f"已保存: {output_name} | 点数: {len(data)} | XYZ范围: {np.ptp(data[:, :3], axis=0)}")

if __name__ == "__main__":
    check_and_convert()