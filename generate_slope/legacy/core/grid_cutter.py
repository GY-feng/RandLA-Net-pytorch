import os
import numpy as np
import laspy
import json
from datetime import datetime

'''
代码直接从 self.las.points[mask] 提取原始点数据，避免了手动构建大矩阵导致的内存溢出风险
自动跳过空块：如果点云是不规则形状，切出来的格子里如果没有点，程序会自动跳过，不会生成无用的空文件。
'''
class LasGridCutter:
    def __init__(self, las_path: str, output_base: str, x_num: int, y_num: int):
        self.las_path = las_path
        self.x_num = x_num
        self.y_num = y_num
        
        # 创建专属输出文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_base, f"Grid_{x_num}x{y_num}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # 读取点云
        print(f"Reading LAS file: {las_path} ...")
        self.las = laspy.read(las_path)

    def cut(self):
        """执行切块操作"""
        # 获取边界
        x_min, x_max = self.las.header.min[0], self.las.header.max[0]
        y_min, y_max = self.las.header.min[1], self.las.header.max[1]

        x_step = (x_max - x_min) / self.x_num
        y_step = (y_max - y_min) / self.y_num

        # 获取坐标数组（内存优化写法）
        lx = np.array(self.las.x)
        ly = np.array(self.las.y)

        blocks_info = []

        for i in range(self.y_num):
            for j in range(self.x_num):
                # 定义当前块的范围
                x_s, x_e = x_min + j * x_step, x_min + (j + 1) * x_step
                y_s, y_e = y_min + i * y_step, y_min + (i + 1) * y_step

                # 筛选属于该块的点
                mask = (lx >= x_s) & (lx < x_e) & (ly >= y_s) & (ly < y_e)
                count = np.sum(mask)

                if count == 0:
                    continue

                # 写入子块文件
                file_name = f"block_{i}_{j}.las"
                out_path = os.path.join(self.output_dir, file_name)
                
                new_las = laspy.LasData(self.las.header.copy())
                new_las.points = self.las.points[mask]
                new_las.write(out_path)

                blocks_info.append({
                    "file": file_name,
                    "grid_index": [i, j],
                    "point_count": int(count),
                    "bounds": {"x": [float(x_s), float(x_e)], "y": [float(y_s), float(y_e)]}
                })
                print(f"Saved: {file_name} (Points: {count})")

        # 保存操作配置文件
        self._save_json(blocks_info)
        print(f"\nGrid cutting finished. Folder: {self.output_dir}")

    def _save_json(self, blocks_info):
        config_path = os.path.join(self.output_dir, "cut_config.json")
        log_data = {
            "source_file": self.las_path,
            "grid_m": self.y_num,
            "grid_n": self.x_num,
            "timestamp": datetime.now().isoformat(),
            "blocks": blocks_info
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4)