import numpy as np
from algorithms.smoothing import get_smooth_weights

class LASVisualEditor:
    def __init__(self, las_data, args):
        self.las = las_data
        self.arg = args

    def apply_noise(self):
        """添加全局高斯噪声"""
        if self.arg.noise_std <= 0: return
        np.random.seed(self.arg.seed)
        n = len(self.las.x)
        noise = np.random.normal(0, self.arg.noise_std, (n, 3))
        # 批量写回提高效率
        self.las.x = np.array(self.las.x) + noise[:, 0]
        self.las.y = np.array(self.las.y) + noise[:, 1]
        self.las.z = np.array(self.las.z) + noise[:, 2]

    def apply_z_offset(self):
        """区域Z轴偏移并打上Classification标签"""

        # self.las.classification[:] = 0  # 将全场点云分类重置为 0，这一步不要了，在batch_run_cutting里面实现了
        
        x, y, z = np.array(self.las.x), np.array(self.las.y), np.array(self.las.z)
        dist = np.sqrt((x - self.arg.region_x)**2 + (y - self.arg.region_y)**2)
        mask = dist <= self.arg.radius
        
        if not np.any(mask):
            print("Warning: No points found in the specified region.")
            return

        # 1. 计算位移权重
        weights = get_smooth_weights(dist[mask], self.arg.radius, self.arg.smooth_type)
        
        # 2. 确定位移方向和数值
        is_up = self.arg.z_direction.lower() == "up"
        actual_dz = abs(self.arg.dz) if is_up else -abs(self.arg.dz)
        
        # 3. 应用位移
        z[mask] += actual_dz * weights
        self.las.z = z
        
        # 定义标签映射表
        # grass + up -> 1 | grass + down -> 2 | bare + up -> 3 | bare + down -> 4
        
        # 4. 根据条件确定标签值 (Classification)
        status = self.arg.slope_status.lower()
        label = None  # 初始化为空，避免误伤

        if status == "grass":
            label = 1 if is_up else 2
        elif status == "bare":
            label = 3 if is_up else 4
        else:
            # 记录未知状态但不对点云数据做任何修改
            print(f"Warning: Unknown slope_status '{status}'. Classification values remained unchanged.")

        # 5. 仅在 label 成功分配时更新数据
        if label is not None:
            # 直接在原始 classification 数组上修改受影响区域
            # 这会自动保留 mask 之外区域的原始标签（如 0, 1 等）
            self.las.classification[mask] = label
            print(f"Region updated: Classification = {label} (Status: {status}, Dir: {self.arg.z_direction})")