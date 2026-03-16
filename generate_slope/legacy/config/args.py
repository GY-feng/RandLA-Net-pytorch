from dataclasses import dataclass, asdict

@dataclass
class Args:
    # 路径配置
    in_path: str = r"D:\Feng\cutting_PC\DJI_202411061039_202_K603-562-632_3x4_20260110_181447\block_2_0.las"
    out_dir_base: str = r"C:\Users\zjf\Desktop\模拟结果"
    
    # 参数：边坡状态 (有草覆盖 / 无草覆盖)# "grass": 有草覆盖, "bare": 无草覆盖 (裸土/岩石)
    slope_status: str = "grass"

    # 模拟参数
    noise_std: float = 0.02
    region_x: float = 609981.83
    region_y: float = 2499710.01
    radius: float = 3.0
    dz: float = 2.0
    seed: int = 2025
    smooth_type: str = "gaussian"  # linear, quadratic, gaussian
    z_direction: str = "up"      # up, down

    def to_dict(self):
        return asdict(self)