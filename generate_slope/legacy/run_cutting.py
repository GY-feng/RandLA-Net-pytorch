from core.grid_cutter import LasGridCutter

# ================= 配置区 =================
INPUT_LAS = r"E:\项目\点云无人机\数据模拟\惠清边坡点云数据_模拟实验\cloud_merged.las"
OUTPUT_DIR = r"E:\项目\点云无人机\数据模拟\模拟结果"
M = 4  # Y方向切几块
N = 3  # X方向切几块
# ==========================================

def start_process():
    try:
        cutter = LasGridCutter(
            las_path=INPUT_LAS,
            output_base=OUTPUT_DIR,
            x_num=N,
            y_num=M
        )
        cutter.cut()
    except Exception as e:
        print(f"Error during cutting: {e}")

if __name__ == "__main__":
    start_process()