import os
import random
import numpy as np
from datetime import datetime
from config.args import Args
from core.io_manager import IOManager
from core.editor import LASVisualEditor

# ==================== 配置区 ====================
TARGET_RATIO = 0.4        # 目标缺陷点占比 (0.4 代表 40%)
BOUNDARY_MARGIN = 0.02    # 边界留白比例
MAX_RETRY_LOCATIONS = 700   # 寻找不重叠位置的最大尝试次数
MAX_TOTAL_DEFECTS = 500    # 单个文件中允许生成的最大缺陷数量
INPUT_DIR = r"D:\Feng\裸露_原始data"

'''1-带草:r"D:\Feng\Train_PC_data\DJI_202510231007_350_肇阳广西方向-K47-320-490-点云\lidars\terra_las"
    2-带草：r"D:\Feng\Train_PC_data\DJI_202510231109_357_云罗罗定方向-K330-206-316-点云\lidars\terra_las"
    3-带草：r"D:\Feng\Train_PC_data\DJI_202510231407_366_云罗广西方向-K339-655-900-点云\lidars\terra_las"
'''
OUTPUT_DIR = r"D:\Feng\裸露_原始data_第一次模拟"
# ===============================================

def get_center_range(las, margin):
    """返回采样范围"""
    xs, ys = np.array(las.x), np.array(las.y)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_l = x_min + margin * (x_max - x_min)
    x_r = x_max - margin * (x_max - x_min)
    y_l = y_min + margin * (y_max - y_min)
    y_r = y_max - margin * (y_max - y_min)
    return (x_l, x_r), (y_l, y_r)

def is_overlapping(new_x, new_y, new_r, existing_defects):
    """检测重叠"""
    for (ex, ey, er) in existing_defects:
        distance = np.sqrt((new_x - ex)**2 + (new_y - ey)**2)
        if distance < (new_r + er): 
            return True
    return False

def get_current_defect_ratio(las):
    """计算非0标签点比例"""
    total_points = len(las.x)
    if total_points == 0: return 0
    # 注意：如果点云本来就有分类，这里建议只统计我们新打的标签（1,2,3,4）
    defect_points = np.count_nonzero(np.isin(las.classification, [1, 2, 3, 4]))
    return defect_points / total_points

def process_one_las_multi_defects(las_path, output_dir):
    file_name = os.path.splitext(os.path.basename(las_path))[0]
    print(f"\n▶ Processing: {file_name}")

    las_data = IOManager.load_las(las_path)
    # 重置分类标签，确保比例计算准确
    las_data.classification[:] = 0 
    
    (x_l, x_r), (y_l, y_r) = get_center_range(las_data, BOUNDARY_MARGIN)
    
    existing_defects_meta = []
    file_log_content = [f"File: {file_name}.las"]
    
    defect_count = 0
    current_direction = None 

    # 初始化编辑器并记录Args基础参数
    init_args = Args()
    file_log_content.append(f"  [Base Args] NoiseStd: {init_args.noise_std}, SlopeStatus: {init_args.slope_status}, InitSeed: {init_args.seed}")
    
    editor = LASVisualEditor(las_data, init_args)
    editor.apply_noise()

    while True:
        current_ratio = get_current_defect_ratio(las_data)
        if current_ratio >= TARGET_RATIO:
            break
        if defect_count >= MAX_TOTAL_DEFECTS:
            file_log_content.append(f"  [Limit] Reached MAX_TOTAL_DEFECTS ({MAX_TOTAL_DEFECTS})")
            break

        found_location = False
        temp_args = Args()
        
        for attempt in range(MAX_RETRY_LOCATIONS):
            test_x = random.uniform(x_l, x_r)
            test_y = random.uniform(y_l, y_r)
            test_r = random.uniform(2.0, 4.0)
            if not is_overlapping(test_x, test_y, test_r, existing_defects_meta):
                temp_args.region_x, temp_args.region_y, temp_args.radius = test_x, test_y, test_r
                found_location = True
                break
        
        if not found_location:
            file_log_content.append(f"  [Warning] No space found after {MAX_RETRY_LOCATIONS} attempts.")
            break

        temp_args.dz = random.uniform(1.0, 2.0)
        temp_args.smooth_type = random.choice(["linear", "quadratic", "gaussian"])
        
        if defect_count == 0:
            current_direction = temp_args.z_direction.lower()
        else:
            current_direction = "down" if current_direction == "up" else "up"
        temp_args.z_direction = current_direction

        editor.arg = temp_args
        editor.apply_z_offset()
        
        existing_defects_meta.append((temp_args.region_x, temp_args.region_y, temp_args.radius))
        defect_count += 1
        
        # 实时打印进度
        if defect_count % 10 == 0:
            print(f"  Progress: {defect_count} defects, Ratio: {get_current_defect_ratio(las_data):.2%}")

    final_ratio = get_current_defect_ratio(las_data)
    file_log_content.append(f"  [Summary] Total Defects: {defect_count}, Final Ratio: {final_ratio:.4%}")

    # 保存文件
    out_las_path = os.path.join(output_dir, f"{file_name}_multi.las")
    IOManager.save_las_direct(las_data, out_las_path)
    
    return "\n".join(file_log_content)

def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    las_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".las")]
    
    # 初始化全局日志文件
    summary_log_path = os.path.join(output_dir, "generation_summary.txt")
    
    with open(summary_log_path, 'w', encoding='utf-8') as log_file:
        # 写入全局配置信息
        log_file.write("================================================\n")
        log_file.write(f"BATCH PROCESSING LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("================================================\n")
        log_file.write(f"Global Config:\n")
        log_file.write(f" - Target Ratio: {TARGET_RATIO}\n")
        log_file.write(f" - Boundary Margin: {BOUNDARY_MARGIN}\n")
        log_file.write(f" - Max Defects Per File: {MAX_TOTAL_DEFECTS}\n")
        log_file.write(f" - Max Retry: {MAX_RETRY_LOCATIONS}\n")
        log_file.write("================================================\n\n")

        for las_path in las_files:
            try:
                # 处理文件并获取该文件的日志文本
                file_report = process_one_las_multi_defects(las_path, output_dir)
                log_file.write(file_report + "\n")
                log_file.write("-" * 50 + "\n")
                log_file.flush() # 实时刷新写入，防止程序中途崩溃丢日志
            except Exception as e:
                error_msg = f"File: {os.path.basename(las_path)} - FAILED with error: {str(e)}"
                print(f"❌ {error_msg}")
                log_file.write(error_msg + "\n" + "-" * 50 + "\n")

    print(f"\n✅ All finished. Summary log saved to: {summary_log_path}")

if __name__ == "__main__":
    batch_process(INPUT_DIR, OUTPUT_DIR)