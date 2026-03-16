import os
import sys
import time
import random
import numpy as np
from datetime import datetime
from config.args import Args
from core.io_manager import IOManager
from core.editor import LASVisualEditor

# ==================== 配置区 ====================
TARGET_RATIO = 0.5        # 目标缺陷点占比 (0.4 代表 40%)
BOUNDARY_MARGIN = 0.03    # 边界留白比例
MAX_RETRY_LOCATIONS = 600   # 寻找不重叠位置的最大尝试次数
MAX_TOTAL_DEFECTS = 500    # 单个文件中允许生成的最大缺陷数量

# 新增：从此 txt 文件读取父文件夹路径列表（每行一个父文件夹地址）
TXT_LIST_PATH = r"C:\Users\zjf\Desktop\PointCloud_github\2026\PointCloudProject\tools\folder_list.txt"

OUTPUT_DIR = r"D:\Feng\裸露(类3，4)_up_down"
# ===============================================

def format_time(seconds):
    """格式化秒为 H:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def print_progress_bar(file_idx, total_files, file_name, current_ratio, target_ratio, file_start_time, global_start_time, bar_len=30):
    """在一行中打印/刷新一个进度条，表现为 PyTorch 风格"""
    now = time.time()
    progress = min(current_ratio / target_ratio, 1.0) if target_ratio > 0 else 1.0
    percent_of_target = min(100.0 * current_ratio / target_ratio, 100.0) if target_ratio > 0 else 100.0
    percent_actual = 100.0 * current_ratio
    filled = int(progress * bar_len)
    if filled >= bar_len:
        bar = "[" + "=" * bar_len + "]"
    else:
        bar = "[" + "=" * filled + ">" + "." * (bar_len - filled - 1) + "]"
    file_elapsed = now - file_start_time
    total_elapsed = now - global_start_time
    # 限制文件名长度避免行过长
    short_name = file_name if len(file_name) <= 30 else ("..." + file_name[-27:])
    sys.stdout.write(
        f"\rFile {file_idx}/{total_files} {short_name} {bar} {percent_actual:6.2f}% (of target {percent_of_target:6.2f}%) | "
        f"File time: {format_time(file_elapsed)} | Total time: {format_time(total_elapsed)}"
    )
    sys.stdout.flush()

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

def process_one_las_multi_defects(las_path, folder_name, output_dir, file_idx, total_files, global_start_time):

    """
    与原 process_one_las_multi_defects 相同逻辑，但用单行进度条显示进度。
    返回 (file_log_text, final_ratio, file_elapsed_seconds, file_basename)
    """
    file_name = folder_name

    # 记录文件处理起始时间（用于进度显示）
    file_start_time = time.time()

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

    # 在进入循环前显示一次初始进度（0%）
    current_ratio = get_current_defect_ratio(las_data)
    print_progress_bar(file_idx, total_files, file_name, current_ratio, TARGET_RATIO, file_start_time, global_start_time)

    while True:
        current_ratio = get_current_defect_ratio(las_data)
        if current_ratio >= TARGET_RATIO:
            # 刷新显示为完成（100%）
            print_progress_bar(file_idx, total_files, file_name, current_ratio, TARGET_RATIO, file_start_time, global_start_time)
            break
        if defect_count >= MAX_TOTAL_DEFECTS:
            file_log_content.append(f"  [Limit] Reached MAX_TOTAL_DEFECTS ({MAX_TOTAL_DEFECTS})")
            # 刷新一次并退出
            print_progress_bar(file_idx, total_files, file_name, current_ratio, TARGET_RATIO, file_start_time, global_start_time)
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
            # 刷新一次并退出
            print_progress_bar(file_idx, total_files, file_name, current_ratio, TARGET_RATIO, file_start_time, global_start_time)
            break

        temp_args.dz = random.uniform(2.0, 4.0)
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

        # 刷新单行进度显示（不会换行）
        current_ratio = get_current_defect_ratio(las_data)
        print_progress_bar(file_idx, total_files, file_name, current_ratio, TARGET_RATIO, file_start_time, global_start_time)

    final_ratio = get_current_defect_ratio(las_data)
    file_log_content.append(f"  [Summary] Total Defects: {defect_count}, Final Ratio: {final_ratio:.4%}")

    # 保存文件（和原逻辑完全一样）
    out_las_path = os.path.join(output_dir, f"{file_name}.las")
    IOManager.save_las_direct(las_data, out_las_path)

    # 在文件处理完成后换行以便下一次输出不覆盖
    sys.stdout.write("\n")
    sys.stdout.flush()

    file_elapsed = time.time() - file_start_time
    return ("\n".join(file_log_content), final_ratio, file_elapsed, file_name)

def batch_process_from_txt(txt_list_path, output_dir):
    """
    从 txt 中读取父文件夹列表（每行一个），对每个父文件夹构造:
      <parent_folder>/lidars/terra_las/cloud_merged.las
    并对存在的 las 文件执行原先的处理流程。
    额外生成一个 final_ratios.txt，放在 output_dir，只包含：全局参数（一次）、文件名、最后实际占比。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读 txt 列表
    las_files = []
    if not os.path.isfile(txt_list_path):
        raise FileNotFoundError(f"TXT list file not found: {txt_list_path}")

    with open(txt_list_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            # 跳过注释行（如果有）
            if line.startswith("#"):
                continue
            # 规范化路径（处理混合斜杠）
            parent_folder = os.path.normpath(line)
            # 构造目标 las 路径
            las_path = os.path.join(parent_folder, "lidars", "terra_las", "cloud_merged.las")
            if os.path.isfile(las_path):
                folder_name = os.path.basename(parent_folder)
                las_files.append((las_path, folder_name))

            else:
                print(f"⚠️ Warning: Expected file not found, skipping: {las_path}")

    total_files = len(las_files)

    # 初始化全局日志文件（保留原有详细日志）
    summary_log_path = os.path.join(output_dir, "generation_summary.txt")

    # 新：创建仅包含 final ratios 的单一 txt（只写入指定信息）
    final_summary_path = os.path.join(output_dir, "final_ratios.txt")
    # 写头（覆盖同名旧文件）——只写一次全局参数
    with open(final_summary_path, 'w', encoding='utf-8') as final_f:
        final_f.write("Final Ratios Summary\n")
        final_f.write("=====================\n")
        final_f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        final_f.write("Global Config:\n")
        final_f.write(f" - Target Ratio: {TARGET_RATIO}\n")
        final_f.write(f" - Boundary Margin: {BOUNDARY_MARGIN}\n")
        final_f.write(f" - Max Defects Per File: {MAX_TOTAL_DEFECTS}\n")
        final_f.write(f" - Max Retry: {MAX_RETRY_LOCATIONS}\n")
        final_f.write("\n")
        final_f.write("Filename,Final_Defect_Ratio\n")  # CSV-like lines follow

    global_start_time = time.time()

    with open(summary_log_path, 'w', encoding='utf-8') as log_file, open(final_summary_path, 'a', encoding='utf-8') as final_f:
        # 写入全局配置信息（详细日志）
        log_file.write("================================================\n")
        log_file.write(f"BATCH PROCESSING LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("================================================\n")
        log_file.write(f"Global Config:\n")
        log_file.write(f" - Target Ratio: {TARGET_RATIO}\n")
        log_file.write(f" - Boundary Margin: {BOUNDARY_MARGIN}\n")
        log_file.write(f" - Max Defects Per File: {MAX_TOTAL_DEFECTS}\n")
        log_file.write(f" - Max Retry: {MAX_RETRY_LOCATIONS}\n")
        log_file.write("================================================\n\n")

        for idx, (las_path, folder_name) in enumerate(las_files, start=1):
            try:
                # 处理文件并获取该文件的日志文本、最终 ratio 和耗时
                file_report, final_ratio, file_elapsed, file_basename = process_one_las_multi_defects(las_path, folder_name, output_dir, idx, total_files, global_start_time)

                # 写入详细日志（原有行为）
                log_file.write(file_report + "\n")
                log_file.write("-" * 50 + "\n")
                log_file.flush()  # 实时刷新写入，防止程序中途崩溃丢日志

                # 向 final_ratios.txt 添加一行：文件名, final_ratio
                final_f.write(f"{file_basename},{final_ratio:.6f}\n")
                final_f.flush()
            except Exception as e:
                error_msg = f"File: {os.path.basename(las_path)} - FAILED with error: {str(e)}"
                # 保留原有错误打印/日志
                print(f"❌ {error_msg}")
                log_file.write(error_msg + "\n" + "-" * 50 + "\n")

    total_elapsed = time.time() - global_start_time
    print(f"\n✅ All finished. Summary log saved to: {summary_log_path}")
    print(f"✅ Final ratios saved to: {final_summary_path}")
    print(f"Total run time: {format_time(total_elapsed)}")

if __name__ == "__main__":
    # 主入口：从 TXT 读取父文件夹列表并处理
    batch_process_from_txt(TXT_LIST_PATH, OUTPUT_DIR)

'''
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202507101026_307_云梧-K340-955-995-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202508221426_217_英怀阳江方向-K86-850-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202508231013_222_英怀怀集方向-K26-000-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510101155_155_清云湛江方向-K516-550-650-重复
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510101620_164_清云湛江方向-K603-562-632-点云-原始-重复
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510140938_173_清云湛江方向-K491-840-980-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510141004_176_清云湛江方向-K492-818-K493-015-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510141004_179_清云湛江方向-K493-160-275-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510141129_185_清云湛江方向-K514-700-860-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510141554_186_清云汕头方向-K520-200-300-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510141634_191_清云湛江方向-K516-540-740-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510141634_194_清云湛江方向-K516-260-440-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510141717_197_清云湛江方向-K516-050-155-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510141754_200_清云湛江方向-K493-570-875-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510150923_203_清云湛江方向-K603-132-272-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510150923_205_清云湛江方向-K603-312-392-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510150948_207_清云湛江方向-K603-562-632-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510150948_209_清云湛江方向-K603-772-862-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510151015_211_清云湛江方向-K603-982-K604-038-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510151025_213_清云湛江方向-K604-452-542-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510151039_215_清云湛江方向-K604-817-882-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510151052_217_清云湛江方向-K605-852-992-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510151332_219_清云汕头方向-K531-880-980-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510151441_223_清云汕头方向-K493-570-875-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510160911_227_英怀阳江方向-K86-850-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510161051_229_英怀怀集方向-K25-730-K26-100-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510161541_239_241_清云湛江方向-K603-562-632-点云01
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510161713_240_禾联岗互通-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510200903_243_云梧广州方向-k130-450-505-路肩点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510200943_248_云梧广州方向-K136-274-404-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201011_250_云梧梧州方向-K150-478-654-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201041_252_云梧广州方向-K163-270-610-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201041_254_云梧广州方向-K163-750-860-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201041_256_云梧梧州方向-K163-745-K164-0045-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201453_260_云梧梧州方向-K187-960-K188-180-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201521_263_云梧广州方向-K181-995-K182-125-路肩墙点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201537_265_云梧梧州方向-K180-217-357-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201553_267_云梧梧州方向-178-027-122-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201612_270_云梧梧州方向-K169-545-K170-645-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201654_272_云梧梧州方向-K164-527-612-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201654_274_云梧梧州方向-K164-090-237点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510201747_276_云梧广州方向-K136-510-550-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510210931_281_双凤枢纽立交广州方向-K122-491-747-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510211112_288_云梧广西方向-K98-892-K99-137-路堤墙点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510211138_290_云梧广州方向-K98-100-255-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510211429_293_云梧梧州方向-K96-430-730-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510211456_297_云梧广州方向-K98-050-490-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510211548_300_云梧梧州方向-K113-510-828-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510211625_303_肇阳广西方向-K31-790-940-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510211647_307_肇阳广州方向-K34-280-340-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510211701_309_肇阳广西方向-K37-558-K38-000-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510211733_311_肇阳广西方向-K39-610-680-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510220905_314_肇阳广州方向-K44-930-980-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510220946_317_云罗广西方向-K317-876-k318-006-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221002_319_云罗罗定方向-K322-121-236-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221022_321_云梧罗定方向-K323-916-324-100-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221036_323_云罗罗定方向-K324-426-626-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221057_325_云罗罗定方向-K327-745-945-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221112_327_云罗罗定方向-K328-676-846-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221112_329_云罗罗定方向-K329-026-166-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221441_332_云罗广西方向-K345-610-800-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221458_334_云罗广西方向-K344-196-356-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221458_336_云罗广西方向-K343-976-K344-066-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221522_338_云罗罗定方向-K342-376-516-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510221522_340_云罗罗定方向-K342-216-316-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510230901_342_肇阳广西方向-K42-670-K43-000-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510230938_347_肇阳广西方向-K46-760-K47-050-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231007_350_肇阳广西方向-K47-320-490-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231047_353_云罗广西方向-K329-350-550-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231109_355_云罗罗定方向-K329-882-912-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231109_357_云罗罗定方向-K330-206-316-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231109_359_云罗罗定方向-K330-075-225-路堤墙点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231349_362_云罗广西方向-K338-340-680-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231407_364_云罗罗定方向-K339-296-735-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231407_366_云罗广西方向-K339-655-900-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231452_369_云罗罗定方向-K340-275-415-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231452_371_云罗广西方向-K340-416-710-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231452_373_云罗罗定方向-K340-576-768-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231533_376_云罗罗定方向-K340-955-995-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510231616_378_云罗罗定方向-K332-950-333-250-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510241018_382_云梧建城互通-EK0-012-150-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510241113_387_肇阳广西方向-宋桂互通DK0-045-383-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510241225_395_云罗附城互通-DK0-060-240-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510241321_398_云梧高村互通-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202510281419_176_清云湛江方向-K603-562-632-点云-实验
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202511061517_190_清云湛江方向-K514-700-860-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202511061547_192_清云湛江方向-K516-550-650-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202511061604_194_清云汕头方向-K520-200-300-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202511061630_197_清云汕头方向-K531-880-980-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202511131641_198_实验20251113清云湛江方向-K516-540-740-重复01
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011038_206_惠清南昆山-AK0-810-920-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011127_209_惠清湛江方向-K307-600-752-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011148_211_惠清湛江方向-K306-790-K307-052-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011148_213_惠清湛江-K306-320-432-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011232_215_惠清打鼓岭-AK1-175-450-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011522_221_惠清湛江方向-K308-270-460-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011522_223_惠清汕头方向-K308-441-619-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011611_225_惠清湛江方向-K308-804-K309-012-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011611_231_惠清汕头方向-K309-036-159-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512011700_234_惠清汕头方向-K318-820-K319-020-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512020924_237_惠清石岭隧道清远端口-K347-950-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512020956_240_惠清石岭隧道惠州端口-K347-562-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512020956_243_惠清湛江方向-K347-437-552-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512021032_246_惠清汕头方向-K346-850-K347-015-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512021058_249_惠清湛江方向-K343-550-670-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512021058_252_惠清湛江方向-K343-358-545-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512021156_255_惠清湛江方向-K304-036-144-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512021404_258_惠清汕头方向-K359-680-800-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512021421_261_惠清惠州方向-K361-300-530-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512150934_266_惠清惠州方向-K359-680-800-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151003_268_惠清高速湛江方向-K367-340-450-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151033_271_惠清湛江方向-K368-710-850-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151033_274_惠清汕头方向-K368-850-980-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151121_277_惠清湛江方向-K376-545-695-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151141_279_惠清汕头方向-K382-045-295-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151202_281_惠清湛江方向-K386-050-210-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151202_283_惠清汕头方向-K386-280-450-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151241_285_惠清湛江方向-K387-810-865-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151307_288_惠清湛江方向-K395-010-205-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151511_290_惠清太和洞隧道仰坡-K423-875-K424-007-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151631_292_惠清三门互通-AK1-600-850-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512151702_295_惠清高山顶隧道惠州端口点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512160856_297_惠清杨梅隧道清远端口点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512160938_304_惠清湛江方向-K351-310-700-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161015_308_惠清湛江方向-K349-020-070-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161055_313_惠清湛江方向-K343-358-545-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161114_314_惠清湛江方向-K336-435-662-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161140_317_惠清汕头方向-K329-810-K330-020-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161200_319_惠清汕头方向-K327-059-109-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161200_322_惠清湛江方向-K326-960-K327-082-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161229_324_惠清湛江方向-K325-470-610-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161254_327_惠清汕头方向-K320-410-510-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161254_331_惠清湛江方向-K320-410-520-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161436_334_惠清汕头方向-K308-000-050-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161436_335_惠清汕头方向-K308-220-280-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161455_336_惠清汕头方向-K308-952-K309-020-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161455_337_惠清汕头方向-K309-036-159-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161550_339_惠清湛江方向-K358-350-560-点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202512161627_342_高山顶清远端口点云
D:/CloudPointProcessing/PCGSPRO_1761030020/wappe2007@qq.com\DJI_202601131513_208_清云湛江方向-K603-562-632-点云-实验0113重复

'''