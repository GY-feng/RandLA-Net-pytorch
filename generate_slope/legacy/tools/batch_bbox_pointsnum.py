import os
import laspy
from pathlib import Path

# ================= 配置区 =================
# 在这里指定你的文件夹路径
TARGET_FOLDER = r'D:\Feng\带草_原始data' 
# =========================================

def process_las_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"错误: 找不到文件夹 {folder_path}")
        return

    # 结果存储
    file_results = []
    total_points = 0
    total_l, total_w, total_h = 0, 0, 0
    
    # 获取目录下所有 .las 和 .laz 文件
    las_files = list(folder.glob('*.las')) + list(folder.glob('*.laz'))
    
    if not las_files:
        print("文件夹内没有找到 .las 或 .laz 文件。")
        return

    print(f"开始处理，共找到 {len(las_files)} 个文件...")

    for file_path in las_files:
        try:
            # 使用 laspy.open 只读取文件头（header），速度极快，无需加载整个点云
            with laspy.open(file_path) as fh:
                header = fh.header
                
                # 计算长宽高 (Max - Min)
                length = header.x_max - header.x_min
                width = header.y_max - header.y_min
                height = header.z_max - header.z_min
                point_count = header.point_count
                
                file_results.append({
                    'name': file_path.name,
                    'l': length, 'w': width, 'h': height,
                    'count': point_count
                })
                
                # 累加用于后续平均值计算
                total_points += point_count
                total_l += length
                total_w += width
                total_h += height
                
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")

    # 计算平均值
    num_files = len(file_results)
    avg_l = total_l / num_files
    avg_w = total_w / num_files
    avg_h = total_h / num_files
    avg_points = total_points / num_files

    # 准备写入内容
    output_lines = []
    output_lines.append(f"--- 单个文件详细信息 ---")
    for res in file_results:
        line = (f"文件名: {res['name']} | "
                f"长: {res['l']:.3f}, 宽: {res['w']:.3f}, 高: {res['h']:.3f} | "
                f"点云数量: {res['count']}")
        output_lines.append(line)

    output_lines.append("\n" + "="*50 + "\n")
    output_lines.append(f"--- 总体统计信息 ---")
    output_lines.append(f"处理文件总数: {num_files}")
    output_lines.append(f"点云总数量: {total_points}")
    output_lines.append(f"平均点云数量: {avg_points:.2f}")
    output_lines.append(f"平均尺寸 (长x宽x高): {avg_l:.3f} x {avg_w:.3f} x {avg_h:.3f}")

    # 写入 TXT 文件 (保存在文件夹同级目录)
    output_file = folder.parent / f"{folder.name}_统计报告.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    print(f"处理完成！统计报告已生成至: {output_file}")

if __name__ == "__main__":
    process_las_files(TARGET_FOLDER)