import os
import re
import laspy
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from datetime import datetime

def read_folder_list_txt(txt_path: Path):
    """
    读取 folder_list.txt，每一行是一个工程目录路径，
    自动拼接 lidars/terra_las/cloud_merged.las
    """
    las_paths = []

    if not txt_path.exists():
        print(f"❌ 未找到 {txt_path}")
        return las_paths

    with txt_path.open('r', encoding='utf-8') as f:
        for line in f:
            base_dir = line.strip()
            if not base_dir:
                continue

            base_path = Path(base_dir)
            las_path = base_path / "lidars" / "terra_las" / "cloud_merged.las"

            if las_path.exists():
                las_paths.append(las_path.resolve())
            else:
                print(f"⚠️ 未找到: {las_path}")

    print(f"✅ 从 {txt_path.name} 解析到 {len(las_paths)} 个 cloud_merged.las")
    return las_paths


def make_clean_name(las_file: Path):
    """
    用完整路径生成唯一文件名
    """
    clean = re.sub(r'[:\\\/]+', '_', str(las_file))
    clean = re.sub(r'\.las$|\.laz$', '', clean, flags=re.IGNORECASE)
    return clean


def process_las_files(las_files, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"➡️ 输出目录: {output_dir}")

    for las_file in las_files:
        try:
            print(f"---\n🔍 处理中: {las_file}")

            with laspy.open(las_file) as f:
                total = f.header.point_count
                step = max(1, total // 2000000)
                las = f.read()

                x, y = np.array(las.x)[::step], np.array(las.y)[::step]

                if not hasattr(las, 'red'):
                    print(f"⚠️ 跳过: {las_file.name} 无颜色信息")
                    continue

                r = np.array(las.red)[::step].astype(np.float32)
                g = np.array(las.green)[::step].astype(np.float32)
                b = np.array(las.blue)[::step].astype(np.float32)

            # --- 自动亮度对比度拉伸 ---
            for channel in [r, g, b]:
                low, high = np.percentile(channel, (2, 98))
                np.clip(channel, low, high, out=channel)
                channel -= low
                if high - low > 0:
                    channel *= (255.0 / (high - low))

            r, g, b = r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

            # --- 坐标映射 ---
            img_size = 1500
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            scale = (img_size - 1) / max(x_max - x_min, y_max - y_min, 1e-9)
            px = ((x - x_min) * scale).astype(np.int32)
            py = ((y - y_min) * scale).astype(np.int32)

            img_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            px = np.clip(px, 0, img_size - 1)
            py = np.clip(py, 0, img_size - 1)

            img_rgb[py, px, 0] = r
            img_rgb[py, px, 1] = g
            img_rgb[py, px, 2] = b

            img = Image.fromarray(np.flipud(img_rgb))

            # --- 点增强 ---
            img = img.filter(ImageFilter.MaxFilter(size=3))

            clean_name = make_clean_name(las_file)
            save_path = output_dir / f"{clean_name}_Enhanced.png"
            img.save(save_path)

            print(f"✨ 已保存: {save_path.name}")

        except Exception as e:
            print(f"❌ 错误: {las_file} | {str(e)}")

    print(f"\n✅ 全部任务完成！请检查目录: {output_dir}")


def generate_bright_bold_topview():
    # 读取脚本同目录下的 folder_list.txt
    script_dir = Path(__file__).resolve().parent
    folder_list_path = script_dir / "folder_list.txt"

    las_files = read_folder_list_txt(folder_list_path)
    if not las_files:
        print("❌ 没有可处理的 LAS 文件，程序退出。")
        return

    # 输出到桌面新建文件夹
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    desktop = Path.home() / "Desktop"
    output_dir = desktop / f"LAS_Previews_{ts}"

    process_las_files(las_files, output_dir)


if __name__ == "__main__":
    generate_bright_bold_topview()
