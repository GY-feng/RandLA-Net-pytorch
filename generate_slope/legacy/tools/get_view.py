import os
import laspy
import numpy as np
from PIL import Image, ImageFilter
from tkinter import filedialog, Tk
from pathlib import Path

def generate_bright_bold_topview():
    root = Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="请选择包含 LAS 文件的根文件夹")
    
    if not input_dir:
        return

    input_path = Path(input_dir)
    output_dir = input_path.parent / f"{input_path.name}_previews_RGB_Enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    las_files = list(input_path.rglob("*.las")) + list(input_path.rglob("*.laz"))
    print(f"✅ 扫描完成，共 {len(las_files)} 个文件。正在进行亮度增强与加粗渲染...")

    for las_file in las_files:
        try:
            print(f"---\n🔍 处理中: {las_file.name}")
            
            with laspy.open(las_file) as f:
                total = f.header.point_count
                # 为了让点显得更密，采样步长稍微调小一点，保留更多点
                step = max(1, total // 2000000) 
                las = f.read()
                
                x, y = np.array(las.x)[::step], np.array(las.y)[::step]
                
                if not hasattr(las, 'red'):
                    print(f"⚠️ 跳过: {las_file.name} 无颜色信息")
                    continue
                
                # 读取原始 RGB (处理 16-bit)
                r = np.array(las.red)[::step].astype(np.float32)
                g = np.array(las.green)[::step].astype(np.float32)
                b = np.array(las.blue)[::step].astype(np.float32)

            # --- 改进1：自动亮度对比度拉伸 ---
            # 这里的逻辑是找到 2% 和 98% 分位数，过滤掉极端噪点并把亮度撑满
            for channel in [r, g, b]:
                low, high = np.percentile(channel, (2, 98))
                np.clip(channel, low, high, out=channel)
                channel -= low
                if high - low > 0:
                    channel *= (255.0 / (high - low))

            r, g, b = r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

            # --- 坐标映射 ---
            img_size = 1500 # 分辨率提高
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            
            scale = (img_size - 1) / max(x_max - x_min, y_max - y_min, 1e-9)
            px = ((x - x_min) * scale).astype(np.int32)
            py = ((y - y_min) * scale).astype(np.int32)
            
            # --- 创建并填充画布 ---
            img_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            img_rgb[py, px, 0] = r
            img_rgb[py, px, 1] = g
            img_rgb[py, px, 2] = b
            
            # 转换为 PIL 对象
            img = Image.fromarray(np.flipud(img_rgb))

            # --- 改进2：点增强（让点变大） ---
            # 使用 MaxFilter (最大值滤波) 模拟形态学膨胀，size=3 代表点会变粗
            img = img.filter(ImageFilter.MaxFilter(size=3))

            # 命名并保存
            relative_path = las_file.relative_to(input_path)
            clean_name = str(relative_path).replace(os.sep, '_').replace('.las', '').replace('.laz', '')
            save_path = output_dir / f"{clean_name}_Enhanced.png"
            
            img.save(save_path)
            print(f"✨ 增强版预览已保存: {save_path.name}")

        except Exception as e:
            print(f"❌ 错误: {las_file.name} | {str(e)}")

    print(f"\n全部任务完成！请检查目录: {output_dir}")

if __name__ == "__main__":
    generate_bright_bold_topview()