import sys
from pathlib import Path
import exifread

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger

def check_exif_data(image_path):
    logger.info(f"正在从 {image_path} 提取元数据...")
    # 使用 exifread 提取详细元数据
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            if not tags:
                logger.warning("未通过exifread找到EXIF数据")
            else:
                logger.info("=== 详细EXIF数据(exifread提取) ===")
                for tag in tags:
                    logger.info(f"{tag}: {tags[tag]}")
    except Exception as e:
        logger.error(f"使用exifread读取时出错: {e}")

if __name__ == "__main__":
    img_path = "./data/AK0_A.JPG"
    check_exif_data(img_path)