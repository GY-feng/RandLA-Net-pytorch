import sys
from pathlib import Path
import exiftool

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger

def check_exif_data_with_exiftool(image_path, exiftool_path = "../../ExifTool/exiftool", is_print=True):
    try:
        with exiftool.ExifTool(executable=exiftool_path) as et:
            # 正确用法：使用 execute_json() 获取所有元数据
            metadata = et.execute_json(image_path)
            
            if not metadata:
                logger.warning("未通过pyexiftool找到EXIF数据")
            elif is_print:
                logger.info("=== 详细EXIF数据(pyexiftool提取) ===")
                # metadata 是一个包含单个字典的列表
                for tag, value in metadata[0].items():
                    logger.info(f"{tag}: {value}")
                    
            return metadata[0]
                    
    except Exception as e:
        logger.error(f"使用pyexiftool读取时出错: {e}")

if __name__ == "__main__":
    img_path = "./data/AK0_A.JPG"
    check_exif_data_with_exiftool(img_path)