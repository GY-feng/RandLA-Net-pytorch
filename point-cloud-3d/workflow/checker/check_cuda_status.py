import sys
from pathlib import Path
import cupy as cp
import open3d as o3d
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger

def check_cuda_status(
    log: bool = True,
    check_o3d: bool = True,
    check_cv2: bool = True,
    check_cupy: bool = True
    ):
    
    if not o3d.core.cuda.is_available() and check_o3d:
        logger.error("错误: Open3D CUDA支持不可用")
        return False
    if not cp.cuda.is_available() and check_cupy:
        logger.error("错误: CuPy CUDA支持不可用")
        return False
    # if not cv2.cuda.getCudaEnabledDeviceCount() and check_cv2:
    #     logger.error("错误: OpenCV CUDA支持不可用")
    #     return False
    
    if log:
        logger.info('当前系统CUDA依赖完备')
    return True

if __name__ == "__main__":
    if not check_cuda_status():
        exit(1)