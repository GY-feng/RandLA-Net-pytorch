import laspy
import numpy as np


def compute_classification_stats(las_path):
    """
    统计单个 LAS / LAZ 文件中 classification 的点数和比例

    Args:
        las_path (str): .las 或 .laz 文件路径

    Returns:
        dict: {
            "total_points": int,
            "class_stats": {
                class_id: {
                    "count": int,
                    "ratio": float   # 0~1
                }
            }
        }
    """
    las = laspy.read(las_path)

    total_points = len(las.points)
    classes, counts = np.unique(las.classification, return_counts=True)

    class_stats = {}
    for cls, count in zip(classes, counts):
        class_stats[int(cls)] = {
            "count": int(count),
            "ratio": count / total_points if total_points > 0 else 0.0
        }

    return {
        "total_points": total_points,
        "class_stats": class_stats
    }
