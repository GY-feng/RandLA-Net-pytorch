import sys
from pathlib import Path
import open3d as o3d
import cupy as cp
from typing import Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC
from utils.logger import logger

def point_cloud_icp_registration(
        src: PC,
        dst: PC, 
        method = "point_to_point",
        max_correspondence_distance = 0.2,
        max_iteration = 200
) -> Tuple[cp.ndarray, cp.ndarray, float, float]:
    """
    点云的ICP配准

    参数:
    source: PointCloud源点云
    target: PointCloud目标点云
    method: 配准方法, 可选"plane"和"point"
    
    返回:
    result: 配准结果
    
    """

    if method == "point_to_plane":
        if src.normals is None or dst.normals is None:
            logger.error("点云法向量缺失, 无法进行配准")
            raise ValueError("点云法向量缺失, 无法进行配准")
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif method == "point_to_point":
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif method == "generalized_icp":
        estimation_method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    elif method == "colored_icp":
        estimation_method = o3d.pipelines.registration.TransformationEstimationForColoredICP()
    else:
        logger.error("无效的配准方法, 请选择'point_to_plane'、'point_to_point'、'generalized_icp'或'colored_icp'")
        raise ValueError("无效的配准方法, 请选择'point_to_plane'、'point_to_point'、'generalized_icp'或'colored_icp'")

    
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(cp.vstack((src.x, src.y, src.z)).T.get())
    target.points = o3d.utility.Vector3dVector(cp.vstack((dst.x, dst.y, dst.z)).T.get())

    if method == "point_to_plane":
        source.normals = o3d.utility.Vector3dVector(src.normals.get())
        target.normals = o3d.utility.Vector3dVector(dst.normals.get())
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance,
        estimation_method=estimation_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=max_iteration
        )
    )

    return result

if __name__ == "__main__":  

    import numpy as np
    from scipy.spatial import cKDTree

    from checker.check_cuda_status import check_cuda_status
    from checker.check_point_cloud import check_point_cloud
    from workflow.filter.by_classification import filter_by_classification
    if not check_cuda_status(log=False):
        exit(1)
    source = PC()
    target = PC()
    target.load_from_las("/home/CloudPointProcessing/点云实验20250708/DJI_202507081616_149_点云实验20250708重复扫描40m-90度-2cm/raw/las/LAS.las")
    source.load_from_las("/home/CloudPointProcessing/点云实验20250708/DJI_202507081616_152_点云实验20250708重复扫描80m-90度-2cm/raw/las/LAS.las")

    # 提取源点云和目标点云的地面点
    src_ground = filter_by_classification(source)
    tar_ground = filter_by_classification(target)

    points = cp.column_stack((src_ground.x, src_ground.y, src_ground.z)).get()
    # 创建KD树
    tree = cKDTree(points)
    # 查询每个点的最近邻（不包括自身）
    distances, indices = tree.query(points, k=2)  # k=2表示获取最近的两个点（自己和最近邻）
    # 提取最近邻的距离（第二近的点，因为第一近的点是自己）
    nearest_distances = distances[:, 1]
    # 计算平均最近邻距离
    average_nearest_distance = np.mean(nearest_distances)

    logger.info(f"源点云平均最近邻距离: {average_nearest_distance}")

    # check_point_cloud(src_ground)
    # check_point_cloud(tar_ground)

    src_ground.estimate_normals((0.1, 200), fast_normal_computation=False)
    tar_ground.estimate_normals((0.1, 200), fast_normal_computation=False)

    src_ground.transform_to(tar_ground.crs)

    result = point_cloud_icp_registration(src_ground, tar_ground, method="point_to_point", max_correspondence_distance=0.2, max_iteration=200)

    transformation = cp.asarray(result.transformation)
    tz = transformation[2, 3]
    logger.info(f"ICP配准结果: \n{transformation} {result.fitness} {result.inlier_rmse}")
    logger.info(f"垂直平移量: {tz:.3f} 米")

    points = cp.vstack((source.x, source.y, source.z)).T  # 形状为 (N, 3)
    R = transformation[:3, :3]  # 形状为 (3, 3)
    T = transformation[:3, 3]  # 形状为 (3,)

    transformed_points = (points @ R.T) + T

    x, y, z = transformed_points.T

    logger.info(f"转换前原点云平均高度{source.z.mean():.3f}, 目标点云平均高度{target.z.mean():.3f}, 高程差{target.z.mean() - source.z.mean():.8f}")
    logger.info(f"转换后原点云平均高度{z.mean():.3f}, 目标点云平均高度{target.z.mean():.3f}, 高程差{target.z.mean() - z.mean():.8f}")

else:
    from checker.check_cuda_status import check_cuda_status
    if not check_cuda_status(log=False, check_cv2=False):
        raise ImportError(f"CUDA依赖缺失, 无法导入{__name__}")