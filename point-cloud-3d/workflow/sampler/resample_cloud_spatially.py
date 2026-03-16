import sys
from pathlib import Path
import open3d as o3d
import numpy as np
import cupy as cp

sys.path.append(str(Path(__file__).parent.parent.parent))

from pointcloud import PointCloud as PC
from workflow.utils.logger import logger

def resample_cloud_spatially(pc: PC, min_dist_between_points: float) -> PC:
    """
    Spatially resample the input cloud to ensure that core points are at least 'min_dist_between_points' away from each other.

    :param min_dist_between_points: Minimum distance between core points.
    :param cal_normals: Whether to compute normals before resampling.
    
    :return: The subsampled new PointCloud.
    """
    # Step 1: Convert to Open3D PointCloud object
    points = np.vstack((pc.x.get(), pc.y.get(), pc.z.get())).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    # Step 2: Calculate the average distance between points
    total_distance = 0
    num_neighbors = 0

    # Use KDTree to calculate distances between all points
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i, point in enumerate(points):
        _, neighbors, distances = pcd_tree.search_knn_vector_3d(point, knn=2)  # Find 2 nearest neighbors (including self)
        if len(neighbors) > 1:
            total_distance += np.sqrt(distances[1])  # Exclude self, use the second distance
            num_neighbors += 1

    average_distance = total_distance / num_neighbors if num_neighbors > 0 else 0
    logger.info(f"平均最近邻距离: {average_distance} 米")

    # Step 3: Calculate the normal estimation radius as twice the average distance
    normal_estimation_radius = 2 * average_distance

    # Step 4: Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    logger.info(f"法向向量: {np.asarray(pcd.normals).shape}")

    pc.normals = cp.asarray(pcd.normals)
    
    # Step 5: Initialize marker array
    markers = np.ones(pc.point_nums, dtype=bool)  # All points are initially marked

    # Step 6: Perform resampling
    for i, point in enumerate(points):
        if markers[i]:  # Only consider marked points
            # Look for neighbors and unmark them
            _, neighbours, _ = pcd_tree.search_radius_vector_3d(point, min_dist_between_points)
            for idx in neighbours:
                if idx != i:
                    markers[idx] = False  # Unmark the neighbor

    # Step 7: Apply the markers to select the subsampled points
    sampled_pc = pc[markers]

    return sampled_pc, normal_estimation_radius
