import os
import glob
import pickle
import numpy as np
import laspy
from sklearn.neighbors import KDTree
from os.path import join, exists

# ====== 配置区（你可以调的只有这里） ======
DATASET_NAME = "SlopeLAS"

RAW_LAS_DIR = "../datasets/{}/raw_las".format(DATASET_NAME)
OUT_ROOT = "../datasets/{}".format(DATASET_NAME)

GRID_SIZE = 0.04          # 第一次 subsample压缩原始超密点云，防止爆内存.调小到 4cm，确保灾害点不被物理剔除
FIRST_SUBSAMPLE = 0.02    # 第2次 subsample 对应调小
# ============================================

ORIGINAL_PLY_DIR = join(OUT_ROOT, "original_ply")
SUB_PC_DIR = join(OUT_ROOT, "input_{:.2f}".format(GRID_SIZE))

os.makedirs(ORIGINAL_PLY_DIR, exist_ok=True)
os.makedirs(SUB_PC_DIR, exist_ok=True)


def grid_sub_sampling(points, labels, grid_size):
    """
    简化版 grid subsampling（只处理 xyz + label）

    这是一个最简体素网格下采样（voxel grid subsampling）：

    用 grid_size 把空间划分为规则立方体（体素）

    每个体素只保留 一个点

    点的 label 与点一起保留
    """
    coords = np.floor(points / grid_size).astype(np.int32)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)

    sub_points = points[unique_idx]
    sub_labels = labels[unique_idx]

    return sub_points, sub_labels


def write_ply_xyz_label(filename, xyz, labels):
    """
    写最简 ply：x y z class

    手写一个最简单的 ASCII PLY

    每个点只包含：

    坐标 (x, y, z)

    语义标签 class
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar class\n")
        f.write("end_header\n")

        for p, l in zip(xyz, labels):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(l)}\n")


def process_las(las_path):
    name = os.path.basename(las_path).replace(".las", "")
    print(f"\nProcessing {name}")

    # ===== 1. 读取 LAS =====
    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    # laspy 2.x：classification 是 SubFieldView，必须转 numpy
    labels = np.array(las.classification, dtype=np.uint8)


    # ===== 2. 第一次 subsample（粗）=====
    '''
    去掉冗余密集点
    但仍保留灾害的几何细节

    这个结果用于：
    后续第二次 subsample
    projection index 的“原始点代理”
    '''
    xyz_1, labels_1 = grid_sub_sampling(xyz, labels, FIRST_SUBSAMPLE)

    # 保存 original ply（用于 debug / 可视化）
    full_ply_path = join(ORIGINAL_PLY_DIR, name + ".ply")
    write_ply_xyz_label(full_ply_path, xyz_1, labels_1)

    # ===== 3. 第二次 subsample（训练用）=====
    '''
    “真正的训练点云”
    点数更少
    空间分布更均匀
    网络计算量可控
    '''
    sub_xyz, sub_labels = grid_sub_sampling(xyz_1, labels_1, GRID_SIZE)

    sub_ply_path = join(SUB_PC_DIR, name + ".ply")
    write_ply_xyz_label(sub_ply_path, sub_xyz, sub_labels)

    # ===== 4. KDTree =====
    '''
    KDTree 用来干什么？

    最近邻搜索

    后面两类任务会用：
    训练时的 neighborhood 查询
    推理 / 评估时 label 投影
    ->:
    网络只对“稀疏点”预测标签，但评估要在“更密的点 / 原始点”上进行，于是需要一个映射把预测结果投影回去。
    做法：label projection（标签投影）
        对每一个 原始点

        找到它最近的一个 训练点

        使用该训练点的预测 label
    '''
    tree = KDTree(sub_xyz, leaf_size=50)
    kd_tree_file = join(SUB_PC_DIR, name + "_KDTree.pkl")
    with open(kd_tree_file, "wb") as f:
        pickle.dump(tree, f)

    # ===== 5. Projection index（评估用）=====
    '''

    为什么要这一步：
    RandLA-Net 的核心特点：
    随机下采样 + 局部聚合
    网络只在 sub_xyz 上预测
    但评估时：
    IoU / OA / F1
    都是对 更密点 / 原始点算的

    推理阶段：pred_sub = model(sub_xyz)  # shape: [N_sub]
    投影回密点：pred_dense = pred_sub[proj_idx]

    
    如何做：

    xyz_1：较密点（第一次 subsample 后）
    sub_xyz：训练点（第二次 subsample 后）

    proj_idx[i] = j表示：第 i 个 xyz_1 点，最近的训练点是 sub_xyz[j]，xyz_1  ──NN──▶  sub_xyz
    保存的是 [proj_idx, labels_1]
    评估时 不需要再读 LAS，直接load pkl，算指标
    '''
    proj_idx = np.squeeze(tree.query(xyz_1, return_distance=False)).astype(np.int32)

    proj_file = join(SUB_PC_DIR, name + "_proj.pkl")
    with open(proj_file, "wb") as f:
        pickle.dump([proj_idx, labels_1], f)

    print(f"Finished {name}")


if __name__ == "__main__":
    las_files = glob.glob(join(RAW_LAS_DIR, "*.las"))
    assert len(las_files) > 0, "No LAS files found!"

    for las_file in las_files:
        process_las(las_file)

    print("\nAll slope LAS files processed.")
