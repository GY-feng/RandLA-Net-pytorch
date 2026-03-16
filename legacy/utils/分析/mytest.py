#!/usr/bin/env python3
"""
test.py
- 批量推理一个文件夹下的所有 .las
- 使用训练好的 checkpoint 做局部块推理 + 投票融合
- 将预测写回新的 .las（classification 字段），文件名为 <orig>_pred.las

运行:
    python test.py

配置请在脚本顶部的 CONFIG 区修改（已写死为你要求的默认路径）
"""
import os
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree
import laspy
from tqdm import tqdm

# ------------------ CONFIG 区（按需修改） ------------------
# 模型 checkpoint（相对于脚本运行目录）
MODEL_PATH = Path("runs/2026-01-19_04-32/checkpoint_100.pth")

# 输入 LAS 文件夹（脚本会遍历该目录下所有 .las）
INPUT_DIR = Path("datasets/drone_highway/test")

# 每次送入模型的点数（默认 = 65536 * 2）
NUM_POINTS = 65536 * 2

# 类别数
NUM_CLASSES = 5

# 输出文件后缀（新文件名 = 原名 + OUT_SUFFIX + .las）
OUT_SUFFIX = "_pred"

# 是否使用 GPU（若可用则启用）
USE_CUDA_IF_AVAILABLE = True

# 默认 KDTREE 查询是否使用全点（如点数 < NUM_POINTS，会退化）
# ----------------------------------------------------------

# 导入模型定义（假设 model.py 与本脚本在同一项目根目录）
try:
    from model import RandLANet
except Exception as e:
    print("无法导入 RandLANet (model.py)。请确保 test.py 在项目根目录且 model.py 可用。")
    raise

# ----------------- 功能函数 -----------------
def normalize_features_from_las(las):
    """
    将 laspy LasData -> features numpy (N,6)
    返回 float32 范围 0..1 的特征: [r,g,b,intensity, return_number, number_of_returns]
    针对你描述的数据范围做了鲁棒处理：
      - RGB 在 0-256 => 除 255（若 max<=256）
      - intensity 可能为 0..65536，使用 99% 分位截断再归一化
      - 回波按最大 5 归一化（若 observed max <5 也按5）

      
    """
    n_points = len(las.x)

    # color
    try:
        r = np.asarray(las.red,   dtype=np.float32) / 255.0
        g = np.asarray(las.green, dtype=np.float32) / 255.0
        b = np.asarray(las.blue,  dtype=np.float32) / 255.0
    except AttributeError:
        print("警告: LAS文件没有颜色字段，填充为0")
        r = g = b = np.zeros(n_points, dtype=np.float32)
    # has_color = hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')
    # if has_color:
    #     r = np.asarray(las.red, dtype=np.float32)
    #     g = np.asarray(las.green, dtype=np.float32)
    #     b = np.asarray(las.blue, dtype=np.float32)
    #     max_color = max(r.max() if r.size else 0, g.max() if g.size else 0, b.max() if b.size else 0)
    #     # If values are in 0-256 range (common), divide by 255, else fallback to 65535
    #     if max_color <= 256:
    #         denom = 255.0
    #     else:
    #         denom = 65535.0
    #     r = r / (denom + 1e-9)
    #     g = g / (denom + 1e-9)
    #     b = b / (denom + 1e-9)
    # else:
    #     r = g = b = np.zeros(n_points, dtype=np.float32)

    # intensity

    intensity=np.asarray(las.intensity, dtype=np.float32) / 65536.0
    # if hasattr(las, 'intensity'):
    #     intensity = np.asarray(las.intensity, dtype=np.float32)
    #     # clip at 99 percentile to reduce outlier influence (robust)
    #     if intensity.size > 0:
    #         p99 = np.percentile(intensity, 99)
    #         p99 = p99 if p99 > 0 else (intensity.max() if intensity.max()>0 else 1.0)
    #         intensity = np.clip(intensity, 0, p99) / (p99 + 1e-9)
    #     else:
    #         intensity = np.zeros(n_points, dtype=np.float32)
    # else:
    #     intensity = np.zeros(n_points, dtype=np.float32)

    # returns

    ret_n = np.array(las.return_number, dtype=np.float32) / 5.0
    n_ret = np.array(las.number_of_returns, dtype=np.float32) / 5.0
    # if hasattr(las, 'return_number') and hasattr(las, 'number_of_returns'):
    #     ret_n = np.asarray(las.return_number, dtype=np.float32)
    #     n_ret = np.asarray(las.number_of_returns, dtype=np.float32)
    #     # use safe max (user indicated <=5)
    #     max_ret = int(max(5, ret_n.max() if ret_n.size else 5, n_ret.max() if n_ret.size else 5))
    #     ret_n = ret_n / float(max_ret)
    #     n_ret = n_ret / float(max_ret)
    # else:
    #     ret_n = np.zeros(n_points, dtype=np.float32)
    #     n_ret = np.zeros(n_points, dtype=np.float32)

    feats = np.vstack([r, g, b, intensity, ret_n, n_ret]).T.astype(np.float32)
    return feats

def predict_full_cloud(model, cloud_xyz, cloud_features, num_classes, device, num_points=65536, verbose=True):
    """
    总体思路：

    用 KDTree 切块（以未被看过次数最少的点为中心），把块送入模型，
    得到每个块内点的类别概率（softmax），把这些概率累加到对应的原始点上。
    最终每个原始点取累加概率的 argmax 作为最终类别。


    投票式全场景推理（块推理 + 概率累加）
    返回 final_labels (N,) np.int32
    """
    model.eval()
    n_points = cloud_xyz.shape[0]

    # score accumulation; float32 确保内存可控
    # 为每个原始点准备一个长度为 num_classes 的分数箱，用来累加概率
    score_flat = np.zeros((n_points, num_classes), dtype=np.float32)
    # 记录每个点被“看过”的次数或权重。初始化为小随机值，目的是“优先处理最少被看过的点”。
    possibility = np.random.rand(n_points) * 1e-3

    # KDTree for neighbor queries
    # 用来快速找到某个中心点附近的 k 个点（k=num_points），比遍历全场快很多。
    tree = KDTree(cloud_xyz)

    iters = max(1, n_points // max(1, (num_points // 2)))
    if verbose:
        print(f"[predict_full_cloud] n_points={n_points}, num_points={num_points}, iters≈{iters}")

    for i in range(iters):
        # 选当前“最少被看过”的点作为新窗口中心。
        center_idx = int(np.argmin(possibility))
        # 找出中心点周围 k 个最近的原始点（索引用来映射回原始点）。
        center_point = cloud_xyz[center_idx].reshape(1, -1)

        k = min(num_points, n_points)
        _, idxs = tree.query(center_point, k=k)
        idxs = idxs[0]

        pts = torch.from_numpy(cloud_xyz[idxs].astype(np.float32)).unsqueeze(0).to(device)  # (1,k,3)
        feat = torch.from_numpy(cloud_features[idxs].astype(np.float32)).unsqueeze(0).to(device)  # (1,k,D)
        # local coords
        center_t = torch.from_numpy(center_point.astype(np.float32)).to(device)
        # 训练时网络期望局部坐标（平移不变），所以推理里要减去中心点，使模型看到的坐标分布和训练一致。
        pts = pts - center_t

        # 把坐标与特征拼成 (1, k, 3 + feature_dim) 的输入张量。
        input_tensor = torch.cat([pts, feat], dim=-1)  # (1,k,3+D)

        with torch.no_grad():
            logits = model(input_tensor)
            # normalize to (k, C)
            if logits.dim() == 3 and logits.shape[1] == num_classes:
                # (1, C, k)
                logits = logits.transpose(1, 2).reshape(-1, num_classes)
            elif logits.dim() == 3 and logits.shape[2] == num_classes:
                # (1, k, C)
                logits = logits.reshape(-1, num_classes)
            else:
                logits = logits.reshape(-1, num_classes)
            scores = F.softmax(logits, dim=-1).cpu().numpy()

        score_flat[idxs] += scores
        possibility[idxs] += 1.0

        if verbose and (i % 10 == 0 or i == iters - 1):
            mean_cov = float(possibility.mean())
            print(f"[predict] {i+1}/{iters} center_idx={center_idx} mean_coverage={mean_cov:.3f}")

    final_labels = np.argmax(score_flat, axis=1).astype(np.int32)
    return final_labels

# ----------------- 主流程 -----------------
def main():
    print("=== test.py batch LAS prediction ===")
    # device
    device = torch.device('cuda:0' if (torch.cuda.is_available() and USE_CUDA_IF_AVAILABLE) else 'cpu')
    print(f"[main] device = {device}")

    # check model
    if not MODEL_PATH.exists():
        print(f"[ERROR] MODEL_PATH not found: {MODEL_PATH}")
        return

    # check input dir
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"[ERROR] INPUT_DIR not found or not a directory: {INPUT_DIR}")
        return

    las_files = sorted(INPUT_DIR.glob("*.las"))
    if len(las_files) == 0:
        print(f"[ERROR] no .las files found in {INPUT_DIR}")
        return

    # load checkpoint
    print(f"[main] loading checkpoint: {MODEL_PATH}")
    ckpt = torch.load(str(MODEL_PATH), map_location='cpu')

    # We will instantiate model for d_in = 3 + feature_dim (from first las)
    # Load first LAS to infer feature dim
    sample_las = laspy.read(str(las_files[0]))
    sample_feats = normalize_features_from_las(sample_las)
    d_in = 3 + sample_feats.shape[1]
    print(f"[main] inferred d_in = {d_in} (3 + {sample_feats.shape[1]})")

    # instantiate model
    model = RandLANet(d_in, NUM_CLASSES, num_neighbors=16, decimation=4, device=device)
    # try to load weights
    try:
        model.load_state_dict(ckpt['model_state_dict'])
    except Exception:
        try:
            if 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                # assume ckpt itself is a state_dict
                model.load_state_dict(ckpt)
        except Exception as e:
            print("[ERROR] failed to load checkpoint into model:", e)
            return
    model.to(device).eval()

    # Process each LAS file in folder
    for las_path in las_files:
        print(f"\n[main] Processing {las_path.name} ...")
        las = laspy.read(str(las_path))

        # load xyz + features (normalized)
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        feats = normalize_features_from_las(las)  # (N,6)
        print(f"[main] points: {xyz.shape[0]}, features dim: {feats.shape[1]}")

        # run prediction
        labels = predict_full_cloud(model, xyz, feats, NUM_CLASSES, device, num_points=NUM_POINTS, verbose=True)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"[main] prediction unique labels: {dict(zip(unique.tolist(), counts.tolist()))}")

        # write to new LAS
        out_path = las_path.with_name(las_path.stem + OUT_SUFFIX + las_path.suffix)
        print(f"[main] writing predicted LAS -> {out_path}")
        # ensure dtype fits in classification field
        labels_u8 = labels.astype(np.uint8)
        las.classification = labels_u8
        las.write(str(out_path))
        print(f"[main] saved {out_path}")

    print("\n[main] All done.")

if __name__ == "__main__":
    main()


# import numpy as np
# import torch
# import torch.nn.functional as F
# from sklearn.neighbors import KDTree # 用于最后的标签投影

# def predict_full_cloud(model, cloud_xyz, cloud_features, num_classes, device, num_points=65536):
#     """
#     cloud_xyz: 原始点云坐标 (N, 3)
#     cloud_features: 原始点云特征 (N, D), 如 RGB
#     """
#     model.eval()
#     n_points = cloud_xyz.shape[0]
    
#     # 1. 初始化得分矩阵和采样可能性
#     # score_flat 会存储每个点在每个类别上的累加得分
#     score_flat = np.zeros((n_points, num_classes))
#     # possibility 记录每个点被覆盖的频率（越小代表越没被看过）
#     possibility = np.random.rand(n_points) * 1e-3 
    
#     # 建立一个快速搜索树
#     search_tree = KDTree(cloud_xyz)

#     print("开始全场景循环推理...")
#     # 2. 循环推理，直到所有点都被覆盖到足够的次数（例如覆盖 2 次）
#     # 这里为了演示，我们设置一个固定的步数，实际可根据可能性判断
#     iters = n_points // (num_points // 2) # 粗略估计需要多少步能覆盖全场
    
#     for i in range(iters):
#         # 选择当前最没被看过的点作为中心
#         center_idx = np.argmin(possibility)
#         center_point = cloud_xyz[center_idx].reshape(1, -1)
        
#         # 搜索最近的 num_points 个点
#         _, indices = search_tree.query(center_point, k=num_points)
#         indices = indices[0]
        
#         # 准备模型输入
#         pts = torch.from_numpy(cloud_xyz[indices]).float().unsqueeze(0).to(device)
#         feat = torch.from_numpy(cloud_features[indices]).float().unsqueeze(0).to(device)
#         # 注意：这里需要把坐标减去中心点，保持模型平移不变性
#         pts = pts - torch.from_numpy(center_point).to(device) 
        
#         input_tensor = torch.cat([pts, feat], dim=-1) # 形状 (1, N, 6)

#         with torch.no_grad():
#             logits = model(input_tensor) # 输出 (1, num_classes, N)
#             logits = logits.transpose(1, 2).reshape(-1, num_classes) # 转成 (N, num_classes)
#             scores = F.softmax(logits, dim=-1).cpu().numpy()

#         # 3. 投票累加：把预测的分数加回到原始索引位置
#         score_flat[indices] += scores
        
#         # 更新可能性：增加被选中点的权重，让它们下次不被选中
#         possibility[indices] += 1.0 
        
#         if i % 10 == 0:
#             print(f"进度: {i}/{iters}")

#     # 4. 最终决策：取每个点累加得分最高的类别
#     final_labels = np.argmax(score_flat, axis=1)
#     return final_labels

# # --- 映射逻辑：如何实现真正意义上的逐点预测 ---
# def map_labels_to_raw(downsampled_xyz, downsampled_labels, raw_xyz):
#     """
#     downsampled_xyz: 模型跑过的点
#     downsampled_labels: 模型跑出来的结果
#     raw_xyz: 还没跑过的、原始最精细的点
#     """
#     # 建立搜索树，找到原始点在下采样点中最近的那个点
#     tree = KDTree(downsampled_xyz)
#     _, nearest_idx = tree.query(raw_xyz, k=1)
    
#     # 把最近点的标签直接传给原始点
#     raw_labels = downsampled_labels[nearest_idx.flatten()]
#     return raw_labels