"""
mytest.py

批量推理 LAS 文件，输出 classification 字段 (0/1/2)。
Run:
    python mytest.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree
import laspy
from tqdm import tqdm

from model import RandLANet

# ------------------ CONFIG ------------------
MODEL_PATH = Path("runs/your_run/checkpoint_20.pth")
INPUT_DIR = Path("datasets/SlopeLAS/infer_las")
OUT_SUFFIX = "_pred"

NUM_POINTS = 65536
NUM_CLASSES = 3
BLOCK_SIZE = 20.0
USE_CUDA_IF_AVAILABLE = True
# -------------------------------------------


def normalize_block(points: np.ndarray, block_size: float) -> np.ndarray:
    center = points.mean(axis=0)
    points = points - center
    scale = float(block_size) * 0.5
    if scale > 1e-6:
        points = points / scale
    return points


def predict_full_cloud(model, cloud_xyz, num_classes, device, num_points=65536, block_size=20.0, verbose=True):
    """
    KDTree + 多块投票融合。
    返回每个点的预测标签。
    """
    model.eval()
    n_points = cloud_xyz.shape[0]

    score_flat = np.zeros((n_points, num_classes), dtype=np.float32)
    possibility = np.random.rand(n_points) * 1e-3

    tree = KDTree(cloud_xyz)
    iters = max(1, n_points // max(1, (num_points // 2)))
    if verbose:
        print(f"[predict] n_points={n_points}, num_points={num_points}, iters={iters}")

    for i in range(iters):
        center_idx = int(np.argmin(possibility))
        center_point = cloud_xyz[center_idx].reshape(1, -1)

        k = min(num_points, n_points)
        _, idxs = tree.query(center_point, k=k)
        idxs = idxs[0]

        block_xyz = cloud_xyz[idxs]
        block_xyz = normalize_block(block_xyz, block_size)

        pts = torch.from_numpy(block_xyz.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(pts)
            if logits.dim() == 3 and logits.shape[1] == num_classes:
                logits = logits.transpose(1, 2).reshape(-1, num_classes)
            elif logits.dim() == 3 and logits.shape[2] == num_classes:
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


def load_model(device):
    model = RandLANet(d_in=3, num_classes=NUM_CLASSES, num_neighbors=32, decimation=4, device=device)
    model.to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def main():
    print("=== mytest.py batch LAS prediction ===")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA_IF_AVAILABLE) else "cpu")
    print("Device:", device)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {INPUT_DIR}")

    las_files = sorted(INPUT_DIR.glob("*.las"))
    if len(las_files) == 0:
        raise RuntimeError(f"No .las files found in {INPUT_DIR}")

    model = load_model(device)

    for las_path in tqdm(las_files, desc="Predict LAS"):
        las = laspy.read(las_path)
        xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

        pred = predict_full_cloud(
            model,
            xyz,
            num_classes=NUM_CLASSES,
            device=device,
            num_points=NUM_POINTS,
            block_size=BLOCK_SIZE,
            verbose=False,
        )

        las.classification = pred.astype(las.classification.dtype)
        out_path = las_path.with_name(las_path.stem + OUT_SUFFIX + las_path.suffix)
        las.write(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
