"""
check_dataset.py

快速检查 SlopeLAS 数据集：
- .npy 形状
- 标签范围/分布
- 坐标归一化范围

运行:
    python check_dataset.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from tqdm import tqdm

CFG = {
    "dataset_root": Path("datasets/SlopeLAS"),
    "splits": ("train", "val", "test"),
    "max_files_per_split": -1,  # -1 表示全量
}


def main():
    root = CFG["dataset_root"]
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for split in CFG["splits"]:
        split_dir = root / split
        files = sorted(split_dir.glob("*.npy"))
        if len(files) == 0:
            print(f"[{split}] no files")
            continue

        if CFG["max_files_per_split"] > 0:
            files = files[: CFG["max_files_per_split"]]

        label_counts = np.zeros(3, dtype=np.int64)
        xyz_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        xyz_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        for f in tqdm(files, desc=f"Checking {split}"):
            data = np.load(f)
            if data.ndim != 2 or data.shape[1] != 4:
                raise ValueError(f"Bad npy shape {data.shape} in {f}")

            points = data[:, :3]
            labels = data[:, 3].astype(np.int64)

            if labels.size > 0:
                label_counts += np.bincount(labels, minlength=3)

            xyz_min = np.minimum(xyz_min, points.min(axis=0))
            xyz_max = np.maximum(xyz_max, points.max(axis=0))

        print(f"\n[{split}] files={len(files)}")
        print(f"label counts: {label_counts} (ratio: {label_counts / max(1, label_counts.sum())})")
        print(f"xyz min: {xyz_min}")
        print(f"xyz max: {xyz_max}")

        if np.any(np.abs(xyz_min) > 2.5) or np.any(np.abs(xyz_max) > 2.5):
            print("[WARN] 坐标范围可能未正确归一化，期望约在 [-1, 1] 附近。")


if __name__ == "__main__":
    main()
