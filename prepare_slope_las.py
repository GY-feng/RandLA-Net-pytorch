"""
prepare_slope_las.py

将 SlopeLAS 的 .las 点云预处理为 RandLA-Net 可训练的分块 .npy 数据。
输出每个样本为 (N, 4): [x, y, z, label]

默认标签映射:
- 0: 正常
- 1: 凸起
- 2: 凹陷

运行:
    python prepare_slope_las.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import laspy
from tqdm import tqdm

# --------------------------- CONFIG ---------------------------
CFG = {
    # 数据集根目录
    "dataset_root": Path("datasets/SlopeLAS"),
    "raw_las_dir": Path("datasets/SlopeLAS/raw_las"),

    # 输出分块目录
    "output_splits": ("train", "val", "test"),

    # 点云处理
    "use_voxel_subsample": True,
    "grid_size": 0.04,          # 体素下采样尺寸 (米)

    # 滑窗切块参数
    "use_sliding_window": True,
    "block_size": 20.0,         # 每块边长 (米)
    "stride": 10.0,             # 滑动步长 (米)
    "min_points": 4096,         # 少于该点数的块丢弃

    # 训练输入
    "num_points": 65536,        # 每块固定采样点数

    # 数据集划分比例
    "split_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},

    # 标签映射
    "label_map": {0: 0, 1: 1, 2: 2},
    "unknown_to_background": True,

    # 随机种子
    "seed": 42,
}
# -------------------------------------------------------------


def grid_sub_sampling(points: np.ndarray, labels: np.ndarray, grid_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化版体素下采样：每个体素保留一个点。
    仅处理 xyz + label。
    """
    if grid_size is None or grid_size <= 0:
        return points, labels

    coords = np.floor(points / grid_size).astype(np.int32)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    sub_points = points[unique_idx]
    sub_labels = labels[unique_idx]
    return sub_points, sub_labels


def normalize_block(points: np.ndarray, block_size: float) -> np.ndarray:
    """
    块内中心化 + 归一化：
    1) 减去块中心
    2) 按 block_size/2 缩放到约 [-1, 1]
    """
    center = points.mean(axis=0)
    points = points - center
    scale = float(block_size) * 0.5
    if scale > 1e-6:
        points = points / scale
    return points


def sample_block(points: np.ndarray, labels: np.ndarray, num_points: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    保证每个样本点数一致。
    """
    n = points.shape[0]
    if n >= num_points:
        idx = rng.choice(n, num_points, replace=False)
    else:
        idx = rng.choice(n, num_points, replace=True)
    return points[idx], labels[idx]


def get_window_starts(min_v: float, max_v: float, block: float, stride: float) -> list:
    """
    生成滑窗起点，确保覆盖尾部区域。
    """
    if max_v - min_v <= block:
        return [min_v]

    starts = list(np.arange(min_v, max_v - block + 1e-6, stride))
    if starts[-1] + block < max_v:
        starts.append(max_v - block)
    return starts


def map_labels(labels: np.ndarray, label_map: Dict[int, int], unknown_to_background: bool = True) -> Tuple[np.ndarray, int]:
    """
    将原始标签映射到目标标签空间。
    返回: (mapped_labels, unknown_count)
    """
    mapped = np.full(labels.shape, 0, dtype=np.int32)
    known_mask = np.zeros(labels.shape, dtype=bool)

    for src, dst in label_map.items():
        mask = labels == src
        mapped[mask] = int(dst)
        known_mask |= mask

    unknown_count = int((~known_mask).sum())
    if not unknown_to_background and unknown_count > 0:
        # 若不允许未知标签，直接丢弃（此处改为强制背景，保持样本量）
        pass

    return mapped, unknown_count


def write_block(out_dir: Path, stem: str, block_idx: int, points: np.ndarray, labels: np.ndarray) -> None:
    out_path = out_dir / f"{stem}_block_{block_idx:05d}.npy"
    data = np.hstack([points.astype(np.float32), labels.reshape(-1, 1).astype(np.int32)])
    np.save(out_path, data)


def process_las(las_path: Path, out_dirs: Dict[str, Path], rng: np.random.RandomState) -> Dict[str, int]:
    """
    处理单个 LAS 文件，返回各 split 的样本数。
    """
    stats = {"train": 0, "val": 0, "test": 0, "skipped": 0}

    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    raw_labels = np.array(las.classification, dtype=np.int32)

    mapped_labels, unknown_count = map_labels(raw_labels, CFG["label_map"], CFG["unknown_to_background"])
    if unknown_count > 0:
        print(f"[WARN] {las_path.name}: 未知标签 {unknown_count} 个，已映射为背景(0)")

    # 体素下采样
    if CFG["use_voxel_subsample"]:
        xyz, mapped_labels = grid_sub_sampling(xyz, mapped_labels, CFG["grid_size"])

    if xyz.shape[0] == 0:
        stats["skipped"] += 1
        return stats

    # 滑窗切块
    if CFG["use_sliding_window"]:
        x_min, y_min, _ = np.min(xyz, axis=0)
        x_max, y_max, _ = np.max(xyz, axis=0)

        x_starts = get_window_starts(x_min, x_max, CFG["block_size"], CFG["stride"])
        y_starts = get_window_starts(y_min, y_max, CFG["block_size"], CFG["stride"])

        block_idx = 0
        for x0 in x_starts:
            x1 = x0 + CFG["block_size"]
            x_mask = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1)
            for y0 in y_starts:
                y1 = y0 + CFG["block_size"]
                mask = x_mask & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)

                if mask.sum() < CFG["min_points"]:
                    continue

                block_xyz = xyz[mask]
                block_lbl = mapped_labels[mask]

                block_xyz = normalize_block(block_xyz, CFG["block_size"])
                block_xyz, block_lbl = sample_block(block_xyz, block_lbl, CFG["num_points"], rng)

                split = pick_split(rng, CFG["split_ratio"])
                write_block(out_dirs[split], las_path.stem, block_idx, block_xyz, block_lbl)
                stats[split] += 1
                block_idx += 1

    else:
        # 整云当作单块
        if xyz.shape[0] >= CFG["min_points"]:
            block_xyz = normalize_block(xyz, CFG["block_size"])
            block_xyz, block_lbl = sample_block(block_xyz, mapped_labels, CFG["num_points"], rng)
            split = pick_split(rng, CFG["split_ratio"])
            write_block(out_dirs[split], las_path.stem, 0, block_xyz, block_lbl)
            stats[split] += 1
        else:
            stats["skipped"] += 1

    return stats


def pick_split(rng: np.random.RandomState, ratios: Dict[str, float]) -> str:
    r = rng.rand()
    if r < ratios["train"]:
        return "train"
    if r < ratios["train"] + ratios["val"]:
        return "val"
    return "test"


def main() -> None:
    dataset_root = CFG["dataset_root"]
    raw_dir = CFG["raw_las_dir"]

    if not raw_dir.exists():
        raise FileNotFoundError(f"未找到原始数据目录: {raw_dir}")

    # 创建输出目录
    out_dirs = {}
    for split in CFG["output_splits"]:
        out_dir = dataset_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[split] = out_dir

    # 保存配置副本，便于追溯
    cfg_path = dataset_root / "prepare_config.json"
    cfg_dump = {}
    for k, v in CFG.items():
        if isinstance(v, Path):
            cfg_dump[k] = str(v)
        else:
            cfg_dump[k] = v
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, ensure_ascii=False, indent=2)

    las_files = sorted(raw_dir.glob("*.las"))
    if len(las_files) == 0:
        raise RuntimeError(f"目录下没有 .las 文件: {raw_dir}")

    rng = np.random.RandomState(CFG["seed"])

    total = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    print(f"找到 {len(las_files)} 个 LAS 文件，开始处理...")

    for las_file in tqdm(las_files, desc="Processing LAS"):
        stats = process_las(las_file, out_dirs, rng)
        for k in total:
            total[k] += stats.get(k, 0)

    print("\n处理完成：")
    print(f"  train: {total['train']} blocks")
    print(f"  val:   {total['val']} blocks")
    print(f"  test:  {total['test']} blocks")
    print(f"  skipped: {total['skipped']} blocks")
    print(f"配置文件已保存: {cfg_path}")


if __name__ == "__main__":
    main()
