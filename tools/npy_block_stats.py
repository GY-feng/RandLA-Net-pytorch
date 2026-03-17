"""
npy_block_stats.py

统计 prepare 之后每个 .npy 的点数量与密度，并输出到终端和 tools/log/*.txt。
Run:
  python tools/npy_block_stats.py --dataset_root datasets/SlopeLAS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np


def _load_block_size(dataset_root: Path) -> float | None:
    cfg_path = dataset_root / "prepare_config.json"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return float(cfg.get("block_size", None))
    except Exception:
        return None


def _iter_npy_files(dataset_root: Path, splits: list[str]):
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] split dir not found: {split_dir}")
            continue
        for f in sorted(split_dir.glob("*.npy")):
            yield split, f


def _safe_area(v: float) -> float:
    return v if v > 1e-12 else 1e-12


def main():
    parser = argparse.ArgumentParser(description="统计 .npy 块点数与密度")
    parser.add_argument("--dataset_root", default="datasets/SlopeLAS")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--block_size", type=float, default=0.0, help="可选，覆盖 prepare_config.json")
    parser.add_argument("--use_mmap", action="store_true", default=False)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    block_size = float(args.block_size) if args.block_size > 0 else _load_block_size(dataset_root)
    if block_size is None:
        print("[WARN] block_size 未找到（prepare_config.json），将只输出归一化密度")

    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"npy_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def write(line: str):
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    total_files = 0
    total_points = 0
    points_list = []

    write(f"Dataset root: {dataset_root}")
    write(f"Splits: {splits}")
    write(f"Block size: {block_size if block_size is not None else 'N/A'}")
    write("-" * 80)

    for split, f in _iter_npy_files(dataset_root, splits):
        data = np.load(f, mmap_mode="r" if args.use_mmap else None)
        if data.ndim != 2 or data.shape[1] != 4:
            write(f"[WARN] Bad shape {data.shape} -> {f}")
            continue

        n = int(data.shape[0])
        x = data[:, 0]
        y = data[:, 1]

        range_x = float(np.max(x) - np.min(x))
        range_y = float(np.max(y) - np.min(y))

        area_norm = _safe_area(range_x) * _safe_area(range_y)
        density_norm = n / area_norm

        if block_size is not None:
            scale = (block_size * 0.5)
            area_bbox = _safe_area(range_x * scale) * _safe_area(range_y * scale)
            density_bbox = n / area_bbox
            density_block = n / _safe_area(block_size * block_size)
            write(
                f"[{split}] {f.name} | N={n} | "
                f"density_bbox={density_bbox:.2f} pts/m^2 | "
                f"density_block={density_block:.2f} pts/m^2"
            )
        else:
            write(f"[{split}] {f.name} | N={n} | density_norm={density_norm:.2f} pts/unit^2")

        total_files += 1
        total_points += n
        points_list.append(n)

    write("-" * 80)
    if total_files == 0:
        write("No .npy files found.")
    else:
        points_arr = np.array(points_list, dtype=np.int64)
        write(f"Total files: {total_files}")
        write(f"Total points: {total_points}")
        write(f"Mean points: {points_arr.mean():.1f}")
        write(f"Median points: {np.median(points_arr):.1f}")
        write(f"Min points: {points_arr.min()}")
        write(f"Max points: {points_arr.max()}")
        write(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
