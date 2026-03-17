"""
1cut_blocks_from_dir.py

Split large LAS into smaller blocks by recursively cutting along the longest
box edge. Supports overlap and min_points filtering.

Run:
  python 1cut_blocks_from_dir.py --config config/cut_blocks.yaml
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import laspy

try:
    import yaml
except Exception as e:
    print("Missing dependency: PyYAML. Please install with: pip install pyyaml")
    raise


@dataclass
class Box:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def width(self) -> float:
        return self.xmax - self.xmin

    def height(self) -> float:
        return self.ymax - self.ymin

    def long_short(self) -> Tuple[float, float]:
        w = self.width()
        h = self.height()
        return (w, h) if w >= h else (h, w)


def list_las_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(root.rglob("*.las"))
    return sorted(root.glob("*.las"))


def compute_bbox(x: np.ndarray, y: np.ndarray) -> Box:
    return Box(float(x.min()), float(x.max()), float(y.min()), float(y.max()))


def should_stop(box: Box, a: float, b: float) -> bool:
    long_side, short_side = box.long_short()
    return (long_side <= a) or (short_side <= b)


def split_box(box: Box, overlap_ratio: float) -> Tuple[Box, Box]:
    w = box.width()
    h = box.height()
    overlap_ratio = max(0.0, min(0.5, overlap_ratio))

    if w >= h:
        mid = (box.xmin + box.xmax) * 0.5
        overlap = w * overlap_ratio
        left = Box(box.xmin, mid + overlap * 0.5, box.ymin, box.ymax)
        right = Box(mid - overlap * 0.5, box.xmax, box.ymin, box.ymax)
    else:
        mid = (box.ymin + box.ymax) * 0.5
        overlap = h * overlap_ratio
        left = Box(box.xmin, box.xmax, box.ymin, mid + overlap * 0.5)
        right = Box(box.xmin, box.xmax, mid - overlap * 0.5, box.ymax)

    return left, right


def recursive_split(box: Box, a: float, b: float, overlap_ratio: float) -> List[Box]:
    # BFS/stack splitting to avoid recursion depth issues
    result: List[Box] = []
    stack: List[Box] = [box]
    while stack:
        cur = stack.pop()
        if should_stop(cur, a, b):
            result.append(cur)
            continue
        left, right = split_box(cur, overlap_ratio)
        # If either child is too small, keep current as leaf (do not split)
        if should_stop(left, a, b) or should_stop(right, a, b):
            result.append(cur)
            continue
        stack.append(left)
        stack.append(right)
    return result


def mask_points(x: np.ndarray, y: np.ndarray, box: Box) -> np.ndarray:
    return (x >= box.xmin) & (x <= box.xmax) & (y >= box.ymin) & (y <= box.ymax)


def save_block(las: laspy.LasData, mask: np.ndarray, out_path: Path) -> int:
    new_las = laspy.LasData(las.header.copy())
    new_las.points = las.points[mask]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_las.write(out_path)
    return len(new_las.points)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Split large LAS into smaller blocks.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "cut_blocks.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        # Try resolving relative to this script
        alt = Path(__file__).parent / cfg_path
        if alt.exists():
            cfg_path = alt
        else:
            raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_dir = Path(cfg.get("input", {}).get("dir", "D:/Feng/myoriginaldatas"))
    recursive = bool(cfg.get("input", {}).get("recursive", False))
    output_dir = Path(cfg.get("output", {}).get("dir", "D:/Feng/cut_blocks"))

    a = float(cfg.get("cut", {}).get("max_long_edge", 100.0))
    b = float(cfg.get("cut", {}).get("min_short_edge", 30.0))
    overlap_ratio = float(cfg.get("cut", {}).get("overlap_ratio", 0.1))
    min_points = int(cfg.get("cut", {}).get("min_points", 0))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    files = list_las_files(input_dir, recursive)
    if not files:
        raise RuntimeError("No .las files found.")

    print(f"Input dir: {input_dir}")
    print(f"Total files: {len(files)}")
    print(f"Split rules: max_long_edge={a}, min_short_edge={b}, overlap_ratio={overlap_ratio}")
    print(f"Min points per block: {min_points}")

    total_blocks = 0
    saved_blocks = 0

    for i, fpath in enumerate(files, start=1):
        print(f"正在处理第 {i}/{len(files)} 个文件: {fpath.name}")
        las = laspy.read(fpath)
        x = np.asarray(las.x)
        y = np.asarray(las.y)

        root_box = compute_bbox(x, y)
        boxes = recursive_split(root_box, a, b, overlap_ratio)

        if not boxes:
            boxes = [root_box]

        kept_any = False
        for idx, box in enumerate(boxes):
            mask = mask_points(x, y, box)
            n = int(mask.sum())
            total_blocks += 1
            if min_points > 0 and n < min_points:
                continue
            out_path = output_dir / f"{fpath.stem}_cut_{idx:05d}.las"
            saved_n = save_block(las, mask, out_path)
            saved_blocks += 1
            kept_any = True
            print(f"  block {idx:05d}: {saved_n} points -> {out_path.name}")

        if not kept_any:
            # If everything was filtered out, save original as a single block
            out_path = output_dir / f"{fpath.stem}_cut_00000.las"
            new_las = laspy.LasData(las.header.copy())
            new_las.points = las.points
            out_path.parent.mkdir(parents=True, exist_ok=True)
            new_las.write(out_path)
            saved_blocks += 1
            print(f"  no blocks kept, saved original -> {out_path.name}")

    print(f"Total candidate blocks: {total_blocks}")
    print(f"Saved blocks: {saved_blocks}")
    print(f"执行完毕，用时 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
