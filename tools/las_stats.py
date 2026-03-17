"""
las_stats.py

统计 LAS 原始点云的点数与密度，并输出到终端和 tools/log/*.txt。
Run:
  python tools/las_stats.py --las_dir datasets/SlopeLAS/raw_las
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import laspy
import numpy as np


def _safe_area(v: float) -> float:
    return v if v > 1e-12 else 1e-12


def list_las_files(root: Path, recursive: bool):
    if recursive:
        return sorted(root.rglob("*.las"))
    return sorted(root.glob("*.las"))


def main():
    parser = argparse.ArgumentParser(description="统计 LAS 点数与密度")
    parser.add_argument("--las_dir", default="datasets/SlopeLAS/raw_las")
    parser.add_argument("--recursive", action="store_true", default=False)
    args = parser.parse_args()

    las_dir = Path(args.las_dir)
    if not las_dir.exists():
        raise FileNotFoundError(f"LAS dir not found: {las_dir}")

    files = list_las_files(las_dir, args.recursive)
    if not files:
        raise RuntimeError(f"No .las files found in {las_dir}")

    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"las_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def write(line: str):
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    write(f"LAS dir: {las_dir}")
    write(f"Recursive: {args.recursive}")
    write(f"Total files: {len(files)}")
    write("-" * 80)

    total_points = 0
    densities = []
    points_list = []

    for f in files:
        with laspy.open(f) as reader:
            header = reader.header
            n = int(header.point_count)
            minx, miny, _ = header.mins
            maxx, maxy, _ = header.maxs
            area = _safe_area((maxx - minx)) * _safe_area((maxy - miny))
            density = n / area

        write(f"{f.name} | N={n} | area={area:.2f} m^2 | density={density:.2f} pts/m^2")
        total_points += n
        densities.append(density)
        points_list.append(n)

    write("-" * 80)
    dens_arr = np.array(densities, dtype=np.float64)
    pts_arr = np.array(points_list, dtype=np.int64)
    write(f"Total points: {total_points}")
    write(f"Mean points: {pts_arr.mean():.1f}")
    write(f"Median points: {np.median(pts_arr):.1f}")
    write(f"Min points: {pts_arr.min()}")
    write(f"Max points: {pts_arr.max()}")
    write(f"Mean density: {dens_arr.mean():.2f} pts/m^2")
    write(f"Median density: {np.median(dens_arr):.2f} pts/m^2")
    write(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
