#!/usr/bin/env python3
# grid_cutter_auto.py
import os
import math
import json
import argparse
from datetime import datetime

import numpy as np
import laspy

"""
Grid cutter that auto decides grid resolution based on point cloud size and area.
Class name and basic API kept compatible with original LasGridCutter:
    LasGridCutter(las_path, output_base, x_num=None, y_num=None, ...)
If x_num/y_num omitted -> automatic decision.
"""

class LasGridCutter:
    def __init__(self,
                 las_path: str,
                 output_base: str,
                 x_num: int = None,
                 y_num: int = None,
                 target_points_per_block: int = None,
                 min_points_per_block: int = 50000,
                 max_points_per_block: int = 300000,
                 max_grid_dim: int = 200,
                 timestamp: str = None):
        """
        :param las_path: input LAS file path
        :param output_base: directory in which to create Grid_* folder
        :param x_num, y_num: optional fixed grid dims (int). If None, auto compute.
        :param target_points_per_block: optional preferred target points per block.
               if None will choose geometric mean of min/max (sqrt)
        :param min_points_per_block: lower bound for points per output block (for autosizing)
        :param max_points_per_block: upper bound for points per output block (for autosizing)
        :param max_grid_dim: maximum allowed grid dimension (avoids absurd splits)
        """
        self.las_path = las_path
        self.output_base = output_base
        self.requested_x = x_num
        self.requested_y = y_num
        self.min_ppb = int(min_points_per_block)
        self.max_ppb = int(max_points_per_block)
        self.max_grid_dim = int(max_grid_dim)

        if target_points_per_block is None:
            # geometric mean is a reasonable default center
            self.target_ppb = int(math.sqrt(self.min_ppb * self.max_ppb))
        else:
            self.target_ppb = int(target_points_per_block)

        # Read point cloud header and basic stats first (lazy reading)
        print(f"Reading LAS file: {las_path} ...")
        self.las = laspy.read(las_path)

        # compute bounding and counts
        self.total_points = len(self.las.points)
        # handle header bounds fallback to coordinates
        try:
            x_min, y_min = float(self.las.header.min[0]), float(self.las.header.min[1])
            x_max, y_max = float(self.las.header.max[0]), float(self.las.header.max[1])
        except Exception:
            # fallback if header doesn't have min/max
            coords_x = np.array(self.las.x)
            coords_y = np.array(self.las.y)
            x_min, x_max = float(coords_x.min()), float(coords_x.max())
            y_min, y_max = float(coords_y.min()), float(coords_y.max())

        self.bounds = (x_min, x_max, y_min, y_max)
        self.width = x_max - x_min if (x_max - x_min) > 0 else 1.0
        self.height = y_max - y_min if (y_max - y_min) > 0 else 1.0
        self.aspect = self.width / self.height

        # Decide grid counts if not provided
        if (self.requested_x is None) or (self.requested_y is None):
            self.x_num, self.y_num = self._auto_grid()
            self.auto = True
        else:
            self.x_num, self.y_num = int(self.requested_x), int(self.requested_y)
            self.auto = False

        # create output dir
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_base, f"Grid_{self.x_num}x{self.y_num}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Total points: {self.total_points:,}")
        print(f"Bounds: x[{self.bounds[0]:.3f}, {self.bounds[1]:.3f}] y[{self.bounds[2]:.3f}, {self.bounds[3]:.3f}]")
        if self.auto:
            print(f"Auto decided grid: {self.x_num} x {self.y_num}  (target ~{self.target_ppb:,} pts/block)")
        else:
            print(f"Using provided grid: {self.x_num} x {self.y_num}")

    def _auto_grid(self):
        """
        Auto decide the grid (x_num, y_num) based on total points and bounding box area.
        Strategy:
          - Compute blocks_needed = ceil(total_points / target_ppb)
          - Compute base cols proportional to sqrt(blocks_needed * aspect_ratio)
          - rows = ceil(blocks_needed / cols)
          - clamp to [1, max_grid_dim]
        """
        # guard
        N = max(1, int(self.total_points))
        area = max(1e-6, self.width * self.height)

        # initial blocks needed by target points
        blocks_by_target = max(1, math.ceil(N / float(self.target_ppb)))

        # ensure blocks_by_target is not absurdly huge by bounding target_ppb into min/max
        # if N is tiny, make at least 1 block
        # compute cols attempt using aspect ratio
        cols = int(round(math.sqrt(blocks_by_target * self.aspect)))
        cols = max(1, cols)
        rows = int(math.ceil(blocks_by_target / cols))

        # clamp dims
        cols = min(cols, self.max_grid_dim)
        rows = min(rows, self.max_grid_dim)

        # if after clamp still not enough blocks to achieve target_ppb, adjust target_ppb downward
        blocks_final = cols * rows
        est_ppb = N / float(blocks_final)
        # If est_ppb > max_ppb, we need more blocks -> increase grid dims proportionally
        if est_ppb > self.max_ppb:
            factor = est_ppb / float(self.max_ppb)
            scale = math.sqrt(factor)
            cols = min(self.max_grid_dim, max(1, int(round(cols * scale))))
            rows = min(self.max_grid_dim, max(1, int(round(rows * scale))))
            blocks_final = cols * rows

        # If est_ppb < min_ppb, we can reduce blocks (coarsen)
        est_ppb = N / float(cols * rows)
        if est_ppb < self.min_ppb:
            factor = self.min_ppb / float(est_ppb) if est_ppb > 0 else 1.0
            scale = math.sqrt(factor)
            cols = max(1, int(round(cols / scale)))
            rows = max(1, int(round(rows / scale)))

        # final clamp
        cols = min(max(1, cols), self.max_grid_dim)
        rows = min(max(1, rows), self.max_grid_dim)

        return int(cols), int(rows)

    def cut(self):
        """Perform cutting into grid and write block .las files. Skip empty blocks."""
        x_min, x_max, y_min, y_max = self.bounds
        x_step = (x_max - x_min) / float(self.x_num)
        y_step = (y_max - y_min) / float(self.y_num)

        # Access coords as numpy arrays to avoid creating huge temporary matrices
        lx = np.array(self.las.x)
        ly = np.array(self.las.y)

        blocks_info = []
        block_counter = 0

        for i in range(self.y_num):
            for j in range(self.x_num):
                x_s, x_e = x_min + j * x_step, x_min + (j + 1) * x_step
                y_s, y_e = y_min + i * y_step, y_min + (i + 1) * y_step

                mask = (lx >= x_s) & (lx < x_e) & (ly >= y_s) & (ly < y_e)
                count = int(np.sum(mask))

                if count == 0:
                    # skip empty block
                    continue

                file_name = f"block_{i}_{j}.las"
                out_path = os.path.join(self.output_dir, file_name)

                # create new las with same header but only selected points
                new_header = self.las.header.copy()
                new_las = laspy.LasData(new_header)
                # assign points; laspy supports direct assignment of structured array
                new_las.points = self.las.points[mask]
                new_las.write(out_path)

                blocks_info.append({
                    "file": file_name,
                    "grid_index": [int(i), int(j)],
                    "point_count": int(count),
                    "bounds": {"x": [float(x_s), float(x_e)], "y": [float(y_s), float(y_e)]}
                })
                block_counter += 1
                print(f"Saved: {file_name} (Points: {count:,})")

        # Save config/log
        self._save_json(blocks_info)
        print(f"\nGrid cutting finished. Folder: {self.output_dir}")
        return block_counter

    def _save_json(self, blocks_info):
        config_path = os.path.join(self.output_dir, "cut_config.json")
        log_data = {
            "source_file": self.las_path,
            "grid_m": self.y_num,
            "grid_n": self.x_num,
            "timestamp": datetime.now().isoformat(),
            "total_points": int(self.total_points),
            "target_points_per_block": int(self.target_ppb),
            "min_points_per_block": int(self.min_ppb),
            "max_points_per_block": int(self.max_ppb),
            "blocks": blocks_info
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4)

# Simple CLI for convenience, preserving compatibility with original usage style
def parse_args():
    p = argparse.ArgumentParser(description="Auto grid cutter for LAS files")
    p.add_argument('--las', required=True, help="Input LAS file")
    p.add_argument('--out', required=True, help="Output base directory")
    p.add_argument('--x_num', type=int, default=None, help="Grid X count (optional)")
    p.add_argument('--y_num', type=int, default=None, help="Grid Y count (optional)")
    p.add_argument('--min_ppb', type=int, default=50000, help="Min points per block (auto mode)")
    p.add_argument('--max_ppb', type=int, default=300000, help="Max points per block (auto mode)")
    p.add_argument('--target_ppb', type=int, default=None, help="Target points per block (optional)")
    p.add_argument('--max_grid_dim', type=int, default=200, help="Maximum grid dim (safeguard)")
    p.add_argument('--timestamp', type=str, default=None, help="Force timestamp string for output folder")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cutter = LasGridCutter(
        las_path=args.las,
        output_base=args.out,
        x_num=args.x_num,
        y_num=args.y_num,
        target_points_per_block=args.target_ppb,
        min_points_per_block=args.min_ppb,
        max_points_per_block=args.max_ppb,
        max_grid_dim=args.max_grid_dim,
        timestamp=args.timestamp
    )
    cutter.cut()
