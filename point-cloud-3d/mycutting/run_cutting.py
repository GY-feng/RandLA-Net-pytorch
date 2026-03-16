from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from polygon_io import collect_json_index, pick_json_for_key, load_polygons_from_json
from geometry import points_in_polygons
from csf_ground import csf_ground_indices
from las_io import read_las, get_xy, get_xyz, write_las_subset


def _match_pairs(
    las_dir: Path,
    json_dir: Path,
    *,
    prefer_geo: bool = True,
) -> List[Tuple[Path, Path, str, str]]:
    """
    Returns list of (json_path, las_path, key, coord_type).
    Matching rule: key contained in LAS filename.
    """
    index = collect_json_index(json_dir)
    las_files = sorted(las_dir.glob("*.las"))
    pairs: List[Tuple[Path, Path, str, str]] = []

    for key in sorted(index.keys()):
        json_path, coord_type = pick_json_for_key(index, key, prefer_geo=prefer_geo)
        if json_path is None:
            continue
        matches = [p for p in las_files if key in p.stem]
        if not matches:
            print(f"[WARN] No LAS matched for key: {key}")
            continue
        if len(matches) > 1:
            print(f"[WARN] Multiple LAS matched for key {key}, using first: {matches[0].name}")
        pairs.append((json_path, matches[0], key, coord_type))

    return pairs


def _cut_one(
    las_path: Path,
    json_path: Path,
    *,
    coord_type: str,
    out_seg_dir: Path,
) -> Tuple[Path, int, int]:
    las = read_las(las_path)
    xy = get_xy(las)
    bbox = (float(xy[:, 0].min()), float(xy[:, 0].max()), float(xy[:, 1].min()), float(xy[:, 1].max()))

    polygons = load_polygons_from_json(json_path, coord_type=coord_type, bbox=bbox if coord_type == "percent" else None)
    if not polygons:
        raise RuntimeError(f"No polygons found in {json_path}")

    mask = points_in_polygons(xy, polygons)
    out_path = out_seg_dir / f"{las_path.stem}_seg.las"
    new_las = write_las_subset(las, mask, out_path)
    return out_path, len(las.points), len(new_las.points)


def _csf_one(
    las_path: Path,
    *,
    out_ground_dir: Path,
    csf_params: dict,
) -> Tuple[Path, int, int]:
    las = read_las(las_path)
    xyz = get_xyz(las)
    ground_idx = csf_ground_indices(xyz, **csf_params)
    mask = np.zeros(len(las.points), dtype=bool)
    mask[ground_idx] = True
    out_path = out_ground_dir / f"{las_path.stem}_ground.las"
    new_las = write_las_subset(las, mask, out_path)
    return out_path, len(las.points), len(new_las.points)


def main():
    parser = argparse.ArgumentParser(description="Cut point clouds by jurisdiction JSON and extract ground by CSF.")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--las_dir", type=str, default="")
    parser.add_argument("--json_dir", type=str, default="")
    parser.add_argument("--las_path", type=str, default="")
    parser.add_argument("--json_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--prefer_geo", action="store_true", default=True)

    # CSF params
    parser.add_argument("--bSloopSmooth", action="store_true", default=False)
    parser.add_argument("--time_step", type=float, default=0.65)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--cloth_resolution", type=float, default=1.0)
    parser.add_argument("--rigidness", type=int, default=3)
    parser.add_argument("--interation", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_seg_dir = out_dir / "seg"
    out_ground_dir = out_dir / "ground"

    csf_params = dict(
        bSloopSmooth=args.bSloopSmooth,
        time_step=args.time_step,
        class_threshold=args.class_threshold,
        cloth_resolution=args.cloth_resolution,
        rigidness=args.rigidness,
        interation=args.interation,
    )

    if args.mode == "single":
        if not args.las_path or not args.json_path:
            raise ValueError("single mode requires --las_path and --json_path")

        las_path = Path(args.las_path)
        json_path = Path(args.json_path)
        coord_type = "geo"
        if "百分比坐标" in json_path.stem:
            coord_type = "percent"

        start = time.perf_counter()
        seg_path, total_pts, seg_pts = _cut_one(las_path, json_path, coord_type=coord_type, out_seg_dir=out_seg_dir)
        print(f"[Cut] {las_path.name} -> {seg_path.name} points {seg_pts}/{total_pts} time {time.perf_counter()-start:.2f}s")

        start = time.perf_counter()
        ground_path, seg_total, ground_pts = _csf_one(seg_path, out_ground_dir=out_ground_dir, csf_params=csf_params)
        print(f"[CSF] {seg_path.name} -> {ground_path.name} points {ground_pts}/{seg_total} time {time.perf_counter()-start:.2f}s")
        return

    # batch mode
    las_dir = Path(args.las_dir) if args.las_dir else None
    json_dir = Path(args.json_dir) if args.json_dir else None
    if las_dir is None or json_dir is None:
        raise ValueError("batch mode requires --las_dir and --json_dir")
    if not las_dir.exists():
        raise FileNotFoundError(f"LAS dir not found: {las_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON dir not found: {json_dir}")

    pairs = _match_pairs(las_dir, json_dir, prefer_geo=args.prefer_geo)
    if not pairs:
        raise RuntimeError("No matched pairs found.")

    cut_bar = tqdm(total=len(pairs), desc="[Cut]", position=0)
    csf_bar = tqdm(total=len(pairs), desc="[CSF]", position=1)

    seg_paths: List[Tuple[Path, str]] = []

    for json_path, las_path, key, coord_type in pairs:
        cut_bar.set_postfix(file=key)
        t0 = time.perf_counter()
        seg_path, total_pts, seg_pts = _cut_one(las_path, json_path, coord_type=coord_type, out_seg_dir=out_seg_dir)
        t1 = time.perf_counter()
        print(f"[Cut] {key}: {seg_pts}/{total_pts} points, {t1 - t0:.2f}s -> {seg_path.name}")
        seg_paths.append((seg_path, key))
        cut_bar.update(1)

    for seg_path, key in seg_paths:
        csf_bar.set_postfix(file=key)
        t0 = time.perf_counter()
        ground_path, seg_total, ground_pts = _csf_one(seg_path, out_ground_dir=out_ground_dir, csf_params=csf_params)
        t1 = time.perf_counter()
        print(f"[CSF] {key}: {ground_pts}/{seg_total} points, {t1 - t0:.2f}s -> {ground_path.name}")
        csf_bar.update(1)

    cut_bar.close()
    csf_bar.close()


if __name__ == "__main__":
    main()
