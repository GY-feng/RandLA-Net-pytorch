"""
ground_sim_from_dir.py

Batch pipeline:
1) Read LAS from input dir
2) Keep ground points by classification value
3) Reset classification to background
4) Run defect simulation (same as step3_generate)

Run:
  python ground_sim_from_dir.py --config config/ground_sim.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import laspy

try:
    import yaml
except Exception as e:
    print("Missing dependency: PyYAML. Please install with: pip install pyyaml")
    raise

from app.pipeline import process_one_las
from app.report import build_report, add_file_report, ratio_str
from app.utils import dump_json, ensure_dir, get
from app.labels import reset_classification, count_labels
from app.defects import apply_radial_offset, apply_noise
from app.sampler import compute_bounds, sample_center
from app.utils import rand_range
from app.io import save_las
from app.pipeline import DEFAULTS as PIPE_DEFAULTS


def list_las_files(root: Path, recursive: bool):
    if recursive:
        return sorted(root.rglob("*.las"))
    return sorted(root.glob("*.las"))


def filter_ground_and_reset(las_path: Path, out_dir: Path, ground_class: int, reset_class: int) -> Path | None:
    las = laspy.read(las_path)
    cls = np.asarray(las.classification)
    mask = cls == ground_class

    if mask.sum() == 0:
        return None

    new_las = laspy.LasData(las.header.copy())
    new_las.points = las.points[mask]
    new_las.classification = np.full(len(new_las.points), reset_class, dtype=new_las.classification.dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{las_path.stem}.las"
    new_las.write(out_path)
    return out_path


def filter_ground_and_reset_in_memory(las_path: Path, ground_class: int, reset_class: int):
    las = laspy.read(las_path)
    cls = np.asarray(las.classification)
    mask = cls == ground_class
    if mask.sum() == 0:
        return None
    new_las = laspy.LasData(las.header.copy())
    new_las.points = las.points[mask]
    new_las.classification = np.full(len(new_las.points), reset_class, dtype=new_las.classification.dtype)
    return new_las


def _merge_defaults(cfg):
    merged = {}
    for k, v in PIPE_DEFAULTS.items():
        if isinstance(v, dict):
            merged[k] = {**v, **(cfg.get(k, {}) if isinstance(cfg.get(k), dict) else {})}
        else:
            merged[k] = cfg.get(k, v)
    for k, v in cfg.items():
        if k not in merged:
            merged[k] = v
    return merged


def _pick_defect_type(bump_need, dep_need):
    if bump_need <= 0 and dep_need <= 0:
        return None
    if bump_need <= 0:
        return "depression"
    if dep_need <= 0:
        return "bump"
    return "bump" if bump_need >= dep_need else "depression"


def simulate_on_las(las, out_dir: Path, cfg, *, seed_key: str, source_stem: str):
    cfg = _merge_defaults(cfg)

    label_policy = cfg["label_policy"]
    ratio_cfg = cfg["ratio"]
    defect_cfg = cfg["defect"]
    noise_cfg = cfg.get("noise", {})

    base_seed = int(cfg.get("seed", 2025))
    file_seed = base_seed + (abs(hash(seed_key)) % 100000)
    rng = np.random.RandomState(file_seed)

    total_points = len(las.points)

    if label_policy.get("reset_all", True):
        reset_classification(las, int(label_policy.get("background", 0)))

    apply_noise(las, float(noise_cfg.get("std", 0.0)), rng)

    x = np.asarray(las.x)
    y = np.asarray(las.y)
    x_range, y_range = compute_bounds(x, y, float(defect_cfg.get("margin_ratio", 0.02)))

    abnormal_ratio = float(ratio_cfg.get("abnormal_ratio", 0.05))
    bump_ratio = float(ratio_cfg.get("bump_ratio", 0.6))
    dep_ratio = float(ratio_cfg.get("depression_ratio", 0.4))
    ratio_sum = max(1e-6, bump_ratio + dep_ratio)
    bump_ratio /= ratio_sum
    dep_ratio /= ratio_sum

    target_abnormal = int(total_points * abnormal_ratio)
    target_bump = int(target_abnormal * bump_ratio)
    target_dep = int(target_abnormal * dep_ratio)

    bump_label = int(label_policy.get("bump", 1))
    dep_label = int(label_policy.get("depression", 2))

    existing = []
    defect_count = 0
    bump_points = 0
    dep_points = 0

    max_defects = int(defect_cfg.get("max_defects", 500))
    max_attempts = int(defect_cfg.get("max_attempts", 2000))
    overlap_policy = defect_cfg.get("overlap_policy", "avoid")

    for _ in range(max_attempts):
        abnormal_points = bump_points + dep_points
        if abnormal_points >= target_abnormal or defect_count >= max_defects:
            break

        bump_need = target_bump - bump_points
        dep_need = target_dep - dep_points
        defect_type = _pick_defect_type(bump_need, dep_need)
        if defect_type is None:
            break

        r_min, r_max = defect_cfg.get("radius", [2.0, 4.0])
        radius = rand_range(rng, float(r_min), float(r_max))

        cx, cy = sample_center(
            rng,
            x_range,
            y_range,
            existing,
            radius,
            overlap_policy=overlap_policy,
            max_attempts=max_attempts,
        )
        if cx is None:
            break

        dz_min, dz_max = defect_cfg.get("dz", [0.5, 2.0])
        dz = rand_range(rng, float(dz_min), float(dz_max))

        smooth_type = defect_cfg.get("smooth_type", "gaussian")
        if isinstance(smooth_type, list) and len(smooth_type) > 0:
            smooth_type = smooth_type[int(rng.randint(0, len(smooth_type)))]

        if defect_type == "bump":
            label = bump_label
            dz = abs(dz)
        else:
            label = dep_label
            dz = -abs(dz)

        affected = apply_radial_offset(
            las,
            center_x=cx,
            center_y=cy,
            radius=radius,
            dz=dz,
            smooth_type=smooth_type,
            label_value=label,
        )

        if affected == 0:
            continue

        existing.append((cx, cy, radius))
        defect_count += 1
        if defect_type == "bump":
            bump_points += affected
        else:
            dep_points += affected

    out_dir = Path(out_dir)
    suffix = str(get(cfg, "output.suffix", "_sim"))
    out_path = out_dir / f"{source_stem}{suffix}.las"
    save_las(las, out_path)

    labels_count = count_labels(las, labels=(label_policy.get("background", 0), bump_label, dep_label))
    abnormal_points = labels_count.get(bump_label, 0) + labels_count.get(dep_label, 0)

    return {
        "input": f"<memory:{source_stem}>",
        "output": str(out_path),
        "total_points": int(total_points),
        "defect_count": int(defect_count),
        "bump_points": int(labels_count.get(bump_label, 0)),
        "depression_points": int(labels_count.get(dep_label, 0)),
        "abnormal_points": int(abnormal_points),
        "abnormal_ratio": float(abnormal_points) / float(total_points) if total_points > 0 else 0.0,
    }


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Filter ground points then simulate defects.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "ground_sim.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        # Try resolving relative to this script's directory
        script_dir = Path(__file__).parent
        alt_path = script_dir / cfg_path
        if alt_path.exists():
            cfg_path = alt_path
        else:
            # Common mistake: running inside generate_slope but still prefixing "generate_slope/"
            if len(cfg_path.parts) > 1 and cfg_path.parts[0] == "generate_slope":
                alt_path2 = script_dir / Path(*cfg_path.parts[1:])
                if alt_path2.exists():
                    cfg_path = alt_path2
                else:
                    raise FileNotFoundError(f"Config not found: {cfg_path}")
            else:
                raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_dir = Path(get(cfg, "input.dir", "D:/Feng/myoriginaldatas"))
    recursive = bool(get(cfg, "input.recursive", False))

    use_classification = bool(get(cfg, "ground_filter.use_classification", True))
    ground_class = int(get(cfg, "ground_filter.class_value", 2))
    reset_class = int(get(cfg, "ground_filter.reset_classification", 0))
    ground_out_dir = Path(get(cfg, "ground_filter.output_dir", "D:/Feng/ground_points"))
    save_intermediate = bool(get(cfg, "ground_filter.save_intermediate", True))
    min_points = int(get(cfg, "ground_filter.min_points", 0))

    out_dir = Path(get(cfg, "output.dir", "D:/Feng/sim_output"))
    ensure_dir(out_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    files = list_las_files(input_dir, recursive)
    if len(files) == 0:
        raise RuntimeError("No .las files found.")

    report = build_report()

    print(f"Input dir: {input_dir}")
    print(f"Total files: {len(files)}")

    total = len(files)
    for i, fpath in enumerate(files, start=1):
        print(f"正在处理第 {i}/{total} 个文件: {fpath.name}")

        virtual_ground_path = ground_out_dir / f"{fpath.stem}.las"

        if use_classification:
            if save_intermediate:
                filtered_path = filter_ground_and_reset(
                    fpath,
                    ground_out_dir,
                    ground_class=ground_class,
                    reset_class=reset_class,
                )
                if filtered_path is None:
                    print(f"  skip: no ground points (class={ground_class})")
                    continue
                if min_points > 0:
                    las = laspy.read(filtered_path)
                    if len(las.points) < min_points:
                        print(f"  skip: ground points {len(las.points)} < min_points {min_points}")
                        continue
                item = process_one_las(filtered_path, out_dir, cfg)
            else:
                filtered_las = filter_ground_and_reset_in_memory(
                    fpath,
                    ground_class=ground_class,
                    reset_class=reset_class,
                )
                if filtered_las is None:
                    print(f"  skip: no ground points (class={ground_class})")
                    continue
                if min_points > 0 and len(filtered_las.points) < min_points:
                    print(f"  skip: ground points {len(filtered_las.points)} < min_points {min_points}")
                    continue
                item = simulate_on_las(
                    filtered_las,
                    out_dir,
                    cfg,
                    seed_key=str(virtual_ground_path),
                    source_stem=fpath.stem,
                )
        else:
            # Input LAS already contains only ground points
            las = laspy.read(fpath)
            if min_points > 0 and len(las.points) < min_points:
                print(f"  skip: points {len(las.points)} < min_points {min_points}")
                continue
            item = simulate_on_las(
                las,
                out_dir,
                cfg,
                seed_key=str(fpath),
                source_stem=fpath.stem,
            )
        add_file_report(report, item)
        print(
            f"  defects={item['defect_count']} abnormal={ratio_str(item['abnormal_points'], item['total_points'])} "
            f"-> {Path(item['output']).name}"
        )

    if get(cfg, "logging.save_log", True):
        log_name = get(cfg, "logging.log_name", "run_log.json")
        dump_json(report, out_dir / log_name)
        print(f"Log saved: {out_dir / log_name}")

    print(f"执行完毕，用时 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
