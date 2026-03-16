from pathlib import Path
from typing import Dict, Any

import numpy as np

from .io import load_las, save_las
from .labels import reset_classification, count_labels
from .defects import apply_radial_offset, apply_noise
from .sampler import compute_bounds, sample_center
from .utils import get, rand_range


DEFAULTS = {
    "label_policy": {"reset_all": True, "background": 0, "bump": 1, "depression": 2},
    "ratio": {"abnormal_ratio": 0.05, "bump_ratio": 0.6, "depression_ratio": 0.4},
    "defect": {
        "radius": [2.0, 4.0],
        "dz": [0.5, 2.0],
        "smooth_type": "gaussian",
        "margin_ratio": 0.02,
        "overlap_policy": "avoid",
        "max_defects": 500,
        "max_attempts": 2000,
    },
    "noise": {"std": 0.0},
    "seed": 2025,
    "output": {"suffix": "_sim"},
}


def _merge_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = {}
    for k, v in DEFAULTS.items():
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


def process_one_las(las_path: Path, out_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _merge_defaults(cfg)

    label_policy = cfg["label_policy"]
    ratio_cfg = cfg["ratio"]
    defect_cfg = cfg["defect"]
    noise_cfg = cfg.get("noise", {})

    base_seed = int(cfg.get("seed", 2025))
    file_seed = base_seed + (abs(hash(str(las_path))) % 100000)
    rng = np.random.RandomState(file_seed)

    las = load_las(las_path)
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
    out_path = out_dir / f"{las_path.stem}{suffix}{las_path.suffix}"
    save_las(las, out_path)

    labels_count = count_labels(las, labels=(label_policy.get("background", 0), bump_label, dep_label))
    abnormal_points = labels_count.get(bump_label, 0) + labels_count.get(dep_label, 0)

    return {
        "input": str(las_path),
        "output": str(out_path),
        "total_points": int(total_points),
        "defect_count": int(defect_count),
        "bump_points": int(labels_count.get(bump_label, 0)),
        "depression_points": int(labels_count.get(dep_label, 0)),
        "abnormal_points": int(abnormal_points),
        "abnormal_ratio": float(abnormal_points) / float(total_points) if total_points > 0 else 0.0,
    }
