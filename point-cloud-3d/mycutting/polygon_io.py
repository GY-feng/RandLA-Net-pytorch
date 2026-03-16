from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np


def _extract_polygons_from_labelstudio(data: dict) -> List[np.ndarray]:
    polys: List[np.ndarray] = []
    for ann in data.get("annotations", []) or []:
        for res in ann.get("result", []) or []:
            val = res.get("value", {}) or {}
            pts = val.get("points")
            if not pts:
                continue
            arr = np.asarray(pts, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            polys.append(arr)
    return polys


def _convert_percent_to_geo(
    pts_percent: np.ndarray,
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> np.ndarray:
    # Label Studio uses top-left origin for image Y; convert to geo Y with flip
    px = pts_percent[:, 0] / 100.0
    py = pts_percent[:, 1] / 100.0
    x = min_x + (max_x - min_x) * px
    y = max_y - (max_y - min_y) * py
    return np.column_stack([x, y]).astype(np.float64, copy=False)


def load_polygons_from_json(
    json_path: Path,
    *,
    coord_type: str = "geo",
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[np.ndarray]:
    """
    Load polygons from Label Studio JSON.
    coord_type: "geo" | "percent"
    bbox: (min_x, max_x, min_y, max_y) required when coord_type="percent".
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polys = _extract_polygons_from_labelstudio(data)
    if coord_type == "geo":
        return polys

    if coord_type != "percent":
        raise ValueError(f"Unknown coord_type: {coord_type}")
    if bbox is None:
        raise ValueError("bbox is required for percent coordinates")

    min_x, max_x, min_y, max_y = bbox
    out: List[np.ndarray] = []
    for p in polys:
        out.append(_convert_percent_to_geo(p, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y))
    return out


def collect_json_index(json_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Build index: key -> {"geo": Path?, "percent": Path?}
    key: filename without suffix markers.
    """
    index: Dict[str, Dict[str, Path]] = {}
    for p in sorted(json_dir.glob("*.json")):
        name = p.stem
        if name.endswith("_地理坐标"):
            key = name.replace("_地理坐标", "")
            index.setdefault(key, {})["geo"] = p
        elif name.endswith("_百分比坐标"):
            key = name.replace("_百分比坐标", "")
            index.setdefault(key, {})["percent"] = p
        else:
            # unknown type -> treat as geo
            key = name
            index.setdefault(key, {})["geo"] = p
    return index


def pick_json_for_key(index: Dict[str, Dict[str, Path]], key: str, prefer_geo: bool = True) -> Tuple[Optional[Path], str]:
    """
    Return (json_path, coord_type) where coord_type in {"geo","percent"}.
    """
    entry = index.get(key, {})
    if prefer_geo and "geo" in entry:
        return entry["geo"], "geo"
    if "percent" in entry:
        return entry["percent"], "percent"
    if "geo" in entry:
        return entry["geo"], "geo"
    return None, "geo"
