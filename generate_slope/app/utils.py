import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safe nested getter with dot path.
    """
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def dump_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def rand_range(rng, low: float, high: float) -> float:
    if low > high:
        low, high = high, low
    return float(rng.uniform(low, high))
