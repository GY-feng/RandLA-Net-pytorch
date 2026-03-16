import argparse
from pathlib import Path
import json
import re
import os

try:
    import yaml
except Exception:
    print("Missing dependency: PyYAML. Please install with: pip install pyyaml")
    raise


def _score_name(name: str, prefer_non_repeat: bool, prefer_no_suffix_digit: bool) -> int:
    score = 0
    if prefer_non_repeat and ("重复" in name):
        score += 10
    if prefer_no_suffix_digit and re.search(r"(?:_|-|\s)(\d+)$", name):
        score += 3
    return score


def _is_target_dir(path: Path, target_parts) -> bool:
    parts = [p.lower() for p in path.parts]
    tail = [p.lower() for p in target_parts]
    if len(parts) < len(tail):
        return False
    return parts[-len(tail):] == tail


def main():
    parser = argparse.ArgumentParser(description="Step1: scan and dedupe by route id.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scan_cfg = cfg.get("scan", {})
    dedupe_cfg = cfg.get("dedupe", {})

    root_dir = Path(scan_cfg.get("root_dir", ""))
    target_rel = scan_cfg.get("target_relpath", "lidars/terra_las/cloud_merged.las")
    out_json = Path(scan_cfg.get("output_json", "scan_list.json"))

    route_regex = dedupe_cfg.get("route_regex", r"K\d{1,4}-\d{1,4}-\d{1,4}")
    prefer_non_repeat = bool(dedupe_cfg.get("prefer_non_repeat", True))
    prefer_no_suffix_digit = bool(dedupe_cfg.get("prefer_no_suffix_digit", True))

    if not root_dir.exists():
        raise FileNotFoundError(f"Scan root not found: {root_dir}")

    target_parts = Path(target_rel).parts
    target_parent_parts = target_parts[:-1]
    target_filename = target_parts[-1]

    # Faster scan: only walk directories and check for target parent path
    # Instead of rglob("*.las"), which enumerates all LAS files.
    candidates = []
    for dirpath, dirnames, filenames in os.walk(str(root_dir)):
        if not dirnames and not filenames:
            continue
        p = Path(dirpath)
        if _is_target_dir(p, target_parent_parts):
            if target_filename in filenames:
                candidates.append(p / target_filename)

    total = len(candidates)
    items = []
    for idx, p in enumerate(candidates, start=1):
        print(f"正在处理第 {idx}/{total} 个文件: {p.name}")
        try:
            name = p.parents[len(target_parent_parts) - 1].name
        except Exception:
            name = p.parent.name
        items.append(
            {
                "name": name,
                "las_path": str(p),
                "selected": True,
            }
        )

    # dedupe by route id
    grouped = {}
    for it in items:
        name = it.get("name", "")
        m = re.search(route_regex, name)
        route_id = m.group(0) if m else name
        it["route_id"] = route_id
        grouped.setdefault(route_id, []).append(it)

    kept = 0
    for route_id, group in grouped.items():
        if len(group) == 1:
            group[0]["selected"] = True
            kept += 1
            continue

        scored = []
        for it in group:
            name = it.get("name", "")
            score = _score_name(name, prefer_non_repeat, prefer_no_suffix_digit)
            scored.append((score, len(name), it))
        scored.sort(key=lambda x: (x[0], x[1]))
        best = scored[0][2]

        for _, _, it in scored:
            if it is best:
                it["selected"] = True
                it["duplicate_of"] = None
            else:
                it["selected"] = False
                it["duplicate_of"] = best.get("name")
        kept += 1

    payload = {
        "root_dir": str(root_dir),
        "target_relpath": target_rel,
        "items": items,
        "summary": {
            "total": len(items),
            "unique_routes": len(grouped),
            "kept": kept,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Found {len(items)} LAS files.")
    print(f"Unique routes: {len(grouped)}")
    print(f"List saved to: {out_json}")


if __name__ == "__main__":
    main()
