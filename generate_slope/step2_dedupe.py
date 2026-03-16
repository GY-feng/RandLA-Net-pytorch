import argparse
from pathlib import Path
import json
import re

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


def main():
    parser = argparse.ArgumentParser(description="Step2: dedupe by route id.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scan_cfg = cfg.get("scan", {})
    dedupe_cfg = cfg.get("dedupe", {})

    input_json = Path(dedupe_cfg.get("input_json", scan_cfg.get("output_json", "scan_list.json")))
    output_json = Path(dedupe_cfg.get("output_json", "scan_list_dedup.json"))
    route_regex = dedupe_cfg.get("route_regex", r"K\d{1,4}-\d{1,4}-\d{1,4}")
    prefer_non_repeat = bool(dedupe_cfg.get("prefer_non_repeat", True))
    prefer_no_suffix_digit = bool(dedupe_cfg.get("prefer_no_suffix_digit", True))

    if not input_json.exists():
        raise FileNotFoundError(f"Input json not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])

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

        # choose best item by score then shortest name
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

    output_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "input_json": str(input_json),
        "route_regex": route_regex,
        "items": items,
        "summary": {
            "total": len(items),
            "unique_routes": len(grouped),
            "kept": kept,
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Total items: {len(items)}")
    print(f"Unique routes: {len(grouped)}")
    print(f"Output saved: {output_json}")


if __name__ == "__main__":
    main()
