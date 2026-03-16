import argparse
from pathlib import Path
import json

try:
    import yaml
except Exception:
    print("Missing dependency: PyYAML. Please install with: pip install pyyaml")
    raise

import laspy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Step2: filter class=2 points and save.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    step2_cfg = cfg.get("step2", {})
    selected_json = Path(step2_cfg.get("selected_json", "scan_list.json"))
    output_dir = Path(step2_cfg.get("output_dir", cfg.get("output", {}).get("dir", "output")))

    if not selected_json.exists():
        raise FileNotFoundError(f"Selected json not found: {selected_json}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(selected_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    selected_items = [it for it in items if it.get("selected") is True]

    log = {"selected_count": len(selected_items), "items": []}

    total = len(selected_items)
    for idx, it in enumerate(selected_items, start=1):
        name = it.get("name")
        las_path = Path(it.get("las_path", ""))
        print(f"正在处理第 {idx}/{total} 个文件: {name}")
        if not las_path.exists():
            log["items"].append({"name": name, "status": "missing", "las_path": str(las_path)})
            continue

        las = laspy.read(las_path)
        cls = np.asarray(las.classification)
        mask = cls == 2

        if mask.sum() == 0:
            log["items"].append({"name": name, "status": "no_class2", "points": 0})
            continue

        new_las = laspy.LasData(las.header.copy())
        new_las.points = las.points[mask]

        out_path = output_dir / f"{name}.las"
        new_las.write(out_path)

        log["items"].append(
            {
                "name": name,
                "status": "ok",
                "points": int(mask.sum()),
                "output": str(out_path),
            }
        )

    log_path = output_dir / "step2_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(selected_items)} items. Log: {log_path}")


if __name__ == "__main__":
    main()
