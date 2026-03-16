import argparse
from pathlib import Path
import json

try:
    import yaml
except Exception:
    print("Missing dependency: PyYAML. Please install with: pip install pyyaml")
    raise


def main():
    parser = argparse.ArgumentParser(description="Step1: scan wappe folder and build list.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scan_cfg = cfg.get("scan", {})
    root_dir = Path(scan_cfg.get("root_dir", ""))
    target_rel = scan_cfg.get("target_relpath", "lidars/terra_las/cloud_merged.las")
    out_json = Path(scan_cfg.get("output_json", "scan_list.json"))

    if not root_dir.exists():
        raise FileNotFoundError(f"Scan root not found: {root_dir}")

    items = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        las_path = child / Path(target_rel)
        if las_path.exists():
            items.append(
                {
                    "name": child.name,
                    "las_path": str(las_path),
                    "selected": False,
                }
            )

    payload = {"root_dir": str(root_dir), "target_relpath": target_rel, "items": items}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Found {len(items)} LAS files.")
    print(f"List saved to: {out_json}")


if __name__ == "__main__":
    main()
