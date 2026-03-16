import argparse
from pathlib import Path
import sys

try:
    import yaml
except Exception as e:
    print("Missing dependency: PyYAML. Please install with: pip install pyyaml")
    raise

from app.pipeline import process_one_las
from app.report import build_report, add_file_report, ratio_str
from app.utils import dump_json, ensure_dir, get


def list_las_files(root: Path, recursive: bool):
    if recursive:
        return sorted(root.rglob("*.las"))
    return sorted(root.glob("*.las"))


def main():
    parser = argparse.ArgumentParser(description="Generate slope defects from LAS using YAML config.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    mode = get(cfg, "input.mode", "single")
    out_dir = Path(get(cfg, "output.dir", "output"))
    ensure_dir(out_dir)

    if mode == "single":
        single_file = get(cfg, "input.single_file")
        if not single_file:
            raise ValueError("input.single_file is required for single mode")
        files = [Path(single_file)]
    else:
        batch_dir = get(cfg, "input.batch_dir")
        if not batch_dir:
            raise ValueError("input.batch_dir is required for batch mode")
        recursive = bool(get(cfg, "input.recursive", False))
        files = list_las_files(Path(batch_dir), recursive)

    if len(files) == 0:
        raise RuntimeError("No .las files found.")

    report = build_report()

    print(f"Mode: {mode}")
    print(f"Total files: {len(files)}")

    total = len(files)
    for i, fpath in enumerate(files, start=1):
        print(f"正在处理第 {i}/{total} 个文件: {fpath.name}")
        item = process_one_las(fpath, out_dir, cfg)
        add_file_report(report, item)
        print(
            f"  defects={item['defect_count']} abnormal={ratio_str(item['abnormal_points'], item['total_points'])} "
            f"-> {Path(item['output']).name}"
        )

    if get(cfg, "logging.save_log", True):
        log_name = get(cfg, "logging.log_name", "run_log.json")
        dump_json(report, out_dir / log_name)
        print(f"Log saved: {out_dir / log_name}")


if __name__ == "__main__":
    main()
