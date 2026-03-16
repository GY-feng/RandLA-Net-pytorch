"""
overfit_check.py

Overfit a single sample/batch to sanity-check data + model.
Run:
  python overfit_check.py --config config/slope_config.yaml
  python overfit_check.py --npy_path datasets/SlopeLAS/train/xxx.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn

from model import RandLANet

try:
    import yaml
except Exception:
    yaml = None


def _load_yaml_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("Missing dependency: PyYAML. Please install with: pip install pyyaml")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("train", {})


def _merge_cfg(defaults: dict, overrides: dict) -> dict:
    merged = dict(defaults)
    merged.update(overrides or {})
    return merged


DEFAULT_CONFIG = {
    "dataset_root": Path("datasets/SlopeLAS"),
    "num_points": 65536,
    "adam_lr": 5e-4,
    "neighbors": 32,
    "decimation": 4,
    "use_cuda": True,
}


def _pick_npy(dataset_root: Path) -> Path:
    train_dir = dataset_root / "train"
    files = sorted(train_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy found in {train_dir}")
    return files[0]


def _load_single_sample(npy_path: Path, num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(npy_path)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Bad npy shape {data.shape} in {npy_path}, expected (N,4)")
    points = data[:, :3].astype(np.float32)
    labels = data[:, 3].astype(np.int64)

    n = points.shape[0]
    if n >= num_points:
        choices = np.random.choice(n, num_points, replace=False)
    else:
        choices = np.random.choice(n, num_points, replace=True)

    pts = torch.from_numpy(points[choices]).float().unsqueeze(0)  # (1, N, 3)
    lbs = torch.from_numpy(labels[choices]).long().unsqueeze(0)    # (1, N)
    return pts, lbs


def main():
    parser = argparse.ArgumentParser(description="Overfit a single sample to validate pipeline.")
    parser.add_argument("--config", default=str(Path("config/slope_config.yaml")))
    parser.add_argument("--npy_path", default="", help="Optional: path to a specific .npy")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num_points", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--use_class_weights", action="store_true", default=False)
    args = parser.parse_args()

    cfg = _merge_cfg(DEFAULT_CONFIG, _load_yaml_config(Path(args.config)))

    dataset_root = Path(cfg["dataset_root"])
    num_points = int(args.num_points) if args.num_points > 0 else int(cfg["num_points"])
    lr = float(args.lr) if args.lr > 0 else float(cfg["adam_lr"])
    neighbors = int(cfg["neighbors"])
    decimation = int(cfg["decimation"])
    use_cuda = bool(cfg["use_cuda"])

    device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    npy_path = Path(args.npy_path) if args.npy_path else _pick_npy(dataset_root)
    print(f"Using sample: {npy_path}")

    pts, lbs = _load_single_sample(npy_path, num_points)
    pts = pts.to(device)
    lbs = lbs.to(device)

    classes_file = dataset_root / "classes.json"
    if classes_file.exists():
        with open(classes_file, encoding="utf-8") as f:
            class_names = json.load(f)
        num_classes = len(class_names.keys())
    else:
        num_classes = int(lbs.max().item()) + 1

    model = RandLANet(d_in=3, num_classes=num_classes, num_neighbors=neighbors, decimation=decimation, device=device)
    model.to(device)

    if args.use_class_weights:
        # Simple weights from this single batch
        counts = torch.bincount(lbs.view(-1), minlength=num_classes).float().to(device)
        ratio = counts / counts.sum()
        weights = 1.0 / (ratio + 1e-2)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Using class weights: {weights.detach().cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        scores = model(pts)
        if scores.dim() == 3 and scores.shape[1] != num_classes:
            scores = scores.transpose(1, 2)
        loss = criterion(scores, lbs)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = scores.argmax(dim=1)
            acc = (preds == lbs).float().mean().item()

        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"Step {step:04d}/{args.steps} | Loss {loss.item():.4f} | Acc {acc:.4f}")

    t1 = time.time()
    print(f"Done. Time: {t1 - t0:.1f}s")


if __name__ == "__main__":
    main()
