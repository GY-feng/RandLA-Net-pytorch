"""
plot_training_metrics.py

Plot training metrics from runs/<exp>/metrics.csv.
Run:
  python tools/plot_training_metrics.py --log_dir runs/2026-03-17_13-03
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_metrics(csv_path: Path):
    epochs = []
    train_loss = []
    val_loss = []
    val_oa = []
    val_miou = []
    ious = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        iou_keys = [k for k in reader.fieldnames if k.startswith("iou_")]
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_oa.append(float(row["val_OA"]))
            val_miou.append(float(row["val_mIoU"]))
            ious.append([float(row[k]) for k in iou_keys])

    return epochs, train_loss, val_loss, val_oa, val_miou, iou_keys, ious


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from metrics.csv")
    parser.add_argument("--log_dir", required=True, help="runs/<exp_name>")
    parser.add_argument("--out", default="training_metrics.png", help="output plot file name")
    parser.add_argument("--no_show", action="store_true", default=False)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    csv_path = log_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from e

    epochs, train_loss, val_loss, val_oa, val_miou, iou_keys, ious = load_metrics(csv_path)

    plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss, label="Val Loss")
    ax1.set_title("Loss (lower is better)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(epochs, val_oa, label="Val OA (higher is better)")
    ax2.plot(epochs, val_miou, label="Val mIoU (higher is better)")

    # Plot per-class IoU if available
    if ious and iou_keys:
        ious_arr = list(zip(*ious))
        for k, vals in zip(iou_keys, ious_arr):
            ax2.plot(epochs, vals, label=f"{k} (higher is better)", alpha=0.6)

    ax2.set_title("Accuracy / IoU (higher is better)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    out_path = log_dir / args.out
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"Plot saved: {out_path}")
    print("Guidance:")
    print("  Loss: lower is better; steady decrease indicates learning.")
    print("  Val OA / mIoU / per-class IoU: higher is better; mIoU > 0.3 usually indicates usable signal.")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
