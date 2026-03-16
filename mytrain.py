"""
mytrain.py

XYZ-only RandLA-Net training script for SlopeLAS.
Run:
    python mytrain.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data import data_loaders
from model import RandLANet

# ----------------------- CONFIG -----------------------
CONFIG = {
    # Paths
    "dataset_root": Path("datasets/SlopeLAS"),
    "logs_dir": Path("runs"),
    "exp_name": None,  # None -> timestamp

    # Training params
    "epochs": 20,
    "batch_size": 6,
    "num_workers": 4,
    "num_points": 65536,
    "adam_lr": 5e-4,
    "scheduler_gamma": 0.98,
    "save_freq": 5,

    # Model params
    "neighbors": 32,
    "decimation": 4,

    # Device
    "use_cuda": True,

    # Class weight
    "max_class_weight": 15.0,
    "class_weight_smooth_eps": 0.02,
}
# ------------------------------------------------------


def compute_class_weights(loader, num_classes, device, smooth_eps=0.02, max_weight=5.0):
    """
    Scan loader to compute class weights as torch tensor on `device`.
    weights = 1 / (ratio + smooth_eps), clamped to max_weight.
    """
    print("正在统计训练集类别分布以计算权重 (可能耗时)...")
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for _, labels in tqdm(loader, desc="统计类别", leave=False):
        labels_np = labels.view(-1).cpu().numpy()
        counts = np.bincount(labels_np.astype(np.int64), minlength=num_classes)
        class_counts += counts

    print(f"类别统计: {class_counts.astype(int)}")

    class_counts = class_counts.astype(np.float32) + 1e-6
    n_samples = torch.tensor(class_counts, dtype=torch.float32, device=device)
    ratio_samples = n_samples / n_samples.sum()

    weights = 1.0 / (ratio_samples + smooth_eps)
    weights = torch.clamp(weights, max=max_weight)

    return weights


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    losses = []

    total_correct = 0
    total_seen = 0
    total_seen_class = np.zeros(num_classes, dtype=np.int64)
    total_correct_class = np.zeros(num_classes, dtype=np.int64)
    total_union_class = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Validation", leave=False):
            points = points.to(device)
            labels = labels.to(device)

            scores = model(points)
            if scores.dim() == 3 and scores.shape[1] != num_classes:
                scores = scores.transpose(1, 2)

            loss = criterion(scores, labels)
            losses.append(loss.item())

            preds = scores.argmax(dim=1)
            correct = (preds == labels).long().sum().item()
            total_correct += correct
            total_seen += labels.numel()

            preds_np = preds.cpu().numpy().reshape(-1)
            labels_np = labels.cpu().numpy().reshape(-1)
            for c in range(num_classes):
                gt_mask = labels_np == c
                pred_mask = preds_np == c
                total_seen_class[c] += int(gt_mask.sum())
                total_correct_class[c] += int(((labels_np == c) & (preds_np == c)).sum())
                total_union_class[c] += int(((gt_mask) | (pred_mask)).sum())

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    OA = float(total_correct) / float(total_seen) if total_seen > 0 else 0.0

    ious = []
    for c in range(num_classes):
        denom = total_union_class[c]
        if denom == 0:
            ious.append(np.nan)
        else:
            ious.append(total_correct_class[c] / float(denom))

    return mean_loss, OA, np.array(ious)


def train(cfg):
    dataset_root = Path(cfg["dataset_root"])
    logs_dir = Path(cfg["logs_dir"])
    name = cfg["exp_name"] or datetime.now().strftime("%Y-%m-%d_%H-%M")

    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    num_points = int(cfg["num_points"])
    adam_lr = float(cfg["adam_lr"])
    scheduler_gamma = float(cfg["scheduler_gamma"])
    neighbors = int(cfg["neighbors"])
    decimation = int(cfg["decimation"])
    save_freq = int(cfg["save_freq"])
    use_cuda = bool(cfg["use_cuda"])
    max_class_weight = float(cfg["max_class_weight"])
    smooth_eps = float(cfg["class_weight_smooth_eps"])

    device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    exp_logs = logs_dir / name
    exp_logs.mkdir(parents=True, exist_ok=True)

    classes_file = dataset_root / "classes.json"
    if not classes_file.exists():
        default_classes = {"0": "Background", "1": "Bump", "2": "Depression"}
        with open(classes_file, "w", encoding="utf-8") as f:
            json.dump(default_classes, f, ensure_ascii=False, indent=2)
        print(f"已生成 {classes_file}")

    with open(classes_file, encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names.keys())
    print(f"检测到 {num_classes} 个类别")

    print("Preparing data loaders...")
    train_loader, val_loader = data_loaders(
        dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        num_points=num_points,
        pin_memory=True,
    )

    # auto detect d_in
    sample_batch = next(iter(train_loader))
    pts, lbs = sample_batch
    d_in = pts.size(-1)
    if d_in != 3:
        raise RuntimeError(f"Expected d_in=3, got {d_in}")
    print(f"Input dim d_in: {d_in}")

    model = RandLANet(
        d_in,
        num_classes,
        num_neighbors=neighbors,
        decimation=decimation,
        device=device,
    )
    model.to(device)

    weights = compute_class_weights(
        train_loader, num_classes, device, smooth_eps=smooth_eps, max_weight=max_class_weight
    )
    print(f"Class weights: {weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    # sanity check forward
    try:
        model.eval()
        with torch.no_grad():
            dbg_scores = model(pts[:1].to(device))
            print(
                "DEBUG scores shape:",
                dbg_scores.shape,
                "min/max:",
                float(dbg_scores.min().item()),
                float(dbg_scores.max().item()),
            )
        model.train()
    except Exception as e:
        print("DEBUG forward failed:", e)
        model.train()

    with SummaryWriter(exp_logs) as writer:
        for epoch in range(1, epochs + 1):
            print(f"\n=== EPOCH {epoch}/{epochs} ===")
            t0 = time.time()
            model.train()
            train_losses = []

            pbar = tqdm(train_loader, desc="Training", leave=False)
            for points, labels in pbar:
                points = points.to(device)
                labels = labels.long().to(device)

                optimizer.zero_grad()
                scores = model(points)

                if scores.dim() == 3 and scores.shape[1] != num_classes:
                    scores = scores.transpose(1, 2)

                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            scheduler.step()

            val_loss, val_OA, val_ious = evaluate(model, val_loader, criterion, device, num_classes)
            mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            mean_val_iou = float(np.nanmean(val_ious))

            t1 = time.time()
            print(f"Time: {t1 - t0:.1f}s")
            print(f"Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val OA: {val_OA:.4f} | Val mIoU: {mean_val_iou:.4f}")

            print("Per-Class IoU:")
            for i, iou in enumerate(val_ious):
                c_name = class_names.get(str(i), str(i))
                print(f"  {c_name}: {np.nan_to_num(iou):.4f}")

            writer.add_scalar("Loss/train", mean_train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metric/OA", val_OA, epoch)
            writer.add_scalar("Metric/mIoU", mean_val_iou, epoch)
            for i, iou in enumerate(val_ious):
                writer.add_scalar(f"Class_IoU/{i}", float(np.nan_to_num(iou)), epoch)

            if epoch % save_freq == 0 or epoch == epochs:
                save_path = exp_logs / f"checkpoint_{epoch:02d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    save_path,
                )
                print(f"Checkpoint saved to {save_path}")

    print("Training finished.")


if __name__ == "__main__":
    print("Using configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    train(CONFIG)
