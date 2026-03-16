"""
data.py

XYZ-only 数据加载：每个 .npy 样本格式为 (N, 4): [x, y, z, label]
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CloudDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        num_points: int = 65536,
        epoch_multiplier: int = 1,
        use_mmap: bool = False,
        strict: bool = True,
    ) -> None:
        self.path = Path(root) / split
        self.files = sorted(list(self.path.glob("*.npy")))
        self.split = split
        self.num_points = int(num_points)
        self.epoch_multiplier = int(max(1, epoch_multiplier))
        self.use_mmap = bool(use_mmap)
        self.strict = bool(strict)
        self._warned = False

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {self.path}. Run prepare_slope_las.py first.")

        if self.num_points <= 0:
            raise ValueError("num_points must be > 0")

    def __len__(self) -> int:
        return len(self.files) * self.epoch_multiplier

    def _load_npy(self, file_path: Path) -> np.ndarray:
        if self.use_mmap:
            data = np.load(file_path, mmap_mode="r")
        else:
            data = np.load(file_path)
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 允许 epoch_multiplier 扩充采样
        file_path = self.files[idx % len(self.files)]

        data = self._load_npy(file_path)
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(f"Bad npy shape {data.shape} in {file_path}, expected (N,4)")

        points = data[:, :3].astype(np.float32)
        labels = data[:, 3].astype(np.int64)

        if self.strict:
            lbl_min = int(labels.min()) if labels.size else 0
            lbl_max = int(labels.max()) if labels.size else 0
            if lbl_min < 0 or lbl_max > 2:
                raise ValueError(
                    f"Label out of range in {file_path}: min={lbl_min}, max={lbl_max}. Expected 0/1/2."
                )

        n = points.shape[0]
        if n >= self.num_points:
            choices = np.random.choice(n, self.num_points, replace=False)
        else:
            choices = np.random.choice(n, self.num_points, replace=True)

        pts = torch.from_numpy(points[choices]).float()
        lbs = torch.from_numpy(labels[choices]).long()

        return pts, lbs


def collate_fn(batch):
    points = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return points, labels


def data_loaders(
    dataset_root: Path,
    batch_size: int = 4,
    num_workers: int = 4,
    num_points: int = 65536,
    pin_memory: bool = True,
):
    train_dataset = CloudDataset(dataset_root, "train", num_points=num_points)
    val_dataset = CloudDataset(dataset_root, "val", num_points=num_points)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader
