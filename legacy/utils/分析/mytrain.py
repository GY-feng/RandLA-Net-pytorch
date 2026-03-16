# mytrain.py (重构版)
"""
Single-entry training script.
Run:
    python mytrain.py

All settings are in the CONFIG block below.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# project modules (assume your project structure)
from data import data_loaders
from model import RandLANet

# ----------------------- CONFIG (修改这里) -----------------------
CONFIG = {
    # Paths
    "dataset_root": Path("datasets/drone_highway"),
    "train_dir": "train",
    "val_dir": "val",
    "logs_dir": Path("runs"),
    # experiment name (None -> use timestamp)
    "exp_name": None,  

    # Training params
    "epochs": 50,
    "batch_size": 6,
    "num_workers": 8,
    "adam_lr": 1e-3,# 5e-4,
    "scheduler_gamma": 0.98,
    "save_freq": 5,  # save checkpoint every n epochs

    # Model params
    "neighbors": 8,# 这个地方，若n=16，配合n_points=131072,则运行速度要去到9800s/epoch（2.7h），这个地方为8，配合n_points=131072，一轮5000s
    "decimation": 4,

    # Device
    "use_cuda": True,  # if True and cuda available -> use GPU

    # Misc
    "max_class_weight": 5.0,   # clamp max class weight
    "class_weight_smooth_eps": 0.02,  # +epsilon in denominator for stability
    "dataset_sampling": "active_learning",  # forwarded to data_loaders
}
# -----------------------------------------------------------------

def compute_class_weights(loader, num_classes, device, smooth_eps=0.02, max_weight=5.0):
    """
    Scan loader to compute class weights as torch tensor on `device`.
    weights = 1 / (ratio + smooth_eps), clamped to max_weight.
    """
    print('正在统计训练集类别分布以计算权重 (这可能需要几分钟)...')
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for _, _, labels in tqdm(loader, desc="统计类别", leave=False):
        # labels shape: [B, N]
        labels_np = labels.view(-1).cpu().numpy()
        counts = np.bincount(labels_np.astype(np.int64), minlength=num_classes)
        class_counts += counts

    print(f"类别统计结果: {class_counts.astype(int)}")

    class_counts = class_counts.astype(np.float32) + 1e-6
    n_samples = torch.tensor(class_counts, dtype=torch.float32, device=device)
    ratio_samples = n_samples / n_samples.sum()

    weights = 1.0 / (ratio_samples + smooth_eps)
    weights = torch.clamp(weights, max=max_weight)

    return weights

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    losses = []

    # accumulators
    total_correct = 0
    total_seen = 0
    total_seen_class = np.zeros(num_classes, dtype=np.int64)
    total_correct_class = np.zeros(num_classes, dtype=np.int64)
    total_union_class = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for points, features, labels in tqdm(loader, desc='Validation', leave=False):
            points = points.to(device)
            features = features.to(device)
            labels = labels.to(device)  # [B, N]

            input_tensor = torch.cat([points, features], dim=-1)  # [B, N, d_in]
            scores = model(input_tensor)  # could be (B,C,N) or (B,N,C)

            # normalize to (B, C, N)
            if scores.dim() == 3 and scores.shape[1] != num_classes:
                # likely (B, N, C)
                scores = scores.transpose(1, 2)

            loss = criterion(scores, labels)
            losses.append(loss.item())

            # predictions: take argmax along classes
            # scores shape: (B, C, N) -> pred shape (B, N)
            preds = scores.argmax(dim=1)

            # compute totals
            correct = (preds == labels).long().sum().item()
            total_correct += correct
            total_seen += labels.numel()

            # per-class stats
            preds_np = preds.cpu().numpy().reshape(-1)
            labels_np = labels.cpu().numpy().reshape(-1)
            for c in range(num_classes):
                gt_mask = (labels_np == c)
                pred_mask = (preds_np == c)
                total_seen_class[c] += int(gt_mask.sum())
                total_correct_class[c] += int(((labels_np == c) & (preds_np == c)).sum())
                total_union_class[c] += int(((gt_mask) | (pred_mask)).sum())

    mean_loss = float(np.mean(losses)) if len(losses) > 0 else float('nan')
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
    # Unpack config
    dataset_root = Path(cfg["dataset_root"])
    train_dir = cfg["train_dir"]
    val_dir = cfg["val_dir"]
    logs_dir = Path(cfg["logs_dir"])
    name = cfg["exp_name"] or datetime.now().strftime('%Y-%m-%d_%H-%M')
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    adam_lr = float(cfg["adam_lr"])
    scheduler_gamma = float(cfg["scheduler_gamma"])
    neighbors = int(cfg["neighbors"])
    decimation = int(cfg["decimation"])
    save_freq = int(cfg["save_freq"])
    use_cuda = bool(cfg["use_cuda"])
    dataset_sampling = cfg.get("dataset_sampling", "active_learning")
    max_class_weight = float(cfg.get("max_class_weight", 5.0))
    smooth_eps = float(cfg.get("class_weight_smooth_eps", 0.02))

    # device
    device = torch.device('cuda:0' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    print("Device:", device)

    # prepare dirs
    exp_logs = logs_dir / name
    exp_logs.mkdir(parents=True, exist_ok=True)

    # classes.json auto create
    classes_file = dataset_root / 'classes.json'
    if not classes_file.exists():
        default_classes = {
            "0": "Background",
            "1": "Defect_1",
            "2": "Defect_2",
            "3": "Defect_3",
            "4": "Defect_4"
        }
        with open(classes_file, 'w') as f:
            json.dump(default_classes, f, indent=4)
        print(f"已生成: {classes_file}")

    with open(classes_file) as f:
        class_names = json.load(f)
    num_classes = len(class_names.keys())
    print(f"检测到 {num_classes} 个类别。")

    # Data loaders
    print("Preparing data loaders...")
    train_loader, val_loader = data_loaders(
        dataset_root,
        dataset_sampling,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    # auto detect d_in
    try:
        sample_batch = next(iter(train_loader))
        pts, fts, lbs = sample_batch
        d_in = pts.size(-1) + fts.size(-1)
        print(f"自动检测输入特征维度 d_in: {d_in}")
    except StopIteration:
        raise RuntimeError("训练集为空，请检查数据路径或预处理脚本。")

    # model init
    model = RandLANet(
        d_in,
        num_classes,
        num_neighbors=neighbors,
        decimation=decimation,
        device=device
    )
    model.to(device)

    # compute class weights
    weights = compute_class_weights(train_loader, num_classes, device, smooth_eps=smooth_eps, max_weight=max_class_weight)
    print(f'计算得出的类别权重: {weights.cpu().numpy()}')
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    # try to load latest checkpoint in logs_dir/name if exists (optional)
    ckpt_list = sorted(list(exp_logs.glob('checkpoint_*.pth')))
    first_epoch = 1
    if len(ckpt_list) > 0:
        # by default, don't auto-load; you can uncomment to auto resume from latest
        # latest = ckpt_list[-1]
        # print(f'自动加载检查点 {latest} ...')
        # ckpt = torch.load(latest, map_location='cpu')
        # model.load_state_dict(ckpt['model_state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        pass

    # training loop
    with SummaryWriter(exp_logs) as writer:
        for epoch in range(first_epoch, epochs + 1):
            print(f'\n=== EPOCH {epoch}/{epochs} ===')
            t0 = time.time()
            model.train()
            train_losses = []

            # debug model sanity check: run a single forward on sample batch (no grad)
            try:
                pts_dbg, fts_dbg, _ = sample_batch
                debug_input = torch.cat([pts_dbg.to(device), fts_dbg.to(device)], dim=-1)
                model.eval()
                with torch.no_grad():
                    dbg_scores = model(debug_input[:1])
                    print("DEBUG scores shape:", dbg_scores.shape,
                          "min/max:", float(dbg_scores.min().item()), float(dbg_scores.max().item()))
                model.train()
            except Exception as e:
                print("DEBUG forward failed:", e)
                model.train()

            pbar = tqdm(train_loader, desc='Training', leave=False)
            for points, features, labels in pbar:
                points = points.to(device)
                features = features.to(device)
                labels = labels.long().to(device)

                input_tensor = torch.cat([points, features], dim=-1)  # [B,N,d_in]
                optimizer.zero_grad()
                scores = model(input_tensor)  # may be (B,C,N) or (B,N,C)

                if scores.dim() == 3 and scores.shape[1] != num_classes:
                    scores = scores.transpose(1, 2)  # make (B,C,N)

                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            scheduler.step()

            # validation
            val_loss, val_OA, val_ious = evaluate(model, val_loader, criterion, device, num_classes)

            mean_train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else float('nan')
            mean_val_iou = float(np.nanmean(val_ious))

            t1 = time.time()
            print(f'Time: {t1 - t0:.1f}s')
            print(f'Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'Val OA: {val_OA:.4f} | Val mIoU: {mean_val_iou:.4f}')

            print('Per-Class IoU:')
            for i, iou in enumerate(val_ious):
                c_name = class_names.get(str(i), str(i))
                print(f'  {c_name}: {np.nan_to_num(iou):.4f}')

            # tensorboard logging
            writer.add_scalar('Loss/train', mean_train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metric/OA', val_OA, epoch)
            writer.add_scalar('Metric/mIoU', mean_val_iou, epoch)
            for i, iou in enumerate(val_ious):
                writer.add_scalar(f'Class_IoU/{i}', float(np.nan_to_num(iou)), epoch)

            # save
            if epoch % save_freq == 0 or epoch == epochs:
                save_path = exp_logs / f'checkpoint_{epoch:02d}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, save_path)
                print(f'Checkpoint saved to {save_path}')

    print("Training finished.")

if __name__ == "__main__":
    # run
    print("Using configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    train(CONFIG)


# import argparse
# from datetime import datetime
# import json
# import numpy as np
# from pathlib import Path
# import time
# from tqdm import tqdm
# import warnings
# import sys

# import torch
# import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

# # 引入项目模块
# from data import data_loaders
# from model import RandLANet
# from utils.metrics import accuracy, intersection_over_union
# from utils.tools import Config as cfg

# def delete_compute_class_weights(loader, num_classes, device):
#     """
#     自动扫描训练集以计算类别权重。
#     解决类别不平衡问题（例如背景点多，缺陷点少）。

#     用途：遍历训练集统计每个类别出现次数并计算类别权重以缓解类别不平衡。
#     实现要点：

#     用 np.bincount 聚合每个 batch 的标签计数（先展平 labels）。

#     防止零计数后加 1e-6，然后计算样本比例 ratio_samples。

#     权重公式：weights = 1 / (ratio_samples + 0.02)（低频类别权重大）。
#     返回：一个在 device 上的 torch.tensor 权重向量（可直接传给 nn.CrossEntropyLoss(weight=...)）。

#     注意：遍历整个 loader 会耗时间（脚本有提示）。
#     """
#     print('正在统计训练集类别分布以计算权重 (这可能需要几分钟)...')
#     class_counts = np.zeros(num_classes, dtype=np.float32)
    
#     # 只需要遍历一遍 loader
#     for _, _, labels in tqdm(loader, desc="统计类别", leave=False):
#         # labels: [Batch, N]
#         labels = labels.view(-1).cpu().numpy()
#         counts = np.bincount(labels.astype(np.int32), minlength=num_classes)
#         class_counts += counts

#     print(f"类别统计结果: {class_counts.astype(int)}")
    
#     # 防止除以0
#     class_counts += 1e-6 
    
#     # 核心公式：频率越低，权重越高
#     # 1 / (log(1.2 + probability)) 是另一种常用变体，这里沿用原项目的平滑倒数法
#     n_samples = torch.tensor(class_counts, dtype=torch.float, device=device)
#     ratio_samples = n_samples / n_samples.sum()
#     # weights = 1 / (ratio_samples + 0.02)

#     weights = torch.tensor(weights, device=device)
#     weights = torch.clamp(weights, max=5.0)  # 不要超过5
    
#     return weights

# def compute_class_weights(loader, num_classes, device):
#     """
#     自动扫描训练集以计算类别权重（返回 torch.tensor, 放在 device 上）。
#     使用平滑倒数并限制最大权重，避免极端权值导致训练不稳定。
#     """
#     import numpy as np
#     import torch

#     print('正在统计训练集类别分布以计算权重 (这可能需要几分钟)...')
#     class_counts = np.zeros(num_classes, dtype=np.float64)

#     # 遍历 loader，累加每个 batch 的标签计数
#     for _, _, labels in tqdm(loader, desc="统计类别", leave=False):
#         labels_np = labels.view(-1).cpu().numpy()
#         counts = np.bincount(labels_np.astype(np.int64), minlength=num_classes)
#         class_counts += counts

#     print(f"类别统计结果: {class_counts.astype(int)}")

#     # 防止除以0
#     class_counts = class_counts.astype(np.float32) + 1e-6

#     # 转为 torch tensor，放到指定 device
#     n_samples = torch.tensor(class_counts, dtype=torch.float32, device=device)
#     ratio_samples = n_samples / n_samples.sum()

#     # ===== 主公式（平滑倒数） =====
#     weights = 1.0 / (ratio_samples + 0.02)

#     # ===== 可选的平滑替代（更温和） =====
#     # 下面是备用公式，如果你想更温和地放大稀有类，可用它：
#     # weights = 1.0 / (torch.log(1.02 + ratio_samples))

#     # 限制最大权重，避免极端值导致训练不稳定
#     weights = torch.clamp(weights, max=5.0)

#     # 最终返回 device 上的 float tensor
#     return weights

# def evaluate(model, loader, criterion, device, num_classes):
#     model.eval()
#     losses = []
#     # 初始化累加器
#     total_correct = 0
#     total_seen = 0
#     total_seen_class = [0] * num_classes
#     total_correct_class = [0] * num_classes
#     total_union_class = [0] * num_classes

#     with torch.no_grad():
#         for points, features, labels in tqdm(loader, desc='Validation', leave=False):
#             points = points.to(device)
#             features = features.to(device)
#             labels = labels.to(device)
            
#             input_tensor = torch.cat([points, features], dim=-1)
#             scores = model(input_tensor)

#             loss = criterion(scores, labels)
#             losses.append(loss.item())
            
#             # 计算指标所需数据
#             pred_choice = scores.max(1)[1]
#             correct = pred_choice.eq(labels.long()).cpu().sum()
#             total_correct += correct.item()
#             total_seen += points.shape[0] * points.shape[1] # Batch * N

#             # 逐类 IoU 计算
#             for l in range(num_classes):
#                 total_seen_class[l] += ((labels == l).sum().item())
#                 total_correct_class[l] += ((pred_choice == l) & (labels == l)).sum().item()
#                 total_union_class[l] += (((pred_choice == l) | (labels == l)).sum().item())

#     # 计算最终指标
#     mean_loss = np.mean(losses)
#     OA = total_correct / float(total_seen)
    
#     # 计算 mIoU
#     ious = []
#     for l in range(num_classes):
#         denom = total_union_class[l]
#         if denom == 0:
#             ious.append(np.nan)
#         else:
#             ious.append(total_correct_class[l] / float(denom))
    
#     return mean_loss, OA, np.array(ious)

# def train(args):
#     # 1. 路径与目录设置
#     train_path = args.dataset / args.train_dir
#     val_path = args.dataset / args.val_dir
#     logs_dir = args.logs_dir / args.name
#     logs_dir.mkdir(exist_ok=True, parents=True)

#     # 2. 类别管理
#     # 尝试自动生成 classes.json 如果不存在
#     classes_file = args.dataset / 'classes.json'
#     if not classes_file.exists():
#         print("未找到 classes.json，正在为您生成默认配置...")
#         default_classes = {
#             "0": "Background",
#             "1": "Defect_1",
#             "2": "Defect_2",
#             "3": "Defect_3",
#             "4": "Defect_4"
#         }
#         with open(classes_file, 'w') as f:
#             json.dump(default_classes, f, indent=4)
#         print(f"已生成: {classes_file}")

#     with open(classes_file) as f:
#         class_names = json.load(f)
#         num_classes = len(class_names.keys())

#     print(f"检测到 {num_classes} 个类别。")

#     # 3. 数据加载
#     # 注意：pin_memory=True 加速 CPU->GPU 传输
#     train_loader, val_loader = data_loaders(
#         args.dataset,
#         args.dataset_sampling,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True 
#     )

#     # 自动获取输入维度 (适配你的 d_in=9)
#     try:
#         sample_batch = next(iter(train_loader))
#         pts, fts, lbs = sample_batch
#         # RandLA-Net 的输入通常是 [Batch, N, D]，这里 D = pts(3) + fts(6) = 9
#         d_in = pts.size(-1) + fts.size(-1) 
#         print(f"自动检测输入特征维度 d_in: {d_in}")
#     except StopIteration:
#         print("错误: 数据集似乎是空的，请检查路径。")
#         return

#     # 4. 模型初始化
#     model = RandLANet(
#         d_in,
#         num_classes,
#         num_neighbors=args.neighbors,
#         decimation=args.decimation,
#         device=args.gpu
#     )

#     # 5. 权重计算 (自动适配你的数据)
#     weights = compute_class_weights(train_loader, num_classes, args.gpu)
#     print(f'计算得出的类别权重: {weights.cpu().numpy()}')
    
#     criterion = nn.CrossEntropyLoss(weight=weights)

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.scheduler_gamma)

#     # 6. 断点续训逻辑
#     first_epoch = 1
#     if args.load:
#         path = max(list((args.logs_dir / args.load).glob('*.pth')))
#         print(f'正在加载检查点 {path}...')
#         checkpoint = torch.load(path)
#         first_epoch = checkpoint['epoch'] + 1
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

#     # 7. 训练主循环
#     with SummaryWriter(logs_dir) as writer:
#         for epoch in range(first_epoch, args.epochs + 1):
#             print(f'\n=== EPOCH {epoch:d}/{args.epochs:d} ===')
#             t0 = time.time()
            
#             # --- Train ---
#             model.train()
#             train_losses = []
            
#             # debug check (remove after verifying)
#             # 使用之前取到的 sample_batch（在 d_in 自动检测时已取过）
#             try:
#                 pts_dbg, fts_dbg, lbs_dbg = sample_batch
#             except NameError:
#                 # 如果没有 sample_batch（极少情况），从 train_loader 获取一个batch
#                 sample_iter = iter(train_loader)
#                 pts_dbg, fts_dbg, lbs_dbg = next(sample_iter)

#             pts_dbg = pts_dbg.to(args.gpu)
#             fts_dbg = fts_dbg.to(args.gpu)
#             debug_input = torch.cat([pts_dbg, fts_dbg], dim=-1)
#             model.eval()
#             with torch.no_grad():
#                 test_scores = model(debug_input[:1])
#                 print("DEBUG scores shape:", test_scores.shape,
#                     "min/max:", test_scores.min().item(), test_scores.max().item())
#             model.train()  # 恢复训练模式

#             # 使用 tqdm 显示进度条
#             pbar = tqdm(train_loader, desc='Training', leave=False)
#             for points, features, labels in pbar:
#                 points = points.to(args.gpu)        # [B, N, 3]
#                 features = features.to(args.gpu)    # [B, N, 6]
#                 labels = labels.long().to(args.gpu) # [B, N], long on device

#                 input_tensor = torch.cat([points, features], dim=-1)  # [B, N, d_in]
                
#                 optimizer.zero_grad()
#                 scores = model(input_tensor)  # expected raw logits

#                 # 确保 shape 是 [B, C, N]；若 model 返回 [B, N, C] 就 transpose
#                 if scores.dim() == 3 and scores.shape[1] != num_classes:
#                     scores = scores.transpose(1, 2)

#                 loss = criterion(scores, labels)  # CrossEntropy expects logits [B,C,N] and labels [B,N]
#                 loss.backward()
#                 optimizer.step()

#                 train_losses.append(loss.item())
#                 pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

#             scheduler.step()

#             # --- Validation ---
#             # 每跑完一个 Epoch 验证一次
#             val_loss, val_OA, val_ious = evaluate(
#                 model, val_loader, criterion, args.gpu, num_classes
#             )

#             # --- Logging & Display ---
#             mean_train_loss = np.mean(train_losses)
#             mean_val_iou = np.nanmean(val_ious)
            
#             t1 = time.time()
#             d = t1 - t0

#             print(f'Time: {d:.1f}s')
#             print(f'Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f}')
#             print(f'Val OA: {val_OA:.4f} | Val mIoU: {mean_val_iou:.4f}')
            
#             print('Per-Class IoU:')
#             for i, iou in enumerate(val_ious):
#                 c_name = class_names.get(str(i), str(i))
#                 print(f'  {c_name}: {iou:.4f}')

#             # Tensorboard
#             writer.add_scalar('Loss/train', mean_train_loss, epoch)
#             writer.add_scalar('Loss/val', val_loss, epoch)
#             writer.add_scalar('Metric/OA', val_OA, epoch)
#             writer.add_scalar('Metric/mIoU', mean_val_iou, epoch)
            
#             for i, iou in enumerate(val_ious):
#                 writer.add_scalar(f'Class_IoU/{i}', iou, epoch)

#             # Save Checkpoint
#             if epoch % args.save_freq == 0 or epoch == args.epochs:
#                 save_path = logs_dir / f'checkpoint_{epoch:02d}.pth'
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'scheduler_state_dict': scheduler.state_dict()
#                 }, save_path)
#                 print(f'Checkpoint saved to {save_path}')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
    
#     # 核心路径配置
#     parser.add_argument('--dataset', type=Path, default='datasets/drone_highway', 
#                         help='数据集根目录')
#     parser.add_argument('--logs_dir', type=Path, default='runs', 
#                         help='Tensorboard日志目录')

#     # 训练超参数 (针对 4080s 优化)
#     parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
#     parser.add_argument('--batch_size', type=int, default=6, 
#                         help='Batch Size (根据 tools.py num_points=65536，6-8 是 4080s 的合理范围)')
#     parser.add_argument('--num_workers', type=int, default=8, 
#                         help='数据加载线程数 (64G内存可设为 8-12)')
#     parser.add_argument('--adam_lr', type=float, default=0.01, help='初始学习率')
#     parser.add_argument('--scheduler_gamma', type=float, default=0.95, help='学习率衰减')
    
#     # 模型参数
#     parser.add_argument('--decimation', type=int, default=4, help='下采样倍率')
#     parser.add_argument('--neighbors', type=int, default=16, help='KNN邻居数')
    
#     # 杂项
#     parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
#     parser.add_argument('--name', type=str, default=None, help='实验名称 (默认使用时间戳)')
#     parser.add_argument('--load', type=str, default='', help='加载预训练模型名称 (如 2023-10-xx)')
#     parser.add_argument('--save_freq', type=int, default=5, help='保存频率')
    
#     # 目录结构 (保持默认即可)
#     parser.add_argument('--train_dir', type=str, default='train')
#     parser.add_argument('--val_dir', type=str, default='val')
#     parser.add_argument('--dataset_sampling', type=str, default='active_learning')

#     args = parser.parse_args()

#     # 设备检查
#     if torch.cuda.is_available():
#         args.gpu = torch.device(f'cuda:{args.gpu}')
#         print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         args.gpu = torch.device('cpu')
#         print("警告: 未检测到 GPU，将使用 CPU 训练 (极慢)")

#     if args.name is None:
#         args.name = datetime.now().strftime('%Y-%m-%d_%H-%M')

#     train(args)