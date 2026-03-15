import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from utils.tools import Config as cfg
import random
import time

class CloudDataset(Dataset):
    def __init__(self, path, split='train', epoch_multiplier=5, use_mmap=False):
        self.path = Path(path) / split
        self.files = sorted(list(self.path.glob('*.npy')))
        self.split = split
        self.use_mmap = use_mmap
        self.epoch_multiplier = epoch_multiplier

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {self.path}. Run prepare_drone_highway.py first.")

        # simple placeholders for future active-learning fields (optional)
        self.possibility = {f: None for f in self.files}
        self.min_possibility = {f: None for f in self.files}

        # ensure cfg has num_points
        try:
            self.num_points = int(cfg.num_points)
            assert self.num_points > 0
        except Exception as e:
            raise RuntimeError("cfg.num_points must be an integer > 0") from e

    def __len__(self):
        return len(self.files) * self.epoch_multiplier

    def __getitem__(self, idx):
        # 如果想完全随机选文件，用 random.choice
        file_path = random.choice(self.files)

        # 使用 mmap 模式可以在文件较大时节约内存（按需读取）
        if self.use_mmap:
            data = np.load(file_path, mmap_mode='r')
        else:
            data = np.load(file_path)

        # 保持和 prepare 保存的格式一致
        points = data[:, :3]
        features = data[:, 3:9]
        labels = data[:, 9].astype(np.int64)

        N = points.shape[0]
        if N >= self.num_points:
            choices = np.random.choice(N, self.num_points, replace=False)
        else:
            choices = np.random.choice(N, self.num_points, replace=True)

        pts = torch.from_numpy(points[choices]).float()
        fts = torch.from_numpy(features[choices]).float()
        lbs = torch.from_numpy(labels[choices]).long()

        return pts, fts, lbs

# class CloudDataset(Dataset):
#     def __init__(self, path, split='train'):
#         self.path = Path(path) / split
#         # 1. 这里的核心修改：寻找 .npy 文件
#         self.files = sorted(list(self.path.glob('*.npy')))
        
#         if len(self.files) == 0:
#             print(f"警告: 在 {self.path} 下没有找到 .npy 文件！")
#             print("请检查 prepare_drone_highway.py 是否成功运行，或路径是否正确。")
#         else:
#             print(f"Size of {split} : {len(self.files)}")

#         self.split = split
#         self.possibility = {}
#         self.min_possibility = {}
        
#         # 初始化采样概率 (Active Learning)
#         for f in self.files:
#             # 这里的 num_points 是你数据里的实际点数，预处理时不定长，
#             # 但为了初始化我们可以先赋一个随机值，或者懒加载。
#             # 这里采用懒加载策略：只记录文件名
#             self.possibility[f] = []
#             self.min_possibility[f] = []

#         # 预加载机制：为了加速，我们可以只维护一个索引列表
#         self.num_points = cfg.num_points

#     def __len__(self):
#         return len(self.files) * 5 # 这里的乘数决定了一个 Epoch 遍历多少次数据块

#     def __getitem__(self, idx):
#         # 随机选择一个文件（Active Learning 策略通常在 Generator 里做，这里简化为随机读取）
#         # 真正的 RandLA-Net 逻辑是在 Generator 里维护 possibility，
#         # 但为了适配你的 .npy 并快速跑通，我们先用纯随机采样测试。
        
#         file_idx = idx % len(self.files)
#         file_path = self.files[file_idx]
        
#         # 2. 这里的核心修改：使用 np.load
#         data = np.load(file_path)
        
#         # [N, 10] -> xyz, feats, label
#         points = data[:, :3]
#         features = data[:, 3:9]
#         labels = data[:, 9].astype(np.int64)

#         # 3. 运行时随机采样 (确保输入点数固定为 cfg.num_points)
#         N = points.shape[0]
#         if N >= self.num_points:
#             choices = np.random.choice(N, self.num_points, replace=False)
#         else:
#             choices = np.random.choice(N, self.num_points, replace=True)

#         return torch.from_numpy(points[choices]).float(), \
#                torch.from_numpy(features[choices]).float(), \
#                torch.from_numpy(labels[choices]).long()

def collate_fn(batch):
    # 简单的拼接
    points = torch.stack([item[0] for item in batch])
    features = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return points, features, labels

def data_loaders(dataset_path, sampling_method='active_learning', batch_size=4, num_workers=8, pin_memory=True):
    # 训练集
    train_dataset = CloudDataset(dataset_path, 'train')
    # 验证集
    val_dataset = CloudDataset(dataset_path, 'val')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )

    return train_loader, val_loader

# import pickle, time, warnings
# import numpy as np
# import torch
# from torch.utils.data import Dataset, IterableDataset, DataLoader
# from utils.tools import Config as cfg
# from utils.tools import DataProcessing as DP
# from sklearn.neighbors import KDTree as SKKDTree

# class CloudsDataset(Dataset):
#     def __init__(self, dir, data_type='ply'):
#         self.path = dir
#         self.paths = list(dir.glob(f'*.{data_type}'))
#         self.size = len(self.paths)
#         self.data_type = data_type
#         self.input_trees = {'training': [], 'validation': []}
#         self.input_colors = {'training': [], 'validation': []}
#         self.input_labels = {'training': [], 'validation': []}
#         self.input_names = {'training': [], 'validation': []}
#         self.val_proj = []
#         self.val_labels = []
        
#         # 验证集划分：文件名 81-100
#         self.val_names = [str(i) for i in range(81, 101)]

#         self.load_data()
#         print('Size of training : ', len(self.input_colors['training']))
#         print('Size of validation : ', len(self.input_colors['validation']))

#     def load_data(self):
#         for i, file_path in enumerate(self.paths):
#             t0 = time.time()
#             cloud_name = file_path.stem

#             kd_tree_file = self.path / '{:s}_KDTree.pkl'.format(cloud_name)
#             proj_file = self.path / '{:s}_proj.pkl'.format(cloud_name)

#             # 1. 加载原始 KDTree
#             with open(kd_tree_file, 'rb') as f:
#                 old_tree = pickle.load(f)
            
#             # 2. 提取点云并处理大地坐标偏移（归心化）
#             # 注意：KDTree.data 返回的是 memoryview，必须转为 numpy array
#             sub_xyz = np.array(old_tree.data, copy=True).astype(np.float32)
#             offset = np.mean(sub_xyz, axis=0)
#             sub_xyz -= offset

#             # 3. 核心修正：由于原树只读，必须使用归心化后的坐标重建新树
#             search_tree = SKKDTree(sub_xyz)

#             # 4. 加载标签
#             with open(proj_file, 'rb') as f:
#                 _, sub_labels = pickle.load(f)

#             # 5. 确保标签和点云数量严格对齐
#             if len(sub_labels) != len(sub_xyz):
#                 min_len = min(len(sub_labels), len(sub_xyz))
#                 sub_labels = sub_labels[:min_len]
#                 sub_xyz = sub_xyz[:min_len]
#                 # 如果发生了长度截断，需要再次重建树以保持索引一致
#                 search_tree = SKKDTree(sub_xyz)
            
#             sub_labels = sub_labels.astype(np.int64) 
            
#             # 6. 特征处理：使用相对 Z 坐标作为初始特征（第一个通道）
#             sub_colors = np.zeros((len(sub_xyz), 3), dtype=np.float32)
#             sub_colors[:, 0] = sub_xyz[:, 2] 

#             # 7. 判定并归类到训练或验证集
#             cloud_split = 'validation' if cloud_name in self.val_names else 'training'
            
#             self.input_trees[cloud_split].append(search_tree)
#             self.input_colors[cloud_split].append(sub_colors)
#             self.input_labels[cloud_split].append(sub_labels)
#             self.input_names[cloud_split].append(cloud_name)
            
#             print('{:s} loaded and centered in {:.1f}s'.format(cloud_name, time.time() - t0))

#     def __len__(self):
#         return self.size

# class ActiveLearningSampler(IterableDataset):
#     def __init__(self, dataset, batch_size=6, split='training'):
#         self.dataset = dataset
#         self.split = split
#         self.batch_size = batch_size
#         self.possibility = {}
#         self.min_possibility = {}

#         if split == 'training':
#             self.n_samples = cfg.train_steps
#         else:
#             self.n_samples = cfg.val_steps

#         self.possibility[split] = []
#         self.min_possibility[split] = []
#         for i, colors in enumerate(self.dataset.input_colors[split]):
#             self.possibility[split] += [np.random.rand(colors.shape[0]) * 1e-3]
#             self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

#     def __iter__(self):
#         return self.spatially_regular_gen()

#     def __len__(self):
#         return self.n_samples

#     def spatially_regular_gen(self):
#         for i in range(self.n_samples * self.batch_size):
#             cloud_idx = int(np.argmin(self.min_possibility[self.split]))
#             point_ind = np.argmin(self.possibility[self.split][cloud_idx])

#             # 这里的 points 已经是归心化后的坐标了
#             points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)
#             center_point = points[point_ind, :].reshape(1, -1)
            
#             # 采样位置加入轻微噪声
#             noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
#             pick_point = center_point + noise.astype(center_point.dtype)

#             # KNN 采样
#             if len(points) < cfg.num_points:
#                 queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
#             else:
#                 queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

#             queried_idx = DP.shuffle_idx(queried_idx)
            
#             # 局部化坐标
#             queried_pc_xyz = points[queried_idx]
#             queried_pc_xyz = queried_pc_xyz - pick_point 
            
#             queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
#             queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

#             # 训练集数据增强
#             if self.split == 'training':
#                 # 随机旋转 (围绕 Z 轴)
#                 theta = np.random.uniform(0, 2 * np.pi)
#                 rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
#                                        [np.sin(theta), np.cos(theta), 0],
#                                        [0, 0, 1]])
#                 queried_pc_xyz = queried_pc_xyz @ rot_matrix
#                 # 随机缩放
#                 queried_pc_xyz *= np.random.uniform(0.9, 1.1)

#             # 更新 Active Learning 采样概率
#             dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
#             delta = np.square(1 - dists / np.max(dists))
#             self.possibility[self.split][cloud_idx][queried_idx] += delta
#             self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

#             # 补全采样点数
#             if len(points) < cfg.num_points:
#                 queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
#                     DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

#             # 转为 Tensor
#             queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
#             queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
#             queried_pc_labels = torch.from_numpy(queried_pc_labels).long()

#             # 拼接 [xyz, features] 作为网络输入
#             combined_features = torch.cat((queried_pc_xyz, queried_pc_colors), 1)

#             yield combined_features, queried_pc_labels

# def data_loaders(dir, sampling_method='active_learning', **kwargs):
#     # 统一从 'train' 子目录加载，CloudsDataset 内部通过文件名区分验证集
#     dataset = CloudsDataset(dir / 'train') 
#     batch_size = kwargs.get('batch_size', 4)
    
#     train_sampler = ActiveLearningSampler(dataset, batch_size=batch_size, split='training')
#     val_sampler = ActiveLearningSampler(dataset, batch_size=batch_size, split='validation')
    
#     return DataLoader(train_sampler, **kwargs), DataLoader(val_sampler, **kwargs)