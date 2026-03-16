import numpy as np
import os, pickle
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import Dataset

class SlopeDataset(Dataset):
    def __init__(self, root_path, split='train', grid_size=0.04):
        self.path = os.path.join(root_path, f'input_{grid_size:.2f}')
        self.all_files = sorted([f[:-11] for f in os.listdir(self.path) if f.endswith('_KDTree.pkl')])
        
        # 划分逻辑：81-100 为 val，其余为 train
        val_names = [str(i) for i in range(81, 101)]
        if split == 'train':
            self.file_list = [f for f in self.all_files if f not in val_names]
        else:
            self.file_list = [f for f in self.all_files if f in val_names]

        self.data_list = []
        for name in self.file_list:
            self.data_list.append(os.path.join(self.path, name + '.ply'))
        
        # 类别权重：针对 <1% 的灾害点进行补偿
        # 权重公式：1 / log(1.2 + 频率)
        self.ignored_labels = [] 
        # 暂时手动设定权重，后续可根据统计结果微调
        self.class_weights = torch.FloatTensor([1.0, 50.0, 50.0, 50.0, 50.0])

    def __getitem__(self, item):
        # 实际 RandLA-Net 的训练通常采用随机切片采样
        # 这里仅展示核心加载逻辑，需配合 RandLA 的采样器使用
        pass

    def center_coords(self, xyz):
        """处理大地坐标偏移：减去均值"""
        offset = np.mean(xyz, axis=0)
        return xyz - offset, offset