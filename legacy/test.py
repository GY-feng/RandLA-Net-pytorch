import numpy as np
from pathlib import Path
import time
import torch
from model import RandLANet
from utils.ply import write_ply

t0 = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. 路径设置
model_path = 'runs/4080_S3DIS_Final/checkpoint_50.pth'
# 我们直接手动指定一个存在的 npy 文件进行测试
data_path = 'datasets/s3dis/subsampled/test/5_office_1.npy'

print(f'Loading model from {model_path}...')
d_in = 6
num_classes = 13 
model = RandLANet(d_in, num_classes, 16, 4, device)

checkpoint = torch.load(model_path)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

print(f'Loading data from {data_path}...')
# S3DIS npy 通常格式: [N, 7] -> x, y, z, r, g, b, label
raw_data = np.load(data_path)
xyzrgb = raw_data[:, :6]
labels_raw = raw_data[:, 6]

# 随机采样 40960 个点，防止一次性输入太多导致显存溢出
n_points = xyzrgb.shape[0]
# n_sample = min(n_points, 40960)
n_sample = 65536
# 确保点数足够，不足则重复采样，足够则不重复采样
replace_mode = n_points < n_sample
idx = np.random.choice(n_points, n_sample, replace=(n_points < n_sample))

points = torch.from_numpy(xyzrgb[idx]).float().unsqueeze(0).to(device)
labels = torch.from_numpy(labels_raw[idx]).long().to(device)

print('Predicting...')


print('Predicting...')
with torch.no_grad():
    scores = model(points)
    
    # 自动识别维度
    if scores.shape[1] == num_classes: # [Batch, 13, N]
        predictions = torch.max(scores, dim=1).indices
    else: # [Batch, N, 13]
        predictions = torch.max(scores, dim=2).indices
    
    # 强制将维度对齐以便对比
    predictions = predictions.view(-1)
    labels = labels.view(-1)
    
    accuracy = (predictions == labels).float().mean()
    print(f'Patch Accuracy: {accuracy.item():.4f}')
    
    # 打印一些样本对比一下
    print("Sample Predictions:", predictions[:10].cpu().numpy())
    print("Sample Labels:     :", labels[:10].cpu().numpy())
    
    # 准备保存数据
    predictions_np = predictions.squeeze(0).cpu().numpy()
    cloud_np = points.squeeze(0)[:, :3].cpu().numpy()

print('Writing results to MiniDijon9.ply...')
write_ply('MiniDijon9.ply', [cloud_np, predictions_np.astype(np.int32)], ['x', 'y', 'z', 'class'])

t1 = time.time()
print('Done. Time elapsed: {:.1f}s'.format(t1-t0))