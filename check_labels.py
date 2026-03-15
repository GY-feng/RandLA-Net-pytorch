import numpy as np
from pathlib import Path

data_path = Path('datasets/s3dis/subsampled/train')
files = list(data_path.glob('*.npy'))

f = files[0]
data = np.load(f)
print(f"数据形状 (Shape): {data.shape}")
print(f"前 5 行数据预览:\n{data[:5]}")

# 尝试找出哪一列全是整数（那通常就是标签）
for i in range(data.shape[1]):
    col = data[:, i]
    if np.all(np.mod(col, 1) == 0):
        print(f"第 {i} 列可能是标签列，范围: {col.min()} 到 {col.max()}")