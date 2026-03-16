# SlopeLAS 使用说明

## 1. 数据要求

`prepare_slope_las.py` 读取 LAS 文件，要求至少包含：
- `x, y, z` 坐标
- `classification` 语义标签字段

标签编码默认约定：
- 0: 正常
- 1: 凸起
- 2: 凹陷

如需映射不同编码，请修改 `prepare_slope_las.py` 中的 `CFG["label_map"]`。

## 2. 目录结构

```
E:\项目\点云无人机\myrandla-planA\RandLA-Net-pytorch
├─ datasets
│  └─ SlopeLAS
│     ├─ raw_las
│     │  ├─ xxx.las
│     │  └─ yyy.las
│     ├─ train
│     ├─ val
│     └─ test
```

## 3. 运行步骤

1) 数据预处理

```sh
python prepare_slope_las.py
```

2) 训练

```sh
python mytrain.py
```

3) 推理

```sh
python mytest.py
```

## 4. 输出格式

- 每个 `.npy` 样本为 `(N, 4)`：`[x, y, z, label]`
- 块内坐标已完成：中心化 + 归一化
- 标签范围严格为 `0/1/2`

推理输入默认目录为：

```
datasets/SlopeLAS/infer_las/*.las
```

## 5. 常见问题

- 不收敛：优先检查 `check_dataset.py` 输出，确认标签比例与坐标范围。
- 标签不在 0/1/2：修改 `label_map` 或检查 LAS 原始标注。
