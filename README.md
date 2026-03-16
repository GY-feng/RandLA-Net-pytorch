# RandLA-Net (SlopeLAS XYZ-only)

本项目用于高速公路边坡点云的灾害检测（凸起/凹陷）。
当前版本已精简为 **仅 XYZ 输入** + **三分类标签 0/1/2**，数据格式为分块 `.npy`。

## 快速开始

1. 安装依赖

```sh
pip install -r requirements.txt
```

2. 准备数据

将原始 `.las` 放入：

```
datasets/SlopeLAS/raw_las/*.las
```

然后执行：

```sh
python prepare_slope_las.py
```

3. 训练

```sh
python mytrain.py
```

4. 推理

```sh
python mytest.py
```

## 文档

- 使用说明：`docs/SLOPE_USAGE.md`
- 数据检查：`python check_dataset.py`

## 说明

- 推理输入目录默认：`datasets/SlopeLAS/infer_las/*.las`
- 标签编码默认：0=正常，1=凸起，2=凹陷
- 每个样本固定点数：`65536`
- 块内中心化并按 `block_size/2` 归一化
