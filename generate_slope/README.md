# Generate Slope (Windows)

本工具用于在 Windows 下对 LAS 点云进行灾害模拟（凸起/凹陷），输出分类标签 0/1/2。
生成的 .las 可直接复制到 `datasets/SlopeLAS/raw_las`，再运行 `prepare_slope_las.py`。

## 依赖

```sh
pip install -r requirements.txt
```

## Step1: 扫描 + 去重（默认全选）

```sh
python step1_scan_dedupe.py --config config/default.yaml
```

编辑 `scan.output_json`，把需要的条目 `selected` 改为 true/false。

## Step2: 过滤地面点 (class=2)

```sh
python step2_filter_ground.py --config config/default.yaml
```

输出文件命名为：`子文件夹名.las`，目录由 `step2.output_dir` 决定。

## Step3: 批量生成（凸起/凹陷）

```sh
python step3_generate.py --config config/default.yaml
```

兼容入口：`python run_generate.py` / `python main.py` / `python batch_edit_las.py`

## 与训练对齐

将输出 `.las` 复制到：

```
RandLA-Net-pytorch/datasets/SlopeLAS/raw_las
```

然后执行：

```sh
python prepare_slope_las.py
```
