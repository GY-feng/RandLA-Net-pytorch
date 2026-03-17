# 项目当前进度摘要

## 项目目标
基于 RandLA‑Net 对高速公路边坡点云进行灾害（凸起/凹陷）检测。输入为 XYZ 点云，标签为 0/1/2。

## 核心流程（推荐）
1. **模拟灾害（Windows 环境）**
   - `generate_slope/ground_sim_from_dir.py`
   - 读取原始 LAS，按 classification 过滤地面点（class=2），重置为 0，再进行凸起/凹陷模拟。
   - 支持 `save_intermediate=false`，不落地中间 ground 文件。
2. **数据准备（WSL 环境）**
   - `prepare_slope_las.py --config config/slope_config.yaml`
   - 生成 `.npy` 块（N,4）用于训练。
3. **训练**
   - `mytrain.py --config config/slope_config.yaml`
4. **测试/推理**
   - `mytest.py --config config/slope_config.yaml`
   - 支持将预测写回 LAS classification 并输出到指定目录。

## 重要配置
### 训练/准备/测试统一配置
`config/slope_config.yaml`

### 灾害模拟配置
`generate_slope/config/ground_sim.yaml`
- `save_intermediate` 控制是否保存中间地面点
- `ground_filter.min_points` 控制地面点不足直接跳过
- `defect.progress_every` 控制模拟进度输出频率

## 近期新增脚本
### 训练健壮性验证
`overfit_check.py`
- 单样本过拟合，验证模型/标签链路是否通

### 统计工具
`tools/npy_block_stats.py`
- 统计 prepare 后每个 .npy 的点数与密度，输出到 `tools/log`

`tools/las_stats.py`
- 统计原始 LAS 的点数与密度，输出到 `tools/log`

## 目录说明
- `generate_slope/`：灾害模拟（Windows 运行）
- `datasets/SlopeLAS/`：训练数据输出
- `runs/`：训练日志与权重
- `tools/`：统计脚本与日志

## 关键约定
- 标签：0=背景，1=凸起，2=凹陷
- 输入：XYZ-only
- `prepare_slope_las.py` 会将块中心化并按 `block_size/2` 归一化

## 已知现象
- `.npy` 约 2MB/块是正常现象（hstack 导致 float64）
- 训练 mIoU 波动较大，属于数据分布+随机采样导致，需进一步稳定化

