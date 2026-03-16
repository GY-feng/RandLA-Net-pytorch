"""示例：使用 PointCloud 类与三个 CSF 实现版本进行比对测试。

用法:
  在工作目录下运行：
  python python/test_csf_py_demo.py

输出文件（每个版本）：
  - cloth_nodes_csf_py.txt       （纯 Python 版 cloth 节点）
  - ground_indices_csf_py.txt    （纯 Python 版 ground 索引）
  - cloth_nodes_csf_py_gpu.txt   （GPU 版 cloth 节点）
  - ground_indices_csf_py_gpu.txt（GPU 版 ground 索引）
  - cloth_nodes_csf_py_cupy.txt  （CuPy 版 cloth 节点）
  - ground_indices_csf_py_cupy.txt（CuPy 版 ground 索引）
  - comparison_report.txt        （对比报告）

说明：脚本按 `CSF` 的标准流程执行：
  1. 使用 PointCloud 类生成随机点云（包含地面和一些高程噪声/建筑点）
  2. 分别使用三个版本的 CSF 实现过滤
  3. 逐点对比结果，统计分类差异
  4. 保存 cloth 节点与地面索引，便于可视化或进一步分析

此脚本展示 csf_py 的执行原理，并验证多个实现版本的一致性。
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from workflow.pointcloud import PointCloud as PC
from workflow.processor.classify.classify_ground_point_by_csf import classify_ground_point_by_csf
from csf_py import CSF as CSF_py


def generate_random_point_cloud(n_points=5000, seed=0, type='CPU'):
    """使用 PointCloud 类生成带有地面与非地面点的随机点云。

    参数：
      n_points: 点数
      seed: 随机种子
      type: 'CPU' 或 'GPU'

    返回：
      PointCloud 对象
    """
    rng = np.random.RandomState(seed)
    # 横向范围
    x = rng.uniform(0.0, 200.0, n_points)
    z = rng.uniform(0.0, 200.0, n_points)

    # 模拟地面高度（平滑变化）
    ground_height = 2.0 * np.sin(x * 0.02) + 0.5 * np.cos(z * 0.03)
    # 基本噪声
    noise = rng.normal(scale=0.05, size=n_points)
    y_ground = ground_height + noise

    # 随机把部分点抬高为非地面（建筑/植被）
    is_off = rng.rand(n_points) < 0.12
    y = y_ground + is_off * rng.uniform(1.0, 6.0, size=n_points)

    # 创建 PointCloud 对象
    pc = PC(type=type)
    pc.x = x
    pc.y = y
    pc.z = z
    pc._point_nums = n_points

    return pc


def run_csf_version(name, pc):
    """运行单个 CSF 版本。

    参数：
      name: 版本名称（用于输出文件
      pc: PointCloud 对象

    返回：
      (ground_idx, off_idx, cloth_coords, elapsed_time)
    """
    import time
    pts_array = np.vstack([pc.x, pc.y, pc.z]).T
    start = time.time()

    if name == 'csf_py':
        csf = CSF_py()
        # 设置参数（统一参数确保公平对比）
        csf.params.cloth_resolution = 1.0
        csf.params.time_step = 0.65
        csf.params.class_threshold = 0.5
        csf.params.rigidness = 3
        csf.params.interations = 500

        # 运行过滤
        csf.set_point_cloud(pts_array)
        ground_idx, off_idx = csf.filter()
        cloth_coords = csf.export_cloth()

    elif name == 'csf_c':
        pc, ground_idx = classify_ground_point_by_csf(pc, 
                                                      cloth_resolution=1.0,
                                                      rigidness=3,
                                                      class_threshold=0.5,
                                                      iterations=500,
                                                      time_step=0.65
                                                      )

    else:
        raise ValueError(f"Unknown CSF version: {name}")

    elapsed = time.time() - start

    # 保存输出
    # cloth_arr = np.array(cloth_coords).reshape(-1, 3)
    # np.savetxt(f'cloth_nodes_{name}.txt', cloth_arr, fmt='%.8f')
    # np.savetxt(f'ground_indices_{name}.txt', np.array(ground_idx, dtype=int), fmt='%d')

    print(f"[{name}] 地面点：{len(ground_idx)}, 耗时：{elapsed:.2f}s")

    return ground_idx, elapsed


def compare_results(results_dict, pc):
    """对比多个 CSF 版本的结果。

    参数：
      results_dict: {'name': (ground_idx, off_idx, cloth_coords, time)}
      pc: PointCloud 对象（用于点数对比）
    """
    names = list(results_dict.keys())
    report = []
    report.append("=" * 80)
    report.append("CSF 版本对比报告")
    report.append("=" * 80)
    report.append(f"点云数量：{len(pc)} 点\n")

    # 逐版本输出统计
    for name, (ground, elapsed) in results_dict.items():
        report.append(f"[{name}]")
        report.append(f"  - 地面点数：{len(ground)} ({len(ground)*100/len(pc):.1f}%)")
        # report.append(f"  - 非地面点数：{len(off)} ({len(off)*100/len(pc):.1f}%)")
        report.append(f"  - 执行时间：{elapsed:.3f}s")

    # 两两对比
    report.append("\n" + "-" * 80)
    report.append("版本间对比：")
    report.append("-" * 80)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_i = names[i]
            name_j = names[j]
            ground_i, _ = results_dict[name_i]
            ground_j, _ = results_dict[name_j]

            # 计算差异
            diff_ground = len(set(ground_i) ^ set(ground_j))
            # diff_off = len(set(off_i) ^ set(off_j))
            total_diff = diff_ground

            report.append(f"\n{name_i} vs {name_j}:")
            report.append(f"  - 地面点分类差异：{diff_ground} 个点")
            # report.append(f"  - 非地面点分类差异：{diff_off} 个点")
            report.append(f"  - 总体一致性：{(1 - total_diff/len(pc))*100:.2f}%")

            # 对于 cloth 节点的对比
            cloth_i = np.array(results_dict[name_i][2]).reshape(-1, 3)
            cloth_j = np.array(results_dict[name_j][2]).reshape(-1, 3)
            cloth_diff = np.abs(cloth_i - cloth_j)
            report.append(f"  - Cloth 节点最大差异：{np.max(cloth_diff):.2e}")
            report.append(f"  - Cloth 节点均差异：{np.mean(cloth_diff):.2e}")

    report_text = "\n".join(report)
    print("\n" + report_text)
    with open('comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("\n已保存对比报告到 comparison_report.txt")


def main():
    # 生成测试点云
    print("生成测试点云...")
    pc = generate_random_point_cloud(n_points=50000, seed=42, type='CPU')
    print(f"生成点云：{len(pc)} 点")

    # 运行三个版本
    results = {}
    print("运行 CSF 过滤...")
    results['csf_py'] = run_csf_version('csf_py', pc)
    results['csf_c'] = run_csf_version('csf_c', pc)

    # 对比结果
    compare_results(results, pc)


if __name__ == '__main__':
    main()
