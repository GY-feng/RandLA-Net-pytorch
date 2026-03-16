import cupy as cp
import numpy as np
from scipy.stats import qmc
from typing import Literal

class GridVolumeComputer:
    def __init__(self, XI, YI, ZA, ZB, n_samples, qmc_type='sobol', debug_mode=False):
        """
        初始化：预编译 Kernel 并预分配显存。
        """
        # copy=False 保证了如果类型已正确，则不会产生额外的显存开销
        self.XI = XI.astype(cp.float32, copy=False)
        self.YI = YI.astype(cp.float32, copy=False)
        self.ZA = ZA.astype(cp.float32, copy=False)
        self.ZB = ZB.astype(cp.float32, copy=False)
        self.rows, self.cols = ZA.shape  # 假设 ZA 是 (rows, cols) 的 2D 数组
        self.n_samples = n_samples

        # 预编译 CUDA Kernel (atomicAdd 版本)
        self.kernel_code = r'''
        extern "C" __global__
        void sdf_kernel(
            const float* rx, const float* ry, const float* rz,
            const float* ZA, const float* ZB,
            float x_min, float x_max, float y_min, float y_max,
            int rows, int cols, int n_samples,
            unsigned int* count,
            int* mask) 
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= n_samples) return;

            float x = rx[i]; float y = ry[i]; float z = rz[i];
            float dx = (x_max - x_min) / (cols - 1);
            float dy = (y_max - y_min) / (rows - 1);
            
            int c = max(0, min((int)((x - x_min) / dx), cols - 2));
            int r = max(0, min((int)((y - y_min) / dy), rows - 2));

            float u = (x - (x_min + c * dx)) / dx;
            float v = (y - (y_min + r * dy)) / dy;

            // Water-tight SDF
            auto get_z = [&](const float* Z) {
                float z00 = Z[r * cols + c];
                float z10 = Z[r * cols + (c + 1)];
                float z01 = Z[(r + 1) * cols + c];
                float z11 = Z[(r + 1) * cols + (c + 1)];
                if (u + v <= 1.0f) return z00 + u*(z10-z00) + v*(z01-z00);
                return z11 + (1.0f-u)*(z01-z11) + (1.0f-v)*(z10-z11);
            };

            bool is_inside = (z - get_z(ZA)) * (z - get_z(ZB)) <= 0.0f;

            if (is_inside) {
                atomicAdd(count, 1);
            }
            if (mask != nullptr) {
                mask[i] = is_inside ? 1 : 0;
            }
        }
        '''
        self.mod = cp.RawModule(code=self.kernel_code)
        self.sdf_kernel = self.mod.get_function('sdf_kernel')
        
        # 预分配：QMC 采样点显存 (避免循环内 Malloc)
        self.rand_x = cp.zeros(n_samples, dtype=cp.float32)
        self.rand_y = cp.zeros(n_samples, dtype=cp.float32)
        self.rand_z = cp.zeros(n_samples, dtype=cp.float32)
        self.inside_count = cp.zeros(1, dtype=cp.uint32)
        
        # 预生成 QMC 序列基础数据 (Sobol/Halton)
        self.qmc_base = self._generate_qmc_base(n_samples, qmc_type)

        self.debug_mode = debug_mode
        if self.debug_mode:
            self.inside_mask = cp.zeros(n_samples, dtype=cp.int32)
        else:
            self.inside_mask = None # 传入 CUDA 将作为 nullptr

    def get_prism_params(self):
        '''可以用一个倾斜平面加上下边界和上边界的偏移量来描述整体的体积范围变化。'''
        # 1. 拼接两个曲面的所有点
        xy_all = cp.concatenate([cp.stack([self.XI.ravel(), self.YI.ravel()], axis=1)] * 2, axis=0)
        z_all = cp.concatenate([self.ZA.ravel(), self.ZB.ravel()])

        # 2. 最小二乘法拟合整体趋势面: z = ax + by + d
        A = cp.stack([xy_all[:, 0], xy_all[:, 1], cp.ones_like(xy_all[:, 0])], axis=1)  # 构建矩阵 A = [x, y, 1]
        w, _, _, _ = cp.linalg.lstsq(A, z_all)  # 解线性方程 A*w = z -> w = [a, b, d_avg]
        a, b, _ = w.get()

        # 3. 寻找上下边界偏移量 (截距 d)
        d_vals = z_all - (a * xy_all[:, 0] + b * xy_all[:, 1])  # 对于每个点，计算其相对于趋势面的残差: d_i = z_i - (ax_i + by_i)
        d_min, d_max = cp.min(d_vals).item(), cp.max(d_vals).item()

        # 4. 计算 XY 范围边界
        x_min, x_max = cp.min(self.XI).item(), cp.max(self.XI).item()
        y_min, y_max = cp.min(self.YI).item(), cp.max(self.YI).item()

        return {
                "plane_coeffs": (float(a), float(b)),
                "d_top": float(d_max),
                "d_bottom": float(d_min),
                "x_range": (float(x_min), float(x_max)),
                "y_range": (float(y_min), float(y_max))
            }

    def _generate_qmc_base(self, n, q_type:Literal['sobol', 'halton'] = 'sobol'):
        if q_type == 'sobol':
            m = int(np.log2(n))
            return cp.asarray(qmc.Sobol(d=3, scramble=True).random_base2(m=m), dtype=cp.float32)
        elif q_type == 'halton':
            return cp.asarray(qmc.Halton(d=3, scramble=True).random(n=n), dtype=cp.float32)
        else:
            raise ValueError(f"Unsupported QMC type: {q_type}")

    def compute_volume(self, confidence_level=1.96):
        # 获取斜棱柱参数
        params = self.get_prism_params()
        a, b = params["plane_coeffs"]
        d_min, d_max = params["d_bottom"], params["d_top"]
        x_min, x_max = params["x_range"]
        y_min, y_max = params["y_range"]

        # 映射 QMC 采样点到当前网格的斜棱柱空间 (向量化操作，在 GPU 完成)
        # x, y 映射到网格矩形
        self.rand_x = x_min + self.qmc_base[:, 0] * (x_max - x_min)
        self.rand_y = y_min + self.qmc_base[:, 1] * (y_max - y_min)
        # z 映射到斜面约束范围: z = a*x + b*y + d_min + u*(d_max-d_min)
        self.rand_z = (a * self.rand_x + b * self.rand_y + d_min + 
                         self.qmc_base[:, 2] * (d_max - d_min))

        # 重置当前grid的计数器并启动 Kernel
        self.inside_count.fill(0)
        threads = 256
        blocks = (self.n_samples + threads - 1) // threads

        # Python float64与CUDA Kernel float32不符，参数压栈的字节数不匹配，导致 Kernel 内部读取到的全是乱码或 0
        self.sdf_kernel(
            (blocks,), (threads,),
            (self.rand_x, self.rand_y, self.rand_z, 
            self.ZA, self.ZB,
            np.float32(x_min), np.float32(x_max), np.float32(y_min), np.float32(y_max), # 必须包装类型
            np.int32(self.rows), np.int32(self.cols), np.int32(self.n_samples), 
            self.inside_count,
            self.inside_mask)
        )

        # 计算体积: V = V_prism * (hit / total)
        # V_prism = Area * thickness
        v_prism = (x_max - x_min) * (y_max - y_min) * (d_max - d_min)
        hit_ratio = float(self.inside_count[0]) / self.n_samples
        est_volume = v_prism * hit_ratio

        # 标准差计算 (基于伯努利分布)
        if 0 < hit_ratio < 1:
            std_hit = np.sqrt(hit_ratio * (1 - hit_ratio) / self.n_samples)
        else:
            std_hit = 0.0
            
        # 绝对误差波动 (95%置信度下)
        error_margin = v_prism * confidence_level * std_hit
        
        # 误差占计算体积的比例 (相对误差)
        error_percentage = (error_margin / est_volume * 100) if est_volume > 0 else 0

        return {
            "est_volume": est_volume,
            "error_margin": error_margin,        # 绝对波动值 (±)
            "error_percentage": error_percentage,  # 相对误差百分比
        }