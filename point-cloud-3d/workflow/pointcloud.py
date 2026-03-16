import sys
import os
from pathlib import Path
from typing import Union, Tuple, List, override, Optional
from scipy.spatial import cKDTree

import numpy as np
import laspy
import copy
import pyproj
import rasterio
import open3d as o3d

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger

class PointCloud:
    def __init__(self, type='GPU'):
        self._point_nums = 0
        self._x = None
        self._y = None
        self._z = None
        self._red = None
        self._green = None
        self._blue = None
        self._intensity = None
        self._return_number = None
        self._classification = None
        self._normals = None

        self._crs: pyproj.CRS = None
        self._offsets = None
        self._scales = None

        self._exist_dimensions = [
            'x', 'y', 'z',
            'red', 'green', 'blue',
            'intensity', 'return_number', 'classification',
            'normals']

        self._status = type

        self._local_offsets = None  # 记录局部坐标的偏移量 (dx, dy, 0)

    def __len__(self):
        return self._point_nums

    def __getitem__(self, mask) -> 'PointCloud':
        """
        通过索引获取点云子集
        
        参数:
            mask: 可以是以下类型之一:
                - slice对象 (如 [1:10])
                - bool数组 (numpy或cupy)
                - 整数数组/列表 (离散下标)
                - 单个整数
        
        返回:
            新的PointCloud对象，包含筛选后的点
        """
        # 创建新对象
        new_pc = PointCloud(type=self.status)
        
        # 处理不同类型的mask
        if isinstance(mask, slice):
            # 切片直接传递
            pass
        elif isinstance(mask, (int, np.integer)):
            # 单个整数索引转换为长度为1的数组
            mask = [mask]
        elif self.status == "CPU":
            if isinstance(mask, np.ndarray):
                if mask.dtype == bool:
                    pass  # bool数组直接使用
                elif np.issubdtype(mask.dtype, np.integer):
                    # 整数数组转换为bool数组
                    bool_mask = np.zeros(len(self), dtype=bool)
                    bool_mask[mask] = True
                    mask = bool_mask
                else:
                    raise TypeError("数组索引必须是bool或整数类型")
            elif isinstance(mask, list):
                if all(isinstance(x, bool) for x in mask):
                    mask = np.asarray(mask, dtype=bool)
                else:
                    # 整数列表转换为bool数组
                    bool_mask = np.zeros(len(self), dtype=bool)
                    bool_mask[mask] = True
                    mask = bool_mask
            else:
                raise TypeError(f"不支持的索引类型: {type(mask)}")
        elif self.status == "GPU":
            import cupy as cp
            if isinstance(mask, cp.ndarray):
                if mask.dtype == bool:
                    pass  # bool数组直接使用
                elif cp.issubdtype(mask.dtype, cp.integer):
                    # 整数数组转换为bool数组
                    bool_mask = cp.zeros(len(self), dtype=bool)
                    bool_mask[mask] = True
                    mask = bool_mask
                else:
                    raise TypeError("数组索引必须是bool或整数类型")
            elif isinstance(mask, np.ndarray):
                if mask.dtype == bool:
                    mask = cp.asarray(mask)
                elif np.issubdtype(mask.dtype, np.integer):
                    # 整数数组转换为bool数组
                    bool_mask = cp.zeros(len(self), dtype=bool)
                    bool_mask[cp.asarray(mask)] = True
                    mask = bool_mask
                else:
                    raise TypeError("数组索引必须是bool或整数类型")
            elif isinstance(mask, list):
                if all(isinstance(x, bool) for x in mask):
                    mask = cp.asarray(mask, dtype=bool)
                else:
                    # 整数列表转换为bool数组
                    bool_mask = cp.zeros(len(self), dtype=bool)
                    bool_mask[cp.asarray(mask)] = True
                    mask = bool_mask
            else:
                raise TypeError(f"不支持的索引类型: {type(mask)}")
        
        # 复制筛选后的属性
        for attr in self.exist_dimensions:
            value = getattr(self, attr)
            setattr(new_pc, attr, value[mask])

        new_pc.crs = self.crs
        new_pc.offsets = self.offsets
        new_pc.scales = self.scales
        return new_pc

    def __repr__(self):
        string = f"PointCloud(point_nums={self._point_nums},"
        string += f"x={"Present" if self.x is not None else "None"},"
        string += f"y={"Present" if self.y is not None else "None"},"
        string += f"z={"Present" if self.z is not None else "None"},"
        string += f"red={"Present" if self.red is not None else "None"},"
        string += f"green={"Present" if self.green is not None else "None"},"
        string += f"blue={"Present" if self.blue is not None else "None"},"
        string += f"intensity={"Present" if self.intensity is not None else "None"},"
        string += f"return_number={"Present" if self.return_number is not None else "None"},"
        string += f"classification={"Present" if self.classification is not None else "None"},"
        string += f"normals={"Present" if self.normals is not None else "None"}"
        string += ")"
        return string

    def __add__(self, other: 'PointCloud') -> 'PointCloud':
        """
        通过+运算符合并两个点云
        
        参数:
            other: 另一个PointCloud对象
            
        返回:
            新的合并后的PointCloud对象
        """
        if self.point_nums == 0:
            return other
        if other.point_nums == 0:
            return self

        if self.status != other.status:
            raise ValueError("无法合并点云：设备不一致")
        
        if self.status == 'CPU':
            import numpy as arr
        elif self.status == 'GPU':
            import cupy as arr

        # 检查坐标系是否一致
        if self.crs != other.crs:
            raise ValueError("无法合并点云：坐标系不一致")
        
        # 创建新对象
        new_pc = PointCloud()
    
        for dim in set(self.exist_dimensions + other.exist_dimensions):
            empty_shape = self.point_nums if dim != 'normals' else (self.point_nums, 3)
            # 左值在左，右值在右
            if dim not in self.exist_dimensions:
                dim_data = arr.concatenate([arr.zeros(empty_shape), getattr(other, dim)])
            if dim not in other.exist_dimensions:
                dim_data = arr.concatenate([getattr(self, dim), arr.zeros(empty_shape)])
            else:
                dim_data = arr.concatenate([getattr(self, dim), getattr(other, dim)])

            setattr(new_pc, dim, dim_data)
        
        # 设置元数据
        new_pc.crs = self.crs
        new_pc.offsets = self.offsets
        new_pc.scales = self.scales
        
        return new_pc

    def __iadd__(self, other: 'PointCloud') -> 'PointCloud':
        """
        通过+=运算符合并两个点云(原地操作)
        
        参数:
            other: 另一个PointCloud对象
            
        返回:
            self (支持链式调用)
        """
        if self.point_nums == 0:
            # 如果当前点云为空，直接复制其他点云的属性
            for dim in other.exist_dimensions:
                setattr(self, dim, getattr(other, dim))
            self.crs = other.crs
            self.offsets = other.offsets
            self.scales = other.scales
            return self
        
        if other.point_nums == 0:
            return self
            
        if self.status != other.status:
            raise ValueError("无法合并点云：设备不一致")
        
        if self.status == 'CPU':
            import numpy as arr
        elif self.status == 'GPU':
            import cupy as arr
            
        # 检查坐标系是否一致
        if self.crs != other.crs:
            raise ValueError("无法合并点云：坐标系不一致")
        
        # 合并所有存在的维度
        for dim in set(self.exist_dimensions + other.exist_dimensions):
            empty_shape = self.point_nums if dim != 'normals' else (self.point_nums, 3)
            if dim not in self.exist_dimensions:
                dim_data = arr.concatenate([arr.zeros(empty_shape), getattr(other, dim)])
            if dim not in other.exist_dimensions:
                dim_data = arr.concatenate([getattr(self, dim), arr.zeros(empty_shape)])
            else:
                dim_data = arr.concatenate([getattr(self, dim), getattr(other, dim)])
            
            setattr(self, '_' + dim, dim_data) # 绕过检查强制更新
        
        # 更新点云数量
        self._point_nums = self._point_nums + other._point_nums
        
        return self

    # 使用 @staticmethod，因为 __new__ 是类方法
    @staticmethod
    def _rebuild(cls, state_data):
        """反序列化助手函数，用于重建对象。"""
        new_pc = cls(type='CPU') # 初始状态为CPU
        new_pc.__setstate__(state_data)
        return new_pc

    def clone(self) -> 'PointCloud':
        """
        创建一个点云的深拷贝，以便在进行 CPU 转换时不影响原始对象
        """
        new_pc = PointCloud(type=self.status)
        
        for dim in self._exist_dimensions:
            value = getattr(self, '_' + dim)
            if value is not None:
                setattr(new_pc, '_' + dim, value.copy())
 
        new_pc._crs = self.crs
        
        if self.offsets is not None:
            new_pc._offsets = self.offsets.copy()
        if self.scales is not None:
            new_pc._scales = self.scales.copy()
            
        new_pc._point_nums = self._point_nums
        return new_pc
    
    def __reduce__(self):
        """
        定义如何序列化PointCloud对象。
        在序列化时，数据必须位于CPU (NumPy)。
        """
        # 如果当前状态是GPU，必须先转换为CPU。
        # 这里必须使用克隆+转换，否则会改变原始对象！
        if self.status == "GPU":
            logger.warning("PointCloud在GPU状态下被序列化，将克隆并转换为CPU进行序列化，原始对象状态不变。")
            cpu_pc = self.clone().to_cpu() 
        else:
            cpu_pc = self
        
        state = {}
        for dim in self._exist_dimensions:
            state[dim] = getattr(cpu_pc, dim) if getattr(cpu_pc, dim) is not None else None
        
        state['crs'] = cpu_pc._crs.to_wkt() if cpu_pc._crs is not None else None
        state['offsets'] = cpu_pc._offsets if cpu_pc._offsets is not None else None
        state['scales'] = cpu_pc._scales if cpu_pc._scales is not None else None
        state['point_nums'] = cpu_pc._point_nums
        
        # 返回重建对象所需的函数和参数
        # return (
        #     self.__class__.__new__,  # 用于创建新对象的函数
        #     (self.__class__,),       # 传递给__new__的参数（这里是类本身）
        #     state                    # 传递给__setstate__的状态字典
        # )
        return (
            PointCloud._rebuild,  # 重建函数（静态方法）
            (self.__class__, state) # 传递给_rebuild的参数：类本身和状态字典
        )

    def __setstate__(self, state):
        """
        定义如何从状态字典恢复PointCloud对象。
        """
        self.__init__(type='CPU')
        # 恢复点云维度数据
        for dim in self._exist_dimensions:
            # 使用setter来恢复属性，并自动更新_point_nums
            setattr(self, dim, state[dim])
            
        if state['crs'] is not None:
            self.crs = pyproj.CRS.from_wkt(state['crs'])
        
        self.offsets = state['offsets']
        self.scales = state['scales']
        
        # 确保点数正确恢复（虽然setter会更新，但显式设置更安全）
        self._point_nums = state['point_nums']

    def _clear(self) -> None:
        for dim in self.exist_dimensions:
            setattr(self, dim, None)

        self.crs = None
        self.offsets = None
        self.scales = None

    @property
    def point_nums(self) -> int:
        return self._point_nums

    def _check_type_with_status(self, value):
        if self.status == "CPU":
            assert isinstance(value, np.ndarray) or value is None, f"不支持的类型{type(value)}"
        else:
            import cupy as cp
            assert isinstance(value, (np.ndarray, cp.ndarray)) or value is None, f"不支持的类型{type(value)}"
            if isinstance(value, np.ndarray):
                value = cp.asarray(value)

        return value
    
    def to_local_origin(self) -> 'PointCloud':
        """
        将坐标转换为局部坐标系（xy 平移使最小点位于 (0,0) 附近）
        安全、可重复调用（已有偏移记录时直接返回）
        """
        if self._x is None or self._y is None:
            return self

        if self._local_offsets is not None:
            logger.warning("XY已是局部坐标，不再转换")
            return self

        xp = np if self.status == "CPU" else __import__('cupy')

        min_x = xp.min(self._x)
        min_y = xp.min(self._y)

        self._x = self._x - min_x
        self._y = self._y - min_y

        self._local_offsets = xp.asarray([
            float(min_x.get() if hasattr(min_x, 'get') else min_x),
            float(min_y.get() if hasattr(min_y, 'get') else min_y),
            0.0
        ])

        return self

    def to_global_origin(self) -> 'PointCloud':
        """
        从局部坐标恢复到全局原始坐标
        仅当有偏移记录时生效，否则无操作
        """
        if self._local_offsets is None:
            logger.warning("XY已是全局坐标")
            return self

        if self._x is None or self._y is None:
            self._local_offsets = None
            return self

        dx, dy, _ = self._local_offsets

        self._x = self._x + dx
        self._y = self._y + dy

        # 恢复后清空记录
        self._local_offsets = None

        return self
    
    @property
    def is_local_origin(self) -> bool:
        return self._local_offsets is not None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value) -> None:

        value = self._check_type_with_status(value)

        if value is None:
            del self.x
            return

        if value.dtype != np.float64:
            value = value.astype(np.float64)
            logger.warning("x坐标已转换为float64双精度")

        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的X坐标数组与点数不匹配，传入的形状为：{value.shape}")

        self._x = value

    @x.deleter
    def x(self) -> None:
        self._x = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.y
            return
        
        if value.dtype != np.float64:
            value = value.astype(np.float64)
            logger.warning("y坐标已转换为float64双精度")

        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的Y坐标数组与点数不匹配，传入的形状为：{value.shape}")

        self._y = value

    @y.deleter
    def y(self) -> None:
        self._y = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value) -> None:

        value = self._check_type_with_status(value)

        if value is None:
            del self.z
            return
        
        if value.dtype != np.float64:
            value = value.astype(np.float64)
            logger.warning("z坐标已转换为float64双精度")

        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的Z坐标数组与点数不匹配，传入的形状为：{value.shape}")

        self._z = value

    @z.deleter
    def z(self) -> None:
        self._z = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def red(self):
        return self._red

    @red.setter
    def red(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.red
            return
        
        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的红色通道数组与点数不匹配，传入的形状为：{value.shape}")

        self._red = value

    @red.deleter
    def red(self) -> None:
        self._red = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def green(self):
        return self._green

    @green.setter
    def green(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.green
            return
        
        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的绿色通道数组与点数不匹配，传入的形状为：{value.shape}")

        self._green = value

    @green.deleter
    def green(self) -> None:
        self._green = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def blue(self):
        return self._blue

    @blue.setter
    def blue(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.blue
            return
        
        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的蓝色通道数组与点数不匹配，传入的形状为：{value.shape}")

        self._blue = value

    @blue.deleter
    def blue(self) -> None:
        self._blue = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.intensity
            return
        
        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的强度数组与点数不匹配，传入的形状为：{value.shape}")

        self._intensity = value

    @intensity.deleter
    def intensity(self) -> None:
        self._intensity = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def return_number(self):
        return self._return_number

    @return_number.setter
    def return_number(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.return_number
            return
        
        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的回波次数数组与点数不匹配，传入的形状为：{value.shape}")

        self._return_number = value

    @return_number.deleter
    def return_number(self) -> None:
        self._return_number = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def classification(self):
        return self._classification

    @classification.setter
    def classification(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.classification
            return
        
        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的类型数组与点数不匹配，传入的形状为：{value.shape}")

        self._classification = value

    @classification.deleter
    def classification(self) -> None:
        self._classification = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def normals(self):
        return self._normals

    @normals.setter
    def normals(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.normals
            return
        
        if len(value.shape) != 2 or value.shape[1] != 3:
            raise ValueError(f"法向量数组形状必须为[n,3]，传入的形状为：{value.shape}")
            
        if self._point_nums == 0:
            self._point_nums = value.shape[0]
        elif self._point_nums != value.shape[0]:
            raise ValueError(f"传入的法向量数组与点数不匹配，传入的形状为：{value.shape}")

        self._normals = value

    @normals.deleter
    def normals(self) -> None:
        self._normals = None
        if len(self.exist_dimensions) == 0:
            self._point_nums = 0

    @property
    def crs(self) -> pyproj.CRS:
        return self._crs
        
    @crs.setter
    def crs(self, value: Union[pyproj.CRS, None]) -> None:
        if value is None:
            del self.crs
            return

        self._crs = value
        
    @crs.deleter
    def crs(self) -> None:
        self._crs = None

    @property
    def offsets(self):
        return self._offsets
        
    @offsets.setter
    def offsets(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.offsets
            return

        if value.shape != (3,):
            raise ValueError("offsets数组形状必须为(3,)")

        self._offsets = value
        
    @offsets.deleter
    def offsets(self) -> None:
        if self.status == "CPU":
            self._offsets = np.asarray([0,0,0])
        else:
            import cupy as cp
            self._offsets = cp.asarray([0,0,0])

    @property
    def scales(self):        
        return self._scales
        
    @scales.setter
    def scales(self, value) -> None:

        value = self._check_type_with_status(value)
        
        if value is None:
            del self.scales
            return
        
        if value.shape != (3,):
            raise ValueError("scales数组形状必须为(3,)")

        self._scales = value
        
    @scales.deleter
    def scales(self) -> None:
        if self.status == "CPU":
            self._scales = np.asarray([1,1,1])
        else:
            import cupy as cp
            self._scales = cp.asarray([1,1,1])

    @property
    def exist_dimensions(self) -> List[str]:
        dims = []
        for dim in self._exist_dimensions:
            if getattr(self, dim) is not None:
                dims.append(dim)
        return dims

    def estimate_normals(
        self,
        search_param: Tuple[float, int] = (0.3, 30),
        fast_normal_computation: bool = True,
        exist_ok = False
    ) -> None:
        """
        使用open3d估计点云法向量
        
        参数:
            search_param: KD树搜索参数，默认为o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            fast_normal_computation: 是否使用快速法向量计算
        """

        if self.status == "GPU":
            import cupy as cp
            xyz = cp.vstack((self._x, self._y, self._z)).get().T
        else:
            xyz = np.vstack((self._x, self._y, self._z)).T

        if self._x is None or self._y is None or self._z is None:
            raise ValueError("无法计算法向量：点坐标数据缺失")

        if self._normals is not None and not exist_ok:
            # raise ValueError("法向量已存在")
            logger.warning("法向量已存在，跳过计算")
            return
            
        # 将点坐标从GPU内存转到CPU内存
        
        # 创建open3d点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # 设置默认搜索参数

        param = o3d.geometry.KDTreeSearchParamHybrid(radius=search_param[0], max_nn=search_param[1])
        
        # 计算法向量
        pcd.estimate_normals(
            search_param=param,
            fast_normal_computation=fast_normal_computation
        )
        self.normals = np.asarray(pcd.normals)

    def load_from_las(self, las: Union[str, laspy.LasData]) -> None:
        """
        从LAS文件加载点云数据

        参数:
            las: LAS文件路径或LasData对象
        """
        
        self._clear()
        
        if isinstance(las, (str, Path)):
            las = laspy.read(las)
            
        try:
            self.crs = las.header.parse_crs(prefer_wkt=False)
            self.offsets = np.asarray(las.header.offsets)
            self.scales = np.asarray(las.header.scales)
            
            self.x = np.asarray(las.x, dtype=np.float64)
            self.y = np.asarray(las.y, dtype=np.float64)
            self.z = np.asarray(las.z, dtype=np.float64)

            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                # 保持颜色处于UINT16范围，GPU对INT16比FP32更快
                self.red = np.asarray(las.red)
                self.green = np.asarray(las.green)
                self.blue = np.asarray(las.blue)
                
            # 设置强度数据（如果存在）
            if hasattr(las, 'intensity'):
                self.intensity = np.asarray(las.intensity)
                
            # 设置回波次数（如果存在）
            if hasattr(las, 'return_number'):
                self.return_number = np.asarray(las.return_number)

            # 设置类型（如果存在）
            if hasattr(las,'classification'):
                self.classification = np.asarray(las.classification)
        except Exception as e:
            logger.error(f"加载LAS文件时出错: {e}")
            self._clear()
            raise e

    def load_from_dem(self, dem_path: Union[str, Path]) -> None:
        """
        从DEM文件(TIFF格式)加载点云数据
        
        参数:
            dem_path: DEM文件路径
        """

        self._clear()
        
        try:
            # 使用rasterio打开DEM文件
            with rasterio.open(dem_path) as src:
                # 读取高程数据
                dem_data = src.read(1)  # 读取第一个波段
                
                # 获取地理参考信息
                transform = src.transform
                crs = src.crs
                
                # 创建行索引和列索引网格
                rows, cols = np.indices(dem_data.shape)
                
                # 使用仿射变换计算地理坐标
                xs, ys = transform * (cols + 0.5, rows + 0.5)  # 计算像素中心坐标
                
                # 展平数组
                xs = xs.ravel()
                ys = ys.ravel()
                zs = dem_data.ravel()
                
                # 创建有效数据掩码（排除NaN值）
                valid_mask = (~np.isnan(zs)) & (zs != -9999)
                
                # 应用掩码过滤无效点
                xs = xs[valid_mask]
                ys = ys[valid_mask]
                zs = zs[valid_mask]
                
                # 设置点云属性
                self.crs = pyproj.CRS.from_wkt(crs.wkt) if crs else None
                self.x = xs
                self.y = ys
                self.z = zs
                
                # 设置偏移和缩放因子
                self.offsets = np.asarray([0, 0, 0])
                self.scales = np.asarray([1, 1, 1])
                
        except Exception as e:
            logger.error(f"加载DEM文件时出错: {str(e)}")
            self._clear()
            raise e

    def transform_to(self, crs: pyproj.CRS) -> None:
        """
        对点云进行坐标变换

        参数:
            CRS: 目标坐标系
        """        
        if self._x is None or self._y is None:
            raise ValueError("无法进行坐标变换：点坐标数据缺失")

        if self._crs is None:
            raise ValueError("无法进行坐标变换：源坐标系缺失")

        T = pyproj.Transformer.from_crs(self._crs, crs, always_xy=True)
        if self.status == 'CPU':
            self.x, self.y = T.transform(self.x, self.y)
        else:
            self.x, self.y = T.transform(self.x.get(), self.y.get())
        self.crs = crs

    def icp_to(self, target: 'PointCloud', method = 'plane'):
        assert self.normals is not None, "源点云法向量缺失"
        assert target.normals is not None, "目标点云法向量缺失"

        from processor.converter.point_cloud_icp_registration import point_cloud_icp_registration
        import cupy as cp
        icp_result = point_cloud_icp_registration(self, target, method=method, max_correspondence_distance=0.1, max_iteration=2000)

        transformation = cp.asarray(icp_result.transformation)
        tz = transformation[2, 3]
        logger.info(f"ICP配准结果\n变换矩阵：\n{transformation}\n拟合分数：{icp_result.fitness} 均方根误差：{icp_result.inlier_rmse}")
        logger.info(f"垂直平移量: {tz:.3f} 米")

        R = transformation[:3, :3]  # 形状为 (3, 3)
        T = transformation[:3, 3]  # 形状为 (3,)
        transformed_points = (cp.vstack((self.x, self.y, self.z)).T @ R.T) + T
        self.x, self.y, self.z = transformed_points.T

    @property
    def status(self) -> str:
        """
        返回点云数据的存储位置（CPU或GPU）

        返回:
            str: "CPU" 或 "GPU"
        """
        return self._status
    
    @status.setter
    def status(self, value: str) -> None:
        raise ValueError("status属性不可直接设置，请使用to_cpu()或to_gpu()方法")

    @status.deleter
    def status(self) -> None:
        raise ValueError("status属性不可直接删除，请使用to_cpu()或to_gpu()方法")

    def to_gpu(self) -> 'PointCloud':
        """
        将点云中的所有数组转换为cupy数组（GPU内存）
        
        返回:
            self (支持链式调用)
        """
        if self.status == "GPU":
            return self
                    
        self._status = "GPU"

        import cupy as cp
            
        for dim in self.exist_dimensions:
            cur_dim = getattr(self, dim)
            if dim in ['x', 'y', 'z']:
                setattr(self, dim, cp.asarray(cur_dim, dtype=cp.float64))
            else:
                setattr(self, dim, cp.asarray(cur_dim))
        
        if self.offsets is not None:
            self.offsets = cp.asarray(self.offsets, dtype=cp.float64)
        if self.scales is not None:
            self.scales = cp.asarray(self.scales, dtype=cp.float64)

        return self

    def to_cpu(self) -> 'PointCloud':
        """
        将点云中的所有数组转换为numpy数组（CPU内存）
        
        返回:
            self (支持链式调用)
        """
        if self.status == "CPU":
            return self
            
        self._status = "CPU"

        import numpy as np
            
        for dim in self.exist_dimensions:
            setattr(self, dim, getattr(self, dim).get())
        
        # 处理offsets和scales
        if self.offsets is not None:
            self.offsets = self._offsets.get()
        if self.scales is not None:
            self.scales = self._scales.get()
        return self

    def export_to_las(self, save_path):
        
        status_changed = False
        if self.status == "GPU":
            self.to_cpu()
            status_changed = True
        
        try:
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            header = laspy.LasHeader(point_format=3, version="1.2")
            
            if self.crs is not None:
                header.add_crs(self.crs)
                header.offset = self.offsets
                header.scale = self.scales

            # las = laspy.create(point_format=3, version="1.2")
            las = laspy.LasData(header)
            las.x = self.x
            las.y = self.y
            las.z = self.z

            if self.red is not None and self.green is not None and self.blue is not None:
                las.red = self.red
                las.green = self.green
                las.blue = self.blue
            
            if self.intensity is not None:
                las.intensity = self.intensity
                
            if self.return_number is not None:
                las.return_number = np.round(self.return_number).astype(int)  # 在minz求地面点的情况，回波次数是浮点数，需强转类型

            if self.classification is not None:
                las.classification = self.classification

            las.write(save_path)
            logger.info(f"LAS文件成功导出到：{save_path}")

            if status_changed:  # 只有原来是在gpu才会改变
                self.to_gpu()

        except Exception as e:
            logger.error(f"导出LAS文件时出错: {e}")
            raise e

if __name__ == "__main__":
    import pickle
    import hashlib
    import time
    import cupy as cp
    from checker.check_cuda_status import check_cuda_status

    test_las_file = "/home/gary/CloudPointProcessing/云梧高速/DJI_202506240857_181_云梧-K136-274-404-点云/raw/las/cloud_merged.las"
    test_dem_file = "/home/gary/CloudPointProcessing/云梧高速/DJI_202506231055_160_云梧-K169-545-K170-645-点云/raw/dem/dem.tif"
    pc = PointCloud(type='GPU')

    try:
        start = time.time()
        pc.load_from_las(test_las_file)
        logger.info(f"成功加载LAS文件，点数: {pc.point_nums}")
        logger.info(f"坐标系: {pc.crs} 当前状态:{pc.status}")
        logger.info(f"加载las消耗时间:{(time.time() -  start):.2f} s")
        print("status 属性:", pc.status)
        print("x 是 numpy 吗？", isinstance(pc.x, np.ndarray))
        print("x 是 cupy 吗？", isinstance(pc.x, cp.ndarray))
        print(pc.x[0])
        print(pc.y[0])
        print(pc.z[0])
        # pc.load_from_dem(test_dem_file)
        # logger.info(f"成功加载DEM文件，点数: {pc.point_nums}")
        # logger.info(f"坐标系: {pc.crs}")
        
        # logger.info(f"当前状态: {pc.status}")
        # logger.info(f"转移至CPU准备测试序列化")
        # pc.to_cpu()
        # logger.info(f"当前状态: {pc.status}")
        
        # serialize = pickle.dumps(pc)
        # hash = hashlib.sha256(serialize).hexdigest()
        # logger.info(f"序列化后的SHA256哈希: {hash}")
        # with open(Path("./data/AK0.dump"), "wb") as f:  # 注意必须是二进制模式 'wb'
        #     f.write(serialize)

        # pc: PointCloud = pickle.load(open(Path("./data/AK0.dump"), "rb"))

        # logger.info(f"序列化已测试完成，转移至GPU")
        # pc.to_gpu()
        # logger.info(f"当前状态: {pc.status}, {type(pc.scales)}")
        # pc.transform_to(pyproj.CRS.from_epsg(32649))
        # logger.info(f"转换后坐标系：{pc.crs}")


    #     if pc.x is not None:
    #         logger.info(f"X通道类型: {type(pc.x.dtype)}")
    #     if pc.y is not None:
    #         logger.info(f"Y通道类型: {type(pc.y.dtype)}")
    #     if pc.z is not None:
    #         logger.info(f"Z通道类型: {type(pc.z.dtype)}")

    #     if pc.red is not None:
    #         logger.info(f"Red通道类型: {type(pc.red.dtype)}")
    #     if pc.green is not None:
    #         logger.info(f"Green通道类型: {type(pc.green.dtype)}")
    #     if pc.blue is not None:
    #         logger.info(f"Blue通道类型: {type(pc.blue.dtype)}")

    #     if pc.intensity is not None:
    #         logger.info(f"Intensity通道类型: {type(pc.intensity.dtype)}")
    #     if pc.return_number is not None:
    #         logger.info(f"Return Number通道类型: {type(pc.return_number.dtype)}")
    #     if pc.normals is not None:
    #         logger.info(f"Normals通道类型: {type(pc.normals.dtype)}")
    #     if pc.classification is not None:
    #         logger.info(f"Classification通道类型: {type(pc.classification.dtype)}")

    #     # 测试索引切割
    #     pcA = pc[0:10000]
    #     logger.info(f"pcA切割后点数: {pcA.point_nums}")
    #     mask = np.random.choice(
    #                 [True, False],  # 选择 True 或 False
    #                 size=pc.point_nums,  # 长度
    #                 p=[0.2, 1 - 0.2]  # 概率
    #             )
    #     pcB = pc[40000:50000]
    #     logger.info(f"pcB切割后点数: {pcB.point_nums}")
        
    #     logger.info(f"耗时: {time.time()-start:.2f}s")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
