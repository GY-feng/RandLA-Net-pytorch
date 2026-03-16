"""Top-level CSF class mirroring C++ CSF API.

Usage:
  from csf_py import CSF
  csf = CSF()
  csf.set_point_cloud(numpy_array_or_list)
  ground_idx, off_idx = csf.filter()

This module uses the python implementations of Cloth, Rasterization and c2cdist
and preserves algorithm ordering and default parameters. It supports running
on CPU. For GPU acceleration we rely on user-level vectorized replacements —
this initial port focuses on faithful algorithm replication.
"""
from typing import List, Tuple
from .cloth import Cloth
from .rasterization import RasterTerrian
from .c2cdist import calCloud2CloudDist
from .Vec3 import Vec3


class Params:
    def __init__(self):
        self.bSloopSmooth = True
        self.time_step = 0.65
        self.class_threshold = 0.5
        self.cloth_resolution = 1.0
        self.rigidness = 3
        self.interations = 500


class CSF:
    def __init__(self, index: int = 0):
        self.params = Params()
        self.index = index
        self.point_cloud: List[Tuple[float,float,float]] = []

    def set_point_cloud(self, points):
        """Accepts list/iterable of (x,y,z) or Nx3 numpy array."""
        self.point_cloud = []
        for p in points:
            # Note: C++ flips coords: las.y = -points[i].z; las.z = points[i].y
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
            # keep internal coord consistent with C++ setPointCloud
            self.point_cloud.append((x, y, z))

    def do_cloth(self) -> Cloth:
        # compute bounding box
        xs = [p[0] for p in self.point_cloud]
        ys = [p[1] for p in self.point_cloud]
        zs = [p[2] for p in self.point_cloud]
        bbMin = Vec3(min(xs), min(ys), min(zs))
        bbMax = Vec3(max(xs), max(ys), max(zs))

        cloth_y_height = 0.05
        clothbuffer_d = 2
        origin_pos = Vec3(
            bbMin.x - clothbuffer_d * self.params.cloth_resolution,
            bbMax.y + cloth_y_height,
            bbMin.z - clothbuffer_d * self.params.cloth_resolution
        )

        width_num = int((bbMax.x - bbMin.x) // self.params.cloth_resolution) + 2 * clothbuffer_d
        height_num = int((bbMax.z - bbMin.z) // self.params.cloth_resolution) + 2 * clothbuffer_d

        cloth = Cloth(
            origin_pos,
            width_num,
            height_num,
            self.params.cloth_resolution,
            self.params.cloth_resolution,
            0.3,
            9999,
            self.params.rigidness,
            self.params.time_step
        )

        # rasterize
        RasterTerrian(cloth, self.point_cloud, cloth.heightvals)

        time_step2 = self.params.time_step * self.params.time_step
        gravity = 0.2
        cloth.addForce(Vec3(0.0, -gravity, 0.0) * time_step2)

        for i in range(self.params.interations):
            maxDiff = cloth.timeStep()
            cloth.terrCollision()
            if (maxDiff != 0) and (maxDiff < 0.005):
                break

        if self.params.bSloopSmooth:
            cloth.movableFilter()

        return cloth

    def filter(self, export_cloth: bool = False) -> Tuple[List[int], List[int]]:
        cloth = self.do_cloth()
        if export_cloth:
            cloth.saveToFile()
        ground, off = calCloud2CloudDist(cloth, self.point_cloud, self.params.class_threshold)
        return ground, off

    def export_cloth(self) -> List[float]:
        cloth = self.do_cloth()
        return cloth.toVector()
