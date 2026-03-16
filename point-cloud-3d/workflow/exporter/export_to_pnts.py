import sys
import os
from pathlib import Path
import cupy as cp
import numpy as np
import pyproj

from py3dtiles.tileset import TileSet, Tile, BoundingVolumeBox
from py3dtiles.tileset.content import Pnts
from py3dtiles.tileset.content.pnts_feature_table import PntsFeatureTableHeader, SemanticPoint

sys.path.append(str(Path(__file__).parent.parent))

from pointcloud import PointCloud as PC

def export_to_pnts(pc: PC, export_path, export_name, uniq_color=None):
    """
    将点云转为PNTS
    
    参数:
        pc: 点云对象
        export_path: 导出路径
        export_name: 导出文件名
        uniq_color: 唯一颜色，默认为None,使用原色
    """
    # 获取点数据并转换为float32(3D Tiles规范要求)
    uniq_color = cp.asarray(uniq_color) if uniq_color is not None else None
    # 获取颜色信息
    if uniq_color is not None:
        colors = cp.full((pc.point_nums, 3), uniq_color).astype(cp.uint8)
    elif all(dim in pc.exist_dimensions for dim in ['red', 'green', 'blue']):
        # 处理颜色数据 (假设原始颜色是16位的)
        colors = cp.vstack((
            pc.red / 256, 
            pc.green / 256, 
            pc.blue / 256
        )).T.astype(cp.uint8)
    else:
        # 默认白色
        colors = cp.full((pc.point_nums, 3), 255, dtype=cp.uint8)
    
    T = pyproj.Transformer.from_crs(pc.crs,"EPSG:4978",always_xy=True)  # 全球三维坐标系（也叫地心坐标系）
    x, y, z = T.transform(pc.x.get(), pc.y.get(), pc.z.get())
    points = np.column_stack([x, y, z]).astype(np.float32)

    pnts = Pnts.from_features(
        PntsFeatureTableHeader.from_semantic(SemanticPoint.POSITION, SemanticPoint.RGB, None, nb_points = len(points)),
        points.flatten(),
        colors.get().flatten()
    )
    pnts.save_as(Path(os.path.join(export_path, export_name + '.pnts')))

    # 4. 创建 tileset.json
    tile = Tile()  # 创建一个 Tile 对象，表示一个 tile（切片）
    tile.content_uri = Path(export_name + '.pnts')
    tile.bounding_volume = BoundingVolumeBox.from_points(points)  # 根据点云的坐标计算 包围盒，即点云数据的空间范围，用来描述该切片的空间范围
    tileset = TileSet()  # 创建一个 TileSet 对象，它代表一个包含多个切片的集合。这里只创建了一个根切片（root_tile）
    tileset.root_tile = tile
    tileset.write_as_json(Path(os.path.join(export_path, export_name + '.json')))

if __name__ == "__main__":
    # 示例用法
    pc = PC()
    pc.load_from_las('./data/AK0.las')
    pnts_data = export_to_pnts(pc, './data', 'AK0', [255, 0, 0])