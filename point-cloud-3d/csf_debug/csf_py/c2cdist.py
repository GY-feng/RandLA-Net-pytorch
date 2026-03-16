"""Cloud-to-cloth distance classification (exact port of c2cdist.cpp).

Given a cloth (with particle positions) and a point cloud (list of (x,y,z)),
compute bilinear-interpolated cloth height at point location and threshold.
"""
from typing import List, Tuple


def calCloud2CloudDist(cloth, pc: List[Tuple[float,float,float]], class_threshold: float):
    groundIndexes = []
    offGroundIndexes = []

    for i in range(len(pc)):
        pc_x, pc_y, pc_z = pc[i]
        deltaX = pc_x - cloth.origin_pos.x
        deltaZ = pc_z - cloth.origin_pos.z
        col0 = int(deltaX / cloth.step_x)
        row0 = int(deltaZ / cloth.step_y)
        col1 = col0 + 1
        row1 = row0
        col2 = col0 + 1
        row2 = row0 + 1
        col3 = col0
        row3 = row0 + 1

        subdeltaX = (deltaX - col0 * cloth.step_x) / cloth.step_x
        subdeltaZ = (deltaZ - row0 * cloth.step_y) / cloth.step_y

        fxy = (cloth.getParticle(col0, row0).pos.y * (1 - subdeltaX) * (1 - subdeltaZ)
             + cloth.getParticle(col3, row3).pos.y * (1 - subdeltaX) * subdeltaZ
             + cloth.getParticle(col2, row2).pos.y * subdeltaX * subdeltaZ
             + cloth.getParticle(col1, row1).pos.y * subdeltaX * (1 - subdeltaZ))
        height_var = fxy - pc_y
        if abs(height_var) < class_threshold:
            groundIndexes.append(i)
        else:
            offGroundIndexes.append(i)

    return groundIndexes, offGroundIndexes
