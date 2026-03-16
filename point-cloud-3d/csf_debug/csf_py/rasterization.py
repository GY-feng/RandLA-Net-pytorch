"""Rasterization logic ported from C++ Rasterization.*

This module assigns lidar points to cloth particles (nearest in grid cell),
then fills missing heights using scanline/neighbor policy exactly as C++.
"""
from .particle import MIN_INF
from .cloth import Cloth


def findHeightValByNeighbor(particle):
    from collections import deque
    nqueue = deque()
    pbacklist = []

    for nb in particle.neighborsList:
        particle.isVisited = True
        nqueue.append(nb)

    while nqueue:
        pneighbor = nqueue.popleft()
        pbacklist.append(pneighbor)
        if pneighbor.nearestPointHeight > MIN_INF:
            for pb in pbacklist:
                pb.isVisited = False
            while nqueue:
                pp = nqueue.popleft()
                pp.isVisited = False
            return pneighbor.nearestPointHeight
        else:
            for ptmp in pneighbor.neighborsList:
                if not ptmp.isVisited:
                    ptmp.isVisited = True
                    nqueue.append(ptmp)
    return MIN_INF


def findHeightValByScanline(p, cloth: Cloth):
    xpos = p.pos_x
    ypos = p.pos_y
    for i in range(xpos + 1, cloth.num_particles_width):
        ch = cloth.getParticle(i, ypos).nearestPointHeight
        if ch > MIN_INF:
            return ch
    for i in range(xpos - 1, -1, -1):
        ch = cloth.getParticle(i, ypos).nearestPointHeight
        if ch > MIN_INF:
            return ch
    for j in range(ypos - 1, -1, -1):
        ch = cloth.getParticle(xpos, j).nearestPointHeight
        if ch > MIN_INF:
            return ch
    for j in range(ypos + 1, cloth.num_particles_height):
        ch = cloth.getParticle(xpos, j).nearestPointHeight
        if ch > MIN_INF:
            return ch
    return findHeightValByNeighbor(p)


def RasterTerrian(cloth: Cloth, pc: list, heightVal: list):
    # pc is a list of points with attributes x,y,z or tuple
    for i in range(len(pc)):
        pt = pc[i]
        pc_x = pt[0]
        pc_z = pt[2]
        deltaX = pc_x - cloth.origin_pos.x
        deltaZ = pc_z - cloth.origin_pos.z
        col = int(deltaX / cloth.step_x + 0.5)
        row = int(deltaZ / cloth.step_y + 0.5)
        if (col >= 0) and (row >= 0):
            particle = cloth.getParticle(col, row)
            particle.correspondingLidarPointList.append(i)
            pc2particleDist = (pc_x - particle.getPos().x) ** 2 + (pc_z - particle.getPos().z) ** 2
            if pc2particleDist < particle.tmpDist:
                particle.tmpDist = pc2particleDist
                particle.nearestPointHeight = pt[1]
                particle.nearestPointIndex = i

    # fill heightVal
    heightVal[:] = [MIN_INF] * cloth.getSize()
    for i in range(cloth.getSize()):
        pcur = cloth.getParticle1d(i)
        nearestHeight = pcur.nearestPointHeight
        if nearestHeight > MIN_INF:
            heightVal[i] = nearestHeight
        else:
            heightVal[i] = findHeightValByScanline(pcur, cloth)
