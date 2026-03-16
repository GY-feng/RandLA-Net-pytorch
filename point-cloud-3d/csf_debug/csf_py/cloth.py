"""Cloth class reproducing C++ Cloth behavior and ordering.

This implementation follows the original loops and neighbor connections
including primary and secondary constraints. It uses the Python `Particle`
class and preserves the per-iteration ordering to improve reproducibility
with the C++ reference.
"""
from typing import List
from .Vec3 import Vec3
from .particle import Particle, MAX_INF, MIN_INF

MAX_PARTICLE_FOR_POSTPROCESSIN = 50

class XY:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Cloth:
    def __init__(self,
                 origin_pos: Vec3,
                 num_particles_width: int,
                 num_particles_height: int,
                 step_x: float,
                 step_y: float,
                 smoothThreshold: float,
                 heightThreshold: float,
                 rigidness: int,
                 time_step: float):

        self.constraint_iterations = rigidness
        self.time_step = time_step
        self.smoothThreshold = smoothThreshold
        self.heightThreshold = heightThreshold
        self.origin_pos = origin_pos
        self.step_x = step_x
        self.step_y = step_y
        self.num_particles_width = num_particles_width
        self.num_particles_height = num_particles_height

        self.particles: List[Particle] = [None] * (num_particles_width * num_particles_height)

        time_step2 = time_step * time_step

        # create particles
        for i in range(num_particles_width):
            for j in range(num_particles_height):
                pos = Vec3(origin_pos.x + i * step_x,
                           origin_pos.y,
                           origin_pos.z + j * step_y)
                p = Particle(pos, time_step2)
                p.pos_x = i
                p.pos_y = j
                self.particles[j * num_particles_width + i] = p

        # immediate neighbors
        for x in range(num_particles_width):
            for y in range(num_particles_height):
                if x < num_particles_width - 1:
                    self.makeConstraint(self.getParticle(x, y), self.getParticle(x + 1, y))
                if y < num_particles_height - 1:
                    self.makeConstraint(self.getParticle(x, y), self.getParticle(x, y + 1))
                if (x < num_particles_width - 1) and (y < num_particles_height - 1):
                    self.makeConstraint(self.getParticle(x, y), self.getParticle(x + 1, y + 1))
                if (x < num_particles_width - 1) and (y < num_particles_height - 1):
                    self.makeConstraint(self.getParticle(x + 1, y), self.getParticle(x, y + 1))

        # secondary neighbors (distance 2)
        for x in range(num_particles_width):
            for y in range(num_particles_height):
                if x < num_particles_width - 2:
                    self.makeConstraint(self.getParticle(x, y), self.getParticle(x + 2, y))
                if y < num_particles_height - 2:
                    self.makeConstraint(self.getParticle(x, y), self.getParticle(x, y + 2))
                if (x < num_particles_width - 2) and (y < num_particles_height - 2):
                    self.makeConstraint(self.getParticle(x, y), self.getParticle(x + 2, y + 2))
                if (x < num_particles_width - 2) and (y < num_particles_height - 2):
                    self.makeConstraint(self.getParticle(x + 2, y), self.getParticle(x, y + 2))

        self.smoothThreshold = smoothThreshold
        self.heightThreshold = heightThreshold
        self.origin_pos = origin_pos
        self.heightvals: List[float] = [MIN_INF] * self.getSize()

    def getSize(self) -> int:
        return self.num_particles_width * self.num_particles_height

    def getParticle(self, x: int, y: int) -> Particle:
        return self.particles[y * self.num_particles_width + x]

    def getParticle1d(self, index: int) -> Particle:
        return self.particles[index]

    def makeConstraint(self, p1: Particle, p2: Particle):
        p1.neighborsList.append(p2)
        p2.neighborsList.append(p1)

    def timeStep(self) -> float:
        # call timeStep for each particle
        for i in range(self.getSize()):
            self.particles[i].timeStep()

        # satisfy constraints
        for j in range(self.getSize()):
            self.particles[j].satisfyConstraintSelf(self.constraint_iterations)

        maxDiff = 0.0
        for i in range(self.getSize()):
            if self.particles[i].isMovable():
                diff = abs(self.particles[i].old_pos.y - self.particles[i].pos.y)
                if diff > maxDiff:
                    maxDiff = diff
        return maxDiff

    def addForce(self, direction: Vec3):
        for p in self.particles:
            p.addForce(direction)

    def terrCollision(self):
        for i in range(self.getSize()):
            v = self.particles[i].getPos()
            if v.y < self.heightvals[i]:
                self.particles[i].offsetPos(Vec3(0.0, self.heightvals[i] - v.y, 0.0))
                self.particles[i].makeUnmovable()

    def movableFilter(self):
        for x in range(self.num_particles_width):
            for y in range(self.num_particles_height):
                ptc = self.getParticle(x, y)
                if ptc.isMovable() and (not ptc.isVisited):
                    que = []
                    connected: List[XY] = []
                    neibors = []
                    sumc = 1
                    connected.append(XY(x, y))
                    ptc.isVisited = True
                    que.append(y * self.num_particles_width + x)

                    while que:
                        idx = que.pop(0)
                        ptc_f = self.particles[idx]
                        cur_x = ptc_f.pos_x
                        cur_y = ptc_f.pos_y
                        neibor = []

                        # left
                        if cur_x > 0:
                            pleft = self.getParticle(cur_x - 1, cur_y)
                            if pleft.isMovable():
                                if not pleft.isVisited:
                                    sumc += 1
                                    pleft.isVisited = True
                                    connected.append(XY(cur_x - 1, cur_y))
                                    que.append(cur_y * self.num_particles_width + cur_x - 1)
                                    neibor.append(sumc - 1)
                                    pleft.c_pos = sumc - 1
                                else:
                                    neibor.append(pleft.c_pos)

                        # right
                        if cur_x < self.num_particles_width - 1:
                            pright = self.getParticle(cur_x + 1, cur_y)
                            if pright.isMovable():
                                if not pright.isVisited:
                                    sumc += 1
                                    pright.isVisited = True
                                    connected.append(XY(cur_x + 1, cur_y))
                                    que.append(cur_y * self.num_particles_width + cur_x + 1)
                                    neibor.append(sumc - 1)
                                    pright.c_pos = sumc - 1
                                else:
                                    neibor.append(pright.c_pos)

                        # bottom
                        if cur_y > 0:
                            pbot = self.getParticle(cur_x, cur_y - 1)
                            if pbot.isMovable():
                                if not pbot.isVisited:
                                    sumc += 1
                                    pbot.isVisited = True
                                    connected.append(XY(cur_x, cur_y - 1))
                                    que.append((cur_y - 1) * self.num_particles_width + cur_x)
                                    neibor.append(sumc - 1)
                                    pbot.c_pos = sumc - 1
                                else:
                                    neibor.append(pbot.c_pos)

                        # top
                        if cur_y < self.num_particles_height - 1:
                            ptop = self.getParticle(cur_x, cur_y + 1)
                            if ptop.isMovable():
                                if not ptop.isVisited:
                                    sumc += 1
                                    ptop.isVisited = True
                                    connected.append(XY(cur_x, cur_y + 1))
                                    que.append((cur_y + 1) * self.num_particles_width + cur_x)
                                    neibor.append(sumc - 1)
                                    ptop.c_pos = sumc - 1
                                else:
                                    neibor.append(ptop.c_pos)

                        neibors.append(neibor)

                    if sumc > MAX_PARTICLE_FOR_POSTPROCESSIN:
                        edgePoints = self.findUnmovablePoint(connected)
                        self.handle_slop_connected(edgePoints, connected, neibors)

    def findUnmovablePoint(self, connected: List[XY]) -> List[int]:
        edgePoints: List[int] = []
        for i in range(len(connected)):
            x = connected[i].x
            y = connected[i].y
            index = y * self.num_particles_width + x
            ptc = self.getParticle(x, y)

            # check neighbors and conditions exactly as C++
            if x > 0:
                ptc_x = self.getParticle(x - 1, y)
                if not ptc_x.isMovable():
                    index_ref = y * self.num_particles_width + x - 1
                    if (abs(self.heightvals[index] - self.heightvals[index_ref]) < self.smoothThreshold) and (ptc.getPos().y - self.heightvals[index] < self.heightThreshold):
                        offsetVec = Vec3(0.0, self.heightvals[index] - ptc.getPos().y, 0.0)
                        self.particles[index].offsetPos(offsetVec)
                        ptc.makeUnmovable()
                        edgePoints.append(i)
                        continue

            if x < self.num_particles_width - 1:
                ptc_x = self.getParticle(x + 1, y)
                if not ptc_x.isMovable():
                    index_ref = y * self.num_particles_width + x + 1
                    if (abs(self.heightvals[index] - self.heightvals[index_ref]) < self.smoothThreshold) and (ptc.getPos().y - self.heightvals[index] < self.heightThreshold):
                        offsetVec = Vec3(0.0, self.heightvals[index] - ptc.getPos().y, 0.0)
                        self.particles[index].offsetPos(offsetVec)
                        ptc.makeUnmovable()
                        edgePoints.append(i)
                        continue

            if y > 0:
                ptc_y = self.getParticle(x, y - 1)
                if not ptc_y.isMovable():
                    index_ref = (y - 1) * self.num_particles_width + x
                    if (abs(self.heightvals[index] - self.heightvals[index_ref]) < self.smoothThreshold) and (ptc.getPos().y - self.heightvals[index] < self.heightThreshold):
                        offsetVec = Vec3(0.0, self.heightvals[index] - ptc.getPos().y, 0.0)
                        self.particles[index].offsetPos(offsetVec)
                        ptc.makeUnmovable()
                        edgePoints.append(i)
                        continue

            if y < self.num_particles_height - 1:
                ptc_y = self.getParticle(x, y + 1)
                if not ptc_y.isMovable():
                    index_ref = (y + 1) * self.num_particles_width + x
                    if (abs(self.heightvals[index] - self.heightvals[index_ref]) < self.smoothThreshold) and (ptc.getPos().y - self.heightvals[index] < self.heightThreshold):
                        offsetVec = Vec3(0.0, self.heightvals[index] - ptc.getPos().y, 0.0)
                        self.particles[index].offsetPos(offsetVec)
                        ptc.makeUnmovable()
                        edgePoints.append(i)
                        continue

        return edgePoints

    def handle_slop_connected(self, edgePoints: List[int], connected: List[XY], neibors: List[List[int]]):
        visited = [False] * len(connected)
        que = []
        for e in edgePoints:
            que.append(e)
            visited[e] = True

        while que:
            index = que.pop(0)
            index_center = connected[index].y * self.num_particles_width + connected[index].x
            for i in range(len(neibors[index])):
                ni = neibors[index][i]
                index_neibor = connected[ni].y * self.num_particles_width + connected[ni].x
                if (abs(self.heightvals[index_center] - self.heightvals[index_neibor]) < self.smoothThreshold) and (abs(self.particles[index_neibor].getPos().y - self.heightvals[index_neibor]) < self.heightThreshold):
                    offsetVec = Vec3(0.0, self.heightvals[index_neibor] - self.particles[index_neibor].getPos().y, 0.0)
                    self.particles[index_neibor].offsetPos(offsetVec)
                    self.particles[index_neibor].makeUnmovable()
                    if not visited[ni]:
                        que.append(ni)
                        visited[ni] = True

    def toVector(self) -> List[float]:
        out = []
        for p in self.particles:
            out.append(p.getPos().x)
            out.append(p.getPos().z)
            out.append(-p.getPos().y)
        return out

    def saveToFile(self, path: str = ""):
        filepath = path if path else "cloth_nodes.txt"
        with open(filepath, 'w') as f:
            for p in self.particles:
                f.write(f"{p.getPos().x:.8f}\t{p.getPos().z:.8f}\t{-p.getPos().y:.8f}\n")

    def saveMovableToFile(self, path: str = ""):
        filepath = path if path else "cloth_movable.txt"
        with open(filepath, 'w') as f:
            for p in self.particles:
                if p.isMovable():
                    f.write(f"{p.getPos().x:.8f}\t{p.getPos().z:.8f}\t{-p.getPos().y:.8f}\n")
