"""Particle implementation mirroring C++ Particle semantics.

This class intentionally follows the C++ data layout and per-particle
operations (verlet integration, neighbor-based constraints) to make it
straightforward to verify bit-for-bit numerical parity when using
float64 and identical ordering.
"""
from typing import List
from .Vec3 import Vec3

DAMPING = 0.01
MAX_INF = 9.999999999e18
MIN_INF = -MAX_INF

singleMove1 = [0, 0.3, 0.51, 0.657, 0.7599, 0.83193, 0.88235, 0.91765, 0.94235, 0.95965, 0.97175, 0.98023, 0.98616, 0.99031, 0.99322]
doubleMove1 = [0, 0.3, 0.42, 0.468, 0.4872, 0.4949, 0.498, 0.4992, 0.4997, 0.4999, 0.4999, 0.5, 0.5, 0.5, 0.5]


class Particle:
    def __init__(self, pos: Vec3, time_step2: float):
        self.movable = True
        self.mass = 1.0
        self.acceleration = Vec3(0.0, 0.0, 0.0)
        self.time_step2 = float(time_step2)

        self.pos = Vec3(pos.x, pos.y, pos.z)
        self.old_pos = Vec3(pos.x, pos.y, pos.z)

        self.isVisited = False
        self.neibor_count = 0
        self.pos_x = 0
        self.pos_y = 0
        self.c_pos = 0

        self.neighborsList: List['Particle'] = []

        self.correspondingLidarPointList: List[int] = []
        self.nearestPointIndex = None
        self.nearestPointHeight = MIN_INF
        self.tmpDist = MAX_INF

    def isMovable(self) -> bool:
        return self.movable

    def addForce(self, f: Vec3):
        # acceleration += f / mass
        self.acceleration = Vec3(self.acceleration.x + f.x / self.mass,
                                 self.acceleration.y + f.y / self.mass,
                                 self.acceleration.z + f.z / self.mass)

    def timeStep(self):
        if self.movable:
            temp = Vec3(self.pos.x, self.pos.y, self.pos.z)
            self.pos = Vec3(
                self.pos.x + (self.pos.x - self.old_pos.x) * (1.0 - DAMPING) + self.acceleration.x * self.time_step2,
                self.pos.y + (self.pos.y - self.old_pos.y) * (1.0 - DAMPING) + self.acceleration.y * self.time_step2,
                self.pos.z + (self.pos.z - self.old_pos.z) * (1.0 - DAMPING) + self.acceleration.z * self.time_step2
            )
            self.old_pos = temp

    def offsetPos(self, v: Vec3):
        if self.movable:
            self.pos = Vec3(self.pos.x + v.x, self.pos.y + v.y, self.pos.z + v.z)

    def makeUnmovable(self):
        self.movable = False

    def resetAcceleration(self):
        self.acceleration = Vec3(0.0, 0.0, 0.0)

    def satisfyConstraintSelf(self, constraintTimes: int):
        p1 = self
        for p2 in self.neighborsList:
            # correctionVector only in y as in C++
            correction_y = p2.pos.y - p1.pos.y
            # determine move fraction
            if p1.isMovable() and p2.isMovable():
                frac = 0.5 if constraintTimes > 14 else doubleMove1[constraintTimes]
                half = frac * correction_y
                p1.offsetPos(Vec3(0.0, half, 0.0))
                p2.offsetPos(Vec3(0.0, -half, 0.0))
            elif p1.isMovable() and (not p2.isMovable()):
                frac = 1.0 if constraintTimes > 14 else singleMove1[constraintTimes]
                p1.offsetPos(Vec3(0.0, frac * correction_y, 0.0))
            elif (not p1.isMovable()) and p2.isMovable():
                frac = 1.0 if constraintTimes > 14 else singleMove1[constraintTimes]
                p2.offsetPos(Vec3(0.0, -frac * correction_y, 0.0))

    def getPos(self) -> Vec3:
        return self.pos

    def getPosCopy(self) -> Vec3:
        return Vec3(self.pos.x, self.pos.y, self.pos.z)

    def addToNormal(self, normal: Vec3):
        pass

    def resetNormal(self):
        pass

    def printself(self, s: str = ""):
        print(f"{s}: {self.getPos().x} movable: {self.movable}")
