"""Utilities for CuPy/NumPy arrays with Vec3 class matching C++ semantics."""
import math
from typing import Tuple, Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    import numpy as cp  # type: ignore
    CUPY_AVAILABLE = False

import numpy as np


def get_xp(use_gpu: bool = True):
    """Return array module: cupy if available+requested else numpy."""
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


def to_array(data, xp=None, dtype=None):
    """Convert list/ndarray to xp array with dtype (float64)"""
    if xp is None:
        xp = get_xp()
    arr = xp.array(data, dtype=dtype if dtype is not None else xp.float64)
    return arr


def is_cupy(xp):
    return CUPY_AVAILABLE and xp is not None and xp.__name__ == 'cupy'


class Vec3:
    """3D vector class matching C++ Vec3 semantics from Vec3.h.
    
    All operations use Python float (double precision) for deterministic behavior.
    Stores three doubles (f[0], f[1], f[2]) accessible as x, y, z.
    
    Supports all C++ operations:
      - Arithmetic: +, -, *, /, +=
      - Vector ops: length(), normalized(), dot(), cross()
      - Unary: negation (-)
    """
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialize vector with three double-precision floats.
        
        Args:
            x: X component (default 0.0)
            y: Y component (default 0.0)
            z: Z component (default 0.0)
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return (x, y, z) as tuple."""
        return (self.x, self.y, self.z)

    def length(self) -> float:
        """Compute Euclidean norm: sqrt(x^2 + y^2 + z^2)."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> 'Vec3':
        """Return unit vector in same direction.
        
        Returns:
            Vec3 with length 1.0
        """
        l = self.length()
        if l == 0.0:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def dot(self, other: 'Vec3') -> float:
        """Dot product: x*other.x + y*other.y + z*other.z."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        """Cross product: this × other.
        
        Returns:
            Vec3(y*other.z - z*other.y,
                 z*other.x - x*other.z,
                 x*other.y - y*other.x)
        """
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def __add__(self, other: Union['Vec3', float]) -> 'Vec3':
        """Vector addition or scalar addition."""
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vec3(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other: Union['Vec3', float]) -> 'Vec3':
        """Vector subtraction or scalar subtraction."""
        if isinstance(other, Vec3):
            return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vec3(self.x - other, self.y - other, self.z - other)

    def __mul__(self, s: float) -> 'Vec3':
        """Scalar multiplication: (x*s, y*s, z*s)."""
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: float) -> 'Vec3':
        """Right multiplication: s * vec3 == vec3 * s."""
        return self.__mul__(s)

    def __truediv__(self, a: float) -> 'Vec3':
        """Scalar division: (x/a, y/a, z/a)."""
        if a == 0.0:
            raise ValueError("Division by zero")
        return Vec3(self.x / a, self.y / a, self.z / a)

    def __neg__(self) -> 'Vec3':
        """Unary negation: (-x, -y, -z)."""
        return Vec3(-self.x, -self.y, -self.z)

    def __iadd__(self, other: 'Vec3') -> 'Vec3':
        """In-place addition (+=)."""
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __repr__(self) -> str:
        """String representation."""
        return f"Vec3({self.x}, {self.y}, {self.z})"

    def __eq__(self, other: object) -> bool:
        """Equality comparison (exact match, no tolerance)."""
        if not isinstance(other, Vec3):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self) -> int:
        """Hash for use in sets/dicts (based on truncated components)."""
        return hash((int(self.x * 1e6), int(self.y * 1e6), int(self.z * 1e6)))
