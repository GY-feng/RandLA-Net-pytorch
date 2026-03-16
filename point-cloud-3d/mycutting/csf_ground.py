from __future__ import annotations

from typing import Tuple
import numpy as np


def csf_ground_indices(
    xyz: np.ndarray,
    *,
    bSloopSmooth: bool = False,
    time_step: float = 0.65,
    class_threshold: float = 0.5,
    cloth_resolution: float = 1.0,
    rigidness: int = 3,
    interation: int = 500,
) -> np.ndarray:
    """
    Run CSF to get ground indices. Returns 1D numpy array of indices.
    """
    try:
        import CSF  # type: ignore
    except Exception as e:
        raise RuntimeError("CSF module not available. Install CSF in this environment.") from e

    pts = np.asarray(xyz, dtype=np.float64)

    csf = CSF.CSF()
    csf.params.bSloopSmooth = bSloopSmooth
    csf.params.time_step = time_step
    csf.params.class_threshold = class_threshold
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.interation = interation

    csf.setPointCloud(pts)
    ground_idx = CSF.VecInt()
    offground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, offground_idx, exportCloth=False)

    ground = np.fromiter(ground_idx, dtype=np.int64, count=len(ground_idx))
    return ground
