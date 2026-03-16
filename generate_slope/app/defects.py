import numpy as np
from .smoothing import get_smooth_weights


def apply_noise(las_data, std: float, rng) -> None:
    if std is None or std <= 0:
        return
    n = len(las_data.points)
    noise = rng.normal(0.0, float(std), (n, 3))
    las_data.x = np.asarray(las_data.x) + noise[:, 0]
    las_data.y = np.asarray(las_data.y) + noise[:, 1]
    las_data.z = np.asarray(las_data.z) + noise[:, 2]


def apply_radial_offset(
    las_data,
    center_x: float,
    center_y: float,
    radius: float,
    dz: float,
    smooth_type: str,
    label_value: int,
) -> int:
    """
    Apply radial z offset to points within radius and assign label.
    Returns number of affected points.
    """
    x = np.asarray(las_data.x)
    y = np.asarray(las_data.y)
    z = np.asarray(las_data.z)
    labels = np.asarray(las_data.classification).copy()

    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = dist <= radius
    if not np.any(mask):
        return 0

    weights = get_smooth_weights(dist[mask], radius, smooth_type)
    z[mask] = z[mask] + float(dz) * weights
    labels[mask] = int(label_value)

    las_data.z = z
    las_data.classification = labels

    return int(mask.sum())
