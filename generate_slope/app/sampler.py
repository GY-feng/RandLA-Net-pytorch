import numpy as np


def compute_bounds(x, y, margin_ratio: float):
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    mx = (x_max - x_min) * float(margin_ratio)
    my = (y_max - y_min) * float(margin_ratio)
    return (x_min + mx, x_max - mx), (y_min + my, y_max - my)


def is_overlapping(cx, cy, r, existing):
    for ex, ey, er in existing:
        d = np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
        if d < (r + er):
            return True
    return False


def sample_center(rng, x_range, y_range, existing, radius, overlap_policy="avoid", max_attempts=2000):
    for _ in range(int(max_attempts)):
        cx = float(rng.uniform(x_range[0], x_range[1]))
        cy = float(rng.uniform(y_range[0], y_range[1]))
        if overlap_policy != "avoid":
            return cx, cy
        if not is_overlapping(cx, cy, radius, existing):
            return cx, cy
    return None, None
