import numpy as np


def get_smooth_weights(r, R, smooth_type="linear"):
    if smooth_type == "linear":
        return 1 - r / R
    if smooth_type == "quadratic":
        return (1 - r / R) ** 2
    if smooth_type == "gaussian":
        sigma = R / 2
        return np.exp(-(r ** 2) / (2 * sigma ** 2))
    raise ValueError(f"Unknown smooth type: {smooth_type}")
