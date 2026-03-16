import numpy as np


def reset_classification(las_data, value: int = 0) -> None:
    """Reset all classification values to a single label."""
    n = len(las_data.points)
    labels = np.full(n, int(value), dtype=np.uint8)
    las_data.classification = labels


def count_labels(las_data, labels=(0, 1, 2)):
    arr = np.asarray(las_data.classification)
    return {int(l): int((arr == l).sum()) for l in labels}
