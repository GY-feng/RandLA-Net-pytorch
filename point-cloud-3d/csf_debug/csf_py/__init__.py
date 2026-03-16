"""CSF Python core module

This package provides an independent, double-precision, CPU/GPU-capable
implementation of the CSF cloth-simulation filtering core. It aims to
follow the original C++ implementation exactly (algorithm steps and
ordering) while offering an option to run computations on CUDA using
PyTorch (dtype=torch.float64) to maintain numerical parity.
"""

from .csf import CSF

__all__ = ["CSF"]
