"""
Acceleration Backends
====================

Different computation backends for high-performance pitch control calculations.
"""

from .numba_backend import NumbaBackend
from .vectorized import NumpyVectorizedBackend

__all__ = ["NumbaBackend", "NumpyVectorizedBackend"]
