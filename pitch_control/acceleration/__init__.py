"""
Acceleration Backends
====================

Different computation backends for high-performance pitch control calculations.
"""

from .numba_backend import NumbaBackend
from .vectorized import VectorizedBackend

__all__ = ["NumbaBackend", "VectorizedBackend"]