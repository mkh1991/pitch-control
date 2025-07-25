"""
Numba JIT compilation backend for maximum performance.
"""

import numba
from numba import jit, prange
import numpy as np


class NumbaBackend:
    """Numba-accelerated computation backend"""

    @staticmethod
    def is_available() -> bool:
        """Check if Numba is available and working"""
        try:
            @jit(nopython=True)
            def test_func(x):
                return x + 1

            test_func(1.0)
            return True
        except Exception:
            return False

    @staticmethod
    def warm_up():
        """Compile functions with dummy data"""
        from ..models.spearman import _calculate_times_vectorized

        dummy_pos = np.array([[0.0, 0.0]])
        dummy_vel = np.array([[1.0, 0.0]])
        dummy_grid = np.array([[5.0, 5.0]])
        dummy_speeds = np.array([8.0])
        dummy_accel = np.array([4.0])
        dummy_reaction = np.array([0.5])

        _calculate_times_vectorized(
            dummy_pos, dummy_vel, dummy_grid,
            dummy_speeds, dummy_accel, dummy_reaction
        )