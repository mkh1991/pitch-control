from dataclasses import dataclass
import numpy as np

from ..core import PitchControlModel


@dataclass
class ModelConfig:
    """Base configuration for pitch control models"""

    grid_resolution: tuple[int, int] = (105, 68)
    use_numba: bool = True
    parallel: bool = True
    cache_grid: bool = True


class OptimizedPitchControlModel(PitchControlModel):
    """Base class for optimized pitch control models"""

    def __init__(self, config: ModelConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or ModelConfig()
        self._compiled_functions = {}

    def _warm_up_numba(self):
        """Warm up Numba compilation with dummy data"""
        if not self.config.use_numba:
            return

        # Small dummy calculation to trigger compilation
        dummy_positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        dummy_velocities = np.array([[1.0, 0.0], [0.0, 1.0]])
        dummy_grid = np.array([[5.0, 5.0]])

        # This will compile the functions
        try:
            self._calculate_times_vectorized(
                dummy_positions,
                dummy_velocities,
                dummy_grid,
                max_speeds=np.array([8.0, 8.0]),
                accelerations=np.array([4.0, 4.0]),
                reaction_times=np.array([0.5, 0.5]),
            )
        except Exception:
            # If compilation fails, fall back to non-numba
            self.config.use_numba = False
