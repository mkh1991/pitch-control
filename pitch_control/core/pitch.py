from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class Point:
    """2D point representation"""

    x: float
    y: float

    def distance(self, other: "Point") -> float:
        """Euclidean distance to another point"""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)


@dataclass
class PitchDimensions:
    """Standard pitch dimensions and coordinate system"""

    length: float = 105.0  # meters
    width: float = 68.0  # meters
    origin: str = "center"  # "center", "corner"

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get pitch boundaries as ((x_min, x_max), (y_min, y_max))"""
        if self.origin == "center":
            return (
                (-self.length / 2, self.length / 2),
                (-self.width / 2, self.width / 2),
            )
        else:  # corner
            return ((0, self.length), (0, self.width))


class Pitch:
    """Pitch representation with coordinate system and grid management"""

    def __init__(
        self,
        dimensions: PitchDimensions = None,
        grid_resolution: Tuple[int, int] = (105, 68),
    ):
        """
        Initialize pitch with dimensions and grid.

        Args:
            dimensions: Pitch dimensions and coordinate system
            grid_resolution: Number of grid points (length, width)
        """
        self.dimensions = dimensions or PitchDimensions()
        self.grid_resolution = grid_resolution
        self._grid_cache = {}

    def create_grid(
        self, resolution: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create coordinate grid for pitch control calculations.

        Args:
            resolution: Grid resolution override

        Returns:
            Tuple of (X, Y) coordinate grids
        """
        res = resolution or self.grid_resolution
        cache_key = res

        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        (x_min, x_max), (y_min, y_max) = self.dimensions.bounds

        x = np.linspace(x_min, x_max, res[0])
        y = np.linspace(y_min, y_max, res[1])
        X, Y = np.meshgrid(x, y)

        self._grid_cache[cache_key] = (X, Y)
        return X, Y

    def is_valid_position(self, point: Point) -> bool:
        """Check if point is within pitch boundaries"""
        (x_min, x_max), (y_min, y_max) = self.dimensions.bounds
        return x_min <= point.x <= x_max and y_min <= point.y <= y_max

    def get_grid_points_as_array(
        self, resolution: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Get all grid points as a flat array for vectorized operations.

        Returns:
            Array of shape (N, 2) where N = resolution[0] * resolution[1]
        """
        X, Y = self.create_grid(resolution)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        return points


@dataclass
class ControlSurface:
    """Represents calculated pitch control probabilities"""

    home_control: np.ndarray
    away_control: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    timestamp: float

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the control surface"""
        return self.home_control.shape

    @property
    def uncertainty(self) -> np.ndarray:
        """Calculate uncertainty (areas where control is contested)"""
        return 1.0 - np.abs(self.home_control - self.away_control)

    def get_control_at_point(self, point: Point) -> Tuple[float, float]:
        """
        Get control probabilities at a specific point.

        Returns:
            Tuple of (home_control, away_control) at the point
        """
        # Find nearest grid point (simple approach)
        x_idx = np.argmin(np.abs(self.grid_x[0, :] - point.x))
        y_idx = np.argmin(np.abs(self.grid_y[:, 0] - point.y))

        return self.home_control[y_idx, x_idx], self.away_control[y_idx, x_idx]
