# File: pitch_control/acceleration/vectorized.py

"""
Pure NumPy vectorized backend for pitch control calculations.
"""

import numpy as np
from typing import List, Tuple
from ..core import PlayerState, Point


class NumpyVectorizedBackend:
    """Pure NumPy vectorized computation backend (no JIT compilation)"""

    @staticmethod
    def is_available() -> bool:
        """Check if NumPy is available (should always be True)"""
        try:
            import numpy

            return True
        except ImportError:
            return False

    @staticmethod
    def calculate_times_vectorized(
        player_positions: np.ndarray,
        player_velocities: np.ndarray,
        grid_points: np.ndarray,
        max_speeds: np.ndarray,
        accelerations: np.ndarray,
        reaction_times: np.ndarray,
    ) -> np.ndarray:
        """
        Pure NumPy implementation of time-to-intercept calculations.

        Same signature as Numba version but using only NumPy operations.
        """
        n_players, _ = player_positions.shape
        n_grid, _ = grid_points.shape

        # Broadcast for vectorized distance calculation
        # Shape: (n_players, n_grid, 2)
        pos_expanded = player_positions[:, np.newaxis, :]
        grid_expanded = grid_points[np.newaxis, :, :]

        # Calculate distances: (n_players, n_grid)
        diff = grid_expanded - pos_expanded
        distances = np.sqrt(np.sum(diff**2, axis=2))

        # Avoid division by zero
        distances = np.maximum(distances, 1e-6)

        # Direction vectors: (n_players, n_grid, 2)
        directions = diff / distances[..., np.newaxis]

        # Current velocity toward target: (n_players, n_grid)
        vel_expanded = player_velocities[:, np.newaxis, :]
        vel_toward = np.maximum(0, np.sum(vel_expanded * directions, axis=2))

        # Physics calculations (vectorized)
        reaction_distances = vel_toward * reaction_times[:, np.newaxis]
        remaining_distances = np.maximum(0, distances - reaction_distances)

        # Simplified acceleration model (vectorized)
        speed_diff = max_speeds[:, np.newaxis] - vel_toward
        time_to_max_speed = np.maximum(
            0, speed_diff / np.maximum(accelerations[:, np.newaxis], 0.1)
        )

        # Distance covered during acceleration
        accel_distance = (
            vel_toward * time_to_max_speed
            + 0.5 * accelerations[:, np.newaxis] * time_to_max_speed**2
        )

        # Where we reach target before max speed
        before_max_mask = accel_distance >= remaining_distances

        # Case 1: Reach target during acceleration
        # Solve: d = v0*t + 0.5*a*t^2 using quadratic formula
        a_coeff = 0.5 * accelerations[:, np.newaxis]
        b_coeff = vel_toward
        c_coeff = -remaining_distances

        discriminant = b_coeff**2 + 4 * a_coeff * c_coeff
        discriminant = np.maximum(discriminant, 0)  # Avoid negative sqrt

        time_accel = np.where(
            a_coeff > 1e-6,
            (-b_coeff + np.sqrt(discriminant)) / (2 * a_coeff),
            remaining_distances / np.maximum(vel_toward, 0.1),
        )

        # Case 2: Accelerate to max speed, then constant speed
        remaining_at_max = remaining_distances - accel_distance
        time_at_max = remaining_at_max / np.maximum(max_speeds[:, np.newaxis], 0.1)
        time_constant = time_to_max_speed + time_at_max

        # Choose appropriate calculation
        total_time = np.where(before_max_mask, time_accel, time_constant)

        # Add reaction time
        total_time += reaction_times[:, np.newaxis]

        return total_time

    @staticmethod
    def calculate_ball_travel_times(
        ball_position: np.ndarray, grid_points: np.ndarray, ball_speed: float
    ) -> np.ndarray:
        """Calculate ball travel times using pure NumPy"""
        distances = np.sqrt(np.sum((grid_points - ball_position) ** 2, axis=1))
        return distances / ball_speed
