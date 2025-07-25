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
        Vectorized implementation that produces identical results to the Numba version.

        Follows Laurie Shaw's approach:
        1. Continue at current velocity during reaction time
        2. Calculate distance from reaction position to target
        3. Handle acceleration from reaction position (starting from zero velocity)

        This should produce IDENTICAL results to your Numba implementation.
        """
        n_players, _ = player_positions.shape
        n_grid, _ = grid_points.shape

        # STEP 1: Calculate reaction positions (where players will be after reaction time)
        # Shape: (n_players, 2) - same as: reaction_pos_x = pos_x + vel_x * reaction_time
        reaction_positions = (
            player_positions + player_velocities * reaction_times[:, np.newaxis]
        )

        # STEP 2: Vectorized distance calculation from reaction positions to targets
        # Shape: (n_players, n_grid, 2)
        reaction_pos_expanded = reaction_positions[
            :, np.newaxis, :
        ]  # (n_players, 1, 2)
        grid_expanded = grid_points[np.newaxis, :, :]  # (1, n_grid, 2)

        # Calculate distances from reaction positions to grid points
        # Same as: dx = target_x - reaction_pos_x; dy = target_y - reaction_pos_y; distance = sqrt(dx*dx + dy*dy)
        diff = grid_expanded - reaction_pos_expanded
        distances = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (n_players, n_grid)

        # Handle very small distances (same as: if distance < 1e-6)
        very_close_mask = distances < 1e-6
        distances = np.maximum(distances, 1e-6)  # Avoid division by zero

        # STEP 3: Handle acceleration cases
        max_speeds_safe = np.maximum(max_speeds, 0.1)  # Same as: max(max_speed, 0.1)
        accelerations_safe = np.maximum(accelerations, 0.1)  # Avoid division by zero

        # Expand arrays for broadcasting
        max_speeds_expanded = max_speeds_safe[:, np.newaxis]  # (n_players, 1)
        accelerations_expanded = accelerations_safe[:, np.newaxis]  # (n_players, 1)
        reaction_times_expanded = reaction_times[:, np.newaxis]  # (n_players, 1)

        # Case 1: No acceleration (acceleration <= 0)
        no_accel_mask = accelerations[:, np.newaxis] <= 0

        # Simple sprint time: distance / max_speed
        simple_sprint_times = distances / max_speeds_expanded

        # Case 2: With acceleration (acceleration > 0)
        # Same as: time_to_max_speed = max_speed / acceleration
        time_to_max_speed = max_speeds_expanded / accelerations_expanded

        # Same as: distance_during_accel = 0.5 * acceleration * time_to_max_speed * time_to_max_speed
        distance_during_accel = 0.5 * accelerations_expanded * time_to_max_speed**2

        # Check if we reach target during acceleration phase
        # Same as: if distance_during_accel >= distance
        reaches_during_accel = distance_during_accel >= distances

        # Case 2a: Reach target during acceleration
        # Same as: accel_time = sqrt(2.0 * distance / acceleration)
        accel_only_time = np.sqrt(2.0 * distances / accelerations_expanded)

        # Case 2b: Accelerate then constant speed
        # Same as: remaining_distance = distance - distance_during_accel
        remaining_distance = distances - distance_during_accel
        # Same as: const_time = remaining_distance / max_speed
        const_time = remaining_distance / max_speeds_expanded
        # Same as: total = time_to_max_speed + const_time
        two_phase_time = time_to_max_speed + const_time

        # Choose between acceleration cases
        accel_sprint_times = np.where(
            reaches_during_accel, accel_only_time, two_phase_time
        )

        # Choose between no acceleration vs acceleration
        sprint_times = np.where(no_accel_mask, simple_sprint_times, accel_sprint_times)

        # Add reaction time (same as: times[p, g] = reaction_time + sprint_time)
        total_times = reaction_times_expanded + sprint_times

        # Handle very close cases (same as: if distance < 1e-6: times[p, g] = reaction_time)
        total_times = np.where(very_close_mask, reaction_times_expanded, total_times)

        return total_times

    @staticmethod
    def calculate_ball_travel_times(
        ball_position: np.ndarray, grid_points: np.ndarray, ball_speed: float
    ) -> np.ndarray:
        """Calculate ball travel times using pure NumPy"""
        distances = np.sqrt(np.sum((grid_points - ball_position) ** 2, axis=1))
        return distances / ball_speed
