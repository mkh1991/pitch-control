from dataclasses import dataclass
from typing import List, Tuple
import time
import numpy as np
import numba
from numba import jit, prange

from ..speedup import NumpyVectorizedBackend
from ..core import PlayerState, Point, ControlSurface, PitchControlResult, Pitch
from .base import OptimizedPitchControlModel, ModelConfig


@dataclass
class SpearmanConfig(ModelConfig):
    """Configuration for Spearman pitch control model"""

    sigma: float = 0.45  # Controls steepness of probability curve
    lambda_att: float = 4.3  # Attacking advantage parameter
    lambda_def: float = 4.3  # Defensive advantage parameter
    average_ball_speed: float = 15.0  # m/s
    integration_window: float = 10.0  # seconds


# Numba-compiled functions for maximum performance
@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _calculate_times_vectorized(
    player_positions,
    player_velocities,
    grid_points,
    max_speeds,
    accelerations,
    reaction_times,
):
    """
    Corrected implementation that follows Laurie Shaw's approach exactly
    but includes optional acceleration (though Laurie notes it's "not used")
    """
    n_players, n_grid = player_positions.shape[0], grid_points.shape[0]
    times = np.zeros((n_players, n_grid))

    for p in prange(n_players):
        pos_x, pos_y = player_positions[p, 0], player_positions[p, 1]
        vel_x, vel_y = player_velocities[p, 0], player_velocities[p, 1]
        max_speed = max_speeds[p]
        acceleration = accelerations[p]
        reaction_time = reaction_times[p]

        for g in range(n_grid):
            target_x, target_y = grid_points[g, 0], grid_points[g, 1]

            # Step 1: Continue at current velocity during reaction time
            reaction_pos_x = pos_x + vel_x * reaction_time
            reaction_pos_y = pos_y + vel_y * reaction_time

            # Step 2: Calculate distance from reaction position to target
            dx = target_x - reaction_pos_x
            dy = target_y - reaction_pos_y
            distance = np.sqrt(dx * dx + dy * dy)

            if distance < 1e-6:
                times[p, g] = reaction_time
                continue

            # Step 3: Sprint from reaction position to target
            # Laurie's implementation: just use max speed (no acceleration)
            if acceleration <= 0:
                sprint_time = distance / max(max_speed, 0.1)
                times[p, g] = reaction_time + sprint_time
            else:
                time_to_max_speed = max_speed / acceleration
                distance_during_accel = 0.5 * acceleration * time_to_max_speed * time_to_max_speed

                if distance_during_accel >= distance:
                    # Reach target during acceleration
                    accel_time = np.sqrt(2.0 * distance / acceleration)
                    times[p, g] = reaction_time + accel_time
                else:
                    # Accelerate then constant speed
                    remaining_distance = distance - distance_during_accel
                    const_time = remaining_distance / max_speed
                    times[p, g] = reaction_time + time_to_max_speed + const_time

    return times


@jit(nopython=True, parallel=True, cache=True)
def _calculate_ball_travel_times(ball_position, grid_points, ball_speed):
    """
    Calculate ball travel times to all grid points.

    Args:
        ball_position: (2,) array of ball position
        grid_points: (n_grid, 2) array of grid coordinates
        ball_speed: scalar ball speed

    Returns:
        (n_grid,) array of ball travel times
    """
    n_grid = grid_points.shape[0]
    times = np.zeros(n_grid)

    ball_x, ball_y = ball_position[0], ball_position[1]

    for g in prange(n_grid):
        target_x, target_y = grid_points[g, 0], grid_points[g, 1]
        distance = np.sqrt((target_x - ball_x) ** 2 + (target_y - ball_y) ** 2)
        times[g] = distance / ball_speed

    return times


@jit(nopython=True, parallel=True, cache=True)
def _calculate_control_probabilities(player_times, team_ids, sigma):
    """
    Calculate control probabilities using logistic function.

    Args:
        player_times: (n_players, n_grid) array of arrival times
        team_ids: (n_players,) array of team IDs (0 for home, 1 for away)
        sigma: logistic function parameter

    Returns:
        Tuple of (home_control, away_control) arrays of shape (n_grid,)
    """
    n_players, n_grid = player_times.shape
    home_control = np.zeros(n_grid)
    away_control = np.zeros(n_grid)

    for g in prange(n_grid):
        min_home_time = np.inf
        min_away_time = np.inf

        # Find fastest player from each team
        for p in range(n_players):
            time = player_times[p, g]
            if team_ids[p] == 0:  # Home team
                if time < min_home_time:
                    min_home_time = time
            else:  # Away team
                if time < min_away_time:
                    min_away_time = time

        # Calculate probabilities using logistic function
        if min_home_time == np.inf:
            home_prob = 0.0
        elif min_away_time == np.inf:
            home_prob = 1.0
        else:
            time_diff = min_away_time - min_home_time
            home_prob = 1.0 / (1.0 + np.exp(-time_diff / sigma))

        home_control[g] = home_prob
        away_control[g] = 1.0 - home_prob

    return home_control, away_control


class SpearmanModel(OptimizedPitchControlModel):
    """
    High-performance implementation of Spearman's pitch control model.

    Features:
    - Full vectorization with NumPy
    - Numba JIT compilation for critical loops
    - Parallel processing for grid calculations
    - Realistic physics with acceleration phases
    """

    def __init__(self, config: SpearmanConfig = None, **kwargs):
        config = config or SpearmanConfig()
        if "pitch" not in kwargs:
            kwargs["pitch"] = Pitch()
        super().__init__(config, **kwargs)
        self.spearman_config = config

        # Warm up Numba compilation
        if config.use_numba:
            self._warm_up_numba()

    def _prepare_player_data(
        self, players: List[PlayerState]
    ) -> Tuple[np.ndarray, ...]:
        """
        Convert player list to vectorized arrays for fast computation.

        Returns:
            Tuple of arrays: positions, velocities, max_speeds, accelerations,
                           reaction_times, team_ids
        """
        n_players = len(players)

        positions = np.zeros((n_players, 2))
        velocities = np.zeros((n_players, 2))
        max_speeds = np.zeros(n_players)
        accelerations = np.zeros(n_players)
        reaction_times = np.zeros(n_players)
        team_ids = np.zeros(n_players, dtype=np.int32)

        for i, player in enumerate(players):
            positions[i] = [player.position.x, player.position.y]
            velocities[i] = [player.velocity.x, player.velocity.y]
            max_speeds[i] = player.physics.max_speed * player.physics.fatigue_factor
            accelerations[i] = (
                player.physics.acceleration * player.physics.fatigue_factor
            )
            reaction_times[i] = player.physics.reaction_time
            team_ids[i] = 0 if player.team.lower() == "home" else 1

        return (
            positions,
            velocities,
            max_speeds,
            accelerations,
            reaction_times,
            team_ids,
        )

    def _calculate_times_vectorized(
        self,
        positions,
        velocities,
        grid_points,
        max_speeds,
        accelerations,
        reaction_times,
    ):
        return _calculate_times_vectorized(
            positions,
            velocities,
            grid_points,
            max_speeds,
            accelerations,
            reaction_times,
        )

    def calculate(
        self, players: List[PlayerState], ball_position: Point, **kwargs
    ) -> PitchControlResult:
        """
        Calculate pitch control using vectorized Spearman model.

        Args:
            players: List of all players on pitch
            ball_position: Current ball position
            **kwargs: Additional parameters

        Returns:
            PitchControlResult with control surface and metadata
        """
        start_time = time.time()

        # Validate inputs
        self.validate_inputs(players, ball_position)

        # Create grid
        grid_x, grid_y = self.pitch.create_grid(self.config.grid_resolution)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # Prepare player data for vectorized operations
        (positions, velocities, max_speeds, accelerations, reaction_times, team_ids) = (
            self._prepare_player_data(players)
        )

        # Calculate player arrival times (vectorized)
        if self.config.use_numba:
            player_times = _calculate_times_vectorized(
                positions,
                velocities,
                grid_points,
                max_speeds,
                accelerations,
                reaction_times,
            )
        else:
            player_times = NumpyVectorizedBackend.calculate_times_vectorized(
                positions,
                velocities,
                grid_points,
                max_speeds,
                accelerations,
                reaction_times,
            )

        # Calculate ball travel times
        ball_pos_array = np.array([ball_position.x, ball_position.y])
        if self.config.use_numba:
            ball_times = _calculate_ball_travel_times(
                ball_pos_array, grid_points, self.spearman_config.average_ball_speed
            )
        else:
            ball_times = self._calculate_ball_times_numpy(ball_pos_array, grid_points)

        # Adjust player times for ball travel time
        adjusted_times = player_times + ball_times[np.newaxis, :]

        # home_control_flat, away_control_flat = _calculate_control_probabilities(
        #     adjusted_times, team_ids, self.spearman_config.sigma
        # )

        # Calculate control probabilities
        if self.config.use_numba:
            home_control_flat, away_control_flat = _calculate_control_probabilities(
                adjusted_times, team_ids, self.spearman_config.sigma
            )
        else:
            home_control_flat, away_control_flat = self._calculate_probabilities_numpy(
                adjusted_times, team_ids
            )

        # Reshape to grid
        grid_shape = grid_x.shape
        home_control = home_control_flat.reshape(grid_shape)
        away_control = away_control_flat.reshape(grid_shape)

        # Create result
        control_surface = ControlSurface(
            home_control=home_control,
            away_control=away_control,
            grid_x=grid_x,
            grid_y=grid_y,
            timestamp=kwargs.get("timestamp", 0.0),
        )

        calculation_time = time.time() - start_time

        metadata = {
            "model": "Spearman",
            "grid_resolution": self.config.grid_resolution,
            "n_players": len(players),
            "use_numba": self.config.use_numba,
            "ball_position": (ball_position.x, ball_position.y),
            "config": self.spearman_config,
        }

        return PitchControlResult(
            control_surface=control_surface,
            calculation_time=calculation_time,
            metadata=metadata,
        )

    def _calculate_times_numpy(
        self,
        positions,
        velocities,
        grid_points,
        max_speeds,
        accelerations,
        reaction_times,
    ):
        """NumPy fallback for time calculations (slower than Numba)"""
        n_players, _ = positions.shape
        n_grid, _ = grid_points.shape

        # Broadcast for vectorized distance calculation
        # Shape: (n_players, n_grid, 2)
        pos_expanded = positions[:, np.newaxis, :]
        grid_expanded = grid_points[np.newaxis, :, :]

        # Calculate distances: (n_players, n_grid)
        diff = grid_expanded - pos_expanded
        distances = np.sqrt(np.sum(diff**2, axis=2))

        # Direction vectors: (n_players, n_grid, 2)
        directions = np.zeros_like(diff)
        mask = distances > 1e-6
        directions[mask] = diff[mask] / distances[mask, np.newaxis]

        # Current velocity toward target: (n_players, n_grid)
        vel_expanded = velocities[:, np.newaxis, :]
        vel_toward = np.maximum(0, np.sum(vel_expanded * directions, axis=2))

        # Physics calculations (vectorized)
        reaction_distances = vel_toward * reaction_times[:, np.newaxis]
        remaining_distances = np.maximum(0, distances - reaction_distances)

        # Simplified time calculation (can be made more sophisticated)
        effective_speeds = np.minimum(
            max_speeds[:, np.newaxis], vel_toward + accelerations[:, np.newaxis] * 2.0
        )

        times = reaction_times[:, np.newaxis] + remaining_distances / np.maximum(
            effective_speeds, 0.1
        )

        return times

    def _calculate_ball_times_numpy(self, ball_position, grid_points):
        """NumPy fallback for ball travel times"""
        distances = np.sqrt(np.sum((grid_points - ball_position) ** 2, axis=1))
        return distances / self.spearman_config.average_ball_speed

    def _calculate_probabilities_numpy(self, player_times, team_ids):
        """NumPy fallback for probability calculations"""
        n_grid = player_times.shape[1]
        home_control = np.zeros(n_grid)
        away_control = np.zeros(n_grid)

        home_mask = team_ids == 0
        away_mask = team_ids == 1

        for g in range(n_grid):
            home_times = player_times[home_mask, g]
            away_times = player_times[away_mask, g]

            min_home = np.min(home_times) if len(home_times) > 0 else np.inf
            min_away = np.min(away_times) if len(away_times) > 0 else np.inf

            if min_home == np.inf:
                home_prob = 0.0
            elif min_away == np.inf:
                home_prob = 1.0
            else:
                time_diff = min_away - min_home
                home_prob = 1.0 / (
                    1.0 + np.exp(-time_diff / self.spearman_config.sigma)
                )

            home_control[g] = home_prob
            away_control[g] = 1.0 - home_prob

        return home_control, away_control
