from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
from .pitch import Point, Pitch
from .player import PlayerState


@dataclass
class BallPhysics:
    """Ball movement physics parameters"""
    max_speed: float = 30.0  # m/s (professional kick)
    air_resistance: float = 0.05  # drag coefficient
    gravity: float = 9.81  # m/s² (for trajectory calculations)
    bounce_factor: float = 0.7  # energy retained after bounce

    def trajectory_time(self, start: Point, end: Point,
                        initial_speed: float = 15.0) -> float:
        """
        Calculate time for ball to travel from start to end.

        Uses simplified physics with air resistance.
        """
        distance = start.distance_to(end)
        if distance == 0:
            return 0.0

        # Simplified model: constant deceleration due to air resistance
        # v(t) = v0 * exp(-k*t) where k is drag coefficient
        # Integrate to get position
        k = self.air_resistance

        if k == 0:
            return distance / initial_speed

        # Solve: distance = (v0/k) * (1 - exp(-k*t))
        # This gives: t = -ln(1 - k*distance/v0) / k
        ratio = k * distance / initial_speed
        if ratio >= 1.0:
            # Ball doesn't have enough speed to reach target
            return float('inf')

        return -np.log(1 - ratio) / k

    def reachable_area(self, current_pos: Point, time_horizon: float = 3.0,
                       initial_speed: float = 15.0) -> float:
        """
        Calculate maximum distance ball can travel in given time.

        Returns:
            Maximum reachable distance in meters
        """
        k = self.air_resistance
        if k == 0:
            return initial_speed * time_horizon

        # Maximum distance with air resistance
        return (initial_speed / k) * (1 - np.exp(-k * time_horizon))


class PhysicsEngine:
    """Main physics calculation engine"""

    def __init__(self, ball_physics: BallPhysics = None):
        self.ball_physics = ball_physics or BallPhysics()

    def calculate_interception_times(self, players: List[PlayerState],
                                     target: Point,
                                     ball_arrival_time: float = 0.0) -> np.ndarray:
        """
        Calculate interception times for all players to a target point.

        Args:
            players: List of player states
            target: Target position
            ball_arrival_time: When ball arrives at target

        Returns:
            Array of interception times for each player
        """
        times = np.zeros(len(players))
        for i, player in enumerate(players):
            times[i] = player.time_to_intercept(target, ball_arrival_time)

        return times

    def find_controlling_player(self, players: List[PlayerState],
                                target: Point, ball_arrival_time: float = 0.0) -> Tuple[
        int, float]:
        """
        Find which player is most likely to control the ball at target.

        Returns:
            Tuple of (player_index, control_probability)
        """
        times = self.calculate_interception_times(players, target, ball_arrival_time)

        # Find fastest player
        fastest_idx = np.argmin(times)
        fastest_time = times[fastest_idx]

        # Find second fastest for probability calculation
        times_sorted = np.sort(times)
        second_fastest_time = times_sorted[1] if len(times_sorted) > 1 else float('inf')

        # Calculate control probability
        control_prob = players[fastest_idx].control_probability(
            target, fastest_time, second_fastest_time
        )

        return fastest_idx, control_prob