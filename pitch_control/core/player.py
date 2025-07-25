from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from .pitch import Point


class Position(Enum):
    """Player positions"""

    GK = "Goalkeeper"
    CB = "Centre Back"
    FB = "Full Back"
    WB = "Wing Back"
    CM = "Central Midfielder"
    DM = "Defensive Midfielder"
    AM = "Attacking Midfielder"
    WM = "Wide Midfielder"
    FW = "Forward"
    CF = "Centre Forward"


@dataclass
class PlayerPhysics:
    """Physical attributes and capabilities of a player"""

    max_speed: float = 8.0  # m/s
    acceleration: float = 4.0  # m/s²
    reaction_time: float = 0.5  # seconds
    deceleration_factor: float = 0.8  # for direction changes
    fatigue_factor: float = 1.0  # 0-1 based on game time/fitness
    height: float = 1.80  # meters (affects aerial ability)
    agility: float = 1.0  # 0-1, affects turning speed

    @classmethod
    def for_position(cls, position: Position) -> "PlayerPhysics":
        """Create physics parameters based on typical position requirements"""
        position_params = {
            Position.GK: cls(max_speed=6.0, acceleration=3.0, reaction_time=0.3),
            Position.CB: cls(max_speed=7.0, acceleration=3.5, reaction_time=0.6),
            Position.FB: cls(max_speed=8.5, acceleration=4.2, reaction_time=0.5),
            Position.WB: cls(max_speed=9.0, acceleration=4.5, reaction_time=0.45),
            Position.CM: cls(max_speed=8.0, acceleration=4.0, reaction_time=0.5),
            Position.DM: cls(max_speed=7.5, acceleration=3.8, reaction_time=0.55),
            Position.AM: cls(max_speed=8.2, acceleration=4.2, reaction_time=0.45),
            Position.WM: cls(max_speed=8.8, acceleration=4.3, reaction_time=0.4),
            Position.FW: cls(max_speed=8.5, acceleration=4.4, reaction_time=0.4),
            Position.CF: cls(max_speed=8.0, acceleration=4.0, reaction_time=0.45),
        }
        return position_params.get(position, cls())


@dataclass
class PlayerState:
    """Current state of a player at a specific moment"""

    player_id: str
    position: Point
    velocity: Point
    team: str  # 'Home' or 'Away'
    jersey_number: int
    position_type: Position
    physics: PlayerPhysics = field(default_factory=PlayerPhysics)
    timestamp: float = 0.0

    def time_to_intercept(self, target: Point, ball_arrival_time: float = 0.0) -> float:
        """
        Calculate time for player to reach target position.

        Uses realistic physics with acceleration phase and reaction time.

        Args:
            target: Target position to reach
            ball_arrival_time: When ball will arrive at target (for comparison)

        Returns:
            Time in seconds to reach target
        """
        # Distance to target
        distance = self.position.distance(target)

        # Current speed in direction of target
        direction = Point(target.x - self.position.x, target.y - self.position.y)
        distance_norm = max(distance, 1e-6)  # Avoid division by zero
        unit_direction = Point(direction.x / distance_norm, direction.y / distance_norm)

        # Current velocity component toward target
        current_speed_toward_target = max(
            0.0, self.velocity.x * unit_direction.x + self.velocity.y * unit_direction.y
        )

        # Account for reaction time
        reaction_distance = current_speed_toward_target * self.physics.reaction_time
        remaining_distance = max(0.0, distance - reaction_distance)

        if remaining_distance == 0:
            return self.physics.reaction_time

        # Calculate time with acceleration
        max_speed = self.physics.max_speed * self.physics.fatigue_factor
        acceleration = self.physics.acceleration * self.physics.fatigue_factor

        # Time to reach max speed
        speed_diff = max_speed - current_speed_toward_target
        time_to_max_speed = speed_diff / acceleration if speed_diff > 0 else 0
        distance_during_acceleration = (
            current_speed_toward_target * time_to_max_speed
            + 0.5 * acceleration * time_to_max_speed**2
        )

        if distance_during_acceleration >= remaining_distance:
            # Reach target before reaching max speed
            # Solve: distance = v0*t + 0.5*a*t^2
            a, b, c = (
                0.5 * acceleration,
                current_speed_toward_target,
                -remaining_distance,
            )
            discriminant = b**2 + 4 * a * c
            if discriminant >= 0:
                time_accelerating = (
                    (-b + np.sqrt(discriminant)) / (2 * a)
                    if a > 0
                    else remaining_distance / max(current_speed_toward_target, 0.1)
                )
            else:
                time_accelerating = remaining_distance / max(
                    current_speed_toward_target, 0.1
                )
        else:
            # Accelerate to max speed, then constant speed
            remaining_at_max_speed = remaining_distance - distance_during_acceleration
            time_at_max_speed = remaining_at_max_speed / max_speed
            time_accelerating = time_to_max_speed + time_at_max_speed

        return self.physics.reaction_time + time_accelerating

    def control_probability(
        self,
        target: Point,
        arrival_time: float,
        opponent_arrival_time: float = float("inf"),
    ) -> float:
        """
        Calculate probability of controlling ball at target given arrival time.

        Args:
            target: Position where ball will be
            arrival_time: When this player arrives
            opponent_arrival_time: When nearest opponent arrives

        Returns:
            Probability (0-1) of gaining control
        """
        time_advantage = opponent_arrival_time - arrival_time

        # Sigmoid function to convert time advantage to probability
        # Parameters tuned based on empirical analysis
        sigma = 0.45  # Controls steepness of probability curve
        probability = 1.0 / (1.0 + np.exp(-time_advantage / sigma))

        # Adjust for position-specific abilities
        position_factors = {
            Position.GK: 1.2,  # Better hands, positioning in box
            Position.CB: 1.0,  # Standard
            Position.FB: 0.95,  # Slightly less comfortable in tight spaces
            Position.CM: 1.1,  # Good all-around
            Position.FW: 1.05,  # Good close control
        }

        position_bonus = position_factors.get(self.position_type, 1.0)
        probability *= position_bonus

        # Fatigue effect
        probability *= 0.7 + 0.3 * self.physics.fatigue_factor

        return np.clip(probability, 0.0, 1.0)


@dataclass
class Player:
    """Complete player representation combining static info and current state"""

    player_id: str
    name: str
    team: str
    position_type: Position
    jersey_number: int
    physics: PlayerPhysics = field(default_factory=PlayerPhysics)

    def create_state(
        self, position: Point, velocity: Point, timestamp: float = 0.0
    ) -> PlayerState:
        """Create a PlayerState instance for this player"""
        return PlayerState(
            player_id=self.player_id,
            position=position,
            velocity=velocity,
            team=self.team,
            jersey_number=self.jersey_number,
            position_type=self.position_type,
            physics=self.physics,
            timestamp=timestamp,
        )
