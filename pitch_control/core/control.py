from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from .pitch import ControlSurface, Pitch, Point
from .player import PlayerState
from .physics import PhysicsEngine


@dataclass
class PitchControlResult:
    """Result of pitch control calculation"""
    control_surface: ControlSurface
    calculation_time: float
    metadata: Dict[str, Any]

    def get_team_control_percentage(self, team: str = "home") -> float:
        """Get percentage of pitch controlled by team"""
        if team.lower() == "home":
            control = self.control_surface.home_control
        else:
            control = self.control_surface.away_control

        return np.mean(control > 0.5) * 100.0

    def get_high_value_areas(self, threshold: float = 0.7) -> np.ndarray:
        """Get areas where team has high control"""
        return self.control_surface.home_control > threshold


class PitchControlModel(ABC):
    """Abstract base class for pitch control models"""

    def __init__(self, pitch: Pitch, physics_engine: PhysicsEngine = None):
        self.pitch = pitch
        self.physics_engine = physics_engine or PhysicsEngine()
        self.config = {}

    @abstractmethod
    def calculate(self, players: List[PlayerState],
                  ball_position: Point, **kwargs) -> PitchControlResult:
        """
        Calculate pitch control for given game state.

        Args:
            players: List of all players on pitch
            ball_position: Current ball position
            **kwargs: Model-specific parameters

        Returns:
            PitchControlResult with control surface and metadata
        """
        pass

    def set_config(self, **config):
        """Update model configuration"""
        self.config.update(config)

    def validate_inputs(self, players: List[PlayerState], ball_position: Point):
        """Validate inputs before calculation"""
        if not players:
            raise ValueError("No players provided")

        if not self.pitch.is_valid_position(ball_position):
            raise ValueError(f"Ball position {ball_position} is outside pitch bounds")

        # Check team distribution
        teams = set(player.team for player in players)
        if len(teams) != 2:
            raise ValueError(f"Expected 2 teams, got {len(teams)}: {teams}")