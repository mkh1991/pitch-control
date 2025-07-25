"""
Advanced Pitch Control Model - Core Module
===========================================

A high-performance, extensible implementation of football pitch control models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
#
from .player import Player, PlayerPhysics, PlayerState, Position
from .pitch import Pitch, Point, ControlSurface
from .physics import PhysicsEngine, BallPhysics
from .control import PitchControlModel, PitchControlResult

__all__ = [
    "Player", "PlayerPhysics", "PlayerState", "Position",
    "Pitch", "Point", "ControlSurface",
    "PhysicsEngine", "BallPhysics",
    "PitchControlModel", "PitchControlResult",
]