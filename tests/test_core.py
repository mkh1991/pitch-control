import pytest
import numpy as np
from pitch_control.core import (
    Player,
    PlayerState,
    Point,
    Pitch,
    PlayerPhysics,
    Position,
)


class TestPoint:
    def test_distance_calculation(self):
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert p1.distance(p2) == 5.0

    def test_point_arithmetic(self):
        p1 = Point(1, 2)
        p2 = Point(3, 4)

        result = p1 + p2
        assert result.x == 4 and result.y == 6

        result = p2 - p1
        assert result.x == 2 and result.y == 2


class TestPlayerPhysics:
    def test_position_specific_physics(self):
        gk_physics = PlayerPhysics.for_position(Position.GK)
        fw_physics = PlayerPhysics.for_position(Position.FW)

        # Goalkeepers should be slower but have better reaction times
        assert gk_physics.max_speed < fw_physics.max_speed
        assert gk_physics.reaction_time < fw_physics.reaction_time


class TestPlayerState:
    def test_time_to_intercept_stationary(self):
        player = PlayerState(
            player_id="test",
            position=Point(0, 0),
            velocity=Point(0, 0),
            team="Home",
            jersey_number=1,
            position_type=Position.CM,
        )

        target = Point(10, 0)  # 10 meters away
        time = player.time_to_intercept(target)

        # Should be reaction time + acceleration time
        assert time > player.physics.reaction_time
        assert time < 5.0  # Reasonable upper bound

    def test_control_probability(self):
        player = PlayerState(
            player_id="test",
            position=Point(0, 0),
            velocity=Point(0, 0),
            team="Home",
            jersey_number=1,
            position_type=Position.CM,
        )

        target = Point(1, 0)

        # Player arrives first should have high probability
        prob1 = player.control_probability(target, 1.0, 2.0)
        assert prob1 > 0.7

        # Player arrives second should have low probability
        prob2 = player.control_probability(target, 2.0, 1.0)
        assert prob2 < 0.3


class TestPitch:
    def test_grid_creation(self):
        pitch = Pitch(grid_resolution=(10, 6))
        X, Y = pitch.create_grid()

        assert X.shape == (6, 10)  # Note: Y dimension first
        assert Y.shape == (6, 10)

    def test_boundary_validation(self):
        pitch = Pitch()

        # Point inside pitch
        assert pitch.is_valid_position(Point(0, 0))

        # Point outside pitch
        assert not pitch.is_valid_position(Point(100, 100))
