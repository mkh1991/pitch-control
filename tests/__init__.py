import pytest
import numpy as np
import time
from pitch_control.core import (
    Player,
    PlayerState,
    Point,
    Pitch,
    PlayerPhysics,
    Position,
)
from pitch_control.models import SpearmanModel, SpearmanConfig


class TestSpearmanModel:

    @pytest.fixture
    def basic_setup(self):
        """Basic test setup"""
        pitch = Pitch()
        config = SpearmanConfig(grid_resolution=(21, 14), use_numba=True)
        model = SpearmanModel(config=config, pitch=pitch)

        # Simple 2v2 scenario
        players = [
            PlayerState("H1", Point(-10, 0), Point(2, 0), "Home", 1, Position.CM),
            PlayerState("H2", Point(-5, 5), Point(1, -1), "Home", 2, Position.FW),
            PlayerState("A1", Point(10, 0), Point(-2, 0), "Away", 1, Position.CM),
            PlayerState("A2", Point(5, -5), Point(-1, 1), "Away", 2, Position.FW),
        ]

        ball_position = Point(0, 0)

        return model, players, ball_position

    def test_basic_calculation(self, basic_setup):
        """Test basic pitch control calculation"""
        model, players, ball_position = basic_setup

        result = model.calculate(players, ball_position)

        # Check result structure
        assert result.control_surface is not None
        assert result.calculation_time > 0
        assert "model" in result.metadata
        assert result.metadata["model"] == "Spearman"

        # Check control surface properties
        surface = result.control_surface
        assert surface.home_control.shape == surface.away_control.shape
        assert surface.home_control.shape == (
            14,
            21,
        )  # grid_resolution swapped due to meshgrid

        # Probabilities should sum to 1 (approximately)
        total_control = surface.home_control + surface.away_control
        np.testing.assert_allclose(total_control, 1.0, atol=1e-10)

        # All probabilities should be in [0, 1]
        assert np.all(surface.home_control >= 0)
        assert np.all(surface.home_control <= 1)
        assert np.all(surface.away_control >= 0)
        assert np.all(surface.away_control <= 1)

    def test_numba_vs_numpy_consistency(self, basic_setup):
        """Test that Numba and NumPy backends give same results"""
        model_numba, players, ball_position = basic_setup

        # Create NumPy model
        config_numpy = SpearmanConfig(grid_resolution=(21, 14), use_numba=False)
        model_numpy = SpearmanModel(config=config_numpy, pitch=model_numba.pitch)

        # Calculate with both
        result_numba = model_numba.calculate(players, ball_position)
        result_numpy = model_numpy.calculate(players, ball_position)

        # Results should be very close
        np.testing.assert_allclose(
            result_numba.control_surface.home_control,
            result_numpy.control_surface.home_control,
            rtol=1e-10,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            result_numba.control_surface.away_control,
            result_numpy.control_surface.away_control,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_performance_improvement(self, basic_setup):
        """Test that Numba is faster than NumPy"""
        model_numba, players, ball_position = basic_setup

        # Create larger grid for meaningful performance test
        config_large = SpearmanConfig(grid_resolution=(63, 42), use_numba=True)
        model_numba_large = SpearmanModel(config=config_large, pitch=model_numba.pitch)

        config_numpy = SpearmanConfig(grid_resolution=(63, 42), use_numba=False)
        model_numpy_large = SpearmanModel(config=config_numpy, pitch=model_numba.pitch)

        # Warm up Numba
        model_numba_large.calculate(players, ball_position)

        # Time Numba
        start = time.time()
        for _ in range(3):
            model_numba_large.calculate(players, ball_position)
        numba_time = (time.time() - start) / 3

        # Time NumPy
        start = time.time()
        for _ in range(3):
            model_numpy_large.calculate(players, ball_position)
        numpy_time = (time.time() - start) / 3

        speedup = numpy_time / numba_time
        print(
            f"Speedup: {speedup:.1f}x (Numba: {numba_time:.3f}s, NumPy: {numpy_time:.3f}s)"
        )

        # Numba should be faster (at least 1.5x for this grid size)
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.1f}x"

    def test_ball_position_effect(self, basic_setup):
        """Test that ball position affects control distribution"""
        model, players, _ = basic_setup

        # Ball near home team
        result_home = model.calculate(players, Point(-20, 0))
        home_control_near_home = np.mean(result_home.control_surface.home_control)

        # Ball near away team
        result_away = model.calculate(players, Point(20, 0))
        home_control_near_away = np.mean(result_away.control_surface.home_control)

        # Home team should have more control when ball is near them
        assert home_control_near_home > home_control_near_away

    def test_player_velocity_effect(self, basic_setup):
        """Test that player velocity affects control"""
        model, players, ball_position = basic_setup

        # Calculate with original velocities
        result_original = model.calculate(players, ball_position)

        # Increase home team velocities toward ball
        modified_players = []
        for player in players:
            if player.team == "Home":
                # Velocity toward ball
                direction = Point(
                    ball_position.x - player.position.x,
                    ball_position.y - player.position.y,
                )
                norm = max(direction.distance(Point(0, 0)), 1e-6)
                new_velocity = Point(direction.x / norm * 5, direction.y / norm * 5)

                modified_player = PlayerState(
                    player.player_id,
                    player.position,
                    new_velocity,
                    player.team,
                    player.jersey_number,
                    player.position_type,
                    player.physics,
                    player.timestamp,
                )
                modified_players.append(modified_player)
            else:
                modified_players.append(player)

        result_modified = model.calculate(modified_players, ball_position)

        # Home team should have more control with improved velocities
        original_home_control = np.mean(result_original.control_surface.home_control)
        modified_home_control = np.mean(result_modified.control_surface.home_control)

        assert modified_home_control > original_home_control

    def test_edge_cases(self, basic_setup):
        """Test edge cases and boundary conditions"""
        model, players, _ = basic_setup

        # Ball at player position
        result = model.calculate(players, players[0].position)
        assert result.control_surface is not None

        # Ball at pitch boundary
        bounds = model.pitch.dimensions.bounds
        corner = Point(bounds[0][1], bounds[1][1])  # Top-right corner
        result = model.calculate(players, corner)
        assert result.control_surface is not None

        # Players with zero velocity
        stationary_players = []
        for player in players:
            stationary_player = PlayerState(
                player.player_id,
                player.position,
                Point(0, 0),
                player.team,
                player.jersey_number,
                player.position_type,
                player.physics,
                player.timestamp,
            )
            stationary_players.append(stationary_player)

        result = model.calculate(stationary_players, Point(0, 0))
        assert result.control_surface is not None

    def test_configuration_parameters(self):
        """Test different configuration parameters"""
        pitch = Pitch()
        players = [
            PlayerState("H1", Point(-5, 0), Point(1, 0), "Home", 1, Position.CM),
            PlayerState("A1", Point(5, 0), Point(-1, 0), "Away", 1, Position.CM),
        ]
        ball_position = Point(0, 0)

        # Test different sigma values
        config1 = SpearmanConfig(sigma=0.2, grid_resolution=(21, 14))
        config2 = SpearmanConfig(sigma=0.8, grid_resolution=(21, 14))

        model1 = SpearmanModel(config=config1, pitch=pitch)
        model2 = SpearmanModel(config=config2, pitch=pitch)

        result1 = model1.calculate(players, ball_position)
        result2 = model2.calculate(players, ball_position)

        # Different sigma should give different results
        assert not np.allclose(
            result1.control_surface.home_control, result2.control_surface.home_control
        )

        # Lower sigma should create sharper boundaries
        uncertainty1 = np.mean(result1.control_surface.uncertainty)
        uncertainty2 = np.mean(result2.control_surface.uncertainty)
        assert uncertainty1 < uncertainty2  # Lower sigma = less uncertainty

    def test_metadata_completeness(self, basic_setup):
        """Test that result metadata is complete"""
        model, players, ball_position = basic_setup

        result = model.calculate(players, ball_position, timestamp=123.45)

        metadata = result.metadata
        required_keys = [
            "model",
            "grid_resolution",
            "n_players",
            "use_numba",
            "ball_position",
            "config",
        ]

        for key in required_keys:
            assert key in metadata, f"Missing metadata key: {key}"

        assert metadata["model"] == "Spearman"
        assert metadata["n_players"] == len(players)
        assert metadata["ball_position"] == (ball_position.x, ball_position.y)
        assert isinstance(metadata["config"], SpearmanConfig)
        assert result.control_surface.timestamp == 123.45
