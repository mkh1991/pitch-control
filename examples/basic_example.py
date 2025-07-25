"""
Basic example of using the vectorized pitch control model.
"""

import numpy as np
import matplotlib.pyplot as plt

from pitch_control.core import Player, Point, Pitch, PlayerPhysics, Position
from pitch_control.models import SpearmanModel, SpearmanConfig
from pitch_control.utils.visualization import plot_combined_control
import random


def create_sample_players():
    """Create a sample set of players for demonstration"""
    players = []

    # Home team (left side)
    home_positions = [
        (-45, 0),  # GK
        (-30, -15),
        (-30, 15),  # CBs
        (-20, -25),
        (-20, 25),  # FBs
        (-10, -10),
        (-10, 10),  # CMs
        (0, -20),
        (0, 20),  # Wings
        (10, -5),
        (15, 5),  # Forwards
    ]

    home_velocities = [
        (0, 0),  # GK stationary
        (2, 1),
        (2, -1),  # CBs moving up
        (3, 2),
        (3, -2),  # FBs pushing forward
        (1, 3),
        (1, -3),  # CMs spreading
        (4, 1),
        (4, -1),  # Wings running
        (2, 2),
        (3, -1),  # Forwards
    ]

    positions_types = [
        Position.GK,
        Position.CB,
        Position.CB,
        Position.FB,
        Position.FB,
        Position.CM,
        Position.CM,
        Position.WM,
        Position.WM,
        Position.FW,
        Position.CF,
    ]

    for i, ((x, y), (vx, vy), pos_type) in enumerate(
        zip(home_positions, home_velocities, positions_types)
    ):
        physics = PlayerPhysics.for_position(pos_type)
        player = Player(
            f"H{i + 1}", f"Home Player {i + 1}", "Home", pos_type, i + 1, physics
        )
        player_state = player.create_state(Point(x, y), Point(vx, vy))
        players.append(player_state)

    # Away team (right side, mirrored, with some randomness)
    away_positions = [(x, -y + random.randint(-5, 5)) for x, y in home_positions]  #
    away_velocities = [(-vx, -vy) for vx, vy in home_velocities]  # Reverse velocities

    for i, ((x, y), (vx, vy), pos_type) in enumerate(
        zip(away_positions, away_velocities, positions_types)
    ):
        physics = PlayerPhysics.for_position(pos_type)
        player = Player(
            f"A{i + 1}", f"Away Player {i + 1}", "Away", pos_type, i + 1, physics
        )
        player_state = player.create_state(Point(x, y), Point(vx, vy))
        players.append(player_state)

    return players


def main():
    """Run basic pitch control example"""
    print("Setting up pitch control calculation...")

    # Create pitch and model
    pitch = Pitch()
    config = SpearmanConfig(
        grid_resolution=(105*2, 68*2),  # Reduced for faster calculation
        use_numba=True,
        parallel=True,
    )
    model = SpearmanModel(config=config, pitch=pitch)

    # Create players
    players = create_sample_players()
    print(
        f"Created {len(players)} players ({len([p for p in players if p.team == 'Home'])} home, "
        f"{len([p for p in players if p.team == 'Away'])} away)"
    )

    # Ball position (slightly towards home side)
    ball_position = Point(-5, 2)

    print("Calculating pitch control...")

    # Calculate pitch control
    result = model.calculate(players, ball_position)

    print(f"Calculation completed in {result.calculation_time:.3f} seconds")
    print(f"Grid resolution: {result.metadata['grid_resolution']}")
    print(f"Used Numba: {result.metadata['use_numba']}")

    # Display statistics
    home_control_pct = result.get_team_control_percentage("home")
    away_control_pct = result.get_team_control_percentage("away")

    print(f"Home team controls {home_control_pct:.1f}% of the pitch")
    print(f"Away team controls {away_control_pct:.1f}% of the pitch")

    # Visualize
    print("Creating visualization...")
    fig = plot_combined_control(result, "Vectorized Spearman Pitch Control")

    # Add ball position
    ax = fig.gca()
    ax.plot(ball_position.x, ball_position.y, "ko", markersize=8, label="Ball")

    # Add player positions
    home_players = [p for p in players if p.team == "Home"]
    away_players = [p for p in players if p.team == "Away"]

    home_x = [p.position.x for p in home_players]
    home_y = [p.position.y for p in home_players]
    away_x = [p.position.x for p in away_players]
    away_y = [p.position.y for p in away_players]

    ax.scatter(home_x, home_y, c="darkred", s=60, alpha=0.8, label="Home Players")
    ax.scatter(away_x, away_y, c="darkblue", s=60, alpha=0.8, label="Away Players")

    # Add velocity vectors
    for player in players[:6]:  # Show vectors for subset to avoid clutter
        dx = player.velocity.x * 2  # Scale for visibility
        dy = player.velocity.y * 2
        color = "darkred" if player.team == "Home" else "darkblue"
        ax.arrow(
            player.position.x,
            player.position.y,
            dx,
            dy,
            head_width=1,
            head_length=0.5,
            fc=color,
            ec=color,
            alpha=0.6,
        )

    ax.legend()
    plt.show()

    # Performance benchmark
    print("\nRunning performance benchmark...")
    benchmark_performance(model, players, ball_position)


def benchmark_performance(model, players, ball_position, n_runs=10):
    """Benchmark the model performance"""
    import time

    times = []
    for i in range(n_runs):
        start = time.time()
        result = model.calculate(players, ball_position)
        end = time.time()
        times.append(end - start)
        print(f"Run {i + 1}: {times[-1]:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nBenchmark Results ({n_runs} runs):")
    print(f"Average time: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"Fastest run: {min(times):.3f} seconds")
    print(
        f"Grid points: {model.config.grid_resolution[0] * model.config.grid_resolution[1]}"
    )
    print(
        f"Points per second: {(model.config.grid_resolution[0] * model.config.grid_resolution[1] * len(players)) / avg_time:.0f}"
    )


if __name__ == "__main__":
    main()
