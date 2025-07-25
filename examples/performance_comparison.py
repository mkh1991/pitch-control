"""
Compare performance between different backends and configurations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pitch_control.core import Pitch, Point
from pitch_control.models import SpearmanModel, SpearmanConfig
from examples.basic_example import create_sample_players


def compare_backends():
    """Compare Numba vs NumPy performance"""

    players = create_sample_players()
    ball_position = Point(0, 0)
    pitch = Pitch()

    # Test different grid resolutions
    resolutions = [(42, 27), (84, 54), (105, 68), (168, 108)]
    numba_times = []
    numpy_times = []

    print("Performance Comparison: Numba vs NumPy")
    print("=" * 50)

    for res in resolutions:
        grid_size = res[0] * res[1]
        print(f"\nGrid Resolution: {res[0]}x{res[1]} ({grid_size} points)")

        # Test Numba backend
        config_numba = SpearmanConfig(grid_resolution=res, use_numba=True)
        model_numba = SpearmanModel(config=config_numba, pitch=pitch)

        # Warm up
        model_numba.calculate(players, ball_position)

        # Benchmark Numba
        start = time.time()
        for _ in range(5):
            model_numba.calculate(players, ball_position)
        numba_time = (time.time() - start) / 5
        numba_times.append(numba_time)

        # Test NumPy backend
        config_numpy = SpearmanConfig(grid_resolution=res, use_numba=False)
        model_numpy = SpearmanModel(config=config_numpy, pitch=pitch)

        # Benchmark NumPy
        start = time.time()
        for _ in range(5):
            model_numpy.calculate(players, ball_position)
        numpy_time = (time.time() - start) / 5
        numpy_times.append(numpy_time)

        speedup = numpy_time / numba_time
        print(f"  Numba:  {numba_time:.3f}s")
        print(f"  NumPy:  {numpy_time:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    grid_sizes = [r[0] * r[1] for r in resolutions]

    # Time comparison
    ax1.plot(grid_sizes, numba_times, "ro-", label="Numba", linewidth=2)
    ax1.plot(grid_sizes, numpy_times, "bo-", label="NumPy", linewidth=2)
    ax1.set_xlabel("Grid Points")
    ax1.set_ylabel("Calculation Time (seconds)")
    ax1.set_title("Performance Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Speedup
    speedups = [n / nb for n, nb in zip(numpy_times, numba_times)]
    ax2.plot(grid_sizes, speedups, "go-", linewidth=2, markersize=8)
    ax2.set_xlabel("Grid Points")
    ax2.set_ylabel("Speedup Factor")
    ax2.set_title("Numba Speedup vs NumPy")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="No speedup")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return resolutions, numba_times, numpy_times


def test_scalability():
    """Test how performance scales with number of players"""

    base_players = create_sample_players()
    ball_position = Point(0, 0)
    pitch = Pitch()
    config = SpearmanConfig(grid_resolution=(84, 54), use_numba=True)
    model = SpearmanModel(config=config, pitch=pitch)

    # Ensure we always have players from both teams
    home_players = [p for p in base_players if p.team == "Home"]
    away_players = [p for p in base_players if p.team == "Away"]

    # Test with balanced teams
    team_sizes = [4, 6, 8, 11]  # Players per team
    player_counts = [size * 2 for size in team_sizes]  # Total players
    times = []

    print("\nScalability Test: Performance vs Number of Players")
    print("=" * 55)

    for i, team_size in enumerate(team_sizes):
        # Take equal numbers from each team
        selected_home = home_players[:team_size]
        selected_away = away_players[:team_size]
        players = selected_home + selected_away

        n_players = len(players)

        # Warm up
        model.calculate(players, ball_position)

        # Benchmark
        start = time.time()
        for _ in range(3):
            model.calculate(players, ball_position)
        avg_time = (time.time() - start) / 3
        times.append(avg_time)

        points_per_sec = (84 * 54 * n_players) / avg_time
        print(
            f"  {n_players:2d} players ({team_size}v{team_size}): {avg_time:.3f}s ({points_per_sec:.0f} calculations/sec)"
        )

    # Plot scalability
    plt.figure(figsize=(8, 6))
    plt.plot(player_counts, times, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Players")
    plt.ylabel("Calculation Time (seconds)")
    plt.title("Performance Scalability")
    plt.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(player_counts, times, 1)
    p = np.poly1d(z)
    plt.plot(
        player_counts,
        p(player_counts),
        "r--",
        alpha=0.8,
        label=f"Linear fit (slope: {z[0]:.4f})",
    )
    plt.legend()
    plt.show()


def memory_usage_test():
    """Test memory usage for different configurations"""
    try:
        import psutil
        import os
    except ImportError:
        print("psutil not available, skipping memory test")
        return

    process = psutil.Process(os.getpid())
    players = create_sample_players()
    ball_position = Point(0, 0)
    pitch = Pitch()

    print("\nMemory Usage Test")
    print("=" * 20)

    resolutions = [(42, 27), (84, 54), (168, 108), (336, 216)]

    for res in resolutions:
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        config = SpearmanConfig(grid_resolution=res, use_numba=True)
        model = SpearmanModel(config=config, pitch=pitch)

        # Calculate and measure peak memory
        result = model.calculate(players, ball_position)
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_used = peak_memory - baseline_memory
        grid_points = res[0] * res[1]
        memory_per_point = memory_used / grid_points * 1024  # KB per grid point

        print(
            f"  {res[0]:3d}x{res[1]:3d} ({grid_points:6d} points): "
            f"{memory_used:5.1f}MB ({memory_per_point:.2f}KB/point)"
        )

        # Clean up
        del model, result


if __name__ == "__main__":
    compare_backends()
    test_scalability()
    memory_usage_test()
