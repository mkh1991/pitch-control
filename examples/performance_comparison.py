"""
Compare performance between different backends and configurations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pitch_control.core import Pitch, Point
from pitch_control.models import SpearmanModel, SpearmanConfig
from examples.basic_example import create_sample_players
from typing import List, Tuple


def profile_backend(
    backend: str,
    players: List = None,
    ball_position: Point = None,
    grid_resolution: Tuple[int, int] = (84, 54),
    n_runs: int = 3,
):
    """
    Profile a specific backend (numba or numpy) with detailed timing.

    Args:
        backend: "numba" or "numpy"
        players: List of PlayerState objects (if None, creates sample players)
        ball_position: Ball position Point (if None, uses (0, 0))
        grid_resolution: Grid size for calculation
        n_runs: Number of runs to average over
    """
    # Set up defaults
    if players is None:
        from examples.basic_example import create_sample_players

        players = create_sample_players()

    if ball_position is None:
        ball_position = Point(0, 0)

    use_numba = backend.lower() == "numba"
    pitch = Pitch()
    config = SpearmanConfig(grid_resolution=grid_resolution, use_numba=use_numba)
    model = SpearmanModel(config=config, pitch=pitch)

    # Proper warm-up for Numba (critical for accurate benchmarking)
    if use_numba:
        print(f"Warming up {backend} (compiling functions)...")
        # Multiple warm-up runs to ensure compilation is complete
        for i in range(3):
            start_warmup = time.time()
            model.calculate(players, ball_position)
            warmup_time = time.time() - start_warmup
            print(f"  Warm-up run {i + 1}: {warmup_time * 1000:.1f}ms")

        # Additional warm-up for individual components to ensure they're compiled
        positions, velocities, max_speeds, accelerations, reaction_times, team_ids = (
            model._prepare_player_data(players)
        )
        grid_x, grid_y = model.pitch.create_grid(model.config.grid_resolution)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        ball_pos_array = np.array([ball_position.x, ball_position.y])

        # Force compilation of individual Numba functions
        from pitch_control.models.spearman import (
            calculate_times_vectorized,
            calculate_ball_travel_times_numba,
            _calc_control_probs_numba,
        )

        _ = calculate_times_vectorized(
            positions,
            velocities,
            grid_points,
            max_speeds,
            accelerations,
            reaction_times,
        )
        _ = calculate_ball_travel_times_numba(
            ball_pos_array, grid_points, model.spearman_config.average_ball_speed
        )
        adjusted_times = np.random.random((len(players), len(grid_points))) * 2.0
        _ = _calc_control_probs_numba(
            adjusted_times, team_ids, model.spearman_config.sigma
        )

        print(f"  Compilation complete!")

    print(f"\nProfiling {backend.upper()} backend ({n_runs} runs):")
    print("=" * 50)

    # Storage for timing results
    times = {
        "data_prep": [],
        "grid_creation": [],
        "physics": [],
        "ball": [],
        "probability": [],
        "total_core": [],
    }

    for run in range(n_runs):
        print(f"Run {run + 1}:")

        # Time data preparation
        start = time.time()
        positions, velocities, max_speeds, accelerations, reaction_times, team_ids = (
            model._prepare_player_data(players)
        )
        data_prep_time = time.time() - start
        times["data_prep"].append(data_prep_time)

        # Time grid creation
        start = time.time()
        grid_x, grid_y = model.pitch.create_grid(model.config.grid_resolution)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        grid_time = time.time() - start
        times["grid_creation"].append(grid_time)

        # Time player time calculation (backend-specific)
        start = time.time()
        player_times = model._calculate_player_times(
            positions,
            velocities,
            grid_points,
            max_speeds,
            accelerations,
            reaction_times,
        )
        physics_time = time.time() - start
        times["physics"].append(physics_time)

        # Time ball calculation (backend-specific)
        start = time.time()
        ball_pos_array = np.array([ball_position.x, ball_position.y])
        ball_times = model._calculate_ball_travel_times(
            ball_pos_array,
            grid_points,
        )

        adjusted_times = player_times + ball_times[np.newaxis, :]
        ball_time = time.time() - start
        times["ball"].append(ball_time)

        # Time probability calculation (backend-specific)
        start = time.time()
        home_control_flat, away_control_flat = model._calculate_control_probs(
            adjusted_times, team_ids, model.spearman_config.sigma
        )
        prob_time = time.time() - start
        times["probability"].append(prob_time)

        # Calculate total core computation time
        total_core = physics_time + ball_time + prob_time
        times["total_core"].append(total_core)

        # Print this run's results
        print(f"  Data preparation: {data_prep_time * 1000:.1f}ms")
        print(f"  Grid creation: {grid_time * 1000:.1f}ms")
        print(f"  Physics calculation: {physics_time * 1000:.1f}ms")
        print(f"  Ball calculation: {ball_time * 1000:.1f}ms")
        print(f"  Probability calculation: {prob_time * 1000:.1f}ms")
        print(f"  Total core computation: {total_core * 1000:.1f}ms")
        print()

    # Calculate and print averages
    print(f"AVERAGE RESULTS ({n_runs} runs):")
    print("-" * 30)
    avg_data_prep = np.mean(times["data_prep"]) * 1000
    avg_grid = np.mean(times["grid_creation"]) * 1000
    avg_physics = np.mean(times["physics"]) * 1000
    avg_ball = np.mean(times["ball"]) * 1000
    avg_prob = np.mean(times["probability"]) * 1000
    avg_total = np.mean(times["total_core"]) * 1000

    print(f"Data preparation: {avg_data_prep:.1f}ms")
    print(f"Grid creation: {avg_grid:.1f}ms")
    print(f"Physics calculation: {avg_physics:.1f}ms")
    print(f"Ball calculation: {avg_ball:.1f}ms")
    print(f"Probability calculation: {avg_prob:.1f}ms")
    print(f"Total core computation: {avg_total:.1f}ms")

    # Return the averages for comparison
    return {
        "data_prep": avg_data_prep,
        "grid_creation": avg_grid,
        "physics": avg_physics,
        "ball": avg_ball,
        "probability": avg_prob,
        "total_core": avg_total,
    }


def compare_backends(n_runs=20):
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

        # Profile NumPy
        numpy_results = profile_backend("numpy", players, ball_position, res, n_runs)

        # Profile Numba
        numba_results = profile_backend("numba", players, ball_position, res, n_runs)
        # Print comparison
        print("\nCOMPARISON SUMMARY:")
        print("=" * 50)
        print(
            f"{'Component':<20} {'NumPy (ms)':<12} {'Numba (ms)':<12} {'Speedup':<10}"
        )
        print("-" * 54)

        components = ["physics", "ball", "probability", "total_core"]
        for comp in components:
            numpy_time = numpy_results[comp]
            numba_time = numba_results[comp]
            speedup = numpy_time / numba_time if numba_time > 0 else 0

            comp_name = comp.replace("_", " ").title()
            if comp == "total_core":
                comp_name = "TOTAL CORE"
                print("-" * 54)

            print(
                f"{comp_name:<20} {numpy_time:>8.1f}     {numba_time:>8.1f}     {speedup:>6.1f}x"
            )

        numba_time, numpy_time = (
            numba_results["total_core"],
            numpy_results["total_core"],
        )
        speedup = numpy_time / numba_time
        numba_times.append(numba_time)
        numpy_times.append(numpy_time)

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
