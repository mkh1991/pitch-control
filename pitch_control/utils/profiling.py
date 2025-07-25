# File: pitch_control/utils/profiling.py

"""
Profiling utilities for comparing NumPy vs Numba performance.
"""

import time
import numpy as np
from typing import List
from ..core import Point, Pitch
from ..models import SpearmanModel, SpearmanConfig


def profile_backend(backend: str, players: List = None, ball_position: Point = None,
                    grid_resolution: tuple = (84, 54), n_runs: int = 20):
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

    # Warm up (especially important for Numba)
    if use_numba:
        print(f"Warming up {backend} (compiling functions)...")
        model.calculate(players, ball_position)
        model.calculate(players, ball_position)  # Second warm-up

    print(f"\nProfiling {backend.upper()} backend ({n_runs} runs):")
    print("=" * 50)

    # Storage for timing results
    times = {
        'data_prep': [],
        'grid_creation': [],
        'physics': [],
        'ball': [],
        'probability': [],
        'total_core': []
    }

    for run in range(n_runs):
        print(f"Run {run + 1}:")

        # Time data preparation
        start = time.time()
        positions, velocities, max_speeds, accelerations, reaction_times, team_ids = model._prepare_player_data(
            players)
        data_prep_time = time.time() - start
        times['data_prep'].append(data_prep_time)

        # Time grid creation
        start = time.time()
        grid_x, grid_y = model.pitch.create_grid(model.config.grid_resolution)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        grid_time = time.time() - start
        times['grid_creation'].append(grid_time)

        # Time player time calculation (backend-specific)
        start = time.time()
        if use_numba:
            from ..models.spearman import _calculate_times_vectorized
            player_times = _calculate_times_vectorized(
                positions, velocities, grid_points, max_speeds, accelerations,
                reaction_times
            )
        else:
            player_times = model._calculate_times_numpy(
                positions, velocities, grid_points, max_speeds, accelerations,
                reaction_times
            )
        physics_time = time.time() - start
        times['physics'].append(physics_time)

        # Time ball calculation (backend-specific)
        start = time.time()
        ball_pos_array = np.array([ball_position.x, ball_position.y])
        if use_numba:
            from ..models.spearman import _calculate_ball_travel_times
            ball_times = _calculate_ball_travel_times(
                ball_pos_array, grid_points, model.spearman_config.average_ball_speed
            )
        else:
            ball_times = model._calculate_ball_times_numpy(ball_pos_array, grid_points)

        adjusted_times = player_times + ball_times[np.newaxis, :]
        ball_time = time.time() - start
        times['ball'].append(ball_time)

        # Time probability calculation (backend-specific)
        start = time.time()
        if use_numba:
            from ..models.spearman import _calculate_control_probabilities
            home_control_flat, away_control_flat = _calculate_control_probabilities(
                adjusted_times, team_ids, model.spearman_config.sigma
            )
        else:
            home_control_flat, away_control_flat = model._calculate_probabilities_numpy(
                adjusted_times, team_ids
            )
        prob_time = time.time() - start
        times['probability'].append(prob_time)

        # Calculate total core computation time
        total_core = physics_time + ball_time + prob_time
        times['total_core'].append(total_core)

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
    avg_data_prep = np.mean(times['data_prep']) * 1000
    avg_grid = np.mean(times['grid_creation']) * 1000
    avg_physics = np.mean(times['physics']) * 1000
    avg_ball = np.mean(times['ball']) * 1000
    avg_prob = np.mean(times['probability']) * 1000
    avg_total = np.mean(times['total_core']) * 1000

    print(f"Data preparation: {avg_data_prep:.1f}ms")
    print(f"Grid creation: {avg_grid:.1f}ms")
    print(f"Physics calculation: {avg_physics:.1f}ms")
    print(f"Ball calculation: {avg_ball:.1f}ms")
    print(f"Probability calculation: {avg_prob:.1f}ms")
    print(f"Total core computation: {avg_total:.1f}ms")

    # Return the averages for comparison
    return {
        'data_prep': avg_data_prep,
        'grid_creation': avg_grid,
        'physics': avg_physics,
        'ball': avg_ball,
        'probability': avg_prob,
        'total_core': avg_total
    }


def compare_backends(players: List = None, ball_position: Point = None,
                     grid_resolution: tuple = (84, 54), n_runs: int = 10):
    """
    Compare NumPy vs Numba backends side by side.

    Args:
        players: List of PlayerState objects (if None, creates sample players)
        ball_position: Ball position Point (if None, uses (0, 0))
        grid_resolution: Grid size for calculation
        n_runs: Number of runs to average over
    """
    print(f"BACKEND COMPARISON")
    print(
        f"Grid Resolution: {grid_resolution[0]}x{grid_resolution[1]} ({grid_resolution[0] * grid_resolution[1]:,} points)")
    print("=" * 70)

    # Profile NumPy
    numpy_results = profile_backend("numpy", players, ball_position, grid_resolution,
                                    n_runs)

    # Profile Numba
    numba_results = profile_backend("numba", players, ball_position, grid_resolution,
                                    n_runs)

    # Print comparison
    print("\nCOMPARISON SUMMARY:")
    print("=" * 50)
    print(f"{'Component':<20} {'NumPy (ms)':<12} {'Numba (ms)':<12} {'Speedup':<10}")
    print("-" * 54)

    components = ['physics', 'ball', 'probability', 'total_core']
    for comp in components:
        numpy_time = numpy_results[comp]
        numba_time = numba_results[comp]
        speedup = numpy_time / numba_time if numba_time > 0 else 0

        comp_name = comp.replace('_', ' ').title()
        if comp == 'total_core':
            comp_name = 'TOTAL CORE'
            print("-" * 54)

        print(
            f"{comp_name:<20} {numpy_time:>8.1f}     {numba_time:>8.1f}     {speedup:>6.1f}x")

    overall_speedup = numpy_results['total_core'] / numba_results['total_core']
    print(f"\nOverall speedup: {overall_speedup:.1f}x")

    return numpy_results, numba_results


# Usage examples:
if __name__ == "__main__":
    # Profile individual backend
    # profile_backend("numpy")
    # profile_backend("numba")

    # Compare both backends
    compare_backends(n_runs=10)