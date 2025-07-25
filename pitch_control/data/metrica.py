#!/usr/bin/env python3
"""
Metrica Sports Data Fetcher with Coordinate & Velocity Extraction
================================================================

Fetch Metrica Sports data and extract ball/player coordinates with velocity calculations.

Usage:
    from metrica import fetch_data, extract_coordinates

    # Fetch data
    dataset = fetch_data("game1_tracking")

    # Extract coordinates and velocities
    coords = extract_coordinates(dataset)
    print(coords['players'].columns)  # Player position/velocity data
    print(coords['ball'].columns)    # Ball position/velocity data

Requirements:
    pip install kloppy pandas numpy
"""

import pandas as pd
import numpy as np
from kloppy import metrica


def fetch_data(dataset_name, **kwargs):
    """
    Fetch Metrica Sports sample data by dataset name.

    Args:
        dataset_name (str): Name of the dataset to fetch
            - "game1_tracking": Sample Game 1 tracking data (CSV)
            - "game2_tracking": Sample Game 2 tracking data (CSV)
            - "game3_tracking": Sample Game 3 tracking data (EPTS FIFA)
            - "game3_events": Sample Game 3 event data (JSON)
        **kwargs: Optional parameters (sample_rate, limit, coordinates, event_types)

    Returns:
        Kloppy dataset object
    """

    base_url = (
        "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
    )

    # Default parameters
    sample_rate = kwargs.get("sample_rate", None)
    limit = kwargs.get("limit", None)
    coordinates = kwargs.get("coordinates", "metrica")
    event_types = kwargs.get("event_types", None)

    if dataset_name == "game1_tracking":
        return metrica.load_tracking_csv(
            home_data=f"{base_url}/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
            away_data=f"{base_url}/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
            sample_rate=sample_rate,
            limit=limit,
            coordinates=coordinates,
        )

    elif dataset_name == "game2_tracking":
        return metrica.load_tracking_csv(
            home_data=f"{base_url}/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
            away_data=f"{base_url}/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
            sample_rate=sample_rate,
            limit=limit,
            coordinates=coordinates,
        )

    elif dataset_name == "game3_tracking":
        return metrica.load_tracking_epts(
            meta_data=f"{base_url}/Sample_Game_3/Sample_Game_3_metadata.xml",
            raw_data=f"{base_url}/Sample_Game_3/Sample_Game_3_tracking.txt",
            sample_rate=sample_rate,
            limit=limit,
            coordinates=coordinates,
        )

    elif dataset_name == "game3_events":
        return metrica.load_event(
            event_data=f"{base_url}/Sample_Game_3/Sample_Game_3_events.json",
            meta_data=f"{base_url}/Sample_Game_3/Sample_Game_3_metadata.xml",
            coordinates=coordinates,
            event_types=event_types,
        )

    else:
        available_datasets = [
            "game1_tracking",
            "game2_tracking",
            "game3_tracking",
            "game3_events",
        ]
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {available_datasets}"
        )


def calculate_velocity(x, y, time, window=3):
    """
    Calculate velocity from position data using a rolling window.

    Args:
        x, y: Position arrays
        time: Time array
        window: Window size for smoothing (frames)

    Returns:
        vx, vy: Velocity components
        speed: Speed magnitude
    """
    # Convert to numpy arrays and handle time properly
    x = np.array(x)
    y = np.array(y)

    # Convert time to numeric (seconds) if it's pandas Timedelta
    if hasattr(time, "dtype") and "timedelta" in str(time.dtype):
        time = pd.Series(time).dt.total_seconds().values
    elif isinstance(time, pd.Series):
        # Handle pandas timestamps by converting to numeric
        if pd.api.types.is_datetime64_any_dtype(time):
            time = time.astype("int64") / 1e9  # Convert to seconds
        else:
            time = time.values
    else:
        time = np.array(time, dtype=float)

    # Initialize velocity arrays
    vx = np.full_like(x, np.nan)
    vy = np.full_like(y, np.nan)

    # Calculate velocities using central difference with smoothing
    for i in range(window, len(x) - window):
        if not (
            np.isnan(x[i - window : i + window + 1]).any()
            or np.isnan(y[i - window : i + window + 1]).any()
        ):
            # Use linear regression over the window for smoother velocity
            t_window = time[i - window : i + window + 1]
            x_window = x[i - window : i + window + 1]
            y_window = y[i - window : i + window + 1]

            # Simple linear fit
            dt = float(t_window[-1] - t_window[0])
            if dt > 0:
                vx[i] = (x_window[-1] - x_window[0]) / dt
                vy[i] = (y_window[-1] - y_window[0]) / dt

    # Calculate speed magnitude
    speed = np.sqrt(vx**2 + vy**2)

    return vx, vy, speed


def calculate_acceleration(vx, vy, time, window=3):
    """
    Calculate acceleration from velocity data.

    Args:
        vx, vy: Velocity components
        time: Time array
        window: Window size for smoothing

    Returns:
        ax, ay: Acceleration components
        accel_mag: Acceleration magnitude
    """
    ax, ay, accel_mag = calculate_velocity(vx, vy, time, window)
    return ax, ay, accel_mag


def extract_coordinates(
    dataset, include_velocity=True, include_acceleration=False, velocity_window=3
):
    """
    Extract player and ball coordinates with optional velocity/acceleration calculations.

    Args:
        dataset: Kloppy tracking dataset
        include_velocity: Whether to calculate velocity
        include_acceleration: Whether to calculate acceleration
        velocity_window: Window size for velocity smoothing

    Returns:
        dict: {
            'players': DataFrame with player data,
            'ball': DataFrame with ball data,
            'metadata': dict with dataset info
        }
    """

    # Convert to DataFrame
    df = dataset.to_df()

    # Extract metadata
    if len(df) > 1:
        time_diff = df["timestamp"].diff().median()
        frame_rate = (
            1.0 / time_diff.total_seconds()
            if hasattr(time_diff, "total_seconds")
            else 1.0 / float(time_diff)
        )
        duration = df["timestamp"].max() - df["timestamp"].min()
        duration = (
            duration.total_seconds()
            if hasattr(duration, "total_seconds")
            else float(duration)
        )
    else:
        frame_rate = 25.0
        duration = 0

    metadata = {
        "frame_rate": frame_rate,
        "total_frames": len(df),
        "duration": duration,
        "field_dimensions": getattr(dataset.metadata, "pitch_dimensions", None),
    }

    results = {"metadata": metadata}

    # Extract ball data
    ball_data = {"timestamp": df["timestamp"]}

    if "ball_x" in df.columns and "ball_y" in df.columns:
        ball_data["x"] = df["ball_x"]
        ball_data["y"] = df["ball_y"]

        if include_velocity:
            vx, vy, speed = calculate_velocity(
                df["ball_x"].values,
                df["ball_y"].values,
                df["timestamp"].values,
                window=velocity_window,
            )
            ball_data["vx"] = vx
            ball_data["vy"] = vy
            ball_data["speed"] = speed

            if include_acceleration:
                ax, ay, accel = calculate_acceleration(
                    vx, vy, df["timestamp"].values, velocity_window
                )
                ball_data["ax"] = ax
                ball_data["ay"] = ay
                ball_data["acceleration"] = accel

    results["ball"] = pd.DataFrame(ball_data)

    # Extract player data
    player_columns = [col for col in df.columns if col.endswith("_x")]
    player_data = []

    for x_col in player_columns:
        # Extract player info
        player_id = x_col.replace("_x", "")
        y_col = x_col.replace("_x", "_y")

        if y_col not in df.columns:
            continue

        # Determine team (basic heuristic - you may need to adjust based on your data)
        team = (
            "home"
            if "Home" in player_id
            or any(h in player_id.lower() for h in ["home", "h_"])
            else "away"
        )

        # Base player data
        player_df = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "player_id": player_id,
                "team": team,
                "x": df[x_col],
                "y": df[y_col],
            }
        )

        # Add velocity if requested
        if include_velocity:
            vx, vy, speed = calculate_velocity(
                df[x_col].values,
                df[y_col].values,
                df["timestamp"].values,
                window=velocity_window,
            )
            player_df["vx"] = vx
            player_df["vy"] = vy
            player_df["speed"] = speed

            # Add acceleration if requested
            if include_acceleration:
                ax, ay, accel = calculate_acceleration(
                    vx, vy, df["timestamp"].values, velocity_window
                )
                player_df["ax"] = ax
                player_df["ay"] = ay
                player_df["acceleration"] = accel

        player_data.append(player_df)

    # Combine all player data
    if player_data:
        results["players"] = pd.concat(player_data, ignore_index=True)
    else:
        results["players"] = pd.DataFrame()

    return results


def get_player_at_timestamp(coords_data, player_id, timestamp):
    """
    Get specific player's data at a given timestamp.

    Args:
        coords_data: Output from extract_coordinates()
        player_id: Player identifier
        timestamp: Target timestamp

    Returns:
        dict: Player data at timestamp or None if not found
    """
    players = coords_data["players"]
    mask = (players["player_id"] == player_id) & (players["timestamp"] == timestamp)

    if mask.any():
        return players[mask].iloc[0].to_dict()

    # Find closest timestamp if exact match not found
    player_data = players[players["player_id"] == player_id]
    if len(player_data) > 0:
        closest_idx = (player_data["timestamp"] - timestamp).abs().idxmin()
        return player_data.loc[closest_idx].to_dict()

    return None


def get_all_players_at_timestamp(coords_data, timestamp, tolerance=0.1):
    """
    Get all players' data at a given timestamp.

    Args:
        coords_data: Output from extract_coordinates()
        timestamp: Target timestamp
        tolerance: Time tolerance for matching

    Returns:
        DataFrame: All players' data at timestamp
    """
    players = coords_data["players"]
    mask = abs(players["timestamp"] - timestamp) <= tolerance
    return players[mask].copy()


def list_datasets():
    """List all available datasets."""
    datasets = {
        "game1_tracking": "Sample Game 1 tracking data (CSV format)",
        "game2_tracking": "Sample Game 2 tracking data (CSV format)",
        "game3_tracking": "Sample Game 3 tracking data (EPTS FIFA format)",
        "game3_events": "Sample Game 3 event data (JSON format)",
    }

    print("Available Metrica Sports datasets:")
    for name, description in datasets.items():
        print(f"  {name}: {description}")

    return list(datasets.keys())


# Example usage
if __name__ == "__main__":
    # Fetch data
    print("Fetching game1_tracking data...")
    dataset = fetch_data("game1_tracking", limit=1000)

    # Extract coordinates with velocity
    print("Extracting coordinates and calculating velocities...")
    coords = extract_coordinates(
        dataset, include_velocity=True, include_acceleration=True
    )

    # Display results
    print(f"\nMetadata:")
    for key, value in coords["metadata"].items():
        print(f"  {key}: {value}")

    print(f"\nBall data shape: {coords['ball'].shape}")
    print(f"Ball columns: {list(coords['ball'].columns)}")

    print(f"\nPlayer data shape: {coords['players'].shape}")
    print(f"Player columns: {list(coords['players'].columns)}")
    print(f"Unique players: {coords['players']['player_id'].nunique()}")

    # Show sample data
    if not coords["ball"].empty:
        print(f"\nSample ball data:")
        print(coords["ball"].head(3))

    if not coords["players"].empty:
        print(f"\nSample player data:")
        print(coords["players"].head(3))

        # Show speed statistics
        print(f"\nPlayer speed statistics:")
        speed_stats = coords["players"].groupby("team")["speed"].describe()
        print(speed_stats)
