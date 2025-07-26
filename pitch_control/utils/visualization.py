"""
Visualization utilities for pitch control surfaces.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from ..core import ControlSurface, PitchControlResult
from ..core.pitch import PitchDimensions
from matplotlib.patches import Rectangle


def plot_control_surface(
    result: PitchControlResult,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    alpha: float = 0.7,
    show_uncertainty: bool = True,
) -> plt.Figure:
    """
    Plot pitch control surface with home/away control and uncertainty.

    Args:
        result: PitchControlResult to visualize
        title: Plot title
        figsize: Figure size
        alpha: Transparency for control surfaces
        show_uncertainty: Whether to show uncertainty overlay

    Returns:
        Matplotlib figure
    """
    surface = result.control_surface

    fig, axes = plt.subplots(1, 3 if show_uncertainty else 2, figsize=figsize)
    if not show_uncertainty:
        axes = [axes[0], axes[1]]

    # Home team control
    im1 = axes[0].contourf(
        surface.grid_x,
        surface.grid_y,
        surface.home_control,
        levels=20,
        cmap="Reds",
        alpha=alpha,
    )
    axes[0].set_title("Home Team Control")
    axes[0].set_xlabel("X Position (m)")
    axes[0].set_ylabel("Y Position (m)")
    plt.colorbar(im1, ax=axes[0])

    # Away team control
    im2 = axes[1].contourf(
        surface.grid_x,
        surface.grid_y,
        surface.away_control,
        levels=20,
        cmap="Blues",
        alpha=alpha,
    )
    axes[1].set_title("Away Team Control")
    axes[1].set_xlabel("X Position (m)")
    axes[1].set_ylabel("Y Position (m)")
    plt.colorbar(im2, ax=axes[1])

    # Uncertainty (contested areas)
    if show_uncertainty:
        uncertainty = surface.uncertainty
        im3 = axes[2].contourf(
            surface.grid_x,
            surface.grid_y,
            uncertainty,
            levels=20,
            cmap="Greys",
            alpha=alpha,
        )
        axes[2].set_title("Uncertainty (Contested Areas)")
        axes[2].set_xlabel("X Position (m)")
        axes[2].set_ylabel("Y Position (m)")
        plt.colorbar(im3, ax=axes[2])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_combined_control_with_pitch(
    result: PitchControlResult,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:

    fig, ax = plot_pitch(figsize=figsize)
    return plot_combined_control(result, title, figsize, fig, ax)


def plot_combined_control(
    result: PitchControlResult,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    plot_legend: bool = True,
) -> plt.Figure:
    """
    Plot combined control surface with home=red, away=blue, contested=white.
    """
    surface = result.control_surface

    # Create RGB image
    height, width = surface.home_control.shape
    rgb_image = np.zeros((height, width, 3))

    # Red channel for home team
    rgb_image[:, :, 0] = surface.home_control

    # Blue channel for away team
    rgb_image[:, :, 2] = surface.away_control

    # Green channel for contested areas (optional)
    contested = surface.uncertainty
    rgb_image[:, :, 1] = contested * 0.3

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(
        rgb_image,
        extent=[
            surface.grid_x.min(),
            surface.grid_x.max(),
            surface.grid_y.min(),
            surface.grid_y.max(),
        ],
        origin="lower",
        aspect="equal",
        alpha=0.9,
    )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    if title:
        ax.set_title(title)
    else:
        calc_time = result.calculation_time
        model = result.metadata.get("model", "Unknown")
        ax.set_title(f"{model} Pitch Control (calculated in {calc_time:.3f}s)")

    if plot_legend:
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor="red", alpha=0.8, label="Home Team"),
            Rectangle((0, 0), 1, 1, facecolor="blue", alpha=0.8, label="Away Team"),
            Rectangle((0, 0), 1, 1, facecolor="white", alpha=0.9, label="Contested"),
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.0), loc="upper right")

    plt.tight_layout()
    return fig


def plot_pitch(
    origin="center",
    figsize: Tuple[int, int] = (12, 8),
    pitch_dims_m: Tuple[float, float] | None = None,
):
    """
    Plot a football pitch with all important lines and markings.

    Parameters:
    origin (str): 'bottom_left' or 'center' - sets the coordinate system origin
                  'bottom_left': origin at bottom-left corner of pitch
                  'center': origin at center of pitch
    """
    if pitch_dims_m is None:
        pitch_dims_m = (PitchDimensions.length, PitchDimensions.width)

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Football pitch dimensions (in meters)
    pitch_length, pitch_width = pitch_dims_m  # Can be 90-120m for length and 45-90m
    # for width

    # Calculate offset based on origin choice
    if origin == "center":
        x_offset = -pitch_length / 2
        y_offset = -pitch_width / 2
    else:  # bottom_left
        x_offset = 0
        y_offset = 0

    # Draw the outer boundary
    ax.plot(
        [
            x_offset,
            x_offset,
            x_offset + pitch_length,
            x_offset + pitch_length,
            x_offset,
        ],
        [y_offset, y_offset + pitch_width, y_offset + pitch_width, y_offset, y_offset],
        "k-",
        linewidth=2,
    )

    # Draw the center line
    ax.plot(
        [x_offset + pitch_length / 2, x_offset + pitch_length / 2],
        [y_offset, y_offset + pitch_width],
        "k-",
        linewidth=2,
    )

    # Draw the center circle
    center_circle = plt.Circle(
        (x_offset + pitch_length / 2, y_offset + pitch_width / 2),
        9.15,
        fill=False,
        color="black",
        linewidth=2,
    )
    ax.add_patch(center_circle)

    # Draw the center spot
    ax.plot(x_offset + pitch_length / 2, y_offset + pitch_width / 2, "ko", markersize=3)

    # Goal dimensions
    goal_width = 7.32
    goal_height = 2.44

    # Draw goals
    # Left goal
    ax.plot(
        [x_offset, x_offset - goal_height, x_offset - goal_height, x_offset],
        [
            y_offset + (pitch_width - goal_width) / 2,
            y_offset + (pitch_width - goal_width) / 2,
            y_offset + (pitch_width + goal_width) / 2,
            y_offset + (pitch_width + goal_width) / 2,
        ],
        "k-",
        linewidth=2,
    )
    # Right goal
    ax.plot(
        [
            x_offset + pitch_length,
            x_offset + pitch_length + goal_height,
            x_offset + pitch_length + goal_height,
            x_offset + pitch_length,
        ],
        [
            y_offset + (pitch_width - goal_width) / 2,
            y_offset + (pitch_width - goal_width) / 2,
            y_offset + (pitch_width + goal_width) / 2,
            y_offset + (pitch_width + goal_width) / 2,
        ],
        "k-",
        linewidth=2,
    )

    # Penalty areas (18-yard box)
    penalty_area_length = 16.5
    penalty_area_width = 40.32

    # Left penalty area
    ax.plot(
        [
            x_offset,
            x_offset + penalty_area_length,
            x_offset + penalty_area_length,
            x_offset,
        ],
        [
            y_offset + (pitch_width - penalty_area_width) / 2,
            y_offset + (pitch_width - penalty_area_width) / 2,
            y_offset + (pitch_width + penalty_area_width) / 2,
            y_offset + (pitch_width + penalty_area_width) / 2,
        ],
        "k-",
        linewidth=2,
    )

    # Right penalty area
    ax.plot(
        [
            x_offset + pitch_length,
            x_offset + pitch_length - penalty_area_length,
            x_offset + pitch_length - penalty_area_length,
            x_offset + pitch_length,
        ],
        [
            y_offset + (pitch_width - penalty_area_width) / 2,
            y_offset + (pitch_width - penalty_area_width) / 2,
            y_offset + (pitch_width + penalty_area_width) / 2,
            y_offset + (pitch_width + penalty_area_width) / 2,
        ],
        "k-",
        linewidth=2,
    )

    # Goal areas (6-yard box)
    goal_area_length = 5.5
    goal_area_width = 18.32

    # Left goal area
    ax.plot(
        [x_offset, x_offset + goal_area_length, x_offset + goal_area_length, x_offset],
        [
            y_offset + (pitch_width - goal_area_width) / 2,
            y_offset + (pitch_width - goal_area_width) / 2,
            y_offset + (pitch_width + goal_area_width) / 2,
            y_offset + (pitch_width + goal_area_width) / 2,
        ],
        "k-",
        linewidth=2,
    )

    # Right goal area
    ax.plot(
        [
            x_offset + pitch_length,
            x_offset + pitch_length - goal_area_length,
            x_offset + pitch_length - goal_area_length,
            x_offset + pitch_length,
        ],
        [
            y_offset + (pitch_width - goal_area_width) / 2,
            y_offset + (pitch_width - goal_area_width) / 2,
            y_offset + (pitch_width + goal_area_width) / 2,
            y_offset + (pitch_width + goal_area_width) / 2,
        ],
        "k-",
        linewidth=2,
    )

    # Penalty spots
    penalty_spot_distance = 11
    ax.plot(
        x_offset + penalty_spot_distance, y_offset + pitch_width / 2, "ko", markersize=3
    )
    ax.plot(
        x_offset + pitch_length - penalty_spot_distance,
        y_offset + pitch_width / 2,
        "ko",
        markersize=3,
    )

    # Penalty arcs
    penalty_arc_radius = 9.15
    # Left penalty arc
    theta1 = np.arccos(
        (penalty_area_length - penalty_spot_distance) / penalty_arc_radius
    )
    theta_left = np.linspace(-theta1, theta1, 50)
    x_left = x_offset + penalty_spot_distance + penalty_arc_radius * np.cos(theta_left)
    y_left = y_offset + pitch_width / 2 + penalty_arc_radius * np.sin(theta_left)
    # Only draw the part outside the penalty area
    mask_left = x_left >= x_offset + penalty_area_length
    ax.plot(x_left[mask_left], y_left[mask_left], "k-", linewidth=2)

    # Right penalty arc
    theta_right = np.linspace(np.pi - theta1, np.pi + theta1, 50)
    x_right = (
        x_offset + pitch_length - penalty_spot_distance
    ) + penalty_arc_radius * np.cos(theta_right)
    y_right = y_offset + pitch_width / 2 + penalty_arc_radius * np.sin(theta_right)
    # Only draw the part outside the penalty area
    mask_right = x_right <= (x_offset + pitch_length - penalty_area_length)
    ax.plot(x_right[mask_right], y_right[mask_right], "k-", linewidth=2)

    # Corner arcs
    corner_radius = 1
    # Bottom left corner
    theta_corner = np.linspace(0, np.pi / 2, 25)
    x_corner_bl = x_offset + corner_radius * np.cos(theta_corner)
    y_corner_bl = y_offset + corner_radius * np.sin(theta_corner)
    ax.plot(x_corner_bl, y_corner_bl, "k-", linewidth=2)

    # Top left corner
    x_corner_tl = x_offset + corner_radius * np.cos(theta_corner)
    y_corner_tl = y_offset + pitch_width - corner_radius * np.sin(theta_corner)
    ax.plot(x_corner_tl, y_corner_tl, "k-", linewidth=2)

    # Bottom right corner
    x_corner_br = x_offset + pitch_length - corner_radius * np.cos(theta_corner)
    y_corner_br = y_offset + corner_radius * np.sin(theta_corner)
    ax.plot(x_corner_br, y_corner_br, "k-", linewidth=2)

    # Top right corner
    x_corner_tr = x_offset + pitch_length - corner_radius * np.cos(theta_corner)
    y_corner_tr = y_offset + pitch_width - corner_radius * np.sin(theta_corner)
    ax.plot(x_corner_tr, y_corner_tr, "k-", linewidth=2)

    # Set equal aspect ratio and limits with buffer
    buffer = 10
    ax.set_aspect("equal")
    ax.set_xlim(x_offset - buffer, x_offset + pitch_length + buffer)
    ax.set_ylim(y_offset - buffer, y_offset + pitch_width + buffer)

    # Add labels and title
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    if origin == "center":
        ax.set_title("Football Pitch Layout (Origin at Center)")
    else:
        ax.set_title("Football Pitch Layout (Origin at Bottom-Left)")

    # Add grid for reference
    ax.grid(True, alpha=0.3)

    # Set appropriate tick marks based on origin
    if origin == "center":
        x_ticks = np.arange(-50, 51, 10)
        y_ticks = np.arange(-30, 35, 10)
    else:
        x_ticks = np.arange(0, pitch_length + 1, 10)
        y_ticks = np.arange(0, pitch_width + 1, 10)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    plt.tight_layout()

    return fig, ax


# # Example usage - plot with origin at bottom-left (default)
# plot_football_pitch('bottom_left')
# plt.show()

# Uncomment the line below to plot with origin at center
# plot_football_pitch('center')
# plt.show()
