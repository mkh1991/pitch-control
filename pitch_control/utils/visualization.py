"""
Visualization utilities for pitch control surfaces.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from ..core import ControlSurface, PitchControlResult


def plot_control_surface(result: PitchControlResult,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8),
                         alpha: float = 0.7,
                         show_uncertainty: bool = True) -> plt.Figure:
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
    im1 = axes[0].contourf(surface.grid_x, surface.grid_y, surface.home_control,
                           levels=20, cmap='Reds', alpha=alpha)
    axes[0].set_title('Home Team Control')
    axes[0].set_xlabel('X Position (m)')
    axes[0].set_ylabel('Y Position (m)')
    plt.colorbar(im1, ax=axes[0])

    # Away team control
    im2 = axes[1].contourf(surface.grid_x, surface.grid_y, surface.away_control,
                           levels=20, cmap='Blues', alpha=alpha)
    axes[1].set_title('Away Team Control')
    axes[1].set_xlabel('X Position (m)')
    axes[1].set_ylabel('Y Position (m)')
    plt.colorbar(im2, ax=axes[1])

    # Uncertainty (contested areas)
    if show_uncertainty:
        uncertainty = surface.uncertainty
        im3 = axes[2].contourf(surface.grid_x, surface.grid_y, uncertainty,
                               levels=20, cmap='Greys', alpha=alpha)
        axes[2].set_title('Uncertainty (Contested Areas)')
        axes[2].set_xlabel('X Position (m)')
        axes[2].set_ylabel('Y Position (m)')
        plt.colorbar(im3, ax=axes[2])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_combined_control(result: PitchControlResult,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 7)) -> plt.Figure:
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

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb_image, extent=[
        surface.grid_x.min(), surface.grid_x.max(),
        surface.grid_y.min(), surface.grid_y.max()
    ], origin='lower', aspect='equal')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')

    if title:
        ax.set_title(title)
    else:
        calc_time = result.calculation_time
        model = result.metadata.get('model', 'Unknown')
        ax.set_title(f'{model} Pitch Control (calculated in {calc_time:.3f}s)')

    # Add color legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Home Team'),
        Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label='Away Team'),
        Rectangle((0, 0), 1, 1, facecolor='white', alpha=0.7, label='Contested')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig