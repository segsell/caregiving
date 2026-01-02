"""Plot savings grid points.

Creates two plots showing the asset grid points:
1. End of period assets grid
2. Savings grid (normalized)
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.wealth_and_budget.savings_grid import (
    create_end_of_period_assets,
    create_savings_grid_deprecated,
)


@pytask.mark.pre_estimation
def task_plot_end_of_period_assets(
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "pre_estimation"
    / "end_of_period_assets_grid.png",
):
    """Plot end of period assets grid points.

    Creates a plot showing the asset grid points from create_end_of_period_assets().
    The grid values are in actual currency units (multiplied by 1000).

    Parameters
    ----------
    path_to_save_plot : Path
        Path to save the plot

    """

    # Generate the grid
    assets_grid = create_end_of_period_assets()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot grid points as scatter plot
    ax.scatter(
        assets_grid,
        range(len(assets_grid)),
        s=50,
        alpha=0.6,
        color="steelblue",
        edgecolors="darkblue",
        linewidths=1,
    )

    # Also plot as a line to show the progression
    ax.plot(
        assets_grid,
        range(len(assets_grid)),
        color="steelblue",
        alpha=0.3,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Asset Value (€)", fontsize=12)
    ax.set_ylabel("Grid Point Index", fontsize=12)
    ax.set_title("End of Period Assets Grid Points", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    # Format x-axis with thousands separator
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Add text with grid statistics
    n_points = len(assets_grid)
    min_val = np.min(assets_grid)
    max_val = np.max(assets_grid)
    stats_text = f"Total points: {n_points}\nMin: €{min_val:,.0f}\nMax: €{max_val:,.0f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)

    print(f"End of period assets grid plot saved to {path_to_save_plot}")


@pytask.mark.pre_estimation
def task_plot_savings_grid(
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "pre_estimation"
    / "savings_grid.png",
):
    """Plot savings grid points.

    Creates a plot showing the asset grid points from create_savings_grid().
    The grid values are in thousands (normalized units).

    Parameters
    ----------
    path_to_save_plot : Path
        Path to save the plot

    """
    # Generate the grid
    savings_grid = create_savings_grid_deprecated()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot grid points as scatter plot
    ax.scatter(
        savings_grid,
        range(len(savings_grid)),
        s=50,
        alpha=0.6,
        color="darkgreen",
        edgecolors="darkolivegreen",
        linewidths=1,
    )

    # Also plot as a line to show the progression
    ax.plot(
        savings_grid,
        range(len(savings_grid)),
        color="darkgreen",
        alpha=0.3,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Savings Value (thousands)", fontsize=12)
    ax.set_ylabel("Grid Point Index", fontsize=12)
    ax.set_title("Savings Grid Points (Normalized)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    # Format x-axis with thousands separator
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Add text with grid statistics
    n_points = len(savings_grid)
    min_val = np.min(savings_grid)
    max_val = np.max(savings_grid)
    stats_text = f"Total points: {n_points}\nMin: {min_val:,.0f}\nMax: {max_val:,.0f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    plt.tight_layout()
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)

    print(f"Savings grid plot saved to {path_to_save_plot}")
