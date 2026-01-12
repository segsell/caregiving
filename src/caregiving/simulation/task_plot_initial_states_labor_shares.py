"""Plot labor shares from initial states at age 29 (first period)."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD


@pytask.mark.initial_conditions
@pytask.mark.plot_initial_conditions
def task_plot_initial_states_labor_shares(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "initial_conditions"
    / "labor_shares_age_29.png",
) -> None:
    """Plot labor shares (unemployed, part-time, full-time) by education at age 29.

    This task plots the share of agents in each labor state (unemployed, part-time,
    full-time) differentiated by education type for the first period (age 29),
    using the lagged_choice variable from the initial states object.

    Parameters
    ----------
    path_to_specs : Path
        Path to model specifications pickle file.
    path_to_initial_states : Path
        Path to initial states pickle file.
    path_to_save_plot : Path
        Path to save the generated plot.
    """
    # Load specs and initial states
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    with path_to_initial_states.open("rb") as f:
        states = pickle.load(f)

    # Extract lagged_choice and education arrays
    lagged_choice = np.array(states["lagged_choice"])
    education = np.array(states["education"])

    # Get education and choice labels from specs
    education_labels = specs["education_labels"]
    choice_labels = specs["choice_labels"]

    # Labor states to plot (excluding retirement = 0)
    # 1 = unemployed, 2 = part-time, 3 = full-time
    labor_states = [1, 2, 3]
    labor_labels = [
        choice_labels[1],  # "Unemployed"
        choice_labels[2],  # "Part-time"
        choice_labels[3],  # "Full-time"
    ]

    # Calculate shares by education and labor state
    n_edu = len(education_labels)
    n_labor = len(labor_states)

    shares = np.zeros((n_edu, n_labor))

    for edu_idx in range(n_edu):
        edu_mask = education == edu_idx
        n_agents_edu = np.sum(edu_mask)

        if n_agents_edu > 0:
            for labor_idx, labor_state in enumerate(labor_states):
                # Count agents with this education and labor state
                count = np.sum((edu_mask) & (lagged_choice == labor_state))
                shares[edu_idx, labor_idx] = count / n_agents_edu

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up bar positions
    x = np.arange(len(education_labels))
    width = 0.25  # Width of bars

    # Create bars for each labor state
    for i, labor_label in enumerate(labor_labels):
        offset = (i - 1) * width  # Center the bars
        values = [shares[edu_idx, i] for edu_idx in range(n_edu)]
        ax.bar(x + offset, values, width, label=labor_label)

    # Customize plot
    ax.set_xlabel("Education", fontsize=12)
    ax.set_ylabel("Share", fontsize=12)
    ax.set_title("Labor Shares by Education at Age 29 (Initial Period)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(education_labels)
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Ensure directory exists
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


@pytask.mark.initial_conditions
@pytask.mark.plot_initial_conditions
def task_plot_initial_states_average_wealth(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "initial_conditions"
    / "average_wealth_age_29.png",
) -> None:
    """Plot average wealth by education at age 29.

    This task plots the average wealth of agents differentiated by education type
    for the first period (age 29), using the assets_begin_of_period variable
    from the initial states object.

    Parameters
    ----------
    path_to_specs : Path
        Path to model specifications pickle file.
    path_to_initial_states : Path
        Path to initial states pickle file.
    path_to_save_plot : Path
        Path to save the generated plot.
    """
    # Load specs and initial states
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    with path_to_initial_states.open("rb") as f:
        states = pickle.load(f)

    # Extract assets_begin_of_period and education arrays
    wealth = np.array(states["assets_begin_of_period"])
    education = np.array(states["education"])

    # Get education labels from specs
    education_labels = specs["education_labels"]

    # Calculate average wealth by education
    n_edu = len(education_labels)
    average_wealth = np.zeros(n_edu)

    for edu_idx in range(n_edu):
        edu_mask = education == edu_idx
        if np.sum(edu_mask) > 0:
            average_wealth[edu_idx] = np.mean(wealth[edu_mask])

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create bars
    x = np.arange(len(education_labels))
    ax.bar(x, average_wealth, width=0.6, alpha=0.7, edgecolor="black", linewidth=1.2)

    # Customize plot
    ax.set_xlabel("Education", fontsize=12)
    ax.set_ylabel("Average Wealth", fontsize=12)
    ax.set_title("Average Wealth by Education at Age 29 (Initial Period)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(education_labels)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, avg_wealth in enumerate(average_wealth):
        ax.text(
            i,
            avg_wealth,
            f"{avg_wealth:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    # Ensure directory exists
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)
