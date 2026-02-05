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
    n_total = len(lagged_choice)

    # Add one row for "All" category
    shares = np.zeros((n_edu + 1, n_labor))

    # Calculate shares for "All" category (no education conditioning)
    if n_total > 0:
        for labor_idx, labor_state in enumerate(labor_states):
            count_all = np.sum(lagged_choice == labor_state)
            shares[0, labor_idx] = count_all / n_total

    # Calculate shares for each education level
    for edu_idx in range(n_edu):
        edu_mask = education == edu_idx
        n_agents_edu = np.sum(edu_mask)

        if n_agents_edu > 0:
            for labor_idx, labor_state in enumerate(labor_states):
                # Count agents with this education and labor state
                count = np.sum((edu_mask) & (lagged_choice == labor_state))
                shares[edu_idx + 1, labor_idx] = count / n_agents_edu

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up bar positions (include "All" as first category)
    labels_with_all = ["All"] + education_labels
    x = np.arange(len(labels_with_all))
    width = 0.25  # Width of bars

    # Create bars for each labor state
    for i, labor_label in enumerate(labor_labels):
        offset = (i - 1) * width  # Center the bars
        values = [shares[cat_idx, i] for cat_idx in range(len(labels_with_all))]
        ax.bar(x + offset, values, width, label=labor_label)

    # Customize plot
    ax.set_xlabel("Education", fontsize=12)
    ax.set_ylabel("Share", fontsize=12)
    ax.set_title("Labor Shares by Education at Age 29 (Initial Period)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_with_all)
    ax.set_ylim([0, 1])
    # Add horizontal grid lines at every 0.1
    ax.set_yticks(np.arange(0, 1.1, 0.1))
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


@pytask.mark.initial_conditions
@pytask.mark.plot_initial_conditions
def task_plot_initial_states_median_wealth(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "initial_conditions"
    / "median_wealth_age_29.png",
) -> None:
    """Plot median wealth by education at age 29.

    This task plots the median wealth of agents differentiated by education type
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

    # Calculate median wealth by education
    n_edu = len(education_labels)
    median_wealth = np.zeros(n_edu)

    for edu_idx in range(n_edu):
        edu_mask = education == edu_idx
        if np.sum(edu_mask) > 0:
            median_wealth[edu_idx] = np.median(wealth[edu_mask])

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create bars
    x = np.arange(len(education_labels))
    ax.bar(x, median_wealth, width=0.6, alpha=0.7, edgecolor="black", linewidth=1.2)

    # Customize plot
    ax.set_xlabel("Education", fontsize=12)
    ax.set_ylabel("Median Wealth", fontsize=12)
    ax.set_title("Median Wealth by Education at Age 29 (Initial Period)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(education_labels)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, med_wealth in enumerate(median_wealth):
        ax.text(
            i,
            med_wealth,
            f"{med_wealth:.2f}",
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


@pytask.mark.initial_conditions
@pytask.mark.plot_initial_conditions
def task_plot_initial_states_education_shares(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "initial_conditions"
    / "education_shares_from_initial_states.png",
) -> None:
    """Plot education shares (low and high education) from initial_states.pkl.

    This task plots the share of agents with low and high education directly
    from the initial_states.pkl file.

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

    # Extract education array
    education = np.array(states["education"])

    # Calculate shares
    education_counts = np.bincount(education)
    total = len(education)
    shares = education_counts / total

    # Create labels
    labels = [specs["education_labels"][idx] for idx in range(len(shares))]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, shares, color=["#1f77b4", "#ff7f0e"], alpha=0.7)
    ax.set_ylabel("Share", fontsize=12)
    ax.set_title(
        "Education Distribution from Initial States", fontsize=14, fontweight="bold"
    )
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, share in zip(bars, shares, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{share:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()

    # Ensure directory exists
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


@pytask.mark.initial_conditions
@pytask.mark.plot_initial_conditions
def task_plot_initial_states_job_offer_shares(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "initial_conditions"
    / "job_offer_shares_from_initial_states.png",
) -> None:
    """Plot job offer shares for all, low educ, and high educ from initial_states.pkl.

    This task plots the share of agents with positive job offer (job_offer == 1)
    for all agents, low education, and high education directly from the
    initial_states.pkl file.

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

    # Extract job_offer and education arrays
    job_offer = np.array(states["job_offer"])
    education = np.array(states["education"])

    # Calculate shares
    # All
    share_all = np.mean(job_offer == 1)

    # Low education
    data_low = job_offer[education == 0]
    share_low = np.mean(data_low == 1) if len(data_low) > 0 else 0.0

    # High education
    data_high = job_offer[education == 1]
    share_high = np.mean(data_high == 1) if len(data_high) > 0 else 0.0

    # Create labels
    labels = [
        "All",
        specs["education_labels"][0],
        specs["education_labels"][1],
    ]
    shares = [share_all, share_low, share_high]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, shares, color=["#2ca02c", "#1f77b4", "#ff7f0e"], alpha=0.7)
    ax.set_ylabel("Share with Positive Job Offer", fontsize=12)
    ax.set_title(
        "Job Offer Distribution by Education from Initial States",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, share in zip(bars, shares, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{share:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()

    # Ensure directory exists
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)
