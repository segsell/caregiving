"""Plot weighted ADL state transitions and shares by age."""

import pickle
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import END_YEAR_PARENT_GENERATION


@pytask.mark.specs
def task_plot_weighted_adl_transitions(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_parent_child_sample: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "specs"
    / "weighted_adl_transitions_and_shares.png",
    start_age: int = 60,
) -> None:
    """Plot weighted ADL transitions and shares by age for women.

    Left panel: Weighted ADL state transition probabilities
    Right panel: Manually weighted ADL shares by age
    """
    # Load specs from pickle file
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    # Load parent-child sample for initial shares
    df_sample = pd.read_csv(path_to_parent_child_sample)
    df_sample = df_sample[df_sample["yrbirth"] < END_YEAR_PARENT_GENERATION].copy()

    plot_weighted_adl_transitions_and_shares(
        specs=specs,
        df_sample=df_sample,
        path_to_save_plot=path_to_save_plot,
        start_age=start_age,
    )


def plot_weighted_adl_transitions_and_shares(  # noqa: PLR0912, PLR0915
    specs: dict,
    df_sample: Optional[pd.DataFrame] = None,
    path_to_save_plot: Optional[Path | str] = None,
    start_age: int = 60,
) -> plt.Figure:
    """Plot weighted ADL transitions and shares by age for women.

    Left panel: Weighted ADL state transition probabilities for women
    Right panel: Manually weighted ADL shares by age using survival probabilities

    Parameters
    ----------
    specs : dict
        Full specs dictionary containing:
        - adl_state_transition_mat_weighted: weighted transition matrix
        - survival_by_age_mat: survival matrix of shape [sex, age_index]
        - survival_min_age: minimum age in survival matrix (age_index 0)
        - adl_labels, sex_labels, start_age_parents, end_age
    path_to_save_plot
        Optional path to save the plot.
    start_age
        Starting age for the projection (default: 60).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Extract data from specs
    adl_trans_mat_weighted = np.array(specs["adl_state_transition_mat_weighted"])
    adl_trans_mat_unweighted = np.array(specs["adl_state_transition_mat"])
    survival_mat = np.array(specs["survival_by_age_mat"])
    survival_min_age = specs["survival_min_age"]
    survival_max_age = survival_min_age + survival_mat.shape[1] - 1
    adl_labels = specs["adl_labels"]  # ["No ADL", "ADL 1", "ADL 2", "ADL 3"]
    sex_labels = specs["sex_labels"]  # ["Men", "Women"]
    start_age_parents = specs["start_age_parents"]
    end_age = specs["end_age"]

    # Map sex to index: "Men" = 0, "Women" = 1
    sex_to_idx = {sex: idx for idx, sex in enumerate(sex_labels)}
    women_idx = sex_to_idx["Women"]

    # ──────────────────────────────────────────────────────────────────────────
    # Compute initial shares at start_age from data (used by both panels)
    # ──────────────────────────────────────────────────────────────────────────
    if df_sample is not None:
        adl_map_num_to_str = {
            0: "No ADL",
            1: "ADL 1",
            2: "ADL 2",
            3: "ADL 3",
        }
        gender_map = {1: "Men", 2: "Women"}

        df_obs = df_sample.copy()
        df_obs = df_obs[df_obs["age"] == start_age].copy()
        df_obs = df_obs.dropna(subset=["adl_cat"])
        df_obs["sex"] = df_obs["gender"].map(gender_map)
        df_obs["adl_str"] = df_obs["adl_cat"].map(adl_map_num_to_str)

        # Compute shares for women
        df_women = df_obs[df_obs["sex"] == "Women"]
        if len(df_women) > 0:
            initial_shares = (
                df_women.groupby("adl_str").size().rename("count").reset_index()
            )
            total = len(df_women)
            initial_shares["share"] = initial_shares["count"] / total

            initial_shares_dict = {}
            for adl_label in adl_labels:
                row = initial_shares[initial_shares["adl_str"] == adl_label]
                if len(row) > 0:
                    initial_shares_dict[adl_label] = row["share"].iloc[0]
                else:
                    initial_shares_dict[adl_label] = 0.0
        else:
            # Fallback to equal shares if no data
            initial_shares_dict = {adl: 0.25 for adl in adl_labels}
    else:
        # Fallback to equal shares if no data provided
        initial_shares_dict = {adl: 0.25 for adl in adl_labels}

    # Prepare data for plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Left panel: ADL shares using weighted transition matrix
    # ──────────────────────────────────────────────────────────────────────────
    ax_left = axes[0]

    # Project shares forward using WEIGHTED transition matrix
    # (already includes survival probabilities)
    ages = np.arange(start_age, end_age + 1)
    shares_by_age_weighted = {adl: [] for adl in adl_labels}

    # Initialize with shares at start_age
    current_shares = initial_shares_dict.copy()

    for age in ages:
        # Store current shares
        for adl_label in adl_labels:
            shares_by_age_weighted[adl_label].append(current_shares[adl_label])

        if age < end_age:
            # Compute next period shares using WEIGHTED transition matrix
            period = age - start_age_parents
            next_shares = {adl: 0.0 for adl in adl_labels}

            for from_adl_idx, from_adl in enumerate(adl_labels):
                for to_adl_idx, to_adl in enumerate(adl_labels):
                    # Get weighted transition probability (already includes survival)
                    trans_prob = adl_trans_mat_weighted[
                        women_idx, period, from_adl_idx, to_adl_idx
                    ]
                    next_shares[to_adl] += current_shares[from_adl] * trans_prob

            current_shares = next_shares

    # Plot ADL shares (exclude No ADL for readability)
    adl_labels_to_plot = [label for label in adl_labels if label != "No ADL"]
    adl_colors = {
        "ADL 1": "green",
        "ADL 2": "orange",
        "ADL 3": "red",
    }

    for adl_label in adl_labels_to_plot:
        ax_left.plot(
            ages,
            shares_by_age_weighted[adl_label],
            label=adl_label,
            color=adl_colors[adl_label],
            linewidth=2,
        )

    # Calculate "Any ADL" (sum of ADL 1, 2, 3)
    any_adl_weighted = []
    for age_idx in range(len(ages)):
        any_adl_sum = (
            shares_by_age_weighted["ADL 1"][age_idx]
            + shares_by_age_weighted["ADL 2"][age_idx]
            + shares_by_age_weighted["ADL 3"][age_idx]
        )
        any_adl_weighted.append(any_adl_sum)

    ax_left.plot(
        ages,
        any_adl_weighted,
        label="Any ADL",
        color="blue",
        linewidth=2,
        linestyle="--",
    )

    # Find max for y-axis limit
    max_share_weighted = 0.0
    for adl_label in adl_labels_to_plot:
        adl_max = max(shares_by_age_weighted[adl_label])
        max_share_weighted = max(max_share_weighted, adl_max)
    if any_adl_weighted:
        max_share_weighted = max(max_share_weighted, *any_adl_weighted)

    ax_left.set_xlabel("Age")
    ax_left.set_ylabel("Share in Population")
    ax_left.set_title("Weighted ADL Shares (Women, from Weighted Matrix)")
    ax_left.set_xlim(start_age, end_age)
    ax_left.set_ylim(0, max_share_weighted * 1.1)
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(title="Care level", fontsize=9, title_fontsize=10, loc="best")

    # ──────────────────────────────────────────────────────────────────────────
    # Right panel: Manually weighted ADL shares by age
    # ──────────────────────────────────────────────────────────────────────────
    ax_right = axes[1]

    # Project shares forward using transition matrix and survival probabilities
    ages = np.arange(start_age, end_age + 1)
    shares_by_age = {adl: [] for adl in adl_labels}

    # Initialize with shares at start_age (among alive)
    current_shares_alive = initial_shares_dict.copy()

    for age in ages:
        # Get survival probability for women at this age
        # Survival matrix shape: [sex, age_index] where age_index = age - min_age
        if survival_min_age <= age <= survival_max_age:
            age_idx = age - survival_min_age
            survival_prob = survival_mat[women_idx, age_idx]
        else:
            survival_prob = 0.0

        # Weight shares by survival probability
        # ADL 0 (No ADL) includes both alive with no ADL and dead
        # ADL 1, 2, 3 only apply to alive population
        shares_by_age["No ADL"].append(
            (1 - survival_prob) + survival_prob * current_shares_alive["No ADL"]
        )
        shares_by_age["ADL 1"].append(survival_prob * current_shares_alive["ADL 1"])
        shares_by_age["ADL 2"].append(survival_prob * current_shares_alive["ADL 2"])
        shares_by_age["ADL 3"].append(survival_prob * current_shares_alive["ADL 3"])

        if age < end_age:
            # Compute next period shares among alive using UNWEIGHTED transition matrix
            # (transitions among alive population only)
            period = age - start_age_parents
            next_shares_alive = {adl: 0.0 for adl in adl_labels}

            for from_adl_idx, from_adl in enumerate(adl_labels):
                for to_adl_idx, to_adl in enumerate(adl_labels):
                    # Get unweighted transition probability (among alive)
                    trans_prob = adl_trans_mat_unweighted[
                        women_idx, period, from_adl_idx, to_adl_idx
                    ]
                    next_shares_alive[to_adl] += (
                        current_shares_alive[from_adl] * trans_prob
                    )

            current_shares_alive = next_shares_alive

    # Plot ADL shares (exclude No ADL for readability)
    adl_labels_to_plot = [label for label in adl_labels if label != "No ADL"]
    for adl_label in adl_labels_to_plot:
        ax_right.plot(
            ages,
            shares_by_age[adl_label],
            color=adl_colors[adl_label],
            linewidth=2,
            label=adl_label,
        )

    # Calculate and plot "Any ADL" (sum of ADL 1, 2, 3)
    any_adl = [
        shares_by_age["ADL 1"][i]
        + shares_by_age["ADL 2"][i]
        + shares_by_age["ADL 3"][i]
        for i in range(len(ages))
    ]
    ax_right.plot(
        ages,
        any_adl,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Any ADL",
    )

    # Find max for y-axis limit
    max_share = 0.0
    for adl_label in adl_labels_to_plot:
        adl_max = max(shares_by_age[adl_label])
        max_share = max(max_share, adl_max)
    if any_adl:
        max_share = max(max_share, *any_adl)

    ax_right.set_xlabel("Age")
    ax_right.set_ylabel("Share in Population")
    ax_right.set_title("Weighted ADL Shares (Women, Manual Weighting)")
    ax_right.set_xlim(start_age, end_age)
    ax_right.set_ylim(0, max_share * 1.1)
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(title="Care level", fontsize=9, title_fontsize=10, loc="best")

    # Add faint grid lines
    ax_right.set_xticks(np.arange(start_age, end_age + 1, 5), minor=True)
    ax_right.set_yticks(np.arange(0, max_share * 1.1 + 0.02, 0.02), minor=True)
    ax_right.grid(True, which="minor", alpha=0.3, linestyle="-", linewidth=0.5)
    ax_right.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.5)

    fig.tight_layout()

    if path_to_save_plot:
        path_to_save_plot = Path(path_to_save_plot)
        path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_to_save_plot, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {path_to_save_plot}")

    plt.close(fig)
    return fig


@pytask.mark.specs
def task_plot_light_intensive_adl_transitions(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_parent_child_sample: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "specs"
    / "light_intensive_adl_transitions_and_shares.png",
    start_age: int = 60,
) -> None:
    """Plot light/intensive ADL transitions and shares by age for women.

    Left panel: Light/intensive ADL categories (0, 1, 2) manually weighted
    Right panel: Original ADL categories (0, 1, 2, 3) manually weighted

    """
    # Load specs from pickle file
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    # Load parent-child sample for initial shares
    df_sample = pd.read_csv(path_to_parent_child_sample)
    df_sample = df_sample[df_sample["yrbirth"] < END_YEAR_PARENT_GENERATION].copy()

    plot_light_intensive_adl_transitions_and_shares(
        specs=specs,
        df_sample=df_sample,
        path_to_save_plot=path_to_save_plot,
        start_age=start_age,
    )


def plot_light_intensive_adl_transitions_and_shares(  # noqa: PLR0912, PLR0915
    specs: dict,
    df_sample: Optional[pd.DataFrame] = None,
    path_to_save_plot: Optional[Path | str] = None,
    start_age: int = 60,
) -> plt.Figure:
    """Plot light/intensive ADL transitions and shares by age for women.

    Left panel: Light/intensive ADL categories (0, 1, 2) manually weighted by survival
    Right panel: Original ADL categories (0, 1, 2, 3) manually weighted by survival

    Parameters
    ----------
    specs : dict
        Full specs dictionary containing:
        - adl_state_transition_mat_light_intensive: 3-category transition matrix
        - adl_state_transition_mat: 4-category transition matrix
        - survival_by_age_mat: survival matrix of shape [sex, age_index]
        - survival_min_age: minimum age in survival matrix (age_index 0)
        - adl_labels, sex_labels, start_age_parents, end_age
    path_to_save_plot
        Optional path to save the plot.
    start_age
        Starting age for the projection (default: 60).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Extract data from specs
    adl_trans_mat_light_intensive = np.array(
        specs["adl_state_transition_mat_light_intensive"]
    )
    adl_trans_mat_unweighted = np.array(specs["adl_state_transition_mat"])
    survival_mat = np.array(specs["survival_by_age_mat"])
    survival_min_age = specs["survival_min_age"]
    survival_max_age = survival_min_age + survival_mat.shape[1] - 1
    adl_labels = specs["adl_labels"]  # ["No ADL", "ADL 1", "ADL 2", "ADL 3"]
    sex_labels = specs["sex_labels"]  # ["Men", "Women"]
    start_age_parents = specs["start_age_parents"]
    end_age = specs["end_age"]

    # Map sex to index: "Men" = 0, "Women" = 1
    sex_to_idx = {sex: idx for idx, sex in enumerate(sex_labels)}
    women_idx = sex_to_idx["Women"]

    # Collapsed ADL labels for light/intensive
    adl_labels_light_intensive = ["No ADL", "ADL 1", "ADL 2 or ADL 3"]

    # ──────────────────────────────────────────────────────────────────────────
    # Compute initial shares at start_age from data
    # ──────────────────────────────────────────────────────────────────────────
    if df_sample is not None:
        adl_map_num_to_str = {
            0: "No ADL",
            1: "ADL 1",
            2: "ADL 2",
            3: "ADL 3",
        }
        gender_map = {1: "Men", 2: "Women"}

        df_obs = df_sample.copy()
        df_obs = df_obs[df_obs["age"] == start_age].copy()
        df_obs = df_obs.dropna(subset=["adl_cat"])
        df_obs["sex"] = df_obs["gender"].map(gender_map)
        df_obs["adl_str"] = df_obs["adl_cat"].map(adl_map_num_to_str)

        # Compute shares for women (4 categories)
        df_women = df_obs[df_obs["sex"] == "Women"]
        if len(df_women) > 0:
            initial_shares = (
                df_women.groupby("adl_str").size().rename("count").reset_index()
            )
            total = len(df_women)
            initial_shares["share"] = initial_shares["count"] / total

            initial_shares_dict = {}
            for adl_label in adl_labels:
                row = initial_shares[initial_shares["adl_str"] == adl_label]
                if len(row) > 0:
                    initial_shares_dict[adl_label] = row["share"].iloc[0]
                else:
                    initial_shares_dict[adl_label] = 0.0

            # Collapse to 3 categories for light/intensive
            initial_shares_light_intensive = {
                "No ADL": initial_shares_dict["No ADL"],
                "ADL 1": initial_shares_dict["ADL 1"],
                "ADL 2 or ADL 3": (
                    initial_shares_dict["ADL 2"] + initial_shares_dict["ADL 3"]
                ),
            }
        else:
            # Fallback to equal shares if no data
            initial_shares_dict = {adl: 0.25 for adl in adl_labels}
            initial_shares_light_intensive = {
                "No ADL": 0.25,
                "ADL 1": 0.25,
                "ADL 2 or ADL 3": 0.5,
            }
    else:
        # Fallback to equal shares if no data provided
        initial_shares_dict = {adl: 0.25 for adl in adl_labels}
        initial_shares_light_intensive = {
            "No ADL": 0.25,
            "ADL 1": 0.25,
            "ADL 2 or ADL 3": 0.5,
        }

    # Prepare data for plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Left panel: Light/intensive ADL shares (3 categories) manually weighted
    # ──────────────────────────────────────────────────────────────────────────
    ax_left = axes[0]

    ages = np.arange(start_age, end_age + 1)
    shares_by_age_light_intensive = {adl: [] for adl in adl_labels_light_intensive}

    # Initialize with shares at start_age (among alive)
    current_shares_alive = initial_shares_light_intensive.copy()

    for age in ages:
        # Get survival probability for women at this age
        if survival_min_age <= age <= survival_max_age:
            age_idx = age - survival_min_age
            survival_prob = survival_mat[women_idx, age_idx]
        else:
            survival_prob = 0.0

        # Weight shares by survival probability
        # ADL 0 (No ADL) includes both alive with no ADL and dead
        # ADL 1, ADL 2 or ADL 3 only apply to alive population
        shares_by_age_light_intensive["No ADL"].append(
            (1 - survival_prob) + survival_prob * current_shares_alive["No ADL"]
        )
        shares_by_age_light_intensive["ADL 1"].append(
            survival_prob * current_shares_alive["ADL 1"]
        )
        shares_by_age_light_intensive["ADL 2 or ADL 3"].append(
            survival_prob * current_shares_alive["ADL 2 or ADL 3"]
        )

        if age < end_age:
            # Compute next period shares among alive using light/intensive matrix
            period = age - start_age_parents
            next_shares_alive = {adl: 0.0 for adl in adl_labels_light_intensive}

            for from_adl_idx, from_adl in enumerate(adl_labels_light_intensive):
                for to_adl_idx, to_adl in enumerate(adl_labels_light_intensive):
                    # Get unweighted transition probability (among alive)
                    trans_prob = adl_trans_mat_light_intensive[
                        women_idx, period, from_adl_idx, to_adl_idx
                    ]
                    next_shares_alive[to_adl] += (
                        current_shares_alive[from_adl] * trans_prob
                    )

            current_shares_alive = next_shares_alive

    # Plot ADL shares (exclude No ADL for readability)
    adl_labels_to_plot = [
        label for label in adl_labels_light_intensive if label != "No ADL"
    ]
    adl_colors_light_intensive = {
        "ADL 1": "green",
        "ADL 2 or ADL 3": "red",
    }

    for adl_label in adl_labels_to_plot:
        ax_left.plot(
            ages,
            shares_by_age_light_intensive[adl_label],
            color=adl_colors_light_intensive[adl_label],
            linewidth=2,
            label=adl_label,
        )

    # Calculate and plot "Any ADL" (sum of ADL 1 and ADL 2 or ADL 3)
    any_adl_light_intensive = [
        shares_by_age_light_intensive["ADL 1"][i]
        + shares_by_age_light_intensive["ADL 2 or ADL 3"][i]
        for i in range(len(ages))
    ]
    ax_left.plot(
        ages,
        any_adl_light_intensive,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Any ADL",
    )

    # Find max for y-axis limit (left panel) - will compute global max after right panel
    max_share_light_intensive = 0.0
    for adl_label in adl_labels_to_plot:
        adl_max = max(shares_by_age_light_intensive[adl_label])
        max_share_light_intensive = max(max_share_light_intensive, adl_max)
    if any_adl_light_intensive:
        max_share_light_intensive = max(
            max_share_light_intensive, *any_adl_light_intensive
        )

    ax_left.set_xlabel("Age")
    ax_left.set_ylabel("Share in Population")
    ax_left.set_title("Weighted ADL Shares (Women, Light/Intensive, Manual Weighting)")
    ax_left.set_xlim(start_age, end_age)
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(title="Care level", fontsize=9, title_fontsize=10, loc="best")

    # ──────────────────────────────────────────────────────────────────────────
    # Right panel: Original ADL shares (4 categories) manually weighted
    # ──────────────────────────────────────────────────────────────────────────
    ax_right = axes[1]

    # Project shares forward using transition matrix and survival probabilities
    ages = np.arange(start_age, end_age + 1)
    shares_by_age = {adl: [] for adl in adl_labels}

    # Initialize with shares at start_age (among alive)
    current_shares_alive = initial_shares_dict.copy()

    for age in ages:
        # Get survival probability for women at this age
        if survival_min_age <= age <= survival_max_age:
            age_idx = age - survival_min_age
            survival_prob = survival_mat[women_idx, age_idx]
        else:
            survival_prob = 0.0

        # Weight shares by survival probability
        # ADL 0 (No ADL) includes both alive with no ADL and dead
        # ADL 1, 2, 3 only apply to alive population
        shares_by_age["No ADL"].append(
            (1 - survival_prob) + survival_prob * current_shares_alive["No ADL"]
        )
        shares_by_age["ADL 1"].append(survival_prob * current_shares_alive["ADL 1"])
        shares_by_age["ADL 2"].append(survival_prob * current_shares_alive["ADL 2"])
        shares_by_age["ADL 3"].append(survival_prob * current_shares_alive["ADL 3"])

        if age < end_age:
            # Compute next period shares among alive using UNWEIGHTED transition matrix
            period = age - start_age_parents
            next_shares_alive = {adl: 0.0 for adl in adl_labels}

            for from_adl_idx, from_adl in enumerate(adl_labels):
                for to_adl_idx, to_adl in enumerate(adl_labels):
                    # Get unweighted transition probability (among alive)
                    trans_prob = adl_trans_mat_unweighted[
                        women_idx, period, from_adl_idx, to_adl_idx
                    ]
                    next_shares_alive[to_adl] += (
                        current_shares_alive[from_adl] * trans_prob
                    )

            current_shares_alive = next_shares_alive

    # Plot ADL shares (exclude No ADL for readability)
    adl_labels_to_plot_right = [label for label in adl_labels if label != "No ADL"]
    adl_colors = {
        "ADL 1": "green",
        "ADL 2": "orange",
        "ADL 3": "red",
    }
    for adl_label in adl_labels_to_plot_right:
        ax_right.plot(
            ages,
            shares_by_age[adl_label],
            color=adl_colors[adl_label],
            linewidth=2,
            label=adl_label,
        )

    # Calculate and plot "Any ADL" (sum of ADL 1, 2, 3)
    any_adl = [
        shares_by_age["ADL 1"][i]
        + shares_by_age["ADL 2"][i]
        + shares_by_age["ADL 3"][i]
        for i in range(len(ages))
    ]
    ax_right.plot(
        ages,
        any_adl,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Any ADL",
    )

    # Find max for y-axis limit (right panel)
    max_share = 0.0
    for adl_label in adl_labels_to_plot_right:
        adl_max = max(shares_by_age[adl_label])
        max_share = max(max_share, adl_max)
    if any_adl:
        max_share = max(max_share, *any_adl)

    # Use the same y-axis limit for both panels
    max_share_global = max(max_share_light_intensive, max_share)
    y_max = max_share_global * 1.1

    # Set y-axis limits for both panels to be identical
    # (sharey=True ensures they stay in sync)
    ax_left.set_ylim(0, y_max)

    ax_right.set_xlabel("Age")
    ax_right.set_ylabel("Share in Population")
    ax_right.set_title("Weighted ADL Shares (Women, Original, Manual Weighting)")
    ax_right.set_xlim(start_age, end_age)
    # Show y-axis ticks and labels on the right plot as well
    ax_right.tick_params(left=True, labelleft=True)
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(title="Care level", fontsize=9, title_fontsize=10, loc="best")

    # Add faint grid lines (use global max for consistency)
    ax_right.set_xticks(np.arange(start_age, end_age + 1, 5), minor=True)
    ax_right.set_yticks(np.arange(0, y_max + 0.02, 0.02), minor=True)
    ax_right.grid(True, which="minor", alpha=0.3, linestyle="-", linewidth=0.5)
    ax_right.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.5)

    fig.tight_layout()

    if path_to_save_plot:
        path_to_save_plot = Path(path_to_save_plot)
        path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_to_save_plot, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {path_to_save_plot}")

    plt.close(fig)
    return fig
