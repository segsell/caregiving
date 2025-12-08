"""Plot pie charts showing labor supply transitions before and after first care demand.

For each agent, determines the age of first care demand (t=0).
Then groups agents by their labor supply state at t=-5 (5 years before first care demand).
For each group, creates pie charts showing the distribution of labor supply states
at t=1, t=3, t=5, and t=10 (years after first care demand).
"""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.counterfactual.plotting_utils import _ensure_agent_period
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    PART_TIME,
    RETIREMENT,
    UNEMPLOYED,
)


def _add_distance_to_first_care_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care_demand column.

    Sets 0 as first time care_demand > 0.
    """
    # Flatten any existing index to avoid column/index name ambiguity
    df = df.reset_index(drop=True)
    df = _ensure_agent_period(df)
    # Find first period where care_demand > 0
    care_demand_mask = df["care_demand"] > 0
    first_care_demand = (
        df.loc[care_demand_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_care_demand_period"})
    )
    out = df.merge(first_care_demand, on="agent", how="left")
    out["distance_to_first_care_demand"] = (
        out["period"] - out["first_care_demand_period"]
    )
    return out


def _add_distance_to_first_care(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care column.

    Sets 0 as first time choice is in INFORMAL_CARE.
    """
    # Flatten any existing index to avoid column/index name ambiguity
    df = df.reset_index(drop=True)
    df = _ensure_agent_period(df)
    # Find first period where choice is in INFORMAL_CARE
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df["choice"].isin(care_codes)
    first_care = (
        df.loc[caregiving_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_care_period"})
    )
    out = df.merge(first_care, on="agent", how="left")
    out["distance_to_first_care"] = out["period"] - out["first_care_period"]
    return out


def get_labor_supply_state(choice: int) -> str:
    """Map choice value to labor supply state.

    Args:
        choice: Choice value from the model

    Returns:
        Labor supply state: 'retired', 'unemployed', 'part_time', or 'full_time'
    """
    # Convert JAX arrays to lists for comparison
    retirement_values = np.asarray(RETIREMENT).ravel().tolist()
    unemployed_values = np.asarray(UNEMPLOYED).ravel().tolist()
    part_time_values = np.asarray(PART_TIME).ravel().tolist()
    full_time_values = np.asarray(FULL_TIME).ravel().tolist()

    if choice in retirement_values:
        return "retired"
    if choice in unemployed_values:
        return "unemployed"
    if choice in part_time_values:
        return "part_time"
    if choice in full_time_values:
        return "full_time"
    return "unknown"


@pytask.mark.post_estimation
def task_plot_labor_supply_transitions_after_care_demand(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_t1: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t1_after_care_demand.png",
    path_to_plot_t3: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t3_after_care_demand.png",
    path_to_plot_t5: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t5_after_care_demand.png",
    path_to_plot_t10: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t10_after_care_demand.png",
) -> None:
    """Plot pie charts showing labor supply transitions after first care demand.

    For each agent, determines the age of first care demand (t=0).
    Groups agents by their labor supply state at t=-5 (5 years before first care demand).
    For each group, creates pie charts showing the distribution of labor supply states
    at t=1, t=3, t=5, and t=10 (years after first care demand).

    Creates 4 plots (one for each time point), each with 1 row x 4 columns
    (one column for each labor supply state at t=-5).

    """
    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Restrict to alive
    df = df[df["health"] != DEAD].copy()

    # Ensure agent/period columns
    df = _ensure_agent_period(df)

    # Add distance to first care demand
    df = _add_distance_to_first_care_demand(df)

    # Filter to agents who have a first care demand period
    df = df[df["first_care_demand_period"].notna()].copy()

    # Get labor supply state for each period
    df["labor_supply_state"] = df["choice"].apply(get_labor_supply_state)

    # Filter to relevant time points: t=-5, t=1, t=3, t=5, t=10
    time_points = [-5, 1, 3, 5, 10]
    df_filtered = df[df["distance_to_first_care_demand"].isin(time_points)].copy()

    # Get labor supply at t=-5 for each agent
    df_t_minus_5 = df_filtered[
        df_filtered["distance_to_first_care_demand"] == -5
    ].copy()
    df_t_minus_5 = df_t_minus_5[["agent", "labor_supply_state"]].rename(
        columns={"labor_supply_state": "labor_supply_t_minus_5"}
    )

    # Merge back to get labor supply at t=-5 for all relevant periods
    df_with_t_minus_5 = df_filtered.merge(df_t_minus_5, on="agent", how="inner")

    # Filter to only agents who have data at t=-5
    df_with_t_minus_5 = df_with_t_minus_5[
        df_with_t_minus_5["labor_supply_t_minus_5"].notna()
    ].copy()

    # Define labor supply states and their order
    labor_states = ["retired", "unemployed", "part_time", "full_time"]
    state_labels = {
        "retired": "Retired",
        "unemployed": "Unemployed",
        "part_time": "Part-time",
        "full_time": "Full-time",
    }

    # Colors for pie charts
    colors = {
        "retired": JET_COLOR_MAP[0],
        "unemployed": JET_COLOR_MAP[1],
        "part_time": JET_COLOR_MAP[2],
        "full_time": JET_COLOR_MAP[3],
    }

    # Create plots for each time point
    for time_point, path_to_plot in (
        (1, path_to_plot_t1),
        (3, path_to_plot_t3),
        (5, path_to_plot_t5),
        (10, path_to_plot_t10),
    ):
        # Filter to this time point
        df_time = df_with_t_minus_5[
            df_with_t_minus_5["distance_to_first_care_demand"] == time_point
        ].copy()

        # Create figure with 1 row x 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for col_idx, state_at_t_minus_5 in enumerate(labor_states):
            ax = axes[col_idx]

            # Filter to agents who were in this state at t=-5
            df_state = df_time[
                df_time["labor_supply_t_minus_5"] == state_at_t_minus_5
            ].copy()

            if len(df_state) == 0:
                # No data for this group
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"{state_labels[state_at_t_minus_5]}\nat t=-5",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.axis("off")
                continue

            # Count labor supply states at this time point
            state_counts = df_state["labor_supply_state"].value_counts()

            # Ensure all states are represented (fill missing with 0)
            state_counts = state_counts.reindex(labor_states, fill_value=0)

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                state_counts.values,
                labels=[state_labels[s] for s in state_counts.index],
                autopct="%1.1f%%",
                colors=[colors[s] for s in state_counts.index],
                startangle=90,
                textprops={"fontsize": 10},
            )

            # Adjust autopct font size
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight("bold")

            # Set title
            n_agents = len(df_state)
            ax.set_title(
                f"{state_labels[state_at_t_minus_5]}\nat t=-5\n(n={n_agents})",
                fontsize=14,
                fontweight="bold",
            )

        # Add overall title
        fig.suptitle(
            f"Labor Supply Transitions {time_point} Year{'s' if time_point > 1 else ''} After First Care Demand\n(Grouped by Labor Supply at t=-5)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
        plt.close()


@pytask.mark.post_estimation
def task_plot_labor_supply_transitions_after_caregiving_t_minus_5(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_t1: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t1_after_caregiving_t_minus_5.png",
    path_to_plot_t3: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t3_after_caregiving_t_minus_5.png",
    path_to_plot_t5: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t5_after_caregiving_t_minus_5.png",
    path_to_plot_t10: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t10_after_caregiving_t_minus_5.png",
) -> None:
    """Plot pie charts showing labor supply transitions after first caregiving spell.

    For each agent, determines the age of first caregiving spell (t=0).
    Groups agents by their labor supply state at t=-5 (5 years before first caregiving).
    For each group, creates pie charts showing the distribution of labor supply states
    at t=1, t=3, t=5, and t=10 (years after first caregiving).

    Creates 4 plots (one for each time point), each with 1 row x 4 columns
    (one column for each labor supply state at t=-5).

    """
    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Restrict to alive
    df = df[df["health"] != DEAD].copy()

    # Ensure agent/period columns
    df = _ensure_agent_period(df)

    # Add distance to first caregiving
    df = _add_distance_to_first_care(df)

    # Filter to agents who have a first caregiving period
    df = df[df["first_care_period"].notna()].copy()

    # Get labor supply state for each period
    df["labor_supply_state"] = df["choice"].apply(get_labor_supply_state)

    # Filter to relevant time points: t=-5, t=1, t=3, t=5, t=10
    time_points = [-5, 1, 3, 5, 10]
    df_filtered = df[df["distance_to_first_care"].isin(time_points)].copy()

    # Get labor supply at t=-5 for each agent
    df_t_minus_5 = df_filtered[df_filtered["distance_to_first_care"] == -5].copy()
    df_t_minus_5 = df_t_minus_5[["agent", "labor_supply_state"]].rename(
        columns={"labor_supply_state": "labor_supply_t_minus_5"}
    )

    # Merge back to get labor supply at t=-5 for all relevant periods
    df_with_t_minus_5 = df_filtered.merge(df_t_minus_5, on="agent", how="inner")

    # Filter to only agents who have data at t=-5
    df_with_t_minus_5 = df_with_t_minus_5[
        df_with_t_minus_5["labor_supply_t_minus_5"].notna()
    ].copy()

    # Define labor supply states and their order
    labor_states = ["retired", "unemployed", "part_time", "full_time"]
    state_labels = {
        "retired": "Retired",
        "unemployed": "Unemployed",
        "part_time": "Part-time",
        "full_time": "Full-time",
    }

    # Colors for pie charts
    colors = {
        "retired": JET_COLOR_MAP[0],
        "unemployed": JET_COLOR_MAP[1],
        "part_time": JET_COLOR_MAP[2],
        "full_time": JET_COLOR_MAP[3],
    }

    # Create plots for each time point
    for time_point, path_to_plot in (
        (1, path_to_plot_t1),
        (3, path_to_plot_t3),
        (5, path_to_plot_t5),
        (10, path_to_plot_t10),
    ):
        # Filter to this time point
        df_time = df_with_t_minus_5[
            df_with_t_minus_5["distance_to_first_care"] == time_point
        ].copy()

        # Create figure with 1 row x 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for col_idx, state_at_t_minus_5 in enumerate(labor_states):
            ax = axes[col_idx]

            # Filter to agents who were in this state at t=-5
            df_state = df_time[
                df_time["labor_supply_t_minus_5"] == state_at_t_minus_5
            ].copy()

            if len(df_state) == 0:
                # No data for this group
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"{state_labels[state_at_t_minus_5]}\nat t=-5",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.axis("off")
                continue

            # Count labor supply states at this time point
            state_counts = df_state["labor_supply_state"].value_counts()

            # Ensure all states are represented (fill missing with 0)
            state_counts = state_counts.reindex(labor_states, fill_value=0)

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                state_counts.values,
                labels=[state_labels[s] for s in state_counts.index],
                autopct="%1.1f%%",
                colors=[colors[s] for s in state_counts.index],
                startangle=90,
                textprops={"fontsize": 10},
            )

            # Adjust autopct font size
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight("bold")

            # Set title
            n_agents = len(df_state)
            ax.set_title(
                f"{state_labels[state_at_t_minus_5]}\nat t=-5\n(n={n_agents})",
                fontsize=14,
                fontweight="bold",
            )

        # Add overall title
        fig.suptitle(
            f"Labor Supply Transitions {time_point} Year{'s' if time_point > 1 else ''} After First Caregiving Spell\n(Grouped by Labor Supply at t=-5)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
        plt.close()


@pytask.mark.post_estimation
def task_plot_labor_supply_transitions_after_caregiving_t_minus_1(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_t1: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t1_after_caregiving_t_minus_1.png",
    path_to_plot_t3: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t3_after_caregiving_t_minus_1.png",
    path_to_plot_t5: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t5_after_caregiving_t_minus_1.png",
    path_to_plot_t10: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t10_after_caregiving_t_minus_1.png",
) -> None:
    """Plot pie charts showing labor supply transitions after first caregiving spell.

    For each agent, determines the age of first caregiving spell (t=0).
    Groups agents by their labor supply state at t=-1 (1 year before first caregiving).
    For each group, creates pie charts showing the distribution of labor supply states
    at t=1, t=3, t=5, and t=10 (years after first caregiving).

    Creates 4 plots (one for each time point), each with 1 row x 4 columns
    (one column for each labor supply state at t=-1).

    """
    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Restrict to alive
    df = df[df["health"] != DEAD].copy()

    # Ensure agent/period columns
    df = _ensure_agent_period(df)

    # Add distance to first caregiving
    df = _add_distance_to_first_care(df)

    # Filter to agents who have a first caregiving period
    df = df[df["first_care_period"].notna()].copy()

    # Get labor supply state for each period
    df["labor_supply_state"] = df["choice"].apply(get_labor_supply_state)

    # Filter to relevant time points: t=-1, t=1, t=3, t=5, t=10
    time_points = [-1, 1, 3, 5, 10]
    df_filtered = df[df["distance_to_first_care"].isin(time_points)].copy()

    # Get labor supply at t=-1 for each agent
    df_t_minus_1 = df_filtered[df_filtered["distance_to_first_care"] == -1].copy()
    df_t_minus_1 = df_t_minus_1[["agent", "labor_supply_state"]].rename(
        columns={"labor_supply_state": "labor_supply_t_minus_1"}
    )

    # Merge back to get labor supply at t=-1 for all relevant periods
    df_with_t_minus_1 = df_filtered.merge(df_t_minus_1, on="agent", how="inner")

    # Filter to only agents who have data at t=-1
    df_with_t_minus_1 = df_with_t_minus_1[
        df_with_t_minus_1["labor_supply_t_minus_1"].notna()
    ].copy()

    # Define labor supply states and their order
    labor_states = ["retired", "unemployed", "part_time", "full_time"]
    state_labels = {
        "retired": "Retired",
        "unemployed": "Unemployed",
        "part_time": "Part-time",
        "full_time": "Full-time",
    }

    # Colors for pie charts
    colors = {
        "retired": JET_COLOR_MAP[0],
        "unemployed": JET_COLOR_MAP[1],
        "part_time": JET_COLOR_MAP[2],
        "full_time": JET_COLOR_MAP[3],
    }

    # Create plots for each time point
    for time_point, path_to_plot in (
        (1, path_to_plot_t1),
        (3, path_to_plot_t3),
        (5, path_to_plot_t5),
        (10, path_to_plot_t10),
    ):
        # Filter to this time point
        df_time = df_with_t_minus_1[
            df_with_t_minus_1["distance_to_first_care"] == time_point
        ].copy()

        # Create figure with 1 row x 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for col_idx, state_at_t_minus_1 in enumerate(labor_states):
            ax = axes[col_idx]

            # Filter to agents who were in this state at t=-1
            df_state = df_time[
                df_time["labor_supply_t_minus_1"] == state_at_t_minus_1
            ].copy()

            if len(df_state) == 0:
                # No data for this group
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"{state_labels[state_at_t_minus_1]}\nat t=-1",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.axis("off")
                continue

            # Count labor supply states at this time point
            state_counts = df_state["labor_supply_state"].value_counts()

            # Ensure all states are represented (fill missing with 0)
            state_counts = state_counts.reindex(labor_states, fill_value=0)

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                state_counts.values,
                labels=[state_labels[s] for s in state_counts.index],
                autopct="%1.1f%%",
                colors=[colors[s] for s in state_counts.index],
                startangle=90,
                textprops={"fontsize": 10},
            )

            # Adjust autopct font size
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight("bold")

            # Set title
            n_agents = len(df_state)
            ax.set_title(
                f"{state_labels[state_at_t_minus_1]}\nat t=-1\n(n={n_agents})",
                fontsize=14,
                fontweight="bold",
            )

        # Add overall title
        fig.suptitle(
            f"Labor Supply Transitions {time_point} Year{'s' if time_point > 1 else ''} After First Caregiving Spell\n(Grouped by Labor Supply at t=-1)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
        plt.close()


@pytask.mark.post_estimation
def task_plot_labor_supply_transitions_after_care_demand_t_minus_1(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_t1: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t1_after_care_demand_t_minus_1.png",
    path_to_plot_t3: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t3_after_care_demand_t_minus_1.png",
    path_to_plot_t5: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t5_after_care_demand_t_minus_1.png",
    path_to_plot_t10: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t10_after_care_demand_t_minus_1.png",
) -> None:
    """Plot pie charts showing labor supply transitions after first care demand.

    For each agent, determines the age of first care demand (t=0).
    Groups agents by their labor supply state at t=-1 (1 year before first care demand).
    For each group, creates pie charts showing the distribution of labor supply states
    at t=1, t=3, t=5, and t=10 (years after first care demand).

    Creates 4 plots (one for each time point), each with 1 row x 4 columns
    (one column for each labor supply state at t=-1).

    """
    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Restrict to alive
    df = df[df["health"] != DEAD].copy()

    # Ensure agent/period columns
    df = _ensure_agent_period(df)

    # Add distance to first care demand
    df = _add_distance_to_first_care_demand(df)

    # Filter to agents who have a first care demand period
    df = df[df["first_care_demand_period"].notna()].copy()

    # Get labor supply state for each period
    df["labor_supply_state"] = df["choice"].apply(get_labor_supply_state)

    # Filter to relevant time points: t=-1, t=1, t=3, t=5, t=10
    time_points = [-1, 1, 3, 5, 10]
    df_filtered = df[df["distance_to_first_care_demand"].isin(time_points)].copy()

    # Get labor supply at t=-1 for each agent
    df_t_minus_1 = df_filtered[
        df_filtered["distance_to_first_care_demand"] == -1
    ].copy()
    df_t_minus_1 = df_t_minus_1[["agent", "labor_supply_state"]].rename(
        columns={"labor_supply_state": "labor_supply_t_minus_1"}
    )

    # Merge back to get labor supply at t=-1 for all relevant periods
    df_with_t_minus_1 = df_filtered.merge(df_t_minus_1, on="agent", how="inner")

    # Filter to only agents who have data at t=-1
    df_with_t_minus_1 = df_with_t_minus_1[
        df_with_t_minus_1["labor_supply_t_minus_1"].notna()
    ].copy()

    # Define labor supply states and their order
    labor_states = ["retired", "unemployed", "part_time", "full_time"]
    state_labels = {
        "retired": "Retired",
        "unemployed": "Unemployed",
        "part_time": "Part-time",
        "full_time": "Full-time",
    }

    # Colors for pie charts
    colors = {
        "retired": JET_COLOR_MAP[0],
        "unemployed": JET_COLOR_MAP[1],
        "part_time": JET_COLOR_MAP[2],
        "full_time": JET_COLOR_MAP[3],
    }

    # Create plots for each time point
    for time_point, path_to_plot in (
        (1, path_to_plot_t1),
        (3, path_to_plot_t3),
        (5, path_to_plot_t5),
        (10, path_to_plot_t10),
    ):
        # Filter to this time point
        df_time = df_with_t_minus_1[
            df_with_t_minus_1["distance_to_first_care_demand"] == time_point
        ].copy()

        # Create figure with 1 row x 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for col_idx, state_at_t_minus_1 in enumerate(labor_states):
            ax = axes[col_idx]

            # Filter to agents who were in this state at t=-1
            df_state = df_time[
                df_time["labor_supply_t_minus_1"] == state_at_t_minus_1
            ].copy()

            if len(df_state) == 0:
                # No data for this group
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"{state_labels[state_at_t_minus_1]}\nat t=-1",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.axis("off")
                continue

            # Count labor supply states at this time point
            state_counts = df_state["labor_supply_state"].value_counts()

            # Ensure all states are represented (fill missing with 0)
            state_counts = state_counts.reindex(labor_states, fill_value=0)

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                state_counts.values,
                labels=[state_labels[s] for s in state_counts.index],
                autopct="%1.1f%%",
                colors=[colors[s] for s in state_counts.index],
                startangle=90,
                textprops={"fontsize": 10},
            )

            # Adjust autopct font size
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight("bold")

            # Set title
            n_agents = len(df_state)
            ax.set_title(
                f"{state_labels[state_at_t_minus_1]}\nat t=-1\n(n={n_agents})",
                fontsize=14,
                fontweight="bold",
            )

        # Add overall title
        fig.suptitle(
            f"Labor Supply Transitions {time_point} Year{'s' if time_point > 1 else ''} After First Care Demand\n(Grouped by Labor Supply at t=-1)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
        plt.close()


@pytask.mark.post_estimation
def task_plot_labor_supply_transitions_after_caregiving_t_minus_5(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_t1: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t1_after_caregiving_t_minus_5.png",
    path_to_plot_t3: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t3_after_caregiving_t_minus_5.png",
    path_to_plot_t5: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t5_after_caregiving_t_minus_5.png",
    path_to_plot_t10: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t10_after_caregiving_t_minus_5.png",
) -> None:
    """Plot pie charts showing labor supply transitions after first caregiving spell.

    For each agent, determines the age of first caregiving spell (t=0).
    Groups agents by their labor supply state at t=-5 (5 years before first caregiving).
    For each group, creates pie charts showing the distribution of labor supply states
    at t=1, t=3, t=5, and t=10 (years after first caregiving).

    Creates 4 plots (one for each time point), each with 1 row x 4 columns
    (one column for each labor supply state at t=-5).

    """
    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Restrict to alive
    df = df[df["health"] != DEAD].copy()

    # Ensure agent/period columns
    df = _ensure_agent_period(df)

    # Add distance to first caregiving
    df = _add_distance_to_first_care(df)

    # Filter to agents who have a first caregiving period
    df = df[df["first_care_period"].notna()].copy()

    # Get labor supply state for each period
    df["labor_supply_state"] = df["choice"].apply(get_labor_supply_state)

    # Filter to relevant time points: t=-5, t=1, t=3, t=5, t=10
    time_points = [-5, 1, 3, 5, 10]
    df_filtered = df[df["distance_to_first_care"].isin(time_points)].copy()

    # Get labor supply at t=-5 for each agent
    df_t_minus_5 = df_filtered[df_filtered["distance_to_first_care"] == -5].copy()
    df_t_minus_5 = df_t_minus_5[["agent", "labor_supply_state"]].rename(
        columns={"labor_supply_state": "labor_supply_t_minus_5"}
    )

    # Merge back to get labor supply at t=-5 for all relevant periods
    df_with_t_minus_5 = df_filtered.merge(df_t_minus_5, on="agent", how="inner")

    # Filter to only agents who have data at t=-5
    df_with_t_minus_5 = df_with_t_minus_5[
        df_with_t_minus_5["labor_supply_t_minus_5"].notna()
    ].copy()

    # Define labor supply states and their order
    labor_states = ["retired", "unemployed", "part_time", "full_time"]
    state_labels = {
        "retired": "Retired",
        "unemployed": "Unemployed",
        "part_time": "Part-time",
        "full_time": "Full-time",
    }

    # Colors for pie charts
    colors = {
        "retired": JET_COLOR_MAP[0],
        "unemployed": JET_COLOR_MAP[1],
        "part_time": JET_COLOR_MAP[2],
        "full_time": JET_COLOR_MAP[3],
    }

    # Create plots for each time point
    for time_point, path_to_plot in (
        (1, path_to_plot_t1),
        (3, path_to_plot_t3),
        (5, path_to_plot_t5),
        (10, path_to_plot_t10),
    ):
        # Filter to this time point
        df_time = df_with_t_minus_5[
            df_with_t_minus_5["distance_to_first_care"] == time_point
        ].copy()

        # Create figure with 1 row x 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for col_idx, state_at_t_minus_5 in enumerate(labor_states):
            ax = axes[col_idx]

            # Filter to agents who were in this state at t=-5
            df_state = df_time[
                df_time["labor_supply_t_minus_5"] == state_at_t_minus_5
            ].copy()

            if len(df_state) == 0:
                # No data for this group
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"{state_labels[state_at_t_minus_5]}\nat t=-5",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.axis("off")
                continue

            # Count labor supply states at this time point
            state_counts = df_state["labor_supply_state"].value_counts()

            # Ensure all states are represented (fill missing with 0)
            state_counts = state_counts.reindex(labor_states, fill_value=0)

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                state_counts.values,
                labels=[state_labels[s] for s in state_counts.index],
                autopct="%1.1f%%",
                colors=[colors[s] for s in state_counts.index],
                startangle=90,
                textprops={"fontsize": 10},
            )

            # Adjust autopct font size
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight("bold")

            # Set title
            n_agents = len(df_state)
            ax.set_title(
                f"{state_labels[state_at_t_minus_5]}\nat t=-5\n(n={n_agents})",
                fontsize=14,
                fontweight="bold",
            )

        # Add overall title
        fig.suptitle(
            f"Labor Supply Transitions {time_point} Year{'s' if time_point > 1 else ''} After First Caregiving Spell\n(Grouped by Labor Supply at t=-5)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
        plt.close()


@pytask.mark.post_estimation
def task_plot_labor_supply_transitions_after_caregiving_t_minus_1(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_t1: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t1_after_caregiving_t_minus_1.png",
    path_to_plot_t3: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t3_after_caregiving_t_minus_1.png",
    path_to_plot_t5: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t5_after_caregiving_t_minus_1.png",
    path_to_plot_t10: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "labor_supply_transitions_t10_after_caregiving_t_minus_1.png",
) -> None:
    """Plot pie charts showing labor supply transitions after first caregiving spell.

    For each agent, determines the age of first caregiving spell (t=0).
    Groups agents by their labor supply state at t=-1 (1 year before first caregiving).
    For each group, creates pie charts showing the distribution of labor supply states
    at t=1, t=3, t=5, and t=10 (years after first caregiving).

    Creates 4 plots (one for each time point), each with 1 row x 4 columns
    (one column for each labor supply state at t=-1).

    """
    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Restrict to alive
    df = df[df["health"] != DEAD].copy()

    # Ensure agent/period columns
    df = _ensure_agent_period(df)

    # Add distance to first caregiving
    df = _add_distance_to_first_care(df)

    # Filter to agents who have a first caregiving period
    df = df[df["first_care_period"].notna()].copy()

    # Get labor supply state for each period
    df["labor_supply_state"] = df["choice"].apply(get_labor_supply_state)

    # Filter to relevant time points: t=-1, t=1, t=3, t=5, t=10
    time_points = [-1, 1, 3, 5, 10]
    df_filtered = df[df["distance_to_first_care"].isin(time_points)].copy()

    # Get labor supply at t=-1 for each agent
    df_t_minus_1 = df_filtered[df_filtered["distance_to_first_care"] == -1].copy()
    df_t_minus_1 = df_t_minus_1[["agent", "labor_supply_state"]].rename(
        columns={"labor_supply_state": "labor_supply_t_minus_1"}
    )

    # Merge back to get labor supply at t=-1 for all relevant periods
    df_with_t_minus_1 = df_filtered.merge(df_t_minus_1, on="agent", how="inner")

    # Filter to only agents who have data at t=-1
    df_with_t_minus_1 = df_with_t_minus_1[
        df_with_t_minus_1["labor_supply_t_minus_1"].notna()
    ].copy()

    # Define labor supply states and their order
    labor_states = ["retired", "unemployed", "part_time", "full_time"]
    state_labels = {
        "retired": "Retired",
        "unemployed": "Unemployed",
        "part_time": "Part-time",
        "full_time": "Full-time",
    }

    # Colors for pie charts
    colors = {
        "retired": JET_COLOR_MAP[0],
        "unemployed": JET_COLOR_MAP[1],
        "part_time": JET_COLOR_MAP[2],
        "full_time": JET_COLOR_MAP[3],
    }

    # Create plots for each time point
    for time_point, path_to_plot in (
        (1, path_to_plot_t1),
        (3, path_to_plot_t3),
        (5, path_to_plot_t5),
        (10, path_to_plot_t10),
    ):
        # Filter to this time point
        df_time = df_with_t_minus_1[
            df_with_t_minus_1["distance_to_first_care"] == time_point
        ].copy()

        # Create figure with 1 row x 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for col_idx, state_at_t_minus_1 in enumerate(labor_states):
            ax = axes[col_idx]

            # Filter to agents who were in this state at t=-1
            df_state = df_time[
                df_time["labor_supply_t_minus_1"] == state_at_t_minus_1
            ].copy()

            if len(df_state) == 0:
                # No data for this group
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"{state_labels[state_at_t_minus_1]}\nat t=-1",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.axis("off")
                continue

            # Count labor supply states at this time point
            state_counts = df_state["labor_supply_state"].value_counts()

            # Ensure all states are represented (fill missing with 0)
            state_counts = state_counts.reindex(labor_states, fill_value=0)

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                state_counts.values,
                labels=[state_labels[s] for s in state_counts.index],
                autopct="%1.1f%%",
                colors=[colors[s] for s in state_counts.index],
                startangle=90,
                textprops={"fontsize": 10},
            )

            # Adjust autopct font size
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight("bold")

            # Set title
            n_agents = len(df_state)
            ax.set_title(
                f"{state_labels[state_at_t_minus_1]}\nat t=-1\n(n={n_agents})",
                fontsize=14,
                fontweight="bold",
            )

        # Add overall title
        fig.suptitle(
            f"Labor Supply Transitions {time_point} Year{'s' if time_point > 1 else ''} After First Caregiving Spell\n(Grouped by Labor Supply at t=-1)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
        plt.close()
