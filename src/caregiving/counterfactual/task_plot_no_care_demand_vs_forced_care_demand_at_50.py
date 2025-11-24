"""Plot differences between no care demand counterfactual and forced care demand at 50.

This module compares the no care demand counterfactual (baseline) to the forced care
demand at age 50 counterfactual, plotting labor supply differences from age 50 to 70.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.counterfactual.plotting_utils import (
    _ensure_agent_period,
    calculate_outcomes,
    calculate_working_hours_weekly,
    prepare_dataframes_for_comparison,
    prepare_single_dataframe,
)
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    N_WEEKS_IN_YEAR,
    PART_TIME,
    WORK,
)
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)


@pytask.mark.counterfactual_differences_forced_care_demand_at_50
def task_plot_differences_no_care_demand_vs_forced_care_demand_at_50(
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_forced_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_50.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "forced_care_demand"
    / "differences_no_care_demand_vs_forced_care_demand_at_50.png",
    start_age: int = 43,
    end_age: int = 73,
) -> None:
    """Plot differences between no care demand and forced care demand at 50.

    Compares the forced care demand at age 50 counterfactual to the no care demand
    counterfactual. Plots labor supply differences from start_age to end_age,
    with ages on the x-axis.

    Steps:
      1) Restrict to alive agents.
      2) Ensure agent/period columns.
      3) Filter to age range [start_age, end_age].
      4) Build per-period outcomes (work, ft, pt) for both scenarios.
      5) Merge on (agent, period) and compute differences (forced - no_care).
      6) Average diffs by age and plot three series.

    Args:
        path_to_no_care_demand_data: Path to no care demand simulation data
        path_to_forced_care_demand_data: Path to forced care demand simulation data
        path_to_plot: Path to save the plot
        start_age: Starting age for the plot (default: 43)
        end_age: Ending age for the plot (default: 73)
    """
    # Load and prepare data (no ever_caregivers filter for this comparison)
    df_no_care, df_forced = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_no_care_demand_data),
        pd.read_pickle(path_to_forced_care_demand_data),
        ever_caregivers=False,
    )

    # Filter to age range [start_age, end_age]
    df_no_care = df_no_care[
        (df_no_care["age"] >= start_age) & (df_no_care["age"] <= end_age)
    ].copy()
    df_forced = df_forced[
        (df_forced["age"] >= start_age) & (df_forced["age"] <= end_age)
    ].copy()

    # Calculate outcomes
    no_care_outcomes = calculate_outcomes(df_no_care, choice_set_type="no_care_demand")
    forced_outcomes = calculate_outcomes(df_forced, choice_set_type="original")

    # Prepare for merging (using custom column names for this comparison)
    no_care_cols = df_no_care[["agent", "period", "age"]].copy()
    no_care_cols["work_no_care"] = no_care_outcomes["work"]
    no_care_cols["ft_no_care"] = no_care_outcomes["ft"]
    no_care_cols["pt_no_care"] = no_care_outcomes["pt"]

    forced_cols = df_forced[["agent", "period"]].copy()
    forced_cols["work_forced"] = forced_outcomes["work"]
    forced_cols["ft_forced"] = forced_outcomes["ft"]
    forced_cols["pt_forced"] = forced_outcomes["pt"]

    # Merge on (agent, period) to get matched differences
    # Difference = forced_care_demand - no_care_demand
    merged = no_care_cols.merge(forced_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_forced"] - merged["work_no_care"]
    merged["diff_ft"] = merged["ft_forced"] - merged["ft_no_care"]
    merged["diff_pt"] = merged["pt_forced"] - merged["pt_no_care"]

    # Average differences by age
    prof = (
        merged.groupby("age", observed=False)[["diff_work", "diff_ft", "diff_pt"]]
        .mean()
        .reset_index()
        .sort_values("age")
    )

    # Plot
    plt.figure(figsize=(12, 7))

    plt.plot(
        prof["age"],
        prof["diff_work"],
        label="Working",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        prof["age"],
        prof["diff_ft"],
        label="Full Time",
        color=JET_COLOR_MAP[1],
        linewidth=2,
    )
    plt.plot(
        prof["age"],
        prof["diff_pt"],
        label="Part Time",
        color=JET_COLOR_MAP[0],
        linewidth=2,
    )

    plt.axvline(x=50, color="k", linestyle=":", alpha=0.5)
    plt.axhline(y=0, color="k", linestyle="-", linewidth=2.5, alpha=0.8)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(
        "Proportion Working\n(Forced Care Demand at 50 - No Care Demand)", fontsize=16
    )
    plt.xlim(start_age, end_age)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to every 2 years to avoid overcrowding
    plt.xticks(range(start_age, end_age + 1, 2), fontsize=16)
    plt.yticks(fontsize=16)

    # Adjust y-limits to accommodate all data with padding
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Add legend
    plt.legend(
        ["Working", "Full Time", "Part Time"],
        loc="best",
        prop={"size": 16},
    )
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_forced_care_demand_at_50
def task_plot_full_time_differences_by_forced_age(
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_forced_care_demand_45: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_45.pkl",
    path_to_forced_care_demand_50: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_50.pkl",
    path_to_forced_care_demand_54: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_54.pkl",
    path_to_forced_care_demand_58: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_58.pkl",
    path_to_forced_care_demand_62: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_62.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "forced_care_demand"
    / "full_time_differences_by_forced_age.png",
    start_age: int = 43,
    end_age: int = 73,
) -> None:
    """Plot full-time differences for all forced starting ages.

    Compares no care demand to forced care demand at ages 45, 50, 54, 58, and 62.
    Plots full-time labor supply differences from start_age to end_age.

    Args:
        path_to_no_care_demand_data: Path to no care demand simulation data
        path_to_forced_care_demand_45: Path to forced care demand at 45 data
        path_to_forced_care_demand_50: Path to forced care demand at 50 data
        path_to_forced_care_demand_54: Path to forced care demand at 54 data
        path_to_forced_care_demand_58: Path to forced care demand at 58 data
        path_to_forced_care_demand_62: Path to forced care demand at 62 data
        path_to_plot: Path to save the plot
        start_age: Starting age for the plot (default: 43)
        end_age: Ending age for the plot (default: 73)
    """
    forced_ages = [45, 50, 54, 58, 62]
    forced_data_paths = [
        path_to_forced_care_demand_45,
        path_to_forced_care_demand_50,
        path_to_forced_care_demand_54,
        path_to_forced_care_demand_58,
        path_to_forced_care_demand_62,
    ]

    # Load no care demand data
    df_no_care = pd.read_pickle(path_to_no_care_demand_data)
    df_no_care = df_no_care[df_no_care["health"] != DEAD].copy()
    df_no_care = _ensure_agent_period(df_no_care)
    if isinstance(df_no_care.index, pd.MultiIndex):
        df_no_care = df_no_care.reset_index()
    df_no_care = df_no_care.reset_index(drop=True)
    df_no_care = df_no_care[
        (df_no_care["age"] >= start_age) & (df_no_care["age"] <= end_age)
    ].copy()

    # Compute no care demand full-time
    no_care_ft = (
        df_no_care["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    no_care_cols = df_no_care[["agent", "period", "age"]].copy()
    no_care_cols["ft_no_care"] = no_care_ft

    # Plot
    plt.figure(figsize=(12, 7))

    colors = [
        "#9467bd",  # purple (age 45)
        "#1f77b4",  # blue (age 50)
        "#ff7f0e",  # orange (age 54)
        "#2ca02c",  # green (age 58)
        "#d62728",  # red (age 62)
    ]

    for forced_age, forced_path, color in zip(
        forced_ages, forced_data_paths, colors, strict=False
    ):
        # Load forced care demand data
        df_forced = pd.read_pickle(forced_path)
        df_forced = df_forced[df_forced["health"] != DEAD].copy()
        df_forced = _ensure_agent_period(df_forced)
        if isinstance(df_forced.index, pd.MultiIndex):
            df_forced = df_forced.reset_index()
        df_forced = df_forced.reset_index(drop=True)
        df_forced = df_forced[
            (df_forced["age"] >= start_age) & (df_forced["age"] <= end_age)
        ].copy()

        # Compute forced care demand full-time
        forced_ft = (
            df_forced["choice"]
            .isin(np.asarray(FULL_TIME).ravel().tolist())
            .astype(float)
        )
        forced_cols = df_forced[["agent", "period"]].copy()
        forced_cols["ft_forced"] = forced_ft

        # Merge and compute differences
        merged = no_care_cols.merge(forced_cols, on=["agent", "period"], how="inner")
        merged["diff_ft"] = merged["ft_forced"] - merged["ft_no_care"]

        # Average by age
        prof = (
            merged.groupby("age", observed=False)["diff_ft"]
            .mean()
            .reset_index()
            .sort_values("age")
        )

        plt.plot(
            prof["age"],
            prof["diff_ft"],
            label=f"Forced at age {forced_age}",
            color=color,
            linewidth=2,
        )

        # Add circle marker at forced age
        forced_age_value = prof[prof["age"] == forced_age]["diff_ft"]
        if not forced_age_value.empty:
            plt.scatter(
                [forced_age],
                [forced_age_value.iloc[0]],
                color=color,
                s=100,  # size of marker
                zorder=5,  # make sure marker is on top
            )

    plt.axhline(y=0, color="k", linestyle="-", linewidth=2.5, alpha=0.8)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(
        "Full-Time Difference\n(Forced Care Demand - No Care Demand)", fontsize=16
    )
    plt.xlim(start_age, end_age)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to integer ages only
    plt.xticks(range(start_age, end_age + 1), fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="best", prop={"size": 14})
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_forced_care_demand_at_50
def task_plot_employment_rate_differences_by_forced_age(
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_forced_care_demand_45: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_45.pkl",
    path_to_forced_care_demand_50: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_50.pkl",
    path_to_forced_care_demand_54: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_54.pkl",
    path_to_forced_care_demand_58: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_58.pkl",
    path_to_forced_care_demand_62: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_62.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "forced_care_demand"
    / "employment_rate_differences_by_forced_age.png",
    start_age: int = 43,
    end_age: int = 73,
) -> None:
    """Plot employment rate differences for all forced starting ages.

    Compares no care demand to forced care demand at ages 45, 50, 54, 58, and 62.
    Plots employment rate (working = part-time or full-time) differences.

    Args:
        path_to_no_care_demand_data: Path to no care demand simulation data
        path_to_forced_care_demand_45: Path to forced care demand at 45 data
        path_to_forced_care_demand_50: Path to forced care demand at 50 data
        path_to_forced_care_demand_54: Path to forced care demand at 54 data
        path_to_forced_care_demand_58: Path to forced care demand at 58 data
        path_to_forced_care_demand_62: Path to forced care demand at 62 data
        path_to_plot: Path to save the plot
        start_age: Starting age for the plot (default: 43)
        end_age: Ending age for the plot (default: 73)
    """
    forced_ages = [45, 50, 54, 58, 62]
    forced_data_paths = [
        path_to_forced_care_demand_45,
        path_to_forced_care_demand_50,
        path_to_forced_care_demand_54,
        path_to_forced_care_demand_58,
        path_to_forced_care_demand_62,
    ]

    # Load no care demand data
    df_no_care = pd.read_pickle(path_to_no_care_demand_data)
    df_no_care = df_no_care[df_no_care["health"] != DEAD].copy()
    df_no_care = _ensure_agent_period(df_no_care)
    if isinstance(df_no_care.index, pd.MultiIndex):
        df_no_care = df_no_care.reset_index()
    df_no_care = df_no_care.reset_index(drop=True)
    df_no_care = df_no_care[
        (df_no_care["age"] >= start_age) & (df_no_care["age"] <= end_age)
    ].copy()

    # Compute no care demand employment rate (working = part-time or full-time)
    no_care_work = (
        df_no_care["choice"]
        .isin(np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    no_care_cols = df_no_care[["agent", "period", "age"]].copy()
    no_care_cols["work_no_care"] = no_care_work

    # Plot
    plt.figure(figsize=(12, 7))

    colors = [
        "#9467bd",  # purple (age 45)
        "#1f77b4",  # blue (age 50)
        "#ff7f0e",  # orange (age 54)
        "#2ca02c",  # green (age 58)
        "#d62728",  # red (age 62)
    ]

    for forced_age, forced_path, color in zip(
        forced_ages, forced_data_paths, colors, strict=False
    ):
        # Load forced care demand data
        df_forced = pd.read_pickle(forced_path)
        df_forced = df_forced[df_forced["health"] != DEAD].copy()
        df_forced = _ensure_agent_period(df_forced)
        if isinstance(df_forced.index, pd.MultiIndex):
            df_forced = df_forced.reset_index()
        df_forced = df_forced.reset_index(drop=True)
        df_forced = df_forced[
            (df_forced["age"] >= start_age) & (df_forced["age"] <= end_age)
        ].copy()

        # Compute forced care demand employment rate (working = part-time or full-time)
        forced_work = (
            df_forced["choice"].isin(np.asarray(WORK).ravel().tolist()).astype(float)
        )
        forced_cols = df_forced[["agent", "period"]].copy()
        forced_cols["work_forced"] = forced_work

        # Merge and compute differences
        merged = no_care_cols.merge(forced_cols, on=["agent", "period"], how="inner")
        merged["diff_work"] = merged["work_forced"] - merged["work_no_care"]

        # Average by age
        prof = (
            merged.groupby("age", observed=False)["diff_work"]
            .mean()
            .reset_index()
            .sort_values("age")
        )

        plt.plot(
            prof["age"],
            prof["diff_work"],
            label=f"Forced at age {forced_age}",
            color=color,
            linewidth=2,
        )

        # Add circle marker at forced age
        forced_age_value = prof[prof["age"] == forced_age]["diff_work"]
        if not forced_age_value.empty:
            plt.scatter(
                [forced_age],
                [forced_age_value.iloc[0]],
                color=color,
                s=100,  # size of marker
                zorder=5,  # make sure marker is on top
            )

    plt.axhline(y=0, color="k", linestyle="-", linewidth=2.5, alpha=0.8)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(
        "Employment Rate Difference\n(Forced Care Demand - No Care Demand)", fontsize=16
    )
    plt.xlim(start_age, end_age)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to integer ages only
    plt.xticks(range(start_age, end_age + 1), fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="best", prop={"size": 14})
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_forced_care_demand_at_50
def task_plot_part_time_differences_by_forced_age(
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_forced_care_demand_45: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_45.pkl",
    path_to_forced_care_demand_50: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_50.pkl",
    path_to_forced_care_demand_54: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_54.pkl",
    path_to_forced_care_demand_58: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_58.pkl",
    path_to_forced_care_demand_62: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_62.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "forced_care_demand"
    / "part_time_differences_by_forced_age.png",
    start_age: int = 43,
    end_age: int = 73,
) -> None:
    """Plot part-time differences for all forced starting ages.

    Compares no care demand to forced care demand at ages 45, 50, 54, 58, and 62.
    Plots part-time labor supply differences from start_age to end_age.

    Args:
        path_to_no_care_demand_data: Path to no care demand simulation data
        path_to_forced_care_demand_45: Path to forced care demand at 45 data
        path_to_forced_care_demand_50: Path to forced care demand at 50 data
        path_to_forced_care_demand_54: Path to forced care demand at 54 data
        path_to_forced_care_demand_58: Path to forced care demand at 58 data
        path_to_forced_care_demand_62: Path to forced care demand at 62 data
        path_to_plot: Path to save the plot
        start_age: Starting age for the plot (default: 43)
        end_age: Ending age for the plot (default: 73)
    """
    forced_ages = [45, 50, 54, 58, 62]
    forced_data_paths = [
        path_to_forced_care_demand_45,
        path_to_forced_care_demand_50,
        path_to_forced_care_demand_54,
        path_to_forced_care_demand_58,
        path_to_forced_care_demand_62,
    ]

    # Load no care demand data
    df_no_care = pd.read_pickle(path_to_no_care_demand_data)
    df_no_care = df_no_care[df_no_care["health"] != DEAD].copy()
    df_no_care = _ensure_agent_period(df_no_care)
    if isinstance(df_no_care.index, pd.MultiIndex):
        df_no_care = df_no_care.reset_index()
    df_no_care = df_no_care.reset_index(drop=True)
    df_no_care = df_no_care[
        (df_no_care["age"] >= start_age) & (df_no_care["age"] <= end_age)
    ].copy()

    # Compute no care demand part-time
    no_care_pt = (
        df_no_care["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    no_care_cols = df_no_care[["agent", "period", "age"]].copy()
    no_care_cols["pt_no_care"] = no_care_pt

    # Plot
    plt.figure(figsize=(12, 7))

    colors = [
        "#9467bd",  # purple (age 45)
        "#1f77b4",  # blue (age 50)
        "#ff7f0e",  # orange (age 54)
        "#2ca02c",  # green (age 58)
        "#d62728",  # red (age 62)
    ]

    for forced_age, forced_path, color in zip(
        forced_ages, forced_data_paths, colors, strict=False
    ):
        # Load forced care demand data
        df_forced = pd.read_pickle(forced_path)
        df_forced = df_forced[df_forced["health"] != DEAD].copy()
        df_forced = _ensure_agent_period(df_forced)
        if isinstance(df_forced.index, pd.MultiIndex):
            df_forced = df_forced.reset_index()
        df_forced = df_forced.reset_index(drop=True)
        df_forced = df_forced[
            (df_forced["age"] >= start_age) & (df_forced["age"] <= end_age)
        ].copy()

        # Compute forced care demand part-time
        forced_pt = (
            df_forced["choice"]
            .isin(np.asarray(PART_TIME).ravel().tolist())
            .astype(float)
        )
        forced_cols = df_forced[["agent", "period"]].copy()
        forced_cols["pt_forced"] = forced_pt

        # Merge and compute differences
        merged = no_care_cols.merge(forced_cols, on=["agent", "period"], how="inner")
        merged["diff_pt"] = merged["pt_forced"] - merged["pt_no_care"]

        # Average by age
        prof = (
            merged.groupby("age", observed=False)["diff_pt"]
            .mean()
            .reset_index()
            .sort_values("age")
        )

        plt.plot(
            prof["age"],
            prof["diff_pt"],
            label=f"Forced at age {forced_age}",
            color=color,
            linewidth=2,
        )

        # Add circle marker at forced age
        forced_age_value = prof[prof["age"] == forced_age]["diff_pt"]
        if not forced_age_value.empty:
            plt.scatter(
                [forced_age],
                [forced_age_value.iloc[0]],
                color=color,
                s=100,  # size of marker
                zorder=5,  # make sure marker is on top
            )

    plt.axhline(y=0, color="k", linestyle="-", linewidth=2.5, alpha=0.8)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(
        "Part-Time Difference\n(Forced Care Demand - No Care Demand)", fontsize=16
    )
    plt.xlim(start_age, end_age)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to integer ages only
    plt.xticks(range(start_age, end_age + 1), fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="best", prop={"size": 14})
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_forced_care_demand_at_50
def task_plot_informal_care_differences_by_forced_age(
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_forced_care_demand_45: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_45.pkl",
    path_to_forced_care_demand_50: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_50.pkl",
    path_to_forced_care_demand_54: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_54.pkl",
    path_to_forced_care_demand_58: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_58.pkl",
    path_to_forced_care_demand_62: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_62.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "forced_care_demand"
    / "informal_care_differences_by_forced_age.png",
    start_age: int = 43,
    end_age: int = 73,
) -> None:
    """Plot informal care probability differences for all forced starting ages.

    Compares no care demand to forced care demand at ages 45, 50, 54, 58, and 62.
    Plots informal care probability differences from start_age to end_age.

    Args:
        path_to_no_care_demand_data: Path to no care demand simulation data
        path_to_forced_care_demand_45: Path to forced care demand at 45 data
        path_to_forced_care_demand_50: Path to forced care demand at 50 data
        path_to_forced_care_demand_54: Path to forced care demand at 54 data
        path_to_forced_care_demand_58: Path to forced care demand at 58 data
        path_to_forced_care_demand_62: Path to forced care demand at 62 data
        path_to_plot: Path to save the plot
        start_age: Starting age for the plot (default: 43)
        end_age: Ending age for the plot (default: 73)
    """
    forced_ages = [45, 50, 54, 58, 62]
    forced_data_paths = [
        path_to_forced_care_demand_45,
        path_to_forced_care_demand_50,
        path_to_forced_care_demand_54,
        path_to_forced_care_demand_58,
        path_to_forced_care_demand_62,
    ]

    # Load no care demand data
    df_no_care = pd.read_pickle(path_to_no_care_demand_data)
    df_no_care = df_no_care[df_no_care["health"] != DEAD].copy()
    df_no_care = _ensure_agent_period(df_no_care)
    if isinstance(df_no_care.index, pd.MultiIndex):
        df_no_care = df_no_care.reset_index()
    df_no_care = df_no_care.reset_index(drop=True)
    df_no_care = df_no_care[
        (df_no_care["age"] >= start_age) & (df_no_care["age"] <= end_age)
    ].copy()

    # No care demand scenario has no informal care (reduced choice set)
    # So probability is always 0
    no_care_cols = df_no_care[["agent", "period", "age"]].copy()
    no_care_cols["care_no_care"] = 0.0

    # Plot
    plt.figure(figsize=(12, 7))

    colors = [
        "#9467bd",  # purple (age 45)
        "#1f77b4",  # blue (age 50)
        "#ff7f0e",  # orange (age 54)
        "#2ca02c",  # green (age 58)
        "#d62728",  # red (age 62)
    ]

    for forced_age, forced_path, color in zip(
        forced_ages, forced_data_paths, colors, strict=False
    ):
        # Load forced care demand data
        df_forced = pd.read_pickle(forced_path)
        df_forced = df_forced[df_forced["health"] != DEAD].copy()
        df_forced = _ensure_agent_period(df_forced)
        if isinstance(df_forced.index, pd.MultiIndex):
            df_forced = df_forced.reset_index()
        df_forced = df_forced.reset_index(drop=True)
        df_forced = df_forced[
            (df_forced["age"] >= start_age) & (df_forced["age"] <= end_age)
        ].copy()

        # Compute forced care demand informal care probability
        forced_care = (
            df_forced["choice"]
            .isin(np.asarray(INFORMAL_CARE).ravel().tolist())
            .astype(float)
        )
        forced_cols = df_forced[["agent", "period"]].copy()
        forced_cols["care_forced"] = forced_care

        # Merge and compute differences
        merged = no_care_cols.merge(forced_cols, on=["agent", "period"], how="inner")
        merged["diff_care"] = merged["care_forced"] - merged["care_no_care"]

        # Average by age
        prof = (
            merged.groupby("age", observed=False)["diff_care"]
            .mean()
            .reset_index()
            .sort_values("age")
        )

        plt.plot(
            prof["age"],
            prof["diff_care"],
            label=f"Forced at age {forced_age}",
            color=color,
            linewidth=2,
        )

        # Add circle marker at forced age
        forced_age_value = prof[prof["age"] == forced_age]["diff_care"]
        if not forced_age_value.empty:
            plt.scatter(
                [forced_age],
                [forced_age_value.iloc[0]],
                color=color,
                s=100,  # size of marker
                zorder=5,  # make sure marker is on top
            )

    plt.axhline(y=0, color="k", linestyle="-", linewidth=2.5, alpha=0.8)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(
        "Informal Care Probability Difference\n(Forced Care Demand - No Care Demand)",
        fontsize=16,
    )
    plt.xlim(start_age, end_age)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to integer ages only
    plt.xticks(range(start_age, end_age + 1), fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="best", prop={"size": 14})
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_forced_care_demand_at_50
def task_plot_job_offer_probability_differences_by_forced_age(
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_forced_care_demand_45: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_45.pkl",
    path_to_forced_care_demand_50: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_50.pkl",
    path_to_forced_care_demand_54: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_54.pkl",
    path_to_forced_care_demand_58: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_58.pkl",
    path_to_forced_care_demand_62: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_62.pkl",
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "forced_care_demand"
    / "job_offer_probability_differences_by_forced_age.png",
    start_age: int = 43,
    end_age: int = 73,
) -> None:
    """Plot job offer probability differences for all forced starting ages.

    Compares no care demand to forced care demand at ages 45, 50, 54, 58, and 62.
    Plots positive job offer probability (job_offer == 1) differences.

    Args:
        path_to_no_care_demand_data: Path to no care demand simulation data
        path_to_forced_care_demand_45: Path to forced care demand at 45 data
        path_to_forced_care_demand_50: Path to forced care demand at 50 data
        path_to_forced_care_demand_54: Path to forced care demand at 54 data
        path_to_forced_care_demand_58: Path to forced care demand at 58 data
        path_to_forced_care_demand_62: Path to forced care demand at 62 data
        path_to_options: Path to model options (not used but kept for consistency)
        path_to_plot: Path to save the plot
        start_age: Starting age for the plot (default: 43)
        end_age: Ending age for the plot (default: 73)
    """
    forced_ages = [45, 50, 54, 58, 62]
    forced_data_paths = [
        path_to_forced_care_demand_45,
        path_to_forced_care_demand_50,
        path_to_forced_care_demand_54,
        path_to_forced_care_demand_58,
        path_to_forced_care_demand_62,
    ]

    # Load no care demand data
    df_no_care = pd.read_pickle(path_to_no_care_demand_data)
    df_no_care = df_no_care[df_no_care["health"] != DEAD].copy()
    df_no_care = _ensure_agent_period(df_no_care)
    if isinstance(df_no_care.index, pd.MultiIndex):
        df_no_care = df_no_care.reset_index()
    df_no_care = df_no_care.reset_index(drop=True)
    df_no_care = df_no_care[
        (df_no_care["age"] >= start_age) & (df_no_care["age"] <= end_age)
    ].copy()

    # No care demand scenario: compute positive job offer probability (job_offer == 1)
    no_care_job_offer = (df_no_care["job_offer"] == 1).astype(float)
    no_care_cols = df_no_care[["agent", "period", "age"]].copy()
    no_care_cols["job_offer_no_care"] = no_care_job_offer

    # Plot
    plt.figure(figsize=(12, 7))

    colors = [
        "#9467bd",  # purple (age 45)
        "#1f77b4",  # blue (age 50)
        "#ff7f0e",  # orange (age 54)
        "#2ca02c",  # green (age 58)
        "#d62728",  # red (age 62)
    ]

    for forced_age, forced_path, color in zip(
        forced_ages, forced_data_paths, colors, strict=False
    ):
        # Load forced care demand data
        df_forced = pd.read_pickle(forced_path)
        df_forced = df_forced[df_forced["health"] != DEAD].copy()
        df_forced = _ensure_agent_period(df_forced)
        if isinstance(df_forced.index, pd.MultiIndex):
            df_forced = df_forced.reset_index()
        df_forced = df_forced.reset_index(drop=True)
        df_forced = df_forced[
            (df_forced["age"] >= start_age) & (df_forced["age"] <= end_age)
        ].copy()

        # Compute forced care demand positive job offer probability (job_offer == 1)
        forced_job_offer = (df_forced["job_offer"] == 1).astype(float)
        forced_cols = df_forced[["agent", "period"]].copy()
        forced_cols["job_offer_forced"] = forced_job_offer

        # Merge and compute differences
        merged = no_care_cols.merge(forced_cols, on=["agent", "period"], how="inner")
        merged["diff_job_offer"] = (
            merged["job_offer_forced"] - merged["job_offer_no_care"]
        )

        # Average by age
        prof = (
            merged.groupby("age", observed=False)["diff_job_offer"]
            .mean()
            .reset_index()
            .sort_values("age")
        )

        plt.plot(
            prof["age"],
            prof["diff_job_offer"],
            label=f"Forced at age {forced_age}",
            color=color,
            linewidth=2,
        )

        # Add circle marker at forced age
        forced_age_value = prof[prof["age"] == forced_age]["diff_job_offer"]
        if not forced_age_value.empty:
            plt.scatter(
                [forced_age],
                [forced_age_value.iloc[0]],
                color=color,
                s=100,  # size of marker
                zorder=5,  # make sure marker is on top
            )

    plt.axhline(y=0, color="k", linestyle="-", linewidth=2.5, alpha=0.8)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(
        "Job Offer Probability Difference\n(Forced Care Demand - No Care Demand)",
        fontsize=16,
    )
    plt.xlim(start_age, end_age)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to integer ages only
    plt.xticks(range(start_age, end_age + 1), fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="best", prop={"size": 14})
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_forced_care_demand_at_50
def task_plot_working_hours_differences_by_forced_age(
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_forced_care_demand_45: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_45.pkl",
    path_to_forced_care_demand_50: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_50.pkl",
    path_to_forced_care_demand_54: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_54.pkl",
    path_to_forced_care_demand_58: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_58.pkl",
    path_to_forced_care_demand_62: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_62.pkl",
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "forced_care_demand"
    / "working_hours_differences_by_forced_age.png",
    start_age: int = 43,
    end_age: int = 73,
) -> None:
    """Plot working hours differences for all forced starting ages.

    Compares no care demand to forced care demand at ages 45, 50, 54, 58, and 62.
    Plots average weekly working hours differences from start_age to end_age.

    Args:
        path_to_no_care_demand_data: Path to no care demand simulation data
        path_to_forced_care_demand_45: Path to forced care demand at 45 data
        path_to_forced_care_demand_50: Path to forced care demand at 50 data
        path_to_forced_care_demand_54: Path to forced care demand at 54 data
        path_to_forced_care_demand_58: Path to forced care demand at 58 data
        path_to_forced_care_demand_62: Path to forced care demand at 62 data
        path_to_options: Path to model options to access working hours parameters
        path_to_plot: Path to save the plot
        start_age: Starting age for the plot (default: 43)
        end_age: Ending age for the plot (default: 73)
    """
    import pickle

    # Load options to access model_params for working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]

    forced_ages = [45, 50, 54, 58, 62]
    forced_data_paths = [
        path_to_forced_care_demand_45,
        path_to_forced_care_demand_50,
        path_to_forced_care_demand_54,
        path_to_forced_care_demand_58,
        path_to_forced_care_demand_62,
    ]

    # Load and prepare no care demand data
    df_no_care = prepare_single_dataframe(
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=False,
    )
    df_no_care = df_no_care[
        (df_no_care["age"] >= start_age) & (df_no_care["age"] <= end_age)
    ].copy()

    # Calculate working hours for no care demand scenario
    no_care_hours_weekly = calculate_working_hours_weekly(
        df_no_care, model_params, choice_set_type="no_care_demand"
    )
    no_care_cols = df_no_care[["agent", "period", "age"]].copy()
    no_care_cols["hours_no_care"] = no_care_hours_weekly

    # Collect all data for plotting
    colors = [
        "#9467bd",  # purple (age 45)
        "#1f77b4",  # blue (age 50)
        "#ff7f0e",  # orange (age 54)
        "#2ca02c",  # green (age 58)
        "#d62728",  # red (age 62)
    ]

    all_prof_data = []
    for forced_age, forced_path in zip(forced_ages, forced_data_paths, strict=False):
        # Load and prepare forced care demand data
        df_forced = prepare_single_dataframe(
            pd.read_pickle(forced_path),
            ever_caregivers=False,
        )
        df_forced = df_forced[
            (df_forced["age"] >= start_age) & (df_forced["age"] <= end_age)
        ].copy()

        # Calculate working hours for forced care demand scenario
        forced_hours_weekly = calculate_working_hours_weekly(
            df_forced, model_params, choice_set_type="original"
        )
        forced_cols = df_forced[["agent", "period"]].copy()
        forced_cols["hours_forced"] = forced_hours_weekly

        # Merge and compute differences
        merged = no_care_cols.merge(forced_cols, on=["agent", "period"], how="inner")
        merged["diff_hours"] = merged["hours_forced"] - merged["hours_no_care"]

        # Average by age
        prof = (
            merged.groupby("age", observed=False)["diff_hours"]
            .mean()
            .reset_index()
            .sort_values("age")
        )
        prof["forced_age"] = forced_age
        all_prof_data.append(prof)

    # Combine all profile data
    combined_prof = pd.concat(all_prof_data, ignore_index=True)

    # Plot
    plt.figure(figsize=(12, 7))
    for forced_age, color in zip(forced_ages, colors, strict=False):
        prof_age = combined_prof[combined_prof["forced_age"] == forced_age].copy()
        plt.plot(
            prof_age["age"],
            prof_age["diff_hours"],
            label=f"Forced at age {forced_age}",
            color=color,
            linewidth=2,
        )
        # Add circle marker at forced age
        forced_age_value = prof_age[prof_age["age"] == forced_age]["diff_hours"]
        if not forced_age_value.empty:
            plt.scatter(
                [forced_age],
                [forced_age_value.iloc[0]],
                color=color,
                s=100,
                zorder=5,
            )

    plt.axhline(y=0, color="k", linestyle="-", linewidth=2.5, alpha=0.8)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(
        "Working Hours Difference\n(Forced Care Demand - No Care Demand)", fontsize=16
    )
    plt.xlim(start_age, end_age)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(start_age, end_age + 1), fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="best", prop={"size": 14})
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()
