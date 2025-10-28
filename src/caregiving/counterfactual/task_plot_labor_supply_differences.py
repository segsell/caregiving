"""Plot differences in labor supply by age between scenarios.

Original and no-care-demand scenario.

"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    PART_TIME,
    RETIREMENT,
    SEX,
    UNEMPLOYED,
    WORK,
)
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)


@pytask.mark.counterfactual_differences
def task_plot_labor_supply_differences(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "labor_supply_differences_by_age.png",
    ever_caregivers: bool = True,
) -> None:
    """Plot differences in labor supply by age between scenarios."""

    # Load data
    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure an 'agent' column exists (source data often indexed by
    # MultiIndex agent, period)
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Optional sample restriction to ever-caregivers in the original scenario
    if ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    # Compute labor supply shares by age for both scenarios
    original_shares = compute_labor_supply_shares_by_age(df_original, is_original=True)
    no_care_demand_shares = compute_labor_supply_shares_by_age(
        df_no_care_demand, is_original=False
    )

    # Compute differences
    differences = compute_labor_supply_differences(
        original_shares, no_care_demand_shares
    )
    # Create plot
    create_labor_supply_difference_plot(differences, path_to_plot)


def compute_labor_supply_shares_by_age(
    df: pd.DataFrame, is_original: bool = True
) -> pd.DataFrame:
    """Compute labor supply shares by age using value_counts approach.

    Mirrors the approach in the model-fit plotting: map raw choices into
    4 aggregate groups (retired, unemployed, part-time, full-time), then
    compute age-specific normalized counts.
    """

    # Ensure required columns
    df_local = df[["choice", "age"]].copy()

    if is_original:
        choice_groups = {
            0: np.asarray(RETIREMENT).ravel().tolist(),
            1: np.asarray(UNEMPLOYED).ravel().tolist(),
            2: np.asarray(PART_TIME).ravel().tolist(),
            3: np.asarray(FULL_TIME).ravel().tolist(),
        }
    else:
        choice_groups = {
            0: np.asarray(RETIREMENT_NO_CARE_DEMAND).ravel().tolist(),
            1: np.asarray(UNEMPLOYED_NO_CARE_DEMAND).ravel().tolist(),
            2: np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist(),
            3: np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist(),
        }

    # Build a flat map from raw codes to aggregate group id
    code_to_group = {}
    for group_id, codes in choice_groups.items():
        for code in codes:
            code_to_group[code] = group_id

    # Map raw choice to group id
    df_local["choice_group"] = (
        df_local["choice"].map(code_to_group).fillna(0).astype(int)
    )

    # Compute normalized shares per age
    shares_by_age = (
        df_local.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .rename(columns={0: "retired", 1: "unemployed", 2: "part_time", 3: "full_time"})
        .reset_index()
    )

    # Construct output with desired columns
    out = pd.DataFrame(
        {
            "age": shares_by_age["age"],
            "is_full_time": shares_by_age.get("full_time", 0),
            "is_part_time": shares_by_age.get("part_time", 0),
        }
    )
    out["is_working"] = out["is_full_time"] + out["is_part_time"]
    out["is_not_working"] = 1 - out["is_working"]

    return out


def compute_labor_supply_differences(
    original_shares: pd.DataFrame, no_care_demand_shares: pd.DataFrame
) -> pd.DataFrame:
    """Compute differences in labor supply shares between scenarios."""

    # Merge on age only (assuming single sex model)
    merged = pd.merge(
        original_shares,
        no_care_demand_shares,
        on=["age"],
        suffixes=("_original", "_no_care_demand"),
    )

    # Compute differences (no_care_demand - original)
    merged["working_diff"] = (
        merged["is_working_no_care_demand"] - merged["is_working_original"]
    )
    merged["part_time_diff"] = (
        merged["is_part_time_no_care_demand"] - merged["is_part_time_original"]
    )
    merged["full_time_diff"] = (
        merged["is_full_time_no_care_demand"] - merged["is_full_time_original"]
    )

    return merged


def create_labor_supply_difference_plot(
    differences: pd.DataFrame, path_to_plot: Path
) -> None:
    """Create plot showing labor supply differences by age."""

    # Create figure with subplots (single row since only women)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Labor Supply Differences by Age: No Care Demand vs Original (Women)",
        fontsize=16,
    )

    # Plot 1: Working (any employment)
    axes[0].plot(differences["age"], differences["working_diff"], "b-", linewidth=2)
    axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_title("Working Rate Difference")
    axes[0].set_ylabel("Difference (No Care - Original)")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Part-time vs Full-time
    axes[1].plot(
        differences["age"],
        differences["part_time_diff"],
        "g-",
        linewidth=2,
        label="Part-time",
    )
    axes[1].plot(
        differences["age"],
        differences["full_time_diff"],
        "r-",
        linewidth=2,
        label="Full-time",
    )
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_title("Part-time vs Full-time Differences")
    axes[1].set_ylabel("Difference (No Care - Original)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Calculate common y-axis range for both plots
    working_diff = differences["working_diff"].values
    part_time_diff = differences["part_time_diff"].values
    full_time_diff = differences["full_time_diff"].values

    # Find the overall min and max across all difference series
    all_values = np.concatenate([working_diff, part_time_diff, full_time_diff])
    common_min = np.min(all_values)
    common_max = np.max(all_values)

    # Add padding
    common_range = common_max - common_min
    padding = common_range * 0.1
    common_y_min = common_min - padding
    common_y_max = common_max + padding

    # Set x-axis properties and common y-axis range
    axes[0].set_xlabel("Age")
    axes[0].set_xlim(30, 70)
    axes[0].set_ylim(common_y_min, common_y_max)

    axes[1].set_xlabel("Age")
    axes[1].set_xlim(30, 70)
    axes[1].set_ylim(common_y_min, common_y_max)

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Labor supply difference plot saved to: {path_to_plot}")


def create_labor_supply_age_profile_plot(
    df_original: pd.DataFrame,
    df_no_care_demand: pd.DataFrame,
    path_to_plot: Path,
) -> None:
    """Plot FT, PT and Not Working age profiles for both scenarios in one plot."""

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    shares_original = compute_labor_supply_shares_by_age(df_original, is_original=True)
    shares_ncd = compute_labor_supply_shares_by_age(
        df_no_care_demand, is_original=False
    )

    merged = pd.merge(
        shares_original,
        shares_ncd,
        on=["age"],
        suffixes=("_original", "_no_care_demand"),
    )

    plt.figure(figsize=(12, 7))
    plt.title("Labor Supply Age Profiles: Original vs No Care Demand")

    # Full-time
    plt.plot(
        merged["age"],
        merged["is_full_time_original"],
        label="Full-time (Original)",
        color="tab:red",
        linewidth=2,
    )
    plt.plot(
        merged["age"],
        merged["is_full_time_no_care_demand"],
        label="Full-time (No Care)",
        color="tab:red",
        linestyle="--",
        linewidth=2,
    )

    # Part-time
    plt.plot(
        merged["age"],
        merged["is_part_time_original"],
        label="Part-time (Original)",
        color="tab:green",
        linewidth=2,
    )
    plt.plot(
        merged["age"],
        merged["is_part_time_no_care_demand"],
        label="Part-time (No Care)",
        color="tab:green",
        linestyle="--",
        linewidth=2,
    )

    # Not working (unemployed or retired)
    plt.plot(
        merged["age"],
        merged["is_not_working_original"],
        label="Not working (Original)",
        color="tab:blue",
        linewidth=2,
    )
    plt.plot(
        merged["age"],
        merged["is_not_working_no_care_demand"],
        label="Not working (No Care)",
        color="tab:blue",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Age")
    plt.ylabel("Share")
    plt.xlim(30, 70)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences
def task_plot_labor_supply_age_profiles(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "labor_supply_age_profiles.png",
    ever_caregivers: bool = True,
) -> None:
    """Task: plot FT, PT, Not Working age profiles for both scenarios."""

    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Ensure 'agent' exists
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Optional sample restriction to ever-caregivers in the original scenario
    if ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    create_labor_supply_age_profile_plot(
        df_original=df_original,
        df_no_care_demand=df_no_care_demand,
        path_to_plot=path_to_plot,
    )
