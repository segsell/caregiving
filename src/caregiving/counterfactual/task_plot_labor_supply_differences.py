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
from caregiving.model.shared import FULL_TIME, PART_TIME, SEX, WORK
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)


def task_plot_labor_supply_differences(
    path_to_original_data: Path = BLD / "solve_and_simulate" / "simulated_data.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "labor_supply_differences_by_age.png",
) -> None:
    """Plot differences in labor supply by age between scenarios."""

    # Load data
    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

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
    """Compute labor supply shares by age - corrected version."""

    # Select only the columns we need to minimize memory usage
    subset_df = df[["choice", "age"]].copy()

    if is_original:
        work_choices = np.asarray(WORK).ravel().tolist()
        part_time_choices = np.asarray(PART_TIME).ravel().tolist()
        full_time_choices = np.asarray(FULL_TIME).ravel().tolist()
    else:  # No-care-demand model
        work_choices = np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist()
        part_time_choices = np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist()
        full_time_choices = np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist()

    # Create boolean columns using the converted arrays
    subset_df["is_working"] = subset_df["choice"].isin(work_choices)
    subset_df["is_part_time"] = subset_df["choice"].isin(part_time_choices)
    subset_df["is_full_time"] = subset_df["choice"].isin(full_time_choices)

    # Use pandas' optimized groupby with mean aggregation
    shares = (
        subset_df.groupby("age")[["is_working", "is_part_time", "is_full_time"]]
        .mean()
        .reset_index()
    )

    return shares


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
    axes[0].set_xlim(30, 80)
    axes[0].set_ylim(common_y_min, common_y_max)

    axes[1].set_xlabel("Age")
    axes[1].set_xlim(30, 80)
    axes[1].set_ylim(common_y_min, common_y_max)

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Labor supply difference plot saved to: {path_to_plot}")
