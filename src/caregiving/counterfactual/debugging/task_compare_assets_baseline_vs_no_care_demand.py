"""Compare assets by age between baseline (estimated_params) and no-care-demand models.

This module creates a comparison plot of average assets_begin_of_period by age
for two scenarios:
- Baseline model (estimated_params)
- No care demand model

This helps debug differences in asset accumulation between these two counterfactuals.
"""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import DEAD, FULL_TIME, INFORMAL_CARE, PART_TIME


@pytask.mark.debug_assets_baseline
def task_compare_assets_baseline_vs_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "assets_by_age_baseline_vs_no_care_demand.png",
) -> None:
    """Compare average assets_begin_of_period by age between two scenarios.

    Creates a plot comparing average assets_begin_of_period by age for:
    - Baseline model (estimated_params)
    - No care demand model

    This helps understand how removing care demand affects asset accumulation
    in the baseline model.

    Args:
        path_to_specs: Path to model specifications
        path_to_baseline_data: Path to baseline (estimated_params) simulated data
        path_to_no_care_demand_data: Path to no-care-demand
            simulated data
        path_to_plot: Path to save the comparison plot
    """
    # Load specifications
    specs = pickle.load(path_to_specs.open("rb"))

    # Load simulated data
    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Filter to alive agents
    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure age column exists (if not, compute from period)
    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    # Verify assets_begin_of_period column exists
    if "assets_begin_of_period" not in df_baseline.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in baseline data. "
            f"Available columns: {df_baseline.columns.tolist()}"
        )
    if "assets_begin_of_period" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in no_care_demand data. "
            f"Available columns: {df_no_care_demand.columns.tolist()}"
        )

    # Calculate average assets by age for each scenario
    avg_assets_baseline = (
        df_baseline.groupby("age", observed=False)["assets_begin_of_period"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_baseline.columns = ["age", "avg_assets"]

    avg_assets_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["assets_begin_of_period"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_no_care_demand.columns = ["age", "avg_assets"]

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Plot no inheritance
    plt.plot(
        avg_assets_baseline["age"],
        avg_assets_baseline["avg_assets"],
        label="Baseline",
        color="steelblue",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4,
    )

    # Plot no care demand and no inheritance
    plt.plot(
        avg_assets_no_care_demand["age"],
        avg_assets_no_care_demand["avg_assets"],
        label="No Care Demand & Baseline",
        color="darkorange",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=4,
    )

    # Formatting
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Average Assets Begin of Period", fontsize=16)
    plt.title(
        "Average Assets by Age: Baseline vs No Care Demand & Baseline",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


# @pytask.mark.debug_assets_baseline
def task_compare_asset_differences_baseline_vs_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "asset_differences_by_age_baseline_vs_no_care_demand.png",
) -> None:
    """Plot asset differences by age between baseline and no-care-demand scenarios.

    Creates a plot of the difference in average assets_begin_of_period by age:

        diff(age) = avg_assets_baseline(age)
                    - avg_assets_no_care_demand(age)

    Positive values indicate higher assets in the baseline model relative
    to the no-care-demand model at that age.
    """
    # Load specifications
    specs = pickle.load(path_to_specs.open("rb"))

    # Load simulated data
    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Filter to alive agents
    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure age column exists (if not, compute from period)
    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    # Verify assets_begin_of_period column exists
    if "assets_begin_of_period" not in df_baseline.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in baseline data. "
            f"Available columns: {df_baseline.columns.tolist()}"
        )
    if "assets_begin_of_period" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in "
            "no_care_demand data. "
            f"Available columns: {df_no_care_demand.columns.tolist()}"
        )

    # Calculate average assets by age for each scenario
    avg_assets_baseline = (
        df_baseline.groupby("age", observed=False)["assets_begin_of_period"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_baseline.columns = ["age", "avg_assets_baseline"]

    avg_assets_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["assets_begin_of_period"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_no_care_demand.columns = [
        "age",
        "avg_assets_no_care_demand",
    ]

    # Merge on age to compute differences
    merged = avg_assets_baseline.merge(avg_assets_no_care_demand, on="age", how="inner")
    merged["diff_assets"] = (
        merged["avg_assets_baseline"] - merged["avg_assets_no_care_demand"]
    )

    # Create the difference plot
    plt.figure(figsize=(12, 7))

    plt.plot(
        merged["age"],
        merged["diff_assets"],
        label="Baseline - No Care Demand & Baseline",
        color="purple",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4,
    )

    # Add horizontal zero line for reference
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.7)

    # Formatting
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Difference in Average Assets Begin of Period", fontsize=16)
    plt.title(
        "Difference in Average Assets by Age\n" "Baseline vs No Care Demand & Baseline",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot_diff, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.debug_accumulated_difference
@pytask.mark.debug_assets_baseline
def task_compare_accumulated_difference_baseline_vs_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_consumption: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "accumulated_consumption_difference_by_age.png",
    path_to_plot_savings_dec: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "accumulated_savings_dec_difference_by_age.png",
) -> None:
    """Plot accumulated differences by age.

    Compares baseline and no-care-demand scenarios.

    Creates plots of the cumulative sum of differences by age for:
    - Consumption: cumulative sum of consumption differences
    - Savings Decision: cumulative sum of savings decision differences

    For each outcome:
    - First computes the difference in average value at each age:
        diff(age) = avg_baseline(age) - avg_no_care_demand(age)
    - Then computes the cumulative sum (accumulated difference):
        accumulated_diff(age) = sum(diff(age') for age' <= age)

    This shows the total accumulated difference up to each age,
    which helps understand the cumulative impact of care demand over the
    life cycle.

    Creates a plot of the cumulative sum of consumption differences by age:
    - First computes the difference in average consumption at each age:
        diff(age) = avg_consumption_baseline(age) - avg_consumption_no_care_demand(age)
    - Then computes the cumulative sum (accumulated difference):
        accumulated_diff(age) = sum(diff(age') for age' <= age)

    This shows the total accumulated difference in consumption up to each age,
    which helps understand the cumulative impact of care demand on consumption
    over the life cycle.

    Args:
        path_to_specs: Path to model specifications
        path_to_baseline_data: Path to baseline (estimated_params) simulated data
        path_to_no_care_demand_data: Path to no-care-demand simulated data
        path_to_plot_consumption: Path to save the accumulated consumption
            difference plot
        path_to_plot_savings_dec: Path to save the accumulated savings decision
            difference plot
    """
    # Load specifications
    specs = pickle.load(path_to_specs.open("rb"))

    # Load simulated data
    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Filter to alive agents
    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure age column exists (if not, compute from period)
    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    # ============================================================================
    # Plot 1: Accumulated Consumption Difference
    # ============================================================================
    # Extract consumption variable (may be in aux dictionary)
    if "consumption" not in df_baseline.columns:
        df_baseline["consumption"] = _extract_aux_variable(df_baseline, "consumption")
    if "consumption" not in df_no_care_demand.columns:
        df_no_care_demand["consumption"] = _extract_aux_variable(
            df_no_care_demand, "consumption"
        )

    # Verify consumption column exists
    if "consumption" not in df_baseline.columns:
        raise ValueError(
            "Column 'consumption' not found in baseline data. "
            f"Available columns: {df_baseline.columns.tolist()}"
        )
    if "consumption" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'consumption' not found in no_care_demand data. "
            f"Available columns: {df_no_care_demand.columns.tolist()}"
        )

    # Calculate average consumption by age for each scenario
    avg_consumption_baseline = (
        df_baseline.groupby("age", observed=False)["consumption"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_consumption_baseline.columns = ["age", "avg_consumption_baseline"]

    avg_consumption_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["consumption"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_consumption_no_care_demand.columns = ["age", "avg_consumption_no_care_demand"]

    # Merge on age to compute differences
    merged_consumption = avg_consumption_baseline.merge(
        avg_consumption_no_care_demand, on="age", how="inner"
    )
    merged_consumption["diff_consumption"] = (
        merged_consumption["avg_consumption_baseline"]
        - merged_consumption["avg_consumption_no_care_demand"]
    )

    # Compute accumulated (cumulative) difference
    merged_consumption = merged_consumption.sort_values("age").reset_index(drop=True)
    merged_consumption["accumulated_diff_consumption"] = merged_consumption[
        "diff_consumption"
    ].cumsum()

    # Create the accumulated consumption difference plot
    plt.figure(figsize=(12, 7))
    plt.plot(
        merged_consumption["age"],
        merged_consumption["accumulated_diff_consumption"],
        label="Accumulated Consumption Difference (Baseline - No Care Demand)",
        color="darkgreen",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4,
    )
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.7)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Accumulated Consumption Difference", fontsize=16)
    plt.title(
        "Accumulated Consumption Difference by Age\n"
        "Baseline vs No Care Demand & Baseline\n"
        "(Cumulative sum of consumption differences)",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot_consumption, dpi=300, bbox_inches="tight")
    plt.close()

    # ============================================================================
    # Plot 2: Accumulated Savings Decision Difference
    # ============================================================================
    # Extract savings_dec variable (may be in aux dictionary)
    if "savings_dec" not in df_baseline.columns:
        df_baseline["savings_dec"] = _extract_aux_variable(df_baseline, "savings_dec")
    if "savings_dec" not in df_no_care_demand.columns:
        df_no_care_demand["savings_dec"] = _extract_aux_variable(
            df_no_care_demand, "savings_dec"
        )

    # Verify savings_dec column exists
    if "savings_dec" not in df_baseline.columns:
        raise ValueError(
            "Column 'savings_dec' not found in baseline data. "
            f"Available columns: {df_baseline.columns.tolist()}"
        )
    if "savings_dec" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'savings_dec' not found in no_care_demand data. "
            f"Available columns: {df_no_care_demand.columns.tolist()}"
        )

    # Calculate average savings_dec by age for each scenario
    avg_savings_dec_baseline = (
        df_baseline.groupby("age", observed=False)["savings_dec"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_savings_dec_baseline.columns = ["age", "avg_savings_dec_baseline"]

    avg_savings_dec_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["savings_dec"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_savings_dec_no_care_demand.columns = ["age", "avg_savings_dec_no_care_demand"]

    # Merge on age to compute differences
    merged_savings_dec = avg_savings_dec_baseline.merge(
        avg_savings_dec_no_care_demand, on="age", how="inner"
    )
    merged_savings_dec["diff_savings_dec"] = (
        merged_savings_dec["avg_savings_dec_baseline"]
        - merged_savings_dec["avg_savings_dec_no_care_demand"]
    )

    # Compute accumulated (cumulative) difference
    merged_savings_dec = merged_savings_dec.sort_values("age").reset_index(drop=True)
    merged_savings_dec["accumulated_diff_savings_dec"] = merged_savings_dec[
        "diff_savings_dec"
    ].cumsum()

    # Create the accumulated savings decision difference plot
    plt.figure(figsize=(12, 7))
    plt.plot(
        merged_savings_dec["age"],
        merged_savings_dec["accumulated_diff_savings_dec"],
        label="Accumulated Savings Decision Difference (Baseline - No Care Demand)",
        color="darkblue",
        linewidth=2.5,
        linestyle="-",
        marker="s",
        markersize=4,
    )
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.7)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Accumulated Savings Decision Difference", fontsize=16)
    plt.title(
        "Accumulated Savings Decision Difference by Age\n"
        "Baseline vs No Care Demand & Baseline\n"
        "(Cumulative sum of savings decision differences)",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot_savings_dec, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.debug_accumulated_difference
@pytask.mark.debug_assets_baseline
def task_compare_rate_differences_baseline_vs_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_consumption_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "consumption_rate_difference_by_age.png",
) -> None:
    """Plot consumption rate difference by age.

    Compares baseline and no-care-demand scenarios.

    Creates a plot of the consumption rate difference at each age:
    - First computes consumption_rate = consumption / total_income for each
      scenario
    - Then computes the difference in average consumption_rate at each age:
        diff(age) = avg_consumption_rate_baseline(age)
                  - avg_consumption_rate_no_care_demand(age)

    Note: consumption_rate is the complement of savings_rate:
        consumption_rate = consumption / total_income
        savings_rate = savings_dec / total_income
                      = (total_income - consumption) / total_income
        Therefore: consumption_rate + savings_rate = 1

    This shows the difference in consumption rate at each age,
    which helps understand how care demand affects the fraction of income
    consumed at different ages.

    Args:
        path_to_specs: Path to model specifications
        path_to_baseline_data: Path to baseline (estimated_params) simulated
            data
        path_to_no_care_demand_data: Path to no-care-demand simulated data
        path_to_plot_consumption_rate: Path to save the consumption rate
            difference plot
    """
    # Load specifications
    specs = pickle.load(path_to_specs.open("rb"))

    # Load simulated data
    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Filter to alive agents
    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure age column exists (if not, compute from period)
    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    # Extract consumption and total_income variables (may be in aux dictionary)
    if "consumption" not in df_baseline.columns:
        df_baseline["consumption"] = _extract_aux_variable(df_baseline, "consumption")
    if "consumption" not in df_no_care_demand.columns:
        df_no_care_demand["consumption"] = _extract_aux_variable(
            df_no_care_demand, "consumption"
        )

    if "total_income" not in df_baseline.columns:
        df_baseline["total_income"] = _extract_aux_variable(df_baseline, "total_income")
    if "total_income" not in df_no_care_demand.columns:
        df_no_care_demand["total_income"] = _extract_aux_variable(
            df_no_care_demand, "total_income"
        )

    # Verify required columns exist
    required_cols = ["consumption", "total_income"]
    for col in required_cols:
        if col not in df_baseline.columns:
            raise ValueError(
                f"Column '{col}' not found in baseline data. "
                f"Available columns: {df_baseline.columns.tolist()}"
            )
        if col not in df_no_care_demand.columns:
            raise ValueError(
                f"Column '{col}' not found in no_care_demand data. "
                f"Available columns: {df_no_care_demand.columns.tolist()}"
            )

    # Compute consumption_rate = consumption / total_income
    # Handle division by zero: set to NaN where total_income is 0 or negative
    df_baseline["consumption_rate"] = np.where(
        df_baseline["total_income"] > 0,
        df_baseline["consumption"] / df_baseline["total_income"],
        np.nan,
    )
    df_no_care_demand["consumption_rate"] = np.where(
        df_no_care_demand["total_income"] > 0,
        df_no_care_demand["consumption"] / df_no_care_demand["total_income"],
        np.nan,
    )

    # Calculate average consumption_rate by age for each scenario
    # Exclude NaN values from the mean calculation
    avg_consumption_rate_baseline = (
        df_baseline.groupby("age", observed=False)["consumption_rate"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_consumption_rate_baseline.columns = ["age", "avg_consumption_rate_baseline"]

    avg_consumption_rate_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["consumption_rate"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_consumption_rate_no_care_demand.columns = [
        "age",
        "avg_consumption_rate_no_care_demand",
    ]

    # Merge on age to compute differences
    merged = avg_consumption_rate_baseline.merge(
        avg_consumption_rate_no_care_demand, on="age", how="inner"
    )
    merged["diff_consumption_rate"] = (
        merged["avg_consumption_rate_baseline"]
        - merged["avg_consumption_rate_no_care_demand"]
    )

    # Sort by age for plotting
    merged = merged.sort_values("age").reset_index(drop=True)

    # Create the consumption rate difference plot
    plt.figure(figsize=(12, 7))
    plt.plot(
        merged["age"],
        merged["diff_consumption_rate"],
        label="Consumption Rate Difference (Baseline - No Care Demand)",
        color="darkred",
        linewidth=2.5,
        linestyle="-",
        marker="^",
        markersize=4,
    )
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.7)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Consumption Rate Difference", fontsize=16)
    plt.title(
        "Consumption Rate Difference by Age\n"
        "Baseline vs No Care Demand & Baseline\n"
        "(consumption_rate = consumption / total_income)",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot_consumption_rate, dpi=300, bbox_inches="tight")
    plt.close()


def _extract_aux_variable(df_sim: pd.DataFrame, var_name: str) -> pd.Series:
    """Extract a variable from either direct column or aux dictionary."""
    if var_name in df_sim.columns:
        return df_sim[var_name]
    elif "aux" in df_sim.columns:
        return df_sim["aux"].apply(
            lambda x: (x.get(var_name, np.nan) if isinstance(x, dict) else np.nan)
        )
    else:
        return pd.Series(np.nan, index=df_sim.index)


def _compute_working_shares_from_choice(
    df: pd.DataFrame, outcome_col: str
) -> pd.Series:
    """Compute working shares from choice variable.

    Args:
        df: DataFrame with 'choice' and 'age' columns
        outcome_col: One of 'share_working_part_time',
            'share_working_full_time', 'share_not_working'

    Returns:
        Series with computed shares
    """
    part_time_codes = np.asarray(PART_TIME).ravel().tolist()
    full_time_codes = np.asarray(FULL_TIME).ravel().tolist()

    if outcome_col == "share_working_part_time":
        return (df["choice"].isin(part_time_codes)).astype(float)
    elif outcome_col == "share_working_full_time":
        return (df["choice"].isin(full_time_codes)).astype(float)
    elif outcome_col == "share_not_working":
        return (~df["choice"].isin(part_time_codes + full_time_codes)).astype(float)
    else:
        raise ValueError(f"Unknown outcome_col: {outcome_col}")


def _prepare_data_for_comparison(
    path_to_specs: Path,
    path_to_baseline_data: Path,
    path_to_no_care_demand_data: Path,
    outcome_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and prepare data for comparison.

    Only extracts the specific outcome needed to reduce memory usage.

    Args:
        path_to_specs: Path to model specifications
        path_to_baseline_data: Path to baseline (estimated_params) simulated
            data
        path_to_no_care_demand_data: Path to no-care-demand data
        outcome_col: Specific outcome column to extract. If None, extracts
            base columns only.

    Returns:
        Tuple of (df_baseline, df_no_care_demand, specs)
    """
    # Load specifications
    specs = pickle.load(path_to_specs.open("rb"))

    # Load simulated data
    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Filter to alive agents
    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure age column exists (if not, compute from period)
    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    # Base required columns
    base_cols = ["agent", "period", "age", "choice"]

    # Extract only the specific outcome needed (if provided)
    if outcome_col is not None:
        # Extract aux variable if needed
        if outcome_col not in df_baseline.columns:
            df_baseline[outcome_col] = _extract_aux_variable(df_baseline, outcome_col)
        if outcome_col not in df_no_care_demand.columns:
            df_no_care_demand[outcome_col] = _extract_aux_variable(
                df_no_care_demand, outcome_col
            )
        base_cols = base_cols + [outcome_col]

    # Verify base columns exist
    for col in base_cols:
        if col not in df_baseline.columns:
            raise ValueError(
                f"Column '{col}' not found in baseline data. "
                f"Available columns: {df_baseline.columns.tolist()}"
            )
        if col not in df_no_care_demand.columns:
            raise ValueError(
                f"Column '{col}' not found in no_care_demand data. "
                f"Available columns: {df_no_care_demand.columns.tolist()}"
            )

    # Keep only relevant columns to reduce memory usage
    df_baseline = df_baseline[base_cols].copy()
    df_no_care_demand = df_no_care_demand[base_cols].copy()

    return df_baseline, df_no_care_demand, specs


def _compute_and_plot_level_and_diff(
    df_baseline: pd.DataFrame,
    df_no_care_demand: pd.DataFrame,
    outcome_col: str,
    ylabel: str,
    path_level: Path,
    path_diff: Path,
    title_suffix: str,
) -> None:
    """Compute and plot levels and differences by age for a given outcome."""
    # Average outcome by age for each scenario
    avg_baseline = (
        df_baseline.groupby("age", observed=False)[outcome_col]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_baseline.columns = ["age", "avg_outcome_baseline"]

    avg_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)[outcome_col]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_no_care_demand.columns = ["age", "avg_outcome_no_care_demand"]

    # Merge and compute differences
    merged = avg_baseline.merge(avg_no_care_demand, on="age", how="inner")
    merged["diff_outcome"] = (
        merged["avg_outcome_baseline"] - merged["avg_outcome_no_care_demand"]
    )

    # Level plot
    plt.figure(figsize=(12, 7))
    plt.plot(
        merged["age"],
        merged["avg_outcome_baseline"],
        label="Baseline",
        color="steelblue",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4,
    )
    plt.plot(
        merged["age"],
        merged["avg_outcome_no_care_demand"],
        label="No Care Demand & Baseline",
        color="darkorange",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=4,
    )
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(
        f"Average {ylabel} by Age: Baseline vs No Care Demand\n"
        f"Subgroup: {title_suffix}",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_level, dpi=300, bbox_inches="tight")
    plt.close()

    # Difference plot
    plt.figure(figsize=(12, 7))
    plt.plot(
        merged["age"],
        merged["diff_outcome"],
        label="Baseline - No Care Demand & Baseline",
        color="purple",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4,
    )
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.7)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(f"Difference in {ylabel}", fontsize=16)
    plt.title(
        f"Difference in {ylabel} by Age\n"
        "Baseline vs No Care Demand\n"
        f"Subgroup: {title_suffix}",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_diff, dpi=300, bbox_inches="tight")
    plt.close()


# @pytask.mark.debug_assets_baseline
def task_compare_outcomes_current_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_assets_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "assets_by_age_level.png",
    path_to_plot_assets_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "assets_by_age_diff.png",
    path_to_plot_savings_dec_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "savings_dec_by_age_level.png",
    path_to_plot_savings_dec_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "savings_dec_by_age_diff.png",
    path_to_plot_savings_rate_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "savings_rate_by_age_level.png",
    path_to_plot_savings_rate_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "savings_rate_by_age_diff.png",
    path_to_plot_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "own_income_after_ssc_by_age_level.png",
    path_to_plot_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "own_income_after_ssc_by_age_diff.png",
    path_to_plot_net_hh_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "net_hh_income_by_age_level.png",
    path_to_plot_net_hh_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "net_hh_income_by_age_diff.png",
    path_to_plot_hh_net_income_wo_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "hh_net_income_wo_interest_by_age_level.png",
    path_to_plot_hh_net_income_wo_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "hh_net_income_wo_interest_by_age_diff.png",
    path_to_plot_consumption_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "consumption_by_age_level.png",
    path_to_plot_consumption_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "consumption_by_age_diff.png",
    path_to_plot_gross_partner_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "gross_partner_income_by_age_level.png",
    path_to_plot_gross_partner_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "gross_partner_income_by_age_diff.png",
    path_to_plot_gross_partner_pension_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "gross_partner_pension_by_age_level.png",
    path_to_plot_gross_partner_pension_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "gross_partner_pension_by_age_diff.png",
    path_to_plot_gross_retirement_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "gross_retirement_income_by_age_level.png",
    path_to_plot_gross_retirement_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "gross_retirement_income_by_age_diff.png",
    path_to_plot_gross_labor_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "gross_labor_income_by_age_level.png",
    path_to_plot_gross_labor_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "gross_labor_income_by_age_diff.png",
    path_to_plot_household_unemployment_benefits_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "household_unemployment_benefits_by_age_level.png",
    path_to_plot_household_unemployment_benefits_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "household_unemployment_benefits_by_age_diff.png",
    path_to_plot_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "interest_by_age_level.png",
    path_to_plot_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "interest_by_age_diff.png",
    path_to_plot_share_working_part_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "share_working_part_time_by_age_level.png",
    path_to_plot_share_working_part_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "share_working_part_time_by_age_diff.png",
    path_to_plot_share_working_full_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "share_working_full_time_by_age_level.png",
    path_to_plot_share_working_full_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "share_working_full_time_by_age_diff.png",
    path_to_plot_share_not_working_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "share_not_working_by_age_level.png",
    path_to_plot_share_not_working_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_caregivers"
    / "share_not_working_by_age_diff.png",
) -> None:
    """Compare all outcomes for currently providing care subgroup."""
    # Ensure output directories exist
    path_to_plot_assets_level.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base columns only (agent, period, age, choice)
    df_baseline_base, df_no_care_demand_base, specs = _prepare_data_for_comparison(
        path_to_specs,
        path_to_baseline_data,
        path_to_no_care_demand_data,
        outcome_col=None,
    )

    # Step 2: Filter baseline to currently providing care
    # (NOT matched - counterfactual unchanged)
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    df_baseline_filtered = df_baseline_base[
        df_baseline_base["choice"].isin(care_codes)
    ].copy()
    df_no_care_demand_filtered = (
        df_no_care_demand_base.copy()
    )  # Counterfactual unchanged

    # Step 3: Delete original dataframes to free memory
    del df_baseline_base, df_no_care_demand_base

    # Plot configurations: (outcome_col, ylabel, path_level, path_diff)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Average Assets Begin of Period",
            path_to_plot_assets_level,
            path_to_plot_assets_diff,
        ),
        (
            "savings_dec",
            "Average Savings Decision",
            path_to_plot_savings_dec_level,
            path_to_plot_savings_dec_diff,
        ),
        (
            "savings_rate",
            "Average Savings Rate",
            path_to_plot_savings_rate_level,
            path_to_plot_savings_rate_diff,
        ),
        (
            "own_income_after_ssc",
            "Average Own Income After SSC",
            path_to_plot_own_income_level,
            path_to_plot_own_income_diff,
        ),
        (
            "net_hh_income",
            "Average Net Household Income",
            path_to_plot_net_hh_income_level,
            path_to_plot_net_hh_income_diff,
        ),
        (
            "hh_net_income_wo_interest",
            "Average HH Net Income Without Interest",
            path_to_plot_hh_net_income_wo_interest_level,
            path_to_plot_hh_net_income_wo_interest_diff,
        ),
        (
            "consumption",
            "Average Consumption",
            path_to_plot_consumption_level,
            path_to_plot_consumption_diff,
        ),
        (
            "gross_partner_income",
            "Average Gross Partner Income",
            path_to_plot_gross_partner_income_level,
            path_to_plot_gross_partner_income_diff,
        ),
        (
            "gross_partner_pension",
            "Average Gross Partner Pension",
            path_to_plot_gross_partner_pension_level,
            path_to_plot_gross_partner_pension_diff,
        ),
        (
            "gross_retirement_income",
            "Average Gross Retirement Income",
            path_to_plot_gross_retirement_income_level,
            path_to_plot_gross_retirement_income_diff,
        ),
        (
            "gross_labor_income",
            "Average Gross Labor Income",
            path_to_plot_gross_labor_income_level,
            path_to_plot_gross_labor_income_diff,
        ),
        (
            "household_unemployment_benefits",
            "Average Household Unemployment Benefits",
            path_to_plot_household_unemployment_benefits_level,
            path_to_plot_household_unemployment_benefits_diff,
        ),
        (
            "interest",
            "Average Interest",
            path_to_plot_interest_level,
            path_to_plot_interest_diff,
        ),
        (
            "share_working_part_time",
            "Share Working Part Time",
            path_to_plot_share_working_part_time_level,
            path_to_plot_share_working_part_time_diff,
        ),
        (
            "share_working_full_time",
            "Share Working Full Time",
            path_to_plot_share_working_full_time_level,
            path_to_plot_share_working_full_time_diff,
        ),
        (
            "share_not_working",
            "Share Not Working",
            path_to_plot_share_not_working_level,
            path_to_plot_share_not_working_diff,
        ),
    ]

    # Step 4: Process outcomes one at a time
    for outcome_col, ylabel, path_level, path_diff in plot_configs:
        # Check if this is a computed outcome from choice variable
        if outcome_col.startswith("share_"):
            # Compute shares directly from choice variable in filtered data
            df_baseline_final = df_baseline_filtered.copy()
            df_baseline_final[outcome_col] = _compute_working_shares_from_choice(
                df_baseline_final, outcome_col
            )

            df_no_care_demand_final = df_no_care_demand_filtered.copy()
            df_no_care_demand_final[outcome_col] = _compute_working_shares_from_choice(
                df_no_care_demand_final, outcome_col
            )
        else:
            # Load only the outcome column we need
            df_baseline_outcome, df_no_care_demand_outcome, _ = (
                _prepare_data_for_comparison(
                    path_to_specs,
                    path_to_baseline_data,
                    path_to_no_care_demand_data,
                    outcome_col=outcome_col,
                )
            )

            # Merge outcome with filtered base
            # (baseline already filtered, counterfactual full)
            df_baseline_final = df_baseline_filtered.merge(
                df_baseline_outcome[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )
            df_no_care_demand_final = df_no_care_demand_filtered.merge(
                df_no_care_demand_outcome[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )

            # Free memory
            del df_baseline_outcome, df_no_care_demand_outcome

        if (
            outcome_col in df_baseline_final.columns
            and outcome_col in df_no_care_demand_final.columns
        ):
            _compute_and_plot_level_and_diff(
                df_baseline_final,
                df_no_care_demand_final,
                outcome_col=outcome_col,
                ylabel=ylabel,
                path_level=path_level,
                path_diff=path_diff,
                title_suffix="Currently Providing Care (Baseline Only - Not Matched)",
            )

        # Free memory
        del df_baseline_final, df_no_care_demand_final


# @pytask.mark.debug_assets_baseline
def task_compare_outcomes_current_non_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_assets_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "assets_by_age_level.png",
    path_to_plot_assets_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "assets_by_age_diff.png",
    path_to_plot_savings_dec_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "savings_dec_by_age_level.png",
    path_to_plot_savings_dec_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "savings_dec_by_age_diff.png",
    path_to_plot_savings_rate_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "savings_rate_by_age_level.png",
    path_to_plot_savings_rate_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "savings_rate_by_age_diff.png",
    path_to_plot_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "own_income_after_ssc_by_age_level.png",
    path_to_plot_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "own_income_after_ssc_by_age_diff.png",
    path_to_plot_net_hh_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "net_hh_income_by_age_level.png",
    path_to_plot_net_hh_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "net_hh_income_by_age_diff.png",
    path_to_plot_hh_net_income_wo_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "hh_net_income_wo_interest_by_age_level.png",
    path_to_plot_hh_net_income_wo_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "hh_net_income_wo_interest_by_age_diff.png",
    path_to_plot_consumption_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "consumption_by_age_level.png",
    path_to_plot_consumption_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "consumption_by_age_diff.png",
    path_to_plot_gross_partner_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "gross_partner_income_by_age_level.png",
    path_to_plot_gross_partner_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "gross_partner_income_by_age_diff.png",
    path_to_plot_gross_partner_pension_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "gross_partner_pension_by_age_level.png",
    path_to_plot_gross_partner_pension_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "gross_partner_pension_by_age_diff.png",
    path_to_plot_gross_retirement_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "gross_retirement_income_by_age_level.png",
    path_to_plot_gross_retirement_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "gross_retirement_income_by_age_diff.png",
    path_to_plot_gross_labor_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "gross_labor_income_by_age_level.png",
    path_to_plot_gross_labor_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "gross_labor_income_by_age_diff.png",
    path_to_plot_household_unemployment_benefits_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "household_unemployment_benefits_by_age_level.png",
    path_to_plot_household_unemployment_benefits_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "household_unemployment_benefits_by_age_diff.png",
    path_to_plot_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "interest_by_age_level.png",
    path_to_plot_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "interest_by_age_diff.png",
    path_to_plot_share_working_part_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "share_working_part_time_by_age_level.png",
    path_to_plot_share_working_part_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "share_working_part_time_by_age_diff.png",
    path_to_plot_share_working_full_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "share_working_full_time_by_age_level.png",
    path_to_plot_share_working_full_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "share_working_full_time_by_age_diff.png",
    path_to_plot_share_not_working_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "share_not_working_by_age_level.png",
    path_to_plot_share_not_working_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "current_non_caregivers"
    / "share_not_working_by_age_diff.png",
) -> None:
    """Compare all outcomes for currently not providing care subgroup."""
    # Ensure output directories exist
    path_to_plot_assets_level.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base columns only (agent, period, age, choice)
    df_baseline_base, df_no_care_demand_base, specs = _prepare_data_for_comparison(
        path_to_specs,
        path_to_baseline_data,
        path_to_no_care_demand_data,
        outcome_col=None,
    )

    # Step 2: Filter baseline to currently NOT providing care
    # (NOT matched - counterfactual unchanged)
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    df_baseline_filtered = df_baseline_base[
        ~df_baseline_base["choice"].isin(care_codes)
    ].copy()
    df_no_care_demand_filtered = (
        df_no_care_demand_base.copy()
    )  # Counterfactual unchanged

    # Step 3: Delete original dataframes to free memory
    del df_baseline_base, df_no_care_demand_base

    # Plot configurations: (outcome_col, ylabel, path_level, path_diff)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Average Assets Begin of Period",
            path_to_plot_assets_level,
            path_to_plot_assets_diff,
        ),
        (
            "savings_dec",
            "Average Savings Decision",
            path_to_plot_savings_dec_level,
            path_to_plot_savings_dec_diff,
        ),
        (
            "savings_rate",
            "Average Savings Rate",
            path_to_plot_savings_rate_level,
            path_to_plot_savings_rate_diff,
        ),
        (
            "own_income_after_ssc",
            "Average Own Income After SSC",
            path_to_plot_own_income_level,
            path_to_plot_own_income_diff,
        ),
        (
            "net_hh_income",
            "Average Net Household Income",
            path_to_plot_net_hh_income_level,
            path_to_plot_net_hh_income_diff,
        ),
        (
            "hh_net_income_wo_interest",
            "Average HH Net Income Without Interest",
            path_to_plot_hh_net_income_wo_interest_level,
            path_to_plot_hh_net_income_wo_interest_diff,
        ),
        (
            "consumption",
            "Average Consumption",
            path_to_plot_consumption_level,
            path_to_plot_consumption_diff,
        ),
        (
            "gross_partner_income",
            "Average Gross Partner Income",
            path_to_plot_gross_partner_income_level,
            path_to_plot_gross_partner_income_diff,
        ),
        (
            "gross_partner_pension",
            "Average Gross Partner Pension",
            path_to_plot_gross_partner_pension_level,
            path_to_plot_gross_partner_pension_diff,
        ),
        (
            "gross_retirement_income",
            "Average Gross Retirement Income",
            path_to_plot_gross_retirement_income_level,
            path_to_plot_gross_retirement_income_diff,
        ),
        (
            "gross_labor_income",
            "Average Gross Labor Income",
            path_to_plot_gross_labor_income_level,
            path_to_plot_gross_labor_income_diff,
        ),
        (
            "household_unemployment_benefits",
            "Average Household Unemployment Benefits",
            path_to_plot_household_unemployment_benefits_level,
            path_to_plot_household_unemployment_benefits_diff,
        ),
        (
            "interest",
            "Average Interest",
            path_to_plot_interest_level,
            path_to_plot_interest_diff,
        ),
        (
            "share_working_part_time",
            "Share Working Part Time",
            path_to_plot_share_working_part_time_level,
            path_to_plot_share_working_part_time_diff,
        ),
        (
            "share_working_full_time",
            "Share Working Full Time",
            path_to_plot_share_working_full_time_level,
            path_to_plot_share_working_full_time_diff,
        ),
        (
            "share_not_working",
            "Share Not Working",
            path_to_plot_share_not_working_level,
            path_to_plot_share_not_working_diff,
        ),
    ]

    # Step 4: Process outcomes one at a time
    for outcome_col, ylabel, path_level, path_diff in plot_configs:
        # Check if this is a computed outcome from choice variable
        if outcome_col.startswith("share_"):
            # Compute shares directly from choice variable in filtered data
            df_baseline_final = df_baseline_filtered.copy()
            df_baseline_final[outcome_col] = _compute_working_shares_from_choice(
                df_baseline_final, outcome_col
            )

            df_no_care_demand_final = df_no_care_demand_filtered.copy()
            df_no_care_demand_final[outcome_col] = _compute_working_shares_from_choice(
                df_no_care_demand_final, outcome_col
            )
        else:
            # Load only the outcome column we need
            df_baseline_outcome, df_no_care_demand_outcome, _ = (
                _prepare_data_for_comparison(
                    path_to_specs,
                    path_to_baseline_data,
                    path_to_no_care_demand_data,
                    outcome_col=outcome_col,
                )
            )

            # Merge outcome with filtered base
            # (baseline already filtered, counterfactual full)
            df_baseline_final = df_baseline_filtered.merge(
                df_baseline_outcome[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )
            df_no_care_demand_final = df_no_care_demand_filtered.merge(
                df_no_care_demand_outcome[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )

            # Free memory
            del df_baseline_outcome, df_no_care_demand_outcome

        if (
            outcome_col in df_baseline_final.columns
            and outcome_col in df_no_care_demand_final.columns
        ):
            _compute_and_plot_level_and_diff(
                df_baseline_final,
                df_no_care_demand_final,
                outcome_col=outcome_col,
                ylabel=ylabel,
                path_level=path_level,
                path_diff=path_diff,
                title_suffix=(
                    "Currently Not Providing Care (Baseline Only - Not Matched)"
                ),
            )

        # Free memory
        del df_baseline_final, df_no_care_demand_final


@pytask.mark.debug_assets_baseline
def task_compare_outcomes_ever_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_assets_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "assets_by_age_level.png",
    path_to_plot_assets_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "assets_by_age_diff.png",
    path_to_plot_savings_dec_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "savings_dec_by_age_level.png",
    path_to_plot_savings_dec_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "savings_dec_by_age_diff.png",
    path_to_plot_savings_rate_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "savings_rate_by_age_level.png",
    path_to_plot_savings_rate_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "savings_rate_by_age_diff.png",
    path_to_plot_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "own_income_after_ssc_by_age_level.png",
    path_to_plot_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "own_income_after_ssc_by_age_diff.png",
    path_to_plot_net_hh_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "net_hh_income_by_age_level.png",
    path_to_plot_net_hh_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "net_hh_income_by_age_diff.png",
    path_to_plot_hh_net_income_wo_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "hh_net_income_wo_interest_by_age_level.png",
    path_to_plot_hh_net_income_wo_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "hh_net_income_wo_interest_by_age_diff.png",
    path_to_plot_consumption_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "consumption_by_age_level.png",
    path_to_plot_consumption_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "consumption_by_age_diff.png",
    path_to_plot_gross_partner_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_partner_income_by_age_level.png",
    path_to_plot_gross_partner_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_partner_income_by_age_diff.png",
    path_to_plot_gross_partner_pension_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_partner_pension_by_age_level.png",
    path_to_plot_gross_partner_pension_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_partner_pension_by_age_diff.png",
    path_to_plot_gross_retirement_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_retirement_income_by_age_level.png",
    path_to_plot_gross_retirement_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_retirement_income_by_age_diff.png",
    path_to_plot_gross_labor_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_labor_income_by_age_level.png",
    path_to_plot_gross_labor_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_labor_income_by_age_diff.png",
    path_to_plot_household_unemployment_benefits_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "household_unemployment_benefits_by_age_level.png",
    path_to_plot_household_unemployment_benefits_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "household_unemployment_benefits_by_age_diff.png",
    path_to_plot_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "interest_by_age_level.png",
    path_to_plot_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "interest_by_age_diff.png",
    path_to_plot_share_working_part_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "share_working_part_time_by_age_level.png",
    path_to_plot_share_working_part_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "share_working_part_time_by_age_diff.png",
    path_to_plot_share_working_full_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "share_working_full_time_by_age_level.png",
    path_to_plot_share_working_full_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "share_working_full_time_by_age_diff.png",
    path_to_plot_share_not_working_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "share_not_working_by_age_level.png",
    path_to_plot_share_not_working_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "share_not_working_by_age_diff.png",
    path_to_plot_total_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "total_income_by_age_level.png",
    path_to_plot_total_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "total_income_by_age_diff.png",
    path_to_plot_gross_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_own_income_by_age_level.png",
    path_to_plot_gross_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "gross_own_income_by_age_diff.png",
    path_to_plot_experience_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "experience_by_age_level.png",
    path_to_plot_experience_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "experience_by_age_diff.png",
    path_to_plot_exp_years_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "exp_years_by_age_level.png",
    path_to_plot_exp_years_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_caregivers"
    / "exp_years_by_age_diff.png",
) -> None:
    """Compare all outcomes for ever caregivers subgroup (matched IDs)."""
    # Step 1: Load base columns only (agent, period, age, choice)
    df_baseline_base, df_no_care_demand_base, specs = _prepare_data_for_comparison(
        path_to_specs,
        path_to_baseline_data,
        path_to_no_care_demand_data,
        outcome_col=None,
    )

    # Step 2: MATCHING - Identify ever caregivers from baseline model
    # This identifies agents who EVER provided care at any point in their life
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiver_ids = (
        df_baseline_base.loc[df_baseline_base["choice"].isin(care_codes), "agent"]
        .unique()
        .tolist()
    )
    caregiver_ids_set = set(caregiver_ids)  # For faster lookup

    # Step 3: MATCHING - Filter BOTH datasets to the SAME agent IDs
    # This ensures we compare the same individuals across both scenarios
    df_baseline_filtered = df_baseline_base[
        df_baseline_base["agent"].isin(caregiver_ids_set)
    ].copy()
    df_no_care_demand_filtered = df_no_care_demand_base[
        df_no_care_demand_base["agent"].isin(caregiver_ids_set)
    ].copy()

    # Step 4: Delete original dataframes to free memory
    del df_baseline_base, df_no_care_demand_base

    # Ensure output directories exist
    path_to_plot_assets_level.parent.mkdir(parents=True, exist_ok=True)

    # Plot configurations: (outcome_col, ylabel, path_level, path_diff)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Average Assets Begin of Period",
            path_to_plot_assets_level,
            path_to_plot_assets_diff,
        ),
        (
            "savings_dec",
            "Average Savings Decision",
            path_to_plot_savings_dec_level,
            path_to_plot_savings_dec_diff,
        ),
        (
            "savings_rate",
            "Average Savings Rate",
            path_to_plot_savings_rate_level,
            path_to_plot_savings_rate_diff,
        ),
        (
            "own_income_after_ssc",
            "Average Own Income After SSC",
            path_to_plot_own_income_level,
            path_to_plot_own_income_diff,
        ),
        (
            "net_hh_income",
            "Average Net Household Income",
            path_to_plot_net_hh_income_level,
            path_to_plot_net_hh_income_diff,
        ),
        (
            "hh_net_income_wo_interest",
            "Average HH Net Income Without Interest",
            path_to_plot_hh_net_income_wo_interest_level,
            path_to_plot_hh_net_income_wo_interest_diff,
        ),
        (
            "consumption",
            "Average Consumption",
            path_to_plot_consumption_level,
            path_to_plot_consumption_diff,
        ),
        (
            "gross_partner_income",
            "Average Gross Partner Income",
            path_to_plot_gross_partner_income_level,
            path_to_plot_gross_partner_income_diff,
        ),
        (
            "gross_partner_pension",
            "Average Gross Partner Pension",
            path_to_plot_gross_partner_pension_level,
            path_to_plot_gross_partner_pension_diff,
        ),
        (
            "gross_retirement_income",
            "Average Gross Retirement Income",
            path_to_plot_gross_retirement_income_level,
            path_to_plot_gross_retirement_income_diff,
        ),
        (
            "gross_labor_income",
            "Average Gross Labor Income",
            path_to_plot_gross_labor_income_level,
            path_to_plot_gross_labor_income_diff,
        ),
        (
            "household_unemployment_benefits",
            "Average Household Unemployment Benefits",
            path_to_plot_household_unemployment_benefits_level,
            path_to_plot_household_unemployment_benefits_diff,
        ),
        (
            "interest",
            "Average Interest",
            path_to_plot_interest_level,
            path_to_plot_interest_diff,
        ),
        (
            "share_working_part_time",
            "Share Working Part Time",
            path_to_plot_share_working_part_time_level,
            path_to_plot_share_working_part_time_diff,
        ),
        (
            "share_working_full_time",
            "Share Working Full Time",
            path_to_plot_share_working_full_time_level,
            path_to_plot_share_working_full_time_diff,
        ),
        (
            "share_not_working",
            "Share Not Working",
            path_to_plot_share_not_working_level,
            path_to_plot_share_not_working_diff,
        ),
        (
            "total_income",
            "Average Total Income",
            path_to_plot_total_income_level,
            path_to_plot_total_income_diff,
        ),
        (
            "gross_own_income",
            "Average Gross Own Income",
            path_to_plot_gross_own_income_level,
            path_to_plot_gross_own_income_diff,
        ),
        (
            "experience",
            "Average Experience",
            path_to_plot_experience_level,
            path_to_plot_experience_diff,
        ),
        (
            "exp_years",
            "Average Experience Years",
            path_to_plot_exp_years_level,
            path_to_plot_exp_years_diff,
        ),
    ]

    # Step 5: Process outcomes one at a time
    for outcome_col, ylabel, path_level, path_diff in plot_configs:
        # Check if this is a computed outcome from choice variable
        if outcome_col.startswith("share_"):
            # Compute shares directly from choice variable in filtered data
            df_baseline_final = df_baseline_filtered.copy()
            df_baseline_final[outcome_col] = _compute_working_shares_from_choice(
                df_baseline_final, outcome_col
            )

            df_no_care_demand_final = df_no_care_demand_filtered.copy()
            df_no_care_demand_final[outcome_col] = _compute_working_shares_from_choice(
                df_no_care_demand_final, outcome_col
            )
        else:
            # Load only the outcome column we need
            df_baseline_outcome, df_no_care_demand_outcome, _ = (
                _prepare_data_for_comparison(
                    path_to_specs,
                    path_to_baseline_data,
                    path_to_no_care_demand_data,
                    outcome_col=outcome_col,
                )
            )

            # Filter outcome data to match our filtered base
            # (already matched to caregiver IDs)
            df_baseline_outcome_filtered = df_baseline_outcome[
                df_baseline_outcome["agent"].isin(caregiver_ids_set)
            ].copy()
            df_no_care_demand_outcome_filtered = df_no_care_demand_outcome[
                df_no_care_demand_outcome["agent"].isin(caregiver_ids_set)
            ].copy()

            # Merge outcome with filtered base
            df_baseline_final = df_baseline_filtered.merge(
                df_baseline_outcome_filtered[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )
            df_no_care_demand_final = df_no_care_demand_filtered.merge(
                df_no_care_demand_outcome_filtered[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )

            # Free memory
            del (
                df_baseline_outcome,
                df_no_care_demand_outcome,
                df_baseline_outcome_filtered,
                df_no_care_demand_outcome_filtered,
            )

        if (
            outcome_col in df_baseline_final.columns
            and outcome_col in df_no_care_demand_final.columns
        ):
            _compute_and_plot_level_and_diff(
                df_baseline_final,
                df_no_care_demand_final,
                outcome_col=outcome_col,
                ylabel=ylabel,
                path_level=path_level,
                path_diff=path_diff,
                title_suffix=(
                    "Ever Caregivers (Matched: Same Agent IDs in Both Scenarios)"
                ),
            )

        # Free memory
        del df_baseline_final, df_no_care_demand_final


# @pytask.mark.debug_assets_baseline
def task_compare_outcomes_ever_non_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_assets_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "assets_by_age_level.png",
    path_to_plot_assets_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "assets_by_age_diff.png",
    path_to_plot_savings_dec_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "savings_dec_by_age_level.png",
    path_to_plot_savings_dec_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "savings_dec_by_age_diff.png",
    path_to_plot_savings_rate_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "savings_rate_by_age_level.png",
    path_to_plot_savings_rate_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "savings_rate_by_age_diff.png",
    path_to_plot_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "own_income_after_ssc_by_age_level.png",
    path_to_plot_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "own_income_after_ssc_by_age_diff.png",
    path_to_plot_net_hh_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "net_hh_income_by_age_level.png",
    path_to_plot_net_hh_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "net_hh_income_by_age_diff.png",
    path_to_plot_hh_net_income_wo_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "hh_net_income_wo_interest_by_age_level.png",
    path_to_plot_hh_net_income_wo_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "hh_net_income_wo_interest_by_age_diff.png",
    path_to_plot_consumption_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "consumption_by_age_level.png",
    path_to_plot_consumption_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "consumption_by_age_diff.png",
    path_to_plot_gross_partner_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "gross_partner_income_by_age_level.png",
    path_to_plot_gross_partner_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "gross_partner_income_by_age_diff.png",
    path_to_plot_gross_partner_pension_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "gross_partner_pension_by_age_level.png",
    path_to_plot_gross_partner_pension_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "gross_partner_pension_by_age_diff.png",
    path_to_plot_gross_retirement_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "gross_retirement_income_by_age_level.png",
    path_to_plot_gross_retirement_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "gross_retirement_income_by_age_diff.png",
    path_to_plot_gross_labor_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "gross_labor_income_by_age_level.png",
    path_to_plot_gross_labor_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "gross_labor_income_by_age_diff.png",
    path_to_plot_household_unemployment_benefits_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "household_unemployment_benefits_by_age_level.png",
    path_to_plot_household_unemployment_benefits_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "household_unemployment_benefits_by_age_diff.png",
    path_to_plot_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "interest_by_age_level.png",
    path_to_plot_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "interest_by_age_diff.png",
    path_to_plot_share_working_part_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "share_working_part_time_by_age_level.png",
    path_to_plot_share_working_part_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "share_working_part_time_by_age_diff.png",
    path_to_plot_share_working_full_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "share_working_full_time_by_age_level.png",
    path_to_plot_share_working_full_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "share_working_full_time_by_age_diff.png",
    path_to_plot_share_not_working_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "share_not_working_by_age_level.png",
    path_to_plot_share_not_working_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging_baseline_vs_no_care_demand"
    / "ever_non_caregivers"
    / "share_not_working_by_age_diff.png",
) -> None:
    """Compare all outcomes for ever non-caregivers subgroup (matched IDs)."""
    # Step 1: Load base columns only (agent, period, age, choice)
    df_baseline_base, df_no_care_demand_base, specs = _prepare_data_for_comparison(
        path_to_specs,
        path_to_baseline_data,
        path_to_no_care_demand_data,
        outcome_col=None,
    )

    # Step 2: MATCHING - Identify ever caregivers from baseline model
    # This identifies agents who EVER provided care at any point in their life
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiver_ids = (
        df_baseline_base.loc[df_baseline_base["choice"].isin(care_codes), "agent"]
        .unique()
        .tolist()
    )
    caregiver_ids_set = set(caregiver_ids)

    # Step 3: MATCHING - Identify non-caregivers (all agents who NEVER provided care)
    all_agents = df_baseline_base["agent"].unique().tolist()
    non_caregiver_ids = [a for a in all_agents if a not in caregiver_ids_set]
    non_caregiver_ids_set = set(non_caregiver_ids)  # For faster lookup

    # Step 4: MATCHING - Filter BOTH datasets to the SAME agent IDs
    # This ensures we compare the same individuals across both scenarios
    df_baseline_filtered = df_baseline_base[
        df_baseline_base["agent"].isin(non_caregiver_ids_set)
    ].copy()
    df_no_care_demand_filtered = df_no_care_demand_base[
        df_no_care_demand_base["agent"].isin(non_caregiver_ids_set)
    ].copy()

    # Step 5: Delete original dataframes to free memory
    del df_baseline_base, df_no_care_demand_base

    # Ensure output directories exist
    path_to_plot_assets_level.parent.mkdir(parents=True, exist_ok=True)

    # Plot configurations: (outcome_col, ylabel, path_level, path_diff)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Average Assets Begin of Period",
            path_to_plot_assets_level,
            path_to_plot_assets_diff,
        ),
        (
            "savings_dec",
            "Average Savings Decision",
            path_to_plot_savings_dec_level,
            path_to_plot_savings_dec_diff,
        ),
        (
            "savings_rate",
            "Average Savings Rate",
            path_to_plot_savings_rate_level,
            path_to_plot_savings_rate_diff,
        ),
        (
            "own_income_after_ssc",
            "Average Own Income After SSC",
            path_to_plot_own_income_level,
            path_to_plot_own_income_diff,
        ),
        (
            "net_hh_income",
            "Average Net Household Income",
            path_to_plot_net_hh_income_level,
            path_to_plot_net_hh_income_diff,
        ),
        (
            "hh_net_income_wo_interest",
            "Average HH Net Income Without Interest",
            path_to_plot_hh_net_income_wo_interest_level,
            path_to_plot_hh_net_income_wo_interest_diff,
        ),
        (
            "consumption",
            "Average Consumption",
            path_to_plot_consumption_level,
            path_to_plot_consumption_diff,
        ),
        (
            "gross_partner_income",
            "Average Gross Partner Income",
            path_to_plot_gross_partner_income_level,
            path_to_plot_gross_partner_income_diff,
        ),
        (
            "gross_partner_pension",
            "Average Gross Partner Pension",
            path_to_plot_gross_partner_pension_level,
            path_to_plot_gross_partner_pension_diff,
        ),
        (
            "gross_retirement_income",
            "Average Gross Retirement Income",
            path_to_plot_gross_retirement_income_level,
            path_to_plot_gross_retirement_income_diff,
        ),
        (
            "gross_labor_income",
            "Average Gross Labor Income",
            path_to_plot_gross_labor_income_level,
            path_to_plot_gross_labor_income_diff,
        ),
        (
            "household_unemployment_benefits",
            "Average Household Unemployment Benefits",
            path_to_plot_household_unemployment_benefits_level,
            path_to_plot_household_unemployment_benefits_diff,
        ),
        (
            "interest",
            "Average Interest",
            path_to_plot_interest_level,
            path_to_plot_interest_diff,
        ),
        (
            "share_working_part_time",
            "Share Working Part Time",
            path_to_plot_share_working_part_time_level,
            path_to_plot_share_working_part_time_diff,
        ),
        (
            "share_working_full_time",
            "Share Working Full Time",
            path_to_plot_share_working_full_time_level,
            path_to_plot_share_working_full_time_diff,
        ),
        (
            "share_not_working",
            "Share Not Working",
            path_to_plot_share_not_working_level,
            path_to_plot_share_not_working_diff,
        ),
    ]

    # Step 6: Process outcomes one at a time
    for outcome_col, ylabel, path_level, path_diff in plot_configs:
        # Check if this is a computed outcome from choice variable
        if outcome_col.startswith("share_"):
            # Compute shares directly from choice variable in filtered data
            df_baseline_final = df_baseline_filtered.copy()
            df_baseline_final[outcome_col] = _compute_working_shares_from_choice(
                df_baseline_final, outcome_col
            )

            df_no_care_demand_final = df_no_care_demand_filtered.copy()
            df_no_care_demand_final[outcome_col] = _compute_working_shares_from_choice(
                df_no_care_demand_final, outcome_col
            )
        else:
            # Load only the outcome column we need
            df_baseline_outcome, df_no_care_demand_outcome, _ = (
                _prepare_data_for_comparison(
                    path_to_specs,
                    path_to_baseline_data,
                    path_to_no_care_demand_data,
                    outcome_col=outcome_col,
                )
            )

            # Filter outcome data to match our filtered base (already matched to non-caregiver IDs)
            df_baseline_outcome_filtered = df_baseline_outcome[
                df_baseline_outcome["agent"].isin(non_caregiver_ids_set)
            ].copy()
            df_no_care_demand_outcome_filtered = df_no_care_demand_outcome[
                df_no_care_demand_outcome["agent"].isin(non_caregiver_ids_set)
            ].copy()

            # Merge outcome with filtered base
            df_baseline_final = df_baseline_filtered.merge(
                df_baseline_outcome_filtered[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )
            df_no_care_demand_final = df_no_care_demand_filtered.merge(
                df_no_care_demand_outcome_filtered[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )

            # Free memory
            del (
                df_baseline_outcome,
                df_no_care_demand_outcome,
                df_baseline_outcome_filtered,
                df_no_care_demand_outcome_filtered,
            )

        if (
            outcome_col in df_baseline_final.columns
            and outcome_col in df_no_care_demand_final.columns
        ):
            _compute_and_plot_level_and_diff(
                df_baseline_final,
                df_no_care_demand_final,
                outcome_col=outcome_col,
                ylabel=ylabel,
                path_level=path_level,
                path_diff=path_diff,
                title_suffix="Ever Non-Caregivers (Matched: Same Agent IDs in Both Scenarios)",
            )

        # Free memory
        del df_baseline_final, df_no_care_demand_final
