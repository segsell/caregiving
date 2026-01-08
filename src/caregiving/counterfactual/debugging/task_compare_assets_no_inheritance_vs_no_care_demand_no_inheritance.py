"""Compare assets by age.

Compares no-inheritance and no-care-demand-no-inheritance models.

This module creates a comparison plot of average assets_begin_of_period by age
for two scenarios:
- No inheritance model
- No care demand and no inheritance model

This helps debug differences in asset accumulation between these two counterfactuals.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import DEAD, FULL_TIME, INFORMAL_CARE, PART_TIME


@pytask.mark.debug_assets
def task_compare_assets_no_inheritance_vs_no_care_demand_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_demand_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "assets_by_age_no_inheritance_vs_no_care_demand_no_inheritance.png",
) -> None:
    """Compare average assets_begin_of_period by age between two scenarios.

    Creates a plot comparing average assets_begin_of_period by age for:
    - No inheritance model (baseline without inheritance)
    - No care demand and no inheritance model

    This helps understand how removing care demand affects asset accumulation
    when inheritance is already removed.

    Args:
        path_to_specs: Path to model specifications
        path_to_no_inheritance_data: Path to no-inheritance simulated data
        path_to_no_care_demand_no_inheritance_data: Path to
            no-care-demand-no-inheritance simulated data
        path_to_plot: Path to save the comparison plot
    """
    import pickle

    # Load specifications
    specs = pickle.load(path_to_specs.open("rb"))

    # Load simulated data
    df_no_inheritance = pd.read_pickle(path_to_no_inheritance_data)
    df_no_care_demand_no_inheritance = pd.read_pickle(
        path_to_no_care_demand_no_inheritance_data
    )

    # Filter to alive agents
    df_no_inheritance = df_no_inheritance[df_no_inheritance["health"] != DEAD].copy()
    df_no_care_demand_no_inheritance = df_no_care_demand_no_inheritance[
        df_no_care_demand_no_inheritance["health"] != DEAD
    ].copy()

    # Ensure age column exists (if not, compute from period)
    if "age" not in df_no_inheritance.columns:
        df_no_inheritance["age"] = df_no_inheritance["period"] + specs["start_age"]
    if "age" not in df_no_care_demand_no_inheritance.columns:
        df_no_care_demand_no_inheritance["age"] = (
            df_no_care_demand_no_inheritance["period"] + specs["start_age"]
        )

    # Verify assets_begin_of_period column exists
    if "assets_begin_of_period" not in df_no_inheritance.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in no_inheritance data. "
            f"Available columns: {df_no_inheritance.columns.tolist()}"
        )
    if "assets_begin_of_period" not in df_no_care_demand_no_inheritance.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in "
            "no_care_demand_no_inheritance data. "
            f"Available columns: {df_no_care_demand_no_inheritance.columns.tolist()}"
        )

    # Calculate average assets by age for each scenario
    avg_assets_no_inheritance = (
        df_no_inheritance.groupby("age", observed=False)["assets_begin_of_period"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_no_inheritance.columns = ["age", "avg_assets"]

    avg_assets_no_care_demand_no_inheritance = (
        df_no_care_demand_no_inheritance.groupby("age", observed=False)[
            "assets_begin_of_period"
        ]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_no_care_demand_no_inheritance.columns = ["age", "avg_assets"]

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Plot no inheritance
    plt.plot(
        avg_assets_no_inheritance["age"],
        avg_assets_no_inheritance["avg_assets"],
        label="No Inheritance",
        color="steelblue",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4,
    )

    # Plot no care demand and no inheritance
    plt.plot(
        avg_assets_no_care_demand_no_inheritance["age"],
        avg_assets_no_care_demand_no_inheritance["avg_assets"],
        label="No Care Demand & No Inheritance",
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
        "Average Assets by Age: No Inheritance vs No Care Demand & No Inheritance",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.debug_assets
def task_compare_asset_differences_no_inheritance_vs_no_care_demand_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_demand_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_plot_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "asset_differences_by_age_no_inheritance_vs_no_care_demand_no_inheritance.png",
) -> None:
    """Plot asset differences by age between two no-inheritance scenarios.

    Creates a plot of the difference in average assets_begin_of_period by age:

        diff(age) = avg_assets_no_inheritance(age)
                    - avg_assets_no_care_demand_no_inheritance(age)

    Positive values indicate higher assets in the no-inheritance model relative
    to the no-care-demand-no-inheritance model at that age.
    """
    import pickle

    # Load specifications
    specs = pickle.load(path_to_specs.open("rb"))

    # Load simulated data
    df_no_inheritance = pd.read_pickle(path_to_no_inheritance_data)
    df_no_care_demand_no_inheritance = pd.read_pickle(
        path_to_no_care_demand_no_inheritance_data
    )

    # Filter to alive agents
    df_no_inheritance = df_no_inheritance[df_no_inheritance["health"] != DEAD].copy()
    df_no_care_demand_no_inheritance = df_no_care_demand_no_inheritance[
        df_no_care_demand_no_inheritance["health"] != DEAD
    ].copy()

    # Ensure age column exists (if not, compute from period)
    if "age" not in df_no_inheritance.columns:
        df_no_inheritance["age"] = df_no_inheritance["period"] + specs["start_age"]
    if "age" not in df_no_care_demand_no_inheritance.columns:
        df_no_care_demand_no_inheritance["age"] = (
            df_no_care_demand_no_inheritance["period"] + specs["start_age"]
        )

    # Verify assets_begin_of_period column exists
    if "assets_begin_of_period" not in df_no_inheritance.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in no_inheritance data. "
            f"Available columns: {df_no_inheritance.columns.tolist()}"
        )
    if "assets_begin_of_period" not in df_no_care_demand_no_inheritance.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in "
            "no_care_demand_no_inheritance data. "
            f"Available columns: {df_no_care_demand_no_inheritance.columns.tolist()}"
        )

    # Calculate average assets by age for each scenario
    avg_assets_no_inheritance = (
        df_no_inheritance.groupby("age", observed=False)["assets_begin_of_period"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_no_inheritance.columns = ["age", "avg_assets_no_inh"]

    avg_assets_no_care_demand_no_inheritance = (
        df_no_care_demand_no_inheritance.groupby("age", observed=False)[
            "assets_begin_of_period"
        ]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_no_care_demand_no_inheritance.columns = [
        "age",
        "avg_assets_no_care_demand_no_inh",
    ]

    # Merge on age to compute differences
    merged = avg_assets_no_inheritance.merge(
        avg_assets_no_care_demand_no_inheritance, on="age", how="inner"
    )
    merged["diff_assets"] = (
        merged["avg_assets_no_inh"] - merged["avg_assets_no_care_demand_no_inh"]
    )

    # Create the difference plot
    plt.figure(figsize=(12, 7))

    plt.plot(
        merged["age"],
        merged["diff_assets"],
        label="No Inheritance - No Care Demand & No Inheritance",
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
        "Difference in Average Assets by Age\n"
        "No Inheritance vs No Care Demand & No Inheritance",
        fontsize=18,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot_diff, dpi=300, bbox_inches="tight")
    plt.close()


def _extract_aux_variable(df_sim: pd.DataFrame, var_name: str) -> pd.Series:
    """Extract a variable from either direct column or aux dictionary."""
    import numpy as np

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
    import numpy as np

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
    path_to_no_inheritance_data: Path,
    path_to_no_care_demand_no_inheritance_data: Path,
    outcome_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and prepare data for comparison.

    Only extracts the specific outcome needed to reduce memory usage.

    Args:
        path_to_specs: Path to model specifications
        path_to_no_inheritance_data: Path to no-inheritance simulated data
        path_to_no_care_demand_no_inheritance_data: Path to
            no-care-demand-no-inheritance data
        outcome_col: Specific outcome column to extract. If None, extracts
            base columns only.

    Returns:
        Tuple of (df_no_inheritance, df_no_care_demand_no_inheritance, specs)
    """
    import pickle

    import numpy as np

    # Load specifications
    specs = pickle.load(path_to_specs.open("rb"))

    # Load simulated data
    df_no_inheritance = pd.read_pickle(path_to_no_inheritance_data)
    df_no_care_demand_no_inheritance = pd.read_pickle(
        path_to_no_care_demand_no_inheritance_data
    )

    # Filter to alive agents
    df_no_inheritance = df_no_inheritance[df_no_inheritance["health"] != DEAD].copy()
    df_no_care_demand_no_inheritance = df_no_care_demand_no_inheritance[
        df_no_care_demand_no_inheritance["health"] != DEAD
    ].copy()

    # Ensure age column exists (if not, compute from period)
    if "age" not in df_no_inheritance.columns:
        df_no_inheritance["age"] = df_no_inheritance["period"] + specs["start_age"]
    if "age" not in df_no_care_demand_no_inheritance.columns:
        df_no_care_demand_no_inheritance["age"] = (
            df_no_care_demand_no_inheritance["period"] + specs["start_age"]
        )

    # Base required columns
    base_cols = ["agent", "period", "age", "choice"]

    # Extract only the specific outcome needed (if provided)
    if outcome_col is not None:
        # Extract aux variable if needed
        if outcome_col not in df_no_inheritance.columns:
            df_no_inheritance[outcome_col] = _extract_aux_variable(
                df_no_inheritance, outcome_col
            )
        if outcome_col not in df_no_care_demand_no_inheritance.columns:
            df_no_care_demand_no_inheritance[outcome_col] = _extract_aux_variable(
                df_no_care_demand_no_inheritance, outcome_col
            )
        base_cols = base_cols + [outcome_col]

    # Verify base columns exist
    for col in base_cols:
        if col not in df_no_inheritance.columns:
            raise ValueError(
                f"Column '{col}' not found in no_inheritance data. "
                f"Available columns: {df_no_inheritance.columns.tolist()}"
            )
        if col not in df_no_care_demand_no_inheritance.columns:
            raise ValueError(
                f"Column '{col}' not found in no_care_demand_no_inheritance data. "
                f"Available columns: "
                f"{df_no_care_demand_no_inheritance.columns.tolist()}"
            )

    # Keep only relevant columns to reduce memory usage
    df_no_inheritance = df_no_inheritance[base_cols].copy()
    df_no_care_demand_no_inheritance = df_no_care_demand_no_inheritance[
        base_cols
    ].copy()

    return df_no_inheritance, df_no_care_demand_no_inheritance, specs


def _compute_and_plot_level_and_diff(
    df_ni: pd.DataFrame,
    df_ncd_ni: pd.DataFrame,
    outcome_col: str,
    ylabel: str,
    path_level: Path,
    path_diff: Path,
    title_suffix: str,
) -> None:
    """Compute and plot levels and differences by age for a given outcome."""
    # Average outcome by age for each scenario
    avg_ni = (
        df_ni.groupby("age", observed=False)[outcome_col]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_ni.columns = ["age", "avg_outcome_no_inh"]

    avg_ncd_ni = (
        df_ncd_ni.groupby("age", observed=False)[outcome_col]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_ncd_ni.columns = ["age", "avg_outcome_no_care_demand_no_inh"]

    # Merge and compute differences
    merged = avg_ni.merge(avg_ncd_ni, on="age", how="inner")
    merged["diff_outcome"] = (
        merged["avg_outcome_no_inh"] - merged["avg_outcome_no_care_demand_no_inh"]
    )

    # Level plot
    plt.figure(figsize=(12, 7))
    plt.plot(
        merged["age"],
        merged["avg_outcome_no_inh"],
        label="No Inheritance",
        color="steelblue",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4,
    )
    plt.plot(
        merged["age"],
        merged["avg_outcome_no_care_demand_no_inh"],
        label="No Care Demand & No Inheritance",
        color="darkorange",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=4,
    )
    plt.xlabel("Age", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(
        f"Average {ylabel} by Age: No Inheritance vs No Care Demand & No Inheritance\n"
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
        label="No Inheritance - No Care Demand & No Inheritance",
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
        "No Inheritance vs No Care Demand & No Inheritance\n"
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


# @pytask.mark.debug_assets
def task_compare_outcomes_current_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_demand_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_plot_assets_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "assets_by_age_level.png",
    path_to_plot_assets_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "assets_by_age_diff.png",
    path_to_plot_savings_dec_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "savings_dec_by_age_level.png",
    path_to_plot_savings_dec_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "savings_dec_by_age_diff.png",
    path_to_plot_savings_rate_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "savings_rate_by_age_level.png",
    path_to_plot_savings_rate_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "savings_rate_by_age_diff.png",
    path_to_plot_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "own_income_after_ssc_by_age_level.png",
    path_to_plot_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "own_income_after_ssc_by_age_diff.png",
    path_to_plot_net_hh_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "net_hh_income_by_age_level.png",
    path_to_plot_net_hh_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "net_hh_income_by_age_diff.png",
    path_to_plot_hh_net_income_wo_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "hh_net_income_wo_interest_by_age_level.png",
    path_to_plot_hh_net_income_wo_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "hh_net_income_wo_interest_by_age_diff.png",
    path_to_plot_consumption_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "consumption_by_age_level.png",
    path_to_plot_consumption_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "consumption_by_age_diff.png",
    path_to_plot_gross_partner_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_partner_income_by_age_level.png",
    path_to_plot_gross_partner_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_partner_income_by_age_diff.png",
    path_to_plot_gross_partner_pension_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_partner_pension_by_age_level.png",
    path_to_plot_gross_partner_pension_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_partner_pension_by_age_diff.png",
    path_to_plot_gross_retirement_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_retirement_income_by_age_level.png",
    path_to_plot_gross_retirement_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_retirement_income_by_age_diff.png",
    path_to_plot_gross_labor_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_labor_income_by_age_level.png",
    path_to_plot_gross_labor_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_labor_income_by_age_diff.png",
    path_to_plot_household_unemployment_benefits_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "household_unemployment_benefits_by_age_level.png",
    path_to_plot_household_unemployment_benefits_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "household_unemployment_benefits_by_age_diff.png",
    path_to_plot_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "interest_by_age_level.png",
    path_to_plot_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "interest_by_age_diff.png",
    path_to_plot_share_working_part_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "share_working_part_time_by_age_level.png",
    path_to_plot_share_working_part_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "share_working_part_time_by_age_diff.png",
    path_to_plot_share_working_full_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "share_working_full_time_by_age_level.png",
    path_to_plot_share_working_full_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "share_working_full_time_by_age_diff.png",
    path_to_plot_share_not_working_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "share_not_working_by_age_level.png",
    path_to_plot_share_not_working_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "share_not_working_by_age_diff.png",
    path_to_plot_total_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "total_income_by_age_level.png",
    path_to_plot_total_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "total_income_by_age_diff.png",
    path_to_plot_gross_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_own_income_by_age_level.png",
    path_to_plot_gross_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "gross_own_income_by_age_diff.png",
    path_to_plot_experience_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "experience_by_age_level.png",
    path_to_plot_experience_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "experience_by_age_diff.png",
    path_to_plot_exp_years_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "exp_years_by_age_level.png",
    path_to_plot_exp_years_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_caregivers"
    / "exp_years_by_age_diff.png",
) -> None:
    """Compare all outcomes for currently providing care subgroup."""
    import numpy as np

    # Ensure output directories exist
    path_to_plot_assets_level.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base columns only (agent, period, age, choice)
    df_ni_base, df_ncd_ni_base, specs = _prepare_data_for_comparison(
        path_to_specs,
        path_to_no_inheritance_data,
        path_to_no_care_demand_no_inheritance_data,
        outcome_col=None,
    )

    # Step 2: Filter baseline to currently providing care
    # (NOT matched - counterfactual unchanged)
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    df_ni_filtered = df_ni_base[df_ni_base["choice"].isin(care_codes)].copy()
    df_ncd_ni_filtered = df_ncd_ni_base.copy()  # Counterfactual unchanged

    # Step 3: Delete original dataframes to free memory
    del df_ni_base, df_ncd_ni_base

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

    # Step 4: Process outcomes one at a time
    for outcome_col, ylabel, path_level, path_diff in plot_configs:
        # Check if this is a computed outcome from choice variable
        if outcome_col.startswith("share_"):
            # Compute shares directly from choice variable in filtered data
            df_ni_final = df_ni_filtered.copy()
            df_ni_final[outcome_col] = _compute_working_shares_from_choice(
                df_ni_final, outcome_col
            )

            df_ncd_ni_final = df_ncd_ni_filtered.copy()
            df_ncd_ni_final[outcome_col] = _compute_working_shares_from_choice(
                df_ncd_ni_final, outcome_col
            )
        else:
            # Load only the outcome column we need
            df_ni_outcome, df_ncd_ni_outcome, _ = _prepare_data_for_comparison(
                path_to_specs,
                path_to_no_inheritance_data,
                path_to_no_care_demand_no_inheritance_data,
                outcome_col=outcome_col,
            )

            # Merge outcome with filtered base (baseline already filtered, counterfactual full)
            df_ni_final = df_ni_filtered.merge(
                df_ni_outcome[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )
            df_ncd_ni_final = df_ncd_ni_filtered.merge(
                df_ncd_ni_outcome[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )

            # Free memory
            del df_ni_outcome, df_ncd_ni_outcome

        if (
            outcome_col in df_ni_final.columns
            and outcome_col in df_ncd_ni_final.columns
        ):
            _compute_and_plot_level_and_diff(
                df_ni_final,
                df_ncd_ni_final,
                outcome_col=outcome_col,
                ylabel=ylabel,
                path_level=path_level,
                path_diff=path_diff,
                title_suffix="Currently Providing Care (Baseline Only - Not Matched)",
            )

        # Free memory
        del df_ni_final, df_ncd_ni_final


# @pytask.mark.debug_assets
def task_compare_outcomes_current_non_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_demand_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_plot_assets_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "assets_by_age_level.png",
    path_to_plot_assets_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "assets_by_age_diff.png",
    path_to_plot_savings_dec_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "savings_dec_by_age_level.png",
    path_to_plot_savings_dec_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "savings_dec_by_age_diff.png",
    path_to_plot_savings_rate_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "savings_rate_by_age_level.png",
    path_to_plot_savings_rate_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "savings_rate_by_age_diff.png",
    path_to_plot_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "own_income_after_ssc_by_age_level.png",
    path_to_plot_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "own_income_after_ssc_by_age_diff.png",
    path_to_plot_net_hh_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "net_hh_income_by_age_level.png",
    path_to_plot_net_hh_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "net_hh_income_by_age_diff.png",
    path_to_plot_hh_net_income_wo_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "hh_net_income_wo_interest_by_age_level.png",
    path_to_plot_hh_net_income_wo_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "hh_net_income_wo_interest_by_age_diff.png",
    path_to_plot_consumption_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "consumption_by_age_level.png",
    path_to_plot_consumption_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "consumption_by_age_diff.png",
    path_to_plot_gross_partner_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_partner_income_by_age_level.png",
    path_to_plot_gross_partner_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_partner_income_by_age_diff.png",
    path_to_plot_gross_partner_pension_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_partner_pension_by_age_level.png",
    path_to_plot_gross_partner_pension_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_partner_pension_by_age_diff.png",
    path_to_plot_gross_retirement_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_retirement_income_by_age_level.png",
    path_to_plot_gross_retirement_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_retirement_income_by_age_diff.png",
    path_to_plot_gross_labor_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_labor_income_by_age_level.png",
    path_to_plot_gross_labor_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_labor_income_by_age_diff.png",
    path_to_plot_household_unemployment_benefits_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "household_unemployment_benefits_by_age_level.png",
    path_to_plot_household_unemployment_benefits_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "household_unemployment_benefits_by_age_diff.png",
    path_to_plot_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "interest_by_age_level.png",
    path_to_plot_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "interest_by_age_diff.png",
    path_to_plot_share_working_part_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "share_working_part_time_by_age_level.png",
    path_to_plot_share_working_part_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "share_working_part_time_by_age_diff.png",
    path_to_plot_share_working_full_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "share_working_full_time_by_age_level.png",
    path_to_plot_share_working_full_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "share_working_full_time_by_age_diff.png",
    path_to_plot_share_not_working_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "share_not_working_by_age_level.png",
    path_to_plot_share_not_working_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "share_not_working_by_age_diff.png",
    path_to_plot_total_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "total_income_by_age_level.png",
    path_to_plot_total_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "total_income_by_age_diff.png",
    path_to_plot_gross_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_own_income_by_age_level.png",
    path_to_plot_gross_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "gross_own_income_by_age_diff.png",
    path_to_plot_experience_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "experience_by_age_level.png",
    path_to_plot_experience_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "experience_by_age_diff.png",
    path_to_plot_exp_years_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "exp_years_by_age_level.png",
    path_to_plot_exp_years_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "current_non_caregivers"
    / "exp_years_by_age_diff.png",
) -> None:
    """Compare all outcomes for currently not providing care subgroup."""
    import numpy as np

    # Ensure output directories exist
    path_to_plot_assets_level.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base columns only (agent, period, age, choice)
    df_ni_base, df_ncd_ni_base, specs = _prepare_data_for_comparison(
        path_to_specs,
        path_to_no_inheritance_data,
        path_to_no_care_demand_no_inheritance_data,
        outcome_col=None,
    )

    # Step 2: Filter baseline to currently NOT providing care (NOT matched - counterfactual unchanged)
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    df_ni_filtered = df_ni_base[~df_ni_base["choice"].isin(care_codes)].copy()
    df_ncd_ni_filtered = df_ncd_ni_base.copy()  # Counterfactual unchanged

    # Step 3: Delete original dataframes to free memory
    del df_ni_base, df_ncd_ni_base

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

    # Step 4: Process outcomes one at a time
    for outcome_col, ylabel, path_level, path_diff in plot_configs:
        # Check if this is a computed outcome from choice variable
        if outcome_col.startswith("share_"):
            # Compute shares directly from choice variable in filtered data
            df_ni_final = df_ni_filtered.copy()
            df_ni_final[outcome_col] = _compute_working_shares_from_choice(
                df_ni_final, outcome_col
            )

            df_ncd_ni_final = df_ncd_ni_filtered.copy()
            df_ncd_ni_final[outcome_col] = _compute_working_shares_from_choice(
                df_ncd_ni_final, outcome_col
            )
        else:
            # Load only the outcome column we need
            df_ni_outcome, df_ncd_ni_outcome, _ = _prepare_data_for_comparison(
                path_to_specs,
                path_to_no_inheritance_data,
                path_to_no_care_demand_no_inheritance_data,
                outcome_col=outcome_col,
            )

            # Merge outcome with filtered base (baseline already filtered, counterfactual full)
            df_ni_final = df_ni_filtered.merge(
                df_ni_outcome[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )
            df_ncd_ni_final = df_ncd_ni_filtered.merge(
                df_ncd_ni_outcome[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )

            # Free memory
            del df_ni_outcome, df_ncd_ni_outcome

        if (
            outcome_col in df_ni_final.columns
            and outcome_col in df_ncd_ni_final.columns
        ):
            _compute_and_plot_level_and_diff(
                df_ni_final,
                df_ncd_ni_final,
                outcome_col=outcome_col,
                ylabel=ylabel,
                path_level=path_level,
                path_diff=path_diff,
                title_suffix="Currently Not Providing Care (Baseline Only - Not Matched)",
            )

        # Free memory
        del df_ni_final, df_ncd_ni_final


@pytask.mark.debug_assets
def task_compare_outcomes_ever_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_demand_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_plot_assets_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "assets_by_age_level.png",
    path_to_plot_assets_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "assets_by_age_diff.png",
    path_to_plot_savings_dec_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "savings_dec_by_age_level.png",
    path_to_plot_savings_dec_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "savings_dec_by_age_diff.png",
    path_to_plot_savings_rate_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "savings_rate_by_age_level.png",
    path_to_plot_savings_rate_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "savings_rate_by_age_diff.png",
    path_to_plot_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "own_income_after_ssc_by_age_level.png",
    path_to_plot_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "own_income_after_ssc_by_age_diff.png",
    path_to_plot_net_hh_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "net_hh_income_by_age_level.png",
    path_to_plot_net_hh_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "net_hh_income_by_age_diff.png",
    path_to_plot_hh_net_income_wo_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "hh_net_income_wo_interest_by_age_level.png",
    path_to_plot_hh_net_income_wo_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "hh_net_income_wo_interest_by_age_diff.png",
    path_to_plot_consumption_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "consumption_by_age_level.png",
    path_to_plot_consumption_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "consumption_by_age_diff.png",
    path_to_plot_gross_partner_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_partner_income_by_age_level.png",
    path_to_plot_gross_partner_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_partner_income_by_age_diff.png",
    path_to_plot_gross_partner_pension_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_partner_pension_by_age_level.png",
    path_to_plot_gross_partner_pension_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_partner_pension_by_age_diff.png",
    path_to_plot_gross_retirement_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_retirement_income_by_age_level.png",
    path_to_plot_gross_retirement_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_retirement_income_by_age_diff.png",
    path_to_plot_gross_labor_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_labor_income_by_age_level.png",
    path_to_plot_gross_labor_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_labor_income_by_age_diff.png",
    path_to_plot_household_unemployment_benefits_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "household_unemployment_benefits_by_age_level.png",
    path_to_plot_household_unemployment_benefits_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "household_unemployment_benefits_by_age_diff.png",
    path_to_plot_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "interest_by_age_level.png",
    path_to_plot_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "interest_by_age_diff.png",
    path_to_plot_share_working_part_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "share_working_part_time_by_age_level.png",
    path_to_plot_share_working_part_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "share_working_part_time_by_age_diff.png",
    path_to_plot_share_working_full_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "share_working_full_time_by_age_level.png",
    path_to_plot_share_working_full_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "share_working_full_time_by_age_diff.png",
    path_to_plot_share_not_working_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "share_not_working_by_age_level.png",
    path_to_plot_share_not_working_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "share_not_working_by_age_diff.png",
    path_to_plot_total_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "total_income_by_age_level.png",
    path_to_plot_total_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "total_income_by_age_diff.png",
    path_to_plot_gross_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_own_income_by_age_level.png",
    path_to_plot_gross_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "gross_own_income_by_age_diff.png",
    path_to_plot_experience_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "experience_by_age_level.png",
    path_to_plot_experience_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "experience_by_age_diff.png",
    path_to_plot_exp_years_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "exp_years_by_age_level.png",
    path_to_plot_exp_years_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_caregivers"
    / "exp_years_by_age_diff.png",
) -> None:
    """Compare all outcomes for ever caregivers subgroup (matched IDs)."""
    import numpy as np

    # Step 1: Load base columns only (agent, period, age, choice)
    df_ni_base, df_ncd_ni_base, specs = _prepare_data_for_comparison(
        path_to_specs,
        path_to_no_inheritance_data,
        path_to_no_care_demand_no_inheritance_data,
        outcome_col=None,
    )

    # Step 2: MATCHING - Identify ever caregivers from baseline (no inheritance) model
    # This identifies agents who EVER provided care at any point in their life
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiver_ids = (
        df_ni_base.loc[df_ni_base["choice"].isin(care_codes), "agent"].unique().tolist()
    )
    caregiver_ids_set = set(caregiver_ids)  # For faster lookup

    # Step 3: MATCHING - Filter BOTH datasets to the SAME agent IDs
    # This ensures we compare the same individuals across both scenarios
    df_ni_filtered = df_ni_base[df_ni_base["agent"].isin(caregiver_ids_set)].copy()
    df_ncd_ni_filtered = df_ncd_ni_base[
        df_ncd_ni_base["agent"].isin(caregiver_ids_set)
    ].copy()

    # Step 4: Delete original dataframes to free memory
    del df_ni_base, df_ncd_ni_base

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

    # Step 4: Process outcomes one at a time
    for outcome_col, ylabel, path_level, path_diff in plot_configs:
        # Check if this is a computed outcome from choice variable
        if outcome_col.startswith("share_"):
            # Compute shares directly from choice variable in filtered data
            df_ni_final = df_ni_filtered.copy()
            df_ni_final[outcome_col] = _compute_working_shares_from_choice(
                df_ni_final, outcome_col
            )

            df_ncd_ni_final = df_ncd_ni_filtered.copy()
            df_ncd_ni_final[outcome_col] = _compute_working_shares_from_choice(
                df_ncd_ni_final, outcome_col
            )
        else:
            # Load only the outcome column we need
            df_ni_outcome, df_ncd_ni_outcome, _ = _prepare_data_for_comparison(
                path_to_specs,
                path_to_no_inheritance_data,
                path_to_no_care_demand_no_inheritance_data,
                outcome_col=outcome_col,
            )

            # Filter outcome data to match our filtered base (already matched to caregiver IDs)
            df_ni_outcome_filtered = df_ni_outcome[
                df_ni_outcome["agent"].isin(caregiver_ids_set)
            ].copy()
            df_ncd_ni_outcome_filtered = df_ncd_ni_outcome[
                df_ncd_ni_outcome["agent"].isin(caregiver_ids_set)
            ].copy()

            # Merge outcome with filtered base
            df_ni_final = df_ni_filtered.merge(
                df_ni_outcome_filtered[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )
            df_ncd_ni_final = df_ncd_ni_filtered.merge(
                df_ncd_ni_outcome_filtered[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )

            # Free memory
            del (
                df_ni_outcome,
                df_ncd_ni_outcome,
                df_ni_outcome_filtered,
                df_ncd_ni_outcome_filtered,
            )

        if (
            outcome_col in df_ni_final.columns
            and outcome_col in df_ncd_ni_final.columns
        ):
            _compute_and_plot_level_and_diff(
                df_ni_final,
                df_ncd_ni_final,
                outcome_col=outcome_col,
                ylabel=ylabel,
                path_level=path_level,
                path_diff=path_diff,
                title_suffix="Ever Caregivers (Matched: Same Agent IDs in Both Scenarios)",
            )

        # Free memory
        del df_ni_final, df_ncd_ni_final


# @pytask.mark.debug_assets
def task_compare_outcomes_ever_non_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_demand_no_inheritance_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_plot_assets_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "assets_by_age_level.png",
    path_to_plot_assets_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "assets_by_age_diff.png",
    path_to_plot_savings_dec_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "savings_dec_by_age_level.png",
    path_to_plot_savings_dec_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "savings_dec_by_age_diff.png",
    path_to_plot_savings_rate_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "savings_rate_by_age_level.png",
    path_to_plot_savings_rate_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "savings_rate_by_age_diff.png",
    path_to_plot_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "own_income_after_ssc_by_age_level.png",
    path_to_plot_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "own_income_after_ssc_by_age_diff.png",
    path_to_plot_net_hh_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "net_hh_income_by_age_level.png",
    path_to_plot_net_hh_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "net_hh_income_by_age_diff.png",
    path_to_plot_hh_net_income_wo_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "hh_net_income_wo_interest_by_age_level.png",
    path_to_plot_hh_net_income_wo_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "hh_net_income_wo_interest_by_age_diff.png",
    path_to_plot_consumption_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "consumption_by_age_level.png",
    path_to_plot_consumption_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "consumption_by_age_diff.png",
    path_to_plot_gross_partner_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_partner_income_by_age_level.png",
    path_to_plot_gross_partner_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_partner_income_by_age_diff.png",
    path_to_plot_gross_partner_pension_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_partner_pension_by_age_level.png",
    path_to_plot_gross_partner_pension_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_partner_pension_by_age_diff.png",
    path_to_plot_gross_retirement_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_retirement_income_by_age_level.png",
    path_to_plot_gross_retirement_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_retirement_income_by_age_diff.png",
    path_to_plot_gross_labor_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_labor_income_by_age_level.png",
    path_to_plot_gross_labor_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_labor_income_by_age_diff.png",
    path_to_plot_household_unemployment_benefits_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "household_unemployment_benefits_by_age_level.png",
    path_to_plot_household_unemployment_benefits_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "household_unemployment_benefits_by_age_diff.png",
    path_to_plot_interest_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "interest_by_age_level.png",
    path_to_plot_interest_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "interest_by_age_diff.png",
    path_to_plot_share_working_part_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "share_working_part_time_by_age_level.png",
    path_to_plot_share_working_part_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "share_working_part_time_by_age_diff.png",
    path_to_plot_share_working_full_time_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "share_working_full_time_by_age_level.png",
    path_to_plot_share_working_full_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "share_working_full_time_by_age_diff.png",
    path_to_plot_share_not_working_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "share_not_working_by_age_level.png",
    path_to_plot_share_not_working_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "share_not_working_by_age_diff.png",
    path_to_plot_total_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "total_income_by_age_level.png",
    path_to_plot_total_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "total_income_by_age_diff.png",
    path_to_plot_gross_own_income_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_own_income_by_age_level.png",
    path_to_plot_gross_own_income_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "gross_own_income_by_age_diff.png",
    path_to_plot_experience_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "experience_by_age_level.png",
    path_to_plot_experience_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "experience_by_age_diff.png",
    path_to_plot_exp_years_level: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "exp_years_by_age_level.png",
    path_to_plot_exp_years_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "debugging"
    / "ever_non_caregivers"
    / "exp_years_by_age_diff.png",
) -> None:
    """Compare all outcomes for ever non-caregivers subgroup (matched IDs)."""
    import numpy as np

    # Step 1: Load base columns only (agent, period, age, choice)
    df_ni_base, df_ncd_ni_base, specs = _prepare_data_for_comparison(
        path_to_specs,
        path_to_no_inheritance_data,
        path_to_no_care_demand_no_inheritance_data,
        outcome_col=None,
    )

    # Step 2: MATCHING - Identify ever caregivers from baseline (no inheritance) model
    # This identifies agents who EVER provided care at any point in their life
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiver_ids = (
        df_ni_base.loc[df_ni_base["choice"].isin(care_codes), "agent"].unique().tolist()
    )
    caregiver_ids_set = set(caregiver_ids)

    # Step 3: MATCHING - Identify non-caregivers (all agents who NEVER provided care)
    all_agents = df_ni_base["agent"].unique().tolist()
    non_caregiver_ids = [a for a in all_agents if a not in caregiver_ids_set]
    non_caregiver_ids_set = set(non_caregiver_ids)  # For faster lookup

    # Step 4: MATCHING - Filter BOTH datasets to the SAME agent IDs
    # This ensures we compare the same individuals across both scenarios
    df_ni_filtered = df_ni_base[df_ni_base["agent"].isin(non_caregiver_ids_set)].copy()
    df_ncd_ni_filtered = df_ncd_ni_base[
        df_ncd_ni_base["agent"].isin(non_caregiver_ids_set)
    ].copy()

    # Step 5: Delete original dataframes to free memory
    del df_ni_base, df_ncd_ni_base

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

    # Step 4: Process outcomes one at a time
    for outcome_col, ylabel, path_level, path_diff in plot_configs:
        # Check if this is a computed outcome from choice variable
        if outcome_col.startswith("share_"):
            # Compute shares directly from choice variable in filtered data
            df_ni_final = df_ni_filtered.copy()
            df_ni_final[outcome_col] = _compute_working_shares_from_choice(
                df_ni_final, outcome_col
            )

            df_ncd_ni_final = df_ncd_ni_filtered.copy()
            df_ncd_ni_final[outcome_col] = _compute_working_shares_from_choice(
                df_ncd_ni_final, outcome_col
            )
        else:
            # Load only the outcome column we need
            df_ni_outcome, df_ncd_ni_outcome, _ = _prepare_data_for_comparison(
                path_to_specs,
                path_to_no_inheritance_data,
                path_to_no_care_demand_no_inheritance_data,
                outcome_col=outcome_col,
            )

            # Filter outcome data to match our filtered base (already matched to non-caregiver IDs)
            df_ni_outcome_filtered = df_ni_outcome[
                df_ni_outcome["agent"].isin(non_caregiver_ids_set)
            ].copy()
            df_ncd_ni_outcome_filtered = df_ncd_ni_outcome[
                df_ncd_ni_outcome["agent"].isin(non_caregiver_ids_set)
            ].copy()

            # Merge outcome with filtered base
            df_ni_final = df_ni_filtered.merge(
                df_ni_outcome_filtered[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )
            df_ncd_ni_final = df_ncd_ni_filtered.merge(
                df_ncd_ni_outcome_filtered[["agent", "period", outcome_col]],
                on=["agent", "period"],
                how="inner",
            )

            # Free memory
            del (
                df_ni_outcome,
                df_ncd_ni_outcome,
                df_ni_outcome_filtered,
                df_ncd_ni_outcome_filtered,
            )

        if (
            outcome_col in df_ni_final.columns
            and outcome_col in df_ncd_ni_final.columns
        ):
            _compute_and_plot_level_and_diff(
                df_ni_final,
                df_ncd_ni_final,
                outcome_col=outcome_col,
                ylabel=ylabel,
                path_level=path_level,
                path_diff=path_diff,
                title_suffix="Ever Non-Caregivers (Matched: Same Agent IDs in Both Scenarios)",
            )

        # Free memory
        del df_ni_final, df_ncd_ni_final
