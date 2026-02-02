"""Helper functions for plotting counterfactual differences.

This module provides reusable functions to reduce code duplication in plotting tasks.
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def ensure_agent_period(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure agent and period are columns, not index levels."""
    if isinstance(df.index, pd.MultiIndex):
        if "agent" in df.index.names or "period" in df.index.names:
            df = df.reset_index()
    return df


def prepare_single_dataframe(
    df: pd.DataFrame,
    ever_caregivers: bool = False,
) -> pd.DataFrame:
    """Prepare a single DataFrame for analysis by standardizing structure.

    This function:
    1. Filters to alive agents
    2. Ensures agent/period columns exist
    3. Flattens MultiIndex if needed
    4. Optionally filters to ever-caregivers

    Args:
        df: DataFrame to prepare
        ever_caregivers: If True, filter to agents who ever provided care

    Returns:
        Prepared DataFrame
    """
    # Alive restriction
    df = df[df["health"] != DEAD].copy()

    # Ensure agent/period
    df = ensure_agent_period(df)

    # Fully flatten any residual index levels named 'agent' or 'period'
    if isinstance(df.index, pd.MultiIndex):
        idx_names = {n for n in df.index.names if n is not None}
        if ("agent" in idx_names) or ("period" in idx_names):
            df = df.reset_index()

    # Ensure no index name collisions remain (fully flatten)
    df = df.reset_index(drop=True)

    # Ever-caregiver restriction
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df.loc[df["choice"].isin(care_codes), "agent"].unique()
        df = df[df["agent"].isin(caregiver_ids)].copy()

    return df


def prepare_dataframes_for_comparison(
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare DataFrames for comparison by standardizing structure.

    This function:
    1. Filters to alive agents
    2. Ensures agent/period columns exist
    3. Flattens MultiIndex if needed
    4. Optionally filters to ever-caregivers
    5. Optionally filters to agents who ever experience care demand

    Args:
        df_o: Original scenario DataFrame
        df_c: Counterfactual scenario DataFrame
        ever_caregivers: If True, filter to agents who ever provided care
        ever_care_demand: If True, filter to agents who ever experienced care demand

    Returns:
        Tuple of (df_o_prepared, df_c_prepared)
    """
    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()
    df_c = df_c[df_c["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = ensure_agent_period(df_o)
    df_c = ensure_agent_period(df_c)

    # Fully flatten any residual index levels named 'agent' or 'period'
    if isinstance(df_o.index, pd.MultiIndex):
        idx_names_o = {n for n in df_o.index.names if n is not None}
        if ("agent" in idx_names_o) or ("period" in idx_names_o):
            df_o = df_o.reset_index()
    if isinstance(df_c.index, pd.MultiIndex):
        idx_names_c = {n for n in df_c.index.names if n is not None}
        if ("agent" in idx_names_c) or ("period" in idx_names_c):
            df_c = df_c.reset_index()

    # Ensure no index name collisions remain (fully flatten)
    df_o = df_o.reset_index(drop=True)
    df_c = df_c.reset_index(drop=True)

    # Ever-caregiver restriction
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_o.loc[df_o["choice"].isin(care_codes), "agent"].unique()
        df_o = df_o[df_o["agent"].isin(caregiver_ids)].copy()
        df_c = df_c[df_c["agent"].isin(caregiver_ids)].copy()

    # Ever-care-demand restriction
    if ever_care_demand:
        care_demand_ids = df_o.loc[df_o["care_demand"] > 0, "agent"].unique()
        df_o = df_o[df_o["agent"].isin(care_demand_ids)].copy()
        df_c = df_c[df_c["agent"].isin(care_demand_ids)].copy()

    return df_o, df_c


def calculate_outcomes(
    df: pd.DataFrame,
    choice_set_type: Literal[
        "original", "no_care_demand", "job_retention"
    ] = "original",
) -> dict[str, np.ndarray]:
    """Calculate work, ft, pt, job_offer, and care outcomes from choice data.

    Args:
        df: DataFrame with 'choice' and 'job_offer' columns
        choice_set_type: 'original' uses 8-choice structure,
            'no_care_demand' uses 4-choice structure,
            'job_retention' uses 8-choice structure (same as original)

    Returns:
        Dictionary with keys: 'work', 'ft', 'pt', 'job_offer', 'care'
        Values are numpy arrays of floats (0 or 1)
    """
    if choice_set_type == "original":
        work_values = np.asarray(WORK).ravel().tolist()
        ft_values = np.asarray(FULL_TIME).ravel().tolist()
        pt_values = np.asarray(PART_TIME).ravel().tolist()
        care_values = np.asarray(INFORMAL_CARE).ravel().tolist()
    elif choice_set_type == "no_care_demand":
        work_values = np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist()
        ft_values = np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist()
        pt_values = np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist()
        care_values = []  # No care choices in no_care_demand
    elif choice_set_type == "job_retention":
        # Job retention uses same choice structure as original
        work_values = np.asarray(WORK).ravel().tolist()
        ft_values = np.asarray(FULL_TIME).ravel().tolist()
        pt_values = np.asarray(PART_TIME).ravel().tolist()
        care_values = np.asarray(INFORMAL_CARE).ravel().tolist()
    else:
        raise ValueError(
            f"choice_set_type must be 'original', 'no_care_demand', "
            f"or 'job_retention', got {choice_set_type}"
        )

    outcomes = {
        "work": df["choice"].isin(work_values).astype(float).values,
        "ft": df["choice"].isin(ft_values).astype(float).values,
        "pt": df["choice"].isin(pt_values).astype(float).values,
        "job_offer": (df["job_offer"] == 1).astype(float).values,
        "care": df["choice"].isin(care_values).astype(float).values,
    }

    return outcomes


def calculate_additional_outcomes(  # noqa: PLR0912
    df: pd.DataFrame,
    specs: dict,
) -> dict[str, np.ndarray]:
    """Extract additional outcomes for plotting.

    Args:
        df: DataFrame with 'gross_labor_income', 'savings_dec',
            'assets_begin_of_period', 'savings_rate', and optionally
            'consumption'.
        specs: Model specifications dict containing 'wealth_unit' key.

    Returns:
        Dictionary with keys:
            'gross_labor_income', 'savings', 'wealth', 'savings_rate', 'consumption',
            'net_government_budget', 'total_tax_revenue', 'bequest_from_parent',
            'caregiving_leave_top_up', 'gross_retirement_income', 'exp_years'
        Values are numpy arrays (zeros if the underlying column is missing).
    """
    outcomes = {}
    n = len(df)

    # Gross labor income (convert from annual to monthly, then scale by wealth_unit)
    # Note: gross_labor_income is ALWAYS stored as ANNUAL in the simulated data
    # (calculated using av_annual_hours_pt/ft), so we divide by 12 to get monthly.
    # Then multiply by wealth_unit since it's scaled down inside the model.
    if "gross_labor_income" in df.columns:
        outcomes["gross_labor_income"] = (
            df["gross_labor_income"] / 12 * specs["wealth_unit"]
        ).values
    else:
        # If column doesn't exist, create zeros
        outcomes["gross_labor_income"] = np.zeros(n)

    # Savings decision
    if "savings_dec" in df.columns:
        outcomes["savings"] = df["savings_dec"].values
    else:
        outcomes["savings"] = np.zeros(n)

    # Wealth at beginning of period
    if "assets_begin_of_period" in df.columns:
        outcomes["wealth"] = df["assets_begin_of_period"].values
    else:
        outcomes["wealth"] = np.zeros(n)

    # Savings rate
    if "savings_rate" in df.columns:
        outcomes["savings_rate"] = df["savings_rate"].values
    else:
        outcomes["savings_rate"] = np.zeros(n)

    # Consumption (level)
    # If consumption is not available in the simulated data, fall back to zeros so
    # that plots are still generated but flat at 0.
    if "consumption" in df.columns:
        outcomes["consumption"] = df["consumption"].values
    else:
        outcomes["consumption"] = np.zeros(n)

    # Net government budget
    if "net_government_budget" in df.columns:
        outcomes["net_government_budget"] = df["net_government_budget"].values
    else:
        outcomes["net_government_budget"] = np.zeros(n)

    # Total tax revenue
    if "total_tax_revenue" in df.columns:
        outcomes["total_tax_revenue"] = df["total_tax_revenue"].values
    else:
        outcomes["total_tax_revenue"] = np.zeros(n)

    # Bequest from parent (inheritance)
    if "bequest_from_parent" in df.columns:
        outcomes["bequest_from_parent"] = df["bequest_from_parent"].values
    else:
        outcomes["bequest_from_parent"] = np.zeros(n)

    # # Caregiving leave top-up (convert from annual to monthly, then scale by we
    # alth_unit)
    # # Note: caregiving_leave_top_up is stored as annual / wealth_unit in the by
    # dget equation,
    # # so we multiply by wealth_unit to get back to annual scale, then divide by
    #  12 for monthly
    if "caregiving_leave_top_up" in df.columns:
        outcomes["caregiving_leave_top_up"] = (
            df["caregiving_leave_top_up"] / 12 * specs["wealth_unit"]
        ).values
    else:
        outcomes["caregiving_leave_top_up"] = np.zeros(n)

    # Gross retirement income
    if "gross_retirement_income" in df.columns:
        outcomes["gross_retirement_income"] = df["gross_retirement_income"].values
    else:
        outcomes["gross_retirement_income"] = np.zeros(n)

    # Experience years
    if "exp_years" in df.columns:
        outcomes["exp_years"] = df["exp_years"].values
    else:
        outcomes["exp_years"] = np.zeros(n)

    return outcomes


def calculate_working_hours_weekly(
    df: pd.DataFrame,
    model_params: dict,
    choice_set_type: Literal["original", "no_care_demand"] = "original",
) -> np.ndarray:
    """Calculate weekly working hours from choice and education.

    Args:
        df: DataFrame with 'choice' and 'education' columns
        model_params: Model parameters dict with 'av_annual_hours_ft' and
            'av_annual_hours_pt' arrays and 'n_education_types' key
        choice_set_type: 'original' uses 8-choice structure,
            'no_care_demand' uses 4-choice structure

    Returns:
        Array of weekly working hours (annual hours / N_WEEKS_IN_YEAR)
    """
    # Check if working_hours already exists
    if "working_hours" in df.columns:
        hours_annual = df["working_hours"].copy().values
    else:
        # Calculate working hours based on choice and education
        hours_annual = np.zeros(len(df), dtype=float)
        sex_var = 1  # Women only

        if choice_set_type == "original":
            part_time_values = np.asarray(PART_TIME).ravel().tolist()
            full_time_values = np.asarray(FULL_TIME).ravel().tolist()
        elif choice_set_type == "no_care_demand":
            part_time_values = np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist()
            full_time_values = np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist()
        else:
            raise ValueError(
                f"choice_set_type must be 'original' or 'no_care_demand', "
                f"got {choice_set_type}"
            )

        for edu_var in range(model_params["n_education_types"]):
            # Full-time
            ft_mask = df["choice"].isin(full_time_values) & (df["education"] == edu_var)
            hours_annual[ft_mask] = model_params["av_annual_hours_ft"][sex_var, edu_var]

            # Part-time
            pt_mask = df["choice"].isin(part_time_values) & (df["education"] == edu_var)
            hours_annual[pt_mask] = model_params["av_annual_hours_pt"][sex_var, edu_var]

    # Convert annual hours to weekly hours
    hours_weekly = hours_annual / N_WEEKS_IN_YEAR

    return hours_weekly


def create_outcome_columns(
    df: pd.DataFrame,
    outcomes_dict: dict[str, np.ndarray],
    suffix: str,
) -> pd.DataFrame:
    """Create outcome columns DataFrame for merging.

    Args:
        df: Source DataFrame with 'agent' and 'period' columns
        outcomes_dict: Dictionary of outcome arrays (e.g., {'work': array, ...})
        suffix: Suffix for column names ('_o' or '_c')

    Returns:
        DataFrame with agent, period, (optionally age), and outcome columns
        with suffix.
    """
    cols = df[["agent", "period", "age"]].copy()

    for outcome_name, outcome_array in outcomes_dict.items():
        cols[f"{outcome_name}{suffix}"] = outcome_array

    return cols


def merge_and_compute_differences(
    o_cols: pd.DataFrame,
    c_cols: pd.DataFrame,
    outcome_names: list[str],
) -> pd.DataFrame:
    """Merge DataFrames and compute differences.

    Args:
        o_cols: Original scenario columns DataFrame
        c_cols: Counterfactual scenario columns DataFrame
        outcome_names: List of outcome names (e.g., ['work', 'ft', 'pt', ...])

    Returns:
        Merged DataFrame with difference columns (diff_*).
    """
    # Merge on (agent, period) to get matched differences; include age if present.
    merged = o_cols.merge(c_cols, on=["agent", "period", "age"], how="inner")

    # Compute differences (original - counterfactual)
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    return merged


def plot_differences_by_age_with_markers(
    prof_data: pd.DataFrame,
    age_col: str,
    diff_col: str,
    forced_ages: list[int],
    colors: list[str],
    path_to_plot: Path,
    start_age: int,
    end_age: int,
    ylabel: str,
    xlabel: str = "Age",
) -> None:
    """Plot differences by age with markers at forced ages.

    Args:
        prof_data: DataFrame with age and difference columns
        age_col: Name of age column
        diff_col: Name of difference column
        forced_ages: List of ages to mark with circles
        colors: List of colors for each line
        path_to_plot: Path to save the plot
        start_age: Starting age for x-axis
        end_age: Ending age for x-axis
        ylabel: Y-axis label
        xlabel: X-axis label (default: "Age")
    """
    plt.figure(figsize=(12, 7))

    for forced_age, color in zip(forced_ages, colors, strict=False):
        prof_age = prof_data[prof_data[age_col] == forced_age].copy()
        if not prof_age.empty:
            plt.plot(
                prof_age[age_col],
                prof_age[diff_col],
                label=f"Forced at age {forced_age}",
                color=color,
                linewidth=2,
            )
            # Add circle marker at forced age
            plt.scatter(
                [forced_age],
                [prof_age[diff_col].iloc[0]],
                color=color,
                s=100,
                zorder=5,
            )

    plt.axhline(y=0, color="k", linestyle="-", linewidth=2.5, alpha=0.8)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlim(start_age, end_age)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(start_age, end_age + 1), fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="best", prop={"size": 14})
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()
