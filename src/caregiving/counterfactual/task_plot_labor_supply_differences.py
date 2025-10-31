"""Plot differences in labor supply by age between scenarios.

Original and no-care-demand scenario.

"""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from linearmodels.panel import PanelOLS
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    FULL_TIME_CHOICES,
    INFORMAL_CARE,
    PART_TIME,
    PART_TIME_CHOICES,
    RETIREMENT,
    RETIREMENT_CHOICES,
    SEX,
    UNEMPLOYED,
    UNEMPLOYED_CHOICES,
    WORK,
)
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)
from caregiving.model.wealth_and_budget.transfers import (
    calc_care_benefits_and_costs,
    calc_child_benefits,
    calc_unemployment_benefits,
)
from caregiving.model.wealth_and_budget.wages import (
    calculate_gross_labor_income as calculate_gross_labor_income_baseline,
)
from caregiving.model.wealth_and_budget.wages_no_care_demand import (
    calculate_gross_labor_income as calculate_gross_labor_income_ncd,
)

jax.config.update("jax_enable_x64", True)


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

    # Compute differences (original - no_care_demand) to match matched-diffs
    merged["working_diff"] = (
        merged["is_working_original"] - merged["is_working_no_care_demand"]
    )
    merged["part_time_diff"] = (
        merged["is_part_time_original"] - merged["is_part_time_no_care_demand"]
    )
    merged["full_time_diff"] = (
        merged["is_full_time_original"] - merged["is_full_time_no_care_demand"]
    )

    return merged


def create_labor_supply_difference_plot(
    differences: pd.DataFrame, path_to_plot: Path
) -> None:
    """Create plot showing labor supply differences by age."""

    # Create single plot with 3 lines
    plt.figure(figsize=(10, 6))

    # Plot Working (black, dashed)
    plt.plot(
        differences["age"],
        differences["working_diff"],
        color="black",
        linewidth=2,
        linestyle="--",
        label="Working",
    )

    # Plot Full-time (JET_COLOR_MAP[1])
    plt.plot(
        differences["age"],
        differences["full_time_diff"],
        color=JET_COLOR_MAP[1],
        linewidth=2,
        label="Full Time",
    )

    # Plot Part-time (JET_COLOR_MAP[2])
    plt.plot(
        differences["age"],
        differences["part_time_diff"],
        color=JET_COLOR_MAP[0],
        linewidth=2,
        label="Part Time",
    )

    # Add horizontal line at zero (more prominent)
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.85)

    # Set labels and formatting with increased font sizes
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Proportion Working\nDeviation from Counterfactual", fontsize=16)
    plt.xlim(30, 70)
    plt.ylim(None, 0.05)  # Set upper y-limit to 0.05
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend in specified order with increased font size
    plt.legend(["Working", "Full Time", "Part Time"], prop={"size": 16})

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Labor supply difference plot saved to: {path_to_plot}")


def compute_wealth_savings_by_age(df: pd.DataFrame, outcome_var: str) -> pd.DataFrame:
    """Compute mean wealth or savings by age."""
    df_local = df[["age", outcome_var]].copy()

    # Compute mean by age
    means_by_age = (
        df_local.groupby("age", observed=False)[outcome_var].mean().reset_index()
    )

    return means_by_age


def compute_wealth_savings_differences(
    original_data: pd.DataFrame, no_care_demand_data: pd.DataFrame, outcome_var: str
) -> pd.DataFrame:
    """Compute differences in wealth or savings between scenarios."""

    # Compute means by age for both scenarios
    original_means = compute_wealth_savings_by_age(original_data, outcome_var)
    no_care_means = compute_wealth_savings_by_age(no_care_demand_data, outcome_var)

    # Merge on age
    merged = pd.merge(
        original_means,
        no_care_means,
        on=["age"],
        suffixes=("_original", "_no_care_demand"),
    )

    # Compute differences (original - no_care_demand)
    merged[f"{outcome_var}_diff"] = (
        merged[f"{outcome_var}_original"] - merged[f"{outcome_var}_no_care_demand"]
    )

    return merged


def create_wealth_difference_plot(
    differences: pd.DataFrame, path_to_plot: Path
) -> None:
    """Create plot showing wealth differences by age."""

    plt.figure(figsize=(10, 6))

    # Plot wealth difference
    plt.plot(
        differences["age"],
        differences["wealth_at_beginning_diff"],
        color="black",
        linewidth=2,
        label="Wealth Difference",
    )

    # Add horizontal line at zero (more prominent)
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.85)

    # Set labels and formatting with increased font sizes
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Wealth Difference\n(Original - No Care Demand)", fontsize=16)
    plt.xlim(30, 70)
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend with increased font size
    plt.legend(prop={"size": 16})

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Wealth difference plot saved to: {path_to_plot}")


def create_savings_difference_plot(
    differences: pd.DataFrame, path_to_plot: Path
) -> None:
    """Create plot showing savings differences by age."""

    plt.figure(figsize=(10, 6))

    # Plot savings difference
    plt.plot(
        differences["age"],
        differences["savings_dec_diff"],
        color="black",
        linewidth=2,
        label="Savings Difference",
    )

    # Add horizontal line at zero (more prominent)
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.85)

    # Set labels and formatting with increased font sizes
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Savings Difference\n(Original - No Care Demand)", fontsize=16)
    plt.xlim(30, 70)
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend with increased font size
    plt.legend(prop={"size": 16})

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Savings difference plot saved to: {path_to_plot}")


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


# ===================================================================================
# Distance to first care spell profiles
# ===================================================================================


def _ensure_agent_period(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that 'agent' and 'period' are columns (not index levels)."""
    if "agent" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("agent" in df.index.names):
            df = df.reset_index(level=["agent"])  # keep period indexed if present
        else:
            df = df.reset_index()
    if "period" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("period" in df.index.names):
            df = df.reset_index(level=["period"])  # keep any other levels
        else:
            if "age" in df.columns:
                df["period"] = df.groupby("agent")["age"].transform(
                    lambda s: s - s.min()
                )
            else:
                df["period"] = df.groupby("agent").cumcount()

    return df


def _add_distance_to_first_care(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care column to original data where 0 is first care."""
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = _ensure_agent_period(df)
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


def _compute_total_income_like_sim(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total_income per period as in simulate career-costs builder.

    total_income = 0 if dead else max(total_net_income, unemployment_benefits)
    """
    raise RuntimeError(
        "_compute_total_income_like_sim now requires model_options. "
        "Use _compute_total_income_like_sim_with_options."
    )


def _ensure_hours_column(
    df: pd.DataFrame, *, model_options: dict, out_name: str, is_original: bool
) -> pd.DataFrame:
    df = df.copy()
    if out_name in df.columns:
        return df
    if "working_hours" in df.columns and out_name != "working_hours":
        df[out_name] = df["working_hours"]
        return df
    # Compute from options using FT/PT mapping
    mp = (
        model_options["model_params"]
        if "model_params" in model_options
        else model_options
    )
    if is_original:
        part_time_values = np.asarray(PART_TIME).ravel().tolist()
        full_time_values = np.asarray(FULL_TIME).ravel().tolist()
    else:
        part_time_values = np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist()
        full_time_values = np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist()
    df[out_name] = 0.0
    sex_var = SEX
    for edu_var in range(mp["n_education_types"]):
        df.loc[
            df["choice"].isin(full_time_values) & (df["education"] == edu_var),
            out_name,
        ] = mp["av_annual_hours_ft"][sex_var, edu_var]
        df.loc[
            df["choice"].isin(part_time_values) & (df["education"] == edu_var),
            out_name,
        ] = mp["av_annual_hours_pt"][sex_var, edu_var]
    return df


def _compute_total_income_like_sim_with_options(
    df: pd.DataFrame, *, model_options: dict
) -> pd.DataFrame:
    """Compute total_income like in simulate builder.

    Derives missing pieces via options.
    """
    df = df.copy()
    mp = (
        model_options["model_params"]
        if "model_params" in model_options
        else model_options
    )

    # Ensure gross labor income (reuse existing helper)
    # Decide scenario by presence of NC choices in data; default to baseline
    is_original = True
    df = _compute_gross_income_columns(
        df, is_original=is_original, model_options=model_options
    )

    # Prepare arrays
    savings_array = np.asarray(df.get("savings", np.zeros(len(df), dtype=float)))
    education_array = np.asarray(df["education"])
    has_partner_int_array = np.asarray((df.get("partner_state", 0) > 0).astype(int))
    periods_array = (
        np.asarray(df["period"])
        if "period" in df.columns
        else np.asarray(df.index.get_level_values("period"))
    )
    lagged_choice_array = (
        np.asarray(df["lagged_choice"])
        if "lagged_choice" in df.columns
        else np.asarray(df["choice"])
    )
    has_sister_array = np.asarray(df.get("has_sister", np.zeros(len(df), dtype=int)))
    care_demand_array = np.asarray(df.get("care_demand", np.zeros(len(df), dtype=int)))

    # unemployment benefits
    v_unemp = jax.vmap(
        lambda s, edu, hp, p: calc_unemployment_benefits(
            savings=s,
            sex=SEX,
            education=edu,
            has_partner_int=hp,
            period=p,
            options=mp,
        )
    )
    df["unemployment_benefits_calc"] = v_unemp(
        savings_array, education_array, has_partner_int_array, periods_array
    )

    # child benefits
    v_child = jax.vmap(
        lambda edu, hp, p: calc_child_benefits(
            education=edu, sex=SEX, has_partner_int=hp, period=p, options=mp
        )
    )
    df["child_benefits_calc"] = v_child(
        education_array, has_partner_int_array, periods_array
    )

    # care benefits and costs
    v_care = jax.vmap(
        lambda lc, edu, hs, cd: calc_care_benefits_and_costs(
            lagged_choice=lc, education=edu, has_sister=hs, care_demand=cd, options=mp
        )
    )
    df["care_benefits_and_costs_calc"] = v_care(
        lagged_choice_array, education_array, has_sister_array, care_demand_array
    )

    # total net income
    # (no pension term here to mirror builder usage in career costs task)
    df["total_net_income_calc"] = (
        np.asarray(df["gross_labor_income_calc"])
        + np.asarray(df["child_benefits_calc"])
        + np.asarray(df["care_benefits_and_costs_calc"])
    )

    # total income final
    df["total_income_calc"] = np.where(
        df["health"] != DEAD,
        np.maximum(df["total_net_income_calc"], df["unemployment_benefits_calc"]),
        0,
    )

    return df


def _compute_gross_income_columns(
    df: pd.DataFrame, *, is_original: bool, model_options: dict
) -> pd.DataFrame:
    """Compute gross labor income using wage functions for each scenario."""
    df = df.copy()
    mp = (
        model_options["model_params"]
        if "model_params" in model_options
        else model_options
    )
    func = (
        calculate_gross_labor_income_baseline
        if is_original
        else calculate_gross_labor_income_ncd
    )

    # Prepare arrays
    lagged_choice_array = np.asarray(df["lagged_choice"])  # shape (N,)
    experience_years_array = np.asarray(df["exp_years"])  # shape (N,)
    education_array = np.asarray(df["education"])  # shape (N,)
    income_shock_array = np.asarray(df["income_shock"])  # shape (N,)

    # Vectorized jax function
    vectorized_calc_gross = jax.vmap(
        lambda lc, exp, edu, shock: func(
            lagged_choice=lc,
            experience_years=exp,
            education=edu,
            sex=SEX,
            income_shock=shock,
            options=mp,
        )
    )
    gross_array = vectorized_calc_gross(
        lagged_choice_array,
        experience_years_array,
        education_array,
        income_shock_array,
    )

    # Apply working mask consistent with scenario
    work_values = (
        np.asarray(WORK).ravel().tolist()
        if is_original
        else np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist()
    )
    work_mask = df["lagged_choice"].isin(work_values).to_numpy()
    df["gross_labor_income_calc"] = np.asarray(gross_array) * work_mask

    return df


def _compute_hourly_wage(df: pd.DataFrame, gross_col: str, hours_col: str) -> pd.Series:
    hours = df[hours_col]
    return np.where(hours > 0, df[gross_col] / hours, np.nan)


def _matched_diff_profile_by_distance(
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    outcome_col_o: str,
    outcome_col_c: str,
    *,
    ever_caregivers: bool,
    window: int,
) -> pd.DataFrame:
    """Generic matched-diff-by-distance builder for a numeric outcome."""
    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()
    df_c = df_c[df_c["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = _ensure_agent_period(df_o)
    df_c = _ensure_agent_period(df_c)

    # Flatten any residual index
    df_o = df_o.reset_index(drop=True)
    df_c = df_c.reset_index(drop=True)

    # Ever-caregiver restriction
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_o.loc[df_o["choice"].isin(care_codes), "agent"].unique()
        df_o = df_o[df_o["agent"].isin(caregiver_ids)].copy()
        df_c = df_c[df_c["agent"].isin(caregiver_ids)].copy()

    # Keep required columns and avoid name collisions by renaming outcome columns
    o_cols = df_o[["agent", "period", outcome_col_o]].copy()
    o_cols = o_cols.rename(columns={outcome_col_o: "outcome_o"})
    c_cols = df_c[["agent", "period", outcome_col_c]].copy()
    c_cols = c_cols.rename(columns={outcome_col_c: "outcome_c"})

    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_outcome"] = merged["outcome_o"] - merged["outcome_c"]

    # Compute distance in original and attach
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care", observed=False)[["diff_outcome"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )
    return prof


def _plot_single_diff_profile(
    prof: pd.DataFrame, ylabel: str, path_to_plot: Path
) -> None:
    plt.figure(figsize=(12, 7))
    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_outcome"],
        color="black",
        linewidth=2,
    )
    plt.axvline(x=0, color="k", linestyle=":", linewidth=2, alpha=0.85)
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.85)
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


def create_outcome_profiles_by_distance_plot(
    df_original_with_dist: pd.DataFrame,
    df_no_care_demand_with_dist: pd.DataFrame,
    path_to_plot: Path,
    x_window: int = 16,
) -> None:
    """Plot mean work, ft, pt by distance_to_first_care for both scenarios (6 lines)."""

    # Build indicators
    work_o = (
        df_original_with_dist["choice"]
        .isin(np.asarray(WORK).ravel().tolist())
        .astype(float)
    )
    ft_o = (
        df_original_with_dist["choice"]
        .isin(np.asarray(FULL_TIME).ravel().tolist())
        .astype(float)
    )
    pt_o = (
        df_original_with_dist["choice"]
        .isin(np.asarray(PART_TIME).ravel().tolist())
        .astype(float)
    )

    work_c = (
        df_no_care_demand_with_dist["choice"]
        .isin(np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    ft_c = (
        df_no_care_demand_with_dist["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    pt_c = (
        df_no_care_demand_with_dist["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    df_o = pd.DataFrame(
        {
            "distance": df_original_with_dist["distance_to_first_care"],
            "work": work_o,
            "ft": ft_o,
            "pt": pt_o,
        }
    )
    df_c = pd.DataFrame(
        {
            "distance": df_no_care_demand_with_dist["distance_to_first_care"],
            "work": work_c,
            "ft": ft_c,
            "pt": pt_c,
        }
    )

    # Restrict window if requested
    df_o = df_o[(df_o["distance"] >= -x_window) & (df_o["distance"] <= x_window)]
    df_c = df_c[(df_c["distance"] >= -x_window) & (df_c["distance"] <= x_window)]

    prof_o = df_o.groupby("distance", observed=False).mean().reset_index()
    prof_c = df_c.groupby("distance", observed=False).mean().reset_index()

    plt.figure(figsize=(12, 7))
    plt.title("Outcomes by Distance to First Care Spell (Original vs No Care)")

    # six lines
    plt.plot(
        prof_o["distance"],
        prof_o["work"],
        label="Work (Original)",
        color="tab:blue",
        linewidth=2,
    )
    plt.plot(
        prof_c["distance"],
        prof_c["work"],
        label="Work (No Care)",
        color="tab:blue",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        prof_o["distance"],
        prof_o["ft"],
        label="FT (Original)",
        color="tab:red",
        linewidth=2,
    )
    plt.plot(
        prof_c["distance"],
        prof_c["ft"],
        label="FT (No Care)",
        color="tab:red",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        prof_o["distance"],
        prof_o["pt"],
        label="PT (Original)",
        color="tab:green",
        linewidth=2,
    )
    plt.plot(
        prof_c["distance"],
        prof_c["pt"],
        label="PT (No Care)",
        color="tab:green",
        linestyle="--",
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", linewidth=2, alpha=0.85)
    plt.xlabel("Distance to first care spell (periods)")
    plt.ylabel("Mean outcome")
    plt.xlim(-x_window, x_window)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences
def task_plot_outcomes_by_distance_to_first_care(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "outcomes_by_distance_to_first_care.png",
    ever_caregivers: bool = True,
    window: int = 16,
) -> None:
    """Create distance_to_first_care and plot mean outcomes by distance (6 lines)."""

    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Restrict to alive periods
    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure agent/period exist
    df_original = _ensure_agent_period(df_original)
    df_no_care_demand = _ensure_agent_period(df_no_care_demand)

    # Optional restriction to ever-caregivers
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(care_codes), "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    # Compute distance in original and copy to counterfactual
    df_original = _add_distance_to_first_care(df_original)
    # Merge the per-agent first_care_period and compute distance in counterfactual
    dist_map = (
        df_original.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    df_no_care_demand = df_no_care_demand.merge(dist_map, on="agent", how="left")
    df_no_care_demand["distance_to_first_care"] = (
        df_no_care_demand["period"] - df_no_care_demand["first_care_period"]
    )

    # Plot
    create_outcome_profiles_by_distance_plot(
        df_original_with_dist=df_original,
        df_no_care_demand_with_dist=df_no_care_demand,
        path_to_plot=path_to_plot,
        x_window=window,
    )


@pytask.mark.counterfactual_differences
def task_plot_matched_differences_by_distance(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_by_distance.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (orig - no-care), then average by distance.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care from original, attach to merged.
      6) Average diffs by distance and plot three series.

    """

    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()
    df_c = df_c[df_c["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = _ensure_agent_period(df_o)
    df_c = _ensure_agent_period(df_c)

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

    # Outcomes per period
    o_work = df_o["choice"].isin(np.asarray(WORK).ravel().tolist()).astype(float)
    o_ft = df_o["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    o_pt = df_o["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    c_work = (
        df_c["choice"]
        .isin(np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_ft = (
        df_c["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_pt = (
        df_c["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    o_cols = df_o[["agent", "period"]].copy()
    o_cols["work_o"] = o_work
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["work_c"] = c_work
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt

    # Merge on (agent, period) to get matched differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_o"] - merged["work_c"]
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]

    # Compute distance in original and attach
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care", observed=False)[
            ["diff_work", "diff_ft", "diff_pt"]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Plot
    plt.figure(figsize=(12, 7))

    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_work"],
        label="Working",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_ft"],
        label="Full Time",
        color=JET_COLOR_MAP[1],
        linewidth=2,
    )
    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_pt"],
        label="Part Time",
        color=JET_COLOR_MAP[0],
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel("Proportion Working\nDeviation from Counterfactual", fontsize=16)
    plt.xlim(-window, window)
    plt.ylim(-0.125, 0.025)
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend with increased font size, positioned closer to (0,0) point
    plt.legend(
        ["Working", "Full Time", "Part Time"],
        loc="lower left",
        bbox_to_anchor=(0.05, 0.05),
        prop={"size": 16},
    )
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences
def task_plot_wealth_differences(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "wealth_differences_by_age.png",
    ever_caregivers: bool = True,
) -> None:
    """Plot differences in wealth by age between scenarios."""

    # Load data
    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure an 'agent' column exists
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
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

    # Compute wealth differences
    differences = compute_wealth_savings_differences(
        df_original, df_no_care_demand, "wealth_at_beginning"
    )

    # Create plot
    create_wealth_difference_plot(differences, path_to_plot)


@pytask.mark.counterfactual_differences
def task_plot_savings_differences(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "savings_differences_by_age.png",
    ever_caregivers: bool = True,
) -> None:
    """Plot differences in savings by age between scenarios."""

    # Load data
    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure an 'agent' column exists
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
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

    # Compute savings differences
    differences = compute_wealth_savings_differences(
        df_original, df_no_care_demand, "savings_dec"
    )

    # Create plot
    create_savings_difference_plot(differences, path_to_plot)


# @pytask.mark.counterfactual_differences
@pytask.mark.skip()
def task_plot_total_income_matched_differences_by_distance(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_baseline_options: Path = BLD / "model" / "options.pkl",
    path_to_no_care_options: Path = BLD / "model" / "options_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_total_income_by_distance.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Matched diffs (orig - no-care) of total_income by distance to first care."""
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    options_o = pkl.load(path_to_baseline_options.open("rb"))
    options_c = pkl.load(path_to_no_care_options.open("rb"))

    # Ensure gross income exists prior to total income calc
    df_o = _compute_gross_income_columns(
        df_o, is_original=True, model_options=options_o
    )
    df_c = _compute_gross_income_columns(
        df_c, is_original=False, model_options=options_c
    )

    df_o = _compute_total_income_like_sim_with_options(df_o, model_options=options_o)
    df_c = _compute_total_income_like_sim_with_options(df_c, model_options=options_c)

    prof = _matched_diff_profile_by_distance(
        df_o,
        df_c,
        outcome_col_o="total_income_calc",
        outcome_col_c="total_income_calc",
        ever_caregivers=ever_caregivers,
        window=window,
    )

    _plot_single_diff_profile(
        prof, ylabel="Deviation from Counterfactual", path_to_plot=path_to_plot
    )


@pytask.mark.counterfactual_differences_new
def task_plot_gross_income_matched_differences_by_distance(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_baseline_options: Path = BLD / "model" / "options.pkl",
    path_to_no_care_options: Path = BLD / "model" / "options_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_gross_income_by_distance.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Matched diffs (orig - no-care) of gross labor income by distance."""
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    options_o = pkl.load(path_to_baseline_options.open("rb"))
    options_c = pkl.load(path_to_no_care_options.open("rb"))

    df_o = _compute_gross_income_columns(
        df_o, is_original=True, model_options=options_o
    )
    df_c = _compute_gross_income_columns(
        df_c, is_original=False, model_options=options_c
    )
    # No hours needed here; this task plots gross income directly

    prof = _matched_diff_profile_by_distance(
        df_o,
        df_c,
        outcome_col_o="gross_labor_income_calc",
        outcome_col_c="gross_labor_income_calc",
        ever_caregivers=ever_caregivers,
        window=window,
    )

    # Convert from annual to monthly
    prof["diff_outcome"] = prof["diff_outcome"] / 12

    _plot_single_diff_profile(
        prof,
        ylabel="Deviation from Counterfactual",
        path_to_plot=path_to_plot,
    )


@pytask.mark.counterfactual_differences_new
def task_plot_hourly_wage_matched_differences_by_distance(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_baseline_options: Path = BLD / "model" / "options.pkl",
    path_to_no_care_options: Path = BLD / "model" / "options_no_care_demand.pkl",
    hours_column: str = "average_hours",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_hourly_wage_by_distance.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Matched diffs (orig - no-care) of hourly wage by distance.

    Hourly wage = gross labor income / average_hours.
    """
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    options_o = pkl.load(path_to_baseline_options.open("rb"))
    options_c = pkl.load(path_to_no_care_options.open("rb"))

    df_o = _compute_gross_income_columns(
        df_o, is_original=True, model_options=options_o
    )
    df_c = _compute_gross_income_columns(
        df_c, is_original=False, model_options=options_c
    )

    # Ensure hours column exists; fall back to working_hours or compute from options
    df_o = _ensure_hours_column(
        df_o, model_options=options_o, out_name=hours_column, is_original=True
    )
    df_c = _ensure_hours_column(
        df_c, model_options=options_c, out_name=hours_column, is_original=False
    )

    df_o["hourly_wage_calc"] = _compute_hourly_wage(
        df_o, "gross_labor_income_calc", hours_column
    )
    df_c["hourly_wage_calc"] = _compute_hourly_wage(
        df_c, "gross_labor_income_calc", hours_column
    )

    prof = _matched_diff_profile_by_distance(
        df_o,
        df_c,
        outcome_col_o="hourly_wage_calc",
        outcome_col_c="hourly_wage_calc",
        ever_caregivers=ever_caregivers,
        window=window,
    )

    _plot_single_diff_profile(
        prof, ylabel="Deviation from Counterfactual", path_to_plot=path_to_plot
    )


@pytask.mark.counterfactual_differences_new
def task_plot_retired_matched_differences_by_distance(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_retired_by_distance.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Matched diffs (orig - no-care) of retired indicator by distance."""
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    df_o = df_o.copy()
    df_c = df_c.copy()
    df_o["retired_ind"] = (
        df_o["choice"].isin(np.asarray(RETIREMENT).ravel().tolist()).astype(float)
    )
    df_c["retired_ind"] = (
        df_c["choice"]
        .isin(np.asarray(RETIREMENT_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    prof = _matched_diff_profile_by_distance(
        df_o,
        df_c,
        outcome_col_o="retired_ind",
        outcome_col_c="retired_ind",
        ever_caregivers=ever_caregivers,
        window=window,
    )

    _plot_single_diff_profile(
        prof, ylabel="Deviation from Counterfactual", path_to_plot=path_to_plot
    )


@pytask.mark.counterfactual_differences_new
def task_plot_savings_rate_matched_differences_by_distance(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_savings_rate_by_distance.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Matched diffs (orig - no-care) of savings_rate by distance."""
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    prof = _matched_diff_profile_by_distance(
        df_o,
        df_c,
        outcome_col_o="savings_rate",
        outcome_col_c="savings_rate",
        ever_caregivers=ever_caregivers,
        window=window,
    )

    _plot_single_diff_profile(
        prof, ylabel="Deviation from Counterfactual", path_to_plot=path_to_plot
    )


@pytask.mark.counterfactual_differences_new
def task_plot_savings_dec_matched_differences_by_distance(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_savings_dec_by_distance.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Matched diffs (orig - no-care) of savings_dec by distance."""
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    prof = _matched_diff_profile_by_distance(
        df_o,
        df_c,
        outcome_col_o="savings_dec",
        outcome_col_c="savings_dec",
        ever_caregivers=ever_caregivers,
        window=window,
    )

    _plot_single_diff_profile(
        prof, ylabel="Deviation from Counterfactual", path_to_plot=path_to_plot
    )


@pytask.mark.bar_chart
def task_plot_pre_care_to_at_care_transitions(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "pre_care_to_at_care_transitions.png",
) -> None:
    """Baseline-only: show where pre-care labor choices move at first care (t=0).

    For each pre-care category (t=-1): retired, unemployed, part-time, full-time,
    show the distribution of labor choices at t=0 (first caregiving spell).
    Uses stacked bar charts, percentages sum to 1 per pre-care category.
    """
    df = pd.read_pickle(path_to_original_data)

    # Ensure agent/period columns and compute distance to first care
    df = _ensure_agent_period(df)
    df = _add_distance_to_first_care(df)

    # Keep only first-care rows (caregivers at t=0)
    df0 = df[df["distance_to_first_care"] == 0].copy()
    if df0.empty:
        print("No caregivers found for transitions plot.")
        return

    # Determine pre-care choice strictly via lagged_choice at t=0
    if "lagged_choice" not in df0.columns:
        print("lagged_choice not found; cannot compute pre-care transitions.")
        return
    # Drop rows without a valid lagged_choice
    df0 = df0[~df0["lagged_choice"].isna()].copy()
    if df0.empty:
        print("No valid lagged choices at first care to plot transitions.")
        return

    # Map raw choices to aggregate groups (baseline scenario)
    def map_group(series: pd.Series) -> pd.Series:
        retired = np.asarray(RETIREMENT).ravel().tolist()
        unemployed = np.asarray(UNEMPLOYED).ravel().tolist()
        part_time = np.asarray(PART_TIME).ravel().tolist()
        full_time = np.asarray(FULL_TIME).ravel().tolist()
        cat = pd.Series(index=series.index, dtype=object)
        cat.loc[series.isin(retired)] = "Retired"
        cat.loc[series.isin(unemployed)] = "Unemployed"
        cat.loc[series.isin(part_time)] = "Part-time"
        cat.loc[series.isin(full_time)] = "Full-time"
        return cat.fillna("Other")

    df0["prev_group"] = map_group(df0["lagged_choice"])
    df0["curr_group"] = map_group(df0["choice"])

    # Filter to the four intended pre-care groups
    valid_prev = ["Retired", "Unemployed", "Part-time", "Full-time"]
    df0 = df0[df0["prev_group"].isin(valid_prev)].copy()

    # Build normalized distribution of current groups within each prev_group
    curr_order = ["Retired", "Unemployed", "Part-time", "Full-time"]
    table = (
        df0.groupby(["prev_group", "curr_group"])  # counts
        .size()
        .unstack(fill_value=0)
        .reindex(columns=curr_order, fill_value=0)
    )
    # Normalize to percentages per prev_group
    table = table.div(table.sum(axis=1), axis=0).reindex(valid_prev)

    # Plot: four stacked bars
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(valid_prev), dtype=float)
    colors = {
        "Retired": "tab:gray",
        "Unemployed": "tab:blue",
        "Part-time": "tab:green",
        "Full-time": "tab:red",
    }
    for curr in curr_order:
        vals = table[curr].to_numpy()
        plt.bar(
            valid_prev, vals, bottom=bottom, color=colors.get(curr, "gray"), label=curr
        )
        bottom += vals

    plt.ylabel("Share at t=0 (First Care)", fontsize=14)
    plt.xlabel("Pre-care (t=-1) labor category", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(title="t=0 category")
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.line_chart
def task_plot_raw_shares_by_distance_for_future_caregivers(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "raw_shares_by_distance_future_caregivers.png",
    window: int = 16,
) -> None:
    """Baseline-only: raw shares by distance for prospective caregivers.

    Sample: agents who begin informal caregiving at event time t=0 (first care spell).
    Outcome lines: share Retired, Unemployed, Part-time, Full-time by distance.
    """
    df = pd.read_pickle(path_to_original_data)

    # Restrict to alive and ensure agent/period
    df = df[df["health"] != DEAD].copy()
    df = _ensure_agent_period(df)

    # Compute event-time (distance to first care) in baseline
    df = _add_distance_to_first_care(df)

    # Identify agents who start informal caregiving at t=0
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    starts_care = df[
        (df["distance_to_first_care"] == 0) & (df["choice"].isin(care_codes))
    ]["agent"].unique()
    if len(starts_care) == 0:
        print("No prospective caregivers starting at t=0 found.")
        return

    df = df[df["agent"].isin(starts_care)].copy()

    # Map choices to four aggregate groups (baseline definitions)
    retired_codes = np.asarray(RETIREMENT).ravel().tolist()
    unemployed_codes = np.asarray(UNEMPLOYED).ravel().tolist()
    part_time_codes = np.asarray(PART_TIME).ravel().tolist()
    full_time_codes = np.asarray(FULL_TIME).ravel().tolist()

    def to_group(s: pd.Series) -> pd.Series:
        out = pd.Series(index=s.index, dtype=object)
        out.loc[s.isin(retired_codes)] = "Retired"
        out.loc[s.isin(unemployed_codes)] = "Unemployed"
        out.loc[s.isin(part_time_codes)] = "Part-time"
        out.loc[s.isin(full_time_codes)] = "Full-time"
        return out.fillna("Other")

    df["group"] = to_group(df["choice"]).astype(str)

    # Keep distances within window
    df = df[
        (df["distance_to_first_care"] >= -window)
        & (df["distance_to_first_care"] <= window)
    ]

    # Compute normalized shares by distance using a simple value_counts(normalize=True)
    order = ["Retired", "Unemployed", "Part-time", "Full-time"]
    shares = (
        df.groupby("distance_to_first_care", observed=False)["group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
        .reset_index()
        .sort_values("distance_to_first_care")
    )
    # Ensure numeric distance for plotting
    shares["distance_to_first_care"] = pd.to_numeric(
        shares["distance_to_first_care"], errors="coerce"
    )

    # Plot: four lines
    plt.figure(figsize=(12, 7))
    for col, color in zip(
        order, ["tab:gray", "tab:blue", "tab:green", "tab:red"], strict=False
    ):
        plt.plot(
            shares["distance_to_first_care"],
            shares[col],
            label=col,
            linewidth=2,
            color=color,
        )
    plt.axvline(x=0, color="k", linestyle=":", linewidth=2, alpha=0.85)
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.85)
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel("Share in population of prospective caregivers", fontsize=16)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.line_chart_emp
def task_plot_empirical_raw_shares_by_distance_for_future_caregivers(
    path_to_empirical_data: Path = BLD / "data" / "soep_event_study_sample.csv",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "raw_shares_by_distance_future_caregivers_empirical.png",
    window: int = 10,
) -> None:
    """Empirical data: raw shares by distance for prospective caregivers.

    - Event time t=0 is the first period where any_care == 1 for an agent.
    - Sample restricted to agents with a valid t=0 (first-time caregivers).
    - Outcome lines: share Retired, Unemployed, Part-time, Full-time
      by distance in [-10,10].
    """
    df = pd.read_csv(path_to_empirical_data, index_col=[0])

    # Ensure id and period exist; id can be 'agent' or 'pid'
    if ("agent" not in df.columns) and ("pid" not in df.columns):
        df = df.reset_index()
    id_col = "agent" if "agent" in df.columns else "pid"
    if "period" not in df.columns:
        if "age" in df.columns:
            df["period"] = df.groupby(id_col)["age"].transform(lambda s: s - s.min())
        else:
            # Fallback generic sequence order per id
            df = df.sort_values([id_col]).copy()
            df["period"] = df.groupby(id_col).cumcount()

    # Compute first care period per agent using any_care (handle missing/non-numeric)
    any_care_numeric = pd.to_numeric(df["any_care"], errors="coerce").fillna(0)
    care_mask = any_care_numeric.astype(int) == 1
    first_care = (
        df.loc[care_mask, [id_col, "period"]]
        .sort_values([id_col, "period"])
        .drop_duplicates(id_col)
        .rename(columns={"period": "first_care_period"})
    )
    df = df.merge(first_care, on=id_col, how="left")

    # Keep only agents with a valid t=0 (prospective caregivers)
    df = df[~df["first_care_period"].isna()].copy()
    if df.empty:
        print("No first-time caregivers found in empirical data.")
        return

    # Event time
    df["distance_to_first_care"] = df["period"] - df["first_care_period"]

    # Restrict to window
    df = df[
        (df["distance_to_first_care"] >= -window)
        & (df["distance_to_first_care"] <= window)
    ].copy()

    # Map choice to four aggregate groups via *_CHOICES from shared
    def to_group(series: pd.Series) -> pd.Series:
        out = pd.Series(index=series.index, dtype=object)
        out.loc[series.isin(np.asarray(RETIREMENT_CHOICES).ravel().tolist())] = (
            "Retired"
        )
        out.loc[series.isin(np.asarray(UNEMPLOYED_CHOICES).ravel().tolist())] = (
            "Unemployed"
        )
        out.loc[series.isin(np.asarray(PART_TIME_CHOICES).ravel().tolist())] = (
            "Part-time"
        )
        out.loc[series.isin(np.asarray(FULL_TIME_CHOICES).ravel().tolist())] = (
            "Full-time"
        )
        return out.fillna("Other")

    df["group"] = to_group(df["choice"]).astype(str)

    # Shares by distance
    order = ["Retired", "Unemployed", "Part-time", "Full-time"]
    shares = (
        df.groupby("distance_to_first_care", observed=False)["group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
        .reset_index()
        .sort_values("distance_to_first_care")
    )
    shares["distance_to_first_care"] = pd.to_numeric(
        shares["distance_to_first_care"], errors="coerce"
    )

    # Plot
    plt.figure(figsize=(12, 7))
    for col, color in zip(
        order, ["tab:gray", "tab:blue", "tab:green", "tab:red"], strict=False
    ):
        plt.plot(
            shares["distance_to_first_care"],
            shares[col],
            label=col,
            linewidth=2,
            color=color,
        )
    plt.axvline(x=0, color="k", linestyle=":", linewidth=2, alpha=0.85)
    plt.axhline(y=0, color="k", linestyle="--", linewidth=2, alpha=0.85)
    plt.xlabel("Year relative to start of first care spell (empirical)", fontsize=16)
    plt.ylabel("Share among first-time caregivers cohort", fontsize=16)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.bar_chart_emp
def task_plot_empirical_pre_care_to_at_care_transitions(  # noqa: PLR0915
    path_to_empirical_data: Path = BLD / "data" / "soep_event_study_sample.csv",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "emp_pre_care_to_at_care_transitions.png",
) -> None:
    """Empirical baseline-only transitions: lagged_choice (t=-1) to choice (t=0).

    Sample: first-time caregivers at t=0 (any_care turns 1). Stacked bars show the
    composition of t=0 labor choices within each pre-care group at t=-1.
    """
    df = pd.read_csv(path_to_empirical_data, index_col=[0])

    # Ensure id and period exist; detect id column
    if ("agent" not in df.columns) and ("pid" not in df.columns):
        df = df.reset_index()
    id_col = "agent" if "agent" in df.columns else "pid"
    if "period" not in df.columns:
        if "age" in df.columns:
            df["period"] = df.groupby(id_col)["age"].transform(lambda s: s - s.min())
        else:
            df = df.sort_values([id_col]).copy()
            df["period"] = df.groupby(id_col).cumcount()

    # Identify first-time caregivers at t=0
    any_care_numeric = pd.to_numeric(df["any_care"], errors="coerce").fillna(0)
    care_mask = any_care_numeric.astype(int) == 1
    first_care = (
        df.loc[care_mask, [id_col, "period"]]
        .sort_values([id_col, "period"])
        .drop_duplicates(id_col)
        .rename(columns={"period": "first_care_period"})
    )
    df = df.merge(first_care, on=id_col, how="left")

    # Keep only t=0 rows
    df0 = df[df["period"] == df["first_care_period"]].copy()
    if df0.empty:
        print("No first-time caregivers at t=0 found in empirical data.")
        return

    # Require valid lagged_choice at t=0
    if "lagged_choice" not in df0.columns:
        print("lagged_choice not found in empirical data for transitions plot.")
        return
    df0 = df0[~df0["lagged_choice"].isna()].copy()
    if df0.empty:
        print("No valid lagged_choice at t=0 for transitions plot.")
        return

    # Map to labor groups via *_CHOICES
    def map_group(series: pd.Series) -> pd.Series:
        out = pd.Series(index=series.index, dtype=object)
        out.loc[series.isin(np.asarray(RETIREMENT_CHOICES).ravel().tolist())] = (
            "Retired"
        )
        out.loc[series.isin(np.asarray(UNEMPLOYED_CHOICES).ravel().tolist())] = (
            "Unemployed"
        )
        out.loc[series.isin(np.asarray(PART_TIME_CHOICES).ravel().tolist())] = (
            "Part-time"
        )
        out.loc[series.isin(np.asarray(FULL_TIME_CHOICES).ravel().tolist())] = (
            "Full-time"
        )
        return out.fillna("Other")

    df0["prev_group"] = map_group(df0["lagged_choice"]).astype(str)
    df0["curr_group"] = map_group(df0["choice"]).astype(str)

    # Restrict to four main groups
    prev_order = ["Retired", "Unemployed", "Part-time", "Full-time"]
    curr_order = ["Retired", "Unemployed", "Part-time", "Full-time"]
    df0 = df0[df0["prev_group"].isin(prev_order)].copy()

    # Build normalized distribution of curr_group within each prev_group
    tab = df0.groupby(["prev_group", "curr_group"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=curr_order, fill_value=0)
    tab = tab.div(tab.sum(axis=1), axis=0).reindex(prev_order)

    # Plot stacked bars
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(prev_order), dtype=float)
    colors = {
        "Retired": "tab:gray",
        "Unemployed": "tab:blue",
        "Part-time": "tab:green",
        "Full-time": "tab:red",
    }
    for curr in curr_order:
        vals = tab[curr].to_numpy()
        plt.bar(
            prev_order, vals, bottom=bottom, color=colors.get(curr, "gray"), label=curr
        )
        bottom += vals

    plt.ylabel("Share at t=0 (First Care)", fontsize=14)
    plt.xlabel("Pre-care (t=-1) labor category", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(title="t=0 category")
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.bar_chart_emp
def task_plot_empirical_tminus1_to_tplus1_transitions(  # noqa: PLR0915
    path_to_empirical_data: Path = BLD / "data" / "soep_event_study_sample.csv",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "emp_tminus1_to_tplus1_transitions.png",
) -> None:
    """Empirical transitions: t=-1 (lagged_choice at t=0) to t=+1 (choice at t=1).

    Sample: first-time caregivers (any_care turns 1) at event time t=0, comparing
    their pre-care category to the category in the next period.
    """
    df = pd.read_csv(path_to_empirical_data, index_col=[0])

    # Ensure id and period; detect id column
    if ("agent" not in df.columns) and ("pid" not in df.columns):
        df = df.reset_index()
    id_col = "agent" if "agent" in df.columns else "pid"
    if "period" not in df.columns:
        if "age" in df.columns:
            df["period"] = df.groupby(id_col)["age"].transform(lambda s: s - s.min())
        else:
            df = df.sort_values([id_col]).copy()
            df["period"] = df.groupby(id_col).cumcount()

    # Identify first-time caregivers at t=0
    any_care_numeric = pd.to_numeric(df["any_care"], errors="coerce").fillna(0)
    care_mask = any_care_numeric.astype(int) == 1
    first_care = (
        df.loc[care_mask, [id_col, "period"]]
        .sort_values([id_col, "period"])
        .drop_duplicates(id_col)
        .rename(columns={"period": "first_care_period"})
    )
    df = df.merge(first_care, on=id_col, how="left")

    # Rows at t=0 and t=+1
    df0 = df[df["period"] == df["first_care_period"]].copy()
    df1 = df[df["period"] == (df["first_care_period"] + 1)].copy()
    if df0.empty or df1.empty:
        print("Insufficient first-time caregiver rows at t=0 or t=+1.")
        return

    # Need valid lagged_choice at t=0 for pre-care (t=-1)
    if "lagged_choice" not in df0.columns:
        print("lagged_choice not found in empirical data for t-1 to t+1 plot.")
        return
    df0 = df0[~df0["lagged_choice"].isna()].copy()
    if df0.empty:
        print("No valid lagged_choice at t=0 for t-1 to t+1 plot.")
        return

    # Map to labor groups via *_CHOICES
    def map_group(series: pd.Series) -> pd.Series:
        out = pd.Series(index=series.index, dtype=object)
        out.loc[series.isin(np.asarray(RETIREMENT_CHOICES).ravel().tolist())] = (
            "Retired"
        )
        out.loc[series.isin(np.asarray(UNEMPLOYED_CHOICES).ravel().tolist())] = (
            "Unemployed"
        )
        out.loc[series.isin(np.asarray(PART_TIME_CHOICES).ravel().tolist())] = (
            "Part-time"
        )
        out.loc[series.isin(np.asarray(FULL_TIME_CHOICES).ravel().tolist())] = (
            "Full-time"
        )
        return out.fillna("Other")

    df0["prev_group"] = map_group(df0["lagged_choice"]).astype(str)
    df1 = df1[[id_col, "choice"]].copy()
    df1["next_group"] = map_group(df1["choice"]).astype(str)

    # Merge prev and next
    merged = df0[[id_col, "prev_group"]].merge(
        df1[[id_col, "next_group"]], on=id_col, how="inner"
    )
    prev_order = ["Retired", "Unemployed", "Part-time", "Full-time"]
    next_order = ["Retired", "Unemployed", "Part-time", "Full-time"]
    merged = merged[merged["prev_group"].isin(prev_order)].copy()

    # Distribution of next_group within prev_group
    tab = merged.groupby(["prev_group", "next_group"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=next_order, fill_value=0)
    tab = tab.div(tab.sum(axis=1), axis=0).reindex(prev_order)

    # Plot stacked bars
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(prev_order), dtype=float)
    colors = {
        "Retired": "tab:gray",
        "Unemployed": "tab:blue",
        "Part-time": "tab:green",
        "Full-time": "tab:red",
    }
    for nxt in next_order:
        vals = tab[nxt].to_numpy()
        plt.bar(
            prev_order, vals, bottom=bottom, color=colors.get(nxt, "gray"), label=nxt
        )
        bottom += vals

    plt.ylabel("Share at t=+1", fontsize=14)
    plt.xlabel("Pre-care (t=-1) labor category", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(title="t=+1 category")
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_new
def task_plot_total_income_matched_differences_by_distance_from_df(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_total_income_by_distance_from_df.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Matched diffs (orig - no-care) of total_income (from df) by distance."""
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    prof = _matched_diff_profile_by_distance(
        df_o,
        df_c,
        outcome_col_o="total_income",
        outcome_col_c="total_income",
        ever_caregivers=ever_caregivers,
        window=window,
    )

    _plot_single_diff_profile(
        prof, ylabel="Deviation from Counterfactual", path_to_plot=path_to_plot
    )
