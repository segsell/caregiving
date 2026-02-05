"""Task to explore government budget components in baseline and counterfactual scenarios."""

import pickle
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    FULL_TIME,
    PART_TIME,
    RETIREMENT,
    UNEMPLOYED,
    WORK,
)
from caregiving.model.wealth_and_budget.tax_and_ssc import calc_health_ltc_contr


@pytask.mark.tables
def task_explore_government_budget(
    path_to_baseline_sim: Path = BLD / "solve_and_simulate"
    # / "simulated_data_no_inheritance.pkl",
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    # / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "government_budget_comparison.csv",
) -> None:
    """Create table comparing government budget components across scenarios.

    For each scenario (baseline and no care demand), compute:
    - Revenue: income_tax, own_ssc, partner_ssc
    - Expenditures: child_benefits, care_benefits (positive only),
      household_unemployment_benefits, net_pension_payout

    Net pension payout = gross pension income (for retirees) - pension contributions
    (pension contributions are already in SSC revenue, so net pension is the
    net expenditure on pensions).

    Parameters
    ----------
    path_to_baseline_sim : Path
        Path to baseline simulated data
    path_to_no_care_demand_sim : Path
        Path to no care demand counterfactual simulated data
    path_to_specs : Path
        Path to model specs (needed for wealth_unit conversion)
    path_to_save_table : Path
        Path to save the output CSV table
    """

    # Load initial states
    initial_states = pickle.load(path_to_initial_states.open("rb"))

    # Load simulated data
    baseline_df = pd.read_pickle(path_to_baseline_sim)
    no_care_demand_df = pd.read_pickle(path_to_no_care_demand_sim)

    # Load specs for wealth_unit conversion
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = specs["wealth_unit"]

    baseline_df = convert_to_currency(baseline_df, wealth_unit)
    no_care_demand_df = convert_to_currency(no_care_demand_df, wealth_unit)
    breakpoint()

    # Compare income_tax_single for working choices only
    # Use numpy isin for much faster filtering
    work_choices = np.asarray(WORK)
    baseline_working_mask = np.isin(baseline_df["choice"].values, work_choices)
    no_care_demand_working_mask = np.isin(
        no_care_demand_df["choice"].values, work_choices
    )
    baseline_working = baseline_df[baseline_working_mask].copy()
    no_care_demand_working = no_care_demand_df[no_care_demand_working_mask].copy()

    # Check if income_tax_single is in wealth_unit or actual currency
    # It should be in wealth_unit based on budget_equation.py line 216
    baseline_income_tax_single_currency = (
        baseline_working["income_tax_single"] * wealth_unit
    )
    no_care_demand_income_tax_single_currency = (
        no_care_demand_working["income_tax_single"] * wealth_unit
    )
    breakpoint()

    baseline_income_tax_single_sum = baseline_income_tax_single_currency.sum() / 100_000
    no_care_demand_income_tax_single_sum = (
        no_care_demand_income_tax_single_currency.sum() / 100_000
    )

    # Diagnostic: Check for negative tax values (shouldn't happen but let's verify)
    baseline_negative_tax = (baseline_income_tax_single_currency < 0).sum()
    no_care_demand_negative_tax = (no_care_demand_income_tax_single_currency < 0).sum()

    # Check total tax from workers above threshold
    baseline_tax_above_threshold = (
        baseline_income_tax_single_currency[baseline_above_threshold_mask].sum()
        / 100_000
    )
    no_care_demand_tax_above_threshold = (
        no_care_demand_income_tax_single_currency[
            no_care_demand_above_threshold_mask
        ].sum()
        / 100_000
    )

    # Diagnostic: Compare average incomes and other factors
    baseline_avg_income = baseline_working["gross_labor_income"].mean()
    no_care_demand_avg_income = no_care_demand_working["gross_labor_income"].mean()

    baseline_avg_income_after_ssc = baseline_working["own_income_after_ssc"].mean()
    no_care_demand_avg_income_after_ssc = no_care_demand_working[
        "own_income_after_ssc"
    ].mean()

    baseline_avg_tax = baseline_income_tax_single_currency.mean()
    no_care_demand_avg_tax = no_care_demand_income_tax_single_currency.mean()

    # Check experience
    baseline_avg_experience = baseline_working["experience"].mean()
    no_care_demand_avg_experience = no_care_demand_working["experience"].mean()

    # Check income distribution and tax threshold
    # Tax threshold is typically around 8004 (thresholds[1])
    # Income values are in wealth_unit, so need to convert to actual currency
    tax_threshold = specs.get("income_tax_brackets", [0.0, 8004, 13469, 52881, 250730])[
        1
    ]

    # Convert income to actual currency for comparison
    baseline_income_currency = baseline_working["own_income_after_ssc"] * wealth_unit
    no_care_demand_income_currency = (
        no_care_demand_working["own_income_after_ssc"] * wealth_unit
    )

    baseline_below_threshold = (baseline_income_currency <= tax_threshold).sum()
    no_care_demand_below_threshold = (
        no_care_demand_income_currency <= tax_threshold
    ).sum()

    baseline_pct_below_threshold = (
        baseline_below_threshold / baseline_working_mask.sum() * 100
    )
    no_care_demand_pct_below_threshold = (
        no_care_demand_below_threshold / no_care_demand_working_mask.sum() * 100
    )

    # Check workers above threshold (the ones actually paying tax)
    baseline_above_threshold_mask = baseline_income_currency > tax_threshold
    no_care_demand_above_threshold_mask = no_care_demand_income_currency > tax_threshold

    baseline_above_threshold = baseline_working[baseline_above_threshold_mask]
    no_care_demand_above_threshold = no_care_demand_working[
        no_care_demand_above_threshold_mask
    ]

    baseline_avg_income_above = baseline_income_currency[
        baseline_above_threshold_mask
    ].mean()
    no_care_demand_avg_income_above = no_care_demand_income_currency[
        no_care_demand_above_threshold_mask
    ].mean()
    baseline_avg_tax_above = baseline_income_tax_single_currency[
        baseline_above_threshold_mask
    ].mean()
    no_care_demand_avg_tax_above = no_care_demand_income_tax_single_currency[
        no_care_demand_above_threshold_mask
    ].mean()

    breakpoint()

    # Check part-time vs full-time composition
    baseline_pt = (baseline_working["choice"].isin(PART_TIME)).sum()
    baseline_ft = (baseline_working["choice"].isin(FULL_TIME)).sum()
    no_care_demand_pt = (no_care_demand_working["choice"].isin(PART_TIME)).sum()
    no_care_demand_ft = (no_care_demand_working["choice"].isin(FULL_TIME)).sum()

    print("\n=== Income Tax Single Comparison (Working Choices Only) ===")
    print(
        f"Baseline income_tax_single sum (in 100k): {baseline_income_tax_single_sum:.2f}"
    )
    print(
        f"No care demand income_tax_single sum (in 100k): {no_care_demand_income_tax_single_sum:.2f}"
    )
    print(
        f"Difference (in 100k): {baseline_income_tax_single_sum - no_care_demand_income_tax_single_sum:.2f}"
    )
    print(f"\nBaseline workers: {baseline_working_mask.sum()}")
    print(f"No care demand workers: {no_care_demand_working_mask.sum()}")
    print(f"\nBaseline avg gross labor income: {baseline_avg_income:.2f}")
    print(f"No care demand avg gross labor income: {no_care_demand_avg_income:.2f}")
    print(f"\nBaseline avg income after SSC: {baseline_avg_income_after_ssc:.2f}")
    print(
        f"No care demand avg income after SSC: {no_care_demand_avg_income_after_ssc:.2f}"
    )
    print(f"\nBaseline avg income_tax_single (currency): {baseline_avg_tax:.2f}")
    print(
        f"No care demand avg income_tax_single (currency): {no_care_demand_avg_tax:.2f}"
    )
    print(f"\nBaseline negative tax count: {baseline_negative_tax}")
    print(f"No care demand negative tax count: {no_care_demand_negative_tax}")
    print(
        f"\nBaseline tax from workers above threshold (100k): {baseline_tax_above_threshold:.2f}"
    )
    print(
        f"No care demand tax from workers above threshold (100k): {no_care_demand_tax_above_threshold:.2f}"
    )
    print(f"\nBaseline avg experience: {baseline_avg_experience:.2f}")
    print(f"No care demand avg experience: {no_care_demand_avg_experience:.2f}")
    print(
        f"\nBaseline workers below tax threshold ({tax_threshold}): {baseline_below_threshold} ({baseline_pct_below_threshold:.1f}%)"
    )
    print(
        f"No care demand workers below tax threshold ({tax_threshold}): {no_care_demand_below_threshold} ({no_care_demand_pct_below_threshold:.1f}%)"
    )
    print(
        f"\nBaseline workers above threshold: {len(baseline_above_threshold)} (avg income: {baseline_avg_income_above:.2f}, avg tax: {baseline_avg_tax_above:.2f})"
    )
    print(
        f"No care demand workers above threshold: {len(no_care_demand_above_threshold)} (avg income: {no_care_demand_avg_income_above:.2f}, avg tax: {no_care_demand_avg_tax_above:.2f})"
    )
    print(f"\nBaseline: PT={baseline_pt}, FT={baseline_ft}")
    print(f"No care demand: PT={no_care_demand_pt}, FT={no_care_demand_ft}")

    breakpoint()

    # Agent-year comparison
    # Merge on agent and period/age to compare
    merge_cols = ["agent", "period"]
    if "age" in baseline_df.columns and "age" in no_care_demand_df.columns:
        merge_cols.append("age")

    comparison_cols = [
        "income_tax",
        "own_ssc",
        "partner_ssc",
        "child_benefits",
        "care_benefits_and_costs",
        "household_unemployment_benefits",
        "gross_retirement_income",
        "choice",
    ]

    # Select only columns that exist in BOTH dataframes
    available_cols = [
        col
        for col in comparison_cols
        if col in baseline_df.columns and col in no_care_demand_df.columns
    ]

    baseline_subset = baseline_df[merge_cols + available_cols].copy()
    no_care_demand_subset = no_care_demand_df[merge_cols + available_cols].copy()

    # Merge for comparison
    comparison = baseline_subset.merge(
        no_care_demand_subset,
        on=merge_cols,
        suffixes=("_baseline", "_no_care_demand"),
        how="outer",
        indicator=True,
    )

    # Calculate differences for numeric columns
    for col in available_cols:
        if col in ["choice"]:  # Skip non-numeric comparison columns
            continue
        baseline_col = f"{col}_baseline"
        no_care_col = f"{col}_no_care_demand"
        if baseline_col in comparison.columns and no_care_col in comparison.columns:
            comparison[f"{col}_diff"] = (
                comparison[baseline_col] - comparison[no_care_col]
            )

    print("\n=== Agent-Year Comparison ===")
    print(f"Total rows baseline: {len(baseline_df)}")
    print(f"Total rows no_care_demand: {len(no_care_demand_df)}")
    print(f"Matched rows: {len(comparison[comparison['_merge'] == 'both'])}")
    print(f"Only in baseline: {len(comparison[comparison['_merge'] == 'left_only'])}")
    print(
        f"Only in no_care_demand: {len(comparison[comparison['_merge'] == 'right_only'])}"
    )

    if len(comparison[comparison["_merge"] == "both"]) > 0:
        matched = comparison[comparison["_merge"] == "both"].copy()
        print("\n=== Summary Statistics for Matched Agent-Year Observations ===")
        diff_cols = [col for col in comparison.columns if col.endswith("_diff")]
        if diff_cols:
            print(matched[diff_cols].describe())

    breakpoint()

    # Calculate net pension payout for each scenario
    baseline_df["net_pension_payout"] = calculate_net_pension_payout(baseline_df)
    no_care_demand_df["net_pension_payout"] = calculate_net_pension_payout(
        no_care_demand_df
    )

    # Aggregate for each scenario
    baseline_fiscal = aggregate_fiscal_components(baseline_df, "baseline")
    no_care_demand_fiscal = aggregate_fiscal_components(
        no_care_demand_df, "no_care_demand"
    )

    # Combine into single table
    combined_table = pd.concat([baseline_fiscal, no_care_demand_fiscal])

    # Reset index to have age as a column
    combined_table = combined_table.reset_index()

    # Reorder columns
    column_order = [
        "scenario",
        "age",
        "revenue_income_tax",
        "revenue_own_ssc",
        "revenue_partner_ssc",
        "revenue_total",
        "expenditure_child_benefits",
        "expenditure_care_benefits",
        "expenditure_unemployment_benefits",
        "expenditure_net_pension",
        "expenditure_total",
        "net_budget",
    ]
    combined_table = combined_table[column_order]

    breakpoint()

    # Save to CSV
    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)
    combined_table.to_csv(path_to_save_table, index=False)


def calculate_net_pension_payout(df):
    """Calculate net pension payout from government perspective.

    Net pension payout = gross pension income - health/LTC SSC contributions.
    This follows the same logic as calc_pensions_after_ssc:
    retirement_income_after_ssc = gross_retirement_income - calc_health_ltc_contr(gross_retirement_income)

    For retirees: net pension = gross pension - health/LTC SSC (what government pays out net)
    For workers/unemployed: no pension payout (0)

    Note: This is only for own pension, not partner pension.
    """
    df = df.copy()

    # Identify employment status from choice
    is_retired = df["choice"].isin(RETIREMENT)
    gross_pension = df["gross_retirement_income"] * is_retired

    # Health/LTC SSC contributions for retirees using the actual function
    # Retirees only pay health/LTC SSC (not pension/unemployment SSC)
    # Use vmap to vectorize the function call
    vmap_calc_health_ltc = jax.vmap(calc_health_ltc_contr)
    gross_pension_jax = jnp.asarray(gross_pension)
    health_ltc_ssc_jax = vmap_calc_health_ltc(gross_pension_jax)
    health_ltc_ssc = np.asarray(health_ltc_ssc_jax) * is_retired

    # Net pension payout = gross pension - health/LTC SSC
    # This is the net amount government pays out (gross pension minus SSC received)
    net_pension_payout = gross_pension - health_ltc_ssc

    return net_pension_payout


def aggregate_fiscal_components(df, scenario_name):
    """Aggregate fiscal components by age for a given scenario."""
    # Identify employment status
    is_retired = df["choice"].isin(RETIREMENT)
    is_working = df["choice"].isin(PART_TIME) | df["choice"].isin(FULL_TIME)
    is_unemployed = df["choice"].isin(UNEMPLOYED)

    # Revenue components
    revenue_income_tax = df["income_tax"] * is_working
    revenue_own_ssc = df["own_ssc"] * (is_working | is_retired)
    revenue_partner_ssc = df["partner_ssc"] * df["partner_state"] > 0

    # Expenditure components
    # Care benefits: only positive values (benefits, not costs)
    care_benefits = df["care_benefits_and_costs"] * (df["care_benefits_and_costs"] > 0)

    expenditure_child_benefits = df["child_benefits"]
    expenditure_care_benefits = care_benefits
    expenditure_unemployment_benefits = (
        df["household_unemployment_benefits"] * is_unemployed
    )
    expenditure_net_pension = df["net_pension_payout"]

    # Aggregate by age
    df_agg = pd.DataFrame(
        {
            "age": df["age"],
            "revenue_income_tax": revenue_income_tax,
            "revenue_own_ssc": revenue_own_ssc,
            "revenue_partner_ssc": revenue_partner_ssc,
            "expenditure_child_benefits": expenditure_child_benefits,
            "expenditure_care_benefits": expenditure_care_benefits,
            "expenditure_unemployment_benefits": expenditure_unemployment_benefits,
            "expenditure_net_pension": expenditure_net_pension,
        }
    )

    # Group by age and calculate means
    fiscal_by_age = df_agg.groupby("age").agg(
        {
            "revenue_income_tax": "mean",
            "revenue_own_ssc": "mean",
            "revenue_partner_ssc": "mean",
            "expenditure_child_benefits": "mean",
            "expenditure_care_benefits": "mean",
            "expenditure_unemployment_benefits": "mean",
            "expenditure_net_pension": "mean",
        }
    )

    # Calculate totals
    fiscal_by_age["revenue_total"] = (
        fiscal_by_age["revenue_income_tax"]
        + fiscal_by_age["revenue_own_ssc"]
        + fiscal_by_age["revenue_partner_ssc"]
    )

    fiscal_by_age["expenditure_total"] = (
        fiscal_by_age["expenditure_child_benefits"]
        + fiscal_by_age["expenditure_care_benefits"]
        + fiscal_by_age["expenditure_unemployment_benefits"]
        + fiscal_by_age["expenditure_net_pension"]
    )

    fiscal_by_age["net_budget"] = (
        fiscal_by_age["revenue_total"] - fiscal_by_age["expenditure_total"]
    )

    # Add scenario name
    fiscal_by_age["scenario"] = scenario_name

    return fiscal_by_age


def convert_to_currency(df, wealth_unit):
    """Convert model units to actual currency."""
    df = df.copy()
    monetary_cols = [
        "income_tax",
        "total_tax_revenue",
        "own_ssc",
        "partner_ssc",
        "child_benefits",
        "care_benefits_and_costs",
        "household_unemployment_benefits",
        "gross_retirement_income",
        "gross_partner_pension",
        "gross_labor_income",
        "gross_partner_income",
    ]
    for col in monetary_cols:
        if col in df.columns:
            df[col] = df[col] * wealth_unit
    return df
