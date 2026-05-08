#!/usr/bin/env python3
"""Debug script to investigate tax revenue discrepancy."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pickle

import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.model.shared import FULL_TIME, PART_TIME, WORK

# Load simulated data
path_to_baseline_sim = (
    BLD / "solve_and_simulate" / "simulated_data_estimated_params.pkl"
)
path_to_no_care_demand_sim = (
    BLD / "solve_and_simulate" / "simulated_data_no_care_demand.pkl"
)
path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"

print("Loading baseline data...")
baseline_df = pd.read_pickle(path_to_baseline_sim)
print(
    f"Baseline loaded: {len(baseline_df)} rows, {baseline_df.memory_usage(deep=True).sum() / 1e9:.2f} GB"
)

print("Loading no care demand data...")
no_care_demand_df = pd.read_pickle(path_to_no_care_demand_sim)
print(
    f"No care demand loaded: {len(no_care_demand_df)} rows, {no_care_demand_df.memory_usage(deep=True).sum() / 1e9:.2f} GB"
)

# Load specs for wealth_unit conversion
specs = pickle.load(path_to_specs.open("rb"))
wealth_unit = specs["wealth_unit"]

# Convert to currency (if convert_to_currency function exists, otherwise just use as is)
# For now, assume columns are already in wealth_unit and we'll convert manually
baseline_df_currency = baseline_df.copy()
no_care_demand_df_currency = no_care_demand_df.copy()

# Compare income_tax_single for working choices only
# Use numpy isin for much faster filtering
work_choices = np.asarray(WORK)
baseline_working_mask = np.isin(baseline_df["choice"].values, work_choices)
no_care_demand_working_mask = np.isin(no_care_demand_df["choice"].values, work_choices)
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

baseline_income_tax_single_sum = baseline_income_tax_single_currency.sum() / 100_000
no_care_demand_income_tax_single_sum = (
    no_care_demand_income_tax_single_currency.sum() / 100_000
)

# Diagnostic: Check for negative tax values (shouldn't happen but let's verify)
baseline_negative_tax = (baseline_income_tax_single_currency < 0).sum()
no_care_demand_negative_tax = (no_care_demand_income_tax_single_currency < 0).sum()

# Check income distribution and tax threshold
# Tax threshold is typically around 8004 (thresholds[1])
# Income values are in wealth_unit, so need to convert to actual currency
tax_threshold = specs.get("income_tax_brackets", [0.0, 8004, 13469, 52881, 250730])[1]

# Convert income to actual currency for comparison
baseline_income_currency = baseline_working["own_income_after_ssc"] * wealth_unit
no_care_demand_income_currency = (
    no_care_demand_working["own_income_after_ssc"] * wealth_unit
)

baseline_below_threshold = (baseline_income_currency <= tax_threshold).sum()
no_care_demand_below_threshold = (no_care_demand_income_currency <= tax_threshold).sum()

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

# Check total tax from workers above threshold
baseline_tax_above_threshold = (
    baseline_income_tax_single_currency[baseline_above_threshold_mask].sum() / 100_000
)
no_care_demand_tax_above_threshold = (
    no_care_demand_income_tax_single_currency[no_care_demand_above_threshold_mask].sum()
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

# # Check part-time vs full-time composition
# baseline_pt = (baseline_working["choice"].isin(PART_TIME)).sum()
# baseline_ft = (baseline_working["choice"].isin(FULL_TIME)).sum()
# no_care_demand_pt = (no_care_demand_working["choice"].isin(PART_TIME)).sum()
# no_care_demand_ft = (no_care_demand_working["choice"].isin(FULL_TIME)).sum()

# print("\n=== Income Tax Single Comparison (Working Choices Only) ===")
# print(
#     f"Baseline income_tax_single sum (in 100k): {baseline_income_tax_single_sum:.2f}"
# )
# print(
#     f"No care demand income_tax_single sum (in 100k): {no_care_demand_income_tax_single_sum:.2f}"
# )
# print(
#     f"Difference (in 100k): {baseline_income_tax_single_sum - no_care_demand_income_tax_single_sum:.2f}"
# )
# print(f"\nBaseline workers: {baseline_working_mask.sum()}")
# print(f"No care demand workers: {no_care_demand_working_mask.sum()}")
# print(f"\nBaseline avg gross labor income: {baseline_avg_income:.2f}")
# print(f"No care demand avg gross labor income: {no_care_demand_avg_income:.2f}")
# print(f"\nBaseline avg income after SSC: {baseline_avg_income_after_ssc:.2f}")
# print(
#     f"No care demand avg income after SSC: {no_care_demand_avg_income_after_ssc:.2f}"
# )
# print(f"\nBaseline avg income_tax_single (currency): {baseline_avg_tax:.2f}")
# print(
#     f"No care demand avg income_tax_single (currency): {no_care_demand_avg_tax:.2f}"
# )
# print(f"\nBaseline negative tax count: {baseline_negative_tax}")
# print(f"No care demand negative tax count: {no_care_demand_negative_tax}")
# print(
#     f"\nBaseline tax from workers above threshold (100k): {baseline_tax_above_threshold:.2f}"
# )
# print(
#     f"No care demand tax from workers above threshold (100k): {no_care_demand_tax_above_threshold:.2f}"
# )
# print(f"\nBaseline avg experience: {baseline_avg_experience:.2f}")
# print(f"No care demand avg experience: {no_care_demand_avg_experience:.2f}")
# print(
#     f"\nBaseline workers below tax threshold ({tax_threshold}): {baseline_below_threshold} ({baseline_pct_below_threshold:.1f}%)"
# )
# print(
#     f"No care demand workers below tax threshold ({tax_threshold}): {no_care_demand_below_threshold} ({no_care_demand_pct_below_threshold:.1f}%)"
# )
# print(
#     f"\nBaseline workers above threshold: {len(baseline_above_threshold)} (avg income: {baseline_avg_income_above:.2f}, avg tax: {baseline_avg_tax_above:.2f})"
# )
# print(
#     f"No care demand workers above threshold: {len(no_care_demand_above_threshold)} (avg income: {no_care_demand_avg_income_above:.2f}, avg tax: {no_care_demand_avg_tax_above:.2f})"
# )
# print(f"\nBaseline: PT={baseline_pt}, FT={baseline_ft}")
# print(f"No care demand: PT={no_care_demand_pt}, FT={no_care_demand_ft}")

# Additional diagnostics
print("\n=== ADDITIONAL DIAGNOSTICS ===")
print(f"Wealth unit: {wealth_unit}")
print(f"Tax threshold: {tax_threshold}")

# Check if there's a difference in the distribution
print("\n=== Income Distribution Comparison ===")
print(
    f"Baseline income after SSC - min: {baseline_income_currency.min():.2f}, max: {baseline_income_currency.max():.2f}, median: {baseline_income_currency.median():.2f}"
)
print(
    f"No care demand income after SSC - min: {no_care_demand_income_currency.min():.2f}, max: {no_care_demand_income_currency.max():.2f}, median: {no_care_demand_income_currency.median():.2f}"
)

# Check tax distribution
print("\n=== Tax Distribution Comparison ===")
print(
    f"Baseline tax - min: {baseline_income_tax_single_currency.min():.2f}, max: {baseline_income_tax_single_currency.max():.2f}, median: {baseline_income_tax_single_currency.median():.2f}"
)
print(
    f"No care demand tax - min: {no_care_demand_income_tax_single_currency.min():.2f}, max: {no_care_demand_income_tax_single_currency.max():.2f}, median: {no_care_demand_income_tax_single_currency.median():.2f}"
)

# Check if the issue is with workers below threshold paying tax (shouldn't happen)
baseline_tax_below_threshold = (
    baseline_income_tax_single_currency[~baseline_above_threshold_mask].sum() / 100_000
)
no_care_demand_tax_below_threshold = (
    no_care_demand_income_tax_single_currency[
        ~no_care_demand_above_threshold_mask
    ].sum()
    / 100_000
)
print(
    f"\nBaseline tax from workers BELOW threshold (should be ~0): {baseline_tax_below_threshold:.2f}"
)
print(
    f"No care demand tax from workers BELOW threshold (should be ~0): {no_care_demand_tax_below_threshold:.2f}"
)

# Check wage components
print("\n=== Wage Component Analysis ===")
print(f"Baseline avg experience_years: {baseline_working['exp_years'].mean():.2f}")
print(
    f"No care demand avg experience_years: {no_care_demand_working['exp_years'].mean():.2f}"
)

print(
    f"\nBaseline avg gross_labor_income: {baseline_working['gross_labor_income'].mean():.2f}"
)
print(
    f"No care demand avg gross_labor_income: {no_care_demand_working['gross_labor_income'].mean():.2f}"
)

# Check if there's a difference in the distribution of experience
print(
    f"\nBaseline experience_years - min: {baseline_working['exp_years'].min():.2f}, max: {baseline_working['exp_years'].max():.2f}, median: {baseline_working['exp_years'].median():.2f}"
)
print(
    f"No care demand experience_years - min: {no_care_demand_working['exp_years'].min():.2f}, max: {no_care_demand_working['exp_years'].max():.2f}, median: {no_care_demand_working['exp_years'].median():.2f}"
)

# Check if there's a difference in education distribution
if "education" in baseline_working.columns:
    print("\nBaseline education distribution:")
    print(baseline_working["education"].value_counts(normalize=True).sort_index())
    print("No care demand education distribution:")
    print(no_care_demand_working["education"].value_counts(normalize=True).sort_index())

# Check if there's a difference in partner_state (affects tax calculation)
if "partner_state" in baseline_working.columns:
    print("\nBaseline partner_state distribution:")
    print(baseline_working["partner_state"].value_counts(normalize=True).sort_index())
    print("No care demand partner_state distribution:")
    print(
        no_care_demand_working["partner_state"]
        .value_counts(normalize=True)
        .sort_index()
    )

    # Check if partner income affects the tax calculation
    if "partner_income_after_ssc" in baseline_working.columns:
        print(
            f"\nBaseline avg partner_income_after_ssc: {baseline_working['partner_income_after_ssc'].mean():.2f}"
        )
        print(
            f"No care demand avg partner_income_after_ssc: {no_care_demand_working['partner_income_after_ssc'].mean():.2f}"
        )
