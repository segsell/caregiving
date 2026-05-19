#!/usr/bin/env python3
"""Debug script to compare education distribution at initial period vs working population."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pickle

import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.model.shared import (
    FULL_TIME,
    FULL_TIME_NO_CARE,
    PART_TIME,
    PART_TIME_NO_CARE,
    RETIREMENT,
    RETIREMENT_NO_CARE,
    UNEMPLOYED,
    UNEMPLOYED_NO_CARE,
    WORK,
)

# Load simulated data
path_to_baseline_sim = (
    BLD / "solve_and_simulate" / "simulated_data_estimated_params.pkl"
)
path_to_no_care_demand_sim = (
    BLD / "solve_and_simulate" / "simulated_data_no_care_demand.pkl"
)
path_to_initial_states = BLD / "model" / "initial_conditions" / "initial_states.pkl"
path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"

print("Loading data...")
baseline_df = pd.read_pickle(path_to_baseline_sim)
no_care_demand_df = pd.read_pickle(path_to_no_care_demand_sim)

# Load specs to get start_age
specs = pickle.load(path_to_specs.open("rb"))
start_age = specs["start_age"]

print(f"Start age: {start_age}")

# Load initial states
print("Loading initial states...")
initial_states = pickle.load(path_to_initial_states.open("rb"))

# Extract education from initial states
# Initial states is a dict with keys like 'education', 'period', etc.
if isinstance(initial_states, dict):
    initial_education = initial_states.get("education", None)
    initial_period = initial_states.get("period", None)

    if initial_education is None:
        print("Warning: 'education' not found in initial_states dict")
        print(f"Available keys: {list(initial_states.keys())}")
        # Try to find it in a nested structure
        if "states_initial" in initial_states:
            states_initial = initial_states["states_initial"]
            if isinstance(states_initial, dict):
                initial_education = states_initial.get("education", None)
                initial_period = states_initial.get("period", None)
                print(f"Found in states_initial, keys: {list(states_initial.keys())}")

    if initial_education is not None:
        # Convert to numpy array if needed
        if hasattr(initial_education, "values"):
            initial_education = initial_education.values
        initial_education = np.asarray(initial_education)

        # Filter to period 0 (age 30)
        if initial_period is not None:
            if hasattr(initial_period, "values"):
                initial_period = initial_period.values
            initial_period = np.asarray(initial_period)
            period_0_mask = initial_period == 0
            initial_education_p0 = initial_education[period_0_mask]
        else:
            # Assume all are period 0 if period not provided
            initial_education_p0 = initial_education

        # Calculate shares
        n_total = len(initial_education_p0)
        n_low_edu = (initial_education_p0 == 0).sum()
        n_high_edu = (initial_education_p0 == 1).sum()

        pct_low_edu_initial = n_low_edu / n_total * 100
        pct_high_edu_initial = n_high_edu / n_total * 100

        print(f"\n=== Initial Conditions (Period 0, Age {start_age}) ===")
        print(f"Total agents: {n_total}")
        print(f"Low education (0): {n_low_edu} ({pct_low_edu_initial:.2f}%)")
        print(f"High education (1): {n_high_edu} ({pct_high_edu_initial:.2f}%)")
    else:
        print("ERROR: Could not find education in initial_states")
        print(f"Initial states type: {type(initial_states)}")
        if isinstance(initial_states, dict):
            print(f"Keys: {list(initial_states.keys())}")
else:
    print(f"Initial states is not a dict, type: {type(initial_states)}")
    # Try to convert to DataFrame if it's a DataFrame
    if hasattr(initial_states, "columns"):
        print("Initial states appears to be a DataFrame")
        print(f"Columns: {list(initial_states.columns)}")
        if "education" in initial_states.columns:
            if "period" in initial_states.columns:
                initial_education_p0 = initial_states[initial_states["period"] == 0][
                    "education"
                ]
            else:
                initial_education_p0 = initial_states["education"]

            n_total = len(initial_education_p0)
            n_low_edu = (initial_education_p0 == 0).sum()
            n_high_edu = (initial_education_p0 == 1).sum()

            pct_low_edu_initial = n_low_edu / n_total * 100
            pct_high_edu_initial = n_high_edu / n_total * 100

            print(f"\n=== Initial Conditions (Period 0, Age {start_age}) ===")
            print(f"Total agents: {n_total}")
            print(f"Low education (0): {n_low_edu} ({pct_low_edu_initial:.2f}%)")
            print(f"High education (1): {n_high_edu} ({pct_high_edu_initial:.2f}%)")

# Now compute working population shares at age 30 (period 0)
print("\n=== Working Population at Age 30 (Period 0) ===")

# Filter to period 0 and working
baseline_p0 = baseline_df[baseline_df["period"] == 0].copy()
no_care_demand_p0 = no_care_demand_df[no_care_demand_df["period"] == 0].copy()

work_choices = np.asarray(WORK)
baseline_working_p0 = baseline_p0[baseline_p0["choice"].isin(work_choices)].copy()
no_care_demand_working_p0 = no_care_demand_p0[
    no_care_demand_p0["choice"].isin(work_choices)
].copy()

# Calculate education shares for working population at period 0
baseline_edu_dist_p0 = (
    baseline_working_p0["education"].value_counts(normalize=True).sort_index()
)
no_care_demand_edu_dist_p0 = (
    no_care_demand_working_p0["education"].value_counts(normalize=True).sort_index()
)

print("\nBaseline working at period 0:")
print(f"  Total workers: {len(baseline_working_p0)}")
print(f"  Low education (0): {baseline_edu_dist_p0.get(0, 0)*100:.2f}%")
print(f"  High education (1): {baseline_edu_dist_p0.get(1, 0)*100:.2f}%")

print("\nNo care demand working at period 0:")
print(f"  Total workers: {len(no_care_demand_working_p0)}")
print(f"  Low education (0): {no_care_demand_edu_dist_p0.get(0, 0)*100:.2f}%")
print(f"  High education (1): {no_care_demand_edu_dist_p0.get(1, 0)*100:.2f}%")

# Also compute overall working population shares (all periods) for comparison
print("\n=== Overall Working Population (All Periods) ===")
work_choices = np.asarray(WORK)
baseline_working_mask = np.isin(baseline_df["choice"].values, work_choices)
no_care_demand_working_mask = np.isin(no_care_demand_df["choice"].values, work_choices)
baseline_working = baseline_df[baseline_working_mask].copy()
no_care_demand_working = no_care_demand_df[no_care_demand_working_mask].copy()

baseline_edu_dist = (
    baseline_working["education"].value_counts(normalize=True).sort_index()
)
no_care_demand_edu_dist = (
    no_care_demand_working["education"].value_counts(normalize=True).sort_index()
)

print("\nBaseline working (all periods):")
print(f"  Total workers: {len(baseline_working)}")
print(f"  Low education (0): {baseline_edu_dist.get(0, 0)*100:.2f}%")
print(f"  High education (1): {baseline_edu_dist.get(1, 0)*100:.2f}%")

print("\nNo care demand working (all periods):")
print(f"  Total workers: {len(no_care_demand_working)}")
print(f"  Low education (0): {no_care_demand_edu_dist.get(0, 0)*100:.2f}%")
print(f"  High education (1): {no_care_demand_edu_dist.get(1, 0)*100:.2f}%")

# Summary comparison
print("\n=== SUMMARY COMPARISON ===")
if "pct_low_edu_initial" in locals():
    print("\nInitial conditions (period 0):")
    print(f"  Low education: {pct_low_edu_initial:.2f}%")
    print(f"  High education: {pct_high_edu_initial:.2f}%")

print("\nBaseline working at period 0:")
print(f"  Low education: {baseline_edu_dist_p0.get(0, 0)*100:.2f}%")
print(f"  High education: {baseline_edu_dist_p0.get(1, 0)*100:.2f}%")

print("\nNo care demand working at period 0:")
print(f"  Low education: {no_care_demand_edu_dist_p0.get(0, 0)*100:.2f}%")
print(f"  High education: {no_care_demand_edu_dist_p0.get(1, 0)*100:.2f}%")

print("\nBaseline working (all periods):")
print(f"  Low education: {baseline_edu_dist.get(0, 0)*100:.2f}%")
print(f"  High education: {baseline_edu_dist.get(1, 0)*100:.2f}%")

print("\nNo care demand working (all periods):")
print(f"  Low education: {no_care_demand_edu_dist.get(0, 0)*100:.2f}%")
print(f"  High education: {no_care_demand_edu_dist.get(1, 0)*100:.2f}%")

# Compare lagged_choice at period 0
print("\n=== Lagged Choice Distribution at Age 30 (Period 0) ===")


# Helper function to count choices
def count_choices(choice_array, choice_set, total):
    """Count choices in array that match choice_set."""
    if hasattr(choice_array, "values"):
        choice_array = choice_array.values
    choice_array = np.asarray(choice_array)
    count = np.isin(choice_array, choice_set).sum()
    return count, count / total * 100 if total > 0 else 0


# Initial states lagged_choice
if isinstance(initial_states, dict):
    initial_lagged_choice = initial_states.get("lagged_choice", None)
    initial_period = initial_states.get("period", None)

    if initial_lagged_choice is None and "states_initial" in initial_states:
        states_initial = initial_states["states_initial"]
        if isinstance(states_initial, dict):
            initial_lagged_choice = states_initial.get("lagged_choice", None)
            initial_period = states_initial.get("period", None)

    if initial_lagged_choice is not None:
        if hasattr(initial_lagged_choice, "values"):
            initial_lagged_choice = initial_lagged_choice.values
        initial_lagged_choice = np.asarray(initial_lagged_choice)

        if initial_period is not None:
            if hasattr(initial_period, "values"):
                initial_period = initial_period.values
            initial_period = np.asarray(initial_period)
            period_0_mask = initial_period == 0
            initial_lagged_choice_p0 = initial_lagged_choice[period_0_mask]
        else:
            initial_lagged_choice_p0 = initial_lagged_choice

        initial_total_p0 = len(initial_lagged_choice_p0)
        initial_unemployed_p0, initial_unemployed_pct = count_choices(
            initial_lagged_choice_p0, UNEMPLOYED, initial_total_p0
        )
        initial_pt_p0, initial_pt_pct = count_choices(
            initial_lagged_choice_p0, PART_TIME, initial_total_p0
        )
        initial_ft_p0, initial_ft_pct = count_choices(
            initial_lagged_choice_p0, FULL_TIME, initial_total_p0
        )
        initial_retired_p0, initial_retired_pct = count_choices(
            initial_lagged_choice_p0, RETIREMENT, initial_total_p0
        )

        print("\nInitial states (period 0):")
        print(f"  Total: {initial_total_p0}")
        print(f"  Unemployed: {initial_unemployed_p0} ({initial_unemployed_pct:.2f}%)")
        print(f"  Part-time: {initial_pt_p0} ({initial_pt_pct:.2f}%)")
        print(f"  Full-time: {initial_ft_p0} ({initial_ft_pct:.2f}%)")
        print(f"  Retired: {initial_retired_p0} ({initial_retired_pct:.2f}%)")
elif hasattr(initial_states, "columns"):
    if "lagged_choice" in initial_states.columns:
        if "period" in initial_states.columns:
            initial_lagged_choice_p0 = initial_states[initial_states["period"] == 0][
                "lagged_choice"
            ]
        else:
            initial_lagged_choice_p0 = initial_states["lagged_choice"]

        initial_total_p0 = len(initial_lagged_choice_p0)
        initial_unemployed_p0 = (initial_lagged_choice_p0.isin(UNEMPLOYED)).sum()
        initial_pt_p0 = (initial_lagged_choice_p0.isin(PART_TIME)).sum()
        initial_ft_p0 = (initial_lagged_choice_p0.isin(FULL_TIME)).sum()
        initial_retired_p0 = (initial_lagged_choice_p0.isin(RETIREMENT)).sum()

        print("\nInitial states (period 0):")
        print(f"  Total: {initial_total_p0}")
        print(
            f"  Unemployed: {initial_unemployed_p0} ({initial_unemployed_p0/initial_total_p0*100:.2f}%)"
        )
        print(
            f"  Part-time: {initial_pt_p0} ({initial_pt_p0/initial_total_p0*100:.2f}%)"
        )
        print(
            f"  Full-time: {initial_ft_p0} ({initial_ft_p0/initial_total_p0*100:.2f}%)"
        )
        print(
            f"  Retired: {initial_retired_p0} ({initial_retired_p0/initial_total_p0*100:.2f}%)"
        )

# Baseline and no-care-demand at period 0
# Note: lagged_choice at period 0 is from period -1 (doesn't exist), so compare choice instead
# At period 0, choices might be in simplified space (0,1,2,3) or full space
baseline_p0_choice = baseline_p0["choice"]
no_care_demand_p0_choice = no_care_demand_p0["choice"]

baseline_total_p0 = len(baseline_p0)
# At period 0, choices are in simplified space: 0=retired, 1=unemployed, 2=part-time, 3=full-time
# Also check full choice space in case some choices are in full space
baseline_unemployed_p0 = max(
    (baseline_p0_choice.isin(UNEMPLOYED)).sum(),
    (baseline_p0_choice == 1).sum(),  # Simplified: 1 = unemployed
)
baseline_pt_p0 = max(
    (baseline_p0_choice.isin(PART_TIME)).sum(),
    (baseline_p0_choice == 2).sum(),  # Simplified: 2 = part-time
)
baseline_ft_p0 = max(
    (baseline_p0_choice.isin(FULL_TIME)).sum(),
    (baseline_p0_choice == 3).sum(),  # Simplified: 3 = full-time
)
baseline_retired_p0 = max(
    (baseline_p0_choice.isin(RETIREMENT)).sum(),
    (baseline_p0_choice == 0).sum(),  # Simplified: 0 = retired
)

no_care_demand_total_p0 = len(no_care_demand_p0)
no_care_demand_unemployed_p0 = max(
    (no_care_demand_p0_choice.isin(UNEMPLOYED)).sum(),
    (no_care_demand_p0_choice == 1).sum(),
)
no_care_demand_pt_p0 = max(
    (no_care_demand_p0_choice.isin(PART_TIME)).sum(),
    (no_care_demand_p0_choice == 2).sum(),
)
no_care_demand_ft_p0 = max(
    (no_care_demand_p0_choice.isin(FULL_TIME)).sum(),
    (no_care_demand_p0_choice == 3).sum(),
)
no_care_demand_retired_p0 = max(
    (no_care_demand_p0_choice.isin(RETIREMENT)).sum(),
    (no_care_demand_p0_choice == 0).sum(),
)

# Check actual choice values
print("\nBaseline choice values at period 0:")
print(baseline_p0_choice.value_counts().sort_index().head(20))
print("\nNo care demand choice values at period 0:")
print(no_care_demand_p0_choice.value_counts().sort_index().head(20))

print("\nBaseline simulated CHOICE (period 0):")
print(f"  Total: {baseline_total_p0}")
print(
    f"  Unemployed: {baseline_unemployed_p0} ({baseline_unemployed_p0/baseline_total_p0*100:.2f}%)"
)
print(f"  Part-time: {baseline_pt_p0} ({baseline_pt_p0/baseline_total_p0*100:.2f}%)")
print(f"  Full-time: {baseline_ft_p0} ({baseline_ft_p0/baseline_total_p0*100:.2f}%)")
print(
    f"  Retired: {baseline_retired_p0} ({baseline_retired_p0/baseline_total_p0*100:.2f}%)"
)

print("\nNo care demand simulated CHOICE (period 0):")
print(f"  Total: {no_care_demand_total_p0}")
print(
    f"  Unemployed: {no_care_demand_unemployed_p0} ({no_care_demand_unemployed_p0/no_care_demand_total_p0*100:.2f}%)"
)
print(
    f"  Part-time: {no_care_demand_pt_p0} ({no_care_demand_pt_p0/no_care_demand_total_p0*100:.2f}%)"
)
print(
    f"  Full-time: {no_care_demand_ft_p0} ({no_care_demand_ft_p0/no_care_demand_total_p0*100:.2f}%)"
)
print(
    f"  Retired: {no_care_demand_retired_p0} ({no_care_demand_retired_p0/no_care_demand_total_p0*100:.2f}%)"
)

# Compare job_offer at period 0
print("\n=== Job Offer Distribution at Age 30 (Period 0) ===")

# Check if job_offer exists in initial states
if isinstance(initial_states, dict):
    initial_job_offer = initial_states.get("job_offer", None)
    if initial_job_offer is None and "states_initial" in initial_states:
        states_initial = initial_states["states_initial"]
        if isinstance(states_initial, dict):
            initial_job_offer = states_initial.get("job_offer", None)

    if initial_job_offer is not None:
        if hasattr(initial_job_offer, "values"):
            initial_job_offer = initial_job_offer.values
        initial_job_offer = np.asarray(initial_job_offer)

        if initial_period is not None:
            if hasattr(initial_period, "values"):
                initial_period = initial_period.values
            initial_period = np.asarray(initial_period)
            period_0_mask = initial_period == 0
            initial_job_offer_p0 = initial_job_offer[period_0_mask]
        else:
            initial_job_offer_p0 = initial_job_offer

        if len(initial_job_offer_p0) > 0:
            initial_job_offer_dist = (
                pd.Series(initial_job_offer_p0)
                .value_counts(normalize=True)
                .sort_index()
            )
            print("\nInitial states job_offer (period 0):")
            for val, pct in initial_job_offer_dist.items():
                print(f"  {val}: {pct*100:.2f}%")
    else:
        print("Warning: job_offer not found in initial_states")
        if isinstance(initial_states, dict):
            print(f"Available keys: {list(initial_states.keys())}")
            if "states_initial" in initial_states:
                print(
                    f"states_initial keys: {list(initial_states['states_initial'].keys())}"
                )
elif hasattr(initial_states, "columns"):
    if "job_offer" in initial_states.columns:
        if "period" in initial_states.columns:
            initial_job_offer_p0 = initial_states[initial_states["period"] == 0][
                "job_offer"
            ]
        else:
            initial_job_offer_p0 = initial_states["job_offer"]

        initial_job_offer_dist = initial_job_offer_p0.value_counts(
            normalize=True
        ).sort_index()
        print("\nInitial states job_offer (period 0):")
        for val, pct in initial_job_offer_dist.items():
            print(f"  {val}: {pct*100:.2f}%")
    else:
        print("Warning: job_offer column not found in initial_states DataFrame")
        print(f"Available columns: {list(initial_states.columns)}")

# Baseline and no-care-demand job_offer at period 0
if "job_offer" in baseline_p0.columns:
    baseline_job_offer_p0_dist = (
        baseline_p0["job_offer"].value_counts(normalize=True).sort_index()
    )
    print("\nBaseline simulated job_offer (period 0):")
    for val, pct in baseline_job_offer_p0_dist.items():
        print(f"  {val}: {pct*100:.2f}%")
else:
    print("\nBaseline: job_offer column not found")

if "job_offer" in no_care_demand_p0.columns:
    no_care_demand_job_offer_p0_dist = (
        no_care_demand_p0["job_offer"].value_counts(normalize=True).sort_index()
    )
    print("\nNo care demand simulated job_offer (period 0):")
    for val, pct in no_care_demand_job_offer_p0_dist.items():
        print(f"  {val}: {pct*100:.2f}%")
else:
    print("\nNo care demand: job_offer column not found")
