"""Debug the source of NaN values in simulated data."""

import operator
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from caregiving.config import BLD


def analyze_simulated_data():  # noqa: PLR0912, PLR0915
    """Analyze the simulated data to understand where NaN values come from."""

    print("=== NaN Source Analysis ===\n")

    # Load simulated data
    print("1. Loading simulated data...")
    with open(BLD / "solve_and_simulate" / "simulated_data.pkl", "rb") as f:
        sim_df = pickle.load(f)

    print(f"Simulated data shape: {sim_df.shape}")
    print(f"Columns: {list(sim_df.columns)}")

    # Analyze NaN values by column
    print("\n2. Analyzing NaN values by column...")

    nan_analysis = []
    for col in sim_df.columns:
        if sim_df[col].dtype in ("float64", "int64"):
            total_values = len(sim_df[col])
            nan_count = sim_df[col].isna().sum()
            inf_count = np.isinf(sim_df[col]).sum()
            nan_pct = (nan_count / total_values) * 100

            if nan_count > 0 or inf_count > 0:
                nan_analysis.append(
                    {
                        "column": col,
                        "total": total_values,
                        "nan_count": nan_count,
                        "inf_count": inf_count,
                        "nan_pct": nan_pct,
                        "dtype": sim_df[col].dtype,
                        "range": (
                            (sim_df[col].min(), sim_df[col].max())
                            if nan_count < total_values
                            else (np.nan, np.nan)
                        ),
                    }
                )

    # Sort by NaN percentage
    nan_analysis.sort(key=operator.itemgetter("nan_pct"), reverse=True)

    print(f"Found {len(nan_analysis)} columns with NaN/inf values:")
    for analysis in nan_analysis[:15]:  # Show top 15
        print(
            f"   {analysis['column']:30} | NaN: {analysis['nan_count']:8,} "
            f"({analysis['nan_pct']:5.1f}%) | inf: {analysis['inf_count']:6,}"
        )

    # Analyze patterns in NaN values
    print("\n3. Analyzing NaN patterns...")

    # Check if NaN values are correlated with certain conditions
    key_columns = ["choice", "age", "education", "health", "care_demand"]
    available_key_columns = [col for col in key_columns if col in sim_df.columns]

    if available_key_columns:
        print(f"Checking NaN patterns by: {available_key_columns}")

        for col in nan_analysis[:5]:  # Check top 5 problematic columns
            col_name = col["column"]
            if col_name in sim_df.columns:
                print(f"\n   Analyzing {col_name}:")

                # Check NaN by choice
                if "choice" in sim_df.columns:
                    choice_nan = sim_df.groupby("choice")[col_name].apply(
                        lambda x: x.isna().sum()
                    )
                    choice_total = sim_df.groupby("choice")[col_name].apply(len)
                    choice_pct = (choice_nan / choice_total * 100).round(1)

                    print("NaN by choice:")
                    for choice_val in choice_nan.index:
                        if choice_nan[choice_val] > 0:
                            print(
                                f"       Choice {choice_val}: "
                                f"{choice_nan[choice_val]:,} NaN "
                                f"({choice_pct[choice_val]:.1f}%)"
                            )

                # Check NaN by age
                if "age" in sim_df.columns:
                    age_nan = sim_df.groupby("age")[col_name].apply(
                        lambda x: x.isna().sum()
                    )
                    age_total = sim_df.groupby("age")[col_name].apply(len)
                    age_pct = (age_nan / age_total * 100).round(1)

                    print("NaN by age (showing ages with >10% NaN):")
                    for age_val in age_nan.index:
                        AGE_THRESHOLD = 10
                        if age_pct[age_val] > AGE_THRESHOLD:
                            print(
                                f"       Age {age_val}: "
                                f"{age_nan[age_val]:,} NaN ({age_pct[age_val]:.1f}%)"
                            )

    # Check if certain combinations of states lead to NaN values
    print("\n4. Checking state combinations that lead to NaN...")

    # Focus on the most problematic column
    if nan_analysis:
        most_problematic = nan_analysis[0]["column"]
        print(f"Analyzing {most_problematic} (most problematic column)")

        # Create a subset with non-NaN values
        non_nan_mask = ~sim_df[most_problematic].isna()

        if non_nan_mask.sum() > 0:
            print(
                f"   Non-NaN values: {non_nan_mask.sum():,} "
                f"({non_nan_mask.mean()*100:.1f}%)"
            )

            # Check what states are associated with NaN values
            if "choice" in sim_df.columns and "age" in sim_df.columns:
                nan_states = (
                    sim_df[~non_nan_mask][["choice", "age"]].value_counts().head(10)
                )
                print("Most common states with NaN values:")
                for (choice, age), count in nan_states.items():
                    print(f"Choice {choice}, Age {age}: {count:,} observations")

        # Check the range of non-NaN values
        non_nan_values = sim_df[most_problematic][non_nan_mask]
        if len(non_nan_values) > 0:
            print(
                f"   Non-NaN value range: "
                f"[{non_nan_values.min():.6f}, {non_nan_values.max():.6f}]"
            )
            print(f"Non-NaN value mean: {non_nan_values.mean():.6f}")

    return nan_analysis


def check_solution_data():
    """Check if the solution data has issues."""

    print("\n\n=== Solution Data Analysis ===\n")

    # Load solution data
    print("1. Loading solution data...")
    try:
        with open(BLD / "solve_and_simulate" / "solution.pkl", "rb") as f:
            solution = pickle.load(f)

        print("Solution loaded")
        print(f"Solution keys: {list(solution.keys())}")

        # Check each component of the solution
        for key, value in solution.items():
            if hasattr(value, "shape"):
                print(f"{key}: shape={value.shape}")

                if hasattr(value, "dtype") and value.dtype in ("float64", "float32"):
                    has_nan = np.any(np.isnan(value))
                    has_inf = np.any(np.isinf(value))

                    if has_nan or has_inf:
                        print(
                            f"     WARNING: {key} contains NaN={has_nan}, inf={has_inf}"
                        )
                        if has_nan:
                            print(f"NaN count: {np.sum(np.isnan(value)):,}")
                        if has_inf:
                            print(f"inf count: {np.sum(np.isinf(value)):,}")
                    else:
                        print(f"{key} is numerically stable")
                        print(
                            f"       Range: "
                            f"[{np.nanmin(value):.6f}, {np.nanmax(value):.6f}]"
                        )

    except Exception as e:
        print(f"ERROR: Error loading solution: {e}")


def check_initial_conditions():
    """Check if initial conditions have issues."""

    print("\n\n=== Initial Conditions Analysis ===\n")

    # Load initial conditions
    print("1. Loading initial conditions...")
    try:
        with open(BLD / "model" / "initial_conditions" / "states.pkl", "rb") as f:
            initial_states = pickle.load(f)

        print("Initial states loaded")
        print(f"Initial states keys: {list(initial_states.keys())}")

        # Check each component
        for key, value in initial_states.items():
            if hasattr(value, "shape"):
                print(f"{key}: shape={value.shape}")

                if hasattr(value, "dtype") and value.dtype in (
                    "float64",
                    "float32",
                    "int64",
                    "int32",
                ):
                    if value.dtype in ("float64", "float32"):
                        has_nan = np.any(np.isnan(value))
                        has_inf = np.any(np.isinf(value))

                        if has_nan or has_inf:
                            print(
                                f"     WARNING: {key} contains "
                                f"NaN={has_nan}, inf={has_inf}"
                            )
                        else:
                            print(f"{key} is numerically stable")
                            print(
                                f"       Range: "
                                f"[{np.nanmin(value):.6f}, {np.nanmax(value):.6f}]"
                            )
                    else:
                        print(f"{key} is numerically stable")
                        print(f"Range: [{np.min(value)}, {np.max(value)}]")

        # Load wealth data
        wealth_df = pd.read_csv(BLD / "model" / "initial_conditions" / "wealth.csv")
        print(f"Wealth data loaded: shape={wealth_df.shape}")

        if "wealth" in wealth_df.columns:
            wealth_data = wealth_df["wealth"]
            has_nan = np.any(np.isnan(wealth_data))
            has_inf = np.any(np.isinf(wealth_data))

            if has_nan or has_inf:
                print(f"WARNING: wealth contains NaN={has_nan}, inf={has_inf}")
            else:
                print("wealth is numerically stable")
                print(
                    f"       Range: [{wealth_data.min():.6f}, {wealth_data.max():.6f}]"
                )

    except Exception as e:
        print(f"ERROR: Error loading initial conditions: {e}")
