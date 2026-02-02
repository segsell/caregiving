"""Lightweight diagnostic script to investigate retirement wealth spike.

This script loads wealth data and checks how experience scaling affects
fresh retirees when adjusting wealth.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import dcegm
from caregiving.model.experience_baseline_model import (
    construct_experience_years,
    scale_experience_years,
)
from caregiving.model.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)
from caregiving.model.shared import RETIREMENT, WORK
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint


def load_data_and_model():
    """Load necessary data and model components."""
    bld = project_root / "bld"

    # Load specs
    specs_path = bld / "model" / "specs" / "specs_full.pkl"
    with specs_path.open("rb") as f:
        specs = pickle.load(f)

    # Load params
    params_path = bld / "estimation" / "result.pkl"
    with params_path.open("rb") as f:
        result = pickle.load(f)
        params = result["x"]

    # Load model config
    model_config_path = bld / "model" / "model_config.yaml"
    import yaml

    with model_config_path.open("r") as f:
        model_config = yaml.safe_load(f)

    # Setup model class
    model_class = dcegm.setup_model(
        model_config=model_config,
        model_specs=specs,
    )

    # Load wealth data (assuming it exists)
    # Try to load from SOEP moments task output or create a sample
    return specs, params, model_class


def identify_fresh_retirees(
    df, person_id_col="pid", age_col="age", choice_col="choice"
):
    """Identify fresh retirees (first period of retirement)."""
    df = df.copy()
    df = df.sort_values([person_id_col, age_col]).reset_index(drop=True)

    retirement_values = np.asarray(RETIREMENT).ravel().tolist()

    # Current period retirement status
    df["is_retired_now"] = df[choice_col].isin(retirement_values)

    # Previous period retirement status
    df["lagged_choice"] = df.groupby(person_id_col)[choice_col].shift(1)
    df["is_retired_prev"] = df["lagged_choice"].isin(retirement_values).fillna(False)

    # Fresh retirees: retired now but not in previous period
    df["is_fresh_retiree"] = df["is_retired_now"] & (~df["is_retired_prev"])

    return df


def analyze_experience_scaling(df, specs, model_class):
    """Analyze how experience scaling affects fresh retirees."""
    results = []

    fresh_retirees = df[df["is_fresh_retiree"]].copy()

    if len(fresh_retirees) == 0:
        print("No fresh retirees found in the data.")
        return None

    print(f"\nFound {len(fresh_retirees)} fresh retirees")
    print(
        f"Age range: {fresh_retirees['age'].min()} to {fresh_retirees['age'].max()}\n"
    )

    for idx, row in fresh_retirees.iterrows():
        age = row["age"]
        period = age - specs["start_age"]
        experience_norm = row["experience"]
        lagged_choice = int(row["lagged_choice"])

        # Get scale factors
        max_exp_working = specs["max_exps_period_working"][period]
        max_pp_retirement = specs["max_pp_retirement"]

        # Calculate experience years with different assumptions
        # Assumption 1: Experience is still on working scale
        exp_years_if_working_scale = experience_norm * max_exp_working

        # Assumption 2: Experience is already on retirement scale
        exp_years_if_retirement_scale = experience_norm * max_pp_retirement

        # What construct_experience_years would return (with is_retired=True)
        exp_years_actual = construct_experience_years(
            float_experience=experience_norm,
            period=period,
            is_retired=True,  # Because lagged_choice is retirement
            model_specs=specs,
        )

        # Calculate pension points from working experience (correct way)
        pension_points_correct = calc_pension_points_for_experience(
            period=period,
            experience_years=exp_years_if_working_scale,
            sex=specs.get("sex", 0),  # Assuming female
            partner_state=row.get("partner_state", 0),
            education=row.get("education", 0),
            model_specs=specs,
        )

        # What budget_constraint would use (incorrect if experience is on wrong scale)
        pension_points_used = (
            exp_years_actual  # This is what gets passed to calc_pensions_after_ssc
        )

        results.append(
            {
                "age": age,
                "period": period,
                "experience_norm": experience_norm,
                "max_exp_working": max_exp_working,
                "max_pp_retirement": max_pp_retirement,
                "exp_years_if_working_scale": exp_years_if_working_scale,
                "exp_years_if_retirement_scale": exp_years_if_retirement_scale,
                "exp_years_actual_used": exp_years_actual,
                "pension_points_correct": pension_points_correct,
                "pension_points_used": pension_points_used,
                "difference": pension_points_used - pension_points_correct,
                "ratio": (
                    pension_points_used / pension_points_correct
                    if pension_points_correct > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(results)


def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("Retirement Wealth Spike Diagnostic")
    print("=" * 80)

    try:
        specs, params, model_class = load_data_and_model()
        print("✓ Loaded specs, params, and model class")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nTrying to create sample data for testing...")
        # Create minimal sample for testing
        return

    # Try to load actual wealth data
    bld = project_root / "bld"
    wealth_data_paths = [
        bld / "moments" / "wealth_data.csv",
        bld / "data" / "soep_wealth.csv",
    ]

    df_wealth = None
    for path in wealth_data_paths:
        if path.exists():
            try:
                df_wealth = pd.read_csv(path)
                print(f"✓ Loaded wealth data from {path}")
                break
            except Exception as e:
                print(f"✗ Could not load {path}: {e}")

    if df_wealth is None:
        print("\n⚠ Could not find wealth data file.")
        print("Please ensure wealth data exists or modify paths in the script.")
        print("\nCreating synthetic data for demonstration...")

        # Create synthetic data for demonstration
        np.random.seed(42)
        n_obs = 1000
        df_wealth = pd.DataFrame(
            {
                "pid": np.repeat(range(100), 10),
                "age": np.tile(range(40, 50), 100),
                "choice": np.random.choice([0, 1, 2, 3, 4], n_obs),
                "experience": np.random.uniform(0.3, 0.9, n_obs),
                "wealth": np.random.uniform(10000, 200000, n_obs),
                "sex": 0,  # Female
                "education": np.random.choice([0, 1], n_obs),
                "partner_state": np.random.choice([0, 1, 2], n_obs),
            }
        )
        print("✓ Created synthetic data for testing")

    # Identify fresh retirees
    df_wealth = identify_fresh_retirees(df_wealth)

    # Analyze experience scaling
    results_df = analyze_experience_scaling(df_wealth, specs, model_class)

    if results_df is not None and len(results_df) > 0:
        print("\n" + "=" * 80)
        print("Analysis Results")
        print("=" * 80)

        print("\nSummary Statistics:")
        print(f"  Number of fresh retirees: {len(results_df)}")
        print("\nExperience scaling factors:")
        print(f"  Mean max_exp_working: {results_df['max_exp_working'].mean():.2f}")
        print(f"  Mean max_pp_retirement: {results_df['max_pp_retirement'].mean():.2f}")
        print(
            f"  Ratio (retirement/working): {results_df['max_pp_retirement'].mean() / results_df['max_exp_working'].mean():.2f}"
        )

        print("\nExperience years calculations:")
        print(
            f"  If on working scale - Mean: {results_df['exp_years_if_working_scale'].mean():.2f}"
        )
        print(
            f"  If on retirement scale - Mean: {results_df['exp_years_if_retirement_scale'].mean():.2f}"
        )
        print(
            f"  Actually used (is_retired=True) - Mean: {results_df['exp_years_actual_used'].mean():.2f}"
        )

        print("\nPension points:")
        print(
            f"  Correct (from working exp) - Mean: {results_df['pension_points_correct'].mean():.2f}"
        )
        print(
            f"  Used (from actual) - Mean: {results_df['pension_points_used'].mean():.2f}"
        )
        print(f"  Difference - Mean: {results_df['difference'].mean():.2f}")
        print(f"  Ratio (used/correct) - Mean: {results_df['ratio'].mean():.2f}")

        # Check if there's a systematic issue
        if results_df["ratio"].mean() > 1.1:
            print(
                "\n⚠ WARNING: Pension points being used are significantly higher than correct!"
            )
            print("  This suggests experience is being incorrectly scaled.")
        elif results_df["ratio"].mean() < 0.9:
            print(
                "\n⚠ WARNING: Pension points being used are significantly lower than correct!"
            )
        else:
            print("\n✓ Pension points appear to be calculated correctly.")

        # Save detailed results
        output_path = bld / "moments" / "retirement_spike_diagnostic.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Detailed results saved to: {output_path}")

        # Show sample of problematic cases
        if results_df["ratio"].max() > 1.1:
            print("\nSample of cases with high ratio (potential issues):")
            high_ratio = results_df[results_df["ratio"] > 1.1].head(5)
            print(
                high_ratio[
                    [
                        "age",
                        "experience_norm",
                        "exp_years_actual_used",
                        "pension_points_correct",
                        "pension_points_used",
                        "ratio",
                    ]
                ].to_string()
            )

    print("\n" + "=" * 80)
    print("Diagnostic complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
