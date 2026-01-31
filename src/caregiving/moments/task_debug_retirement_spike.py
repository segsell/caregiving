"""Lightweight diagnostic task to investigate retirement wealth spike.

This task loads wealth data and checks how experience scaling affects
fresh retirees when adjusting wealth.

Can be run as:
    python -m caregiving.moments.task_debug_retirement_spike
Or as a pytask task.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Annotated

try:
    from pytask import Product
    from build import BLD
    PYTASK_AVAILABLE = True
except ImportError:
    PYTASK_AVAILABLE = False
    # Define BLD for standalone execution
    BLD = Path(__file__).parent.parent.parent.parent / "bld"
from caregiving.moments.task_create_soep_moments import (
    create_df_wealth,
    SEX,
)
from caregiving.model.shared import RETIREMENT
from caregiving.model.experience_baseline_model import construct_experience_years
from caregiving.model.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)


def identify_fresh_retirees(df, person_id_col="pid", age_col="age", choice_col="choice"):
    """Identify fresh retirees (first period of retirement)."""
    df = df.copy()
    df = df.sort_values([person_id_col, age_col]).reset_index(drop=True)
    
    retirement_values = np.asarray(RETIREMENT).ravel().tolist()
    
    # Current period retirement status
    df["is_retired_now"] = df[choice_col].isin(retirement_values)
    
    # Previous period retirement status
    df["lagged_choice"] = df.groupby(person_id_col)[choice_col].shift(1)
    df["is_retired_prev"] = (
        df["lagged_choice"].isin(retirement_values).fillna(False)
    )
    
    # Fresh retirees: retired now but not in previous period
    df["is_fresh_retiree"] = df["is_retired_now"] & (~df["is_retired_prev"])
    
    return df


def analyze_experience_scaling(df, specs):
    """Analyze how experience scaling affects fresh retirees."""
    results = []
    
    fresh_retirees = df[df["is_fresh_retiree"]].copy()
    
    if len(fresh_retirees) == 0:
        print("No fresh retirees found in the data.")
        return None
    
    print(f"\nFound {len(fresh_retirees)} fresh retirees")
    print(f"Age range: {fresh_retirees['age'].min()} to {fresh_retirees['age'].max()}\n")
    
    for idx, row in fresh_retirees.iterrows():
        age = row["age"]
        period = age - specs["start_age"]
        experience_norm = row["experience"]
        lagged_choice_val = row["lagged_choice"]
        if pd.isna(lagged_choice_val):
            lagged_choice = 0  # Default to first choice if missing
        else:
            lagged_choice = int(lagged_choice_val)
        
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
        partner_state_val = np.array(row.get("partner_state", 0))
        education_val = np.array(row.get("education", 0))
        
        pension_points_correct = calc_pension_points_for_experience(
            period=period,
            experience_years=exp_years_if_working_scale,
            sex=SEX,
            partner_state=partner_state_val,
            education=education_val,
            model_specs=specs,
        )
        
        # What budget_constraint would use (incorrect if experience is on wrong scale)
        pension_points_used = exp_years_actual  # This is what gets passed to calc_pensions_after_ssc
        
        results.append({
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
            "ratio": pension_points_used / pension_points_correct if pension_points_correct > 0 else np.nan,
        })
    
    return pd.DataFrame(results)


def debug_retirement_spike(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_model: Path = BLD / "model" / "model.pkl",
    path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
    path_to_main_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_output: Path = BLD / "moments" / "retirement_spike_diagnostic.csv",
) -> None:
    """Diagnostic task to investigate retirement wealth spike."""
    import yaml
    import dcegm
    from caregiving.model.state_space import create_state_space_functions
    from caregiving.model.utility.utility_functions_additive import create_utility_functions
    from caregiving.model.utility.bequest_utility import (
        create_final_period_utility_functions,
    )
    from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
    from caregiving.model.taste_shocks import shock_function_dict
    from caregiving.model.task_specify_model import create_stochastic_states_transitions
    
    print("=" * 80)
    print("Retirement Wealth Spike Diagnostic")
    print("=" * 80)
    
    # Load data
    specs = pickle.load(path_to_specs.open("rb"))
    params = yaml.safe_load(path_to_params.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))
    
    model_class = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_model,
    )
    
    print("✓ Loaded specs, params, and model class")
    
    # Load wealth data
    df_full = pd.read_csv(path_to_main_sample, index_col=[0])
    print("✓ Loaded main sample data")
    
    # Create wealth dataframe (without adjustment first to see raw data)
    df_wealth = create_df_wealth(
        df_full=df_full,
        specs=specs,
        params=params,
        model_class=model_class,
        adjust_wealth=False,  # Don't adjust yet, just get the data
    )
    print(f"✓ Created wealth dataframe with {len(df_wealth)} observations")
    
    # Identify fresh retirees
    df_wealth = identify_fresh_retirees(df_wealth)
    
    # Analyze experience scaling
    results_df = analyze_experience_scaling(df_wealth, specs)
    
    if results_df is not None and len(results_df) > 0:
        print("\n" + "=" * 80)
        print("Analysis Results")
        print("=" * 80)
        
        print("\nSummary Statistics:")
        print(f"  Number of fresh retirees: {len(results_df)}")
        print(f"\nExperience scaling factors:")
        print(f"  Mean max_exp_working: {results_df['max_exp_working'].mean():.2f}")
        print(f"  Mean max_pp_retirement: {results_df['max_pp_retirement'].mean():.2f}")
        ratio_scales = results_df['max_pp_retirement'].mean() / results_df['max_exp_working'].mean()
        print(f"  Ratio (retirement/working): {ratio_scales:.2f}")
        
        print(f"\nExperience years calculations:")
        print(f"  If on working scale - Mean: {results_df['exp_years_if_working_scale'].mean():.2f}")
        print(f"  If on retirement scale - Mean: {results_df['exp_years_if_retirement_scale'].mean():.2f}")
        print(f"  Actually used (is_retired=True) - Mean: {results_df['exp_years_actual_used'].mean():.2f}")
        
        print(f"\nPension points:")
        print(f"  Correct (from working exp) - Mean: {results_df['pension_points_correct'].mean():.2f}")
        print(f"  Used (from actual) - Mean: {results_df['pension_points_used'].mean():.2f}")
        print(f"  Difference - Mean: {results_df['difference'].mean():.2f}")
        mean_ratio = results_df['ratio'].mean()
        print(f"  Ratio (used/correct) - Mean: {mean_ratio:.2f}")
        
        # Check if there's a systematic issue
        if mean_ratio > 1.1:
            print("\n⚠ WARNING: Pension points being used are significantly higher than correct!")
            print("  This suggests experience is being incorrectly scaled.")
            print("  The issue: Experience normalized value is being rescaled with retirement")
            print("  scale even though it might still be on working scale.")
        elif mean_ratio < 0.9:
            print("\n⚠ WARNING: Pension points being used are significantly lower than correct!")
        else:
            print("\n✓ Pension points appear to be calculated correctly.")
        
        # Save detailed results
        path_to_output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(path_to_output, index=False)
        print(f"\n✓ Detailed results saved to: {path_to_output}")
        
        # Show sample of problematic cases
        if results_df['ratio'].max() > 1.1:
            print("\nSample of cases with high ratio (potential issues):")
            high_ratio = results_df[results_df['ratio'] > 1.1].head(5)
            print(high_ratio[['age', 'experience_norm', 'exp_years_actual_used', 
                            'pension_points_correct', 'pension_points_used', 'ratio']].to_string())
    
    print("\n" + "=" * 80)
    print("Diagnostic complete")
    print("=" * 80)


# Pytask task wrapper
if PYTASK_AVAILABLE:
    def task_debug_retirement_spike(
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_model_config: Path = BLD / "model" / "model_config.pkl",
        path_to_model: Path = BLD / "model" / "model.pkl",
        path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
        path_to_main_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
        path_to_output: Annotated[Path, Product] = BLD / "moments" / "retirement_spike_diagnostic.csv",
    ) -> None:
        """Pytask task wrapper."""
        debug_retirement_spike(
            path_to_specs=path_to_specs,
            path_to_model_config=path_to_model_config,
            path_to_model=path_to_model,
            path_to_params=path_to_params,
            path_to_main_sample=path_to_main_sample,
            path_to_output=path_to_output,
        )


# Allow running as standalone script
if __name__ == "__main__":
    debug_retirement_spike()
