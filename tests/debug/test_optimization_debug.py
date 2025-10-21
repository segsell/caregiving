"""Test function to debug the full optimization process and identify overflow issues."""

import pickle
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from caregiving.config import BLD
from caregiving.estimation.estimation_setup import (
    estimate_model,
    get_msm_optimization_function,
    simulate_moments,
)
from caregiving.simulation.simulate_moments import simulate_moments_pandas

# Constants
LARGE_DISUTILITY_THRESHOLD = 50


def test_optimization_with_extreme_parameters():  # noqa: PLR0915
    """Test optimization with various parameter values to identify overflow issues."""

    print("=== Optimization Debug Test with Extreme Parameters ===\n")

    # Load required data
    print("1. Loading data...")
    with open(BLD / "solve_and_simulate" / "simulated_data.pkl", "rb") as f:
        sim_df = pickle.load(f)

    with open(BLD / "moments" / "moments_full.csv") as f:
        empirical_moments = np.array(pd.read_csv(f, index_col=0).squeeze())

    with open(BLD / "moments" / "variances_full.csv") as f:
        empirical_variances = np.array(pd.read_csv(f, index_col=0).squeeze())

    with open(BLD / "model" / "params" / "start_params_model.yaml", "rb") as f:
        start_params = yaml.safe_load(f)

    with open(BLD / "model" / "options.pkl", "rb") as f:
        options = pickle.load(f)

    print(f"Loaded data: moments={empirical_moments.shape}, params={len(start_params)}")

    # Fix weights computation
    empirical_variances_reg = empirical_variances.copy()
    MACHINE_ZERO = 1e-12
    close_to_zero = empirical_variances_reg < MACHINE_ZERO
    empirical_variances_reg[close_to_zero] = 1e-6
    weight_elements = 1 / empirical_variances_reg
    weight_elements = np.sqrt(weight_elements)
    diagonal_weights = np.diag(weight_elements)
    _ = np.linalg.cholesky(diagonal_weights)

    print(
        f"Fixed weights: {diagonal_weights.shape}, zero vars: {np.sum(close_to_zero)}"
    )

    # Create mock functions
    def mock_simulate_moments(params):
        """Mock function that returns moments from pre-simulated data."""
        try:
            simulated_moments = simulate_moments_pandas(sim_df, options)
            moments_array = np.asarray(simulated_moments.to_numpy())
            out = np.nan_to_num(moments_array, nan=0.0, posinf=0.0, neginf=0.0)
            return out
        except Exception as e:
            print(f"Error in mock_simulate_moments: {e}")
            return np.zeros_like(empirical_moments)

    # Test different parameter scenarios
    param_scenarios = [
        ("Original parameters", start_params),
        (
            "Small parameters",
            {k: 0.1 if v > 0 else -0.1 for k, v in start_params.items()},
        ),
        (
            "Large parameters",
            {k: 5.0 if v > 0 else -5.0 for k, v in start_params.items()},
        ),
        (
            "Extreme parameters",
            {k: 50.0 if v > 0 else -50.0 for k, v in start_params.items()},
        ),
    ]

    for scenario_name, params in param_scenarios:
        print(f"\n2. Testing scenario: {scenario_name}")
        print(
            f"   Parameter range: "
            f"[{min(params.values()):.3f}, {max(params.values()):.3f}]"
        )

        try:
            # Convert to array
            params_array = np.array([params[key] for key in sorted(params.keys())])

            # Test criterion function
            opt_func = get_msm_optimization_function(
                simulate_moments=mock_simulate_moments,
                empirical_moments=empirical_moments,
                weights=diagonal_weights,
                cholesky=True,
                relative_deviations=False,
                least_squares=True,
            )

            result = opt_func(params_array)

            print(f"Criterion result shape: {result.shape}")
            print(
                f"Criterion range: [{np.nanmin(result):.6f}, {np.nanmax(result):.6f}]"
            )

            # Check for problematic values
            has_nan = np.any(np.isnan(result))
            has_inf = np.any(np.isinf(result))
            LARGE_VALUE_THRESHOLD = 1e10
            has_large = np.any(np.abs(result) > LARGE_VALUE_THRESHOLD)

            if has_nan:
                print(f"WARNING: Found {np.sum(np.isnan(result))} NaN values")
            if has_inf:
                print(f"WARNING: Found {np.sum(np.isinf(result))} infinite values")
            if has_large:
                print(
                    f"WARNING: Found "
                    f"{np.sum(np.abs(result) > LARGE_VALUE_THRESHOLD)} "
                    f"very large values"
                )

            # Test if result would cause issues in optimization
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                print("ERROR: This scenario would cause optimization failure")
            else:
                print("This scenario is numerically stable")

        except Exception as e:
            print(f"ERROR: Error in scenario: {e}")
            import traceback

            traceback.print_exc()


def test_parameter_bounds_issues():  # noqa: PLR0912, PLR0915
    """Test if parameter bounds are causing the optimization issues."""

    print("\n\n=== Parameter Bounds Debug Test ===\n")

    # Load bounds
    print("1. Loading parameter bounds...")

    try:
        with open(BLD / "model" / "params" / "start_params_model.yaml", "rb") as f:
            start_params = yaml.safe_load(f)

        # Check if bounds files exist
        lower_bounds_path = BLD / "model" / "params" / "lower_bounds_model.yaml"
        upper_bounds_path = BLD / "model" / "params" / "upper_bounds_model.yaml"

        if lower_bounds_path.exists():
            with open(lower_bounds_path, "rb") as f:
                lower_bounds = yaml.safe_load(f)
            print(f"Lower bounds loaded: {len(lower_bounds)} parameters")
        else:
            print(f"WARNING: Lower bounds file not found: {lower_bounds_path}")
            lower_bounds = {}

        if upper_bounds_path.exists():
            with open(upper_bounds_path, "rb") as f:
                upper_bounds = yaml.safe_load(f)
            print(f"Upper bounds loaded: {len(upper_bounds)} parameters")
        else:
            print(f"WARNING: Upper bounds file not found: {upper_bounds_path}")
            upper_bounds = {}

        # Check for problematic bounds
        print("\n2. Analyzing parameter bounds...")

        problematic_params = []

        for param_name, start_val in start_params.items():
            lower_val = lower_bounds.get(param_name, -np.inf)
            upper_val = upper_bounds.get(param_name, np.inf)

            # Check if start value is at or near bounds
            MACHINE_EPSILON = 1e-10
            if abs(start_val - lower_val) < MACHINE_EPSILON:
                problematic_params.append(
                    (param_name, "at_lower_bound", start_val, lower_val, upper_val)
                )
            elif abs(start_val - upper_val) < MACHINE_EPSILON:
                problematic_params.append(
                    (param_name, "at_upper_bound", start_val, lower_val, upper_val)
                )
            elif start_val < lower_val + 1e-6:
                problematic_params.append(
                    (param_name, "near_lower_bound", start_val, lower_val, upper_val)
                )
            elif start_val > upper_val - 1e-6:
                problematic_params.append(
                    (param_name, "near_upper_bound", start_val, lower_val, upper_val)
                )

            # Check for very small bounds
            if lower_val > -np.inf and lower_val < MACHINE_EPSILON:
                problematic_params.append(
                    (param_name, "very_small_lower", start_val, lower_val, upper_val)
                )
            if upper_val < np.inf and upper_val < MACHINE_EPSILON:
                problematic_params.append(
                    (param_name, "very_small_upper", start_val, lower_val, upper_val)
                )

        if problematic_params:
            print(
                f"WARNING: Found {len(problematic_params)} "
                f"potentially problematic parameters:"
            )
            for (
                param_name,
                issue,
                start_val,
                lower_val,
                upper_val,
            ) in problematic_params:
                print(
                    f"{param_name}: {issue} (start={start_val:.6f}, "
                    f"bounds=[{lower_val:.6f}, {upper_val:.6f}])"
                )
        else:
            print("No obviously problematic parameter bounds found")

        # Check for parameters that could cause numerical issues
        print("\n3. Checking for numerical issues...")

        numerical_issues = []

        for param_name, start_val in start_params.items():
            lower_val = lower_bounds.get(param_name, -np.inf)
            upper_val = upper_bounds.get(param_name, np.inf)

            # Check for parameters that could cause division by zero
            if param_name.startswith("rho") and (
                lower_val <= MACHINE_EPSILON or start_val <= MACHINE_EPSILON
            ):
                numerical_issues.append((param_name, "rho_near_zero", start_val))

            # Check for parameters that could cause overflow in exp()
            if "disutil" in param_name and (
                upper_val > LARGE_DISUTILITY_THRESHOLD
                or start_val > LARGE_DISUTILITY_THRESHOLD
            ):
                numerical_issues.append((param_name, "large_disutility", start_val))

            # Check for lambda parameters that could cause issues
            if param_name.startswith("lambda") and (
                lower_val <= MACHINE_EPSILON or start_val <= MACHINE_EPSILON
            ):
                numerical_issues.append((param_name, "lambda_near_zero", start_val))

        if numerical_issues:
            print(
                f"WARNING: Found {len(numerical_issues)} parameters "
                f"with potential numerical issues:"
            )
            for param_name, issue, start_val in numerical_issues:
                print(f"{param_name}: {issue} (value={start_val:.6f})")
        else:
            print("No obvious numerical issues in parameters")

    except Exception as e:
        print(f"ERROR: Error analyzing bounds: {e}")
        import traceback

        traceback.print_exc()
