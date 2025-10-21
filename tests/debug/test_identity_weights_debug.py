"""Test optimization with identity weights to isolate the real issue."""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from caregiving.config import BLD, SRC
from caregiving.estimation.estimation_setup import (
    get_msm_optimization_function,
    simulate_moments,
)


def test_with_identity_weights():  # noqa: PLR0915
    """Test optimization with identity weights to isolate the real issue."""

    print("=== Identity Weights Debug Test ===\n")

    # Load required data
    print("1. Loading data...")
    with open(BLD / "solve_and_simulate" / "simulated_data.pkl", "rb") as f:
        sim_df = pickle.load(f)

    with open(BLD / "moments" / "moments_full.csv") as f:
        empirical_moments = np.array(pd.read_csv(f, index_col=0).squeeze())

    with open(BLD / "model" / "params" / "start_params_model.yaml", "rb") as f:
        start_params = yaml.safe_load(f)

    with open(BLD / "model" / "options.pkl", "rb") as f:
        options = pickle.load(f)

    print(
        f"   Loaded data: moments={empirical_moments.shape}, "
        f"params={len(start_params)}"
    )

    # Use identity weights (no variance issues)
    identity_weights = np.identity(empirical_moments.shape[0])
    print(f"Using identity weights: {identity_weights.shape}")

    # Create mock simulate_moments function
    def mock_simulate_moments(params):
        """Mock function that returns moments from pre-simulated data."""
        try:
            from caregiving.simulation.simulate_moments import simulate_moments_pandas

            simulated_moments = simulate_moments_pandas(sim_df, options)
            moments_array = np.asarray(simulated_moments.to_numpy())
            out = np.nan_to_num(moments_array, nan=0.0, posinf=0.0, neginf=0.0)
            return out
        except Exception as e:
            print(f"Error in mock_simulate_moments: {e}")
            return np.zeros_like(empirical_moments)

    # Test with different parameter perturbations
    print("\n2. Testing with parameter perturbations...")

    base_params = start_params.copy()
    param_names = sorted(base_params.keys())

    # Test scenarios
    test_scenarios = [
        ("Original parameters", base_params),
        ("Small perturbation", {k: v + 0.01 for k, v in base_params.items()}),
        ("Larger perturbation", {k: v + 0.1 for k, v in base_params.items()}),
        ("Negative perturbation", {k: v - 0.1 for k, v in base_params.items()}),
    ]

    for scenario_name, params in test_scenarios:
        print(f"\n   Testing: {scenario_name}")

        try:
            # Convert to array
            params_array = np.array([params[key] for key in param_names])

            # Test the optimization function
            opt_func = get_msm_optimization_function(
                simulate_moments=mock_simulate_moments,
                empirical_moments=empirical_moments,
                weights=identity_weights,
                cholesky=False,  # No Cholesky with identity weights
                relative_deviations=False,
                least_squares=True,
            )

            result = opt_func(params_array)

            print(f"Result shape: {result.shape}")
            print(
                f"   Result range: [{np.nanmin(result):.6f}, {np.nanmax(result):.6f}]"
            )

            # Check for issues
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
                    f"   WARNING: Found "
                    f"{np.sum(np.abs(result) > LARGE_VALUE_THRESHOLD)} "
                    f"very large values"
                )

            if not (has_nan or has_inf or has_large):
                print("Scenario is numerically stable")
            else:
                print("ERROR: Scenario has numerical issues")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()


def test_actual_simulation_process():  # noqa: PLR0915
    """Test the actual simulation process to find where overflow occurs."""

    print("\n\n=== Actual Simulation Process Debug Test ===\n")

    # Load required data
    print("1. Loading all required data...")

    try:
        # Load start parameters
        with open(BLD / "model" / "params" / "start_params_model.yaml", "rb") as f:
            start_params = yaml.safe_load(f)

        # Load options
        with open(BLD / "model" / "options.pkl", "rb") as f:
            options = pickle.load(f)

        # Load initial conditions
        with open(BLD / "model" / "initial_conditions" / "states.pkl", "rb") as f:
            initial_states = pickle.load(f)

        wealth_agents = np.array(
            pd.read_csv(
                BLD / "model" / "initial_conditions" / "wealth.csv", usecols=["wealth"]
            ).squeeze()
        )

        # Load model for simulation
        with open(BLD / "model" / "model_for_solution.pkl", "rb") as f:
            model_for_simulation = pickle.load(f)

        print("Loaded all data")

    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        return

    # Create a mock solve function that might cause issues
    def mock_solve_func(params):
        """Mock solve function that could cause overflow."""
        print(f"Mock solve called with {len(params)} parameters")

        # Check for problematic parameter values
        LARGE_PARAM_THRESHOLD = 100
        EXTREME_PARAM_THRESHOLD = 1000
        has_large = np.any(np.abs(params) > LARGE_PARAM_THRESHOLD)
        has_extreme = np.any(np.abs(params) > EXTREME_PARAM_THRESHOLD)
        has_inf = np.any(np.isinf(params))
        has_nan = np.any(np.isnan(params))

        if has_inf:
            print("WARNING: Parameters contain infinite values")
        if has_nan:
            print("WARNING: Parameters contain NaN values")
        if has_extreme:
            print(
                f"   WARNING: Parameters contain extreme values: "
                f"{np.max(np.abs(params))}"
            )
        elif has_large:
            print(
                f"   WARNING: Parameters contain large values: "
                f"{np.max(np.abs(params))}"
            )

        # Return dummy solution
        n_states = 1000
        n_choices = 10

        return (
            np.random.randn(n_states, n_choices),  # value function
            np.random.randint(0, n_choices, (n_states,)),  # policy
            np.random.randn(n_states, n_choices),  # endogenous grid
        )

    # Test simulate_moments function with actual parameters
    print("\n2. Testing simulate_moments function...")

    try:
        # Convert start_params to array
        param_names = sorted(start_params.keys())
        params_array = np.array([start_params[key] for key in param_names])

        print(f"Parameter array shape: {params_array.shape}")
        print(
            f"   Parameter range: "
            f"[{np.min(params_array):.6f}, {np.max(params_array):.6f}]"
        )

        # Check for problematic parameters
        problematic_params = []
        LARGE_PARAM_THRESHOLD = 100
        for _, (name, value) in enumerate(zip(param_names, params_array, strict=False)):
            if np.isnan(value) or np.isinf(value):
                problematic_params.append((name, "nan_or_inf", value))
            elif abs(value) > LARGE_PARAM_THRESHOLD:
                problematic_params.append((name, "very_large", value))

        if problematic_params:
            print(
                f"   WARNING: Found {len(problematic_params)} problematic parameters:"
            )
            for name, issue, value in problematic_params:
                print(f"{name}: {issue} = {value}")
        else:
            print("No obviously problematic parameters")

        # Test the actual simulate_moments function
        from caregiving.simulation.simulate import simulate_scenario
        from caregiving.simulation.simulate_moments import simulate_moments_pandas

        def mock_simulate_scenario(
            model, solution, initial_states, wealth_agents, params, options, seed
        ):
            """Mock simulate scenario that uses pre-computed data."""
            with open(BLD / "solve_and_simulate" / "simulated_data.pkl", "rb") as f:
                return pickle.load(f)

        # Call simulate_moments
        simulated_moments = simulate_moments(
            params=params_array,
            solve_func=mock_solve_func,
            initial_states=initial_states,
            wealth_agents=wealth_agents,
            model_for_simulation=model_for_simulation,
            options=options,
            fixed_seed=options["model_params"]["seed"],
            seed_generator=None,
            simulate_scenario_func=mock_simulate_scenario,
            simulate_moments_func=simulate_moments_pandas,
        )

        print(f"simulate_moments successful: shape={simulated_moments.shape}")
        print(
            f"   Moments range: "
            f"[{np.nanmin(simulated_moments):.6f}, {np.nanmax(simulated_moments):.6f}]"
        )

        # Check for issues in the result
        has_nan = np.any(np.isnan(simulated_moments))
        has_inf = np.any(np.isinf(simulated_moments))
        LARGE_VALUE_THRESHOLD = 1e10
        has_large = np.any(np.abs(simulated_moments) > LARGE_VALUE_THRESHOLD)

        if has_nan:
            print(
                f"   WARNING: Found {np.sum(np.isnan(simulated_moments))} "
                f"NaN values in moments"
            )
        if has_inf:
            print(
                f"   WARNING: Found {np.sum(np.isinf(simulated_moments))} "
                f"infinite values in moments"
            )
        if has_large:
            print(
                f"   WARNING: Found "
                f"{np.sum(np.abs(simulated_moments) > LARGE_VALUE_THRESHOLD)} "
                f"very large values in moments"
            )

    except Exception as e:
        print(f"ERROR: Error in simulate_moments: {e}")
        import traceback

        traceback.print_exc()


def test_moments_computation():  # noqa: PLR0912, PLR0915
    """Test the moments computation specifically."""

    print("\n\n=== Moments Computation Debug Test ===\n")

    # Load simulated data
    print("1. Loading simulated data...")
    with open(BLD / "solve_and_simulate" / "simulated_data.pkl", "rb") as f:
        sim_df = pickle.load(f)

    with open(BLD / "model" / "options.pkl", "rb") as f:
        options = pickle.load(f)

    print(f"Simulated data shape: {sim_df.shape}")
    print(f"Simulated data columns: {list(sim_df.columns)[:10]}...")

    # Check for issues in simulated data
    print("\n2. Checking simulated data for issues...")

    # Check for infinite or NaN values in key columns
    problematic_columns = []
    for col in sim_df.columns:
        if sim_df[col].dtype in ("float64", "int64"):
            has_inf = np.any(np.isinf(sim_df[col]))
            has_nan = np.any(np.isnan(sim_df[col]))
            if has_inf or has_nan:
                problematic_columns.append((col, has_inf, has_nan))

    if problematic_columns:
        print(f"WARNING: Found {len(problematic_columns)} columns with issues:")
        for col, has_inf, has_nan in problematic_columns:
            issues = []
            if has_inf:
                issues.append("inf")
            if has_nan:
                issues.append("nan")
            print(f"{col}: {', '.join(issues)}")
    else:
        print("No issues found in simulated data")

    # Test moments computation
    print("\n3. Testing moments computation...")

    try:
        from caregiving.simulation.simulate_moments import simulate_moments_pandas

        moments_df = simulate_moments_pandas(sim_df, options)
        moments_array = np.asarray(moments_df.to_numpy())

        print(f"Moments computed: shape={moments_array.shape}")
        print(
            f"   Moments range: "
            f"[{np.nanmin(moments_array):.6f}, {np.nanmax(moments_array):.6f}]"
        )

        # Check for issues in moments
        has_nan = np.any(np.isnan(moments_array))
        has_inf = np.any(np.isinf(moments_array))
        LARGE_VALUE_THRESHOLD = 1e10
        has_large = np.any(np.abs(moments_array) > LARGE_VALUE_THRESHOLD)

        if has_nan:
            print(
                f"   WARNING: Found {np.sum(np.isnan(moments_array))} "
                f"NaN values in moments"
            )
            # Show which moments are NaN
            nan_indices = np.where(np.isnan(moments_array))[0]
            print(f"NaN moment indices: {nan_indices[:10]}...")  # Show first 10

        if has_inf:
            print(
                f"   WARNING: Found {np.sum(np.isinf(moments_array))} "
                f"infinite values in moments"
            )
            # Show which moments are infinite
            inf_indices = np.where(np.isinf(moments_array))[0]
            print(f"Infinite moment indices: {inf_indices[:10]}...")  # Show first 10

        if has_large:
            print(
                f"   WARNING: Found "
                f"{np.sum(np.abs(moments_array) > LARGE_VALUE_THRESHOLD)} "
                f"very large values in moments"
            )
            # Show which moments are very large
            large_indices = np.where(np.abs(moments_array) > LARGE_VALUE_THRESHOLD)[0]
            print(f"Large moment indices: {large_indices[:10]}...")  # Show first 10
            print(f"Largest moment values: {moments_array[large_indices[:5]]}")

        if not (has_nan or has_inf or has_large):
            print("Moments computation is numerically stable")

    except Exception as e:
        print(f"ERROR: Error in moments computation: {e}")
        import traceback

        traceback.print_exc()
