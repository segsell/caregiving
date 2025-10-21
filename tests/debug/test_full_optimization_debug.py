"""Test the full optimization process to identify overflow issues."""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from caregiving.config import BLD, SRC
from caregiving.estimation.estimation_setup import (
    estimate_model,
    get_msm_optimization_function,
    simulate_moments,
)
from caregiving.simulation.simulate_moments import simulate_moments_pandas


def test_full_optimization_process():  # noqa: PLR0912, PLR0915
    """Test the complete optimization process with actual bounds and constraints."""

    print("=== Full Optimization Process Debug Test ===\n")

    # Load all required data
    print("1. Loading all required data...")

    # Load bounds from source (not built files)
    lower_bounds_path = (
        SRC / "estimation" / "start_params_and_bounds" / "lower_bounds.yaml"
    )
    upper_bounds_path = (
        SRC / "estimation" / "start_params_and_bounds" / "upper_bounds.yaml"
    )
    start_params_path = (
        SRC / "estimation" / "start_params_and_bounds" / "start_params.yaml"
    )

    try:
        with open(lower_bounds_path, "rb") as f:
            lower_bounds = yaml.safe_load(f)
        print(f"Lower bounds loaded: {len(lower_bounds)} parameters")

        with open(upper_bounds_path, "rb") as f:
            upper_bounds = yaml.safe_load(f)
        print(f"Upper bounds loaded: {len(upper_bounds)} parameters")

        with open(start_params_path, "rb") as f:
            start_params = yaml.safe_load(f)
        print(f"Start parameters loaded: {len(start_params)} parameters")

    except Exception as e:
        print(f"ERROR: Error loading parameter files: {e}")
        return

    # Load empirical data
    try:
        empirical_moments = np.array(
            pd.read_csv(BLD / "moments" / "moments_full.csv", index_col=0).squeeze()
        )
        empirical_variances = np.array(
            pd.read_csv(BLD / "moments" / "variances_full.csv", index_col=0).squeeze()
        )
        print(f"Empirical data loaded: moments={empirical_moments.shape}")
    except Exception as e:
        print(f"ERROR: Error loading empirical data: {e}")
        return

    # Load other required data
    try:
        with open(BLD / "model" / "options.pkl", "rb") as f:
            options = pickle.load(f)
        print("Model options loaded")
    except Exception as e:
        print(f"ERROR: Error loading options: {e}")
        return

    # Analyze the bounds
    print("\n2. Analyzing parameter bounds...")

    # Find parameters that are in start_params but not in bounds
    missing_in_bounds = []
    for param_name in start_params.keys():
        if param_name not in lower_bounds:
            missing_in_bounds.append((param_name, "missing_lower"))
        if param_name not in upper_bounds:
            missing_in_bounds.append((param_name, "missing_upper"))

    if missing_in_bounds:
        print(f"WARNING: Found {len(missing_in_bounds)} missing bounds:")
        for param_name, bound_type in missing_in_bounds:
            print(f"{param_name}: {bound_type}")

    # Check for problematic bounds
    problematic_bounds = []
    for param_name, start_val in start_params.items():
        lower_val = lower_bounds.get(param_name, -np.inf)
        upper_val = upper_bounds.get(param_name, np.inf)

        # Check if bounds are too restrictive
        if lower_val >= start_val:
            problematic_bounds.append(
                (param_name, "lower_bound_too_high", start_val, lower_val)
            )
        if upper_val <= start_val:
            problematic_bounds.append(
                (param_name, "upper_bound_too_low", start_val, upper_val)
            )

        # Check for extremely small bounds that could cause numerical issues
        MACHINE_EPSILON = 1e-10
        if lower_val > -np.inf and lower_val < MACHINE_EPSILON:
            problematic_bounds.append(
                (param_name, "extremely_small_lower", start_val, lower_val)
            )
        if upper_val < np.inf and upper_val < MACHINE_EPSILON:
            problematic_bounds.append(
                (param_name, "extremely_small_upper", start_val, upper_val)
            )

        # Check if bounds are too close together
        if (
            upper_val < np.inf
            and lower_val > -np.inf
            and (upper_val - lower_val) < MACHINE_EPSILON
        ):
            problematic_bounds.append(
                (param_name, "bounds_too_close", start_val, lower_val, upper_val)
            )

    if problematic_bounds:
        print(f"WARNING: Found {len(problematic_bounds)} problematic bounds:")
        for item in problematic_bounds:
            ITEM_LENGTH_4 = 4
            if len(item) == ITEM_LENGTH_4:
                param_name, issue, start_val, bound_val = item
                print(
                    f"      {param_name}: {issue} "
                    f"(start={start_val:.6f}, bound={bound_val:.6f})"
                )
            else:
                param_name, issue, start_val, lower_val, upper_val = item
                print(
                    f"      {param_name}: {issue} "
                    f"(start={start_val:.6f}, "
                    f"bounds=[{lower_val:.6f}, {upper_val:.6f}])"
                )
    else:
        print("No obviously problematic bounds found")

    # Test the weights computation that's causing issues
    print("\n3. Testing weights computation...")

    try:
        # This is the problematic computation from estimation_setup.py
        empirical_variances_reg = empirical_variances.copy()
        MACHINE_ZERO = 1e-12
        close_to_zero = empirical_variances_reg < MACHINE_ZERO
        print(f"Zero variances found: {np.sum(close_to_zero)}")

        # This is where the overflow happens - division by zero
        weight_elements = 1 / empirical_variances_reg
        weight_elements[close_to_zero] = 0.0
        weight_elements = np.sqrt(weight_elements)

        print(
            f"   Weight elements range: "
            f"[{np.nanmin(weight_elements):.6f}, {np.nanmax(weight_elements):.6f}]"
        )

        # This is where the Cholesky decomposition fails
        diagonal_weights = np.diag(weight_elements)

        # Check if matrix is positive definite
        eigenvals = np.linalg.eigvals(diagonal_weights)
        min_eigenval = np.min(eigenvals)
        print(f"Minimum eigenvalue: {min_eigenval:.6f}")

        if min_eigenval <= 0:
            print(
                "   WARNING: Matrix is not positive definite - "
                "this will cause Cholesky to fail"
            )
        else:
            print("Matrix is positive definite")

        # Try Cholesky decomposition
        try:
            _ = np.linalg.cholesky(diagonal_weights)
            print("Cholesky decomposition successful")
        except np.linalg.LinAlgError as e:
            print(f"ERROR: Cholesky decomposition failed: {e}")
            print("This is likely the source of the optimization failure")

    except Exception as e:
        print(f"ERROR: Error in weights computation: {e}")
        import traceback

        traceback.print_exc()

    # Test with a simple optimization setup
    print("\n4. Testing simple optimization setup...")

    try:
        # Create a simple mock solve function
        def mock_solve_func(params):
            """Mock solve function that returns dummy solution."""
            # Return dummy arrays with reasonable shapes
            n_states = 1000
            n_choices = 10

            return (
                np.random.randn(n_states, n_choices),  # value function
                np.random.randint(0, n_choices, (n_states,)),  # policy
                np.random.randn(n_states, n_choices),  # endogenous grid
            )

        # Create a simple mock simulate function
        def mock_simulate_func(
            model, solution, initial_states, wealth_agents, params, options, seed
        ):
            """Mock simulate function that returns the pre-computed data."""
            with open(BLD / "solve_and_simulate" / "simulated_data.pkl", "rb") as f:
                return pickle.load(f)

        # Create a simple mock moments function
        def mock_moments_func(sim_df, options):
            """Mock moments function that returns pre-computed moments."""
            return simulate_moments_pandas(sim_df, options)

        # Test the simulate_moments function
        print("Testing simulate_moments function...")

        # Load required data for simulate_moments
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

        # Test simulate_moments with start parameters
        params_array = np.array(
            [start_params[key] for key in sorted(start_params.keys())]
        )

        try:
            simulated_moments = simulate_moments(
                params=params_array,
                solve_func=mock_solve_func,
                initial_states=initial_states,
                wealth_agents=wealth_agents,
                model_for_simulation=model_for_simulation,
                options=options,
                fixed_seed=options["model_params"]["seed"],
                seed_generator=None,
                simulate_scenario_func=mock_simulate_func,
                simulate_moments_func=mock_moments_func,
            )

            print(f"simulate_moments successful: shape={simulated_moments.shape}")
            print(
                f"   Moments range: "
                f"[{np.nanmin(simulated_moments):.6f}, "
                f"{np.nanmax(simulated_moments):.6f}]"
            )

        except Exception as e:
            print(f"ERROR: simulate_moments failed: {e}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"ERROR: Error in optimization setup test: {e}")
        import traceback

        traceback.print_exc()
