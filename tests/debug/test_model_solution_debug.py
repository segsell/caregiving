"""Test the model solution and simulation process to find NaN values."""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.solve import get_solve_func_for_model

from caregiving.config import BLD

# Constants
LARGE_VALUE_THRESHOLD = 1e10


def test_model_solution():  # noqa: PLR0912, PLR0915
    """Test the model solution process to find where NaN values originate."""

    print("=== Model Solution Debug Test ===\n")

    # Load required data
    print("1. Loading model and parameters...")

    try:
        # Load start parameters
        with open(BLD / "model" / "params" / "start_params_model.yaml", "rb") as f:
            start_params = yaml.safe_load(f)

        # Load options
        with open(BLD / "model" / "options.pkl", "rb") as f:
            _ = pickle.load(f)

        # Load model for solution
        with open(BLD / "model" / "model_for_solution.pkl", "rb") as f:
            model_for_solution = pickle.load(f)

        print("Loaded model and parameters")

    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        return

    # Test the solve function
    print("\n2. Testing model solution...")

    try:
        # Get solve function
        solve_func = get_solve_func_for_model(model_for_solution)
        print("Solve function created")

        # Convert parameters to the format expected by solve function
        param_names = sorted(start_params.keys())
        params_array = np.array([start_params[key] for key in param_names])

        print(f"Parameter array shape: {params_array.shape}")
        print(
            f"   Parameter range: "
            f"[{np.min(params_array):.6f}, {np.max(params_array):.6f}]"
        )

        # Check for problematic parameters before solving
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

        # Solve the model
        print("Solving model...")
        value_func, policy_func, endog_grid = solve_func(params_array)

        print("Model solved successfully")
        print(f"Value function shape: {value_func.shape}")
        print(f"Policy function shape: {policy_func.shape}")
        print(f"Endogenous grid shape: {endog_grid.shape}")

        # Check for issues in solution
        print("\n3. Checking solution for issues...")

        # Check value function
        value_has_nan = np.any(np.isnan(value_func))
        value_has_inf = np.any(np.isinf(value_func))
        LARGE_VALUE_THRESHOLD = 1e10
        value_has_large = np.any(np.abs(value_func) > LARGE_VALUE_THRESHOLD)

        print("Value function:")
        print(f"Range: [{np.nanmin(value_func):.6f}, {np.nanmax(value_func):.6f}]")
        if value_has_nan:
            print(f"WARNING: Contains {np.sum(np.isnan(value_func))} NaN values")
        if value_has_inf:
            print(
                f"     WARNING: Contains {np.sum(np.isinf(value_func))} infinite values"
            )
        if value_has_large:
            print(
                f"     WARNING: Contains "
                f"{np.sum(np.abs(value_func) > LARGE_VALUE_THRESHOLD)} "
                f"very large values"
            )

        # Check policy function
        policy_has_nan = np.any(np.isnan(policy_func))
        policy_has_inf = np.any(np.isinf(policy_func))
        policy_range = (np.min(policy_func), np.max(policy_func))

        print("Policy function:")
        print(f"Range: {policy_range}")
        if policy_has_nan:
            print(f"WARNING: Contains {np.sum(np.isnan(policy_func))} NaN values")
        if policy_has_inf:
            print(
                f"     WARNING: Contains "
                f"{np.sum(np.isinf(policy_func))} infinite values"
            )

        # Check endogenous grid
        endog_has_nan = np.any(np.isnan(endog_grid))
        endog_has_inf = np.any(np.isinf(endog_grid))
        endog_has_large = np.any(np.abs(endog_grid) > LARGE_VALUE_THRESHOLD)

        print("Endogenous grid:")
        print(f"Range: [{np.nanmin(endog_grid):.6f}, {np.nanmax(endog_grid):.6f}]")
        if endog_has_nan:
            print(f"WARNING: Contains {np.sum(np.isnan(endog_grid))} NaN values")
        if endog_has_inf:
            print(
                f"     WARNING: Contains {np.sum(np.isinf(endog_grid))} infinite values"
            )
        if endog_has_large:
            print(
                f"     WARNING: Contains "
                f"{np.sum(np.abs(endog_grid) > LARGE_VALUE_THRESHOLD)} "
                f"very large values"
            )

        # Overall assessment
        if value_has_nan or policy_has_nan or endog_has_nan:
            print(
                "\n   ERROR: SOLUTION CONTAINS NaN VALUES - "
                "This is the source of the optimization failure"
            )
        elif value_has_inf or policy_has_inf or endog_has_inf:
            print(
                "\n   WARNING: SOLUTION CONTAINS INFINITE VALUES - "
                "This could cause optimization issues"
            )
        else:
            print("\n   Solution appears numerically stable")

        return {
            "value_func": value_func,
            "policy_func": policy_func,
            "endog_grid": endog_grid,
            "has_issues": value_has_nan
            or policy_has_nan
            or endog_has_nan
            or value_has_inf
            or policy_has_inf
            or endog_has_inf,
        }

    except Exception as e:
        print(f"ERROR: Error solving model: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_simulation_with_solution():  # noqa: PLR0912, PLR0915
    """Test simulation process with the solution to see where NaN values come from."""

    print("\n\n=== Simulation Process Debug Test ===\n")

    # First get the solution
    solution_result = test_model_solution()
    if solution_result is None or solution_result["has_issues"]:
        print("Skipping simulation test due to solution issues")
        return

    print("1. Testing simulation with solution...")

    try:
        # Load required data
        with open(BLD / "model" / "params" / "start_params_model.yaml", "rb") as f:
            start_params = yaml.safe_load(f)

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

        print("Loaded simulation data")

        # Create solution dictionary
        solution_dict = {
            "value": solution_result["value_func"],
            "policy": solution_result["policy_func"],
            "endog_grid": solution_result["endog_grid"],
        }

        # Test simulation
        from caregiving.simulation.simulate import simulate_scenario

        print("Running simulation...")

        # Convert parameters
        param_names = sorted(start_params.keys())
        params_array = np.array([start_params[key] for key in param_names])

        # Run simulation
        sim_df = simulate_scenario(
            model=model_for_simulation,
            solution=solution_dict,
            initial_states=initial_states,
            wealth_agents=wealth_agents,
            params=params_array,
            options=options,
            seed=options["model_params"]["seed"],
        )

        print(f"Simulation completed: shape={sim_df.shape}")

        # Check simulation results for NaN values
        print("\n2. Checking simulation results...")

        nan_columns = []
        for col in sim_df.columns:
            if sim_df[col].dtype in ("float64", "int64"):
                has_nan = np.any(np.isnan(sim_df[col]))
                has_inf = np.any(np.isinf(sim_df[col]))
                if has_nan or has_inf:
                    nan_columns.append((col, has_nan, has_inf))

        if nan_columns:
            print(f"WARNING: Found {len(nan_columns)} columns with NaN/inf values:")
            for col, has_nan, has_inf in nan_columns:
                issues = []
                if has_nan:
                    issues.append("NaN")
                if has_inf:
                    issues.append("inf")
                print(f"{col}: {', '.join(issues)}")
        else:
            print("No NaN or infinite values in simulation results")

        # Check specific problematic columns from previous test
        problematic_cols = [
            "consumption",
            "experience",
            "savings",
            "utility",
            "value_max",
        ]
        print("\n3. Checking specific problematic columns...")

        for col in problematic_cols:
            if col in sim_df.columns:
                col_data = sim_df[col]
                has_nan = np.any(np.isnan(col_data))
                has_inf = np.any(np.isinf(col_data))
                if has_nan or has_inf:
                    print(
                        f"   {col}: NaN={has_nan}, inf={has_inf}, "
                        f"range=[{np.nanmin(col_data):.6f}, {np.nanmax(col_data):.6f}]"
                    )
                else:
                    print(
                        f"   {col}: OK, "
                        f"range=[{np.nanmin(col_data):.6f}, {np.nanmax(col_data):.6f}]"
                    )

    except Exception as e:
        print(f"ERROR: Error in simulation: {e}")
        import traceback

        traceback.print_exc()


# def test_parameter_sensitivity():
#     """Test how sensitive the solution is to parameter changes."""

#     print("\n\n=== Parameter Sensitivity Test ===\n")

#     # Load base parameters
#     with open(BLD / "model" / "params" / "start_params_model.yaml", "rb") as f:
#         base_params = yaml.safe_load(f)

#     with open(BLD / "model" / "options.pkl", "rb") as f:
#         options = pickle.load(f)

#     with open(BLD / "model" / "model_for_solution.pkl", "rb") as f:
#         model_for_solution = pickle.load(f)

#     # Test with different parameter modifications
#     test_cases = [
#         ("Original", base_params),
#         ("Smaller rho", {**base_params, "rho_low": 1.0, "rho_high": 1.0}),
#         ("Larger rho", {**base_params, "rho_low": 3.0, "rho_high": 3.0}),
#         (
#             "Smaller disutility",
#             {k: v / 2 if "disutil" in k else v for k, v in base_params.items()},
#         ),
#         (
#             "Larger disutility",
#             {k: v * 2 if "disutil" in k else v for k, v in base_params.items()},
#         ),
#     ]

#     solve_func = get_solve_func_for_model(model_for_solution)

#     for case_name, params in test_cases:
#         print(f"\nTesting: {case_name}")

#         try:
#             param_names = sorted(params.keys())
#             params_array = np.array([params[key] for key in param_names])

#             value_func, policy_func, endog_grid = solve_func(params_array)

#             # Check for issues
#             has_issues = (
#                 np.any(np.isnan(value_func))
#                 or np.any(np.isinf(value_func))
#                 or np.any(np.isnan(policy_func))
#                 or np.any(np.isnan(endog_grid))
#             )

#             if has_issues:
#                 print("   ERROR: Issues found in solution")
#                 if np.any(np.isnan(value_func)):
#                     print(
#                         f"     Value function has "
#                         f"{np.sum(np.isnan(value_func))} NaN values"
#                     )
#                 if np.any(np.isnan(policy_func)):
#                     print(
#                         f"     Policy function has "
#                         f"{np.sum(np.isnan(policy_func))} NaN values"
#                     )
#                 if np.any(np.isnan(endog_grid)):
#                     print(
#                         f"     Endogenous grid has "
#                         f"{np.sum(np.isnan(endog_grid))} NaN values"
#                     )
#             else:
#                 print("   Solution is numerically stable")

#         except Exception as e:
#             print(f"   ERROR: {e}")
