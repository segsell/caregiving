"""Test function to debug MSM optimization issues."""

import pickle
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from caregiving.config import BLD
from caregiving.estimation.estimation_setup import (
    get_msm_optimization_function,
    msm_criterion,
    msm_criterion_scalar,
    simulate_moments,
)
from caregiving.simulation.simulate_moments import simulate_moments_pandas


def test_msm_with_simulated_data():  # noqa: PLR0912, PLR0915
    """Test MSM criterion function with pre-simulated data to identify issues."""

    print("=== MSM Optimization Debug Test ===\n")

    # 1. Load simulated data
    print("1. Loading simulated data...")
    sim_data_path = BLD / "solve_and_simulate" / "simulated_data.pkl"

    try:
        with open(sim_data_path, "rb") as f:
            sim_df = pickle.load(f)
        print(f"Loaded simulated data: {type(sim_df)}, shape: {sim_df.shape}")
        print(f"Columns: {list(sim_df.columns)[:10]}...")  # Show first 10 columns
    except Exception as e:
        print(f"ERROR: Failed to load simulated data: {e}")
        return

    # 2. Load empirical moments and variances
    print("\n2. Loading empirical moments...")
    try:
        empirical_moments = np.array(
            pd.read_csv(BLD / "moments" / "moments_full.csv", index_col=0).squeeze()
        )
        empirical_variances = np.array(
            pd.read_csv(BLD / "moments" / "variances_full.csv", index_col=0).squeeze()
        )
        print(f"Empirical moments shape: {empirical_moments.shape}")
        print(f"Empirical variances shape: {empirical_variances.shape}")
        print(
            f"   Moments range: "
            f"[{np.nanmin(empirical_moments):.6f}, {np.nanmax(empirical_moments):.6f}]"
        )
        print(
            f"   Variances range: "
            f"[{np.nanmin(empirical_variances):.6f}, "
            f"{np.nanmax(empirical_variances):.6f}]"
        )
    except Exception as e:
        print(f"ERROR: Failed to load empirical data: {e}")
        return

    # 3. Load start parameters
    print("\n3. Loading start parameters...")
    try:
        start_params_path = BLD / "model" / "params" / "start_params_model.yaml"
        with open(start_params_path, "rb") as f:
            start_params = yaml.safe_load(f)
        print(f"Loaded {len(start_params)} parameters")
        print(f"Sample params: {dict(list(start_params.items())[:5])}")
    except Exception as e:
        print(f"ERROR: Failed to load start parameters: {e}")
        return

    # 4. Load options
    print("\n4. Loading model options...")
    try:
        options_path = BLD / "model" / "options.pkl"
        with open(options_path, "rb") as f:
            options = pickle.load(f)
        print("Loaded model options")
    except Exception as e:
        print(f"ERROR: Failed to load options: {e}")
        return

    # 5. Create mock simulate_moments function that uses pre-simulated data
    print("\n5. Creating mock simulate_moments function...")

    def mock_simulate_moments(params):
        """Mock function that returns moments from pre-simulated data."""
        print(
            f"   Called with params shape: "
            f"{len(params) if hasattr(params, '__len__') else 'scalar'}"
        )

        try:
            # Compute moments from the pre-simulated data
            simulated_moments = simulate_moments_pandas(sim_df, options)
            moments_array = np.asarray(simulated_moments.to_numpy())

            print(f"Computed moments shape: {moments_array.shape}")
            print(
                f"   Moments range: "
                f"[{np.nanmin(moments_array):.6f}, {np.nanmax(moments_array):.6f}]"
            )

            # Check for problematic values
            has_nan = np.any(np.isnan(moments_array))
            has_inf = np.any(np.isinf(moments_array))
            LARGE_VALUE_THRESHOLD = 1e10
            has_large = np.any(np.abs(moments_array) > LARGE_VALUE_THRESHOLD)

            if has_nan:
                print(f"WARNING: Found {np.sum(np.isnan(moments_array))} NaN values")
            if has_inf:
                print(
                    f"   WARNING: Found "
                    f"{np.sum(np.isinf(moments_array))} infinite values"
                )
            if has_large:
                print(
                    f"   WARNING: Found "
                    f"{np.sum(np.abs(moments_array) > LARGE_VALUE_THRESHOLD)} "
                    f"very large values"
                )

            # Apply the same NaN handling as in the original function
            out = np.nan_to_num(moments_array, nan=0.0, posinf=0.0, neginf=0.0)

            print(
                f"   After NaN handling: [{np.nanmin(out):.6f}, {np.nanmax(out):.6f}]"
            )

            return out

        except Exception as e:
            print(f"ERROR: Error computing moments: {e}")
            # Return zeros as fallback
            return np.zeros_like(empirical_moments)

    # 6. Test weights computation
    print("\n6. Testing weights computation...")
    try:
        # Test identity weights
        identity_weights = np.identity(empirical_moments.shape[0])
        print(f"Identity weights shape: {identity_weights.shape}")

        # Test diagonal weights - fix the zero variance issue
        empirical_variances_reg = empirical_variances.copy()
        MACHINE_ZERO = 1e-12
        close_to_zero = empirical_variances_reg < MACHINE_ZERO
        print(f"Zero variances found: {np.sum(close_to_zero)}")

        # Replace zero variances with a small positive value to avoid division by zero
        empirical_variances_reg[close_to_zero] = 1e-6

        weight_elements = 1 / empirical_variances_reg
        weight_elements = np.sqrt(weight_elements)
        diagonal_weights = np.diag(weight_elements)

        print(f"Diagonal weights shape: {diagonal_weights.shape}")
        print(
            f"   Weight elements range: "
            f"[{np.nanmin(weight_elements):.6f}, {np.nanmax(weight_elements):.6f}]"
        )

        # Test Cholesky decomposition
        chol_weights = np.linalg.cholesky(diagonal_weights)
        print(f"Cholesky weights shape: {chol_weights.shape}")
        print(
            f"   Cholesky weights range: "
            f"[{np.nanmin(chol_weights):.6f}, {np.nanmax(chol_weights):.6f}]"
        )

    except Exception as e:
        print(f"ERROR: Error computing weights: {e}")
        print("Using identity weights as fallback")
        diagonal_weights = identity_weights
        chol_weights = identity_weights

    # 7. Test MSM criterion functions
    print("\n7. Testing MSM criterion functions...")

    # Convert start_params to array format expected by criterion functions
    params_array = np.array([start_params[key] for key in sorted(start_params.keys())])
    print(f"Parameter array shape: {params_array.shape}")
    print(
        f"   Parameter range: "
        f"[{np.nanmin(params_array):.6f}, {np.nanmax(params_array):.6f}]"
    )

    try:
        # Test scalar criterion
        print("\n   Testing scalar criterion...")
        scalar_result = msm_criterion_scalar(
            params=params_array,
            simulate_moments=mock_simulate_moments,
            flat_empirical_moments=empirical_moments,
            chol_weights=chol_weights,
            relative_deviations=False,
        )
        print(f"Scalar criterion result: {scalar_result}")

        if np.isnan(scalar_result) or np.isinf(scalar_result):
            print(f"WARNING: WARNING: Scalar criterion is {scalar_result}")

    except Exception as e:
        print(f"ERROR: Error in scalar criterion: {e}")
        import traceback

        traceback.print_exc()

    try:
        # Test vector criterion (least squares)
        print("\n   Testing vector criterion...")
        vector_result = msm_criterion(
            params=params_array,
            simulate_moments=mock_simulate_moments,
            flat_empirical_moments=empirical_moments,
            chol_weights=chol_weights,
            relative_deviations=False,
        )
        print(f"Vector criterion shape: {vector_result.shape}")
        print(
            f"    Vector criterion range: "
            f"[{np.nanmin(vector_result):.6f}, {np.nanmax(vector_result):.6f}]"
        )

        if np.any(np.isnan(vector_result)):
            print(
                f"   WARNING: Found "
                f"{np.sum(np.isnan(vector_result))} NaN values in vector criterion"
            )
        if np.any(np.isinf(vector_result)):
            print(
                f"   WARNING: Found "
                f"{np.sum(np.isinf(vector_result))} infinite values in vector criterion"
            )

    except Exception as e:
        print(f"ERROR: Error in vector criterion: {e}")
        import traceback

        traceback.print_exc()

    # 8. Test optimization function creation
    print("\n8. Testing optimization function creation...")
    try:
        opt_func = get_msm_optimization_function(
            simulate_moments=mock_simulate_moments,
            empirical_moments=empirical_moments,
            weights=diagonal_weights,
            cholesky=True,
            relative_deviations=False,
            least_squares=True,
        )
        print("Optimization function created successfully")

        # Test calling the optimization function
        print("\n   Testing optimization function call...")
        opt_result = opt_func(params_array)
        print(f"Optimization function result shape: {opt_result.shape}")
        print(
            f"    Optimization function result range: "
            f"[{np.nanmin(opt_result):.6f}, {np.nanmax(opt_result):.6f}]"
        )

        if np.any(np.isnan(opt_result)):
            print(
                f"   WARNING: Found "
                f"{np.sum(np.isnan(opt_result))} NaN values in optimization result"
            )
        if np.any(np.isinf(opt_result)):
            print(
                f"   WARNING: Found "
                f"{np.sum(np.isinf(opt_result))} infinite values in optimization result"
            )

    except Exception as e:
        print(f"ERROR: Error in optimization function: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== Test Complete ===")
