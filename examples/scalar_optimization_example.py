"""Example of using scalar optimization in the estimate_model function."""

# Example usage of the modified estimate_model function with scalar optimization

# When using scalar optimization (least_squares=False):
# - The criterion function returns a single scalar value (sum of squared residuals)
# - Use om.mark.scalar() instead of om.mark.least_squares()
# - Suitable for scalar optimizers like 'scipy_lbfgsb', 'scipy_neldermead', etc.

# Example call:
"""
from caregiving.estimation.estimation_setup import estimate_model

# For scalar optimization
estimate_model(
    model_for_simulation=model_for_simulation,
    start_params=start_params,
    solve_func=solve_func,
    options=options,
    algo="scipy_lbfgsb",  # Scalar optimizer
    algo_options={"maxiter": 1000},
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    least_squares=False,  # Enable scalar optimization
    weighting_method="diagonal",
    use_cholesky_weights=True,
    relative_deviations=False,
)

# For least squares optimization (default behavior)
estimate_model(
    model_for_simulation=model_for_simulation,
    start_params=start_params,
    solve_func=solve_func,
    options=options,
    algo="tranquilo_ls",  # Least squares optimizer
    algo_options={"maxiter": 1000},
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    least_squares=True,  # Use least squares optimization (default)
    weighting_method="diagonal",
    use_cholesky_weights=True,
    relative_deviations=False,
)
"""

print("Scalar optimization example - see comments above for usage")
