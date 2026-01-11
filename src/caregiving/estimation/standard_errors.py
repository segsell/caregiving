"""Functions for computing analytical standard errors."""

from functools import partial
from typing import Any, Callable, Dict

import jax.numpy as jnp
import numpy as np
from optimagic.differentiation.derivatives import first_derivative

from caregiving.estimation.estimation_setup import simulate_moments
from caregiving.simulation.simulate import simulate_scenario
from caregiving.simulation.simulate_moments import simulate_moments_pandas


def get_analytical_standard_errors(
    params: Dict[str, float],
    options: Dict[str, Any],
    emp_moments: jnp.ndarray,
    emp_var: jnp.ndarray,
    model_loaded: Dict[str, Any],
    solve_func: Callable,
    initial_states: Dict[str, jnp.ndarray],
    wealth_agents: jnp.ndarray,
):
    """Get analytical standard errors using asymptotic distribution theory.

    Computes standard errors using the sandwich formula:
    Var(θ) = (G'W G)^(-1) * G'W Σ W'G * (G'W G)^(-1)

    where:
    - G is the Jacobian (sensitivity) matrix of moments to parameters
    - W is the weighting matrix
    - Σ is the covariance matrix of moments

    Args:
        params: Dictionary of estimated parameters
        params_fixed: Dictionary of fixed parameters
        options: Model options
        emp_moments: Empirical moments
        emp_var: Empirical variances
        model_loaded: Loaded model dictionary
        solve_func: Function to solve the model
        initial_states: Initial states for simulation
        wealth_agents: Initial wealth for agents

    Returns:
        Array of standard errors for each parameter
    """
    # Handle zero variances to avoid numerical issues
    emp_var = np.maximum(emp_var, 1e-12)
    covariance = jnp.diag(emp_var)
    weighting_mat = jnp.diag(emp_var ** (-1))

    # Create partial function for moment errors
    get_error_partial = partial(
        get_moment_error_vec,
        options=options,
        emp_moments=emp_moments,
        model_loaded=model_loaded,
        solve_func=solve_func,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
    )

    # Compute Jacobian (sensitivity of moments to parameters)
    jac = first_derivative(
        func=get_error_partial,
        params=params,
        step_size=0.01,  # Updated from deprecated base_steps
        method="forward",
    )
    _jacobian = list(jac.derivative.values())

    # Debug: Check dimensions
    print(f"Number of parameters: {len(params)}")
    print(f"Number of moments: {len(emp_moments)}")
    print(f"Jacobian derivative values length: {len(_jacobian)}")
    if _jacobian:
        print(f"First derivative shape: {_jacobian[0].shape}")

    jacobian = jnp.stack(_jacobian).T
    print(f"Final Jacobian shape: {jacobian.shape}")
    print(f"Weighting matrix shape: {weighting_mat.shape}")
    print(f"Covariance matrix shape: {covariance.shape}")

    # Sandwich formula: (G'W G)^(-1) * G'W Σ W'G * (G'W G)^(-1)

    # Check dimension compatibility
    if jacobian.shape[0] != len(emp_moments):
        raise ValueError(
            f"Jacobian first dimension ({jacobian.shape[0]}) must match "
            f"number of moments ({len(emp_moments)})"
        )
    if jacobian.shape[1] != len(params):
        raise ValueError(
            f"Jacobian second dimension ({jacobian.shape[1]}) must match "
            f"number of parameters ({len(params)})"
        )

    bread = jnp.linalg.inv(jacobian.T @ weighting_mat @ jacobian)
    butter = jacobian.T @ weighting_mat @ covariance @ weighting_mat @ jacobian
    variance = bread @ butter @ bread

    return jnp.sqrt(jnp.diag(variance))


def get_moment_error_vec(
    params: Dict[str, float],
    options: Dict[str, Any],
    model_loaded: Dict[str, Any],
    emp_moments: jnp.ndarray,
    solve_func: Callable,
    initial_states: Dict[str, jnp.ndarray],
    wealth_agents: jnp.ndarray,
) -> jnp.ndarray:
    """Compute moment error vector for a given set of parameters.

    Simulates the model with given parameters and computes the difference
    between simulated and empirical moments.

    Args:
        params: Dictionary of parameters to evaluate
        params_fixed: Dictionary of fixed parameters
        options: Model options
        model_loaded: Loaded model dictionary
        emp_moments: Empirical moments
        solve_func: Function to solve the model
        initial_states: Initial states for simulation
        wealth_agents: Initial wealth for agents

    Returns:
        Array of moment errors (simulated - empirical)
    """

    # Create partial function for simulate_moments with fixed arguments
    simulate_moments_partial = partial(
        simulate_moments,
        solve_func=solve_func,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        model_for_simulation=model_loaded,
        options=options,
        fixed_seed=42,  # Use a fixed seed for reproducibility
        seed_generator=None,  # Use fixed seed instead of generator
        simulate_scenario_func=simulate_scenario,
        simulate_moments_func=simulate_moments_pandas,
    )

    # Simulate moments using the existing function
    sim_moments_array = simulate_moments_partial(params)

    # Return moment errors
    return sim_moments_array - emp_moments
