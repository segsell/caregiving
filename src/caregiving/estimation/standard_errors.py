from functools import partial

import jax.numpy as jnp
from optimagic.differentiation.derivatives import first_derivative

# from caregiving.estimate import get_moment_error_vec


def get_analytical_standard_errors(
    params: dict,
    params_fixed: dict,
    options: dict,
    emp_moments: jnp.array,
    emp_var: jnp.array,
    model_loaded: dict,
    solve_func: callable,
    initial_states: dict,
    initial_resources: jnp.array,
):
    """Get analytical standard errors.

    weighting_mat = jnp.linalg.inv(covariance)

    """
    covariance = jnp.diag(emp_var)
    weighting_mat = jnp.diag(emp_var ** (-1))

    get_error_partial = partial(
        get_moment_error_vec,
        params_fixed=params_fixed,
        options=options,
        emp_moments=emp_moments,
        model_loaded=model_loaded,
        solve_func=solve_func,
        initial_states=initial_states,
        initial_resources=initial_resources,
    )

    jac = first_derivative(
        func=get_error_partial,
        params=params,
        base_steps=0.01,
        method="forward",
    )
    _jacobian = list(jac["derivative"].values())

    jacobian = jnp.stack(_jacobian).T

    bread = jnp.linalg.inv(jacobian.T @ weighting_mat @ jacobian)
    butter = jacobian.T @ weighting_mat @ covariance @ weighting_mat @ jacobian
    variance = bread @ butter @ bread

    return jnp.sqrt(jnp.diag(variance))


def get_moment_error_vec(
    params,
    params_fixed,
    options,
    model_loaded,
    emp_moments,
    solve_func,
    initial_states,
    initial_resources,
):
    """Criterion function for the estimation.

    chol_weights = jnp.eye(len(emp_moments))

    err = sim_moments - emp_moments
    # crit_val = jnp.dot(jnp.dot(err.T, chol_weights), err)

    # deviations = sim_moments - np.array(emp_moments)
    root_contribs = err @ chol_weights
    crit_val = root_contribs @ root_contribs

    """
    seed = int(time.time())

    params_all = params | params_fixed

    value, policy, endog_grid = solve_func(params_all)

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=initial_resources,
        n_periods=options["model_params"]["n_periods"],
        params=params_all,
        seed=seed,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model=model_loaded,
    )

    data = create_simulation_df_from_dict(sim_dict)
    arr, idx = create_simulation_array_from_df(
        data=data,
        options=options,
        params=params_all,
    )

    _sim_raw = simulate_moments(arr, idx)

    return _sim_raw - emp_moments
