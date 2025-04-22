"""Functions for pre and post estimation setup."""

import pickle
import time
from functools import partial
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import yaml
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.wealth_correction import adjust_observed_wealth

from caregiving.config import BLD, SRC
from caregiving.model.shared import RETIREMENT
from caregiving.model.state_space import (
    create_state_space_functions,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.simulate import simulate_scenario
from caregiving.simulation.simulate_moments import simulate_moments_jax

jax.config.update("jax_enable_x64", True)


def estimate_model(
    model_for_solution: Dict[str, Any],
    model_for_simulation: Dict[str, Any],
    solve_func: callable,
    options: Dict[str, Any],
    algo: str,
    algo_options: Dict[str, Any],
    params_to_estimate_names: list[str],
    path_to_discrete_states: str = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_wealth: str = BLD / "model" / "initial_conditions" / "wealth.csv",
    path_to_empirical_moments: str = BLD
    / "estimation"
    / "empirical_moments"
    / "empirical_moments.csv",
    path_to_empirical_variance: str = BLD
    / "estimation"
    / "empirical_moments"
    / "empirical_variance.csv",
    path_to_updated_start_params: str = BLD
    / "model"
    / "params"
    / "start_params_updated.yaml",
    path_to_lower_bounds: str = SRC
    / "estimation"
    / "start_params_and_bounds"
    / "lower_bounds.yaml",
    path_to_upper_bounds: str = SRC
    / "estimation"
    / "start_params_and_bounds"
    / "upper_bounds.yaml",
    last_estimate: Optional[Dict[str, Any]] = None,
) -> None:
    """Estimate the model based on empirical data and starting parameters."""

    # ! random seed !
    # seed = int(time.time())

    initial_states = pickle.load(path_to_discrete_states.open("rb"))
    wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

    # Load empirical data
    empirical_moments = jnp.array(
        pd.read_csv(path_to_empirical_moments, index_col=0).squeeze()
    )
    empirical_covariance = jnp.array(
        pd.read_csv(path_to_empirical_variance, index_col=0).squeeze()
    )
    _diagonal_values = 1 / jnp.diagonal(empirical_covariance)
    weights = jnp.diag(_diagonal_values)

    # if method == "optimal":
    #     array_weights = robust_inverse(_internal_cov)
    # elif method == "diagonal":
    #     diagonal_values = 1 / np.clip(np.diagonal(_internal_cov), clip_value, np.inf)
    #     array_weights = np.diag(diagonal_values)
    # elif method == "identity":
    #     array_weights = np.identity(_internal_cov.shape[0])

    with open(path_to_updated_start_params) as file:
        start_params_all = yaml.safe_load(file)

    # Assign start params from before
    if last_estimate is not None:
        for key in last_estimate.keys():
            if key in ["sigma", "interest_rate", "beta"]:
                continue
            try:
                print(
                    f"Start params value of {key} was {start_params_all[key]} and is"
                    f"replaced by {last_estimate[key]}"
                )
            except:
                raise ValueError(f"Key {key} not found in start params.")
            start_params_all[key] = last_estimate[key]

    start_params = {name: start_params_all[name] for name in params_to_estimate_names}

    lower_bounds_all = yaml.safe_load(open(path_to_lower_bounds, "rb"))
    lower_bounds = {name: lower_bounds_all[name] for name in params_to_estimate_names}

    upper_bounds_all = yaml.safe_load(open(path_to_upper_bounds, "rb"))
    upper_bounds = {name: upper_bounds_all[name] for name in params_to_estimate_names}

    bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)

    # # Solve and simulate
    # solution_dict = solve_func(model_for_solution, params=start_params)

    # sim_df = simulate_scenario(
    #     model_for_simulation,
    #     solution=solution_dict,
    #     initial_states=initial_states,
    #     wealth_agents=wealth_agents,
    #     params=start_params,
    #     options=options,
    #     seed=options["model_params"]["seed"],
    # )
    # simulated_moments = simulate_moments_jax(sim_df, options=options)

    simulate_moments_given_params = partial(
        simulate_moments,
        solve_func=solve_func,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        model_for_solution=model_for_solution,
        model_for_simulation=model_for_simulation,
        options=options,
    )

    # Minimize
    criterion_func = get_msm_optimization_function(
        simulate_moments=simulate_moments_given_params,
        empirical_moments=empirical_moments,
        weights=weights,
    )

    result = om.minimize(
        fun=criterion_func,
        params=start_params,
        bounds=bounds,
        algorithm=algo,
        algo_options=algo_options,
        # multistart=om.MultistartOptions(n_samples=100, seed=0, n_cores=4),
        # logging="test_log.db",
        error_handling="continue",
    )

    return result

    # pickle.dump(
    #     result, open(path_dict["struct_results"] + f"em_result_{file_append}.pkl", "wb")
    # )
    # start_params_all.update(result.params)

    # pickle.dump(
    #     start_params_all,
    #     open(path_dict["struct_results"] + f"est_params_{file_append}.pkl", "wb"),
    # )


# =====================================================================================
# Criterion function
# =====================================================================================


def simulate_moments(
    params: jnp.ndarray,
    solve_func: callable,
    initial_states: Dict[str, Any],
    wealth_agents: jnp.ndarray,
    model_for_solution: Dict[str, Any],
    model_for_simulation: Dict[str, Any],
    options: Dict[str, Any],
):
    """Solve the model and simulate moments."""

    solution_dict = solve_func(model_for_solution, params=params)

    sim_df = simulate_scenario(
        model=model_for_simulation,
        solution=solution_dict,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options,
        seed=options["model_params"]["seed"],
    )

    simulated_moments = simulate_moments_jax(sim_df, options=options)

    return simulated_moments


def get_msm_optimization_function(
    simulate_moments: callable,
    empirical_moments: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:

    chol_weights = jnp.linalg.cholesky(weights)

    criterion = om.mark.least_squares(
        partial(
            msm_criterion,
            simulate_moments=simulate_moments,
            flat_empirical_moments=empirical_moments,
            chol_weights=chol_weights,
        )
    )

    # out = {"fun": criterion}
    return criterion


def msm_criterion(
    params: jnp.ndarray,
    simulate_moments: callable,
    flat_empirical_moments: jnp.ndarray,
    chol_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the raw criterion based on simulated and empirical moments."""

    simulated_flat = simulate_moments(params)

    deviations = simulated_flat - flat_empirical_moments
    residuals = deviations @ chol_weights

    # LeastSquaresFunctionValue(value=residuals)
    return residuals


# def solve_and_simulate(
#     params,
#     solve_func: callable,
#     model_for_solution: Dict[str, Any],
#     model_for_simulation: Dict[str, Any],
#     initial_states: Dict[str, Any],
#     wealth_agents: jnp.ndarray,
#     options: Dict[str, Any],
# ):

#     # Solve and simulate
#     solution_dict = solve_func(model_for_solution, params=params)

#     sim_df = simulate_scenario(
#         model_for_simulation,
#         solution=solution_dict,
#         initial_states=initial_states,
#         wealth_agents=wealth_agents,
#         params=params,
#         options=options,
#         seed=options["model_params"]["seed"],
#     )
#     simulated_moments = simulate_moments_jax(sim_df, options=options)

#     return simulated_moments


# def get_criterion_function(
#     model: Dict[str, Any],
#     empirical_moments: np.ndarray,
#     empirical_variance: np.ndarray,
#     initial_states: Dict[str, Any],
#     initial_resources: np.ndarray,
#     params_fixed: Dict[str, Any],
#     params_to_estimate: Dict[str, Any],
#     params_to_estimate_names: list[str],
#     params_to_estimate_values: np.ndarray,
#     params_to_estimate_bounds: Dict[str, Any],
#     params_to_estimate_bounds_names: list[str],
# ) -> np.ndarray:
#     """Get the criterion function for the estimation."""

#     # Update params with new values
#     params = params_fixed.copy()
#     for name, value in zip(params_to_estimate_names, params_to_estimate_values):
#         params[name] = value

#     # Get the solve function
#     solve_func = get_solve_func_for_model(
#         model=model,
#         initial_states=initial_states,
#         initial_resources=initial_resources,
#         params=params,
#         endog_grid_solved=None,
#         value_solved=None,
#         policy_solved=None,
#     )

#     # Get the moment error vector
#     moment_error_vec = get_moment_error_vec(
#         params_fixed=params_fixed,
#         options=model["options"],
#         emp_moments=empirical_moments,
#         model_loaded=model,
#         solve_func=solve_func,
#         initial_states=initial_states,
#         initial_resources=initial_resources,
#     )

#     # Calculate the criterion function
#     criterion_function = (
#         moment_error_vec.T @ np.linalg.inv(empirical_variance) @ moment_error_vec
#     )

#     return criterion_function


# def get_solve_func_for_model(
#     model: Dict[str, Any],
#     initial_states: Dict[str, Any],
#     initial_resources: np.ndarray,
#     params: Dict[str, Any],
#     endog_grid_solved: Optional[np.ndarray] = None,
#     value_solved: Optional[np.ndarray] = None,
#     policy_solved: Optional[np.ndarray] = None,
# ) -> Any:
#     """Get the solve function for the model."""

#     # Get the solve function
#     solve_func = model["solve_func"]

#     # Solve the model
#     if endog_grid_solved is None or value_solved is None or policy_solved is None:
#         (
#             endog_grid_solved,
#             value_solved,
#             policy_solved,
#         ) = solve_func(
#             initial_states=initial_states,
#             initial_resources=initial_resources,
#             params=params,
#         )

#     return endog_grid_solved, value_solved, policy_solved


# def get_moment_error_vec(
#     params_fixed: Dict[str, Any],
#     options: Dict[str, Any],
#     emp_moments: np.ndarray,
#     model_loaded: Dict[str, Any],
#     solve_func: Any,
#     initial_states: Dict[str, Any],
#     initial_resources: np.ndarray,
# ) -> np.ndarray:
#     """Get the moment error vector for the estimation."""

#     # Get the model parameters
#     params = params_fixed.copy()

#     # Get the empirical moments
#     empirical_moments = emp_moments

#     # Get the model moments
#     model_moments = solve_func(
#         initial_states=initial_states,
#         initial_resources=initial_resources,
#         params=params,
#     )

#     # Calculate the moment error vector
#     moment_error_vec = empirical_moments - model_moments

#     return moment_error_vec


# =====================================================================================
# Preparation for estimation
# =====================================================================================


def load_and_setup_full_model_for_solution(options, path_to_model) -> Dict[str, Any]:
    """Load and setup full model for solution."""

    model_full = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_model,
        sim_model=False,
    )

    return model_full


def load_and_prep_data(data_emp, model, start_params, drop_retirees=True):
    """Load and prepare empirical data to compare with simulated data."""
    # specs = generate_derived_and_data_derived_specs(path_dict)
    # data_decision = pd.read_pickle(path_dict["struct_est_sample"])

    specs = model["options"]["model_params"]
    # We need to filter observations in period 0 because of job offer
    # weighting from last period
    data_emp = data_emp[data_emp["period"] > 0].copy()

    # Also already retired individuals hold no identification
    if drop_retirees:
        data_emp = data_emp[~data_emp["lagged_choice"].isin(RETIREMENT.tolist())]

    data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_emp.loc[:, "age_bin"] = np.floor(data_emp["age"] / 10)
    data_emp.loc[data_emp["age_bin"] > 6, "age_bin"] = 6  # noqa: PLR2004

    age_bin_av_size = data_emp.shape[0] / data_emp["age_bin"].nunique()
    data_emp.loc[:, "age_weights"] = 1.0
    data_emp.loc[:, "age_weights"] = age_bin_av_size / data_emp.groupby("age_bin")[
        "age_weights"
    ].transform("sum")

    # Transform experience
    max_init_exp = specs["max_exp_diffs_per_period"][data_emp["period"].values]
    exp_denominator = data_emp["period"].values + max_init_exp
    data_emp["experience"] = data_emp["experience"] / exp_denominator

    # We can adjust wealth outside, as it does not depend on estimated parameters
    # (only on interest rate)
    # Now transform for dcegm
    states_dict = {
        name: data_emp[name].values
        for name in model["model_structure"]["discrete_states_names"]
    }
    states_dict["experience"] = data_emp["experience"].values
    states_dict["wealth"] = data_emp["wealth"].values / specs["wealth_unit"]

    adjusted_wealth = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=start_params,
        model=model,
    )
    data_emp.loc[:, "adjusted_wealth"] = adjusted_wealth
    states_dict["wealth"] = data_emp["adjusted_wealth"].values

    return data_emp, states_dict
