"""Solve and simulate the job retention model for estimated parameters."""

import pickle
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import pandas as pd
import pytask
import yaml
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.solve import get_solve_func_for_model
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.simulate_counterfactual import (
    simulate_counterfactual_npv,
)
from caregiving.estimation.prepare_estimation import (
    load_and_setup_full_model_for_solution,
)
from caregiving.model.state_space_job_retention import (
    create_state_space_functions,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.simulate import simulate_scenario

jax.config.update("jax_enable_x64", True)


def task_solve_and_simulate_job_retention_estimated_params(
    path_to_solution_model: Path = BLD / "model" / "model_job_retention.pkl",
    path_to_options: Path = BLD / "model" / "options_job_retention.pkl",
    path_to_estimated_params: Path = BLD
    / "model"
    / "params"
    / "estimated_params_model.yaml",
    path_to_discrete_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "states_job_retention.pkl",
    path_to_wealth: Path = BLD
    / "model"
    / "initial_conditions"
    / "wealth_job_retention.csv",
    path_to_save_solution: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "solution_job_retention_estimated_params.pkl",
    # path_to_save_simulation_model: Annotated[Path, Product] = BLD
    # / "model"
    # / "model_for_simulation_job_retention_estimated_params.pkl",
    path_to_save_simulated_data: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    # path_to_save_simulated_data_jax: Annotated[Path, Product] = BLD
    # / "solve_and_simulate"
    # / "simulated_data_jax_job_retention_estimated_params.pkl",
) -> None:
    """Solve and simulate the job retention model with estimated parameters.

    This function solves and simulates the job retention counterfactual model
    using estimated parameters. The job retention model implements a policy
    where caregivers can keep their jobs even when they reduce hours or become
    unemployed due to caregiving activities.

    Args:
        path_to_solution_model: Path to the job retention model for solution
        path_to_options: Path to the job retention model options
        path_to_estimated_params: Path to the estimated parameters
        path_to_discrete_states: Path to discrete initial states
        path_to_wealth: Path to initial wealth data
        path_to_save_solution: Path to save the solution
        path_to_save_simulated_data: Path to save simulated data
    """

    options = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_estimated_params.open("rb"))

    model_for_solution = load_and_setup_full_model_for_solution(
        options, path_to_model=path_to_solution_model
    )

    # 1) Solve
    solution_dict = {}
    (
        solution_dict["value"],
        solution_dict["policy"],
        solution_dict["endog_grid"],
    ) = get_solve_func_for_model(model_for_solution)(params)
    # value, policy, endog_grid = get_solve_func_for_model(model_for_solution)(params)

    pickle.dump(solution_dict, path_to_save_solution.open("wb"))

    # 2) Simulate
    initial_states = pickle.load(path_to_discrete_states.open("rb"))
    wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

    model_for_simulation = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_solution_model,
        sim_model=True,
    )
    # pickle.dump(model_for_simulation, path_to_save_simulation_model.open("wb"))

    sim_df = simulate_scenario(
        model_for_simulation,
        solution=solution_dict,
        # solution_endog_grid=solution_dict["endog_grid"],
        # solution_value=solution_dict["value"],
        # solution_policy=solution_dict["policy"],
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options,
        seed=options["model_params"]["seed"],
    )

    # sim_df.to_csv(path_to_save_simulated_data, index=True)
    sim_df.to_pickle(path_to_save_simulated_data)

    # sim_df_npv = simulate_counterfactual_npv(
    #     model_for_simulation,
    #     solution=solution_dict,
    #     initial_states=initial_states,
    #     wealth_agents=wealth_agents,
    #     params=params,
    #     options=options,
    #     seed=options["model_params"]["seed"],
    # )
