"""Solve and simulate the counterfactual without care demand."""

import pickle
from pathlib import Path
from typing import Annotated, Any, Dict

import jax
import jax.numpy as jnp
import pandas as pd
import pytask
import yaml
from pytask import Product

from caregiving.config import BLD
from caregiving.model.state_space_no_care_demand import create_state_space_functions
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive_no_care_demand import (
    create_utility_functions,
)
from caregiving.model.wealth_and_budget.budget_equation_no_care_demand import (
    budget_constraint,
)
from caregiving.simulation.simulate_no_care_demand import (
    simulate_scenario_no_care_demand,
)
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.solve import get_solve_func_for_model

jax.config.update("jax_enable_x64", True)


def task_solve_and_simulate_no_care_demand(
    path_to_solution_model: Path = BLD / "model" / "model_no_care_demand.pkl",
    path_to_options: Path = BLD / "model" / "options_no_care_demand.pkl",
    path_to_params: Path = BLD
    / "model"
    / "params"
    / "start_params_model_no_care_demand.yaml",
    # / "params_estimated_no_care_demand.yaml",
    path_to_discrete_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "states_no_care_demand.pkl",
    path_to_wealth: Path = BLD
    / "model"
    / "initial_conditions"
    / "wealth_no_care_demand.csv",
    path_to_save_solution: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "solution_no_care_demand.pkl",
    path_to_save_simulated_data: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
):
    """Solve and simulate the counterfactual and save the DataFrame."""

    options = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_params.open("rb"))

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
        path=path_to_solution_model,
        sim_model=True,
    )

    sim_df = simulate_scenario_no_care_demand(
        model_for_simulation,
        solution=solution_dict,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options,
        seed=options["model_params"]["seed"],
    )
    sim_df.to_pickle(path_to_save_simulated_data)


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
