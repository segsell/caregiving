"""Solve and simulate the model for estimated parameters."""

import pickle
from pathlib import Path
from typing import Annotated

import jax
import yaml
from pytask import Product

import dcegm
from caregiving.config import BLD
from caregiving.estimation.estimation_setup import draw_caregiving_type_from_params
from caregiving.model.state_space_no_inheritance import create_state_space_functions

# from caregiving.simulation.simulate import simulate_scenario
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation_no_inheritance import (
    budget_constraint,
)
from caregiving.simulation.simulate import simulate_scenario

jax.config.update("jax_enable_x64", True)


# # @pytask.mark.no_inheritance
# @pytask.mark.baseline_model_no_inheritance
# def task_solve_and_simulate_no_inheritance(
#     path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
#     path_to_model: Path = BLD / "model" / "model_no_inheritance.pkl",
#     path_to_model_config: Path = BLD / "model" / "model_config_no_inheritance.pkl",
#     path_to_estimated_params: Path = BLD
#     / "model"
#     / "params"
#     / "estimated_params_model.yaml",
#     path_to_initial_states: Path = (
#         BLD / "model" / "initial_conditions" / "initial_states.pkl"
#     ),
#     path_to_save_solution: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "solution_no_inheritance.pkl",
#     # path_to_save_simulation_model: Annotated[Path, Product] = BLD
#     # / "model"
#     # / "model_for_simulation_estimated_params.pkl",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_no_inheritance.pkl",
#     # path_to_save_simulated_data_jax: Annotated[Path, Product] = BLD
#     # / "solve_and_simulate"
#     # / "simulated_data_jax_estimated_params.pkl",
# ) -> None:
#     """Solve and simulate the model for estimated parameters."""

#     solve_and_simulate_no_inheritance(
#         path_to_specs=path_to_specs,
#         path_to_model=path_to_model,
#         path_to_model_config=path_to_model_config,
#         path_to_estimated_params=path_to_estimated_params,
#         path_to_initial_states=path_to_initial_states,
#         path_to_save_solution=path_to_save_solution,
#         path_to_save_simulated_data=path_to_save_simulated_data,
#     )


def solve_and_simulate_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model: Path = BLD / "model" / "model_no_inheritance.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config_no_inheritance.pkl",
    path_to_estimated_params: Path = BLD
    / "model"
    / "params"
    / "estimated_params_model.yaml",
    path_to_initial_states: Path = (
        BLD / "model" / "initial_conditions" / "initial_states.pkl"
    ),
    path_to_save_solution: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "solution_no_inheritance.pkl",
    # path_to_save_simulation_model: Annotated[Path, Product] = BLD
    # / "model"
    # / "model_for_simulation_estimated_params.pkl",
    path_to_save_simulated_data: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    # path_to_save_simulated_data_jax: Annotated[Path, Product] = BLD
    # / "solve_and_simulate"
    # / "simulated_data_jax_estimated_params.pkl",
) -> None:
    """Solve and simulate the model for estimated parameters."""

    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))
    params = yaml.safe_load(path_to_estimated_params.open("rb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_model,
    )
    # 1) Solve
    model_solved = model.solve(params, save_sol_path=path_to_save_solution)

    # 2) Simulate
    initial_states = pickle.load(path_to_initial_states.open("rb"))

    initial_states_adjusted = draw_caregiving_type_from_params(
        initial_states, params, specs["seed"]
    )

    # =================================================================================
    # Simulate scenario
    # =================================================================================
    # sim_df = model_solved.simulate(
    #     states_initial=initial_states,
    #     seed=specs["seed"],
    # )

    # sim_df = sim_df[sim_df["health"] != DEAD].copy()
    # sim_df.reset_index(inplace=True)

    # # sim_df["age"] = sim_df["period"] + specs["start_age"]
    # sim_df = create_additional_variables(sim_df, specs)
    sim_df = simulate_scenario(model_solved, initial_states_adjusted, specs)

    # =================================================================================

    # sim_df.to_csv(path_to_save_simulated_data, index=True)
    sim_df.to_pickle(path_to_save_simulated_data)
