# """Simulate the model with estimated parameters for 1,000,000 agents.

# Uses the existing solution from the baseline simulation.
# """

# import pickle
# from pathlib import Path
# from typing import Annotated

# import jax
# import jax.numpy as jnp
# import pandas as pd
# import pytask
# import yaml
# from pytask import Product

# from caregiving.config import BLD
# from caregiving.model.state_space import (
#     create_state_space_functions,
# )
# from caregiving.model.utility.bequest_utility import (
#     create_final_period_utility_functions,
# )
# from caregiving.model.utility.utility_functions_additive import create_utility_functions
# from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
# from caregiving.simulation.simulate import simulate_scenario
# from dcegm.pre_processing.setup_model import load_and_setup_model

# jax.config.update("jax_enable_x64", True)


# @pytask.mark.simulate_1m
# def task_simulate_estimated_params_1m(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_estimated_params: Path = BLD
#     / "model"
#     / "params"
#     / "estimated_params_model.yaml",
#     path_to_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_estimated_params.pkl",
#     path_to_discrete_states: Path = BLD
#     / "model"
#     / "initial_conditions"
#     / "1m"
#     / "states_1m.pkl",
#     path_to_wealth: Path = BLD
#     / "model"
#     / "initial_conditions"
#     / "1m"
#     / "wealth_1m.csv",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_estimated_params_1m.pkl",
# ) -> None:
#     """Simulate the model with estimated parameters for 1,000,000 agents.

#     Loads the existing solution from the baseline simulation and only performs
#     the simulation step with 1m initial conditions.
#     """

#     options = pickle.load(path_to_options.open("rb"))
#     params = yaml.safe_load(path_to_estimated_params.open("rb"))

#     # Load existing solution from baseline
#     solution_dict = pickle.load(path_to_solution.open("rb"))

#     # Simulate with 1m initial conditions
#     initial_states = pickle.load(path_to_discrete_states.open("rb"))
#     wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

#     model_for_simulation = load_and_setup_model(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         path=path_to_solution_model,
#         sim_model=True,
#     )

#     sim_df = simulate_scenario(
#         model_for_simulation,
#         solution=solution_dict,
#         initial_states=initial_states,
#         wealth_agents=wealth_agents,
#         params=params,
#         options=options,
#         seed=options["model_params"]["seed"],
#     )

#     sim_df.to_pickle(path_to_save_simulated_data)
