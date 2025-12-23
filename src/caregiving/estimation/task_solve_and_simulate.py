# """Solve and simulate the model for start parameters."""

# import pickle
# from pathlib import Path
# from typing import Annotated, Any, Dict, List, Tuple

# import jax
# import jax.numpy as jnp
# import pandas as pd
# import pytask
# import yaml

# # from dcegm.pre_processing.setup_model import load_and_setup_model
# # from dcegm.solve import get_solve_func_for_model
# # from pytask import Product

# # from caregiving.config import BLD
# # from caregiving.counterfactual.simulate_counterfactual import (
# #     simulate_counterfactual_npv,
# # )
# # from caregiving.estimation.prepare_estimation import (
# #     load_and_setup_full_model_for_solution,
# # )
# # from caregiving.model.state_space import (
# #     create_state_space_functions,
# # )
# # from caregiving.model.utility.bequest_utility import (
# #     create_final_period_utility_functions,
# # )
# # from caregiving.model.utility.utility_functions_additive import create_utility_functions
# # from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
# # from caregiving.simulation.simulate import simulate_scenario

# # jax.config.update("jax_enable_x64", True)


# # @pytask.mark.skip()
# # @pytask.mark.baseline_model
# # def task_solve_and_simulate_start_params(
# #     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
# #     path_to_options: Path = BLD / "model" / "options.pkl",
# #     path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
# #     path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
# #     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
# #     path_to_save_solution: Annotated[Path, Product] = BLD
# #     / "solve_and_simulate"
# #     / "solution.pkl",
# #     # path_to_save_simulation_model: Annotated[Path, Product] = BLD
# #     # / "model"
# #     # / "model_for_simulation.pkl",
# #     path_to_save_simulated_data: Annotated[Path, Product] = BLD
# #     / "solve_and_simulate"
# #     / "simulated_data.pkl",
# #     # path_to_save_simulated_data_jax: Annotated[Path, Product] = BLD
# #     # / "solve_and_simulate"
# #     # / "simulated_data_jax.pkl",
# # ) -> None:

# #     options = pickle.load(path_to_options.open("rb"))
# #     params = yaml.safe_load(path_to_start_params.open("rb"))

# #     model_for_solution = load_and_setup_full_model_for_solution(
# #         options, path_to_model=path_to_solution_model
# #     )

# #     # 1) Solve
# #     solution_dict = {}
# #     (
# #         solution_dict["value"],
# #         solution_dict["policy"],
# #         solution_dict["endog_grid"],
# #     ) = get_solve_func_for_model(model_for_solution)(params)
# #     # value, policy, endog_grid = get_solve_func_for_model(model_for_solution)(params)

# #     pickle.dump(solution_dict, path_to_save_solution.open("wb"))

# #     # 2) Simulate
# #     initial_states = pickle.load(path_to_discrete_states.open("rb"))
# #     wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

# #     model_for_simulation = load_and_setup_model(
# #         options=options,
# #         state_space_functions=create_state_space_functions(),
# #         utility_functions=create_utility_functions(),
# #         utility_functions_final_period=create_final_period_utility_functions(),
# #         budget_constraint=budget_constraint,
# #         # shock_functions=shock_function_dict(),
# #         path=path_to_solution_model,
# #         sim_model=True,
# #     )
# #     # pickle.dump(model_for_simulation, path_to_save_simulation_model.open("wb"))

# #     sim_df = simulate_scenario(
# #         model_for_simulation,
# #         solution=solution_dict,
# #         # solution_endog_grid=solution_dict["endog_grid"],
# #         # solution_value=solution_dict["value"],
# #         # solution_policy=solution_dict["policy"],
# #         initial_states=initial_states,
# #         wealth_agents=wealth_agents,
# #         params=params,
# #         options=options,
# #         seed=options["model_params"]["seed"],
# #     )

# #     # sim_df.to_csv(path_to_save_simulated_data, index=True)
# #     sim_df.to_pickle(path_to_save_simulated_data)

# #     # sim_df_npv = simulate_counterfactual_npv(
# #     #     model_for_simulation,
# #     #     solution=solution_dict,
# #     #     initial_states=initial_states,
# #     #     wealth_agents=wealth_agents,
# #     #     params=params,
# #     #     options=options,
# #     #     seed=options["model_params"]["seed"],
# #     # )
