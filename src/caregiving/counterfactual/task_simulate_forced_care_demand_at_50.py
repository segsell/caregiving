# """Solve and simulate the counterfactual with forced care_demand == 2 at age 50."""

# import pickle
# from pathlib import Path
# from typing import Annotated

# import jax
# import jax.numpy as jnp
# import pandas as pd
# import pytask
# import yaml

# from dcegm.pre_processing.setup_model import load_and_setup_model
# from pytask import Product

# from caregiving.config import BLD
# from caregiving.model.state_space import create_state_space_functions
# from caregiving.model.utility.bequest_utility import (
#     create_final_period_utility_functions,
# )
# from caregiving.model.utility.utility_functions_additive import create_utility_functions
# from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
# from caregiving.simulation.simulate import setup_model_for_simulation_baseline
# from caregiving.simulation.simulate_forced_care_demand_at_50 import (
#     simulate_scenario_forced_care_demand_at_50,
# )

# jax.config.update("jax_enable_x64", True)


# def _simulate_forced_care_demand(
#     path_to_solution_model: Path,
#     path_to_options: Path,
#     path_to_params: Path,
#     path_to_baseline_solution: Path,
#     path_to_discrete_states: Path,
#     path_to_wealth: Path,
#     path_to_save_simulated_data: Path,
#     forced_age: int,
# ) -> None:
#     """Helper function to simulate forced care demand at a given age.

#     This uses the baseline solution and initial states, but forces care_demand = 2
#     (care demand and no other supply) and mother_health = 0 (bad health) at the
#     specified age. All other state variables and stochastic processes proceed normally.

#     Args:
#         path_to_solution_model: Path to baseline model structure
#         path_to_options: Path to baseline options
#         path_to_params: Path to model parameters
#         path_to_baseline_solution: Path to baseline solution (value, policy, endog_grid)
#         path_to_discrete_states: Path to baseline initial discrete states
#         path_to_wealth: Path to baseline initial wealth
#         path_to_save_simulated_data: Path to save simulation results
#         forced_age: Age at which to force care_demand == 2 and mother_health == 0
#     """
#     # Load options and params
#     options = pickle.load(path_to_options.open("rb"))
#     params = yaml.safe_load(path_to_params.open("rb"))

#     # Load baseline solution (we use the baseline solution, not a new one)
#     solution_dict = pickle.load(path_to_baseline_solution.open("rb"))

#     # Load initial states and wealth
#     initial_states = pickle.load(path_to_discrete_states.open("rb"))
#     wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

#     # Setup model for simulation (using baseline model structure)
#     model_for_simulation = setup_model_for_simulation_baseline(
#         path_to_model=path_to_solution_model,
#         options=options,
#     )

#     # Simulate with forced care_demand at age 50
#     sim_df = simulate_scenario_forced_care_demand_at_50(
#         model=model_for_simulation,
#         solution=solution_dict,
#         initial_states=initial_states,
#         wealth_agents=wealth_agents,
#         params=params,
#         options=options,
#         seed=options["model_params"]["seed"],
#         forced_age=forced_age,
#     )

#     # Save simulation results
#     sim_df.to_pickle(path_to_save_simulated_data)


# @pytask.mark.counterfactual_forced_care_demand_at_50
# def task_simulate_forced_care_demand_at_45(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
#     path_to_baseline_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_estimated_params.pkl",
#     path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_forced_care_demand_at_45.pkl",
#     forced_age: int = 45,
# ) -> None:
#     """Simulate the counterfactual with forced care_demand == 2 at age 45."""
#     _simulate_forced_care_demand(
#         path_to_solution_model=path_to_solution_model,
#         path_to_options=path_to_options,
#         path_to_params=path_to_params,
#         path_to_baseline_solution=path_to_baseline_solution,
#         path_to_discrete_states=path_to_discrete_states,
#         path_to_wealth=path_to_wealth,
#         path_to_save_simulated_data=path_to_save_simulated_data,
#         forced_age=forced_age,
#     )


# @pytask.mark.counterfactual_forced_care_demand_at_50
# def task_simulate_forced_care_demand_at_50(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
#     path_to_baseline_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_estimated_params.pkl",
#     path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_forced_care_demand_at_50.pkl",
#     forced_age: int = 50,
# ) -> None:
#     """Simulate the counterfactual with forced care_demand == 2 at age 50."""
#     _simulate_forced_care_demand(
#         path_to_solution_model=path_to_solution_model,
#         path_to_options=path_to_options,
#         path_to_params=path_to_params,
#         path_to_baseline_solution=path_to_baseline_solution,
#         path_to_discrete_states=path_to_discrete_states,
#         path_to_wealth=path_to_wealth,
#         path_to_save_simulated_data=path_to_save_simulated_data,
#         forced_age=forced_age,
#     )


# @pytask.mark.counterfactual_forced_care_demand_at_50
# def task_simulate_forced_care_demand_at_54(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
#     path_to_baseline_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_estimated_params.pkl",
#     path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_forced_care_demand_at_54.pkl",
#     forced_age: int = 54,
# ) -> None:
#     """Simulate the counterfactual with forced care_demand == 2 at age 54."""
#     _simulate_forced_care_demand(
#         path_to_solution_model=path_to_solution_model,
#         path_to_options=path_to_options,
#         path_to_params=path_to_params,
#         path_to_baseline_solution=path_to_baseline_solution,
#         path_to_discrete_states=path_to_discrete_states,
#         path_to_wealth=path_to_wealth,
#         path_to_save_simulated_data=path_to_save_simulated_data,
#         forced_age=forced_age,
#     )


# @pytask.mark.counterfactual_forced_care_demand_at_50
# def task_simulate_forced_care_demand_at_58(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
#     path_to_baseline_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_estimated_params.pkl",
#     path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_forced_care_demand_at_58.pkl",
#     forced_age: int = 58,
# ) -> None:
#     """Simulate the counterfactual with forced care_demand == 2 at age 58."""
#     _simulate_forced_care_demand(
#         path_to_solution_model=path_to_solution_model,
#         path_to_options=path_to_options,
#         path_to_params=path_to_params,
#         path_to_baseline_solution=path_to_baseline_solution,
#         path_to_discrete_states=path_to_discrete_states,
#         path_to_wealth=path_to_wealth,
#         path_to_save_simulated_data=path_to_save_simulated_data,
#         forced_age=forced_age,
#     )


# @pytask.mark.counterfactual_forced_care_demand_at_50
# def task_simulate_forced_care_demand_at_62(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
#     path_to_baseline_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_estimated_params.pkl",
#     path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_forced_care_demand_at_62.pkl",
#     forced_age: int = 62,
# ) -> None:
#     """Simulate the counterfactual with forced care_demand == 2 at age 62."""
#     _simulate_forced_care_demand(
#         path_to_solution_model=path_to_solution_model,
#         path_to_options=path_to_options,
#         path_to_params=path_to_params,
#         path_to_baseline_solution=path_to_baseline_solution,
#         path_to_discrete_states=path_to_discrete_states,
#         path_to_wealth=path_to_wealth,
#         path_to_save_simulated_data=path_to_save_simulated_data,
#         forced_age=forced_age,
#     )
