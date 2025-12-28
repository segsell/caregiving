# """Temporary task to recreate simulated data frames without solving.

# This task is needed because the simulation functions changed to correctly create
# the gross_labor_income variable. We need to recreate the data frames without
# running the expensive solve step.
# """

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
# from caregiving.model.state_space_no_care_demand import (
#     create_state_space_functions as create_state_space_functions_no_care_demand,
# )
# from caregiving.model.utility.bequest_utility import (
#     create_final_period_utility_functions,
# )
# from caregiving.model.utility.utility_functions_additive import (  # noqa: E501
#     create_utility_functions,
# )
# from caregiving.model.utility.utility_functions_additive_no_care_demand import (
#     create_utility_functions as create_utility_functions_no_care_demand,
# )
# from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
# from caregiving.model.wealth_and_budget.budget_equation_higher_formal_care_costs import (  # noqa: E501
#     budget_constraint as budget_constraint_higher_formal_care_costs,
# )
# from caregiving.model.wealth_and_budget.budget_equation_lower_formal_care_costs import (  # noqa: E501
#     budget_constraint as budget_constraint_lower_formal_care_costs,
# )
# from caregiving.model.wealth_and_budget.budget_equation_no_care_demand import (
#     budget_constraint as budget_constraint_no_care_demand,
# )
# from caregiving.simulation.simulate import simulate_scenario
# from caregiving.simulation.simulate_no_care_demand import (
#     simulate_scenario_no_care_demand,
# )

# jax.config.update("jax_enable_x64", True)


# @pytask.mark.temp_recreate_data
# def task_recreate_simulated_data_baseline(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_estimated_params: Path = BLD
#     / "model"
#     / "params"
#     / "estimated_params_model.yaml",
#     path_to_discrete_states: Path = BLD  # noqa: E501
#     / "model"
#     / "initial_conditions"
#     / "states.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_load_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_estimated_params.pkl",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_estimated_params.pkl",
# ) -> None:
#     """Recreate baseline simulated data using existing solution."""
#     # Load existing solution
#     solution_dict = pickle.load(path_to_load_solution.open("rb"))

#     # Load parameters and options
#     params = yaml.safe_load(path_to_estimated_params.open("rb"))
#     options = pickle.load(path_to_options.open("rb"))
#     initial_states = pickle.load(path_to_discrete_states.open("rb"))

#     # Load wealth
#     wealth_agents = jnp.array(  # noqa: E501
#         pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze()
#     )

#     # Setup model for simulation
#     model_for_simulation = load_and_setup_model(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         path=path_to_solution_model,
#         sim_model=True,
#     )

#     # Simulate
#     sim_df = simulate_scenario(
#         model_for_simulation,
#         solution=solution_dict,
#         initial_states=initial_states,
#         wealth_agents=wealth_agents,
#         params=params,
#         options=options,
#         seed=options["model_params"]["seed"],
#     )

#     # Save recreated data
#     sim_df.to_pickle(path_to_save_simulated_data)


# @pytask.mark.temp_recreate_data
# def task_recreate_simulated_data_no_care_demand(
#     path_to_solution_model: Path = BLD / "model" / "model_no_care_demand.pkl",
#     path_to_options: Path = BLD / "model" / "options_no_care_demand.pkl",
#     path_to_estimated_params: Path = BLD
#     / "model"
#     / "params"
#     / "estimated_params_model.yaml",
#     path_to_discrete_states: Path = BLD
#     / "model"
#     / "initial_conditions"
#     / "states_no_care_demand.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_load_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_no_care_demand.pkl",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_no_care_demand.pkl",
# ) -> None:
#     """Recreate no care demand simulated data using existing solution."""
#     # Load existing solution
#     solution_dict = pickle.load(path_to_load_solution.open("rb"))

#     # Load parameters and options
#     params = yaml.safe_load(path_to_estimated_params.open("rb"))
#     options = pickle.load(path_to_options.open("rb"))
#     initial_states = pickle.load(path_to_discrete_states.open("rb"))

#     # Load wealth
#     wealth_agents = jnp.array(  # noqa: E501
#         pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze()
#     )

#     # Setup model for simulation using NO CARE DEMAND functions
#     model_for_simulation = load_and_setup_model(
#         options=options,
#         state_space_functions=create_state_space_functions_no_care_demand(),
#         utility_functions=create_utility_functions_no_care_demand(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint_no_care_demand,
#         path=path_to_solution_model,
#         sim_model=True,
#     )

#     # Simulate using no care demand simulation function
#     sim_df = simulate_scenario_no_care_demand(
#         model_for_simulation,
#         solution=solution_dict,
#         initial_states=initial_states,
#         wealth_agents=wealth_agents,
#         params=params,
#         options=options,
#         seed=options["model_params"]["seed"],
#     )

#     # Save recreated data
#     sim_df.to_pickle(path_to_save_simulated_data)


# @pytask.mark.temp_recreate_data
# def task_recreate_simulated_data_job_retention(
#     path_to_solution_model: Path = BLD / "model" / "model_job_retention.pkl",
#     path_to_options: Path = BLD / "model" / "options_job_retention.pkl",
#     path_to_estimated_params: Path = BLD
#     / "model"
#     / "params"
#     / "estimated_params_model.yaml",
#     path_to_discrete_states: Path = BLD
#     / "model"
#     / "initial_conditions"
#     / "states_job_retention.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_load_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_job_retention_estimated_params.pkl",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_job_retention_estimated_params.pkl",
# ) -> None:
#     """Recreate job retention simulated data using existing solution."""
#     # Load existing solution
#     solution_dict = pickle.load(path_to_load_solution.open("rb"))

#     # Load parameters and options
#     params = yaml.safe_load(path_to_estimated_params.open("rb"))
#     options = pickle.load(path_to_options.open("rb"))
#     initial_states = pickle.load(path_to_discrete_states.open("rb"))

#     # Load wealth
#     wealth_agents = jnp.array(  # noqa: E501
#         pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze()
#     )

#     # Setup model for simulation
#     model_for_simulation = load_and_setup_model(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         path=path_to_solution_model,
#         sim_model=True,
#     )

#     # Simulate
#     sim_df = simulate_scenario(
#         model_for_simulation,
#         solution=solution_dict,
#         initial_states=initial_states,
#         wealth_agents=wealth_agents,
#         params=params,
#         options=options,
#         seed=options["model_params"]["seed"],
#     )

#     # Save recreated data
#     sim_df.to_pickle(path_to_save_simulated_data)


# @pytask.mark.temp_recreate_data
# def task_recreate_simulated_data_higher_formal_care_costs(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_estimated_params: Path = BLD
#     / "model"
#     / "params"
#     / "estimated_params_model.yaml",
#     path_to_discrete_states: Path = BLD  # noqa: E501
#     / "model"
#     / "initial_conditions"
#     / "states.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_load_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_higher_formal_care_costs_estimated_params.pkl",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_higher_formal_care_costs_estimated_params.pkl",
# ) -> None:
#     """Recreate higher formal care costs simulated data using existing solution."""
#     # Load existing solution
#     solution_dict = pickle.load(path_to_load_solution.open("rb"))

#     # Load parameters and options
#     params = yaml.safe_load(path_to_estimated_params.open("rb"))
#     options = pickle.load(path_to_options.open("rb"))
#     initial_states = pickle.load(path_to_discrete_states.open("rb"))

#     # Load wealth
#     wealth_agents = jnp.array(  # noqa: E501
#         pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze()
#     )

#     # Setup model for simulation
#     model_for_simulation = load_and_setup_model(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint_higher_formal_care_costs,
#         path=path_to_solution_model,
#         sim_model=True,
#     )

#     # Simulate
#     sim_df = simulate_scenario(
#         model_for_simulation,
#         solution=solution_dict,
#         initial_states=initial_states,
#         wealth_agents=wealth_agents,
#         params=params,
#         options=options,
#         seed=options["model_params"]["seed"],
#     )

#     # Save recreated data
#     sim_df.to_pickle(path_to_save_simulated_data)


# @pytask.mark.temp_recreate_data
# def task_recreate_simulated_data_lower_formal_care_costs(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_estimated_params: Path = BLD
#     / "model"
#     / "params"
#     / "estimated_params_model.yaml",
#     path_to_discrete_states: Path = BLD  # noqa: E501
#     / "model"
#     / "initial_conditions"
#     / "states.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_load_solution: Path = BLD
#     / "solve_and_simulate"
#     / "solution_lower_formal_care_costs_estimated_params.pkl",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data_lower_formal_care_costs_estimated_params.pkl",
# ) -> None:
#     """Recreate lower formal care costs simulated data using existing solution."""
#     # Load existing solution
#     solution_dict = pickle.load(path_to_load_solution.open("rb"))

#     # Load parameters and options
#     params = yaml.safe_load(path_to_estimated_params.open("rb"))
#     options = pickle.load(path_to_options.open("rb"))
#     initial_states = pickle.load(path_to_discrete_states.open("rb"))

#     # Load wealth
#     wealth_agents = jnp.array(  # noqa: E501
#         pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze()
#     )

#     # Setup model for simulation
#     model_for_simulation = load_and_setup_model(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint_lower_formal_care_costs,
#         path=path_to_solution_model,
#         sim_model=True,
#     )

#     # Simulate
#     sim_df = simulate_scenario(
#         model_for_simulation,
#         solution=solution_dict,
#         initial_states=initial_states,
#         wealth_agents=wealth_agents,
#         params=params,
#         options=options,
#         seed=options["model_params"]["seed"],
#     )

#     # Save recreated data
#     sim_df.to_pickle(path_to_save_simulated_data)
