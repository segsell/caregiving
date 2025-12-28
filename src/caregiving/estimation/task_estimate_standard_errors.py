# """Estimate standard errors for model parameters."""

# import pickle
# from pathlib import Path
# from typing import Annotated, Any, Callable, Dict

# import jax.numpy as jnp
# import numpy as np
# import pandas as pd
# import pytask
# import yaml

# # TEMPORARILY COMMENTED OUT - TO BE RE-ENABLED STEP BY STEP
# raise NotImplementedError(
#     "This module is temporarily commented out due to dcegm API changes. "
#     "Need to update imports to use new dcegm API."
# )
# # from dcegm.pre_processing.setup_model import load_and_setup_model
# # from dcegm.solve import get_solve_func_for_model
# # from pytask import Product

# # from caregiving.config import BLD
# # from caregiving.estimation.prepare_estimation import (
# #     load_and_setup_full_model_for_solution,
# # )
# # from caregiving.estimation.standard_errors import get_analytical_standard_errors
# # from caregiving.model.state_space import create_state_space_functions
# # from caregiving.model.utility.bequest_utility import (
# #     create_final_period_utility_functions,
# # )
# # from caregiving.model.utility.utility_functions_additive import create_utility_functions
# # from caregiving.model.wealth_and_budget.budget_equation import budget_constraint


# # @pytask.mark.skip(reason="Multiple solve and simulate calls needed")
# # @pytask.mark.standard_errors
# # def task_estimate_standard_errors(
# #     path_to_options: Path = BLD / "model" / "options.pkl",
# #     path_to_model: Path = BLD / "model" / "model_for_solution.pkl",
# #     path_to_estimated_params: Path = BLD
# #     / "model"
# #     / "params"
# #     / "estimated_params_model.yaml",
# #     path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
# #     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
# #     path_to_empirical_moments: Path = BLD / "moments" / "moments_full.csv",
# #     path_to_empirical_variance: Path = BLD / "moments" / "variances_full.csv",
# #     path_to_save_standard_errors: Annotated[Path, Product] = BLD
# #     / "estimation"
# #     / "standard_errors.csv",
# # ) -> None:
# #     """Estimate analytical standard errors for model parameters.

# #     This function computes analytical standard errors using the asymptotic
# #     distribution theory for simulated method of moments estimators.

# #     Args:
# #         path_to_options: Path to model options
# #         path_to_model: Path to model for solution
# #         path_to_estimated_params: Path to estimated parameters
# #         path_to_discrete_states: Path to discrete initial states
# #         path_to_wealth: Path to initial wealth data
# #         path_to_empirical_moments: Path to empirical moments
# #         path_to_empirical_variance: Path to empirical variances
# #         path_to_save_standard_errors: Path to save standard errors
# #     """
# #     options = pickle.load(path_to_options.open("rb"))
# #     params = yaml.safe_load(path_to_estimated_params.open("rb"))

# #     model_for_solution = load_and_setup_full_model_for_solution(
# #         options, path_to_model=path_to_model
# #     )

# #     # Load empirical moments
# #     emp_moments = pd.read_csv(path_to_empirical_moments, index_col=[0]).squeeze(
# #         "columns"
# #     )
# #     emp_var = pd.read_csv(path_to_empirical_variance, index_col=[0]).squeeze("columns")

# #     # Load initial conditions
# #     initial_states = pickle.load(path_to_discrete_states.open("rb"))
# #     wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

# #     solve_func = get_solve_func_for_model(model_for_solution)

# #     model_for_simulation = load_and_setup_model(
# #         options=options,
# #         state_space_functions=create_state_space_functions(),
# #         utility_functions=create_utility_functions(),
# #         utility_functions_final_period=create_final_period_utility_functions(),
# #         budget_constraint=budget_constraint,
# #         path=path_to_model,
# #         sim_model=True,
# #     )

# #     # Compute standard errors
# #     standard_errors = get_analytical_standard_errors(
# #         params=params,
# #         options=options,
# #         emp_moments=jnp.array(emp_moments.values),
# #         emp_var=jnp.array(emp_var.values),
# #         model_loaded=model_for_simulation,
# #         solve_func=solve_func,
# #         initial_states=initial_states,
# #         wealth_agents=wealth_agents,
# #     )

# #     # Save results
# #     results_df = pd.DataFrame(
# #         {
# #             "parameter": list(params.keys()),
# #             "value": list(params.values()),
# #             "standard_error": standard_errors,
# #         }
# #     )
# #     results_df.to_csv(path_to_save_standard_errors, index=False)
