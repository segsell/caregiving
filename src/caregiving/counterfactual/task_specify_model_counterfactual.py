# """Specify model for estimation."""

# import copy
# import pickle
# from pathlib import Path
# from typing import Annotated

# import jax.numpy as jnp
# import numpy as np
# import pytask
# import yaml
# from dcegm.pre_processing.setup_model import setup_and_save_model
# from pytask import Product

# from caregiving.config import BLD
# from caregiving.model.state_space import create_state_space_functions_counterfactual
# from caregiving.model.stochastic_processes.caregiving_transition import (
#     care_demand_transition,
#     care_demand_with_exog_supply_transition,
#     exog_care_supply_transition,
#     health_transition_good_medium_bad,
# )
# from caregiving.model.stochastic_processes.health_transition import (
#     health_transition,
# )
# from caregiving.model.stochastic_processes.job_transition import (
#     job_offer_process_transition,
# )
# from caregiving.model.stochastic_processes.partner_transition
# import partner_transition
# from caregiving.model.utility.bequest_utility import (
#     create_final_period_utility_functions,
# )
# from caregiving.model.utility.utility_functions import create_utility_functions
# from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
# from caregiving.model.wealth_and_budget.savings_grid import create_savings_grid


# @pytask.mark.skip()
# def task_specify_model(
#     # load_model=False,
#     # model_type="solution",
#     path_to_derived_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
#     path_to_start_params: Path =
# BLD / "model" / "params" / "start_params_updated.yaml",
#     path_to_save_options: Annotated[Path, Product] = BLD
#     / "model"
#     / "options_counterfactual.pkl",
#     path_to_save_model: Annotated[Path, Product] = BLD
#     / "model"
#     / "model_for_solution_counterfactual.pkl",
# ):
#     """Generate model and options dictionaries."""

#     with path_to_derived_specs.open("rb") as f:
#         specs = pickle.load(f)

#     params = yaml.safe_load(path_to_start_params.open("rb"))

#     # Assign income shock scale to start_params_all
#     params["sigma"] = float(specs["income_shock_scale"])
#     params["interest_rate"] = float(specs["interest_rate"])
#     params["beta"] = float(specs["discount_factor"])

#     # Load specifications
#     n_periods = specs["n_periods"]
#     choices = np.arange(specs["n_choices"], dtype=int)

#     # Savings grid
#     savings_grid = create_savings_grid()

#     # Experience grid
#     experience_grid = jnp.linspace(0, 1, specs["n_experience_grid_points"])

#     specs = copy.deepcopy(specs)
#     specs["formal_care_costs"] = 0

#     options = {
#         "state_space": {
#             "min_period_batch_segments": [33, 44],
#             "n_periods": n_periods,
#             "choices": choices,
#             "endogenous_states": {
#                 "education": np.arange(specs["n_education_types"], dtype=int),
#                 # "sex": np.arange(specs["n_sexes"], dtype=int),
#                 "already_retired": np.arange(2, dtype=int),
#                 "has_sister": np.arange(2, dtype=int),
#             },
#             "exogenous_processes": {
#                 "job_offer": {
#                     "transition": job_offer_process_transition,
#                     "states": np.arange(2, dtype=int),
#                 },
#                 "partner_state": {
#                     "transition": partner_transition,
#                     "states": np.arange(specs["n_partner_states"], dtype=int),
#                 },
#                 "health": {
#                     "transition": health_transition,
#                     "states": np.arange(specs["n_health_states"], dtype=int),
#                 },
#                 "mother_health": {
#                     "transition": health_transition_good_medium_bad,
#                     "states": np.arange(specs["n_health_states_three"], dtype=int),
#                 },
#                 "care_demand": {
#                     "transition": care_demand_with_exog_supply_transition,
#                     "states": np.arange(2, dtype=int),
#                 },
#                 # "care_demand": {
#                 #     "transition": care_demand_transition,
#                 #     "states": np.arange(2, dtype=int),
#                 # },
#                 # "care_supply": {
#                 #     "transition": exog_care_transition,
#                 #     "states": np.arange(2, dtype=int),
#                 # },
#             },
#             "continuous_states": {
#                 "wealth": savings_grid,
#                 "experience": experience_grid,
#             },
#         },
#         "model_params": specs,
#     }
#     pickle.dump(options, path_to_save_options.open("wb"))

#     model = setup_and_save_model(
#         options=options,
#         state_space_functions=create_state_space_functions_counterfactual(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         # shock_functions=shock_function_dict(),
#         path=path_to_save_model,
#         sim_model=False,
#     )

#     print("Model specified.")
#     return model, params
