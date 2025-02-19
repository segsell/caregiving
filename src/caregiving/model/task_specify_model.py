"""Specify model for estimation."""

# from pathlib import Path
# from typing import Annotated, Any

# import numpy as np
# import pytask
# import yaml
# from dcegm.pre_processing.setup_model import setup_and_save_model
# from pytask import Product

# from caregiving.config import BLD, SRC
# from caregiving.model.budget import budget_constraint
# from caregiving.model.shared import ALL
# from caregiving.model.state_space import (
#     create_state_space_functions,
#     sparsity_condition,
# )
# from caregiving.model.stochastic_processes import (
#     exog_health_transition_mother_with_survival,
#     prob_full_time_offer,
#     prob_part_time_offer,
# )
# from caregiving.model.utility_functions import (
#     create_final_period_utility_functions,
#     create_utility_functions,
# )
# from caregiving.stochastic_processes.task_create_exog_processes_soep import (
#     task_create_exog_wage,
#     task_create_spousal_income,
# )
# from caregiving.utils import load_dict_from_pickle


# # @pytask.mark.skip(reason="Respecifying model.")
# def task_specify_and_setup_model(
#     path_to_specs: Path = SRC / "model" / "specs.yaml",
#     path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
#     path_to_save: Annotated[Path, Product] = BLD / "model" / "model.pkl",
# ) -> dict[str, Any]:
#     """Generate options and setup model.

#     start_params["sigma"] = specs["income_shock_scale"]

#     """
#     options = get_options_dict(path_to_specs, path_to_exog)

#     return setup_and_save_model(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         path=path_to_save,
#     )


# def get_options_dict(
#     path_to_specs: Path = SRC / "model" / "specs.yaml",
#     path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
# ):
#     specs, _wage_params = load_specs(path_to_specs)

#     exog_params = load_dict_from_pickle(path_to_exog)
#     spousal_params = task_create_spousal_income()

#     # exog_params = {
#     #     "part_time_constant": -2.102635900186225,
#     #     "part_time_not_working_last_period": -1.0115255914421664,
#     #     "part_time_high_education": 0.48013160890989515,
#     #     "part_time_above_retirement_age": -2.110713962590601,
#     #     "full_time_constant": -1.9425261133765783,
#     #     "full_time_not_working_last_period": -2.097935912953995,
#     #     "full_time_high_education": 0.8921957457184644,
#     #     "full_time_above_retirement_age": -3.1212459549307496,
#     # }

#     n_periods = specs["n_periods"]
#     choices = np.arange(len(ALL), dtype=np.uint8)

#     exog_processes = {
#         "part_time_offer": {
#             "states": np.arange(2, dtype=np.uint8),
#             "transition": prob_part_time_offer,
#         },
#         "full_time_offer": {
#             "states": np.arange(2, dtype=np.uint8),
#             "transition": prob_full_time_offer,
#         },
#         "mother_health": {
#             "states": np.arange(3, dtype=np.uint8),
#             "transition": exog_health_transition_mother_with_survival,
#         },
#     }

#     return {
#         "state_space": {
#             "n_periods": n_periods,
#             "choices": choices,
#             "income_shock_scale": specs["income_shock_scale"],
#             "endogenous_states": {
#                 # "high_educ": np.arange(2, dtype=np.uint8),
#                 "experience": np.arange(
#                     start=10,  # 5 * 2
#                     stop=specs["experience_cap"] + 1,
#                     dtype=np.uint8,
#                 ),
#                 "sparsity_condition": sparsity_condition,
#             },
#             "exogenous_processes": exog_processes,
#         },
#         "model_params": specs
#         | exog_params
#         | spousal_params
#         | {"interest_rate": 0.04, "bequest_scale": 1.3},
#     }


# def load_specs(path_to_specs):
#     specs = yaml.safe_load(Path.open(path_to_specs))

#     specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
#     specs["n_choices"] = len(ALL)

#     wage_params = task_create_exog_wage()

#     specs["income_shock_scale"] = wage_params.pop("wage_std_regression_residual")

#     return specs, wage_params


#######################################################################################

import pickle
from pathlib import Path
from typing import Annotated, Any, Dict

import jax.numpy as jnp
import numpy as np
import yaml
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.stochastic_processes.job_transition import (
    job_offer_process_transition,
)
from caregiving.model.stochastic_processes.partner_transition import partner_transition
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.savings_grid import create_savings_grid
from dcegm.pre_processing.setup_model import load_and_setup_model, setup_and_save_model
from dcegm.solve import get_solve_func_for_model


def task_specify_model(
    # load_model=False,
    # model_type="solution",
    path_to_derived_specs: Path = BLD / "model" / "specs" / "specs.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_updated.yaml",
    path_to_save_model: Annotated[Path, Product] = BLD
    / "model"
    / "model_for_solution.pkl",
):
    """Generate model and options dictionaries."""

    with path_to_derived_specs.open("rb") as f:
        specs = pickle.load(f)

    params = yaml.safe_load(path_to_start_params.open("rb"))

    # Assign income shock scale to start_params_all
    params["sigma"] = specs["income_shock_scale"]
    params["interest_rate"] = specs["interest_rate"]
    params["beta"] = specs["discount_factor"]

    # Load specifications
    n_periods = specs["n_periods"]
    choices = np.arange(specs["n_choices"], dtype=int)

    # Create savings grid
    savings_grid = create_savings_grid()

    # Experience grid
    experience_grid = jnp.linspace(0, 1, specs["n_experience_grid_points"])

    options = {
        "state_space": {
            "min_period_batch_segments": [33, 44],
            "n_periods": n_periods,
            "choices": choices,
            "endogenous_states": {
                "education": np.arange(specs["n_education_types"], dtype=int),
                # "sex": np.arange(specs["n_sexes"], dtype=int),
            },
            "exogenous_processes": {
                "job_offer": {
                    "transition": job_offer_process_transition,
                    "states": np.arange(2, dtype=int),
                },
                "partner_state": {
                    "transition": partner_transition,
                    "states": np.arange(specs["n_partner_states"], dtype=int),
                },
            },
            "continuous_states": {
                "wealth": savings_grid,
                "experience": experience_grid,
            },
        },
        "model_params": specs,
    }

    model = setup_and_save_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_save_model,
        sim_model=False,
    )

    print("Model specified.")
    return model, params
