"""Specify model for estimation and simulation."""

from pathlib import Path
from typing import Annotated, Any

import numpy as np
import yaml
from pytask import Product

from dcegm.pre_processing.setup_model import setup_and_save_model
from elder_care.config import BLD, SRC
from elder_care.exogenous_processes.task_create_exog_processes_soep import (
    task_create_exog_wage,
)
from elder_care.model.budget import budget_constraint
from elder_care.model.exogenous_processes import (
    prob_full_time_offer,
    prob_part_time_offer,
)
from elder_care.model.shared import ALL
from elder_care.model.state_space import (
    create_state_space_functions,
    sparsity_condition,
)
from elder_care.model.utility_functions import (
    create_final_period_utility_functions,
    create_utility_functions,
)


# @pytask.mark.skip(reason="Respecifying model.")
def task_specify_and_setup_model(
    path_to_specs: Path = SRC / "model" / "specs.yaml",
    # path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
    path_to_save: Annotated[Path, Product] = BLD / "model" / "model.pkl",
) -> dict[str, Any]:
    """Generate options and setup model.

    start_params["sigma"] = specs["income_shock_scale"]

    """
    options = get_options_dict(path_to_specs)

    return setup_and_save_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_save,
    )


def get_options_dict(
    path_to_specs: Path = SRC / "model" / "specs.yaml",
    # path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
):

    specs, wage_params = load_specs(path_to_specs)

    # exog_params = load_dict_from_pickle(path_to_exog)

    # exog_params = {
    #     "part_time_constant": -2.102635900186225,
    #     "part_time_not_working_last_period": -1.0115255914421664,
    #     "part_time_high_education": 0.48013160890989515,
    #     "part_time_above_retirement_age": -2.110713962590601,
    #     "full_time_constant": -1.9425261133765783,
    #     "full_time_not_working_last_period": -2.097935912953995,
    #     "full_time_high_education": 0.8921957457184644,
    #     "full_time_above_retirement_age": -3.1212459549307496,
    # }

    n_periods = specs["n_periods"]
    choices = np.arange(len(ALL), dtype=np.uint8)

    exog_processes = {
        "part_time_offer": {
            "states": np.arange(2, dtype=np.uint8),
            "transition": prob_part_time_offer,
        },
        "full_time_offer": {
            "states": np.arange(2, dtype=np.uint8),
            "transition": prob_full_time_offer,
        },
        # "mother_health": {
        #     "states": np.arange(3, dtype=np.int8),
        #     "transition": exog_health_transition_mother_with_survival,
        # },
    }

    return {
        "state_space": {
            "n_periods": n_periods,
            "choices": choices,
            "income_shock_scale": specs["income_shock_scale"],
            "endogenous_states": {
                "high_educ": np.arange(2, dtype=np.uint8),
                "experience": np.arange(
                    start=10,  # 5 * 2
                    stop=specs["experience_cap"] + 1,
                    dtype=np.uint8,
                ),
                "sparsity_condition": sparsity_condition,
            },
            "exogenous_processes": exog_processes,
        },
        "model_params": specs | {"interest_rate": 0.04, "bequest_scale": 1.3},
    }


def load_specs(path_to_specs):
    specs = yaml.safe_load(Path.open(path_to_specs))

    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
    specs["n_choices"] = len(ALL)

    wage_params = task_create_exog_wage()

    specs["income_shock_scale"] = wage_params.pop("wage_std_regression_residual")

    return specs, wage_params
