from pathlib import Path

import jax
import numpy as np
import pytask
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.solve import get_solve_func_for_model

from caregiving.config import BLD, SRC
from caregiving.model.budget import budget_constraint, create_savings_grid
from caregiving.model.state_space import (
    create_state_space_functions,
    sparsity_condition,
)
from caregiving.model.stochastic_processes import (
    exog_health_transition_mother,
    prob_exog_care_demand,
    prob_full_time_offer,
    prob_part_time_offer,
    prob_survival_mother,
)
from caregiving.model.task_specify_model import load_specs
from caregiving.model.utility_functions import (
    create_final_period_utility_functions,
    create_utility_functions,
)
from caregiving.utils import load_dict_from_pickle

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


@pytask.mark.skip()
def task_solve_trial_run(
    path_to_specs: Path = SRC / "model" / "specs.yaml",
    path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
    path_to_model: Path = BLD / "model" / "model_exp_without_father_correct_util.pkl",
):
    specs, wage_params = load_specs(path_to_specs)

    exog_params = load_dict_from_pickle(path_to_exog)

    n_periods = specs["n_periods"]
    choices = np.arange(specs["n_choices"], dtype=np.int8)

    more_exog_params = {
        "part_time_constant": -2.568584,
        "part_time_not_working_last_period": 0.3201395,
        "part_time_high_education": 0.1691369,
        "part_time_above_retirement_age": -1.9976496,
        "full_time_constant": -2.445238,
        "full_time_not_working_last_period": -0.9964007,
        "full_time_high_education": 0.3019138,
        "full_time_above_retirement_age": -2.6571659,
    }

    exog_processes = {
        "part_time_offer": {
            "states": np.arange(2, dtype=np.int8),
            "transition": prob_part_time_offer,
        },
        "full_time_offer": {
            "states": np.arange(2, dtype=np.int8),
            "transition": prob_full_time_offer,
        },
        "care_demand": {
            "states": np.arange(2, dtype=np.int8),
            "transition": prob_exog_care_demand,
        },
        "mother_alive": {
            "states": np.arange(2, dtype=np.int8),
            "transition": prob_survival_mother,
        },
        "mother_health": {
            "states": np.arange(3, dtype=np.int8),
            "transition": exog_health_transition_mother,
        },
    }

    options = {
        "state_space": {
            "n_periods": n_periods,
            "choices": choices,
            "income_shock_scale": specs["income_shock_scale"],
            "taste_shock_scale": specs["lambda"],
            "endogenous_states": {
                "high_educ": np.arange(2, dtype=np.uint8),
                "has_sibling": np.arange(2, dtype=np.uint8),
                "experience": np.arange(
                    stop=specs["experience_cap"] + 1,
                    step=1,
                    dtype=np.uint8,
                ),
                "sparsity_condition": sparsity_condition,
            },
            "exogenous_processes": exog_processes,
        },
        "model_params": specs
        | wage_params
        | exog_params
        | more_exog_params
        | {"interest_rate": 0.04, "bequest_scale": 1.3},
    }

    params = {
        "beta": 0.95,
        "rho": 2,
        "disutility_part_time": -2,
        "disutility_full_time": -4,
        "utility_informal_care_parent_good_health": 1,
        "utility_informal_care_parent_medium_health": 1,
        "utility_informal_care_parent_bad_health": 1,
        "utility_formal_care_parent_good_health": 1,
        "utility_formal_care_parent_medium_health": 1,
        "utility_formal_care_parent_bad_health": 1,
        "utility_combination_care_parent_good_health": 1,
        "utility_combination_care_parent_medium_health": 1,
        "utility_combination_care_parent_bad_health": 1,
        "utility_informal_care_sibling": 1,
        "utility_formal_care_sibling": 1,
        "utility_combination_care_sibling": 1,
    }

    model_loaded = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_model,
    )

    exog_savings_grid = create_savings_grid()

    func = get_solve_func_for_model(
        model=model_loaded,
        exog_savings_grid=exog_savings_grid,
        options=options,
    )

    return func(params)
