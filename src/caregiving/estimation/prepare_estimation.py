"""Functions for preparing estimation setup."""

from typing import Any, Dict

import numpy as np
import pandas as pd
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.wealth_correction import adjust_observed_wealth

from caregiving.model.shared import RETIREMENT
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint


def load_and_setup_full_model_for_solution(options, path_to_model) -> Dict[str, Any]:
    """Load and setup full caregiving model for solution."""

    model_full = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_model,
        sim_model=False,
    )

    return model_full


def load_and_prep_data(data_emp, model, start_params, drop_retirees=True):
    """Load and prepare empirical data to compare with simulated data."""

    specs = model["options"]["model_params"]
    # We need to filter observations in period 0 because of job offer
    # weighting from last period
    data_emp = data_emp[data_emp["period"] > 0].copy()

    # Also already retired individuals hold no identification
    if drop_retirees:
        data_emp = data_emp[~data_emp["lagged_choice"].isin(RETIREMENT.tolist())]

    data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_emp.loc[:, "age_bin"] = np.floor(data_emp["age"] / 10)
    data_emp.loc[data_emp["age_bin"] > 6, "age_bin"] = 6  # noqa: PLR2004

    age_bin_av_size = data_emp.shape[0] / data_emp["age_bin"].nunique()
    data_emp.loc[:, "age_weights"] = 1.0
    data_emp.loc[:, "age_weights"] = age_bin_av_size / data_emp.groupby("age_bin")[
        "age_weights"
    ].transform("sum")

    # Transform experience
    max_init_exp = specs["max_exp_diffs_per_period"][data_emp["period"].values]
    exp_denominator = data_emp["period"].values + max_init_exp
    data_emp["experience"] = data_emp["experience"] / exp_denominator

    # We can adjust wealth outside, as it does not depend on estimated parameters
    # (only on interest rate)
    # Now transform for dcegm
    states_dict = {
        name: data_emp[name].values
        for name in model["model_structure"]["discrete_states_names"]
        if name not in ("mother_health", "care_demand", "care_supply")
    }
    states_dict["care_demand"] = np.zeros_like(data_emp["wealth"])
    states_dict["experience"] = data_emp["experience"].values
    states_dict["wealth"] = data_emp["wealth"].values / specs["wealth_unit"]

    adjusted_wealth = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=start_params,
        model=model,
    )
    data_emp.loc[:, "adjusted_wealth"] = adjusted_wealth
    states_dict["wealth"] = data_emp["adjusted_wealth"].values

    return data_emp, states_dict
