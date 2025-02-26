"""Initial conditions for the simulation."""

import pickle
from pathlib import Path
from typing import Annotated, Any, Dict

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.wealth_correction import adjust_observed_wealth
from pytask import Product
from scipy.stats import pareto

from caregiving.config import BLD, SRC
from caregiving.model.shared import SEX
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint


def task_generate_start_states_for_solution(  # noqa: PLR0915
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_save_discrete_states: Annotated[Path, Product] = BLD
    / "model"
    / "states.pkl",
    path_to_save_wealth: Annotated[Path, Product] = BLD / "model" / "wealth.csv",
) -> None:
    sex_var = SEX

    observed_data = pd.read_csv(path_to_sample, index_col=[0])

    options = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))

    model = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_model,
        sim_model=False,
    )

    specs = options["model_params"]
    n_agents = specs["n_agents"]
    seed = specs["seed"]

    np.random.seed(seed)

    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()
    start_period_data = start_period_data[start_period_data["wealth"].notnull()].copy()

    states_dict = {
        name: start_period_data[name].values
        for name in model["model_structure"]["discrete_states_names"]
    }

    states_dict["wealth"] = start_period_data["wealth"].values / specs["wealth_unit"]
    states_dict["experience"] = start_period_data["experience"].values
    start_period_data.loc[:, "adjusted_wealth"] = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=params,
        model=model,
    )

    # # Generate container
    # sex_agents = np.array([], np.uint8)
    # education_agents = np.array([], np.uint8)
    # for sex_var in range(specs["n_sexes"]):
    #     if specs["n_sexes"] > 1:
    #         if sex_var == 0:
    #             n_agents_sex = n_agents - n_agents // 2
    #         else:
    #             n_agents_sex = n_agents // 2
    #     else:
    #         n_agents_sex = n_agents

    #     sex_vars = np.ones(n_agents_sex, np.uint8) * sex_var
    #     sex_agents = np.append(sex_agents, sex_vars)

    #     # Restrict start data
    #     start_data_sex = start_period_data[start_period_data["sex"] == sex_var]

    #     # Generate education level
    #     edu_shares = start_data_sex["education"].value_counts(normalize=True)
    #     n_agents_edu_types = np.round(edu_shares.sort_index() * n_agents_sex).astype(
    #         int
    #     )

    #     # Generate education array
    #     edu_agents_per_sex = np.repeat(edu_shares.index, n_agents_edu_types)
    #     education_agents = np.append(education_agents, edu_agents_per_sex)

    # All agents have sex == 1
    sex_agents = np.full(n_agents, sex_var, dtype=np.uint8)

    # Restrict to start data for sex == 1
    start_data_sex = start_period_data[start_period_data["sex"] == 1]

    # Generate education distribution
    edu_shares = start_data_sex["education"].value_counts(normalize=True).sort_index()
    n_agents_edu_types = np.round(edu_shares * n_agents).astype(int)

    # Create the education array
    education_agents = np.repeat(edu_shares.index, n_agents_edu_types)

    # Generate containers
    wealth_agents = np.empty(n_agents, np.float64)
    exp_agents = np.empty(n_agents, np.float64)
    lagged_choice = np.empty(n_agents, np.uint8)
    partner_states = np.empty(n_agents, np.uint8)

    # for sex_var in range(specs["n_sexes"]):
    for edu in range(specs["n_education_types"]):
        type_mask = (sex_agents == sex_var) & (education_agents == edu)
        start_period_data_edu = start_period_data[
            (start_period_data["sex"] == sex_var)
            & (start_period_data["education"] == edu)
        ]

        n_agents_edu = np.sum(type_mask)

        # Restrict dataset on education level

        wealth_start_edu = draw_start_wealth_dist(start_period_data_edu, n_agents_edu)
        wealth_agents[type_mask] = wealth_start_edu

        # Generate type specific initial experience distribution
        exp_max_edu = start_period_data_edu["experience"].max()
        empirical_exp_probs = start_period_data_edu["experience"].value_counts(
            normalize=True
        )
        exp_probs = pd.Series(index=np.arange(0, exp_max_edu + 1), data=0, dtype=float)
        exp_probs.update(empirical_exp_probs)
        exp_agents[type_mask] = np.random.choice(
            exp_max_edu + 1, size=n_agents_edu, p=exp_probs.values
        )

        # Generate type specific initial lagged choice distribution
        empirical_lagged_choice_probs = start_period_data_edu[
            "lagged_choice"
        ].value_counts(normalize=True)
        lagged_choice_probs = pd.Series(
            index=np.arange(0, specs["n_choices"]), data=0, dtype=float
        )
        lagged_choice_probs.update(empirical_lagged_choice_probs)
        lagged_choice_edu = np.random.choice(
            specs["n_choices"], size=n_agents_edu, p=lagged_choice_probs.values
        )
        lagged_choice[type_mask] = lagged_choice_edu

        # Get type specific partner states
        empirical_partner_probs = start_period_data_edu["partner_state"].value_counts(
            normalize=True
        )
        partner_probs = pd.Series(
            index=np.arange(specs["n_partner_states"]), data=0, dtype=float
        )
        partner_probs.update(empirical_partner_probs)
        partner_states_edu = np.random.choice(
            specs["n_partner_states"], size=n_agents_edu, p=partner_probs.values
        )
        partner_states[type_mask] = partner_states_edu

    # Transform it to be between 0 and 1
    exp_agents /= specs["max_exp_diffs_per_period"][0]

    # Set lagged choice to 1(unemployment) if experience is 0
    exp_zero_mask = exp_agents == 0
    lagged_choice[exp_zero_mask] = 1

    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "experience": jnp.array(exp_agents, dtype=jnp.float64),
        "education": jnp.array(education_agents, dtype=jnp.uint8),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.uint8),
        "job_offer": jnp.ones_like(exp_agents, dtype=jnp.uint8),
        "partner_state": jnp.array(partner_states, dtype=jnp.uint8),
    }

    # Save initial discrete states and wealth
    with path_to_save_discrete_states.open("wb") as f:
        pickle.dump(states, f)

    wealth_agents = pd.DataFrame(wealth_agents, columns=["wealth"])
    wealth_agents.to_csv(path_to_save_wealth, index=False)

    return states, wealth_agents


def draw_start_wealth_dist(start_period_data_edu, n_agents_edu):
    """Draw uniform wealth distribution from 30 to 70th quantile."""
    wealth_start_edu = np.random.uniform(
        start_period_data_edu["adjusted_wealth"].quantile(0.3),
        start_period_data_edu["adjusted_wealth"].quantile(0.7),
        n_agents_edu,
    )
    return wealth_start_edu
    # if edu == 1:
    #     # Filter out high outliers for high
    #     wealth_edu = wealth_edu[wealth_edu < np.quantile(wealth_edu, 0.85)]
    #
    # median = np.quantile(wealth_edu, 0.5)
    # fscale = min_unemployment_benefits - 0.01
    #
    # # # Adjust shape to ensure the median is as desired
    # # adjusted_shape = np.log(2) / np.log(median / fscale)
    #
    # # Estimate pareto wealth distribution.
    # # Take single unemployment benefits as minimum.
    # shape_param, loc_param, scale_param = pareto.fit(wealth_edu, fscale=fscale)
    #
    # wealth_agents[edu_agents == edu] = pareto.rvs(
    # shape_param, loc=loc_param, scale=fscale, size=n_agents_edu
    # )
    # breakpoint()
