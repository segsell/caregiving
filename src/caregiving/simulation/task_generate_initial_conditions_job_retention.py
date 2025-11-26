"""Initial conditions for the job retention simulation."""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
import yaml
from dcegm.pre_processing.setup_model import load_model_dict
from dcegm.wealth_correction import adjust_observed_wealth
from pytask import Product
from scipy import stats
from sklearn.neighbors import KernelDensity

from caregiving.config import BLD
from caregiving.model.shared import (
    ALL_NO_CARE,
    INITIAL_CONDITIONS_AGE_HIGH,
    INITIAL_CONDITIONS_AGE_LOW,
    INITIAL_CONDITIONS_COHORT_HIGH,
    INITIAL_CONDITIONS_COHORT_LOW,
    MOTHER,
    PARENT_DEAD,
    SEX,
    WORK_AND_UNEMPLOYED_NO_CARE,
)
from caregiving.model.state_space_job_retention import create_state_space_functions
from caregiving.model.stochastic_processes.job_transition_job_retention import (
    job_offer_process_transition_with_job_retention,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.utils import table


def task_generate_start_states_for_solution_job_retention(  # noqa: PLR0915
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_lifetable: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "lifetable.csv",
    path_to_health_sample: Path = BLD
    / "data"
    / "health_transition_estimation_sample_good_medium_bad.pkl",
    path_to_options: Path = BLD / "model" / "options_job_retention.pkl",
    path_to_model: Path = BLD / "model" / "model_job_retention.pkl",
    path_to_start_params: Path = BLD
    / "model"
    / "params"
    / "start_params_model_job_retention.yaml",
    path_to_save_discrete_states: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "states_job_retention.pkl",
    path_to_save_wealth: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "wealth_job_retention.csv",
) -> None:
    """Generate initial conditions for job retention model simulation.

    This function creates initial conditions specifically for the job retention
    counterfactual model, including the job_before_caregiving state variable.
    """

    observed_data = pd.read_csv(path_to_sample, index_col=[0])
    lifetable = pd.read_csv(path_to_lifetable)
    health_sample = pd.read_pickle(path_to_health_sample)

    options = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))

    model = load_model_dict(
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

    # =================================================================================
    # Static state variables
    sex_data = observed_data.loc[observed_data["sex"] == SEX]

    sister_cohort = sex_data.loc[
        (sex_data["gebjahr"] >= INITIAL_CONDITIONS_COHORT_LOW)
        & (sex_data["gebjahr"] <= INITIAL_CONDITIONS_COHORT_HIGH)
        & (sex_data["age"] >= INITIAL_CONDITIONS_AGE_LOW)
        & (sex_data["age"] <= INITIAL_CONDITIONS_AGE_HIGH)
    ].copy()
    # The fact that a woman has obtained higher education correlates with the
    # presence of a sister.
    sister_shares = (
        sister_cohort.groupby("education")["has_sister"]
        .value_counts(normalize=True)  # proportions within each group
        .unstack(fill_value=0)  # make 0/1 the columns
        .rename(columns={0: "no_sister", 1: "has_sister"})
        .sort_index()  # optional: sort education levels
    )

    lifetable = lifetable.sort_values(["sex", "age"])  # ensure order
    lifetable["cum_survival_prob"] = (
        (1 - lifetable["death_prob"]).groupby(lifetable["sex"]).cumprod()
    )

    # =================================================================================

    states_dict = {
        name: start_period_data[name].values
        for name in model["model_structure"]["discrete_states_names"]
        if name
        not in ("mother_health", "care_demand", "care_supply", "job_before_caregiving")
    }

    states_dict["care_demand"] = np.zeros_like(start_period_data["wealth"])
    states_dict["wealth"] = start_period_data["wealth"].values / specs["wealth_unit"]
    states_dict["experience"] = start_period_data["experience"].values
    start_period_data.loc[:, "adjusted_wealth"] = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=params,
        model=model,
    )

    # Generate container
    sex_agents = np.array([], np.uint8)
    education_agents = np.array([], np.uint8)
    for sex_var in range(specs["n_sexes"]):
        if specs["n_sexes"] > 1:
            if sex_var == 0:
                n_agents_sex = n_agents - n_agents // 2
            else:
                n_agents_sex = n_agents // 2
        else:
            n_agents_sex = n_agents

        sex_vars = np.ones(n_agents_sex, np.uint8) * sex_var
        sex_agents = np.append(sex_agents, sex_vars)

        # Restrict start data
        start_data_sex = start_period_data[start_period_data["sex"] == sex_var]

        # Education
        education_shares = (
            start_data_sex["education"].value_counts(normalize=True).sort_index()
        )
        education_agents_sex = np.random.choice(
            specs["n_education_types"], size=n_agents_sex, p=education_shares.values
        )
        education_agents = np.append(education_agents, education_agents_sex)

    # Sister
    has_sister_agents = np.zeros(n_agents, dtype=np.uint8)
    for edu in range(specs["n_education_types"]):
        type_mask = education_agents == edu
        n_agents_edu = np.sum(type_mask)
        if n_agents_edu > 0:
            sister_probs = sister_shares.loc[edu]
            has_sister_agents[type_mask] = np.random.choice(
                [0, 1], size=n_agents_edu, p=sister_probs.values
            )

    # Partner state
    partner_states = np.zeros(n_agents, dtype=np.uint8)

    # Mother health
    mother_health_agents = np.zeros(n_agents, dtype=np.uint8)

    # Job offer
    job_offer_agents = np.ones(n_agents, dtype=np.uint8)

    # Experience and lagged choice
    exp_agents = np.zeros(n_agents, dtype=np.float64)
    lagged_choice = np.zeros(n_agents, dtype=np.uint8)

    # Wealth
    wealth_agents = np.zeros(n_agents, dtype=np.float64)

    for edu in range(specs["n_education_types"]):
        type_mask = education_agents == edu
        n_agents_edu = np.sum(type_mask)
        if n_agents_edu > 0:
            start_data_edu = start_period_data[
                start_period_data["education"] == edu
            ].copy()

            # Experience
            exp_agents[type_mask] = draw_start_experience_dist(
                start_data_edu, n_agents_edu
            )

            # Lagged choice
            lagged_choice[type_mask] = draw_start_lagged_choice_dist(
                start_data_edu, n_agents_edu
            )

            # Wealth
            wealth_agents[type_mask] = draw_start_wealth_dist(
                start_data_edu, n_agents_edu
            )

    # Health
    health_agents = np.zeros(n_agents, dtype=np.uint8)
    for edu in range(specs["n_education_types"]):
        type_mask = education_agents == edu
        n_agents_edu = np.sum(type_mask)
        if n_agents_edu > 0:
            health_probs = (
                health_sample[health_sample["education"] == edu]["health"]
                .value_counts(normalize=True)
                .sort_index()
            )
            health_states_edu = np.random.choice(
                specs["n_health_states"], size=n_agents_edu, p=health_probs.values
            )
            health_agents[type_mask] = health_states_edu

    # Transform it to be between 0 and 1
    exp_agents /= specs["max_exp_diffs_per_period"][0]

    # Set lagged choice to 1(unemployment) if experience is 0
    exp_zero_mask = exp_agents == 0
    lagged_choice[exp_zero_mask] = 1

    n_care = len(specs["caregiving_labels"])
    lagged_choice_model = lagged_choice * n_care

    # Build states including job_before_caregiving for job retention model
    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "education": jnp.array(education_agents, dtype=jnp.uint8),
        "health": jnp.array(health_agents, dtype=jnp.uint8),
        "lagged_choice": jnp.array(lagged_choice_model, dtype=jnp.uint8),
        "already_retired": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "job_before_caregiving": jnp.zeros_like(
            exp_agents, dtype=jnp.uint8
        ),  # Key addition for job retention
        "experience": jnp.array(exp_agents, dtype=jnp.float64),
        "job_offer": jnp.array(job_offer_agents, dtype=jnp.uint8),
        "partner_state": jnp.array(partner_states, dtype=jnp.uint8),
        "has_sister": jnp.array(has_sister_agents, dtype=jnp.uint8),
        "mother_health": jnp.array(mother_health_agents, dtype=jnp.uint8),
        "care_demand": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
    }

    # Save initial discrete states and wealth
    with path_to_save_discrete_states.open("wb") as f:
        pickle.dump(states, f)

    wealth_agents = pd.DataFrame(wealth_agents, columns=["wealth"])
    wealth_agents.to_csv(path_to_save_wealth, index=False)


def draw_start_wealth_dist(start_period_data_edu, n_agents_edu, method="kde"):
    """Draw start wealth distribution."""
    if method == "kde":
        kde = KernelDensity(kernel="gaussian", bandwidth=0.1)
        kde.fit(start_period_data_edu["adjusted_wealth"].values.reshape(-1, 1))
        wealth_agents_edu = kde.sample(n_agents_edu).flatten()
    else:
        wealth_agents_edu = np.random.choice(
            start_period_data_edu["adjusted_wealth"].values, size=n_agents_edu
        )
    return wealth_agents_edu


def draw_start_experience_dist(start_period_data_edu, n_agents_edu):
    """Draw start experience distribution."""
    exp_agents_edu = np.random.choice(
        start_period_data_edu["experience"].values, size=n_agents_edu
    )
    return exp_agents_edu


def draw_start_lagged_choice_dist(start_period_data_edu, n_agents_edu):
    """Draw start lagged choice distribution."""
    lagged_choice_agents_edu = np.random.choice(
        start_period_data_edu["lagged_choice"].values, size=n_agents_edu
    )
    return lagged_choice_agents_edu
