"""Initial conditions for the simulation."""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.wealth_correction import adjust_observed_wealth
from pytask import Product
from scipy import stats
from sklearn.neighbors import KernelDensity

from caregiving.config import BLD
from caregiving.model.shared import SEX
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.stochastic_processes.job_transition import (
    job_offer_process_transition_initial_conditions,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.utils import table


def task_generate_start_states_for_solution(  # noqa: PLR0915
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_save_discrete_states: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "states.pkl",
    path_to_save_wealth: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "wealth.csv",
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
    health_agents = np.empty(n_agents, np.uint8)
    job_offer_agents = np.empty(n_agents, np.uint8)

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

        # Generate job offer probabilities
        job_offer_probs = job_offer_process_transition_initial_conditions(
            params=params,
            options=specs,
            # sex=jnp.ones_like(lagged_choice_edu) * sex_var,
            education=jnp.ones_like(lagged_choice_edu) * edu,
            period=jnp.zeros_like(lagged_choice_edu),
            choice=lagged_choice_edu,
        ).T
        # Job offer probs is n_agents x 2. Choose for each row the job offer state
        # with np random choice
        job_offer_edu = np.array(
            [np.random.choice(a=len(p), p=p) for p in job_offer_probs]
        )
        job_offer_agents[type_mask] = job_offer_edu

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

        # Generate health states
        empirical_health_probs = start_period_data_edu["health"].value_counts(
            normalize=True
        )
        health_probs = pd.Series(
            index=np.arange(specs["n_health_states"]), data=0, dtype=float
        )
        health_probs.update(empirical_health_probs)
        health_states_edu = np.random.choice(
            specs["n_health_states"], size=n_agents_edu, p=health_probs.values
        )
        health_agents[type_mask] = health_states_edu

    # Transform it to be between 0 and 1
    exp_agents /= specs["max_exp_diffs_per_period"][0]

    # Set lagged choice to 1(unemployment) if experience is 0
    exp_zero_mask = exp_agents == 0
    lagged_choice[exp_zero_mask] = 1

    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "education": jnp.array(education_agents, dtype=jnp.uint8),
        "health": jnp.array(health_agents, dtype=jnp.uint8),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.uint8),
        "already_retired": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "experience": jnp.array(exp_agents, dtype=jnp.float64),
        "job_offer": jnp.array(job_offer_agents, dtype=jnp.uint8),
        "partner_state": jnp.array(partner_states, dtype=jnp.uint8),
    }

    # Save initial discrete states and wealth
    with path_to_save_discrete_states.open("wb") as f:
        pickle.dump(states, f)

    wealth_agents = pd.DataFrame(wealth_agents, columns=["wealth"])
    wealth_agents.to_csv(path_to_save_wealth, index=False)


def draw_start_wealth_dist(start_period_data_edu, n_agents_edu, method="kde"):
    """Draws samples from the starting wealth distribution using different methods.

    Methods:
    - "uniform": Uniform sampling between the 30th and 70th percentiles.
    - "lognormal": Fit a shifted lognormal distribution and sample from it.
    - "kde": Kernel Density Estimation (KDE) based sampling.
    - "pareto": Fit a shifted Pareto distribution and sample from it.

    Parameters:
        start_period_data_edu (pd.DataFrame): Data containing "adjusted_wealth".
        n_agents_edu (int): Number of samples to draw.
        method (str): Sampling method ("uniform", "lognormal", "kde", "pareto").

    Returns:
        np.ndarray: Sampled wealth values.
    """

    wealth_data = start_period_data_edu["adjusted_wealth"]

    if method == "uniform":
        # Existing uniform sampling between 30th and 70th quantiles
        wealth_start = np.random.uniform(
            wealth_data.quantile(0.3), wealth_data.quantile(0.7), n_agents_edu
        )

    elif method == "lognormal":
        # Fit a shifted lognormal distribution
        min_val = wealth_data.min()
        shifted_data = wealth_data - min_val + 1e-6  # Avoid log(0)
        shape, loc, scale = stats.lognorm.fit(
            shifted_data, floc=0
        )  # Fix location at zero
        samples = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=n_agents_edu)
        wealth_start = samples + min_val - 1e-6  # Shift back

    elif method == "kde":
        # Kernel Density Estimation (KDE) sampling
        kde = KernelDensity(kernel="gaussian", bandwidth=0.1 * wealth_data.std()).fit(
            wealth_data.values.reshape(-1, 1)
        )
        wealth_start = kde.sample(n_agents_edu).flatten()

    elif method == "pareto":
        # Fit a Pareto-like distribution (Shifted Pareto)
        min_val = wealth_data.min()
        shifted_data = wealth_data - min_val + 1e-6  # Shift data to avoid 0
        shape, loc, scale = stats.pareto.fit(
            shifted_data, floc=0
        )  # Fix location at zero
        samples = stats.pareto.rvs(shape, loc=loc, scale=scale, size=n_agents_edu)
        wealth_start = samples + min_val - 1e-6  # Shift back

    wealth_start_clipped = np.clip(
        wealth_start,
        a_min=wealth_data.quantile(0),  # wealth_data.min()
        a_max=wealth_data.quantile(0.98),
    )

    return wealth_start_clipped
