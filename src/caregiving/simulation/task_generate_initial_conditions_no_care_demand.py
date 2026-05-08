"""Initial conditions for the simulation (no-care-demand counterfactual)."""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
import yaml
from dcegm.asset_correction import adjust_observed_assets
from pytask import Product

import dcegm
from caregiving.config import BLD
from caregiving.model.experience_baseline_model import scale_experience_years
from caregiving.model.shared import (
    MOTHER,
    PARENT_HEALTH_DEAD,
    PARENT_LONGER_DEAD,
    SEX,
)
from caregiving.model.state_space_no_care_demand import create_state_space_functions
from caregiving.model.stochastic_processes.job_transition import (
    job_offer_process_transition_initial_conditions,
)
from caregiving.model.task_specify_model_no_care_demand import (
    create_stochastic_states_transitions,
)
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive_no_care_demand import (
    create_utility_functions,
)
from caregiving.model.wealth_and_budget.budget_equation_no_care_demand import (
    budget_constraint,
)
from caregiving.simulation.task_generate_initial_conditions import (
    draw_mother_health,
    draw_start_wealth_dist,
)


@pytask.mark.skip()
def task_generate_start_states_for_solution_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_lifetable: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "lifetable.csv",
    path_to_health_sample: Path = BLD
    / "data"
    / "health_transition_estimation_sample_good_medium_bad.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config_no_care_demand.pkl",
    path_to_model: Path = BLD / "model" / "model_no_care_demand.pkl",
    path_to_start_params: Path = BLD
    / "model"
    / "params"
    / "start_params_model_no_care_demand.yaml",
    path_to_save_discrete_states: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "initial_states_no_care_demand.pkl",
) -> None:
    sex_var = SEX

    observed_data = pd.read_csv(path_to_sample, index_col=[0])
    lifetable = pd.read_csv(path_to_lifetable)
    health_sample = pd.read_pickle(path_to_health_sample)

    specs = pickle.load(path_to_specs.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model_class = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_model,
    )
    model_specs = model_class.model_specs
    model_structure = model_class.model_structure
    n_agents = model_specs["n_agents"]
    seed = model_specs["seed"]

    np.random.seed(seed)

    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()
    start_period_data = start_period_data[start_period_data["wealth"].notnull()].copy()

    # =================================================================================
    # Static state variables
    lifetable = lifetable.sort_values(["sex", "age"])  # ensure order
    lifetable["cum_survival_prob"] = (
        (1 - lifetable["death_prob"]).groupby(lifetable["sex"]).cumprod()
    )

    # =================================================================================

    states_dict = {
        name: start_period_data[name].values
        for name in model_structure["discrete_states_names"]
        if name
        not in (
            "mother_dead",
            "caregiving_type",
        )
    }

    states_dict["experience"] = start_period_data["experience"].values
    # Initialize mother_dead to 0 (alive) for all agents at initial period
    states_dict["mother_dead"] = np.zeros_like(
        start_period_data["wealth"], dtype=np.uint8
    )
    states_dict["assets_begin_of_period"] = (
        start_period_data["wealth"].values / specs["wealth_unit"]
    )
    start_period_data.loc[:, "adjusted_wealth"] = adjust_observed_assets(
        observed_states_dict=states_dict,
        params=params,
        model_class=model_class,
    )

    # All agents have sex == 1
    sex_agents = np.full(n_agents, sex_var, dtype=np.uint8)

    # Restrict to start data for sex == 1
    start_data_sex = start_period_data[start_period_data["sex"] == sex_var]

    # Generate education distribution
    edu_shares = start_data_sex["education"].value_counts(normalize=True).sort_index()
    n_agents_edu_types = np.round(edu_shares * n_agents).astype(int)

    # Create the education array
    education_agents = np.repeat(edu_shares.index, n_agents_edu_types)

    survival_by_age = lifetable.loc[lifetable["sex"] == sex_var].set_index("age")[
        "cum_survival_prob"
    ]
    # b) P(health = 0/1/2 | alive, age)  → DataFrame indexed by age
    health_prob_by_age = (
        health_sample.loc[health_sample["sex"] == MOTHER]
        .groupby("age")["health"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)  # columns 0,1,2
        .sort_index()
    )
    health_prob_by_age.index = health_prob_by_age.index.astype(int)

    # =================================================================================

    # Generate containers
    wealth_agents = np.empty(n_agents, np.float64)
    exp_agents = np.empty(n_agents, np.float64)
    lagged_choice = np.empty(n_agents, np.uint8)
    partner_states = np.empty(n_agents, np.uint8)
    health_agents = np.empty(n_agents, np.uint8)
    job_offer_agents = np.empty(n_agents, np.uint8)
    mother_health_agents = np.empty(n_agents, np.uint8)
    mother_dead_agents = np.zeros(n_agents, dtype=np.uint8)
    caregiving_type_agents = np.empty(n_agents, dtype=np.uint8)

    # for sex_var in range(specs["n_sexes"]):
    for edu in range(model_specs["n_education_types"]):

        # Restrict dataset on education level
        type_mask = (sex_agents == sex_var) & (education_agents == edu)
        start_period_data_edu = start_period_data[
            (start_period_data["sex"] == sex_var)
            & (start_period_data["education"] == edu)
        ]

        n_agents_edu = np.sum(type_mask)

        # mother health
        mother_age_diff = model_specs["mother_age_diff"][edu]
        mother_age_scalar = int(
            np.asarray(model_specs["start_age"] + mother_age_diff.round().astype(int))
        )
        # Create array of ages (one per agent in this education group)
        mother_ages_array = np.full(n_agents_edu, mother_age_scalar, dtype=int)

        mother_health_agents[type_mask] = draw_mother_health(
            mother_ages_array,
            survival_by_age,
            health_prob_by_age,
        )

        # mother_dead if mother_health == PARENT_HEALTH_DEAD (3), else 0
        # In initial conditions, if mother is dead, we set to "longer dead"
        # because we don't know if death was recent, so no inheritance
        mother_dead_agents[type_mask] = np.where(
            mother_health_agents[type_mask] == PARENT_HEALTH_DEAD,
            PARENT_LONGER_DEAD,  # Set to longer dead
            0,  # Set to alive
        )

        # Wealth distribution
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
            index=np.arange(0, model_specs["n_choices"]), data=0, dtype=float
        )
        lagged_choice_probs.update(empirical_lagged_choice_probs)
        lagged_choice_edu = np.random.choice(
            model_specs["n_choices"], size=n_agents_edu, p=lagged_choice_probs.values
        )
        lagged_choice[type_mask] = lagged_choice_edu

        # Generate job offer probabilities
        job_offer_probs = job_offer_process_transition_initial_conditions(
            params=params,
            model_specs=model_specs,
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
            index=np.arange(model_specs["n_partner_states"]), data=0, dtype=float
        )
        partner_probs.update(empirical_partner_probs)
        partner_states_edu = np.random.choice(
            model_specs["n_partner_states"], size=n_agents_edu, p=partner_probs.values
        )
        partner_states[type_mask] = partner_states_edu

        # Generate health states
        empirical_health_probs = start_period_data_edu["health"].value_counts(
            normalize=True
        )
        health_probs = pd.Series(
            index=np.arange(model_specs["n_health_states"]), data=0, dtype=float
        )
        health_probs.update(empirical_health_probs)
        health_states_edu = np.random.choice(
            model_specs["n_health_states"], size=n_agents_edu, p=health_probs.values
        )
        health_agents[type_mask] = health_states_edu

        # Generate caregiving_type: 50% type 0, 50% type 1 (regardless of education)
        caregiving_type_edu = np.random.choice([0, 1], size=n_agents_edu, p=[0.5, 0.5])
        caregiving_type_agents[type_mask] = caregiving_type_edu

    # Transform it to be between 0 and 1
    exp_agents_scaled = scale_experience_years(
        experience_years=exp_agents,
        period=jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        is_retired=jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        model_specs=model_specs,
    )
    # Set lagged choice to 1(unemployment) if experience is 0
    exp_zero_mask = exp_agents == 0
    lagged_choice[exp_zero_mask] = 1

    # # Show share of observations for each discrete outcome in lagged_choice
    # lagged_choice_counts = (
    #     pd.Series(lagged_choice).value_counts(normalize=True).sort_index()
    # )
    # print("Lagged choice shares:")
    # for choice, share in lagged_choice_counts.items():
    #     print(f"  Choice {choice}: {share:.4f} ({share*100:.2f}%)")

    # Build states dict
    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "education": jnp.array(education_agents, dtype=jnp.uint8),
        "health": jnp.array(health_agents, dtype=jnp.uint8),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.uint8),
        "already_retired": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "experience": jnp.array(exp_agents_scaled, dtype=jnp.float64),
        "job_offer": jnp.array(job_offer_agents, dtype=jnp.uint8),
        "partner_state": jnp.array(partner_states, dtype=jnp.uint8),
        "mother_dead": jnp.array(mother_dead_agents, dtype=jnp.uint8),
        "caregiving_type": jnp.array(caregiving_type_agents, dtype=jnp.uint8),
        # "gets_inheritance": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "assets_begin_of_period": wealth_agents,
    }

    # Save initial discrete states and wealth
    with path_to_save_discrete_states.open("wb") as f:
        pickle.dump(states, f)
