"""Initial conditions for the simulation."""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
import yaml
from pytask import Product
from scipy import stats
from sklearn.neighbors import KernelDensity

import dcegm
from caregiving.config import BLD
from caregiving.model.experience_baseline_model import scale_experience_years
from caregiving.model.shared import (
    END_YEAR_PARENT_GENERATION,
    FEMALE,
    MOTHER,
    PARENT_HEALTH_DEAD,
    PARENT_LONGER_DEAD,
    SEX,
    WEALTH_END_YEAR,
    WEALTH_QUANTILE_CUTOFF,
    WEALTH_START_YEAR,
)
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.stochastic_processes.job_transition import (
    job_offer_process_transition_initial_conditions,
)
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.moments.task_create_soep_moments import (
    create_df_non_caregivers,
    create_df_wealth,
)
from caregiving.moments.transform_data import load_and_scale_correct_data
from dcegm.asset_correction import adjust_observed_assets


@pytask.mark.initial_conditions
def task_generate_start_states_for_solution(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_lifetable: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "lifetable.csv",
    path_to_health_sample: Path = BLD
    / "data"
    / "health_transition_estimation_sample_good_medium_bad.pkl",
    path_to_parent_child_sample: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_model: Path = BLD / "model" / "model.pkl",
    path_to_start_params: Path = BLD
    / "model"
    / "params"
    / "estimated_params_model.yaml",
    path_to_save_health_by_age: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "health_by_age.csv",
    path_to_save_survival_by_age: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "survival_by_age.csv",
    path_to_save_initial_states: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
) -> None:
    sex_var = SEX

    observed_data = pd.read_csv(path_to_sample, index_col=[0])

    lifetable = pd.read_csv(path_to_lifetable)
    health_sample = pd.read_pickle(path_to_health_sample)
    parent_child_data = pd.read_csv(path_to_parent_child_sample)

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
        # alternative_sim_specifications=alternative_sim_specifications,
        # debug_info=debug_info,
        # use_stochastic_sparsity=True,
    )
    model_specs = model_class.model_specs
    model_structure = model_class.model_structure

    n_agents = model_specs["n_agents"]
    seed = model_specs["seed"]

    np.random.seed(seed)

    # # Define start data and adjust wealth
    # min_period = observed_data["period"].min()
    # start_period_data = observed_data[
    #     observed_data["period"].isin([min_period])
    # ].copy()
    # start_period_data = start_period_data[
    #     start_period_data["wealth"].notnull()
    # ].copy()

    # Use create_df_non_caregivers to match the moments calculation
    # (moments use non-caregivers only, so initial conditions should too)
    moments_data = create_df_non_caregivers(
        df_full=observed_data,
        specs=model_specs,
        start_year=2001,
        end_year=2019,
        end_age=model_specs["end_age_msm"],
    )
    start_period_data = moments_data[
        moments_data["age"] == model_specs["start_age"]
    ].copy()

    # observed_wealth = create_df_wealth(
    #     df_full=observed_data,
    #     specs=model_specs,
    #     params=params,
    #     model_class=model_class,
    #     adjust_wealth=False,
    #     trim_quantile=False,
    #     wealth_var="lagged_wealth",
    # )
    observed_wealth_corrected = load_and_scale_correct_data(
        data_decision=observed_data,
        model_class=model_class,
    )
    observed_wealth = observed_wealth_corrected[
        (observed_wealth_corrected["syear"] >= WEALTH_START_YEAR)
        & (observed_wealth_corrected["syear"] <= WEALTH_END_YEAR - 1)  # 2019
    ].copy()
    observed_wealth["adjusted_wealth"] = observed_wealth["assets_begin_of_period"]

    # start_age_wealth = observed_wealth[
    #     observed_wealth["age"] == model_specs["start_age"]
    # ].copy()

    min_period = observed_data["period"].min()
    start_period_data_wealth = observed_wealth[
        observed_wealth["period"].isin([min_period])
    ].copy()
    start_age_wealth = start_period_data_wealth[
        start_period_data_wealth["wealth"].notnull()
    ].copy()

    # =================================================================================
    # Static state variables
    lifetable = lifetable.sort_values(["sex", "age"])  # ensure order
    lifetable["cum_survival_prob"] = (
        (1 - lifetable["death_prob"]).groupby(lifetable["sex"]).cumprod()
    )

    # =================================================================================

    states_dict = {
        name: start_age_wealth[name].values
        for name in model_structure["discrete_states_names"]
        if name
        not in (
            "mother_health",
            "mother_adl",
            "mother_dead",
            "care_demand",
            "care_supply",
            "caregiving_type",
        )
    }

    states_dict["care_demand"] = np.zeros_like(start_age_wealth["wealth"])
    states_dict["experience"] = start_age_wealth["experience"].values
    # Initialize mother_dead to 0 (alive) for all agents at initial period
    # (will be drawn later based on mother health)
    states_dict["mother_dead"] = np.zeros_like(
        start_age_wealth["wealth"], dtype=np.uint8
    )

    # states_dict["assets_begin_of_period"] = (
    #     start_age_wealth["wealth"].values / specs["wealth_unit"]
    # )
    # start_age_wealth.loc[:, "adjusted_wealth"] = adjust_observed_assets(
    #     observed_states_dict=states_dict,
    #     params=params,
    #     model_class=model_class,
    # )

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
    # health_prob_by_age.index.name = "mother_age"

    survival_by_age.to_csv(path_to_save_survival_by_age, index=True)
    health_prob_by_age.to_csv(path_to_save_health_by_age, index=True)

    # =================================================================================
    # # Generate policy_state values for synthetic agents based on empirical SRA shares

    # # 1. Get empirical distribution of policy_state (already created)
    # sra_counts = (
    #     observed_data.loc[
    #         (observed_data["gebjahr"] >= specs["min_birth_year"])
    #         & (
    #         observed_data["gebjahr"] < specs["end_year"] - specs["min_ret_age"] + 20
    #         ),  # 1974
    #         "policy_state",
    #     ]
    #     .value_counts(normalize=True)
    #     .sort_index()
    # )

    # # 2. Sample SRA values for all agents
    # available_sras = sra_counts.index.to_numpy()
    # sra_probs = sra_counts.to_numpy()
    # drawn_sras = np.random.choice(available_sras, size=n_agents, p=sra_probs)

    # # 3. Map sampled SRA values to grid indices
    # sra_grid_size = options["model_params"]["SRA_grid_size"]
    # n_policy_states = options["model_params"]["n_policy_states"]

    # # Validate SRA range
    # assert drawn_sras.min() >= 0
    # assert drawn_sras.max() < n_policy_states * sra_grid_size

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
    mother_adl_agents = np.zeros(
        n_agents, dtype=np.uint8
    )  # Initialize to 0 (dead = no ADL)
    caregiving_type_agents = np.empty(n_agents, dtype=np.uint8)

    # for sex_var in range(specs["n_sexes"]):
    for edu in range(model_specs["n_education_types"]):

        # Restrict dataset on education level
        type_mask = (sex_agents == sex_var) & (education_agents == edu)
        start_period_data_edu = start_period_data[
            (start_period_data["sex"] == sex_var)
            & (start_period_data["education"] == edu)
        ]
        wealth_edu = start_age_wealth[
            start_age_wealth["education"] == edu
        ].copy()  # already women only

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

        # mother_adl: draw from empirical ADL distribution by age in parent_child data
        # If dead, ADL = 0 (No ADL). If alive, draw from empirical distribution
        mother_adl_agents[type_mask] = draw_mother_adl(
            mother_ages_array,
            mother_dead_agents[type_mask],
            parent_child_data,
            model_specs,
        )

        # Wealth distribution
        # wealth_start_edu = draw_start_wealth_dist(start_period_data_edu, n_agents_edu)
        wealth_start_edu = draw_start_wealth_dist(wealth_edu, n_agents_edu)

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
            # "choice"
            "lagged_choice"
        ].value_counts(normalize=True)
        lagged_choice_probs = pd.Series(index=np.arange(0, 4), data=0, dtype=float)
        lagged_choice_probs.update(empirical_lagged_choice_probs)
        lagged_choice_edu = np.random.choice(
            4, size=n_agents_edu, p=lagged_choice_probs.values
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

    # In the first period, only NO_CARE choices are available (0, 1, 2, 3),
    # which correspond to retirement, unemployed, part-time, full-time.
    # The empirical lagged_choice values
    # (0=retirement, 1=unemployed, 2=part-time, 3=full-time)
    # map directly to NO_CARE choices in the model (0, 1, 2, 3).
    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "education": jnp.array(education_agents, dtype=jnp.uint8),
        "health": jnp.array(health_agents, dtype=jnp.uint8),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.uint8),
        # "policy_state": jnp.array(drawn_sras, dtype=jnp.uint8),
        "already_retired": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "experience": jnp.array(exp_agents_scaled, dtype=jnp.float64),
        "job_offer": jnp.array(job_offer_agents, dtype=jnp.uint8),
        "partner_state": jnp.array(partner_states, dtype=jnp.uint8),
        # "mother_health": jnp.array(mother_health_agents, dtype=jnp.uint8),
        "mother_dead": jnp.array(mother_dead_agents, dtype=jnp.uint8),
        "mother_adl": jnp.array(mother_adl_agents, dtype=jnp.uint8),
        "care_demand": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        # "care_demand": jnp.where(
        #     jnp.array(mother_dead_agents, dtype=jnp.uint8) == 1,
        #     NO_CARE_DEMAND_DEAD,
        #     NO_CARE_DEMAND_ALIVE,
        # ),
        "caregiving_type": jnp.array(caregiving_type_agents, dtype=jnp.uint8),
        # "gets_inheritance": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "assets_begin_of_period": wealth_agents,
    }
    # type_mask_low = (sex_agents == sex_var) & (education_agents == 0)
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # df = pd.DataFrame(
    #     {
    #         "exp_empirical": start_period_data["experience"].values,
    #     }
    # )

    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # sns.histplot(data=df, x="exp_empirical", ax=ax, kde=True)
    # ax.set_title("Experience (Empirical)")
    # plt.tight_layout()
    # plt.show()

    with path_to_save_initial_states.open("wb") as f:
        pickle.dump(states, f)


def draw_mother_adl(
    mother_age: np.ndarray,
    mother_dead: np.ndarray,
    parent_child_data: pd.DataFrame,
    specs: dict,
) -> np.ndarray:
    """
    Draw initial ADL states for mothers using empirical distribution from
    parent_child data.

    Parameters
    ----------
    mother_age : np.ndarray
        Array of mother ages
    mother_dead : np.ndarray
        Array of mother death status (1=dead, 0=alive)
    parent_child_data : pd.DataFrame
        Parent-child dataset with columns: age, gender, adl_cat, yrbirth
    specs : dict
        Specs dictionary (not used currently, but kept for consistency)

    Returns
    -------
    np.ndarray
        Array of ADL states (0=No ADL, 1=ADL 1, 2=ADL 2 or ADL 3)
    """
    rng = np.random.default_rng()

    # For alive mothers, draw from empirical ADL distribution by age
    # Dead mothers will have ADL = 0 (initialized in result array below)
    alive_mask = mother_dead == 0

    # Filter parent_child data to women only and valid ADL categories (once)
    df_obs = parent_child_data[
        (parent_child_data["gender"] == FEMALE)
        & (parent_child_data["yrbirth"] < END_YEAR_PARENT_GENERATION)
        & (parent_child_data["adl_cat"].notna())
    ].copy()

    # Precompute ADL distributions by age using groupby (efficient)
    # Map to collapsed categories: 0=No ADL, 1=ADL 1, 2=ADL 2 or ADL 3
    df_obs["adl_collapsed"] = df_obs["adl_cat"].replace({3: 2})  # Combine 2 and 3

    # Compute probabilities by age
    age_adl_probs = (
        df_obs.groupby("age")["adl_collapsed"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .sort_index()
    )

    # Ensure all 3 categories (0, 1, 2) are present
    for col in (0, 1, 2):
        if col not in age_adl_probs.columns:
            age_adl_probs[col] = 0.0

    age_adl_probs = age_adl_probs[[0, 1, 2]]  # Ensure correct order

    # Convert index to int to ensure consistent indexing
    age_adl_probs.index = age_adl_probs.index.astype(int)

    # Get overall distribution as fallback
    overall_probs = (
        df_obs["adl_collapsed"]
        .value_counts(normalize=True)
        .sort_index()
        .reindex([0, 1, 2], fill_value=0.0)
    )
    overall_probs = overall_probs.values

    # Get unique ages for lookup (convert index to numpy array of ints)
    unique_ages = age_adl_probs.index.values.astype(int)

    # Convert mother_age to numpy array if it's a JAX array
    mother_age_np = np.asarray(mother_age, dtype=int)
    # Ensure it's 1D
    if mother_age_np.ndim == 0:
        mother_age_np = np.array([mother_age_np.item()])
    mother_age_np = mother_age_np.flatten()

    # For each alive mother, get ADL distribution at her age
    alive_ages = mother_age_np[alive_mask]
    mother_adl_alive = np.empty(len(alive_ages), dtype=np.uint8)

    # Convert unique_ages to a set for faster lookup
    unique_ages_set = set(unique_ages)

    for i, age in enumerate(alive_ages):
        # Convert to Python int for pandas indexing
        age_int = int(age)

        # Find closest age in data
        if age_int in unique_ages_set:
            # Age exists in data, use it directly
            # Use .loc with explicit index access
            try:
                probs = age_adl_probs.loc[age_int].values
            except (KeyError, IndexError):
                # Fallback: use overall distribution
                probs = overall_probs.copy()
        else:
            # Find closest age
            age_idx = np.argmin(np.abs(unique_ages - age_int))
            age_to_use = unique_ages[age_idx]
            try:
                probs = age_adl_probs.loc[age_to_use].values
            except (KeyError, IndexError):
                # Fallback: use overall distribution
                probs = overall_probs.copy()

        # Normalize probabilities (in case they don't sum to 1 due to rounding)
        probs = probs / probs.sum() if probs.sum() > 0 else overall_probs

        # Sample ADL state
        mother_adl_alive[i] = rng.choice(3, p=probs)

    # Create result array (dead mothers = 0, alive mothers = sampled values)
    mother_adl = np.zeros(len(mother_age), dtype=np.uint8)
    mother_adl[alive_mask] = mother_adl_alive

    return mother_adl


def draw_mother_health(
    mother_age: int,
    survival_by_age: pd.Series,
    health_prob_by_age: pd.DataFrame,
) -> int:
    """
    Draw one of the four mother-health states for a single age.

    Returns
    -------
    int  -  0: good, 1: medium, 2: bad, 3: dead
    """
    rng = np.random.default_rng()

    ages = np.asarray(mother_age, dtype=int)
    # Ensure ages is 1D array (handle scalar/0-d array case)
    if ages.ndim == 0:
        ages = np.array([ages.item()])
    ages = ages.flatten()  # Ensure 1D

    prob_alive = pd.Series(ages).map(survival_by_age).to_numpy()
    health = health_prob_by_age.reindex(ages).to_numpy()  # shape (n, 3)

    # 3 ── full 4-state probability rows
    probs_alive = health * prob_alive[:, None]  # (n, 3)
    probs = np.hstack(
        [probs_alive, (1 - prob_alive)[:, None]]
    )  # add “dead”, shape (n, 4)

    # probs /= probs.sum(axis=1, keepdims=True)  # row-wise normalise
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Probability rows do not sum to 1 after normalisation.")

    # 4 ── sample one state per agent (readable version)
    mother_health = np.array([rng.choice(4, p=row) for row in probs], dtype=np.uint8)

    return mother_health


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
        a_min=wealth_data.quantile(0),
        a_max=wealth_data.quantile(WEALTH_QUANTILE_CUTOFF),
    )

    return wealth_start_clipped


# def scale_experience_years(experience_years, period, model_specs):
#     """Scale experience between 0 and 1."""
#     # If period is past the last working period, then we take the maximum experience
#     scale_not_retired = jnp.take(
#         model_specs["max_exps_period_working"], period, mode="clip"
#     )
#     # scale_retired = model_specs["max_pp_retirement"]
#     # scale = is_retired * scale_retired + (1 - is_retired) * scale_not_retired

#     return experience_years / scale_not_retired
