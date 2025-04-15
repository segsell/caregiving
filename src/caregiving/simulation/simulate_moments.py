"""Simulate using JAX numpy."""

import jax.numpy as jnp
import numpy as np
import pandas as pd

from caregiving.model.shared import (
    FULL_TIME,
    NOT_WORKING,
    PART_TIME,
    RETIREMENT,
    SEX,
    UNEMPLOYED,
)
from caregiving.model.state_space import construct_experience_years
from caregiving.utils import table
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods


# =====================================================================================
# Pandas (like empirical moments)
# =====================================================================================
def simulate_moments_pandas_like_empirical(
    df,
    options,
) -> pd.DataFrame:
    """Simulate the model for given parametrization and model solution."""

    model_params = options["model_params"]
    start_age = model_params["start_age"]
    end_age = model_params["end_age_msm"]

    return create_moments_pandas_like_empirical(df, start_age, end_age)


def create_moments_pandas_like_empirical(sim_df, start_age, end_age):
    """
    Create simulation moments (means only) following the coding logic
    used in the empirical SOEP moments module.

    This function performs two main tasks:

    A) Computes labor share moments by age. For each age between start_age and end_age,
       it computes:
         - share of agents working full-time,
         - share working part-time,
         - share unemployed,
         - share retired, and
         - share not working (unemployed or retired).
       Keys are named like "share_working_full_time_age_40".

    B) Computes overall year-to-year transition probabilities between the states
       full_time, part_time, and not_working. Keys are named like
       "transition_full_time_to_part_time".

    Parameters:
        sim_df (pd.DataFrame): The simulation DataFrame, assumed to contain at least:
                               'age', 'choice', and 'lagged_choice'.
        start_age (int): Lower bound (inclusive) of the age range.
        end_age (int): Upper bound (inclusive) of the age range.

    Returns:
        pd.Series: A Series with descriptive moment names as the index and the
        computed means as values.
    """
    moments = {}

    # A) Labor shares by age.
    for age in range(start_age, end_age + 1):
        subdf = sim_df[sim_df["age"] == age]
        shares = _compute_labor_shares_means(subdf)
        moments[f"share_retired_age_{age}"] = shares["retired"]
        moments[f"share_unemployed_age_{age}"] = shares["unemployed"]
        moments[f"share_working_part_time_age_{age}"] = shares["part_time"]
        moments[f"share_working_full_time_age_{age}"] = shares["full_time"]

    # B) Overall transition probabilities.
    trans_moments = _compute_transition_means(
        sim_df,
        not_working=NOT_WORKING,
        part_time=PART_TIME,
        full_time=FULL_TIME,
        choice="choice",
        lagged_choice="lagged_choice",
    )
    moments.update(trans_moments)

    return pd.Series(moments)


def _compute_labor_shares_means(subdf):
    """
    Compute labor shares (means) for a given subsample.

    Parameters:
        subdf (pd.DataFrame): Subsample of the simulation DataFrame.

    Returns:
        dict: Dictionary with the mean shares for:
            - "full_time": share working full-time,
            - "part_time": share working part-time,
            - "unemployed": share unemployed,
            - "retired": share retired,
            - "not_working": share not working (unemployed or retired).
            If no observations are available, each value is set to np.nan.
    """
    total = len(subdf)
    if total == 0:
        return {
            "full_time": np.nan,
            "part_time": np.nan,
            "unemployed": np.nan,
            "retired": np.nan,
            "not_working": np.nan,
        }

    full_time = subdf["choice"].isin(FULL_TIME.tolist()).astype(float).mean()
    part_time = subdf["choice"].isin(PART_TIME.tolist()).astype(float).mean()
    unemployed = subdf["choice"].isin(UNEMPLOYED.tolist()).astype(float).mean()
    retired = subdf["choice"].isin(RETIREMENT.tolist()).astype(float).mean()
    not_working = (
        subdf["choice"]
        .isin(UNEMPLOYED.tolist() + RETIREMENT.tolist())
        .astype(float)
        .mean()
    )

    return {
        "full_time": full_time,
        "part_time": part_time,
        "unemployed": unemployed,
        "retired": retired,
        "not_working": not_working,
    }


def _compute_transition_means(
    df,
    not_working,
    part_time,
    full_time,
    choice="choice",
    lagged_choice="lagged_choice",
):
    """
    Compute overall year-to-year labor supply transition probabilities (means).

    The function creates a row-normalized transition matrix using pd.crosstab,
    and then maps numeric codes to descriptive labels using the provided constants.

    Parameters:
       df (pd.DataFrame): DataFrame containing the columns specified by `lagged_choice`
                          and `choice`.
       full_time, part_time, not_working: Array-like codes for the respective states.
       choice (str): Column name for the current period’s state.
       lagged_choice (str): Column name for the previous period’s state.

    Returns:
        dict: A dictionary with keys like "transition_full_time_to_part_time"
            corresponding to the computed probabilities.
    """
    # # Create a row-normalized transition matrix.
    # transition_matrix = pd.crosstab(df[lagged_choice], df[choice], normalize="index")

    # # Build a mapping from numeric codes to descriptive labels.
    # choice_map = {code: "not_working" for code in not_working.tolist()}
    # choice_map.update({code: "part_time" for code in part_time.tolist()})
    # choice_map.update({code: "full_time" for code in full_time.tolist()})

    # trans_moments = {}
    # for lag_val in transition_matrix.index:
    #     for current_val in transition_matrix.columns:
    #         from_state = choice_map.get(lag_val, lag_val)
    #         to_state = choice_map.get(current_val, current_val)
    #         key = f"transition_{from_state}_to_{to_state}"
    #         trans_moments[key] = transition_matrix.loc[lag_val, current_val]

    trans_moments = {}
    # --- (b) Transition probabilities ---
    # Define the states of interest for transitions
    states = {
        "not_working": NOT_WORKING,
        "part_time": PART_TIME,
        "full_time": FULL_TIME,
    }

    # For each "from" state, filter rows where lagged_choice is in that state,
    # and for each "to" state, compute the probability that 'choice' is in that state.
    for from_label, from_val in states.items():
        # Use isin() to safely compare even if from_val is an array.
        subset = df[df["lagged_choice"].isin(np.atleast_1d(from_val))]
        for to_label, to_val in states.items():
            if len(subset) > 0:
                probability = subset["choice"].isin(np.atleast_1d(to_val)).mean()
            else:
                probability = np.nan  # or 0 if that is preferable
            trans_moments[f"trans_{from_label}_to_{to_label}"] = probability

    return trans_moments


# =====================================================================================
# Pandas
# =====================================================================================


def simulate_moments_pandas(
    df,
    options,
) -> pd.DataFrame:
    """Simulate the model for given parametrization and model solution."""

    model_params = options["model_params"]
    start_age = model_params["start_age"]
    end_age = model_params["end_age_msm"]

    return create_moments_pandas(df, start_age, end_age)


def create_moments_pandas(df, start_age, end_age):
    """
    Create a Pandas Series of simulation moments.

    This function performs two tasks:

      (a) Age-specific shares: For each age between start_age and end_age,
          compute the share of agents (from df) whose 'choice' indicates they
          are retired, unemployed, working part-time, or working full-time.
          The resulting keys are named for example, "share_retired_age_40".

      (b) Transition probabilities: Compute nine transition probabilities from
          the previous period (lagged_choice) to the current period (choice) for
          the following states: NOT_WORKING, PART_TIME, and FULL_TIME.
          The keys are named like "trans_not_working_to_full_time".

    Assumes that the DataFrame `df` contains at least the following columns:
      - age
      - choice
      - lagged_choice

    Parameters:
        df (pd.DataFrame): The simulation DataFrame.
        start_age (int): The starting age (inclusive) for computing age-specific shares.
        end_age (int): The ending age (inclusive) for computing age-specific shares.

    Returns:
        pd.Series: A Series with moment names as the index and computed moments
            as the values.
    """
    moments = {}

    # --- (a) Age-specific shares ---
    # Create the desired age range
    age_range = range(start_age, end_age + 1)

    # Group by 'age' over the entire dataframe (assumes df already has an 'age' column)
    age_groups = df.groupby("age")

    # Compute the proportion for each status using vectorized operations
    retired_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(RETIREMENT)).mean()
    )
    unemployed_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(UNEMPLOYED)).mean()
    )
    part_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(PART_TIME)).mean()
    )
    full_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(FULL_TIME)).mean()
    )

    # Reindex to ensure that every age between start_age and end_age is included;
    # missing ages will be filled with NaN
    retired_shares = retired_shares.reindex(age_range, fill_value=np.nan)
    unemployed_shares = unemployed_shares.reindex(age_range, fill_value=np.nan)
    part_time_shares = part_time_shares.reindex(age_range, fill_value=np.nan)
    full_time_shares = full_time_shares.reindex(age_range, fill_value=np.nan)

    # Populate the moments dictionary for age-specific shares
    for age in age_range:
        moments[f"share_retired_age_{age}"] = retired_shares.loc[age]
        # for age in age_range:
        moments[f"share_unemployed_age_{age}"] = unemployed_shares.loc[age]
        # for age in age_range:
        moments[f"share_part_time_age_{age}"] = part_time_shares.loc[age]
        # for age in age_range:
        moments[f"share_full_time_age_{age}"] = full_time_shares.loc[age]

    # --- (b) Transition probabilities ---
    # Define the states of interest for transitions
    states = {
        "not_working": NOT_WORKING,
        "part_time": PART_TIME,
        "full_time": FULL_TIME,
    }

    # For each "from" state, filter rows where lagged_choice is in that state,
    # and for each "to" state, compute the probability that 'choice' is in that state.
    for from_label, from_val in states.items():
        # Use isin() to safely compare even if from_val is an array.
        subset = df[df["lagged_choice"].isin(np.atleast_1d(from_val))]
        for to_label, to_val in states.items():
            if len(subset) > 0:
                probability = subset["choice"].isin(np.atleast_1d(to_val)).mean()
            else:
                probability = np.nan  # or 0 if that is preferable
            moments[f"trans_{from_label}_to_{to_label}"] = probability

    return pd.Series(moments)


# =====================================================================================
# JAX numpy
# =====================================================================================


def simulate_moments_jax(
    df,
    options,
) -> pd.DataFrame:
    """Simulate the model for given parametrization and model solution."""

    model_params = options["model_params"]
    start_age = model_params["start_age"]
    end_age = model_params["end_age_msm"]

    # df["age"] = df.index.get_level_values("period") + model_params["start_age"]

    return create_moments_jax(df, start_age, end_age)


def create_moments_jax(sim_df, min_age, max_age):

    column_indices = {col: idx for idx, col in enumerate(sim_df.columns)}
    idx = column_indices.copy()
    arr = jnp.asarray(sim_df)

    share_retired_by_age = get_share_by_age(
        arr, ind=idx, choice=RETIREMENT, min_age=min_age, max_age=max_age
    )  # 15
    share_unemployed_by_age = get_share_by_age(
        arr, ind=idx, choice=UNEMPLOYED, min_age=min_age, max_age=max_age
    )  # 15
    share_working_part_time_by_age = get_share_by_age(
        arr, ind=idx, choice=PART_TIME, min_age=min_age, max_age=max_age
    )  # 15
    share_working_full_time_by_age = get_share_by_age(
        arr, ind=idx, choice=FULL_TIME, min_age=min_age, max_age=max_age
    )  # 15

    # Work transitions
    no_work_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=NOT_WORKING,
        current_choice=NOT_WORKING,
    )
    no_work_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=NOT_WORKING,
        current_choice=PART_TIME,
    )
    no_work_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=NOT_WORKING,
        current_choice=FULL_TIME,
    )

    part_time_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=NOT_WORKING,
    )
    part_time_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=PART_TIME,
    )
    part_time_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=FULL_TIME,
    )

    full_time_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=NOT_WORKING,
    )
    full_time_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=PART_TIME,
    )
    full_time_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=FULL_TIME,
    )

    return jnp.asarray(
        share_retired_by_age
        + share_unemployed_by_age
        + share_working_part_time_by_age
        + share_working_full_time_by_age
        + no_work_to_no_work
        + no_work_to_part_time
        + no_work_to_full_time
        + part_time_to_no_work
        + part_time_to_part_time
        + part_time_to_full_time
        + full_time_to_no_work
        + full_time_to_part_time
        + full_time_to_full_time
    )


# ==============================================================================
# Auxiliary
# ==============================================================================


def get_share_by_age(df_arr, ind, choice, min_age, max_age):
    """Get share of agents choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    shares = []
    for age in range(min_age, max_age + 1):
        age_mask = df_arr[:, ind["age"]] == age

        share = jnp.sum(age_mask & lagged_choice_mask) / jnp.sum(age_mask)
        # period count is always larger than 0! otherwise error
        shares.append(share)

    return shares


def get_share_by_age_bin(df_arr, ind, choice, age_bins):
    """Get share of agents choosing lagged choice by age bin."""
    return [
        jnp.mean(
            jnp.isin(df_arr[:, ind["choice"]], choice)
            & (df_arr[:, ind["period"]] > age_bin[0])
            & (df_arr[:, ind["period"]] <= age_bin[1]),
        )
        for age_bin in age_bins
    ]


def get_mean_by_age_bin_for_lagged_choice(df_arr, ind, var, choice, age_bins):
    """Get mean of agents choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    means = []

    for age_bin in age_bins:
        age_bin_mask = (df_arr[:, ind["period"]] > age_bin[0]) & (
            df_arr[:, ind["period"]] <= age_bin[1]
        )
        means += [jnp.mean(df_arr[lagged_choice_mask & age_bin_mask, ind[var]])]

    return means


def get_share_by_type_by_age_bin(df_arr, ind, choice, care_type, age_bins):
    """Get share of agents of given care type choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    care_type_mask = jnp.isin(df_arr[:, ind["choice"]], care_type)

    shares = []
    for age_bin in age_bins:
        age_bin_mask = (df_arr[:, ind["period"]] > age_bin[0]) & (
            df_arr[:, ind["period"]] <= age_bin[1]
        )
        share = jnp.sum(lagged_choice_mask & care_type_mask & age_bin_mask) / jnp.sum(
            care_type_mask & age_bin_mask,
        )
        shares.append(share)

    return shares


def get_share_by_type(df_arr, ind, choice, care_type):
    """Get share of agents of given care type choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    care_type_mask = jnp.isin(df_arr[:, ind["choice"]], care_type)

    return jnp.sum(lagged_choice_mask & care_type_mask) / jnp.sum(care_type_mask)


def get_transition(df_arr, ind, lagged_choice, current_choice):
    """Get transition probability from lagged choice to current choice."""
    return [
        jnp.sum(
            jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
            & jnp.isin(df_arr[:, ind["choice"]], current_choice),
        )
        / jnp.sum(jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)),
    ]
