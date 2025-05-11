"""Simulate moments."""

import re
from itertools import product
from typing import Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.model.shared import (
    FULL_TIME,
    INFORMAL_CARE,
    NOT_WORKING_CARE,
    PART_TIME,
    RETIREMENT,
    UNEMPLOYED,
    WORK,
)

SEX = 1

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

    age_range = range(start_age, end_age + 1)
    age_bins = (
        list(range(40, 75, 5)),  # [40, 45, … , 70]
        [f"{s}_{s+4}" for s in range(40, 70, 5)],  # "40_44", …a
    )

    df_low = df[df["education"] == 0]
    df_high = df[df["education"] == 1]

    df_caregivers = df[df["choice"].isin(np.asarray(INFORMAL_CARE))]
    # df_caregivers_low = df_caregivers[df_caregivers["education"] == 0]
    # df_caregivers_high = df_caregivers[df_caregivers["education"] == 1]

    moments = {}

    moments = create_labor_share_moments_pandas(df, moments, age_range=age_range)
    moments = create_labor_share_moments_pandas(
        df_low, moments, age_range=age_range, label="low_education"
    )
    moments = create_labor_share_moments_pandas(
        df_high, moments, age_range=age_range, label="high_education"
    )

    moments = create_choice_shares_by_age_bin_pandas(
        df, moments, choice_set=INFORMAL_CARE, age_bins=age_bins
    )
    moments["share_informal_care_high_educ"] = df.loc[
        df["choice"].isin(np.atleast_1d(INFORMAL_CARE)), "education"
    ].mean()

    moments = create_labor_share_moments_by_age_bin(
        df_caregivers, moments, label="caregivers", age_bins=age_bins
    )

    # states = {
    #     "not_working": NOT_WORKING,
    #     "part_time": PART_TIME,
    #     "full_time": FULL_TIME,
    # }
    # moments = compute_transition_moments_pandas(df, moments, age_range, states=states)

    # states_work_no_work = {
    #     "not_working": NOT_WORKING,
    #     "working": WORK,
    # }
    # moments = compute_transition_moments_pandas_for_age_bins(
    #     df_low, moments, age_range, states=states_work_no_work, label="low_education"
    # )
    # moments = compute_transition_moments_pandas_for_age_bins(
    #     df_high,
    # moments,
    # age_range,
    # states=states_work_no_work,
    # label="high_education"
    # )

    return pd.Series(moments)


def create_labor_share_moments_pandas(df, moments, age_range, label=None):
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

    if label is None:
        label = ""
    else:
        label = "_" + label

    # 1) Labor shares
    # Create the desired age range

    # Group by 'age' over the entire dataframe (assumes df already has an 'age' column)
    age_groups = df.groupby("age")

    # Compute the proportion for each status using vectorized operations
    # retired_shares = age_groups["choice"].apply(
    #     lambda x: x.isin(np.atleast_1d(RETIREMENT)).mean()
    # )
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
    # retired_shares = retired_shares.reindex(age_range, fill_value=np.nan)
    unemployed_shares = unemployed_shares.reindex(age_range, fill_value=np.nan)
    part_time_shares = part_time_shares.reindex(age_range, fill_value=np.nan)
    full_time_shares = full_time_shares.reindex(age_range, fill_value=np.nan)

    # Populate the moments dictionary for age-specific shares
    # for age in age_range:
    #     moments[f"share_retired{label}_age_{age}"] = retired_shares.loc[age]
    for age in age_range:
        moments[f"share_unemployed{label}_age_{age}"] = unemployed_shares.loc[age]
    for age in age_range:
        moments[f"share_part_time{label}_age_{age}"] = part_time_shares.loc[age]
    for age in age_range:
        moments[f"share_full_time{label}_age_{age}"] = full_time_shares.loc[age]

    return moments


def create_labor_share_moments_by_age_bin(
    df: pd.DataFrame,
    moments: dict,
    age_bins: tuple[list[int], list[str]] | None = None,
    label: str | None = None,
):
    """
    Like `create_labor_share_moments_pandas`, but aggregates by *age-bin*
    instead of single ages.

    Parameters
    ----------
    df : DataFrame
        Must contain ``age`` (int) and ``choice`` (categorical / int).
    moments : dict
        Updated **in-place** with the new statistics.
    age_bins : tuple[list[int], list[str]] | None
        Optional ``(bin_edges, bin_labels)`` passed to ``pd.cut``.
        Default edges are ``[40, 45, 50, 55, 60, 65, 70]`` which yield
        labels ``["40_44", "45_49", … , "65_69"]``.
    label : str | None
        Extra label inserted in every key (prefixed with “_” if given).

    Returns
    -------
    dict
        The same *moments* dict, for convenience.
    """

    # ---------- 1.  Pre-processing ------------------------------------------------
    label = f"_{label}" if label else ""

    if age_bins is None:
        bin_edges = list(range(40, 75, 5))  # [40, 45, … , 70]
        bin_labels = [f"{s}_{s+4}" for s in bin_edges[:-1]]  # "40_44", …
    else:
        bin_edges, bin_labels = age_bins

    # Work on a copy that contains only the relevant ages
    df = df[df["age"].between(bin_edges[0], bin_edges[-1] - 1)].copy()

    df["age_bin"] = pd.cut(
        df["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # [40,45) ⇒ 40–44, etc.
    )

    age_groups = df.groupby("age_bin", observed=False)

    unemployed_shares = (
        age_groups["choice"]
        .apply(lambda x: x.isin(np.atleast_1d(UNEMPLOYED)).mean())
        .reindex(bin_labels, fill_value=np.nan)
    )
    part_time_shares = (
        age_groups["choice"]
        .apply(lambda x: x.isin(np.atleast_1d(PART_TIME)).mean())
        .reindex(bin_labels, fill_value=np.nan)
    )
    full_time_shares = (
        age_groups["choice"]
        .apply(lambda x: x.isin(np.atleast_1d(FULL_TIME)).mean())
        .reindex(bin_labels, fill_value=np.nan)
    )

    for age_bin in bin_labels:
        moments[f"share_unemployed{label}_age_bin_{age_bin}"] = unemployed_shares.loc[
            age_bin
        ]
    for age_bin in bin_labels:
        moments[f"share_part_time{label}_age_bin_{age_bin}"] = part_time_shares.loc[
            age_bin
        ]
    for age_bin in bin_labels:
        moments[f"share_full_time{label}_age_bin_{age_bin}"] = full_time_shares.loc[
            age_bin
        ]

    return moments


def create_choice_shares_by_age_pandas(
    df: pd.DataFrame,
    moments: dict,
    choice_set: jnp.ndarray,
    age_range,
    label: str | None = None,
):
    """
    Update *moments* in=place with the share of agents whose “choice”
    lies in INFORMAL_CARE, computed separately for every age in
    *age_range*.

    """
    label = f"_{label}" if label else ""

    share_by_age = (
        df.groupby("age")["choice"]
        .apply(lambda x: x.isin(np.atleast_1d(choice_set)).mean())
        .reindex(age_range, fill_value=np.nan)
    )

    for age in age_range:
        moments[f"share_informal_care{label}_age_{age}"] = share_by_age.loc[age]

    return moments


def create_choice_shares_by_age_bin_pandas(
    df: pd.DataFrame,
    moments: dict,
    choice_set: jnp.ndarray,
    age_bins: tuple[list[int], list[str]] | None = None,
    label: str | None = None,
):
    """
    Update *moments* in-place with the share of agents whose ``choice`` lies
    in *choice_set*, computed separately for every *age-bin*.

    Parameters
    ----------
    df : DataFrame
        Must contain the columns ``age`` (int) and ``choice`` (categorical/int).
    moments : dict
        Updated **in-place** with the new statistics.
    choice_set : jnp.ndarray
        Set of codes/categories to be counted (e.g. INFORMAL_CARE choices).
    age_bins : tuple[list[int], list[str]] | None
        Optional ``(bin_edges, bin_labels)`` passed to ``pd.cut``.
        Defaults to 5-year bins ``[40, 45, …, 70]`` with labels
        ``["40_44", "45_49", …, "65_69"]``.
    label : str | None
        Extra label inserted in every key (prefixed with “_” if given).

    Returns
    -------
    dict
        The same *moments* dict (for convenience).
    """
    # -------- 1. Pre-processing ---------------------------------------------------
    label = f"_{label}" if label else ""

    if age_bins is None:
        bin_edges = list(range(40, 75, 5))  # [40, 45, …, 70]
        bin_labels = [f"{s}_{s + 4}" for s in bin_edges[:-1]]
    else:
        bin_edges, bin_labels = age_bins

    # Work on a copy limited to the covered age range
    df = df[df["age"].between(bin_edges[0], bin_edges[-1] - 1)].copy()

    df["age_bin"] = pd.cut(
        df["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # [40,45) ⇒ 40–44, etc.
    )

    age_groups = df.groupby("age_bin", observed=False)

    # -------- 2. Compute the statistic -------------------------------------------
    share_by_bin = (
        age_groups["choice"]
        .apply(lambda x: x.isin(np.atleast_1d(choice_set)).mean())
        .reindex(bin_labels, fill_value=np.nan)  # keep bins even if empty
    )

    # -------- 3. Write into *moments* --------------------------------------------
    for age_bin in bin_labels:
        moments[f"share_informal_care{label}_age_bin_{age_bin}"] = share_by_bin.loc[
            age_bin
        ]

    return moments


def compute_transition_moments_pandas_for_age_bins(
    df,
    moments,
    age_range,
    states,
    label=None,
    bin_width: int = 5,
):
    """
    Compute same-state transition probabilities aggregated into 5-year age bins.

    Processes one state fully across all bins before moving to the next.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'age', 'lagged_choice', and 'choice'.
    moments : dict
        Pre-existing moments dict to append to.
    age_range : iterable of int
        Sequence of ages to include in bins.
    states : dict
        Mapping of state labels to their codes in the data.
    label : str, optional
        Suffix label for keys.
    bin_width : int, default 5
        Width of each age bin.

    Returns
    -------
    moments : dict
        Updated with keys 'trans_<state>_to_<state>_<label>_age_<start>_<end>'.
    """
    suffix = f"_{label}" if label else ""
    start_age = min(age_range)
    end_age = max(age_range)

    # Build complete bins
    bins = []
    start = start_age
    while start + bin_width - 1 <= end_age:
        end = start + bin_width - 1
        bins.append((start, end))
        start += bin_width

    # Compute transitions per state then per bin
    for state_label, state_val in states.items():
        for start, end in bins:
            df_bin = df[(df["age"] >= start) & (df["age"] <= end)]
            subset = df_bin[df_bin["lagged_choice"].isin(np.atleast_1d(state_val))]
            if len(subset) > 0:
                prob = subset["choice"].isin(np.atleast_1d(state_val)).mean()
            else:
                prob = np.nan

            key = f"trans_{state_label}_to_{state_label}{suffix}_age_{start}_{end}"
            moments[key] = prob

    return moments


def compute_transition_moments_pandas(df, moments, age_range, states, label=None):
    # 2) Transition probabilities
    # Define the states of interest for transitions

    # For each "from" state, filter rows where lagged_choice is in that state,
    # and for each "to" state, compute the probability that 'choice' is in that state.
    # for from_label, from_val in states.items():
    #     # Use isin() to safely compare even if from_val is an array.
    #     subset = df[df["lagged_choice"].isin(np.atleast_1d(from_val))]
    #     for to_label, to_val in states.items():
    #         if len(subset) > 0:
    #             probability = subset["choice"].isin(np.atleast_1d(to_val)).mean()
    #         else:
    #             probability = np.nan  # or 0 if that is preferable
    #         moments[f"trans_{from_label}_to_{to_label}"] = probability

    if label is None:
        label = ""
    else:
        label = "_" + label

    for (from_label, from_val), (to_label, to_val) in product(
        states.items(), states.items()
    ):
        for age in age_range:
            df_age = df[df["age"] == age]
            subset = df_age[df_age["lagged_choice"].isin(np.atleast_1d(from_val))]
            if len(subset) > 0:
                bool_series = subset["choice"].isin(np.atleast_1d(to_val))
                probability = bool_series.mean()
            else:
                probability = np.nan

            key = f"trans_{from_label}_to_{to_label}{label}_age_{age}"
            moments[key] = probability

    return moments


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

    # df_low_educ = sim_df.loc[sim_df["education"] == 0]
    # df_high_educ = sim_df.loc[sim_df["education"] == 1]

    arr_low_educ = arr[arr[:, idx["education"]] == 0]
    arr_high_educ = arr[arr[:, idx["education"]] == 1]

    _care_mask = jnp.isin(arr[:, idx["choice"]], INFORMAL_CARE)
    arr_caregivers = arr[_care_mask]

    age_bins = [(40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)]

    # share_retired_by_age = get_share_by_age(
    #     arr, ind=idx, choice=RETIREMENT, min_age=min_age, max_age=max_age
    # )
    share_unemployed_by_age = get_share_by_age(
        arr, ind=idx, choice=UNEMPLOYED, min_age=min_age, max_age=max_age
    )
    share_working_part_time_by_age = get_share_by_age(
        arr, ind=idx, choice=PART_TIME, min_age=min_age, max_age=max_age
    )
    share_working_full_time_by_age = get_share_by_age(
        arr, ind=idx, choice=FULL_TIME, min_age=min_age, max_age=max_age
    )

    # share_retired_by_age_low_educ = get_share_by_age(
    #     arr_low_educ, ind=idx, choice=RETIREMENT, min_age=min_age, max_age=max_age
    # )
    share_unemployed_by_age_low_educ = get_share_by_age(
        arr_low_educ, ind=idx, choice=UNEMPLOYED, min_age=min_age, max_age=max_age
    )
    share_working_part_time_by_age_low_educ = get_share_by_age(
        arr_low_educ, ind=idx, choice=PART_TIME, min_age=min_age, max_age=max_age
    )
    share_working_full_time_by_age_low_educ = get_share_by_age(
        arr_low_educ, ind=idx, choice=FULL_TIME, min_age=min_age, max_age=max_age
    )

    # share_retired_by_age_high_educ = get_share_by_age(
    #     arr_high_educ, ind=idx, choice=RETIREMENT, min_age=min_age, max_age=max_age
    # )
    share_unemployed_by_age_high_educ = get_share_by_age(
        arr_high_educ, ind=idx, choice=UNEMPLOYED, min_age=min_age, max_age=max_age
    )
    share_working_part_time_by_age_high_educ = get_share_by_age(
        arr_high_educ, ind=idx, choice=PART_TIME, min_age=min_age, max_age=max_age
    )
    share_working_full_time_by_age_high_educ = get_share_by_age(
        arr_high_educ, ind=idx, choice=FULL_TIME, min_age=min_age, max_age=max_age
    )

    share_caregivers_by_age_bin = get_share_by_age_bin(
        arr, ind=idx, choice=INFORMAL_CARE, bins=age_bins
    )
    education_mask = arr[:, idx["education"]] == 1
    care_type_mask = jnp.isin(arr[:, idx["choice"]], INFORMAL_CARE)
    share_caregivers_high_educ = jnp.sum(education_mask & care_type_mask) / jnp.sum(
        care_type_mask
    )

    share_unemployed_by_age_bin_caregivers = get_share_by_age_bin(
        arr_caregivers, ind=idx, choice=UNEMPLOYED, bins=age_bins
    )
    share_working_part_time_by_age_bin_caregivers = get_share_by_age_bin(
        arr_caregivers, ind=idx, choice=PART_TIME, bins=age_bins
    )
    share_working_full_time_by_age_bin_caregivers = get_share_by_age_bin(
        arr_caregivers, ind=idx, choice=FULL_TIME, bins=age_bins
    )

    # Work transitions
    # work_to_work_low_educ_by_age = get_transition_for_age_bins(
    #     arr_low_educ,
    #     ind=idx,
    #     lagged_choice=WORK,
    #     current_choice=WORK,
    #     min_age=min_age,
    #     max_age=max_age,
    # )
    # no_work_to_no_work_low_educ_by_age = get_transition_for_age_bins(
    #     arr_low_educ,
    #     ind=idx,
    #     lagged_choice=NOT_WORKING,
    #     current_choice=NOT_WORKING,
    #     min_age=min_age,
    #     max_age=max_age,
    # )

    # work_to_work_high_educ_by_age = get_transition_for_age_bins(
    #     arr_high_educ,
    #     ind=idx,
    #     lagged_choice=WORK,
    #     current_choice=WORK,
    #     min_age=min_age,
    #     max_age=max_age,
    # )
    # no_work_to_no_work_high_educ_by_age = get_transition_for_age_bins(
    #     arr_high_educ,
    #     ind=idx,
    #     lagged_choice=NOT_WORKING,
    #     current_choice=NOT_WORKING,
    #     min_age=min_age,
    #     max_age=max_age,
    # )
    # #

    # no_work_to_part_time_by_age = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NOT_WORKING,
    #     current_choice=PART_TIME,
    #     min_age=min_age,
    #     max_age=max_age,
    # )
    # no_work_to_full_time = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NOT_WORKING,
    #     current_choice=FULL_TIME,
    #     min_age=min_age,
    #     max_age=max_age,
    # )

    # part_time_to_no_work = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=PART_TIME,
    #     current_choice=NOT_WORKING,
    #     min_age=min_age,
    #     max_age=max_age,
    # )
    # part_time_to_part_time = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=PART_TIME,
    #     current_choice=PART_TIME,
    #     min_age=min_age,
    #     max_age=max_age,
    # )
    # part_time_to_full_time = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=PART_TIME,
    #     current_choice=FULL_TIME,
    #     min_age=min_age,
    #     max_age=max_age,
    # )

    # full_time_to_no_work = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=FULL_TIME,
    #     current_choice=NOT_WORKING,
    #     min_age=min_age,
    #     max_age=max_age,
    # )
    # full_time_to_part_time = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=FULL_TIME,
    #     current_choice=PART_TIME,
    #     min_age=min_age,
    #     max_age=max_age,
    # )
    # full_time_to_full_time = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=FULL_TIME,
    #     current_choice=FULL_TIME,
    #     min_age=min_age,
    #     max_age=max_age,
    # )

    return jnp.asarray(
        # share_retired_by_age
        share_unemployed_by_age
        + share_working_part_time_by_age
        + share_working_full_time_by_age
        # + share_retired_by_age_low_educ
        + share_unemployed_by_age_low_educ
        + share_working_part_time_by_age_low_educ
        + share_working_full_time_by_age_low_educ
        # + share_retired_by_age_high_educ
        + share_unemployed_by_age_high_educ
        + share_working_part_time_by_age_high_educ
        + share_working_full_time_by_age_high_educ
        # caregivers
        + share_caregivers_by_age_bin
        + [share_caregivers_high_educ]
        + share_unemployed_by_age_bin_caregivers
        + share_working_part_time_by_age_bin_caregivers
        + share_working_full_time_by_age_bin_caregivers
        # # work to work transitions
        # + no_work_to_no_work_low_educ_by_age
        # + work_to_work_low_educ_by_age
        # + no_work_to_no_work_high_educ_by_age
        # + work_to_work_high_educ_by_age
        #
        # + work_to_work_by_age
        # + work_to_no_work_by_age
        # + no_work_to_work_by_age
        # + no_work_to_no_work_by_age
        # + no_work_to_part_time_by_age
        # + no_work_to_full_time
        # + part_time_to_no_work
        # + part_time_to_part_time
        # + part_time_to_full_time
        # + full_time_to_no_work
        # + full_time_to_part_time
        # + full_time_to_full_time
    )


# ==============================================================================
# Auxiliary
# ==============================================================================


def get_static_share(df_arr, ind, choice):
    """Compute static share of agents choosing one of the given choices."""
    choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    share = jnp.mean(choice_mask)
    return share


def get_share_by_age(df_arr, ind, choice, min_age, max_age):
    """Get share of agents choosing choice by age bin."""
    choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    shares = []
    for age in range(min_age, max_age + 1):
        age_mask = df_arr[:, ind["age"]] == age

        share = jnp.sum(age_mask & choice_mask) / jnp.sum(age_mask)
        # period count is always larger than 0! otherwise error
        shares.append(share)

    return shares


def get_share_by_age_bin(df_arr, ind, choice, bins):
    """Get share of agents choosing choice by age bin."""
    age_col = df_arr[:, ind["age"]]
    choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)

    shares: list[jnp.ndarray] = []
    for bin_start, bin_end in bins:
        age_mask = (age_col >= bin_start) & (age_col < bin_end)
        share = jnp.sum(age_mask & choice_mask) / jnp.sum(age_mask)
        shares.append(share)

    return shares


def _get_share_by_type(df_arr, ind, choice, care_type):
    """Get share of agents of given care type choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    care_type_mask = jnp.isin(df_arr[:, ind["choice"]], care_type)

    return jnp.sum(lagged_choice_mask & care_type_mask) / jnp.sum(care_type_mask)


def get_transition_for_age_bins(
    df_arr,
    ind,
    lagged_choice,
    current_choice,
    min_age,
    max_age,
    bin_width: int = 5,
):
    """
    Get transition probability from lagged_choice to current_choice,
    aggregated into fixed-width age bins (default 5-year bins).

    Only complete bins are returned (no partial final bin).

    Parameters
    ----------
    df_arr : np.ndarray or jnp.ndarray
        2D array where rows are observations and columns indexed by `ind`.
    ind : dict
        Mapping of column names ('age', 'lagged_choice', 'choice') to integer indices.
    lagged_choice : scalar or iterable
        Value(s) representing the previous state.
    current_choice : scalar or iterable
        Value(s) representing the current state.
    min_age : int
        Minimum age (inclusive) for binning.
    max_age : int
        Maximum age (inclusive) for binning.
    bin_width : int, default 5
        Width of each age bin in years.

    Returns
    -------
    List of transition probabilities for each 5-year bin.
    """
    probs = []
    # Build only complete bins
    bins = []
    start = min_age
    while start + bin_width - 1 <= max_age:
        end = start + bin_width - 1
        bins.append((start, end))
        start += bin_width

    # Compute per-bin transition
    for bin_start, bin_end in bins:
        # Masks for age within the bin
        ages = df_arr[:, ind["age"]]
        age_mask = (ages >= bin_start) & (ages <= bin_end)
        # Masks for lagged and current states
        lagged_mask = jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
        current_mask = jnp.isin(df_arr[:, ind["choice"]], current_choice)

        num = jnp.sum(age_mask & lagged_mask & current_mask)
        den = jnp.sum(age_mask & lagged_mask)
        prob = num / den if den > 0 else jnp.nan
        probs.append(prob)

    return probs


def get_transition(df_arr, ind, lagged_choice, current_choice, min_age, max_age):
    """Get transition probability from lagged choice to current choice."""
    # return [
    #     jnp.sum(
    #         jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
    #         & jnp.isin(df_arr[:, ind["choice"]], current_choice),
    #     )
    #     / jnp.sum(jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)),
    # ]
    probs = []
    for age in range(min_age, max_age + 1):
        # Create a mask selecting only the rows for the current age.
        age_mask = df_arr[:, ind["age"]] == age

        # Create masks for the lagged and current state. The use of jnp.isin
        # allows lagged_choice and current_choice to be single values or iterables.
        lagged_mask = jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
        current_mask = jnp.isin(df_arr[:, ind["choice"]], current_choice)

        num = jnp.sum(age_mask & lagged_mask & current_mask)
        den = jnp.sum(age_mask & lagged_mask)

        # Handle potential division by zero.
        prob = num / den if den > 0 else jnp.nan
        probs.append(prob)

    return probs


def _get_share_by_age_bin(df_arr, ind, choice, age_bins):
    """Get share of agents choosing lagged choice by age bin."""
    return [
        jnp.mean(
            jnp.isin(df_arr[:, ind["choice"]], choice)
            & (df_arr[:, ind["period"]] > age_bin[0])
            & (df_arr[:, ind["period"]] <= age_bin[1]),
        )
        for age_bin in age_bins
    ]


def _get_mean_by_age_bin_for_lagged_choice(df_arr, ind, var, choice, age_bins):
    """Get mean of agents choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    means = []

    for age_bin in age_bins:
        age_bin_mask = (df_arr[:, ind["period"]] > age_bin[0]) & (
            df_arr[:, ind["period"]] <= age_bin[1]
        )
        means += [jnp.mean(df_arr[lagged_choice_mask & age_bin_mask, ind[var]])]

    return means


def _get_share_by_type_by_age_bin(df_arr, ind, choice, care_type, age_bins):
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


# =====================================================================================
# Plotting
# =====================================================================================


def plot_model_fit_labor_moments_pandas_by_education(
    moms_emp: pd.Series,
    moms_sim: pd.Series,
    specs: dict,
    path_to_save_plot: Optional[str] = None,
) -> None:
    """
    Plots the age specific labor supply shares (choice shares) for four states:
    retired, unemployed, part-time, and full-time based on the empirical
    and simulated moments.

    Both data_emp and data_sim are pandas Series indexed by moment names in the format:
      "share_{state}_age_{age}"
    e.g., "share_retired_age_30", "share_unemployed_age_40", etc.

    Parameters
    ----------
    data_emp : pd.Series
        Empirical moments with keys like "share_retired_age_30", etc.
    data_sim : pd.Series
        Simulated moments with the same key naming convention.
    path_to_save_plot : str
        File path to save the generated plot.
    """

    choices = ["retired", "unemployed", "part_time", "full_time"]

    fig, axs = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)

    for edu_var, edu_label in enumerate(specs["education_labels"]):

        for choice_var, choice_label in enumerate(specs["choice_labels"]):

            ax = axs[edu_var, choice_var]

            emp_keys = [
                k
                for k in moms_emp.index
                if k.startswith(f"share_{choices[choice_var]}_")
                and str(edu_label.lower().replace(" ", "_")) in k
            ]
            sim_keys = [
                k
                for k in moms_sim.index
                if k.startswith(f"share_{choices[choice_var]}_")
                and str(edu_label.lower().replace(" ", "_")) in k
            ]

            # Sort the keys by age.
            emp_keys_sorted = sorted(emp_keys, key=_extract_age)
            sim_keys_sorted = sorted(sim_keys, key=_extract_age)

            # Build lists of ages and corresponding values.
            emp_ages = [_extract_age(k) for k in emp_keys_sorted]
            emp_values = [moms_emp[k] for k in emp_keys_sorted]
            sim_ages = [_extract_age(k) for k in sim_keys_sorted]
            sim_values = [moms_sim[k] for k in sim_keys_sorted]

            # Plot empirical and simulated shares.
            ax.plot(sim_ages, sim_values, label="Simulated")
            ax.plot(emp_ages, emp_values, label="Observed", ls="--")

            ax.set_xlabel("Age")
            ax.set_ylim([0, 1])

            # if edu_var == 0:
            ax.set_title(choice_label)
            ax.tick_params(labelbottom=True)

            if choice_var == 0:
                ax.set_ylabel(edu_label + "\nShare")
                ax.legend()
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, transparent=False, dpi=300)


def plot_model_fit_labor_moments_pandas(
    moms_emp: pd.Series, moms_sim: pd.Series, path_to_save_plot: Optional[str] = None
) -> None:
    """
    Plots the age specific labor supply shares (choice shares) for four states:
    retired, unemployed, part-time, and full-time based on the empirical
    and simulated moments.

    Both data_emp and data_sim are pandas Series indexed by moment names in the format:
      "share_{state}_age_{age}"
    e.g., "share_retired_age_30", "share_unemployed_age_40", etc.

    Parameters
    ----------
    data_emp : pd.Series
        Empirical moments with keys like "share_retired_age_30", etc.
    data_sim : pd.Series
        Simulated moments with the same key naming convention.
    path_to_save_plot : str
        File path to save the generated plot.
    """

    # Define the states of interest.
    states = ["retired", "unemployed", "part_time", "full_time"]

    fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharex=True, sharey=True)
    axs = axs.flatten()

    # Loop through each state to plot its empirical and simulated shares.
    for ax, state in zip(axs, states, strict=False):
        # Filter the keys for the given state for both empirical and simulated data.
        emp_keys = [k for k in moms_emp.index if k.startswith(f"share_{state}_age_")]
        sim_keys = [k for k in moms_sim.index if k.startswith(f"share_{state}_age_")]

        # Sort the keys by age.
        emp_keys_sorted = sorted(emp_keys, key=_extract_age)
        sim_keys_sorted = sorted(sim_keys, key=_extract_age)

        # Build lists of ages and corresponding values.
        emp_ages = [_extract_age(k) for k in emp_keys_sorted]
        emp_values = [moms_emp[k] for k in emp_keys_sorted]
        sim_ages = [_extract_age(k) for k in sim_keys_sorted]
        sim_values = [moms_sim[k] for k in sim_keys_sorted]

        # Plot empirical and simulated shares.
        ax.plot(sim_ages, sim_values, label="Simulated")
        ax.plot(emp_ages, emp_values, label="Observed", ls="--")

        ax.set_title(state.capitalize())
        ax.set_xlabel("Age")
        ax.set_ylim([0, 1])

        if state == "retired":
            ax.set_ylabel("Share")
            ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=True)


def plot_transition_shares_by_age_bins(
    moms_emp: pd.Series,
    moms_sim: pd.Series,
    specs: dict,
    states: dict,
    state_labels: Optional[dict] = None,
    path_to_save_plot: Optional[str] = None,
    bin_width: int = 5,
) -> None:
    """
    Plot age-specific transition probabilities for each from-to state pair by age bins.

    Parameters
    ----------
    moms_emp : pd.Series
        Empirical transition moments, indexed by keys like
        'trans_{state}_to_{state}_{label}_age_{start}_{end}'.
    moms_sim : pd.Series
        Simulated transition moments, same indexing convention.
    specs : dict
        Specification dict containing 'education_labels'.
    states : dict
        Mapping of state-labels to their underlying values.
    state_labels : dict, optional
        Pretty names for each state key. If None, will use .capitalize().
    path_to_save_plot : str, optional
        If given, where to save the resulting figure (PNG).
    bin_width : int, default 5
        Width of each age bin (for axis labeling).
    """

    # Default pretty names
    if state_labels is None:
        state_labels = {s: s.replace("_", " ").capitalize() for s in states}

    edu_levels = specs.get("education_labels", [""])
    n_edu = len(edu_levels)
    n_states = len(states)

    fig, axs = plt.subplots(
        n_edu, n_states, figsize=(4 * n_states, 3 * n_edu), sharex=True, sharey=True
    )
    axs = np.atleast_2d(axs)

    for edu_var, edu_label in enumerate(edu_levels):
        suffix = edu_label.lower().replace(" ", "_")

        for i, state_key in enumerate(states):
            ax = axs[edu_var, i]
            prefix = f"trans_{state_key}_to_{state_key}"
            emp_keys = sorted(
                [k for k in moms_emp.index if k.startswith(prefix) and suffix in k],
                key=_bin_start,
            )
            sim_keys = sorted(
                [k for k in moms_sim.index if k.startswith(prefix) and suffix in k],
                key=_bin_start,
            )
            x_emp = [_bin_start(k) for k in emp_keys]
            y_emp = [moms_emp[k] for k in emp_keys]
            x_sim = [_bin_start(k) for k in sim_keys]
            y_sim = [moms_sim[k] for k in sim_keys]

            ax.plot(x_sim, y_sim, label="Simulated")
            ax.plot(x_emp, y_emp, ls="--", label="Observed")

            ax.set_title(f"{state_labels[state_key]} → {state_labels[state_key]}")
            ax.set_xlabel("Age Bin")

            if i == 0:
                ax.set_ylabel(
                    f"{edu_label}\nTransition Rate" if edu_label else "Transition Rate"
                )
                ax.legend()

            ax.set_ylim(0, 1)

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=True)


# =====================================================================================
# JAX numpy
# =====================================================================================


def plot_model_fit_labor_moments_by_education_pandas_jax(
    moms_emp: pd.Series,
    moms_sim: jnp.ndarray,
    specs: dict,
    path_to_save_plot: Optional[str] = None,
) -> None:
    """
    Plots the age specific labor supply shares (choice shares) for four states:
    retired, unemployed, part-time, and full-time based on the empirical
    and simulated moments.

    Both data_emp and data_sim are pandas Series indexed by moment names in the format:
      "share_{state}_age_{age}"
    e.g., "share_retired_age_30", "share_unemployed_age_40", etc.

    Parameters
    ----------
    data_emp : pd.Series
        Empirical moments with keys like "share_retired_age_30", etc.
    data_sim : pd.Series
        Simulated moments with the same key naming convention.
    path_to_save_plot : str
        File path to save the generated plot.
    """

    choices = ["retired", "unemployed", "part_time", "full_time"]

    sim_array = np.asarray(moms_sim)

    fig, axs = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)

    for edu_var, edu_label in enumerate(specs["education_labels"]):

        for choice_var, choice_label in enumerate(specs["choice_labels"]):

            ax = axs[edu_var, choice_var]

            # Get positions where keys match the current state. This preserves the order
            # of the empirical Series.
            indices = [
                i
                for i, k in enumerate(moms_emp.index)
                # if key.startswith(f"share_{state}_age_")
                if k.startswith(f"share_{choices[choice_var]}_")
                and str(edu_label.lower().replace(" ", "_")) in k
            ]
            # Retrieve the keys for these positions.
            keys = [moms_emp.index[i] for i in indices]
            # Extract the ages from these keys.
            ages = [_extract_age(key) for key in keys]

            # Extract empirical values using iloc and simulated values from the jax
            # array using the same indices.
            emp_values = moms_emp.iloc[indices].values
            sim_values = sim_array[indices]

            # Plot empirical and simulated shares.
            ax.plot(ages, sim_values, label="Simulated")
            ax.plot(ages, emp_values, label="Observed", ls="--")

            ax.set_xlabel("Age")
            ax.set_ylim([0, 1])

            # if edu_var == 0:
            ax.set_title(choice_label)
            ax.tick_params(labelbottom=True)

            if choice_var == 0:
                ax.set_ylabel(edu_label + "\nShare")
                ax.legend()
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, transparent=False, dpi=300)


def plot_model_fit_labor_moments_pandas_jax(
    moms_emp: pd.Series, moms_sim: jnp.ndarray, path_to_save_plot: Optional[str] = None
) -> None:
    """
    Plots the age specific labor supply shares (choice shares) for four states:
    retired, unemployed, part-time, and full-time based on the empirical
    and simulated moments.

    Both data_emp and data_sim are pandas Series indexed by moment names in the format:
      "share_{state}_age_{age}"
    e.g., "share_retired_age_30", "share_unemployed_age_40", etc.

    Parameters
    ----------
    data_emp : pd.Series
        Empirical moments with keys like "share_retired_age_30", etc.
    data_sim : pd.Series
        Simulated moments with the same key naming convention.
    path_to_save_plot : str
        File path to save the generated plot.
    """

    # Define the states of interest.
    states = ["retired", "unemployed", "part_time", "full_time"]

    # Convert the simulated array to a NumPy array (if needed).
    sim_array = np.asarray(moms_sim)

    fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharex=True, sharey=True)
    axs = axs.flatten()

    # Loop through each state to plot its empirical and simulated shares.
    for ax, state in zip(axs, states, strict=False):

        # Get positions where keys match the current state. This preserves the order
        # of the empirical Series.
        indices = [
            i
            for i, key in enumerate(moms_emp.index)
            if key.startswith(f"share_{state}_age_")
        ]
        # Retrieve the keys for these positions.
        keys = [moms_emp.index[i] for i in indices]
        # Extract the ages from these keys.
        ages = [_extract_age(key) for key in keys]

        # Extract empirical values using iloc and simulated values from the jax array
        # using the same indices.
        emp_values = moms_emp.iloc[indices].values
        sim_values = sim_array[indices]

        # Plot empirical and simulated shares.
        ax.plot(ages, sim_values, label="Simulated")
        ax.plot(ages, emp_values, label="Observed", ls="--")

        ax.set_title(state.capitalize())
        ax.set_xlabel("Age")
        ax.set_ylim([0, 1])

        if state == "retired":
            ax.set_ylabel("Share")
            ax.legend()

    plt.tight_layout()
    plt.savefig(path_to_save_plot, transparent=True, dpi=300)


# Helper function to extract age from key (assumes format: "share_{state}_age_{age}")


def _extract_age(key: str) -> int:
    # The age is assumed to be the number after the last underscore.
    return int(key.split("_")[-1])


def _bin_start(key: str) -> int:
    m = re.search(r"_age_(\d+)_", key)
    return int(m.group(1)) if m else 0
