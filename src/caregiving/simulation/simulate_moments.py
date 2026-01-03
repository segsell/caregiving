"""Simulate moments."""

import re
from itertools import product
from typing import Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.data_management.share.task_create_parent_child_data_set import (
    AGE_BINS_PARENTS,
    AGE_LABELS_PARENTS,
)
from caregiving.model.shared import (  # NURSING_HOME_CARE,
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    CARE_DEMAND_AND_OTHER_SUPPLY,
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    INFORMAL_CARE_OR_OTHER_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_CARE,
    NO_INFORMAL_CARE,
    NO_NURSING_HOME_CARE,
    NOT_WORKING,
    NOT_WORKING_CARE,
    PART_TIME,
    RETIREMENT,
    SCALE_CAREGIVER_SHARE,
    SEX,
    UNEMPLOYED,
    WEALTH_MOMENTS_SCALE,
    WORK,
)

FILL_VALUE_MISSING_AGE = 0  # np.nan

# =====================================================================================
# Pandas
# =====================================================================================


def simulate_moments_pandas(  # noqa: PLR0915
    df_full,
    model_specs,
) -> pd.DataFrame:
    """Simulate the model for given parametrization and model solution."""

    start_age = model_specs["start_age"]
    # Prefer directly stored caregiving start age if available
    start_age_caregivers = model_specs["start_age_caregiving"]
    end_age = model_specs["end_age_msm"]

    age_range = range(start_age, end_age + 1)
    age_range_caregivers = range(start_age_caregivers, end_age + 1)
    age_range_wealth = range(start_age, model_specs["end_age_wealth"] + 1)

    age_bins_caregivers_5year = (
        list(range(45, 75, 5)),  # [40, 45, 50, 55, 60, 65, 70]
        [
            f"{s}_{s+4}" for s in range(45, 70, 5)
        ],  # ["40_44", "45_49", "50_54", "55_59", "60_64", "65_69"]
    )
    # age_bins_75 = (
    #     list(range(40, 80, 5)),  # [40, 45, … , 70]
    #     [f"{s}_{s+4}" for s in range(40, 75, 5)],  # "40_44", …a
    # )

    df_full = df_full.loc[df_full["health"] != DEAD].copy()
    df_full["mother_age"] = (
        df_full["age"].to_numpy()
        + model_specs["mother_age_diff"][df_full["education"].to_numpy()]
    )

    # Only non-caregivers
    df_non_caregivers = df_full[
        df_full["choice"].isin(np.asarray(NO_INFORMAL_CARE).tolist())
    ].copy()

    df_low = df_non_caregivers[df_non_caregivers["education"] == 0]
    df_high = df_non_caregivers[df_non_caregivers["education"] == 1]

    df_wealth_low = df_full[df_full["education"] == 0]
    df_wealth_high = df_full[df_full["education"] == 1]

    df_caregivers = df_full[
        df_full["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
    ].copy()
    df_caregivers_low = df_caregivers[df_caregivers["education"] == 0]
    df_caregivers_high = df_caregivers[df_caregivers["education"] == 1]

    # =================================================================================
    df_light_caregivers = df_full[
        df_full["choice"].isin(np.asarray(LIGHT_INFORMAL_CARE).tolist())
    ].copy()
    df_light_caregivers_low = df_light_caregivers[df_light_caregivers["education"] == 0]
    df_light_caregivers_high = df_light_caregivers[
        df_light_caregivers["education"] == 1
    ]

    df_intensive_caregivers = df_full[
        df_full["choice"].isin(np.asarray(INTENSIVE_INFORMAL_CARE).tolist())
    ].copy()
    df_intensive_caregivers_low = df_intensive_caregivers[
        df_intensive_caregivers["education"] == 0
    ]
    df_intensive_caregivers_high = df_intensive_caregivers[
        df_intensive_caregivers["education"] == 1
    ]

    # includes "no care", which means other informal care if positive care demand
    # if other care_supply == 0 and no personal care provided
    # --> formal home care (implicit)
    # df_domestic = df_full.loc[
    #     (df_full["choice"].isin(
    # np.asarray(NO_NURSING_HOME_CARE))) & (df_full["care_demand"] >= 1)
    # ].copy()
    # df_parent_bad_health = df_full[
    #     df_full["mother_health"] == PARENT_BAD_HEALTH
    # ].copy()
    # =================================================================================

    moments = {}

    # =================================================================================
    # Wealth moments
    moments = create_mean_by_age(
        df_wealth_low,
        moments,
        variable="assets_begin_of_period",
        age_range=age_range_wealth,
        label="low_education",
    )
    moments = create_mean_by_age(
        df_wealth_high,
        moments,
        variable="assets_begin_of_period",
        age_range=age_range_wealth,
        label="high_education",
    )

    # =================================================================================
    moments = create_labor_share_moments_pandas(
        df_non_caregivers, moments, age_range=age_range
    )
    moments = create_labor_share_moments_pandas(
        df_low, moments, age_range=age_range, label="low_education"
    )
    moments = create_labor_share_moments_pandas(
        df_high, moments, age_range=age_range, label="high_education"
    )

    moments = create_choice_shares_by_age_bin_pandas(
        df_full,
        moments,
        choice_set=INFORMAL_CARE,
        age_bins_and_labels=age_bins_caregivers_5year,
        label="informal_care",
        scale=SCALE_CAREGIVER_SHARE,
    )
    # =================================================================================

    # moments = create_choice_shares_by_age_bin_pandas(
    #     df, moments, choice_set=LIGHT_INFORMAL_CARE, age_bins=age_bins_75
    # )
    # moments = create_choice_shares_by_age_bin_pandas(
    #     df, moments, choice_set=INTENSIVE_INFORMAL_CARE, age_bins=age_bins_75
    # )

    # ================================================================================
    moments["share_informal_care_high_educ"] = df_caregivers["education"].mean()
    # ================================================================================

    # Labor caregiver shares using 3-year age bins
    age_bins_caregivers_3year = (
        list(
            range(start_age_caregivers, end_age + 1, 3)
        ),  # [40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70]
        [
            f"{s}_{s+2}" for s in range(start_age_caregivers, end_age - 1, 3)
        ],  # ["40_42", "43_45", "46_48", "49_51", "52_54", "55_57",
        # "58_60", "61_63", "64_66", "67_69"]
    )

    moments = create_labor_share_moments_by_age_bin_pandas(
        df_caregivers, moments, age_bins=age_bins_caregivers_3year, label="caregivers"
    )
    moments = create_labor_share_moments_by_age_bin_pandas(
        df_caregivers_low,
        moments,
        age_bins=age_bins_caregivers_3year,
        label="caregivers_low_education",
    )
    moments = create_labor_share_moments_by_age_bin_pandas(
        df_caregivers_high,
        moments,
        age_bins=age_bins_caregivers_3year,
        label="caregivers_high_education",
    )

    # Light informal caregiving
    moments = create_labor_share_moments_by_age_bin_pandas(
        df_light_caregivers,
        moments,
        label="light_caregivers",
        age_bins=age_bins_caregivers_3year,
    )
    moments = create_labor_share_moments_by_age_bin_pandas(
        df_light_caregivers_low,
        moments,
        label="light_caregivers_low_education",
        age_bins=age_bins_caregivers_3year,
    )
    moments = create_labor_share_moments_by_age_bin_pandas(
        df_light_caregivers_high,
        moments,
        label="light_caregivers_high_education",
        age_bins=age_bins_caregivers_3year,
    )

    # Intensive informal caregiving
    moments = create_labor_share_moments_by_age_bin_pandas(
        df_intensive_caregivers,
        moments,
        label="intensive_caregivers",
        age_bins=age_bins_caregivers_3year,
    )
    moments = create_labor_share_moments_by_age_bin_pandas(
        df_intensive_caregivers_low,
        moments,
        label="intensive_caregivers_low_education",
        age_bins=age_bins_caregivers_3year,
    )
    moments = create_labor_share_moments_by_age_bin_pandas(
        df_intensive_caregivers_high,
        moments,
        label="intensive_caregivers_high_education",
        age_bins=age_bins_caregivers_3year,
    )

    # states = {
    #     "not_working": NOT_WORKING,
    #     "part_time": PART_TIME,
    #     "full_time": FULL_TIME,
    # }
    # moments = compute_transition_moments_pandas(df, moments, age_range, states=states)

    states_work_no_work = {
        "not_working": NOT_WORKING,
        "working": WORK,
    }
    # Use df_with_caregivers to match empirical data (task_create_soep_moments.py)
    # Create df_with_caregivers equivalent (all observations, not just non-caregivers)
    df_with_caregivers = df_full.copy()  # Pooled across all education levels
    df_with_caregivers_low = df_full[df_full["education"] == 0]
    df_with_caregivers_high = df_full[df_full["education"] == 1]

    moments = compute_transition_moments_pandas_for_age_bins(
        df_with_caregivers_low,
        moments,
        age_range,
        states=states_work_no_work,
        label="low_education",
    )
    moments = compute_transition_moments_pandas_for_age_bins(
        df_with_caregivers_high,
        moments,
        age_range,
        states=states_work_no_work,
        label="high_education",
    )

    # Caregiving transitions (informal care to informal care)
    # Use age_range_caregivers (starts at start_age_caregivers) instead of age_range
    # Pool across education levels (all_education) to match empirical moment creation
    # Note: Only compute "caregiving to caregiving" transitions to match empirical
    # (which uses states_caregiving = {"caregiving": 1})
    states_caregiving = {
        "caregiving": INFORMAL_CARE,
    }
    moments = compute_transition_moments_pandas_for_age_bins(
        df_with_caregivers,  # Pooled across all education levels
        moments,
        age_range_caregivers,
        states=states_caregiving,
        label="all_education",
    )
    # states_light_informal = {
    #     "no_light_informal_care": NO_LIGHT_INFORMAL_CARE,
    #     "light_informal_care": LIGHT_INFORMAL_CARE
    # }
    # states_intensive_informal = {
    #     "no_intensive_informal_care": NO_INTENSIVE_INFORMAL_CARE,
    #     "intensive_informal_care": INTENSIVE_INFORMAL_CARE
    # }
    # moments = compute_transition_moments_pandas_for_age_bins(
    #     df, moments, age_range, states=states_informal_care
    # )
    # moments = compute_transition_moments_pandas_for_age_bins(
    #     df, moments, age_range, states=states_light_informal_care
    # )
    # moments = compute_transition_moments_pandas_for_age_bins(
    #     df, moments, age_range, states=states_intensive_informal_care
    # )

    # # ================================================================================
    # # Care mix by parent age

    # age_bins_parents_to_agent = (AGE_BINS_PARENTS, AGE_LABELS_PARENTS)

    # # Create nursing_home indicator: formal care occurs when care_demand exists
    # # and no one else provides care (CARE_DEMAND_AND_NO_OTHER_SUPPLY) AND
    # # the agent chooses NO_CARE (meaning formal care is organized)

    # df_parent_bad_health = df_full[
    #     (df_full["mother_health"] != PARENT_DEAD)
    #     & (df_full["mother_health"] == PARENT_BAD_HEALTH)
    #     & (df_full["care_demand"] > 0)
    # ].copy()
    # no_care_choices = np.asarray(NO_CARE).tolist()
    # df_parent_bad_health["nursing_home"] = (
    #     df_parent_bad_health["choice"].isin(no_care_choices)
    #     & (df_parent_bad_health["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
    # ).astype(int)

    # # Compute shares by age bin for nursing home (matching empirical moment creation)
    # moments = compute_shares_by_age_bin_pandas(
    #     df_parent_bad_health,
    #     moments,
    #     variable="nursing_home",
    #     age_bins=age_bins_parents_to_agent,
    #     label="parents_nursing_home",
    #     age_var="mother_age",
    # )
    # # ================================================================================

    # # moments = create_choice_shares_by_age_bin_pandas(
    # #     df_domestic_care,
    #     moments,
    #     choice_set=INFORMAL_CARE_OR_OTHER_CARE,
    #     age_bins_and_labels=age_bins_parents_to_agent,
    #     label="informal_care_or_other_care",
    #     age_var="mother_age",
    # )
    # moments = create_choice_shares_by_age_bin_pandas(
    #     df_domestic_care,
    #     moments,
    #     choice_set=LIGHT_INFORMAL_CARE,
    #     age_bins_and_labels=age_bins_parents_to_agent,
    #     label="light_informal_care",  # := combination care
    #     age_var="mother_age",
    # )
    # moments = create_choice_shares_by_age_bin_pandas(
    #     df_domestic_care,
    #     moments,
    #     choice_set=NO_INFORMAL_CARE,
    #     age_bins_and_labels=age_bins_parents_to_agent,
    #     label="formal_home_care",
    #     age_var="mother_age",
    # )

    # ==================================================================================

    # # Age bins on mother_age
    # df_domestic["age_bin"] = pd.cut(
    #     df_domestic["mother_age"],
    #     bins=AGE_BINS_PARENTS,  # e.g. [65, 70, 75, 80, 85, 90, np.inf]
    #     labels=AGE_LABELS_PARENTS,  # ["65_69", ..., "90_plus"]
    #     right=False,  # [65,70), [70,75), ...
    # )

    # # --- 2) Build masks on the domestic subset (match JAX logic) ------------------
    # is_intensive_informal = df_domestic["choice"].isin(
    #     np.asarray(INTENSIVE_INFORMAL_CARE)
    # )
    # is_no_care = df_domestic["choice"].isin(np.asarray(NO_CARE))

    # # supply flags from care_demand coding (same as in JAX)
    # is_supply = df_domestic["care_demand"].eq(CARE_DEMAND_AND_OTHER_SUPPLY)
    # # no_other_supply = df_domestic["care_demand"].eq(CARE_DEMAND_AND_NO_OTHER_SUPPLY)

    # # (2) informal_care_or_other_care  == intensive_informal  OR  (no_care & supply)
    # df_domestic["m_informal_or_other"] = is_intensive_informal | (
    #     is_no_care & is_supply
    # )

    # # (3) light_informal_care          == pure choice set LIGHT_INFORMAL_CARE
    # df_domestic["m_light_informal"] = df_domestic["choice"].isin(
    #     np.asarray(LIGHT_INFORMAL_CARE)
    # )

    # # (4) formal_home_care             == (no_care & supply==0)
    # df_domestic["m_formal_home"] = is_no_care & (~is_supply)

    # # --- 3) Group by age bin and take means (shares) ------------------------------
    # grp = df_domestic.groupby("age_bin", observed=True)

    # share_informal_or_other = (
    #     grp["m_informal_or_other"].mean().reindex(AGE_LABELS_PARENTS)
    # )
    # share_light_informal = grp["m_light_informal"].mean().reindex(AGE_LABELS_PARENTS)
    # share_formal_home = grp["m_formal_home"].mean().reindex(AGE_LABELS_PARENTS)

    # # (Optional) write into your moments dict with your naming scheme
    # for age_bin, val in share_informal_or_other.items():
    #     moments[f"share_informal_care_or_other_care_age_bin_{age_bin}"] = val

    # for age_bin, val in share_light_informal.items():
    #     moments[f"share_light_informal_care_age_bin_{age_bin}"] = val

    # for age_bin, val in share_formal_home.items():
    #     moments[f"share_formal_home_care_age_bin_{age_bin}"] = val

    # ==================================================================================

    return pd.Series(moments)


def create_labor_share_moments_pandas(df, moments, age_range, label=None):
    """
    Create a Pandas Series of simulation moments using an optimized method.

    This function computes age-specific shares by creating choice groups and using
    value_counts(normalize=True), which is the same method used in the plotting
    functions for consistency. This version is optimized for performance while
    maintaining correctness.

    Assumes that the DataFrame `df` contains at least the following columns:
      - age
      - choice

    Parameters:
        df (pd.DataFrame): The simulation DataFrame.
        moments (dict): Dictionary to store the computed moments.
        age_range (range): The age range for computing age-specific shares.
        label (str, optional): Label to append to moment names.

    Returns:
        dict: Updated moments dictionary with computed labor share moments.
    """

    if label is None:
        label = ""
    else:
        label = "_" + label

    # Create choice groups mapping
    choice_groups = {
        0: RETIREMENT,
        1: UNEMPLOYED,
        2: PART_TIME,
        3: FULL_TIME,
    }

    # OPTIMIZATION 1: Use vectorized mapping instead of loops
    choice_group_map = {}
    for agg_code, raw_codes in choice_groups.items():
        for code in raw_codes.tolist():  # Convert JAX array to Python list
            choice_group_map[code] = agg_code

    # OPTIMIZATION 2: Use map() instead of copy + multiple loc operations
    df_copy = df.copy()
    df_copy["choice_group"] = (
        df_copy["choice"].map(choice_group_map).fillna(0).astype(int)
    )

    # OPTIMIZATION 3: Single groupby operation
    shares_by_age = (
        df_copy.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # OPTIMIZATION 4: Vectorized reindexing
    shares_by_age = shares_by_age.reindex(age_range, fill_value=0)

    # OPTIMIZATION 5: Vectorized dictionary population
    choice_labels = ["retired", "unemployed", "part_time", "full_time"]

    # Create all moment names and values at once
    moment_data = []
    for choice_var, choice_label in enumerate(choice_labels):
        for age in age_range:
            if choice_var in shares_by_age.columns:
                value = shares_by_age.loc[age, choice_var]
            else:
                value = 0.0
            moment_data.append((f"share_{choice_label}{label}_age_{age}", value))

    # OPTIMIZATION 6: Bulk dictionary update
    moments.update(dict(moment_data))

    return moments


def create_labor_share_moments_pandas_backup(df, moments, age_range, label=None):
    """
    Create a Pandas Series of simulation moments using the same method as
    plot_choice_shares_by_education.

    This function computes age-specific shares by creating choice groups and using
    value_counts(normalize=True), which is the same method used in the plotting
    functions for consistency.

    Assumes that the DataFrame `df` contains at least the following columns:
      - age
      - choice

    Parameters:
        df (pd.DataFrame): The simulation DataFrame.
        moments (dict): Dictionary to store the computed moments.
        age_range (range): The age range for computing age-specific shares.
        label (str, optional): Label to append to moment names.

    Returns:
        dict: Updated moments dictionary with computed labor share moments.
    """

    if label is None:
        label = ""
    else:
        label = "_" + label

    # Create choice groups mapping (same as in plot_choice_shares_by_education)
    choice_groups = {
        0: RETIREMENT,
        1: UNEMPLOYED,
        2: PART_TIME,
        3: FULL_TIME,
    }

    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Map raw choice codes to aggregated choice groups (same as plotting function)
    for agg_code, raw_codes in choice_groups.items():
        df_copy.loc[
            df_copy["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code

    # Fill any missing choice_group values with 0 (retirement)
    df_copy["choice_group"] = df_copy["choice_group"].fillna(0).astype(int)

    # Calculate shares by age using value_counts(normalize=True) - same as plotting
    # function
    shares_by_age = (
        df_copy.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # Reindex to ensure all ages in age_range are included
    shares_by_age = shares_by_age.reindex(age_range, fill_value=0)

    # Populate the moments dictionary for age-specific shares
    choice_labels = ["retired", "unemployed", "part_time", "full_time"]

    for choice_var, choice_label in enumerate(choice_labels):
        for age in age_range:
            if choice_var in shares_by_age.columns:
                moments[f"share_{choice_label}{label}_age_{age}"] = shares_by_age.loc[
                    age, choice_var
                ]
            else:
                moments[f"share_{choice_label}{label}_age_{age}"] = 0.0

    return moments


def create_labor_share_moments_by_age_bin_pandas(
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

    # Create choice groups mapping (same robust approach as other functions)
    choice_groups = {
        0: RETIREMENT,
        1: UNEMPLOYED,
        2: PART_TIME,
        3: FULL_TIME,
    }

    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Map raw choice codes to aggregated choice groups (same as plotting function)
    for agg_code, raw_codes in choice_groups.items():
        df_copy.loc[
            df_copy["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code

    # Fill any missing choice_group values with 0 (retirement)
    df_copy["choice_group"] = df_copy["choice_group"].fillna(0).astype(int)

    # Calculate shares by age bin using value_counts(normalize=True) - robust approach
    shares_by_bin = (
        df_copy.groupby("age_bin", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # Reindex to ensure all bins are included
    shares_by_bin = shares_by_bin.reindex(bin_labels, fill_value=0)

    # Populate the moments dictionary for age bin-specific shares
    choice_labels = ["retired", "unemployed", "part_time", "full_time"]

    for choice_var, choice_label in enumerate(choice_labels):
        for age_bin in bin_labels:
            if choice_var in shares_by_bin.columns:
                value = shares_by_bin.loc[age_bin, choice_var]
            else:
                value = 0.0
            moments[f"share_{choice_label}{label}_age_bin_{age_bin}"] = value

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
        df.groupby("age", observed=False)["choice"]
        .apply(lambda x: x.isin(np.atleast_1d(choice_set)).mean())
        .reindex(age_range, fill_value=np.nan)
    )

    for age in age_range:
        moments[f"share_informal_care{label}_age_{age}"] = share_by_age.loc[age]

    return moments


def create_choice_shares_by_age_bin_pandas(
    df: pd.DataFrame,
    moments: dict,
    *,
    choice_set: jnp.ndarray,
    age_bins_and_labels: tuple[list[int], list[str]] | None = None,
    label: str | None = None,
    age_var: str = "age",
    scale: float = 1.0,
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
    # age_var = age_var or "age"

    if age_bins_and_labels is None:
        bin_edges = list(range(40, 75, 5))  # [40, 45, …, 70]
        bin_labels = [f"{s}_{s + 4}" for s in bin_edges[:-1]]
    else:
        bin_edges, bin_labels = age_bins_and_labels

    # Work on a copy limited to the covered age range
    df = df[df[age_var].between(bin_edges[0], bin_edges[-1] - 1)].copy()

    df["age_bin"] = pd.cut(
        df[age_var],
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
        moments[f"share{label}_age_bin_{age_bin}"] = share_by_bin.loc[age_bin] * scale

    return moments


def compute_shares_by_age_bin_pandas(
    df: pd.DataFrame,
    moments: dict,
    *,
    variable: str,
    age_bins: tuple[list[int], list[str]] | None = None,
    label: str | None = None,
    age_var: str = "age",
):
    """
    Compute shares and sample variances by age-bin for an indicator variable.

    This function mirrors the empirical moment creation function
    `compute_shares_by_age_bin` from task_create_soep_moments.py.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns with the age variable and the indicator column given by
        *variable*. Indicator should be boolean or 0/1.
    moments : dict
        Dictionary updated **in-place** with the new statistics.
    variable : str
        Name of the indicator column in `df` (e.g., "nursing_home").
    age_bins : tuple[list[int], list[str]] | None
        Optional ``(bin_edges, bin_labels)``. If *None*, defaults to:
        edges ``[40, 45, 50, 55, 60, 65, 70]`` and labels ``["40_44", …, "65_69"]``.
        *bin_edges* must include the left edge of the first bin and the right edge
        of the last bin, exactly as required by ``pd.cut``.
    label : str | None
        Optional extra label inserted in every key (prefixed with "_" if given).
    age_var : str
        Name of the age variable column (default: "age").

    Returns
    -------
    dict
        The same *moments* dict (for convenience).
    """
    # 1. Prepare labels and default bin specification
    label = f"_{label}" if label else ""

    if age_bins is None:
        bin_edges = list(range(40, 75, 5))
        bin_labels = [f"{s}_{s+4}" for s in bin_edges[:-1]]
    else:
        bin_edges, bin_labels = age_bins

    # 2. Keep only ages we care about and create an "age_bin" column
    df = df[df[age_var].between(bin_edges[0], bin_edges[-1] - 1)].copy()
    df["age_bin"] = pd.cut(
        df[age_var],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # left-closed / right-open ⇒ 40-44, 45-49, …
    )

    # 3. Group by bins and compute shares (means) of the indicator
    grouped = df.groupby("age_bin", observed=False)[variable]
    shares = grouped.mean().reindex(bin_labels, fill_value=np.nan)

    # 4. Store results with keys mirroring empirical moment style
    #    Keys: share<label>_age_bin_<binlabel>
    #    Note: label already includes the variable name (e.g., "parents_nursing_home")
    for bin_label in bin_labels:
        moments[f"share{label}_age_bin_{bin_label}"] = shares.loc[bin_label]

    return moments


# =====================================================================================
# Wealth
# =====================================================================================


def create_mean_by_age(
    df: pd.DataFrame,
    moments: dict,
    *,
    variable: str,
    age_range: list[int] | np.ndarray,
    label: str | None = None,
    age_var: str = "age",
):
    """
    Compute means by single-year age for a numeric variable and
    store them in `moments` with keys:
        mean_<variable>_<label>_age_<age>

    Parameters
    ----------
    df : DataFrame
        Must contain columns `age_var` and `variable`.
    moments : dict
        Updated in place.
    variable : str
        Column to average (e.g., "assets_begin_of_period").
    age_range : sequence of int
        Ages to include (e.g., range(40, 71)).
    label : str | None
        Optional suffix inserted in the key (prefixed with "_").
    age_var : str, default "age"
        Name of the age column.
    """
    # 1) Label prefix
    label = f"_{label}" if label else ""
    ages = pd.Index(list(age_range), name=age_var)

    # # # 2) Restrict to requested ages (copy to avoid SettingWithCopy warnings)
    # a_min, a_max = int(np.min(age_range)), int(np.max(age_range))
    # df = df[df[age_var].between(a_min, a_max)].copy()
    df_sub = df[df[age_var].isin(ages)]

    # 3) Group by age and compute means
    # mean_by_age = (
    #     df.groupby(age_var, observed=False)[variable]
    #     .mean()
    #     .reindex(age_index, fill_value=np.nan)  # keep all ages even if empty
    # )
    mean_by_age = (
        df_sub.groupby(df_sub[age_var], observed=False)[variable]
        .mean()
        .reindex(ages, fill_value=np.nan)
    )

    # # 3) Warn if any requested ages have no data
    # if mean_by_age.isna().any():
    #     missing = mean_by_age[mean_by_age.isna()].index.tolist()
    #     warnings.warn(
    #         f"Missing mean values for ages (no data): {missing}",
    #         category=UserWarning,
    #         stacklevel=2,
    #     )

    # 4) Write to moments
    for age in ages:
        moments[f"mean_{variable}{label}_age_{age}"] = (
            mean_by_age.loc[age] * WEALTH_MOMENTS_SCALE
        )

    return moments


def create_mean_by_age_bin(
    df: pd.DataFrame,
    moments: dict,
    *,
    variable: str,
    age_bins_and_labels: tuple[list[int], list[str]] | None = None,
    label: str | None = None,
    age_var: str = "age",
):
    """
    Compute means by age-bin for a numeric variable.

    """
    # 1) Label prefix
    label = f"_{label}" if label else ""

    # 2) Default 5-year bins (40–44, ..., 70–74)
    if age_bins_and_labels is None:
        bin_edges = list(range(40, 75, 5))  # [40,45,...,70]
        bin_labels = [f"{s}_{s+4}" for s in bin_edges[:-1]]
    else:
        bin_edges, bin_labels = age_bins_and_labels

    # Work on a copy limited to the covered age range
    df = df[df[age_var].between(bin_edges[0], bin_edges[-1] - 1)].copy()

    df["age_bin"] = pd.cut(
        df[age_var],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # [40,45) ⇒ 40–44, etc.
    )

    age_groups = df.groupby("age_bin", observed=False)

    mean_by_bin = (
        age_groups[variable]
        .mean()
        .reindex(bin_labels, fill_value=np.nan)  # keep bins even if empty
    )

    for age_bin in bin_labels:
        moments[f"mean_{variable}{label}_age_bin_{age_bin}"] = mean_by_bin.loc[age_bin]

    return moments


# =====================================================================================
# Transition moments
# =====================================================================================


def compute_transition_moments_pandas_for_age_bins(
    df,
    moments,
    age_range,
    states,
    label=None,
    bin_width: int = 5,
    choice="choice",
    lagged_choice="lagged_choice",
):
    """
    Compute same-state transition probabilities aggregated into age bins.

    Matches the logic of compute_transition_moments_and_variances_for_age_bins
    in task_create_soep_moments.py, but only returns moments (no variances).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for age, lagged_choice, and choice.
    moments : dict
        Pre-existing moments dict to append to.
    age_range : iterable of int
        Sequence of ages to include in bins.
    states : dict
        Mapping of state labels to their codes in the data.
    label : str, optional
        Suffix label for keys (e.g. education level).
    bin_width : int, default 5
        Width of each age bin (in years).
    choice : str, default "choice"
        Column name for current year's labor supply choice.
    lagged_choice : str, default "lagged_choice"
        Column name for previous year's labor supply choice.

    Returns
    -------
    moments : dict
        Updated with keys 'trans_<state>_to_<state>_<label>_age_<start>_<end>'.
    """
    # Prepare label suffix (matches empirical version)
    suffix = f"_{label}" if label else ""

    # Extract min_age and max_age from age_range (matches empirical structure)
    min_age = min(age_range)
    max_age = max(age_range)

    # Define age bins (exactly as in empirical version)
    bins = []
    start = min_age
    while start + bin_width - 1 <= max_age:
        end = start + bin_width - 1
        bins.append((start, end))
        start += bin_width

    # Loop over states and age bins (matches empirical version)
    for state_label, state_val in states.items():
        for start, end in bins:
            # Filter df for this age bin (matches empirical version)
            df_bin = df[(df["age"] >= start) & (df["age"] <= end)]

            # Subset where last year's state matches (matches empirical version)
            subset = df_bin[df_bin[lagged_choice].isin(np.atleast_1d(state_val))]
            if len(subset) > 0:
                is_same = subset[choice].isin(np.atleast_1d(state_val))
                prob = is_same.mean()
            else:
                prob = np.nan

            # Construct keys including bin range and optional suffix
            # (matches empirical version)
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
    model_specs,
) -> pd.DataFrame:
    """Simulate the model for given parametrization and model solution."""

    # Mirror pandas version: drop periods where the agent is dead.
    df = df.loc[df["health"] != DEAD].copy()

    start_age = model_specs["start_age"]
    end_age = model_specs["end_age_msm"]

    # df["age"] = df.index.get_level_values("period") + model_specs["start_age"]

    return create_moments_jax(df, start_age, end_age, model_specs=model_specs)


def create_moments_jax(sim_df, min_age, max_age, model_specs):  # noqa: PLR0915

    column_indices = {col: idx for idx, col in enumerate(sim_df.columns)}
    idx = column_indices.copy()
    arr_all = jnp.asarray(sim_df)

    # Use model parameters to locate the caregiving start age in the JAX grid
    min_age_caregivers = min_age + model_specs["start_period_caregiving"]
    end_age_wealth = model_specs["end_age_wealth"]

    # df_low_educ = sim_df.loc[sim_df["education"] == 0]
    # df_high_educ = sim_df.loc[sim_df["education"] == 1]

    _no_care_mask = jnp.isin(arr_all[:, idx["choice"]], NO_INFORMAL_CARE)
    arr = arr_all[_no_care_mask]
    # arr = arr_all
    arr_low_educ = arr[arr[:, idx["education"]] == 0]
    arr_high_educ = arr[arr[:, idx["education"]] == 1]

    arr_all_low_educ = arr_all[arr_all[:, idx["education"]] == 0]
    arr_all_high_educ = arr_all[arr_all[:, idx["education"]] == 1]

    _care_mask = jnp.isin(arr_all[:, idx["choice"]], INFORMAL_CARE)
    _light_care_mask = jnp.isin(arr_all[:, idx["choice"]], LIGHT_INFORMAL_CARE)
    _intensive_care_mask = jnp.isin(arr_all[:, idx["choice"]], INTENSIVE_INFORMAL_CARE)
    arr_caregivers = arr_all[_care_mask]
    arr_light_caregivers = arr_all[_light_care_mask]
    arr_intensive_caregivers = arr_all[_intensive_care_mask]

    arr_caregivers_low_educ = arr_caregivers[
        arr_caregivers[:, idx["education"]] == 0
    ].copy()
    arr_caregivers_high_educ = arr_caregivers[
        arr_caregivers[:, idx["education"]] == 1
    ].copy()

    arr_light_caregivers_low_educ = arr_light_caregivers[
        arr_light_caregivers[:, idx["education"]] == 0
    ]
    arr_light_caregivers_high_educ = arr_light_caregivers[
        arr_light_caregivers[:, idx["education"]] == 1
    ]

    arr_intensive_caregivers_low_educ = arr_intensive_caregivers[
        arr_intensive_caregivers[:, idx["education"]] == 0
    ]
    arr_intensive_caregivers_high_educ = arr_intensive_caregivers[
        arr_intensive_caregivers[:, idx["education"]] == 1
    ]

    # Age bins for informal care shares (5-year bins)
    age_bins = [
        (40, 45),
        (45, 50),
        (50, 55),
        (55, 60),
        (60, 65),
        (65, 70),
    ]
    age_bins_75 = [
        (40, 45),
        (45, 50),
        (50, 55),
        (55, 60),
        (60, 65),
        (65, 70),
        (70, 75),
    ]

    # Age bins for caregiver labor shares (3-year bins)
    age_bins_caregivers_3year_jax = [
        (40, 43),
        (43, 46),
        (46, 49),
        (49, 52),
        (52, 55),
        (55, 58),
        (58, 61),
        (61, 64),
        (64, 67),
        (67, 70),
    ]

    # Mean wealth by education and age bin
    mean_wealth_by_age_low_educ = get_mean_by_age(
        arr_all_low_educ,
        ind=idx,
        variable="assets_begin_of_period",
        min_age=min_age,
        max_age=end_age_wealth,
    )
    mean_wealth_by_age_high_educ = get_mean_by_age(
        arr_all_high_educ,
        ind=idx,
        variable="assets_begin_of_period",
        min_age=min_age,
        max_age=end_age_wealth,
    )

    # Labor shares by education and age
    share_retired_by_age = get_share_by_age(
        arr, ind=idx, choice=RETIREMENT, min_age=min_age, max_age=max_age
    )
    share_unemployed_by_age = get_share_by_age(
        arr, ind=idx, choice=UNEMPLOYED, min_age=min_age, max_age=max_age
    )
    share_working_part_time_by_age = get_share_by_age(
        arr, ind=idx, choice=PART_TIME, min_age=min_age, max_age=max_age
    )
    share_working_full_time_by_age = get_share_by_age(
        arr, ind=idx, choice=FULL_TIME, min_age=min_age, max_age=max_age
    )

    share_retired_by_age_low_educ = get_share_by_age(
        arr_low_educ, ind=idx, choice=RETIREMENT, min_age=min_age, max_age=max_age
    )
    share_unemployed_by_age_low_educ = get_share_by_age(
        arr_low_educ, ind=idx, choice=UNEMPLOYED, min_age=min_age, max_age=max_age
    )
    share_working_part_time_by_age_low_educ = get_share_by_age(
        arr_low_educ, ind=idx, choice=PART_TIME, min_age=min_age, max_age=max_age
    )
    share_working_full_time_by_age_low_educ = get_share_by_age(
        arr_low_educ,
        ind=idx,
        choice=FULL_TIME,
        min_age=min_age,
        max_age=max_age,
    )

    share_retired_by_age_high_educ = get_share_by_age(
        arr_high_educ, ind=idx, choice=RETIREMENT, min_age=min_age, max_age=max_age
    )
    share_unemployed_by_age_high_educ = get_share_by_age(
        arr_high_educ, ind=idx, choice=UNEMPLOYED, min_age=min_age, max_age=max_age
    )
    share_working_part_time_by_age_high_educ = get_share_by_age(
        arr_high_educ, ind=idx, choice=PART_TIME, min_age=min_age, max_age=max_age
    )
    share_working_full_time_by_age_high_educ = get_share_by_age(
        arr_high_educ,
        ind=idx,
        choice=FULL_TIME,
        min_age=min_age,
        max_age=max_age,
    )

    # Caregivers labor shares by education and age bin
    share_caregivers_by_age_bin = get_share_by_age_bin(
        arr_all,
        ind=idx,
        choice=INFORMAL_CARE,
        bins=age_bins,
        scale=SCALE_CAREGIVER_SHARE,
    )
    # share_caregivers_by_age_bin = get_share_by_age_bin(
    #     arr, ind=idx, choice=LIGHT_INFORMAL_CARE, bins=age_bins_75
    # )
    # share_caregivers_by_age_bin = get_share_by_age_bin(
    #     arr, ind=idx, choice=INFORMAL_CARE, bins=age_bins_75
    # )

    # ================================================================================
    education_mask = arr_all[:, idx["education"]] == 1
    care_type_mask = jnp.isin(arr_all[:, idx["choice"]], INFORMAL_CARE)
    share_caregivers_high_educ = jnp.sum(education_mask & care_type_mask) / jnp.sum(
        care_type_mask
    )
    # ================================================================================

    # All informal caregivers - using 3-year age bins for labor shares
    share_retired_by_age_bin_caregivers = get_share_by_age_bin(
        arr_caregivers, ind=idx, choice=RETIREMENT, bins=age_bins_caregivers_3year_jax
    )
    share_unemployed_by_age_bin_caregivers = get_share_by_age_bin(
        arr_caregivers, ind=idx, choice=UNEMPLOYED, bins=age_bins_caregivers_3year_jax
    )
    share_working_part_time_by_age_bin_caregivers = get_share_by_age_bin(
        arr_caregivers, ind=idx, choice=PART_TIME, bins=age_bins_caregivers_3year_jax
    )
    share_working_full_time_by_age_bin_caregivers = get_share_by_age_bin(
        arr_caregivers, ind=idx, choice=FULL_TIME, bins=age_bins_caregivers_3year_jax
    )

    # Low education caregivers - using 3-year age bins for labor shares
    share_retired_by_age_bin_caregivers_low_educ = get_share_by_age_bin(
        arr_caregivers_low_educ,
        ind=idx,
        choice=RETIREMENT,
        bins=age_bins_caregivers_3year_jax,
    )
    share_unemployed_by_age_bin_caregivers_low_educ = get_share_by_age_bin(
        arr_caregivers_low_educ,
        ind=idx,
        choice=UNEMPLOYED,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_part_time_by_age_bin_caregivers_low_educ = get_share_by_age_bin(
        arr_caregivers_low_educ,
        ind=idx,
        choice=PART_TIME,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_full_time_by_age_bin_caregivers_low_educ = get_share_by_age_bin(
        arr_caregivers_low_educ,
        ind=idx,
        choice=FULL_TIME,
        bins=age_bins_caregivers_3year_jax,
    )

    # High education caregivers - using 3-year age bins for labor shares
    share_retired_by_age_bin_caregivers_high_educ = get_share_by_age_bin(
        arr_caregivers_high_educ,
        ind=idx,
        choice=RETIREMENT,
        bins=age_bins_caregivers_3year_jax,
    )
    share_unemployed_by_age_bin_caregivers_high_educ = get_share_by_age_bin(
        arr_caregivers_high_educ,
        ind=idx,
        choice=UNEMPLOYED,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_part_time_by_age_bin_caregivers_high_educ = get_share_by_age_bin(
        arr_caregivers_high_educ,
        ind=idx,
        choice=PART_TIME,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_full_time_by_age_bin_caregivers_high_educ = get_share_by_age_bin(
        arr_caregivers_high_educ,
        ind=idx,
        choice=FULL_TIME,
        bins=age_bins_caregivers_3year_jax,
    )

    # Light caregivers (labor shares in 3-year bins, to mirror pandas version)
    share_retired_by_age_bin_light_caregivers = get_share_by_age_bin(
        arr_light_caregivers,
        ind=idx,
        choice=RETIREMENT,
        bins=age_bins_caregivers_3year_jax,
    )
    share_unemployed_by_age_bin_light_caregivers = get_share_by_age_bin(
        arr_light_caregivers,
        ind=idx,
        choice=UNEMPLOYED,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_part_time_by_age_bin_light_caregivers = get_share_by_age_bin(
        arr_light_caregivers,
        ind=idx,
        choice=PART_TIME,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_full_time_by_age_bin_light_caregivers = get_share_by_age_bin(
        arr_light_caregivers,
        ind=idx,
        choice=FULL_TIME,
        bins=age_bins_caregivers_3year_jax,
    )

    share_retired_by_age_bin_light_caregivers_low_educ = get_share_by_age_bin(
        arr_light_caregivers_low_educ,
        ind=idx,
        choice=RETIREMENT,
        bins=age_bins_caregivers_3year_jax,
    )
    share_unemployed_by_age_bin_light_caregivers_low_educ = get_share_by_age_bin(
        arr_light_caregivers_low_educ,
        ind=idx,
        choice=UNEMPLOYED,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_part_time_by_age_bin_light_caregivers_low_educ = get_share_by_age_bin(
        arr_light_caregivers_low_educ,
        ind=idx,
        choice=PART_TIME,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_full_time_by_age_bin_light_caregivers_low_educ = get_share_by_age_bin(
        arr_light_caregivers_low_educ,
        ind=idx,
        choice=FULL_TIME,
        bins=age_bins_caregivers_3year_jax,
    )

    share_retired_by_age_bin_light_caregivers_high_educ = get_share_by_age_bin(
        arr_light_caregivers_high_educ,
        ind=idx,
        choice=RETIREMENT,
        bins=age_bins_caregivers_3year_jax,
    )
    share_unemployed_by_age_bin_light_caregivers_high_educ = get_share_by_age_bin(
        arr_light_caregivers_high_educ,
        ind=idx,
        choice=UNEMPLOYED,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_part_time_by_age_bin_light_caregivers_high_educ = (
        get_share_by_age_bin(
            arr_light_caregivers_high_educ,
            ind=idx,
            choice=PART_TIME,
            bins=age_bins_caregivers_3year_jax,
        )
    )
    share_working_full_time_by_age_bin_light_caregivers_high_educ = (
        get_share_by_age_bin(
            arr_light_caregivers_high_educ,
            ind=idx,
            choice=FULL_TIME,
            bins=age_bins_caregivers_3year_jax,
        )
    )

    # Intensive caregivers (labor shares in 3-year bins, to mirror pandas version)
    share_retired_by_age_bin_intensive_caregivers = get_share_by_age_bin(
        arr_intensive_caregivers,
        ind=idx,
        choice=RETIREMENT,
        bins=age_bins_caregivers_3year_jax,
    )
    share_unemployed_by_age_bin_intensive_caregivers = get_share_by_age_bin(
        arr_intensive_caregivers,
        ind=idx,
        choice=UNEMPLOYED,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_part_time_by_age_bin_intensive_caregivers = get_share_by_age_bin(
        arr_intensive_caregivers,
        ind=idx,
        choice=PART_TIME,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_full_time_by_age_bin_intensive_caregivers = get_share_by_age_bin(
        arr_intensive_caregivers,
        ind=idx,
        choice=FULL_TIME,
        bins=age_bins_caregivers_3year_jax,
    )

    share_retired_by_age_bin_intensive_caregivers_low_educ = get_share_by_age_bin(
        arr_intensive_caregivers_low_educ,
        ind=idx,
        choice=RETIREMENT,
        bins=age_bins_caregivers_3year_jax,
    )
    share_unemployed_by_age_bin_intensive_caregivers_low_educ = get_share_by_age_bin(
        arr_intensive_caregivers_low_educ,
        ind=idx,
        choice=UNEMPLOYED,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_part_time_by_age_bin_intensive_caregivers_low_educ = (
        get_share_by_age_bin(
            arr_intensive_caregivers_low_educ,
            ind=idx,
            choice=PART_TIME,
            bins=age_bins_caregivers_3year_jax,
        )
    )
    share_working_full_time_by_age_bin_intensive_caregivers_low_educ = (
        get_share_by_age_bin(
            arr_intensive_caregivers_low_educ,
            ind=idx,
            choice=FULL_TIME,
            bins=age_bins_caregivers_3year_jax,
        )
    )

    share_retired_by_age_bin_intensive_caregivers_high_educ = get_share_by_age_bin(
        arr_intensive_caregivers_high_educ,
        ind=idx,
        choice=RETIREMENT,
        bins=age_bins_caregivers_3year_jax,
    )
    share_unemployed_by_age_bin_intensive_caregivers_high_educ = get_share_by_age_bin(
        arr_intensive_caregivers_high_educ,
        ind=idx,
        choice=UNEMPLOYED,
        bins=age_bins_caregivers_3year_jax,
    )
    share_working_part_time_by_age_bin_intensive_caregivers_high_educ = (
        get_share_by_age_bin(
            arr_intensive_caregivers_high_educ,
            ind=idx,
            choice=PART_TIME,
            bins=age_bins_caregivers_3year_jax,
        )
    )
    share_working_full_time_by_age_bin_intensive_caregivers_high_educ = (
        get_share_by_age_bin(
            arr_intensive_caregivers_high_educ,
            ind=idx,
            choice=FULL_TIME,
            bins=age_bins_caregivers_3year_jax,
        )
    )

    # Work transitions
    no_work_to_no_work_low_educ_by_age_bin = get_transition_for_age_bins(
        arr_low_educ,
        ind=idx,
        lagged_choice=NOT_WORKING,
        current_choice=NOT_WORKING,
        min_age=min_age,
        max_age=max_age,
    )
    work_to_work_low_educ_by_age_bin = get_transition_for_age_bins(
        arr_low_educ,
        ind=idx,
        lagged_choice=WORK,
        current_choice=WORK,
        min_age=min_age,
        max_age=max_age,
    )

    no_work_to_no_work_high_educ_by_age_bin = get_transition_for_age_bins(
        arr_high_educ,
        ind=idx,
        lagged_choice=NOT_WORKING,
        current_choice=NOT_WORKING,
        min_age=min_age,
        max_age=max_age,
    )
    work_to_work_high_educ_by_age_bin = get_transition_for_age_bins(
        arr_high_educ,
        ind=idx,
        lagged_choice=WORK,
        current_choice=WORK,
        min_age=min_age,
        max_age=max_age,
    )

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

    # Caregiving transitions by age bin
    # Pool across education levels (use arr_all) and start at min_age_caregivers
    # Note: Only compute "caregiving to caregiving" transitions to match empirical
    # (which uses states_caregiving = {"caregiving": 1})
    informal_to_informal_all_educ_by_age_bin = get_transition_for_age_bins(
        arr_all,  # Pooled across all education levels
        ind=idx,
        lagged_choice=INFORMAL_CARE,
        current_choice=INFORMAL_CARE,
        min_age=min_age_caregivers,
        max_age=max_age,
    )

    # # ==============================================================================  # noqa: E501
    # # Care mix
    # age_bins_parents = [(a, a + 5) for a in range(65, 90, 5)]
    # age_bins_parents.append((90, np.inf))
    # # Condition on parent being alive and in medium or bad health
    # alive_mask = arr_all[:, idx["mother_health"]] != PARENT_DEAD
    # medium_or_bad_mask = jnp.isin(
    #     arr_all[:, idx["mother_health"]],
    #     jnp.array([PARENT_BAD_HEALTH, PARENT_MEDIUM_HEALTH]),
    # )
    # arr_parent_bad_health = arr_all[alive_mask & medium_or_bad_mask]

    # # Create nursing_home indicator: formal care occurs when care_demand exists
    # # and no one else provides care (CARE_DEMAND_AND_NO_OTHER_SUPPLY) AND
    # # the agent chooses NO_CARE (meaning formal care is organized)
    # no_care_mask = jnp.isin(arr_parent_bad_health[:, idx["choice"]], NO_CARE)
    # care_demand_mask = (
    #     arr_parent_bad_health[:, idx["care_demand"]]
    #     == CARE_DEMAND_AND_NO_OTHER_SUPPLY
    # )
    # nursing_home_mask = no_care_mask & care_demand_mask

    # # Compute shares by age bin for nursing home (matching empirical moment creation)
    # share_nursing_home_by_parent_age_bin = get_share_by_age_bin_with_extra_mask(
    #     arr_parent_bad_health,
    #     ind=idx,
    #     bins=age_bins_parents,
    #     extra_mask=nursing_home_mask,
    #     age_var="mother_age",
    # )
    # # ==============================================================================  # noqa: E501

    # # share_pure_informal_care_by_parent_age_bin = get_share_by_age_bin(
    # #     arr_domestic_care,
    # #     ind=idx,
    # #     choice=INFORMAL_CARE_OR_OTHER_CARE,
    # #     bins=AGE_BINS_PARENTS,
    # #     age_var="mother_age",
    # # )
    # # share_combination_care_by_parent_age_bin = get_share_by_age_bin(
    # #     arr_domestic_care,
    # #     ind=idx,
    # #     choice=LIGHT_INFORMAL_CARE,
    # #     bins=AGE_BINS_PARENTS,
    # #     age_var="mother_age",
    # # )
    # # share_pure_formal_care_by_parent_age_bin = get_share_by_age_bin(
    # #     arr_domestic_care,
    # #     ind=idx,
    # #     choice=FORMAL_HOME_CARE,
    # #     bins=AGE_BINS_PARENTS,
    # #     age_var="mother_age",
    # # )

    # Subset for domestic care
    # _mask_no_nursing = jnp.isin(arr[:, idx["choice"]], NO_NURSING_HOME_CARE)
    # _mask_demand = arr[:, idx["care_demand"]] >= 1
    # arr_domestic_care = arr[_mask_no_nursing & _mask_demand]

    # # 1) Nursing home
    # share_nursing_home_by_parent_age_bin = get_share_by_age_bin(
    #     arr_parent_bad_health,
    #     ind=idx,
    #     choice=NURSING_HOME_CARE,
    #     bins=age_bins_parents,
    #     age_var="mother_age",
    # )

    # # Build masks on the domestic subset
    # choice_domestic = arr_domestic_care[:, idx["choice"]]
    # is_intensive_informal_dom = jnp.isin(choice_domestic, INTENSIVE_INFORMAL_CARE)
    # is_no_care_domestic = jnp.isin(choice_domestic, NO_CARE)
    # is_supply_domestic = (
    #     arr_domestic_care[:, idx["care_demand"]] == CARE_DEMAND_AND_OTHER_SUPPLY
    # )

    # mask_intensive_informal_or_other = is_intensive_informal_dom | (
    #     is_no_care_domestic & is_supply_domestic
    # )
    # mask_pure_formal = is_no_care_domestic & ~is_supply_domestic

    # # 2) informal_care_or_other_care  (REPLACED: use extra_mask)
    # share_pure_informal_care_by_parent_age_bin = get_share_by_age_bin_with_extra_mask(
    #     arr_domestic_care,
    #     ind=idx,
    #     bins=age_bins_parents,
    #     extra_mask=mask_intensive_informal_or_other,
    #     age_var="mother_age",
    # )

    # # 3) light_informal_care (UNCHANGED: pure choice set)
    # share_combination_care_by_parent_age_bin = get_share_by_age_bin(
    #     arr_domestic_care,
    #     ind=idx,
    #     choice=LIGHT_INFORMAL_CARE,
    #     bins=age_bins_parents,
    #     age_var="mother_age",
    # )

    # # 4) formal_home_care (REPLACED: use extra_mask for NO_CARE & supply==0)
    # share_pure_formal_care_by_parent_age_bin = get_share_by_age_bin_with_extra_mask(
    #     arr_domestic_care,
    #     ind=idx,
    #     bins=age_bins_parents,
    #     extra_mask=mask_pure_formal,
    #     age_var="mother_age",
    # )
    # =================================================================================

    return jnp.asarray(
        []
        # wealth
        + mean_wealth_by_age_low_educ
        + mean_wealth_by_age_high_educ
        # labor shares all
        + share_retired_by_age
        + share_unemployed_by_age
        + share_working_part_time_by_age
        + share_working_full_time_by_age
        + share_retired_by_age_low_educ
        + share_unemployed_by_age_low_educ
        + share_working_part_time_by_age_low_educ
        + share_working_full_time_by_age_low_educ
        + share_retired_by_age_high_educ
        + share_unemployed_by_age_high_educ
        + share_working_part_time_by_age_high_educ
        + share_working_full_time_by_age_high_educ
        # caregivers
        + share_caregivers_by_age_bin
        + [share_caregivers_high_educ]
        + share_retired_by_age_bin_caregivers
        + share_unemployed_by_age_bin_caregivers
        + share_working_part_time_by_age_bin_caregivers
        + share_working_full_time_by_age_bin_caregivers
        + share_retired_by_age_bin_caregivers_low_educ
        + share_unemployed_by_age_bin_caregivers_low_educ
        + share_working_part_time_by_age_bin_caregivers_low_educ
        + share_working_full_time_by_age_bin_caregivers_low_educ
        + share_retired_by_age_bin_caregivers_high_educ
        + share_unemployed_by_age_bin_caregivers_high_educ
        + share_working_part_time_by_age_bin_caregivers_high_educ
        + share_working_full_time_by_age_bin_caregivers_high_educ
        # light caregivers
        + share_retired_by_age_bin_light_caregivers
        + share_unemployed_by_age_bin_light_caregivers
        + share_working_part_time_by_age_bin_light_caregivers
        + share_working_full_time_by_age_bin_light_caregivers
        + share_retired_by_age_bin_light_caregivers_low_educ
        + share_unemployed_by_age_bin_light_caregivers_low_educ
        + share_working_part_time_by_age_bin_light_caregivers_low_educ
        + share_working_full_time_by_age_bin_light_caregivers_low_educ
        + share_retired_by_age_bin_light_caregivers_high_educ
        + share_unemployed_by_age_bin_light_caregivers_high_educ
        + share_working_part_time_by_age_bin_light_caregivers_high_educ
        + share_working_full_time_by_age_bin_light_caregivers_high_educ
        #
        # intensive caregivers
        + share_retired_by_age_bin_intensive_caregivers
        + share_unemployed_by_age_bin_intensive_caregivers
        + share_working_part_time_by_age_bin_intensive_caregivers
        + share_working_full_time_by_age_bin_intensive_caregivers
        + share_retired_by_age_bin_intensive_caregivers_low_educ
        + share_unemployed_by_age_bin_intensive_caregivers_low_educ
        + share_working_part_time_by_age_bin_intensive_caregivers_low_educ
        + share_working_full_time_by_age_bin_intensive_caregivers_low_educ
        + share_retired_by_age_bin_intensive_caregivers_high_educ
        + share_unemployed_by_age_bin_intensive_caregivers_high_educ
        + share_working_part_time_by_age_bin_intensive_caregivers_high_educ
        + share_working_full_time_by_age_bin_intensive_caregivers_high_educ
        #
        # #
        # # transitions
        # Work transitions (by education)
        + no_work_to_no_work_low_educ_by_age_bin
        + work_to_work_low_educ_by_age_bin
        + no_work_to_no_work_high_educ_by_age_bin
        + work_to_work_high_educ_by_age_bin
        # Caregiving transitions (pooled across education,
        # starting at min_age_caregivers)
        # Note: Only "caregiving to caregiving" to match empirical
        + informal_to_informal_all_educ_by_age_bin
        # #
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
        # # care mix by parent age
        # + share_nursing_home_by_parent_age_bin
        # + share_pure_informal_care_by_parent_age_bin
        # + share_combination_care_by_parent_age_bin
        # + share_pure_formal_care_by_parent_age_bin
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


def get_share_by_age_bin(df_arr, ind, choice, bins, age_var=None, scale=1.0):
    """Get share of agents choosing choice by age bin."""
    age_var = age_var or "age"
    age_col = df_arr[:, ind[age_var]]

    choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)

    shares: list[jnp.ndarray] = []
    for bin_start, bin_end in bins:
        age_mask = (age_col >= bin_start) & (age_col < bin_end)
        share = jnp.sum(age_mask & choice_mask) / jnp.sum(age_mask)
        shares.append(share * scale)

    return shares


def get_share_by_age_bin_with_extra_mask(df_arr, ind, bins, extra_mask, age_var=None):
    """Get share of agents choosing choice by age bin."""
    age_var = age_var or "age"
    age_col = df_arr[:, ind[age_var]]

    shares: list[jnp.ndarray] = []
    for bin_start, bin_end in bins:
        age_mask = (age_col >= bin_start) & (age_col < bin_end)
        # denom = jnp.sum(age_mask)
        # num   = jnp.sum(age_mask & base_mask)
        # share = jnp.where(denom > 0, num / denom, jnp.nan)
        share = jnp.sum(age_mask & extra_mask) / jnp.sum(age_mask)
        shares.append(share)

    return shares


def get_mean_by_age(df_arr, ind, variable, min_age, max_age, age_var=None):
    """Get mean of variable by age bin."""
    age_var = age_var or "age"
    age_col = df_arr[:, ind[age_var]]

    values = df_arr[:, ind[variable]]

    means: list[jnp.ndarray] = []
    for age in range(min_age, max_age + 1):
        age_mask = age_col == age
        mean = jnp.nanmean(jnp.where(age_mask, values, jnp.nan)) * WEALTH_MOMENTS_SCALE
        means.append(mean)

    return means


def get_mean_by_age_bin(df_arr, ind, variable, bins, age_var=None):
    """Get mean of variable by age bin."""
    age_var = age_var or "age"
    age_col = df_arr[:, ind[age_var]]

    values = df_arr[:, ind[variable]]

    means: list[jnp.ndarray] = []
    for bin_start, bin_end in bins:
        age_mask = (age_col >= bin_start) & (age_col < bin_end)
        mean = jnp.nanmean(jnp.where(age_mask, values, jnp.nan))
        means.append(mean)

    return means


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


def plot_model_fit_labor_moments_pandas_by_education(  # noqa: PLR0915
    moms_emp: pd.Series,
    moms_sim: pd.Series,
    specs: dict,
    path_to_save_plot: Optional[str] = None,
    include_caregivers: bool = False,
) -> None:
    """
    Plots the age specific labor supply shares (choice shares) for four states:
    retired, unemployed, part-time, and full-time based on the empirical
    and simulated moments.

    Both data_emp and data_sim are pandas Series indexed by moment names in the format:
      "share_{state}_age_{age}" or "share_{state}_caregivers_age_{age}"
    e.g., "share_retired_age_30", "share_unemployed_caregivers_age_40", etc.

    Parameters
    ----------
    moms_emp : pd.Series
        Empirical moments with keys like "share_retired_age_30", etc.
    moms_sim : pd.Series
        Simulated moments with the same key naming convention.
    specs : dict
        Model specifications containing education_labels and choice_labels.
    path_to_save_plot : str, optional
        File path to save the generated plot.
    include_caregivers : bool, default False
        If True, includes caregiver-specific moments in addition to general moments.
    """

    choices = ["retired", "unemployed", "part_time", "full_time"]

    # Determine number of rows based on whether we include caregivers
    n_rows = (
        2 if not include_caregivers else 4
    )  # general + caregivers for each education level
    fig, axs = plt.subplots(
        n_rows, 4, figsize=(16, 6 * n_rows // 2), sharex=True, sharey=True
    )

    # Ensure axs is always 2D
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    row_idx = 0

    for _edu_var, edu_label in enumerate(specs["education_labels"]):
        # Plot general moments (non-caregivers)
        for choice_var, choice_label in enumerate(specs["choice_labels"]):
            ax = axs[row_idx, choice_var]

            # Filter for general moments (no "caregivers" in the key)
            emp_keys = [
                k
                for k in moms_emp.index
                if k.startswith(f"share_{choices[choice_var]}_")
                and str(edu_label.lower().replace(" ", "_")) in k
                and "caregivers" not in k
            ]
            sim_keys = [
                k
                for k in moms_sim.index
                if k.startswith(f"share_{choices[choice_var]}_")
                and str(edu_label.lower().replace(" ", "_")) in k
                and "caregivers" not in k
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
            ax.plot(sim_ages, sim_values, label="Simulated", color="blue")
            ax.plot(emp_ages, emp_values, label="Observed", ls="--", color="red")

            ax.set_xlabel("Age")
            ax.set_ylim([0, 1])

            ax.set_title(f"{choice_label} - {edu_label}")
            ax.tick_params(labelbottom=True)

            if choice_var == 0:
                ax.set_ylabel("Share")
                ax.legend()
            else:
                ax.set_ylabel("")

        row_idx += 1

        # Plot caregiver moments if requested
        if include_caregivers:
            for choice_var, choice_label in enumerate(specs["choice_labels"]):
                ax = axs[row_idx, choice_var]

                # Filter for caregiver moments
                emp_keys = [
                    k
                    for k in moms_emp.index
                    if k.startswith(f"share_{choices[choice_var]}_caregivers_")
                    and str(edu_label.lower().replace(" ", "_")) in k
                ]
                sim_keys = [
                    k
                    for k in moms_sim.index
                    if k.startswith(f"share_{choices[choice_var]}_caregivers_")
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
                ax.plot(sim_ages, sim_values, label="Simulated", color="blue")
                ax.plot(emp_ages, emp_values, label="Observed", ls="--", color="red")

                ax.set_xlabel("Age")
                ax.set_ylim([0, 1])

                ax.set_title(f"{choice_label} - {edu_label} (Caregivers)")
                ax.tick_params(labelbottom=True)

                if choice_var == 0:
                    ax.set_ylabel("Share")
                    ax.legend()
                else:
                    ax.set_ylabel("")

            row_idx += 1

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
    include_caregivers: bool = False,
) -> None:
    """
    Plots the age specific labor supply shares (choice shares) for four states:
    retired, unemployed, part-time, and full-time based on the empirical
    and simulated moments.

    Both data_emp and data_sim are pandas Series indexed by moment names in the format:
      "share_{state}_age_{age}" or "share_{state}_caregivers_age_{age}"
    e.g., "share_retired_age_30", "share_unemployed_caregivers_age_40", etc.

    Parameters
    ----------
    moms_emp : pd.Series
        Empirical moments with keys like "share_retired_age_30", etc.
    moms_sim : jnp.ndarray
        Simulated moments as JAX array.
    specs : dict
        Model specifications containing education_labels and choice_labels.
    path_to_save_plot : str, optional
        File path to save the generated plot.
    include_caregivers : bool, default False
        If True, includes caregiver-specific moments in addition to general moments.
    """

    choices = ["retired", "unemployed", "part_time", "full_time"]

    sim_array = np.asarray(moms_sim)

    # Determine number of rows based on whether we include caregivers
    n_rows = (
        2 if not include_caregivers else 4
    )  # general + caregivers for each education level
    fig, axs = plt.subplots(
        n_rows, 4, figsize=(16, 6 * n_rows // 2), sharex=True, sharey=True
    )

    # Ensure axs is always 2D
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    row_idx = 0

    for _edu_var, edu_label in enumerate(specs["education_labels"]):
        # Plot general moments (non-caregivers)
        for choice_var, choice_label in enumerate(specs["choice_labels"]):
            ax = axs[row_idx, choice_var]

            # Get positions where keys match the current state. This preserves the order
            # of the empirical Series.
            indices = [
                i
                for i, k in enumerate(moms_emp.index)
                if k.startswith(f"share_{choices[choice_var]}_")
                and str(edu_label.lower().replace(" ", "_")) in k
                and "caregivers" not in k
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
            ax.plot(ages, sim_values, label="Simulated", color="blue")
            ax.plot(ages, emp_values, label="Observed", ls="--", color="red")

            ax.set_xlabel("Age")
            ax.set_ylim([0, 1])

            ax.set_title(f"{choice_label} - {edu_label}")
            ax.tick_params(labelbottom=True)

            if choice_var == 0:
                ax.set_ylabel("Share")
                ax.legend()
            else:
                ax.set_ylabel("")

        row_idx += 1

        # Plot caregiver moments if requested
        if include_caregivers:
            for choice_var, choice_label in enumerate(specs["choice_labels"]):
                ax = axs[row_idx, choice_var]

                # Get positions where keys match the current state for caregivers
                indices = [
                    i
                    for i, k in enumerate(moms_emp.index)
                    if k.startswith(f"share_{choices[choice_var]}_caregivers_")
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
                ax.plot(ages, sim_values, label="Simulated", color="blue")
                ax.plot(ages, emp_values, label="Observed", ls="--", color="red")

                ax.set_xlabel("Age")
                ax.set_ylim([0, 1])

                ax.set_title(f"{choice_label} - {edu_label} (Caregivers)")
                ax.tick_params(labelbottom=True)

                if choice_var == 0:
                    ax.set_ylabel("Share")
                    ax.legend()
                else:
                    ax.set_ylabel("")

            row_idx += 1

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
