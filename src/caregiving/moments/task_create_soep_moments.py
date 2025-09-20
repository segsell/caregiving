"""Create SOEP moments and variances for MSM estimation."""

from itertools import product
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.share.task_create_parent_child_data_set import (
    AGE_BINS_PARENTS,
    AGE_LABELS_PARENTS,
    weighted_shares_and_counts,
)
from caregiving.model.shared import (
    AGE_BINS_WEALTH,
    AGE_LABELS_WEALTH,
    BAD_HEALTH,
    DEAD,
    FULL_TIME_CHOICES,
    GOOD_HEALTH,
    NOT_WORKING,
    PARENT_BAD_HEALTH,
    PARENT_WEIGHTS_SHARE,
    PART_TIME_CHOICES,
    RETIREMENT_CHOICES,
    SCALE_CAREGIVER_SHARE,
    START_PERIOD_CAREGIVING,
    UNEMPLOYED_CHOICES,
    WORK,
    SEX,
)
from caregiving.specs.task_write_specs import read_and_derive_specs
from caregiving.utils import table
from dcegm.wealth_correction import adjust_observed_wealth

DEGREES_OF_FREEDOM = 1


def task_create_soep_moments(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_wealth_sample: Path = BLD / "data" / "soep_wealth_and_personal_data.csv",
    path_to_main_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_caregivers_sample: Path = BLD
    / "data"
    / "soep_structural_caregivers_sample.csv",
    path_to_save_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "soep_moments_new.csv",
    path_to_save_variances: Annotated[Path, Product] = BLD
    / "moments"
    / "soep_variances_new.csv",
) -> None:
    """Create moments for MSM estimation."""

    specs = read_and_derive_specs(path_to_specs)

    start_age = specs["start_age"]
    start_age_caregivers = start_age + START_PERIOD_CAREGIVING
    end_age = specs["end_age_msm"]

    age_range = range(start_age, end_age + 1)
    age_range_caregivers = range(start_age_caregivers, end_age + 1)
    age_range_wealth = range(start_age, specs["end_age"] + 1)

    # =================================================================================
    df_wealth = pd.read_csv(path_to_wealth_sample, index_col=[0])
    df_wealth["adjusted_wealth"] = df_wealth["wealth"] / specs["wealth_unit"]

    # female respondents only
    df_wealth = df_wealth[df_wealth["sex"] == SEX].copy()
    df_wealth_low = df_wealth[df_wealth["education"] == 0]
    df_wealth_high = df_wealth[df_wealth["education"] == 1]

    age_bins_90 = (
        list(range(30, 95, 5)),
        [f"{s}_{s+4}" for s in range(30, 90, 5)],
    )
    # =================================================================================

    _age_bins_75 = (
        list(range(40, 80, 5)),  # [40, 45, … , 70]
        [f"{s}_{s+4}" for s in range(40, 75, 5)],  # "40_44", …a
    )

    df_full = pd.read_csv(path_to_main_sample, index_col=[0])
    df = df_full[
        (df_full["sex"] == SEX)
        & (df_full["age"] <= end_age + 10)
        & (df_full["any_care"] == 0)
    ]  # women only and non-caregivers
    _df_alive = df[df["health"] != DEAD].copy()
    _df_good_health = df[df["health"] == GOOD_HEALTH].copy()
    _df_bad_health = df[df["health"] == BAD_HEALTH].copy()

    df_caregivers_full = pd.read_csv(path_to_caregivers_sample, index_col=[0])
    df_caregivers = df_caregivers_full[
        (df_caregivers_full["sex"] == 1)
        & (df_caregivers_full["age"] <= end_age + 10)
        & (df_caregivers_full["any_care"] == 1)
    ]
    _df_caregivers_alive = df_caregivers[df_caregivers["health"] != DEAD].copy()

    df_light_caregivers = df_caregivers[df_caregivers["light_care"] == 1]
    df_intensive_caregivers = df_caregivers[df_caregivers["intensive_care"] == 1]

    df_year = df[df["syear"] == 2012]  # 2012, 2016 # noqa: PLR2004
    _df_year_caregivers = df_year[
        (df_year["any_care"] == 1) & (df_year["health"] != DEAD)
    ].copy()
    _df_year_alive = df_year[df_year["health"] != DEAD].copy()
    # # df_year = df[df["syear"].between(2012, 2018)]
    _df_year_bad_health = df_year[df_year["health"] == BAD_HEALTH]
    _df_year_good_health = df_year[df_year["health"] == GOOD_HEALTH]

    df["kidage_youngest"] = df["kidage_youngest"] - 1

    df_low = df[df["education"] == 0]
    df_high = df[df["education"] == 1]

    df_caregivers_low = df_caregivers[df_caregivers["education"] == 0]
    df_caregivers_high = df_caregivers[df_caregivers["education"] == 1]
    _df_light_caregivers_low = df_light_caregivers[
        df_light_caregivers["education"] == 0
    ]
    _df_light_caregivers_high = df_light_caregivers[
        df_light_caregivers["education"] == 1
    ]
    _df_intensive_caregivers_low = df_intensive_caregivers[
        df_intensive_caregivers["education"] == 0
    ]
    _df_intensive_caregivers_high = df_intensive_caregivers[
        df_intensive_caregivers["education"] == 1
    ]

    moments = {}
    variances = {}

    # =================================================================================
    # 0) Wealth by education and age bin
    moments, variances = compute_mean_wealth_by_age(
        df_wealth_low,
        moments,
        variances,
        wealth_var="adjusted_wealth",
        age_range=age_range_wealth,
        quantile=0.98,
        label="wealth_low_education",
    )
    moments, variances = compute_mean_wealth_by_age(
        df_wealth_high,
        moments,
        variances,
        wealth_var="adjusted_wealth",
        age_range=age_range_wealth,
        quantile=0.98,
        label="wealth_high_education",
    )

    # A) Moments by age.
    moments, variances = compute_labor_shares_by_age(
        df,
        moments=moments,
        variances=variances,
        age_range=age_range,
    )

    # B1) Moments by age and education.
    moments, variances = compute_labor_shares_by_age(
        df_low,
        moments=moments,
        variances=variances,
        age_range=age_range,
        label="low_education",
    )
    moments, variances = compute_labor_shares_by_age(
        df_high,
        moments=moments,
        variances=variances,
        age_range=age_range,
        label="high_education",
    )
    # =================================================================================

    # # B) Health
    # # moments, variances = compute_shares_by_age_bin(
    # #     df_alive,
    # #     moments,
    # #     variances,
    # #     age_bins=age_bins_75,
    # #     variable="health",
    # #     label="good_health",
    # # )
    # moments, variances = compute_labor_shares_by_age(
    #     df_good_health,
    #     moments=moments,
    #     variances=variances,
    #     age_range=age_range,
    #     label="good_health",
    # )
    # moments, variances = compute_labor_shares_by_age(
    #     df_bad_health,
    #     moments=moments,
    #     variances=variances,
    #     age_range=age_range,
    #     label="bad_health",
    # )

    # B2) Moments by age bin conditional on caregiving
    # moments, variances = compute_share_informal_care_by_age(
    #     df,
    #     moments=moments,
    #     variances=variances,
    #     age_range=age_range,
    # )

    # =================================================================================
    # _, variances = compute_share_informal_care_by_age_bin(
    #     df_year,
    #     moments=moments.copy(),
    #     variances=variances,
    #     weights=PARENT_WEIGHTS_SHARE,
    #     scale=SCALE_CAREGIVER_SHARE,
    # )
    # =================================================================================

    # t, variances = compute_share_informal_care_by_age_bin(
    #     df_year_good_health,
    #     moments=t,
    #     variances=variances,
    #     weights=PARENT_WEIGHTS_SHARE,
    #     scale=SCALE_CAREGIVER_SHARE,
    #     label="good_health",
    # )
    # t, variances = compute_share_informal_care_by_age_bin(
    #     df_year_bad_health,
    #     moments=t,
    #     variances=variances,
    #     weights=PARENT_WEIGHTS_SHARE,
    #     scale=SCALE_CAREGIVER_SHARE,
    #     label="bad_health",
    # )

    # =================================================================================
    # caregiver_shares = {
    #     "share_informal_care_age_bin_40_45": 0.02980982,
    #     "share_informal_care_age_bin_45_50": 0.04036255,
    #     "share_informal_care_age_bin_50_55": 0.05350986,
    #     "share_informal_care_age_bin_55_60": 0.06193384,
    #     "share_informal_care_age_bin_60_65": 0.05304824,
    #     "share_informal_care_age_bin_65_70": 0.03079298,
    #     # "share_informal_care_age_bin_70_75": 0.00155229,
    # }
    # scaled_caregiver_shares = {
    #     k: v * SCALE_CAREGIVER_SHARE for k, v in caregiver_shares.items()
    # }
    # moments.update(scaled_caregiver_shares)
    # =================================================================================

    # moments, variances = compute_shares_by_age_bin(
    #     df_caregivers_alive,
    #     moments,
    #     variances,
    #     variable="health",
    #     label="caregivers_good_health",
    # )

    # share_informal_care_40_45, 0.02980982
    # share_informal_care_45_50, 0.04036255
    # share_informal_care_50_55, 0.05350986
    # share_informal_care_55_60, 0.06193384
    # share_informal_care_60_65, 0.05304824
    # share_informal_care_65_70, 0.03079298
    # share_informal_care_70_75, 0.00155229
    # moments["share_informal_care_high_educ"] = df.loc[
    #     df["any_care"] == 1, "education"
    # ].mean()
    # variances["share_informal_care_high_educ"] = df.loc[
    #     df["any_care"] == 1, "education"
    # ].var(ddof=DEGREES_OF_FREEDOM)

    # Caregiving
    moments, variances = compute_labor_shares_by_age(
        df_caregivers,
        moments=moments,
        variances=variances,
        age_range=age_range_caregivers,
        label="caregivers",
    )
    moments, variances = compute_labor_shares_by_age(
        df_caregivers_low,
        moments=moments,
        variances=variances,
        age_range=age_range_caregivers,
        label="caregivers_low_education",
    )
    moments, variances = compute_labor_shares_by_age(
        df_caregivers_high,
        moments=moments,
        variances=variances,
        age_range=age_range_caregivers,
        label="caregivers_high_education",
    )

    # moments, variances = compute_labor_shares_by_age_bin(
    #     df_caregivers,
    #     moments=moments,
    #     variances=variances,
    #     label="caregivers",
    # )
    # moments, variances = compute_labor_shares_by_age_bin(
    #     df_caregivers_low,
    #     moments=moments,
    #     variances=variances,
    #     label="caregivers_low_education",
    # )
    # moments, variances = compute_labor_shares_by_age_bin(
    #     df_caregivers_high,
    #     moments=moments,
    #     variances=variances,
    #     label="caregivers_high_education",
    # )

    # # B2.2) Light caregiving
    # moments, variances = compute_labor_shares_by_age_bin(
    #     df_light_caregivers,
    #     moments=moments,
    #     variances=variances,
    #     label="light_caregivers",
    # )
    # moments, variances = compute_labor_shares_by_age_bin(
    #     df_light_caregivers_low,
    #     moments=moments,
    #     variances=variances,
    #     label="light_caregivers_low_education",
    # )
    # moments, variances = compute_labor_shares_by_age_bin(
    #     df_light_caregivers_high,
    #     moments=moments,
    #     variances=variances,
    #     label="light_caregivers_high_education",
    # )

    # # B2.3) Intensive caregiving
    # moments, variances = compute_labor_shares_by_age_bin(
    #     df_intensive_caregivers,
    #     moments=moments,
    #     variances=variances,
    #     label="intensive_caregivers",
    # )
    # moments, variances = compute_labor_shares_by_age_bin(
    #     df_intensive_caregivers_low,
    #     moments=moments,
    #     variances=variances,
    #     label="intensive_caregivers_low_education",
    # )
    # moments, variances = compute_labor_shares_by_age_bin(
    #     df_intensive_caregivers_high,
    #     moments=moments,
    #     variances=variances,
    #     label="intensive_caregivers_high_education",
    # )

    # =================================================================================

    # C) Moments by number of children and education.
    # children_groups = {
    #     "0": lambda x: x == 0,
    #     "1": lambda x: x == 1,
    #     "2": lambda x: x == 2,  # noqa: PLR2004
    #     "3_plus": lambda x: x >= 3,  # noqa: PLR2004
    # }
    # moments, variances = (
    #     _get_labor_shares_by_education_level_and_number_of_children_in_hh(
    #         df, moments, variances, children_groups
    #     )
    # )

    # D) Moments by kidage (youngest) and education.
    # kidage_bins = {"0_3": (0, 3), "4_6": (4, 6), "7_9": (7, 9)}
    # moments, variances = _get_labor_shares_by_educ_level_and_age_of_youngest_child(
    #     df, moments, variances, kidage_bins
    # )
    # =================================================================================

    # # E) Year-to-year labor supply transitions
    # states_work_no_work = {
    #     "working": WORK,
    #     "not_working": NOT_WORKING,
    # }
    # transition_moments, transition_variances = (
    #     compute_transition_moments_and_variances_for_age_bins(
    #         df_low,
    #         min_age=start_age,
    #         max_age=end_age,
    #         states=states_work_no_work,
    #         label="low_education",
    #     )
    # )
    # moments.update(transition_moments)
    # variances.update(transition_variances)

    # transition_moments, transition_variances = (
    #     compute_transition_moments_and_variances_for_age_bins(
    #         df_high,
    #         min_age=start_age,
    #         max_age=end_age,
    #         states=states_work_no_work,
    #         label="high_education",
    #     )
    # )
    # moments.update(transition_moments)
    # variances.update(transition_variances)

    # states = {
    #     "not_working": NOT_WORKING,
    #     "part_time": PART_TIME,
    #     "full_time": FULL_TIME_CHOICES,
    # }
    # trans_moments, trans_variances = compute_transition_moments_and_variances(
    #     df,
    #     min_age=start_age + 1,
    #     max_age=end_age + 1,
    #     states=states,
    #     full_time=FULL_TIME_CHOICES,
    #     part_time=PART_TIME,
    #     not_working=NOT_WORKING,
    # )
    # moments.update(trans_moments)
    # variances.update(trans_variances)

    # states_care_to_care = {
    #     "caregiving": 1,
    #     "not_caregiving": 0,
    # }
    # transition_moments, transition_variances = (
    #     compute_transition_moments_and_variances_for_age_bins(
    #         df_low,
    #         min_age=start_age,
    #         max_age=end_age,
    #         states=states_care_to_care,
    #         # label="low_education",
    #         choice="any_care",
    #         lagged_choice="lagged_care",
    #     )
    # )
    # moments.update(transition_moments)
    # variances.update(transition_variances)

    # F) Wealth moments by age and education (NEW)
    # wealth_moments_edu_low, wealth_variances_edu_low = (
    #     compute_wealth_moments_by_five_year_age_bins(
    #         df, edu_level=0, start_age=start_age, end_age=end_age
    #     )
    # )
    # moments.update(wealth_moments_edu_low)
    # variances.update(wealth_variances_edu_low)
    # wealth_moments_edu_high, wealth_variances_edu_high = (
    #     compute_wealth_moments_by_five_year_age_bins(
    #         df, edu_level=1, start_age=start_age, end_age=end_age
    #     )
    # )
    # moments.update(wealth_moments_edu_high)
    # variances.update(wealth_variances_edu_high)
    # plot_wealth_by_age(df, start_age=30, end_age=70, educ_val=1)
    # plot_wealth_by_5yr_bins(df, start_age=30, end_age=70, educ_val=1)

    # =================================================================================
    # df_bad_health = df_caregivers_full.loc[
    #     (df_caregivers_full["health"] == BAD_HEALTH)
    #     & (df_caregivers_full["age"] > AGE_BINS_PARENTS[0])
    # ].copy()

    # moments, variances = compute_shares_by_age_bin(
    #     df_bad_health,
    #     moments,
    #     variances,
    #     variable="nursing_home",
    #     age_bins=(AGE_BINS_PARENTS, AGE_LABELS_PARENTS),
    #     label="parents_nursing_home",
    # )
    # =================================================================================

    moments_df = pd.DataFrame({"value": pd.Series(moments)})
    moments_df.index.name = "moment"

    variances_df = pd.DataFrame({"value": pd.Series(variances)})
    variances_df.index.name = "moment"

    moments_df.to_csv(path_to_save_moments, index=True)
    variances_df.to_csv(path_to_save_variances, index=True)


def compute_labor_shares_by_age(df, moments, variances, age_range, label=None):

    if label is None:
        label = ""
    else:
        label = "_" + label

    age_groups = df.groupby("age", observed=False)

    # Compute the proportion for each status using vectorized operations
    retired_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(RETIREMENT_CHOICES)).mean()
    )
    unemployed_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(UNEMPLOYED_CHOICES)).mean()
    )
    part_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(PART_TIME_CHOICES)).mean()
    )
    full_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(FULL_TIME_CHOICES)).mean()
    )

    retired_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(RETIREMENT_CHOICES)).var(ddof=DEGREES_OF_FREEDOM)
    )
    unemployed_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(UNEMPLOYED_CHOICES)).var(ddof=DEGREES_OF_FREEDOM)
    )
    part_time_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(PART_TIME_CHOICES)).var(ddof=DEGREES_OF_FREEDOM)
    )
    full_time_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(FULL_TIME_CHOICES)).var(ddof=DEGREES_OF_FREEDOM)
    )

    # Reindex to ensure that every age between start_age and end_age is included;
    # missing ages will be filled with NaN
    retired_shares = retired_shares.reindex(age_range, fill_value=np.nan)
    unemployed_shares = unemployed_shares.reindex(age_range, fill_value=np.nan)
    part_time_shares = part_time_shares.reindex(age_range, fill_value=np.nan)
    full_time_shares = full_time_shares.reindex(age_range, fill_value=np.nan)

    retired_vars = retired_vars.reindex(age_range, fill_value=np.nan)
    unemployed_vars = unemployed_vars.reindex(age_range, fill_value=np.nan)
    part_time_vars = part_time_vars.reindex(age_range, fill_value=np.nan)
    full_time_vars = full_time_vars.reindex(age_range, fill_value=np.nan)

    # Populate the moments dictionary for age-specific shares
    for age in age_range:
        moments[f"share_retired{label}_age_{age}"] = retired_shares.loc[age]
        variances[f"share_retired{label}_age_{age}"] = retired_vars.loc[age]

    for age in age_range:
        moments[f"share_unemployed{label}_age_{age}"] = unemployed_shares.loc[age]
        variances[f"share_unemployed{label}_age_{age}"] = unemployed_vars.loc[age]

    for age in age_range:
        moments[f"share_part_time{label}_age_{age}"] = part_time_shares.loc[age]
        variances[f"share_part_time{label}_age_{age}"] = part_time_vars.loc[age]

    for age in age_range:
        moments[f"share_full_time{label}_age_{age}"] = full_time_shares.loc[age]
        variances[f"share_full_time{label}_age_{age}"] = full_time_vars.loc[age]

    return moments, variances


def compute_share_informal_care_by_age(
    df: pd.DataFrame,
    moments: dict,
    variances: dict,
    age_range,
    label: str | None = None,
):
    """
    Update *moments* in=place with the share of agents whose “choice”
    lies in INFORMAL_CARE, computed separately for every age in
    *age_range*.

    """

    label = f"_{label}" if label else ""

    age_groups = df.groupby("age", observed=False)

    share_by_age = age_groups["any_care"].apply(
        lambda s: s.eq(1).sum() / s.notna().sum()
    )
    variance_by_age = age_groups["any_care"].apply(
        lambda s: s.eq(1).astype(int).var(ddof=DEGREES_OF_FREEDOM)
    )

    for age in age_range:
        moments[f"share_informal_care{label}_age_{age}"] = share_by_age.loc[age]
        variances[f"variance_informal_care{label}_age_{age}"] = variance_by_age.loc[age]

    return moments, variances


def compute_share_informal_care_by_age_bin(
    df: pd.DataFrame,
    moments: dict,
    variances: dict,
    age_bins: tuple | None = None,
    weights: dict | None = None,
    label: str | None = None,
    scale: float = 1.0,
):
    """
    Update *moments* in=place with the share of agents whose “choice”
    lies in INFORMAL_CARE, computed separately for every age in
    *age_range*.

    """

    # 1. Prepare labels and default bin specification
    label = f"_{label}" if label else ""

    if age_bins is None:
        # bin edges: 40,45,50,55,60,65,70  (right edge 70 is *exclusive*)
        bin_edges = list(range(40, 75, 5))  # [40,45,50,55,60,65,70]
        bin_labels = [f"{s}_{s+4}" for s in bin_edges[:-1]]
    else:
        bin_edges, bin_labels = age_bins

    # 2. Keep only ages we care about and create an “age_bin” column
    df = df[df["age"].between(bin_edges[0], bin_edges[-1] - 1)].copy()

    df["age_bin"] = pd.cut(
        df["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # left-closed / right-open ⇒ 40-44, 45-49, …
    )
    _counts = (
        df.groupby("age_bin", observed=False)
        .size()
        .reindex(bin_labels, fill_value=np.nan)
    )

    # 3. Group by the new bins and compute shares & variances
    age_groups = df.groupby("age_bin", observed=False)

    share_by_age = age_groups["any_care"].apply(
        lambda s: s.eq(1).sum() / s.notna().sum()
    )
    variance_by_age = age_groups["any_care"].apply(
        lambda s: s.eq(1).astype(int).var(ddof=DEGREES_OF_FREEDOM)
    )
    # variance_by_age = age_groups["any_care"].apply(
    #     lambda x: x.astype(int).var(ddof=DEGREES_OF_FREEDOM)
    # )

    if weights is None:
        adjusted_share_by_age = share_by_age
    else:
        parent_weights = pd.Series(weights, name="parent_weights").reindex(
            bin_labels, fill_value=np.nan
        )
        adjusted_share_by_age = share_by_age * parent_weights

    for age in bin_labels:
        moments[f"share_informal_care{label}_age_bin_{age}"] = (
            adjusted_share_by_age.loc[age] * scale
        )
        variances[f"variance_informal_care{label}_age_bin_{age}"] = variance_by_age.loc[
            age
        ]

    return moments, variances


def compute_labor_shares_by_age_bin(
    df: pd.DataFrame,
    moments: dict,
    variances: dict,
    age_bins: tuple | None = None,
    label: str | None = None,
):
    """
    Compute labour-market status shares and their variances by 5-year age-bin.

    Parameters
    ----------
    df : DataFrame
            Must contain the columns ``age`` (int) and ``choice`` (categorical / int).
    moments, variances : dict
            Dictionaries updated **in-place** with the new statistics.
    age_bins : tuple[list[int], list[str]] | None
            Optional ``(bin_edges, bin_labels)``.
            *bin_edges* must include the left edge of the first bin and the right
            edge of the last bin, exactly as required by ``pd.cut``.
            If *None*, the default edges ``[40, 45, 50, 55, 60, 65, 70]`` and the
            labels ``["40_44", "45_49", … , "65_69"]`` are used.
    label : str | None
            Optional extra label inserted in every key (prefixed with “_” if given).

    Returns
    -------
    moments, variances : dict
            The same objects that were passed in, for convenience.
    """

    # 1. Prepare labels and default bin specification
    label = f"_{label}" if label else ""

    if age_bins is None:
        # bin edges: 40,45,50,55,60,65,70  (right edge 70 is *exclusive*)
        bin_edges = list(range(40, 75, 5))  # [40,45,50,55,60,65,70]
        bin_labels = [f"{s}_{s+4}" for s in bin_edges[:-1]]
    else:
        bin_edges, bin_labels = age_bins

    # 2. Keep only ages we care about and create an “age_bin” column
    df = df[df["age"].between(bin_edges[0], bin_edges[-1] - 1)].copy()

    df["age_bin"] = pd.cut(
        df["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # left-closed / right-open ⇒ 40-44, 45-49, …
    )
    _counts = (
        df.groupby("age_bin", observed=False)
        .size()
        .reindex(bin_labels, fill_value=np.nan)
    )

    # 3. Group by the new bins and compute shares & variances
    age_groups = df.groupby("age_bin")

    retired_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(RETIREMENT_CHOICES)).mean()
    )
    unemployed_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(UNEMPLOYED_CHOICES)).mean()
    )
    part_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(PART_TIME_CHOICES)).mean()
    )
    full_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(FULL_TIME_CHOICES)).mean()
    )

    retired_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(RETIREMENT_CHOICES)).var(ddof=DEGREES_OF_FREEDOM)
    )
    unemployed_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(UNEMPLOYED_CHOICES)).var(ddof=DEGREES_OF_FREEDOM)
    )
    part_time_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(PART_TIME_CHOICES)).var(ddof=DEGREES_OF_FREEDOM)
    )
    full_time_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(FULL_TIME_CHOICES)).var(ddof=DEGREES_OF_FREEDOM)
    )

    # Re-index so every bin is present even if it appears nowhere in the data
    retired_shares = retired_shares.reindex(bin_labels, fill_value=np.nan)
    unemployed_shares = unemployed_shares.reindex(bin_labels, fill_value=np.nan)
    part_time_shares = part_time_shares.reindex(bin_labels, fill_value=np.nan)
    full_time_shares = full_time_shares.reindex(bin_labels, fill_value=np.nan)

    retired_vars = retired_vars.reindex(bin_labels, fill_value=np.nan)
    unemployed_vars = unemployed_vars.reindex(bin_labels, fill_value=np.nan)
    part_time_vars = part_time_vars.reindex(bin_labels, fill_value=np.nan)
    full_time_vars = full_time_vars.reindex(bin_labels, fill_value=np.nan)

    for bin in bin_labels:
        moments[f"share_retired{label}_age_bin_{bin}"] = retired_shares.loc[bin]
        variances[f"share_retired{label}_age_bin_{bin}"] = retired_vars.loc[bin]

    for bin in bin_labels:
        moments[f"share_unemployed{label}_age_bin_{bin}"] = unemployed_shares.loc[bin]
        variances[f"share_unemployed{label}_age_bin_{bin}"] = unemployed_vars.loc[bin]

    for bin in bin_labels:
        moments[f"share_part_time{label}_age_bin_{bin}"] = part_time_shares.loc[bin]
        variances[f"share_part_time{label}_age_bin_{bin}"] = part_time_vars.loc[bin]

    for bin in bin_labels:
        moments[f"share_full_time{label}_age_bin_{bin}"] = full_time_shares.loc[bin]
        variances[f"share_full_time{label}_age_bin_{bin}"] = full_time_vars.loc[bin]

    return moments, variances


def compute_shares_by_age_bin(
    df: pd.DataFrame,
    moments: dict,
    variances: dict,
    variable: str,
    age_bins: tuple | None = None,
    label: str | None = None,
):
    """
    Compute shares and sample variances by age-bin for an indicator variable.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``age`` (int) and the indicator column given by *variable*.
        Indicator should be boolean or 0/1.
    moments, variances : dict
        Dictionaries updated **in-place** with the new statistics.
    variable : str
        Name of the indicator column in `df` (e.g., "nursing_home").
    age_bins : tuple[list[int], list[str]] | None
        Optional ``(bin_edges, bin_labels)``. If *None*, defaults to:
        edges ``[40, 45, 50, 55, 60, 65, 70]`` and labels ``["40_44", …, "65_69"]``.
        *bin_edges* must include the left edge of the first bin and the right edge
        of the last bin, exactly as required by ``pd.cut``.
    label : str | None
        Optional extra label inserted in every key (prefixed with “_” if given).

    Returns
    -------
    moments, variances : dict
            The same objects that were passed in, for convenience.
    """
    # 1. Prepare labels and default bin specification
    label = f"_{label}" if label else ""

    if age_bins is None:
        bin_edges = list(range(40, 75, 5))
        bin_labels = [f"{s}_{s+4}" for s in bin_edges[:-1]]
    else:
        bin_edges, bin_labels = age_bins

    # 2. Keep only ages we care about and create an “age_bin” column
    # df = df[df["age"].between(bin_edges[0], bin_edges[-1] - 1)].copy()
    df["age_bin"] = pd.cut(
        df["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # left-closed / right-open ⇒ 40-44, 45-49, …
    )

    # 3. Group by bins and compute shares (means) & sample variances of the indicator
    # Cast to float to be robust to boolean dtype
    grouped = df.groupby("age_bin", observed=False)[variable]
    shares = grouped.mean().reindex(bin_labels, fill_value=np.nan)
    vars = grouped.var(ddof=DEGREES_OF_FREEDOM).reindex(bin_labels, fill_value=np.nan)

    # 4. Store results with keys mirroring your style
    #    Keys: share_<variable><label>_age_bin_<binlabel>
    for bin in bin_labels:
        moments[f"share{label}_age_bin_{bin}"] = shares.loc[bin]
        variances[f"share{label}_age_bin_{bin}"] = vars.loc[bin]

    return moments, variances


# =====================================================================================
# Wealth
# =====================================================================================


def compute_mean_wealth_by_age(
    df: pd.DataFrame,
    moments: dict,
    variances: dict,
    age_range: list[int] | np.ndarray,
    quantile: float = 0.95,
    *,
    wealth_var: str = "wealth",
    label: str | None = None,
):
    """
    Compute empirical mean + variance of wealth by AGE (not bins) and
    store them into `moments` and `variances` with keys:
        mean_<label>_wealth_age_<age>
        var_<label>_wealth_age_<age>

    Smoothing (empirical-only):
      - Trim top 5% by wealth.
      - Use rolling(3) mean by age.
      - For the first two ages in `age_range`, use the raw mean.
      - For the last 21 ages in `age_range`, use rolling(5).

    Parameters
    ----------
    df : DataFrame
        Must contain columns: 'age' and `wealth_var`.
        Assumes only one sex and one education present.
    moments, variances : dict
        Updated in-place with new entries.
    age_range : sequence of int
        Ages to include (used for reindexing & key creation).
    wealth_var : str, default "wealth"
        Wealth column name.
    quantile : float, default 0.95
        Top quantile to trim (e.g., 0.95 for top 5%
    label : str | None
        Optional suffix in keys (prefixed with '_').

    """
    label = f"_{label}" if label else ""
    age_index = pd.Index(age_range, name="age")

    # 1) Percentile trim
    wealth_mask = df[wealth_var] < df[wealth_var].quantile(quantile)
    trimmed = df.loc[wealth_mask, ["age", wealth_var]].copy()

    # 2) Group by age: raw mean and variance
    base_mean = (
        trimmed.groupby("age", observed=False)[wealth_var]
        .mean()
        .reindex(age_index, fill_value=np.nan)
        .sort_index()
    )
    base_var = (
        trimmed.groupby("age", observed=False)[wealth_var]
        .var(ddof=DEGREES_OF_FREEDOM)
        .reindex(age_index, fill_value=np.nan)
        .sort_index()
    )

    # 3) Rolling smoothing on the mean (index is age, regular spacing assumed)
    roll3 = base_mean.rolling(3, min_periods=1).mean()
    roll5 = base_mean.rolling(5, min_periods=1).mean()

    # First two ages → raw mean
    if len(age_index) >= 1:
        roll3.iloc[0] = base_mean.iloc[0]
    if len(age_index) >= 2:
        roll3.iloc[1] = base_mean.iloc[1]

    # Last 21 ages → rolling(5)
    last_n = min(21, len(age_index))
    if last_n > 0:
        roll3.iloc[-last_n:] = roll5.iloc[-last_n:]

    smoothed_mean = roll3

    # 4) Write out moments/variances with per-age keys
    for age in age_index:
        moments[f"mean{label}_wealth_age_{age}"] = smoothed_mean.loc[age]
        variances[f"var{label}_wealth_age_{age}"] = base_var.loc[age]

    return moments, variances


def compute_mean_by_age_bin(
    df: pd.DataFrame,
    moments: dict,
    variances: dict,
    variable: str,
    age_bins: tuple[list[int], list[str]] | None = None,
    label: str | None = None,
):
    """
    Compute means and sample variances by age-bin for a numeric variable.

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'age' (int) and the numeric column given by *variable*.
    moments, variances : dict
        Dictionaries updated **in-place** with results.
    variable : str
        Name of the numeric column in `df` (e.g., 'income').
    age_bins : tuple[list[int], list[str]] | None
        Optional (bin_edges, bin_labels). If None, defaults to 5-year bins 40–74:
          edges:  [40, 45, 50, 55, 60, 65, 70, 75]
          labels: ['40_44', '45_49', ..., '70_74']
        Note: edges must include both the first left edge and the final right edge.
    label : str | None
        Optional extra label inserted in every key (prefixed with '_' if given).

    Returns
    -------
    moments, variances : dict
        The same objects passed in, updated with new values.
    """
    # 1) Label prefix
    key_label = f"_{label}" if label else ""

    # 2) Default 5-year bins (40–44, ..., 70–74)
    if age_bins is None:
        bin_edges = list(range(40, 75, 5))  # [40,45,...,70]
        bin_labels = [f"{s}_{s+4}" for s in bin_edges[:-1]]
    else:
        bin_edges, bin_labels = age_bins

    # # 3) Ensure numeric dtype (robust to bool/object)
    # x = pd.to_numeric(df[variable], errors="coerce")

    # 4) Assign bins (left-closed, right-open ⇒ 40–44, 45–49, …)
    df["age_bin"] = pd.cut(df["age"], bins=bin_edges, labels=bin_labels, right=False)

    # 5) Group, compute mean & variance
    grouped = df.groupby("age_bin", observed=False)[variable]
    means = grouped.mean().reindex(bin_labels, fill_value=np.nan)
    vars = grouped.var(ddof=DEGREES_OF_FREEDOM).reindex(bin_labels, fill_value=np.nan)

    # 6) Store results with consistent keys
    # Keys: mean_<label>_<var>_age_bin_<bin>, var_<label>_<var>_age_bin_<bin>
    for bin in bin_labels:
        moments[f"mean{key_label}_{variable}_age_bin_{bin}"] = means.loc[bin]
        variances[f"var{key_label}_{variable}_age_bin_{bin}"] = vars.loc[bin]

    return moments, variances


# =====================================================================================
# Transition moments
# =====================================================================================


def compute_transition_moments_and_variances_for_age_bins(
    df,
    states,
    min_age,
    max_age,
    choice="choice",
    lagged_choice="lagged_choice",
    label=None,
    bin_width: int = 5,
):
    """
    Compute year-to-year labor supply transition moments and their variances.

    Aggregated in fixed-width age bins (default 5-year bins).

    Only compute transitions where the initial and final state are the same.

    Parameters
    ----------
    df : pd.DataFrame
            Must contain columns for age, lagged_choice, and choice.
    states : dict
            Mapping of state labels to their codes in the data.
    min_age : int
            Minimum age for binning (inclusive).
    max_age : int
            Maximum age for binning (inclusive).
    choice : str, default "choice"
            Column name for current year's labor supply choice.
    lagged_choice : str, default "lagged_choice"
            Column name for previous year's labor supply choice.
    label : str, optional
            Suffix label for keys (e.g. education level).
    bin_width : int, default 5
            Width of each age bin (in years).

    Returns
    -------
    moments : dict
            Keys like 'trans_working_to_working_low_education_age_30_34',
            values are probabilities.
    variances : dict
            Keys like 'var_trans_working_to_working_low_education_age_30_34',
            values are variances.
    """

    # Prepare label suffix
    suffix = f"_{label}" if label else ""

    moments = {}
    variances = {}

    # Define age bins
    bins = []
    start = min_age
    while start + bin_width - 1 <= max_age:
        end = start + bin_width - 1
        bins.append((start, end))
        start += bin_width

    # Loop over states and age bins
    for state_label, state_val in states.items():
        for start, end in bins:
            # Filter df for this age bin
            df_bin = df[(df["age"] >= start) & (df["age"] <= end)]

            # Subset where last year's state matches
            subset = df_bin[df_bin[lagged_choice].isin(np.atleast_1d(state_val))]
            if len(subset) > 0:
                is_same = subset[choice].isin(np.atleast_1d(state_val))
                prob = is_same.mean()
                var = is_same.astype(float).var(ddof=DEGREES_OF_FREEDOM)
            else:
                prob = np.nan
                var = np.nan

            # Construct keys including bin range and optional suffix
            key = f"trans_{state_label}_to_{state_label}{suffix}_age_{start}_{end}"
            moments[key] = prob
            variances[f"var_{key}"] = var

    return moments, variances


def compute_transition_moments_and_variances(
    df,
    states,
    min_age,
    max_age,
    choice="choice",
    lagged_choice="lagged_choice",
    label=None,
):
    """
    Compute year-to-year labor supply transition moments and their variances.

    Parameters
    ----------
    df : pd.DataFrame
            Must contain the columns specified by lagged_col and current_col.
    FULL_TIME_CHOICES : array-like
            Codes representing full-time in the 'choice' columns.
    PART_TIME_CHOICE : array-like
            Codes representing part-time in the 'choice' columns.
    NOT_WORKING : array-like
            Codes representing not-working states in the 'choice' columns.
    lagged_col : str, default "lagged_choice"
            The column in df representing last year's labor supply choice.
    current_col : str, default "choice"
            The column in df representing current year's labor supply choice.

    Returns
    -------
    moments : dict
            Keys like 'transition_full_time_to_part_time',
            values are transition probabilities.
    variances : dict
            Keys like 'var_transition_full_time_to_part_time',
            values are the corresponding variances of those probabilities.
    """

    if label is None:
        label = ""
    else:
        label = "_" + label

    age_range = range(min_age, max_age + 1)

    moments = {}
    variances = {}

    # # For each "from" state, filter rows where lagged_choice is in that state,
    # # and for each "to" state, compute the probability that 'choice' is in that state.
    # for from_label, from_val in states.items():
    #     # Use isin() to safely compare even if from_val is an array.
    #     subset = df[df["lagged_choice"].isin(np.atleast_1d(from_val))]

    #     for to_label, to_val in states.items():
    #         if len(subset) > 0:
    #             # Create a boolean series for whether 'choice' is in the "to" state.
    #             bool_series = subset["choice"].isin(np.atleast_1d(to_val))
    #             # Probability is the mean of that boolean indicator.
    #             probability = bool_series.mean()
    #             # Compute the sample variance (ddof=1) of the indicator.
    #             variance = bool_series.astype(float).var(ddof=DEGREES_OF_FREEDOM)
    #             # Variance of a Bernoulli variable with sample size N
    #             # variance = probability * (1 - probability) / len(subset)
    #         else:
    #             probability = np.nan
    #             variance = np.nan

    #         moments[f"trans_{from_label}_to_{to_label}"] = probability
    #         variances[f"var_trans_{from_label}_to_{to_label}"] = variance

    # # Loop over each unique age in the DataFrame (in ascending order).
    # for age in age_range:
    #     df_age = df[df["age"] == age]

    #     # For each "from" state, filter rows where lagged_choice is in that state.
    #     for from_label, from_val in states.items():
    #         subset = df_age[df_age[lagged_choice].isin(np.atleast_1d(from_val))]

    #         # For each "to" state, compute the transition probability and its
    #         # variance.
    #         for to_label, to_val in states.items():
    #             if len(subset) > 0:
    #                 # Boolean indicator for whether the current choice is
    #                 # in the "to" state.
    #                 bool_series = subset[choice].isin(np.atleast_1d(to_val))
    #                 probability = bool_series.mean()
    #                 variance = bool_series.astype(float).var(ddof=DEGREES_OF_FREEDOM)
    #             else:
    #                 probability = np.nan
    #                 variance = np.nan

    #             # Save the results with keys that include the age.
    #             key = f"trans_{from_label}_to_{to_label}_age_{age}"
    #             moments[key] = probability
    #             variances[f"var_{key}"] = variance

    # Loop over each transition type.
    # for from_label, from_val in states.items():
    #     for to_label, to_val in states.items():
    #         probs_by_age = []
    #         vars_by_age = []
    #         # Loop over the age range.
    #         for age in age_range:
    #             df_age = df[df["age"] == age]
    #             # Filter observations based on the lagged state.
    #             subset = df_age[df_age[lagged_choice].isin(np.atleast_1d(from_val))]
    #             if len(subset) > 0:
    #                 # Create boolean indicator for whether the current state is in the
    #                 # target "to" state.
    #                 bool_series = subset[choice].isin(np.atleast_1d(to_val))
    #                 probability = bool_series.mean()
    #                 variance = bool_series.astype(float).var(ddof=DEGREES_OF_FREEDOM)
    #             else:
    #                 probability = np.nan
    #                 variance = np.nan

    #             probs_by_age.append(probability)
    #             vars_by_age.append(variance)

    #         # Save results for this transition type.
    #         key = f"trans_{from_label}_to_{to_label}"
    #         moments[key] = probs_by_age
    #         variances[f"var_{key}"] = vars_by_age

    for (from_label, from_val), (to_label, to_val) in product(
        states.items(), states.items()
    ):
        for age in age_range:
            df_age = df[df["age"] == age]
            subset = df_age[df_age[lagged_choice].isin(np.atleast_1d(from_val))]
            if len(subset) > 0:
                bool_series = subset[choice].isin(np.atleast_1d(to_val))
                probability = bool_series.mean()
                variance = bool_series.astype(float).var(ddof=DEGREES_OF_FREEDOM)
            else:
                probability = np.nan
                variance = np.nan

            # key = f"trans_{from_label}_to_{to_label}"
            key = f"trans_{from_label}_to_{to_label}{label}_age_{age}"
            moments[key] = probability
            variances[f"var_{key}"] = variance

    return moments, variances


# def compute_transition_moments_by_five_year_age_bins(
#     df,
#     full_time,
#     part_time,
#     not_working,
#     start_age,
#     end_age,
#     choice="choice",
#     lagged_choice="lagged_choice",
# ):
#     """
#     Compute year-to-year labor supply transition moments and variances
#     by 5-year age bin, e.g. [30,35), [35,40), etc.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain columns 'age', plus the columns in 'choice' and 'lagged_choice'.
#     full_time : array-like
#         Codes representing full-time in the 'choice' columns.
#     part_time : array-like
#         Codes representing part-time in the 'choice' columns.
#     not_working : array-like
#         Codes representing not-working states in the 'choice' columns.
#     start_age : int
#         Lower bound (inclusive) of the age range.
#     end_age : int
#         Upper bound (inclusive) of the age range.
#     choice : str, default "choice"
#         Column name for current year's labor supply choice.
#     lagged_choice : str, default "lagged_choice"
#         Column name for last year's labor supply choice.

#     Returns
#     -------
#     moments : dict
#         A dictionary of transition probabilities for each 5-year bin.
#         Keys look like: "transition_full_time_to_part_time_[30,35)".
#     variances : dict
#         Corresponding variances, with keys like:
#         "var_transition_full_time_to_part_time_[30,35)".
#     """

#     # --- 1) Define the 5-year bins ---
#     # If start_age=30, end_age=42 => bins = [30, 35, 40, 45]
#     # => intervals: [30,35), [35,40), [40,45)
#     bins = list(range(start_age, end_age + 1, 5))
#     if bins[-1] < end_age:
#         bins.append(end_age + 1)  # Ensure we cover up to end_age
#     # Create labels like "[30,35)", "[35,40)", etc.
#     bin_labels = [f"age_{bins[i]}_{bins[i+1] - 1}" for i in range(len(bins) - 1)]

#     # Prepare storage for aggregated results
#     moments = {}
#     variances = {}

#     # --- 2) Loop over each bin, filter df, compute transitions ---
#     for i in range(len(bin_labels)):
#         bin_label = bin_labels[i]
#         age_lower = bins[i]
#         age_upper = bins[i + 1]

#         # Filter rows whose age is in [age_lower, age_upper)
#         subdf = df[(df["age"] >= age_lower) & (df["age"] < age_upper)].copy()

#         # If there's no data in that bin, skip it
#         if subdf.empty:
#             continue

#         # --- 3) Compute transitions for the subsample ---
#         sub_moments, sub_variances = compute_transition_moments_and_variances(
#             subdf,
#             start_age,
#             end_age,
#             choice=choice,
#             lagged_choice=lagged_choice,
#         )

#         # --- 4) Append bin label to the keys and store ---
#         for k, v in sub_moments.items():
#             new_key = f"{k}_{bin_label}"
#             moments[new_key] = v

#         for k, v in sub_variances.items():
#             new_key = f"{k}_{bin_label}"
#             variances[new_key] = v

#     return moments, variances


def compute_wealth_moments_by_five_year_age_bins(df, edu_level, start_age, end_age):
    """
    Compute wealth moments (mean and variance) for 5-year age bins
    between start_age and end_age.

    Parameters
    ----------
    df : pd.DataFrame
            Must have columns 'age' and 'wealth'.
    edu_level : int
            Education level to filter on (0=low, 1=high).
    start_age : int
            Lower bound (inclusive) of the age range.
    end_age : int
            Upper bound (inclusive) of the age range.

    Returns
    -------
    moments : dict
            Dictionary of mean wealth by bin, with keys like 'wealth_[30,35)'.
    variances : dict
            Dictionary of variance of wealth by bin, matching the same keys.
    """

    # 1) Restrict to specified age range and education level
    edu_label = "low" if edu_level == 0 else "high"

    df_filtered = df[
        (df["age"] >= start_age)
        & (df["age"] <= end_age)
        & (df["education"] == edu_level)
    ].copy()

    # 2) Define 5-year age bins
    #    For example, if start_age=30, end_age=50, bins -> [30, 35, 40, 45, 50, 51]
    #    So you get these intervals: [30,35), [35,40), [40,45), [45,50), [50,51)
    #    The last bin may be shorter if end_age is not a multiple of 5.
    bins = list(range(start_age, end_age + 1, 5))
    if bins[-1] < end_age:
        bins.append(end_age + 1)  # ensure coverage up to end_age

    bin_labels = [
        f"age_{bins[i]}_{bins[i+1] - 1}_edu_{edu_label}" for i in range(len(bins) - 1)
    ]

    # 3) Assign each row to a bin
    df_filtered["age_bin"] = pd.cut(
        df_filtered["age"],
        bins=bins,
        right=False,  # left-closed, right-open intervals
        labels=bin_labels,
    )

    # 4) Group by the bin, compute mean & variance
    grouped = df_filtered.groupby("age_bin", observed=False)["wealth"]
    wealth_mean = grouped.mean()
    wealth_var = grouped.var(ddof=1)  # ddof=1 for sample variance

    # 5) Store in dictionaries
    moments = {}
    variances = {}
    for bin_label in wealth_mean.index:
        # e.g. bin_label might be "[30,35)"
        mean_key = f"wealth_{bin_label}"
        moments[mean_key] = wealth_mean[bin_label]
        variances[mean_key] = wealth_var[bin_label]

    return moments, variances


def compute_wealth_moments(subdf):
    """Compute mean wealth and its variance for a given subsample."""
    if len(subdf) == 0:
        return {
            "wealth_mean": np.nan,
            "wealth_var": np.nan,
        }

    wealth_mean = subdf["wealth"].mean()
    wealth_var = np.var(subdf["wealth"], ddof=DEGREES_OF_FREEDOM)

    return {
        "wealth_mean": wealth_mean,
        "wealth_var": wealth_var,
    }


# =====================================================================================
# Auxiliary functions
# =====================================================================================


# def _get_labor_shares_by_education_level_and_number_of_children_in_hh(
#     df, moments, variances, children_groups
# ):
#     for edu in (0, 1):
#         edu_label = "low" if edu == 0 else "high"
#         for grp_label, cond_func in children_groups.items():

#             subdf = df[(df["education"] == edu) & (df["children"].apply(cond_func))]
#             shares = compute_labor_shares(subdf)
#             key = f"children_{grp_label}_edu_{edu_label}"

#             moments[f"share_not_working_{key}"] = shares["not_working"]
#             moments[f"share_part_time_{key}"] = shares["part_time"]
#             moments[f"share_full_time_{key}"] = shares["full_time"]

#             variances[f"share_not_working_{key}"] = shares["not_working_var"]
#             variances[f"share_part_time_{key}"] = shares["part_time_var"]
#             variances[f"share_full_time_{key}"] = shares["full_time_var"]

#     return moments, variances


# def _get_labor_shares_by_educ_level_and_age_of_youngest_child(
#     df, moments, variances, kidage_bins
# ):
#     for edu in (0, 1):
#         edu_label = "low" if edu == 0 else "high"
#         for bin_label, (lb, ub) in kidage_bins.items():

#             subdf = df[
#                 (df["education"] == edu)
#                 & (df["kidage_youngest"] >= lb)
#                 & (df["kidage_youngest"] <= ub)
#             ]
#             shares = compute_labor_shares(subdf)
#             key = f"kidage_{bin_label}_edu_{edu_label}"

#             moments[f"share_full_time_{key}"] = shares["full_time"]
#             moments[f"share_part_time_{key}"] = shares["part_time"]
#             moments[f"share_unemployed_{key}"] = shares["unemployed"]
#             moments[f"share_retired_{key}"] = shares["retired"]

#             variances[f"share_full_time_{key}"] = shares["full_time_var"]
#             variances[f"share_part_time_{key}"] = shares["part_time_var"]
#             variances[f"share_unemployed_{key}"] = shares["unemployed_var"]
#             variances[f"share_retired_{key}"] = shares["retired_var"]

#     return moments, variances


def _get_wealth_moments_by_age_and_education(df, moments, variances, min_age, max_age):
    """
    Loop over age and education to compute mean wealth and variance,
    storing results in the same 'moments' and 'variances' dicts.
    """
    for edu in (0, 1):
        edu_label = "low" if edu == 0 else "high"
        for age in range(min_age, max_age + 1):
            subdf = df[(df["age"] == age) & (df["education"] == edu)]
            wealth = compute_wealth_moments(subdf)

            key = f"wealth_age_{age}_edu_{edu_label}"
            moments[key] = wealth["wealth_mean"]
            variances[key] = wealth["wealth_var"]

    return moments, variances


# ======================================================================================
# SHARE data set
# ======================================================================================


def get_share_informal_care_by_age_bin(
    dat,
    intensive_care_var,
    weight,
    age_bins,
):
    dat["intensive_care_weighted"] = dat[intensive_care_var] * dat[weight]

    share_intensive_care = []
    share_intensive_care += [
        dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            "intensive_care_weighted",
        ].sum()
        / dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]
    return pd.Series(
        {
            f"share_informal_care_{age_bin[0]}_{age_bin[1]}": share_intensive_care[i]
            for i, age_bin in enumerate(age_bins)
        },
    )


# =====================================================================================
# Bootstrap transition probabilities
# =====================================================================================


def compute_transition_moments(data, choice_map):
    """
    Compute transition probabilities (moments) from the data.
    Returns a dictionary with keys like 'transition_full_time_to_part_time'.
    """
    # Compute the transition matrix (normalized by row)
    transition_matrix = pd.crosstab(
        data["lagged_choice"], data["choice"], normalize="index"
    )

    # Store the transition probabilities with descriptive keys
    moments = {}
    for lag in transition_matrix.index:
        for current in transition_matrix.columns:
            from_state = choice_map.get(lag, lag)
            to_state = choice_map.get(current, current)
            moment_name = f"transition_{from_state}_to_{to_state}"
            moments[moment_name] = transition_matrix.loc[lag, current]

    return moments


def bootstrap_transition_variances(df, choice_map, n_bootstrap=1000):
    """
    Perform bootstrapping to estimate the variance of transition probabilities.

    Parameters:
      df         : The original DataFrame.
      n_bootstrap: Number of bootstrap replicates.

    Returns:
      A dictionary where keys are transition moment names
            (e.g., 'transition_full_time_to_part_time')
            and the values are the bootstrap variance estimates.

    """

    boot_estimates = {}

    # Run the bootstrap
    for _ in range(n_bootstrap):
        # Resample the dataframe with replacement
        boot_df = df.sample(frac=1, replace=True)

        # Compute moments from the bootstrap sample
        boot_moments = compute_transition_moments(boot_df, choice_map=choice_map)

        # Collect the bootstrapped estimates
        for key, value in boot_moments.items():
            if key not in boot_estimates:
                boot_estimates[key] = []
            boot_estimates[key].append(value)

    # Calculate the variance for each transition moment
    boot_variances = {
        key: np.var(values, ddof=1) for key, values in boot_estimates.items()
    }

    return boot_variances


# =====================================================================================
# Plotting
# =====================================================================================


# def plot_labor_shares_by_age(
#     df, age_var, start_age, end_age, condition_col=None, condition_val=None
# ):
#     """
#     Plots labor share outcomes by age using a specified age variable.
#     Outcomes include full time, part time, unemployed, retired, and
#     not working (unemployed + retired).

#     Parameters:
#       df (pd.DataFrame): The data frame containing the data.
#       age_var (str): Column to use as the age variable (e.g., "age" or
#           "kidage_youngest").
#       start_age (int): Starting age (inclusive) for the plot.
#       end_age (int): Ending age (inclusive) for the plot.
#       condition_col (str or list, optional): Column name(s) to filter data.
#       condition_val (any or list, optional): Value(s) for filtering.
#           If multiple, supply a list/tuple that matches condition_col.
#     """
#     # Apply conditioning if specified.
#     if condition_col is not None:
#         if isinstance(condition_col, (list, tuple)):
#             if not isinstance(condition_val, (list, tuple)) or (
#                 len(condition_col) != len(condition_val)
#             ):
#                 raise ValueError(
#                     "When condition_col is a list/tuple, condition_val must be a "
#                     "list/tuple of the same length."
#                 )
#             for col, val in zip(condition_col, condition_val, strict=False):
#                 df = df[df[col] == val]
#         else:
#             df = df[df[condition_col] == condition_val]

#     # Filter on the chosen age variable.
#     df = df[(df[age_var] >= start_age) & (df[age_var] <= end_age)]

#     ages = list(range(start_age, end_age + 1))
#     full_time_shares = []
#     part_time_shares = []
#     not_working_shares = []
#     # unemployed_shares = []
#     # retired_shares = []

#     # Loop over each age.
#     for age in ages:
#         subdf = df[df[age_var] == age]
#         shares = compute_labor_shares(subdf)
#         full_time_shares.append(shares["full_time"])
#         part_time_shares.append(shares["part_time"])
#         not_working_shares.append(shares["not_working"])
#         # unemployed_shares.append(shares["unemployed"])
#         # retired_shares.append(shares["retired"])

#     plt.figure(figsize=(10, 6))
#     plt.plot(ages, full_time_shares, marker="o", label="Full Time")
#     plt.plot(ages, part_time_shares, marker="o", label="Part Time")
#     plt.plot(ages, not_working_shares, marker="o", label="Not Working")
#     # plt.plot(ages, unemployed_shares, marker="o", label="Unemployed")
#     # plt.plot(ages, retired_shares, marker="o", label="Retired")

#     plt.xlabel(age_var.title())
#     plt.ylabel("Share")
#     title = (
#         "Labor Shares by "
#         + age_var.title()
#         + " (Ages "
#         + str(start_age)
#         + " to "
#         + str(end_age)
#         + ")"
#     )
#     if condition_col is not None:
#         if isinstance(condition_col, (list, tuple)):
#             conditions = ", ".join(
#                 [
#                     f"{col}={val}"
#                     for col, val in zip(condition_col, condition_val, strict=False)
#                 ]
#             )
#         else:
#             conditions = f"{condition_col}={condition_val}"
#         title += " | Conditions: " + conditions
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def plot_wealth_by_age(df, start_age, end_age, educ_val=None):
    """
    Plot mean wealth by age from start_age to end_age.

    Parameters
    ----------
    df : pd.DataFrame
            Data containing at least the columns 'age', 'education', and 'wealth'.
    start_age : int
            Lower bound (inclusive) of the age range.
    end_age : int
            Upper bound (inclusive) of the age range.
    educ_val : {0, 1, None}, optional
            If 0 or 1, filter the data to only rows where 'education' == educ_val.
            If None (default), use all education levels (unconditional).

    """

    # 1) Restrict to the specified age range
    df_ages = df[(df["age"] >= start_age) & (df["age"] <= end_age)]

    # 2) If educ_val is given (0 or 1), filter to that education category
    if educ_val is not None:
        df_ages = df_ages[df_ages["education"] == educ_val]

    # 3) Group by age and compute mean wealth
    grouped = df_ages.groupby("age", observed=False)["wealth"].mean().sort_index()

    # 4) Plot mean wealth by age
    plt.figure(figsize=(8, 5))
    plt.plot(grouped.index, grouped.values, marker="o", label="Mean Wealth")

    # Labels
    plt.xlabel("Age")
    plt.ylabel("Mean Wealth")

    # Construct a descriptive title
    if educ_val is None:
        title_str = "Mean Wealth by Age (All Education)"
    else:
        title_str = f"Mean Wealth by Age (Education={educ_val})"
    title_str += f" — Ages {start_age} to {end_age}"
    plt.title(title_str)

    plt.grid(True)
    plt.legend()
    plt.show()


def plot_wealth_by_5yr_bins(df, start_age, end_age, educ_val=None):
    """
    Plot mean wealth for 5-year age bins between start_age and end_age
    as a line chart, optionally filtering by education.

    Parameters
    ----------
    df : pd.DataFrame
            Must contain 'age', 'wealth', and 'education' columns.
    start_age : int
            Lower bound (inclusive) of the age range to plot.
    end_age : int
            Upper bound (inclusive) of the age range to plot.
    educ_val : {0, 1, None}, optional
            If 0 or 1, keep only rows where 'education' == edu_val.
            If None (default), use all education levels.

    """

    # 1) Filter rows by age
    df_filtered = df[(df["age"] >= start_age) & (df["age"] <= end_age)].copy()

    # 2) If edu_val is specified (0 or 1), filter further
    if educ_val is not None:
        df_filtered = df_filtered[df_filtered["education"] == educ_val]

    # 3) Define 5-year bins
    #    e.g. if start_age=30, end_age=50 => bins=[30, 35, 40, 45, 50, 51]
    #    Last bin covers [50,51) so that we include all ages <= end_age.
    bins = list(range(start_age, end_age + 1, 5))
    if bins[-1] < end_age:
        bins.append(end_age + 1)  # ensure coverage to end_age

    bin_labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]

    # 4) Assign each row to an interval
    df_filtered["age_bin"] = pd.cut(
        df_filtered["age"],
        bins=bins,
        right=False,  # left-closed, right-open
        labels=bin_labels,
    )

    # 5) Group by bin, compute mean wealth
    grouped = (
        df_filtered.groupby("age_bin", observed=False)["wealth"].mean().reset_index()
    )

    # 6) Plot a line chart with textual interval labels on the x-axis
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["age_bin"], grouped["wealth"], marker="o", linestyle="-")
    plt.xlabel("Age Bin")
    plt.ylabel("Mean Wealth")

    # Build a descriptive title
    if educ_val is None:
        edu_str = "All Education"
    else:
        edu_str = f"Education={educ_val}"
    plt.title(
        f"Mean Wealth by 5-Year Age Bins ({edu_str})\nAges {start_age} to {end_age}"
    )

    plt.xticks(rotation=45)  # rotate labels if they overlap
    plt.grid(True)
    plt.tight_layout()
    plt.show()
