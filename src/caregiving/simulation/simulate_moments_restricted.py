"""Restricted simulation: wealth-only and full with mean wealth by age bin."""

import numpy as np
import pandas as pd

from caregiving.model.shared import (
    DEAD,
    INFORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_INFORMAL_CARE,
    NOT_WORKING,
    SCALE_CAREGIVER_SHARE,
    WORK,
)
from caregiving.simulation.simulate_moments import (
    compute_transition_moments_pandas_for_age_bins,
    create_choice_shares_by_age_bin_pandas,
    create_labor_share_moments_by_age_bin_pandas,
    create_labor_share_moments_pandas,
    create_mean_by_age_bin,
    create_median_by_age_bin,
    create_pure_formal_care_moments_pandas,
)


def simulate_moments_pandas_wealth_only(
    df_full: pd.DataFrame, model_specs: dict
) -> pd.DataFrame:
    """Simulate only wealth moments (median wealth by age bin, by education).

    Same setup and keys as the wealth block of simulate_moments_pandas:
    median_assets_begin_of_period_low_education_age_bin_*, etc.
    """
    age_bins_wealth = model_specs["age_bins_wealth"]

    df_full = df_full.loc[df_full["health"] != DEAD].copy()
    df_full["mother_age"] = (
        df_full["age"].to_numpy()
        + model_specs["mother_age_diff"][df_full["education"].to_numpy()]
    )

    df_full_low = df_full[df_full["education"] == 0]
    df_full_high = df_full[df_full["education"] == 1]

    moments = {}
    moments = create_median_by_age_bin(
        df_full_low,
        moments,
        variable="assets_begin_of_period",
        age_bins_and_labels=age_bins_wealth,
        label="low_education",
    )
    moments = create_median_by_age_bin(
        df_full_high,
        moments,
        variable="assets_begin_of_period",
        age_bins_and_labels=age_bins_wealth,
        label="high_education",
    )

    return pd.Series(moments)


def simulate_moments_pandas_mean_wealth(  # noqa: PLR0915
    df_full: pd.DataFrame, model_specs: dict
) -> pd.Series:
    """Same as simulate_moments_pandas but use mean wealth by age bin instead of median.

    Full copy of simulate_moments_pandas with only the wealth block changed from
    create_median_by_age_bin to create_mean_by_age_bin.
    """
    start_age = model_specs["start_age"]
    end_age = model_specs["end_age_msm"]

    age_range = range(start_age, end_age + 1)

    age_bins_caregivers_5year = model_specs["age_bins_caregivers_5year"]
    age_bins_wealth = model_specs["age_bins_wealth"]
    age_bins_caregivers_3year = model_specs["age_bins_caregivers_3year"]

    df_full = df_full.loc[df_full["health"] != DEAD].copy()
    df_full["mother_age"] = (
        df_full["age"].to_numpy()
        + model_specs["mother_age_diff"][df_full["education"].to_numpy()]
    )

    df_non_caregivers = df_full[
        df_full["choice"].isin(np.asarray(NO_INFORMAL_CARE).tolist())
    ].copy()

    df_low = df_non_caregivers[df_non_caregivers["education"] == 0]
    df_high = df_non_caregivers[df_non_caregivers["education"] == 1]

    df_full_low = df_full[df_full["education"] == 0]
    df_full_high = df_full[df_full["education"] == 1]

    df_caregivers = df_full[
        df_full["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
    ].copy()
    df_caregivers_low = df_caregivers[df_caregivers["education"] == 0]
    df_caregivers_high = df_caregivers[df_caregivers["education"] == 1]

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

    moments = {}

    # Wealth moments: mean by age bin (only difference from simulate_moments_pandas)
    moments = create_mean_by_age_bin(
        df_full_low,
        moments,
        variable="assets_begin_of_period",
        age_bins_and_labels=age_bins_wealth,
        label="low_education",
    )
    moments = create_mean_by_age_bin(
        df_full_high,
        moments,
        variable="assets_begin_of_period",
        age_bins_and_labels=age_bins_wealth,
        label="high_education",
    )

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
        label="informal_care_any",
        scale=SCALE_CAREGIVER_SHARE,
    )
    moments = create_choice_shares_by_age_bin_pandas(
        df_full,
        moments,
        choice_set=LIGHT_INFORMAL_CARE,
        age_bins_and_labels=age_bins_caregivers_5year,
        label="informal_care_light",
        scale=SCALE_CAREGIVER_SHARE,
    )
    moments = create_choice_shares_by_age_bin_pandas(
        df_full,
        moments,
        choice_set=INTENSIVE_INFORMAL_CARE,
        age_bins_and_labels=age_bins_caregivers_5year,
        label="informal_care_intensive",
        scale=SCALE_CAREGIVER_SHARE,
    )

    moments = create_choice_shares_by_age_bin_pandas(
        df_full_low,
        moments,
        choice_set=INFORMAL_CARE,
        age_bins_and_labels=age_bins_caregivers_5year,
        label="informal_care_any_low_educ",
        scale=SCALE_CAREGIVER_SHARE,
    )
    moments = create_choice_shares_by_age_bin_pandas(
        df_full_high,
        moments,
        choice_set=INFORMAL_CARE,
        age_bins_and_labels=age_bins_caregivers_5year,
        label="informal_care_any_high_educ",
        scale=SCALE_CAREGIVER_SHARE,
    )

    moments = create_choice_shares_by_age_bin_pandas(
        df_full_low,
        moments,
        choice_set=LIGHT_INFORMAL_CARE,
        age_bins_and_labels=age_bins_caregivers_5year,
        label="informal_care_light_low_educ",
        scale=SCALE_CAREGIVER_SHARE,
    )
    moments = create_choice_shares_by_age_bin_pandas(
        df_full_high,
        moments,
        choice_set=LIGHT_INFORMAL_CARE,
        age_bins_and_labels=age_bins_caregivers_5year,
        label="informal_care_light_high_educ",
        scale=SCALE_CAREGIVER_SHARE,
    )

    moments = create_choice_shares_by_age_bin_pandas(
        df_full_low,
        moments,
        choice_set=INTENSIVE_INFORMAL_CARE,
        age_bins_and_labels=age_bins_caregivers_5year,
        label="informal_care_intensive_low_educ",
        scale=SCALE_CAREGIVER_SHARE,
    )
    moments = create_choice_shares_by_age_bin_pandas(
        df_full_high,
        moments,
        choice_set=INTENSIVE_INFORMAL_CARE,
        age_bins_and_labels=age_bins_caregivers_5year,
        label="informal_care_intensive_high_educ",
        scale=SCALE_CAREGIVER_SHARE,
    )

    moments["share_informal_care_high_educ"] = df_caregivers["education"].mean()

    moments = create_pure_formal_care_moments_pandas(df_full, moments)

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

    states_work_no_work = {
        "not_working": NOT_WORKING,
        "working": WORK,
    }
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

    states_caregiving = {
        "caregiving": INFORMAL_CARE,
    }
    moments = compute_transition_moments_pandas_for_age_bins(
        df_with_caregivers_low,
        moments,
        range(40, 71),
        states=states_caregiving,
        label="low_education",
        bin_width=31,
    )
    moments = compute_transition_moments_pandas_for_age_bins(
        df_with_caregivers_high,
        moments,
        range(40, 71),
        states=states_caregiving,
        label="high_education",
        bin_width=31,
    )

    return pd.Series(moments)
