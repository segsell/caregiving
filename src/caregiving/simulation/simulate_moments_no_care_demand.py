"""Simulate moments for no care demand counterfactual."""

import numpy as np
import pandas as pd

from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
)
from caregiving.simulation.simulate_moments import (
    create_mean_by_age,
)

FILL_VALUE_MISSING_AGE = np.nan

# =====================================================================================
# Pandas
# =====================================================================================


def simulate_moments_pandas_no_care_demand(
    df_full,
    model_specs,
) -> pd.DataFrame:
    """Simulate the model for given parametrization and model solution.

    Counterfactual without care demand.
    """

    start_age = model_specs["start_age"]
    end_age = model_specs["end_age_msm"]

    age_range = range(start_age, end_age + 1)
    age_range_wealth = range(start_age, model_specs["end_age_wealth"] + 1)

    df_low = df_full[df_full["education"] == 0].copy()
    df_high = df_full[df_full["education"] == 1].copy()

    # =================================================================================
    # Wealth moments
    # =================================================================================

    moments = {}

    # 0) Wealth by education and age bin
    moments = create_mean_by_age(
        df_low,
        moments,
        variable="assets_begin_of_period",
        age_range=age_range_wealth,
        label="low_education",
    )
    moments = create_mean_by_age(
        df_high,
        moments,
        variable="assets_begin_of_period",
        age_range=age_range_wealth,
        label="high_education",
    )

    # A) Moments by age.
    moments = create_labor_share_moments_pandas(
        df_full,
        moments=moments,
        age_range=age_range,
    )

    # B1) Moments by age and education.
    moments = create_labor_share_moments_pandas(
        df_low,
        moments=moments,
        age_range=age_range,
        label="low_education",
    )
    moments = create_labor_share_moments_pandas(
        df_high,
        moments=moments,
        age_range=age_range,
        label="high_education",
    )

    # C) Transitions
    # _states_work_no_work = {
    #     "not_working": NOT_WORKING,
    #     "working": WORK,
    # }
    # moments = compute_transition_moments_pandas_for_age_bins(
    #     df_low, moments, age_range, states=states_work_no_work, label="low_education"
    # )
    # moments = compute_transition_moments_pandas_for_age_bins(
    #     df_high, moments, age_range, states=states_work_no_work,
    # label="high_education"
    # )

    # =================================================================================

    return pd.Series(moments)


# =====================================================================================
# Create moments Pandas
# =====================================================================================


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
    retired_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(RETIREMENT_NO_CARE_DEMAND)).mean()
    )
    unemployed_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(UNEMPLOYED_NO_CARE_DEMAND)).mean()
    )
    part_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(PART_TIME_NO_CARE_DEMAND)).mean()
    )
    full_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(FULL_TIME_NO_CARE_DEMAND)).mean()
    )

    # Reindex to ensure that every age between start_age and end_age is included;
    # missing ages will be filled with NaN
    retired_shares = retired_shares.reindex(
        age_range, fill_value=FILL_VALUE_MISSING_AGE
    )
    unemployed_shares = unemployed_shares.reindex(
        age_range, fill_value=FILL_VALUE_MISSING_AGE
    )
    part_time_shares = part_time_shares.reindex(
        age_range, fill_value=FILL_VALUE_MISSING_AGE
    )
    full_time_shares = full_time_shares.reindex(
        age_range, fill_value=FILL_VALUE_MISSING_AGE
    )

    # Populate the moments dictionary for age-specific shares
    for age in age_range:
        moments[f"share_retired{label}_age_{age}"] = retired_shares.loc[age]
    for age in age_range:
        moments[f"share_unemployed{label}_age_{age}"] = unemployed_shares.loc[age]
    for age in age_range:
        moments[f"share_part_time{label}_age_{age}"] = part_time_shares.loc[age]
    for age in age_range:
        moments[f"share_full_time{label}_age_{age}"] = full_time_shares.loc[age]

    return moments
