"""Restricted simulation: wealth-only and full with mean wealth by age bin."""

import numpy as np
import pandas as pd

from caregiving.model.shared import DEAD
from caregiving.simulation.simulate_moments import (
    create_mean_by_age_bin,
    create_median_by_age_bin,
    simulate_moments_pandas,
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


def simulate_moments_pandas_mean_wealth(
    df_full: pd.DataFrame, model_specs: dict
) -> pd.Series:
    """Same as simulate_moments_pandas but use mean wealth by age bin instead of median.

    All moment keys are identical except wealth: mean_assets_begin_of_period_*_age_bin_*
    instead of median_assets_begin_of_period_*_age_bin_*.
    """
    full_moments = simulate_moments_pandas(df_full, model_specs)
    moments_dict = full_moments.to_dict()

    # Drop median wealth keys (same as in simulate_moments_pandas)
    median_wealth_prefix = "median_assets_begin_of_period_"
    moments_dict = {
        k: v for k, v in moments_dict.items() if not k.startswith(median_wealth_prefix)
    }

    # Recompute wealth block with mean
    age_bins_wealth = model_specs["age_bins_wealth"]
    df_full = df_full.loc[df_full["health"] != DEAD].copy()
    df_full["mother_age"] = (
        df_full["age"].to_numpy()
        + model_specs["mother_age_diff"][df_full["education"].to_numpy()]
    )
    df_full_low = df_full[df_full["education"] == 0]
    df_full_high = df_full[df_full["education"] == 1]

    mean_moments = {}
    mean_moments = create_mean_by_age_bin(
        df_full_low,
        mean_moments,
        variable="assets_begin_of_period",
        age_bins_and_labels=age_bins_wealth,
        label="low_education",
    )
    mean_moments = create_mean_by_age_bin(
        df_full_high,
        mean_moments,
        variable="assets_begin_of_period",
        age_bins_and_labels=age_bins_wealth,
        label="high_education",
    )

    moments_dict.update(mean_moments)
    return pd.Series(moments_dict)
