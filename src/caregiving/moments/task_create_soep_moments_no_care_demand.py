"""Create SOEP moments and variances for MSM estimation.

No care demand counterfactual.
"""

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
    BAD_HEALTH,
    DEAD,
    GOOD_HEALTH,
    SEX,
    WEALTH_MOMENTS_SCALE,
    WEALTH_QUANTILE_CUTOFF,
)
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    NOT_WORKING_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)
from caregiving.moments.task_create_soep_moments import (
    adjust_and_trim_wealth_data,
    compute_labor_shares_by_age,
    compute_mean_wealth_by_age,
)
from caregiving.specs.task_write_specs import read_and_derive_specs

DEGREES_OF_FREEDOM = 1


def task_create_soep_moments_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_main_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "soep_moments_no_care_demand.csv",
    path_to_save_variances: Annotated[Path, Product] = BLD
    / "moments"
    / "soep_variances_no_care_demand.csv",
) -> None:
    """Create moments for MSM estimation - no care demand counterfactual."""

    specs = read_and_derive_specs(path_to_specs)

    start_age = specs["start_age"]
    end_age = specs["end_age_msm"]

    age_range = range(start_age, end_age + 1)
    age_range_wealth = range(start_age, specs["end_age_wealth"] + 1)

    _age_bins_75 = (
        list(range(40, 80, 5)),  # [40, 45, … , 70]
        [f"{s}_{s+4}" for s in range(40, 75, 5)],  # "40_44", …a
    )

    df_full = pd.read_csv(path_to_main_sample, index_col=[0])
    df = df_full[
        (df_full["gebjahr"] >= specs["min_birth_year"])
        & (df_full["gebjahr"] <= specs["max_birth_year"])
        & (df_full["syear"] <= specs["end_year"])
        & (df_full["sex"] == 1)
        & (df_full["age"] <= end_age + 10)
        & (df_full["any_care"] == 0)
    ]  # women only and non-caregivers

    _df_alive = df[df["health"] != DEAD].copy()
    _df_good_health = df[df["health"] == GOOD_HEALTH].copy()
    _df_bad_health = df[df["health"] == BAD_HEALTH].copy()

    df_wealth = df_full[df_full["sex"] == SEX].copy()
    df_wealth = adjust_and_trim_wealth_data(df=df_wealth, specs=specs)

    df_wealth_low = df_wealth[df_wealth["education"] == 0].copy()
    df_wealth_high = df_wealth[df_wealth["education"] == 1].copy()

    df_year = df[df["syear"] == 2012]  # 2012, 2016 # noqa: PLR2004
    _df_year_alive = df_year[df_year["health"] != DEAD].copy()
    _df_year_bad_health = df_year[df_year["health"] == BAD_HEALTH]
    _df_year_good_health = df_year[df_year["health"] == GOOD_HEALTH]

    df["kidage_youngest"] = df["kidage_youngest"] - 1

    df_low = df[df["education"] == 0].copy()
    df_high = df[df["education"] == 1].copy()

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
        label="wealth_low_education",
    )
    moments, variances = compute_mean_wealth_by_age(
        df_wealth_high,
        moments,
        variances,
        wealth_var="adjusted_wealth",
        age_range=age_range_wealth,
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

    # Save moments and variances
    moments_df = pd.DataFrame(list(moments.items()), columns=["moment", "value"])
    moments_df.to_csv(path_to_save_moments, index=False)

    variances_df = pd.DataFrame(list(variances.items()), columns=["moment", "value"])
    variances_df.to_csv(path_to_save_variances, index=False)
