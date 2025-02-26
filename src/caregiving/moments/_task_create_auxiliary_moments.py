"""Create auxiliary SHARE moments."""

from pathlib import Path

import pandas as pd

from caregiving.config import BLD

BASE_YEAR = 2015

MALE = 1
FEMALE = 2

MIN_AGE = 50
MAX_AGE = 65

AGE_50 = 50
AGE_53 = 53
AGE_56 = 56
AGE_59 = 59
AGE_62 = 62

AGE_55 = 55
AGE_60 = 60
AGE_65 = 65
AGE_70 = 70

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_create_share_parental_care(
    path_to_hh_weight: Path = BLD / "data" / "estimation_data_hh_weight.csv",
) -> None:
    """Create share parental care moments.

    intensive_care_no_other / intensive_care_general

    Shares of all intensive
    share parental: 50%
    share parents and parents in law: 66%

    # path_to_save: Annotated[Path, Product] = BLD
    # / "moments"
    # / "share_parental_of_all_intensive_care.csv",


    intensive_care_var_new?

    """
    dat_hh_weight = pd.read_csv(path_to_hh_weight)

    dat = dat_hh_weight.copy()

    weight = "hh_weight"
    intensive_all_care_var = "intensive_care_general"
    intensive_care_var = "intensive_care_no_other"

    intensive_care_outside = "intensive_care_outside"
    intensive_parental_care_outside = "intensive_parental_care_outside_no_other"

    age_bins_coarse = [
        (AGE_50, AGE_55),
        (AGE_55, AGE_60),
        (AGE_60, AGE_65),
    ]

    dat["intensive_all_care_weighted"] = dat[intensive_all_care_var] * dat[weight]
    dat["intensive_parental_care_weighted"] = dat[intensive_care_var] * dat[weight]
    dat["intensive_care_outside_weighted"] = dat[intensive_care_outside] * dat[weight]
    dat["intensive_parental_care_outside_weighted"] = (
        dat[intensive_parental_care_outside] * dat[weight]
    )

    share_intensive_parental_care = []
    share_intensive_parental_care += [
        dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            "intensive_parental_care_weighted",
        ].sum()
        / dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins_coarse
    ]

    share_intensive_all_care = []
    share_intensive_all_care += [
        dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            "intensive_all_care_weighted",
        ].sum()
        / dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins_coarse
    ]

    share_intensive_care_by_age_bin_coarse = turn_share_into_pandas_series(
        share_intensive_parental_care,
        age_bins_coarse,
    )
    share_parental_care_of_all_care_by_age_bin_coarse = (
        turn_share_of_shares_into_pandas_series(
            share_intensive_parental_care,
            share_intensive_all_care,
            age_bins_coarse,
        )
    )

    share_intensive_outside_care = create_share_by_age_bin(
        dat,
        intensive_care_outside,
        weight,
        age_bins_coarse,
    )
    share_intensive_parental_outside_care = create_share_by_age_bin(
        dat,
        intensive_parental_care_outside,
        weight,
        age_bins_coarse,
    )

    share_parental_care_of_all_care_outside = turn_share_of_shares_into_pandas_series(
        share_intensive_parental_outside_care,
        share_intensive_outside_care,
        age_bins_coarse,
    )
    share_parental_care_outside_of_parental = turn_share_of_shares_into_pandas_series(
        share_intensive_parental_outside_care,
        share_intensive_parental_care,
        age_bins_coarse,
    )

    return {
        "intensive_care": share_intensive_care_by_age_bin_coarse,
        "parental_care_of_all_care": share_parental_care_of_all_care_by_age_bin_coarse,
        "parental_care_outside_of_parental": share_parental_care_outside_of_parental,
        "parental_care_of_all_care_outside": share_parental_care_of_all_care_outside,
    }


def create_share_by_age_bin(dat, intensive_care_var, weight, age_bins):
    """Create share of intensive care by age bin.

    share_intensive_care_by_age_bin_coarse = pd.Series(     {
    f"{intensive_care_var}_{age_bin[0]}_{age_bin[1]}": share_intensive_care[i] for i,
    age_bin in enumerate(age_bins)     }, )

    """
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

    return share_intensive_care


def turn_share_into_pandas_series(share, age_bins):
    return pd.Series(
        {
            f"share_informal_care_{age_bin[0]}_{age_bin[1]}": share[i]
            for i, age_bin in enumerate(age_bins)
        },
    )


def turn_share_of_shares_into_pandas_series(subgroup, all_group, age_bins):
    return pd.Series(
        {
            f"share_informal_care_{age_bin[0]}_{age_bin[1]}": subgroup[i] / all_group[i]
            for i, age_bin in enumerate(age_bins)
        },
    )
