"""Create the parent child data set of people older than 65."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import statsmodels.api as sm
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    MAX_AGE_PARENTS,
    MIN_AGE_PARENTS,
    PARENT_BAD_HEALTH,
    PARENT_GOOD_HEALTH,
    PARENT_MEDIUM_HEALTH,
)
from caregiving.utils import count, describe, table

WAVE_1 = 1
WAVE_2 = 2
WAVE_3 = 3
WAVE_4 = 4
WAVE_5 = 5
WAVE_6 = 6
WAVE_7 = 7
WAVE_8 = 8
WAVE_9 = 9

DAILY = 1

FIVE = 5
THREE = 3
TWO = 2
ONE = 1

MALE = 1
FEMALE = 2

HEALTH_EXCELLENT = 1
HEALTH_VERY_GOOD = 2
HEALTH_GOOD = 3
HEALTH_FAIR = 4
HEALTH_POOR = 5

RECEIVED_HELP_DAILY = 1


CHILD_ONE_GAVE_HELP = 10
STEP_CHILD_GAVE_HELP = 11
OTHER_CHILD_GAVE_HELP = 19
SN_PERSON_ONE = 101
SN_PERSON_SEVEN = 107

ANSWER_YES = 1
ANSWER_NO = 5
NO_HELP_FROM_OTHERS_OUTSIDE_HOUSEHOLD = 5
YES_TEMPORARILY = 1
YES_PERMANENTLY = 3

DUMMY_TRUE = 1
AT_LEAST_TWO = 2

# columns that record who gave help
WHO_IS_CAREGIVER_EXTRA_HH = ["sp003_1", "sp003_2", "sp003_3"]

AGE_BINS_PARENTS = [64, 74, 84, np.inf]
AGE_LABELS_PARENTS = ["65_74", "75_84", "85_plus"]

CARE_COLS = [
    "pure_informal_care_general",
    "pure_home_care_general",
    "combination_care_general",
]
DAILY_CARE_COLS = [
    "pure_informal_care_daily",
    "pure_home_care_daily",
    "combination_care_daily",
]
CHILD_CARE_COLS = [
    "pure_informal_care_child",
    "pure_home_care_child",
    "combination_care_child",
]
DAILY_CHILD_CARE_COLS = [
    "pure_informal_care_daily_child",
    "pure_home_care_daily_child",
    "combination_care_daily_child",
]
_CARE_COLS_ALL = [
    "pure_informal_care_general",
    "pure_home_care",
    "combination_care_general",
    "nursing_home",
]


def task_create_parent_child_data(
    path_to_raw_data: Path = BLD / "data" / "share_data_parent_child_merged.csv",
    path_to_save_main: Annotated[Path, Product] = BLD
    / "data"
    / "share_parent_child_data.csv",
    # path_to_design_weight: Annotated[Path, Product] = BLD
    # / "data"
    # / "share_parent_child_data_design_weight.csv",
    # path_to_hh_weight: Annotated[Path, Product] = BLD
    # / "data"
    # / "share_parent_child_data_hh_weight.csv",
    # path_to_ind_weight: Annotated[Path, Product] = BLD
    # / "data"
    # / "share_parent_child_data_ind_weight.csv",
    # parent couple - child
    path_to_save_couple: Annotated[Path, Product] = BLD
    / "data"
    / "share_parent_child_data_couple.csv",
    # path_to_design_weight_couple: Annotated[Path, Product] = BLD
    # / "data"
    # / "share_parent_child_data_couple_design_weight.csv",
    # path_to_hh_weight_couple: Annotated[Path, Product] = BLD
    # / "data"
    # / "share_parent_child_data_couple_hh_weight.csv",
    # path_to_ind_weight_couple: Annotated[Path, Product] = BLD
    # / "data"
    # / "share_parent_child_data_couple_ind_weight.csv",
) -> None:
    """Create the estimation data set."""
    dat = pd.read_csv(path_to_raw_data)

    dat["sex"] = dat["gender"].map({MALE: 0, FEMALE: 1})

    # Make prettier
    dat["age"] = dat.apply(
        lambda row: (
            row["int_year"] - row["yrbirth"]
            if row["int_month"] >= row["mobirth"]
            else row["int_year"] - row["yrbirth"] - 1
        ),
        axis=1,
    )

    dat = dat[(dat["age"] >= MIN_AGE_PARENTS) & (dat["age"] <= MAX_AGE_PARENTS + 5)]

    dat = create_children_information(dat)
    dat = create_married_or_partner_alive(dat)

    # Health, ADL and IADL
    dat = create_health_variables(dat)
    dat = create_limitations_with_adl_categories(dat)

    dat = create_care_variables(dat)
    dat = create_care_combinations(dat, informal_care_var="informal_care_child")

    dat = dat.reset_index(drop=True)
    # dat_design_weight = multiply_rows_with_weight(dat, weight="design_weight")
    # dat_hh_weight = multiply_rows_with_weight(dat, weight="hh_weight")
    # dat_ind_weight = multiply_rows_with_weight(dat, weight="ind_weight")

    # Create couple data
    dat_couple = create_couple_data(dat)
    # dat_couple_design_weight = create_couple_data(dat_design_weight)
    # dat_couple_hh_weight = create_couple_data(dat_hh_weight)
    # dat_couple_ind_weight = create_couple_data(dat_ind_weight)

    # Save
    dat.to_csv(path_to_save_main, index=False)
    # dat_design_weight.to_csv(path_to_design_weight, index=False)
    # dat_hh_weight.to_csv(path_to_hh_weight, index=False)
    # dat_ind_weight.to_csv(path_to_ind_weight, index=False)

    dat_couple.to_csv(path_to_save_couple, index=False)
    # dat_couple_design_weight.to_csv(path_to_design_weight_couple, index=False)
    # dat_couple_hh_weight.to_csv(path_to_hh_weight_couple, index=False)
    # dat_couple_ind_weight.to_csv(path_to_ind_weight_couple, index=False)


def create_couple_data(data):
    """Create data set with couple information of both parents."""
    dat_partner = data.copy()
    dat_female = data.copy()

    dat_partner["mergeid"] = dat_partner["mergeidp"]
    columns_to_keep = [
        "mergeid",
        "int_year",
        "gender",
        "married",
        "age",
        "health",
        "any_care",
    ]
    dat_partner = dat_partner[columns_to_keep]

    dat_female = dat_female[dat_female["gender"] == FEMALE]
    dat_partner_male = dat_partner[dat_partner["gender"] == MALE]

    male_columns = {
        "gender": "father_gender",
        "married": "father_married",
        "age": "father_age",
        "health": "father_health",
        "any_care": "father_any_care",
    }
    dat_partner_male = dat_partner_male.rename(columns=male_columns)

    female_columns = {
        "gender": "mother_gender",
        "married": "mother_married",
        "age": "mother_age",
        "health": "mother_health",
        "any_care": "mother_any_care",
    }
    dat_female = dat_female.rename(columns=female_columns)

    return dat_female.merge(dat_partner_male, on=["mergeid", "int_year"], how="inner")


def multiply_rows_with_weight(dat, weight):  # noqa: PLR0915
    # Create a DataFrame of weights with the same shape as dat
    weights = dat[weight].to_numpy().reshape(-1, 1)

    static_cols = [
        "mergeid",
        "mergeidp",
        "coupleid",
        "gender",
        "int_year",
        "int_month",
        "age",
        "only_informal",
        "only_formal",
        "only_home_care",
        "combination_care_general",
        "nursing_home",
        "informal_care_child",
        "informal_care_general",
        "home_care",
        "formal_care",
        "no_informal_care_child",
        "no_home_care",
        "no_formal_care",
        "no_combination_care",
        "no_only_formal",
        "no_only_informal",
        "no_care",
        "informal_care_child_no_comb",
        "formal_care_no_comb",
        "lagged_home_care",
        "lagged_formal_care",
        "lagged_informal_care_general",
        "lagged_informal_care_child",
        "lagged_combination_care_general",
        "lagged_no_informal_care_child",
        "lagged_no_home_care",
        "lagged_no_formal_care",
        "lagged_no_combination_care",
        "lagged_only_formal",
        "lagged_only_informal",
        "lagged_no_only_formal",
        "lagged_no_only_informal",
        "health",
        "married",
        "has_two_daughters",
        "has_two_children",
        "has_daughter",
        "wave",
        weight,
    ]
    data_columns = dat.drop(columns=static_cols).to_numpy()

    result = data_columns * weights

    dat_weighted = pd.DataFrame(
        result,
        columns=[col for col in dat.columns if col not in static_cols],
    )
    dat_weighted.insert(0, "mergeid", dat["mergeid"])
    dat_weighted.insert(1, "int_year", dat["int_year"])
    dat_weighted.insert(2, "int_month", dat["int_month"])
    dat_weighted.insert(3, "age", dat["age"])
    dat_weighted.insert(4, weight, dat[weight])
    dat_weighted.insert(5, "only_informal", dat["only_informal"])
    dat_weighted.insert(6, "combination_care_general", dat["combination_care_general"])
    dat_weighted.insert(7, "only_home_care", dat["only_home_care"])
    dat_weighted.insert(8, "informal_care_child", dat["informal_care_child"])
    dat_weighted.insert(9, "informal_care_general", dat["informal_care_general"])
    dat_weighted.insert(10, "home_care", dat["home_care"])
    dat_weighted.insert(11, "health", dat["health"])
    dat_weighted.insert(12, "married", dat["married"])
    dat_weighted.insert(13, "wave", dat["wave"])
    dat_weighted.insert(14, "lagged_home_care", dat["lagged_home_care"])
    dat_weighted.insert(
        15,
        "lagged_informal_care_general",
        dat["lagged_informal_care_general"],
    )
    dat_weighted.insert(
        16,
        "lagged_informal_care_child",
        dat["lagged_informal_care_child"],
    )
    dat_weighted.insert(
        17,
        "lagged_combination_care",
        dat["lagged_combination_care"],
    )
    dat_weighted.insert(18, "gender", dat["gender"])
    dat_weighted.insert(
        19,
        "lagged_no_informal_care_child",
        dat["lagged_no_informal_care_child"],
    )
    dat_weighted.insert(20, "lagged_no_home_care", dat["lagged_no_home_care"])
    dat_weighted.insert(21, "no_informal_care_child", dat["no_informal_care_child"])
    dat_weighted.insert(22, "no_home_care", dat["no_home_care"])
    dat_weighted.insert(23, "coupleid", dat["coupleid"])
    dat_weighted.insert(24, "mergeidp", dat["mergeidp"])
    dat_weighted.insert(25, "formal_care", dat["formal_care"])
    dat_weighted.insert(26, "lagged_formal_care", dat["lagged_formal_care"])
    dat_weighted.insert(27, "has_two_daughters", dat["has_two_daughters"])
    dat_weighted.insert(28, "has_two_children", dat["has_two_children"])
    dat_weighted.insert(29, "no_formal_care", dat["no_formal_care"])
    dat_weighted.insert(
        30,
        "lagged_no_combination_care",
        dat["lagged_no_combination_care"],
    )
    dat_weighted.insert(31, "lagged_no_formal_care", dat["lagged_no_formal_care"])
    dat_weighted.insert(32, "only_formal", dat["only_formal"])
    dat_weighted.insert(33, "lagged_only_formal", dat["lagged_only_formal"])
    dat_weighted.insert(34, "lagged_only_informal", dat["lagged_only_informal"])
    dat_weighted.insert(35, "no_only_formal", dat["no_only_formal"])
    dat_weighted.insert(36, "no_only_informal", dat["no_only_informal"])
    dat_weighted.insert(37, "lagged_no_only_formal", dat["lagged_no_only_formal"])
    dat_weighted.insert(38, "lagged_no_only_informal", dat["lagged_no_only_informal"])
    dat_weighted.insert(38, "no_combination_care", dat["no_combination_care"])
    dat_weighted.insert(39, "no_care", dat["no_care"])
    dat_weighted.insert(39, "formal_care_no_comb", dat["formal_care_no_comb"])
    dat_weighted.insert(
        40,
        "informal_care_child_no_comb",
        dat["informal_care_child_no_comb"],
    )
    dat_weighted.insert(41, "has_daughter", dat["has_daughter"])
    dat_weighted.insert(42, "nursing_home", dat["nursing_home"])
    dat_weighted.insert(43, "combination_care_child", dat["combination_care_child"])

    dat_weighted[f"{weight}_avg"] = dat_weighted.groupby("mergeid")[weight].transform(
        "mean",
    )

    return dat_weighted


def create_health_variables(dat):
    """Create dummy for health status.

    Impute missing values!!!

    """
    dat = replace_negative_values_with_nan(dat, "ph003_")

    _cond = [
        (dat["ph003_"] == HEALTH_EXCELLENT)
        | (dat["ph003_"] == HEALTH_VERY_GOOD)
        | (dat["ph003_"] == HEALTH_GOOD),
        (dat["ph003_"] == HEALTH_FAIR),
        (dat["ph003_"] == HEALTH_POOR),
    ]
    _val = [PARENT_GOOD_HEALTH, PARENT_MEDIUM_HEALTH, PARENT_BAD_HEALTH]

    dat["health"] = np.select(_cond, _val, default=np.nan)

    return dat


def create_limitations_with_adl_categories(df):
    """Create limitations with ADL categories.

    0: Care degree 0 or 1
    1: Care degree 2
    2: Care degree 3
    3: Care degree 4 or 5

    Less than 2 ADL and no IADL: category 0
    At least 2 ADL and at least 1 IADL: category 1
    At least 3 ADL and at least 3 IADL: category 2
    At least 5 ADL and at least 5 IADL: category 3

    """

    # Identify rows where both 'adl' and 'iadl' are NaN
    both_nan = df["adl"].isna() & df["iadl"].isna()

    # Conditions for each category
    cond_3 = (df["adl"] >= FIVE) & (df["iadl"] >= FIVE)
    cond_2 = (df["adl"] >= THREE) & (df["iadl"] >= THREE)
    cond_1 = (df["adl"] >= TWO) & (df["iadl"] >= ONE)

    # Create 'adl_cat' using np.select
    df["adl_cat"] = np.select([cond_3, cond_2, cond_1], [3, 2, 1], default=0)

    # Set 'adl_cat' to NaN where both 'adl' and 'iadl' are NaN
    df.loc[both_nan, "adl_cat"] = np.nan

    return df


def create_care_variables(dat):  # noqa: PLR0915
    """Create a dummy for formal care."""
    dat = _process_negative_values(dat)

    # nursing home
    _cond = [
        (dat["hc029_"].isin([YES_TEMPORARILY, YES_PERMANENTLY]) | (dat["hc031_"] > 0)),
        dat["hc029_"] == ANSWER_NO,
    ]
    _val = [1, 0]
    dat["nursing_home"] = np.select(_cond, _val, default=np.nan)

    # # ===============================================================================
    # df = dat.loc[(dat["health"] == 1) | (dat["nursing_home"] == 1)]
    # df = df[df["age"] >= 65]

    # age_labels = ["65-69", "70-74", "75-79", "80-84", "85+"]

    # df["age_bin"] = pd.cut(
    #     df["age"],
    #     bins=[65, 70, 75, 80, 85, np.inf],
    #     right=False,
    #     labels=age_labels,
    #     include_lowest=True,
    # )
    # age_dummies = pd.get_dummies(df["age_bin"], drop_first=True)
    # df = pd.concat([df, age_dummies], axis=1)

    # coeff_nursing_home = weighted_logistic_regression(
    #     df,
    #     age_dummies,
    #     outcome="nursing_home",
    #     weight=weight,
    # )

    # # ===============================================================================
    # RENAME WAVES 1, 2 (3, 4 missing): hc127d1

    # formal home care by professional nursing service
    _cond = [
        (dat["hc032d1"] == 1) | (dat["hc032d2"] == 1) | (dat["hc032d3"] == 1)
        # | (dat["hc032dno"] == 1)
        | (dat["hc127d1"] == 1)
        | (dat["hc127d2"] == 1)
        | (dat["hc127d3"] == 1)
        | (dat["hc127d4"] == 1),
        # | (dat["hc127dno"] == 0),
        (
            dat["hc032d1"].isna()
            & dat["hc032d2"].isna()
            & dat["hc032d3"].isna()
            & dat["hc032dno"].isna()
            & dat["hc127d1"].isna()
            & dat["hc127d2"].isna()
            & dat["hc127d3"].isna()
            & dat["hc127d4"].isna()
        ),
        # (
        #     (dat["hc032d1"] == 0)
        #     & (dat["hc032d2"] == 0)
        #     & (dat["hc032d3"] == 0)
        #     & (dat["hc032dno"] == 0)
        #     & (dat["hc127d1"] == 0)
        #     & (dat["hc127d2"] == 0)
        #     & (dat["hc127d3"] == 0)
        #     & (dat["hc127d4"] == 0)
        # ),
        # & (dat["hc127dno"] == 1),
    ]
    _val = [1, np.nan]
    dat["home_care"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["nursing_home"] == 1) | (dat["home_care"] == 1),
        (dat["nursing_home"].isna()) & (dat["home_care"].isna()),
    ]
    _val = [1, np.nan]
    dat["formal_care"] = np.select(_cond, _val, default=0)

    # informal care by own children
    _cond = [
        dat["sp021d10"] == ANSWER_YES,  # help within household from own children
        # help outside the household from own children
        (
            dat["wave"].isin([WAVE_1, WAVE_2, WAVE_5])
            & (
                (
                    dat["sp003_1"].between(CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP)
                    # & (dat["sp005_1"] == RECEIVED_HELP_DAILY)
                )
                | (
                    dat["sp003_2"].between(CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP)
                    # & (dat["sp005_2"] == RECEIVED_HELP_DAILY)
                )
                | (
                    dat["sp003_3"].between(CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP)
                    # & (dat["sp005_1"] == RECEIVED_HELP_DAILY)
                )
            )
        )
        | (
            (dat["wave"].isin([WAVE_6, WAVE_7, WAVE_8, WAVE_9]))
            & (
                (
                    dat["sp003_1"]
                    == CHILD_ONE_GAVE_HELP
                    # & (dat["sp005_1"] == RECEIVED_HELP_DAILY)
                )
                | (
                    dat["sp003_2"]
                    == CHILD_ONE_GAVE_HELP
                    # & (dat["sp005_2"] == RECEIVED_HELP_DAILY)
                )
                | (
                    dat["sp003_3"]
                    == CHILD_ONE_GAVE_HELP
                    # & (dat["sp005_3"] == RECEIVED_HELP_DAILY)
                )
            )
        ),
        (dat["mstat"].isin([3, 4, 5, 6]))
        & (dat["sp020_"] == ANSWER_NO)
        & (dat["sp002_"] == ANSWER_NO),
        (dat["sp020_"]).isna() & (dat["sp002_"]).isna(),
        ((dat["sp020_"] == ANSWER_YES) & (dat["sp021d10"] == ANSWER_NO))
        & ((dat["sp002_"] == ANSWER_YES) & (dat["sp021d10"] == ANSWER_NO)),
    ]
    _val = [1, 1, 0, np.nan, np.nan]
    dat["informal_care_child"] = np.select(_cond, _val, default=np.nan)

    # informal care general
    _cond = [
        (dat["sp020_"] == 1)  # care from inside the household with personal care
        | (
            dat["sp002_"]
            == 1
            # anyone from the outside the household given any type of help
            # & (
            #     (dat["sp005_1"] == DAILY)
            #     | (dat["sp005_2"] == DAILY)
            #     | (dat["sp005_3"] == DAILY)
            # )
        ),
        (dat["sp020_"].isna()) & (dat["sp002_"].isna()),
        # (dat["sp020_"] == ANSWER_NO) & (dat["sp002_"] == ANSWER_NO),
    ]
    _val = [1, np.nan]
    # _val = [1, 0]
    dat["informal_care_general"] = np.select(_cond, _val, default=0)

    # _cond = [
    #     (dat["sp020_"] == 1)  # care from inside the household with personal care
    #     | (
    #         (
    #             dat["sp002_"] == 1
    #         )  # anyone from the outside the household given any type of help
    #         # & (
    #         #     (dat["sp005_1"] == DAILY)
    #         #     | (dat["sp005_2"] == DAILY)
    #         #     | (dat["sp005_3"] == DAILY)
    #         # )
    #     ),
    #     (dat["sp020_"].isna()) & (dat["sp002_"].isna()),
    #     # (dat["sp020_"] == ANSWER_NO) & (dat["sp002_"] == ANSWER_NO),
    # ]
    # _val = [1, np.nan]
    # # _val = [1, 0]
    # dat["informal_care_general_alt"] = np.select(_cond, _val, default=0)

    # ================================================================================
    # NEW
    # ================================================================================

    outside_daily_count = (
        ((dat["sp003_1"] >= 1) & (dat["sp005_1"] == DAILY)).astype(int)
        + ((dat["sp003_2"] >= 1) & (dat["sp005_2"] == DAILY)).astype(int)
        + ((dat["sp003_3"] >= 1) & (dat["sp005_3"] == DAILY)).astype(int)
    )
    # inside_count = dat.filter(regex=r"^sp020_").eq(1).sum(axis=1)
    inside_count = dat["sp020_"].eq(ANSWER_YES).astype(int)

    _cond = [
        # Case 1:  two or more informal caregivers in total
        (inside_count + outside_daily_count) >= 1,
        # Case 2:  explicitly no informal care at all
        (dat["sp020_"] == ANSWER_NO) & (dat["sp002_"] == ANSWER_NO),
        # (dat.filter(regex=r"^sp020_").eq(ANSWER_NO).all(axis=1))
        # & (dat["sp002_"] == ANSWER_NO),
    ]
    _val = [1, 0]
    dat["informal_care_daily"] = np.select(_cond, _val, default=np.nan)

    _cond = [
        # Case 1:  two or more informal caregivers in total
        (inside_count + outside_daily_count) >= TWO,
        (inside_count + outside_daily_count) == 1,
        # Case 2:  explicitly no informal care at all
        # (dat["sp020_"] == ANSWER_NO) & (dat["sp002_"] == ANSWER_NO),
    ]
    _val = [1, 0]
    dat["informal_care_daily_two"] = np.select(_cond, _val, default=np.nan)

    # table(dat["informal_care_daily_two"])
    # np.mean(dat["informal_care_daily_two"])
    # np.float64(0.1617473435655254)
    # np.var(dat["informal_care_daily_two"])
    # np.float64(0.13558514041502123)
    # np.std(dat["informal_care_daily_two"])
    # np.float64(0.36821887569083317)

    # ============================================================================
    # Children from inside household
    # ============================================================================
    who_is_caregiver_w1_2_5 = [
        f"sp021d{n}" for n in range(CHILD_ONE_GAVE_HELP, CHILD_ONE_GAVE_HELP + 2)
    ]
    at_least_one_child_provides_care_from_inside_hh = (
        # Waves 1, 2, 5  ─ any of sp021d10…19 equals 1
        # (
        #     dat["wave"].isin([WAVE_1, WAVE_2, WAVE_5])
        #     & (dat[child_vars_w1_2_5].eq(1).any(axis=1))
        # )
        (
            dat["wave"].isin([WAVE_1, WAVE_2, WAVE_5])
            & (
                (dat[who_is_caregiver_w1_2_5] >= CHILD_ONE_GAVE_HELP)
                & (dat[who_is_caregiver_w1_2_5] <= OTHER_CHILD_GAVE_HELP)
            ).sum(axis=1)
            >= 1
        )
        |
        # Wave 4  ─ special social-network logic
        ((dat["wave"] == WAVE_4) & (dat.apply(_has_child_in_wave4, axis=1)))
        |
        # Waves 6-9  ─ sp021d10 equals 1
        (dat["wave"].isin([WAVE_6, WAVE_7, WAVE_8, WAVE_9]) & (dat["sp021d10"] == 1))
    )

    # ============================================================================
    # Children from outside household
    # ============================================================================
    at_least_one_child_provides_care_from_outside_hh = (
        # Waves 1, 2, 5 : "child" is any value in
        # [CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP]
        (
            dat["wave"].isin([WAVE_1, WAVE_2, WAVE_5])
            & (
                (
                    (dat[WHO_IS_CAREGIVER_EXTRA_HH] >= CHILD_ONE_GAVE_HELP)
                    & (dat[WHO_IS_CAREGIVER_EXTRA_HH] <= OTHER_CHILD_GAVE_HELP)
                ).sum(axis=1)
                >= 1
            )
        )
        |
        # Wave 4  ─ special social-network coding
        ((dat["wave"] == WAVE_4) & (dat.apply(_count_children_wave4, axis=1) >= 1))
        |
        # Waves 6-9  ─ “child” is exactly CHILD_ONE_GAVE_HELP
        (
            dat["wave"].isin([WAVE_6, WAVE_7, WAVE_8, WAVE_9])
            & (dat[WHO_IS_CAREGIVER_EXTRA_HH].eq(CHILD_ONE_GAVE_HELP).sum(axis=1) >= 1)
        )
    )
    at_least_two_children_provide_care_from_outside_hh = (
        # Waves 1, 2, 5 : "child" is any value in
        # [CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP]
        (
            dat["wave"].isin([WAVE_1, WAVE_2, WAVE_5])
            & (
                (
                    (dat[WHO_IS_CAREGIVER_EXTRA_HH] >= CHILD_ONE_GAVE_HELP)
                    & (dat[WHO_IS_CAREGIVER_EXTRA_HH] <= OTHER_CHILD_GAVE_HELP)
                ).sum(axis=1)
                >= TWO
            )
        )
        |
        # Wave 4  ─ special social-network coding
        ((dat["wave"] == WAVE_4) & (dat.apply(_count_children_wave4, axis=1) >= TWO))
        |
        # Waves 6-9  ─ “child” is exactly CHILD_ONE_GAVE_HELP
        (
            dat["wave"].isin([WAVE_6, WAVE_7, WAVE_8, WAVE_9])
            & (
                dat[WHO_IS_CAREGIVER_EXTRA_HH].eq(CHILD_ONE_GAVE_HELP).sum(axis=1)
                >= TWO
            )
        )
    )

    # ============================================================================

    # ------------------------------------------------------------------
    # Helper: Is <code> a child in WAVE-4?
    # keeps the naming pattern you fixed earlier (sp021d1sn … sp021d7sn)
    # ------------------------------------------------------------------
    def _is_child_wave4(who: int, row) -> bool:
        if who == OTHER_CHILD_GAVE_HELP:  # direct child
            return True
        if SN_PERSON_ONE <= who <= SN_PERSON_SEVEN:  # social-network person 1–7
            return row.get(f"sn005_{who-100}", None) == CHILD_ONE_GAVE_HELP
        return False

    # ------------------------------------------------------------------
    # Helper: count *daily* child caregivers OUTSIDE the household
    # across ALL waves (sp003_1…3 paired with sp005_1…3)
    # ------------------------------------------------------------------
    def _count_daily_children_outside(row) -> int:
        cnt = 0
        wave = row["wave"]

        for idx in (1, 2, 3):
            who = row.get(f"sp003_{idx}", np.nan)
            freq = row.get(f"sp005_{idx}", np.nan)

            # must give help *and* do so daily
            if pd.isna(who) or pd.isna(freq) or freq != DAILY:
                continue

            # ----- Wave-specific child test --------------------------------
            if wave in (WAVE_1, WAVE_2, WAVE_5):
                if CHILD_ONE_GAVE_HELP <= who <= OTHER_CHILD_GAVE_HELP:
                    cnt += 1

            elif wave == WAVE_4:
                if _is_child_wave4(who, row):
                    cnt += 1

            elif wave in (WAVE_6, WAVE_7, WAVE_8, WAVE_9):
                if who == CHILD_ONE_GAVE_HELP:
                    cnt += 1

        return cnt

    # vectorised Series with the per-row counts
    daily_child_outside_cnt = dat.apply(_count_daily_children_outside, axis=1)

    # ------------------------------------------------------------------
    # NEW masks that include the DAILY condition
    # ------------------------------------------------------------------
    _exactly_one_child_provides_care_from_inside_hh = (
        at_least_one_child_provides_care_from_inside_hh & (inside_count == 1)
    )
    _exactly_one_child_provides_care_from_outside_hh = (
        at_least_one_child_provides_care_from_outside_hh
        & (daily_child_outside_cnt == 1)
    )

    _exactly_one_child_provides_care_from_inside_hh = (
        at_least_one_child_provides_care_from_inside_hh.copy()
    )
    _exactly_one_child_provides_care_from_outside_hh = daily_child_outside_cnt == 1

    at_least_one_child_provides_care_from_outside_hh = daily_child_outside_cnt >= 1
    at_least_two_children_provide_care_from_outside_hh = daily_child_outside_cnt >= TWO

    # =============================================================================

    _cond = [
        at_least_one_child_provides_care_from_inside_hh
        | at_least_one_child_provides_care_from_outside_hh,
        (inside_count + outside_daily_count) >= 1,
    ]
    _val = [1, 0]
    dat["informal_care_daily_child"] = np.select(_cond, _val, default=np.nan)

    _cond = [
        at_least_two_children_provide_care_from_outside_hh
        | (
            at_least_one_child_provides_care_from_inside_hh
            & at_least_one_child_provides_care_from_outside_hh
        ),
        (inside_count + outside_daily_count) >= 1,
    ]
    _val = [1, 0]
    dat["informal_care_daily_two_children"] = np.select(_cond, _val, default=np.nan)

    _cond = [
        at_least_two_children_provide_care_from_outside_hh,
        at_least_one_child_provides_care_from_inside_hh
        & at_least_one_child_provides_care_from_outside_hh,
        (
            at_least_one_child_provides_care_from_inside_hh
            | at_least_one_child_provides_care_from_outside_hh
        ),
    ]
    _val = [1, 1, 0]
    dat["informal_care_daily_two_children_versus_one"] = np.select(
        _cond, _val, default=np.nan
    )

    # _cond = [
    #     at_least_two_children_provide_care_from_outside_hh,
    #     at_least_one_child_provides_care_from_inside_hh
    #     & at_least_one_child_provides_care_from_outside_hh,
    #     (inside_count + outside_daily_count) >= 1,
    # ]
    # _val = [1, 1, 0]
    # dat["informal_care_daily_two_children_cond"] = np.select(
    #     _cond, _val, default=np.nan
    # )

    _at_least_one_child_provides_care = (
        at_least_one_child_provides_care_from_inside_hh
        | at_least_one_child_provides_care_from_outside_hh
    )
    _at_least_two_children_provide_care = (
        at_least_two_children_provide_care_from_outside_hh
        | (
            at_least_one_child_provides_care_from_inside_hh
            & at_least_one_child_provides_care_from_outside_hh
        )
    )

    # ────────────────────────────────────────────────────────────────
    # 0 Baseline: receives any informal care (inside YES  OR  outside-daily ≥1)
    # ────────────────────────────────────────────────────────────────
    receives_informal_care = (inside_count + outside_daily_count) >= 1

    # ────────────────────────────────────────────────────────────────
    # 1 Has help from at least ONE child (inside OR outside)
    # ────────────────────────────────────────────────────────────────
    one_child_caregiver = (
        at_least_one_child_provides_care_from_inside_hh
        | at_least_one_child_provides_care_from_outside_hh
    )

    # ────────────────────────────────────────────────────────────────
    # 2 Has help from at least TWO children  (inside + outside combined)
    #     • any combination totalling ≥2 is accepted
    #       – ≥2 children outside
    #       – 1 inside  + 1 outside
    #       – ≥2 children inside (possible in Waves 1,2,5)
    # ────────────────────────────────────────────────────────────────
    # ­--- 2.a  two children outside (mask already exists) ­---
    two_children_outside = at_least_two_children_provide_care_from_outside_hh

    # ­--- 2.b  one child inside  + one child outside ­---
    one_in_one_out = (
        at_least_one_child_provides_care_from_inside_hh
        & at_least_one_child_provides_care_from_outside_hh
    )

    two_children_caregivers = two_children_outside | one_in_one_out

    # ────────────────────────────────────────────────────────────────
    # 3 Shares among everyone who receives informal care
    # ────────────────────────────────────────────────────────────────
    base_n = receives_informal_care.sum()

    _share_one_child = (
        (receives_informal_care & one_child_caregiver).sum() / base_n
    ) * 100
    _share_two_child = (
        (receives_informal_care & two_children_caregivers).sum() / base_n
    ) * 100

    _cond = [
        (dat["nursing_home"] == 1)
        | (dat["home_care"] == 1)
        | dat["informal_care_daily"]
        == 1,
        (dat["nursing_home"].isna())
        & (dat["home_care"].isna())
        & (dat["informal_care_daily"].isna()),
    ]
    _val = [1, np.nan]
    dat["any_care_daily"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["nursing_home"] == 1)
        | (dat["home_care"] == 1)
        | dat["informal_care_daily"]
        == 1,
        (dat["nursing_home"].isna())
        & (dat["home_care"].isna())
        & (dat["informal_care_daily"].isna()),
    ]
    _val = [1, 0]
    dat["any_care_daily_alt"] = np.select(_cond, _val, default=np.nan)

    # informal care from outside the household
    _cond = [
        (dat["sp021d10"] == 1) | (dat["sp002_"] == 1),
        (dat["sp021d20"].isna()) & (dat["sp002_"].isna()),
    ]
    _val = [1, np.nan]
    dat["informal_care_outside"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) & (dat["informal_care_general"] == 1),
        (dat["home_care"].isna()) & (dat["informal_care_general"].isna()),
    ]
    _val = [1, np.nan]
    dat["combination_care_general"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) & (dat["informal_care_daily"] == 1),
        (dat["home_care"].isna()) & (dat["informal_care_daily"].isna()),
    ]
    _val = [1, np.nan]
    dat["combination_care_daily"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) & (dat["informal_care_child"] == 1),
        (dat["home_care"].isna()) & (dat["informal_care_child"].isna()),
    ]
    _val = [1, np.nan]
    dat["combination_care_child"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) & (dat["informal_care_daily_child"] == 1),
        (dat["home_care"].isna()) & (dat["informal_care_daily_child"].isna()),
    ]
    _val = [1, np.nan]
    dat["combination_care_daily_child"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) | (dat["informal_care_general"] == 1),
        (dat["home_care"] == 0) & (dat["informal_care_general"] == 0),
    ]
    _val = [1, 0]
    dat["any_care"] = np.select(_cond, _val, default=np.nan)

    _cond = [
        (dat["home_care"] == 1) | (dat["informal_care_general"] == 1),
        (dat["home_care"].isna())
        & (dat["informal_care_general"].isna())
        & (dat["nursing_home"].isna()),
    ]
    _val = [1, np.nan]
    dat["any_care_no_nursing_home"] = np.select(_cond, _val, default=0)

    # lagged care
    dat = dat.sort_values(by=["mergeid", "int_year"], ascending=[True, True])

    _cond = [dat["informal_care_child"] == 1, dat["informal_care_child"] == 0]
    _val = [0, 1]
    dat["no_informal_care_child"] = np.select(_cond, _val, default=np.nan)

    _cond = [dat["home_care"] == 1, dat["home_care"] == 0]
    _val = [0, 1]
    dat["no_home_care"] = np.select(_cond, _val, default=np.nan)

    _cond = [dat["formal_care"] == 1, dat["formal_care"] == 0]
    _val = [0, 1]
    dat["no_formal_care"] = np.select(_cond, _val, default=np.nan)

    _cond = [dat["combination_care_general"] == 1, dat["combination_care_general"] == 0]
    _val = [0, 1]
    dat["no_combination_care"] = np.select(_cond, _val, default=np.nan)

    _cond = [
        (dat["informal_care_child"] == 1)
        | (dat["home_care"] == 1)
        | (dat["nursing_home"] == 1),
        (dat["informal_care_child"] == 0)
        & (dat["home_care"] == 0)
        & (dat["nursing_home"] == 0),
    ]
    _val = [0, 1]
    dat["no_care"] = np.select(_cond, _val, default=np.nan)

    # =============================================================================
    # Care mix shares with common denominator
    # =============================================================================

    # ────────────────────────────────────────────────────────────────
    # Pure-INFORMAL-care (no home care)
    #     • 1  → informal_care_general == 1  AND  home_care != 1
    #     • NA → both variables NA   (keeps your missing logic)
    #     • 0  → everyone else
    # ────────────────────────────────────────────────────────────────
    _cond = [
        (dat["informal_care_general"] == 1) & (dat["home_care"] != 1),
        (dat["informal_care_general"].isna()) & (dat["home_care"].isna()),
    ]
    _val = [1, np.nan]
    dat["pure_informal_care_general"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["informal_care_daily"] == 1) & (dat["home_care"] != 1),
        (dat["informal_care_daily"].isna()) & (dat["home_care"].isna()),
    ]
    _val = [1, np.nan]
    dat["pure_informal_care_daily"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["informal_care_child"] == 1) & (dat["home_care"] != 1),
        (dat["informal_care_child"].isna()) & (dat["home_care"].isna()),
    ]
    _val = [1, np.nan]
    dat["pure_informal_care_child"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["informal_care_daily_child"] == 1) & (dat["home_care"] != 1),
        (dat["informal_care_daily_child"].isna()) & (dat["home_care"].isna()),
    ]
    _val = [1, np.nan]
    dat["pure_informal_care_daily_child"] = np.select(_cond, _val, default=0)

    # ────────────────────────────────────────────────────────────────
    # Pure-HOME-care (no informal care)
    # ────────────────────────────────────────────────────────────────
    _cond = [
        (dat["home_care"] == 1) & (dat["informal_care_general"] != 1),
        (dat["home_care"].isna()) & (dat["informal_care_general"].isna()),
    ]
    _val = [1, np.nan]
    dat["pure_home_care_general"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) & (dat["informal_care_daily"] != 1),
        (dat["home_care"].isna()) & (dat["informal_care_daily"].isna()),
    ]
    _val = [1, np.nan]
    dat["pure_home_care_daily"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) & (dat["informal_care_child"] != 1),
        (dat["home_care"].isna()) & (dat["informal_care_child"].isna()),
    ]
    _val = [1, np.nan]
    dat["pure_home_care_child"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) & (dat["informal_care_daily_child"] != 1),
        (dat["home_care"].isna()) & (dat["informal_care_daily_child"].isna()),
    ]
    _val = [1, np.nan]
    dat["pure_home_care_daily_child"] = np.select(_cond, _val, default=0)

    # ────────────────────────────────────────────────────────────────
    # Shares that add up to 1  (care-recipient mix)
    #     Denominator = everyone who has ≥1 of the four care types
    # ────────────────────────────────────────────────────────────────

    flags = dat[CARE_COLS].fillna(0).astype(int)  # treat NA as 0

    receives_any = flags.any(axis=1)  # baseline subset
    denom = receives_any.sum()

    shares = (flags[receives_any] == 1).sum() / denom
    print(shares)
    # Example output (sums to 1):
    # pure_informal_care_general    0.65
    # pure_home_care                0.20
    # combination_care_general      0.13
    # nursing_home                  0.02
    # dtype: float64

    # rows with ≥1 of the four flags
    dat["receives_any_care"] = (
        dat[CARE_COLS].fillna(0).astype(int).any(axis=1).astype(int)
    )

    # simple age bands – edit to taste
    dat["age_group"] = pd.cut(
        dat["age"], bins=AGE_BINS_PARENTS, labels=AGE_LABELS_PARENTS, right=True
    )

    # ───────────────────────────────────────────────────────────────
    # 2.  Keep only care recipients, then compute shares
    #     • group by sex (sex == 1) and age_group
    #     • mean() of a 0/1 flag = share
    # ───────────────────────────────────────────────────────────────
    subset = dat[dat["receives_any_care"] == 1].copy()

    shares_by_grp = subset.groupby(["sex", "age_group"], observed=True)[
        CARE_COLS
    ].mean()  # observed=True → no empty rows  # mean of 0/1 flags = proportion

    print(shares_by_grp.head())

    # Survey weights

    # 1. baseline subset: only care recipients
    subset = dat.loc[
        dat[CARE_COLS].fillna(0).astype(int).any(axis=1),  # receives any care
        CARE_COLS + ["sex", "age", "hh_weight"],  # keep needed cols
    ].copy()

    # 2. add age bands (edit breaks to taste)
    # AGE_BINS_PARENTS = [0, 64, 74, 84, np.inf]
    subset["age_group"] = pd.cut(
        subset["age"],
        bins=AGE_BINS_PARENTS,
        labels=AGE_LABELS_PARENTS,
        right=True,
    )

    # 3. weighted shares: one custom lambda is enough
    def wmean(col):
        return (col * subset["hh_weight"]).sum() / subset["hh_weight"].sum()

    shares_w = subset.groupby(["sex", "age_group"], observed=True).apply(
        lambda g: (
            g[CARE_COLS].multiply(g["hh_weight"], axis=0).sum() / g["hh_weight"].sum()
        )
    )

    print(shares_w.head())

    _shares_w_hh = weighted_shares_and_counts(
        dat,
        care_cols=CARE_COLS,
        weight_col="hh_weight",
        group_cols=["sex", "age_group"],
        show_counts=True,
    )

    _shares_w_ind = weighted_shares_and_counts(
        dat,
        CARE_COLS,
        weight_col="ind_weight",
        group_cols=["sex", "age_group"],
    )
    _shares_w_design = weighted_shares_and_counts(
        dat, CARE_COLS, weight_col="design_weight", group_cols=["sex", "age_group"]
    )

    _shares_daily_w_hh = weighted_shares_and_counts(
        dat,
        care_cols=DAILY_CARE_COLS,
        weight_col="hh_weight",
        group_cols=["sex", "age_group"],
        show_counts=True,
    )
    _shares_daily_w_ind = weighted_shares_and_counts(
        dat, DAILY_CARE_COLS, weight_col="ind_weight", group_cols=["sex", "age_group"]
    )
    _shares_daily_w_design = weighted_shares_and_counts(
        dat,
        DAILY_CARE_COLS,
        weight_col="design_weight",
        group_cols=["sex", "age_group"],
    )

    _shares_child_w_hh = weighted_shares_and_counts(
        dat,
        care_cols=CHILD_CARE_COLS,
        weight_col="hh_weight",
        group_cols=["sex", "age_group"],
        show_counts=True,
    )
    _shares_child_w_ind = weighted_shares_and_counts(
        dat, CHILD_CARE_COLS, weight_col="ind_weight", group_cols=["sex", "age_group"]
    )
    _shares_child_w_design = weighted_shares_and_counts(
        dat,
        CHILD_CARE_COLS,
        weight_col="design_weight",
        group_cols=["sex", "age_group"],
    )

    _shares_daily_child_w_hh = weighted_shares_and_counts(
        dat,
        care_cols=DAILY_CHILD_CARE_COLS,
        weight_col="hh_weight",
        group_cols=["sex", "age_group"],
        show_counts=True,
    )
    _shares_daily_child_w_ind = weighted_shares_and_counts(
        dat,
        DAILY_CHILD_CARE_COLS,
        weight_col="ind_weight",
        group_cols=["sex", "age_group"],
    )
    _shares_daily_child_w_design = weighted_shares_and_counts(
        dat,
        DAILY_CHILD_CARE_COLS,
        weight_col="design_weight",
        group_cols=["sex", "age_group"],
    )

    dat = _create_lagged_var(dat, "no_care")
    dat = _create_lagged_var(dat, "home_care")
    dat = _create_lagged_var(dat, "formal_care")
    dat = _create_lagged_var(dat, "informal_care_general")
    dat = _create_lagged_var(dat, "informal_care_child")
    dat = _create_lagged_var(dat, "combination_care_general")
    dat = _create_lagged_var(dat, "any_care")
    dat = _create_lagged_var(dat, "no_informal_care_child")
    dat = _create_lagged_var(dat, "no_formal_care")
    dat = _create_lagged_var(dat, "no_combination_care")
    return _create_lagged_var(dat, "no_home_care")


def weighted_shares_and_counts(
    df: pd.DataFrame,
    care_cols,
    weight_col: str,
    group_cols,
    *,
    show_counts: bool = False,
    show_variances: bool = False,
) -> pd.DataFrame:
    """
    Compute weighted shares (and optionally counts & variances) of mutually
    exclusive 0/1 care-type flags within sub-groups.

    Parameters
    ----------
    df          : full DataFrame
    care_cols   : list[str]
                  column names of mutually-exclusive 0/1 flags
    weight_col  : str
                  name of the weight variable
    group_cols  : list[str]
                  columns to group by (e.g. ["sex", "age_band"])
    show_counts : bool, default False
                  include <flag>_count columns (raw # of 1s)
    show_vars   : bool, default False
                  include <flag>_var   columns (weighted variance)

    Returns
    -------
    DataFrame indexed by *group_cols* with at least
        <flag>_share  columns.
    Additional columns appear depending on the switches.
    """

    def _agg(group: pd.DataFrame) -> pd.Series:
        w = group[weight_col]
        w_sum = w.sum()
        w_sq_sum = (w**2).sum()
        n_eff = (w_sum**2) / w_sq_sum if w_sq_sum > 0 else np.nan

        out = {}

        for col in care_cols:
            share_num = (group[col] * w).sum()
            share = share_num / w_sum if w_sum > 0 else np.nan

            if not show_variances:
                out[f"share_{col}"] = share

            if show_counts:
                out[f"count_{col}"] = group[col].sum()

            if show_variances:
                var = share * (1 - share) / n_eff if n_eff > 0 else np.nan
                out[f"share_{col}"] = var

        return pd.Series(out)

    # keep only rows that receive any care (1 in at least one flag)
    mask_any = df[care_cols].fillna(0).astype(int).any(axis=1)

    return df.loc[mask_any].groupby(group_cols, observed=True).apply(_agg)


# def weighted_shares_and_counts(
#     df: pd.DataFrame, care_cols, weight_col: str, group_cols
# ):
#     """
#     Parameters
#     ----------
#     df          : the full DataFrame
#     care_cols   : list of 0/1 flags that are mutually exclusive
#     weight_col  : name of the survey-weight column to use
#     group_cols  : list of columns to group by (e.g. ["sex", "age_group"])

#     Returns
#     -------
#     DataFrame indexed by group_cols.
#     For every care flag `c` two columns are created:
#         c + "_share"  – weighted share (sums to 1 within a group)
#         c + "_count"  – un-weighted #rows with c == 1 in that group
#     """

#     def _agg(group):
#         w = group[weight_col]
#         weight_sum = w.sum()

#         result = {}
#         for c in care_cols:
#             num = (group[c] * w).sum()  # weighted numerator
#             share = num / weight_sum if weight_sum > 0 else np.nan
#             result[f"{c}_share"] = share
#             result[f"{c}_count"] = group[c].sum()  # simple count of 1-flags
#         return pd.Series(result)

#     # keep only rows that receive at least one type of care
#     mask_any = df[care_cols].fillna(0).astype(int).any(axis=1)
#     return df.loc[mask_any].groupby(group_cols, observed=True).apply(_agg)


def create_care_combinations(dat, informal_care_var):
    # 25.03.2024
    _cond = [
        (dat["formal_care"] == 0) & (dat[informal_care_var] == 1),
        (dat["formal_care"].isna()) & (dat[informal_care_var].isna()),
    ]
    _val = [1, np.nan]
    dat["informal_care_child_no_comb"] = np.select(_cond, _val, default=0)
    dat["only_informal"] = dat["informal_care_child_no_comb"].copy()

    _cond = [
        (dat["formal_care"] == 1) & (dat[informal_care_var] == 0),
        (dat["formal_care"].isna()) & (dat[informal_care_var].isna()),
    ]
    _val = [1, np.nan]
    dat["formal_care_no_comb"] = np.select(_cond, _val, default=0)
    dat["only_formal"] = dat["formal_care_no_comb"].copy()

    _cond = [
        (dat["home_care"] == 1) & (dat[informal_care_var] == 0),
        (dat["home_care"].isna()) & (dat[informal_care_var].isna()),
    ]
    _val = [1, np.nan]
    dat["only_home_care"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["nursing_home"] == 1)
        & (dat["home_care"] == 0)
        & (dat[informal_care_var] == 0),
        (dat["nursing_home"].isna())
        & (dat["home_care"].isna())
        & (dat[informal_care_var].isna()),
    ]
    _val = [1, np.nan]
    dat["only_nursing_home"] = np.select(_cond, _val, default=0)

    _cond = [dat["only_formal"] == 1, dat["only_formal"] == 0]
    _val = [0, 1]
    dat["no_only_formal"] = np.select(_cond, _val, default=np.nan)

    _cond = [dat["only_informal"] == 1, dat["only_informal"] == 0]
    _val = [0, 1]
    dat["no_only_informal"] = np.select(_cond, _val, default=np.nan)

    dat = _create_lagged_var(dat, "only_formal")
    dat = _create_lagged_var(dat, "only_informal")
    dat = _create_lagged_var(dat, "no_only_formal")
    return _create_lagged_var(dat, "no_only_informal")


def replace_negative_values_with_nan(dat, col):
    """Replace negative values with NaN."""
    dat[col] = np.where(dat[col] < 0, np.nan, dat[col])
    return dat


def create_means(dat):
    mean_home_care = dat.loc[dat["any_care"] == 1, "only_home_care"].mean()
    mean_combination_care = dat.loc[dat["any_care"] == 1, "combination_care"].mean()
    mean_informal_care = dat.loc[dat["any_care"] == 1, "only_informal"].mean()
    mean_formal_care = dat.loc[dat["any_care"] == 1, "only_formal"].mean()
    mean_nursing_home = dat.loc[dat["any_care"] == 1, "only_nursing_home"].mean()

    return (
        mean_home_care,
        mean_combination_care,
        mean_informal_care,
        mean_formal_care,
        mean_nursing_home,
    )


def create_married_or_partner_alive(dat):
    """Create married variable."""
    # We use marriage information in SHARE to construct an indicator on the
    # existence of a partner living in the same household.
    # We do not distinguish between marriage and registered partnership.
    # dn014_
    # Widowed

    conditions_married_or_partner = [
        dat["mstat"].isin([1, 2]),
        dat["mstat"].isin([3, 4, 5, 6]),
    ]
    values_married_or_partner = [1, 0]
    # replace with zeros or nans
    dat["married"] = np.select(
        conditions_married_or_partner,
        values_married_or_partner,
        np.nan,
    )

    return dat


def create_children_information(dat):
    """Create information on number of children (and daughters).

    # Handling the all-NaN case separately for both columns ch_gender_cols = [col for
    col in dat.columns if col.startswith("ch_gender_")]

    all_nan_indices = dat[ch_gender_cols].isna().all(axis=1) dat.loc[all_nan_indices,
    ["has_two_daughters", "has_two_children"]] = np.nan

    """
    dat["has_daughter"] = 0
    dat["has_two_daughters"] = 0  # Assuming less than two daughters by default
    dat["has_two_children"] = 0  # Assuming less than two children by default

    # Iterate through the DataFrame rows
    for index, row in dat.iterrows():
        # Counting non-NaN values for 'has_two_children'
        non_nan_count = row.filter(like="ch006_").notna().sum()

        # Counting values equal to 2 for 'has_two_daughters'
        female_count = (row.filter(like="ch005_") == FEMALE).sum()

        # Update 'has_two_children_loop' based on non-NaN count

        if non_nan_count >= AT_LEAST_TWO:
            dat.loc[index, "has_two_children"] = DUMMY_TRUE

        if female_count >= 1:
            dat.loc[index, "has_daughter"] = DUMMY_TRUE

        # Update 'has_two_daughters_loop' based on female count
        if female_count >= AT_LEAST_TWO:
            dat.loc[index, "has_two_daughters"] = DUMMY_TRUE

        if female_count < 1:
            dat.loc[index, "has_two_children"] = np.nan
            dat.loc[index, "has_two_daughters"] = np.nan

    return dat


def _create_lagged_var(dat, var):
    """Create lagged variable by mergeid."""
    dat[f"lagged_{var}"] = dat.groupby("mergeid")[var].shift(1)
    return dat


def _process_negative_values(dat):
    """Replace negative values with NaN."""
    columns_to_replace = [
        "hc029_",
        "hc031_",
        "hc032d1",
        "hc032d2",
        "hc032d3",
        "hc032dno",
        "hc033_",
        "hc034_",
        "hc035_",
        "hc036_",
        "hc127d1",
        "hc127d2",
        "hc127d3",
        "hc127d4",
        "hc127dno",
        "sp020_",
        "sp021d10",
        "sp021d11",
        "sp021d20",
        "sp021d21",
        "sp002_",
        "sp003_1",
        "sp003_2",
        "sp003_3",
    ]

    for col in columns_to_replace:
        dat[col] = np.where(dat[col] < 0, np.nan, dat[col])

    return dat


def _count_children_wave4(row) -> int:
    """Return the number (0-3) of sp003_* positions that are children in Wave 4."""
    cnt = 0
    for col in WHO_IS_CAREGIVER_EXTRA_HH:
        v = row[col]

        # direct code “child”
        if v == OTHER_CHILD_GAVE_HELP:
            cnt += 1

        # social-network person 1-7  →  child if sn005_x == 10
        elif SN_PERSON_ONE <= v <= SN_PERSON_SEVEN:
            sn_idx = v - 100  # 1 … 7
            if row[f"sn005_{sn_idx}"] == CHILD_ONE_GAVE_HELP:
                cnt += 1
    return cnt


def _has_child_in_wave4(row) -> bool:
    """Return True if at least one child in the household gave help (Wave 4 rules)."""
    # direct “child” code
    if row.get("sp021d19", 0) == 1:
        return True

    # social-network persons 1-7 (codes 1-7)
    for idx in range(1, 8):
        if (
            row.get(f"sp021d{idx}sn", 0) == 1  # person gave help
            and row.get(f"sn005_{idx}", None)
            == CHILD_ONE_GAVE_HELP  # that person is a child
        ):
            return True

    return False


def _count_children_inside_hh_wave4(row) -> int:
    """Return how many child caregivers (0-8) helped inside the HH in Wave 4."""
    cnt = 0

    # direct child
    if row.get("sp021d19", 0) == 1:
        cnt += 1

    # social-network 1-7
    for idx in range(1, 8):
        if (
            row.get(f"sp021d{idx}sn", 0) == 1
            and row.get(f"sn005_{idx}", None) == CHILD_ONE_GAVE_HELP
        ):
            cnt += 1

    return cnt
