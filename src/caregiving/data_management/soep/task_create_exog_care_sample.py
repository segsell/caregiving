"""Exognenous care sample."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD

SOEP_IS_ANSWER_NO = 2
MOTHER_OR_FATHER = 2
SOEP_IS_MALE = 1
SOEP_IS_FEMALE = 2


def task_create_exog_care_sample(
    path_to_raw_data: Path = BLD / "data" / "soep_is_exog_care_data_raw.csv",
    path_to_save: Annotated[Path, Product] = (
        BLD / "data" / "exog_care_estimation_sample.pkl"
    ),
):

    df = pd.read_csv(path_to_raw_data, index_col=["pid"])

    df["other_informal_care"] = 0
    df["_other_informal_care"] = 0
    df["formal_care"] = 0

    # mask_parent_care = (
    #     ((df["ip02"] == MOTHER_OR_FATHER) & (df["ip03"] == SOEP_IS_FEMALE))
    #     | ((df["ip02_2"] == MOTHER_OR_FATHER) & (df["ip03_2"] == SOEP_IS_FEMALE))
    #     | ((df["ip02_3"] == MOTHER_OR_FATHER) & (df["ip03_3"] == SOEP_IS_FEMALE))
    #     | ((df["ip02_4"] == MOTHER_OR_FATHER) & (df["ip03_4"] == SOEP_IS_FEMALE))
    # )
    mask_parent_care = (
        (df["ip02"] == MOTHER_OR_FATHER)
        | (df["ip02_2"] == MOTHER_OR_FATHER)
        | (df["ip02_3"] == MOTHER_OR_FATHER)
        | (df["ip02_4"] == MOTHER_OR_FATHER)
    )
    df["parent_care_demand"] = np.where(mask_parent_care, 1, 0)

    # Assign 1 if conditions are met
    df.loc[
        (df["ip05"].isin([1, 2]))  # lives in private household
        & ((df["ip08a1"] == 1) | (df["ip08a4"] == 1)),
        "_other_informal_care",
    ] = 1

    person_1 = (
        df["ip05"].isin([1, 2])
        & (df["ip02"] == MOTHER_OR_FATHER)
        & ((df["ip08a1"] == 1) | (df["ip08a4"] == 1))
    )
    person_2 = (
        df["ip05_2"].isin([1, 2])
        & (df["ip02_2"] == MOTHER_OR_FATHER)
        & ((df["ip08a1_2"] == 1) | (df["ip08a4_2"] == 1))
    )
    person_3 = (
        df["ip05_3"].isin([1, 2])
        & (df["ip02_3"] == MOTHER_OR_FATHER)
        & ((df["ip08a1_3"] == 1) | (df["ip08a4_3"] == 1))
    )
    person_4 = (
        df["ip05_4"].isin([1, 2])
        & (df["ip02_4"] == MOTHER_OR_FATHER)
        & ((df["ip08a1_4"] == 1) | (df["ip08a4_4"] == 1))
    )
    df["other_informal_care"] = np.select(
        condlist=[person_1, person_2, person_3, person_4],
        choicelist=[1, 1, 1, 1],
        default=0,
    )

    df.loc[
        (df["ip08a2"] == 1) | (df["ip08a3"] == 1) | (df["ip08a5"] == 1),
        "formal_care",
    ] = 1

    df["only_own_informal_care"] = 0
    df["only_other_informal_care"] = 0
    df["only_formal_care"] = 0

    # Assign 1 where conditions are met
    df.loc[
        (df["ip06"] == 1) & (df["ip05"].isin([1, 2]))  # lives in private household
        # & (merged_data["ip07w"] >= 1)
        & (df["other_informal_care"] == 0),
        # & (merged_data["formal_care"] == 0),
        "only_own_informal_care",
    ] = 1

    df.loc[
        # ((merged_data["ip06"] == 2) | (merged_data["ip07w"] < 1))
        (df["ip06"] == SOEP_IS_ANSWER_NO) & (df["other_informal_care"] == 1),
        # & (merged_data["formal_care"] == 0),
        "only_other_informal_care",
    ] = 1

    # only_other_informal_care
    # 0    622
    # 1    320

    df.loc[
        (df["ip06"] == SOEP_IS_ANSWER_NO)  # No
        # & (merged_data["other_informal_care"] == 0)
        & (df["formal_care"] == 1),
        "only_formal_care",
    ] = 1

    df["has_sister"] = np.nan  # start with NaN (or use np.nan)
    df.loc[df["l0063"] > 0, "has_sister"] = 1
    df.loc[df["l0063"].isin([0, -2]), "has_sister"] = 0

    # Replace negative values with NaN for summing purposes
    l0062_clean = df["l0062"].where(df["l0062"] >= 0)
    l0063_clean = df["l0063"].where(df["l0063"] >= 0)

    # Calculate n_siblings as sum of non-negative values
    df["n_siblings"] = l0062_clean + l0063_clean

    cols_to_keep = [
        "female",
        "age",
        "education",
        "has_sister",
        "n_siblings",
        "parent_care_demand",
        "other_informal_care",
        "formal_care",
        "only_own_informal_care",
        "only_other_informal_care",
        "only_formal_care",
    ]

    df_to_save = df[cols_to_keep]
    df_to_save.to_pickle(path_to_save)
