"""Exognenous care sample."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD

SOEP_IS_ANSWER_NO = 2


def task_create_exog_care_sample(
    path_to_raw_data: Path = BLD / "data" / "soep_is_exog_care_data_raw.csv",
    path_to_save: Annotated[Path, Product] = (
        BLD / "data" / "exog_care_estimation_sample.pkl"
    ),
):

    df = pd.read_csv(path_to_raw_data, index_col=["pid"])

    # Initialize dummies with -1
    df["other_informal_care"] = 0
    df["formal_care"] = 0

    # Assign 1 if conditions are met
    df.loc[
        (df["ip05"].isin([1, 2]))  # lives in private household
        & ((df["ip08a1"] == 1) | (df["ip08a4"] == 1)),
        "other_informal_care",
    ] = 1

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
        "ip03",
        "other_informal_care",
        "formal_care",
        "only_own_informal_care",
        "only_other_informal_care",
        "only_formal_care",
    ]

    df_to_save = df[cols_to_keep]
    df_to_save.to_pickle(path_to_save)
