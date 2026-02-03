"""Exognenous care sample."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import CARE_DEMAND_INTENSIVE, CARE_DEMAND_LIGHT

SOEP_IS_ANSWER_NO = 2
MOTHER_OR_FATHER = 2
SOEP_IS_MALE = 1
SOEP_IS_FEMALE = 2
SOEP_IS_NURSING_HOME = 3  # ip05 == 3 means person lives in nursing home


@pytask.mark.soep_is
def task_create_care_intensity_sample(
    path_to_raw_data: Path = BLD / "data" / "soep_is_care_raw.csv",
    path_to_save: Annotated[Path, Product] = (BLD / "data" / "soep_is_sample.csv"),
) -> None:
    """Create care intensity variables from SOEP-IS care data.

    Each pid can have up to 4 persons in need of care. For each person,
    create a care_intensity dummy variable:
    - 0 if care level is 0 or 1 (values 1 or 2 in ip04a)
    - 1 if care level is 2 or 3 (values 3 or 4 in ip04a)
    - NaN for other values (negative codes, don't know, etc.)

    """
    df = pd.read_csv(path_to_raw_data, index_col=["pid"])

    # Define care level variables for each of the 4 persons
    care_level_vars = ["ip04a", "ip04a_2", "ip04a_3", "ip04a_4"]
    care_intensity_vars = [
        "care_intensity",
        "care_intensity_2",
        "care_intensity_3",
        "care_intensity_4",
    ]

    # Create care_intensity for each person
    for care_level_var, care_intensity_var in zip(
        care_level_vars, care_intensity_vars, strict=True
    ):
        if care_level_var not in df.columns:
            # If column doesn't exist, create NaN column
            df[care_intensity_var] = np.nan
            continue

        # Initialize with NaN
        df[care_intensity_var] = np.nan

        # 0 if care level is 0 or 1 (values 1 or 2)
        mask_low = df[care_level_var].isin([2])
        df.loc[mask_low, care_intensity_var] = 0

        # 1 if care level is 2 or 3 (values 3 or 4)
        mask_high = df[care_level_var].isin([3, 4])
        df.loc[mask_high, care_intensity_var] = 1

        # All other values remain NaN (negative codes, don't know, etc.)

    # Create pure_formal_care variables for each of the 4 persons
    # This identifies cases where a parent is cared for exclusively by
    # formal care providers (nursing home or formal care services)
    pure_formal_care_vars = [
        "pure_formal_care",
        "pure_formal_care_2",
        "pure_formal_care_3",
        "pure_formal_care_4",
    ]
    # Variable suffixes for persons 2-4
    suffixes = ["", "_2", "_3", "_4"]

    for suffix, pure_formal_var in zip(suffixes, pure_formal_care_vars, strict=True):
        # Initialize with 0
        df[pure_formal_var] = 0

        # Get variable names for this person
        ip02_var = f"ip02{suffix}" if suffix else "ip02"
        ip03_var = f"ip03{suffix}" if suffix else "ip03"
        ip05_var = f"ip05{suffix}" if suffix else "ip05"
        ip06_var = f"ip06{suffix}" if suffix else "ip06"
        ip08_var = f"ip08{suffix}" if suffix else "ip08"
        ip08wn_var = f"ip08wn{suffix}" if suffix else "ip08wn"

        # Only create for mothers (parent AND female)
        is_parent = df[ip02_var] == MOTHER_OR_FATHER
        is_mother = df[ip03_var] == SOEP_IS_FEMALE
        is_mother_parent = is_parent & is_mother

        # Condition 1: Person lives in nursing home (ip05 == 3)
        condition_nursing_home = df[ip05_var] == SOEP_IS_NURSING_HOME

        # Condition 2: No informal care from respondent (ip06 == 2)
        # AND receives additional care (ip08 == 1)
        # AND at least one formal care provider is involved
        # (ip08a2 == 1 OR ip08a3 == 1 OR ip08a5 == 1)
        condition_formal_care_at_home = (
            (df[ip06_var] == SOEP_IS_ANSWER_NO)
            & (df[ip08_var] == 1)
            & (df[ip08wn_var] == 1)
            # & ((df[ip08a2_var] == 1) | (df[ip08a3_var] == 1) | (df[ip08a5_var] == 1))
        )

        # Set to 1 if either condition is met AND person is a mother (parent AND female)
        mask_pure_formal = is_mother_parent & (
            condition_nursing_home | condition_formal_care_at_home
        )
        df.loc[mask_pure_formal, pure_formal_var] = 1

    df.to_csv(path_to_save)


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
