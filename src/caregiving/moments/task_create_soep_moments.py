"""Create moments for MSM estimation."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import (
    create_lagged_and_lead_variables,
    enforce_model_choice_restriction,
    filter_data,
)
from caregiving.data_management.soep.variables import (
    create_choice_variable,
    create_education_type,
    create_experience_variable,
    create_kidage_youngest,
    create_partner_state,
    determine_observed_job_offers,
    generate_job_separation_var,
)
from caregiving.model.shared import (
    FULL_TIME,
    NOT_WORKING,
    PART_TIME,
    RETIREMENT,
    UNEMPLOYED,
    WORK,
)
from caregiving.specs.task_write_specs import read_and_derive_specs


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def compute_labor_shares(subdf):
    """Returns shares for full_time, part_time, unemployed, retired and
    not_working (unemployed or retired)
    from a sub-dataframe.

    """
    total = len(subdf)
    if total == 0:
        return {
            "full_time": np.nan,
            "part_time": np.nan,
            "unemployed": np.nan,
            "retired": np.nan,
            "not_working": np.nan,
        }

    full_time = subdf["choice"].isin(FULL_TIME.tolist()).mean()
    part_time = subdf["choice"].isin(PART_TIME.tolist()).mean()
    unemployed = subdf["choice"].isin(UNEMPLOYED.tolist()).mean()
    retired = subdf["choice"].isin(RETIREMENT.tolist()).mean()
    not_working = unemployed + retired

    return {
        "full_time": full_time,
        "part_time": part_time,
        "unemployed": unemployed,
        "retired": retired,
        "not_working": not_working,
    }


def task_create_structural_estimation_sample(
    # path_to_specs: Path = SRC / "specs.yaml",
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save: Annotated[Path, Product] = BLD / "moments" / "soep_moments.csv",
) -> None:
    """Create moments for MSM estimation."""

    # Load data
    df = pd.read_csv(path_to_sample)
    df = df[df["sex"] == 1]  # women only

    # Initialize a dictionary to store all moments
    moments = {}

    # ---------------------------
    # A) Moments by age (40 to 69)
    # ---------------------------
    for age in range(40, 70):
        subdf = df[df["age"] == age]
        shares = compute_labor_shares(subdf)
        moments[f"share_working_full_time_age_{age}"] = shares["full_time"]
        moments[f"share_working_part_time_age_{age}"] = shares["part_time"]
        moments[f"share_unemployed_age_{age}"] = shares["unemployed"]
        moments[f"share_retired_age_{age}"] = shares["retired"]
        moments[f"share_not_working_age_{age}"] = shares["not_working"]

    # ------------------------------------------------
    # B) Moments by age conditional on education type
    #     (assuming education: 0 = low, 1 = high)
    # ------------------------------------------------
    for edu in (0, 1):
        edu_label = "low" if edu == 0 else "high"
        for age in range(40, 70):

            subdf = df[(df["age"] == age) & (df["education"] == edu)]
            shares = compute_labor_shares(subdf)

            moments[f"share_working_full_time_age_{age}_edu_{edu_label}"] = shares[
                "full_time"
            ]
            moments[f"share_working_part_time_age_{age}_edu_{edu_label}"] = shares[
                "part_time"
            ]
            moments[f"share_being_unemployed_age_{age}_edu_{edu_label}"] = shares[
                "unemployed"
            ]
            moments[f"share_being_retired_age_{age}_edu_{edu_label}"] = shares[
                "retired"
            ]
            moments[f"share_not_working_age_{age}_edu_{edu_label}"] = shares[
                "not_working"
            ]

    # -----------------------------------------------------------
    # C) Moments by number of children (0, 1, 2, 3 or more) and education type
    # -----------------------------------------------------------
    children_groups = {
        "0": lambda x: x == 0,
        "1": lambda x: x == 1,
        "2": lambda x: x == 2,  # noqa: PLR2004
        "3_plus": lambda x: x >= 3,  # noqa: PLR2004
    }

    for edu in (0, 1):
        edu_label = "low" if edu == 0 else "high"
        for group_label, condition in children_groups.items():
            subdf = df[(df["education"] == edu) & (df["children"].apply(condition))]
            shares = compute_labor_shares(subdf)
            moments[f"share_full_time_children_{group_label}_edu_{edu_label}"] = shares[
                "full_time"
            ]
            moments[f"share_part_time_children_{group_label}_edu_{edu_label}"] = shares[
                "part_time"
            ]
            moments[f"share_not_working_children_{group_label}_edu_{edu_label}"] = (
                shares["not_working"]
            )

    # ---------------------------------------------------------------------
    # D) Moments by age of the youngest child (kidage_youngest) and education type
    #     (D1: bins 0-3, 4-6, 7-9; D2: aggregated for ages 0 through 9)
    # ---------------------------------------------------------------------
    kidage_bins = {"0_3": (0, 3), "4_6": (4, 6), "7_9": (7, 9)}

    for edu in (0, 1):
        edu_label = "low" if edu == 0 else "high"
        for bin_label, (lb, ub) in kidage_bins.items():
            subdf = df[
                (df["education"] == edu)
                & (df["kidage_youngest"] >= lb)
                & (df["kidage_youngest"] <= ub)
            ]
            shares = compute_labor_shares(subdf)
            moments[f"share_full_time_kidage_{bin_label}_edu_{edu_label}"] = shares[
                "full_time"
            ]
            moments[f"share_part_time_kidage_{bin_label}_edu_{edu_label}"] = shares[
                "part_time"
            ]
            moments[f"share_unemployed_kidage_{bin_label}_edu_{edu_label}"] = shares[
                "unemployed"
            ]
            moments[f"share_retired_kidage_{bin_label}_edu_{edu_label}"] = shares[
                "retired"
            ]

    # -------------------------------------------------------------------------
    # E) Year-to-year labor supply transitions (lagged_choice -> current choice)
    # -------------------------------------------------------------------------
    transition_matrix = pd.crosstab(
        df["lagged_choice"], df["choice"], normalize="index"
    )

    # Build a mapping dictionary using the imported jax arrays
    choice_map = {code: "full_time" for code in FULL_TIME.tolist()}
    choice_map.update({code: "part_time" for code in PART_TIME.tolist()})
    choice_map.update({code: "not_working" for code in NOT_WORKING.tolist()})

    for lag in transition_matrix.index:
        for current in transition_matrix.columns:
            from_state = choice_map.get(lag, lag)
            to_state = choice_map.get(current, current)
            moment_name = f"transition_{from_state}_to_{to_state}"
            moments[moment_name] = transition_matrix.loc[lag, current]

    # ------------------------------------------
    # Create the final moments DataFrame/Series
    # ------------------------------------------
    moments_df = pd.DataFrame({"value": pd.Series(moments)})
    moments_df.index.name = "moment"

    moments_df.to_csv(path_to_save, index=True)
