"""Create moments for MSM estimation."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
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

    df["kidage_youngest"] = df["kidage_youngest"] - 1

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
    #     Age bins 0-3, 4-6, 7-9
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

    # breakpoint()

    moments_df.to_csv(path_to_save, index=True)


# =====================================================================================
# Plotting
# =====================================================================================


def plot_labor_shares_by_age(
    df, age_var, start_age, end_age, condition_col=None, condition_val=None
):
    """
    Plots labor share outcomes by age using a specified age variable.
    Outcomes include full time, part time, unemployed, retired, and
    not working (unemployed + retired).

    Parameters:
      df (pd.DataFrame): The data frame containing the data.
      age_var (str): Column to use as the age variable (e.g., "age" or
          "kidage_youngest").
      start_age (int): Starting age (inclusive) for the plot.
      end_age (int): Ending age (inclusive) for the plot.
      condition_col (str or list, optional): Column name(s) to filter data.
      condition_val (any or list, optional): Value(s) for filtering.
          If multiple, supply a list/tuple that matches condition_col.
    """
    # Apply conditioning if specified.
    if condition_col is not None:
        if isinstance(condition_col, (list, tuple)):
            if not isinstance(condition_val, (list, tuple)) or (
                len(condition_col) != len(condition_val)
            ):
                raise ValueError(
                    "When condition_col is a list/tuple, condition_val must be a "
                    "list/tuple of the same length."
                )
            for col, val in zip(condition_col, condition_val, strict=False):
                df = df[df[col] == val]
        else:
            df = df[df[condition_col] == condition_val]

    # Filter on the chosen age variable.
    df = df[(df[age_var] >= start_age) & (df[age_var] <= end_age)]

    ages = list(range(start_age, end_age + 1))
    full_time_shares = []
    part_time_shares = []
    not_working_shares = []
    # unemployed_shares = []
    # retired_shares = []

    # Loop over each age.
    for age in ages:
        subdf = df[df[age_var] == age]
        shares = compute_labor_shares(subdf)
        full_time_shares.append(shares["full_time"])
        part_time_shares.append(shares["part_time"])
        not_working_shares.append(shares["not_working"])
        # unemployed_shares.append(shares["unemployed"])
        # retired_shares.append(shares["retired"])

    plt.figure(figsize=(10, 6))
    plt.plot(ages, full_time_shares, marker="o", label="Full Time")
    plt.plot(ages, part_time_shares, marker="o", label="Part Time")
    plt.plot(ages, not_working_shares, marker="o", label="Not Working")
    # plt.plot(ages, unemployed_shares, marker="o", label="Unemployed")
    # plt.plot(ages, retired_shares, marker="o", label="Retired")

    plt.xlabel(age_var.title())
    plt.ylabel("Share")
    title = (
        "Labor Shares by "
        + age_var.title()
        + " (Ages "
        + str(start_age)
        + " to "
        + str(end_age)
        + ")"
    )
    if condition_col is not None:
        if isinstance(condition_col, (list, tuple)):
            conditions = ", ".join(
                [
                    f"{col}={val}"
                    for col, val in zip(condition_col, condition_val, strict=False)
                ]
            )
        else:
            conditions = f"{condition_col}={condition_val}"
        title += " | Conditions: " + conditions
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
