"""Create SOEP moments and variances for MSM estimation."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    FULL_TIME,
    NOT_WORKING,
    PART_TIME,
    RETIREMENT,
    UNEMPLOYED,
)
from caregiving.specs.task_write_specs import read_and_derive_specs

DEGREES_OF_FREEDOM = 1


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def compute_labor_shares(subdf):
    """Compute labor shares and their variances for a given subsample."""
    total = len(subdf)

    if total == 0:
        return {
            "full_time": np.nan,
            "full_time_var": np.nan,
            "part_time": np.nan,
            "part_time_var": np.nan,
            "unemployed": np.nan,
            "unemployed_var": np.nan,
            "retired": np.nan,
            "retired_var": np.nan,
            "not_working": np.nan,
            "not_working_var": np.nan,
        }

    # Full time
    ind_full = subdf["choice"].isin(FULL_TIME.tolist()).astype(float)
    full_time = ind_full.mean()
    full_time_var = np.var(ind_full, ddof=DEGREES_OF_FREEDOM)

    # Part time
    ind_part = subdf["choice"].isin(PART_TIME.tolist()).astype(float)
    part_time = ind_part.mean()
    part_time_var = np.var(ind_part, ddof=DEGREES_OF_FREEDOM)

    # Unemployed
    ind_unemp = subdf["choice"].isin(UNEMPLOYED.tolist()).astype(float)
    unemployed = ind_unemp.mean()
    unemployed_var = np.var(ind_unemp, ddof=DEGREES_OF_FREEDOM)

    # Retired
    ind_ret = subdf["choice"].isin(RETIREMENT.tolist()).astype(float)
    retired = ind_ret.mean()
    retired_var = np.var(ind_ret, ddof=DEGREES_OF_FREEDOM)

    # Not working indicator: either unemployed or retired.
    ind_not_working = (
        subdf["choice"].isin(UNEMPLOYED.tolist() + RETIREMENT.tolist()).astype(float)
    )
    not_working = ind_not_working.mean()
    not_working_var = np.var(ind_not_working, ddof=DEGREES_OF_FREEDOM)

    return {
        "full_time": full_time,
        "full_time_var": full_time_var,
        "part_time": part_time,
        "part_time_var": part_time_var,
        "unemployed": unemployed,
        "unemployed_var": unemployed_var,
        "retired": retired,
        "retired_var": retired_var,
        "not_working": not_working,
        "not_working_var": not_working_var,
    }


def task_create_structural_estimation_sample(
    # path_to_specs: Path = SRC / "specs.yaml",
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "soep_moments.csv",
    path_to_save_variances: Annotated[Path, Product] = BLD
    / "moments"
    / "soep_variances.csv",
) -> None:
    """Create moments for MSM estimation."""

    # Load data
    df = pd.read_csv(path_to_sample)
    df = df[df["sex"] == 1]  # women only

    df["kidage_youngest"] = df["kidage_youngest"] - 1

    # Initialize a dictionary to store all moments
    moments = {}
    variances = {}

    # A) Moments by age.
    moments, variances = _get_labor_shares_by_age(df, moments, variances, 40, 69)

    # B) Moments by age and education.
    moments, variances = _get_labor_shares_by_age_and_education_level(
        df, moments, variances, 40, 69
    )

    # C) Moments by number of children and education.
    children_groups = {
        "0": lambda x: x == 0,
        "1": lambda x: x == 1,
        "2": lambda x: x == 2,  # noqa: PLR2004
        "3_plus": lambda x: x >= 3,  # noqa: PLR2004
    }
    moments, variances = (
        _get_labor_shares_by_education_level_and_number_of_children_in_hh(
            df, moments, variances, children_groups
        )
    )

    # D) Moments by kidage (youngest) and education.
    kidage_bins = {"0_3": (0, 3), "4_6": (4, 6), "7_9": (7, 9)}
    moments, variances = _get_labor_shares_by_education_level_and_age_of_youngest_child(
        df, moments, variances, kidage_bins
    )

    # E) Year-to-year labor supply transitions
    transition_matrix = pd.crosstab(
        df["lagged_choice"], df["choice"], normalize="index"
    )

    # Get raw counts (needed for variance calculation)
    transition_counts = pd.crosstab(df["lagged_choice"], df["choice"])
    transition_variance = (
        transition_matrix * (1 - transition_matrix) / transition_counts
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

            var_name = f"var_{moment_name}"
            variances[var_name] = transition_variance.loc[lag, current]

    # Create the final moments DataFrame/Series
    moments_df = pd.DataFrame({"value": pd.Series(moments)})
    moments_df.index.name = "moment"

    variances_df = pd.DataFrame({"value": pd.Series(variances)})
    variances_df.index.name = "moment"

    moments_df.to_csv(path_to_save_moments, index=True)
    variances_df.to_csv(path_to_save_variances, index=True)


# =====================================================================================
# Auxiliary functions
# =====================================================================================


def _get_labor_shares_by_age(df, moments, variances, min_age, max_age):

    for age in range(min_age, max_age + 1):
        subdf = df[df["age"] == age]
        shares = compute_labor_shares(subdf)

        moments[f"share_working_full_time_age_{age}"] = shares["full_time"]
        moments[f"share_working_part_time_age_{age}"] = shares["part_time"]
        moments[f"share_unemployed_age_{age}"] = shares["unemployed"]
        moments[f"share_retired_age_{age}"] = shares["retired"]
        moments[f"share_not_working_age_{age}"] = shares["not_working"]

        variances[f"share_working_full_time_age_{age}"] = shares["full_time_var"]
        variances[f"share_working_part_time_age_{age}"] = shares["part_time_var"]
        variances[f"share_unemployed_age_{age}"] = shares["unemployed_var"]
        variances[f"share_retired_age_{age}"] = shares["retired_var"]
        variances[f"share_not_working_age_{age}"] = shares["not_working_var"]

    return moments, variances


def _get_labor_shares_by_age_and_education_level(
    df, moments, variances, min_age, max_age
):
    for edu in (0, 1):
        edu_label = "low" if edu == 0 else "high"
        for age in range(min_age, max_age + 1):

            subdf = df[(df["age"] == age) & (df["education"] == edu)]
            shares = compute_labor_shares(subdf)
            key = f"age_{age}_edu_{edu_label}"

            moments[f"share_working_full_time_{key}"] = shares["full_time"]
            moments[f"share_working_part_time_{key}"] = shares["part_time"]
            moments[f"share_being_unemployed_{key}"] = shares["unemployed"]
            moments[f"share_being_retired_{key}"] = shares["retired"]
            moments[f"share_not_working_{key}"] = shares["not_working"]

            variances[f"share_working_full_time_{key}"] = shares["full_time_var"]
            variances[f"share_working_part_time_{key}"] = shares["part_time_var"]
            variances[f"share_being_unemployed_{key}"] = shares["unemployed_var"]
            variances[f"share_being_retired_{key}"] = shares["retired_var"]
            variances[f"share_not_working_{key}"] = shares["not_working_var"]

    return moments, variances


def _get_labor_shares_by_education_level_and_number_of_children_in_hh(
    df, moments, variances, children_groups
):
    for edu in (0, 1):
        edu_label = "low" if edu == 0 else "high"
        for grp_label, cond_func in children_groups.items():

            subdf = df[(df["education"] == edu) & (df["children"].apply(cond_func))]
            shares = compute_labor_shares(subdf)
            key = f"children_{grp_label}_edu_{edu_label}"

            moments[f"share_full_time_{key}"] = shares["full_time"]
            moments[f"share_part_time_{key}"] = shares["part_time"]
            moments[f"share_not_working_{key}"] = shares["not_working"]

            variances[f"share_full_time_{key}"] = shares["full_time_var"]
            variances[f"share_part_time_{key}"] = shares["part_time_var"]
            variances[f"share_not_working_{key}"] = shares["not_working_var"]

    return moments, variances


def _get_labor_shares_by_education_level_and_age_of_youngest_child(
    df, moments, variances, kidage_bins
):
    for edu in (0, 1):
        edu_label = "low" if edu == 0 else "high"
        for bin_label, (lb, ub) in kidage_bins.items():

            subdf = df[
                (df["education"] == edu)
                & (df["kidage_youngest"] >= lb)
                & (df["kidage_youngest"] <= ub)
            ]
            shares = compute_labor_shares(subdf)
            key = f"kidage_{bin_label}_edu_{edu_label}"

            moments[f"share_full_time_{key}"] = shares["full_time"]
            moments[f"share_part_time_{key}"] = shares["part_time"]
            moments[f"share_unemployed_{key}"] = shares["unemployed"]
            moments[f"share_retired_{key}"] = shares["retired"]

            variances[f"share_full_time_{key}"] = shares["full_time_var"]
            variances[f"share_part_time_{key}"] = shares["part_time_var"]
            variances[f"share_unemployed_{key}"] = shares["unemployed_var"]
            variances[f"share_retired_{key}"] = shares["retired_var"]

    return moments, variances


# =====================================================================================
# Bootstrap transition probabilities
# =====================================================================================


def compute_transition_moments(data, choice_map):
    """
    Compute transition probabilities (moments) from the data.
    Returns a dictionary with keys like 'transition_full_time_to_part_time'.
    """
    # Compute the transition matrix (normalized by row)
    transition_matrix = pd.crosstab(
        data["lagged_choice"], data["choice"], normalize="index"
    )

    # Store the transition probabilities with descriptive keys
    moments = {}
    for lag in transition_matrix.index:
        for current in transition_matrix.columns:
            from_state = choice_map.get(lag, lag)
            to_state = choice_map.get(current, current)
            moment_name = f"transition_{from_state}_to_{to_state}"
            moments[moment_name] = transition_matrix.loc[lag, current]

    return moments


def bootstrap_transition_variances(df, choice_map, n_bootstrap=1000):
    """
    Perform bootstrapping to estimate the variance of transition probabilities.

    Parameters:
      df         : The original DataFrame.
      n_bootstrap: Number of bootstrap replicates.

    Returns:
      A dictionary where keys are transition moment names
        (e.g., 'transition_full_time_to_part_time')
        and the values are the bootstrap variance estimates.

    """

    boot_estimates = {}

    # Run the bootstrap
    for _ in range(n_bootstrap):
        # Resample the dataframe with replacement
        boot_df = df.sample(frac=1, replace=True)

        # Compute moments from the bootstrap sample
        boot_moments = compute_transition_moments(boot_df, choice_map=choice_map)

        # Collect the bootstrapped estimates
        for key, value in boot_moments.items():
            if key not in boot_estimates:
                boot_estimates[key] = []
            boot_estimates[key].append(value)

    # Calculate the variance for each transition moment
    boot_variances = {
        key: np.var(values, ddof=1) for key, values in boot_estimates.items()
    }

    return boot_variances


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
