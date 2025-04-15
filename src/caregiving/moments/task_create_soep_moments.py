"""Create SOEP moments and variances for MSM estimation."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
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


def task_create_soep_moments(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "soep_moments.csv",
    path_to_save_variances: Annotated[Path, Product] = BLD
    / "moments"
    / "soep_variances.csv",
) -> None:
    """Create moments for MSM estimation."""

    specs = read_and_derive_specs(path_to_specs)
    start_age = specs["start_age"]
    end_age = specs["end_age_msm"]

    df = pd.read_csv(path_to_sample)
    df = df[(df["sex"] == 1) & (df["age"] <= end_age)]  # women only

    df["kidage_youngest"] = df["kidage_youngest"] - 1

    # Initialize a dictionary to store all moments
    moments = {}
    variances = {}

    # =========================================================

    age_range = range(start_age, end_age + 1)
    age_groups = df.groupby("age")

    # Compute the proportion for each status using vectorized operations
    retired_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(RETIREMENT)).mean()
    )
    unemployed_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(UNEMPLOYED)).mean()
    )
    part_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(PART_TIME)).mean()
    )
    full_time_shares = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(FULL_TIME)).mean()
    )

    retired_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(RETIREMENT)).var(ddof=DEGREES_OF_FREEDOM)
    )
    unemployed_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(UNEMPLOYED)).var(ddof=DEGREES_OF_FREEDOM)
    )
    part_time_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(PART_TIME)).var(ddof=DEGREES_OF_FREEDOM)
    )
    full_time_vars = age_groups["choice"].apply(
        lambda x: x.isin(np.atleast_1d(FULL_TIME)).var(ddof=DEGREES_OF_FREEDOM)
    )

    # Reindex to ensure that every age between start_age and end_age is included;
    # missing ages will be filled with NaN
    retired_shares = retired_shares.reindex(age_range, fill_value=np.nan)
    unemployed_shares = unemployed_shares.reindex(age_range, fill_value=np.nan)
    part_time_shares = part_time_shares.reindex(age_range, fill_value=np.nan)
    full_time_shares = full_time_shares.reindex(age_range, fill_value=np.nan)

    retired_vars = retired_vars.reindex(age_range, fill_value=np.nan)
    unemployed_vars = unemployed_vars.reindex(age_range, fill_value=np.nan)
    part_time_vars = part_time_vars.reindex(age_range, fill_value=np.nan)
    full_time_vars = full_time_vars.reindex(age_range, fill_value=np.nan)

    # Populate the moments dictionary for age-specific shares
    for age in age_range:
        moments[f"share_retired_age_{age}"] = retired_shares.loc[age]
        variances[f"share_retired_age_{age}"] = retired_vars.loc[age]

    for age in age_range:
        moments[f"share_unemployed_age_{age}"] = unemployed_shares.loc[age]
        variances[f"share_unemployed_age_{age}"] = unemployed_vars.loc[age]

    for age in age_range:
        moments[f"share_part_time_age_{age}"] = part_time_shares.loc[age]
        variances[f"share_part_time_age_{age}"] = part_time_vars.loc[age]

    for age in age_range:
        moments[f"share_full_time_age_{age}"] = full_time_shares.loc[age]
        variances[f"share_full_time_age_{age}"] = full_time_vars.loc[age]

    # =========================================================

    # # A) Moments by age.
    # moments, variances = _get_labor_shares_by_age(
    #     df, moments, variances, start_age, end_age
    # )

    # # B) Moments by age and education.
    # moments, variances = _get_labor_shares_by_age_and_education_level(
    #     df, moments, variances, start_age, end_age
    # )

    # C) Moments by number of children and education.
    # children_groups = {
    #     "0": lambda x: x == 0,
    #     "1": lambda x: x == 1,
    #     "2": lambda x: x == 2,  # noqa: PLR2004
    #     "3_plus": lambda x: x >= 3,  # noqa: PLR2004
    # }
    # moments, variances = (
    #     _get_labor_shares_by_education_level_and_number_of_children_in_hh(
    #         df, moments, variances, children_groups
    #     )
    # )

    # D) Moments by kidage (youngest) and education.
    # kidage_bins = {"0_3": (0, 3), "4_6": (4, 6), "7_9": (7, 9)}
    # moments, variances = _get_labor_shares_by_educ_level_and_age_of_youngest_child(
    #     df, moments, variances, kidage_bins
    # )

    # E) Year-to-year labor supply transitions

    transition_moments, transition_variances = compute_transition_moments_and_variances(
        df, full_time=FULL_TIME, part_time=PART_TIME, not_working=NOT_WORKING
    )
    moments.update(transition_moments)
    variances.update(transition_variances)

    # F) Wealth moments by age and education (NEW)
    # wealth_moments_edu_low, wealth_variances_edu_low = (
    #     compute_wealth_moments_by_five_year_age_bins(
    #         df, edu_level=0, start_age=start_age, end_age=end_age
    #     )
    # )
    # moments.update(wealth_moments_edu_low)
    # variances.update(wealth_variances_edu_low)

    # wealth_moments_edu_high, wealth_variances_edu_high = (
    #     compute_wealth_moments_by_five_year_age_bins(
    #         df, edu_level=1, start_age=start_age, end_age=end_age
    #     )
    # )
    # moments.update(wealth_moments_edu_high)
    # variances.update(wealth_variances_edu_high)

    # plot_wealth_by_age(df, start_age=30, end_age=70, educ_val=1)
    # plot_wealth_by_5yr_bins(df, start_age=30, end_age=70, educ_val=1)

    # Create the final moments DataFrame/Series
    moments_df = pd.DataFrame({"value": pd.Series(moments)})
    moments_df.index.name = "moment"

    variances_df = pd.DataFrame({"value": pd.Series(variances)})
    variances_df.index.name = "moment"

    moments_df.to_csv(path_to_save_moments, index=True)
    variances_df.to_csv(path_to_save_variances, index=True)


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


def compute_transition_moments_and_variances(
    df,
    full_time,
    part_time,
    not_working,
    choice="choice",
    lagged_choice="lagged_choice",
):
    """
    Compute year-to-year labor supply transition moments and their variances.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns specified by lagged_col and current_col.
    FULL_TIME : array-like
        Codes representing full-time in the 'choice' columns.
    PART_TIME : array-like
        Codes representing part-time in the 'choice' columns.
    NOT_WORKING : array-like
        Codes representing not-working states in the 'choice' columns.
    lagged_col : str, default "lagged_choice"
        The column in df representing last year's labor supply choice.
    current_col : str, default "choice"
        The column in df representing current year's labor supply choice.

    Returns
    -------
    moments : dict
        Keys like 'transition_full_time_to_part_time',
        values are transition probabilities.
    variances : dict
        Keys like 'var_transition_full_time_to_part_time',
        values are the corresponding variances of those probabilities.
    """

    # 1) Transition matrix (row-normalized by lagged_col)
    transition_matrix = pd.crosstab(df[lagged_choice], df[choice], normalize="index")

    # 2) Raw counts (needed for variance calculation)
    transition_counts = pd.crosstab(df[lagged_choice], df[choice])

    # Variance formula: p * (1 - p) / N
    transition_variance = (
        transition_matrix * (1 - transition_matrix)
    ) / transition_counts

    # 3) Build a mapping dictionary to map numeric codes to textual labels
    choice_map = {code: "full_time" for code in full_time.tolist()}
    choice_map.update({code: "part_time" for code in part_time.tolist()})
    choice_map.update({code: "not_working" for code in not_working.tolist()})

    # 4) Loop over all lag/current combinations to populate the dictionaries
    moments = {}
    variances = {}

    for lag_val in transition_matrix.index:
        for current_val in transition_matrix.columns:
            from_state = choice_map.get(lag_val, lag_val)
            to_state = choice_map.get(current_val, current_val)

            # e.g. 'transition_full_time_to_part_time'
            moment_name = f"transition_{from_state}_to_{to_state}"
            moments[moment_name] = transition_matrix.loc[lag_val, current_val]

            # e.g. 'var_transition_full_time_to_part_time'
            var_name = f"var_{moment_name}"
            variances[var_name] = transition_variance.loc[lag_val, current_val]

    return moments, variances


def compute_transition_moments_by_five_year_age_bins(
    df,
    full_time,
    part_time,
    not_working,
    start_age,
    end_age,
    choice="choice",
    lagged_choice="lagged_choice",
):
    """
    Compute year-to-year labor supply transition moments and variances
    by 5-year age bin, e.g. [30,35), [35,40), etc.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'age', plus the columns in 'choice' and 'lagged_choice'.
    full_time : array-like
        Codes representing full-time in the 'choice' columns.
    part_time : array-like
        Codes representing part-time in the 'choice' columns.
    not_working : array-like
        Codes representing not-working states in the 'choice' columns.
    start_age : int
        Lower bound (inclusive) of the age range.
    end_age : int
        Upper bound (inclusive) of the age range.
    choice : str, default "choice"
        Column name for current year's labor supply choice.
    lagged_choice : str, default "lagged_choice"
        Column name for last year's labor supply choice.

    Returns
    -------
    moments : dict
        A dictionary of transition probabilities for each 5-year bin.
        Keys look like: "transition_full_time_to_part_time_[30,35)".
    variances : dict
        Corresponding variances, with keys like:
        "var_transition_full_time_to_part_time_[30,35)".
    """

    # --- 1) Define the 5-year bins ---
    # If start_age=30, end_age=42 => bins = [30, 35, 40, 45]
    # => intervals: [30,35), [35,40), [40,45)
    bins = list(range(start_age, end_age + 1, 5))
    if bins[-1] < end_age:
        bins.append(end_age + 1)  # Ensure we cover up to end_age
    # Create labels like "[30,35)", "[35,40)", etc.
    bin_labels = [f"age_{bins[i]}_{bins[i+1] - 1}" for i in range(len(bins) - 1)]

    # Prepare storage for aggregated results
    moments = {}
    variances = {}

    # --- 2) Loop over each bin, filter df, compute transitions ---
    for i in range(len(bin_labels)):
        bin_label = bin_labels[i]
        age_lower = bins[i]
        age_upper = bins[i + 1]

        # Filter rows whose age is in [age_lower, age_upper)
        subdf = df[(df["age"] >= age_lower) & (df["age"] < age_upper)].copy()

        # If there's no data in that bin, skip it
        if subdf.empty:
            continue

        # --- 3) Compute transitions for the subsample ---
        sub_moments, sub_variances = compute_transition_moments_and_variances(
            subdf,
            full_time=full_time,
            part_time=part_time,
            not_working=not_working,
            choice=choice,
            lagged_choice=lagged_choice,
        )

        # --- 4) Append bin label to the keys and store ---
        for k, v in sub_moments.items():
            new_key = f"{k}_{bin_label}"
            moments[new_key] = v

        for k, v in sub_variances.items():
            new_key = f"{k}_{bin_label}"
            variances[new_key] = v

    return moments, variances


def compute_wealth_moments_by_five_year_age_bins(df, edu_level, start_age, end_age):
    """
    Compute wealth moments (mean and variance) for 5-year age bins
    between start_age and end_age.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'age' and 'wealth'.
    edu_level : int
        Education level to filter on (0=low, 1=high).
    start_age : int
        Lower bound (inclusive) of the age range.
    end_age : int
        Upper bound (inclusive) of the age range.

    Returns
    -------
    moments : dict
        Dictionary of mean wealth by bin, with keys like 'wealth_[30,35)'.
    variances : dict
        Dictionary of variance of wealth by bin, matching the same keys.
    """

    # 1) Restrict to specified age range and education level
    edu_label = "low" if edu_level == 0 else "high"

    df_filtered = df[
        (df["age"] >= start_age)
        & (df["age"] <= end_age)
        & (df["education"] == edu_level)
    ].copy()

    # 2) Define 5-year age bins
    #    For example, if start_age=30, end_age=50, bins -> [30, 35, 40, 45, 50, 51]
    #    So you get these intervals: [30,35), [35,40), [40,45), [45,50), [50,51)
    #    The last bin may be shorter if end_age is not a multiple of 5.
    bins = list(range(start_age, end_age + 1, 5))
    if bins[-1] < end_age:
        bins.append(end_age + 1)  # ensure coverage up to end_age

    bin_labels = [
        f"age_{bins[i]}_{bins[i+1] - 1}_edu_{edu_label}" for i in range(len(bins) - 1)
    ]

    # 3) Assign each row to a bin
    df_filtered["age_bin"] = pd.cut(
        df_filtered["age"],
        bins=bins,
        right=False,  # left-closed, right-open intervals
        labels=bin_labels,
    )

    # 4) Group by the bin, compute mean & variance
    grouped = df_filtered.groupby("age_bin", observed=False)["wealth"]
    wealth_mean = grouped.mean()
    wealth_var = grouped.var(ddof=1)  # ddof=1 for sample variance

    # 5) Store in dictionaries
    moments = {}
    variances = {}
    for bin_label in wealth_mean.index:
        # e.g. bin_label might be "[30,35)"
        mean_key = f"wealth_{bin_label}"
        moments[mean_key] = wealth_mean[bin_label]
        variances[mean_key] = wealth_var[bin_label]

    return moments, variances


def compute_wealth_moments(subdf):
    """Compute mean wealth and its variance for a given subsample."""
    if len(subdf) == 0:
        return {
            "wealth_mean": np.nan,
            "wealth_var": np.nan,
        }

    wealth_mean = subdf["wealth"].mean()
    wealth_var = np.var(subdf["wealth"], ddof=DEGREES_OF_FREEDOM)

    return {
        "wealth_mean": wealth_mean,
        "wealth_var": wealth_var,
    }


# =====================================================================================
# Auxiliary functions
# =====================================================================================


def _get_labor_shares_by_age(df, moments, variances, min_age, max_age):

    for age in range(min_age, max_age + 1):
        subdf = df[df["age"] == age]
        shares = compute_labor_shares(subdf)

        moments[f"share_retired_age_{age}"] = shares["retired"]
        moments[f"share_unemployed_age_{age}"] = shares["unemployed"]
        moments[f"share_working_part_time_age_{age}"] = shares["part_time"]
        moments[f"share_working_full_time_age_{age}"] = shares["full_time"]
        # moments[f"share_not_working_age_{age}"] = shares["not_working"]

        variances[f"share_retired_age_{age}"] = shares["retired_var"]
        variances[f"share_unemployed_age_{age}"] = shares["unemployed_var"]
        variances[f"share_working_part_time_age_{age}"] = shares["part_time_var"]
        variances[f"share_working_full_time_age_{age}"] = shares["full_time_var"]
        # variances[f"share_not_working_age_{age}"] = shares["not_working_var"]

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

            moments[f"share_being_retired_{key}"] = shares["retired"]
            moments[f"share_being_unemployed_{key}"] = shares["unemployed"]
            moments[f"share_working_part_time_{key}"] = shares["part_time"]
            moments[f"share_working_full_time_{key}"] = shares["full_time"]
            # moments[f"share_not_working_{key}"] = shares["not_working"]

            variances[f"share_being_retired_{key}"] = shares["retired_var"]
            variances[f"share_being_unemployed_{key}"] = shares["unemployed_var"]
            variances[f"share_working_part_time_{key}"] = shares["part_time_var"]
            variances[f"share_working_full_time_{key}"] = shares["full_time_var"]
            # variances[f"share_not_working_{key}"] = shares["not_working_var"]

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

            moments[f"share_not_working_{key}"] = shares["not_working"]
            moments[f"share_part_time_{key}"] = shares["part_time"]
            moments[f"share_full_time_{key}"] = shares["full_time"]

            variances[f"share_not_working_{key}"] = shares["not_working_var"]
            variances[f"share_part_time_{key}"] = shares["part_time_var"]
            variances[f"share_full_time_{key}"] = shares["full_time_var"]

    return moments, variances


def _get_labor_shares_by_educ_level_and_age_of_youngest_child(
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


def _get_wealth_moments_by_age_and_education(df, moments, variances, min_age, max_age):
    """
    Loop over age and education to compute mean wealth and variance,
    storing results in the same 'moments' and 'variances' dicts.
    """
    for edu in (0, 1):
        edu_label = "low" if edu == 0 else "high"
        for age in range(min_age, max_age + 1):
            subdf = df[(df["age"] == age) & (df["education"] == edu)]
            wealth = compute_wealth_moments(subdf)

            key = f"wealth_age_{age}_edu_{edu_label}"
            moments[key] = wealth["wealth_mean"]
            variances[key] = wealth["wealth_var"]

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


def plot_wealth_by_age(df, start_age, end_age, educ_val=None):
    """
    Plot mean wealth by age from start_age to end_age.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing at least the columns 'age', 'education', and 'wealth'.
    start_age : int
        Lower bound (inclusive) of the age range.
    end_age : int
        Upper bound (inclusive) of the age range.
    educ_val : {0, 1, None}, optional
        If 0 or 1, filter the data to only rows where 'education' == educ_val.
        If None (default), use all education levels (unconditional).

    """

    # 1) Restrict to the specified age range
    df_ages = df[(df["age"] >= start_age) & (df["age"] <= end_age)]

    # 2) If educ_val is given (0 or 1), filter to that education category
    if educ_val is not None:
        df_ages = df_ages[df_ages["education"] == educ_val]

    # 3) Group by age and compute mean wealth
    grouped = df_ages.groupby("age")["wealth"].mean().sort_index()

    # 4) Plot mean wealth by age
    plt.figure(figsize=(8, 5))
    plt.plot(grouped.index, grouped.values, marker="o", label="Mean Wealth")

    # Labels
    plt.xlabel("Age")
    plt.ylabel("Mean Wealth")

    # Construct a descriptive title
    if educ_val is None:
        title_str = "Mean Wealth by Age (All Education)"
    else:
        title_str = f"Mean Wealth by Age (Education={educ_val})"
    title_str += f" — Ages {start_age} to {end_age}"
    plt.title(title_str)

    plt.grid(True)
    plt.legend()
    plt.show()


def plot_wealth_by_5yr_bins(df, start_age, end_age, educ_val=None):
    """
    Plot mean wealth for 5-year age bins between start_age and end_age
    as a line chart, optionally filtering by education.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'age', 'wealth', and 'education' columns.
    start_age : int
        Lower bound (inclusive) of the age range to plot.
    end_age : int
        Upper bound (inclusive) of the age range to plot.
    educ_val : {0, 1, None}, optional
        If 0 or 1, keep only rows where 'education' == edu_val.
        If None (default), use all education levels.

    """

    # 1) Filter rows by age
    df_filtered = df[(df["age"] >= start_age) & (df["age"] <= end_age)].copy()

    # 2) If edu_val is specified (0 or 1), filter further
    if educ_val is not None:
        df_filtered = df_filtered[df_filtered["education"] == educ_val]

    # 3) Define 5-year bins
    #    e.g. if start_age=30, end_age=50 => bins=[30, 35, 40, 45, 50, 51]
    #    Last bin covers [50,51) so that we include all ages <= end_age.
    bins = list(range(start_age, end_age + 1, 5))
    if bins[-1] < end_age:
        bins.append(end_age + 1)  # ensure coverage to end_age

    bin_labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]

    # 4) Assign each row to an interval
    df_filtered["age_bin"] = pd.cut(
        df_filtered["age"],
        bins=bins,
        right=False,  # left-closed, right-open
        labels=bin_labels,
    )

    # 5) Group by bin, compute mean wealth
    grouped = df_filtered.groupby("age_bin")["wealth"].mean().reset_index()

    # 6) Plot a line chart with textual interval labels on the x-axis
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["age_bin"], grouped["wealth"], marker="o", linestyle="-")
    plt.xlabel("Age Bin")
    plt.ylabel("Mean Wealth")

    # Build a descriptive title
    if educ_val is None:
        edu_str = "All Education"
    else:
        edu_str = f"Education={educ_val}"
    plt.title(
        f"Mean Wealth by 5-Year Age Bins ({edu_str})\nAges {start_age} to {end_age}"
    )

    plt.xticks(rotation=45)  # rotate labels if they overlap
    plt.grid(True)
    plt.tight_layout()
    plt.show()
