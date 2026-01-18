"""Alternative simulation moments using MultiIndex approach (colleague's style)."""

import numpy as np
import pandas as pd

from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_INFORMAL_CARE,
    PART_TIME,
    RETIREMENT,
    UNEMPLOYED,
)


def simulate_moments_pandas_alternative(
    df_full: pd.DataFrame,
    model_specs: dict,
) -> pd.DataFrame:
    """
    Simulate moments using MultiIndex approach (colleague's style).

    For now, only computes labor moments (not wealth or transitions).

    Parameters
    ----------
    df_full : pd.DataFrame
        Full simulation DataFrame.
    model_specs : dict
        Model specifications.

    Returns
    -------
    pd.DataFrame
        Series with moment names as index and moment values as values.
    """
    start_age = model_specs["start_age"]
    end_age = model_specs["end_age_msm"]
    start_age_caregivers = model_specs["start_age_caregiving"]
    end_age_caregiving = model_specs["end_age_caregiving"]

    age_range = range(start_age, end_age + 1)
    age_range_caregivers = range(start_age_caregivers, end_age_caregiving + 1)

    # Define age bins for caregivers (5-year bins)
    age_bins_caregivers_5year = (
        list(range(40, 75, 5)),  # [40, 45, 50, 55, 60, 65, 70]
        [f"{s}_{s+4}" for s in range(40, 70, 5)],  # ["40_44", "45_49", ..., "65_69"]
    )

    # Define age bins for caregivers (3-year bins)
    bin_edges_caregivers = []
    current_edge = start_age_caregivers
    while current_edge + 3 <= end_age_caregiving + 1:
        bin_edges_caregivers.append(current_edge)
        current_edge += 3
    if bin_edges_caregivers:
        bin_edges_caregivers.append(bin_edges_caregivers[-1] + 3)
    bin_labels_caregivers = [f"{s}_{s+2}" for s in bin_edges_caregivers[:-1]]
    age_bins_caregivers_3year = (bin_edges_caregivers, bin_labels_caregivers)

    # Filter out dead observations
    df_full = df_full.loc[df_full["health"] != DEAD].copy()

    # Create filtered dataframes
    df_non_caregivers = df_full[
        df_full["choice"].isin(np.asarray(NO_INFORMAL_CARE).tolist())
    ].copy()

    df_low = df_non_caregivers[df_non_caregivers["education"] == 0]
    df_high = df_non_caregivers[df_non_caregivers["education"] == 1]

    df_caregivers = df_full[
        df_full["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
    ].copy()
    df_caregivers_low = df_caregivers[df_caregivers["education"] == 0]
    df_caregivers_high = df_caregivers[df_caregivers["education"] == 1]

    df_light_caregivers = df_full[
        df_full["choice"].isin(np.asarray(LIGHT_INFORMAL_CARE).tolist())
    ].copy()
    df_light_caregivers_low = df_light_caregivers[df_light_caregivers["education"] == 0]
    df_light_caregivers_high = df_light_caregivers[
        df_light_caregivers["education"] == 1
    ]

    df_intensive_caregivers = df_full[
        df_full["choice"].isin(np.asarray(INTENSIVE_INFORMAL_CARE).tolist())
    ].copy()
    df_intensive_caregivers_low = df_intensive_caregivers[
        df_intensive_caregivers["education"] == 0
    ]
    df_intensive_caregivers_high = df_intensive_caregivers[
        df_intensive_caregivers["education"] == 1
    ]

    moments = {}

    # =================================================================================
    # Labor moments by age (non-caregivers)
    # =================================================================================
    # All non-caregivers
    shares_all = calc_labor_supply_choice_alternative(df_non_caregivers, model_specs)
    moments.update(_convert_shares_series_to_dict(shares_all, label="", start_age=start_age))

    # Low education
    shares_low = calc_labor_supply_choice_alternative(df_low, model_specs)
    moments.update(_convert_shares_series_to_dict(shares_low, label="low_education", start_age=start_age))

    # High education
    shares_high = calc_labor_supply_choice_alternative(df_high, model_specs)
    moments.update(_convert_shares_series_to_dict(shares_high, label="high_education", start_age=start_age))

    # =================================================================================
    # Labor moments by age bin (caregivers)
    # =================================================================================
    # All caregivers
    shares_caregivers = calc_labor_supply_choice_by_age_bin_alternative(
        df_caregivers, model_specs, *age_bins_caregivers_3year
    )
    moments.update(_convert_shares_series_to_dict(shares_caregivers, label="caregivers"))

    # Caregivers low education
    shares_caregivers_low = calc_labor_supply_choice_by_age_bin_alternative(
        df_caregivers_low, model_specs, *age_bins_caregivers_3year
    )
    moments.update(
        _convert_shares_series_to_dict(shares_caregivers_low, label="caregivers_low_education")
    )

    # Caregivers high education
    shares_caregivers_high = calc_labor_supply_choice_by_age_bin_alternative(
        df_caregivers_high, model_specs, *age_bins_caregivers_3year
    )
    moments.update(
        _convert_shares_series_to_dict(shares_caregivers_high, label="caregivers_high_education")
    )

    # Light caregivers
    shares_light = calc_labor_supply_choice_by_age_bin_alternative(
        df_light_caregivers, model_specs, *age_bins_caregivers_3year
    )
    moments.update(_convert_shares_series_to_dict(shares_light, label="light_caregivers"))

    # Light caregivers low education
    shares_light_low = calc_labor_supply_choice_by_age_bin_alternative(
        df_light_caregivers_low, model_specs, *age_bins_caregivers_3year
    )
    moments.update(
        _convert_shares_series_to_dict(shares_light_low, label="light_caregivers_low_education")
    )

    # Light caregivers high education
    shares_light_high = calc_labor_supply_choice_by_age_bin_alternative(
        df_light_caregivers_high, model_specs, *age_bins_caregivers_3year
    )
    moments.update(
        _convert_shares_series_to_dict(shares_light_high, label="light_caregivers_high_education")
    )

    # Intensive caregivers
    shares_intensive = calc_labor_supply_choice_by_age_bin_alternative(
        df_intensive_caregivers, model_specs, *age_bins_caregivers_3year
    )
    moments.update(
        _convert_shares_series_to_dict(shares_intensive, label="intensive_caregivers")
    )

    # Intensive caregivers low education
    shares_intensive_low = calc_labor_supply_choice_by_age_bin_alternative(
        df_intensive_caregivers_low, model_specs, *age_bins_caregivers_3year
    )
    moments.update(
        _convert_shares_series_to_dict(shares_intensive_low, label="intensive_caregivers_low_education")
    )

    # Intensive caregivers high education
    shares_intensive_high = calc_labor_supply_choice_by_age_bin_alternative(
        df_intensive_caregivers_high, model_specs, *age_bins_caregivers_3year
    )
    moments.update(
        _convert_shares_series_to_dict(shares_intensive_high, label="intensive_caregivers_high_education")
    )

    return pd.Series(moments)



def calc_labor_supply_choice_alternative(
    df: pd.DataFrame,
    specs: dict,
    sex_values: list[int] | None = None,
    education_values: list[int] | None = None,
) -> pd.Series:
    """
    Calculate labor supply choice shares using MultiIndex approach (colleague's style).

    Parameters
    ----------
    df : pd.DataFrame
        Simulation DataFrame. Must contain columns: 'sex', 'education', 'period', 'choice'.
    specs : dict
        Model specifications. Must contain 'start_age' and 'end_age_msm'.
    sex_values : list[int] | None
        List of sex values to include. If None, uses all unique values from df['sex'].
    education_values : list[int] | None
        List of education values to include. If None, uses all unique values from df['education'].

    Returns
    -------
    pd.Series
        Series with MultiIndex (sex, education, period, choice_group) containing
        normalized choice shares (0.0-1.0).
    """
    # Get period range from specs
    start_age = specs["start_age"]
    end_age_msm = specs["end_age_msm"]
    n_periods = end_age_msm - start_age

    # Determine sex and education values
    if sex_values is None:
        sex_values = sorted(df["sex"].unique().tolist())
    if education_values is None:
        education_values = sorted(df["education"].unique().tolist())

    # Create choice groups mapping: map 16 choices (0-15) to 4 labor groups (0-3)
    choice_group_map = {}
    choice_groups = {
        0: np.asarray(RETIREMENT).ravel().tolist(),
        1: np.asarray(UNEMPLOYED).ravel().tolist(),
        2: np.asarray(PART_TIME).ravel().tolist(),
        3: np.asarray(FULL_TIME).ravel().tolist(),
    }
    for agg_code, raw_codes in choice_groups.items():
        for code in raw_codes:
            choice_group_map[code] = agg_code

    # Map raw choices to aggregated choice groups
    df_copy = df.copy()
    df_copy["choice_group"] = (
        df_copy["choice"].map(choice_group_map).fillna(0).astype(int)
    )

    # Create MultiIndex with all combinations
    index_col_first = sex_values
    index = pd.MultiIndex.from_product(
        [
            index_col_first,
            education_values,
            np.arange(0, n_periods),
            [0, 1, 2, 3],  # choice_group: retired, unemployed, part-time, full-time
        ],
        names=["sex", "education", "period", "choice_group"],
    )

    # Group by sex, education, period and compute normalized choice shares
    choice_shares = (
        df_copy.groupby(["sex", "education", "period"], observed=False)["choice_group"]
        .value_counts(normalize=True)
    )

    # Reindex to fill missing combinations with 0.0
    choice_shares_full = choice_shares.reindex(index, fill_value=0.0).fillna(0.0)

    return choice_shares_full


def calc_labor_supply_choice_by_age_bin_alternative(
    df: pd.DataFrame,
    specs: dict,
    bin_edges: list[int],
    bin_labels: list[str],
    sex_values: list[int] | None = None,
    education_values: list[int] | None = None,
) -> pd.Series:
    """
    Calculate labor supply choice shares by age bin using MultiIndex approach.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation DataFrame. Must contain columns: 'sex', 'education', 'age', 'choice'.
    specs : dict
        Model specifications (used for consistency, but age bins are provided directly).
    bin_edges : list[int]
        Age bin edges (e.g., [40, 45, 50, 55, 60, 65, 70]).
    bin_labels : list[str]
        Age bin labels (e.g., ['40_44', '45_49', '50_54', ...]).
    sex_values : list[int] | None
        List of sex values to include. If None, uses all unique values from df['sex'].
    education_values : list[int] | None
        List of education values to include. If None, uses all unique values from df['education'].

    Returns
    -------
    pd.Series
        Series with MultiIndex (sex, education, age_bin, choice_group) containing
        normalized choice shares (0.0-1.0).
    """
    # Determine sex and education values
    if sex_values is None:
        sex_values = sorted(df["sex"].unique().tolist())
    if education_values is None:
        education_values = sorted(df["education"].unique().tolist())

    # Create choice groups mapping: map 16 choices (0-15) to 4 labor groups (0-3)
    choice_group_map = {}
    choice_groups = {
        0: np.asarray(RETIREMENT).ravel().tolist(),
        1: np.asarray(UNEMPLOYED).ravel().tolist(),
        2: np.asarray(PART_TIME).ravel().tolist(),
        3: np.asarray(FULL_TIME).ravel().tolist(),
    }
    for agg_code, raw_codes in choice_groups.items():
        for code in raw_codes:
            choice_group_map[code] = agg_code

    # Filter to relevant ages and create age_bin column
    df_copy = df[
        df["age"].between(bin_edges[0], bin_edges[-1] - 1)
    ].copy()

    # Handle empty dataframe case
    if len(df_copy) == 0:
        index = pd.MultiIndex.from_product(
            [
                sex_values,
                education_values,
                bin_labels,
                [0, 1, 2, 3],
            ],
            names=["sex", "education", "age_bin", "choice_group"],
        )
        return pd.Series(0.0, index=index)

    df_copy["age_bin"] = pd.cut(
        df_copy["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # [40,45) ⇒ 40-44, etc.
    )

    # Map raw choices to aggregated choice groups
    df_copy["choice_group"] = (
        df_copy["choice"].map(choice_group_map).fillna(0).astype(int)
    )

    # Create MultiIndex with all combinations
    index = pd.MultiIndex.from_product(
        [
            sex_values,
            education_values,
            bin_labels,
            [0, 1, 2, 3],  # choice_group: retired, unemployed, part-time, full-time
        ],
        names=["sex", "education", "age_bin", "choice_group"],
    )

    # Group by sex, education, age_bin and compute normalized choice shares
    choice_shares = (
        df_copy.groupby(["sex", "education", "age_bin"], observed=False)["choice_group"]
        .value_counts(normalize=True)
    )

    # Reindex to fill missing combinations with 0.0
    choice_shares_full = choice_shares.reindex(index, fill_value=0.0).fillna(0.0)

    return choice_shares_full




def _convert_shares_series_to_dict(
    shares_series: pd.Series,
    label: str = "",
    start_age: int | None = None,
) -> dict:
    """
    Convert MultiIndex Series to dictionary with moment names.

    Parameters
    ----------
    shares_series : pd.Series
        Series with MultiIndex (sex, education, period/age_bin, choice_group).
    label : str
        Label to append to moment names (prefixed with '_' if given).

    Returns
    -------
    dict
        Dictionary with moment names as keys and values as values.
        Format: "share_{choice_label}{label}_age_{age}" or
                "share_{choice_label}{label}_age_bin_{age_bin}"
    """
    if label:
        label = f"_{label}"

    choice_labels = ["retired", "unemployed", "part_time", "full_time"]
    moments = {}

    for idx, value in shares_series.items():
        sex, education, period_or_age_bin, choice_group = idx

        choice_label = choice_labels[choice_group]

        # Determine if it's period-based or age_bin-based
        if isinstance(period_or_age_bin, (int, np.integer)) and start_age is not None:
            # Period-based: convert period to age
            age = period_or_age_bin + start_age
            moment_key = f"share_{choice_label}{label}_age_{age}"
        elif isinstance(period_or_age_bin, (int, np.integer)):
            # Assume it's already age if no start_age provided
            moment_key = f"share_{choice_label}{label}_age_{period_or_age_bin}"
        else:
            # Age bin-based
            moment_key = f"share_{choice_label}{label}_age_bin_{period_or_age_bin}"

        moments[moment_key] = float(value)

    return moments
