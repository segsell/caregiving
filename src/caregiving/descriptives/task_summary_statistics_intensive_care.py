"""Create summary statistics for daily care parents versus other by age bin."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    AGE_40,
    AGE_45,
    AGE_50,
    AGE_55,
    AGE_60,
    AGE_65,
    AGE_70,
    AGE_75,
)


@pytask.mark.descriptives
def task_summary_statistics_daily_care_by_age_bin(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "descriptives"
    / "daily_care_parents_versus_other_by_age_bin.csv",
) -> None:
    """Create summary statistics for intensive_care_parents_versus_other.

    This task loads the SHARE estimation data and calculates summary statistics
    (mean, standard deviation, and sample size) for the variable
    `intensive_care_parents_versus_other` by age bins from [40, 50)
    to [70, 75).

    Parameters
    ----------
    path_to_estimation_data : Path
        Path to the SHARE estimation data CSV file.
    path_to_save_summary : Path
        Path to save the summary statistics CSV file.
    """
    # Load the estimation data
    df = pd.read_csv(path_to_estimation_data)

    # Define age bins: [40, 50), [50, 55), ..., [70, 75)
    bin_edges = [
        AGE_40,
        AGE_50,
        AGE_55,
        AGE_60,
        AGE_65,
        AGE_70,
        AGE_75,
    ]
    bin_labels = [
        f"[{bin_edges[i]}, {bin_edges[i+1]})" for i in range(len(bin_edges) - 1)
    ]

    # Filter data to age range [40, 75) - ages 40 to 74 inclusive
    df_filtered = df[(df["age"] >= AGE_40) & (df["age"] < AGE_75)].copy()

    # Create age bin column
    df_filtered["age_bin"] = pd.cut(
        df_filtered["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # left-closed, right-open intervals
    )

    # Group by age bin and calculate statistics
    grouped = df_filtered.groupby("age_bin", observed=False)[
        "intensive_care_parents_versus_other"
    ]

    # Calculate mean, standard deviation, and count
    # Note: We exclude NaN values in calculations
    summary_stats = pd.DataFrame(
        {
            "age_bin": bin_labels,
            "mean": grouped.mean().reindex(bin_labels, fill_value=np.nan),
            "std": grouped.std(ddof=1).reindex(bin_labels, fill_value=np.nan),
            "n_observations": grouped.count().reindex(bin_labels, fill_value=0),
        }
    )

    # Ensure directory exists
    path_to_save_summary.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    summary_stats.to_csv(path_to_save_summary, index=False)


@pytask.mark.descriptives
def task_summary_statistics_daily_care_by_age_bin_and_education(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "descriptives"
    / "daily_care_parents_versus_other_by_age_bin_and_education.csv",
) -> None:
    """Create summary statistics for intensive_care_parents_versus_other by education.

    This task loads the SHARE estimation data and calculates summary statistics
    (mean, standard deviation, and sample size) for the variable
    `intensive_care_parents_versus_other` by age bins from [40, 50)
    to [70, 75), separately by education level.

    Statistics are computed for three different education variables:
    - high_educ
    - high_educ_isced
    - high_isced

    Each education variable is split into "low" and "high" categories.
    Within each block (each education variable), low education rows come first,
    followed by high education rows, for readability.

    Parameters
    ----------
    path_to_estimation_data : Path
        Path to the SHARE estimation data CSV file.
    path_to_save_summary : Path
        Path to save the summary statistics CSV file.
    """
    # Load the estimation data
    df = pd.read_csv(path_to_estimation_data)

    # Define age bins: [40, 50), [50, 55), ..., [70, 75)
    bin_edges = [
        AGE_40,
        AGE_50,
        AGE_55,
        AGE_60,
        AGE_65,
        AGE_70,
        AGE_75,
    ]
    bin_labels = [
        f"[{bin_edges[i]}, {bin_edges[i+1]})" for i in range(len(bin_edges) - 1)
    ]

    # Filter data to age range [40, 75) - ages 40 to 74 inclusive
    df_filtered = df[(df["age"] >= AGE_40) & (df["age"] < AGE_75)].copy()

    # Create age bin column
    df_filtered["age_bin"] = pd.cut(
        df_filtered["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # left-closed, right-open intervals
    )

    # Compute statistics for each education variable
    educ_vars = ["high_educ", "high_educ_isced", "high_isced"]
    all_stats = []

    for educ_var in educ_vars:
        if educ_var not in df_filtered.columns:
            continue  # Skip if variable doesn't exist

        stats_df = _compute_stats_by_age_bin_and_education(
            df_filtered, bin_labels, educ_var
        )
        all_stats.append(stats_df)

    # Combine all statistics into one DataFrame
    if all_stats:
        summary_stats = pd.concat(all_stats, ignore_index=True)
        # Reorder: within each educ_variable block, low education first, then high
        # Also ensure columns are in correct order
        summary_stats = summary_stats.sort_values(
            by=["educ_variable", "education", "age_bin"]
        )
        col_order = [
            "age_bin",
            "education",
            "mean",
            "std",
            "n_observations",
            "educ_variable",
        ]
        summary_stats = summary_stats[col_order]
    else:
        # Fallback: create empty DataFrame with correct structure
        summary_stats = pd.DataFrame(
            columns=[
                "age_bin",
                "education",
                "mean",
                "std",
                "n_observations",
                "educ_variable",
            ]
        )

    # Ensure directory exists
    path_to_save_summary.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    summary_stats.to_csv(path_to_save_summary, index=False)


@pytask.mark.descriptives
def task_summary_statistics_daily_care_extra_hh_by_age_bin(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "descriptives"
    / "daily_care_parents_versus_other_extra_hh_by_age_bin.csv",
) -> None:
    """Create summary statistics for intensive_care_parents_versus_other_extra_hh.

    This task loads the SHARE estimation data and calculates summary statistics
    (mean, standard deviation, and sample size) for the variable
    `intensive_care_parents_versus_other_extra_hh` by age bins from
    [40, 50) to [70, 75).

    Parameters
    ----------
    path_to_estimation_data : Path
        Path to the SHARE estimation data CSV file.
    path_to_save_summary : Path
        Path to save the summary statistics CSV file.
    """
    # Load the estimation data
    df = pd.read_csv(path_to_estimation_data)

    # Define age bins: [40, 50), [50, 55), ..., [70, 75)
    bin_edges = [
        AGE_40,
        AGE_50,
        AGE_55,
        AGE_60,
        AGE_65,
        AGE_70,
        AGE_75,
    ]
    bin_labels = [
        f"[{bin_edges[i]}, {bin_edges[i+1]})" for i in range(len(bin_edges) - 1)
    ]

    # Filter data to age range [40, 75) - ages 40 to 74 inclusive
    df_filtered = df[(df["age"] >= AGE_40) & (df["age"] < AGE_75)].copy()

    # Create age bin column
    df_filtered["age_bin"] = pd.cut(
        df_filtered["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # left-closed, right-open intervals
    )

    # Group by age bin and calculate statistics
    grouped = df_filtered.groupby("age_bin", observed=False)[
        "intensive_care_parents_versus_other_extra_hh"
    ]

    # Calculate mean, standard deviation, and count
    # Note: We exclude NaN values in calculations
    summary_stats = pd.DataFrame(
        {
            "age_bin": bin_labels,
            "mean": grouped.mean().reindex(bin_labels, fill_value=np.nan),
            "std": grouped.std(ddof=1).reindex(bin_labels, fill_value=np.nan),
            "n_observations": grouped.count().reindex(bin_labels, fill_value=0),
        }
    )

    # Ensure directory exists
    path_to_save_summary.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    summary_stats.to_csv(path_to_save_summary, index=False)


def _compute_stats_by_age_bin_and_education(
    df_filtered: pd.DataFrame,
    bin_labels: list[str],
    educ_var: str,
) -> pd.DataFrame:
    """Compute statistics by age bin and education level.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Filtered dataframe with age_bin column already created.
    bin_labels : list[str]
        List of age bin labels.
    educ_var : str
        Name of the education variable to use.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: age_bin, education, mean, std, n_observations,
        educ_variable
    """
    # Create education level column (low/high)
    # Filter out NaN values in education variable
    df_educ = df_filtered[df_filtered[educ_var].notna()].copy()
    df_educ["education"] = df_educ[educ_var].map({0: "low", 1: "high"})

    # Group by age bin and education
    grouped = df_educ.groupby(["age_bin", "education"], observed=False)[
        "intensive_care_parents_versus_other"
    ]

    # Create all combinations of age_bin and education
    # Order: low education first for each age bin, then high education
    age_bin_educ_combos = []
    for age_bin in bin_labels:
        age_bin_educ_combos.extend([(age_bin, "low"), (age_bin, "high")])

    # Calculate statistics
    stats_list = []
    for age_bin, educ in age_bin_educ_combos:
        try:
            mean_val = grouped.get_group((age_bin, educ)).mean()
            std_val = grouped.get_group((age_bin, educ)).std(ddof=1)
            count_val = grouped.get_group((age_bin, educ)).count()
        except KeyError:
            # Group doesn't exist (no observations)
            mean_val = np.nan
            std_val = np.nan
            count_val = 0

        stats_list.append(
            {
                "age_bin": age_bin,
                "education": educ,
                "mean": mean_val,
                "std": std_val,
                "n_observations": count_val,
            }
        )

    stats_df = pd.DataFrame(stats_list)
    stats_df["educ_variable"] = educ_var
    return stats_df
