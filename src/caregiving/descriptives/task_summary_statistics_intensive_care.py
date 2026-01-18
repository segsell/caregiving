"""Create summary statistics for intensive care parents versus other by age bin."""

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
def task_summary_statistics_intensive_care_by_age_bin(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "descriptives"
    / "intensive_care_parents_versus_other_by_age_bin.csv",
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
def task_summary_statistics_intensive_care_extra_hh_by_age_bin(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "descriptives"
    / "intensive_care_parents_versus_other_extra_hh_by_age_bin.csv",
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
