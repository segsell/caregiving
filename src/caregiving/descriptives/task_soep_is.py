from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.data_management.soep.task_create_innovation_sample import (
    MOTHER_OR_FATHER,
    SOEP_IS_FEMALE,
)
from caregiving.model.shared import MAX_AGE_SIM, MIN_AGE_SIM


@pytask.mark.soep_is
def task_summary_statistics_pure_formal_care(
    path_to_data: Path = BLD / "data" / "soep_is_sample.csv",
    path_to_save: Annotated[Path, Product] = (
        BLD / "descriptives" / "pure_formal_care_by_age_educ_intensity.csv"
    ),
) -> None:
    """Compute summary statistics for pure formal care.

    Statistics are computed by age bin, education, and care intensity.

    This function handles multiple care recipients per pid by reshaping the data
    to long format (one row per care recipient). Each care recipient is then
    analyzed with their associated respondent characteristics (age, education).

    Parameters
    ----------
    path_to_data : Path
        Path to the CSV file with care_intensity and pure_formal_care variables.
    path_to_save : Path
        Path to save the summary statistics CSV file.

    """
    # Load data
    df = pd.read_csv(path_to_data, index_col=["pid"])

    # Create age bins for respondent: 40-49, 50-59, 60-70
    # Using [60, 71) to include ages 60-70 inclusive
    bin_edges = [40, 50, 60, 71]
    bin_labels = ["40-49", "50-59", "60-70"]

    # Filter to age range [MIN_AGE_SIM, MAX_AGE_SIM+1) to include ages 40-70
    df_filtered = df[(df["age"] >= MIN_AGE_SIM) & (df["age"] < MAX_AGE_SIM + 1)].copy()

    # Create age bin column for respondent
    df_filtered["age_bin"] = pd.cut(
        df_filtered["age"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,  # left-closed, right-open intervals
    )

    # Reshape data to long format: one row per care recipient
    # This handles the case where one pid has multiple care recipients
    care_recipient_rows = []

    # Process each of the 4 possible care recipients
    suffixes = ["", "_2", "_3", "_4"]
    for suffix in suffixes:
        # Get variable names for this care recipient
        ip02_var = f"ip02{suffix}" if suffix else "ip02"
        ip03_var = f"ip03{suffix}" if suffix else "ip03"
        care_intensity_var = f"care_intensity{suffix}" if suffix else "care_intensity"
        pure_formal_var = f"pure_formal_care{suffix}" if suffix else "pure_formal_care"

        # Check if this care recipient exists and is a parent
        if ip02_var not in df_filtered.columns:
            continue

        # Filter to rows where this care recipient is a mother (parent AND female)
        is_parent = df_filtered[ip02_var] == MOTHER_OR_FATHER
        is_mother = df_filtered[ip03_var] == SOEP_IS_FEMALE
        mask_mother = is_parent & is_mother

        # Create subset for this care recipient
        subset = df_filtered[mask_mother].copy()

        if len(subset) == 0:
            continue

        # Extract care recipient-specific variables
        subset = subset[
            [
                "age_bin",
                "education",
                care_intensity_var,
                pure_formal_var,
            ]
        ].copy()

        # Rename to standard names
        subset = subset.rename(
            columns={
                care_intensity_var: "care_intensity",
                pure_formal_var: "pure_formal_care",
            }
        )

        # Drop rows where care_intensity is NaN (invalid care level)
        subset = subset[subset["care_intensity"].notna()].copy()

        if len(subset) > 0:
            care_recipient_rows.append(subset)

    # Combine all care recipients into one DataFrame
    if not care_recipient_rows:
        # Create empty DataFrame with correct structure
        df_long = pd.DataFrame(
            columns=["age_bin", "education", "care_intensity", "pure_formal_care"]
        )
    else:
        df_long = pd.concat(care_recipient_rows, ignore_index=True)

    # Group by age_bin, education, and care_intensity
    # Compute mean, std, and N for pure_formal_care
    grouped = df_long.groupby(
        ["age_bin", "education", "care_intensity"], observed=False
    )["pure_formal_care"]

    summary_stats = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(),
            "n_observations": grouped.count(),
        }
    ).reset_index()

    # Map education to string labels for readability
    summary_stats["education"] = summary_stats["education"].map({0: "low", 1: "high"})

    # Map care_intensity to string labels for readability
    summary_stats["care_intensity"] = summary_stats["care_intensity"].map(
        {0: "light", 1: "intensive"}
    )

    # Compute aggregated statistics for age bin 40-70
    # Group by education and care_intensity only (aggregating across all ages)
    grouped_all_ages = df_long.groupby(["education", "care_intensity"], observed=False)[
        "pure_formal_care"
    ]

    summary_stats_all_ages = pd.DataFrame(
        {
            "mean": grouped_all_ages.mean(),
            "std": grouped_all_ages.std(),
            "n_observations": grouped_all_ages.count(),
        }
    ).reset_index()

    # Add age_bin column with "40-70" for all rows
    summary_stats_all_ages["age_bin"] = "40-70"

    # Map education to string labels for readability
    summary_stats_all_ages["education"] = summary_stats_all_ages["education"].map(
        {0: "low", 1: "high"}
    )

    # Map care_intensity to string labels for readability
    summary_stats_all_ages["care_intensity"] = summary_stats_all_ages[
        "care_intensity"
    ].map({0: "light", 1: "intensive"})

    # Reorder columns to match the main summary_stats
    summary_stats_all_ages = summary_stats_all_ages[
        ["age_bin", "education", "care_intensity", "mean", "std", "n_observations"]
    ]

    # Combine the age-specific stats with the aggregated stats
    summary_stats_combined = pd.concat(
        [summary_stats, summary_stats_all_ages], ignore_index=True
    )

    # Sort for readability: age_bin, education, care_intensity
    # Use a custom sort order to ensure 40-70 appears at the bottom
    age_bin_order = ["40-49", "50-59", "60-70", "40-70"]
    summary_stats_combined["age_bin"] = pd.Categorical(
        summary_stats_combined["age_bin"], categories=age_bin_order, ordered=True
    )
    summary_stats_combined = summary_stats_combined.sort_values(
        by=["age_bin", "education", "care_intensity"]
    )

    # Save to CSV
    summary_stats_combined.to_csv(path_to_save, index=False)
