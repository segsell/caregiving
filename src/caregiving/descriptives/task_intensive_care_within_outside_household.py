"""Create summary statistics for intensive care within vs outside household.

This module computes summary statistics (mean, standard deviation, sample size)
for intensive caregiving to parents, distinguishing between:
- Co-residential care (within same household)
- Non-residential care (outside household)

Statistics are computed conditionally on providing any intensive care
(within OR outside), so shares sum to 100%.

Statistics are computed separately for:
- Care to mother only
- Care to mother OR father
- All respondents (men and women)
- Only women (adult daughters)
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import AGE_50, AGE_60, AGE_70, FEMALE


@pytask.mark.descriptives
@pytask.mark.within_outside_intensive_care
def task_summary_statistics_intensive_care_within_outside(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "descriptives"
    / "intensive_care_within_outside_household.csv",
) -> None:
    """Create summary statistics for care within vs outside household.

    Computes conditional shares (mean, standard deviation, and sample size) for:
    - Intensive care to mother within/outside household
      (conditional on any intensive care)
    - Intensive care to mother OR father within/outside household
      (conditional on any)
    - General care to mother within/outside household
      (conditional on any general care)
    - General care to mother OR father within/outside household
      (conditional on any)

    Statistics are computed for women only (adult daughters), as the sample
    only contains women.

    The conditional shares for within/outside sum to 100% (among those providing
    any care of the respective type).

    Parameters
    ----------
    path_to_estimation_data : Path
        Path to the SHARE estimation data CSV file.
    path_to_save : Path
        Path to save the summary statistics CSV file.

    """
    # Load the estimation data
    df = pd.read_csv(path_to_estimation_data)

    # Create care to mother OR father variables if they don't exist
    if "care_to_mother_or_father_intensive_within" not in df.columns:
        df["care_to_mother_or_father_intensive_within"] = (
            (df["care_to_mother_intensive_within"] == 1)
            | (df["care_to_father_intensive_within"] == 1)
        ).astype(int)

    if "care_to_mother_or_father_intensive_outside" not in df.columns:
        df["care_to_mother_or_father_intensive_outside"] = (
            (df["care_to_mother_intensive_outside"] == 1)
            | (df["care_to_father_intensive_outside"] == 1)
        ).astype(int)

    # Create "any intensive care" variables (within OR outside)
    df["care_to_mother_intensive_any"] = (
        (df["care_to_mother_intensive_within"] == 1)
        | (df["care_to_mother_intensive_outside"] == 1)
    ).astype(int)

    df["care_to_mother_or_father_intensive_any"] = (
        (df["care_to_mother_or_father_intensive_within"] == 1)
        | (df["care_to_mother_or_father_intensive_outside"] == 1)
    ).astype(int)

    # Create care to mother OR father variables for general (non-intensive) care
    if "care_to_mother_or_father_within" not in df.columns:
        df["care_to_mother_or_father_within"] = (
            (df["care_to_mother_within"] == 1) | (df["care_to_father_within"] == 1)
        ).astype(int)

    if "care_to_mother_or_father_outside" not in df.columns:
        df["care_to_mother_or_father_outside"] = (
            (df["care_to_mother_outside"] == 1) | (df["care_to_father_outside"] == 1)
        ).astype(int)

    # Create "any general care" variables (within OR outside)
    df["care_to_mother_any"] = (
        (df["care_to_mother_within"] == 1) | (df["care_to_mother_outside"] == 1)
    ).astype(int)

    df["care_to_mother_or_father_any"] = (
        (df["care_to_mother_or_father_within"] == 1)
        | (df["care_to_mother_or_father_outside"] == 1)
    ).astype(int)

    # Compute statistics for women only (sample only contains women)
    df_women = df[df["gender"] == FEMALE].copy()

    # Define age bins and education levels
    all_stats = []

    # All ages, all education levels
    all_stats.extend(
        compute_conditional_stats(
            df_women, "women_only", age_min=None, age_max=None, education=None
        )
    )

    # All ages, by education
    all_stats.extend(
        compute_conditional_stats(
            df_women, "women_only", age_min=None, age_max=None, education=0
        )
    )
    all_stats.extend(
        compute_conditional_stats(
            df_women, "women_only", age_min=None, age_max=None, education=1
        )
    )

    # Age 50-59, all education
    all_stats.extend(
        compute_conditional_stats(
            df_women, "women_only", age_min=AGE_50, age_max=AGE_60 - 1, education=None
        )
    )

    # Age 50-59, by education
    all_stats.extend(
        compute_conditional_stats(
            df_women, "women_only", age_min=AGE_50, age_max=AGE_60 - 1, education=0
        )
    )
    all_stats.extend(
        compute_conditional_stats(
            df_women, "women_only", age_min=AGE_50, age_max=AGE_60 - 1, education=1
        )
    )

    # Age 60-70 (inclusive), all education
    all_stats.extend(
        compute_conditional_stats(
            df_women, "women_only", age_min=AGE_60, age_max=AGE_70, education=None
        )
    )

    # Age 60-70 (inclusive), by education
    all_stats.extend(
        compute_conditional_stats(
            df_women, "women_only", age_min=AGE_60, age_max=AGE_70, education=0
        )
    )
    all_stats.extend(
        compute_conditional_stats(
            df_women, "women_only", age_min=AGE_60, age_max=AGE_70, education=1
        )
    )

    # Create summary statistics DataFrame
    summary_stats = pd.DataFrame(all_stats)

    # Ensure directory exists
    path_to_save.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    summary_stats.to_csv(path_to_save, index=False)


def compute_conditional_stats(  # noqa: PLR0912, PLR0915
    data, respondent_group, age_min=None, age_max=None, education=None
):
    """Compute conditional statistics for within/outside household care.

    Conditions on providing care (within OR outside), then computes
    the share providing care within vs outside household.

    Computes statistics for both intensive care and general (non-intensive) care.

    Parameters
    ----------
    data : pd.DataFrame
        Data to compute statistics from.
    respondent_group : str
        Label for respondent group (e.g., "women_only").
    age_min : int, optional
        Minimum age for filtering (inclusive). If None, no lower bound.
    age_max : int, optional
        Maximum age for filtering (inclusive). If None, no upper bound.
    education : int, optional
        Education level to filter (0 = low, 1 = high). If None, all education levels.

    Returns
    -------
    list
        List of dictionaries with statistics.
    """
    # Filter by age if specified
    data_filtered = data.copy()
    if age_min is not None:
        data_filtered = data_filtered[data_filtered["age"] >= age_min]
    if age_max is not None:
        data_filtered = data_filtered[data_filtered["age"] <= age_max]

    # Filter by education if specified
    if education is not None:
        if "high_isced" not in data_filtered.columns:
            # If variable doesn't exist, return empty list
            return []
        data_filtered = data_filtered[data_filtered["high_isced"] == education]

    # Create age and education labels for the output
    if age_min is None is age_max:
        age_label = "all_ages"
    elif age_min == AGE_50 and age_max == AGE_60 - 1:
        age_label = "age_50_59"
    elif age_min == AGE_60 and age_max == AGE_70:
        age_label = "age_60_70"
    else:
        age_label = f"age_{age_min}_{age_max}"

    if education is None:
        educ_label = "all_education"
    elif education == 0:
        educ_label = "low_education"
    elif education == 1:
        educ_label = "high_education"
    else:
        educ_label = f"education_{education}"

    stats = []

    # ===================================================================
    # INTENSIVE CARE
    # ===================================================================

    # Care to mother: conditional on any intensive care to mother
    mask_mother_any = data_filtered["care_to_mother_intensive_any"] == 1
    n_mother_any = mask_mother_any.sum()

    if n_mother_any > 0:
        # Share within among those providing any intensive care to mother
        mean_within = data_filtered.loc[
            mask_mother_any, "care_to_mother_intensive_within"
        ].mean()
        std_within = data_filtered.loc[
            mask_mother_any, "care_to_mother_intensive_within"
        ].std(ddof=1)

        # Share outside among those providing any intensive care to mother
        mean_outside = data_filtered.loc[
            mask_mother_any, "care_to_mother_intensive_outside"
        ].mean()
        std_outside = data_filtered.loc[
            mask_mother_any, "care_to_mother_intensive_outside"
        ].std(ddof=1)

        stats.append(
            {
                "respondent_group": respondent_group,
                "age_bin": age_label,
                "education": educ_label,
                "care_variable": "care_to_mother_intensive_within",
                "mean": mean_within,
                "std": std_within,
                "n_observations": n_mother_any,
            }
        )
        stats.append(
            {
                "respondent_group": respondent_group,
                "age_bin": age_label,
                "education": educ_label,
                "care_variable": "care_to_mother_intensive_outside",
                "mean": mean_outside,
                "std": std_outside,
                "n_observations": n_mother_any,
            }
        )

    # Care to mother OR father: conditional on any intensive care to parent
    mask_parent_any = data_filtered["care_to_mother_or_father_intensive_any"] == 1
    n_parent_any = mask_parent_any.sum()

    if n_parent_any > 0:
        # Share within among those providing any intensive care to parent
        mean_within = data_filtered.loc[
            mask_parent_any, "care_to_mother_or_father_intensive_within"
        ].mean()
        std_within = data_filtered.loc[
            mask_parent_any, "care_to_mother_or_father_intensive_within"
        ].std(ddof=1)

        # Share outside among those providing any intensive care to parent
        mean_outside = data_filtered.loc[
            mask_parent_any, "care_to_mother_or_father_intensive_outside"
        ].mean()
        std_outside = data_filtered.loc[
            mask_parent_any, "care_to_mother_or_father_intensive_outside"
        ].std(ddof=1)

        stats.append(
            {
                "respondent_group": respondent_group,
                "age_bin": age_label,
                "education": educ_label,
                "care_variable": "care_to_mother_or_father_intensive_within",
                "mean": mean_within,
                "std": std_within,
                "n_observations": n_parent_any,
            }
        )
        stats.append(
            {
                "respondent_group": respondent_group,
                "age_bin": age_label,
                "education": educ_label,
                "care_variable": "care_to_mother_or_father_intensive_outside",
                "mean": mean_outside,
                "std": std_outside,
                "n_observations": n_parent_any,
            }
        )

    # ===================================================================
    # GENERAL (NON-INTENSIVE) CARE
    # ===================================================================

    # Care to mother: conditional on any general care to mother
    mask_mother_any = data_filtered["care_to_mother_any"] == 1
    n_mother_any = mask_mother_any.sum()

    if n_mother_any > 0:
        # Share within among those providing any general care to mother
        mean_within = data_filtered.loc[mask_mother_any, "care_to_mother_within"].mean()
        std_within = data_filtered.loc[mask_mother_any, "care_to_mother_within"].std(
            ddof=1
        )

        # Share outside among those providing any general care to mother
        mean_outside = data_filtered.loc[
            mask_mother_any, "care_to_mother_outside"
        ].mean()
        std_outside = data_filtered.loc[mask_mother_any, "care_to_mother_outside"].std(
            ddof=1
        )

        stats.append(
            {
                "respondent_group": respondent_group,
                "age_bin": age_label,
                "education": educ_label,
                "care_variable": "care_to_mother_within",
                "mean": mean_within,
                "std": std_within,
                "n_observations": n_mother_any,
            }
        )
        stats.append(
            {
                "respondent_group": respondent_group,
                "age_bin": age_label,
                "education": educ_label,
                "care_variable": "care_to_mother_outside",
                "mean": mean_outside,
                "std": std_outside,
                "n_observations": n_mother_any,
            }
        )

    # Care to mother OR father: conditional on any general care to parent
    mask_parent_any = data_filtered["care_to_mother_or_father_any"] == 1
    n_parent_any = mask_parent_any.sum()

    if n_parent_any > 0:
        # Share within among those providing any general care to parent
        mean_within = data_filtered.loc[
            mask_parent_any, "care_to_mother_or_father_within"
        ].mean()
        std_within = data_filtered.loc[
            mask_parent_any, "care_to_mother_or_father_within"
        ].std(ddof=1)

        # Share outside among those providing any general care to parent
        mean_outside = data_filtered.loc[
            mask_parent_any, "care_to_mother_or_father_outside"
        ].mean()
        std_outside = data_filtered.loc[
            mask_parent_any, "care_to_mother_or_father_outside"
        ].std(ddof=1)

        stats.append(
            {
                "respondent_group": respondent_group,
                "age_bin": age_label,
                "education": educ_label,
                "care_variable": "care_to_mother_or_father_within",
                "mean": mean_within,
                "std": std_within,
                "n_observations": n_parent_any,
            }
        )
        stats.append(
            {
                "respondent_group": respondent_group,
                "age_bin": age_label,
                "education": educ_label,
                "care_variable": "care_to_mother_or_father_outside",
                "mean": mean_outside,
                "std": std_outside,
                "n_observations": n_parent_any,
            }
        )

    return stats
