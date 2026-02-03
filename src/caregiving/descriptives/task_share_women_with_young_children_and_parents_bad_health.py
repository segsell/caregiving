"""Calculate percentage of women with young children and parents in bad health."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import AGE_40, AGE_70

# Age thresholds for youngest child
CHILD_AGE_THRESHOLDS = [8, 10, 12, 14, 16, 18]

# Health status: 0 = good, 1 = fair, 2 = poor
# "Bad health" is defined as health >= 1 (fair or poor)
BAD_HEALTH_THRESHOLD = 2


@pytask.mark.descriptives
def task_share_women_young_children_parents_bad_health(  # noqa: PLR0915
    path_to_share_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "descriptives"
    / "share_women_young_children_parents_bad_health.tex",
) -> None:
    """Calculate percentage of women with young children and parents in bad health.

    This task analyzes women aged 40-70 and calculates the percentage who have:
    - A youngest child below various age thresholds (8, 10, 12, 14, 16, 18)
    - AND at least one parent (mother or father) in bad health

    Results are reported separately for:
    - Any parent (mother OR father) in bad health
    - Only mother in bad health

    Parameters
    ----------
    path_to_share_data : Path
        Path to the SHARE estimation data CSV file.
    path_to_save_table : Path
        Path to save the LaTeX table with results.
    """
    # Load the SHARE data
    df = pd.read_csv(path_to_share_data)

    # Verify required columns exist
    required_cols = [
        "age",
        "age_youngest_child",
        "mother_health",
        "father_health",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in data: {missing_cols}. "
            "Please ensure age_youngest_child is created and saved correctly."
        )

    # Filter to women aged 40-70
    df_40_70 = df[(df["age"] >= AGE_40) & (df["age"] <= AGE_70)].copy()

    # Create indicators for bad health
    df_40_70["mother_bad_health"] = (
        df_40_70["mother_health"] >= BAD_HEALTH_THRESHOLD
    ).astype(float)
    df_40_70["father_bad_health"] = (
        df_40_70["father_health"] >= BAD_HEALTH_THRESHOLD
    ).astype(float)
    df_40_70["any_parent_bad_health"] = (
        (df_40_70["mother_bad_health"] == 1) | (df_40_70["father_bad_health"] == 1)
    ).astype(float)

    # Set to NaN where health status is missing
    df_40_70.loc[df_40_70["mother_health"].isna(), "mother_bad_health"] = np.nan
    df_40_70.loc[df_40_70["father_health"].isna(), "father_bad_health"] = np.nan
    df_40_70.loc[
        df_40_70["mother_health"].isna() & df_40_70["father_health"].isna(),
        "any_parent_bad_health",
    ] = np.nan

    # Initialize results list
    results_list = []

    # Calculate for each child age threshold
    for child_age_threshold in CHILD_AGE_THRESHOLDS:
        # Filter to women with youngest child below threshold
        # Only include if age_youngest_child is not missing
        mask_child = df_40_70["age_youngest_child"].notna() & (
            df_40_70["age_youngest_child"] < child_age_threshold
        )

        # Calculate for "any parent in bad health"
        # Subset: age_youngest_child < threshold AND any_parent_bad_health not missing
        mask_any_parent = mask_child & df_40_70["any_parent_bad_health"].notna()
        n_any_parent = mask_any_parent.sum()
        if n_any_parent > 0:
            pct_any_parent = (
                (df_40_70.loc[mask_any_parent, "any_parent_bad_health"] == 1).sum()
                / n_any_parent
                * 100
            )
        else:
            pct_any_parent = np.nan

        # Calculate for "mother in bad health"
        # Subset: age_youngest_child < threshold AND mother_bad_health not missing
        mask_mother = mask_child & df_40_70["mother_bad_health"].notna()
        n_mother = mask_mother.sum()
        if n_mother > 0:
            pct_mother = (
                (df_40_70.loc[mask_mother, "mother_bad_health"] == 1).sum()
                / n_mother
                * 100
            )
        else:
            pct_mother = np.nan

        results_list.append(
            {
                "Specification": f"Child age < {child_age_threshold}",
                "Any Parent Bad Health (%)": pct_any_parent,
                "N (Any Parent)": int(n_any_parent),
                "Mother Bad Health (%)": pct_mother,
                "N (Mother)": int(n_mother),
            }
        )

    # Create results DataFrame
    results = pd.DataFrame(results_list)

    # Format percentages (round to 2 decimal places, handle NaN)
    results["Any Parent Bad Health (%)"] = results["Any Parent Bad Health (%)"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    results["Mother Bad Health (%)"] = results["Mother Bad Health (%)"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )

    # Ensure directory exists
    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)

    # Save as LaTeX table
    latex_table = results.to_latex(
        index=False,
        float_format="%.2f",
        escape=False,
    )

    # Write to file
    path_to_save_table.open("w").write(latex_table)
