"""Analyze differential moving behavior by care status and covariates."""

from contextlib import suppress
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product, mark, task
from statsmodels.stats.proportion import proportions_ztest

from caregiving.config import BLD
from caregiving.model.shared import AGE_40, AGE_50, AGE_60, AGE_70

# Significance level constants
P_VALUE_HIGHLY_SIGNIFICANT = 0.001
P_VALUE_VERY_SIGNIFICANT = 0.01
P_VALUE_SIGNIFICANT = 0.05
P_VALUE_MARGINALLY_SIGNIFICANT = 0.1

# Moving variable scenarios
MOVING_VARIABLES = {
    "hh_has_moved_a": {
        "path_to_save": BLD
        / "descriptives"
        / "moving_behavior_analysis_hh_has_moved_a.csv",
    },
    "hh_has_moved_b": {
        "path_to_save": BLD
        / "descriptives"
        / "moving_behavior_analysis_hh_has_moved_b.csv",
    },
}

for moving_var, var_params in MOVING_VARIABLES.items():

    @mark.descriptives
    @task(
        name=f"task_moving_behavior_analysis_{moving_var}",
        kwargs={
            "path_to_caregivers_sample": Path(
                BLD / "data" / "soep_structural_caregivers_sample.csv"
            ),
            "moving_variable": moving_var,
            "path_to_save_analysis": Path(var_params["path_to_save"]),
        },
    )
    def task_moving_behavior_analysis(  # noqa: PLR0912,PLR0915
        path_to_caregivers_sample: Path,
        moving_variable: str,
        path_to_save_analysis: Annotated[Path, Product],
    ) -> None:
        """Calculate moving behavior rates by care status and covariates.

        This task analyzes moving behavior (household moves) and calculates
        moving rates by care status, age, sex, education, and other covariates.
        It tests for significant differences in moving behavior between
        caregivers and non-caregivers.

        Note: Uses the caregivers sample which has proper panel structure
        (multiple observations per person).

        Parameters
        ----------
        path_to_caregivers_sample : Path
            Path to the SOEP caregivers sample CSV file (has panel structure).
        moving_variable : str
            Name of the moving variable to analyze (hh_has_moved_a or hh_has_moved_b).
        path_to_save_analysis : Path
            Path to save the moving behavior analysis CSV file.
        """
        # Load the caregivers sample (has panel structure)
        df = pd.read_csv(path_to_caregivers_sample)

        # The caregivers sample should have pid as a column
        if "pid" not in df.columns and "Unnamed: 0" in df.columns:
            df_temp = df[["Unnamed: 0", "syear"]].drop_duplicates()
            if df_temp["Unnamed: 0"].nunique() > len(df) * 0.5:
                df = df.rename(columns={"Unnamed: 0": "pid"})

        if "pid" not in df.columns:
            raise ValueError(
                "pid column not found in caregivers sample. " "Expected 'pid' column."
            )

        # Ensure we have syear column
        if "syear" not in df.columns:
            raise ValueError("syear column not found in caregivers sample")

        # Check if we have panel structure (multiple observations per pid)
        pid_counts = df.groupby("pid").size()
        if (pid_counts > 1).sum() == 0:
            raise ValueError(
                "Caregivers sample does not have panel structure. "
                "Each person appears only once. Cannot calculate moving behavior."
            )

        # Check if moving variable exists
        if moving_variable not in df.columns:
            raise ValueError(
                f"Moving variable '{moving_variable}' not found in caregivers sample."
            )

        # Ensure data is sorted by pid and syear
        df = df.sort_values(["pid", "syear"]).copy()

        # Create lagged care status
        df["lagged_any_care"] = df.groupby("pid")["any_care"].shift(1)

        # Only analyze observations where we have lagged care status
        df_analysis = df[df["lagged_any_care"].notna()].copy()

        # Create care status categories
        def categorize_care_status(val):
            if pd.isna(val):
                return "Missing"
            if val == 1:
                return "Caregiver"
            if val == 0:
                return "Non-caregiver"
            return "Unknown"

        df_analysis["lagged_care_status"] = df_analysis["lagged_any_care"].apply(
            categorize_care_status
        )

        # Create age groups
        def categorize_age(age):
            if pd.isna(age):
                return "Missing"
            if age < AGE_40:
                return "30-39"
            if age < AGE_50:
                return "40-49"
            if age < AGE_60:
                return "50-59"
            if age < AGE_70:
                return "60-69"
            return "70+"

        df_analysis["age_group"] = df_analysis["age"].apply(categorize_age)

        # Create education groups
        def categorize_education(edu):
            if pd.isna(edu):
                return "Missing"
            if edu == 0:
                return "Low"
            if edu == 1:
                return "High"
            return "Unknown"

        df_analysis["education_group"] = df_analysis["education"].apply(
            categorize_education
        )

        # Create sex groups
        def categorize_sex(sex_val):
            if pd.isna(sex_val):
                return "Missing"
            if sex_val == 0:
                return "Male"
            if sex_val == 1:
                return "Female"
            return "Unknown"

        df_analysis["sex_group"] = df_analysis["sex"].apply(categorize_sex)

        # Filter to non-missing moving variable
        df_analysis = df_analysis[df_analysis[moving_variable].notna()].copy()

        # Calculate moving rates by different groupings
        results_list = []

        # 1. By lagged care status only
        moving_care = (
            df_analysis.groupby("lagged_care_status")[moving_variable]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        for _, row in moving_care.iterrows():
            results_list.append(
                {
                    "Grouping": "Lagged Care Status",
                    "Category": row["lagged_care_status"],
                    "Subcategory": "All",
                    "Moving Rate (%)": row["mean"] * 100,
                    "N Observations": int(row["count"]),
                    "N Moved": int(row["sum"]),
                }
            )

        # 2. By lagged care status and age group
        moving_care_age = (
            df_analysis.groupby(["lagged_care_status", "age_group"])[moving_variable]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        for _, row in moving_care_age.iterrows():
            results_list.append(
                {
                    "Grouping": "Lagged Care Status x Age",
                    "Category": row["lagged_care_status"],
                    "Subcategory": row["age_group"],
                    "Moving Rate (%)": row["mean"] * 100,
                    "N Observations": int(row["count"]),
                    "N Moved": int(row["sum"]),
                }
            )

        # 3. By lagged care status and education
        moving_care_edu = (
            df_analysis.groupby(["lagged_care_status", "education_group"])[
                moving_variable
            ]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        for _, row in moving_care_edu.iterrows():
            results_list.append(
                {
                    "Grouping": "Lagged Care Status x Education",
                    "Category": row["lagged_care_status"],
                    "Subcategory": row["education_group"],
                    "Moving Rate (%)": row["mean"] * 100,
                    "N Observations": int(row["count"]),
                    "N Moved": int(row["sum"]),
                }
            )

        # 4. By lagged care status and sex
        moving_care_sex = (
            df_analysis.groupby(["lagged_care_status", "sex_group"])[moving_variable]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        for _, row in moving_care_sex.iterrows():
            results_list.append(
                {
                    "Grouping": f"Lagged Care Status x Sex ({row['sex_group']})",
                    "Category": row["lagged_care_status"],
                    "Subcategory": "All",
                    "Moving Rate (%)": row["mean"] * 100,
                    "N Observations": int(row["count"]),
                    "N Moved": int(row["sum"]),
                }
            )

        # 5. By age group only (for reference)
        moving_age = (
            df_analysis.groupby("age_group")[moving_variable]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        for _, row in moving_age.iterrows():
            results_list.append(
                {
                    "Grouping": "Age Group",
                    "Category": row["age_group"],
                    "Subcategory": "All",
                    "Moving Rate (%)": row["mean"] * 100,
                    "N Observations": int(row["count"]),
                    "N Moved": int(row["sum"]),
                }
            )

        # 6. By lagged care status, age group, and sex
        # (separate analyses for male and female)
        moving_care_age_sex = (
            df_analysis.groupby(["lagged_care_status", "age_group", "sex_group"])[
                moving_variable
            ]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        for _, row in moving_care_age_sex.iterrows():
            results_list.append(
                {
                    "Grouping": f"Lagged Care Status x Age x Sex ({row['sex_group']})",
                    "Category": row["lagged_care_status"],
                    "Subcategory": row["age_group"],
                    "Moving Rate (%)": row["mean"] * 100,
                    "N Observations": int(row["count"]),
                    "N Moved": int(row["sum"]),
                }
            )

        # 7. By lagged care status, education, and sex
        # (separate analyses for male and female)
        moving_care_edu_sex = (
            df_analysis.groupby(["lagged_care_status", "education_group", "sex_group"])[
                moving_variable
            ]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        for _, row in moving_care_edu_sex.iterrows():
            results_list.append(
                {
                    "Grouping": (
                        f"Lagged Care Status x Education x Sex ({row['sex_group']})"
                    ),
                    "Category": row["lagged_care_status"],
                    "Subcategory": row["education_group"],
                    "Moving Rate (%)": row["mean"] * 100,
                    "N Observations": int(row["count"]),
                    "N Moved": int(row["sum"]),
                }
            )

        # 8. By sex only (without care status)
        moving_sex = (
            df_analysis.groupby("sex_group")[moving_variable]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        for _, row in moving_sex.iterrows():
            results_list.append(
                {
                    "Grouping": f"Sex ({row['sex_group']})",
                    "Category": "All",
                    "Subcategory": "All",
                    "Moving Rate (%)": row["mean"] * 100,
                    "N Observations": int(row["count"]),
                    "N Moved": int(row["sum"]),
                }
            )

        # 9. By education only (without care status)
        moving_edu = (
            df_analysis.groupby("education_group")[moving_variable]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        for _, row in moving_edu.iterrows():
            results_list.append(
                {
                    "Grouping": "Education",
                    "Category": row["education_group"],
                    "Subcategory": "All",
                    "Moving Rate (%)": row["mean"] * 100,
                    "N Observations": int(row["count"]),
                    "N Moved": int(row["sum"]),
                }
            )

        # 10. Age 40-70: By lagged care status
        df_age_40_70 = df_analysis[
            (df_analysis["age"] >= AGE_40) & (df_analysis["age"] <= AGE_70)
        ].copy()
        if len(df_age_40_70) > 0:
            moving_care_40_70 = (
                df_age_40_70.groupby("lagged_care_status")[moving_variable]
                .agg(["mean", "count", "sum"])
                .reset_index()
            )
            for _, row in moving_care_40_70.iterrows():
                results_list.append(
                    {
                        "Grouping": "Lagged Care Status (Age 40-70)",
                        "Category": row["lagged_care_status"],
                        "Subcategory": "40-70",
                        "Moving Rate (%)": row["mean"] * 100,
                        "N Observations": int(row["count"]),
                        "N Moved": int(row["sum"]),
                    }
                )

            # 11. Age 40-70: By lagged care status and sex
            moving_care_sex_40_70 = (
                df_age_40_70.groupby(["lagged_care_status", "sex_group"])[
                    moving_variable
                ]
                .agg(["mean", "count", "sum"])
                .reset_index()
            )
            for _, row in moving_care_sex_40_70.iterrows():
                results_list.append(
                    {
                        "Grouping": (
                            f"Lagged Care Status x Sex ({row['sex_group']}) (Age 40-70)"
                        ),
                        "Category": row["lagged_care_status"],
                        "Subcategory": "40-70",
                        "Moving Rate (%)": row["mean"] * 100,
                        "N Observations": int(row["count"]),
                        "N Moved": int(row["sum"]),
                    }
                )

            # 12. Age 40-70: By lagged care status and education
            moving_care_edu_40_70 = (
                df_age_40_70.groupby(["lagged_care_status", "education_group"])[
                    moving_variable
                ]
                .agg(["mean", "count", "sum"])
                .reset_index()
            )
            for _, row in moving_care_edu_40_70.iterrows():
                results_list.append(
                    {
                        "Grouping": ("Lagged Care Status x Education (Age 40-70)"),
                        "Category": row["lagged_care_status"],
                        "Subcategory": row["education_group"],
                        "Moving Rate (%)": row["mean"] * 100,
                        "N Observations": int(row["count"]),
                        "N Moved": int(row["sum"]),
                    }
                )

        # Create results DataFrame
        results = pd.DataFrame(results_list)

        # Reorder results: group by gender (Female first, then Male)
        # within each grouping
        def extract_sex_from_grouping(grouping_str):
            """Extract sex from grouping string."""
            if "(Female)" in grouping_str:
                return "Female"
            if "(Male)" in grouping_str:
                return "Male"
            return "All"

        # Add a temporary column for sorting
        results["_sex_sort"] = results["Grouping"].apply(extract_sex_from_grouping)

        # Define sort order: Female = 0, Male = 1, All = 2
        sex_sort_order = {"Female": 0, "Male": 1, "All": 2}
        results["_sex_sort_num"] = results["_sex_sort"].map(sex_sort_order)

        # Define subcategory sort order for age groups
        age_order = {
            "30-39": 0,
            "40-49": 1,
            "50-59": 2,
            "60-69": 3,
            "70+": 4,
            "All": 5,
            "Missing": 6,
        }
        education_order = {"Low": 0, "High": 1, "All": 2, "Missing": 3}

        def get_subcategory_sort_num(row):
            """Get sort order for subcategory based on grouping type."""
            grouping = row["Grouping"]
            subcat = row["Subcategory"]

            if "Age" in grouping or "(Age 40-70)" in grouping:
                return age_order.get(subcat, 99)
            if "Education" in grouping:
                return education_order.get(subcat, 99)
            # Handle "40-70" as a special subcategory
            if subcat == "40-70":
                return 4  # Between 60-69 and 70+
            return 0

        results["_subcat_sort_num"] = results.apply(get_subcategory_sort_num, axis=1)

        # Define category sort order based on grouping type
        def get_category_sort_num(row):
            """Get sort order for category based on grouping type."""
            grouping = row["Grouping"]
            cat = row["Category"]

            # For education groupings, use education order
            if "Education" in grouping and cat in ("Low", "High"):
                return education_order.get(cat, 99)
            # For care status groupings, use care status order
            care_status_order = {
                "Caregiver": 0,
                "Non-caregiver": 1,
                "All": 2,
                "Missing": 3,
            }
            if cat in care_status_order:
                return care_status_order[cat]
            # For age groups
            if cat in age_order:
                return age_order[cat]
            return 99

        results["_cat_sort_num"] = results.apply(get_category_sort_num, axis=1)

        # Sort: first by Grouping, then by sex (Female first),
        # then by subcategory, then by category
        results = results.sort_values(
            ["Grouping", "_sex_sort_num", "_subcat_sort_num", "_cat_sort_num"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

        # Drop temporary sorting columns
        results = results.drop(
            columns=["_sex_sort", "_sex_sort_num", "_subcat_sort_num", "_cat_sort_num"]
        )

        # Add significance tests for each grouping that has caregivers
        # and non-caregivers
        def add_significance_to_results(results_df, df_analysis_data):
            """Add significance test columns comparing caregivers vs non-caregivers."""
            # Initialize columns
            results_df["Difference vs Non-Caregiver (pp)"] = "N/A"
            results_df["P-value"] = "N/A"

            # For each grouping, test if there are both caregivers and non-caregivers
            for grouping in results_df["Grouping"].unique():
                grouping_data = results_df[results_df["Grouping"] == grouping]

                # Check if this grouping has both caregivers and non-caregivers
                has_caregivers = (grouping_data["Category"] == "Caregiver").any()
                has_non_caregivers = (
                    grouping_data["Category"] == "Non-caregiver"
                ).any()

                if not (has_caregivers and has_non_caregivers):  # noqa: PLR1714
                    continue

                # For each subcategory (or "All"), test the difference
                for subcat in grouping_data["Subcategory"].unique():
                    subcat_data = grouping_data[grouping_data["Subcategory"] == subcat]

                    caregivers_row = subcat_data[subcat_data["Category"] == "Caregiver"]
                    non_caregivers_row = subcat_data[
                        subcat_data["Category"] == "Non-caregiver"
                    ]

                    if 0 in (len(caregivers_row), len(non_caregivers_row)):
                        continue

                    # Extract counts
                    n1 = int(caregivers_row["N Observations"].iloc[0])
                    x1 = int(caregivers_row["N Moved"].iloc[0])
                    n2 = int(non_caregivers_row["N Observations"].iloc[0])
                    x2 = int(non_caregivers_row["N Moved"].iloc[0])

                    # Calculate proportions
                    p1 = x1 / n1 if n1 > 0 else 0
                    p2 = x2 / n2 if n2 > 0 else 0

                    # Two-proportion z-test
                    with suppress(Exception):
                        z_stat, p_value = proportions_ztest(
                            [x1, x2], [n1, n2], alternative="two-sided"
                        )

                        # Calculate difference
                        diff = (p1 - p2) * 100  # Convert to percentage points

                        # Format significance
                        if p_value < P_VALUE_HIGHLY_SIGNIFICANT:
                            sig_level = "***"
                        elif p_value < P_VALUE_VERY_SIGNIFICANT:
                            sig_level = "**"
                        elif p_value < P_VALUE_SIGNIFICANT:
                            sig_level = "*"
                        elif p_value < P_VALUE_MARGINALLY_SIGNIFICANT:
                            sig_level = "."
                        else:
                            sig_level = ""

                        diff_str = f"{diff:.2f} pp{sig_level}"
                        pval_str = f"{p_value:.4f}"

                        # Update results for caregivers row
                        mask = (
                            (results_df["Grouping"] == grouping)
                            & (results_df["Subcategory"] == subcat)
                            & (results_df["Category"] == "Caregiver")
                        )
                        results_df.loc[mask, "Difference vs Non-Caregiver (pp)"] = (
                            diff_str
                        )
                        results_df.loc[mask, "P-value"] = pval_str

            return results_df

        # Add significance tests
        results = add_significance_to_results(results, df_analysis)

        # Ensure directory exists
        path_to_save_analysis.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        results.to_csv(path_to_save_analysis, index=False)
