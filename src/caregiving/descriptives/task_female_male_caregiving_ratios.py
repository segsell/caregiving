"""Create female-male ratios for caregiving by age groups."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD


@pytask.mark.descriptives
def task_female_male_caregiving_ratios(
    path_to_caregivers_sample: Path = BLD
    / "data"
    / "soep_structural_caregivers_sample.csv",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "descriptives"
    / "female_male_caregiving_ratios.tex",
) -> None:
    """Compute female-male ratios for caregiving by age groups.

    This task loads the SOEP caregivers sample and calculates female-male ratios
    for:
    1. Intensive caregiving (conditioning on intensive_care == True for both sexes)
    2. Light caregiving
    3. Overall caregiving (any_care)

    For each ratio, the analysis is performed on three age ranges:
    - Age 30 to <= 70
    - Age 21 to <= 100
    - Age 50 to <= 70

    The results are saved as a LaTeX table with ratios and observation counts.

    Parameters
    ----------
    path_to_caregivers_sample : Path
        Path to the SOEP caregivers sample CSV file.
    path_to_save_table : Path
        Path to save the LaTeX table.
    """
    # Load the caregivers sample
    df = pd.read_csv(path_to_caregivers_sample)

    # Define age ranges
    age_ranges = {
        "30-70": (30, 70),
        "21-100": (21, 100),
        "50-70": (50, 70),
        "40-70": (40, 70),
        "30-39": (30, 39, True),  # < 40 (exclusive upper bound)
        "40-49": (40, 49, True),  # < 50 (exclusive upper bound)
        "50-59": (50, 59, True),  # < 60 (exclusive upper bound)
        "60-69": (60, 69, True),  # < 70 (exclusive upper bound)
        "30-59": (30, 59, True),  # < 60 (exclusive upper bound)
        "40-59": (40, 59, True),  # < 60 (exclusive upper bound)
        "21-59": (21, 59, True),  # < 60 (exclusive upper bound)
    }

    # Define caregiving types
    care_types = {
        "Intensive care": {
            "condition": lambda x: x["intensive_care"] == 1,
            "description": "Intensive caregiving (intensive_care == 1)",
        },
        "Light care": {
            "condition": lambda x: x["light_care"] == 1,
            "description": "Light caregiving",
        },
        "Overall care": {
            "condition": lambda x: x["any_care"] == 1,
            "description": "Overall caregiving (any_care == 1)",
        },
    }

    # Store results
    results = []

    # Compute ratios for each care type and age range
    for care_name, care_info in care_types.items():
        for age_label, age_range_tuple in age_ranges.items():
            # Handle age ranges with exclusive upper bounds
            if len(age_range_tuple) == 3 and age_range_tuple[2]:
                age_min, age_max, exclusive = age_range_tuple
                # Filter by age range (exclusive upper bound: < age_max)
                df_age = df[(df["age"] >= age_min) & (df["age"] < age_max)].copy()
            else:
                age_min, age_max = age_range_tuple[:2]
                # Filter by age range (inclusive upper bound: <= age_max)
                df_age = df[(df["age"] >= age_min) & (df["age"] <= age_max)].copy()

            # Apply caregiving condition (exclude NaN values)
            # Check that the care variable is not NaN before applying condition
            care_var = None
            if care_name == "Intensive care":
                care_var = "intensive_care"
            elif care_name == "Light care":
                care_var = "light_care"
            elif care_name == "Overall care":
                care_var = "any_care"

            # Filter to non-missing care variable and sex
            df_filtered = df_age[
                df_age[care_var].notna() & df_age["sex"].notna()
            ].copy()

            # Apply caregiving condition
            mask = care_info["condition"](df_filtered)
            df_care = df_filtered[mask].copy()

            # Count by sex (0 = male, 1 = female)
            counts = df_care["sex"].value_counts().sort_index()
            n_female = counts.get(1, 0)
            n_male = counts.get(0, 0)

            # Calculate ratio (female/male)
            if n_male > 0:
                ratio = n_female / n_male
            else:
                ratio = float("inf") if n_female > 0 else float("nan")

            # Calculate female share percentage
            total = n_female + n_male
            if total > 0:
                female_share = (n_female / total) * 100
            else:
                female_share = float("nan")

            results.append(
                {
                    "Care type": care_name,
                    "Age range": age_label,
                    "Female": n_female,
                    "Male": n_male,
                    "Ratio (F/M)": ratio,
                    "Female share (%)": female_share,
                }
            )

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Format ratio for LaTeX (handle infinity and NaN)
    def format_ratio(val):
        if pd.isna(val):
            return "---"
        if val == float("inf"):
            return r"$\infty$"
        return f"{val:.2f}"

    results_df["Ratio (F/M)"] = results_df["Ratio (F/M)"].apply(format_ratio)

    # Format female share percentage
    def format_share(val):
        if pd.isna(val):
            return "---"
        return f"{val:.1f}\\%"

    results_df["Female share (%)"] = results_df["Female share (%)"].apply(format_share)

    # Create LaTeX table
    # Format numeric columns for LaTeX
    results_df_latex = results_df.copy()
    results_df_latex["Female"] = results_df_latex["Female"].apply(
        lambda x: f"{int(x):,}"
    )
    results_df_latex["Male"] = results_df_latex["Male"].apply(lambda x: f"{int(x):,}")

    latex_table = results_df_latex.to_latex(
        index=False,
        escape=False,
        column_format="lcccccc",
    )

    # Add booktabs formatting
    latex_table = latex_table.replace(
        "\\begin{tabular}{lcccccc}",
        "\\begin{tabular}{lcccccc}\n\\toprule",
    )
    # Ensure midrule and bottomrule are present
    if "\\midrule" not in latex_table:
        # Find the line after header and add midrule
        lines = latex_table.split("\n")
        for i, line in enumerate(lines):
            if "Care type" in line and "Age range" in line:
                lines.insert(i + 2, "\\midrule")
                break
        latex_table = "\n".join(lines)

    if "\\bottomrule" not in latex_table:
        latex_table = latex_table.replace(
            "\\end{tabular}", "\\bottomrule\n\\end{tabular}"
        )

    # Ensure directory exists
    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)

    # Save LaTeX table
    with open(path_to_save_table, "w", encoding="utf-8") as f:
        f.write(latex_table)


@pytask.mark.descriptives
def task_explore_sample_attrition(
    path_to_caregivers_sample: Path = BLD
    / "data"
    / "soep_structural_caregivers_sample.csv",
    path_to_save_report: Annotated[Path, Product] = BLD
    / "descriptives"
    / "sample_attrition_analysis.md",
) -> None:
    """Explore sample attrition and non-response patterns.

    This task analyzes the caregivers sample to:
    1. Verify the distribution of `any_care` values (0, 1, NaN)
    2. Propose methods to check differential attrition/non-response
       between caregivers and non-caregivers

    The task creates a markdown report documenting:
    - Distribution of `any_care` values
    - Proposed methods for checking differential attrition
    - Summary statistics on missing data patterns

    Parameters
    ----------
    path_to_caregivers_sample : Path
        Path to the SOEP caregivers sample CSV file.
    path_to_save_report : Path
        Path to save the markdown report with analysis and proposals.
    """
    # Load the caregivers sample
    df = pd.read_csv(path_to_caregivers_sample)

    # Verify any_care variable values
    any_care_values = df["any_care"].value_counts(dropna=False)
    any_care_unique = df["any_care"].unique()

    # Create report content
    report_lines = [
        "# Sample Attrition and Non-Response Analysis",
        "",
        "## 1. Verification of `any_care` Variable",
        "",
        "### Unique Values:",
        f"- Values found: {sorted([v for v in any_care_unique if pd.notna(v)])}",
        f"- NaN count: {df['any_care'].isna().sum():,}",
        "",
        "### Distribution:",
        "```",
        str(any_care_values),
        "```",
        "",
        "### Verification:",
        f"- Contains 0: {0 in any_care_unique}",
        f"- Contains 1: {1 in any_care_unique}",
        f"- Contains NaN: {df['any_care'].isna().any()}",
        "",
        "## 2. Proposed Methods for Checking Differential Attrition",
        "",
        "### Method 1: Panel Retention Rates by Care Status",
        "",
        "**Description**: Compare how many caregivers vs non-caregivers continue",
        "in the panel over time.",
        "",
        "**Implementation approach**:",
        "- For each person, identify their care status in period t",
        "- Track whether they appear in period t+1",
        "- Calculate retention rates separately for:",
        "  - Those with `any_care == 1` in period t (caregivers)",
        "  - Those with `any_care == 0` in period t (non-caregivers)",
        "  - Those with `any_care == NaN` in period t (missing/not asked)",
        "- Compare retention rates across these groups",
        "",
        "**Expected output**:",
        "- Retention rate by lagged care status",
        "- Statistical tests for differences in retention",
        "",
        "---",
        "",
        "### Method 2: Missing Data Patterns Conditional on Lagged Care Status",
        "",
        "**Description**: Compare rates of missing `any_care` for those who were",
        "caregivers vs non-caregivers in previous periods.",
        "",
        "**Implementation approach**:",
        "- Create lagged `any_care` variable (care status in previous period)",
        "- For each period, calculate:",
        "  - P(`any_care` is NaN | lagged `any_care` == 1)",
        "  - P(`any_care` is NaN | lagged `any_care` == 0)",
        "- Test if missingness is independent of lagged care status",
        "",
        "**Expected output**:",
        "- Conditional probabilities of missingness",
        "- Tests for differential non-response",
        "",
        "---",
        "",
        "### Method 3: Wave-to-Wave Retention Analysis",
        "",
        "**Description**: Compare retention rates between consecutive survey waves",
        "for caregivers vs non-caregivers.",
        "",
        "**Implementation approach**:",
        "- Identify consecutive survey years (syear) for each person",
        "- For each transition from year t to t+1:",
        "  - Check if person appears in both years",
        "  - If yes, check if `any_care` is observed in both",
        "- Calculate retention rates by:",
        "  - Care status in year t",
        "  - Age group",
        "  - Other observable characteristics",
        "",
        "**Expected output**:",
        "- Wave-to-wave retention rates by care status",
        "- Retention rates by age and other covariates",
        "",
        "---",
        "",
        "### Method 4: Covariate Balance Analysis",
        "",
        "**Description**: Compare observable characteristics of those with missing",
        "`any_care` who were previously caregivers vs non-caregivers.",
        "",
        "**Implementation approach**:",
        "- Among observations with `any_care == NaN` in period t:",
        "  - Compare mean characteristics (age, education, health, etc.)",
        "  - Separate by lagged care status (caregiver vs non-caregiver)",
        "- Test if missingness is related to observable characteristics",
        "- Check if relationship differs by prior care status",
        "",
        "**Expected output**:",
        "- Mean characteristics by missingness and lagged care status",
        "- Tests for covariate balance",
        "",
        "---",
        "",
        "### Method 5: Transition Probabilities to Missing State",
        "",
        "**Description**: Analyze transitions from observed states (caregiver/non-caregiver)",
        "to missing state.",
        "",
        "**Implementation approach**:",
        "- Create transition matrix:",
        "  - From: `any_care` in period t (0, 1, or NaN)",
        "  - To: `any_care` in period t+1 (0, 1, or NaN)",
        "- Calculate transition probabilities:",
        "  - P(missing in t+1 | caregiver in t)",
        "  - P(missing in t+1 | non-caregiver in t)",
        "  - P(missing in t+1 | missing in t)",
        "- Compare these probabilities",
        "",
        "**Expected output**:",
        "- Transition probability matrix",
        "- Tests for differential attrition",
        "",
        "---",
        "",
        "### Method 6: Survival Analysis of Panel Participation",
        "",
        "**Description**: Use survival analysis to model time until dropout,",
        "conditioning on care status.",
        "",
        "**Implementation approach**:",
        "- Define 'event' as last observed period in panel",
        "- Use care status as time-varying covariate",
        "- Estimate hazard of dropout by care status",
        "- Control for other observable characteristics",
        "",
        "**Expected output**:",
        "- Hazard ratios for dropout by care status",
        "- Survival curves by care status",
        "",
        "---",
        "",
        "## 3. Summary Statistics",
        "",
        "### Overall Missing Data:",
        f"- Total observations: {len(df):,}",
        f"- Missing `any_care`: {df['any_care'].isna().sum():,}",
        f"- Missing rate: {df['any_care'].isna().mean() * 100:.2f}%",
        "",
        "### By Care Status (when observed):",
    ]

    # Add statistics by care status
    if 0 in any_care_unique:
        n_non_caregivers = (df["any_care"] == 0).sum()
        report_lines.append(f"- Non-caregivers (`any_care == 0`): {n_non_caregivers:,}")

    if 1 in any_care_unique:
        n_caregivers = (df["any_care"] == 1).sum()
        report_lines.append(f"- Caregivers (`any_care == 1`): {n_caregivers:,}")

    report_lines.extend(
        [
            "",
            "### Panel Structure:",
            f"- Unique persons (pid): {df['pid'].nunique() if 'pid' in df.columns else 'N/A'}",
            f"- Unique survey years: {df['syear'].nunique() if 'syear' in df.columns else 'N/A'}",
            "",
            "## 4. Next Steps",
            "",
            "The methods proposed above can be implemented to:",
            "1. Quantify the extent of differential attrition",
            "2. Test whether attrition is systematically related to care status",
            "3. Inform imputation strategies if needed",
            "4. Assess potential bias in estimates",
            "",
        ]
    )

    # Ensure directory exists
    path_to_save_report.parent.mkdir(parents=True, exist_ok=True)

    # Save report
    with open(path_to_save_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


@pytask.mark.descriptives
def task_panel_retention_rates_by_care_status(
    path_to_estimation_sample: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_save_retention: Annotated[Path, Product] = BLD
    / "descriptives"
    / "panel_retention_rates_by_care_status.csv",
) -> None:
    """Calculate panel retention rates by care status (Method 1).

    This task implements Method 1: Panel Retention Rates by Care Status.
    It compares retention rates for caregivers vs non-caregivers over time
    by tracking whether individuals appear in subsequent periods by care status.

    Note: Uses the estimation sample which has proper panel structure
    (multiple observations per person).

    Parameters
    ----------
    path_to_estimation_sample : Path
        Path to the SOEP estimation sample CSV file (has panel structure).
    path_to_save_retention : Path
        Path to save the retention rates CSV file.
    """
    # Load the estimation sample (has panel structure)
    df = pd.read_csv(path_to_estimation_sample)

    # The caregivers sample may have pid as "Unnamed: 0" (from index) or as "pid"
    if "pid" not in df.columns and "Unnamed: 0" in df.columns:
        # Rename Unnamed: 0 to pid (it's likely the pid from the index)
        df = df.rename(columns={"Unnamed: 0": "pid"})
    elif "pid" not in df.columns:
        raise ValueError(
            "pid column not found in caregivers sample. "
            "Expected 'pid' or 'Unnamed: 0' column."
        )

    # Ensure we have syear column
    if "syear" not in df.columns:
        raise ValueError("syear column not found in caregivers sample")

    # Ensure data is sorted by pid and syear
    df = df.sort_values(["pid", "syear"]).copy()

    # Create lagged care status
    df["lagged_any_care"] = df.groupby("pid")["any_care"].shift(1)

    # Identify if person appears in next period
    # Create next year for each person
    df["next_syear"] = df.groupby("pid")["syear"].shift(-1)
    df["expected_next_syear"] = df["syear"] + 1

    # Check if person appears in next period (retained)
    # Person is retained if next_syear equals expected_next_syear
    df["retained"] = (df["next_syear"] == df["expected_next_syear"]).astype(int)

    # Only analyze transitions where we have lagged care status
    # (i.e., not the first observation for each person)
    df_transitions = df[df["lagged_any_care"].notna()].copy()

    # Create care status categories for lagged care
    def categorize_care_status(val):
        if pd.isna(val):
            return "Missing"
        if val == 1:
            return "Caregiver"
        if val == 0:
            return "Non-caregiver"
        return "Unknown"

    df_transitions["lagged_care_status"] = df_transitions["lagged_any_care"].apply(
        categorize_care_status
    )

    # Calculate retention rates by lagged care status
    retention_by_care = (
        df_transitions.groupby("lagged_care_status")["retained"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    retention_by_care.columns = [
        "Lagged Care Status",
        "Retention Rate",
        "N Observations",
        "N Retained",
    ]
    retention_by_care["Retention Rate"] = retention_by_care["Retention Rate"] * 100

    # Also calculate by current care status (for comparison)
    df_transitions["current_care_status"] = df_transitions["any_care"].apply(
        categorize_care_status
    )

    # Calculate retention rates by current care status
    retention_by_current_care = (
        df_transitions.groupby("current_care_status")["retained"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    retention_by_current_care.columns = [
        "Current Care Status",
        "Retention Rate",
        "N Observations",
        "N Retained",
    ]
    retention_by_current_care["Retention Rate"] = (
        retention_by_current_care["Retention Rate"] * 100
    )

    # Combine results
    results = pd.DataFrame(
        {
            "Analysis Type": ["Lagged Care Status"] * len(retention_by_care)
            + ["Current Care Status"] * len(retention_by_current_care),
            "Care Status": list(retention_by_care["Lagged Care Status"])
            + list(retention_by_current_care["Current Care Status"]),
            "Retention Rate (%)": list(retention_by_care["Retention Rate"])
            + list(retention_by_current_care["Retention Rate"]),
            "N Observations": list(retention_by_care["N Observations"])
            + list(retention_by_current_care["N Observations"]),
            "N Retained": list(retention_by_care["N Retained"])
            + list(retention_by_current_care["N Retained"]),
        }
    )

    # Ensure directory exists
    path_to_save_retention.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    results.to_csv(path_to_save_retention, index=False)


@pytask.mark.descriptives
def task_wave_to_wave_retention_analysis(
    path_to_estimation_sample: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_save_retention: Annotated[Path, Product] = BLD
    / "descriptives"
    / "wave_to_wave_retention_analysis.csv",
) -> None:
    """Calculate wave-to-wave retention rates by care status and covariates (Method 3).

    This task implements Method 3: Wave-to-Wave Retention Analysis.
    It analyzes retention between consecutive survey waves and calculates
    retention rates by care status, age, and other covariates.

    Note: Uses the estimation sample which has proper panel structure
    (multiple observations per person).

    Parameters
    ----------
    path_to_estimation_sample : Path
        Path to the SOEP estimation sample CSV file (has panel structure).
    path_to_save_retention : Path
        Path to save the retention analysis CSV file.
    """
    # Load the estimation sample (has panel structure)
    df = pd.read_csv(path_to_estimation_sample)

    # The estimation sample should have pid as a column
    if "pid" not in df.columns and "Unnamed: 0" in df.columns:
        # Check if Unnamed: 0 might be pid
        df_temp = df[["Unnamed: 0", "syear"]].drop_duplicates()
        # If Unnamed: 0 has many unique values similar to expected pid count, use it
        if df_temp["Unnamed: 0"].nunique() > len(df) * 0.5:
            df = df.rename(columns={"Unnamed: 0": "pid"})

    if "pid" not in df.columns:
        raise ValueError(
            "pid column not found in estimation sample. " "Expected 'pid' column."
        )

    # Ensure we have syear column
    if "syear" not in df.columns:
        raise ValueError("syear column not found in estimation sample")

    # Check if we have panel structure (multiple observations per pid)
    pid_counts = df.groupby("pid").size()
    if (pid_counts > 1).sum() == 0:
        raise ValueError(
            "Estimation sample does not have panel structure. "
            "Each person appears only once. Cannot calculate retention rates."
        )

    # Ensure data is sorted by pid and syear
    df = df.sort_values(["pid", "syear"]).copy()

    # Create lagged care status
    df["lagged_any_care"] = df.groupby("pid")["any_care"].shift(1)

    # Identify if person appears in next wave
    df["next_syear"] = df.groupby("pid")["syear"].shift(-1)
    df["expected_next_syear"] = df["syear"] + 1
    df["retained"] = (df["next_syear"] == df["expected_next_syear"]).astype(int)

    # Only analyze transitions where we have lagged care status
    df_transitions = df[df["lagged_any_care"].notna()].copy()

    # Create care status categories
    def categorize_care_status(val):
        if pd.isna(val):
            return "Missing"
        if val == 1:
            return "Caregiver"
        if val == 0:
            return "Non-caregiver"
        return "Unknown"

    df_transitions["lagged_care_status"] = df_transitions["lagged_any_care"].apply(
        categorize_care_status
    )

    # Create age groups
    def categorize_age(age):
        if pd.isna(age):
            return "Missing"
        if age < 40:
            return "30-39"
        if age < 50:
            return "40-49"
        if age < 60:
            return "50-59"
        if age < 70:
            return "60-69"
        return "70+"

    df_transitions["age_group"] = df_transitions["age"].apply(categorize_age)

    # Create education groups
    def categorize_education(edu):
        if pd.isna(edu):
            return "Missing"
        if edu == 0:
            return "Low"
        if edu == 1:
            return "High"
        return "Unknown"

    df_transitions["education_group"] = df_transitions["education"].apply(
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

    df_transitions["sex_group"] = df_transitions["sex"].apply(categorize_sex)

    # Calculate retention rates by different groupings
    results_list = []

    # 1. By lagged care status only
    retention_care = (
        df_transitions.groupby("lagged_care_status")["retained"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    for _, row in retention_care.iterrows():
        results_list.append(
            {
                "Grouping": "Lagged Care Status",
                "Category": row["lagged_care_status"],
                "Subcategory": "All",
                "Retention Rate (%)": row["mean"] * 100,
                "N Observations": int(row["count"]),
                "N Retained": int(row["sum"]),
            }
        )

    # 2. By lagged care status and age group
    retention_care_age = (
        df_transitions.groupby(["lagged_care_status", "age_group"])["retained"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    for _, row in retention_care_age.iterrows():
        results_list.append(
            {
                "Grouping": "Lagged Care Status × Age",
                "Category": row["lagged_care_status"],
                "Subcategory": row["age_group"],
                "Retention Rate (%)": row["mean"] * 100,
                "N Observations": int(row["count"]),
                "N Retained": int(row["sum"]),
            }
        )

    # 3. By lagged care status and education
    retention_care_edu = (
        df_transitions.groupby(["lagged_care_status", "education_group"])["retained"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    for _, row in retention_care_edu.iterrows():
        results_list.append(
            {
                "Grouping": "Lagged Care Status × Education",
                "Category": row["lagged_care_status"],
                "Subcategory": row["education_group"],
                "Retention Rate (%)": row["mean"] * 100,
                "N Observations": int(row["count"]),
                "N Retained": int(row["sum"]),
            }
        )

    # 4. By lagged care status and sex
    retention_care_sex = (
        df_transitions.groupby(["lagged_care_status", "sex_group"])["retained"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    for _, row in retention_care_sex.iterrows():
        results_list.append(
            {
                "Grouping": "Lagged Care Status × Sex",
                "Category": row["lagged_care_status"],
                "Subcategory": row["sex_group"],
                "Retention Rate (%)": row["mean"] * 100,
                "N Observations": int(row["count"]),
                "N Retained": int(row["sum"]),
            }
        )

    # 5. By age group only (for reference)
    retention_age = (
        df_transitions.groupby("age_group")["retained"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    for _, row in retention_age.iterrows():
        results_list.append(
            {
                "Grouping": "Age Group",
                "Category": row["age_group"],
                "Subcategory": "All",
                "Retention Rate (%)": row["mean"] * 100,
                "N Observations": int(row["count"]),
                "N Retained": int(row["sum"]),
            }
        )

    # Create results DataFrame
    results = pd.DataFrame(results_list)

    # Ensure directory exists
    path_to_save_retention.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    results.to_csv(path_to_save_retention, index=False)
