"""Plot differences in labor supply for caregiving leave vs baseline counterfactual.

Caregiving leave with job retention and baseline scenario.
"""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.counterfactual.plotting_helpers import (
    get_age_at_first_event,
    get_distinct_colors,
    plot_multi_line_differences_by_group,
    plot_three_line_differences,
)
from caregiving.counterfactual.plotting_utils import (
    calculate_additional_outcomes,
    calculate_outcomes,
    calculate_working_hours_weekly,
    create_outcome_columns,
    ensure_agent_period,
    merge_and_compute_differences,
    prepare_dataframes_for_comparison,
)
from caregiving.counterfactual.task_plot_labor_supply_differences import (
    _add_distance_to_first_care,
)
from caregiving.model.shared import DEAD, INFORMAL_CARE

# ============================================================================
# Distance calculation helpers
# ============================================================================


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_by_distance(  # noqa: PLR0915, E501
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "matched_differences_by_distance_caregiving_leave_with_job_retention.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (caregiving leave - baseline).

    Averages by distance.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care from caregiving leave, attach to merged.
      6) Average diffs by distance and plot three series.

    """
    # Load and prepare data
    df_cg, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_caregiving_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate outcomes
    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    specs = pickle.load(path_to_specs.open("rb"))
    cg_additional = calculate_additional_outcomes(df_cg, specs)
    baseline_additional = calculate_additional_outcomes(df_baseline, specs)
    cg_outcomes.update(cg_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns
    o_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    c_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Merge and compute differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_o"] - merged["work_c"]
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]
    merged["diff_gross_labor_income"] = (
        merged["gross_labor_income_o"] - merged["gross_labor_income_c"]
    )
    merged["diff_savings"] = merged["savings_o"] - merged["savings_c"]
    merged["diff_wealth"] = merged["wealth_o"] - merged["wealth_c"]
    merged["diff_savings_rate"] = merged["savings_rate_o"] - merged["savings_rate_c"]

    # Compute distance in caregiving leave and attach
    df_cg_dist = _add_distance_to_first_care(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care", observed=False)[
            [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Plot
    plot_three_line_differences(
        prof=prof,
        x_col="distance_to_first_care",
        path_to_plot=path_to_plot,
        xlabel="Year relative to start of first care spell",
        window=window,
    )


def _process_data_for_age_at_first_care(
    path_to_caregiving_leave_data: Path,
    path_to_baseline_data: Path,
    path_to_specs: Path,
    outcome_name: str,
    ever_caregivers: bool,
    ever_care_demand: bool,
    ages_at_first_care: list[int],
    window: int,
) -> pd.DataFrame:
    """Helper function to process data for age_at_first_care tasks.

    Args:
        path_to_caregiving_leave_data: Path to caregiving leave data
        path_to_baseline_data: Path to baseline data
        path_to_specs: Path to specs file
        outcome_name: Name of outcome to calculate (e.g., 'pt', 'ft', 'work', etc.)
        ever_caregivers: Filter to ever-caregivers
        ever_care_demand: Filter to ever-care-demand
        ages_at_first_care: List of ages at first care
        window: Window size for distance filtering

    Returns:
        Profile DataFrame with aggregated differences by distance and age_at_first_care
    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_caregiving_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate only the outcome needed (plus dependencies)
    o_outcomes = {}
    c_outcomes = {}

    # Basic outcomes from calculate_outcomes
    if outcome_name in ("work", "ft", "pt", "job_offer", "care"):
        full_outcomes_o = calculate_outcomes(df_o, choice_set_type="original")
        full_outcomes_c = calculate_outcomes(df_c, choice_set_type="original")
        o_outcomes[outcome_name] = full_outcomes_o[outcome_name]
        c_outcomes[outcome_name] = full_outcomes_c[outcome_name]
    elif outcome_name == "hours_weekly":
        specs = pickle.load(path_to_specs.open("rb"))
        o_outcomes["hours_weekly"] = calculate_working_hours_weekly(
            df_o, specs, choice_set_type="original"
        )
        c_outcomes["hours_weekly"] = calculate_working_hours_weekly(
            df_c, specs, choice_set_type="original"
        )
    elif outcome_name in (
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ):
        specs = pickle.load(path_to_specs.open("rb"))
        o_additional = calculate_additional_outcomes(df_o, specs)
        c_additional = calculate_additional_outcomes(df_c, specs)
        o_outcomes[outcome_name] = o_additional[outcome_name]
        c_outcomes[outcome_name] = c_additional[outcome_name]
    else:
        raise ValueError(f"Unknown outcome_name: {outcome_name}")

    # Compute distance and age at first care from original (before merging)
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_o["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_o, caregiving_mask, "age_at_first_care"
    )

    del df_o_dist

    # Create outcome columns
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_o.columns:
        o_cols["education"] = df_o["education"].values
    if "caregiving_type" in df_o.columns:
        o_cols["caregiving_type"] = df_o["caregiving_type"].values

    # Clean up original dataframes before merging (keep only what we need)
    del df_o, df_c

    # Merge and compute differences
    merged = merge_and_compute_differences(o_cols, c_cols, [outcome_name])

    # Clean up intermediate dataframes
    del o_cols, c_cols

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    del dist_map, first_care_with_age

    # Filter to specific ages at first care
    merged = merged[merged["age_at_first_care"].isin(ages_at_first_care)]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # # Return full merged dataframe (not aggregated) so calling functions can do
    #  nested splits
    return merged


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_pt_by_age_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_part_time_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences for part-time work by age at first care."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Get merged dataframe from helper
    merged = _process_data_for_age_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="pt",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        ages_at_first_care=ages_at_first_care,
        window=window,
    )

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    colors = get_distinct_colors(len(ages_at_first_care))

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_at_first_care
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_at_first_care"], observed=False
                )[["diff_pt"]]
                .mean()
                .reset_index()
                .sort_values(["age_at_first_care", "distance_to_first_care"])
            )

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_at_first_care",
                diff_col="diff_pt",
                groups=ages_at_first_care,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Proportion Part-Time Working\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_ft_by_age_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_full_time_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences for full-time work by age at first care."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Get merged dataframe from helper
    merged = _process_data_for_age_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="ft",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        ages_at_first_care=ages_at_first_care,
        window=window,
    )

    colors = get_distinct_colors(len(ages_at_first_care))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_at_first_care
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_at_first_care"], observed=False
                )[["diff_ft"]]
                .mean()
                .reset_index()
                .sort_values(["age_at_first_care", "distance_to_first_care"])
            )

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_at_first_care",
                diff_col="diff_ft",
                groups=ages_at_first_care,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Proportion Full-Time Working\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_work_by_age_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_employment_rate_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences for employment rate by age at first care."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Get merged dataframe from helper
    merged = _process_data_for_age_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="work",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        ages_at_first_care=ages_at_first_care,
        window=window,
    )

    colors = get_distinct_colors(len(ages_at_first_care))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_at_first_care
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_at_first_care"], observed=False
                )[["diff_work"]]
                .mean()
                .reset_index()
                .sort_values(["age_at_first_care", "distance_to_first_care"])
            )

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_at_first_care",
                diff_col="diff_work",
                groups=ages_at_first_care,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Proportion Working\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_job_offer_by_age_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_job_offer_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences for job offer by age at first care."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Get merged dataframe from helper
    merged = _process_data_for_age_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="job_offer",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        ages_at_first_care=ages_at_first_care,
        window=window,
    )

    colors = get_distinct_colors(len(ages_at_first_care))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_at_first_care
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_at_first_care"], observed=False
                )[["diff_job_offer"]]
                .mean()
                .reset_index()
                .sort_values(["age_at_first_care", "distance_to_first_care"])
            )

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_at_first_care",
                diff_col="diff_job_offer",
                groups=ages_at_first_care,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Job Offer Probability\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_hours_weekly_by_age_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_working_hours_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences for working hours by age at first care."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Get merged dataframe from helper
    merged = _process_data_for_age_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="hours_weekly",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        ages_at_first_care=ages_at_first_care,
        window=window,
    )

    colors = get_distinct_colors(len(ages_at_first_care))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_at_first_care
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_at_first_care"], observed=False
                )[["diff_hours_weekly"]]
                .mean()
                .reset_index()
                .sort_values(["age_at_first_care", "distance_to_first_care"])
            )

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_at_first_care",
                diff_col="diff_hours_weekly",
                groups=ages_at_first_care,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Weekly Working Hours\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_gross_labor_income_by_age_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_gross_labor_income_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences for gross labor income by age at first care."""  # noqa: E501
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Get merged dataframe from helper
    merged = _process_data_for_age_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="gross_labor_income",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        ages_at_first_care=ages_at_first_care,
        window=window,
    )

    colors = get_distinct_colors(len(ages_at_first_care))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_at_first_care
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_at_first_care"], observed=False
                )[["diff_gross_labor_income"]]
                .mean()
                .reset_index()
                .sort_values(["age_at_first_care", "distance_to_first_care"])
            )

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_at_first_care",
                diff_col="diff_gross_labor_income",
                groups=ages_at_first_care,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Gross Labor Income (Monthly)\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_savings_by_age_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_savings_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences for savings by age at first care."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Get merged dataframe from helper
    merged = _process_data_for_age_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="savings",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        ages_at_first_care=ages_at_first_care,
        window=window,
    )

    colors = get_distinct_colors(len(ages_at_first_care))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_at_first_care
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_at_first_care"], observed=False
                )[["diff_savings"]]
                .mean()
                .reset_index()
                .sort_values(["age_at_first_care", "distance_to_first_care"])
            )

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_at_first_care",
                diff_col="diff_savings",
                groups=ages_at_first_care,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Savings Decision\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_wealth_by_age_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_wealth_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences for wealth by age at first care."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Get merged dataframe from helper
    merged = _process_data_for_age_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="wealth",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        ages_at_first_care=ages_at_first_care,
        window=window,
    )

    colors = get_distinct_colors(len(ages_at_first_care))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_at_first_care
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_at_first_care"], observed=False
                )[["diff_wealth"]]
                .mean()
                .reset_index()
                .sort_values(["age_at_first_care", "distance_to_first_care"])
            )

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_at_first_care",
                diff_col="diff_wealth",
                groups=ages_at_first_care,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Wealth at Beginning of Period\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_savings_rate_by_age_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_savings_rate_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences for savings rate by age at first care."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    prof = _process_data_for_age_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="savings_rate",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        ages_at_first_care=ages_at_first_care,
        window=window,
    )

    colors = get_distinct_colors(len(ages_at_first_care))

    plot_multi_line_differences_by_group(
        prof=prof,
        x_col="distance_to_first_care",
        group_col="age_at_first_care",
        diff_col="diff_savings_rate",
        groups=ages_at_first_care,
        colors=colors,
        path_to_plot=path_to_plot,
        xlabel="Year relative to start of first care spell",
        ylabel="Savings Rate\nDeviation from Baseline",
        window=window,
        legend_title="Age at first care",
    )


# NOTE: This task has been split into individual tasks per outcome to reduce
# memory usage. See tasks below starting with task_plot_matched_differences_*
# _by_age_bins_at_first_care
@pytask.mark.skip()
@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_by_age_bins_at_first_care_deprecated(
    # noqa: E501
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_part_time_by_age_bins_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_full_time_by_age_bins_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_employment_rate_by_age_bins_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_job_offer_by_age_bins_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_working_hours_by_age_bins_at_first_care.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """DEPRECATED: Split into individual tasks per outcome to reduce memory usage."""


# ============================================================================
# Split tasks by outcome type for age_bins_at_first_care
# ============================================================================


def _process_data_for_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path,
    path_to_baseline_data: Path,
    path_to_specs: Path,
    outcome_name: str,
    ever_caregivers: bool,
    ever_care_demand: bool,
    min_age: int,
    max_age: int,
    bin_width: int,
    window: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Helper function to process data for age_bins_at_first_care tasks.

    Args:
        path_to_caregiving_leave_data: Path to caregiving leave data
        path_to_baseline_data: Path to baseline data
        path_to_specs: Path to specs file
        outcome_name: Name of outcome to calculate (e.g., 'pt', 'ft', 'work', etc.)
        ever_caregivers: Filter to ever-caregivers
        ever_care_demand: Filter to ever-care-demand
        min_age: Minimum age for binning
        max_age: Maximum age for binning
        bin_width: Width of age bins
        window: Window size for distance filtering

    Returns:
        Tuple of (profile DataFrame, list of unique bin labels in order)
    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_caregiving_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate only the outcome needed (plus dependencies)
    o_outcomes = {}
    c_outcomes = {}

    # Basic outcomes from calculate_outcomes
    if outcome_name in ("work", "ft", "pt", "job_offer", "care"):
        full_outcomes_o = calculate_outcomes(df_o, choice_set_type="original")
        full_outcomes_c = calculate_outcomes(df_c, choice_set_type="original")
        o_outcomes[outcome_name] = full_outcomes_o[outcome_name]
        c_outcomes[outcome_name] = full_outcomes_c[outcome_name]
    elif outcome_name == "hours_weekly":
        specs = pickle.load(path_to_specs.open("rb"))
        o_outcomes["hours_weekly"] = calculate_working_hours_weekly(
            df_o, specs, choice_set_type="original"
        )
        c_outcomes["hours_weekly"] = calculate_working_hours_weekly(
            df_c, specs, choice_set_type="original"
        )
    elif outcome_name in (
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ):
        specs = pickle.load(path_to_specs.open("rb"))
        o_additional = calculate_additional_outcomes(df_o, specs)
        c_additional = calculate_additional_outcomes(df_c, specs)
        o_outcomes[outcome_name] = o_additional[outcome_name]
        c_outcomes[outcome_name] = c_additional[outcome_name]
    else:
        raise ValueError(f"Unknown outcome_name: {outcome_name}")

    # Compute distance and age at first care from original (before merging)
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_o["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_o, caregiving_mask, "age_at_first_care"
    )

    del df_o_dist

    # Create outcome columns
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_o.columns:
        o_cols["education"] = df_o["education"].values
    if "caregiving_type" in df_o.columns:
        o_cols["caregiving_type"] = df_o["caregiving_type"].values

    # Clean up original dataframes before merging (keep only what we need)
    del df_o, df_c

    # Merge and compute differences
    merged = merge_and_compute_differences(o_cols, c_cols, [outcome_name])

    # Clean up intermediate dataframes
    del o_cols, c_cols

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    del dist_map, first_care_with_age

    # Filter to age range
    merged = merged[
        (merged["age_at_first_care"] >= min_age)
        & (merged["age_at_first_care"] <= max_age)
    ]

    # Create age bins
    merged["age_bin_start"] = (
        (merged["age_at_first_care"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Get unique age bins in order (sorted by bin start) before returning
    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )

    # # Return full merged dataframe (not aggregated) so calling functions can do
    #  nested splits
    return merged, unique_bins


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_pt_by_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_part_time_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for part-time work by age bins at first care."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="pt",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[["diff_pt"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_bin_label",
                diff_col="diff_pt",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Proportion Part-Time Working\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_ft_by_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_full_time_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for full-time work by age bins at first care."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="ft",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[["diff_ft"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_bin_label",
                diff_col="diff_ft",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Proportion Full-Time Working\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_work_by_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_employment_rate_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for employment rate by age bins at first care."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="work",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[["diff_work"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_bin_label",
                diff_col="diff_work",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Proportion Working\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_job_offer_by_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_job_offer_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for job offer by age bins at first care."""
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="job_offer",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[["diff_job_offer"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_bin_label",
                diff_col="diff_job_offer",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Job Offer Probability\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_hours_weekly_by_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_working_hours_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for working hours by age bins at first care."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="hours_weekly",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[["diff_hours_weekly"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_bin_label",
                diff_col="diff_hours_weekly",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Weekly Working Hours\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_gross_labor_income_by_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_gross_labor_income_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for gross labor income by age bins at first care."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="gross_labor_income",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[["diff_gross_labor_income"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_bin_label",
                diff_col="diff_gross_labor_income",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Gross Labor Income (Monthly)\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_savings_by_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_savings_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for savings by age bins at first care."""
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="savings",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[["diff_savings"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_bin_label",
                diff_col="diff_savings",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Savings Decision\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_wealth_by_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_wealth_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for wealth by age bins at first care."""
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="wealth",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[["diff_wealth"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_bin_label",
                diff_col="diff_wealth",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Wealth at Beginning of Period\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_savings_rate_by_age_bins_at_first_care(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_savings_rate_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for savings rate by age bins at first care."""
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="savings_rate",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[["diff_savings_rate"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care",
                group_col="age_bin_label",
                diff_col="diff_savings_rate",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to start of first care spell",
                ylabel="Savings Rate\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_first_care_start_by_age(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "first_care_start_by_age.png",
    path_to_csv: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "first_care_start_by_age.csv",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Count how many people start giving care for the first time at each age.

    Creates a plot showing the distribution of first care start ages and saves
    the counts to a CSV file.

    Args:
        path_to_caregiving_leave_data: Path to caregiving leave simulated data
        path_to_plot: Path to save the plot
        path_to_csv: Path to save the CSV with counts by age
        min_age: Minimum age to include in analysis
        max_age: Maximum age to include in analysis

    """
    # Load data
    df_o = pd.read_pickle(path_to_caregiving_leave_data)

    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = ensure_agent_period(df_o)

    # Fully flatten any residual index levels
    if isinstance(df_o.index, pd.MultiIndex):
        idx_names_o = {n for n in df_o.index.names if n is not None}
        if ("agent" in idx_names_o) or ("period" in idx_names_o):
            df_o = df_o.reset_index()

    # Ensure no index name collisions remain
    df_o = df_o.reset_index(drop=True)

    # Find first care period for each agent
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_o["choice"].isin(care_codes)

    first_care = (
        df_o.loc[caregiving_mask, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(columns={"period": "first_care_period", "age": "age_at_first_care"})
    )

    # Filter to age range
    first_care = first_care[
        (first_care["age_at_first_care"] >= min_age)
        & (first_care["age_at_first_care"] <= max_age)
    ]

    # Count by age
    counts_by_age = (
        first_care["age_at_first_care"]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"age_at_first_care": "age"})
    )

    # Ensure all ages in range are represented (fill missing with 0)
    all_ages = pd.DataFrame({"age": range(min_age, max_age + 1)})
    counts_by_age = all_ages.merge(counts_by_age, on="age", how="left").fillna(0)
    counts_by_age["count"] = counts_by_age["count"].astype(int)

    # Save to CSV
    counts_by_age.to_csv(path_to_csv, index=False)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.bar(
        counts_by_age["age"],
        counts_by_age["count"],
        color=JET_COLOR_MAP[1],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Age at first care spell", fontsize=16)
    plt.ylabel("Number of people", fontsize=16)
    plt.title("Distribution of First Care Start by Age", fontsize=18)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


def _add_distance_to_first_care_demand(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care_demand column.

    Sets 0 as first time care_demand > 0.
    """
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = ensure_agent_period(df)
    # Find first period where care_demand > 0
    care_demand_mask = df["care_demand"] > 0
    first_care_demand = (
        df.loc[care_demand_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_care_demand_period"})
    )
    out = df.merge(first_care_demand, on="agent", how="left")
    out["distance_to_first_care_demand"] = (
        out["period"] - out["first_care_demand_period"]
    )
    return out


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_by_distance_by_care_demand(  # noqa: PLR0915, E501
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_by_distance_by_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_care_by_distance_by_care_demand.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (caregiving leave - baseline).

    Averages by distance.

    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt, care) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand from caregiving leave, attach to merged.
      6) Average diffs by distance and plot three series for labor outcomes.
      7) Plot care probability separately.

    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_caregiving_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate outcomes
    o_outcomes = calculate_outcomes(df_o, choice_set_type="original")
    c_outcomes = calculate_outcomes(df_c, choice_set_type="original")

    # Load specs for additional outcomes
    import pickle as pkl

    specs = pkl.load(path_to_specs.open("rb"))

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    o_additional = calculate_additional_outcomes(df_o, specs)
    c_additional = calculate_additional_outcomes(df_c, specs)
    o_outcomes.update(o_additional)
    c_outcomes.update(c_additional)

    # Create outcome columns
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Merge and compute differences
    outcome_names = [
        "work",
        "ft",
        "pt",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
    ]
    merged = merge_and_compute_differences(o_cols, c_cols, outcome_names)

    # Compute distance to first care demand in original and attach
    df_o_dist = _add_distance_to_first_care_demand(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care_demand", observed=False)[
            [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care_demand")
    )

    # Plot labor outcomes
    plot_three_line_differences(
        prof=prof,
        x_col="distance_to_first_care_demand",
        path_to_plot=path_to_plot,
        xlabel="Year relative to first care demand",
        window=window,
    )

    # Plot care probability
    prof_care = (
        merged.groupby("distance_to_first_care_demand", observed=False)[["diff_care"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care_demand")
    )

    plt.figure(figsize=(12, 7))
    plt.plot(
        prof_care["distance_to_first_care_demand"],
        prof_care["diff_care"],
        label="Care",
        color="green",
        linewidth=2,
    )
    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to first care demand", fontsize=16)
    plt.ylabel("Probability of Providing Care\nDeviation from Baseline", fontsize=16)
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        ["Care"], loc="lower left", bbox_to_anchor=(0.05, 0.05), prop={"size": 16}
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_care, dpi=300, bbox_inches="tight")
    plt.close()


# NOTE: This task has been split into individual tasks per outcome to reduce
# memory usage. See tasks below starting with task_plot_matched_differences_*
# _by_age_at_first_care_demand
@pytask.mark.skip()
@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_by_age_at_first_care_demand_deprecated(
    # noqa: E501
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_part_time_by_age_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_full_time_by_age_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_employment_rate_by_age_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_job_offer_by_age_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_working_hours_by_age_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_care_by_age_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care_demand: list[int] | None = None,
) -> None:
    """DEPRECATED: Split into individual tasks per outcome to reduce memory usage."""


# ============================================================================
# Split tasks by outcome type for age_at_first_care_demand
# ============================================================================


def _process_data_for_age_at_first_care_demand(
    path_to_caregiving_leave_data: Path,
    path_to_baseline_data: Path,
    path_to_specs: Path,
    outcome_name: str,
    ever_caregivers: bool,
    ever_care_demand: bool,
    ages_at_first_care_demand: list[int],
    window: int,
) -> pd.DataFrame:
    """Helper function to process data for age_at_first_care_demand tasks.

    Args:
        path_to_caregiving_leave_data: Path to caregiving leave data
        path_to_baseline_data: Path to baseline data
        path_to_specs: Path to specs file
        outcome_name: Name of outcome to calculate (e.g., 'pt', 'ft', 'work', etc.)
        ever_caregivers: Filter to ever-caregivers
        ever_care_demand: Filter to ever-care-demand
        ages_at_first_care_demand: List of ages to filter to
        window: Window size for distance filtering

    Returns:
        Profile DataFrame with aggregated differences
    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_caregiving_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate only the outcome needed (plus dependencies)
    o_outcomes = {}
    c_outcomes = {}

    # Basic outcomes from calculate_outcomes
    if outcome_name in ("work", "ft", "pt", "job_offer", "care"):
        full_outcomes_o = calculate_outcomes(df_o, choice_set_type="original")
        full_outcomes_c = calculate_outcomes(df_c, choice_set_type="original")
        o_outcomes[outcome_name] = full_outcomes_o[outcome_name]
        c_outcomes[outcome_name] = full_outcomes_c[outcome_name]
    elif outcome_name == "hours_weekly":
        specs = pickle.load(path_to_specs.open("rb"))
        o_outcomes["hours_weekly"] = calculate_working_hours_weekly(
            df_o, specs, choice_set_type="original"
        )
        c_outcomes["hours_weekly"] = calculate_working_hours_weekly(
            df_c, specs, choice_set_type="original"
        )
    elif outcome_name in (
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ):
        specs = pickle.load(path_to_specs.open("rb"))
        o_additional = calculate_additional_outcomes(df_o, specs)
        c_additional = calculate_additional_outcomes(df_c, specs)
        o_outcomes[outcome_name] = o_additional[outcome_name]
        c_outcomes[outcome_name] = c_additional[outcome_name]
    else:
        raise ValueError(f"Unknown outcome_name: {outcome_name}")

    # Compute distance and age at first care demand from original (before merging)
    df_o_dist = _add_distance_to_first_care_demand(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    care_demand_mask = df_o["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_o, care_demand_mask, "age_at_first_care_demand"
    )

    del df_o_dist

    # Create outcome columns
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Clean up original dataframes before merging (keep only what we need)
    del df_o, df_c

    # Merge and compute differences
    merged = merge_and_compute_differences(o_cols, c_cols, [outcome_name])

    # Clean up intermediate dataframes
    del o_cols, c_cols

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    del dist_map, first_care_demand_with_age

    # Filter to specific ages at first care demand
    merged = merged[merged["age_at_first_care_demand"].isin(ages_at_first_care_demand)]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # # Return full merged dataframe (not aggregated) so calling functions can do
    #  nested splits
    return merged


# NOTE: This task has been split into individual tasks per outcome to reduce
# memory usage. See tasks below starting with task_plot_matched_differences_*
# _by_age_bins_at_first_care_demand
@pytask.mark.skip()
@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_by_age_bins_at_first_care_demand_deprecated(
    # noqa: E501
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_part_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_full_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_employment_rate_by_age_bins_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_job_offer_by_age_bins_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_working_hours_by_age_bins_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "_matched_differences_care_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """DEPRECATED: Split into individual tasks per outcome to reduce memory usage."""


# ============================================================================
# Split tasks by outcome type for age_bins_at_first_care_demand
# ============================================================================


def _process_data_for_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path,
    path_to_baseline_data: Path,
    path_to_specs: Path,
    outcome_name: str,
    ever_caregivers: bool,
    ever_care_demand: bool,
    min_age: int,
    max_age: int,
    bin_width: int,
    window: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Helper function to process data for age_bins_at_first_care_demand tasks.

    Args:
        path_to_caregiving_leave_data: Path to caregiving leave data
        path_to_baseline_data: Path to baseline data
        path_to_specs: Path to specs file
        outcome_name: Name of outcome to calculate (e.g., 'pt', 'ft', 'work', etc.)
        ever_caregivers: Filter to ever-caregivers
        ever_care_demand: Filter to ever-care-demand
        min_age: Minimum age for binning
        max_age: Maximum age for binning
        bin_width: Width of age bins
        window: Window size for distance filtering

    Returns:
        Tuple of (profile DataFrame, list of unique bin labels in order)
    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_caregiving_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate only the outcome needed (plus dependencies)
    o_outcomes = {}
    c_outcomes = {}

    # Basic outcomes from calculate_outcomes
    if outcome_name in ("work", "ft", "pt", "job_offer", "care"):
        full_outcomes_o = calculate_outcomes(df_o, choice_set_type="original")
        full_outcomes_c = calculate_outcomes(df_c, choice_set_type="original")
        o_outcomes[outcome_name] = full_outcomes_o[outcome_name]
        c_outcomes[outcome_name] = full_outcomes_c[outcome_name]
    elif outcome_name == "hours_weekly":
        specs = pickle.load(path_to_specs.open("rb"))
        o_outcomes["hours_weekly"] = calculate_working_hours_weekly(
            df_o, specs, choice_set_type="original"
        )
        c_outcomes["hours_weekly"] = calculate_working_hours_weekly(
            df_c, specs, choice_set_type="original"
        )
    elif outcome_name in (
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ):
        specs = pickle.load(path_to_specs.open("rb"))
        o_additional = calculate_additional_outcomes(df_o, specs)
        c_additional = calculate_additional_outcomes(df_c, specs)
        o_outcomes[outcome_name] = o_additional[outcome_name]
        c_outcomes[outcome_name] = c_additional[outcome_name]
    else:
        raise ValueError(f"Unknown outcome_name: {outcome_name}")

    # Compute distance and age at first care demand from original (before merging)
    df_o_dist = _add_distance_to_first_care_demand(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    care_demand_mask = df_o["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_o, care_demand_mask, "age_at_first_care_demand"
    )

    del df_o_dist

    # Create outcome columns
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_o.columns:
        o_cols["education"] = df_o["education"].values
    if "caregiving_type" in df_o.columns:
        o_cols["caregiving_type"] = df_o["caregiving_type"].values

    # Clean up original dataframes before merging (keep only what we need)
    del df_o, df_c

    # Merge and compute differences
    merged = merge_and_compute_differences(o_cols, c_cols, [outcome_name])

    # Clean up intermediate dataframes
    del o_cols, c_cols

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    del dist_map, first_care_demand_with_age

    # Filter to age range
    merged = merged[
        (merged["age_at_first_care_demand"] >= min_age)
        & (merged["age_at_first_care_demand"] <= max_age)
    ]

    # Create age bins
    merged["age_bin_start"] = (
        (merged["age_at_first_care_demand"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Get unique age bins in order (sorted by bin start) before returning
    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )

    # # Return full merged dataframe (not aggregated) so calling functions can do
    #  nested splits
    return merged, unique_bins


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_pt_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_part_time_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for part-time work by age bins at first care demand."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="pt",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_pt"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_pt",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Proportion Part-Time Working\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_ft_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_full_time_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for full-time work by age bins at first care demand."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="ft",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_ft"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_ft",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Proportion Full-Time Working\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_work_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_employment_rate_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for employment rate by age bins at first care demand."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="work",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_work"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_work",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Proportion Working\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_job_offer_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_job_offer_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for job offer by age bins at first care demand."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="job_offer",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_job_offer"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_job_offer",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Job Offer Probability\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_hours_weekly_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_working_hours_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for working hours by age bins at first care demand."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="hours_weekly",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_hours_weekly"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_hours_weekly",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Weekly Working Hours\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_care_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_care_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for care by age bins at first care demand."""
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="care",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_care"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_care",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Proportion Providing Care\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_gross_labor_income_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_gross_labor_income_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for gross labor income by age bins at first care demand."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="gross_labor_income",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_gross_labor_income"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_gross_labor_income",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Gross Labor Income (Monthly)\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_savings_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_savings_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for savings by age bins at first care demand."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="savings",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_savings"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_savings",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Savings Decision\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_wealth_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_wealth_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for wealth by age bins at first care demand."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="wealth",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_wealth"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_wealth",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Wealth at Beginning of Period\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_savings_rate_by_age_bins_at_first_care_demand(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_savings_rate_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences for savings rate by age bins at first care demand."""  # noqa: E501
    # Get merged dataframe and unique bins from helper
    merged, unique_bins = _process_data_for_age_bins_at_first_care_demand(
        path_to_caregiving_leave_data=path_to_caregiving_leave_data,
        path_to_baseline_data=path_to_baseline_data,
        path_to_specs=path_to_specs,
        outcome_name="savings_rate",
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
        window=window,
    )

    colors = get_distinct_colors(len(unique_bins))

    # Create nested splits: education (all, low, high) × caregiving_type (all, 0, 1)
    education_specs = [
        ("all", None, "all"),
        ("low_education", 0, "low_education"),
        ("high_education", 1, "high_education"),
    ]
    caregiving_type_specs = [
        ("all", None, "all"),
        ("caregiving_type0", 0, "caregiving_type0"),
        ("caregiving_type1", 1, "caregiving_type1"),
    ]

    # Loop over nested splits
    for _edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for _cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / path_to_plot.name

            # Average differences by distance and age_bin_label
            prof = (
                merged_spec.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[["diff_savings_rate"]]
                .mean()
                .reset_index()
            )
            # Add age_bin_start for sorting
            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])
            prof = prof.drop(columns=["age_bin_start"])

            plot_multi_line_differences_by_group(
                prof=prof,
                x_col="distance_to_first_care_demand",
                group_col="age_bin_label",
                diff_col="diff_savings_rate",
                groups=unique_bins,
                colors=colors,
                path_to_plot=plot_path,
                xlabel="Year relative to first care demand",
                ylabel="Savings Rate\nDeviation from Baseline",
                window=window,
                legend_title="Age at first care demand",
            )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_first_care_demand_start_by_age(
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "first_care_demand_start_by_age.png",
    path_to_csv: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "first_care_demand_start_by_age.csv",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Count how many people first experience care_demand > 0 at each age.

    Creates a plot showing the distribution of first care demand start ages and saves
    the counts to a CSV file.

    Args:
        path_to_caregiving_leave_data: Path to caregiving leave simulated data
        path_to_plot: Path to save the plot
        path_to_csv: Path to save the CSV with counts by age
        min_age: Minimum age to include in analysis
        max_age: Maximum age to include in analysis
    """
    # Load data
    df_o = pd.read_pickle(path_to_caregiving_leave_data)

    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = ensure_agent_period(df_o)

    # Fully flatten any residual index levels
    if isinstance(df_o.index, pd.MultiIndex):
        idx_names_o = {n for n in df_o.index.names if n is not None}
        if ("agent" in idx_names_o) or ("period" in idx_names_o):
            df_o = df_o.reset_index()

    # Ensure no index name collisions remain
    df_o = df_o.reset_index(drop=True)

    # Check for age column
    if "age" not in df_o.columns:
        raise ValueError("Age column required but not found in data")

    # Find first period where care_demand > 0 for each agent
    care_demand_mask = df_o["care_demand"] > 0

    first_care_demand = (
        df_o.loc[care_demand_mask, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(
            columns={
                "period": "first_care_demand_period",
                "age": "age_at_first_care_demand",
            }
        )
    )

    # Filter to age range
    first_care_demand = first_care_demand[
        (first_care_demand["age_at_first_care_demand"] >= min_age)
        & (first_care_demand["age_at_first_care_demand"] <= max_age)
    ]

    # Count by age
    counts_by_age = (
        first_care_demand["age_at_first_care_demand"]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"age_at_first_care_demand": "age"})
    )

    # Ensure all ages in range are represented (fill missing with 0)
    all_ages = pd.DataFrame({"age": range(min_age, max_age + 1)})
    counts_by_age = all_ages.merge(counts_by_age, on="age", how="left").fillna(0)
    counts_by_age["count"] = counts_by_age["count"].astype(int)

    # Save to CSV
    counts_by_age.to_csv(path_to_csv, index=False)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.bar(
        counts_by_age["age"],
        counts_by_age["count"],
        color=JET_COLOR_MAP[1],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Age at first care demand", fontsize=16)
    plt.ylabel("Number of people", fontsize=16)
    plt.title("Distribution of First Care Demand Start by Age", fontsize=18)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.skip()
@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_vs_baseline
def task_plot_matched_differences_forced_care_demand_at_50(  # noqa: PLR0915, E501
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_forced_care_demand_at_50.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    forced_age: int = 50,
) -> None:
    """Compute matched period differences (baseline - forced care demand at 50).

    Uses t=0 as age 50 (where everyone experiences care demand in counterfactual).
    Since everyone experiences care demand at age 50 in the counterfactual,
    age 50 is at the center of the x-axis with a vertical line.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance from age 50 (age - forced_age) for x-axis.
      6) Average diffs by distance and plot three series.

    """
    # Load and prepare data
    df_cg, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_caregiving_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate outcomes - both use original choice set
    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Load specs for additional outcomes
    specs = pickle.load(path_to_specs.open("rb"))

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    cg_additional = calculate_additional_outcomes(df_cg, specs)
    baseline_additional = calculate_additional_outcomes(df_baseline, specs)
    cg_outcomes.update(cg_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns (include age from caregiving leave)
    o_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    c_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Add age column from caregiving leave
    if "age" in df_cg.columns:
        o_cols["age"] = df_cg["age"].values

    # Merge and compute differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_o"] - merged["work_c"]
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]
    merged["diff_gross_labor_income"] = (
        merged["gross_labor_income_o"] - merged["gross_labor_income_c"]
    )
    merged["diff_savings"] = merged["savings_o"] - merged["savings_c"]
    merged["diff_wealth"] = merged["wealth_o"] - merged["wealth_c"]
    merged["diff_savings_rate"] = merged["savings_rate_o"] - merged["savings_rate_c"]

    # Compute distance from forced_age (age 50)
    merged["distance_from_forced_age"] = merged["age"] - forced_age

    # Trim to window around forced_age
    merged = merged[
        (merged["distance_from_forced_age"] >= -window)
        & (merged["distance_from_forced_age"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_from_forced_age", observed=False)[
            [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_from_forced_age")
    )

    # Plot
    plot_three_line_differences(
        prof=prof,
        x_col="distance_from_forced_age",
        path_to_plot=path_to_plot,
        xlabel=f"Year relative to age {forced_age}",
        window=window,
    )
