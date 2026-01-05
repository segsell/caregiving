"""Plot differences in labor supply for caregiving-leave-with-job-retention.

Counterfactual.

Compares caregiving-leave-with-job-retention counterfactual vs:
1. No care demand counterfactual
2. Baseline scenario
"""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    calculate_simple_outcomes,
    get_age_at_first_event,
    get_distinct_colors,
    plot_all_outcomes_by_age,
    plot_all_outcomes_by_group,
    plot_three_line_differences,
    prepare_dataframes_simple,
)
from caregiving.counterfactual.plotting_utils import (
    _ensure_agent_period,
    calculate_additional_outcomes,
    calculate_outcomes,
    calculate_working_hours_weekly,
    create_outcome_columns,
    merge_and_compute_differences,
    prepare_dataframes_for_comparison,
)
from caregiving.counterfactual.task_plot_labor_supply_differences import (
    _add_distance_to_first_care,
)
from caregiving.counterfactual.task_plot_labor_supply_differences_no_care_demand import (  # noqa: E501
    _add_distance_to_first_care_demand,
)
from caregiving.model.shared import DEAD, INFORMAL_CARE


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_distance_caregiving_leave(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences"
    / "all"
    / "all"
    / "matched_differences_by_distance_caregiving_leave_with_job_retention.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Matched period differences by distance to first care.

    Compares cg-leave-jr vs no-care-demand.
    """

    # Load specs
    specs = pickle.load(path_to_specs.open("rb"))

    # Load and prepare data
    df_cg, df_ncd = prepare_dataframes_simple(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers,
    )

    # Calculate outcomes
    cg_work, cg_ft, cg_pt = calculate_simple_outcomes(df_cg, "job_retention")
    ncd_work, ncd_ft, ncd_pt = calculate_simple_outcomes(df_ncd, "no_care_demand")

    # Additional outcomes (with specs for scaling)
    cg_additional = calculate_additional_outcomes(df_cg, specs)
    ncd_additional = calculate_additional_outcomes(df_ncd, specs)

    # Outcome columns
    cg_cols = df_cg[["agent", "period"]].copy()
    cg_cols["work_cg"] = cg_work
    cg_cols["ft_cg"] = cg_ft
    cg_cols["pt_cg"] = cg_pt
    cg_cols["gross_labor_income_cg"] = cg_additional["gross_labor_income"]
    cg_cols["savings_cg"] = cg_additional["savings"]
    cg_cols["wealth_cg"] = cg_additional["wealth"]
    cg_cols["savings_rate_cg"] = cg_additional["savings_rate"]
    cg_cols["consumption_cg"] = cg_additional["consumption"]
    cg_cols["bequest_from_parent_cg"] = cg_additional["bequest_from_parent"]
    if "caregiving_leave_top_up" in cg_additional:
        cg_cols["caregiving_leave_top_up_cg"] = cg_additional["caregiving_leave_top_up"]

    ncd_cols = df_ncd[["agent", "period"]].copy()
    ncd_cols["work_ncd"] = ncd_work
    ncd_cols["ft_ncd"] = ncd_ft
    ncd_cols["pt_ncd"] = ncd_pt
    ncd_cols["gross_labor_income_ncd"] = ncd_additional["gross_labor_income"]
    ncd_cols["savings_ncd"] = ncd_additional["savings"]
    ncd_cols["wealth_ncd"] = ncd_additional["wealth"]
    ncd_cols["savings_rate_ncd"] = ncd_additional["savings_rate"]
    ncd_cols["consumption_ncd"] = ncd_additional["consumption"]
    ncd_cols["bequest_from_parent_ncd"] = ncd_additional["bequest_from_parent"]

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    # Differences (cg-leave - no-care-demand)
    merged = cg_cols.merge(ncd_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_cg"] - merged["work_ncd"]
    merged["diff_ft"] = merged["ft_cg"] - merged["ft_ncd"]
    merged["diff_pt"] = merged["pt_cg"] - merged["pt_ncd"]
    merged["diff_gross_labor_income"] = (
        merged["gross_labor_income_cg"] - merged["gross_labor_income_ncd"]
    )
    merged["diff_savings"] = merged["savings_cg"] - merged["savings_ncd"]
    merged["diff_wealth"] = merged["wealth_cg"] - merged["wealth_ncd"]
    merged["diff_savings_rate"] = merged["savings_rate_cg"] - merged["savings_rate_ncd"]
    merged["diff_consumption"] = merged["consumption_cg"] - merged["consumption_ncd"]
    merged["diff_bequest_from_parent"] = (
        merged["bequest_from_parent_cg"] - merged["bequest_from_parent_ncd"]
    )
    if "caregiving_leave_top_up_cg" in merged.columns:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_cg"]

    # Distance to first care in caregiving-leave scenario
    df_cg_dist = _add_distance_to_first_care(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    # Window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

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

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            # Create profile
            prof = (
                merged_filtered.groupby("distance_to_first_care", observed=False)[
                    [
                        "diff_work",
                        "diff_ft",
                        "diff_pt",
                        "diff_gross_labor_income",
                        "diff_savings",
                        "diff_wealth",
                        "diff_savings_rate",
                        "diff_consumption",
                        "diff_bequest_from_parent",
                    ]
                ]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care")
            )

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in merged_filtered.columns:
                prof_top_up = (
                    merged_filtered.groupby("distance_to_first_care", observed=False)[
                        ["diff_caregiving_leave_top_up"]
                    ]
                    .mean()
                    .reset_index()
                )
                prof = prof.merge(prof_top_up, on="distance_to_first_care", how="left")

            # Create path
            # path_to_plot structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_dir}/{cg_type_dir}/filename.png
            base_path = path_to_plot.parent.parent.parent / edu_dir / cg_type_dir
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = (
                base_path
                / "matched_differences_by_distance_caregiving_leave_with_job_retention.png"
            )

    plot_three_line_differences(
        prof=prof,
        x_col="distance_to_first_care",
        path_to_plot=plot_path,
        xlabel="Year relative to start of first care spell",
        window=window,
    )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_age_at_first_care_cg_leave(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_part_time_by_age_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_full_time_by_age_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_employment_rate_by_age_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_age_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_working_hours_by_age_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Matched differences by age at first caregiving spell (cg-leave vs no care)."""

    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    df_cg, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, specs, choice_set_type="no_care_demand"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    ncd_additional = calculate_additional_outcomes(df_ncd, specs)
    cg_outcomes.update(cg_additional)
    ncd_outcomes.update(ncd_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]

    df_cg_dist = _add_distance_to_first_care(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_cg["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_cg, caregiving_mask, "age_at_first_care"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    merged = merged[merged["age_at_first_care"].isin(ages_at_first_care)]
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

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

    # Helper function to create plot configs with dynamic paths
    def create_plot_configs(base_path):
        """Create plot configs dictionary with dynamic paths."""
        return {
            "pt": {
                "path": base_path
                / "matched_differences_part_time_by_age_at_first_care.png",
                "ylabel": "Proportion Part-Time Working\nDeviation from No Care Demand",
                "diff_col": "diff_pt",
            },
            "ft": {
                "path": base_path
                / "matched_differences_full_time_by_age_at_first_care.png",
                "ylabel": "Proportion Full-Time Working\nDeviation from No Care Demand",
                "diff_col": "diff_ft",
            },
            "work": {
                "path": base_path
                / "matched_differences_employment_rate_by_age_at_first_care.png",
                "ylabel": "Proportion Working\nDeviation from No Care Demand",
                "diff_col": "diff_work",
            },
            "job_offer": {
                "path": base_path
                / "matched_differences_job_offer_by_age_at_first_care.png",
                "ylabel": "Job Offer Probability\nDeviation from No Care Demand",
                "diff_col": "diff_job_offer",
            },
            "hours_weekly": {
                "path": base_path
                / "matched_differences_working_hours_by_age_at_first_care.png",
                "ylabel": "Weekly Working Hours\nDeviation from No Care Demand",
                "diff_col": "diff_hours_weekly",
            },
            "gross_labor_income": {
                "path": base_path
                / "matched_differences_gross_labor_income_by_age_at_first_care.png",
                "ylabel": "Gross Labor Income (Monthly)\nDeviation from No Care Demand",
                "diff_col": "diff_gross_labor_income",
            },
            "savings": {
                "path": base_path
                / "matched_differences_savings_by_age_at_first_care.png",
                "ylabel": "Savings Decision\nDeviation from No Care Demand",
                "diff_col": "diff_savings",
            },
            "wealth": {
                "path": base_path
                / "matched_differences_wealth_by_age_at_first_care.png",
                "ylabel": "Wealth at Beginning of Period\nDeviation from No Care Demand",
                "diff_col": "diff_wealth",
            },
            "savings_rate": {
                "path": base_path
                / "matched_differences_savings_rate_by_age_at_first_care.png",
                "ylabel": "Savings Rate\nDeviation from No Care Demand",
                "diff_col": "diff_savings_rate",
            },
            "consumption": {
                "path": base_path
                / "matched_differences_consumption_by_age_at_first_care.png",
                "ylabel": "Consumption\nDeviation from No Care Demand",
                "diff_col": "diff_consumption",
            },
            "bequest_from_parent": {
                "path": base_path
                / "matched_differences_bequest_from_parent_by_age_at_first_care.png",
                "ylabel": "Bequest from Parent\nDeviation from No Care Demand",
                "diff_col": "diff_bequest_from_parent",
            },
        }

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            # Create profile
            prof_cols = [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
                "diff_consumption",
                "diff_bequest_from_parent",
            ]
            prof = (
                merged_filtered.groupby(
                    ["distance_to_first_care", "age_at_first_care"], observed=False
                )[prof_cols]
                .mean()
                .reset_index()
                .sort_values(["age_at_first_care", "distance_to_first_care"])
            )

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in merged_filtered.columns:
                prof_top_up = (
                    merged_filtered.groupby(
                        ["distance_to_first_care", "age_at_first_care"], observed=False
                    )[["diff_caregiving_leave_top_up"]]
                    .mean()
                    .reset_index()
                )
                prof = prof.merge(
                    prof_top_up,
                    on=["distance_to_first_care", "age_at_first_care"],
                    how="left",
                )
                # Add to plot_configs
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )
                plot_configs["caregiving_leave_top_up"] = {
                    "path": (
                        path_to_plot_work.parent.parent
                        / "matched_differences"
                        / edu_dir
                        / cg_type_dir
                        / "matched_differences_caregiving_leave_top_up_by_age_at_first_care.png"
                    ),
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from No Care Demand",
                    "diff_col": "diff_caregiving_leave_top_up",
                }
            else:
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )

            # Create directory
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

    plot_all_outcomes_by_group(
        prof=prof,
        x_col="distance_to_first_care",
        group_col="age_at_first_care",
        groups=ages_at_first_care,
        colors=colors,
        plot_configs=plot_configs,
        window=window,
        legend_title="Age at first care",
    )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_age_bins_at_first_care_cg_leave(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_part_time_by_age_bins_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_full_time_by_age_bins_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_employment_rate_by_age_bins_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_age_bins_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_working_hours_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care spell.

    Compares cg-leave vs no care.
    """
    df_cg, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, specs, choice_set_type="no_care_demand"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    ncd_additional = calculate_additional_outcomes(df_ncd, specs)
    cg_outcomes.update(cg_additional)
    ncd_outcomes.update(ncd_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]

    df_cg_dist = _add_distance_to_first_care(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_cg["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_cg, caregiving_mask, "age_at_first_care"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    merged = merged[
        (merged["age_at_first_care"] >= min_age)
        & (merged["age_at_first_care"] <= max_age)
    ]

    merged["age_bin_start"] = (
        (merged["age_at_first_care"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

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

    # Helper function to create plot configs with dynamic paths
    def create_plot_configs(base_path):
        """Create plot configs dictionary with dynamic paths."""
        return {
            "pt": {
                "path": base_path
                / "matched_differences_part_time_by_age_bins_at_first_care.png",
                "ylabel": "Proportion Part-Time Working\nDeviation from No Care Demand",
                "diff_col": "diff_pt",
            },
            "ft": {
                "path": base_path
                / "matched_differences_full_time_by_age_bins_at_first_care.png",
                "ylabel": "Proportion Full-Time Working\nDeviation from No Care Demand",
                "diff_col": "diff_ft",
            },
            "work": {
                "path": base_path
                / "matched_differences_employment_rate_by_age_bins_at_first_care.png",
                "ylabel": "Proportion Working\nDeviation from No Care Demand",
                "diff_col": "diff_work",
            },
            "job_offer": {
                "path": base_path
                / "matched_differences_job_offer_by_age_bins_at_first_care.png",
                "ylabel": "Job Offer Probability Difference\nDeviation from No Care Demand",
                "diff_col": "diff_job_offer",
            },
            "hours_weekly": {
                "path": base_path
                / "matched_differences_working_hours_by_age_bins_at_first_care.png",
                "ylabel": "Weekly Working Hours Difference\nDeviation from No Care Demand",
                "diff_col": "diff_hours_weekly",
            },
            "gross_labor_income": {
                "path": base_path
                / "matched_differences_gross_labor_income_by_age_bins_at_first_care.png",
                "ylabel": "Gross Labor Income (Monthly)\nDeviation from No Care Demand",
                "diff_col": "diff_gross_labor_income",
            },
            "savings": {
                "path": base_path
                / "matched_differences_savings_by_age_bins_at_first_care.png",
                "ylabel": "Savings Decision\nDeviation from No Care Demand",
                "diff_col": "diff_savings",
            },
            "wealth": {
                "path": base_path
                / "matched_differences_wealth_by_age_bins_at_first_care.png",
                "ylabel": "Wealth at Beginning of Period\nDeviation from No Care Demand",
                "diff_col": "diff_wealth",
            },
            "savings_rate": {
                "path": base_path
                / "matched_differences_savings_rate_by_age_bins_at_first_care.png",
                "ylabel": "Savings Rate\nDeviation from No Care Demand",
                "diff_col": "diff_savings_rate",
            },
            "consumption": {
                "path": base_path
                / "matched_differences_consumption_by_age_bins_at_first_care.png",
                "ylabel": "Consumption\nDeviation from No Care Demand",
                "diff_col": "diff_consumption",
            },
            "bequest_from_parent": {
                "path": base_path
                / "matched_differences_bequest_from_parent_by_age_bins_at_first_care.png",
                "ylabel": "Bequest from Parent\nDeviation from No Care Demand",
                "diff_col": "diff_bequest_from_parent",
            },
        }

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            # Create profile
            prof_cols = [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
                "diff_consumption",
                "diff_bequest_from_parent",
            ]
            prof = (
                merged_filtered.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[prof_cols]
                .mean()
                .reset_index()
            )

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in merged_filtered.columns:
                prof_top_up = (
                    merged_filtered.groupby(
                        ["distance_to_first_care", "age_bin_label"], observed=False
                    )[["diff_caregiving_leave_top_up"]]
                    .mean()
                    .reset_index()
                )
                prof = prof.merge(
                    prof_top_up,
                    on=["distance_to_first_care", "age_bin_label"],
                    how="left",
                )
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )
                plot_configs["caregiving_leave_top_up"] = {
                    "path": (
                        path_to_plot_work.parent.parent
                        / "matched_differences"
                        / edu_dir
                        / cg_type_dir
                        / "matched_differences_caregiving_leave_top_up_by_age_bins_at_first_care.png"
                    ),
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from No Care Demand",
                    "diff_col": "diff_caregiving_leave_top_up",
                }
            else:
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )

            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])

            unique_bins = (
                merged_filtered[["age_bin_label", "age_bin_start"]]
                .drop_duplicates()
                .sort_values("age_bin_start")["age_bin_label"]
                .tolist()
            )

            colors = get_distinct_colors(len(unique_bins))

            # Create directory
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

    plot_all_outcomes_by_group(
        prof=prof,
        x_col="distance_to_first_care",
        group_col="age_bin_label",
        groups=unique_bins,
        colors=colors,
        plot_configs=plot_configs,
        window=window,
        legend_title="Age at first care",
    )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_age_at_first_care_demand_cg_leave(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_part_time_by_age_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_full_time_by_age_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_employment_rate_by_age_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_age_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_working_hours_by_age_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_care_by_age_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care_demand: list[int] | None = None,
) -> None:
    """Compute matched period differences by age at first care demand.

    Compares cg-leave vs no care.
    """
    if ages_at_first_care_demand is None:
        ages_at_first_care_demand = [45, 50, 55, 60, 65]

    df_cg, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, specs, choice_set_type="no_care_demand"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    ncd_additional = calculate_additional_outcomes(df_ncd, specs)
    cg_outcomes.update(cg_additional)
    ncd_outcomes.update(ncd_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]

    df_cg_dist = _add_distance_to_first_care_demand(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    care_demand_mask = df_cg["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_cg, care_demand_mask, "age_at_first_care_demand"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    merged = merged[merged["age_at_first_care_demand"].isin(ages_at_first_care_demand)]

    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    colors = get_distinct_colors(len(ages_at_first_care_demand))

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

    # Helper function to create plot configs with dynamic paths
    def create_plot_configs(base_path):
        """Create plot configs dictionary with dynamic paths."""
        return {
            "pt": {
                "path": base_path
                / "matched_differences_part_time_by_age_at_first_care_demand.png",
                "ylabel": "Proportion Part-Time Working\nDeviation from No Care Demand",
                "diff_col": "diff_pt",
                "xlabel": "Year relative to first care demand",
            },
            "ft": {
                "path": base_path
                / "matched_differences_full_time_by_age_at_first_care_demand.png",
                "ylabel": "Proportion Full-Time Working\nDeviation from No Care Demand",
                "diff_col": "diff_ft",
                "xlabel": "Year relative to first care demand",
            },
            "work": {
                "path": base_path
                / "matched_differences_employment_rate_by_age_at_first_care_demand.png",
                "ylabel": "Proportion Working\nDeviation from No Care Demand",
                "diff_col": "diff_work",
                "xlabel": "Year relative to first care demand",
            },
            "job_offer": {
                "path": base_path
                / "matched_differences_job_offer_by_age_at_first_care_demand.png",
                "ylabel": "Job Offer Probability Difference\nDeviation from No Care Demand",
                "diff_col": "diff_job_offer",
                "xlabel": "Year relative to first care demand",
            },
            "hours_weekly": {
                "path": base_path
                / "matched_differences_working_hours_by_age_at_first_care_demand.png",
                "ylabel": "Weekly Working Hours Difference\nDeviation from No Care Demand",
                "diff_col": "diff_hours_weekly",
                "xlabel": "Year relative to first care demand",
            },
            "care": {
                "path": base_path
                / "matched_differences_care_by_age_at_first_care_demand.png",
                "ylabel": "Probability of Providing Care\nDeviation from No Care Demand",
                "diff_col": "diff_care",
                "xlabel": "Year relative to first care demand",
            },
            "gross_labor_income": {
                "path": base_path
                / "matched_differences_gross_labor_income_by_age_at_first_care_demand.png",
                "ylabel": "Gross Labor Income (Monthly)\nDeviation from No Care Demand",
                "diff_col": "diff_gross_labor_income",
                "xlabel": "Year relative to first care demand",
            },
            "savings": {
                "path": base_path
                / "matched_differences_savings_by_age_at_first_care_demand.png",
                "ylabel": "Savings Decision\nDeviation from No Care Demand",
                "diff_col": "diff_savings",
                "xlabel": "Year relative to first care demand",
            },
            "wealth": {
                "path": base_path
                / "matched_differences_wealth_by_age_at_first_care_demand.png",
                "ylabel": "Wealth at Beginning of Period\nDeviation from No Care Demand",
                "diff_col": "diff_wealth",
                "xlabel": "Year relative to first care demand",
            },
            "savings_rate": {
                "path": base_path
                / "matched_differences_savings_rate_by_age_at_first_care_demand.png",
                "ylabel": "Savings Rate\nDeviation from No Care Demand",
                "diff_col": "diff_savings_rate",
                "xlabel": "Year relative to first care demand",
            },
            "consumption": {
                "path": base_path
                / "matched_differences_consumption_by_age_at_first_care_demand.png",
                "ylabel": "Consumption\nDeviation from No Care Demand",
                "diff_col": "diff_consumption",
                "xlabel": "Year relative to first care demand",
            },
            "bequest_from_parent": {
                "path": base_path
                / "matched_differences_bequest_from_parent_by_age_at_first_care_demand.png",
                "ylabel": "Bequest from Parent\nDeviation from No Care Demand",
                "diff_col": "diff_bequest_from_parent",
                "xlabel": "Year relative to first care demand",
            },
        }

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            # Create profile
            prof_cols = [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_care",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
                "diff_consumption",
                "diff_bequest_from_parent",
            ]
            prof = (
                merged_filtered.groupby(
                    ["distance_to_first_care_demand", "age_at_first_care_demand"],
                    observed=False,
                )[prof_cols]
                .mean()
                .reset_index()
                .sort_values(
                    ["age_at_first_care_demand", "distance_to_first_care_demand"]
                )
            )

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in merged_filtered.columns:
                prof_top_up = (
                    merged_filtered.groupby(
                        ["distance_to_first_care_demand", "age_at_first_care_demand"],
                        observed=False,
                    )[["diff_caregiving_leave_top_up"]]
                    .mean()
                    .reset_index()
                )
                prof = prof.merge(
                    prof_top_up,
                    on=["distance_to_first_care_demand", "age_at_first_care_demand"],
                    how="left",
                )
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )
                plot_configs["caregiving_leave_top_up"] = {
                    "path": (
                        path_to_plot_work.parent.parent
                        / "matched_differences"
                        / edu_dir
                        / cg_type_dir
                        / "matched_differences_caregiving_leave_top_up_by_age_at_first_care_demand.png"
                    ),
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from No Care Demand",
                    "diff_col": "diff_caregiving_leave_top_up",
                    "xlabel": "Year relative to first care demand",
                }
            else:
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )

            # Create directory
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

    plot_all_outcomes_by_group(
        prof=prof,
        x_col="distance_to_first_care_demand",
        group_col="age_at_first_care_demand",
        groups=ages_at_first_care_demand,
        colors=colors,
        plot_configs=plot_configs,
        window=window,
        legend_title="Age at first care demand",
    )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_age_bins_at_first_care_demand_cg_leave(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_part_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_full_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_employment_rate_by_age_bins_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_age_bins_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_working_hours_by_age_bins_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_care_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care demand.

    Compares cg-leave vs no care.
    """
    df_cg, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, specs, choice_set_type="no_care_demand"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    ncd_additional = calculate_additional_outcomes(df_ncd, specs)
    cg_outcomes.update(cg_additional)
    ncd_outcomes.update(ncd_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]

    df_cg_dist = _add_distance_to_first_care_demand(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    care_demand_mask = df_cg["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_cg, care_demand_mask, "age_at_first_care_demand"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    merged = merged[
        (merged["age_at_first_care_demand"] >= min_age)
        & (merged["age_at_first_care_demand"] <= max_age)
    ]

    merged["age_bin_start"] = (
        (merged["age_at_first_care_demand"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

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

    # Helper function to create plot configs with dynamic paths
    def create_plot_configs(base_path):
        """Create plot configs dictionary with dynamic paths."""
        return {
            "pt": {
                "path": base_path
                / "matched_differences_part_time_by_age_bins_at_first_care_demand.png",
                "ylabel": "Proportion Part-Time Working\nDeviation from No Care Demand",
                "diff_col": "diff_pt",
                "xlabel": "Year relative to first care demand",
            },
            "ft": {
                "path": base_path
                / "matched_differences_full_time_by_age_bins_at_first_care_demand.png",
                "ylabel": "Proportion Full-Time Working\nDeviation from No Care Demand",
                "diff_col": "diff_ft",
                "xlabel": "Year relative to first care demand",
            },
            "work": {
                "path": base_path
                / "matched_differences_employment_rate_by_age_bins_at_first_care_demand.png",
                "ylabel": "Proportion Working\nDeviation from No Care Demand",
                "diff_col": "diff_work",
                "xlabel": "Year relative to first care demand",
            },
            "job_offer": {
                "path": base_path
                / "matched_differences_job_offer_by_age_bins_at_first_care_demand.png",
                "ylabel": "Job Offer Probability Difference\nDeviation from No Care Demand",
                "diff_col": "diff_job_offer",
                "xlabel": "Year relative to first care demand",
            },
            "hours_weekly": {
                "path": base_path
                / "matched_differences_working_hours_by_age_bins_at_first_care_demand.png",
                "ylabel": "Weekly Working Hours Difference\nDeviation from No Care Demand",
                "diff_col": "diff_hours_weekly",
                "xlabel": "Year relative to first care demand",
            },
            "care": {
                "path": base_path
                / "matched_differences_care_by_age_bins_at_first_care_demand.png",
                "ylabel": "Probability of Providing Care\nDeviation from No Care Demand",
                "diff_col": "diff_care",
                "xlabel": "Year relative to first care demand",
            },
            "gross_labor_income": {
                "path": base_path
                / "matched_differences_gross_labor_income_by_age_bins_at_first_care_demand.png",
                "ylabel": "Gross Labor Income (Monthly)\nDeviation from No Care Demand",
                "diff_col": "diff_gross_labor_income",
                "xlabel": "Year relative to first care demand",
            },
            "savings": {
                "path": base_path
                / "matched_differences_savings_by_age_bins_at_first_care_demand.png",
                "ylabel": "Savings Decision\nDeviation from No Care Demand",
                "diff_col": "diff_savings",
                "xlabel": "Year relative to first care demand",
            },
            "wealth": {
                "path": base_path
                / "matched_differences_wealth_by_age_bins_at_first_care_demand.png",
                "ylabel": "Wealth at Beginning of Period\nDeviation from No Care Demand",
                "diff_col": "diff_wealth",
                "xlabel": "Year relative to first care demand",
            },
            "savings_rate": {
                "path": base_path
                / "matched_differences_savings_rate_by_age_bins_at_first_care_demand.png",
                "ylabel": "Savings Rate\nDeviation from No Care Demand",
                "diff_col": "diff_savings_rate",
                "xlabel": "Year relative to first care demand",
            },
            "consumption": {
                "path": base_path
                / "matched_differences_consumption_by_age_bins_at_first_care_demand.png",
                "ylabel": "Consumption\nDeviation from No Care Demand",
                "diff_col": "diff_consumption",
                "xlabel": "Year relative to first care demand",
            },
            "bequest_from_parent": {
                "path": base_path
                / "matched_differences_bequest_from_parent_by_age_bins_at_first_care_demand.png",
                "ylabel": "Bequest from Parent\nDeviation from No Care Demand",
                "diff_col": "diff_bequest_from_parent",
                "xlabel": "Year relative to first care demand",
            },
        }

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            # Create profile
            prof_cols = [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_care",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
                "diff_consumption",
                "diff_bequest_from_parent",
            ]
            prof = (
                merged_filtered.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[prof_cols]
                .mean()
                .reset_index()
            )

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in merged_filtered.columns:
                prof_top_up = (
                    merged_filtered.groupby(
                        ["distance_to_first_care_demand", "age_bin_label"],
                        observed=False,
                    )[["diff_caregiving_leave_top_up"]]
                    .mean()
                    .reset_index()
                )
                prof = prof.merge(
                    prof_top_up,
                    on=["distance_to_first_care_demand", "age_bin_label"],
                    how="left",
                )
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )
                plot_configs["caregiving_leave_top_up"] = {
                    "path": (
                        path_to_plot_work.parent.parent
                        / "matched_differences"
                        / edu_dir
                        / cg_type_dir
                        / "matched_differences_caregiving_leave_top_up_by_age_bins_at_first_care_demand.png"
                    ),
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from No Care Demand",
                    "diff_col": "diff_caregiving_leave_top_up",
                    "xlabel": "Year relative to first care demand",
                }
            else:
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )

            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])

            unique_bins = (
                merged_filtered[["age_bin_label", "age_bin_start"]]
                .drop_duplicates()
                .sort_values("age_bin_start")["age_bin_label"]
                .tolist()
            )

            colors = get_distinct_colors(len(unique_bins))

            # Create directory
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

    plot_all_outcomes_by_group(
        prof=prof,
        x_col="distance_to_first_care_demand",
        group_col="age_bin_label",
        groups=unique_bins,
        colors=colors,
        plot_configs=plot_configs,
        window=window,
        legend_title="Age at first care demand",
    )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_distance_to_first_care_all_outcomes(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_work_by_distance_to_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_ft_by_distance_to_first_care.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_pt_by_distance_to_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_distance_to_first_care.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_hours_weekly_by_distance_to_first_care.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_care_by_distance_to_first_care.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_gross_labor_income_by_distance_to_first_care.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_savings_by_distance_to_first_care.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_wealth_by_distance_to_first_care.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_savings_rate_by_distance_to_first_care.png",
    path_to_plot_consumption: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_consumption_by_distance_to_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Plot all outcomes by distance to first care spell (cg-leave vs no care)."""
    df_cg, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, specs, choice_set_type="no_care_demand"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    ncd_additional = calculate_additional_outcomes(df_ncd, specs)
    cg_outcomes.update(cg_additional)
    ncd_outcomes.update(ncd_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]
        outcome_names.append("caregiving_leave_top_up")

    df_cg_dist = _add_distance_to_first_care(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

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

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            prof = (
                merged_filtered.groupby("distance_to_first_care", observed=False)[
                    [f"diff_{outcome}" for outcome in outcome_names]
                ]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care")
            )

            # Create base path for this nested split
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

            plot_configs = {
                "work": {
                    "path": base_path
                    / "matched_differences_work_by_distance_to_first_care.png",
                    "ylabel": "Proportion Working\nDeviation from No Care Demand",
                    "diff_col": "diff_work",
                },
                "ft": {
                    "path": base_path
                    / "matched_differences_ft_by_distance_to_first_care.png",
                    "ylabel": "Proportion Full-Time\nDeviation from No Care Demand",
                    "diff_col": "diff_ft",
                },
                "pt": {
                    "path": base_path
                    / "matched_differences_pt_by_distance_to_first_care.png",
                    "ylabel": "Proportion Part-Time\nDeviation from No Care Demand",
                    "diff_col": "diff_pt",
                },
                "job_offer": {
                    "path": base_path
                    / "matched_differences_job_offer_by_distance_to_first_care.png",
                    "ylabel": "Job Offer Probability\nDeviation from No Care Demand",
                    "diff_col": "diff_job_offer",
                },
                "hours_weekly": {
                    "path": base_path
                    / "matched_differences_hours_weekly_by_distance_to_first_care.png",
                    "ylabel": "Weekly Working Hours\nDeviation from No Care Demand",
                    "diff_col": "diff_hours_weekly",
                },
                "care": {
                    "path": base_path
                    / "matched_differences_care_by_distance_to_first_care.png",
                    "ylabel": "Care Probability\nDeviation from No Care Demand",
                    "diff_col": "diff_care",
                },
                "gross_labor_income": {
                    "path": base_path
                    / "matched_differences_gross_labor_income_by_distance_to_first_care.png",
                    "ylabel": "Gross Labor Income (Monthly)\nDeviation from No Care Demand",
                    "diff_col": "diff_gross_labor_income",
                },
                "savings": {
                    "path": base_path
                    / "matched_differences_savings_by_distance_to_first_care.png",
                    "ylabel": "Savings (in 1,000€)\nDeviation from No Care Demand",
                    "diff_col": "diff_savings",
                },
                "wealth": {
                    "path": base_path
                    / "matched_differences_wealth_by_distance_to_first_care.png",
                    "ylabel": "Wealth (in 1,000€)\nDeviation from No Care Demand",
                    "diff_col": "diff_wealth",
                },
                "savings_rate": {
                    "path": base_path
                    / "matched_differences_savings_rate_by_distance_to_first_care.png",
                    "ylabel": "Savings Rate\nDeviation from No Care Demand",
                    "diff_col": "diff_savings_rate",
                },
                "consumption": {
                    "path": base_path
                    / "matched_differences_consumption_by_distance_to_first_care.png",
                    "ylabel": "Consumption (in 1,000€)\nDeviation from No Care Demand",
                    "diff_col": "diff_consumption",
                },
                "bequest_from_parent": {
                    "path": base_path
                    / "matched_differences_bequest_from_parent_by_distance_to_first_care.png",
                    "ylabel": "Bequest from Parent (in 1,000€)\nDeviation from No Care Demand",
                    "diff_col": "diff_bequest_from_parent",
                },
            }

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in prof.columns:
                plot_configs["caregiving_leave_top_up"] = {
                    "path": base_path
                    / "matched_differences_caregiving_leave_top_up_by_distance_to_first_care.png",
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from No Care Demand",
                    "diff_col": "diff_caregiving_leave_top_up",
                }

            # Plot each outcome separately
            for config in plot_configs.values():
                diff_col = config["diff_col"]
                if diff_col not in prof.columns:
                    continue

                plt.figure(figsize=(12, 7))
                plt.plot(
                    prof["distance_to_first_care"],
                    prof[diff_col],
                    color="black",
                    linewidth=2,
                )
                plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
                plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
                plt.xlabel("Year relative to start of first care spell", fontsize=16)
                plt.ylabel(config["ylabel"], fontsize=16)
                plt.xlim(-window, window)
                plt.grid(True, alpha=0.3)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                plt.savefig(config["path"], dpi=300, bbox_inches="tight")
                plt.close()


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_distance_to_first_care_demand_all_outcomes(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_work_by_distance_to_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_ft_by_distance_to_first_care_demand.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_pt_by_distance_to_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_distance_to_first_care_demand.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_hours_weekly_by_distance_to_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_care_by_distance_to_first_care_demand.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_gross_labor_income_by_distance_to_first_care_demand.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_savings_by_distance_to_first_care_demand.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_wealth_by_distance_to_first_care_demand.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_savings_rate_by_distance_to_first_care_demand.png",
    path_to_plot_consumption: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_consumption_by_distance_to_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Plot all outcomes by distance to first care demand (cg-leave vs no care)."""
    df_cg, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, specs, choice_set_type="no_care_demand"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    ncd_additional = calculate_additional_outcomes(df_ncd, specs)
    cg_outcomes.update(cg_additional)
    ncd_outcomes.update(ncd_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]
        outcome_names.append("caregiving_leave_top_up")

    df_cg_dist = _add_distance_to_first_care_demand(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )

    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

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

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            prof = (
                merged_filtered.groupby(
                    "distance_to_first_care_demand", observed=False
                )[[f"diff_{outcome}" for outcome in outcome_names]]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )

            # Create base path for this nested split
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

            plot_configs = {
                "work": {
                    "path": base_path
                    / "matched_differences_work_by_distance_to_first_care_demand.png",
                    "ylabel": "Proportion Working\nDeviation from No Care Demand",
                    "diff_col": "diff_work",
                },
                "ft": {
                    "path": base_path
                    / "matched_differences_ft_by_distance_to_first_care_demand.png",
                    "ylabel": "Proportion Full-Time\nDeviation from No Care Demand",
                    "diff_col": "diff_ft",
                },
                "pt": {
                    "path": base_path
                    / "matched_differences_pt_by_distance_to_first_care_demand.png",
                    "ylabel": "Proportion Part-Time\nDeviation from No Care Demand",
                    "diff_col": "diff_pt",
                },
                "job_offer": {
                    "path": base_path
                    / "matched_differences_job_offer_by_distance_to_first_care_demand.png",
                    "ylabel": "Job Offer Probability\nDeviation from No Care Demand",
                    "diff_col": "diff_job_offer",
                },
                "hours_weekly": {
                    "path": base_path
                    / "matched_differences_hours_weekly_by_distance_to_first_care_demand.png",
                    "ylabel": "Weekly Working Hours\nDeviation from No Care Demand",
                    "diff_col": "diff_hours_weekly",
                },
                "care": {
                    "path": base_path
                    / "matched_differences_care_by_distance_to_first_care_demand.png",
                    "ylabel": "Care Probability\nDeviation from No Care Demand",
                    "diff_col": "diff_care",
                },
                "gross_labor_income": {
                    "path": base_path
                    / "matched_differences_gross_labor_income_by_distance_to_first_care_demand.png",
                    "ylabel": "Gross Labor Income (Monthly)\nDeviation from No Care Demand",
                    "diff_col": "diff_gross_labor_income",
                },
                "savings": {
                    "path": base_path
                    / "matched_differences_savings_by_distance_to_first_care_demand.png",
                    "ylabel": "Savings (in 1,000€)\nDeviation from No Care Demand",
                    "diff_col": "diff_savings",
                },
                "wealth": {
                    "path": base_path
                    / "matched_differences_wealth_by_distance_to_first_care_demand.png",
                    "ylabel": "Wealth (in 1,000€)\nDeviation from No Care Demand",
                    "diff_col": "diff_wealth",
                },
                "savings_rate": {
                    "path": base_path
                    / "matched_differences_savings_rate_by_distance_to_first_care_demand.png",
                    "ylabel": "Savings Rate\nDeviation from No Care Demand",
                    "diff_col": "diff_savings_rate",
                },
                "consumption": {
                    "path": base_path
                    / "matched_differences_consumption_by_distance_to_first_care_demand.png",
                    "ylabel": "Consumption (in 1,000€)\nDeviation from No Care Demand",
                    "diff_col": "diff_consumption",
                },
                "bequest_from_parent": {
                    "path": base_path
                    / "matched_differences_bequest_from_parent_by_distance_to_first_care_demand.png",
                    "ylabel": "Bequest from Parent (in 1,000€)\nDeviation from No Care Demand",
                    "diff_col": "diff_bequest_from_parent",
                },
            }

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in prof.columns:
                plot_configs["caregiving_leave_top_up"] = {
                    "path": base_path
                    / "matched_differences_caregiving_leave_top_up_by_distance_to_first_care_demand.png",
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from No Care Demand",
                    "diff_col": "diff_caregiving_leave_top_up",
                }

            # Plot each outcome separately
            for config in plot_configs.values():
                diff_col = config["diff_col"]
                if diff_col not in prof.columns:
                    continue

                plt.figure(figsize=(12, 7))
                plt.plot(
                    prof["distance_to_first_care_demand"],
                    prof[diff_col],
                    color="black",
                    linewidth=2,
                )
                plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
                plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
                plt.xlabel("Year relative to first care demand", fontsize=16)
                plt.ylabel(config["ylabel"], fontsize=16)
                plt.xlim(-window, window)
                plt.grid(True, alpha=0.3)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                plt.savefig(config["path"], dpi=300, bbox_inches="tight")
                plt.close()


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_distance_to_first_care_all_outcomes_vs_baseline(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_work_by_distance_to_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_ft_by_distance_to_first_care.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_pt_by_distance_to_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_job_offer_by_distance_to_first_care.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_hours_weekly_by_distance_to_first_care.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_care_by_distance_to_first_care.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_gross_labor_income_by_distance_to_first_care.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_savings_by_distance_to_first_care.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_wealth_by_distance_to_first_care.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_savings_rate_by_distance_to_first_care.png",
    path_to_plot_consumption: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_consumption_by_distance_to_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Plot all outcomes by distance to first care spell (cg-leave vs baseline)."""
    df_cg, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, specs, choice_set_type="original"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    baseline_additional = calculate_additional_outcomes(df_baseline, specs)
    cg_outcomes.update(cg_additional)
    baseline_outcomes.update(baseline_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]
        outcome_names.append("caregiving_leave_top_up")

    df_cg_dist = _add_distance_to_first_care(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

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

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            prof = (
                merged_filtered.groupby("distance_to_first_care", observed=False)[
                    [f"diff_{outcome}" for outcome in outcome_names]
                ]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care")
            )

            # Create base path for this nested split
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

            plot_configs = {
                "work": {
                    "path": base_path
                    / "matched_differences_work_by_distance_to_first_care.png",
                    "ylabel": "Proportion Working\nDeviation from Baseline",
                    "diff_col": "diff_work",
                },
                "ft": {
                    "path": base_path
                    / "matched_differences_ft_by_distance_to_first_care.png",
                    "ylabel": "Proportion Full-Time\nDeviation from Baseline",
                    "diff_col": "diff_ft",
                },
                "pt": {
                    "path": base_path
                    / "matched_differences_pt_by_distance_to_first_care.png",
                    "ylabel": "Proportion Part-Time\nDeviation from Baseline",
                    "diff_col": "diff_pt",
                },
                "job_offer": {
                    "path": base_path
                    / "matched_differences_job_offer_by_distance_to_first_care.png",
                    "ylabel": "Job Offer Probability\nDeviation from Baseline",
                    "diff_col": "diff_job_offer",
                },
                "hours_weekly": {
                    "path": base_path
                    / "matched_differences_hours_weekly_by_distance_to_first_care.png",
                    "ylabel": "Weekly Working Hours\nDeviation from Baseline",
                    "diff_col": "diff_hours_weekly",
                },
                "care": {
                    "path": base_path
                    / "matched_differences_care_by_distance_to_first_care.png",
                    "ylabel": "Care Probability\nDeviation from Baseline",
                    "diff_col": "diff_care",
                },
                "gross_labor_income": {
                    "path": base_path
                    / "matched_differences_gross_labor_income_by_distance_to_first_care.png",
                    "ylabel": "Gross Labor Income (Monthly)\nDeviation from Baseline",
                    "diff_col": "diff_gross_labor_income",
                },
                "savings": {
                    "path": base_path
                    / "matched_differences_savings_by_distance_to_first_care.png",
                    "ylabel": "Savings (in 1,000€)\nDeviation from Baseline",
                    "diff_col": "diff_savings",
                },
                "wealth": {
                    "path": base_path
                    / "matched_differences_wealth_by_distance_to_first_care.png",
                    "ylabel": "Wealth (in 1,000€)\nDeviation from Baseline",
                    "diff_col": "diff_wealth",
                },
                "savings_rate": {
                    "path": base_path
                    / "matched_differences_savings_rate_by_distance_to_first_care.png",
                    "ylabel": "Savings Rate\nDeviation from Baseline",
                    "diff_col": "diff_savings_rate",
                },
                "consumption": {
                    "path": base_path
                    / "matched_differences_consumption_by_distance_to_first_care.png",
                    "ylabel": "Consumption (in 1,000€)\nDeviation from Baseline",
                    "diff_col": "diff_consumption",
                },
                "bequest_from_parent": {
                    "path": base_path
                    / "matched_differences_bequest_from_parent_by_distance_to_first_care.png",
                    "ylabel": "Bequest from Parent (in 1,000€)\nDeviation from Baseline",
                    "diff_col": "diff_bequest_from_parent",
                },
            }

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in prof.columns:
                plot_configs["caregiving_leave_top_up"] = {
                    "path": base_path
                    / "matched_differences_caregiving_leave_top_up_by_distance_to_first_care.png",
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from Baseline",
                    "diff_col": "diff_caregiving_leave_top_up",
                }

            # Plot each outcome separately
            for config in plot_configs.values():
                diff_col = config["diff_col"]
                if diff_col not in prof.columns:
                    continue

                plt.figure(figsize=(12, 7))
                plt.plot(
                    prof["distance_to_first_care"],
                    prof[diff_col],
                    color="black",
                    linewidth=2,
                )
                plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
                plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
                plt.xlabel("Year relative to start of first care spell", fontsize=16)
                plt.ylabel(config["ylabel"], fontsize=16)
                plt.xlim(-window, window)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(config["path"], dpi=300, bbox_inches="tight")
        plt.close()


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_distance_to_first_care_demand_all_outcomes_vs_baseline(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_work_by_distance_to_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_ft_by_distance_to_first_care_demand.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_pt_by_distance_to_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_job_offer_by_distance_to_first_care_demand.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_hours_weekly_by_distance_to_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_care_by_distance_to_first_care_demand.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_gross_labor_income_by_distance_to_first_care_demand.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_savings_by_distance_to_first_care_demand.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_wealth_by_distance_to_first_care_demand.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_savings_rate_by_distance_to_first_care_demand.png",
    path_to_plot_consumption: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_consumption_by_distance_to_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Plot all outcomes by distance to first care demand (cg-leave vs baseline)."""
    df_cg, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, specs, choice_set_type="original"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    baseline_additional = calculate_additional_outcomes(df_baseline, specs)
    cg_outcomes.update(cg_additional)
    baseline_outcomes.update(baseline_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]
        outcome_names.append("caregiving_leave_top_up")

    df_cg_dist = _add_distance_to_first_care_demand(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )

    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

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

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            prof = (
                merged_filtered.groupby(
                    "distance_to_first_care_demand", observed=False
                )[[f"diff_{outcome}" for outcome in outcome_names]]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )

            # Create base path for this nested split
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

            plot_configs = {
                "work": {
                    "path": base_path
                    / "matched_differences_work_by_distance_to_first_care_demand.png",
                    "ylabel": "Proportion Working\nDeviation from Baseline",
                    "diff_col": "diff_work",
                },
                "ft": {
                    "path": base_path
                    / "matched_differences_ft_by_distance_to_first_care_demand.png",
                    "ylabel": "Proportion Full-Time\nDeviation from Baseline",
                    "diff_col": "diff_ft",
                },
                "pt": {
                    "path": base_path
                    / "matched_differences_pt_by_distance_to_first_care_demand.png",
                    "ylabel": "Proportion Part-Time\nDeviation from Baseline",
                    "diff_col": "diff_pt",
                },
                "job_offer": {
                    "path": base_path
                    / "matched_differences_job_offer_by_distance_to_first_care_demand.png",
                    "ylabel": "Job Offer Probability\nDeviation from Baseline",
                    "diff_col": "diff_job_offer",
                },
                "hours_weekly": {
                    "path": base_path
                    / "matched_differences_hours_weekly_by_distance_to_first_care_demand.png",
                    "ylabel": "Weekly Working Hours\nDeviation from Baseline",
                    "diff_col": "diff_hours_weekly",
                },
                "care": {
                    "path": base_path
                    / "matched_differences_care_by_distance_to_first_care_demand.png",
                    "ylabel": "Care Probability\nDeviation from Baseline",
                    "diff_col": "diff_care",
                },
                "gross_labor_income": {
                    "path": base_path
                    / "matched_differences_gross_labor_income_by_distance_to_first_care_demand.png",
                    "ylabel": "Gross Labor Income (Monthly)\nDeviation from Baseline",
                    "diff_col": "diff_gross_labor_income",
                },
                "savings": {
                    "path": base_path
                    / "matched_differences_savings_by_distance_to_first_care_demand.png",
                    "ylabel": "Savings (in 1,000€)\nDeviation from Baseline",
                    "diff_col": "diff_savings",
                },
                "wealth": {
                    "path": base_path
                    / "matched_differences_wealth_by_distance_to_first_care_demand.png",
                    "ylabel": "Wealth (in 1,000€)\nDeviation from Baseline",
                    "diff_col": "diff_wealth",
                },
                "savings_rate": {
                    "path": base_path
                    / "matched_differences_savings_rate_by_distance_to_first_care_demand.png",
                    "ylabel": "Savings Rate\nDeviation from Baseline",
                    "diff_col": "diff_savings_rate",
                },
                "consumption": {
                    "path": base_path
                    / "matched_differences_consumption_by_distance_to_first_care_demand.png",
                    "ylabel": "Consumption (in 1,000€)\nDeviation from Baseline",
                    "diff_col": "diff_consumption",
                },
                "bequest_from_parent": {
                    "path": base_path
                    / "matched_differences_bequest_from_parent_by_distance_to_first_care_demand.png",
                    "ylabel": "Bequest from Parent (in 1,000€)\nDeviation from Baseline",
                    "diff_col": "diff_bequest_from_parent",
                },
            }

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in prof.columns:
                plot_configs["caregiving_leave_top_up"] = {
                    "path": base_path
                    / "matched_differences_caregiving_leave_top_up_by_distance_to_first_care_demand.png",
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from Baseline",
                    "diff_col": "diff_caregiving_leave_top_up",
                }

            # Plot each outcome separately
            for config in plot_configs.values():
                diff_col = config["diff_col"]
                if diff_col not in prof.columns:
                    continue

                plt.figure(figsize=(12, 7))
                plt.plot(
                    prof["distance_to_first_care_demand"],
                    prof[diff_col],
                    color="black",
                    linewidth=2,
                )
                plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
                plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
                plt.xlabel("Year relative to first care demand", fontsize=16)
                plt.ylabel(config["ylabel"], fontsize=16)
                plt.xlim(-window, window)
                plt.grid(True, alpha=0.3)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                plt.savefig(config["path"], dpi=300, bbox_inches="tight")
                plt.close()


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_age_bins_at_first_care_cg_leave_vs_baseline(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
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
    / "matched_differences_part_time_by_age_bins_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_bins_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_bins_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_bins_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_bins_at_first_care.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care spell.

    Compares cg-leave vs baseline.
    """
    df_cg, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, specs, choice_set_type="original"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    baseline_additional = calculate_additional_outcomes(df_baseline, specs)
    cg_outcomes.update(cg_additional)
    baseline_outcomes.update(baseline_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]

    df_cg_dist = _add_distance_to_first_care(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_cg["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_cg, caregiving_mask, "age_at_first_care"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    merged = merged[
        (merged["age_at_first_care"] >= min_age)
        & (merged["age_at_first_care"] <= max_age)
    ]

    merged["age_bin_start"] = (
        (merged["age_at_first_care"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

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

    # Helper function to create plot configs with dynamic paths
    def create_plot_configs(base_path):
        """Create plot configs dictionary with dynamic paths."""
        return {
            "pt": {
                "path": base_path
                / "matched_differences_part_time_by_age_bins_at_first_care.png",
                "ylabel": "Proportion Part-Time Working\nDeviation from Baseline",
                "diff_col": "diff_pt",
            },
            "ft": {
                "path": base_path
                / "matched_differences_full_time_by_age_bins_at_first_care.png",
                "ylabel": "Proportion Full-Time Working\nDeviation from Baseline",
                "diff_col": "diff_ft",
            },
            "work": {
                "path": base_path
                / "matched_differences_employment_rate_by_age_bins_at_first_care.png",
                "ylabel": "Proportion Working\nDeviation from Baseline",
                "diff_col": "diff_work",
            },
            "job_offer": {
                "path": base_path
                / "matched_differences_job_offer_by_age_bins_at_first_care.png",
                "ylabel": "Job Offer Probability Difference\nDeviation from Baseline",
                "diff_col": "diff_job_offer",
            },
            "hours_weekly": {
                "path": base_path
                / "matched_differences_working_hours_by_age_bins_at_first_care.png",
                "ylabel": "Weekly Working Hours Difference\nDeviation from Baseline",
                "diff_col": "diff_hours_weekly",
            },
            "gross_labor_income": {
                "path": base_path
                / "matched_differences_gross_labor_income_by_age_bins_at_first_care.png",
                "ylabel": "Gross Labor Income (Monthly)\nDeviation from Baseline",
                "diff_col": "diff_gross_labor_income",
            },
            "savings": {
                "path": base_path
                / "matched_differences_savings_by_age_bins_at_first_care.png",
                "ylabel": "Savings Decision\nDeviation from Baseline",
                "diff_col": "diff_savings",
            },
            "wealth": {
                "path": base_path
                / "matched_differences_wealth_by_age_bins_at_first_care.png",
                "ylabel": "Wealth at Beginning of Period\nDeviation from Baseline",
                "diff_col": "diff_wealth",
            },
            "savings_rate": {
                "path": base_path
                / "matched_differences_savings_rate_by_age_bins_at_first_care.png",
                "ylabel": "Savings Rate\nDeviation from Baseline",
                "diff_col": "diff_savings_rate",
            },
            "consumption": {
                "path": base_path
                / "matched_differences_consumption_by_age_bins_at_first_care.png",
                "ylabel": "Consumption\nDeviation from Baseline",
                "diff_col": "diff_consumption",
            },
            "bequest_from_parent": {
                "path": base_path
                / "matched_differences_bequest_from_parent_by_age_bins_at_first_care.png",
                "ylabel": "Bequest from Parent\nDeviation from Baseline",
                "diff_col": "diff_bequest_from_parent",
            },
        }

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            # Create profile
            prof_cols = [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
                "diff_consumption",
                "diff_bequest_from_parent",
            ]
            prof = (
                merged_filtered.groupby(
                    ["distance_to_first_care", "age_bin_label"], observed=False
                )[prof_cols]
                .mean()
                .reset_index()
            )

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in merged_filtered.columns:
                prof_top_up = (
                    merged_filtered.groupby(
                        ["distance_to_first_care", "age_bin_label"], observed=False
                    )[["diff_caregiving_leave_top_up"]]
                    .mean()
                    .reset_index()
                )
                prof = prof.merge(
                    prof_top_up,
                    on=["distance_to_first_care", "age_bin_label"],
                    how="left",
                )
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )
                plot_configs["caregiving_leave_top_up"] = {
                    "path": (
                        path_to_plot_work.parent.parent
                        / "matched_differences"
                        / edu_dir
                        / cg_type_dir
                        / "matched_differences_caregiving_leave_top_up_by_age_bins_at_first_care.png"
                    ),
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from Baseline",
                    "diff_col": "diff_caregiving_leave_top_up",
                }
            else:
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )

            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])

            unique_bins = (
                merged_filtered[["age_bin_label", "age_bin_start"]]
                .drop_duplicates()
                .sort_values("age_bin_start")["age_bin_label"]
                .tolist()
            )

            colors = get_distinct_colors(len(unique_bins))

            # Create directory
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

    plot_all_outcomes_by_group(
        prof=prof,
        x_col="distance_to_first_care",
        group_col="age_bin_label",
        groups=unique_bins,
        colors=colors,
        plot_configs=plot_configs,
        window=window,
        legend_title="Age at first care",
    )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_age_at_first_care_demand_cg_leave_vs_baseline(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
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
    / "matched_differences_part_time_by_age_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_care_by_age_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care_demand: list[int] | None = None,
) -> None:
    """Compute matched period differences by age at first care demand.

    Compares cg-leave vs baseline.
    """
    if ages_at_first_care_demand is None:
        ages_at_first_care_demand = [45, 50, 55, 60, 65]

    df_cg, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, specs, choice_set_type="original"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    baseline_additional = calculate_additional_outcomes(df_baseline, specs)
    cg_outcomes.update(cg_additional)
    baseline_outcomes.update(baseline_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]

    df_cg_dist = _add_distance_to_first_care_demand(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    care_demand_mask = df_cg["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_cg, care_demand_mask, "age_at_first_care_demand"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    merged = merged[merged["age_at_first_care_demand"].isin(ages_at_first_care_demand)]

    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    colors = get_distinct_colors(len(ages_at_first_care_demand))

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

    # Helper function to create plot configs with dynamic paths
    def create_plot_configs(base_path):
        """Create plot configs dictionary with dynamic paths."""
        return {
            "pt": {
                "path": base_path
                / "matched_differences_part_time_by_age_at_first_care_demand.png",
                "ylabel": "Proportion Part-Time Working\nDeviation from Baseline",
                "diff_col": "diff_pt",
                "xlabel": "Year relative to first care demand",
            },
            "ft": {
                "path": base_path
                / "matched_differences_full_time_by_age_at_first_care_demand.png",
                "ylabel": "Proportion Full-Time Working\nDeviation from Baseline",
                "diff_col": "diff_ft",
                "xlabel": "Year relative to first care demand",
            },
            "work": {
                "path": base_path
                / "matched_differences_employment_rate_by_age_at_first_care_demand.png",
                "ylabel": "Proportion Working\nDeviation from Baseline",
                "diff_col": "diff_work",
                "xlabel": "Year relative to first care demand",
            },
            "job_offer": {
                "path": base_path
                / "matched_differences_job_offer_by_age_at_first_care_demand.png",
                "ylabel": "Job Offer Probability Difference\nDeviation from Baseline",
                "diff_col": "diff_job_offer",
                "xlabel": "Year relative to first care demand",
            },
            "hours_weekly": {
                "path": base_path
                / "matched_differences_working_hours_by_age_at_first_care_demand.png",
                "ylabel": "Weekly Working Hours Difference\nDeviation from Baseline",
                "diff_col": "diff_hours_weekly",
                "xlabel": "Year relative to first care demand",
            },
            "care": {
                "path": base_path
                / "matched_differences_care_by_age_at_first_care_demand.png",
                "ylabel": "Probability of Providing Care\nDeviation from Baseline",
                "diff_col": "diff_care",
                "xlabel": "Year relative to first care demand",
            },
            "gross_labor_income": {
                "path": base_path
                / "matched_differences_gross_labor_income_by_age_at_first_care_demand.png",
                "ylabel": "Gross Labor Income (Monthly)\nDeviation from Baseline",
                "diff_col": "diff_gross_labor_income",
                "xlabel": "Year relative to first care demand",
            },
            "savings": {
                "path": base_path
                / "matched_differences_savings_by_age_at_first_care_demand.png",
                "ylabel": "Savings Decision\nDeviation from Baseline",
                "diff_col": "diff_savings",
                "xlabel": "Year relative to first care demand",
            },
            "wealth": {
                "path": base_path
                / "matched_differences_wealth_by_age_at_first_care_demand.png",
                "ylabel": "Wealth at Beginning of Period\nDeviation from Baseline",
                "diff_col": "diff_wealth",
                "xlabel": "Year relative to first care demand",
            },
            "savings_rate": {
                "path": base_path
                / "matched_differences_savings_rate_by_age_at_first_care_demand.png",
                "ylabel": "Savings Rate\nDeviation from Baseline",
                "diff_col": "diff_savings_rate",
                "xlabel": "Year relative to first care demand",
            },
            "consumption": {
                "path": base_path
                / "matched_differences_consumption_by_age_at_first_care_demand.png",
                "ylabel": "Consumption\nDeviation from Baseline",
                "diff_col": "diff_consumption",
                "xlabel": "Year relative to first care demand",
            },
            "bequest_from_parent": {
                "path": base_path
                / "matched_differences_bequest_from_parent_by_age_at_first_care_demand.png",
                "ylabel": "Bequest from Parent\nDeviation from Baseline",
                "diff_col": "diff_bequest_from_parent",
                "xlabel": "Year relative to first care demand",
            },
        }

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            # Create profile
            prof_cols = [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_care",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
                "diff_consumption",
                "diff_bequest_from_parent",
            ]
            prof = (
                merged_filtered.groupby(
                    ["distance_to_first_care_demand", "age_at_first_care_demand"],
                    observed=False,
                )[prof_cols]
                .mean()
                .reset_index()
                .sort_values(
                    ["age_at_first_care_demand", "distance_to_first_care_demand"]
                )
            )

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in merged_filtered.columns:
                prof_top_up = (
                    merged_filtered.groupby(
                        ["distance_to_first_care_demand", "age_at_first_care_demand"],
                        observed=False,
                    )[["diff_caregiving_leave_top_up"]]
                    .mean()
                    .reset_index()
                )
                prof = prof.merge(
                    prof_top_up,
                    on=["distance_to_first_care_demand", "age_at_first_care_demand"],
                    how="left",
                )
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )
                plot_configs["caregiving_leave_top_up"] = {
                    "path": (
                        path_to_plot_work.parent.parent
                        / "matched_differences"
                        / edu_dir
                        / cg_type_dir
                        / "matched_differences_caregiving_leave_top_up_by_age_at_first_care_demand.png"
                    ),
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from Baseline",
                    "diff_col": "diff_caregiving_leave_top_up",
                    "xlabel": "Year relative to first care demand",
                }
            else:
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )

            # Create directory
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

    plot_all_outcomes_by_group(
        prof=prof,
        x_col="distance_to_first_care_demand",
        group_col="age_at_first_care_demand",
        groups=ages_at_first_care_demand,
        colors=colors,
        plot_configs=plot_configs,
        window=window,
        legend_title="Age at first care demand",
    )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_age_bins_at_first_care_demand_cg_leave_vs_baseline(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
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
    / "matched_differences_part_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_bins_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_bins_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_bins_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "matched_differences_care_by_age_bins_at_first_care_demand.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care demand.

    Compares cg-leave vs baseline.
    """
    df_cg, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    specs = pickle.load(path_to_specs.open("rb"))
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, specs, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, specs, choice_set_type="original"
    )

    cg_additional = calculate_additional_outcomes(df_cg, specs)
    baseline_additional = calculate_additional_outcomes(df_baseline, specs)
    cg_outcomes.update(cg_additional)
    baseline_outcomes.update(baseline_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Add education and caregiving_type columns for filtering
    if "education" in df_cg.columns:
        cg_cols["education"] = df_cg["education"].values
    if "caregiving_type" in df_cg.columns:
        cg_cols["caregiving_type"] = df_cg["caregiving_type"].values

    merged = cg_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
        "consumption",
        "bequest_from_parent",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Add caregiving_leave_top_up if available (only in cg scenario)
    if "caregiving_leave_top_up" in cg_outcomes:
        merged["diff_caregiving_leave_top_up"] = merged["caregiving_leave_top_up_o"]

    df_cg_dist = _add_distance_to_first_care_demand(df_cg)
    dist_map = (
        df_cg_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    care_demand_mask = df_cg["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_cg, care_demand_mask, "age_at_first_care_demand"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    merged = merged[
        (merged["age_at_first_care_demand"] >= min_age)
        & (merged["age_at_first_care_demand"] <= max_age)
    ]

    merged["age_bin_start"] = (
        (merged["age_at_first_care_demand"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

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

    # Helper function to create plot configs with dynamic paths
    def create_plot_configs(base_path):
        """Create plot configs dictionary with dynamic paths."""
        return {
            "pt": {
                "path": base_path
                / "matched_differences_part_time_by_age_bins_at_first_care_demand.png",
                "ylabel": "Proportion Part-Time Working\nDeviation from Baseline",
                "diff_col": "diff_pt",
                "xlabel": "Year relative to first care demand",
            },
            "ft": {
                "path": base_path
                / "matched_differences_full_time_by_age_bins_at_first_care_demand.png",
                "ylabel": "Proportion Full-Time Working\nDeviation from Baseline",
                "diff_col": "diff_ft",
                "xlabel": "Year relative to first care demand",
            },
            "work": {
                "path": base_path
                / "matched_differences_employment_rate_by_age_bins_at_first_care_demand.png",
                "ylabel": "Proportion Working\nDeviation from Baseline",
                "diff_col": "diff_work",
                "xlabel": "Year relative to first care demand",
            },
            "job_offer": {
                "path": base_path
                / "matched_differences_job_offer_by_age_bins_at_first_care_demand.png",
                "ylabel": "Job Offer Probability Difference\nDeviation from Baseline",
                "diff_col": "diff_job_offer",
                "xlabel": "Year relative to first care demand",
            },
            "hours_weekly": {
                "path": base_path
                / "matched_differences_working_hours_by_age_bins_at_first_care_demand.png",
                "ylabel": "Weekly Working Hours Difference\nDeviation from Baseline",
                "diff_col": "diff_hours_weekly",
                "xlabel": "Year relative to first care demand",
            },
            "care": {
                "path": base_path
                / "matched_differences_care_by_age_bins_at_first_care_demand.png",
                "ylabel": "Probability of Providing Care\nDeviation from Baseline",
                "diff_col": "diff_care",
                "xlabel": "Year relative to first care demand",
            },
            "gross_labor_income": {
                "path": base_path
                / "matched_differences_gross_labor_income_by_age_bins_at_first_care_demand.png",
                "ylabel": "Gross Labor Income (Monthly)\nDeviation from Baseline",
                "diff_col": "diff_gross_labor_income",
                "xlabel": "Year relative to first care demand",
            },
            "savings": {
                "path": base_path
                / "matched_differences_savings_by_age_bins_at_first_care_demand.png",
                "ylabel": "Savings Decision\nDeviation from Baseline",
                "diff_col": "diff_savings",
                "xlabel": "Year relative to first care demand",
            },
            "wealth": {
                "path": base_path
                / "matched_differences_wealth_by_age_bins_at_first_care_demand.png",
                "ylabel": "Wealth at Beginning of Period\nDeviation from Baseline",
                "diff_col": "diff_wealth",
                "xlabel": "Year relative to first care demand",
            },
            "savings_rate": {
                "path": base_path
                / "matched_differences_savings_rate_by_age_bins_at_first_care_demand.png",
                "ylabel": "Savings Rate\nDeviation from Baseline",
                "diff_col": "diff_savings_rate",
                "xlabel": "Year relative to first care demand",
            },
            "consumption": {
                "path": base_path
                / "matched_differences_consumption_by_age_bins_at_first_care_demand.png",
                "ylabel": "Consumption\nDeviation from Baseline",
                "diff_col": "diff_consumption",
                "xlabel": "Year relative to first care demand",
            },
            "bequest_from_parent": {
                "path": base_path
                / "matched_differences_bequest_from_parent_by_age_bins_at_first_care_demand.png",
                "ylabel": "Bequest from Parent\nDeviation from Baseline",
                "diff_col": "diff_bequest_from_parent",
                "xlabel": "Year relative to first care demand",
            },
        }

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter merged data
            merged_filtered = merged.copy()
            if edu_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["education"] == edu_value
                ]
            if cg_type_value is not None:
                merged_filtered = merged_filtered[
                    merged_filtered["caregiving_type"] == cg_type_value
                ]

            # Create profile
            prof_cols = [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_care",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
                "diff_consumption",
                "diff_bequest_from_parent",
            ]
            prof = (
                merged_filtered.groupby(
                    ["distance_to_first_care_demand", "age_bin_label"], observed=False
                )[prof_cols]
                .mean()
                .reset_index()
            )

            # Add caregiving_leave_top_up if available
            if "diff_caregiving_leave_top_up" in merged_filtered.columns:
                prof_top_up = (
                    merged_filtered.groupby(
                        ["distance_to_first_care_demand", "age_bin_label"],
                        observed=False,
                    )[["diff_caregiving_leave_top_up"]]
                    .mean()
                    .reset_index()
                )
                prof = prof.merge(
                    prof_top_up,
                    on=["distance_to_first_care_demand", "age_bin_label"],
                    how="left",
                )
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )
                plot_configs["caregiving_leave_top_up"] = {
                    "path": (
                        path_to_plot_work.parent.parent
                        / "matched_differences"
                        / edu_dir
                        / cg_type_dir
                        / "matched_differences_caregiving_leave_top_up_by_age_bins_at_first_care_demand.png"
                    ),
                    "ylabel": "Caregiving Leave Top-up (Monthly)\nDeviation from Baseline",
                    "diff_col": "diff_caregiving_leave_top_up",
                    "xlabel": "Year relative to first care demand",
                }
            else:
                plot_configs = create_plot_configs(
                    path_to_plot_work.parent.parent
                    / "matched_differences"
                    / edu_dir
                    / cg_type_dir
                )

            prof["age_bin_start"] = (
                prof["age_bin_label"].str.split("-").str[0].astype(int)
            )
            prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])

            unique_bins = (
                merged_filtered[["age_bin_label", "age_bin_start"]]
                .drop_duplicates()
                .sort_values("age_bin_start")["age_bin_label"]
                .tolist()
            )

            colors = get_distinct_colors(len(unique_bins))

            # Create directory
            base_path = (
                path_to_plot_work.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)

    plot_all_outcomes_by_group(
        prof=prof,
        x_col="distance_to_first_care_demand",
        group_col="age_bin_label",
        groups=unique_bins,
        colors=colors,
        plot_configs=plot_configs,
        window=window,
        legend_title="Age at first care demand",
    )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_first_care_start_by_age_cg_leave(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_first_care_start_by_age.png",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Plot matched differences in distribution of first care start ages.

    Compares cg-leave vs no care.
    """
    # Load data
    df_cg = pd.read_pickle(path_to_cg_leave_data)
    df_ncd = pd.read_pickle(path_to_no_care_demand_data)

    # Alive restriction
    df_cg = df_cg[df_cg["health"] != DEAD].copy()
    df_ncd = df_ncd[df_ncd["health"] != DEAD].copy()

    # Ensure agent/period
    df_cg = _ensure_agent_period(df_cg)
    df_ncd = _ensure_agent_period(df_ncd)

    # Fully flatten any residual index levels
    for df in (df_cg, df_ncd):
        if isinstance(df.index, pd.MultiIndex):
            idx_names = {n for n in df.index.names if n is not None}
            if ("agent" in idx_names) or ("period" in idx_names):
                df.reset_index(inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Check for age column
    if "age" not in df_cg.columns or "age" not in df_ncd.columns:
        raise ValueError("Age column required but not found in data")

    # Find first care period for each agent
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()

    # For cg_leave - include education and caregiving_type if available
    caregiving_mask_cg = df_cg["choice"].isin(care_codes)
    cols_to_select = ["agent", "period", "age"]
    if "education" in df_cg.columns:
        cols_to_select.append("education")
    if "caregiving_type" in df_cg.columns:
        cols_to_select.append("caregiving_type")
    first_care_cg = (
        df_cg.loc[caregiving_mask_cg, cols_to_select]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(columns={"period": "first_care_period", "age": "age_at_first_care"})
    )

    # For no_care_demand (should be empty, but calculate for consistency)
    caregiving_mask_ncd = df_ncd["choice"].isin(care_codes)
    first_care_ncd = (
        df_ncd.loc[caregiving_mask_ncd, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(columns={"period": "first_care_period", "age": "age_at_first_care"})
    )

    # Filter to age range
    first_care_cg = first_care_cg[
        (first_care_cg["age_at_first_care"] >= min_age)
        & (first_care_cg["age_at_first_care"] <= max_age)
    ]
    first_care_ncd = first_care_ncd[
        (first_care_ncd["age_at_first_care"] >= min_age)
        & (first_care_ncd["age_at_first_care"] <= max_age)
    ]

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

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter first_care_cg data
            first_care_cg_filtered = first_care_cg.copy()
            if edu_value is not None and "education" in first_care_cg_filtered.columns:
                first_care_cg_filtered = first_care_cg_filtered[
                    first_care_cg_filtered["education"] == edu_value
                ]
            if (
                cg_type_value is not None
                and "caregiving_type" in first_care_cg_filtered.columns
            ):
                first_care_cg_filtered = first_care_cg_filtered[
                    first_care_cg_filtered["caregiving_type"] == cg_type_value
                ]

            # Count by age
            counts_cg = (
                first_care_cg_filtered["age_at_first_care"]
                .value_counts()
                .sort_index()
                .reset_index(name="count_cg")
                .rename(columns={"age_at_first_care": "age"})
            )
            counts_ncd = (
                first_care_ncd["age_at_first_care"]
                .value_counts()
                .sort_index()
                .reset_index(name="count_ncd")
                .rename(columns={"age_at_first_care": "age"})
            )

            # Ensure all ages in range are represented (fill missing with 0)
            all_ages = pd.DataFrame({"age": range(min_age, max_age + 1)})
            counts_cg = all_ages.merge(counts_cg, on="age", how="left").fillna(0)
            counts_ncd = all_ages.merge(counts_ncd, on="age", how="left").fillna(0)
            counts_cg["count_cg"] = counts_cg["count_cg"].astype(int)
            counts_ncd["count_ncd"] = counts_ncd["count_ncd"].astype(int)

            # Calculate difference
            counts_diff = counts_cg.copy()
            counts_diff["count_diff"] = (
                counts_diff["count_cg"] - counts_ncd["count_ncd"]
            )

            # Create base path for this nested split
            base_path = (
                path_to_plot.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = base_path / "matched_differences_first_care_start_by_age.png"

            # Plot
            plt.figure(figsize=(12, 7))
            colors = [
                "#2E86AB" if x >= 0 else "#A23B72" for x in counts_diff["count_diff"]
            ]
            plt.bar(
                counts_diff["age"],
                counts_diff["count_diff"],
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            plt.axhline(y=0, color="k", linestyle="-", linewidth=1)
            plt.xlabel("Age at first care spell", fontsize=16)
            plt.ylabel(
                "Difference in number of people\n(Caregiving Leave - No Care Demand)",
                fontsize=16,
            )
            title_suffix = (
                f" ({edu_label}, {cg_type_label})"
                if (edu_label != "all" or cg_type_label != "all")
                else ""
            )
            plt.title(
                f"Matched Differences in First Care Start by Age{title_suffix}",
                fontsize=18,
            )
            plt.grid(True, alpha=0.3, axis="y")
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_first_care_demand_start_by_age_cg_leave(  # noqa: PLR0915, E501
    path_to_cg_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "matched_differences_first_care_demand_start_by_age.png",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Plot matched differences in distribution of first care demand start ages.

    Compares cg-leave vs no care.
    """
    # Load data
    df_cg = pd.read_pickle(path_to_cg_leave_data)
    df_ncd = pd.read_pickle(path_to_no_care_demand_data)

    # Alive restriction
    df_cg = df_cg[df_cg["health"] != DEAD].copy()
    df_ncd = df_ncd[df_ncd["health"] != DEAD].copy()

    # Ensure agent/period
    df_cg = _ensure_agent_period(df_cg)
    df_ncd = _ensure_agent_period(df_ncd)

    # Fully flatten any residual index levels
    for df in (df_cg, df_ncd):
        if isinstance(df.index, pd.MultiIndex):
            idx_names = {n for n in df.index.names if n is not None}
            if ("agent" in idx_names) or ("period" in idx_names):
                df.reset_index(inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Check for age column
    if "age" not in df_cg.columns or "age" not in df_ncd.columns:
        raise ValueError("Age column required but not found in data")

    # Find first period where care_demand > 0 for each agent - include education and caregiving_type if available
    care_demand_mask_cg = df_cg["care_demand"] > 0
    cols_to_select = ["agent", "period", "age"]
    if "education" in df_cg.columns:
        cols_to_select.append("education")
    if "caregiving_type" in df_cg.columns:
        cols_to_select.append("caregiving_type")
    first_care_demand_cg = (
        df_cg.loc[care_demand_mask_cg, cols_to_select]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(
            columns={
                "period": "first_care_demand_period",
                "age": "age_at_first_care_demand",
            }
        )
    )

    # For no_care_demand scenario, there is no care_demand column,
    # so create empty dataframe
    if "care_demand" in df_ncd.columns:
        care_demand_mask_ncd = df_ncd["care_demand"] > 0
        first_care_demand_ncd = (
            df_ncd.loc[care_demand_mask_ncd, ["agent", "period", "age"]]
            .sort_values(["agent", "period"])
            .drop_duplicates("agent", keep="first")
            .rename(
                columns={
                    "period": "first_care_demand_period",
                    "age": "age_at_first_care_demand",
                }
            )
        )
    else:
        # No care demand scenario - create empty dataframe with correct structure
        first_care_demand_ncd = pd.DataFrame(
            columns=["agent", "first_care_demand_period", "age_at_first_care_demand"]
        )

    # Filter to age range
    first_care_demand_cg = first_care_demand_cg[
        (first_care_demand_cg["age_at_first_care_demand"] >= min_age)
        & (first_care_demand_cg["age_at_first_care_demand"] <= max_age)
    ]
    if len(first_care_demand_ncd) > 0:
        first_care_demand_ncd = first_care_demand_ncd[
            (first_care_demand_ncd["age_at_first_care_demand"] >= min_age)
            & (first_care_demand_ncd["age_at_first_care_demand"] <= max_age)
        ]

    # Count by age
    if len(first_care_demand_cg) > 0:
        counts_cg = (
            first_care_demand_cg["age_at_first_care_demand"]
            .value_counts()
            .sort_index()
            .reset_index(name="count_cg")
            .rename(columns={"age_at_first_care_demand": "age"})
        )
    else:
        counts_cg = pd.DataFrame(columns=["age", "count_cg"])

    if len(first_care_demand_ncd) > 0:
        counts_ncd = (
            first_care_demand_ncd["age_at_first_care_demand"]
            .value_counts()
            .sort_index()
            .reset_index(name="count_ncd")
            .rename(columns={"age_at_first_care_demand": "age"})
        )
    else:
        counts_ncd = pd.DataFrame(columns=["age", "count_ncd"])

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

    for edu_label, edu_value, edu_dir in education_specs:
        for cg_type_label, cg_type_value, cg_type_dir in caregiving_type_specs:
            # Filter first_care_demand_cg data
            first_care_demand_cg_filtered = first_care_demand_cg.copy()
            if (
                edu_value is not None
                and "education" in first_care_demand_cg_filtered.columns
            ):
                first_care_demand_cg_filtered = first_care_demand_cg_filtered[
                    first_care_demand_cg_filtered["education"] == edu_value
                ]
            if (
                cg_type_value is not None
                and "caregiving_type" in first_care_demand_cg_filtered.columns
            ):
                first_care_demand_cg_filtered = first_care_demand_cg_filtered[
                    first_care_demand_cg_filtered["caregiving_type"] == cg_type_value
                ]

            # Count by age
            if len(first_care_demand_cg_filtered) > 0:
                counts_cg = (
                    first_care_demand_cg_filtered["age_at_first_care_demand"]
                    .value_counts()
                    .sort_index()
                    .reset_index(name="count_cg")
                    .rename(columns={"age_at_first_care_demand": "age"})
                )
            else:
                counts_cg = pd.DataFrame(columns=["age", "count_cg"])

            if len(first_care_demand_ncd) > 0:
                counts_ncd = (
                    first_care_demand_ncd["age_at_first_care_demand"]
                    .value_counts()
                    .sort_index()
                    .reset_index(name="count_ncd")
                    .rename(columns={"age_at_first_care_demand": "age"})
                )
            else:
                counts_ncd = pd.DataFrame(columns=["age", "count_ncd"])

            # Ensure all ages in range are represented (fill missing with 0)
            all_ages = pd.DataFrame({"age": range(min_age, max_age + 1)})
            counts_cg = all_ages.merge(counts_cg, on="age", how="left").fillna(0)
            counts_ncd = all_ages.merge(counts_ncd, on="age", how="left").fillna(0)
            counts_cg["count_cg"] = counts_cg["count_cg"].astype(int)
            counts_ncd["count_ncd"] = counts_ncd["count_ncd"].astype(int)

            # Calculate difference
            counts_diff = counts_cg.copy()
            counts_diff["count_diff"] = (
                counts_diff["count_cg"] - counts_ncd["count_ncd"]
            )

            # Create base path for this nested split
            base_path = (
                path_to_plot.parent.parent
                / "matched_differences"
                / edu_dir
                / cg_type_dir
            )
            base_path.mkdir(parents=True, exist_ok=True)
            plot_path = (
                base_path / "matched_differences_first_care_demand_start_by_age.png"
            )

            # Plot
            plt.figure(figsize=(12, 7))
            colors = [
                "#2E86AB" if x >= 0 else "#A23B72" for x in counts_diff["count_diff"]
            ]
            plt.bar(
                counts_diff["age"],
                counts_diff["count_diff"],
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            plt.axhline(y=0, color="k", linestyle="-", linewidth=1)
            plt.xlabel("Age at first care demand", fontsize=16)
            plt.ylabel(
                "Difference in number of people\n(Caregiving Leave - No Care Demand)",
                fontsize=16,
            )
            title_suffix = (
                f" ({edu_label}, {cg_type_label})"
                if (edu_label != "all" or cg_type_label != "all")
                else ""
            )
            plt.title(
                f"Matched Differences in First Care Demand Start by Age{title_suffix}",
                fontsize=18,
            )
            plt.grid(True, alpha=0.3, axis="y")
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
