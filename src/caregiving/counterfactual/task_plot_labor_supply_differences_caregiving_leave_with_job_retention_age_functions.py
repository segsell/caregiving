"""Age-based plotting functions for caregiving-leave-with-job-retention counterfactual."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import plot_all_outcomes_by_age
from caregiving.counterfactual.plotting_utils import (
    calculate_additional_outcomes,
    calculate_outcomes,
    create_outcome_columns,
    merge_and_compute_differences,
    prepare_dataframes_for_comparison,
)


@pytask.mark.counterfactual_differences_cg_leave_age_profiles
def task_plot_matched_differences_by_age_cg_leave_vs_no_care_demand(  # noqa: PLR0915
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
    / "age_profiles"
    / "matched_differences_employment_rate_by_age.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_full_time_by_age.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_part_time_by_age.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_job_offer_by_age.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_hours_weekly_by_age.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_care_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_gross_labor_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_savings_by_age.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_wealth_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_savings_rate_by_age.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    age_min: int = 30,
    age_max: int = 100,
) -> None:
    """Plot matched differences by age (cg-leave vs no care demand)."""

    df_cg, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    cg_additional = calculate_additional_outcomes(df_cg)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    cg_outcomes.update(cg_additional)
    ncd_outcomes.update(ncd_additional)

    if "working_hours" in df_cg.columns:
        cg_outcomes["hours_weekly"] = df_cg["working_hours"].values / 52
    else:
        cg_outcomes["hours_weekly"] = np.zeros(len(df_cg))

    if "working_hours" in df_ncd.columns:
        ncd_outcomes["hours_weekly"] = df_ncd["working_hours"].values / 52
    else:
        ncd_outcomes["hours_weekly"] = np.zeros(len(df_ncd))

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    if "age" in df_cg.columns:
        cg_cols["age"] = df_cg["age"].values

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
    ]
    merged = merge_and_compute_differences(cg_cols, ncd_cols, outcome_names)

    merged = merged[(merged["age"] >= age_min) & (merged["age"] <= age_max)]

    prof = (
        merged.groupby("age", observed=False)[
            [
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
            ]
        ]
        .mean()
        .reset_index()
    )

    plot_configs = {
        "work": {
            "ylabel": "Proportion Working\nDeviation from No Care Demand",
            "title": "Employment Rate by Age",
            "diff_col": "diff_work",
            "path": path_to_plot_work,
        },
        "ft": {
            "ylabel": "Proportion Full-time\nDeviation from No Care Demand",
            "title": "Full-time Employment by Age",
            "diff_col": "diff_ft",
            "path": path_to_plot_ft,
        },
        "pt": {
            "ylabel": "Proportion Part-time\nDeviation from No Care Demand",
            "title": "Part-time Employment by Age",
            "diff_col": "diff_pt",
            "path": path_to_plot_pt,
        },
        "job_offer": {
            "ylabel": "Job Offer Probability\nDeviation from No Care Demand",
            "title": "Job Offer Probability by Age",
            "diff_col": "diff_job_offer",
            "path": path_to_plot_job_offer,
        },
        "hours_weekly": {
            "ylabel": "Weekly Hours\nDeviation from No Care Demand",
            "title": "Weekly Working Hours by Age",
            "diff_col": "diff_hours_weekly",
            "path": path_to_plot_hours_weekly,
        },
        "care": {
            "ylabel": "Care Probability\nDeviation from No Care Demand",
            "title": "Care Probability by Age",
            "diff_col": "diff_care",
            "path": path_to_plot_care,
        },
        "gross_labor_income": {
            "ylabel": "Gross Labor Income\nDeviation from No Care Demand",
            "title": "Gross Labor Income by Age",
            "diff_col": "diff_gross_labor_income",
            "path": path_to_plot_gross_labor_income,
        },
        "savings": {
            "ylabel": "Savings\nDeviation from No Care Demand",
            "title": "Savings by Age",
            "diff_col": "diff_savings",
            "path": path_to_plot_savings,
        },
        "wealth": {
            "ylabel": "Wealth\nDeviation from No Care Demand",
            "title": "Wealth by Age",
            "diff_col": "diff_wealth",
            "path": path_to_plot_wealth,
        },
        "savings_rate": {
            "ylabel": "Savings Rate\nDeviation from No Care Demand",
            "title": "Savings Rate by Age",
            "diff_col": "diff_savings_rate",
            "path": path_to_plot_savings_rate,
        },
        "consumption": {
            "ylabel": "Consumption\nDeviation from No Care Demand",
            "title": "Consumption by Age",
            "diff_col": "diff_consumption",
            "path": path_to_plot_savings.parent
            / "matched_differences_consumption_by_age.png",
        },
    }

    plot_all_outcomes_by_age(
        prof=prof,
        plot_configs=plot_configs,
        age_min=age_min,
        age_max=age_max,
    )


@pytask.mark.counterfactual_differences_cg_leave_age_profiles
def task_plot_matched_differences_by_age_cg_leave_vs_baseline(  # noqa: PLR0915
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
    / "age_profiles"
    / "matched_differences_employment_rate_by_age.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_full_time_by_age.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_part_time_by_age.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_job_offer_by_age.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_hours_weekly_by_age.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_care_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_gross_labor_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_savings_by_age.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_wealth_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "caregiving_leave_with_job_retention"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_savings_rate_by_age.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    age_min: int = 30,
    age_max: int = 100,
) -> None:
    """Plot matched differences by age (cg-leave vs baseline)."""

    df_cg, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
    )

    cg_outcomes = calculate_outcomes(df_cg, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    cg_additional = calculate_additional_outcomes(df_cg)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    cg_outcomes.update(cg_additional)
    baseline_outcomes.update(baseline_additional)

    if "working_hours" in df_cg.columns:
        cg_outcomes["hours_weekly"] = df_cg["working_hours"].values / 52
    else:
        cg_outcomes["hours_weekly"] = np.zeros(len(df_cg))

    if "working_hours" in df_baseline.columns:
        baseline_outcomes["hours_weekly"] = df_baseline["working_hours"].values / 52
    else:
        baseline_outcomes["hours_weekly"] = np.zeros(len(df_baseline))

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    if "age" in df_cg.columns:
        cg_cols["age"] = df_cg["age"].values

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
    ]
    merged = merge_and_compute_differences(cg_cols, baseline_cols, outcome_names)

    merged = merged[(merged["age"] >= age_min) & (merged["age"] <= age_max)]

    prof = (
        merged.groupby("age", observed=False)[
            [
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
            ]
        ]
        .mean()
        .reset_index()
    )

    plot_configs = {
        "work": {
            "ylabel": "Proportion Working\nDeviation from Baseline",
            "title": "Employment Rate by Age",
            "diff_col": "diff_work",
            "path": path_to_plot_work,
        },
        "ft": {
            "ylabel": "Proportion Full-time\nDeviation from Baseline",
            "title": "Full-time Employment by Age",
            "diff_col": "diff_ft",
            "path": path_to_plot_ft,
        },
        "pt": {
            "ylabel": "Proportion Part-time\nDeviation from Baseline",
            "title": "Part-time Employment by Age",
            "diff_col": "diff_pt",
            "path": path_to_plot_pt,
        },
        "job_offer": {
            "ylabel": "Job Offer Probability\nDeviation from Baseline",
            "title": "Job Offer Probability by Age",
            "diff_col": "diff_job_offer",
            "path": path_to_plot_job_offer,
        },
        "hours_weekly": {
            "ylabel": "Weekly Hours\nDeviation from Baseline",
            "title": "Weekly Working Hours by Age",
            "diff_col": "diff_hours_weekly",
            "path": path_to_plot_hours_weekly,
        },
        "care": {
            "ylabel": "Care Probability\nDeviation from Baseline",
            "title": "Care Probability by Age",
            "diff_col": "diff_care",
            "path": path_to_plot_care,
        },
        "gross_labor_income": {
            "ylabel": "Gross Labor Income\nDeviation from Baseline",
            "title": "Gross Labor Income by Age",
            "diff_col": "diff_gross_labor_income",
            "path": path_to_plot_gross_labor_income,
        },
        "savings": {
            "ylabel": "Savings\nDeviation from Baseline",
            "title": "Savings by Age",
            "diff_col": "diff_savings",
            "path": path_to_plot_savings,
        },
        "wealth": {
            "ylabel": "Wealth\nDeviation from Baseline",
            "title": "Wealth by Age",
            "diff_col": "diff_wealth",
            "path": path_to_plot_wealth,
        },
        "savings_rate": {
            "ylabel": "Savings Rate\nDeviation from Baseline",
            "title": "Savings Rate by Age",
            "diff_col": "diff_savings_rate",
            "path": path_to_plot_savings_rate,
        },
        "consumption": {
            "ylabel": "Consumption\nDeviation from Baseline",
            "title": "Consumption by Age",
            "diff_col": "diff_consumption",
            "path": path_to_plot_savings.parent
            / "matched_differences_consumption_by_age.png",
        },
    }

    plot_all_outcomes_by_age(
        prof=prof,
        plot_configs=plot_configs,
        age_min=age_min,
        age_max=age_max,
    )
