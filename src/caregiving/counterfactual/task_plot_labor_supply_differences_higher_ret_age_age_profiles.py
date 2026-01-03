"""Age-based plotting functions for higher-retirement-age counterfactual vs baseline."""

import pickle
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
    calculate_working_hours_weekly,
    create_outcome_columns,
    merge_and_compute_differences,
    prepare_dataframes_for_comparison,
)
from caregiving.model.shared import INFORMAL_CARE


@pytask.mark.counterfactual_differences_higher_ret_age_age_profiles
def task_plot_matched_differences_by_age_higher_ret_age_vs_baseline(  # noqa: PLR0915, E501
    path_to_higher_ret_age_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_higher_ret_age_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_work_by_age.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_full_time_by_age.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_part_time_by_age.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_job_offer_by_age.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_working_hours_by_age.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_care_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_gross_labor_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_savings_by_age.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_wealth_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_savings_rate_by_age.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    age_min: int = 30,
    age_max: int = 100,
) -> None:
    """Compute matched period differences (higher-ret-age - baseline) by age.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Average diffs by age and plot all outcomes.

    """
    # Load and prepare data
    df_hr, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_higher_ret_age_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
    )

    # Calculate outcomes (both use original choice set)
    hr_outcomes = calculate_outcomes(df_hr, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate working hours
    specs = pickle.load(path_to_specs.open("rb"))
    hr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_hr, specs, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, specs, choice_set_type="original"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth,
    # savings_rate, consumption)
    hr_additional = calculate_additional_outcomes(df_hr)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    hr_outcomes.update(hr_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns and merge
    hr_cols = create_outcome_columns(df_hr, hr_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Add age column to hr_cols for age-based filtering
    if "age" in df_hr.columns:
        hr_cols["age"] = df_hr["age"].values

    # Merge and compute differences
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
    merged = merge_and_compute_differences(hr_cols, baseline_cols, outcome_names)

    # Filter to age range and average by age
    merged = merged[(merged["age"] >= age_min) & (merged["age"] <= age_max)]

    # Average differences by age
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

    # Plot configurations
    plot_configs = {
        "work": {
            "ylabel": "Proportion Working\nDeviation from Baseline",
            "title": "Employment Rate by Age",
            "diff_col": "diff_work",
            "path": path_to_plot_work,
            "age_max": 75,
        },
        "ft": {
            "ylabel": "Proportion Full-time\nDeviation from Baseline",
            "title": "Full-time Employment by Age",
            "diff_col": "diff_ft",
            "path": path_to_plot_ft,
            "age_max": 75,
        },
        "pt": {
            "ylabel": "Proportion Part-time\nDeviation from Baseline",
            "title": "Part-time Employment by Age",
            "diff_col": "diff_pt",
            "path": path_to_plot_pt,
            "age_max": 75,
        },
        "job_offer": {
            "ylabel": "Job Offer Probability\nDeviation from Baseline",
            "title": "Job Offer Probability by Age",
            "diff_col": "diff_job_offer",
            "path": path_to_plot_job_offer,
            "age_max": 75,
        },
        "hours_weekly": {
            "ylabel": "Weekly Hours\nDeviation from Baseline",
            "title": "Weekly Working Hours by Age",
            "diff_col": "diff_hours_weekly",
            "path": path_to_plot_hours_weekly,
            "age_max": 75,
        },
        "care": {
            "ylabel": "Care Probability\nDeviation from Baseline",
            "title": "Care Probability by Age",
            "diff_col": "diff_care",
            "path": path_to_plot_care,
            "age_max": 75,
        },
        "gross_labor_income": {
            "ylabel": "Gross Labor Income\nDeviation from Baseline",
            "title": "Gross Labor Income by Age",
            "diff_col": "diff_gross_labor_income",
            "path": path_to_plot_gross_labor_income,
            "age_max": 90,
        },
        "savings": {
            "ylabel": "Savings (in 1,000€)\nDeviation from Baseline",
            "title": "Savings by Age",
            "diff_col": "diff_savings",
            "path": path_to_plot_savings,
            "age_max": 90,
        },
        "wealth": {
            "ylabel": "Wealth (in 1,000€)\nDeviation from Baseline",
            "title": "Wealth by Age",
            "diff_col": "diff_wealth",
            "path": path_to_plot_wealth,
            "age_max": 90,
        },
        "savings_rate": {
            "ylabel": "Savings Rate\nDeviation from Baseline",
            "title": "Savings Rate by Age",
            "diff_col": "diff_savings_rate",
            "path": path_to_plot_savings_rate,
            "age_max": 90,
        },
        "consumption": {
            "ylabel": "Consumption (in 1,000€)\nDeviation from Baseline",
            "title": "Consumption by Age",
            "diff_col": "diff_consumption",
            "path": path_to_plot_savings.parent
            / "matched_differences_consumption_by_age.png",
            "age_max": 90,
        },
    }

    plot_all_outcomes_by_age(
        prof=prof,
        plot_configs=plot_configs,
        age_min=age_min,
        age_max=age_max,
    )
