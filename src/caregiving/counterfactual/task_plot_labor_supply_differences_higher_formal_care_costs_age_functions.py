"""Age-based plotting functions for higher formal care costs counterfactual."""

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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_age_profiles
@pytask.mark.counterfactual_differences_higher_formal_care_costs_age_profiles
def task_plot_matched_differences_by_age_vs_no_care_demand(  # noqa: PLR0915
    path_to_higher_formal_care_costs_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_higher_formal_care_costs_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_work_by_age.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_full_time_by_age.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_part_time_by_age.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_job_offer_by_age.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_working_hours_by_age.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_care_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_gross_labor_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_savings_by_age.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_wealth_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_savings_rate_by_age.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    age_min: int = 30,
    age_max: int = 100,
) -> None:
    """Compute matched period differences (higher-formal-care-costs - no-care-demand) by age."""
    # Load and prepare data
    df_hfc, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_higher_formal_care_costs_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
    )

    # Calculate outcomes
    hfc_outcomes = calculate_outcomes(df_hfc, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    hfc_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_hfc, model_params, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, model_params, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes
    hfc_additional = calculate_additional_outcomes(df_hfc)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    hfc_outcomes.update(hfc_additional)
    ncd_outcomes.update(ncd_additional)

    # Create outcome columns and merge
    hfc_cols = create_outcome_columns(df_hfc, hfc_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Add age column for age-based filtering
    if "age" in df_hfc.columns:
        hfc_cols["age"] = df_hfc["age"].values

    # Merge and compute differences (include full set of outcomes incl. consumption)
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
    merged = merge_and_compute_differences(hfc_cols, ncd_cols, outcome_names)

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
            "ylabel": "Proportion Working\nDeviation from No Care Demand",
            "title": "Employment Rate by Age",
            "diff_col": "diff_work",
            "path": path_to_plot_work,
            "age_max": 70,
        },
        "ft": {
            "ylabel": "Proportion Full-time\nDeviation from No Care Demand",
            "title": "Full-time Employment by Age",
            "diff_col": "diff_ft",
            "path": path_to_plot_ft,
            "age_max": 70,
        },
        "pt": {
            "ylabel": "Proportion Part-time\nDeviation from No Care Demand",
            "title": "Part-time Employment by Age",
            "diff_col": "diff_pt",
            "path": path_to_plot_pt,
            "age_max": 70,
        },
        "job_offer": {
            "ylabel": "Job Offer Probability\nDeviation from No Care Demand",
            "title": "Job Offer Probability by Age",
            "diff_col": "diff_job_offer",
            "path": path_to_plot_job_offer,
            "age_max": 70,
        },
        "hours_weekly": {
            "ylabel": "Weekly Hours\nDeviation from No Care Demand",
            "title": "Weekly Working Hours by Age",
            "diff_col": "diff_hours_weekly",
            "path": path_to_plot_hours_weekly,
            "age_max": 70,
        },
        "care": {
            "ylabel": "Care Probability\nDeviation from No Care Demand",
            "title": "Care Probability by Age",
            "diff_col": "diff_care",
            "path": path_to_plot_care,
            "age_max": 70,
        },
        "gross_labor_income": {
            "ylabel": "Gross Labor Income\nDeviation from No Care Demand",
            "title": "Gross Labor Income by Age",
            "diff_col": "diff_gross_labor_income",
            "path": path_to_plot_gross_labor_income,
            "age_max": 90,
        },
        "savings": {
            "ylabel": "Savings (in 1,000€)\nDeviation from No Care Demand",
            "title": "Savings by Age",
            "diff_col": "diff_savings",
            "path": path_to_plot_savings,
            "age_max": 90,
        },
        "wealth": {
            "ylabel": "Wealth (in 1,000€)\nDeviation from No Care Demand",
            "title": "Wealth by Age",
            "diff_col": "diff_wealth",
            "path": path_to_plot_wealth,
            "age_max": 90,
        },
        "savings_rate": {
            "ylabel": "Savings Rate\nDeviation from No Care Demand",
            "title": "Savings Rate by Age",
            "diff_col": "diff_savings_rate",
            "path": path_to_plot_savings_rate,
            "age_max": 90,
        },
        "consumption": {
            "ylabel": "Consumption (in 1,000€)\nDeviation from No Care Demand",
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_age_profiles
@pytask.mark.counterfactual_differences_higher_formal_care_costs_age_profiles
def task_plot_matched_differences_by_age_vs_baseline(  # noqa: PLR0915
    path_to_higher_formal_care_costs_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_higher_formal_care_costs_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_work_by_age.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_full_time_by_age.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_part_time_by_age.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_job_offer_by_age.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_working_hours_by_age.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_care_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_gross_labor_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_savings_by_age.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_wealth_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_savings_rate_by_age.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    age_min: int = 30,
    age_max: int = 100,
) -> None:
    """Compute matched period differences (higher-formal-care-costs - baseline) by age."""
    # Load and prepare data
    df_hfc, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_higher_formal_care_costs_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
    )

    # Calculate outcomes
    hfc_outcomes = calculate_outcomes(df_hfc, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    hfc_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_hfc, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    # Calculate additional outcomes
    hfc_additional = calculate_additional_outcomes(df_hfc)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    hfc_outcomes.update(hfc_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns and merge
    hfc_cols = create_outcome_columns(df_hfc, hfc_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Add age column for age-based filtering
    if "age" in df_hfc.columns:
        hfc_cols["age"] = df_hfc["age"].values

    # Merge and compute differences (include full set of outcomes incl. consumption)
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
    merged = merge_and_compute_differences(hfc_cols, baseline_cols, outcome_names)

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
            "age_max": 70,
        },
        "ft": {
            "ylabel": "Proportion Full-time\nDeviation from Baseline",
            "title": "Full-time Employment by Age",
            "diff_col": "diff_ft",
            "path": path_to_plot_ft,
            "age_max": 70,
        },
        "pt": {
            "ylabel": "Proportion Part-time\nDeviation from Baseline",
            "title": "Part-time Employment by Age",
            "diff_col": "diff_pt",
            "path": path_to_plot_pt,
            "age_max": 70,
        },
        "job_offer": {
            "ylabel": "Job Offer Probability\nDeviation from Baseline",
            "title": "Job Offer Probability by Age",
            "diff_col": "diff_job_offer",
            "path": path_to_plot_job_offer,
            "age_max": 70,
        },
        "hours_weekly": {
            "ylabel": "Weekly Hours\nDeviation from Baseline",
            "title": "Weekly Working Hours by Age",
            "diff_col": "diff_hours_weekly",
            "path": path_to_plot_hours_weekly,
            "age_max": 70,
        },
        "care": {
            "ylabel": "Care Probability\nDeviation from Baseline",
            "title": "Care Probability by Age",
            "diff_col": "diff_care",
            "path": path_to_plot_care,
            "age_max": 70,
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_age_profiles
@pytask.mark.counterfactual_differences_higher_formal_care_costs_age_profiles
def task_plot_matched_differences_by_age_vs_no_care_demand_up_to_90(  # noqa: PLR0915
    path_to_higher_formal_care_costs_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_higher_formal_care_costs_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_work_by_age_up_to_90.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_full_time_by_age_up_to_90.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_part_time_by_age_up_to_90.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_job_offer_by_age_up_to_90.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_working_hours_by_age_up_to_90.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_care_by_age_up_to_90.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_gross_labor_income_by_age_up_to_90.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_savings_by_age_up_to_90.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_wealth_by_age_up_to_90.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_no_care_demand"
    / "age_profiles"
    / "matched_differences_savings_rate_by_age_up_to_90.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Compute matched period differences (higher-formal-care-costs - no-care-demand) by age, up to age 90."""
    # Load and prepare data
    df_hfc, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_higher_formal_care_costs_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
    )

    # Calculate outcomes
    hfc_outcomes = calculate_outcomes(df_hfc, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    hfc_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_hfc, model_params, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, model_params, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes
    hfc_additional = calculate_additional_outcomes(df_hfc)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    hfc_outcomes.update(hfc_additional)
    ncd_outcomes.update(ncd_additional)

    # Create outcome columns and merge
    hfc_cols = create_outcome_columns(df_hfc, hfc_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Add age column for age-based filtering
    if "age" in df_hfc.columns:
        hfc_cols["age"] = df_hfc["age"].values

    # Merge and compute differences (include full set of outcomes incl. consumption)
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
    merged = merge_and_compute_differences(hfc_cols, ncd_cols, outcome_names)

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
            "ylabel": "Proportion Working\nDeviation from No Care Demand",
            "title": "Employment Rate by Age",
            "diff_col": "diff_work",
            "path": path_to_plot_work,
            "age_max": 70,
        },
        "ft": {
            "ylabel": "Proportion Full-time\nDeviation from No Care Demand",
            "title": "Full-time Employment by Age",
            "diff_col": "diff_ft",
            "path": path_to_plot_ft,
            "age_max": 70,
        },
        "pt": {
            "ylabel": "Proportion Part-time\nDeviation from No Care Demand",
            "title": "Part-time Employment by Age",
            "diff_col": "diff_pt",
            "path": path_to_plot_pt,
            "age_max": 70,
        },
        "job_offer": {
            "ylabel": "Job Offer Probability\nDeviation from No Care Demand",
            "title": "Job Offer Probability by Age",
            "diff_col": "diff_job_offer",
            "path": path_to_plot_job_offer,
            "age_max": 70,
        },
        "hours_weekly": {
            "ylabel": "Weekly Hours\nDeviation from No Care Demand",
            "title": "Weekly Working Hours by Age",
            "diff_col": "diff_hours_weekly",
            "path": path_to_plot_hours_weekly,
            "age_max": 70,
        },
        "care": {
            "ylabel": "Care Probability\nDeviation from No Care Demand",
            "title": "Care Probability by Age",
            "diff_col": "diff_care",
            "path": path_to_plot_care,
            "age_max": 70,
        },
        "gross_labor_income": {
            "ylabel": "Gross Labor Income\nDeviation from No Care Demand",
            "title": "Gross Labor Income by Age",
            "diff_col": "diff_gross_labor_income",
            "path": path_to_plot_gross_labor_income,
            "age_max": 90,
        },
        "savings": {
            "ylabel": "Savings (in 1,000€)\nDeviation from No Care Demand",
            "title": "Savings by Age",
            "diff_col": "diff_savings",
            "path": path_to_plot_savings,
            "age_max": 90,
        },
        "wealth": {
            "ylabel": "Wealth (in 1,000€)\nDeviation from No Care Demand",
            "title": "Wealth by Age",
            "diff_col": "diff_wealth",
            "path": path_to_plot_wealth,
            "age_max": 90,
        },
        "savings_rate": {
            "ylabel": "Savings Rate\nDeviation from No Care Demand",
            "title": "Savings Rate by Age",
            "diff_col": "diff_savings_rate",
            "path": path_to_plot_savings_rate,
            "age_max": 90,
        },
        "consumption": {
            "ylabel": "Consumption (in 1,000€)\nDeviation from No Care Demand",
            "title": "Consumption by Age",
            "diff_col": "diff_consumption",
            "path": path_to_plot_savings.parent
            / "matched_differences_consumption_by_age_up_to_90.png",
        },
    }

    plot_all_outcomes_by_age(
        prof=prof,
        plot_configs=plot_configs,
        age_min=age_min,
        age_max=age_max,
    )


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_age_profiles
@pytask.mark.counterfactual_differences_higher_formal_care_costs_age_profiles
def task_plot_matched_differences_by_age_vs_baseline_up_to_90(  # noqa: PLR0915
    path_to_higher_formal_care_costs_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_higher_formal_care_costs_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_work_by_age_up_to_90.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_full_time_by_age_up_to_90.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_part_time_by_age_up_to_90.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_job_offer_by_age_up_to_90.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_working_hours_by_age_up_to_90.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_care_by_age_up_to_90.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_gross_labor_income_by_age_up_to_90.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_savings_by_age_up_to_90.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_wealth_by_age_up_to_90.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_formal_care_costs"
    / "vs_baseline"
    / "age_profiles"
    / "matched_differences_savings_rate_by_age_up_to_90.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Compute matched period differences (higher-formal-care-costs - baseline) by age, up to age 90."""
    # Load and prepare data
    df_hfc, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_higher_formal_care_costs_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
    )

    # Calculate outcomes
    hfc_outcomes = calculate_outcomes(df_hfc, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    hfc_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_hfc, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    # Calculate additional outcomes
    hfc_additional = calculate_additional_outcomes(df_hfc)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    hfc_outcomes.update(hfc_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns and merge
    hfc_cols = create_outcome_columns(df_hfc, hfc_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Add age column for age-based filtering
    if "age" in df_hfc.columns:
        hfc_cols["age"] = df_hfc["age"].values

    # Merge and compute differences (include full set of outcomes incl. consumption)
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
    merged = merge_and_compute_differences(hfc_cols, baseline_cols, outcome_names)

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
            "age_max": 70,
        },
        "ft": {
            "ylabel": "Proportion Full-time\nDeviation from Baseline",
            "title": "Full-time Employment by Age",
            "diff_col": "diff_ft",
            "path": path_to_plot_ft,
            "age_max": 70,
        },
        "pt": {
            "ylabel": "Proportion Part-time\nDeviation from Baseline",
            "title": "Part-time Employment by Age",
            "diff_col": "diff_pt",
            "path": path_to_plot_pt,
            "age_max": 70,
        },
        "job_offer": {
            "ylabel": "Job Offer Probability\nDeviation from Baseline",
            "title": "Job Offer Probability by Age",
            "diff_col": "diff_job_offer",
            "path": path_to_plot_job_offer,
            "age_max": 70,
        },
        "hours_weekly": {
            "ylabel": "Weekly Hours\nDeviation from Baseline",
            "title": "Weekly Working Hours by Age",
            "diff_col": "diff_hours_weekly",
            "path": path_to_plot_hours_weekly,
            "age_max": 70,
        },
        "care": {
            "ylabel": "Care Probability\nDeviation from Baseline",
            "title": "Care Probability by Age",
            "diff_col": "diff_care",
            "path": path_to_plot_care,
            "age_max": 70,
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
            / "matched_differences_consumption_by_age_up_to_90.png",
        },
    }

    plot_all_outcomes_by_age(
        prof=prof,
        plot_configs=plot_configs,
        age_min=age_min,
        age_max=age_max,
    )
