"""Age-based plotting functions for no care demand counterfactual."""

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
@pytask.mark.counterfactual_differences_no_care_demand_age_profiles
def task_plot_matched_differences_by_age(  # noqa: PLR0915, E501
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_work_by_age.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_full_time_by_age.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_part_time_by_age.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_job_offer_by_age.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_working_hours_by_age.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_care_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_gross_labor_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_savings_by_age.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_wealth_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "matched_differences_savings_rate_by_age.png",
    # Caregiving type 0 plots
    path_to_plot_work_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_work_by_age.png",
    path_to_plot_ft_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_full_time_by_age.png",
    path_to_plot_pt_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_part_time_by_age.png",
    path_to_plot_job_offer_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_job_offer_by_age.png",
    path_to_plot_hours_weekly_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_working_hours_by_age.png",
    path_to_plot_care_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_care_by_age.png",
    path_to_plot_gross_labor_income_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_gross_labor_income_by_age.png",
    path_to_plot_savings_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_savings_by_age.png",
    path_to_plot_wealth_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_wealth_by_age.png",
    path_to_plot_savings_rate_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type0"
    / "matched_differences_savings_rate_by_age.png",
    # Caregiving type 1 plots
    path_to_plot_work_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_work_by_age.png",
    path_to_plot_ft_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_full_time_by_age.png",
    path_to_plot_pt_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_part_time_by_age.png",
    path_to_plot_job_offer_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_job_offer_by_age.png",
    path_to_plot_hours_weekly_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_working_hours_by_age.png",
    path_to_plot_care_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_care_by_age.png",
    path_to_plot_gross_labor_income_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_gross_labor_income_by_age.png",
    path_to_plot_savings_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_savings_by_age.png",
    path_to_plot_wealth_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_wealth_by_age.png",
    path_to_plot_savings_rate_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "caregiving_type1"
    / "matched_differences_savings_rate_by_age.png",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    age_min: int = 30,
    age_max: int = 100,
) -> None:
    """Compute matched period differences (orig - no-care-demand) by age.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Average diffs by age and plot all outcomes.
      6) Also plot separately for caregiving_type 0 and 1.

    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_original_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
    )

    # Calculate outcomes
    o_outcomes = calculate_outcomes(df_o, choice_set_type="original")
    c_outcomes = calculate_outcomes(df_c, choice_set_type="no_care_demand")

    # Calculate working hours
    specs = pickle.load(path_to_specs.open("rb"))
    o_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_o, specs, choice_set_type="original"
    )
    c_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_c, specs, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    o_additional = calculate_additional_outcomes(df_o)
    c_additional = calculate_additional_outcomes(df_c)
    o_outcomes.update(o_additional)
    c_outcomes.update(c_additional)

    # Create outcome columns and merge
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Add caregiving_type from original dataframe to o_cols for filtering
    if "caregiving_type" in df_o.columns:
        o_cols["caregiving_type"] = df_o["caregiving_type"].values

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
    merged = merge_and_compute_differences(o_cols, c_cols, outcome_names)

    # Filter to age range
    merged = merged[(merged["age"] >= age_min) & (merged["age"] <= age_max)]

    # Average differences by age (for all caregiving types)
    prof_all = (
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

    # Plot configurations for all types
    plot_configs_all = {
        "work": {
            "ylabel": "Proportion Working\nDeviation from Counterfactual",
            "title": "Employment Rate by Age",
            "diff_col": "diff_work",
            "path": path_to_plot_work,
            "age_max": 70,
        },
        "ft": {
            "ylabel": "Proportion Full-time\nDeviation from Counterfactual",
            "title": "Full-time Employment by Age",
            "diff_col": "diff_ft",
            "path": path_to_plot_ft,
            "age_max": 70,
        },
        "pt": {
            "ylabel": "Proportion Part-time\nDeviation from Counterfactual",
            "title": "Part-time Employment by Age",
            "diff_col": "diff_pt",
            "path": path_to_plot_pt,
            "age_max": 70,
        },
        "job_offer": {
            "ylabel": "Job Offer Probability\nDeviation from Counterfactual",
            "title": "Job Offer Probability by Age",
            "diff_col": "diff_job_offer",
            "path": path_to_plot_job_offer,
            "age_max": 70,
        },
        "hours_weekly": {
            "ylabel": "Weekly Hours\nDeviation from Counterfactual",
            "title": "Weekly Working Hours by Age",
            "diff_col": "diff_hours_weekly",
            "path": path_to_plot_hours_weekly,
            "age_max": 70,
        },
        "care": {
            "ylabel": "Care Probability\nDeviation from Counterfactual",
            "title": "Care Probability by Age",
            "diff_col": "diff_care",
            "path": path_to_plot_care,
            "age_max": 70,
        },
        "gross_labor_income": {
            "ylabel": "Gross Labor Income\nDeviation from Counterfactual",
            "title": "Gross Labor Income by Age",
            "diff_col": "diff_gross_labor_income",
            "path": path_to_plot_gross_labor_income,
            "age_max": 90,
        },
        "savings": {
            "ylabel": "Savings (in 1,000€)\nDeviation from Counterfactual",
            "title": "Savings by Age",
            "diff_col": "diff_savings",
            "path": path_to_plot_savings,
            "age_max": 90,
        },
        "wealth": {
            "ylabel": "Wealth (in 1,000€)\nDeviation from Counterfactual",
            "title": "Wealth by Age",
            "diff_col": "diff_wealth",
            "path": path_to_plot_wealth,
            "age_max": 90,
        },
        "savings_rate": {
            "ylabel": "Savings Rate\nDeviation from Counterfactual",
            "title": "Savings Rate by Age",
            "diff_col": "diff_savings_rate",
            "path": path_to_plot_savings_rate,
            "age_max": 90,
        },
        "consumption": {
            "ylabel": "Consumption (in 1,000€)\nDeviation from Counterfactual",
            "title": "Consumption by Age",
            "diff_col": "diff_consumption",
            "path": path_to_plot_savings.parent
            / "matched_differences_consumption_by_age.png",
            "age_max": 90,
        },
    }

    # Plot for all caregiving types
    plot_all_outcomes_by_age(
        prof=prof_all,
        plot_configs=plot_configs_all,
        age_min=age_min,
        age_max=age_max,
    )

    # Plot separately for caregiving_type 0 and 1 if caregiving_type column exists
    if "caregiving_type" in merged.columns:
        # Plot configurations for caregiving_type 0
        plot_configs_type0 = {
            "work": {
                "ylabel": "Proportion Working\nDeviation from Counterfactual",
                "title": "Employment Rate by Age (Caregiving Type 0)",
                "diff_col": "diff_work",
                "path": path_to_plot_work_type0,
                "age_max": 70,
            },
            "ft": {
                "ylabel": "Proportion Full-time\nDeviation from Counterfactual",
                "title": "Full-time Employment by Age (Caregiving Type 0)",
                "diff_col": "diff_ft",
                "path": path_to_plot_ft_type0,
                "age_max": 70,
            },
            "pt": {
                "ylabel": "Proportion Part-time\nDeviation from Counterfactual",
                "title": "Part-time Employment by Age (Caregiving Type 0)",
                "diff_col": "diff_pt",
                "path": path_to_plot_pt_type0,
                "age_max": 70,
            },
            "job_offer": {
                "ylabel": "Job Offer Probability\nDeviation from Counterfactual",
                "title": "Job Offer Probability by Age (Caregiving Type 0)",
                "diff_col": "diff_job_offer",
                "path": path_to_plot_job_offer_type0,
                "age_max": 70,
            },
            "hours_weekly": {
                "ylabel": "Weekly Hours\nDeviation from Counterfactual",
                "title": "Weekly Working Hours by Age (Caregiving Type 0)",
                "diff_col": "diff_hours_weekly",
                "path": path_to_plot_hours_weekly_type0,
                "age_max": 70,
            },
            "care": {
                "ylabel": "Care Probability\nDeviation from Counterfactual",
                "title": "Care Probability by Age (Caregiving Type 0)",
                "diff_col": "diff_care",
                "path": path_to_plot_care_type0,
                "age_max": 70,
            },
            "gross_labor_income": {
                "ylabel": "Gross Labor Income\nDeviation from Counterfactual",
                "title": "Gross Labor Income by Age (Caregiving Type 0)",
                "diff_col": "diff_gross_labor_income",
                "path": path_to_plot_gross_labor_income_type0,
                "age_max": 90,
            },
            "savings": {
                "ylabel": "Savings (in 1,000€)\nDeviation from Counterfactual",
                "title": "Savings by Age (Caregiving Type 0)",
                "diff_col": "diff_savings",
                "path": path_to_plot_savings_type0,
                "age_max": 90,
            },
            "wealth": {
                "ylabel": "Wealth (in 1,000€)\nDeviation from Counterfactual",
                "title": "Wealth by Age (Caregiving Type 0)",
                "diff_col": "diff_wealth",
                "path": path_to_plot_wealth_type0,
                "age_max": 90,
            },
            "savings_rate": {
                "ylabel": "Savings Rate\nDeviation from Counterfactual",
                "title": "Savings Rate by Age (Caregiving Type 0)",
                "diff_col": "diff_savings_rate",
                "path": path_to_plot_savings_rate_type0,
                "age_max": 90,
            },
            "consumption": {
                "ylabel": "Consumption (in 1,000€)\nDeviation from Counterfactual",
                "title": "Consumption by Age (Caregiving Type 0)",
                "diff_col": "diff_consumption",
                "path": path_to_plot_savings_type0.parent
                / "matched_differences_consumption_by_age.png",
                "age_max": 90,
            },
        }

        # Plot configurations for caregiving_type 1
        plot_configs_type1 = {
            "work": {
                "ylabel": "Proportion Working\nDeviation from Counterfactual",
                "title": "Employment Rate by Age (Caregiving Type 1)",
                "diff_col": "diff_work",
                "path": path_to_plot_work_type1,
                "age_max": 70,
            },
            "ft": {
                "ylabel": "Proportion Full-time\nDeviation from Counterfactual",
                "title": "Full-time Employment by Age (Caregiving Type 1)",
                "diff_col": "diff_ft",
                "path": path_to_plot_ft_type1,
                "age_max": 70,
            },
            "pt": {
                "ylabel": "Proportion Part-time\nDeviation from Counterfactual",
                "title": "Part-time Employment by Age (Caregiving Type 1)",
                "diff_col": "diff_pt",
                "path": path_to_plot_pt_type1,
                "age_max": 70,
            },
            "job_offer": {
                "ylabel": "Job Offer Probability\nDeviation from Counterfactual",
                "title": "Job Offer Probability by Age (Caregiving Type 1)",
                "diff_col": "diff_job_offer",
                "path": path_to_plot_job_offer_type1,
                "age_max": 70,
            },
            "hours_weekly": {
                "ylabel": "Weekly Hours\nDeviation from Counterfactual",
                "title": "Weekly Working Hours by Age (Caregiving Type 1)",
                "diff_col": "diff_hours_weekly",
                "path": path_to_plot_hours_weekly_type1,
                "age_max": 70,
            },
            "care": {
                "ylabel": "Care Probability\nDeviation from Counterfactual",
                "title": "Care Probability by Age (Caregiving Type 1)",
                "diff_col": "diff_care",
                "path": path_to_plot_care_type1,
                "age_max": 70,
            },
            "gross_labor_income": {
                "ylabel": "Gross Labor Income\nDeviation from Counterfactual",
                "title": "Gross Labor Income by Age (Caregiving Type 1)",
                "diff_col": "diff_gross_labor_income",
                "path": path_to_plot_gross_labor_income_type1,
                "age_max": 90,
            },
            "savings": {
                "ylabel": "Savings (in 1,000€)\nDeviation from Counterfactual",
                "title": "Savings by Age (Caregiving Type 1)",
                "diff_col": "diff_savings",
                "path": path_to_plot_savings_type1,
                "age_max": 90,
            },
            "wealth": {
                "ylabel": "Wealth (in 1,000€)\nDeviation from Counterfactual",
                "title": "Wealth by Age (Caregiving Type 1)",
                "diff_col": "diff_wealth",
                "path": path_to_plot_wealth_type1,
                "age_max": 90,
            },
            "savings_rate": {
                "ylabel": "Savings Rate\nDeviation from Counterfactual",
                "title": "Savings Rate by Age (Caregiving Type 1)",
                "diff_col": "diff_savings_rate",
                "path": path_to_plot_savings_rate_type1,
                "age_max": 90,
            },
            "consumption": {
                "ylabel": "Consumption (in 1,000€)\nDeviation from Counterfactual",
                "title": "Consumption by Age (Caregiving Type 1)",
                "diff_col": "diff_consumption",
                "path": path_to_plot_savings_type1.parent
                / "matched_differences_consumption_by_age.png",
                "age_max": 90,
            },
        }

        # Filter merged data by caregiving_type and create profiles
        merged_type0 = merged[merged["caregiving_type"] == 0].copy()
        merged_type1 = merged[merged["caregiving_type"] == 1].copy()

        # Average differences by age for caregiving_type 0
        prof_type0 = (
            merged_type0.groupby("age", observed=False)[
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

        # Average differences by age for caregiving_type 1
        prof_type1 = (
            merged_type1.groupby("age", observed=False)[
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

        # Plot for caregiving_type 0
        plot_all_outcomes_by_age(
            prof=prof_type0,
            plot_configs=plot_configs_type0,
            age_min=age_min,
            age_max=age_max,
        )

        # Plot for caregiving_type 1
        plot_all_outcomes_by_age(
            prof=prof_type1,
            plot_configs=plot_configs_type1,
            age_min=age_min,
            age_max=age_max,
        )
