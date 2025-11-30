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


@pytask.mark.counterfactual_differences_no_care_demand_age_profiles
def task_plot_matched_differences_by_age(  # noqa: PLR0915
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
    / "matched_differences_work_by_age.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "matched_differences_full_time_by_age.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "matched_differences_part_time_by_age.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "matched_differences_job_offer_by_age.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "matched_differences_working_hours_by_age.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "matched_differences_care_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "matched_differences_gross_labor_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "matched_differences_savings_by_age.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "matched_differences_wealth_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "matched_differences_savings_rate_by_age.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
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
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    o_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_o, model_params, choice_set_type="original"
    )
    c_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_c, model_params, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    o_additional = calculate_additional_outcomes(df_o)
    c_additional = calculate_additional_outcomes(df_c)
    o_outcomes.update(o_additional)
    c_outcomes.update(c_additional)

    # Create outcome columns and merge
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Add age column to o_cols for age-based filtering
    if "age" in df_o.columns:
        o_cols["age"] = df_o["age"].values

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
    ]
    merged = merge_and_compute_differences(o_cols, c_cols, outcome_names)

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
            ]
        ]
        .mean()
        .reset_index()
    )

    # Plot configurations
    plot_configs = {
        "work": {
            "ylabel": "Proportion Working\nDeviation from Counterfactual",
            "title": "Employment Rate by Age",
            "diff_col": "diff_work",
            "path": path_to_plot_work,
        },
        "ft": {
            "ylabel": "Proportion Full-time\nDeviation from Counterfactual",
            "title": "Full-time Employment by Age",
            "diff_col": "diff_ft",
            "path": path_to_plot_ft,
        },
        "pt": {
            "ylabel": "Proportion Part-time\nDeviation from Counterfactual",
            "title": "Part-time Employment by Age",
            "diff_col": "diff_pt",
            "path": path_to_plot_pt,
        },
        "job_offer": {
            "ylabel": "Job Offer Probability\nDeviation from Counterfactual",
            "title": "Job Offer Probability by Age",
            "diff_col": "diff_job_offer",
            "path": path_to_plot_job_offer,
        },
        "hours_weekly": {
            "ylabel": "Weekly Hours\nDeviation from Counterfactual",
            "title": "Weekly Working Hours by Age",
            "diff_col": "diff_hours_weekly",
            "path": path_to_plot_hours_weekly,
        },
        "care": {
            "ylabel": "Care Probability\nDeviation from Counterfactual",
            "title": "Care Probability by Age",
            "diff_col": "diff_care",
            "path": path_to_plot_care,
        },
        "gross_labor_income": {
            "ylabel": "Gross Labor Income\nDeviation from Counterfactual",
            "title": "Gross Labor Income by Age",
            "diff_col": "diff_gross_labor_income",
            "path": path_to_plot_gross_labor_income,
        },
        "savings": {
            "ylabel": "Savings\nDeviation from Counterfactual",
            "title": "Savings by Age",
            "diff_col": "diff_savings",
            "path": path_to_plot_savings,
        },
        "wealth": {
            "ylabel": "Wealth\nDeviation from Counterfactual",
            "title": "Wealth by Age",
            "diff_col": "diff_wealth",
            "path": path_to_plot_wealth,
        },
        "savings_rate": {
            "ylabel": "Savings Rate\nDeviation from Counterfactual",
            "title": "Savings Rate by Age",
            "diff_col": "diff_savings_rate",
            "path": path_to_plot_savings_rate,
        },
    }

    plot_all_outcomes_by_age(
        prof=prof,
        plot_configs=plot_configs,
        age_min=age_min,
        age_max=age_max,
    )
