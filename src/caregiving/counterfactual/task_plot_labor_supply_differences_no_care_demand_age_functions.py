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
def task_plot_matched_differences_by_age_no_care_demand(  # noqa: PLR0915, E501
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
    / "all"
    / "age_profiles_work_by_age.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_full_time_by_age.png",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_part_time_by_age.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_job_offer_by_age.png",
    path_to_plot_hours_weekly: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_working_hours_by_age.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_care_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_gross_labor_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_savings_by_age.png",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_wealth_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_savings_rate_by_age.png",
    path_to_plot_savings_dec: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_savings_dec_by_age.png",
    # Caregiving type 0 plots
    path_to_plot_work_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_work_by_age.png",
    path_to_plot_ft_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_full_time_by_age.png",
    path_to_plot_pt_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_part_time_by_age.png",
    path_to_plot_job_offer_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_job_offer_by_age.png",
    path_to_plot_hours_weekly_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_working_hours_by_age.png",
    path_to_plot_care_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_care_by_age.png",
    path_to_plot_gross_labor_income_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_gross_labor_income_by_age.png",
    path_to_plot_savings_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_savings_by_age.png",
    path_to_plot_wealth_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_wealth_by_age.png",
    path_to_plot_savings_rate_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_savings_rate_by_age.png",
    path_to_plot_savings_dec_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_savings_dec_by_age.png",
    # Caregiving type 1 plots
    path_to_plot_work_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_work_by_age.png",
    path_to_plot_ft_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_full_time_by_age.png",
    path_to_plot_pt_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_part_time_by_age.png",
    path_to_plot_job_offer_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_job_offer_by_age.png",
    path_to_plot_hours_weekly_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_working_hours_by_age.png",
    path_to_plot_care_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_care_by_age.png",
    path_to_plot_gross_labor_income_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_gross_labor_income_by_age.png",
    path_to_plot_savings_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_savings_by_age.png",
    path_to_plot_wealth_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_wealth_by_age.png",
    path_to_plot_savings_rate_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_savings_rate_by_age.png",
    path_to_plot_savings_dec_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_savings_dec_by_age.png",
    # Additional outcomes - all/all
    path_to_plot_net_hh_income: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_net_hh_income_by_age.png",
    path_to_plot_income_tax: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_income_tax_by_age.png",
    path_to_plot_income_tax_single: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_income_tax_single_by_age.png",
    path_to_plot_total_tax_revenue: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_total_tax_revenue_by_age.png",
    path_to_plot_net_government_budget: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_net_government_budget_by_age.png",
    path_to_plot_own_income_after_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_own_income_after_ssc_by_age.png",
    path_to_plot_exp_years: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_exp_years_by_age.png",
    path_to_plot_caregiving_leave_top_up: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "all"
    / "age_profiles_caregiving_leave_top_up_by_age.png",
    # Additional outcomes - type0
    path_to_plot_net_hh_income_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_net_hh_income_by_age.png",
    path_to_plot_income_tax_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_income_tax_by_age.png",
    path_to_plot_income_tax_single_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_income_tax_single_by_age.png",
    path_to_plot_total_tax_revenue_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_total_tax_revenue_by_age.png",
    path_to_plot_net_government_budget_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_net_government_budget_by_age.png",
    path_to_plot_own_income_after_ssc_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_own_income_after_ssc_by_age.png",
    path_to_plot_exp_years_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_exp_years_by_age.png",
    path_to_plot_caregiving_leave_top_up_type0: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type0"
    / "age_profiles_caregiving_leave_top_up_by_age.png",
    # Additional outcomes - type1
    path_to_plot_net_hh_income_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_net_hh_income_by_age.png",
    path_to_plot_income_tax_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_income_tax_by_age.png",
    path_to_plot_income_tax_single_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_income_tax_single_by_age.png",
    path_to_plot_total_tax_revenue_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_total_tax_revenue_by_age.png",
    path_to_plot_net_government_budget_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_net_government_budget_by_age.png",
    path_to_plot_own_income_after_ssc_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_own_income_after_ssc_by_age.png",
    path_to_plot_exp_years_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_exp_years_by_age.png",
    path_to_plot_caregiving_leave_top_up_type1: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "age_profiles"
    / "all"
    / "caregiving_type1"
    / "age_profiles_caregiving_leave_top_up_by_age.png",
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

    # Load specs for working hours and additional outcomes
    specs = pickle.load(path_to_specs.open("rb"))
    o_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_o, specs, choice_set_type="original"
    )
    c_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_c, specs, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    o_additional = calculate_additional_outcomes(df_o, specs)
    c_additional = calculate_additional_outcomes(df_c, specs)
    o_outcomes.update(o_additional)
    c_outcomes.update(c_additional)

    # Add savings_dec directly from dataframe if available
    if "savings_dec" in df_o.columns:
        o_outcomes["savings_dec"] = df_o["savings_dec"].values
    else:
        o_outcomes["savings_dec"] = np.zeros(len(df_o))
    if "savings_dec" in df_c.columns:
        c_outcomes["savings_dec"] = df_c["savings_dec"].values
    else:
        c_outcomes["savings_dec"] = np.zeros(len(df_c))

    # Add additional outcomes directly from dataframe if available
    additional_outcome_names = [
        "net_hh_income",
        "income_tax",
        "income_tax_single",
        "total_tax_revenue",
        "net_government_budget",
        "own_income_after_ssc",
        "exp_years",
        "caregiving_leave_top_up",
    ]
    for outcome_name in additional_outcome_names:
        if outcome_name in df_o.columns:
            o_outcomes[outcome_name] = df_o[outcome_name].values
        else:
            o_outcomes[outcome_name] = np.zeros(len(df_o))
        if outcome_name in df_c.columns:
            c_outcomes[outcome_name] = df_c[outcome_name].values
        else:
            c_outcomes[outcome_name] = np.zeros(len(df_c))

    # Create outcome columns and merge
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Add caregiving_type and education from original dataframe to o_cols for filtering
    if "caregiving_type" in df_o.columns:
        o_cols["caregiving_type"] = df_o["caregiving_type"].values
    if "education" in df_o.columns:
        o_cols["education"] = df_o["education"].values

    # Merge and compute differences (include full set of outcomes incl. consumption and bequest)
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
        "savings_dec",
        "consumption",
        "bequest_from_parent",
        "net_hh_income",
        "income_tax",
        "income_tax_single",
        "total_tax_revenue",
        "net_government_budget",
        "own_income_after_ssc",
        "exp_years",
        "caregiving_leave_top_up",
    ]
    merged = merge_and_compute_differences(o_cols, c_cols, outcome_names)

    # Filter to age range
    merged = merged[(merged["age"] >= age_min) & (merged["age"] <= age_max)]

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
    def create_plot_configs(base_path, suffix=""):
        """Create plot configs dictionary with dynamic paths."""
        title_suffix = f" ({suffix})" if suffix else ""
        return {
            "work": {
                "ylabel": "Proportion Working\nDeviation from Counterfactual",
                "title": f"Employment Rate by Age{title_suffix}",
                "diff_col": "diff_work",
                "path": base_path / "matched_differences_work_by_age.png",
                "age_max": 70,
            },
            "ft": {
                "ylabel": "Proportion Full-time\nDeviation from Counterfactual",
                "title": f"Full-time Employment by Age{title_suffix}",
                "diff_col": "diff_ft",
                "path": base_path / "matched_differences_full_time_by_age.png",
                "age_max": 70,
            },
            "pt": {
                "ylabel": "Proportion Part-time\nDeviation from Counterfactual",
                "title": f"Part-time Employment by Age{title_suffix}",
                "diff_col": "diff_pt",
                "path": base_path / "matched_differences_part_time_by_age.png",
                "age_max": 70,
            },
            "job_offer": {
                "ylabel": "Job Offer Probability\nDeviation from Counterfactual",
                "title": f"Job Offer Probability by Age{title_suffix}",
                "diff_col": "diff_job_offer",
                "path": base_path / "matched_differences_job_offer_by_age.png",
                "age_max": 70,
            },
            "hours_weekly": {
                "ylabel": "Weekly Hours\nDeviation from Counterfactual",
                "title": f"Weekly Working Hours by Age{title_suffix}",
                "diff_col": "diff_hours_weekly",
                "path": base_path / "matched_differences_working_hours_by_age.png",
                "age_max": 70,
            },
            "care": {
                "ylabel": "Care Probability\nDeviation from Counterfactual",
                "title": f"Care Probability by Age{title_suffix}",
                "diff_col": "diff_care",
                "path": base_path / "matched_differences_care_by_age.png",
                "age_max": 70,
            },
            "gross_labor_income": {
                "ylabel": "Gross Labor Income\nDeviation from Counterfactual",
                "title": f"Gross Labor Income by Age{title_suffix}",
                "diff_col": "diff_gross_labor_income",
                "path": base_path / "matched_differences_gross_labor_income_by_age.png",
                "age_max": 90,
            },
            "savings": {
                "ylabel": "Savings (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Savings by Age{title_suffix}",
                "diff_col": "diff_savings",
                "path": base_path / "matched_differences_savings_by_age.png",
                "age_max": 90,
            },
            "wealth": {
                "ylabel": "Wealth (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Wealth by Age{title_suffix}",
                "diff_col": "diff_wealth",
                "path": base_path / "matched_differences_wealth_by_age.png",
                "age_max": 90,
            },
            "savings_rate": {
                "ylabel": "Savings Rate\nDeviation from Counterfactual",
                "title": f"Savings Rate by Age{title_suffix}",
                "diff_col": "diff_savings_rate",
                "path": base_path / "matched_differences_savings_rate_by_age.png",
                "age_max": 90,
            },
            "savings_dec": {
                "ylabel": "Savings Decision (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Savings Decision by Age{title_suffix}",
                "diff_col": "diff_savings_dec",
                "path": base_path / "matched_differences_savings_dec_by_age.png",
                "age_max": 90,
            },
            "consumption": {
                "ylabel": "Consumption (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Consumption by Age{title_suffix}",
                "diff_col": "diff_consumption",
                "path": base_path / "matched_differences_consumption_by_age.png",
                "age_max": 90,
            },
            "bequest_from_parent": {
                "ylabel": "Bequest from Parent (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Bequest from Parent by Age{title_suffix}",
                "diff_col": "diff_bequest_from_parent",
                "path": base_path
                / "matched_differences_bequest_from_parent_by_age.png",
                "age_max": 90,
            },
            "net_hh_income": {
                "ylabel": "Net Household Income (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Net Household Income by Age{title_suffix}",
                "diff_col": "diff_net_hh_income",
                "path": base_path / "matched_differences_net_hh_income_by_age.png",
                "age_max": 90,
            },
            "income_tax": {
                "ylabel": "Income Tax (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Income Tax by Age{title_suffix}",
                "diff_col": "diff_income_tax",
                "path": base_path / "matched_differences_income_tax_by_age.png",
                "age_max": 90,
            },
            "income_tax_single": {
                "ylabel": "Income Tax Single (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Income Tax Single by Age{title_suffix}",
                "diff_col": "diff_income_tax_single",
                "path": base_path / "matched_differences_income_tax_single_by_age.png",
                "age_max": 90,
            },
            "total_tax_revenue": {
                "ylabel": "Total Tax Revenue (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Total Tax Revenue by Age{title_suffix}",
                "diff_col": "diff_total_tax_revenue",
                "path": base_path / "matched_differences_total_tax_revenue_by_age.png",
                "age_max": 90,
            },
            "net_government_budget": {
                "ylabel": "Net Government Budget (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Net Government Budget by Age{title_suffix}",
                "diff_col": "diff_net_government_budget",
                "path": base_path
                / "matched_differences_net_government_budget_by_age.png",
                "age_max": 90,
            },
            "own_income_after_ssc": {
                "ylabel": "Own Income After SSC (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Own Income After SSC by Age{title_suffix}",
                "diff_col": "diff_own_income_after_ssc",
                "path": base_path
                / "matched_differences_own_income_after_ssc_by_age.png",
                "age_max": 90,
            },
            "exp_years": {
                "ylabel": "Experience Years\nDeviation from Counterfactual",
                "title": f"Experience Years by Age{title_suffix}",
                "diff_col": "diff_exp_years",
                "path": base_path / "matched_differences_exp_years_by_age.png",
                "age_max": 90,
            },
            "caregiving_leave_top_up": {
                "ylabel": "Caregiving Leave Top-up (in 1,000€)\nDeviation from Counterfactual",
                "title": f"Caregiving Leave Top-up by Age{title_suffix}",
                "diff_col": "diff_caregiving_leave_top_up",
                "path": base_path
                / "matched_differences_caregiving_leave_top_up_by_age.png",
                "age_max": 90,
            },
        }

    # Loop over education and caregiving_type combinations
    for edu_name, edu_filter, edu_folder in education_specs:
        # Filter by education if specified
        if edu_filter is not None and "education" in merged.columns:
            merged_edu = merged[merged["education"] == edu_filter].copy()
        else:
            merged_edu = merged.copy()

        for cg_name, cg_filter, cg_folder in caregiving_type_specs:
            # Filter by caregiving_type if specified
            if cg_filter is not None and "caregiving_type" in merged_edu.columns:
                merged_spec = merged_edu[
                    merged_edu["caregiving_type"] == cg_filter
                ].copy()
            else:
                merged_spec = merged_edu.copy()

            # Create base path for this specification
            # path_to_plot_work structure: .../matched_differences/all/all/filename.png
            # We need: .../matched_differences/{edu_folder}/{cg_folder}/filename.png
            base_path = path_to_plot_work.parent.parent.parent / edu_folder / cg_folder
            base_path.mkdir(parents=True, exist_ok=True)

            # Create title suffix
            title_parts = []
            if edu_name != "all":
                title_parts.append(edu_name.replace("_", " ").title())
            if cg_name != "all":
                title_parts.append(cg_name.replace("_", " ").title())
            suffix = ", ".join(title_parts) if title_parts else ""

            # Average differences by age for this specification
            prof = (
                merged_spec.groupby("age", observed=False)[
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
                        "diff_savings_dec",
                        "diff_consumption",
                        "diff_bequest_from_parent",
                        "diff_net_hh_income",
                        "diff_income_tax",
                        "diff_income_tax_single",
                        "diff_total_tax_revenue",
                        "diff_net_government_budget",
                        "diff_own_income_after_ssc",
                        "diff_exp_years",
                        "diff_caregiving_leave_top_up",
                    ]
                ]
                .mean()
                .reset_index()
            )

            # Create plot configs for this specification
            plot_configs = create_plot_configs(base_path, suffix)

            # Update plot_configs paths for "all/all" case to use declared product paths
            if edu_folder == "all" and cg_folder == "all":
                declared_paths_map = {
                    "work": path_to_plot_work,
                    "ft": path_to_plot_ft,
                    "pt": path_to_plot_pt,
                    "job_offer": path_to_plot_job_offer,
                    "hours_weekly": path_to_plot_hours_weekly,
                    "care": path_to_plot_care,
                    "gross_labor_income": path_to_plot_gross_labor_income,
                    "savings": path_to_plot_savings,
                    "wealth": path_to_plot_wealth,
                    "savings_rate": path_to_plot_savings_rate,
                    "savings_dec": path_to_plot_savings_dec,
                    "net_hh_income": path_to_plot_net_hh_income,
                    "income_tax": path_to_plot_income_tax,
                    "income_tax_single": path_to_plot_income_tax_single,
                    "total_tax_revenue": path_to_plot_total_tax_revenue,
                    "net_government_budget": path_to_plot_net_government_budget,
                    "own_income_after_ssc": path_to_plot_own_income_after_ssc,
                    "exp_years": path_to_plot_exp_years,
                    "caregiving_leave_top_up": path_to_plot_caregiving_leave_top_up,
                }
                for outcome_key, declared_path in declared_paths_map.items():
                    if outcome_key in plot_configs:
                        plot_configs[outcome_key]["path"] = declared_path

            # Also update paths for caregiving_type0 and caregiving_type1 cases
            elif edu_folder == "all" and cg_folder == "caregiving_type0":
                declared_paths_map = {
                    "work": path_to_plot_work_type0,
                    "ft": path_to_plot_ft_type0,
                    "pt": path_to_plot_pt_type0,
                    "job_offer": path_to_plot_job_offer_type0,
                    "hours_weekly": path_to_plot_hours_weekly_type0,
                    "care": path_to_plot_care_type0,
                    "gross_labor_income": path_to_plot_gross_labor_income_type0,
                    "savings": path_to_plot_savings_type0,
                    "wealth": path_to_plot_wealth_type0,
                    "savings_rate": path_to_plot_savings_rate_type0,
                    "savings_dec": path_to_plot_savings_dec_type0,
                    "net_hh_income": path_to_plot_net_hh_income_type0,
                    "income_tax": path_to_plot_income_tax_type0,
                    "income_tax_single": path_to_plot_income_tax_single_type0,
                    "total_tax_revenue": path_to_plot_total_tax_revenue_type0,
                    "net_government_budget": path_to_plot_net_government_budget_type0,
                    "own_income_after_ssc": path_to_plot_own_income_after_ssc_type0,
                    "exp_years": path_to_plot_exp_years_type0,
                    "caregiving_leave_top_up": path_to_plot_caregiving_leave_top_up_type0,
                }
                for outcome_key, declared_path in declared_paths_map.items():
                    if outcome_key in plot_configs:
                        plot_configs[outcome_key]["path"] = declared_path

            elif edu_folder == "all" and cg_folder == "caregiving_type1":
                declared_paths_map = {
                    "work": path_to_plot_work_type1,
                    "ft": path_to_plot_ft_type1,
                    "pt": path_to_plot_pt_type1,
                    "job_offer": path_to_plot_job_offer_type1,
                    "hours_weekly": path_to_plot_hours_weekly_type1,
                    "care": path_to_plot_care_type1,
                    "gross_labor_income": path_to_plot_gross_labor_income_type1,
                    "savings": path_to_plot_savings_type1,
                    "wealth": path_to_plot_wealth_type1,
                    "savings_rate": path_to_plot_savings_rate_type1,
                    "savings_dec": path_to_plot_savings_dec_type1,
                    "net_hh_income": path_to_plot_net_hh_income_type1,
                    "income_tax": path_to_plot_income_tax_type1,
                    "income_tax_single": path_to_plot_income_tax_single_type1,
                    "total_tax_revenue": path_to_plot_total_tax_revenue_type1,
                    "net_government_budget": path_to_plot_net_government_budget_type1,
                    "own_income_after_ssc": path_to_plot_own_income_after_ssc_type1,
                    "exp_years": path_to_plot_exp_years_type1,
                    "caregiving_leave_top_up": path_to_plot_caregiving_leave_top_up_type1,
                }
                for outcome_key, declared_path in declared_paths_map.items():
                    if outcome_key in plot_configs:
                        plot_configs[outcome_key]["path"] = declared_path

            # Plot for this specification
            plot_all_outcomes_by_age(
                prof=prof,
                plot_configs=plot_configs,
                age_min=age_min,
                age_max=age_max,
            )
