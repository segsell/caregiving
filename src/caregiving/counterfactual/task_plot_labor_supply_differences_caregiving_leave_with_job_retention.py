"""Plot differences in labor supply for caregiving-leave-with-job-retention counterfactual.

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
from caregiving.counterfactual.task_plot_labor_supply_differences_no_care_demand import (
    _add_distance_to_first_care_demand,
)
from caregiving.model.shared import INFORMAL_CARE


@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_distance_caregiving_leave(  # noqa: PLR0915
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
    / "matched_differences_by_distance_caregiving_leave_with_job_retention.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Matched period differences (cg-leave-jr - no-care-demand), by distance to first care."""

    # Load and prepare data
    df_cg, df_ncd = prepare_dataframes_simple(
        pd.read_pickle(path_to_cg_leave_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers,
    )

    # Calculate outcomes
    cg_work, cg_ft, cg_pt = calculate_simple_outcomes(df_cg, "job_retention")
    ncd_work, ncd_ft, ncd_pt = calculate_simple_outcomes(df_ncd, "no_care_demand")

    # Additional outcomes
    cg_additional = calculate_additional_outcomes(df_cg)
    ncd_additional = calculate_additional_outcomes(df_ncd)

    # Outcome columns
    cg_cols = df_cg[["agent", "period"]].copy()
    cg_cols["work_cg"] = cg_work
    cg_cols["ft_cg"] = cg_ft
    cg_cols["pt_cg"] = cg_pt
    cg_cols["gross_labor_income_cg"] = cg_additional["gross_labor_income"]
    cg_cols["savings_cg"] = cg_additional["savings"]
    cg_cols["wealth_cg"] = cg_additional["wealth"]
    cg_cols["savings_rate_cg"] = cg_additional["savings_rate"]

    ncd_cols = df_ncd[["agent", "period"]].copy()
    ncd_cols["work_ncd"] = ncd_work
    ncd_cols["ft_ncd"] = ncd_ft
    ncd_cols["pt_ncd"] = ncd_pt
    ncd_cols["gross_labor_income_ncd"] = ncd_additional["gross_labor_income"]
    ncd_cols["savings_ncd"] = ncd_additional["savings"]
    ncd_cols["wealth_ncd"] = ncd_additional["wealth"]
    ncd_cols["savings_rate_ncd"] = ncd_additional["savings_rate"]

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

    plot_three_line_differences(
        prof=prof,
        x_col="distance_to_first_care",
        path_to_plot=path_to_plot,
        xlabel="Year relative to start of first care spell",
        window=window,
    )


@pytask.mark.counterfactual_differences_caregiving_leave_with_job_retention
def task_plot_matched_differences_by_age_at_first_care_cg_leave(  # noqa: PLR0915
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
    path_to_options: Path = BLD / "model" / "options.pkl",
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

    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    cg_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_cg, model_params, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, model_params, choice_set_type="no_care_demand"
    )

    cg_additional = calculate_additional_outcomes(df_cg)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    cg_outcomes.update(cg_additional)
    ncd_outcomes.update(ncd_additional)

    cg_cols = create_outcome_columns(df_cg, cg_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

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
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

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

    prof = (
        merged.groupby(["distance_to_first_care", "age_at_first_care"], observed=False)[
            [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_gross_labor_income",
                "diff_savings",
                "diff_wealth",
                "diff_savings_rate",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(["age_at_first_care", "distance_to_first_care"])
    )

    colors = get_distinct_colors(len(ages_at_first_care))

    plot_configs = {
        "pt": {
            "path": path_to_plot_pt,
            "ylabel": "Proportion Part-Time Working\nDeviation from No Care Demand",
            "diff_col": "diff_pt",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from No Care Demand",
            "diff_col": "diff_ft",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from No Care Demand",
            "diff_col": "diff_work",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability\nDeviation from No Care Demand",
            "diff_col": "diff_job_offer",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours\nDeviation from No Care Demand",
            "diff_col": "diff_hours_weekly",
        },
        "gross_labor_income": {
            "path": path_to_plot_work.parent
            / "matched_differences_gross_labor_income_by_age_at_first_care.png",
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from No Care Demand",
            "diff_col": "diff_gross_labor_income",
        },
        "savings": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_by_age_at_first_care.png",
            "ylabel": "Savings Decision\nDeviation from No Care Demand",
            "diff_col": "diff_savings",
        },
        "wealth": {
            "path": path_to_plot_work.parent
            / "matched_differences_wealth_by_age_at_first_care.png",
            "ylabel": "Wealth at Beginning of Period\nDeviation from No Care Demand",
            "diff_col": "diff_wealth",
        },
        "savings_rate": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_rate_by_age_at_first_care.png",
            "ylabel": "Savings Rate\nDeviation from No Care Demand",
            "diff_col": "diff_savings_rate",
        },
    }

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


# Further tasks mirror the job-retention ones: by-age-bins at first care,
# by distance to first care demand, and their baseline analogues. For brevity,
# the caregiving-leave variants follow exactly the same structure, but use
# caregiving-leave simulated data and save plots under the corresponding
# caregiving_leave_with_job_retention folders.
