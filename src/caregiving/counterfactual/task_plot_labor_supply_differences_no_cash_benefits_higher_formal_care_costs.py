"""Plot differences in labor supply for no-cash-benefits higher formal care costs.

Compares the no-cash-benefits higher formal care costs counterfactual vs:
1. No care demand counterfactual
2. Baseline scenario

This mirrors the plotting structure used for higher formal care costs.
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
from caregiving.model.shared import INFORMAL_CARE


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_cash_benefits_higher_formal_care_costs
def task_plot_matched_differences_by_distance_vs_no_care_demand(  # noqa: PLR0915
    path_to_no_cash_benefits_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_cash_benefits_higher_formal_care_costs_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_no_care_demand"
    / "matched_differences_by_distance_no_cash_benefits_higher_formal_care_costs.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (no-cash-benefits - no-care-demand) by distance.

    Averages by distance to first caregiving spell in the no-cash-benefits scenario.
    """
    # Load and prepare data
    df_ncb, df_ncd = prepare_dataframes_simple(
        pd.read_pickle(path_to_no_cash_benefits_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers,
    )

    # Calculate outcomes
    ncb_work, ncb_ft, ncb_pt = calculate_simple_outcomes(df_ncb, "original")
    ncd_work, ncd_ft, ncd_pt = calculate_simple_outcomes(df_ncd, "no_care_demand")

    # Create outcome columns
    ncb_cols = df_ncb[["agent", "period"]].copy()
    ncb_cols["work_ncb"] = ncb_work
    ncb_cols["ft_ncb"] = ncb_ft
    ncb_cols["pt_ncb"] = ncb_pt

    ncd_cols = df_ncd[["agent", "period"]].copy()
    ncd_cols["work_ncd"] = ncd_work
    ncd_cols["ft_ncd"] = ncd_ft
    ncd_cols["pt_ncd"] = ncd_pt

    # Merge and compute differences (no cash benefits - no care demand)
    merged = ncb_cols.merge(ncd_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_ncb"] - merged["work_ncd"]
    merged["diff_ft"] = merged["ft_ncb"] - merged["ft_ncd"]
    merged["diff_pt"] = merged["pt_ncb"] - merged["pt_ncd"]

    # Compute distance to first care in no-cash-benefits scenario and attach
    df_ncb_dist = _add_distance_to_first_care(df_ncb)
    dist_map = (
        df_ncb_dist.groupby("agent", observed=False)["first_care_period"]
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_cash_benefits_higher_formal_care_costs
def task_plot_matched_differences_by_distance_vs_baseline(  # noqa: PLR0915
    path_to_no_cash_benefits_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_cash_benefits_higher_formal_care_costs_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_baseline"
    / "matched_differences_by_distance_no_cash_benefits_higher_formal_care_costs.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (no-cash-benefits - baseline) by distance."""
    # Load and prepare data
    df_ncb, df_baseline = prepare_dataframes_simple(
        pd.read_pickle(path_to_no_cash_benefits_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers,
    )

    # Calculate outcomes
    ncb_work, ncb_ft, ncb_pt = calculate_simple_outcomes(df_ncb, "original")
    base_work, base_ft, base_pt = calculate_simple_outcomes(df_baseline, "original")

    # Create outcome columns
    ncb_cols = df_ncb[["agent", "period"]].copy()
    ncb_cols["work_ncb"] = ncb_work
    ncb_cols["ft_ncb"] = ncb_ft
    ncb_cols["pt_ncb"] = ncb_pt

    base_cols = df_baseline[["agent", "period"]].copy()
    base_cols["work_base"] = base_work
    base_cols["ft_base"] = base_ft
    base_cols["pt_base"] = base_pt

    # Merge and compute differences (no cash benefits - baseline)
    merged = ncb_cols.merge(base_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_ncb"] - merged["work_base"]
    merged["diff_ft"] = merged["ft_ncb"] - merged["ft_base"]
    merged["diff_pt"] = merged["pt_ncb"] - merged["pt_base"]

    # Compute distance to first care in no-cash-benefits scenario and attach
    df_ncb_dist = _add_distance_to_first_care(df_ncb)
    dist_map = (
        df_ncb_dist.groupby("agent", observed=False)["first_care_period"]
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_cash_benefits_higher_formal_care_costs
def task_plot_matched_differences_by_age_at_first_care_vs_no_care_demand(  # noqa: PLR0915
    path_to_no_cash_benefits_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_cash_benefits_higher_formal_care_costs_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_no_care_demand"
    / "matched_differences_part_time_by_age_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_no_care_demand"
    / "matched_differences_full_time_by_age_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_no_care_demand"
    / "matched_differences_employment_rate_by_age_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_age_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_no_care_demand"
    / "matched_differences_working_hours_by_age_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched differences by age at first care (no-cash-benefits - no-care-demand)."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Load and prepare data
    df_ncb, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_no_cash_benefits_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    # Calculate outcomes
    ncb_outcomes = calculate_outcomes(df_ncb, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    ncb_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncb, model_params, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, model_params, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    ncb_additional = calculate_additional_outcomes(df_ncb)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    ncb_outcomes.update(ncb_additional)
    ncd_outcomes.update(ncd_additional)

    # Create outcome columns and merge
    ncb_cols = create_outcome_columns(df_ncb, ncb_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = ncb_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    # Compute differences (no cash benefits - no care demand)
    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Compute distance and age at first care from no-cash-benefits scenario
    df_ncb_dist = _add_distance_to_first_care(df_ncb)

    # Get first care period for each agent
    dist_map = (
        df_ncb_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_ncb["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_ncb, caregiving_mask, "age_at_first_care"
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    # Filter to specific ages at first care
    merged = merged[merged["age_at_first_care"].isin(ages_at_first_care)]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance and age_at_first_care
    prof = (
        merged.groupby(["distance_to_first_care", "age_at_first_care"], observed=False)[
            [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(["age_at_first_care", "distance_to_first_care"])
    )

    # Get colors for ages
    colors = get_distinct_colors(len(ages_at_first_care))

    # Plot all outcomes
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
            "ylabel": "Job Offer Probability Difference\nDeviation from No Care Demand",
            "diff_col": "diff_job_offer",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from No Care Demand",
            "diff_col": "diff_hours_weekly",
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_cash_benefits_higher_formal_care_costs
def task_plot_matched_differences_by_age_at_first_care_vs_baseline(  # noqa: PLR0915
    path_to_no_cash_benefits_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_cash_benefits_higher_formal_care_costs_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_baseline"
    / "matched_differences_part_time_by_age_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_cash_benefits_higher_formal_care_costs"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched differences by age at first care (no-cash-benefits - baseline)."""
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Load and prepare data
    df_ncb, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_no_cash_benefits_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    # Calculate outcomes
    ncb_outcomes = calculate_outcomes(df_ncb, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    ncb_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncb, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    ncb_additional = calculate_additional_outcomes(df_ncb)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    ncb_outcomes.update(ncb_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns and merge
    ncb_cols = create_outcome_columns(df_ncb, ncb_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = ncb_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    # Compute differences (no cash benefits - baseline)
    outcome_names = [
        "work",
        "ft",
        "pt",
        "job_offer",
        "hours_weekly",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Compute distance and age at first care from no-cash-benefits scenario
    df_ncb_dist = _add_distance_to_first_care(df_ncb)

    # Get first care period for each agent
    dist_map = (
        df_ncb_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_ncb["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_ncb, caregiving_mask, "age_at_first_care"
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    # Filter to specific ages at first care
    merged = merged[merged["age_at_first_care"].isin(ages_at_first_care)]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance and age_at_first_care
    prof = (
        merged.groupby(["distance_to_first_care", "age_at_first_care"], observed=False)[
            [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(["age_at_first_care", "distance_to_first_care"])
    )

    # Get colors for ages
    colors = get_distinct_colors(len(ages_at_first_care))

    # Plot all outcomes
    plot_configs = {
        "pt": {
            "path": path_to_plot_pt,
            "ylabel": "Proportion Part-Time Working\nDeviation from Baseline",
            "diff_col": "diff_pt",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from Baseline",
            "diff_col": "diff_ft",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from Baseline",
            "diff_col": "diff_work",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability Difference\nDeviation from Baseline",
            "diff_col": "diff_job_offer",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from Baseline",
            "diff_col": "diff_hours_weekly",
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
