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
    # Skip if required data file doesn't exist
    if not path_to_no_cash_benefits_data.exists():
        return

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
    # Skip if required data file doesn't exist
    if not path_to_no_cash_benefits_data.exists():
        return

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
    # Skip if required data file doesn't exist
    if not path_to_no_cash_benefits_data.exists():
        return

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
    # Skip if required data file doesn't exist
    if not path_to_no_cash_benefits_data.exists():
        return

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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_cash_benefits_higher_formal_care_costs
def task_plot_matched_differences_first_care_start_by_age_no_cash_benefits_higher_formal_care_costs(  # noqa: PLR0915
    path_to_no_cash_benefits_higher_formal_care_costs_data: Path = BLD
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
    / "matched_differences_first_care_start_by_age.png",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Plot matched differences in distribution of first care start ages (No Cash Benefits Higher Formal Care Costs vs no care)."""
    from caregiving.model.shared import DEAD

    # Load data
    df_cf = pd.read_pickle(path_to_no_cash_benefits_higher_formal_care_costs_data)
    df_ncd = pd.read_pickle(path_to_no_care_demand_data)

    # Alive restriction
    df_cf = df_cf[df_cf["health"] != DEAD].copy()
    df_ncd = df_ncd[df_ncd["health"] != DEAD].copy()

    # Ensure agent/period
    df_cf = _ensure_agent_period(df_cf)
    df_ncd = _ensure_agent_period(df_ncd)

    # Fully flatten any residual index levels
    for df in (df_cf, df_ncd):
        if isinstance(df.index, pd.MultiIndex):
            idx_names = {n for n in df.index.names if n is not None}
            if ("agent" in idx_names) or ("period" in idx_names):
                df.reset_index(inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Check for age column
    if "age" not in df_cf.columns or "age" not in df_ncd.columns:
        raise ValueError("Age column required but not found in data")

    # Find first care period for each agent
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()

    # For counterfactual
    caregiving_mask_cf = df_cf["choice"].isin(care_codes)
    first_care_cf = (
        df_cf.loc[caregiving_mask_cf, ["agent", "period", "age"]]
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
    first_care_cf = first_care_cf[
        (first_care_cf["age_at_first_care"] >= min_age)
        & (first_care_cf["age_at_first_care"] <= max_age)
    ]
    first_care_ncd = first_care_ncd[
        (first_care_ncd["age_at_first_care"] >= min_age)
        & (first_care_ncd["age_at_first_care"] <= max_age)
    ]

    # Count by age
    counts_cf = (
        first_care_cf["age_at_first_care"]
        .value_counts()
        .sort_index()
        .reset_index(name="count_cf")
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
    counts_cf = all_ages.merge(counts_cf, on="age", how="left").fillna(0)
    counts_ncd = all_ages.merge(counts_ncd, on="age", how="left").fillna(0)
    counts_cf["count_cf"] = counts_cf["count_cf"].astype(int)
    counts_ncd["count_ncd"] = counts_ncd["count_ncd"].astype(int)

    # Calculate difference
    counts_diff = counts_cf.copy()
    counts_diff["count_diff"] = counts_diff["count_cf"] - counts_ncd["count_ncd"]

    # Plot
    plt.figure(figsize=(12, 7))
    colors = ["#2E86AB" if x >= 0 else "#A23B72" for x in counts_diff["count_diff"]]
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
        "Difference in number of people\n(No Cash Benefits Higher Formal Care Costs - No Care Demand)",
        fontsize=16,
    )
    plt.title("Matched Differences in First Care Start by Age", fontsize=18)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_cash_benefits_higher_formal_care_costs
def task_plot_matched_differences_first_care_demand_start_by_age_no_cash_benefits_higher_formal_care_costs(  # noqa: PLR0915
    path_to_no_cash_benefits_higher_formal_care_costs_data: Path = BLD
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
    / "matched_differences_first_care_demand_start_by_age.png",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Plot matched differences in distribution of first care demand start ages (No Cash Benefits Higher Formal Care Costs vs no care)."""
    from caregiving.model.shared import DEAD

    # Load data
    df_cf = pd.read_pickle(path_to_no_cash_benefits_higher_formal_care_costs_data)
    df_ncd = pd.read_pickle(path_to_no_care_demand_data)

    # Alive restriction
    df_cf = df_cf[df_cf["health"] != DEAD].copy()
    df_ncd = df_ncd[df_ncd["health"] != DEAD].copy()

    # Ensure agent/period
    df_cf = _ensure_agent_period(df_cf)
    df_ncd = _ensure_agent_period(df_ncd)

    # Fully flatten any residual index levels
    for df in [df_cf, df_ncd]:
        if isinstance(df.index, pd.MultiIndex):
            idx_names = {n for n in df.index.names if n is not None}
            if ("agent" in idx_names) or ("period" in idx_names):
                df.reset_index(inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Check for age column
    if "age" not in df_cf.columns or "age" not in df_ncd.columns:
        raise ValueError("Age column required but not found in data")

    # Find first period where care_demand > 0 for each agent
    care_demand_mask_cf = df_cf["care_demand"] > 0

    first_care_demand_cf = (
        df_cf.loc[care_demand_mask_cf, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(
            columns={
                "period": "first_care_demand_period",
                "age": "age_at_first_care_demand",
            }
        )
    )

    # For no_care_demand scenario, there is no care_demand column, so create empty dataframe
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
    first_care_demand_cf = first_care_demand_cf[
        (first_care_demand_cf["age_at_first_care_demand"] >= min_age)
        & (first_care_demand_cf["age_at_first_care_demand"] <= max_age)
    ]
    if len(first_care_demand_ncd) > 0:
        first_care_demand_ncd = first_care_demand_ncd[
            (first_care_demand_ncd["age_at_first_care_demand"] >= min_age)
            & (first_care_demand_ncd["age_at_first_care_demand"] <= max_age)
        ]

    # Count by age
    if len(first_care_demand_cf) > 0:
        counts_cf = (
            first_care_demand_cf["age_at_first_care_demand"]
            .value_counts()
            .sort_index()
            .reset_index(name="count_cf")
            .rename(columns={"age_at_first_care_demand": "age"})
        )
    else:
        counts_cf = pd.DataFrame(columns=["age", "count_cf"])

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
    counts_cf = all_ages.merge(counts_cf, on="age", how="left").fillna(0)
    counts_ncd = all_ages.merge(counts_ncd, on="age", how="left").fillna(0)
    counts_cf["count_cf"] = counts_cf["count_cf"].astype(int)
    counts_ncd["count_ncd"] = counts_ncd["count_ncd"].astype(int)

    # Calculate difference
    counts_diff = counts_cf.copy()
    counts_diff["count_diff"] = counts_diff["count_cf"] - counts_ncd["count_ncd"]

    # Plot
    plt.figure(figsize=(12, 7))
    colors = ["#2E86AB" if x >= 0 else "#A23B72" for x in counts_diff["count_diff"]]
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
        "Difference in number of people\n(No Cash Benefits Higher Formal Care Costs - No Care Demand)",
        fontsize=16,
    )
    plt.title("Matched Differences in First Care Demand Start by Age", fontsize=18)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()
