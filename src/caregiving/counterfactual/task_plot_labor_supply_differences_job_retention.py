"""Plot differences in labor supply for job retention counterfactual.

Compares job retention counterfactual vs:
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
from caregiving.model.shared import INFORMAL_CARE


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_distance(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_by_distance_job_retention.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (job-retention - no-care-demand).

    Averages by distance.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care from job retention, attach to merged.
      6) Average diffs by distance and plot three series.

    """
    # Load and prepare data
    df_jr, df_ncd = prepare_dataframes_simple(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers,
    )

    # Calculate outcomes
    jr_work, jr_ft, jr_pt = calculate_simple_outcomes(df_jr, "job_retention")
    ncd_work, ncd_ft, ncd_pt = calculate_simple_outcomes(df_ncd, "no_care_demand")

    # Calculate additional outcomes (gross labor income, savings, wealth)
    jr_additional = calculate_additional_outcomes(df_jr)
    ncd_additional = calculate_additional_outcomes(df_ncd)

    # Create outcome columns
    jr_cols = df_jr[["agent", "period"]].copy()
    jr_cols["work_jr"] = jr_work
    jr_cols["ft_jr"] = jr_ft
    jr_cols["pt_jr"] = jr_pt
    jr_cols["gross_labor_income_jr"] = jr_additional["gross_labor_income"]
    jr_cols["savings_jr"] = jr_additional["savings"]
    jr_cols["wealth_jr"] = jr_additional["wealth"]
    jr_cols["savings_rate_jr"] = jr_additional["savings_rate"]

    ncd_cols = df_ncd[["agent", "period"]].copy()
    ncd_cols["work_ncd"] = ncd_work
    ncd_cols["ft_ncd"] = ncd_ft
    ncd_cols["pt_ncd"] = ncd_pt
    ncd_cols["gross_labor_income_ncd"] = ncd_additional["gross_labor_income"]
    ncd_cols["savings_ncd"] = ncd_additional["savings"]
    ncd_cols["wealth_ncd"] = ncd_additional["wealth"]
    ncd_cols["savings_rate_ncd"] = ncd_additional["savings_rate"]

    # Merge and compute differences (job retention - no care demand)
    merged = jr_cols.merge(ncd_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_jr"] - merged["work_ncd"]
    merged["diff_ft"] = merged["ft_jr"] - merged["ft_ncd"]
    merged["diff_pt"] = merged["pt_jr"] - merged["pt_ncd"]
    merged["diff_gross_labor_income"] = (
        merged["gross_labor_income_jr"] - merged["gross_labor_income_ncd"]
    )
    merged["diff_savings"] = merged["savings_jr"] - merged["savings_ncd"]
    merged["diff_wealth"] = merged["wealth_jr"] - merged["wealth_ncd"]
    merged["diff_savings_rate"] = merged["savings_rate_jr"] - merged["savings_rate_ncd"]
    merged["diff_savings_rate"] = merged["savings_rate_jr"] - merged["savings_rate_ncd"]

    # Compute distance in job retention and attach
    df_jr_dist = _add_distance_to_first_care(df_jr)
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_period"]
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

    # Plot
    plot_three_line_differences(
        prof=prof,
        x_col="distance_to_first_care",
        path_to_plot=path_to_plot,
        xlabel="Year relative to start of first care spell",
        window=window,
    )


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_age_at_first_care(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_part_time_by_age_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_full_time_by_age_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_employment_rate_by_age_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_age_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_working_hours_by_age_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences by age at first care spell.

    Compares job retention vs no care demand counterfactuals.
    Creates separate plots for part-time and full-time work, with separate lines
    for each age at which caregiving started.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care and age_at_first_care from job retention.
      6) Filter to specific ages at first care.
      7) Average diffs by distance and age_at_first_care.
      8) Plot separate figures for PT and FT with one line per starting age.

    """
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Load and prepare data
    df_jr, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    # Calculate outcomes
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    jr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_jr, model_params, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, model_params, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    jr_additional = calculate_additional_outcomes(df_jr)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    jr_outcomes.update(jr_additional)
    ncd_outcomes.update(ncd_additional)

    # Create outcome columns and merge
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = jr_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    # Compute differences (job retention - no care demand)
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

    # Compute distance and age at first care from job retention
    df_jr_dist = _add_distance_to_first_care(df_jr)

    # Get first care period for each agent
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_jr["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_jr, caregiving_mask, "age_at_first_care"
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


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_age_bins_at_first_care(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_part_time_by_age_bins_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_full_time_by_age_bins_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_employment_rate_by_age_bins_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_age_bins_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_working_hours_by_age_bins_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care spell.

    Compares job retention vs no care demand counterfactuals.
    Creates separate plots for part-time and full-time work, with separate lines
    for each age bin at which caregiving started (e.g., 50-53, 54-57, etc.).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care and age_at_first_care from job retention.
      6) Group ages into bins.
      7) Average diffs by distance and age_bin_at_first_care.
      8) Plot separate figures for PT and FT with one line per age bin.

    """
    # Load and prepare data
    df_jr, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    # Calculate outcomes
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    jr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_jr, model_params, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, model_params, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    jr_additional = calculate_additional_outcomes(df_jr)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    jr_outcomes.update(jr_additional)
    ncd_outcomes.update(ncd_additional)

    # Create outcome columns and merge
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = jr_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    # Compute differences (job retention - no care demand)
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

    # Compute distance and age at first care from job retention
    df_jr_dist = _add_distance_to_first_care(df_jr)

    # Get first care period for each agent
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_jr["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_jr, caregiving_mask, "age_at_first_care"
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    # Filter to age range
    merged = merged[
        (merged["age_at_first_care"] >= min_age)
        & (merged["age_at_first_care"] <= max_age)
    ]

    # Create age bins
    merged["age_bin_start"] = (
        (merged["age_at_first_care"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance and age_bin
    prof = (
        merged.groupby(["distance_to_first_care", "age_bin_label"], observed=False)[
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
    )

    # Add age_bin_start back for sorting
    prof["age_bin_start"] = prof["age_bin_label"].str.split("-").str[0].astype(int)
    prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])

    # Get unique age bins in order (sorted by bin start)
    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )

    # Get colors for age bins
    colors = get_distinct_colors(len(unique_bins))

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
        group_col="age_bin_label",
        groups=unique_bins,
        colors=colors,
        plot_configs=plot_configs,
        window=window,
        legend_title="Age at first care",
    )


# ============================================================================
# Job Retention vs Baseline Comparison
# ============================================================================


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_distance_vs_baseline(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_by_distance_job_retention.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (job-retention - baseline).

    Averages by distance.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care from job retention, attach to merged.
      6) Average diffs by distance and plot three series.

    """
    # Load and prepare data
    df_jr, df_baseline = prepare_dataframes_simple(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers,
    )

    # Calculate outcomes - both use same 8-choice structure
    jr_work, jr_ft, jr_pt = calculate_simple_outcomes(df_jr, "job_retention")
    baseline_work, baseline_ft, baseline_pt = calculate_simple_outcomes(
        df_baseline, "original"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    jr_additional = calculate_additional_outcomes(df_jr)
    baseline_additional = calculate_additional_outcomes(df_baseline)

    # Create outcome columns
    jr_cols = df_jr[["agent", "period"]].copy()
    jr_cols["work_jr"] = jr_work
    jr_cols["ft_jr"] = jr_ft
    jr_cols["pt_jr"] = jr_pt
    jr_cols["gross_labor_income_jr"] = jr_additional["gross_labor_income"]
    jr_cols["savings_jr"] = jr_additional["savings"]
    jr_cols["wealth_jr"] = jr_additional["wealth"]
    jr_cols["savings_rate_jr"] = jr_additional["savings_rate"]

    baseline_cols = df_baseline[["agent", "period"]].copy()
    baseline_cols["work_baseline"] = baseline_work
    baseline_cols["ft_baseline"] = baseline_ft
    baseline_cols["pt_baseline"] = baseline_pt
    baseline_cols["gross_labor_income_baseline"] = baseline_additional[
        "gross_labor_income"
    ]
    baseline_cols["savings_baseline"] = baseline_additional["savings"]
    baseline_cols["wealth_baseline"] = baseline_additional["wealth"]
    baseline_cols["savings_rate_baseline"] = baseline_additional["savings_rate"]

    # Merge and compute differences (job retention - baseline)
    merged = jr_cols.merge(baseline_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_jr"] - merged["work_baseline"]
    merged["diff_ft"] = merged["ft_jr"] - merged["ft_baseline"]
    merged["diff_pt"] = merged["pt_jr"] - merged["pt_baseline"]
    merged["diff_gross_labor_income"] = (
        merged["gross_labor_income_jr"] - merged["gross_labor_income_baseline"]
    )
    merged["diff_savings"] = merged["savings_jr"] - merged["savings_baseline"]
    merged["diff_wealth"] = merged["wealth_jr"] - merged["wealth_baseline"]
    merged["diff_savings_rate"] = (
        merged["savings_rate_jr"] - merged["savings_rate_baseline"]
    )

    # Compute distance in job retention and attach
    df_jr_dist = _add_distance_to_first_care(df_jr)
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_period"]
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

    # Plot
    plot_three_line_differences(
        prof=prof,
        x_col="distance_to_first_care",
        path_to_plot=path_to_plot,
        xlabel="Year relative to start of first care spell",
        window=window,
    )


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_age_at_first_care_vs_baseline(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_part_time_by_age_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences by age at first care spell.

    Compares job retention vs baseline counterfactuals.
    Creates separate plots for part-time and full-time work, with separate lines
    for each age at which caregiving started.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care and age_at_first_care from job retention.
      6) Filter to specific ages at first care.
      7) Average diffs by distance and age_at_first_care.
      8) Plot separate figures for PT and FT with one line per starting age.

    """
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Load and prepare data
    df_jr, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate outcomes - both use same 8-choice structure
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    jr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_jr, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    jr_additional = calculate_additional_outcomes(df_jr)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    jr_outcomes.update(jr_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns and merge
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = jr_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    # Compute differences (job retention - baseline)
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

    # Compute distance and age at first care from job retention
    df_jr_dist = _add_distance_to_first_care(df_jr)

    # Get first care period for each agent
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_jr["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_jr, caregiving_mask, "age_at_first_care"
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
        "gross_labor_income": {
            "path": path_to_plot_work.parent
            / "matched_differences_gross_labor_income_by_age_at_first_care_vs_baseline.png",
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from Baseline",
            "diff_col": "diff_gross_labor_income",
        },
        "savings": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_by_age_at_first_care_vs_baseline.png",
            "ylabel": "Savings Decision\nDeviation from Baseline",
            "diff_col": "diff_savings",
        },
        "wealth": {
            "path": path_to_plot_work.parent
            / "matched_differences_wealth_by_age_at_first_care_vs_baseline.png",
            "ylabel": "Wealth at Beginning of Period\nDeviation from Baseline",
            "diff_col": "diff_wealth",
        },
        "savings_rate": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_rate_by_age_at_first_care_vs_baseline.png",
            "ylabel": "Savings Rate\nDeviation from Baseline",
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


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_age_bins_at_first_care_vs_baseline(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_part_time_by_age_bins_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_bins_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_bins_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_bins_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_bins_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care spell.

    Compares job retention vs baseline counterfactuals.
    Creates separate plots for part-time and full-time work, with separate lines
    for each age bin at which caregiving started (e.g., 50-53, 54-57, etc.).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care and age_at_first_care from job retention.
      6) Group ages into bins.
      7) Average diffs by distance and age_bin_at_first_care.
      8) Plot separate figures for PT and FT with one line per age bin.

    """
    # Load and prepare data
    df_jr, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate outcomes - both use same 8-choice structure
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    jr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_jr, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    jr_additional = calculate_additional_outcomes(df_jr)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    jr_outcomes.update(jr_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns and merge
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = jr_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    # Compute differences (job retention - baseline)
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

    # Compute distance and age at first care from job retention
    df_jr_dist = _add_distance_to_first_care(df_jr)

    # Get first care period for each agent
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_jr["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_jr, caregiving_mask, "age_at_first_care"
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    # Filter to age range
    merged = merged[
        (merged["age_at_first_care"] >= min_age)
        & (merged["age_at_first_care"] <= max_age)
    ]

    # Create age bins
    merged["age_bin_start"] = (
        (merged["age_at_first_care"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance and age_bin
    prof = (
        merged.groupby(["distance_to_first_care", "age_bin_label"], observed=False)[
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
    )

    # Add age_bin_start back for sorting
    prof["age_bin_start"] = prof["age_bin_label"].str.split("-").str[0].astype(int)
    prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])

    # Get unique age bins in order (sorted by bin start)
    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )

    # Get colors for age bins
    colors = get_distinct_colors(len(unique_bins))

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
        "gross_labor_income": {
            "path": path_to_plot_work.parent
            / "matched_differences_gross_labor_income_by_age_bins_at_first_care_vs_baseline.png",
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from Baseline",
            "diff_col": "diff_gross_labor_income",
        },
        "savings": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_by_age_bins_at_first_care_vs_baseline.png",
            "ylabel": "Savings Decision\nDeviation from Baseline",
            "diff_col": "diff_savings",
        },
        "wealth": {
            "path": path_to_plot_work.parent
            / "matched_differences_wealth_by_age_bins_at_first_care_vs_baseline.png",
            "ylabel": "Wealth at Beginning of Period\nDeviation from Baseline",
            "diff_col": "diff_wealth",
        },
        "savings_rate": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_rate_by_age_bins_at_first_care_vs_baseline.png",
            "ylabel": "Savings Rate\nDeviation from Baseline",
            "diff_col": "diff_savings_rate",
        },
    }

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


# ============================================================================
# Job Retention vs No Care Demand - By Care Demand Distance
# ============================================================================


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_distance_by_care_demand(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_by_distance_by_care_demand_job_retention.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_care_by_distance_by_care_demand_job_retention.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (job-retention - no-care-demand).

    Averages by distance to first care demand.

    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt, care) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand from job retention, attach to merged.
      6) Average diffs by distance and plot three series for labor outcomes.
      7) Plot care probability separately.

    """
    # Load and prepare data
    df_jr, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    # Calculate outcomes
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="job_retention")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    jr_additional = calculate_additional_outcomes(df_jr)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    jr_outcomes.update(jr_additional)
    ncd_outcomes.update(ncd_additional)

    # Create outcome columns
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Merge and compute differences
    merged = jr_cols.merge(ncd_cols, on=["agent", "period"], how="inner")
    outcome_names = [
        "work",
        "ft",
        "pt",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Compute distance to first care demand in job retention and attach
    df_jr_dist = _add_distance_to_first_care_demand(df_jr)
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care_demand", observed=False)[
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
        .sort_values("distance_to_first_care_demand")
    )

    # Plot labor outcomes
    plot_three_line_differences(
        prof=prof,
        x_col="distance_to_first_care_demand",
        path_to_plot=path_to_plot,
        xlabel="Year relative to first care demand",
        window=window,
    )

    # Plot care probability
    prof_care = (
        merged.groupby("distance_to_first_care_demand", observed=False)[["diff_care"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care_demand")
    )

    plt.figure(figsize=(12, 7))
    plt.plot(
        prof_care["distance_to_first_care_demand"],
        prof_care["diff_care"],
        label="Care",
        color="green",
        linewidth=2,
    )
    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to first care demand", fontsize=16)
    plt.ylabel(
        "Probability of Providing Care\nDeviation from No Care Demand", fontsize=16
    )
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        ["Care"], loc="lower left", bbox_to_anchor=(0.05, 0.05), prop={"size": 16}
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_care, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_age_at_first_care_demand(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_part_time_by_age_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_full_time_by_age_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_employment_rate_by_age_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_age_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_working_hours_by_age_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_care_by_age_at_first_care_demand.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care_demand: list[int] | None = None,
) -> None:
    """Compute matched period differences by age at first care demand.

    Compares job retention vs no care demand counterfactuals.
    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).
    Creates separate plots for part-time and full-time work, with separate lines
    for each age at which care demand first appeared.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand and age_at_first_care_demand.
      6) Filter to specific ages at first care demand.
      7) Average diffs by distance and age_at_first_care_demand.
      8) Plot separate figures for PT and FT with one line per starting age.

    """
    if ages_at_first_care_demand is None:
        ages_at_first_care_demand = [45, 50, 55, 60]

    # Load and prepare data
    df_jr, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    # Calculate outcomes
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    jr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_jr, model_params, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, model_params, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    jr_additional = calculate_additional_outcomes(df_jr)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    jr_outcomes.update(jr_additional)
    ncd_outcomes.update(ncd_additional)

    # Create outcome columns and merge
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = jr_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    # Compute differences (job retention - no care demand)
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
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Compute distance and age at first care demand from job retention
    df_jr_dist = _add_distance_to_first_care_demand(df_jr)

    # Get first care demand period for each agent
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    care_demand_mask = df_jr["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_jr, care_demand_mask, "age_at_first_care_demand"
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    # Filter to specific ages at first care demand
    merged = merged[merged["age_at_first_care_demand"].isin(ages_at_first_care_demand)]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance and age_at_first_care_demand
    prof = (
        merged.groupby(
            ["distance_to_first_care_demand", "age_at_first_care_demand"],
            observed=False,
        )[
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
        .sort_values(["age_at_first_care_demand", "distance_to_first_care_demand"])
    )

    # Get colors for ages
    colors = get_distinct_colors(len(ages_at_first_care_demand))

    # Plot all outcomes
    plot_configs = {
        "pt": {
            "path": path_to_plot_pt,
            "ylabel": "Proportion Part-Time Working\nDeviation from No Care Demand",
            "diff_col": "diff_pt",
            "xlabel": "Year relative to first care demand",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from No Care Demand",
            "diff_col": "diff_ft",
            "xlabel": "Year relative to first care demand",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from No Care Demand",
            "diff_col": "diff_work",
            "xlabel": "Year relative to first care demand",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability Difference\nDeviation from No Care Demand",
            "diff_col": "diff_job_offer",
            "xlabel": "Year relative to first care demand",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from No Care Demand",
            "diff_col": "diff_hours_weekly",
            "xlabel": "Year relative to first care demand",
        },
        "care": {
            "path": path_to_plot_care,
            "ylabel": "Probability of Providing Care\nDeviation from No Care Demand",
            "diff_col": "diff_care",
            "xlabel": "Year relative to first care demand",
        },
        "gross_labor_income": {
            "path": path_to_plot_work.parent
            / "matched_differences_gross_labor_income_by_age_at_first_care_demand.png",
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from No Care Demand",
            "diff_col": "diff_gross_labor_income",
            "xlabel": "Year relative to first care demand",
        },
        "savings": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_by_age_at_first_care_demand.png",
            "ylabel": "Savings Decision\nDeviation from No Care Demand",
            "diff_col": "diff_savings",
            "xlabel": "Year relative to first care demand",
        },
        "wealth": {
            "path": path_to_plot_work.parent
            / "matched_differences_wealth_by_age_at_first_care_demand.png",
            "ylabel": "Wealth at Beginning of Period\nDeviation from No Care Demand",
            "diff_col": "diff_wealth",
            "xlabel": "Year relative to first care demand",
        },
        "savings_rate": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_rate_by_age_at_first_care_demand.png",
            "ylabel": "Savings Rate\nDeviation from No Care Demand",
            "diff_col": "diff_savings_rate",
            "xlabel": "Year relative to first care demand",
        },
    }

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


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_age_bins_at_first_care_demand(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_part_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_full_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_employment_rate_by_age_bins_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_job_offer_by_age_bins_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_working_hours_by_age_bins_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_no_care_demand"
    / "matched_differences_care_by_age_bins_at_first_care_demand.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care demand.

    Compares job retention vs no care demand counterfactuals.
    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).
    Creates separate plots for part-time and full-time work, with separate lines
    for each age bin at which care demand first appeared (e.g., 50-53, 54-57, etc.).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand and age_at_first_care_demand.
      6) Group ages into bins.
      7) Average diffs by distance and age_bin_at_first_care_demand.
      8) Plot separate figures for PT and FT with one line per age bin.

    """
    # Load and prepare data
    df_jr, df_ncd = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    # Calculate outcomes
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="original")
    ncd_outcomes = calculate_outcomes(df_ncd, choice_set_type="no_care_demand")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    jr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_jr, model_params, choice_set_type="original"
    )
    ncd_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_ncd, model_params, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    jr_additional = calculate_additional_outcomes(df_jr)
    ncd_additional = calculate_additional_outcomes(df_ncd)
    jr_outcomes.update(jr_additional)
    ncd_outcomes.update(ncd_additional)

    # Create outcome columns and merge
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    ncd_cols = create_outcome_columns(df_ncd, ncd_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = jr_cols.merge(ncd_cols, on=["agent", "period"], how="inner")

    # Compute differences (job retention - no care demand)
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
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Compute distance and age at first care demand from job retention
    df_jr_dist = _add_distance_to_first_care_demand(df_jr)

    # Get first care demand period for each agent
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    care_demand_mask = df_jr["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_jr, care_demand_mask, "age_at_first_care_demand"
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    # Filter to age range
    merged = merged[
        (merged["age_at_first_care_demand"] >= min_age)
        & (merged["age_at_first_care_demand"] <= max_age)
    ]

    # Create age bins
    merged["age_bin_start"] = (
        (merged["age_at_first_care_demand"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance and age_bin
    prof = (
        merged.groupby(
            ["distance_to_first_care_demand", "age_bin_label"], observed=False
        )[
            [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_care",
            ]
        ]
        .mean()
        .reset_index()
    )

    # Add age_bin_start back for sorting
    prof["age_bin_start"] = prof["age_bin_label"].str.split("-").str[0].astype(int)
    prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])

    # Get unique age bins in order (sorted by bin start)
    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )

    # Get colors for age bins
    colors = get_distinct_colors(len(unique_bins))

    # Plot all outcomes
    plot_configs = {
        "pt": {
            "path": path_to_plot_pt,
            "ylabel": "Proportion Part-Time Working\nDeviation from No Care Demand",
            "diff_col": "diff_pt",
            "xlabel": "Year relative to first care demand",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from No Care Demand",
            "diff_col": "diff_ft",
            "xlabel": "Year relative to first care demand",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from No Care Demand",
            "diff_col": "diff_work",
            "xlabel": "Year relative to first care demand",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability Difference\nDeviation from No Care Demand",
            "diff_col": "diff_job_offer",
            "xlabel": "Year relative to first care demand",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from No Care Demand",
            "diff_col": "diff_hours_weekly",
            "xlabel": "Year relative to first care demand",
        },
        "care": {
            "path": path_to_plot_care,
            "ylabel": "Probability of Providing Care\nDeviation from No Care Demand",
            "diff_col": "diff_care",
            "xlabel": "Year relative to first care demand",
        },
    }

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


# ============================================================================
# Job Retention vs Baseline - By Care Demand Distance
# ============================================================================


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_distance_by_care_demand_vs_baseline(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_by_distance_by_care_demand_job_retention.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_care_by_distance_by_care_demand_job_retention.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (job-retention - baseline).

    Averages by distance to first care demand.

    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt, care) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand from job retention, attach to merged.
      6) Average diffs by distance and plot three series for labor outcomes.
      7) Plot care probability separately.

    """
    # Load and prepare data
    df_jr, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate outcomes - both use same 8-choice structure
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="job_retention")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    jr_additional = calculate_additional_outcomes(df_jr)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    jr_outcomes.update(jr_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Merge and compute differences
    merged = jr_cols.merge(baseline_cols, on=["agent", "period"], how="inner")
    outcome_names = [
        "work",
        "ft",
        "pt",
        "care",
        "gross_labor_income",
        "savings",
        "wealth",
        "savings_rate",
    ]
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Compute distance to first care demand in job retention and attach
    df_jr_dist = _add_distance_to_first_care_demand(df_jr)
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care_demand", observed=False)[
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
        .sort_values("distance_to_first_care_demand")
    )

    # Plot labor outcomes
    plot_three_line_differences(
        prof=prof,
        x_col="distance_to_first_care_demand",
        path_to_plot=path_to_plot,
        xlabel="Year relative to first care demand",
        window=window,
    )

    # Plot care probability
    prof_care = (
        merged.groupby("distance_to_first_care_demand", observed=False)[["diff_care"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care_demand")
    )

    plt.figure(figsize=(12, 7))
    plt.plot(
        prof_care["distance_to_first_care_demand"],
        prof_care["diff_care"],
        label="Care",
        color="green",
        linewidth=2,
    )
    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to first care demand", fontsize=16)
    plt.ylabel("Probability of Providing Care\nDeviation from Baseline", fontsize=16)
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        ["Care"], loc="lower left", bbox_to_anchor=(0.05, 0.05), prop={"size": 16}
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_care, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_age_at_first_care_demand_vs_baseline(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_part_time_by_age_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_care_by_age_at_first_care_demand.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care_demand: list[int] | None = None,
) -> None:
    """Compute matched period differences by age at first care demand.

    Compares job retention vs baseline counterfactuals.
    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).
    Creates separate plots for part-time and full-time work, with separate lines
    for each age at which care demand first appeared.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand and age_at_first_care_demand.
      6) Filter to specific ages at first care demand.
      7) Average diffs by distance and age_at_first_care_demand.
      8) Plot separate figures for PT and FT with one line per starting age.

    """
    if ages_at_first_care_demand is None:
        ages_at_first_care_demand = [45, 50, 55, 60]

    # Load and prepare data
    df_jr, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate outcomes - both use same 8-choice structure
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    jr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_jr, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    jr_additional = calculate_additional_outcomes(df_jr)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    jr_outcomes.update(jr_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns and merge
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = jr_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    # Compute differences (job retention - baseline)
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
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Compute distance and age at first care demand from job retention
    df_jr_dist = _add_distance_to_first_care_demand(df_jr)

    # Get first care demand period for each agent
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    care_demand_mask = df_jr["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_jr, care_demand_mask, "age_at_first_care_demand"
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    # Filter to specific ages at first care demand
    merged = merged[merged["age_at_first_care_demand"].isin(ages_at_first_care_demand)]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance and age_at_first_care_demand
    prof = (
        merged.groupby(
            ["distance_to_first_care_demand", "age_at_first_care_demand"],
            observed=False,
        )[
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
        .sort_values(["age_at_first_care_demand", "distance_to_first_care_demand"])
    )

    # Get colors for ages
    colors = get_distinct_colors(len(ages_at_first_care_demand))

    # Plot all outcomes
    plot_configs = {
        "pt": {
            "path": path_to_plot_pt,
            "ylabel": "Proportion Part-Time Working\nDeviation from Baseline",
            "diff_col": "diff_pt",
            "xlabel": "Year relative to first care demand",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from Baseline",
            "diff_col": "diff_ft",
            "xlabel": "Year relative to first care demand",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from Baseline",
            "diff_col": "diff_work",
            "xlabel": "Year relative to first care demand",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability Difference\nDeviation from Baseline",
            "diff_col": "diff_job_offer",
            "xlabel": "Year relative to first care demand",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from Baseline",
            "diff_col": "diff_hours_weekly",
            "xlabel": "Year relative to first care demand",
        },
        "care": {
            "path": path_to_plot_care,
            "ylabel": "Probability of Providing Care\nDeviation from Baseline",
            "diff_col": "diff_care",
            "xlabel": "Year relative to first care demand",
        },
    }

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


@pytask.mark.counterfactual_differences_job_retention
def task_plot_matched_differences_by_age_bins_at_first_care_demand_vs_baseline(  # noqa: PLR0915
    path_to_job_retention_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_job_retention_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_part_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_bins_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_bins_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_bins_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "job_retention"
    / "vs_baseline"
    / "matched_differences_care_by_age_bins_at_first_care_demand.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care demand.

    Compares job retention vs baseline counterfactuals.
    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).
    Creates separate plots for part-time and full-time work, with separate lines
    for each age bin at which care demand first appeared (e.g., 50-53, 54-57, etc.).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand and age_at_first_care_demand.
      6) Group ages into bins.
      7) Average diffs by distance and age_bin_at_first_care_demand.
      8) Plot separate figures for PT and FT with one line per age bin.

    """
    # Load and prepare data
    df_jr, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_job_retention_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate outcomes - both use same 8-choice structure
    jr_outcomes = calculate_outcomes(df_jr, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    jr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_jr, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    # Calculate additional outcomes (gross labor income, savings, wealth)
    jr_additional = calculate_additional_outcomes(df_jr)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    jr_outcomes.update(jr_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns and merge
    jr_cols = create_outcome_columns(df_jr, jr_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")

    # Merge on (agent, period) to get matched differences
    merged = jr_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    # Compute differences (job retention - baseline)
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
    for outcome_name in outcome_names:
        merged[f"diff_{outcome_name}"] = (
            merged[f"{outcome_name}_o"] - merged[f"{outcome_name}_c"]
        )

    # Compute distance and age at first care demand from job retention
    df_jr_dist = _add_distance_to_first_care_demand(df_jr)

    # Get first care demand period for each agent
    dist_map = (
        df_jr_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    care_demand_mask = df_jr["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_jr, care_demand_mask, "age_at_first_care_demand"
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    # Filter to age range
    merged = merged[
        (merged["age_at_first_care_demand"] >= min_age)
        & (merged["age_at_first_care_demand"] <= max_age)
    ]

    # Create age bins
    merged["age_bin_start"] = (
        (merged["age_at_first_care_demand"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance and age_bin
    prof = (
        merged.groupby(
            ["distance_to_first_care_demand", "age_bin_label"], observed=False
        )[
            [
                "diff_work",
                "diff_ft",
                "diff_pt",
                "diff_job_offer",
                "diff_hours_weekly",
                "diff_care",
            ]
        ]
        .mean()
        .reset_index()
    )

    # Add age_bin_start back for sorting
    prof["age_bin_start"] = prof["age_bin_label"].str.split("-").str[0].astype(int)
    prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])

    # Get unique age bins in order (sorted by bin start)
    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )

    # Get colors for age bins
    colors = get_distinct_colors(len(unique_bins))

    # Plot all outcomes
    plot_configs = {
        "pt": {
            "path": path_to_plot_pt,
            "ylabel": "Proportion Part-Time Working\nDeviation from Baseline",
            "diff_col": "diff_pt",
            "xlabel": "Year relative to first care demand",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from Baseline",
            "diff_col": "diff_ft",
            "xlabel": "Year relative to first care demand",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from Baseline",
            "diff_col": "diff_work",
            "xlabel": "Year relative to first care demand",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability Difference\nDeviation from Baseline",
            "diff_col": "diff_job_offer",
            "xlabel": "Year relative to first care demand",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from Baseline",
            "diff_col": "diff_hours_weekly",
            "xlabel": "Year relative to first care demand",
        },
        "care": {
            "path": path_to_plot_care,
            "ylabel": "Probability of Providing Care\nDeviation from Baseline",
            "diff_col": "diff_care",
            "xlabel": "Year relative to first care demand",
        },
    }

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
