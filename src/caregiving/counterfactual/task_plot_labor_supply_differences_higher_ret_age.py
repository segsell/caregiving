"""Plot differences in labor supply for higher-retirement-age counterfactual.

Compares higher-retirement-age counterfactual vs baseline scenario.
"""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    calculate_simple_outcomes,
    get_age_at_first_event,
    get_distinct_colors,
    plot_all_outcomes_by_group,
    plot_three_line_differences,
    prepare_dataframes_simple,
)
from caregiving.counterfactual.plotting_utils import (
    calculate_additional_outcomes,
    calculate_outcomes,
    calculate_working_hours_weekly,
    create_outcome_columns,
    prepare_dataframes_for_comparison,
)
from caregiving.model.shared import INFORMAL_CARE


@pytask.mark.counterfactual_differences_higher_ret_age
def task_plot_matched_differences_by_distance_higher_ret_age(  # noqa: PLR0915
    path_to_higher_ret_age_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_higher_ret_age_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_by_distance_higher_ret_age.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (higher-ret-age - baseline).

    Averages by distance to first caregiving spell, similar to the job-retention
    counterfactual plots.
    """

    # Load and prepare data (restrict to alive / ever-caregivers etc.)
    df_hr, df_baseline = prepare_dataframes_simple(
        pd.read_pickle(path_to_higher_ret_age_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers,
    )

    # Calculate basic outcomes (work, part-time, full-time)
    # Both higher-ret-age and baseline use the original 8-choice structure.
    hr_work, hr_ft, hr_pt = calculate_simple_outcomes(df_hr, "original")
    base_work, base_ft, base_pt = calculate_simple_outcomes(df_baseline, "original")

    # Additional outcomes (gross labor income, savings, wealth, savings rate)
    hr_additional = calculate_additional_outcomes(df_hr)
    base_additional = calculate_additional_outcomes(df_baseline)

    # Create outcome columns
    hr_cols = df_hr[["agent", "period"]].copy()
    hr_cols["work_hr"] = hr_work
    hr_cols["ft_hr"] = hr_ft
    hr_cols["pt_hr"] = hr_pt
    hr_cols["gross_labor_income_hr"] = hr_additional["gross_labor_income"]
    hr_cols["savings_hr"] = hr_additional["savings"]
    hr_cols["wealth_hr"] = hr_additional["wealth"]
    hr_cols["savings_rate_hr"] = hr_additional["savings_rate"]

    base_cols = df_baseline[["agent", "period"]].copy()
    base_cols["work_baseline"] = base_work
    base_cols["ft_baseline"] = base_ft
    base_cols["pt_baseline"] = base_pt
    base_cols["gross_labor_income_baseline"] = base_additional["gross_labor_income"]
    base_cols["savings_baseline"] = base_additional["savings"]
    base_cols["wealth_baseline"] = base_additional["wealth"]
    base_cols["savings_rate_baseline"] = base_additional["savings_rate"]

    # Merge and compute differences (higher-ret-age - baseline)
    merged = hr_cols.merge(base_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_hr"] - merged["work_baseline"]
    merged["diff_ft"] = merged["ft_hr"] - merged["ft_baseline"]
    merged["diff_pt"] = merged["pt_hr"] - merged["pt_baseline"]
    merged["diff_gross_labor_income"] = (
        merged["gross_labor_income_hr"] - merged["gross_labor_income_baseline"]
    )
    merged["diff_savings"] = merged["savings_hr"] - merged["savings_baseline"]
    merged["diff_wealth"] = merged["wealth_hr"] - merged["wealth_baseline"]
    merged["diff_savings_rate"] = (
        merged["savings_rate_hr"] - merged["savings_rate_baseline"]
    )

    # Distance to first care: use helper from baseline counterfactual module
    from caregiving.counterfactual.task_plot_labor_supply_differences import (  # noqa: E402
        _add_distance_to_first_care,
    )

    df_hr_dist = _add_distance_to_first_care(df_hr)
    dist_map = (
        df_hr_dist.groupby("agent", observed=False)["first_care_period"]
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

    # Plot three main labor outcomes; additional outcomes still available in `prof`
    plot_three_line_differences(
        prof=prof,
        x_col="distance_to_first_care",
        path_to_plot=path_to_plot,
        xlabel="Year relative to start of first care spell",
        window=window,
    )


@pytask.mark.counterfactual_differences_higher_ret_age
def task_plot_matched_differences_by_age_at_first_care_higher_ret_age_vs_baseline(  # noqa: PLR0915
    path_to_higher_ret_age_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_higher_ret_age_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_part_time_by_age_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Matched differences by age at first care: higher-ret-age vs baseline."""

    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

    # Load and prepare data
    df_hr, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_higher_ret_age_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Outcomes: both use original 8-choice structure
    hr_outcomes = calculate_outcomes(df_hr, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    # Working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    hr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_hr, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    # Additional outcomes
    hr_additional = calculate_additional_outcomes(df_hr)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    hr_outcomes.update(hr_additional)
    baseline_outcomes.update(baseline_additional)

    # Create outcome columns and merge
    hr_cols = create_outcome_columns(df_hr, hr_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")
    merged = hr_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

    # Differences (higher-ret-age - baseline)
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

    # Distance and age at first care from higher-ret-age simulation
    from caregiving.counterfactual.task_plot_labor_supply_differences import (  # noqa: E402
        _add_distance_to_first_care,
    )

    df_hr_dist = _add_distance_to_first_care(df_hr)
    dist_map = (
        df_hr_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_hr["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_hr, caregiving_mask, "age_at_first_care"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    # Filter ages and window
    merged = merged[merged["age_at_first_care"].isin(ages_at_first_care)]
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Group and average
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


@pytask.mark.counterfactual_differences_higher_ret_age
def task_plot_matched_differences_by_age_bins_at_first_care_higher_ret_age_vs_baseline(  # noqa: PLR0915
    path_to_higher_ret_age_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_higher_ret_age_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_part_time_by_age_bins_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_bins_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_bins_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_bins_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_bins_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Matched differences by age-bin at first care: higher-ret-age vs baseline."""

    # Load and prepare data
    df_hr, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_higher_ret_age_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
    )

    # Outcomes (original 8-choice)
    hr_outcomes = calculate_outcomes(df_hr, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    hr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_hr, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    hr_additional = calculate_additional_outcomes(df_hr)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    hr_outcomes.update(hr_additional)
    baseline_outcomes.update(baseline_additional)

    hr_cols = create_outcome_columns(df_hr, hr_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")
    merged = hr_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

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

    from caregiving.counterfactual.task_plot_labor_supply_differences import (  # noqa: E402
        _add_distance_to_first_care,
    )

    df_hr_dist = _add_distance_to_first_care(df_hr)
    dist_map = (
        df_hr_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_hr["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_hr, caregiving_mask, "age_at_first_care"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    # Filter age range and create bins
    merged = merged[
        (merged["age_at_first_care"] >= min_age)
        & (merged["age_at_first_care"] <= max_age)
    ]
    merged["age_bin_start"] = (
        (merged["age_at_first_care"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

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

    prof["age_bin_start"] = prof["age_bin_label"].str.split("-").str[0].astype(int)
    prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])

    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )
    colors = get_distinct_colors(len(unique_bins))

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
        legend_title="Age bin at first care",
    )


@pytask.mark.counterfactual_differences_higher_ret_age
def task_plot_matched_differences_by_age_at_first_care_demand_higher_ret_age_vs_baseline(  # noqa: PLR0915
    path_to_higher_ret_age_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_higher_ret_age_estimated_params.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_part_time_by_age_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_full_time_by_age_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_employment_rate_by_age_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_job_offer_by_age_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_working_hours_by_age_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "higher_ret_age"
    / "vs_baseline"
    / "matched_differences_care_by_age_at_first_care_demand.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care_demand: list[int] | None = None,
) -> None:
    """By age at first care demand: higher-ret-age vs baseline."""

    if ages_at_first_care_demand is None:
        ages_at_first_care_demand = [45, 50, 55, 60]

    df_hr, df_baseline = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_higher_ret_age_data),
        pd.read_pickle(path_to_baseline_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=False,
    )

    hr_outcomes = calculate_outcomes(df_hr, choice_set_type="original")
    baseline_outcomes = calculate_outcomes(df_baseline, choice_set_type="original")

    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    hr_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_hr, model_params, choice_set_type="original"
    )
    baseline_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_baseline, model_params, choice_set_type="original"
    )

    hr_additional = calculate_additional_outcomes(df_hr)
    baseline_additional = calculate_additional_outcomes(df_baseline)
    hr_outcomes.update(hr_additional)
    baseline_outcomes.update(baseline_additional)

    hr_cols = create_outcome_columns(df_hr, hr_outcomes, "_o")
    baseline_cols = create_outcome_columns(df_baseline, baseline_outcomes, "_c")
    merged = hr_cols.merge(baseline_cols, on=["agent", "period"], how="inner")

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

    from caregiving.counterfactual.task_plot_labor_supply_differences_no_care_demand import (  # noqa: E402
        _add_distance_to_first_care_demand,
    )

    df_hr_dist = _add_distance_to_first_care_demand(df_hr)
    dist_map = (
        df_hr_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    care_demand_mask = df_hr["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_hr, care_demand_mask, "age_at_first_care_demand"
    )

    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    merged = merged[merged["age_at_first_care_demand"].isin(ages_at_first_care_demand)]
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

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

    colors = get_distinct_colors(len(ages_at_first_care_demand))

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
        "gross_labor_income": {
            "path": path_to_plot_work.parent
            / "matched_differences_gross_labor_income_by_age_at_first_care_demand_vs_baseline.png",
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from Baseline",
            "diff_col": "diff_gross_labor_income",
        },
        "savings": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_by_age_at_first_care_demand_vs_baseline.png",
            "ylabel": "Savings Decision\nDeviation from Baseline",
            "diff_col": "diff_savings",
        },
        "wealth": {
            "path": path_to_plot_work.parent
            / "matched_differences_wealth_by_age_at_first_care_demand_vs_baseline.png",
            "ylabel": "Wealth at Beginning of Period\nDeviation from Baseline",
            "diff_col": "diff_wealth",
        },
        "savings_rate": {
            "path": path_to_plot_work.parent
            / "matched_differences_savings_rate_by_age_at_first_care_demand_vs_baseline.png",
            "ylabel": "Savings Rate\nDeviation from Baseline",
            "diff_col": "diff_savings_rate",
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
