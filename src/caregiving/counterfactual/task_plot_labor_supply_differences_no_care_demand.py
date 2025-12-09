"""Plot differences in labor supply by distance for no care demand counterfactual.

Original and no-care-demand scenario.
"""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
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
from caregiving.model.shared import DEAD, INFORMAL_CARE

# ============================================================================
# Distance calculation helpers
# ============================================================================


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_matched_differences_by_distance(  # noqa: PLR0915, E501
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_by_distance_no_care_demand.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (orig - no-care-demand).

    Averages by distance.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care from original, attach to merged.
      6) Average diffs by distance and plot three series.

    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_simple(
        pd.read_pickle(path_to_original_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers,
        ever_care_demand,
    )

    # Calculate outcomes
    o_work, o_ft, o_pt = calculate_simple_outcomes(df_o, "original")
    c_work, c_ft, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    o_additional = calculate_additional_outcomes(df_o)
    c_additional = calculate_additional_outcomes(df_c)

    # Create outcome columns
    o_cols = df_o[["agent", "period"]].copy()
    o_cols["work_o"] = o_work
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt
    o_cols["gross_labor_income_o"] = o_additional["gross_labor_income"]
    o_cols["savings_o"] = o_additional["savings"]
    o_cols["wealth_o"] = o_additional["wealth"]
    o_cols["savings_rate_o"] = o_additional["savings_rate"]

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["work_c"] = c_work
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt
    c_cols["gross_labor_income_c"] = c_additional["gross_labor_income"]
    c_cols["savings_c"] = c_additional["savings"]
    c_cols["wealth_c"] = c_additional["wealth"]
    c_cols["savings_rate_c"] = c_additional["savings_rate"]

    # Merge and compute differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_o"] - merged["work_c"]
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]
    merged["diff_gross_labor_income"] = (
        merged["gross_labor_income_o"] - merged["gross_labor_income_c"]
    )
    merged["diff_savings"] = merged["savings_o"] - merged["savings_c"]
    merged["diff_wealth"] = merged["wealth_o"] - merged["wealth_c"]
    merged["diff_savings_rate"] = merged["savings_rate_o"] - merged["savings_rate_c"]

    # Compute distance in original and attach
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_matched_differences_by_age_at_first_care(  # noqa: PLR0915, E501
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_part_time_by_age_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_full_time_by_age_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_employment_rate_by_age_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_job_offer_by_age_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_working_hours_by_age_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] | None = None,
) -> None:
    """Compute matched period differences by age at first care spell.

    Creates separate plots for part-time and full-time work, with separate lines
    for each age at which caregiving started.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care and age_at_first_care from original.
      6) Filter to specific ages at first care.
      7) Average diffs by distance and age_at_first_care.
      8) Plot separate figures for PT and FT with one line per starting age.

    """
    if ages_at_first_care is None:
        ages_at_first_care = [45, 50, 54, 58, 62]

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

    # Calculate additional outcomes (gross labor income, savings, wealth)
    o_additional = calculate_additional_outcomes(df_o)
    c_additional = calculate_additional_outcomes(df_c)
    o_outcomes.update(o_additional)
    c_outcomes.update(c_additional)

    # Create outcome columns and merge
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Merge and compute differences
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
    merged = merge_and_compute_differences(o_cols, c_cols, outcome_names)

    # Compute distance and age at first care from original
    df_o_dist = _add_distance_to_first_care(df_o)

    # Get first care period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_o["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_o, caregiving_mask, "age_at_first_care"
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
            "ylabel": "Proportion Part-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_pt",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_ft",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from Counterfactual",
            "diff_col": "diff_work",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability Difference\nDeviation from Counterfactual",
            "diff_col": "diff_job_offer",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from Counterfactual",
            "diff_col": "diff_hours_weekly",
        },
        "gross_labor_income": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_gross_labor_income_by_age_at_first_care.png"
            ),
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from Counterfactual",
            "diff_col": "diff_gross_labor_income",
        },
        "savings": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_savings_by_age_at_first_care.png"
            ),
            "ylabel": "Savings Decision\nDeviation from Counterfactual",
            "diff_col": "diff_savings",
        },
        "wealth": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_wealth_by_age_at_first_care.png"
            ),
            "ylabel": "Wealth at Beginning of Period\nDeviation from Counterfactual",
            "diff_col": "diff_wealth",
        },
        "savings_rate": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_savings_rate_by_age_at_first_care.png"
            ),
            "ylabel": "Savings Rate\nDeviation from Counterfactual",
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_matched_differences_by_age_bins_at_first_care(  # noqa: PLR0915, E501
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_part_time_by_age_bins_at_first_care.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_full_time_by_age_bins_at_first_care.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_employment_rate_by_age_bins_at_first_care.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_job_offer_by_age_bins_at_first_care.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_working_hours_by_age_bins_at_first_care.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care spell.

    Creates separate plots for part-time and full-time work, with separate lines
    for each age bin at which caregiving started (e.g., 50-53, 54-57, etc.).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care and age_at_first_care from original.
      6) Group ages into bins.
      7) Average diffs by distance and age_bin_at_first_care.
      8) Plot separate figures for PT and FT with one line per age bin.

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

    # Calculate additional outcomes (gross labor income, savings, wealth)
    o_additional = calculate_additional_outcomes(df_o)
    c_additional = calculate_additional_outcomes(df_c)
    o_outcomes.update(o_additional)
    c_outcomes.update(c_additional)

    # Create outcome columns and merge
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Merge and compute differences
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
    merged = merge_and_compute_differences(o_cols, c_cols, outcome_names)

    # Compute distance and age at first care from original
    df_o_dist = _add_distance_to_first_care(df_o)

    # Get first care period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_o["choice"].isin(care_codes)
    first_care_with_age = get_age_at_first_event(
        df_o, caregiving_mask, "age_at_first_care"
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
            "ylabel": "Proportion Part-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_pt",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_ft",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from Counterfactual",
            "diff_col": "diff_work",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability Difference\nDeviation from Counterfactual",
            "diff_col": "diff_job_offer",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from Counterfactual",
            "diff_col": "diff_hours_weekly",
        },
        "gross_labor_income": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_gross_labor_income_by_age_bins_at_first_care.png"
            ),
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from Counterfactual",
            "diff_col": "diff_gross_labor_income",
        },
        "savings": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_savings_by_age_bins_at_first_care.png"
            ),
            "ylabel": "Savings Decision\nDeviation from Counterfactual",
            "diff_col": "diff_savings",
        },
        "wealth": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_wealth_by_age_bins_at_first_care.png"
            ),
            "ylabel": "Wealth at Beginning of Period\nDeviation from Counterfactual",
            "diff_col": "diff_wealth",
        },
        "savings_rate": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_savings_rate_by_age_bins_at_first_care.png"
            ),
            "ylabel": "Savings Rate\nDeviation from Counterfactual",
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_first_care_start_by_age(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "first_care_start_by_age.png",
    path_to_csv: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "first_care_start_by_age.csv",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Count how many people start giving care for the first time at each age.

    Creates a plot showing the distribution of first care start ages and saves
    the counts to a CSV file.

    Args:
        path_to_original_data: Path to original simulated data
        path_to_plot: Path to save the plot
        path_to_csv: Path to save the CSV with counts by age
        min_age: Minimum age to include in analysis
        max_age: Maximum age to include in analysis
    """
    # Load data
    df_o = pd.read_pickle(path_to_original_data)

    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = _ensure_agent_period(df_o)

    # Fully flatten any residual index levels
    if isinstance(df_o.index, pd.MultiIndex):
        idx_names_o = {n for n in df_o.index.names if n is not None}
        if ("agent" in idx_names_o) or ("period" in idx_names_o):
            df_o = df_o.reset_index()

    # Ensure no index name collisions remain
    df_o = df_o.reset_index(drop=True)

    # Check for age column
    if "age" not in df_o.columns:
        raise ValueError("Age column required but not found in data")

    # Find first care period for each agent
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_o["choice"].isin(care_codes)

    first_care = (
        df_o.loc[caregiving_mask, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(columns={"period": "first_care_period", "age": "age_at_first_care"})
    )

    # Filter to age range
    first_care = first_care[
        (first_care["age_at_first_care"] >= min_age)
        & (first_care["age_at_first_care"] <= max_age)
    ]

    # Count by age
    counts_by_age = (
        first_care["age_at_first_care"]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"age_at_first_care": "age"})
    )

    # Ensure all ages in range are represented (fill missing with 0)
    all_ages = pd.DataFrame({"age": range(min_age, max_age + 1)})
    counts_by_age = all_ages.merge(counts_by_age, on="age", how="left").fillna(0)
    counts_by_age["count"] = counts_by_age["count"].astype(int)

    # Save to CSV
    counts_by_age.to_csv(path_to_csv, index=False)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.bar(
        counts_by_age["age"],
        counts_by_age["count"],
        color=JET_COLOR_MAP[1],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Age at first care spell", fontsize=16)
    plt.ylabel("Number of people", fontsize=16)
    plt.title("Distribution of First Care Start by Age", fontsize=18)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


def _add_distance_to_first_care_demand(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care_demand column.

    Sets 0 as first time care_demand > 0.
    """
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = _ensure_agent_period(df)
    # Find first period where care_demand > 0
    care_demand_mask = df["care_demand"] > 0
    first_care_demand = (
        df.loc[care_demand_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_care_demand_period"})
    )
    out = df.merge(first_care_demand, on="agent", how="left")
    out["distance_to_first_care_demand"] = (
        out["period"] - out["first_care_demand_period"]
    )
    return out


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_matched_differences_by_distance_by_care_demand(  # noqa: PLR0915, E501
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_by_distance_by_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_care_by_distance_by_care_demand.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (orig - no-care-demand).

    Averages by distance.

    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt, care) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand from original, attach to merged.
      6) Average diffs by distance and plot three series for labor outcomes.
      7) Plot care probability separately.

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

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    o_additional = calculate_additional_outcomes(df_o)
    c_additional = calculate_additional_outcomes(df_c)
    o_outcomes.update(o_additional)
    c_outcomes.update(c_additional)

    # Create outcome columns
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Merge and compute differences
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
    merged = merge_and_compute_differences(o_cols, c_cols, outcome_names)

    # Compute distance to first care demand in original and attach
    df_o_dist = _add_distance_to_first_care_demand(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
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
        "Probability of Providing Care\nDeviation from Counterfactual", fontsize=16
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_matched_differences_by_age_at_first_care_demand(  # noqa: PLR0915, E501
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_part_time_by_age_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_full_time_by_age_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_employment_rate_by_age_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_job_offer_by_age_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_working_hours_by_age_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_care_by_age_at_first_care_demand.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages_at_first_care_demand: list[int] | None = None,
) -> None:
    """Compute matched period differences by age at first care demand.

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
        ages_at_first_care_demand = [45, 50, 55, 60, 65]

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

    # Calculate additional outcomes (gross labor income, savings, wealth)
    o_additional = calculate_additional_outcomes(df_o)
    c_additional = calculate_additional_outcomes(df_c)
    o_outcomes.update(o_additional)
    c_outcomes.update(c_additional)

    # Create outcome columns and merge
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

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

    # Compute distance and age at first care demand from original
    df_o_dist = _add_distance_to_first_care_demand(df_o)

    # Get first care demand period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    care_demand_mask = df_o["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_o, care_demand_mask, "age_at_first_care_demand"
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
            "ylabel": "Proportion Part-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_pt",
            "xlabel": "Year relative to first care demand",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_ft",
            "xlabel": "Year relative to first care demand",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from Counterfactual",
            "diff_col": "diff_work",
            "xlabel": "Year relative to first care demand",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability Difference\nDeviation from Counterfactual",
            "diff_col": "diff_job_offer",
            "xlabel": "Year relative to first care demand",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from Counterfactual",
            "diff_col": "diff_hours_weekly",
            "xlabel": "Year relative to first care demand",
        },
        "care": {
            "path": path_to_plot_care,
            "ylabel": "Probability of Providing Care\nDeviation from Counterfactual",
            "diff_col": "diff_care",
            "xlabel": "Year relative to first care demand",
        },
        "gross_labor_income": {
            "path": (
                path_to_plot_work.parent
                / (
                    "matched_differences_gross_labor_income_by_age_at_first_care"
                    "_demand.png"
                )
            ),
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from Counterfactual",
            "diff_col": "diff_gross_labor_income",
            "xlabel": "Year relative to first care demand",
        },
        "savings": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_savings_by_age_at_first_care_demand.png"
            ),
            "ylabel": "Savings Decision\nDeviation from Counterfactual",
            "diff_col": "diff_savings",
            "xlabel": "Year relative to first care demand",
        },
        "wealth": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_wealth_by_age_at_first_care_demand.png"
            ),
            "ylabel": "Wealth at Beginning of Period\nDeviation from Counterfactual",
            "diff_col": "diff_wealth",
            "xlabel": "Year relative to first care demand",
        },
        "savings_rate": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_savings_rate_by_age_at_first_care_demand.png"
            ),
            "ylabel": "Savings Rate\nDeviation from Counterfactual",
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_matched_differences_by_age_bins_at_first_care_demand(  # noqa: PLR0915, E501
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_part_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_full_time_by_age_bins_at_first_care_demand.png",
    path_to_plot_work: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_employment_rate_by_age_bins_at_first_care_demand.png",
    path_to_plot_job_offer: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_job_offer_by_age_bins_at_first_care_demand.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_working_hours_by_age_bins_at_first_care_demand.png",
    path_to_plot_care: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_care_by_age_bins_at_first_care_demand.png",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care demand.

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

    # Calculate additional outcomes (gross labor income, savings, wealth)
    o_additional = calculate_additional_outcomes(df_o)
    c_additional = calculate_additional_outcomes(df_c)
    o_outcomes.update(o_additional)
    c_outcomes.update(c_additional)

    # Create outcome columns and merge
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

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

    # Compute distance and age at first care demand from original
    df_o_dist = _add_distance_to_first_care_demand(df_o)

    # Get first care demand period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    care_demand_mask = df_o["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_o, care_demand_mask, "age_at_first_care_demand"
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
            "ylabel": "Proportion Part-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_pt",
            "xlabel": "Year relative to first care demand",
        },
        "ft": {
            "path": path_to_plot_ft,
            "ylabel": "Proportion Full-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_ft",
            "xlabel": "Year relative to first care demand",
        },
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from Counterfactual",
            "diff_col": "diff_work",
            "xlabel": "Year relative to first care demand",
        },
        "job_offer": {
            "path": path_to_plot_job_offer,
            "ylabel": "Job Offer Probability Difference\nDeviation from Counterfactual",
            "diff_col": "diff_job_offer",
            "xlabel": "Year relative to first care demand",
        },
        "hours_weekly": {
            "path": path_to_plot_working_hours,
            "ylabel": "Weekly Working Hours Difference\nDeviation from Counterfactual",
            "diff_col": "diff_hours_weekly",
            "xlabel": "Year relative to first care demand",
        },
        "care": {
            "path": path_to_plot_care,
            "ylabel": "Probability of Providing Care\nDeviation from Counterfactual",
            "diff_col": "diff_care",
            "xlabel": "Year relative to first care demand",
        },
        "gross_labor_income": {
            "path": (
                path_to_plot_work.parent
                / (
                    "matched_differences_gross_labor_income_by_age_bins_at_first_care"
                    "_demand.png"
                )
            ),
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from Counterfactual",
            "diff_col": "diff_gross_labor_income",
            "xlabel": "Year relative to first care demand",
        },
        "savings": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_savings_by_age_bins_at_first_care_demand.png"
            ),
            "ylabel": "Savings Decision\nDeviation from Counterfactual",
            "diff_col": "diff_savings",
            "xlabel": "Year relative to first care demand",
        },
        "wealth": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_wealth_by_age_bins_at_first_care_demand.png"
            ),
            "ylabel": "Wealth at Beginning of Period\nDeviation from Counterfactual",
            "diff_col": "diff_wealth",
            "xlabel": "Year relative to first care demand",
        },
        "savings_rate": {
            "path": (
                path_to_plot_work.parent
                / (
                    "matched_differences_savings_rate_by_age_bins_at_first_care"
                    "_demand.png"
                )
            ),
            "ylabel": "Savings Rate\nDeviation from Counterfactual",
            "diff_col": "diff_savings_rate",
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


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_first_care_demand_start_by_age(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "first_care_demand_start_by_age.png",
    path_to_csv: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "first_care_demand_start_by_age.csv",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Count how many people first experience care_demand > 0 at each age.

    Creates a plot showing the distribution of first care demand start ages and saves
    the counts to a CSV file.

    Args:
        path_to_original_data: Path to original simulated data
        path_to_plot: Path to save the plot
        path_to_csv: Path to save the CSV with counts by age
        min_age: Minimum age to include in analysis
        max_age: Maximum age to include in analysis
    """
    # Load data
    df_o = pd.read_pickle(path_to_original_data)

    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = _ensure_agent_period(df_o)

    # Fully flatten any residual index levels
    if isinstance(df_o.index, pd.MultiIndex):
        idx_names_o = {n for n in df_o.index.names if n is not None}
        if ("agent" in idx_names_o) or ("period" in idx_names_o):
            df_o = df_o.reset_index()

    # Ensure no index name collisions remain
    df_o = df_o.reset_index(drop=True)

    # Check for age column
    if "age" not in df_o.columns:
        raise ValueError("Age column required but not found in data")

    # Find first period where care_demand > 0 for each agent
    care_demand_mask = df_o["care_demand"] > 0

    first_care_demand = (
        df_o.loc[care_demand_mask, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(
            columns={
                "period": "first_care_demand_period",
                "age": "age_at_first_care_demand",
            }
        )
    )

    # Filter to age range
    first_care_demand = first_care_demand[
        (first_care_demand["age_at_first_care_demand"] >= min_age)
        & (first_care_demand["age_at_first_care_demand"] <= max_age)
    ]

    # Count by age
    counts_by_age = (
        first_care_demand["age_at_first_care_demand"]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"age_at_first_care_demand": "age"})
    )

    # Ensure all ages in range are represented (fill missing with 0)
    all_ages = pd.DataFrame({"age": range(min_age, max_age + 1)})
    counts_by_age = all_ages.merge(counts_by_age, on="age", how="left").fillna(0)
    counts_by_age["count"] = counts_by_age["count"].astype(int)

    # Save to CSV
    counts_by_age.to_csv(path_to_csv, index=False)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.bar(
        counts_by_age["age"],
        counts_by_age["count"],
        color=JET_COLOR_MAP[1],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Age at first care demand", fontsize=16)
    plt.ylabel("Number of people", fontsize=16)
    plt.title("Distribution of First Care Demand Start by Age", fontsize=18)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_matched_differences_forced_care_demand_at_50(  # noqa: PLR0915, E501
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_forced_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_50.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_forced_care_demand_at_50.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    forced_age: int = 50,
) -> None:
    """Compute matched period differences (baseline - forced care demand at 50).

    Uses t=0 as age 50 (where everyone experiences care demand in counterfactual).
    Since everyone experiences care demand at age 50 in the counterfactual,
    age 50 is at the center of the x-axis with a vertical line.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance from age 50 (age - forced_age) for x-axis.
      6) Average diffs by distance and plot three series.

    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_simple(
        pd.read_pickle(path_to_original_data),
        pd.read_pickle(path_to_forced_care_demand_data),
        ever_caregivers,
        ever_care_demand,
    )

    # Outcomes per period - both use same 8-choice structure
    o_work, o_ft, o_pt = calculate_simple_outcomes(df_o, "original")
    c_work, c_ft, c_pt = calculate_simple_outcomes(df_c, "original")

    # Calculate additional outcomes (gross labor income, savings, wealth, savings_rate)
    o_additional = calculate_additional_outcomes(df_o)
    c_additional = calculate_additional_outcomes(df_c)

    # Create outcome columns (include age from original)
    o_cols = df_o[["agent", "period", "age"]].copy()
    o_cols["work_o"] = o_work
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt
    o_cols["gross_labor_income_o"] = o_additional["gross_labor_income"]
    o_cols["savings_o"] = o_additional["savings"]
    o_cols["wealth_o"] = o_additional["wealth"]
    o_cols["savings_rate_o"] = o_additional["savings_rate"]

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["work_c"] = c_work
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt
    c_cols["gross_labor_income_c"] = c_additional["gross_labor_income"]
    c_cols["savings_c"] = c_additional["savings"]
    c_cols["wealth_c"] = c_additional["wealth"]
    c_cols["savings_rate_c"] = c_additional["savings_rate"]

    # Merge and compute differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_o"] - merged["work_c"]
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]
    merged["diff_gross_labor_income"] = (
        merged["gross_labor_income_o"] - merged["gross_labor_income_c"]
    )
    merged["diff_savings"] = merged["savings_o"] - merged["savings_c"]
    merged["diff_wealth"] = merged["wealth_o"] - merged["wealth_c"]
    merged["diff_savings_rate"] = merged["savings_rate_o"] - merged["savings_rate_c"]

    # Compute distance from forced_age (age 50)
    merged["distance_from_forced_age"] = merged["age"] - forced_age

    # Trim to window around forced_age
    merged = merged[
        (merged["distance_from_forced_age"] >= -window)
        & (merged["distance_from_forced_age"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_from_forced_age", observed=False)[
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
        .sort_values("distance_from_forced_age")
    )

    # Plot
    plot_three_line_differences(
        prof=prof,
        x_col="distance_from_forced_age",
        path_to_plot=path_to_plot,
        xlabel=f"Year relative to age {forced_age}",
        window=window,
    )


@pytask.mark.counterfactual_differences_no_care_demand
@pytask.mark.skip()  # Temporarily skip to avoid conflict with age_functions module
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
    / "matched_differences_work_by_age.png",
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
        .sort_values("age")
    )

    # Plot all outcomes
    plot_configs = {
        "work": {
            "path": path_to_plot_work,
            "ylabel": "Proportion Working\nDeviation from Counterfactual",
            "diff_col": "diff_work",
        },
        "ft": {
            "path": path_to_plot_work.parent / "matched_differences_ft_by_age.png",
            "ylabel": "Proportion Full-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_ft",
        },
        "pt": {
            "path": path_to_plot_work.parent / "matched_differences_pt_by_age.png",
            "ylabel": "Proportion Part-Time Working\nDeviation from Counterfactual",
            "diff_col": "diff_pt",
        },
        "job_offer": {
            "path": (
                path_to_plot_work.parent / "matched_differences_job_offer_by_age.png"
            ),
            "ylabel": "Job Offer Probability\nDeviation from Counterfactual",
            "diff_col": "diff_job_offer",
        },
        "hours_weekly": {
            "path": (
                path_to_plot_work.parent / "matched_differences_hours_weekly_by_age.png"
            ),
            "ylabel": "Weekly Working Hours\nDeviation from Counterfactual",
            "diff_col": "diff_hours_weekly",
        },
        "care": {
            "path": path_to_plot_work.parent / "matched_differences_care_by_age.png",
            "ylabel": "Probability of Providing Care\nDeviation from Counterfactual",
            "diff_col": "diff_care",
        },
        "gross_labor_income": {
            "path": (
                path_to_plot_work.parent
                / "matched_differences_gross_labor_income_by_age.png"
            ),
            "ylabel": "Gross Labor Income (Monthly)\nDeviation from Counterfactual",
            "diff_col": "diff_gross_labor_income",
        },
        "savings": {
            "path": path_to_plot_work.parent / "matched_differences_savings_by_age.png",
            "ylabel": "Savings Decision\nDeviation from Counterfactual",
            "diff_col": "diff_savings",
        },
        "wealth": {
            "path": path_to_plot_work.parent / "matched_differences_wealth_by_age.png",
            "ylabel": "Wealth at Beginning of Period\nDeviation from Counterfactual",
            "diff_col": "diff_wealth",
        },
        "savings_rate": {
            "path": (
                path_to_plot_work.parent / "matched_differences_savings_rate_by_age.png"
            ),
            "ylabel": "Savings Rate\nDeviation from Counterfactual",
            "diff_col": "diff_savings_rate",
        },
    }

    plot_all_outcomes_by_age(
        prof=prof,
        plot_configs=plot_configs,
        age_min=age_min,
        age_max=age_max,
    )
