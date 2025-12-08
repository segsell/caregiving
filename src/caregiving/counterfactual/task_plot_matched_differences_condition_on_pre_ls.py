"""Plot matched differences event studies conditioned on pre-labor supply.

Creates event study plots for baseline vs no care demand, conditioned on
labor supply state at t=-1 (unemployed, part-time, full-time).
Plots outcomes by age at first care spell or first care demand.
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
from caregiving.counterfactual.plotting_helpers import get_age_at_first_event
from caregiving.counterfactual.plotting_utils import (
    _ensure_agent_period,
    calculate_additional_outcomes,
    calculate_outcomes,
    calculate_working_hours_weekly,
    create_outcome_columns,
    merge_and_compute_differences,
    prepare_dataframes_for_comparison,
)
from caregiving.model.shared import (
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    PART_TIME,
    RETIREMENT,
    UNEMPLOYED,
)


@pytask.mark.counterfactual_differences_no_care_demand
@pytask.mark.counterfactual_differences
@pytask.mark.counterfactual_differences_pre_ls
def task_plot_matched_differences_condition_on_pre_ls(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_options: Path = BLD / "model" / "options.pkl",
    ever_caregivers: bool = False,
    ever_care_demand: bool = True,
    window: int = 20,
    ages: list[int] | None = None,
) -> None:
    """Create event study plots conditioned on pre-labor supply.

    Creates plots for baseline vs no care demand, conditioned on labor supply
    at t=-1 (unemployed, part-time, full-time). Plots outcomes by age at first
    care spell or first care demand.

    Args:
        path_to_original_data: Path to baseline simulated data
        path_to_no_care_demand_data: Path to no care demand simulated data
        path_to_options: Path to model options
        ever_caregivers: Whether to filter to ever-caregivers
        ever_care_demand: Whether to filter to ever-care-demand
        window: Window size for event study
        ages: List of ages at first event (default: [45, 50, 55, 60, 65])
    """
    if ages is None:
        ages = [45, 50, 55, 60, 65]

    # Load and prepare data
    df_o, df_c = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_original_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
        ever_care_demand=ever_care_demand,
    )

    # Load options
    options = pickle.load(path_to_options.open("rb"))

    # Create output directory
    output_dir = (
        BLD / "plots" / "counterfactual" / "no_care_demand" / "condition_on_pre_ls"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Labor supply states to condition on
    labor_supply_states = ["unemployed", "part_time", "full_time"]

    # Event types
    event_types = ["care", "care_demand"]

    # Create plots for each combination
    for event_type in event_types:
        for ls_state in labor_supply_states:
            create_event_study_plots(
                df_o=df_o,
                df_c=df_c,
                options=options,
                event_type=event_type,
                labor_supply_state=ls_state,
                ages=ages,
                window=window,
                output_dir=output_dir,
            )


def _add_distance_to_first_care(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care column to original data where 0 is first care."""
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = _ensure_agent_period(df)
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df["choice"].isin(care_codes)
    first_care = (
        df.loc[caregiving_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_care_period"})
    )
    out = df.merge(first_care, on="agent", how="left")
    out["distance_to_first_care"] = out["period"] - out["first_care_period"]
    return out


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


def get_labor_supply_state(choice: int) -> str:
    """Map choice value to labor supply state.

    Args:
        choice: Choice value from the model

    Returns:
        Labor supply state: 'retired', 'unemployed', 'part_time', or 'full_time'
    """
    # Convert JAX arrays to lists for comparison
    retirement_values = np.asarray(RETIREMENT).ravel().tolist()
    unemployed_values = np.asarray(UNEMPLOYED).ravel().tolist()
    part_time_values = np.asarray(PART_TIME).ravel().tolist()
    full_time_values = np.asarray(FULL_TIME).ravel().tolist()

    if choice in retirement_values:
        return "retired"
    if choice in unemployed_values:
        return "unemployed"
    if choice in part_time_values:
        return "part_time"
    if choice in full_time_values:
        return "full_time"
    return "unknown"


def filter_by_labor_supply_at_t_minus_5_both_scenarios(
    merged_df: pd.DataFrame,
    df_baseline: pd.DataFrame,
    df_counterfactual: pd.DataFrame,
    distance_col: str,
    labor_supply_state: str,
) -> pd.DataFrame:
    """Filter merged DataFrame to agents with same labor supply at t=-5.

    Conditions on BOTH baseline and counterfactual scenarios having the same
    labor supply at t=-5, and that labor supply matches the specified state.

    Args:
        merged_df: Merged DataFrame with differences
            (from merge_and_compute_differences)
        df_baseline: Baseline DataFrame with choice column
        df_counterfactual: Counterfactual (no care demand) DataFrame
            with choice column
        distance_col: Name of distance column (e.g., 'distance_to_first_care')
        labor_supply_state: Labor supply state to filter on
            ('unemployed', 'part_time', 'full_time')

    Returns:
        Filtered merged DataFrame
    """
    # Ensure both DataFrames have agent/period columns
    df_baseline = df_baseline.copy()
    df_baseline = _ensure_agent_period(df_baseline)
    df_counterfactual = df_counterfactual.copy()
    df_counterfactual = _ensure_agent_period(df_counterfactual)

    # Get labor supply state for each period in both scenarios
    df_baseline["labor_supply_state"] = df_baseline["choice"].apply(
        get_labor_supply_state
    )
    df_counterfactual["labor_supply_state"] = df_counterfactual["choice"].apply(
        get_labor_supply_state
    )

    # Get distance information from merged DataFrame to find t=-5
    df_with_dist = merged_df[["agent", "period", distance_col]].drop_duplicates()

    # Add distance to baseline and counterfactual
    df_baseline = df_baseline.merge(df_with_dist, on=["agent", "period"], how="inner")
    df_counterfactual = df_counterfactual.merge(
        df_with_dist, on=["agent", "period"], how="inner"
    )

    # Get labor supply at t=-5 from both scenarios
    T_MINUS_5 = -5
    df_baseline_t_minus_5 = df_baseline[df_baseline[distance_col] == T_MINUS_5].copy()
    df_baseline_t_minus_5 = df_baseline_t_minus_5[
        ["agent", "labor_supply_state"]
    ].rename(columns={"labor_supply_state": "baseline_labor_supply_t_minus_5"})

    df_counterfactual_t_minus_5 = df_counterfactual[
        df_counterfactual[distance_col] == T_MINUS_5
    ].copy()
    df_counterfactual_t_minus_5 = df_counterfactual_t_minus_5[
        ["agent", "labor_supply_state"]
    ].rename(columns={"labor_supply_state": "counterfactual_labor_supply_t_minus_5"})

    # Merge both to get agents where both scenarios have the same labor supply at t=-5
    both_scenarios_ls = df_baseline_t_minus_5.merge(
        df_counterfactual_t_minus_5, on="agent", how="inner"
    )

    # Filter to agents where both scenarios have the SAME labor supply at t=-5
    # AND that labor supply matches the specified state
    both_scenarios_ls = both_scenarios_ls[
        (
            both_scenarios_ls["baseline_labor_supply_t_minus_5"]
            == both_scenarios_ls["counterfactual_labor_supply_t_minus_5"]
        )
        & (both_scenarios_ls["baseline_labor_supply_t_minus_5"] == labor_supply_state)
    ].copy()

    # Merge back to merged DataFrame and filter
    merged_with_ls = merged_df.merge(
        both_scenarios_ls[["agent"]], on="agent", how="inner"
    )

    return merged_with_ls


def plot_event_study_by_age(
    prof_data: pd.DataFrame,
    x_col: str,
    y_col: str,
    age_col: str,
    ages: list[int],
    path_to_plot: Path,
    ylabel: str,
    xlabel: str,
    title: str,
    window: int = 20,
) -> None:
    """Plot event study with separate lines for each age at first event.

    Args:
        prof_data: DataFrame with distance, age, and outcome columns
        x_col: Name of x-axis column (distance)
        y_col: Name of y-axis column (difference)
        age_col: Name of age column
        ages: List of ages to plot as separate lines
        path_to_plot: Path to save plot
        ylabel: Y-axis label
        xlabel: X-axis label
        title: Plot title
        window: Window size for x-axis limits
    """
    plt.figure(figsize=(12, 7))

    colors = [JET_COLOR_MAP[i] for i in range(len(ages))]

    for age, color in zip(ages, colors, strict=False):
        prof_age = prof_data[prof_data[age_col] == age].copy()
        if not prof_age.empty:
            plt.plot(
                prof_age[x_col],
                prof_age[y_col],
                label=f"Age {age}",
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
            )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5, linewidth=1.5)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=1)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.legend(loc="best", prop={"size": 12}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


def create_event_study_plots(
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    options: dict,
    event_type: str,  # 'care' or 'care_demand'
    labor_supply_state: str,  # 'unemployed', 'part_time', 'full_time'
    ages: list[int],
    window: int,
    output_dir: Path,
) -> None:
    """Create event study plots for a given event type and labor supply state.

    Args:
        df_o: Original (baseline) DataFrame
        df_c: Counterfactual (no care demand) DataFrame
        options: Model options
        event_type: 'care' or 'care_demand'
        labor_supply_state: Labor supply state at t=-1
        ages: List of ages at first event
        window: Window size for event study
        output_dir: Directory to save plots
    """
    # Calculate outcomes
    o_outcomes = calculate_outcomes(df_o, choice_set_type="original")
    c_outcomes = calculate_outcomes(df_c, choice_set_type="no_care_demand")

    # Calculate working hours
    model_params = options["model_params"]
    o_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_o, model_params, choice_set_type="original"
    )
    c_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_c, model_params, choice_set_type="no_care_demand"
    )

    # Calculate additional outcomes
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
        "hours_weekly",
        "gross_labor_income",
    ]
    merged = merge_and_compute_differences(o_cols, c_cols, outcome_names)

    # Add distance and age at first event
    if event_type == "care":
        df_o_dist = _add_distance_to_first_care(df_o)
        distance_col = "distance_to_first_care"
        period_col = "first_care_period"
        age_col_name = "age_at_first_care"
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiving_mask = df_o["choice"].isin(care_codes)
    else:  # care_demand
        df_o_dist = _add_distance_to_first_care_demand(df_o)
        distance_col = "distance_to_first_care_demand"
        period_col = "first_care_demand_period"
        age_col_name = "age_at_first_care_demand"
        # CARE_DEMAND_AND_NO_OTHER_SUPPLY: care_demand == 2
        # Filter to first occurrence where care_demand ==
        # CARE_DEMAND_AND_NO_OTHER_SUPPLY
        care_demand_mask = df_o["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY

    # Get first event period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)[period_col].first().reset_index()
    )

    # Get age at first event
    if event_type == "care":
        first_event_with_age = get_age_at_first_event(
            df_o, caregiving_mask, age_col_name
        )
    else:
        first_event_with_age = get_age_at_first_event(
            df_o, care_demand_mask, age_col_name
        )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged[distance_col] = merged["period"] - merged[period_col]
    merged = merged.merge(first_event_with_age, on="agent", how="left")

    # Filter to window
    merged = merged[
        (merged[distance_col] >= -window) & (merged[distance_col] <= window)
    ]

    # Filter by labor supply at t=-5 in BOTH scenarios
    # This conditions on both baseline and counterfactual having
    # the same labor supply at t=-5
    merged_filtered = filter_by_labor_supply_at_t_minus_5_both_scenarios(
        merged, df_o, df_c, distance_col, labor_supply_state
    )

    # Filter to specific ages
    merged_filtered = merged_filtered[merged_filtered[age_col_name].isin(ages)].copy()

    # Average differences by distance and age
    prof = (
        merged_filtered.groupby([distance_col, age_col_name], observed=False)[
            ["diff_work", "diff_gross_labor_income", "diff_hours_weekly"]
        ]
        .mean()
        .reset_index()
        .sort_values([age_col_name, distance_col])
    )

    # Create plots for each outcome
    outcomes_to_plot = {
        "employment_rate": {
            "col": "diff_work",
            "ylabel": "Employment Rate\n(Baseline - No Care Demand)",
        },
        "gross_labor_income": {
            "col": "diff_gross_labor_income",
            "ylabel": "Gross Labor Income (Monthly)\n(Baseline - No Care Demand)",
        },
        "working_hours": {
            "col": "diff_hours_weekly",
            "ylabel": "Working Hours (Weekly)\n(Baseline - No Care Demand)",
        },
    }

    event_label = "First Care Spell" if event_type == "care" else "First Care Demand"
    ls_label = labor_supply_state.replace("_", " ").title()

    for outcome_name, outcome_info in outcomes_to_plot.items():
        xlabel = f"Year relative to {event_label.lower()}"
        title = (
            f"{outcome_info['ylabel'].split(chr(10))[0]}\n"
            f"Conditioned on {ls_label} at t=-1"
        )

        plot_event_study_by_age(
            prof_data=prof,
            x_col=distance_col,
            y_col=outcome_info["col"],
            age_col=age_col_name,
            ages=ages,
            path_to_plot=output_dir
            / f"{outcome_name}_by_age_at_{event_type}_{labor_supply_state}.png",
            ylabel=outcome_info["ylabel"],
            xlabel=xlabel,
            title=title,
            window=window,
        )
