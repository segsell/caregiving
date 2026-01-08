"""Plot overall employment rate by distance to first caregiving spell.

This module creates event study plots comparing baseline vs no-care-demand
employment rates, aligned by distance to first caregiving spell (t=0).
No age bins or age differentiation - overall comparison only.
"""

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
    prepare_dataframes_simple,
)
from caregiving.counterfactual.task_plot_labor_supply_differences import (
    _add_distance_to_first_care,
)
from caregiving.model.shared import (
    INFORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
)


@pytask.mark.counterfactual_differences_end_of_caregiving
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_employment_rate_by_distance_to_first_care(
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
    / "matched_differences_end_of_caregiving"
    / "employment_rate_by_distance_to_first_care.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = False,
    window: int = 20,
) -> None:
    """Plot overall employment rate by distance to first caregiving spell.

    Creates an event study plot comparing baseline vs no-care-demand employment
    rates, where t=0 is the start of the first caregiving spell. No age bins or
    age differentiation - overall comparison across all agents.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers/ever-care-demand.
      2) Ensure agent/period columns.
      3) Calculate employment outcomes (work indicator) for both scenarios.
      4) Merge on (agent, period) to ensure matched comparison.
      5) Compute distance_to_first_care from baseline, attach to merged.
      6) Aggregate employment rates by distance (baseline and counterfactual separately).
      7) Plot both series on same graph.

    Args:
        path_to_original_data: Path to baseline simulated data
        path_to_no_care_demand_data: Path to no-care-demand counterfactual data
        path_to_plot: Path to save the plot
        ever_caregivers: If True, filter to agents who ever provided care
        ever_care_demand: If True, filter to agents who ever experienced care demand
        window: Window size around event (e.g., 20 = -20 to +20 periods)
    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_simple(
        pd.read_pickle(path_to_original_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers,
        ever_care_demand,
    )

    # Calculate employment outcomes
    o_work, _, _ = calculate_simple_outcomes(df_o, "original")
    c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")

    # Create outcome columns
    o_cols = df_o[["agent", "period", "choice"]].copy()
    o_cols["work_o"] = o_work
    # Add current caregiving indicator (1 if currently providing care, 0 otherwise)
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["work_c"] = c_work

    # Merge on (agent, period) to ensure matched comparison
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

    # Compute distance to first care in baseline and attach
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    # Filter to agents with valid first care period (i.e., ever provided care)
    # and trim to window
    merged = merged[
        merged["first_care_period"].notna()
        & (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Aggregate employment rates by distance
    # Group by distance and compute mean employment rate for each scenario
    #
    # BASELINE LINE COMPOSITION:
    # The baseline line includes ALL agents who:
    #   1. Ever provided care (have a valid first_care_period)
    #   2. Are within the window around their first care period
    #   3. Are matched on (agent, period) with the counterfactual
    #
    # At each distance, the baseline line is the average employment rate across:
    #   - People who are currently providing care at that distance
    #   - People who are not currently providing care at that distance
    #   - People who provided care for 1 year, 2 years, 3 years, 4 years, or longer
    #   - People who stopped and resumed care later
    #   - People who never stopped providing care
    #   - People who stopped after different durations
    #
    # This is the MOST INCLUSIVE group - it represents the average employment rate
    # across ALL caregivers at each distance from their first care period, regardless of:
    #   - Current caregiving status
    #   - Caregiving duration
    #   - Whether they stopped/resumed
    #
    # Example at distance = 2 (t=2):
    #   - Person A: Provided care only at t=0 → included
    #   - Person B: Provided care at t=0, t=1, stopped at t=2 → included
    #   - Person C: Provided care at t=0, t=1, t=2, t=3 → included (still providing care)
    #   - Person D: Provided care at t=0, stopped, resumed at t=5 → included
    #   - Person E: Provided care continuously from t=0 to t=10 → included
    #
    # The baseline line is thus a WEIGHTED AVERAGE of all these different caregiving patterns,
    # where the weights are determined by how many people fall into each pattern at each distance.
    prof = (
        merged.groupby("distance_to_first_care", observed=False)[["work_o", "work_c"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Create conditional baseline series:
    # 1. Baseline employment rate, conditioned on current_caregiving == 1, after t=0
    merged_care = merged[
        (merged["current_caregiving"] == 1) & (merged["distance_to_first_care"] >= 0)
    ].copy()
    prof_care = (
        merged_care.groupby("distance_to_first_care", observed=False)[["work_o"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # 2. Baseline employment rate, conditioned on current_caregiving == 0, after t=1
    #
    # HETEROGENEOUS GROUP LOGIC:
    # This group includes ALL people who are NOT currently providing care at each distance >= 1,
    # regardless of their caregiving history. This makes it heterogeneous because it mixes:
    #
    # Example at distance = 2 (t=2):
    #   - Person A: Provided care only at t=0, stopped at t=1 → included (not providing care at t=2)
    #   - Person B: Provided care at t=0, t=1, stopped at t=2 → included (not providing care at t=2)
    #   - Person C: Provided care at t=0, t=1, t=2, stopped at t=3 → NOT included at t=2 (still providing care)
    #   - Person D: Provided care at t=0, stopped at t=1, resumed at t=5, stopped at t=6 → included at t=2 (not providing care)
    #   - Person E: Provided care at t=0, t=1, t=2, t=3, stopped at t=4 → NOT included at t=2 (still providing care)
    #
    # So at each distance, this group is a MIX of:
    #   - People who stopped after 1 period (t=0 only)
    #   - People who stopped after 2 periods (t=0, t=1)
    #   - People who stopped after 3+ periods (but stopped before this distance)
    #   - People who stopped, resumed, and stopped again
    #
    # This is why it's heterogeneous - the composition changes at each distance, and people
    # with different caregiving histories are mixed together.
    # COMMENTED OUT - not plotting heterogeneous group at the moment
    # merged_no_care = merged[
    #     (merged["current_caregiving"] == 0) & (merged["distance_to_first_care"] >= 1)
    # ].copy()
    # prof_no_care = (
    #     merged_no_care.groupby("distance_to_first_care", observed=False)[["work_o"]]
    #     .mean()
    #     .reset_index()
    #     .sort_values("distance_to_first_care")
    # )

    # Identify agents by caregiving duration to create more homogeneous subgroups:
    # - 1-year caregivers: provide care at t=0 only, then stop at t=1
    # - 2-year caregivers: provide care at t=0 and t=1, then stop at t=2
    # - 3-year caregivers: provide care at t=0, t=1, and t=2, then stop at t=3
    # - 4-year caregivers: provide care at t=0, t=1, t=2, and t=3, then stop at t=4
    # Create a pivot table to check caregiving status at each distance for each agent
    agent_care_matrix = merged[merged["distance_to_first_care"] >= 0].pivot_table(
        index="agent",
        columns="distance_to_first_care",
        values="current_caregiving",
        aggfunc="first",
    )

    # Identify 1-year caregivers:
    # - Must provide care at distance 0 (current_caregiving == 1)
    # - Must NOT provide care at distance 1 (current_caregiving == 0)
    # NOTE: We check that they don't resume at distance 2 (the period immediately after stopping)
    # to ensure a "clean break" - this excludes people who have a brief interruption but
    # continue caregiving. However, if they resume much later (e.g., t=10), they're still included.
    agents_1_year = []
    for agent in agent_care_matrix.index:
        # Get caregiving status at each distance, handling missing values
        care_at_0 = (
            agent_care_matrix.loc[agent, 0] == 1
            if 0 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 0])
            else False
        )
        care_at_1 = (
            agent_care_matrix.loc[agent, 1] == 0
            if 1 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 1])
            else True  # If missing, assume they stopped (lenient)
        )
        # Check they don't resume immediately at t=2 (one period after stopping)
        # This ensures a "clean break" rather than a brief interruption
        care_at_2 = (
            agent_care_matrix.loc[agent, 2] == 0
            if 2 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 2])
            else True  # If missing, assume they don't resume (lenient)
        )

        if care_at_0 and care_at_1 and care_at_2:
            agents_1_year.append(agent)

    # Identify 2-year caregivers:
    # - Must provide care at distance 0 and 1 (current_caregiving == 1)
    # - Must NOT provide care at distance 2 (current_caregiving == 0)
    # NOTE: We check that they don't resume at distance 3 (the period immediately after stopping)
    # to ensure a "clean break" - this excludes people who have a brief interruption but
    # continue caregiving. However, if they resume much later (e.g., t=10), they're still included.
    agents_2_year = []
    for agent in agent_care_matrix.index:
        # Get caregiving status at each distance, handling missing values
        care_at_0 = (
            agent_care_matrix.loc[agent, 0] == 1
            if 0 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 0])
            else False
        )
        care_at_1 = (
            agent_care_matrix.loc[agent, 1] == 1
            if 1 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 1])
            else False
        )
        care_at_2 = (
            agent_care_matrix.loc[agent, 2] == 0
            if 2 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 2])
            else True  # If missing, assume they stopped (lenient)
        )
        # Check they don't resume immediately at t=3 (one period after stopping)
        # This ensures a "clean break" rather than a brief interruption
        care_at_3 = (
            agent_care_matrix.loc[agent, 3] == 0
            if 3 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 3])
            else True  # If missing, assume they don't resume (lenient)
        )

        if care_at_0 and care_at_1 and care_at_2 and care_at_3:
            agents_2_year.append(agent)

    # Identify 3-year caregivers:
    # - Must provide care at distance 0, 1, and 2 (current_caregiving == 1)
    # - Must NOT provide care at distance 3 (current_caregiving == 0)
    # NOTE: We check that they don't resume at distance 4 (the period immediately after stopping)
    # to ensure a "clean break" - this excludes people who have a brief interruption but
    # continue caregiving. However, if they resume much later (e.g., t=10), they're still included.
    agents_3_year = []
    for agent in agent_care_matrix.index:
        # Get caregiving status at each distance, handling missing values
        care_at_0 = (
            agent_care_matrix.loc[agent, 0] == 1
            if 0 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 0])
            else False
        )
        care_at_1 = (
            agent_care_matrix.loc[agent, 1] == 1
            if 1 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 1])
            else False
        )
        care_at_2 = (
            agent_care_matrix.loc[agent, 2] == 1
            if 2 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 2])
            else False
        )
        care_at_3 = (
            agent_care_matrix.loc[agent, 3] == 0
            if 3 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 3])
            else True  # If missing, assume they stopped (lenient)
        )
        # Check they don't resume immediately at t=4 (one period after stopping)
        # This ensures a "clean break" rather than a brief interruption
        care_at_4 = (
            agent_care_matrix.loc[agent, 4] == 0
            if 4 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 4])
            else True  # If missing, assume they don't resume (lenient)
        )

        if care_at_0 and care_at_1 and care_at_2 and care_at_3 and care_at_4:
            agents_3_year.append(agent)

    # Identify 4-year caregivers:
    # - Must provide care at distance 0, 1, 2, and 3 (current_caregiving == 1)
    # - Must NOT provide care at distance 4 (current_caregiving == 0)
    # NOTE: We check that they don't resume at distance 5 (the period immediately after stopping)
    # to ensure a "clean break" - this excludes people who have a brief interruption but
    # continue caregiving. However, if they resume much later (e.g., t=10), they're still included.
    agents_4_year = []
    for agent in agent_care_matrix.index:
        # Get caregiving status at each distance, handling missing values
        care_at_0 = (
            agent_care_matrix.loc[agent, 0] == 1
            if 0 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 0])
            else False
        )
        care_at_1 = (
            agent_care_matrix.loc[agent, 1] == 1
            if 1 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 1])
            else False
        )
        care_at_2 = (
            agent_care_matrix.loc[agent, 2] == 1
            if 2 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 2])
            else False
        )
        care_at_3 = (
            agent_care_matrix.loc[agent, 3] == 1
            if 3 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 3])
            else False
        )
        care_at_4 = (
            agent_care_matrix.loc[agent, 4] == 0
            if 4 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 4])
            else True  # If missing, assume they stopped (lenient)
        )
        # Check they don't resume immediately at t=5 (one period after stopping)
        # This ensures a "clean break" rather than a brief interruption
        care_at_5 = (
            agent_care_matrix.loc[agent, 5] == 0
            if 5 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 5])
            else True  # If missing, assume they don't resume (lenient)
        )

        if (
            care_at_0
            and care_at_1
            and care_at_2
            and care_at_3
            and care_at_4
            and care_at_5
        ):
            agents_4_year.append(agent)

    agents_1_year = np.array(agents_1_year)
    agents_2_year = np.array(agents_2_year)
    agents_3_year = np.array(agents_3_year)
    agents_4_year = np.array(agents_4_year)

    # Create conditional series for 1-year caregivers
    merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
    prof_1_year = (
        merged_1_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Create conditional series for 2-year caregivers
    merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
    prof_2_year = (
        merged_2_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Create conditional series for 3-year caregivers
    merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
    prof_3_year = (
        merged_3_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Create conditional series for 4-year caregivers
    merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
    prof_4_year = (
        merged_4_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Plot
    plt.figure(figsize=(12, 7))

    # Plot baseline employment rate
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_o"],
        label="Baseline",
        color="steelblue",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4,
    )

    # Plot no-care-demand employment rate
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_c"],
        label="No Care Demand",
        color="darkorange",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=4,
    )

    # Plot baseline employment rate, conditioned on current_caregiving == 1 (after t=0)
    if len(prof_care) > 0:
        plt.plot(
            prof_care["distance_to_first_care"],
            prof_care["work_o"],
            label="Baseline (Currently Providing Care)",
            color="forestgreen",
            linewidth=2.5,
            linestyle="-.",
            marker="^",
            markersize=4,
        )

    # Plot baseline employment rate for 1-year caregivers (care at t=0 only, then stop)
    if len(prof_1_year) > 0:
        plt.plot(
            prof_1_year["distance_to_first_care"],
            prof_1_year["work_o"],
            label="Baseline (1-Year Caregivers: t=0)",
            color="gold",
            linewidth=2.5,
            linestyle=":",
            marker="*",
            markersize=5,
        )

    # Plot baseline employment rate, conditioned on current_caregiving == 0 (after t=1)
    # COMMENTED OUT - not plotting heterogeneous group at the moment
    # if len(prof_no_care) > 0:
    #     plt.plot(
    #         prof_no_care["distance_to_first_care"],
    #         prof_no_care["work_o"],
    #         label="Baseline (Not Currently Providing Care)",
    #         color="purple",
    #         linewidth=2.5,
    #         linestyle="-.",
    #         marker="v",
    #         markersize=4,
    #     )

    # Plot baseline employment rate for 2-year caregivers (care at t=0 and t=1, then stop)
    if len(prof_2_year) > 0:
        plt.plot(
            prof_2_year["distance_to_first_care"],
            prof_2_year["work_o"],
            label="Baseline (2-Year Caregivers: t=0, t=1)",
            color="crimson",
            linewidth=2.5,
            linestyle=":",
            marker="D",
            markersize=4,
        )

    # Plot baseline employment rate for 3-year caregivers (care at t=0, t=1, t=2, then stop)
    if len(prof_3_year) > 0:
        plt.plot(
            prof_3_year["distance_to_first_care"],
            prof_3_year["work_o"],
            label="Baseline (3-Year Caregivers: t=0, t=1, t=2)",
            color="teal",
            linewidth=2.5,
            linestyle=":",
            marker="X",
            markersize=4,
        )

    # Plot baseline employment rate for 4-year caregivers (care at t=0, t=1, t=2, t=3, then stop)
    if len(prof_4_year) > 0:
        plt.plot(
            prof_4_year["distance_to_first_care"],
            prof_4_year["work_o"],
            label="Baseline (4-Year Caregivers: t=0, t=1, t=2, t=3)",
            color="maroon",
            linewidth=2.5,
            linestyle=":",
            marker="P",
            markersize=4,
        )

    # Add vertical line at t=0 (start of first caregiving spell)
    plt.axvline(
        x=0, color="k", linestyle=":", linewidth=2, alpha=0.7, label="First Care"
    )

    # Formatting
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel("Employment Rate", fontsize=16)
    plt.title(
        "Employment Rate by Distance to First Caregiving Spell\n(Baseline vs No Care Demand)",
        fontsize=18,
    )
    plt.xlim(-window, window)
    plt.ylim(0, 1)  # Employment rate is between 0 and 1
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 14}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_end_of_caregiving_intensity
@pytask.mark.counterfactual_differences_no_care_demand
def task_plot_employment_rate_by_intensity_and_distance(
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
    / "matched_differences_end_of_caregiving"
    / "employment_rate_by_intensity_and_distance.png",
    ever_caregivers: bool = False,
    ever_care_demand: bool = False,
    window: int = 20,
) -> None:
    """Plot employment rate by caregiving intensity and distance to first caregiving spell.

    Creates an event study plot comparing employment rates for people currently providing
    light care vs intensive care at each distance from their first caregiving spell (t=0).
    This shows how employment responds to current caregiving intensity at each point.

    NOTE: This analysis is heterogeneous - the composition of people in each intensity
    group changes at each distance because people can switch between light and intensive
    care, or stop providing care. This is similar to "Currently Providing Care" but
    split by intensity.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers/ever-care-demand.
      2) Ensure agent/period columns.
      3) Calculate employment outcomes (work indicator) for both scenarios.
      4) Merge on (agent, period) to ensure matched comparison.
      5) Compute distance_to_first_care from baseline, attach to merged.
      6) Identify current caregiving intensity (light vs intensive) at each distance.
      7) Aggregate employment rates by distance and intensity (baseline and counterfactual separately).
      8) Plot both intensity groups on same graph.

    Args:
        path_to_original_data: Path to baseline simulated data
        path_to_no_care_demand_data: Path to no-care-demand counterfactual data
        path_to_plot: Path to save the plot
        ever_caregivers: If True, filter to agents who ever provided care
        ever_care_demand: If True, filter to agents who ever experienced care demand
        window: Window size around event (e.g., 20 = -20 to +20 periods)
    """
    # Load and prepare data
    df_o, df_c = prepare_dataframes_simple(
        pd.read_pickle(path_to_original_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers,
        ever_care_demand,
    )

    # Calculate employment outcomes
    o_work, _, _ = calculate_simple_outcomes(df_o, "original")
    c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")

    # Create outcome columns
    o_cols = df_o[["agent", "period", "choice"]].copy()
    o_cols["work_o"] = o_work
    # Add current caregiving intensity indicators
    light_care_codes = np.asarray(LIGHT_INFORMAL_CARE).ravel().tolist()
    intensive_care_codes = np.asarray(INTENSIVE_INFORMAL_CARE).ravel().tolist()
    o_cols["current_light_care"] = o_cols["choice"].isin(light_care_codes).astype(int)
    o_cols["current_intensive_care"] = (
        o_cols["choice"].isin(intensive_care_codes).astype(int)
    )

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["work_c"] = c_work

    # Merge on (agent, period) to ensure matched comparison
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

    # Compute distance to first care in baseline and attach
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    # Filter to agents with valid first care period (i.e., ever provided care)
    # and trim to window
    merged = merged[
        merged["first_care_period"].notna()
        & (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Aggregate employment rates by distance and intensity
    # For baseline: split by current intensity at each distance
    merged_light = merged[
        (merged["current_light_care"] == 1) & (merged["distance_to_first_care"] >= 0)
    ].copy()
    prof_light = (
        merged_light.groupby("distance_to_first_care", observed=False)[["work_o"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    merged_intensive = merged[
        (merged["current_intensive_care"] == 1)
        & (merged["distance_to_first_care"] >= 0)
    ].copy()
    prof_intensive = (
        merged_intensive.groupby("distance_to_first_care", observed=False)[["work_o"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Also compute overall baseline and no-care-demand for reference
    prof = (
        merged.groupby("distance_to_first_care", observed=False)[["work_o", "work_c"]]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Plot
    plt.figure(figsize=(12, 7))

    # Plot baseline overall employment rate (reference)
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_o"],
        label="Baseline (All Caregivers)",
        color="steelblue",
        linewidth=2,
        linestyle="-",
        marker="o",
        markersize=3,
        alpha=0.6,
    )

    # Plot no-care-demand employment rate (reference)
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_c"],
        label="No Care Demand",
        color="darkorange",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=4,
    )

    # Plot baseline employment rate for currently providing light care
    if len(prof_light) > 0:
        plt.plot(
            prof_light["distance_to_first_care"],
            prof_light["work_o"],
            label="Baseline (Currently Providing Light Care)",
            color="lightblue",
            linewidth=2.5,
            linestyle="-",
            marker="^",
            markersize=5,
        )

    # Plot baseline employment rate for currently providing intensive care
    if len(prof_intensive) > 0:
        plt.plot(
            prof_intensive["distance_to_first_care"],
            prof_intensive["work_o"],
            label="Baseline (Currently Providing Intensive Care)",
            color="darkred",
            linewidth=2.5,
            linestyle="-",
            marker="v",
            markersize=5,
        )

    # Add vertical line at t=0 (start of first caregiving spell)
    plt.axvline(
        x=0, color="k", linestyle=":", linewidth=2, alpha=0.7, label="First Care"
    )

    # Formatting
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel("Employment Rate", fontsize=16)
    plt.title(
        "Employment Rate by Caregiving Intensity and Distance to First Caregiving Spell\n"
        "(Baseline: Light vs Intensive Care)",
        fontsize=18,
    )
    plt.xlim(-window, window)
    plt.ylim(0, 1)  # Employment rate is between 0 and 1
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", prop={"size": 12}, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()
