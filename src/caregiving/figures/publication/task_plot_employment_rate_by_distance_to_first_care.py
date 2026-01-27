"""Plot overall employment rate by distance to first caregiving spell.

This module creates event study plots comparing baseline vs no-care-demand
employment rates, aligned by distance to first caregiving spell (t=0).
No age bins or age differentiation - overall comparison only.
"""

from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    calculate_simple_outcomes,
    get_age_at_first_event,
    prepare_dataframes_simple,
)
from caregiving.counterfactual.task_plot_labor_supply_differences import (
    _add_distance_to_first_care,
)
from caregiving.model.shared import (
    INFORMAL_CARE,
)

for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication
    @pytask.task(id=age_label_val)
    def task_plot_employment_rate_by_distance_to_first_care(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / f"employment_rate_by_distance_to_first_care_{age_label_val}.pdf",
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot overall employment rate by distance to first caregiving spell.

        Creates an event study plot comparing baseline vs no-care-demand employment
        rates, where t=0 is the start of the first caregiving spell. Can be filtered
        by age at first care period.

        Steps:
          1) Restrict to alive and (optionally) ever-caregivers/ever-care-demand.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care from baseline, attach to merged.
          6) Filter by age at first care period (if age_min/age_max specified).
          7) Aggregate employment rates by distance (baseline and
          counterfactual separately).
          8) Plot both series on same graph.

        Args:
            age_min: Minimum age at first care period (inclusive).
                If None, no lower bound.
            age_max: Maximum age at first care period (inclusive).
                If None, no upper bound.
            age_label: Label for age group (used in filename)
            path_to_original_data: Path to baseline simulated data
            path_to_no_care_demand_data: Path to no-care-demand counterfactual data
            path_to_plot: Path to save the plot (constructed from age_label)
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
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
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
        merged["distance_to_first_care"] = (
            merged["period"] - merged["first_care_period"]
        )

        # Get age at first care period for filtering
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")

        # Filter to agents with valid first care period (i.e., ever provided care)
        # and trim to window
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]

        # Filter by age at first care period if specified
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

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
        # # across ALL caregivers at each distance from their first care period, rega
        # rdless of:
        #   - Current caregiving status
        #   - Caregiving duration
        #   - Whether they stopped/resumed
        #
        # Example at distance = 2 (t=2):
        #   - Person A: Provided care only at t=0 → included
        #   - Person B: Provided care at t=0, t=1, stopped at t=2 → included
        # #   - Person C: Provided care at t=0, t=1, t=2, t=3 → included (still provi
        # ding care)
        #   - Person D: Provided care at t=0, stopped, resumed at t=5 → included
        #   - Person E: Provided care continuously from t=0 to t=10 → included
        #
        # # The baseline line is thus a WEIGHTED AVERAGE of all these different careg
        # iving patterns,
        # # where the weights are determined by how many people fall into each pattern
        # at each distance.
        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional baseline series:
        # 1. Baseline employment rate, conditioned on current_caregiving == 1, after t=0
        # COMMENTED OUT - not currently used in plotting
        # merged_care = merged[
        #     (merged["current_caregiving"] == 1)
        #     & (merged["distance_to_first_care"] >= 0)
        # ].copy()
        # prof_care = (
        #     merged_care.groupby("distance_to_first_care", observed=False)[["work_o"]]
        #     .mean()
        #     .reset_index()
        #     .sort_values("distance_to_first_care")
        # )

        # 2. Baseline employment rate, conditioned on current_caregiving == 0, after t=1
        #
        # HETEROGENEOUS GROUP LOGIC:
        # # This group includes ALL people who are NOT currently providing care at ea
        # ch distance >= 1,
        # # regardless of their caregiving history. This makes it heterogeneous becau
        # se it mixes:
        #
        # Example at distance = 2 (t=2):  # noqa: E501
        # #   - Person A: Provided care only at t=0, stopped at t=1 → included (not providing care at t=2)  # noqa: E501
        # #   - Person B: Provided care at t=0, t=1, stopped at t=2 → included (not providing care at t=2)  # noqa: E501
        # #   - Person C: Provided care at t=0, t=1, t=2, stopped at t=3 → NOT included at t=2 (still providing care)  # noqa: E501
        # #   - Person D: Provided care at t=0, stopped at t=1, resumed at t=5, stopped at t=6 → included at t=2 (not providing care)  # noqa: E501
        # #   - Person E: Provided care at t=0, t=1, t=2, t=3, stopped at t=4 → NOT included at t=2 (still providing care)  # noqa: E501
        #
        # So at each distance, this group is a MIX of:
        #   - People who stopped after 1 period (t=0 only)
        #   - People who stopped after 2 periods (t=0, t=1)
        #   - People who stopped after 3+ periods (but stopped before this distance)
        #   - People who stopped, resumed, and stopped again
        #
        # # This is why it's heterogeneous - the composition changes at each distance
        # , and people
        # with different caregiving histories are mixed together.
        # COMMENTED OUT - not plotting heterogeneous group at the moment
        # merged_no_care = merged[
        #     (merged["current_caregiving"] == 0)
        #     & (merged["distance_to_first_care"] >= 1)
        # ].copy()
        # prof_no_care = (
        #     merged_no_care.groupby(
        #         "distance_to_first_care", observed=False
        #     )[["work_o"]]
        #     .mean()
        #     .reset_index()
        #     .sort_values("distance_to_first_care")
        # )

        # Identify agents by caregiving duration to create more homogeneous subgroups:
        # - 1-year caregivers: provide care at t=0 only, then stop at t=1
        # - 2-year caregivers: provide care at t=0 and t=1, then stop at t=2
        # - 3-year caregivers: provide care at t=0, t=1, and t=2, then stop at t=3
        # - 4-year caregivers: provide care at t=0, t=1, t=2, and t=3,
        #   then stop at t=4
        # Create a pivot table to check caregiving status at each distance
        # for each agent
        agent_care_matrix = merged[merged["distance_to_first_care"] >= 0].pivot_table(
            index="agent",
            columns="distance_to_first_care",
            values="current_caregiving",
            aggfunc="first",
        )

        # Identify 1-year caregivers:
        # - Must provide care at distance 0 (current_caregiving == 1)
        # - Must NOT provide care at distance 1 (current_caregiving == 0)
        # # NOTE: We check that they don't resume at distance 2 (the period
        # # immediately after stopping)
        # # to ensure a "clean break" - this excludes people who have a brief
        # # interruption but continue caregiving. However, if they resume much
        # # later (e.g., t=10), they're still included.
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
                if 2 in agent_care_matrix.columns  # noqa: PLR2004
                and pd.notna(agent_care_matrix.loc[agent, 2])
                else True  # If missing, assume they don't resume (lenient)
            )

            if care_at_0 and care_at_1 and care_at_2:
                agents_1_year.append(agent)

        # Identify 2-year caregivers:
        # - Must provide care at distance 0 and 1 (current_caregiving == 1)
        # - Must NOT provide care at distance 2 (current_caregiving == 0)
        # # NOTE: We check that they don't resume at distance 3 (the period
        # # immediately after stopping)
        # # to ensure a "clean break" - this excludes people who have a brief
        # # interruption but continue caregiving. However, if they resume much
        # # later (e.g., t=10), they're still included.
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
                if 2 in agent_care_matrix.columns  # noqa: PLR2004
                and pd.notna(agent_care_matrix.loc[agent, 2])
                else True  # If missing, assume they stopped (lenient)
            )
            # Check they don't resume immediately at t=3 (one period after stopping)
            # This ensures a "clean break" rather than a brief interruption
            care_at_3 = (
                agent_care_matrix.loc[agent, 3] == 0
                if 3 in agent_care_matrix.columns  # noqa: PLR2004
                and pd.notna(agent_care_matrix.loc[agent, 3])
                else True  # If missing, assume they don't resume (lenient)
            )

            if care_at_0 and care_at_1 and care_at_2 and care_at_3:
                agents_2_year.append(agent)

        # Identify 3-year caregivers:
        # - Must provide care at distance 0, 1, and 2 (current_caregiving == 1)
        # - Must NOT provide care at distance 3 (current_caregiving == 0)
        # # NOTE: We check that they don't resume at distance 4 (the period
        # # immediately after stopping)
        # # to ensure a "clean break" - this excludes people who have a brief
        # # interruption but continue caregiving. However, if they resume much
        # # later (e.g., t=10), they're still included.
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
                if 2 in agent_care_matrix.columns  # noqa: PLR2004
                and pd.notna(agent_care_matrix.loc[agent, 2])
                else False
            )
            care_at_3 = (
                agent_care_matrix.loc[agent, 3] == 0
                if 3 in agent_care_matrix.columns  # noqa: PLR2004
                and pd.notna(agent_care_matrix.loc[agent, 3])
                else True  # If missing, assume they stopped (lenient)
            )
            # Check they don't resume immediately at t=4 (one period after stopping)
            # This ensures a "clean break" rather than a brief interruption
            care_at_4 = (
                agent_care_matrix.loc[agent, 4] == 0
                if 4 in agent_care_matrix.columns  # noqa: PLR2004
                and pd.notna(agent_care_matrix.loc[agent, 4])
                else True  # If missing, assume they don't resume (lenient)
            )

            if care_at_0 and care_at_1 and care_at_2 and care_at_3 and care_at_4:
                agents_3_year.append(agent)

        # Identify 4-year caregivers:
        # - Must provide care at distance 0, 1, 2, and 3 (current_caregiving == 1)
        # - Must NOT provide care at distance 4 (current_caregiving == 0)
        # # NOTE: We check that they don't resume at distance 5 (the period
        # # immediately after stopping)
        # # to ensure a "clean break" - this excludes people who have a brief
        # # interruption but continue caregiving. However, if they resume much
        # # later (e.g., t=10), they're still included.
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
                if 2 in agent_care_matrix.columns  # noqa: PLR2004
                and pd.notna(agent_care_matrix.loc[agent, 2])
                else False
            )
            care_at_3 = (
                agent_care_matrix.loc[agent, 3] == 1
                if 3 in agent_care_matrix.columns  # noqa: PLR2004
                and pd.notna(agent_care_matrix.loc[agent, 3])
                else False
            )
            care_at_4 = (
                agent_care_matrix.loc[agent, 4] == 0
                if 4 in agent_care_matrix.columns  # noqa: PLR2004
                and pd.notna(agent_care_matrix.loc[agent, 4])
                else True  # If missing, assume they stopped (lenient)
            )
            # Check they don't resume immediately at t=5 (one period after stopping)
            # This ensures a "clean break" rather than a brief interruption
            care_at_5 = (
                agent_care_matrix.loc[agent, 5] == 0
                if 5 in agent_care_matrix.columns  # noqa: PLR2004
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

        # Call plotting function
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            window=window,
            path_to_plot=path_to_plot,
        )


def plot_employment_rate_by_distance(  # noqa: PLR0913
    prof,
    prof_1_year,
    prof_2_year,
    prof_3_year,
    prof_4_year,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
) -> None:
    """Plot employment rate by distance to first caregiving spell.

    Creates an event study plot comparing baseline vs no-care-demand employment
    rates, with separate lines for different caregiving durations.

    Args:
        prof: DataFrame with columns 'distance_to_first_care', 'work_o', 'work_c'
        prof_1_year: DataFrame for 1-year caregivers
        prof_2_year: DataFrame for 2-year caregivers
        prof_3_year: DataFrame for 3-year caregivers
        prof_4_year: DataFrame for 4-year caregivers
        window: Window size around event (e.g., 20 = -20 to +20 periods)
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
    """
    # Plot
    # Increased figure size to maintain visual balance with thinner lines/text
    plt.figure(figsize=(14, 8))

    # # Plot overall baseline employment rate (all caregivers)
    # plt.plot(
    #     prof["distance_to_first_care"],
    #     prof["work_o"],
    #     label="Baseline (All Caregivers)",
    #     color="black",
    #     linewidth=2.0,
    #     linestyle="-",
    #     marker=None,
    # )

    # Plot no-care-demand employment rate
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_c"],
        label="No Care Demand",
        color="black",
        linewidth=2.0,
        linestyle="-",
        marker=None,
    )

    # Plot baseline employment rate for 1-year caregivers (care at t=0 only, then stop)
    if len(prof_1_year) > 0:
        plt.plot(
            prof_1_year["distance_to_first_care"],
            prof_1_year["work_o"],
            label="Baseline (1-Year Caregivers: t=0)",
            color="0.8",
            linewidth=2.0,
            linestyle="-",
            marker="8",  # Octagon
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # # Plot baseline employment rate for 2-year caregivers (care at t=0 and t=1,
    #  then stop)
    if len(prof_2_year) > 0:
        plt.plot(
            prof_2_year["distance_to_first_care"],
            prof_2_year["work_o"],
            label="Baseline (2-Year Caregivers: t=0, t=1)",
            color="0.6",
            linewidth=2.0,
            linestyle="-",
            marker="^",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # # Plot baseline employment rate for 3-year caregivers (care at t=0, t=1, t=
    # 2, then stop)
    if len(prof_3_year) > 0:
        plt.plot(
            prof_3_year["distance_to_first_care"],
            prof_3_year["work_o"],
            label="Baseline (3-Year Caregivers: t=0, t=1, t=2)",
            color="0.4",
            linewidth=2.0,
            linestyle="-",
            marker="D",  # Diamond
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # # Plot baseline employment rate for 4-year caregivers (care at t=0, t=1, t=
    # 2, t=3, then stop)
    if len(prof_4_year) > 0:
        plt.plot(
            prof_4_year["distance_to_first_care"],
            prof_4_year["work_o"],
            label="Baseline (4-Year Caregivers: t=0, t=1, t=2, t=3)",
            color="0.2",
            linewidth=2.0,
            linestyle="-",
            marker="s",  # Hollow square
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Add vertical line at t=0 (start of first caregiving spell)
    # Position at -0.5 with spaced-out dashes
    plt.axvline(
        x=-0.5,
        color="k",
        linestyle=(
            0,
            (7, 7),
        ),  # Custom dash pattern: 7 points on, 7 points off (2/3 of 10)
        linewidth=1.0,
    )

    # Formatting
    plt.xlabel("Year relative to start of first care spell", fontsize=14)
    plt.ylabel("Employment Rate", fontsize=14)
    # Add padding: x-axis extends beyond -window and window, y-axis extends below 0
    plt.xlim(-window - 0.5, window + 0.5)
    plt.ylim(-0.025, 1.0)  # Employment rate is between 0 and 1, with padding below
    plt.grid(True, axis="y", alpha=0.3, linewidth=0.8)  # Only horizontal grid lines
    # Set ticks to original range (no ticks in padding area)
    plt.xticks(range(-window, window + 1, 5), fontsize=12)
    plt.yticks(fontsize=12)
    # plt.legend(loc="best", prop={"size": 12}, framealpha=0.9)  # Temporarily hidden

    # Remove top and right spines (box lines)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Make tick marks longer
    ax.tick_params(axis="both", length=8)

    plt.tight_layout()
    if path_to_plot:
        plt.savefig(path_to_plot, dpi=1200, bbox_inches="tight")
    plt.close()
