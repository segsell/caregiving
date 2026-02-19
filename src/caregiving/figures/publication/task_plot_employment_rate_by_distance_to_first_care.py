"""Plot overall employment rate by distance to first caregiving spell.

This module creates event study plots comparing baseline vs no-care-demand
employment rates, aligned by distance to first caregiving spell (t=0).
No age bins or age differentiation - overall comparison only.
"""

import pickle
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    add_distance_to_first_care,
    add_distance_to_first_care_demand,
    calculate_simple_outcomes,
    ensure_agent_period,
    get_age_at_first_event,
    identify_agents_by_total_caregiving_over_lifecycle,
    prepare_dataframes_simple,
)
from caregiving.figures.publication.plotting_helpers import (
    identify_agents_by_duration_at_least,
)
from caregiving.model.shared import (
    ALL_CARE,
    INFORMAL_CARE,
)


def _identify_agents_by_duration(
    merged: pd.DataFrame,
    distance_col: str,
    duration_type: str = "care_demand",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by duration of care demand or caregiving.

    For care_demand: Identifies agents who experience care_demand > 0 for
    1, 2, 3, or 4 years (includes all agents, not just informal caregivers).
    For caregiving: Identifies agents who provide informal care for
    1, 2, 3, or 4 years (only type_1 agents can provide informal care).

    Args:
        merged: DataFrame with agent, distance, and relevant columns
        distance_col: Name of distance column (e.g., "distance_to_first_care_demand")
        duration_type: "care_demand" or "caregiving"

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year)
        as numpy arrays of agent IDs
    """
    # Create matrix of care status by distance
    if duration_type == "care_demand":
        # Use care_demand > 0 to identify duration
        merged["care_status"] = (merged["care_demand"] > 0).astype(int)
    elif duration_type == "caregiving":
        # Use current_caregiving (informal care) to identify duration
        # current_caregiving should already be in merged
        if "current_caregiving" not in merged.columns:
            raise ValueError(
                "current_caregiving column not found. "
                "Cannot identify caregiving duration."
            )
        merged["care_status"] = merged["current_caregiving"]
    else:
        raise ValueError(
            f"duration_type must be 'care_demand' or 'caregiving', got {duration_type}"
        )

    # Create pivot table of care status by distance
    agent_care_matrix = merged[merged[distance_col] >= 0].pivot_table(
        index="agent",
        columns=distance_col,
        values="care_status",
        aggfunc="first",
    )

    # Identify 1-year: care at t=0 only, then stop
    agents_1_year = []
    for agent in agent_care_matrix.index:
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
            else True
        )
        care_at_2 = (
            agent_care_matrix.loc[agent, 2] == 0
            if 2 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, 2])
            else True
        )

        if care_at_0 and care_at_1 and care_at_2:
            agents_1_year.append(agent)

    # Identify 2-year: care at t=0 and t=1, then stop
    agents_2_year = []
    for agent in agent_care_matrix.index:
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
            else True
        )
        care_at_3 = (
            agent_care_matrix.loc[agent, 3] == 0
            if 3 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, 3])
            else True
        )

        if care_at_0 and care_at_1 and care_at_2 and care_at_3:
            agents_2_year.append(agent)

    # Identify 3-year: care at t=0, t=1, t=2, then stop
    agents_3_year = []
    for agent in agent_care_matrix.index:
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
            else True
        )
        care_at_4 = (
            agent_care_matrix.loc[agent, 4] == 0
            if 4 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, 4])
            else True
        )

        if care_at_0 and care_at_1 and care_at_2 and care_at_3 and care_at_4:
            agents_3_year.append(agent)

    # Identify 4-year: care at t=0, t=1, t=2, t=3, then stop
    agents_4_year = []
    for agent in agent_care_matrix.index:
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
            else True
        )
        care_at_5 = (
            agent_care_matrix.loc[agent, 5] == 0
            if 5 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, 5])
            else True
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

    return (
        np.array(agents_1_year),
        np.array(agents_2_year),
        np.array(agents_3_year),
        np.array(agents_4_year),
    )


# ---------------------------------------------------------------------------
# PLOTS OVERVIEW (sample, event, duration, and who we compare)
# ---------------------------------------------------------------------------
# WHO WE COMPARE (same for all tasks):
#   The same individuals (and same agent-periods) are compared in baseline vs no-care-demand.
#   (1) Sample is defined from the BASELINE (original) data: alive only; then optionally
#       restricted to ever-caregivers and/or ever-care-demand via prepare_dataframes_simple
#       (ever_caregivers, ever_care_demand; default False for both).
#   (2) For type_1_* tasks we further restrict to caregiving_type == 1 (same agent set
#       in both baseline and no-care-demand).
#   (3) We merge baseline outcomes and no-care-demand outcomes on (agent, period) with
#       how="inner", so we keep only agent-periods that exist in BOTH datasets. Thus
#       no-care-demand data is effectively restricted to the same agents (and to the same
#       person-periods after the merge) as the baseline sample.
#
# Lines in the plot (same for all plot types):
#   - Dashed black: "Baseline" = mean outcome (e.g. employment rate) in the ORIGINAL
#     scenario for the full sample, by distance to the event (average caregiver).
#   - Solid black: "No Care Demand" = mean outcome in the no-care-demand counterfactual
#     for that same sample, by distance.
#   - Gray lines with markers: mean outcome for each duration subgroup (1–4 year exact
#     consecutive, or at least 1–4 year as noted below).
#   - Vertical dashed line at t=-0.5: event time.
#
# EVENT: Two definitions used in this module:
#   - "First care demand": t=0 = first period with care_demand > 0 (light or intensive).
#   - "First caregiving spell": t=0 = first period in which the agent provides informal care.
#
# 1) task_plot_employment_rate_by_distance_to_first_care_demand_type_0
#    Sample: caregiving_type == 0 only. Event: First care demand.
#    Duration: 1–4 = EXACT consecutive care demand. duration_type="care_demand".
#
# 2) task_plot_employment_rate_by_distance_to_first_care_demand_all
#    Sample: ALL agents. Event: First care demand.
#    Duration: 1–4 = EXACT consecutive care demand. duration_type="care_demand".
#
# 3) task_plot_employment_rate_by_distance_to_first_care_demand_type_1_cg
#    Sample: caregiving_type == 1 only. Event: First care demand.
#    Duration: 1–4 = EXACT consecutive caregiving. duration_type="caregiving".
#
# 4) task_plot_employment_rate_by_distance_to_first_care_demand_type_1_exact_care_demand
#    (employment only). Sample: caregiving_type == 1 only. Event: First care demand.
#    Duration: 1–4 = EXACT consecutive care demand. duration_type="care_demand".
#
# 5) task_plot_employment_rate_by_distance_to_first_caregiving_spell_at_least
#    Sample: ALL agents. Event: First caregiving spell.
#    Duration: 1–4 = AT LEAST caregiving (overlapping). identify_agents_by_duration_at_least.
#
# 6) task_plot_employment_rate_by_distance_to_first_caregiving_spell_exact
#    Sample: ALL agents. Event: First caregiving spell.
#    Duration: 1–4 = EXACT consecutive caregiving (then stop).
#
# 7) task_plot_*_by_distance_to_first_care_demand_total_caregiving [base / _back_to_Jan7]
#    Employment: base, Jan7, back_to_Jan7 (data in function name and filename).
#    publication_other_check (full_time, part_time, working_hours, labor_income): two data variants each:
#    - Standard: estimated_params, no_care_demand; function/filename without suffix/prefix.
#    - back_to_Jan7: function/filename _back_to_Jan7 and back_to_Jan7_ prefix. Output: .../total_caregiving_years/.
#
# 8) task_plot_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving [/_Jan7 /_back_to_Jan7]
#    Sample: ever_caregivers=True. Event: First caregiving spell. Duration: 1–5+ TOTAL care years.
#    Data variant in function name and filename (base, Jan7_, back_to_Jan7_).
#
# 9) task_plot_*_by_distance_to_first_caregiving_spell_total_caregiving [base / _back_to_Jan7] (publication_other_check)
#    (full_time, part_time, working_hours, labor_income). Two data variants each: standard (estimated_params,
#    no_care_demand; no suffix/prefix) and back_to_Jan7 (_back_to_Jan7 / back_to_Jan7_ prefix).
#
# ---------------------------------------------------------------------------
# TASKS WITH publication_counterfactual + publication_employment_check (run with -m publication_employment_check)
# ---------------------------------------------------------------------------
# All use: original = simulated_data_estimated_params_back_to_Jan7.pkl,
#          no_care_demand = simulated_data_no_care_demand_back_to_Jan7.pkl.
# Subsetting: prepare_dataframes_simple (alive; optional ever_caregivers/ever_care_demand) then
#             restrict to caregiving_type == 1; merge on (agent, period).
#
# | Task function (pattern)                    | Event           | Duration        | Output filename pattern (subdir)        |
# |---------------------------------------------|-----------------|-----------------|----------------------------------------|
# | ..._care_demand_type_1_cg                   | First care demand | Caregiving (exact 1–4y) | employment/..._care_demand_type_1_caregiving_duration_{age}.pdf |
# | ..._care_spell_type_1_cd (employment)      | First care demand | Care demand (exact 1–4y) | employment/..._care_spell_type_1_exact_consecutive_caregiving_spells_{age}.pdf |
# | ..._care_spell_type_1_cd (full_time)      | First care demand | Care demand (exact 1–4y) | full_time/..._care_spell_type_1_exact_consecutive_caregiving_spells_{age}.pdf |
# | ..._care_spell_type_1_cd (part_time)       | First care demand | Care demand (exact 1–4y) | part_time/..._care_spell_type_1_... |
# | ..._care_spell_type_1_cd (working_hours)   | First care demand | Care demand (exact 1–4y) | working_hours/..._care_spell_type_1_... |
# | ..._care_spell_type_1_cd (labor_income)    | First care demand | Care demand (exact 1–4y) | labor_income/..._care_spell_type_1_... |
#
# Naming: type_1_cg = "care demand" event + "caregiving duration" in filename. type_1_cd (and outcomes)
#         = "care spell" + "exact_consecutive_caregiving_spells" in filename (axis label "first care spell").
# ---------------------------------------------------------------------------

# Sanity check: Plot for caregiving_type == 0 (agents who cannot provide informal care)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication_employment
    # @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_type_0_exact_care_demand")
    def task_plot_employment_rate_by_distance_to_first_care_demand_type_0(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / (
            f"employment_rate_by_distance_to_first_care_demand_type_0_"
            f"exact_care_demand_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to first care demand (type 0; sanity check).

        Sample: Same agents in baseline and no-care-demand; from baseline (alive, optional
        ever_caregivers/ever_care_demand), restricted to caregiving_type == 0; merge on
        (agent, period). Event: t=0 = first care demand. Type 0 cannot provide care,
        so caregiving duration lines are empty; duration groups use care demand.
        Can be filtered by age at first care demand period.

        Steps:
          1) Restrict to alive and caregiving_type == 0.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care_demand from baseline, attach to merged.
          6) Filter by age at first care demand period (if age_min/age_max specified).
          7) Aggregate employment rates by distance (baseline and
          counterfactual separately).
          8) Plot both series on same graph.

        Args:
            age_min: Minimum age at first care demand period (inclusive).
                If None, no lower bound.
            age_max: Maximum age at first care demand period (inclusive).
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

        # Filter to caregiving_type == 0 (agents who cannot provide informal care)
        if "caregiving_type" not in df_o.columns:
            raise ValueError(
                "caregiving_type column not found in data. "
                "Cannot filter to type 0 agents."
            )
        type_0_agents = df_o[df_o["caregiving_type"] == 0]["agent"].unique()
        df_o = df_o[df_o["agent"].isin(type_0_agents)].copy()
        df_c = df_c[df_c["agent"].isin(type_0_agents)].copy()

        # Calculate employment outcomes
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")

        # Create outcome columns
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        # Add current caregiving indicator (1 if currently providing informal
        # care, 0 otherwise)
        # Note: For type 0 agents, this will always be 0 since they cannot
        # provide informal care
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work

        # Merge on (agent, period) to ensure matched comparison
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

        # Add care_demand column to merged for duration identification
        # (needed for care_demand duration identification)
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to first care demand in baseline and attach
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )

        # Get age at first care demand period for filtering
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

        # Filter to agents with valid first care demand period
        # and trim to window
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]

        # Filter by age at first care demand period if specified
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        # Aggregate employment rates by distance
        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        # Rename column to match plotting function expectation
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Identify agents by duration of care demand
        # For type_0 version: use care_demand duration to include type_0 agents
        agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
            _identify_agents_by_duration(
                merged,
                distance_col="distance_to_first_care_demand",
                duration_type="care_demand",
            )
        )

        # Create conditional series for 1-year caregivers
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_1_year = prof_1_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 2-year caregivers
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_2_year = prof_2_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 3-year caregivers
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_3_year = prof_3_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 4-year caregivers
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
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
            xlabel="Year relative to start of first care demand",
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication_employment
    # @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_all_exact_care_demand")
    def task_plot_employment_rate_by_distance_to_first_care_demand_all(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / (
            f"employment_rate_by_distance_to_first_care_demand_all_"
            f"exact_care_demand_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to first care demand (all agents).

        Sample: Same agents in baseline and no-care-demand; from baseline (alive, optional
        ever_caregivers/ever_care_demand); merge on (agent, period). Event: t=0 = first
        care demand. Duration: 1–4 year = exact consecutive care demand (then stop).
        Can be filtered by age at first care demand period.

        Steps:
          1) Restrict to alive agents.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care_demand from baseline, attach to merged.
          6) Filter by age at first care demand period (if age_min/age_max specified).
          7) Aggregate employment rates by distance (baseline and
          counterfactual separately).
          8) Plot both series on same graph.

        Args:
            age_min: Minimum age at first care demand period (inclusive).
                If None, no lower bound.
            age_max: Maximum age at first care demand period (inclusive).
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
        # Add current caregiving indicator (1 if currently providing informal care
        # - light or intensive, 0 otherwise)
        # Note: "caregiving" refers to informal care, not formal care
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work

        # Merge on (agent, period) to ensure matched comparison
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

        # Add care_demand column to merged for duration identification
        # (needed for care_demand duration identification)
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to first care demand in baseline and attach
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )

        # Get age at first care demand period for filtering
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

        # Filter to agents with valid first care demand period
        # and trim to window
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]

        # Filter by age at first care demand period if specified
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        # Aggregate employment rates by distance
        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        # Rename column to match plotting function expectation
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Identify agents by duration of care demand
        # For "all" version: use care_demand duration to include type_0 agents
        agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
            _identify_agents_by_duration(
                merged,
                distance_col="distance_to_first_care_demand",
                duration_type="care_demand",
            )
        )

        # Create conditional series for 1-year caregivers
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_1_year = prof_1_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 2-year caregivers
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_2_year = prof_2_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 3-year caregivers
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_3_year = prof_3_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 4-year caregivers
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
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
            xlabel="Year relative to start of first care demand",
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication_employment
    # @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_type_1_exact_caregiving")
    def task_plot_employment_rate_by_distance_to_first_care_demand_type_1_cg(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / (
            f"employment_rate_by_distance_to_first_care_demand_type_1_"
            f"exact_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
        caregiving_type_1: bool = True,
    ) -> None:
        """Plot employment rate by distance to first care demand (type 1, caregiving duration).

        Sample: Same agents in baseline and no-care-demand; from baseline (alive, optional
        ever_caregivers/ever_care_demand), optionally restricted to caregiving_type == 1;
        merge on (agent, period). Event: t=0 = first care demand. Duration: 1–4 year = exact
        consecutive informal caregiving (then stop).
        Can be filtered by age at first care demand period.

        Steps:
          1) Restrict to alive; if caregiving_type_1, restrict to caregiving_type == 1.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care_demand from baseline, attach to merged.
          6) Filter by age at first care demand period (if age_min/age_max specified).
          7) Aggregate employment rates by distance (baseline and
          counterfactual separately).
          8) Plot both series on same graph.

        Args:
            age_min: Minimum age at first care demand period (inclusive).
                If None, no lower bound.
            age_max: Maximum age at first care demand period (inclusive).
                If None, no upper bound.
            age_label: Label for age group (used in filename)
            path_to_original_data: Path to baseline simulated data
            path_to_no_care_demand_data: Path to no-care-demand counterfactual data
            path_to_plot: Path to save the plot (constructed from age_label)
            ever_caregivers: If True, filter to agents who ever provided care
            ever_care_demand: If True, filter to agents who ever experienced care demand
            window: Window size around event (e.g., 20 = -20 to +20 periods)
            caregiving_type_1: If True, restrict to agents with caregiving_type == 1.
                If False, restrict to same agents as in original data (no type filter).

        """
        # Load and prepare data
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )

        # Restrict to same agents in both datasets
        if caregiving_type_1:
            if "caregiving_type" not in df_o.columns:
                raise ValueError(
                    "caregiving_type column not found in data. "
                    "Cannot filter to type 1 agents."
                )
            agents_to_keep = df_o[df_o["caregiving_type"] == 1]["agent"].unique()
        else:
            agents_to_keep = df_o["agent"].unique()
        df_o = df_o[df_o["agent"].isin(agents_to_keep)].copy()
        df_c = df_c[df_c["agent"].isin(agents_to_keep)].copy()

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

        # Add care_demand column to merged for duration identification
        # (needed for care_demand duration identification)
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to first care demand in baseline and attach
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )

        # Get age at first care demand period for filtering
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

        # Filter to agents with valid first care demand period
        # and trim to window
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]

        # Filter by age at first care demand period if specified
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        # Aggregate employment rates by distance
        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        # Rename column to match plotting function expectation
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Identify agents by duration of informal caregiving
        # For type_1 caregiving duration version: use caregiving duration
        agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
            _identify_agents_by_duration(
                merged,
                distance_col="distance_to_first_care_demand",
                duration_type="caregiving",
            )
        )

        # Create conditional series for 1-year caregivers
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_1_year = prof_1_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 2-year caregivers
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_2_year = prof_2_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 3-year caregivers
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_3_year = prof_3_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 4-year caregivers
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
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
            xlabel="Year relative to start of first care demand",
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    #
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication_employment
    # @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_type_1_exact_care_demand")
    def task_plot_employment_rate_by_distance_to_first_care_demand_type_1_exact_care_demand(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / (
            f"employment_rate_by_distance_to_first_care_demand_type_1_"
            f"exact_care_demand_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
        caregiving_type_1: bool = False,
    ) -> None:
        """Plot employment rate by distance to first care demand (type 1, exact care demand duration).

        Sample: Same agents in baseline and no-care-demand; sample defined from baseline
        (alive, optional ever_caregivers/ever_care_demand), optionally restricted to
        caregiving_type == 1. Merge on (agent, period) keeps only matched person-periods.

        Event: t=0 = first period with care demand > 0 (x-axis: first care demand).
        Duration: 1–4 year lines = exact consecutive care demand (then stop).
        Can be filtered by age at first care demand period.

        Steps:
          1) Restrict to alive; if caregiving_type_1, restrict to caregiving_type == 1.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care_demand from baseline, attach to merged.
          6) Filter by age at first care demand period (if age_min/age_max specified).
          7) Aggregate employment rates by distance (baseline and
          counterfactual separately).
          8) Plot both series on same graph.

        Args:
            age_min: Minimum age at first care demand period (inclusive).
                If None, no lower bound.
            age_max: Maximum age at first care demand period (inclusive).
                If None, no upper bound.
            age_label: Label for age group (used in filename)
            path_to_original_data: Path to baseline simulated data
            path_to_no_care_demand_data: Path to no-care-demand counterfactual data
            path_to_plot: Path to save the plot (constructed from age_label)
            ever_caregivers: If True, filter to agents who ever provided care
            ever_care_demand: If True, filter to agents who ever experienced care demand
            window: Window size around event (e.g., 20 = -20 to +20 periods)
            caregiving_type_1: If True, restrict to agents with caregiving_type == 1.
                If False, restrict to same agents as in original data (no type filter).

        """
        # Load and prepare data
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )

        # Restrict to same agents in both datasets
        if caregiving_type_1:
            if "caregiving_type" not in df_o.columns:
                raise ValueError(
                    "caregiving_type column not found in data. "
                    "Cannot filter to type 1 agents."
                )
            agents_to_keep = df_o[df_o["caregiving_type"] == 1]["agent"].unique()
        else:
            agents_to_keep = df_o["agent"].unique()
        df_o = df_o[df_o["agent"].isin(agents_to_keep)].copy()
        df_c = df_c[df_c["agent"].isin(agents_to_keep)].copy()

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

        # Add care_demand column to merged for duration identification
        # (needed for care_demand duration identification)
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to first care demand in baseline and attach
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )

        # Get age at first care demand period for filtering
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

        # Filter to agents with valid first care demand period
        # and trim to window
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]

        # Filter by age at first care demand period if specified
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        # Aggregate employment rates by distance
        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        # Rename column to match plotting function expectation
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Identify agents by duration of care demand
        # For type_1 care_demand duration version: use care_demand duration
        agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
            _identify_agents_by_duration(
                merged,
                distance_col="distance_to_first_care_demand",
                duration_type="care_demand",
            )
        )

        # Create conditional series for 1-year caregivers
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_1_year = prof_1_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 2-year caregivers
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_2_year = prof_2_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 3-year caregivers
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_3_year = prof_3_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Create conditional series for 4-year caregivers
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
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
            xlabel="Year relative to start of first care demand",
        )


# ---------------------------------------------------------------------------
# publication_other_check: by distance to first CARE DEMAND, total care years 1–5+
# Data: estimated_params, no_care_demand (standard)
# ---------------------------------------------------------------------------

# Full-time share by distance to first care demand (standard data; total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_full_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_full_time")
    def task_plot_full_time_share_by_distance_to_first_care_demand_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"full_time_share_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot full-time share by distance to first care demand (ever caregivers, total care years 1–5+).

        Data: estimated_params, no_care_demand. Event: t=0 = first care demand.
        Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_ft, _ = calculate_simple_outcomes(df_o, "original")
        _, c_ft, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["full_time_o"] = o_ft.astype(float)
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["full_time_c"] = c_ft.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["full_time_o", "full_time_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[["full_time_o"]]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Rate",
            subgroup_labels=total_labels,
        )


# Part-time share by distance to first care demand (standard data; total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_part_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_part_time")
    def task_plot_part_time_share_by_distance_to_first_care_demand_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "part_time"
        / "total_caregiving_years"
        / (
            f"part_time_share_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot part-time share by distance to first care demand (ever caregivers, total care years 1–5+).

        Data: estimated_params, no_care_demand. Event: t=0 = first care demand.
        Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_pt = calculate_simple_outcomes(df_o, "original")
        _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["part_time_o"] = o_pt.astype(float)
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["part_time_c"] = c_pt.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["part_time_o", "part_time_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[["part_time_o"]]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            outcome_baseline="part_time_o",
            outcome_counterfactual="part_time_c",
            ylabel="Part-Time Rate",
            subgroup_labels=total_labels,
        )


# Weekly working hours by distance to first care demand (standard data; total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_working_hours
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_working_hours")
    def task_plot_working_hours_by_distance_to_first_care_demand_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"working_hours_weekly_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot weekly working hours by distance to first care demand (ever caregivers, total care years 1–5+).

        Data: estimated_params, no_care_demand. Event: t=0 = first care demand.
        Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        wh_o = (
            df_o["working_hours"].astype(float) / 52.0
            if "working_hours" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        wh_c = (
            df_c["working_hours"].astype(float) / 52.0
            if "working_hours" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["working_hours_weekly_o"] = wh_o.values
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["working_hours_weekly_c"] = wh_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["working_hours_weekly_o", "working_hours_weekly_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[
                    ["working_hours_weekly_o"]
                ]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            outcome_baseline="working_hours_weekly_o",
            outcome_counterfactual="working_hours_weekly_c",
            ylabel="Weekly Working Hours",
            ylim=None,
            subgroup_labels=total_labels,
        )


# Monthly gross labor income by distance to first care demand (standard data; total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_labor_income
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_labor_income")
    def task_plot_labor_income_by_distance_to_first_care_demand_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"monthly_gross_labor_income_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot monthly gross labor income by distance to first care demand (ever caregivers, total care years 1–5+).

        Data: estimated_params, no_care_demand. Event: t=0 = first care demand.
        Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        inc_o = (
            df_o["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        inc_c = (
            df_c["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["monthly_gross_labor_income_o"] = inc_o.values
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["monthly_gross_labor_income_c"] = inc_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["monthly_gross_labor_income_o", "monthly_gross_labor_income_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[
                    ["monthly_gross_labor_income_o"]
                ]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            outcome_baseline="monthly_gross_labor_income_o",
            outcome_counterfactual="monthly_gross_labor_income_c",
            ylabel="Monthly Gross Labor Income",
            ylim=None,
            subgroup_labels=total_labels,
        )


# Full-time share by distance to first care demand (ever caregivers, total care years 1–5+); data: back_to_Jan7
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_full_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_back_to_Jan7_full_time")
    def task_plot_full_time_share_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_full_time_share_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot full-time share by distance to first care demand (ever caregivers, total care years 1–5+).

        Event: t=0 = first care demand. Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_ft, _ = calculate_simple_outcomes(df_o, "original")
        _, c_ft, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["full_time_o"] = o_ft.astype(float)
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["full_time_c"] = c_ft.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["full_time_o", "full_time_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[["full_time_o"]]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Rate",
            subgroup_labels=total_labels,
        )


# Part-time share by distance to first care demand (ever caregivers, total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_part_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_back_to_Jan7_part_time")
    def task_plot_part_time_share_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "part_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_part_time_share_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot part-time share by distance to first care demand (ever caregivers, total care years 1–5+).

        Event: t=0 = first care demand. Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_pt = calculate_simple_outcomes(df_o, "original")
        _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["part_time_o"] = o_pt.astype(float)
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["part_time_c"] = c_pt.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["part_time_o", "part_time_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[["part_time_o"]]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            outcome_baseline="part_time_o",
            outcome_counterfactual="part_time_c",
            ylabel="Part-Time Rate",
            subgroup_labels=total_labels,
        )


# Weekly working hours by distance to first care demand (ever caregivers, total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_working_hours
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_back_to_Jan7_working_hours")
    def task_plot_working_hours_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_working_hours_weekly_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot weekly working hours by distance to first care demand (ever caregivers, total care years 1–5+).

        Event: t=0 = first care demand. Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        wh_o = (
            df_o["working_hours"].astype(float) / 52.0
            if "working_hours" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        wh_c = (
            df_c["working_hours"].astype(float) / 52.0
            if "working_hours" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["working_hours_weekly_o"] = wh_o.values
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["working_hours_weekly_c"] = wh_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["working_hours_weekly_o", "working_hours_weekly_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[
                    ["working_hours_weekly_o"]
                ]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            outcome_baseline="working_hours_weekly_o",
            outcome_counterfactual="working_hours_weekly_c",
            ylabel="Weekly Working Hours",
            ylim=None,
            subgroup_labels=total_labels,
        )


# Monthly gross labor income by distance to first care demand (ever caregivers, total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_labor_income
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_back_to_Jan7_labor_income")
    def task_plot_labor_income_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_monthly_gross_labor_income_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot monthly gross labor income by distance to first care demand (ever caregivers, total care years 1–5+).

        Event: t=0 = first care demand. Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        inc_o = (
            df_o["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        inc_c = (
            df_c["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["monthly_gross_labor_income_o"] = inc_o.values
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["monthly_gross_labor_income_c"] = inc_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["monthly_gross_labor_income_o", "monthly_gross_labor_income_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[
                    ["monthly_gross_labor_income_o"]
                ]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            outcome_baseline="monthly_gross_labor_income_o",
            outcome_counterfactual="monthly_gross_labor_income_c",
            ylabel="Monthly Gross Labor Income",
            ylim=None,
            subgroup_labels=total_labels,
        )


# ---------------------------------------------------------------------------
# publication_other_check: alternative outcomes by distance to FIRST CAREGIVING SPELL
# (ever_caregivers=True; same setup as employment by first caregiving spell)
# ---------------------------------------------------------------------------
# Data: estimated_params, no_care_demand (standard)
# ---------------------------------------------------------------------------

# Full-time share by distance to first caregiving spell (standard data; total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_full_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_full_time")
    def task_plot_full_time_share_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "full_time"
        / (
            f"full_time_share_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot full-time share by distance to first caregiving spell (ever caregivers, total care years 1–5+).

        Data: estimated_params, no_care_demand. Event: t=0 = first caregiving spell.
        Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_ft, _ = calculate_simple_outcomes(df_o, "original")
        _, c_ft, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["full_time_o"] = o_ft.astype(float)
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["full_time_c"] = c_ft.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["full_time_o", "full_time_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Rate",
            subgroup_labels=total_labels,
        )


# Part-time share by distance to first caregiving spell (standard data; total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_part_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_part_time")
    def task_plot_part_time_share_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "part_time"
        / (
            f"part_time_share_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot part-time share by distance to first caregiving spell (ever caregivers, total care years 1–5+).

        Data: estimated_params, no_care_demand. Event: t=0 = first caregiving spell.
        Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_pt = calculate_simple_outcomes(df_o, "original")
        _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["part_time_o"] = o_pt.astype(float)
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["part_time_c"] = c_pt.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["part_time_o", "part_time_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            outcome_baseline="part_time_o",
            outcome_counterfactual="part_time_c",
            ylabel="Part-Time Rate",
            subgroup_labels=total_labels,
        )


# Weekly working hours by distance to first caregiving spell (standard data; total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_working_hours
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_working_hours")
    def task_plot_working_hours_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "working_hours"
        / (
            f"working_hours_weekly_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot weekly working hours by distance to first caregiving spell (ever caregivers, total care years 1–5+).

        Data: estimated_params, no_care_demand. Event: t=0 = first caregiving spell.
        Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        wh_o = (
            df_o["working_hours"].astype(float) / 52.0
            if "working_hours" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        wh_c = (
            df_c["working_hours"].astype(float) / 52.0
            if "working_hours" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["working_hours_weekly_o"] = wh_o.values
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["working_hours_weekly_c"] = wh_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["working_hours_weekly_o", "working_hours_weekly_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            outcome_baseline="working_hours_weekly_o",
            outcome_counterfactual="working_hours_weekly_c",
            ylabel="Weekly Working Hours",
            ylim=None,
            subgroup_labels=total_labels,
        )


# Monthly gross labor income by distance to first caregiving spell (standard data; total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_labor_income
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_labor_income")
    def task_plot_labor_income_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "labor_income"
        / (
            f"monthly_gross_labor_income_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot monthly gross labor income by distance to first caregiving spell (ever caregivers, total care years 1–5+).

        Data: estimated_params, no_care_demand. Event: t=0 = first caregiving spell.
        Duration: 1,2,3,4,5+ = total caregiving years over lifecycle.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        inc_o = (
            df_o["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        inc_c = (
            df_c["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["monthly_gross_labor_income_o"] = inc_o.values
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["monthly_gross_labor_income_c"] = inc_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["monthly_gross_labor_income_o", "monthly_gross_labor_income_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            outcome_baseline="monthly_gross_labor_income_o",
            outcome_counterfactual="monthly_gross_labor_income_c",
            ylabel="Monthly Gross Labor Income",
            ylim=None,
            subgroup_labels=total_labels,
        )


# Full-time share by distance to first caregiving spell (data: back_to_Jan7; total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_full_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_back_to_Jan7_full_time")
    def task_plot_full_time_share_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "full_time"
        / (
            f"back_to_Jan7_full_time_share_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot full-time share by distance to first caregiving spell (ever caregivers, total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_ft, _ = calculate_simple_outcomes(df_o, "original")
        _, c_ft, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["full_time_o"] = o_ft.astype(float)
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["full_time_c"] = c_ft.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["full_time_o", "full_time_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["full_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Rate",
            subgroup_labels=total_labels,
        )


# Part-time share by distance to first caregiving spell (ever caregivers, total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_part_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_back_to_Jan7_part_time")
    def task_plot_part_time_share_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "part_time"
        / (
            f"back_to_Jan7_part_time_share_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot part-time share by distance to first caregiving spell (ever caregivers, total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_pt = calculate_simple_outcomes(df_o, "original")
        _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["part_time_o"] = o_pt.astype(float)
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["part_time_c"] = c_pt.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["part_time_o", "part_time_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["part_time_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            outcome_baseline="part_time_o",
            outcome_counterfactual="part_time_c",
            ylabel="Part-Time Rate",
            subgroup_labels=total_labels,
        )


# Weekly working hours by distance to first caregiving spell (ever caregivers, total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_working_hours
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_back_to_Jan7_working_hours")
    def task_plot_working_hours_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "working_hours"
        / (
            f"back_to_Jan7_working_hours_weekly_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot weekly working hours by distance to first caregiving spell (ever caregivers, total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        wh_o = (
            df_o["working_hours"].astype(float) / 52.0
            if "working_hours" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        wh_c = (
            df_c["working_hours"].astype(float) / 52.0
            if "working_hours" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["working_hours_weekly_o"] = wh_o.values
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["working_hours_weekly_c"] = wh_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["working_hours_weekly_o", "working_hours_weekly_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["working_hours_weekly_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            outcome_baseline="working_hours_weekly_o",
            outcome_counterfactual="working_hours_weekly_c",
            ylabel="Weekly Working Hours",
            ylim=None,
            subgroup_labels=total_labels,
        )


# Monthly gross labor income by distance to first caregiving spell (ever caregivers, total care years 1–5+)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_labor_income
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_back_to_Jan7_labor_income")
    def task_plot_labor_income_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "labor_income"
        / (
            f"back_to_Jan7_monthly_gross_labor_income_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot monthly gross labor income by distance to first caregiving spell (ever caregivers, total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        inc_o = (
            df_o["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        inc_c = (
            df_c["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["monthly_gross_labor_income_o"] = inc_o.values
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["monthly_gross_labor_income_c"] = inc_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["monthly_gross_labor_income_o", "monthly_gross_labor_income_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["monthly_gross_labor_income_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            outcome_baseline="monthly_gross_labor_income_o",
            outcome_counterfactual="monthly_gross_labor_income_c",
            ylabel="Monthly Gross Labor Income",
            ylim=None,
            subgroup_labels=total_labels,
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    # @pytask.mark.publication_counterfactual
    @pytask.mark.publication_employment
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_at_least_caregiving")
    def task_plot_employment_rate_by_distance_to_first_care_at_least(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / (
            f"employment_rate_by_distance_to_first_caregiving_spell_"
            f"at_least_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to first caregiving spell (at least N years).

        Sample: Same agents in baseline and no-care-demand; from baseline (alive, optional
        ever_caregivers/ever_care_demand); merge on (agent, period). Event: t=0 = first
        period providing informal care. Duration: 1–4 = at least 1, 2, 3, 4 years
        caregiving (overlapping groups).

        Homogeneous groups are based on AT LEAST N years of caregiving:
        - At least 1-year: care at t=0
        - At least 2-year: care at t=0 and t=1
        - At least 3-year: care at t=0, t=1, t=2
        - At least 4-year: care at t=0, t=1, t=2, t=3

        Groups overlap (e.g., 4-year agents also appear in 3-year, 2-year, 1-year).

        Can be filtered by age at first care period.

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
        df_o_dist = add_distance_to_first_care(df_o)
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
        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Identify agents by AT LEAST N years of caregiving duration
        agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
            identify_agents_by_duration_at_least(
                merged,
                distance_col="distance_to_first_care",
                duration_type="caregiving",
            )
        )

        # Create conditional series for at least 1-year caregivers
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for at least 2-year caregivers
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for at least 3-year caregivers
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for at least 4-year caregivers
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


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    # @pytask.mark.publication_employment
    # @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_exact_caregiving")
    def task_plot_employment_rate_by_distance_to_first_care(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / (
            f"employment_rate_by_distance_to_first_caregiving_spell_"
            f"exact_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot overall employment rate by distance to first caregiving spell (exact caregiving).

        Sample: Same agents in baseline and no-care-demand; from baseline (alive, optional
        ever_caregivers/ever_care_demand); merge on (agent, period). Event: t=0 = first
        period providing informal care. Duration: 1–4 year = exact consecutive caregiving
        (then stop). Can be filtered by age at first care period.

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
        df_o_dist = add_distance_to_first_care(df_o)
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

        # Identify agents by caregiving duration (informal care)
        # For "caregiving spell" version: use current_caregiving to identify duration
        agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
            _identify_agents_by_duration(
                merged,
                distance_col="distance_to_first_care",
                duration_type="caregiving",
            )
        )

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


# ---------------------------------------------------------------------------
# Tasks: distance to first care demand + TOTAL caregiving years (1,2,3,4,5+) over lifecycle
# Set 1 (base): original = estimated_params, no_care = no_care_demand (no suffix/prefix)
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_employment
    @pytask.mark.publication_employment_check
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving")
    def task_plot_employment_rate_by_distance_to_first_care_demand_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / "total_caregiving_years"
        / (
            f"employment_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to first care demand, grouped by total caregiving years.

        Sample restricted to ever caregivers (same as 1–5+ total care years). Data: estimated_params, no_care_demand.
        Event: t=0 = first care demand. Duration: 1,2,3,4,5+ = TOTAL caregiving years over
        lifecycle (up to end_age_caregiving from specs), not necessarily consecutive.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[["work_o"]]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            subgroup_labels=total_labels,
        )


# ---------------------------------------------------------------------------
# Set 2 (Jan7): original = estimated_params_Jan7, no_care = no_care_demand_back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication_employment
    # @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_Jan7")
    def task_plot_employment_rate_by_distance_to_first_care_demand_total_caregiving_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / "total_caregiving_years"
        / (
            f"Jan7_employment_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to first care demand, grouped by total caregiving years.

        Sample restricted to ever caregivers. Data: estimated_params_Jan7, no_care_demand_back_to_Jan7.
        Event: t=0 = first care demand. Duration: 1,2,3,4,5+ = TOTAL caregiving years over
        lifecycle (up to end_age_caregiving from specs), not necessarily consecutive.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[["work_o"]]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            subgroup_labels=total_labels,
        )


# ---------------------------------------------------------------------------
# Tasks: distance to first caregiving spell + TOTAL caregiving years (1,2,3,4,5+) over lifecycle
# Set 1 (base): original = estimated_params, no_care = no_care_demand (no suffix/prefix)
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_employment
    @pytask.mark.publication_employment_check
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving")
    def task_plot_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / "total_caregiving_years"
        / (
            f"employment_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to first caregiving spell, grouped by total caregiving years.

        Sample restricted to ever caregivers. Data: estimated_params, no_care_demand.
        Event: t=0 = first caregiving spell. Duration: 1,2,3,4,5+ = TOTAL caregiving years
        over lifecycle (up to end_age_caregiving from specs), not necessarily consecutive.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = (
            merged["period"] - merged["first_care_period"]
        )
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            subgroup_labels=total_labels,
        )


# ---------------------------------------------------------------------------
# Set 2 (Jan7): original = estimated_params_Jan7, no_care = no_care_demand_back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication_employment
    # @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_Jan7")
    def task_plot_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / "total_caregiving_years"
        / (
            f"Jan7_employment_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to first caregiving spell, grouped by total caregiving years.

        Sample restricted to ever caregivers. Data: estimated_params_Jan7, no_care_demand_back_to_Jan7.
        Event: t=0 = first caregiving spell. Duration: 1,2,3,4,5+ = TOTAL caregiving years
        over lifecycle (up to end_age_caregiving from specs), not necessarily consecutive.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = (
            merged["period"] - merged["first_care_period"]
        )
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            subgroup_labels=total_labels,
        )


# ---------------------------------------------------------------------------
# Set 3 (back_to_Jan7): original = estimated_params_back_to_Jan7, no_care = no_care_demand_back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_employment
    @pytask.mark.publication_employment_check
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_back_to_Jan7")
    def task_plot_employment_rate_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_employment_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to first care demand, grouped by total caregiving years.

        Sample restricted to ever caregivers. Data: estimated_params_back_to_Jan7, no_care_demand_back_to_Jan7.
        Event: t=0 = first care demand. Duration: 1,2,3,4,5+ = TOTAL caregiving years over
        lifecycle (up to end_age_caregiving from specs), not necessarily consecutive.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care_demand"] = (
            merged["period"] - merged["first_care_demand_period"]
        )
        care_demand_mask = df_o["care_demand"] > 0
        first_care_demand_with_age = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
        merged = merged.merge(first_care_demand_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_demand_period"].notna()
            & (merged["distance_to_first_care_demand"] >= -window)
            & (merged["distance_to_first_care_demand"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care_demand"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care_demand"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof = prof.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        def _prof_for_agents(agents):
            m = merged[merged["agent"].isin(agents)].copy()
            p = (
                m.groupby("distance_to_first_care_demand", observed=False)[["work_o"]]
                .mean()
                .reset_index()
                .sort_values("distance_to_first_care_demand")
            )
            return p.rename(
                columns={"distance_to_first_care_demand": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1_year)
        prof_2_year = _prof_for_agents(agents_2_year)
        prof_3_year = _prof_for_agents(agents_3_year)
        prof_4_year = _prof_for_agents(agents_4_year)
        prof_5_year = _prof_for_agents(agents_5_year)

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            subgroup_labels=total_labels,
        )


# ---------------------------------------------------------------------------
# Set 3 (back_to_Jan7): original = estimated_params_back_to_Jan7, no_care = no_care_demand_back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_employment
    @pytask.mark.publication_employment_check
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_caregiving_spell_total_caregiving_back_to_Jan7")
    def task_plot_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "employment"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_employment_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to first caregiving spell, grouped by total caregiving years.

        Sample restricted to ever caregivers. Data: estimated_params_back_to_Jan7, no_care_demand_back_to_Jan7.
        Event: t=0 = first caregiving spell. Duration: 1,2,3,4,5+ = TOTAL caregiving years
        over lifecycle (up to end_age_caregiving from specs), not necessarily consecutive.
        """
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])

        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        df_o_dist = add_distance_to_first_care(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_care_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_first_care"] = (
            merged["period"] - merged["first_care_period"]
        )
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_care_with_age = get_age_at_first_event(
            df_o, caregiving_mask, "age_at_first_care"
        )
        merged = merged.merge(first_care_with_age, on="agent", how="left")
        merged = merged[
            merged["first_care_period"].notna()
            & (merged["distance_to_first_care"] >= -window)
            & (merged["distance_to_first_care"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_first_care"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_first_care"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_first_care", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        (
            agents_1_year,
            agents_2_year,
            agents_3_year,
            agents_4_year,
            agents_5_year,
        ) = identify_agents_by_total_caregiving_over_lifecycle(
            df_o, start_age, end_age_caregiving
        )

        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_first_care", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        total_labels = (
            "Baseline (1 total care year)",
            "Baseline (2 total care years)",
            "Baseline (3 total care years)",
            "Baseline (4 total care years)",
            "Baseline (5+ total care years)",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            subgroup_labels=total_labels,
        )


def plot_employment_rate_by_distance(  # noqa: PLR0913
    prof,
    prof_1_year,
    prof_2_year,
    prof_3_year,
    prof_4_year,
    prof_5_year=None,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
    xlabel: str = "Year relative to start of first care spell",
    outcome_baseline: str = "work_o",
    outcome_counterfactual: str = "work_c",
    ylabel: str = "Employment Rate",
    ylim: tuple[float, float] | None = (-0.025, 1.0),
    subgroup_labels: Optional[tuple[str, ...]] = None,
) -> None:
    """Plot employment or full-time rate by distance to first care/care demand.

    Creates an event study plot comparing baseline vs no-care-demand rates,
    with separate lines for different caregiving/care-demand durations.

    Args:
        prof: DataFrame with distance_to_first_care and outcome columns
        prof_1_year: DataFrame for 1-year duration subgroup (has outcome_baseline col)
        prof_2_year: DataFrame for 2-year duration subgroup
        prof_3_year: DataFrame for 3-year duration subgroup
        prof_4_year: DataFrame for 4-year duration subgroup
        window: Window size around event (e.g., 20 = -20 to +20 periods)
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
        xlabel: Label for x-axis (default: "Year relative to start of first care spell")
        outcome_baseline: Column name in prof and prof_* for baseline (e.g. work_o, full_time_o)
        outcome_counterfactual: Column name in prof for no-care-demand (e.g. work_c, full_time_c)
        ylabel: Label for y-axis (e.g. "Employment Rate", "Full-Time Rate")
        ylim: Fixed (ymin, ymax) for y-axis. If None, y-axis scale is determined by data.
    """
    # Plot
    # Increased figure size to maintain visual balance with thinner lines/text
    plt.figure(figsize=(14, 8))

    # Plot overall baseline (entire baseline sample) - dashed black line
    plt.plot(
        prof["distance_to_first_care"],
        prof[outcome_baseline],
        label="Baseline",
        color="black",
        linewidth=2.0,
        linestyle="--",
        marker=None,
    )

    # Plot no-care-demand counterfactual
    plt.plot(
        prof["distance_to_first_care"],
        prof[outcome_counterfactual],
        label="No Care Demand",
        color="black",
        linewidth=2.0,
        linestyle="-",
        marker=None,
    )

    default_labels = (
        "Baseline (1-year exact consecutive caregiving spell)",
        "Baseline (2-year exact consecutive caregiving spells)",
        "Baseline (3-year exact consecutive caregiving spells)",
        "Baseline (4-year exact consecutive caregiving spells)",
    )
    labels = subgroup_labels if subgroup_labels is not None else default_labels

    # Plot baseline for 1-year duration subgroup
    if len(prof_1_year) > 0:
        plt.plot(
            prof_1_year["distance_to_first_care"],
            prof_1_year[outcome_baseline],
            label=labels[0],
            color="0.8",
            linewidth=2.0,
            linestyle="-",
            marker="8",  # Octagon
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot baseline for 2-year duration subgroup
    if len(prof_2_year) > 0:
        plt.plot(
            prof_2_year["distance_to_first_care"],
            prof_2_year[outcome_baseline],
            label=labels[1],
            color="0.6",
            linewidth=2.0,
            linestyle="-",
            marker="^",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot baseline for 3-year duration subgroup
    if len(prof_3_year) > 0:
        plt.plot(
            prof_3_year["distance_to_first_care"],
            prof_3_year[outcome_baseline],
            label=labels[2],
            color="0.4",
            linewidth=2.0,
            linestyle="-",
            marker="D",  # Diamond
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot baseline for 4-year duration subgroup
    if len(prof_4_year) > 0:
        plt.plot(
            prof_4_year["distance_to_first_care"],
            prof_4_year[outcome_baseline],
            label=labels[3],
            color="0.2",
            linewidth=2.0,
            linestyle="-",
            marker="s",  # Hollow square
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot baseline for 5-year (5+ total) duration subgroup if provided
    if prof_5_year is not None and len(prof_5_year) > 0 and len(labels) > 4:
        plt.plot(
            prof_5_year["distance_to_first_care"],
            prof_5_year[outcome_baseline],
            label=labels[4],
            color="black",
            linewidth=2.0,
            linestyle="-",
            marker="*",
            markersize=6,
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
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    # Add padding: x-axis extends beyond -window and window
    plt.xlim(-window - 0.5, window + 0.5)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])  # e.g. (0, 1) for shares
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
