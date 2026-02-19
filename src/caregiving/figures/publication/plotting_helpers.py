"""Helper functions for publication plotting modules.

This module contains shared helper functions used across multiple
publication plotting task modules.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.counterfactual.plotting_utils import ensure_agent_period
from caregiving.model.shared import INFORMAL_CARE


def add_distance_to_first_care(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care column; 0 is first period providing informal care."""
    df = df_original.reset_index(drop=True)
    df = ensure_agent_period(df)
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


def add_distance_to_first_care_demand(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care_demand column.

    Sets 0 as first time care_demand > 0 (light or intensive care demand).
    """
    df = df_original.reset_index(drop=True)
    df = ensure_agent_period(df)
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


def identify_agents_by_duration(
    merged: pd.DataFrame,
    distance_col: str,
    duration_type: str = "care_demand",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by duration of care demand or caregiving (exact consecutive years).

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
    merged = merged.copy()
    if duration_type == "care_demand":
        merged["care_status"] = (merged["care_demand"] > 0).astype(int)
    elif duration_type == "caregiving":
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

    agent_care_matrix = merged[merged[distance_col] >= 0].pivot_table(
        index="agent",
        columns=distance_col,
        values="care_status",
        aggfunc="first",
    )

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


def identify_agents_by_duration_at_least(
    merged: pd.DataFrame,
    distance_col: str,
    duration_type: str = "caregiving",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by AT LEAST N years of caregiving duration.

    Identifies agents who provide informal care for AT LEAST 1, 2, 3, or 4 years.
    Groups use "at least" logic with overlap allowed:
    - At least 1-year: care at t=0
    - At least 2-year: care at t=0 and t=1
    - At least 3-year: care at t=0, t=1, t=2
    - At least 4-year: care at t=0, t=1, t=2, t=3

    Args:
        merged: DataFrame with agent, distance, and relevant columns
        distance_col: Name of distance column (e.g., "distance_to_first_care")
        duration_type: "caregiving" (only type supported for "at least" logic)

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year)
        as numpy arrays of agent IDs
    """
    if duration_type != "caregiving":
        raise ValueError(
            f"duration_type must be 'caregiving' for 'at least' logic, "
            f"got {duration_type}"
        )

    # Use current_caregiving (informal care) to identify duration
    if "current_caregiving" not in merged.columns:
        raise ValueError(
            "current_caregiving column not found. "
            "Cannot identify caregiving duration."
        )

    # Create pivot table of caregiving status by distance
    agent_care_matrix = merged[merged[distance_col] >= 0].pivot_table(
        index="agent",
        columns=distance_col,
        values="current_caregiving",
        aggfunc="first",
    )

    # Identify agents with at least N years of caregiving
    agents_1_year = []
    agents_2_year = []
    agents_3_year = []
    agents_4_year = []

    for agent in agent_care_matrix.index:
        # Check caregiving at t=0, t=1, t=2, t=3
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

        # At least 1-year: care at t=0
        if care_at_0:
            agents_1_year.append(agent)
        # At least 2-year: care at t=0 and t=1
        if care_at_0 and care_at_1:
            agents_2_year.append(agent)
        # At least 3-year: care at t=0, t=1, t=2
        if care_at_0 and care_at_1 and care_at_2:
            agents_3_year.append(agent)
        # At least 4-year: care at t=0, t=1, t=2, t=3
        if care_at_0 and care_at_1 and care_at_2 and care_at_3:
            agents_4_year.append(agent)

    return (
        np.array(agents_1_year),
        np.array(agents_2_year),
        np.array(agents_3_year),
        np.array(agents_4_year),
    )


def identify_agents_by_consecutive_duration(  # noqa: PLR0912
    merged: pd.DataFrame,
    distance_col: str,
    duration_type: str = "caregiving",
    last_group_at_least: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by CONSECUTIVE N years of caregiving, then stop.

    Identifies agents who provide informal care for N consecutive years
    starting at t=0, then stop (at least 1 year off):
    - 1-year consecutive: care at t=0, then NOT at t=1 (at least 1 year off)
    - 2-year consecutive: care at t=0 and t=1, then NOT at t=2 (at least 1 year off)
    - 3-year consecutive: care at t=0, t=1, t=2, then NOT at t=3 (at least 1 year off)
    - 4-year consecutive:
        - If last_group_at_least=True: care at t=0, t=1, t=2, t=3 (at least 4 years)
        - If last_group_at_least=False: care at t=0, t=1, t=2, t=3,
          then NOT at t=4, t=5 (exactly 4 years)

    Groups are mutually exclusive (no overlap).

    Args:
        merged: DataFrame with agent, distance, and relevant columns
        distance_col: Name of distance column (e.g., "distance_to_first_care")
        duration_type: "caregiving" (only type supported for consecutive logic)
        last_group_at_least: If True, last group (4-year) contains "at least 4 years".
            If False, last group contains "exactly 4 years" (then stop).

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year)
        as numpy arrays of agent IDs
    """
    if duration_type != "caregiving":
        raise ValueError(
            f"duration_type must be 'caregiving' for consecutive logic, "
            f"got {duration_type}"
        )

    # Use current_caregiving (informal care) to identify duration
    if "current_caregiving" not in merged.columns:
        raise ValueError(
            "current_caregiving column not found. "
            "Cannot identify caregiving duration."
        )

    # Create pivot table of caregiving status by distance
    agent_care_matrix = merged[merged[distance_col] >= 0].pivot_table(
        index="agent",
        columns=distance_col,
        values="current_caregiving",
        aggfunc="first",
    )

    # Identify 1-year consecutive: care at t=0 only, then stop (not at t=1)
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

    # Identify 2-year consecutive: care at t=0 and t=1, then stop (not at t=2)
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

    # Identify 3-year consecutive: care at t=0, t=1, t=2, then stop (not at t=3)
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

    # Identify 4-year consecutive
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

        if last_group_at_least:
            # At least 4 years: care at t=0, t=1, t=2, t=3 (no check for t=4)
            if care_at_0 and care_at_1 and care_at_2 and care_at_3:
                agents_4_year.append(agent)
        else:
            # Exactly 4 years: care at t=0, t=1, t=2, t=3, then NOT at t=4, t=5
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
    plt.figure(figsize=(14, 8))

    plt.plot(
        prof["distance_to_first_care"],
        prof[outcome_baseline],
        label="Baseline",
        color="black",
        linewidth=2.0,
        linestyle="--",
        marker=None,
    )
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

    if len(prof_1_year) > 0:
        plt.plot(
            prof_1_year["distance_to_first_care"],
            prof_1_year[outcome_baseline],
            label=labels[0],
            color="0.8",
            linewidth=2.0,
            linestyle="-",
            marker="8",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )
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
    if len(prof_3_year) > 0:
        plt.plot(
            prof_3_year["distance_to_first_care"],
            prof_3_year[outcome_baseline],
            label=labels[2],
            color="0.4",
            linewidth=2.0,
            linestyle="-",
            marker="D",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )
    if len(prof_4_year) > 0:
        plt.plot(
            prof_4_year["distance_to_first_care"],
            prof_4_year[outcome_baseline],
            label=labels[3],
            color="0.2",
            linewidth=2.0,
            linestyle="-",
            marker="s",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )
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

    plt.axvline(
        x=-0.5,
        color="k",
        linestyle=(0, (7, 7)),
        linewidth=1.0,
    )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlim(-window - 0.5, window + 0.5)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.grid(True, axis="y", alpha=0.3, linewidth=0.8)
    plt.xticks(range(-window, window + 1, 5), fontsize=12)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=8)
    plt.tight_layout()
    if path_to_plot:
        path_to_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_plot, dpi=1200, bbox_inches="tight")
    plt.close()
