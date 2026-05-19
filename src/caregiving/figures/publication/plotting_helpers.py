"""Helper functions for publication plotting modules.

This module contains shared helper functions used across multiple
publication plotting task modules.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.counterfactual.plotting_helpers import (
    PUBLICATION_PLOT_STYLE,
    publication_savefig,
)
from caregiving.counterfactual.plotting_utils import ensure_agent_period
from caregiving.model.shared import INFORMAL_CARE

MAX_DISTANCE_LABELS = 4


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
    """Identify agents by duration of care demand or caregiving.

    Uses exact consecutive years.

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
    window_low: int = 20,
    window_high: int = 20,
    path_to_plot: Path | None = None,
    xlabel: str = "Year relative to start of first care spell",
    outcome_baseline: str = "work_o",
    outcome_counterfactual: str = "work_c",
    ylabel: str = "Employment Rate",
    ylim: tuple[float, float] | None = (-0.025, 1.0),
    subgroup_labels: tuple[str, ...] | None = None,
    extra_vlines: Iterable[float] | None = None,
    style: dict | None = None,
    age_label: str | None = None,
    yticks: list[float] | None = None,
    plot_caregivers_mean: bool = True,
) -> None:
    """Plot employment or full-time rate by distance to first care/care demand.

    Creates a conditional-mean plot comparing baseline vs no-care-demand rates,
    with separate lines for different caregiving/care-demand durations.

    window_low and window_high are positive integers (years before/after t=0);
    internally window_low is negated for the axis range.

    When age_label == "ages_60_70", the 5+ total care years line is not drawn.
    When plot_caregivers_mean is False, the dashed average-caregiver line is omitted.
    """
    w_low = -window_low
    S = style if style is not None else PUBLICATION_PLOT_STYLE
    plt.figure(figsize=S["figsize"])

    if plot_caregivers_mean:
        plt.plot(
            prof["distance_to_first_care"],
            prof[outcome_baseline],
            label="Baseline",
            color="black",
            linewidth=S["linewidth"],
            linestyle="--",
            marker=None,
        )
    plt.plot(
        prof["distance_to_first_care"],
        prof[outcome_counterfactual],
        label="No Care Demand",
        color="black",
        linewidth=S["linewidth"],
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
            linewidth=S["linewidth"],
            linestyle="-",
            marker="8",
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )
    if len(prof_2_year) > 0:
        plt.plot(
            prof_2_year["distance_to_first_care"],
            prof_2_year[outcome_baseline],
            label=labels[1],
            color="0.6",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="^",
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )
    if len(prof_3_year) > 0:
        plt.plot(
            prof_3_year["distance_to_first_care"],
            prof_3_year[outcome_baseline],
            label=labels[2],
            color="0.4",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="D",
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )
    if len(prof_4_year) > 0:
        plt.plot(
            prof_4_year["distance_to_first_care"],
            prof_4_year[outcome_baseline],
            label=labels[3],
            color="0.2",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="s",
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )
    if (
        age_label != "ages_60_70"
        and prof_5_year is not None
        and len(prof_5_year) > 0
        and len(labels) > MAX_DISTANCE_LABELS
    ):
        plt.plot(
            prof_5_year["distance_to_first_care"],
            prof_5_year[outcome_baseline],
            label=labels[4],
            color="black",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="*",
            markersize=S.get("markersize_star", S["markersize"]),
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S.get("markeredgewidth_star", S["markeredgewidth"]),
        )

    plt.axvline(
        x=-0.5,
        color="k",
        linestyle=(0, (7, 7)),
        linewidth=S["axvline_linewidth"],
    )
    if extra_vlines is not None:
        for x in extra_vlines:
            plt.axvline(
                x=x,
                color="gray",
                linestyle=(0, (3, 3)),
                linewidth=0.8,
            )
    plt.xlabel(xlabel, fontsize=S["label_fontsize"], labelpad=S.get("labelpad", 14))
    plt.ylabel(ylabel, fontsize=S["label_fontsize"], labelpad=S.get("labelpad", 14))
    plt.xlim(w_low - 0.5, window_high + 0.5)
    if ylim is not None:
        pad = S.get("y_axis_margin_factor", 0.03) * (ylim[1] - ylim[0])
        plt.ylim(ylim[0] - pad, ylim[1])
    plt.grid(True, axis="y", alpha=S["grid_alpha"], linewidth=S["grid_linewidth"])
    plt.xticks(range(w_low, window_high + 1, 5), fontsize=S["xtick_fontsize"])
    if yticks is not None:
        plt.yticks(yticks, fontsize=S["ytick_fontsize"])
    else:
        plt.yticks(fontsize=S["ytick_fontsize"])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=S["tick_length"], width=S["tick_width"])
    plt.tight_layout()
    plt.subplots_adjust(
        bottom=S.get("subplots_adjust_bottom", 0.11),
        left=S.get("subplots_adjust_left", 0.14),
    )
    if path_to_plot:
        publication_savefig(path_to_plot)
    plt.close()
