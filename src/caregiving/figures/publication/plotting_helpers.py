"""Helper functions for publication plotting modules.

This module contains shared helper functions used across multiple
publication plotting task modules.
"""

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
