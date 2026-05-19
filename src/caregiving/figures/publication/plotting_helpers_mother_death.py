"""Helper functions for plotting tasks related to mother's death.

This module contains helper functions for identifying agents and calculating
distances relative to mother's death (t=0 when mother dies).
"""

import numpy as np
import pandas as pd

from caregiving.counterfactual.plotting_helpers import ensure_agent_period
from caregiving.model.shared import PARENT_RECENTLY_DEAD

CARE_YEARS_1 = 1
CARE_YEARS_2 = 2
CARE_YEARS_3 = 3
CARE_YEARS_4 = 4
CARE_YEARS_5_PLUS = 5
DIST_AT_CARE_3 = -3
DIST_AT_CARE_5 = -5
DIST_AT_CARE_7 = -7
DIST_AT_CARE_10 = -10
DIST_AT_CARE_11_PLUS = -11
DIST_AT_CG_1_4_MIN = -4
DIST_AT_CG_1_4_MAX = -1
DIST_AT_CG_5_9_MIN = -9
DIST_AT_CG_5_9_MAX = -5
DIST_AT_CG_10_14_MIN = -14
DIST_AT_CG_10_14_MAX = -10
DIST_AT_CG_15_PLUS = -15


def add_distance_to_mother_death(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_mother_death column.

    Sets 0 as first time mother_dead == PARENT_RECENTLY_DEAD (mother dies).
    """
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = ensure_agent_period(df)
    # Find first period where mother_dead == PARENT_RECENTLY_DEAD
    death_mask = df["mother_dead"] == PARENT_RECENTLY_DEAD
    first_death = (
        df.loc[death_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_death_period"})
    )
    out = df.merge(first_death, on="agent", how="left")
    out["distance_to_mother_death"] = out["period"] - out["first_death_period"]
    return out


def identify_agents_by_caregiving_before_death(  # noqa: PLR0912, PLR0915
    merged: pd.DataFrame,
    distance_col: str,
    add_five_year: bool = False,
    last_group_at_least: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Identify agents by caregiving duration BEFORE mother's death.

    Identifies agents who provide informal care for 1, 2, 3, or 4 years
    BEFORE mother's death (at t=-1, t=-2, t=-3, t=-4).
    Optionally includes 5-year group (at t=-1, t=-2, t=-3, t=-4, t=-5).

    Groups are mutually exclusive:
    - 1-year: care at t=-1, but NOT at t=-2
    - 2-year: care at t=-1 and t=-2, but NOT at t=-3
    - 3-year: care at t=-1, t=-2, t=-3, but NOT at t=-4
    - 4-year:
        - If last_group_at_least=True: care at t=-1, t=-2, t=-3, t=-4
          (at least 4 years)
        - If last_group_at_least=False: care at t=-1, t=-2, t=-3, t=-4,
          but NOT at t=-5 (exactly 4 years)
    - 5-year: care at t=-1, t=-2, t=-3, t=-4, t=-5 (if add_five_year)

    Args:
        merged: DataFrame with agent, distance, and current_caregiving columns
        distance_col: Name of distance column (e.g., "distance_to_mother_death")
        add_five_year: If True, include 5-year group and exclude 5-year from 4-year
        last_group_at_least: If True, last group (4-year or 5-year) contains
            "at least N years". If False, last group contains "exactly N years"
            (then stop).

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year,
        agents_5_year) as numpy arrays of agent IDs. agents_5_year is None if
        add_five_year is False.
    """
    # Ensure current_caregiving column exists
    if "current_caregiving" not in merged.columns:
        raise ValueError(
            "current_caregiving column not found. "
            "Cannot identify caregiving duration before death."
        )

    # Create pivot table of caregiving status by distance (only negative distances)
    agent_care_matrix = merged[merged[distance_col] < 0].pivot_table(
        index="agent",
        columns=distance_col,
        values="current_caregiving",
        aggfunc="first",
    )

    # Identify agents with caregiving at specific distances before death
    agents_1_year = []
    agents_2_year = []
    agents_3_year = []
    agents_4_year = []
    agents_5_year = [] if add_five_year else None

    for agent in agent_care_matrix.index:
        # Check caregiving at t=-1, t=-2, t=-3, t=-4, t=-5
        care_at_minus_1 = (
            agent_care_matrix.loc[agent, -1] == 1
            if -1 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, -1])
            else False
        )
        care_at_minus_2 = (
            agent_care_matrix.loc[agent, -2] == 1
            if -2 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -2])
            else False
        )
        care_at_minus_3 = (
            agent_care_matrix.loc[agent, -3] == 1
            if -3 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -3])
            else False
        )
        care_at_minus_4 = (
            agent_care_matrix.loc[agent, -4] == 1
            if -4 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -4])
            else False
        )
        # Always check t=-5 if it exists (needed to exclude 5+ year caregivers
        # from 4-year group)
        care_at_minus_5 = (
            agent_care_matrix.loc[agent, -5] == 1
            if -5 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -5])
            else False
        )

        if add_five_year:
            if last_group_at_least:
                # 5-year: care at t=-1, t=-2, t=-3, t=-4, t=-5 (at least 5 years)
                if (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and care_at_minus_5
                ):
                    agents_5_year.append(agent)
                # 4-year: care at t=-1, t=-2, t=-3, t=-4, but NOT at t=-5
                elif (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and not care_at_minus_5
                ):
                    agents_4_year.append(agent)
                # 3-year: care at t=-1, t=-2, t=-3, but NOT at t=-4
                elif (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and not care_at_minus_4
                ):
                    agents_3_year.append(agent)
                # 2-year: care at t=-1, t=-2, but NOT at t=-3
                elif care_at_minus_1 and care_at_minus_2 and not care_at_minus_3:
                    agents_2_year.append(agent)
                # 1-year: care at t=-1, but NOT at t=-2
                elif care_at_minus_1 and not care_at_minus_2:
                    agents_1_year.append(agent)
            else:
                # 5-year: care at t=-1, t=-2, t=-3, t=-4, t=-5,
                # but NOT at t=-6 (exactly 5 years)
                care_at_minus_6 = (
                    agent_care_matrix.loc[agent, -6] == 1
                    if -6 in agent_care_matrix.columns  # noqa: PLR2004
                    and pd.notna(agent_care_matrix.loc[agent, -6])
                    else False
                )
                if (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and care_at_minus_5
                    and not care_at_minus_6
                ):
                    agents_5_year.append(agent)
                # 4-year: care at t=-1, t=-2, t=-3, t=-4, but NOT at t=-5
                elif (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and not care_at_minus_5
                ):
                    agents_4_year.append(agent)
                # 3-year: care at t=-1, t=-2, t=-3, but NOT at t=-4
                elif (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and not care_at_minus_4
                ):
                    agents_3_year.append(agent)
                # 2-year: care at t=-1, t=-2, but NOT at t=-3
                elif care_at_minus_1 and care_at_minus_2 and not care_at_minus_3:
                    agents_2_year.append(agent)
                # 1-year: care at t=-1, but NOT at t=-2
                elif care_at_minus_1 and not care_at_minus_2:
                    agents_1_year.append(agent)
        else:
            if last_group_at_least:
                # 4-year: care at t=-1, t=-2, t=-3, t=-4
                # (at least 4 years, no check for t=-5)
                if (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                ):
                    agents_4_year.append(agent)
            else:
                # 4-year: care at t=-1, t=-2, t=-3, t=-4,
                # but NOT at t=-5 (exactly 4 years)
                if (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and not care_at_minus_5
                ):
                    agents_4_year.append(agent)
            # 3-year: care at t=-1, t=-2, t=-3, but NOT at t=-4
            if (
                care_at_minus_1
                and care_at_minus_2
                and care_at_minus_3
                and not care_at_minus_4
            ):
                agents_3_year.append(agent)
            # 2-year: care at t=-1, t=-2, but NOT at t=-3
            if care_at_minus_1 and care_at_minus_2 and not care_at_minus_3:
                agents_2_year.append(agent)
            # 1-year: care at t=-1, but NOT at t=-2
            if care_at_minus_1 and not care_at_minus_2:
                agents_1_year.append(agent)

    if add_five_year:
        return (
            np.array(agents_1_year),
            np.array(agents_2_year),
            np.array(agents_3_year),
            np.array(agents_4_year),
            np.array(agents_5_year),
        )
    return (
        np.array(agents_1_year),
        np.array(agents_2_year),
        np.array(agents_3_year),
        np.array(agents_4_year),
        None,
    )


def identify_agents_by_care_demand_before_death(  # noqa: PLR0912, PLR0915
    merged: pd.DataFrame,
    distance_col: str,
    add_five_year: bool = False,
    last_group_at_least: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Identify agents by care demand duration BEFORE mother's death.

    Identifies agents who experience care_demand > 0 for 1, 2, 3, or 4 years
    BEFORE mother's death (at t=-1, t=-2, t=-3, t=-4).
    Optionally includes 5-year group (at t=-1, t=-2, t=-3, t=-4, t=-5).

    Groups are mutually exclusive:
    - 1-year: care demand at t=-1, but NOT at t=-2
    - 2-year: care demand at t=-1 and t=-2, but NOT at t=-3
    - 3-year: care demand at t=-1, t=-2, t=-3, but NOT at t=-4
    - 4-year:
        - If last_group_at_least=True: care demand at t=-1, t=-2, t=-3, t=-4
          (at least 4 years)
        - If last_group_at_least=False: care demand at t=-1, t=-2, t=-3, t=-4,
          but NOT at t=-5 (exactly 4 years)
    - 5-year: care demand at t=-1, t=-2, t=-3, t=-4, t=-5 (if add_five_year)

    Args:
        merged: DataFrame with agent, distance, and care_demand columns
        distance_col: Name of distance column (e.g., "distance_to_mother_death")
        add_five_year: If True, include 5-year group and exclude 5-year from 4-year
        last_group_at_least: If True, last group (4-year or 5-year) contains
            "at least N years". If False, last group contains "exactly N years"
            (then stop).

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year,
        agents_5_year) as numpy arrays of agent IDs. agents_5_year is None if
        add_five_year is False.
    """
    # Ensure care_demand column exists
    if "care_demand" not in merged.columns:
        raise ValueError(
            "care_demand column not found. "
            "Cannot identify care demand duration before death."
        )

    # Create pivot table of care demand status by distance (only negative distances)
    # Use care_demand > 0 to identify duration
    merged["care_demand_status"] = (merged["care_demand"] > 0).astype(int)
    agent_care_matrix = merged[merged[distance_col] < 0].pivot_table(
        index="agent",
        columns=distance_col,
        values="care_demand_status",
        aggfunc="first",
    )

    # Identify agents with care demand at specific distances before death
    agents_1_year = []
    agents_2_year = []
    agents_3_year = []
    agents_4_year = []
    agents_5_year = [] if add_five_year else None

    for agent in agent_care_matrix.index:
        # Check care demand at t=-1, t=-2, t=-3, t=-4, t=-5
        care_at_minus_1 = (
            agent_care_matrix.loc[agent, -1] == 1
            if -1 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, -1])
            else False
        )
        care_at_minus_2 = (
            agent_care_matrix.loc[agent, -2] == 1
            if -2 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -2])
            else False
        )
        care_at_minus_3 = (
            agent_care_matrix.loc[agent, -3] == 1
            if -3 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -3])
            else False
        )
        care_at_minus_4 = (
            agent_care_matrix.loc[agent, -4] == 1
            if -4 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -4])
            else False
        )
        # Always check t=-5 if it exists (needed to exclude 5+ year from 4-year group)
        care_at_minus_5 = (
            agent_care_matrix.loc[agent, -5] == 1
            if -5 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -5])
            else False
        )

        if add_five_year:
            if last_group_at_least:
                # 5-year: care demand at t=-1, t=-2, t=-3, t=-4, t=-5 (at least 5 years)
                if (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and care_at_minus_5
                ):
                    agents_5_year.append(agent)
                # 4-year: care demand at t=-1, t=-2, t=-3, t=-4, but NOT at t=-5
                elif (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and not care_at_minus_5
                ):
                    agents_4_year.append(agent)
                # 3-year: care demand at t=-1, t=-2, t=-3, but NOT at t=-4
                elif (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and not care_at_minus_4
                ):
                    agents_3_year.append(agent)
                # 2-year: care demand at t=-1, t=-2, but NOT at t=-3
                elif care_at_minus_1 and care_at_minus_2 and not care_at_minus_3:
                    agents_2_year.append(agent)
                # 1-year: care demand at t=-1, but NOT at t=-2
                elif care_at_minus_1 and not care_at_minus_2:
                    agents_1_year.append(agent)
            else:
                # 5-year: care demand at t=-1, t=-2, t=-3, t=-4, t=-5,
                # but NOT at t=-6 (exactly 5 years)
                care_at_minus_6 = (
                    agent_care_matrix.loc[agent, -6] == 1
                    if -6 in agent_care_matrix.columns  # noqa: PLR2004
                    and pd.notna(agent_care_matrix.loc[agent, -6])
                    else False
                )
                if (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and care_at_minus_5
                    and not care_at_minus_6
                ):
                    agents_5_year.append(agent)
                # 4-year: care demand at t=-1, t=-2, t=-3, t=-4, but NOT at t=-5
                elif (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and not care_at_minus_5
                ):
                    agents_4_year.append(agent)
                # 3-year: care demand at t=-1, t=-2, t=-3, but NOT at t=-4
                elif (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and not care_at_minus_4
                ):
                    agents_3_year.append(agent)
                # 2-year: care demand at t=-1, t=-2, but NOT at t=-3
                elif care_at_minus_1 and care_at_minus_2 and not care_at_minus_3:
                    agents_2_year.append(agent)
                # 1-year: care demand at t=-1, but NOT at t=-2
                elif care_at_minus_1 and not care_at_minus_2:
                    agents_1_year.append(agent)
        else:
            if last_group_at_least:
                # 4-year: care demand at t=-1, t=-2, t=-3, t=-4
                # (at least 4 years, no check for t=-5)
                if (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                ):
                    agents_4_year.append(agent)
            else:
                # 4-year: care demand at t=-1, t=-2, t=-3, t=-4,
                # but NOT at t=-5 (exactly 4 years)
                if (
                    care_at_minus_1
                    and care_at_minus_2
                    and care_at_minus_3
                    and care_at_minus_4
                    and not care_at_minus_5
                ):
                    agents_4_year.append(agent)
            # 3-year: care demand at t=-1, t=-2, t=-3, but NOT at t=-4
            if (
                care_at_minus_1
                and care_at_minus_2
                and care_at_minus_3
                and not care_at_minus_4
            ):
                agents_3_year.append(agent)
            # 2-year: care demand at t=-1, t=-2, but NOT at t=-3
            if care_at_minus_1 and care_at_minus_2 and not care_at_minus_3:
                agents_2_year.append(agent)
            # 1-year: care demand at t=-1, but NOT at t=-2
            if care_at_minus_1 and not care_at_minus_2:
                agents_1_year.append(agent)

    if add_five_year:
        return (
            np.array(agents_1_year),
            np.array(agents_2_year),
            np.array(agents_3_year),
            np.array(agents_4_year),
            np.array(agents_5_year),
        )
    return (
        np.array(agents_1_year),
        np.array(agents_2_year),
        np.array(agents_3_year),
        np.array(agents_4_year),
        None,
    )


def identify_agents_by_care_demand_before_death_at_least(
    merged: pd.DataFrame,
    distance_col: str,
    add_five_year: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Identify agents by care demand duration BEFORE mother's death (at least N years).

    Identifies agents who experience care_demand > 0 for AT LEAST 1, 2, 3, or 4 years
    BEFORE mother's death (at t=-1, t=-2, t=-3, t=-4).
    Optionally includes 5-year group (at t=-1, t=-2, t=-3, t=-4, t=-5).

    Groups use "at least" logic with overlap allowed:
    - At least 1-year: care demand at t=-1
    - At least 2-year: care demand at t=-1 and t=-2
    - At least 3-year: care demand at t=-1, t=-2, t=-3
    - At least 4-year: care demand at t=-1, t=-2, t=-3, t=-4
    - At least 5-year: care demand at t=-1, t=-2, t=-3, t=-4, t=-5 (if add_five_year)

    Args:
        merged: DataFrame with agent, distance, and care_demand columns
        distance_col: Name of distance column (e.g., "distance_to_mother_death")
        add_five_year: If True, include 5-year group

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year,
        agents_5_year) as numpy arrays of agent IDs. agents_5_year is None if
        add_five_year is False.
    """
    # Ensure care_demand column exists
    if "care_demand" not in merged.columns:
        raise ValueError(
            "care_demand column not found. "
            "Cannot identify care demand duration before death."
        )

    # Create pivot table of care demand status by distance (only negative distances)
    # Use care_demand > 0 to identify duration
    merged["care_demand_status"] = (merged["care_demand"] > 0).astype(int)
    agent_care_matrix = merged[merged[distance_col] < 0].pivot_table(
        index="agent",
        columns=distance_col,
        values="care_demand_status",
        aggfunc="first",
    )

    # Identify agents with at least N years of care demand before death
    agents_1_year = []
    agents_2_year = []
    agents_3_year = []
    agents_4_year = []
    agents_5_year = [] if add_five_year else None

    for agent in agent_care_matrix.index:
        # Check care demand at t=-1, t=-2, t=-3, t=-4, t=-5
        care_at_minus_1 = (
            agent_care_matrix.loc[agent, -1] == 1
            if -1 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, -1])
            else False
        )
        care_at_minus_2 = (
            agent_care_matrix.loc[agent, -2] == 1
            if -2 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -2])
            else False
        )
        care_at_minus_3 = (
            agent_care_matrix.loc[agent, -3] == 1
            if -3 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -3])
            else False
        )
        care_at_minus_4 = (
            agent_care_matrix.loc[agent, -4] == 1
            if -4 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -4])
            else False
        )
        care_at_minus_5 = (
            agent_care_matrix.loc[agent, -5] == 1
            if add_five_year
            and -5 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -5])
            else False
        )

        # At least 1-year: care demand at t=-1
        if care_at_minus_1:
            agents_1_year.append(agent)
        # At least 2-year: care demand at t=-1 and t=-2
        if care_at_minus_1 and care_at_minus_2:
            agents_2_year.append(agent)
        # At least 3-year: care demand at t=-1, t=-2, t=-3
        if care_at_minus_1 and care_at_minus_2 and care_at_minus_3:
            agents_3_year.append(agent)
        # At least 4-year: care demand at t=-1, t=-2, t=-3, t=-4
        if care_at_minus_1 and care_at_minus_2 and care_at_minus_3 and care_at_minus_4:
            agents_4_year.append(agent)
        # At least 5-year: care demand at t=-1, t=-2, t=-3, t=-4, t=-5
        if (
            add_five_year
            and care_at_minus_1
            and care_at_minus_2
            and care_at_minus_3
            and care_at_minus_4
            and care_at_minus_5
        ):
            agents_5_year.append(agent)

    if add_five_year:
        return (
            np.array(agents_1_year),
            np.array(agents_2_year),
            np.array(agents_3_year),
            np.array(agents_4_year),
            np.array(agents_5_year),
        )
    return (
        np.array(agents_1_year),
        np.array(agents_2_year),
        np.array(agents_3_year),
        np.array(agents_4_year),
        None,
    )


def identify_agents_by_caregiving_before_death_at_least(
    merged: pd.DataFrame,
    distance_col: str,
    add_five_year: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Identify agents by caregiving duration BEFORE mother's death (at least N years).

    Identifies agents who provide informal care for AT LEAST 1, 2, 3, or 4 years
    BEFORE mother's death (at t=-1, t=-2, t=-3, t=-4).
    Optionally includes 5-year group (at t=-1, t=-2, t=-3, t=-4, t=-5).

    Groups use "at least" logic with overlap allowed:
    - At least 1-year: care at t=-1
    - At least 2-year: care at t=-1 and t=-2
    - At least 3-year: care at t=-1, t=-2, t=-3
    - At least 4-year: care at t=-1, t=-2, t=-3, t=-4
    - At least 5-year: care at t=-1, t=-2, t=-3, t=-4, t=-5 (if add_five_year)

    Args:
        merged: DataFrame with agent, distance, and current_caregiving columns
        distance_col: Name of distance column (e.g., "distance_to_mother_death")
        add_five_year: If True, include 5-year group

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year,
        agents_5_year) as numpy arrays of agent IDs. agents_5_year is None if
        add_five_year is False.
    """
    # Ensure current_caregiving column exists
    if "current_caregiving" not in merged.columns:
        raise ValueError(
            "current_caregiving column not found. "
            "Cannot identify caregiving duration before death."
        )

    # Create pivot table of caregiving status by distance (only negative distances)
    agent_care_matrix = merged[merged[distance_col] < 0].pivot_table(
        index="agent",
        columns=distance_col,
        values="current_caregiving",
        aggfunc="first",
    )

    # Identify agents with at least N years of caregiving before death
    agents_1_year = []
    agents_2_year = []
    agents_3_year = []
    agents_4_year = []
    agents_5_year = [] if add_five_year else None

    for agent in agent_care_matrix.index:
        # Check caregiving at t=-1, t=-2, t=-3, t=-4, t=-5
        care_at_minus_1 = (
            agent_care_matrix.loc[agent, -1] == 1
            if -1 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, -1])
            else False
        )
        care_at_minus_2 = (
            agent_care_matrix.loc[agent, -2] == 1
            if -2 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -2])
            else False
        )
        care_at_minus_3 = (
            agent_care_matrix.loc[agent, -3] == 1
            if -3 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -3])
            else False
        )
        care_at_minus_4 = (
            agent_care_matrix.loc[agent, -4] == 1
            if -4 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -4])
            else False
        )
        care_at_minus_5 = (
            agent_care_matrix.loc[agent, -5] == 1
            if add_five_year
            and -5 in agent_care_matrix.columns  # noqa: PLR2004
            and pd.notna(agent_care_matrix.loc[agent, -5])
            else False
        )

        # At least 1-year: care at t=-1
        if care_at_minus_1:
            agents_1_year.append(agent)
        # At least 2-year: care at t=-1 and t=-2
        if care_at_minus_1 and care_at_minus_2:
            agents_2_year.append(agent)
        # At least 3-year: care at t=-1, t=-2, t=-3
        if care_at_minus_1 and care_at_minus_2 and care_at_minus_3:
            agents_3_year.append(agent)
        # At least 4-year: care at t=-1, t=-2, t=-3, t=-4
        if care_at_minus_1 and care_at_minus_2 and care_at_minus_3 and care_at_minus_4:
            agents_4_year.append(agent)
        # At least 5-year: care at t=-1, t=-2, t=-3, t=-4, t=-5
        if (
            add_five_year
            and care_at_minus_1
            and care_at_minus_2
            and care_at_minus_3
            and care_at_minus_4
            and care_at_minus_5
        ):
            agents_5_year.append(agent)

    if add_five_year:
        return (
            np.array(agents_1_year),
            np.array(agents_2_year),
            np.array(agents_3_year),
            np.array(agents_4_year),
            np.array(agents_5_year),
        )
    return (
        np.array(agents_1_year),
        np.array(agents_2_year),
        np.array(agents_3_year),
        np.array(agents_4_year),
        None,
    )


def identify_agents_by_total_caregiving_before_death(
    merged: pd.DataFrame,
    distance_col: str,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by total (cumulative) caregiving years before mother's death.

    Counts how many periods with caregiving in the window before death
    (distance in [-window, -1]). Groups are mutually exclusive:
    - 1-year: exactly 1 total period with caregiving before death
    - 2-year: exactly 2 total periods
    - 3-year: exactly 3 total periods
    - 4-year: exactly 4 total periods
    - 5-year: 5 or more total periods (5+)

    Not consecutive: periods can be any time in the window (e.g. t=-1 and t=-10).

    Args:
        merged: DataFrame with agent, distance_col, and current_caregiving columns
        distance_col: Name of distance column (e.g., "distance_to_mother_death")
        window: Window size (e.g. 20); uses periods with distance in [-window, -1]

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year,
        agents_5_year) as numpy arrays of agent IDs. agents_5_year is 5+ total years.
    """
    if "current_caregiving" not in merged.columns:
        raise ValueError(
            "current_caregiving column not found. "
            "Cannot identify total caregiving before death."
        )
    before_death = merged[
        (merged[distance_col] >= -window) & (merged[distance_col] <= -1)
    ].copy()
    total_care = (
        before_death.groupby("agent", observed=False)["current_caregiving"]
        .sum()
        .astype(int)
    )
    agents_1_year = total_care[total_care == CARE_YEARS_1].index.to_numpy()
    agents_2_year = total_care[total_care == CARE_YEARS_2].index.to_numpy()
    agents_3_year = total_care[total_care == CARE_YEARS_3].index.to_numpy()
    agents_4_year = total_care[total_care == CARE_YEARS_4].index.to_numpy()
    agents_5_year = total_care[total_care >= CARE_YEARS_5_PLUS].index.to_numpy()
    return (
        agents_1_year,
        agents_2_year,
        agents_3_year,
        agents_4_year,
        agents_5_year,
    )


def identify_agents_by_first_care_demand_timing_before_death(
    df_o: pd.DataFrame,
    first_death_period_by_agent: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by when first care demand occurred relative to mother's death.

    Groups (mutually exclusive):
    - Group 1: first care demand 3 years before mother death (t=-3)
    - Group 2: first care demand 5 years before (t=-5)
    - Group 3: first care demand 7 years before (t=-7)
    - Group 4: first care demand 10 years before (t=-10)
    - Group 5: first care demand 11 years or longer before (t <= -11)

    Args:
        df_o: Baseline DataFrame with agent, period, care_demand columns
        first_death_period_by_agent: DataFrame with columns agent, first_death_period

    Returns:
        Tuple of (agents_3, agents_5, agents_7, agents_10, agents_11_plus) as numpy
        arrays of agent IDs.
    """
    if "care_demand" not in df_o.columns:
        raise ValueError(
            "care_demand column not found in data. "
            "Cannot identify first care demand timing."
        )
    first_care = (
        df_o[df_o["care_demand"] > 0]
        .groupby("agent", observed=False)["period"]
        .min()
        .reset_index()
        .rename(columns={"period": "first_care_demand_period"})
    )
    combined = first_care.merge(
        first_death_period_by_agent[["agent", "first_death_period"]],
        on="agent",
        how="inner",
    )
    combined["distance_at_first_care"] = (
        combined["first_care_demand_period"] - combined["first_death_period"]
    )
    agents_3 = combined[combined["distance_at_first_care"] == DIST_AT_CARE_3][
        "agent"
    ].to_numpy()
    agents_5 = combined[combined["distance_at_first_care"] == DIST_AT_CARE_5][
        "agent"
    ].to_numpy()
    agents_7 = combined[combined["distance_at_first_care"] == DIST_AT_CARE_7][
        "agent"
    ].to_numpy()
    agents_10 = combined[combined["distance_at_first_care"] == DIST_AT_CARE_10][
        "agent"
    ].to_numpy()
    agents_11_plus = combined[
        combined["distance_at_first_care"] <= DIST_AT_CARE_11_PLUS
    ]["agent"].to_numpy()
    return (agents_3, agents_5, agents_7, agents_10, agents_11_plus)


def identify_agents_by_first_caregiving_timing_before_death(
    df_o: pd.DataFrame,
    first_death_period_by_agent: pd.DataFrame,
    informal_care_choices: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by first caregiving spell relative to mother's death.

    Groups (mutually exclusive):
    - Group 1: first caregiving 1-4 years before mother death (distance in [-4, -1])
    - Group 2: first caregiving 5-9 years before (distance in [-9, -5])
    - Group 3: first caregiving 10-14 years before (distance in [-14, -10])
    - Group 4: first caregiving 15+ years before (distance <= -15)

    Args:
        df_o: Baseline DataFrame with agent, period, choice columns
        first_death_period_by_agent: DataFrame with columns agent, first_death_period
        informal_care_choices: List of choice codes that count as informal caregiving

    Returns:
        Tuple of (agents_1_4, agents_5_9, agents_10_14, agents_15_plus) as numpy arrays.
    """
    if "choice" not in df_o.columns:
        raise ValueError(
            "choice column not found in data. "
            "Cannot identify first caregiving spell."
        )
    first_care = (
        df_o[df_o["choice"].isin(informal_care_choices)]
        .groupby("agent", observed=False)["period"]
        .min()
        .reset_index()
        .rename(columns={"period": "first_caregiving_period"})
    )
    combined = first_care.merge(
        first_death_period_by_agent[["agent", "first_death_period"]],
        on="agent",
        how="inner",
    )
    combined["distance_at_first_caregiving"] = (
        combined["first_caregiving_period"] - combined["first_death_period"]
    )
    agents_1_4 = combined[
        (combined["distance_at_first_caregiving"] >= DIST_AT_CG_1_4_MIN)
        & (combined["distance_at_first_caregiving"] <= DIST_AT_CG_1_4_MAX)
    ]["agent"].to_numpy()
    agents_5_9 = combined[
        (combined["distance_at_first_caregiving"] >= DIST_AT_CG_5_9_MIN)
        & (combined["distance_at_first_caregiving"] <= DIST_AT_CG_5_9_MAX)
    ]["agent"].to_numpy()
    agents_10_14 = combined[
        (combined["distance_at_first_caregiving"] >= DIST_AT_CG_10_14_MIN)
        & (combined["distance_at_first_caregiving"] <= DIST_AT_CG_10_14_MAX)
    ]["agent"].to_numpy()
    agents_15_plus = combined[
        combined["distance_at_first_caregiving"] <= DIST_AT_CG_15_PLUS
    ]["agent"].to_numpy()
    return (agents_1_4, agents_5_9, agents_10_14, agents_15_plus)


def identify_agents_by_exact_caregiving_years_in_window(
    merged: pd.DataFrame,
    distance_col: str,
    window_start: int,
    window_end: int,
    include_5_plus: bool = False,
) -> tuple[np.ndarray, ...]:
    """Identify agents by exact number of caregiving years in a distance window.

    Counts periods with current_caregiving==1 where distance_col is in
    [window_start, window_end]. Returns mutually exclusive groups:
    exactly 1, 2, 3, 4 years; optionally a fifth group with >=5 years.

    Args:
        merged: DataFrame with agent, distance_col, and current_caregiving
        distance_col: Name of distance column (e.g. distance_to_mother_death)
        window_start: Inclusive lower bound (e.g. -4)
        window_end: Inclusive upper bound (e.g. -1)
        include_5_plus: If True, return fifth group (agents with >=5 years in window)

    Returns:
        (agents_1_year, agents_2_year, agents_3_year, agents_4_year) or
        (agents_1_year, ..., agents_5_plus) if include_5_plus.
    """
    if "current_caregiving" not in merged.columns:
        raise ValueError(
            "current_caregiving column not found. "
            "Cannot identify exact caregiving years in window."
        )
    in_window = merged[
        (merged[distance_col] >= window_start) & (merged[distance_col] <= window_end)
    ].copy()
    total_care = (
        in_window.groupby("agent", observed=False)["current_caregiving"]
        .sum()
        .astype(int)
    )
    agents_1 = total_care[total_care == CARE_YEARS_1].index.to_numpy()
    agents_2 = total_care[total_care == CARE_YEARS_2].index.to_numpy()
    agents_3 = total_care[total_care == CARE_YEARS_3].index.to_numpy()
    agents_4 = total_care[total_care == CARE_YEARS_4].index.to_numpy()
    if include_5_plus:
        agents_5_plus = total_care[total_care >= CARE_YEARS_5_PLUS].index.to_numpy()
        return (agents_1, agents_2, agents_3, agents_4, agents_5_plus)
    return (agents_1, agents_2, agents_3, agents_4)
