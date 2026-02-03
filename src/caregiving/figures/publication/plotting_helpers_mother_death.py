"""Helper functions for plotting tasks related to mother's death.

This module contains helper functions for identifying agents and calculating
distances relative to mother's death (t=0 when mother dies).
"""

import numpy as np
import pandas as pd

from caregiving.counterfactual.plotting_helpers import ensure_agent_period
from caregiving.model.shared import PARENT_RECENTLY_DEAD


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
