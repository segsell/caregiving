"""Plot employment rate by distance to mother's death.

This module creates event study plots comparing baseline vs no-care-demand
employment rates, aligned by distance to mother's death (t=0).
The analysis is "reverse" - t=0 is when mother dies, and we examine
employment rates before and after death.
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
    _ensure_agent_period,
    calculate_simple_outcomes,
    prepare_dataframes_simple,
)
from caregiving.model.shared import (
    INFORMAL_CARE,
    PARENT_RECENTLY_DEAD,
)


def _add_distance_to_mother_death(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_mother_death column.

    Sets 0 as first time mother_dead == PARENT_RECENTLY_DEAD (mother dies).
    """
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = _ensure_agent_period(df)
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


def _identify_agents_by_caregiving_before_death(
    merged: pd.DataFrame,
    distance_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by caregiving duration BEFORE mother's death.

    Identifies agents who provide informal care for 1, 2, 3, or 4 years
    BEFORE mother's death (at t=-1, t=-2, t=-3, t=-4).

    Groups are mutually exclusive:
    - 1-year: care at t=-1, but NOT at t=-2
    - 2-year: care at t=-1 and t=-2, but NOT at t=-3
    - 3-year: care at t=-1, t=-2, t=-3, but NOT at t=-4
    - 4-year: care at t=-1, t=-2, t=-3, t=-4

    Args:
        merged: DataFrame with agent, distance, and current_caregiving columns
        distance_col: Name of distance column (e.g., "distance_to_mother_death")

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year)
        as numpy arrays of agent IDs
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

    for agent in agent_care_matrix.index:
        # Check caregiving at t=-1, t=-2, t=-3, t=-4
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

        # 4-year: care at t=-1, t=-2, t=-3, t=-4
        if care_at_minus_1 and care_at_minus_2 and care_at_minus_3 and care_at_minus_4:
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

    return (
        np.array(agents_1_year),
        np.array(agents_2_year),
        np.array(agents_3_year),
        np.array(agents_4_year),
    )


def _identify_agents_by_caregiving_before_death_at_least(
    merged: pd.DataFrame,
    distance_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by caregiving duration BEFORE mother's death (at least N years).

    Identifies agents who provide informal care for AT LEAST 1, 2, 3, or 4 years
    BEFORE mother's death (at t=-1, t=-2, t=-3, t=-4).

    Groups use "at least" logic with overlap allowed:
    - At least 1-year: care at t=-1
    - At least 2-year: care at t=-1 and t=-2
    - At least 3-year: care at t=-1, t=-2, t=-3
    - At least 4-year: care at t=-1, t=-2, t=-3, t=-4

    Args:
        merged: DataFrame with agent, distance, and current_caregiving columns
        distance_col: Name of distance column (e.g., "distance_to_mother_death")

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year)
        as numpy arrays of agent IDs
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

    for agent in agent_care_matrix.index:
        # Check caregiving at t=-1, t=-2, t=-3, t=-4
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

    return (
        np.array(agents_1_year),
        np.array(agents_2_year),
        np.array(agents_3_year),
        np.array(agents_4_year),
    )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death")
    def task_plot_employment_rate_by_distance_to_mother_death(  # noqa: PLR0912, PLR0915
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
        / "reverse_employment"
        / f"employment_rate_by_distance_to_mother_death_{age_label_val}.pdf",
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to mother's death.

        Creates an event study plot comparing baseline vs no-care-demand employment
        rates, where t=0 is when mother dies (mother_dead == PARENT_RECENTLY_DEAD).
        The analysis is "reverse" - we examine employment rates before and after
        mother's death.

        Homogeneous groups are based on caregiving duration BEFORE death:
        - 1-year: care at t=-1, but NOT at t=-2
        - 2-year: care at t=-1 and t=-2, but NOT at t=-3
        - 3-year: care at t=-1, t=-2, t=-3, but NOT at t=-4
        - 4-year: care at t=-1, t=-2, t=-3, t=-4

        Can be filtered by age at mother's death period.

        Steps:
          1) Restrict to alive agents.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_mother_death from baseline, attach to merged.
          6) Filter by age at mother's death period (if age_min/age_max specified).
          7) Aggregate employment rates by distance (baseline and
          counterfactual separately).
          8) Plot both series on same graph.

        Args:
            age_min: Minimum age at mother's death period (inclusive).
                If None, no lower bound.
            age_max: Maximum age at mother's death period (inclusive).
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
        # Add current caregiving indicator (1 if currently providing informal care,
        # 0 otherwise)
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work

        # Merge on (agent, period) to ensure matched comparison
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

        # Add mother_dead column to merged for distance calculation
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to mother's death in baseline and attach
        df_o_dist = _add_distance_to_mother_death(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_death_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_mother_death"] = (
            merged["period"] - merged["first_death_period"]
        )

        # Get age at mother's death period for filtering
        death_mask = df_o["mother_dead"] == PARENT_RECENTLY_DEAD
        first_death_with_age = (
            df_o.loc[death_mask, ["agent", "period", "age"]]
            .sort_values(["agent", "period"])
            .drop_duplicates("agent")
            .rename(columns={"period": "first_death_period", "age": "age_at_death"})
        )
        merged = merged.merge(
            first_death_with_age[["agent", "age_at_death"]], on="agent", how="left"
        )

        # Filter to agents with valid first death period
        # and trim to window
        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]

        # Filter by age at mother's death period if specified
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        # Aggregate employment rates by distance
        prof = (
            merged.groupby("distance_to_mother_death", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        # Rename column to match plotting function expectation
        prof = prof.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Identify agents by caregiving duration BEFORE death
        agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
            _identify_agents_by_caregiving_before_death(
                merged,
                distance_col="distance_to_mother_death",
            )
        )

        # Create conditional series for 1-year caregivers (before death)
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_1_year = prof_1_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for 2-year caregivers (before death)
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_2_year = prof_2_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for 3-year caregivers (before death)
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_3_year = prof_3_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for 4-year caregivers (before death)
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Call plotting function
        plot_employment_rate_by_distance_to_mother_death(
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

    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_at_least")
    def task_plot_employment_rate_by_distance_to_mother_death_at_least(  # noqa: PLR0912, PLR0915
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
        / "reverse_employment"
        / (
            f"employment_rate_by_distance_to_mother_death_at_least_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot employment rate by distance to mother's death (at least N years).

        Creates an event study plot comparing baseline vs no-care-demand employment
        rates, where t=0 is when mother dies (mother_dead == PARENT_RECENTLY_DEAD).
        The analysis is "reverse" - we examine employment rates before and after
        mother's death.

        Homogeneous groups are based on AT LEAST N years of caregiving BEFORE death:
        - At least 1-year: care at t=-1
        - At least 2-year: care at t=-1 and t=-2
        - At least 3-year: care at t=-1, t=-2, t=-3
        - At least 4-year: care at t=-1, t=-2, t=-3, t=-4

        Groups overlap (e.g., 4-year agents also appear in 3-year, 2-year, 1-year).

        Can be filtered by age at mother's death period.

        Steps:
          1) Restrict to alive agents.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_mother_death from baseline, attach to merged.
          6) Filter by age at mother's death period (if age_min/age_max specified).
          7) Aggregate employment rates by distance (baseline and
          counterfactual separately).
          8) Plot both series on same graph.

        Args:
            age_min: Minimum age at mother's death period (inclusive).
                If None, no lower bound.
            age_max: Maximum age at mother's death period (inclusive).
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
        # Add current caregiving indicator (1 if currently providing informal care,
        # 0 otherwise)
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work

        # Merge on (agent, period) to ensure matched comparison
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

        # Add mother_dead column to merged for distance calculation
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to mother's death in baseline and attach
        df_o_dist = _add_distance_to_mother_death(df_o)
        dist_map = (
            df_o_dist.groupby("agent", observed=False)["first_death_period"]
            .first()
            .reset_index()
        )
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_mother_death"] = (
            merged["period"] - merged["first_death_period"]
        )

        # Get age at mother's death period for filtering
        death_mask = df_o["mother_dead"] == PARENT_RECENTLY_DEAD
        first_death_with_age = (
            df_o.loc[death_mask, ["agent", "period", "age"]]
            .sort_values(["agent", "period"])
            .drop_duplicates("agent")
            .rename(columns={"period": "first_death_period", "age": "age_at_death"})
        )
        merged = merged.merge(
            first_death_with_age[["agent", "age_at_death"]], on="agent", how="left"
        )

        # Filter to agents with valid first death period
        # and trim to window
        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]

        # Filter by age at mother's death period if specified
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        # Aggregate employment rates by distance
        prof = (
            merged.groupby("distance_to_mother_death", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        # Rename column to match plotting function expectation
        prof = prof.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Identify agents by AT LEAST N years of caregiving BEFORE death
        agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
            _identify_agents_by_caregiving_before_death_at_least(
                merged,
                distance_col="distance_to_mother_death",
            )
        )

        # Create conditional series for at least 1-year caregivers (before death)
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year = (
            merged_1_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_1_year = prof_1_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for at least 2-year caregivers (before death)
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_2_year = prof_2_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for at least 3-year caregivers (before death)
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_3_year = prof_3_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for at least 4-year caregivers (before death)
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Call plotting function
        plot_employment_rate_by_distance_to_mother_death(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            window=window,
            path_to_plot=path_to_plot,
        )


def plot_employment_rate_by_distance_to_mother_death(  # noqa: PLR0913
    prof,
    prof_1_year,
    prof_2_year,
    prof_3_year,
    prof_4_year,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
) -> None:
    """Plot employment rate by distance to mother's death.

    Creates an event study plot comparing baseline vs no-care-demand employment
    rates, with separate lines for different caregiving durations before death.

    Args:
        prof: DataFrame with columns 'distance_to_first_care', 'work_o', 'work_c'
        prof_1_year: DataFrame for 1-year caregivers (before death)
        prof_2_year: DataFrame for 2-year caregivers (before death)
        prof_3_year: DataFrame for 3-year caregivers (before death)
        prof_4_year: DataFrame for 4-year caregivers (before death)
        window: Window size around event (e.g., 20 = -20 to +20 periods)
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
    """
    # Plot
    # Increased figure size to maintain visual balance with thinner lines/text
    plt.figure(figsize=(14, 8))

    # Plot overall baseline employment rate (entire baseline sample) - dashed black line
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_o"],
        label="Baseline",
        color="black",
        linewidth=2.0,
        linestyle="--",
        marker=None,
    )

    # Plot no-care-demand employment rate - solid black line
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_c"],
        label="No Care Demand",
        color="black",
        linewidth=2.0,
        linestyle="-",
        marker=None,
    )

    # Plot baseline employment rate for 1-year caregivers (care at t=-1, but NOT t=-2)
    if len(prof_1_year) > 0:
        plt.plot(
            prof_1_year["distance_to_first_care"],
            prof_1_year["work_o"],
            label="Baseline (1-Year Caregivers: t=-1)",
            color="0.8",
            linewidth=2.0,
            linestyle="-",
            marker="8",  # Octagon
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot baseline employment rate for 2-year caregivers
    # (care at t=-1, t=-2, but NOT t=-3)
    if len(prof_2_year) > 0:
        plt.plot(
            prof_2_year["distance_to_first_care"],
            prof_2_year["work_o"],
            label="Baseline (2-Year Caregivers: t=-1, t=-2)",
            color="0.6",
            linewidth=2.0,
            linestyle="-",
            marker="^",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot baseline employment rate for 3-year caregivers
    # (care at t=-1, t=-2, t=-3, but NOT t=-4)
    if len(prof_3_year) > 0:
        plt.plot(
            prof_3_year["distance_to_first_care"],
            prof_3_year["work_o"],
            label="Baseline (3-Year Caregivers: t=-1, t=-2, t=-3)",
            color="0.4",
            linewidth=2.0,
            linestyle="-",
            marker="D",  # Diamond
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot baseline employment rate for 4-year caregivers
    # (care at t=-1, t=-2, t=-3, t=-4)
    if len(prof_4_year) > 0:
        plt.plot(
            prof_4_year["distance_to_first_care"],
            prof_4_year["work_o"],
            label="Baseline (4-Year Caregivers: t=-1, t=-2, t=-3, t=-4)",
            color="0.2",
            linewidth=2.0,
            linestyle="-",
            marker="s",  # Hollow square
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Add vertical line at t=0 (mother's death)
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
    plt.xlabel("Year relative to mother's death", fontsize=14)
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
