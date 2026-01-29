"""Plot overall caregiving rate by distance to first caregiving spell.

This module creates event study plots comparing baseline vs no-care-demand
caregiving rates (light or intensive informal care), aligned by distance to
first caregiving spell or first care demand (t=0).
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
    ensure_agent_period,
    get_age_at_first_event,
    prepare_dataframes_simple,
)
from caregiving.counterfactual.task_plot_labor_supply_differences import (
    _add_distance_to_first_care,
)
from caregiving.model.shared import (
    INFORMAL_CARE,
)


def _add_distance_to_first_care_demand(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care_demand column.

    Sets 0 as first time care_demand > 0 (light or intensive care demand).
    """
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = ensure_agent_period(df)
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


def _identify_agents_by_duration(
    merged: pd.DataFrame,
    distance_col: str,
    duration_type: str = "care_demand",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by duration of care demand or caregiving.

    For care_demand: Identifies agents who experience care_demand > 0 for
    at least 1, 2, 3, or 4 years (includes all agents, not just informal caregivers).
    For caregiving: Identifies agents who provide informal care for
    at least 1, 2, 3, or 4 years (only type_1 agents can provide informal care).

    Note: For caregiving plots, uses "at least" logic (no requirement for off year).
    Groups are mutually exclusive (4-year includes only those with >=4 years,
    3-year includes those with exactly 3 years, etc.).

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

    # Identify agents with at least N years of care (for caregiving plots)
    # All groups use "at least" logic with no stopping requirement
    # Overlap between groups is allowed (e.g., 4-year agents also appear in
    # 3-year, 2-year, 1-year)

    # Identify 4-year: at least 4 years (care at t=0, t=1, t=2, t=3)
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

        if care_at_0 and care_at_1 and care_at_2 and care_at_3:
            agents_4_year.append(agent)

    # Identify 3-year: at least 3 years (care at t=0, t=1, t=2)
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

        if care_at_0 and care_at_1 and care_at_2:
            agents_3_year.append(agent)

    # Identify 2-year: at least 2 years (care at t=0, t=1)
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

        if care_at_0 and care_at_1:
            agents_2_year.append(agent)

    # Identify 1-year: at least 1 year (care at t=0)
    agents_1_year = []
    for agent in agent_care_matrix.index:
        care_at_0 = (
            agent_care_matrix.loc[agent, 0] == 1
            if 0 in agent_care_matrix.columns
            and pd.notna(agent_care_matrix.loc[agent, 0])
            else False
        )

        if care_at_0:
            agents_1_year.append(agent)

    return (
        np.array(agents_1_year),
        np.array(agents_2_year),
        np.array(agents_3_year),
        np.array(agents_4_year),
    )


def calculate_caregiving_outcomes(
    df: pd.DataFrame, choice_set_type: str
) -> tuple[pd.Series, pd.Series]:
    """Calculate caregiving indicators from choice column.

    Args:
        df: DataFrame with 'choice' column
        choice_set_type: 'original' or 'no_care_demand'

    Returns:
        Tuple of (caregiving_o, caregiving_c) Series
        - caregiving_o: 1 if choice in INFORMAL_CARE, 0 otherwise
        - caregiving_c: 0 for no_care_demand (no care possible),
          same as caregiving_o for original
    """
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_o = df["choice"].isin(care_codes).astype(int)

    if choice_set_type == "no_care_demand":
        # No care demand = no caregiving possible
        caregiving_c = pd.Series(0, index=df.index, dtype=int)
    else:
        caregiving_c = caregiving_o.copy()

    return caregiving_o, caregiving_c


# Sanity check: Plot for caregiving_type == 0 (agents who cannot provide informal care)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_caregiving_rate
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_type_0")
    def task_plot_caregiving_rate_by_distance_to_first_care_demand_type_0(  # noqa: PLR0912, PLR0915
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
        / "caregiving"
        / (
            f"caregiving_rate_by_distance_to_first_care_demand_type_0_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot caregiving rates by distance to first care demand spell.

        Type 0 sanity check.

        Creates an event study plot comparing baseline vs no-care-demand caregiving
        rates, where t=0 is the start of the first care demand (light or intensive).
        Restricted to caregiving_type == 0 (agents who cannot provide informal care).
        This is a sanity check - type 0 agents cannot provide informal care, so
        the caregiver duration lines (1-year, 2-year, etc.) will be empty.
        Can be filtered by age at first care demand period.

        Steps:
          1) Restrict to alive and caregiving_type == 0.
          2) Ensure agent/period columns.
          3) Calculate caregiving outcomes (caregiving indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care_demand from baseline, attach to merged.
          6) Filter by age at first care demand period (if age_min/age_max specified).
          7) Aggregate caregiving rates by distance (baseline and
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

        # Calculate caregiving outcomes
        o_caregiving, _ = calculate_caregiving_outcomes(df_o, "original")
        _, c_caregiving = calculate_caregiving_outcomes(df_c, "no_care_demand")

        # Create outcome columns
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["caregiving_o"] = o_caregiving
        # Add current caregiving indicator (1 if currently providing informal
        # care, 0 otherwise)
        # Note: For type 0 agents, this will always be 0 since they cannot
        # provide informal care
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["caregiving_c"] = c_caregiving

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
        df_o_dist = _add_distance_to_first_care_demand(df_o)
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

        # Aggregate caregiving rates by distance
        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["caregiving_o", "caregiving_c"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Call plotting function
        plot_caregiving_rate_by_distance(
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

    @pytask.mark.publication_caregiving_rate
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_all")
    def task_plot_caregiving_rate_by_distance_to_first_care_demand_all(  # noqa: PLR0912, PLR0915
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
        / "caregiving"
        / f"caregiving_rate_by_distance_to_first_care_demand_all_{age_label_val}.pdf",
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot caregiving rates by distance to first care demand spell (all agents).

        Creates an event study plot comparing baseline vs no-care-demand caregiving
        rates, where t=0 is the start of the first care demand (light or intensive).
        Includes all agents (not restricted to caregiving_type == 1).
        Can be filtered by age at first care demand period.

        Steps:
          1) Restrict to alive agents.
          2) Ensure agent/period columns.
          3) Calculate caregiving outcomes (caregiving indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care_demand from baseline, attach to merged.
          6) Filter by age at first care demand period (if age_min/age_max specified).
          7) Aggregate caregiving rates by distance (baseline and
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

        # Calculate caregiving outcomes
        o_caregiving, _ = calculate_caregiving_outcomes(df_o, "original")
        _, c_caregiving = calculate_caregiving_outcomes(df_c, "no_care_demand")

        # Create outcome columns
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["caregiving_o"] = o_caregiving
        # Add current caregiving indicator (1 if currently providing informal care
        # - light or intensive, 0 otherwise)
        # Note: "caregiving" refers to informal care, not formal care
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["caregiving_c"] = c_caregiving

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
        df_o_dist = _add_distance_to_first_care_demand(df_o)
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

        # Aggregate caregiving rates by distance
        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["caregiving_o", "caregiving_c"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Call plotting function
        plot_caregiving_rate_by_distance(
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

    @pytask.mark.publication_caregiving_rate
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_type_1_caregiving_duration")
    def task_plot_caregiving_rate_by_distance_to_first_care_demand_type_1_cg(  # noqa: PLR0912, PLR0915
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
        / "caregiving"
        / (
            f"caregiving_rate_by_distance_to_first_care_demand_type_1_"
            f"caregiving_duration_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot caregiving rates by distance to first care demand spell.

        Caregiving duration version.

        Creates an event study plot comparing baseline vs no-care-demand caregiving
        rates, where t=0 is the start of the first care demand (light or intensive).
        Restricted to caregiving_type == 1 (agents who can provide informal
        care). Duration lines (1-year, 2-year, 3-year, 4-year) are based on
        informal caregiving duration.
        Can be filtered by age at first care demand period.

        Steps:
          1) Restrict to alive and caregiving_type == 1.
          2) Ensure agent/period columns.
          3) Calculate caregiving outcomes (caregiving indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care_demand from baseline, attach to merged.
          6) Filter by age at first care demand period (if age_min/age_max specified).
          7) Aggregate caregiving rates by distance (baseline and
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

        # Filter to caregiving_type == 1 (agents who can provide informal care)
        if "caregiving_type" not in df_o.columns:
            raise ValueError(
                "caregiving_type column not found in data. "
                "Cannot filter to type 1 agents."
            )
        type_1_agents = df_o[df_o["caregiving_type"] == 1]["agent"].unique()
        df_o = df_o[df_o["agent"].isin(type_1_agents)].copy()
        df_c = df_c[df_c["agent"].isin(type_1_agents)].copy()

        # Calculate caregiving outcomes
        o_caregiving, _ = calculate_caregiving_outcomes(df_o, "original")
        _, c_caregiving = calculate_caregiving_outcomes(df_c, "no_care_demand")

        # Create outcome columns
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["caregiving_o"] = o_caregiving
        # Add current caregiving indicator (1 if currently providing care, 0 otherwise)
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["caregiving_c"] = c_caregiving

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
        df_o_dist = _add_distance_to_first_care_demand(df_o)
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

        # Aggregate caregiving rates by distance
        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["caregiving_o", "caregiving_c"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Call plotting function
        plot_caregiving_rate_by_distance(
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

    @pytask.mark.publication_caregiving_rate
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_type_1_care_demand_duration")
    def task_plot_caregiving_rate_by_distance_to_first_care_demand_type_1_cd(  # noqa: PLR0912, PLR0915
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
        / "caregiving"
        / (
            f"caregiving_rate_by_distance_to_first_care_demand_type_1_"
            f"care_demand_duration_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot caregiving rates by distance to first care demand spell.

        Care demand duration version.

        Creates an event study plot comparing baseline vs no-care-demand caregiving
        rates, where t=0 is the start of the first care demand (light or intensive).
        Restricted to caregiving_type == 1 (agents who can provide informal
        care). Duration lines (1-year, 2-year, 3-year, 4-year) are based on
        care demand duration.
        Can be filtered by age at first care demand period.

        Steps:
          1) Restrict to alive and caregiving_type == 1.
          2) Ensure agent/period columns.
          3) Calculate caregiving outcomes (caregiving indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care_demand from baseline, attach to merged.
          6) Filter by age at first care demand period (if age_min/age_max specified).
          7) Aggregate caregiving rates by distance (baseline and
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

        # Filter to caregiving_type == 1 (agents who can provide informal care)
        if "caregiving_type" not in df_o.columns:
            raise ValueError(
                "caregiving_type column not found in data. "
                "Cannot filter to type 1 agents."
            )
        type_1_agents = df_o[df_o["caregiving_type"] == 1]["agent"].unique()
        df_o = df_o[df_o["agent"].isin(type_1_agents)].copy()
        df_c = df_c[df_c["agent"].isin(type_1_agents)].copy()

        # Calculate caregiving outcomes
        o_caregiving, _ = calculate_caregiving_outcomes(df_o, "original")
        _, c_caregiving = calculate_caregiving_outcomes(df_c, "no_care_demand")

        # Create outcome columns
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["caregiving_o"] = o_caregiving
        # Add current caregiving indicator (1 if currently providing care, 0 otherwise)
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["caregiving_c"] = c_caregiving

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
        df_o_dist = _add_distance_to_first_care_demand(df_o)
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

        # Aggregate caregiving rates by distance
        prof = (
            merged.groupby("distance_to_first_care_demand", observed=False)[
                ["caregiving_o", "caregiving_c"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
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
                ["caregiving_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care_demand")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_first_care_demand": "distance_to_first_care"}
        )

        # Call plotting function
        plot_caregiving_rate_by_distance(
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

    @pytask.mark.publication_caregiving_rate
    @pytask.mark.publication
    @pytask.task(id=age_label_val)
    def task_plot_caregiving_rate_by_distance_to_first_care(  # noqa: PLR0912, PLR0915
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
        / "caregiving"
        / f"caregiving_rate_by_distance_to_first_care_{age_label_val}.pdf",
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot overall caregiving rates by distance to first caregiving spell.

        Creates an event study plot comparing baseline vs no-care-demand employment
        rates, where t=0 is the start of the first caregiving spell. Can be filtered
        by age at first care period.

        Steps:
          1) Restrict to alive and (optionally) ever-caregivers/ever-care-demand.
          2) Ensure agent/period columns.
          3) Calculate caregiving outcomes (caregiving indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care from baseline, attach to merged.
          6) Filter by age at first care period (if age_min/age_max specified).
          7) Aggregate caregiving rates by distance (baseline and
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

        # Calculate caregiving outcomes
        o_caregiving, _ = calculate_caregiving_outcomes(df_o, "original")
        _, c_caregiving = calculate_caregiving_outcomes(df_c, "no_care_demand")

        # Create outcome columns
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["caregiving_o"] = o_caregiving
        # Add current caregiving indicator (1 if currently providing care, 0 otherwise)
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["caregiving_c"] = c_caregiving

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

        # Aggregate caregiving rates by distance
        # Group by distance and compute mean caregiving rates for each scenario
        #
        # BASELINE LINE COMPOSITION:
        # The baseline line includes ALL agents who:
        #   1. Ever provided care (have a valid first_care_period)
        #   2. Are within the window around their first care period
        #   3. Are matched on (agent, period) with the counterfactual
        #
        # At each distance, the baseline line is the average caregiving rates across:
        #   - People who are currently providing care at that distance
        #   - People who are not currently providing care at that distance
        #   - People who provided care for 1 year, 2 years, 3 years, 4 years, or longer
        #   - People who stopped and resumed care later
        #   - People who never stopped providing care
        #   - People who stopped after different durations
        #
        # This is the MOST INCLUSIVE group - it represents the average caregiving rates
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
                ["caregiving_o", "caregiving_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional baseline series:
        # 1. Baseline caregiving rates, conditioned on current_caregiving == 1,
        #    after t=0
        # COMMENTED OUT - not currently used in plotting
        # merged_care = merged[
        #     (merged["current_caregiving"] == 1)
        #     & (merged["distance_to_first_care"] >= 0)
        # ].copy()
        # prof_care = (
        #     merged_care.groupby(
        #         "distance_to_first_care", observed=False
        #     )[["caregiving_o"]]
        #     .mean()
        #     .reset_index()
        #     .sort_values("distance_to_first_care")
        # )

        # 2. Baseline caregiving rates, conditioned on current_caregiving == 0,
        #    after t=1
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
        #     )[["caregiving_o"]]
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
            merged_1_year.groupby("distance_to_first_care", observed=False)[
                ["caregiving_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for 2-year caregivers
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year = (
            merged_2_year.groupby("distance_to_first_care", observed=False)[
                ["caregiving_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for 3-year caregivers
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year = (
            merged_3_year.groupby("distance_to_first_care", observed=False)[
                ["caregiving_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for 4-year caregivers
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year = (
            merged_4_year.groupby("distance_to_first_care", observed=False)[
                ["caregiving_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Call plotting function
        plot_caregiving_rate_by_distance(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            window=window,
            path_to_plot=path_to_plot,
        )


def plot_caregiving_rate_by_distance(  # noqa: PLR0913
    prof,
    prof_1_year,
    prof_2_year,
    prof_3_year,
    prof_4_year,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
    xlabel: str = "Year relative to start of first care spell",
) -> None:
    """Plot caregiving rates by distance to first caregiving spell.

    Creates an event study plot comparing baseline vs no-care-demand caregiving
    rates, with separate lines for different caregiving durations.

    Args:
        prof: DataFrame with columns 'distance_to_first_care', 'caregiving_o',
            'caregiving_c'
        prof_1_year: DataFrame for 1-year caregivers
        prof_2_year: DataFrame for 2-year caregivers
        prof_3_year: DataFrame for 3-year caregivers
        prof_4_year: DataFrame for 4-year caregivers
        window: Window size around event (e.g., 20 = -20 to +20 periods)
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
        xlabel: Label for x-axis (default: "Year relative to start of first care spell")
    """
    # Plot
    # Increased figure size to maintain visual balance with thinner lines/text
    plt.figure(figsize=(14, 8))

    # Plot overall baseline caregiving rate (entire baseline sample) - dashed black line
    plt.plot(
        prof["distance_to_first_care"],
        prof["caregiving_o"],
        label="Baseline",
        color="black",
        linewidth=2.0,
        linestyle="--",
        marker=None,
    )

    # Plot no-care-demand caregiving rate
    plt.plot(
        prof["distance_to_first_care"],
        prof["caregiving_c"],
        label="No Care Demand",
        color="black",
        linewidth=2.0,
        linestyle="-",
        marker=None,
    )

    # Plot baseline caregiving rates for 1-year caregivers (care at t=0 only, then stop)
    if len(prof_1_year) > 0:
        plt.plot(
            prof_1_year["distance_to_first_care"],
            prof_1_year["caregiving_o"],
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

    # # Plot baseline caregiving rates for 2-year caregivers (care at t=0 and t=1,
    #  then stop)
    if len(prof_2_year) > 0:
        plt.plot(
            prof_2_year["distance_to_first_care"],
            prof_2_year["caregiving_o"],
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

    # # Plot baseline caregiving rates for 3-year caregivers (care at t=0, t=1, t=
    # 2, then stop)
    if len(prof_3_year) > 0:
        plt.plot(
            prof_3_year["distance_to_first_care"],
            prof_3_year["caregiving_o"],
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

    # # Plot baseline caregiving rates for 4-year caregivers (care at t=0, t=1, t=
    # 2, t=3, then stop)
    if len(prof_4_year) > 0:
        plt.plot(
            prof_4_year["distance_to_first_care"],
            prof_4_year["caregiving_o"],
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
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Caregiving Rate", fontsize=14)
    # Add padding: x-axis extends beyond -window and window, y-axis extends below 0
    plt.xlim(-window - 0.5, window + 0.5)
    plt.ylim(-0.025, 1.0)  # Caregiving rate is between 0 and 1, with padding below
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
