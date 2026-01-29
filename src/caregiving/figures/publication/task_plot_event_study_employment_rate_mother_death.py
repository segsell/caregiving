"""Plot event study of employment rate differences by distance to mother's death.

This module creates event study plots showing the difference in employment rates
between baseline and no-care-demand counterfactual, aligned by distance to mother's
death (t=0). The analysis is "reverse" - t=0 is when mother dies, and we examine
employment rate differences before and after death.

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
    ensure_agent_period,
    prepare_dataframes_simple,
)
from caregiving.figures.publication.plotting_helpers_mother_death import (
    add_distance_to_mother_death,
    identify_agents_by_care_demand_before_death,
    identify_agents_by_care_demand_before_death_at_least,
    identify_agents_by_caregiving_before_death,
    identify_agents_by_caregiving_before_death_at_least,
)
from caregiving.model.shared import (
    INFORMAL_CARE,
    PARENT_RECENTLY_DEAD,
)

for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_at_least")
    def task_plot_event_study_employment_rate_mother_death(  # noqa: PLR0912, PLR0915
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
        / "event_study_reverse"
        / "employment"
        / (
            f"event_study_employment_rate_by_distance_to_mother_death_at_least_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot event study of employment rate differences by distance to death.

        Creates an event study plot showing the difference in employment rates
        between baseline and no-care-demand counterfactual, where t=0 is when mother
        dies (mother_dead == PARENT_RECENTLY_DEAD).

        Homogeneous groups are based on AT LEAST N years of caregiving BEFORE death:
        - At least 1-year: care at t=-1
        - At least 2-year: care at t=-1 and t=-2
        - At least 3-year: care at t=-1, t=-2, t=-3
        - At least 4-year: care at t=-1, t=-2, t=-3, t=-4

        Groups overlap (e.g., 4-year agents also appear in 3-year, 2-year, 1-year).

        Can be filtered by age at mother's death period.

        Steps:
          1) Restrict to alive and (optionally) ever-caregivers/ever-care-demand.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_mother_death from baseline, attach to merged.
          6) Filter by age at mother's death period (if age_min/age_max specified).
          7) Calculate differences: (work_o - work_c) (raw employment rate difference).
          8) Aggregate differences by distance for each group.
          9) Plot differences on same graph.

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
        # Add current caregiving indicator (1 if currently providing care, 0 otherwise)
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work

        # Merge on (agent, period) to ensure matched comparison
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

        # Add mother_dead and age columns to merged for distance calculation
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to mother's death in baseline and attach
        df_o_dist = add_distance_to_mother_death(df_o)
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

        # Filter to agents with valid first death period (i.e., mother died)
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

        # Calculate difference: (work_o - work_c) for raw employment rate difference
        merged["diff"] = merged["work_o"] - merged["work_c"]

        # Aggregate differences by distance for overall baseline
        prof_diff = (
            merged.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        # Rename column to match plotting function expectation
        prof_diff = prof_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Identify agents by AT LEAST N years of caregiving BEFORE death
        agents_1_year, agents_2_year, agents_3_year, agents_4_year, _ = (
            identify_agents_by_caregiving_before_death_at_least(
                merged,
                distance_col="distance_to_mother_death",
                add_five_year=False,
            )
        )

        # Create conditional series for at least 1-year caregivers (before death)
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year_diff = (
            merged_1_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_1_year_diff = prof_1_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for at least 2-year caregivers (before death)
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year_diff = (
            merged_2_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_2_year_diff = prof_2_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for at least 3-year caregivers (before death)
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year_diff = (
            merged_3_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_3_year_diff = prof_3_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for at least 4-year caregivers (before death)
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year_diff = (
            merged_4_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_4_year_diff = prof_4_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Call plotting function
        plot_employment_rate_difference_by_distance_to_mother_death(
            prof_diff=prof_diff,
            prof_1_year_diff=prof_1_year_diff,
            prof_2_year_diff=prof_2_year_diff,
            prof_3_year_diff=prof_3_year_diff,
            prof_4_year_diff=prof_4_year_diff,
            window=window,
            path_to_plot=path_to_plot,
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_consecutive")
    def task_plot_event_study_employment_rate_mother_death_consecutive(  # noqa: PLR0912, PLR0915
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
        / "event_study_reverse"
        / "employment"
        / (
            f"event_study_employment_rate_by_distance_to_mother_death_consecutive_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot event study of employment rate differences
        (consecutive N years before death).

        Creates an event study plot showing the difference in employment rates
        between baseline and no-care-demand counterfactual, where t=0 is when mother
        dies (mother_dead == PARENT_RECENTLY_DEAD).

        Homogeneous groups are based on CONSECUTIVE N years of caregiving BEFORE death:
        - 1-year consecutive: care at t=-1, then NOT at t=-2
          (at least 1 year off before)
        - 2-year consecutive: care at t=-1 and t=-2, then NOT at t=-3
          (at least 1 year off before)
        - 3-year consecutive: care at t=-1, t=-2, t=-3, then NOT at t=-4
          (at least 1 year off before)
        - 4-year consecutive: care at t=-1, t=-2, t=-3, t=-4

        Groups are mutually exclusive (no overlap).

        Can be filtered by age at mother's death period.

        Steps:
          1) Restrict to alive and (optionally) ever-caregivers/ever-care-demand.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_mother_death from baseline, attach to merged.
          6) Filter by age at mother's death period (if age_min/age_max specified).
          7) Calculate differences: (work_o - work_c) (raw employment rate difference).
          8) Aggregate differences by distance for each group.
          9) Plot differences on same graph.

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
        # Add current caregiving indicator (1 if currently providing care, 0 otherwise)
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work

        # Merge on (agent, period) to ensure matched comparison
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

        # Add mother_dead and age columns to merged for distance calculation
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to mother's death in baseline and attach
        df_o_dist = add_distance_to_mother_death(df_o)
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

        # Filter to agents with valid first death period (i.e., mother died)
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

        # Calculate difference: (work_o - work_c) for raw employment rate difference
        merged["diff"] = merged["work_o"] - merged["work_c"]

        # Aggregate differences by distance for overall baseline
        prof_diff = (
            merged.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        # Rename column to match plotting function expectation
        prof_diff = prof_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Identify agents by CONSECUTIVE N years of caregiving BEFORE death
        agents_1_year, agents_2_year, agents_3_year, agents_4_year, _ = (
            identify_agents_by_caregiving_before_death(
                merged,
                distance_col="distance_to_mother_death",
                add_five_year=False,
                last_group_at_least=True,  # Default: last group "at least" N years
            )
        )

        # Create conditional series for 1-year consecutive caregivers (before death)
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year_diff = (
            merged_1_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_1_year_diff = prof_1_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for 2-year consecutive caregivers (before death)
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year_diff = (
            merged_2_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_2_year_diff = prof_2_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for 3-year consecutive caregivers (before death)
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year_diff = (
            merged_3_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_3_year_diff = prof_3_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Create conditional series for 4-year consecutive caregivers (before death)
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year_diff = (
            merged_4_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_4_year_diff = prof_4_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Call plotting function with consecutive labels
        plot_employment_rate_difference_by_distance_to_mother_death_consecutive(
            prof_diff=prof_diff,
            prof_1_year_diff=prof_1_year_diff,
            prof_2_year_diff=prof_2_year_diff,
            prof_3_year_diff=prof_3_year_diff,
            prof_4_year_diff=prof_4_year_diff,
            window=window,
            path_to_plot=path_to_plot,
        )


# Task functions for care demand duration (restricted to caregiving_type == 1)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_care_demand_exact")
    def task_plot_event_study_employment_rate_mother_death_care_demand_exact(  # noqa: PLR0912, PLR0915
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
        / "event_study_reverse"
        / "employment"
        / (
            f"event_study_employment_rate_by_distance_to_mother_death_"
            f"care_demand_exact_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot event study of employment rate differences (exact care demand duration).

        Creates an event study plot showing the difference in employment rates
        between baseline and no-care-demand counterfactual, where t=0 is when mother
        dies (mother_dead == PARENT_RECENTLY_DEAD).

        Restricted to caregiving_type == 1 (agents who can provide informal care).

        Homogeneous groups are based on EXACT care demand duration BEFORE death:
        - 1-year: care demand at t=-1, but NOT at t=-2
        - 2-year: care demand at t=-1 and t=-2, but NOT at t=-3
        - 3-year: care demand at t=-1, t=-2, t=-3, but NOT at t=-4
        - 4-year: care demand at t=-1, t=-2, t=-3, t=-4, but NOT at t=-5
        - 5-year: care demand at t=-1, t=-2, t=-3, t=-4, t=-5 (at least 5 years)

        Groups are mutually exclusive (no overlap).

        Can be filtered by age at mother's death period.

        Steps:
          1) Restrict to alive and caregiving_type == 1.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_mother_death from baseline, attach to merged.
          6) Filter by age at mother's death period (if age_min/age_max specified).
          7) Calculate differences: (work_o - work_c) (raw employment rate difference).
          8) Aggregate differences by distance for each group.
          9) Plot differences on same graph.

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

        # Filter to caregiving_type == 1 (agents who can provide informal care)
        if "caregiving_type" not in df_o.columns:
            raise ValueError(
                "caregiving_type column not found in data. "
                "Cannot filter to caregiving_type == 1."
            )
        type_1_agents = df_o[df_o["caregiving_type"] == 1]["agent"].unique()
        df_o = df_o[df_o["agent"].isin(type_1_agents)].copy()
        df_c = df_c[df_c["agent"].isin(type_1_agents)].copy()

        # Calculate employment outcomes
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")

        # Create outcome columns
        o_cols = df_o[["agent", "period", "choice", "care_demand"]].copy()
        o_cols["work_o"] = o_work

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work

        # Merge on (agent, period) to ensure matched comparison
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

        # Add mother_dead and age columns to merged for distance calculation
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to mother's death in baseline and attach
        df_o_dist = add_distance_to_mother_death(df_o)
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

        # Filter to agents with valid first death period (i.e., mother died)
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

        # Calculate difference: (work_o - work_c) for raw employment rate difference
        merged["diff"] = merged["work_o"] - merged["work_c"]

        # Aggregate differences by distance for overall baseline
        prof_diff = (
            merged.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        # Rename column to match plotting function expectation
        prof_diff = prof_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Identify agents by EXACT care demand duration BEFORE death
        # (1, 2, 3, 4 years exactly, 5+ years)
        agents_1_year, agents_2_year, agents_3_year, agents_4_year, agents_5_year = (
            identify_agents_by_care_demand_before_death(
                merged,
                distance_col="distance_to_mother_death",
                add_five_year=True,
                last_group_at_least=True,  # 5-year is "at least 5 years"
            )
        )

        # Create conditional series for each group
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year_diff = (
            merged_1_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_1_year_diff = prof_1_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year_diff = (
            merged_2_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_2_year_diff = prof_2_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year_diff = (
            merged_3_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_3_year_diff = prof_3_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year_diff = (
            merged_4_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_4_year_diff = prof_4_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year_diff = (
            merged_5_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_5_year_diff = prof_5_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Call plotting function with 5 groups
        plot_employment_rate_difference_by_distance_to_mother_death_care_demand(
            prof_diff=prof_diff,
            prof_1_year_diff=prof_1_year_diff,
            prof_2_year_diff=prof_2_year_diff,
            prof_3_year_diff=prof_3_year_diff,
            prof_4_year_diff=prof_4_year_diff,
            prof_5_year_diff=prof_5_year_diff,
            window=window,
            path_to_plot=path_to_plot,
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_care_demand_at_least")
    def task_plot_event_study_employment_rate_mother_death_care_demand_at_least(  # noqa: PLR0912, PLR0915
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
        / "event_study_reverse"
        / "employment"
        / (
            f"event_study_employment_rate_by_distance_to_mother_death_"
            f"care_demand_at_least_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot event study of employment rate differences
        (at least care demand duration).

        Creates an event study plot showing the difference in employment rates
        between baseline and no-care-demand counterfactual, where t=0 is when mother
        dies (mother_dead == PARENT_RECENTLY_DEAD).

        Restricted to caregiving_type == 1 (agents who can provide informal care).

        Homogeneous groups are based on AT LEAST N years of care demand BEFORE death:
        - At least 1-year: care demand at t=-1
        - At least 2-year: care demand at t=-1 and t=-2
        - At least 3-year: care demand at t=-1, t=-2, t=-3
        - At least 4-year: care demand at t=-1, t=-2, t=-3, t=-4
        - At least 5-year: care demand at t=-1, t=-2, t=-3, t=-4, t=-5

        Groups overlap (e.g., 5-year agents also appear in 4-year, 3-year, etc.).

        Can be filtered by age at mother's death period.

        Steps:
          1) Restrict to alive and caregiving_type == 1.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_mother_death from baseline, attach to merged.
          6) Filter by age at mother's death period (if age_min/age_max specified).
          7) Calculate differences: (work_o - work_c) (raw employment rate difference).
          8) Aggregate differences by distance for each group.
          9) Plot differences on same graph.

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

        # Filter to caregiving_type == 1 (agents who can provide informal care)
        if "caregiving_type" not in df_o.columns:
            raise ValueError(
                "caregiving_type column not found in data. "
                "Cannot filter to caregiving_type == 1."
            )
        type_1_agents = df_o[df_o["caregiving_type"] == 1]["agent"].unique()
        df_o = df_o[df_o["agent"].isin(type_1_agents)].copy()
        df_c = df_c[df_c["agent"].isin(type_1_agents)].copy()

        # Calculate employment outcomes
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")

        # Create outcome columns
        o_cols = df_o[["agent", "period", "choice", "care_demand"]].copy()
        o_cols["work_o"] = o_work

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work

        # Merge on (agent, period) to ensure matched comparison
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

        # Add mother_dead and age columns to merged for distance calculation
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

        # Compute distance to mother's death in baseline and attach
        df_o_dist = add_distance_to_mother_death(df_o)
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

        # Filter to agents with valid first death period (i.e., mother died)
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

        # Calculate difference: (work_o - work_c) for raw employment rate difference
        merged["diff"] = merged["work_o"] - merged["work_c"]

        # Aggregate differences by distance for overall baseline
        prof_diff = (
            merged.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        # Rename column to match plotting function expectation
        prof_diff = prof_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Identify agents by AT LEAST N years of care demand BEFORE death
        agents_1_year, agents_2_year, agents_3_year, agents_4_year, agents_5_year = (
            identify_agents_by_care_demand_before_death_at_least(
                merged,
                distance_col="distance_to_mother_death",
                add_five_year=True,
            )
        )

        # Create conditional series for each group
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year_diff = (
            merged_1_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_1_year_diff = prof_1_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year_diff = (
            merged_2_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_2_year_diff = prof_2_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year_diff = (
            merged_3_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_3_year_diff = prof_3_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year_diff = (
            merged_4_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_4_year_diff = prof_4_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year_diff = (
            merged_5_year.groupby("distance_to_mother_death", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_5_year_diff = prof_5_year_diff.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Call plotting function with 5 groups
        plot_employment_rate_difference_by_distance_to_mother_death_care_demand(
            prof_diff=prof_diff,
            prof_1_year_diff=prof_1_year_diff,
            prof_2_year_diff=prof_2_year_diff,
            prof_3_year_diff=prof_3_year_diff,
            prof_4_year_diff=prof_4_year_diff,
            prof_5_year_diff=prof_5_year_diff,
            window=window,
            path_to_plot=path_to_plot,
        )


def plot_employment_rate_difference_by_distance_to_mother_death(  # noqa: PLR0913
    prof_diff,
    prof_1_year_diff,
    prof_2_year_diff,
    prof_3_year_diff,
    prof_4_year_diff,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
    xlabel: str = "Year relative to mother's death",
) -> None:
    """Plot employment rate difference by distance to mother's death.

    Creates an event study plot showing the difference in employment rates
    between baseline and no-care-demand counterfactual, with separate lines
    for different caregiving durations before death (at least N years).

    Args:
        prof_diff: DataFrame with columns 'distance_to_first_care', 'diff'
            (raw difference in employment rate)
        prof_1_year_diff: DataFrame for at least 1-year caregivers
        prof_2_year_diff: DataFrame for at least 2-year caregivers
        prof_3_year_diff: DataFrame for at least 3-year caregivers
        prof_4_year_diff: DataFrame for at least 4-year caregivers
        window: Window size around event (e.g., 20 = -20 to +20 periods)
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
        xlabel: Label for x-axis (default: "Year relative to mother's death")
    """
    # Plot
    # Increased figure size to maintain visual balance with thinner lines/text
    plt.figure(figsize=(14, 8))

    # Plot overall baseline difference (entire baseline sample) - dashed black line
    plt.plot(
        prof_diff["distance_to_first_care"],
        prof_diff["diff"],
        label="Baseline",
        color="black",
        linewidth=2.0,
        linestyle="--",
        marker=None,
    )

    # Plot horizontal line at y=0 for reference
    plt.axhline(y=0, color="k", linestyle="-", linewidth=0.8, alpha=0.5)

    # Plot difference for at least 1-year caregivers
    if len(prof_1_year_diff) > 0:
        plt.plot(
            prof_1_year_diff["distance_to_first_care"],
            prof_1_year_diff["diff"],
            label="At Least 1-Year Caregivers",
            color="0.8",
            linewidth=2.0,
            linestyle="-",
            marker="8",  # Octagon
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for at least 2-year caregivers
    if len(prof_2_year_diff) > 0:
        plt.plot(
            prof_2_year_diff["distance_to_first_care"],
            prof_2_year_diff["diff"],
            label="At Least 2-Year Caregivers",
            color="0.6",
            linewidth=2.0,
            linestyle="-",
            marker="^",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for at least 3-year caregivers
    if len(prof_3_year_diff) > 0:
        plt.plot(
            prof_3_year_diff["distance_to_first_care"],
            prof_3_year_diff["diff"],
            label="At Least 3-Year Caregivers",
            color="0.4",
            linewidth=2.0,
            linestyle="-",
            marker="D",  # Diamond
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for at least 4-year caregivers
    if len(prof_4_year_diff) > 0:
        plt.plot(
            prof_4_year_diff["distance_to_first_care"],
            prof_4_year_diff["diff"],
            label="At Least 4-Year Caregivers",
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
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Difference in employment rate", fontsize=14)
    # Add padding: x-axis extends beyond -window and window
    plt.xlim(-window - 0.5, window + 0.5)
    # Y-axis range: adjust based on typical differences (can be negative or positive)
    # Use symmetric range around 0, with some padding
    # Collect all differences from all series
    all_diffs = []
    if len(prof_diff) > 0:
        all_diffs.extend(prof_diff["diff"].tolist())
    if len(prof_1_year_diff) > 0:
        all_diffs.extend(prof_1_year_diff["diff"].tolist())
    if len(prof_2_year_diff) > 0:
        all_diffs.extend(prof_2_year_diff["diff"].tolist())
    if len(prof_3_year_diff) > 0:
        all_diffs.extend(prof_3_year_diff["diff"].tolist())
    if len(prof_4_year_diff) > 0:
        all_diffs.extend(prof_4_year_diff["diff"].tolist())

    if all_diffs:
        y_max = max(abs(min(all_diffs)), abs(max(all_diffs)))
        # Add 10% padding and round up to nearest 0.05
        # (since values are between -1 and 1)
        y_lim = (int(y_max * 1.1 / 0.05) + 1) * 0.05
        # Ensure minimum range of 0.1 for visibility
        y_lim = max(y_lim, 0.05)
    else:
        y_lim = 0.1  # Default range if no data
    plt.ylim(-y_lim, y_lim)
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


def plot_employment_rate_difference_by_distance_to_mother_death_consecutive(  # noqa: PLR0913
    prof_diff,
    prof_1_year_diff,
    prof_2_year_diff,
    prof_3_year_diff,
    prof_4_year_diff,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
    xlabel: str = "Year relative to mother's death",
) -> None:
    """Plot employment rate difference by distance (consecutive N years before death).

    Creates an event study plot showing the difference in employment rates
    between baseline and no-care-demand counterfactual, with separate lines
    for different consecutive caregiving durations before death (N consecutive years).

    Args:
        prof_diff: DataFrame with columns 'distance_to_first_care', 'diff'
            (raw difference in employment rate)
        prof_1_year_diff: DataFrame for 1-year consecutive caregivers
        prof_2_year_diff: DataFrame for 2-year consecutive caregivers
        prof_3_year_diff: DataFrame for 3-year consecutive caregivers
        prof_4_year_diff: DataFrame for 4-year consecutive caregivers
        window: Window size around event (e.g., 20 = -20 to +20 periods)
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
        xlabel: Label for x-axis (default: "Year relative to mother's death")
    """
    # Plot
    # Increased figure size to maintain visual balance with thinner lines/text
    plt.figure(figsize=(14, 8))

    # Plot overall baseline difference (entire baseline sample) - dashed black line
    plt.plot(
        prof_diff["distance_to_first_care"],
        prof_diff["diff"],
        label="Baseline",
        color="black",
        linewidth=2.0,
        linestyle="--",
        marker=None,
    )

    # Plot horizontal line at y=0 for reference
    plt.axhline(y=0, color="k", linestyle="-", linewidth=0.8, alpha=0.5)

    # Plot difference for 1-year consecutive caregivers
    if len(prof_1_year_diff) > 0:
        plt.plot(
            prof_1_year_diff["distance_to_first_care"],
            prof_1_year_diff["diff"],
            label="1-Year Consecutive Caregivers",
            color="0.8",
            linewidth=2.0,
            linestyle="-",
            marker="8",  # Octagon
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 2-year consecutive caregivers
    if len(prof_2_year_diff) > 0:
        plt.plot(
            prof_2_year_diff["distance_to_first_care"],
            prof_2_year_diff["diff"],
            label="2-Year Consecutive Caregivers",
            color="0.6",
            linewidth=2.0,
            linestyle="-",
            marker="^",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 3-year consecutive caregivers
    if len(prof_3_year_diff) > 0:
        plt.plot(
            prof_3_year_diff["distance_to_first_care"],
            prof_3_year_diff["diff"],
            label="3-Year Consecutive Caregivers",
            color="0.4",
            linewidth=2.0,
            linestyle="-",
            marker="D",  # Diamond
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 4-year consecutive caregivers
    if len(prof_4_year_diff) > 0:
        plt.plot(
            prof_4_year_diff["distance_to_first_care"],
            prof_4_year_diff["diff"],
            label="4-Year Consecutive Caregivers",
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
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Difference in employment rate", fontsize=14)
    # Add padding: x-axis extends beyond -window and window
    plt.xlim(-window - 0.5, window + 0.5)
    # Y-axis range: adjust based on typical differences (can be negative or positive)
    # Use symmetric range around 0, with some padding
    # Collect all differences from all series
    all_diffs = []
    if len(prof_diff) > 0:
        all_diffs.extend(prof_diff["diff"].tolist())
    if len(prof_1_year_diff) > 0:
        all_diffs.extend(prof_1_year_diff["diff"].tolist())
    if len(prof_2_year_diff) > 0:
        all_diffs.extend(prof_2_year_diff["diff"].tolist())
    if len(prof_3_year_diff) > 0:
        all_diffs.extend(prof_3_year_diff["diff"].tolist())
    if len(prof_4_year_diff) > 0:
        all_diffs.extend(prof_4_year_diff["diff"].tolist())

    if all_diffs:
        y_max = max(abs(min(all_diffs)), abs(max(all_diffs)))
        # Add 10% padding and round up to nearest 0.05
        # (since values are between -1 and 1)
        y_lim = (int(y_max * 1.1 / 0.05) + 1) * 0.05
        # Ensure minimum range of 0.1 for visibility
        y_lim = max(y_lim, 0.05)
    else:
        y_lim = 0.1  # Default range if no data
    plt.ylim(-y_lim, y_lim)
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


def plot_employment_rate_difference_by_distance_to_mother_death_care_demand(  # noqa: PLR0912, PLR0913
    prof_diff,
    prof_1_year_diff,
    prof_2_year_diff,
    prof_3_year_diff,
    prof_4_year_diff,
    prof_5_year_diff,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
    xlabel: str = "Year relative to mother's death",
) -> None:
    """Plot employment rate difference by distance to mother's death (care demand).

    Creates an event study plot showing the difference in employment rates
    between baseline and no-care-demand counterfactual, with separate lines
    for different care demand durations before death.

    Args:
        prof_diff: DataFrame with columns 'distance_to_first_care', 'diff'
            (raw difference in employment rate)
        prof_1_year_diff: DataFrame for 1-year care demand group
        prof_2_year_diff: DataFrame for 2-year care demand group
        prof_3_year_diff: DataFrame for 3-year care demand group
        prof_4_year_diff: DataFrame for 4-year care demand group
        prof_5_year_diff: DataFrame for 5-year care demand group
        window: Window size around event (e.g., 20 = -20 to +20 periods)
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
        xlabel: Label for x-axis (default: "Year relative to mother's death")
    """
    # Plot
    # Increased figure size to maintain visual balance with thinner lines/text
    plt.figure(figsize=(14, 8))

    # Plot overall baseline difference (entire baseline sample) - dashed black line
    plt.plot(
        prof_diff["distance_to_first_care"],
        prof_diff["diff"],
        label="Baseline",
        color="black",
        linewidth=2.0,
        linestyle="--",
        marker=None,
    )

    # Plot horizontal line at y=0 for reference
    plt.axhline(y=0, color="k", linestyle="-", linewidth=0.8, alpha=0.5)

    # Plot difference for 1-year care demand group
    if len(prof_1_year_diff) > 0:
        plt.plot(
            prof_1_year_diff["distance_to_first_care"],
            prof_1_year_diff["diff"],
            label="1-Year Care Demand",
            color="0.8",
            linewidth=2.0,
            linestyle="-",
            marker="8",  # Octagon
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 2-year care demand group
    if len(prof_2_year_diff) > 0:
        plt.plot(
            prof_2_year_diff["distance_to_first_care"],
            prof_2_year_diff["diff"],
            label="2-Year Care Demand",
            color="0.6",
            linewidth=2.0,
            linestyle="-",
            marker="^",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 3-year care demand group
    if len(prof_3_year_diff) > 0:
        plt.plot(
            prof_3_year_diff["distance_to_first_care"],
            prof_3_year_diff["diff"],
            label="3-Year Care Demand",
            color="0.4",
            linewidth=2.0,
            linestyle="-",
            marker="D",  # Diamond
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 4-year care demand group
    if len(prof_4_year_diff) > 0:
        plt.plot(
            prof_4_year_diff["distance_to_first_care"],
            prof_4_year_diff["diff"],
            label="4-Year Care Demand",
            color="0.2",
            linewidth=2.0,
            linestyle="-",
            marker="s",  # Hollow square
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 5-year care demand group
    if len(prof_5_year_diff) > 0:
        plt.plot(
            prof_5_year_diff["distance_to_first_care"],
            prof_5_year_diff["diff"],
            label="5+ Year Care Demand",
            color="black",
            linewidth=2.0,
            linestyle="-",
            marker="*",  # Star
            markersize=6,
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
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Difference in employment rate", fontsize=14)
    # Add padding: x-axis extends beyond -window and window
    plt.xlim(-window - 0.5, window + 0.5)
    # Y-axis range: adjust based on typical differences (can be negative or positive)
    # Use symmetric range around 0, with some padding
    # Collect all differences from all series
    all_diffs = []
    if len(prof_diff) > 0:
        all_diffs.extend(prof_diff["diff"].tolist())
    if len(prof_1_year_diff) > 0:
        all_diffs.extend(prof_1_year_diff["diff"].tolist())
    if len(prof_2_year_diff) > 0:
        all_diffs.extend(prof_2_year_diff["diff"].tolist())
    if len(prof_3_year_diff) > 0:
        all_diffs.extend(prof_3_year_diff["diff"].tolist())
    if len(prof_4_year_diff) > 0:
        all_diffs.extend(prof_4_year_diff["diff"].tolist())
    if len(prof_5_year_diff) > 0:
        all_diffs.extend(prof_5_year_diff["diff"].tolist())

    if all_diffs:
        y_max = max(abs(min(all_diffs)), abs(max(all_diffs)))
        # Add 10% padding and round up to nearest 0.05
        # (since values are between -1 and 1)
        y_lim = (int(y_max * 1.1 / 0.05) + 1) * 0.05
        # Ensure minimum range of 0.1 for visibility
        y_lim = max(y_lim, 0.05)
    else:
        y_lim = 0.1  # Default range if no data
    plt.ylim(-y_lim, y_lim)
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
