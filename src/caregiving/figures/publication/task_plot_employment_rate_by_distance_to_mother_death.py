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
    PUBLICATION_PLOT_STYLE,
    calculate_simple_outcomes,
    ensure_agent_period,
    prepare_dataframes_simple,
    publication_savefig,
)
from caregiving.figures.publication.plotting_helpers_mother_death import (
    add_distance_to_mother_death,
    identify_agents_by_care_demand_before_death,
    identify_agents_by_care_demand_before_death_at_least,
    identify_agents_by_caregiving_before_death,
    identify_agents_by_caregiving_before_death_at_least,
    identify_agents_by_exact_caregiving_years_in_window,
    identify_agents_by_first_care_demand_timing_before_death,
    identify_agents_by_first_caregiving_timing_before_death,
    identify_agents_by_total_caregiving_before_death,
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

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_exact_caregiving")
    def task_plot_employment_rate_by_distance_to_mother_death_exact_caregiving(  # noqa: PLR0912, PLR0915
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
        / "counterfactual"
        / "reverse_employment"
        / f"employment_rate_by_distance_to_mother_death_exact_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Plot employment rate by distance to mother's death.

        Creates an event study plot comparing baseline vs no-care-demand employment
        rates, where t=0 is when mother dies (mother_dead == PARENT_RECENTLY_DEAD).
        The analysis is "reverse" - we examine employment rates before and after
        mother's death.

        Homogeneous groups are based on EXACT caregiving duration BEFORE death:
        - 1-year: care at t=-1, but NOT at t=-2
        - 2-year: care at t=-1 and t=-2, but NOT at t=-3
        - 3-year: care at t=-1, t=-2, t=-3, but NOT at t=-4
        - 4-year: care at t=-1, t=-2, t=-3, t=-4, but NOT at t=-5
        - 5-year: care at t=-1, t=-2, t=-3, t=-4, t=-5, but NOT at t=-6
          (exactly 5 years)

        Groups are mutually exclusive (no overlap).

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
            window_low: Years before t=0 (positive int).
            window_high: Years after t=0 (positive int).
            window_by_age: Optional per-age (window_low, window_high); keys as in age groups.

        """
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
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

        # Filter to agents with valid first death period
        # and trim to window
        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
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

        # Identify agents by EXACT caregiving duration BEFORE death
        # (1, 2, 3, 4, 5 years exactly)
        agents_1_year, agents_2_year, agents_3_year, agents_4_year, agents_5_year = (
            identify_agents_by_caregiving_before_death(
                merged,
                distance_col="distance_to_mother_death",
                add_five_year=True,
                last_group_at_least=False,  # 5-year is "exactly 5 years"
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

        # Create conditional series for 5-year caregivers (before death)
        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_5_year = prof_5_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Call plotting function with 5 groups
        plot_employment_rate_by_distance_to_mother_death(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_at_least_caregiving")
    def task_plot_employment_rate_by_distance_to_mother_death_at_least_caregiving(  # noqa: PLR0912, PLR0915
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
        / "counterfactual"
        / "reverse_employment"
        / (
            f"employment_rate_by_distance_to_mother_death_at_least_caregiving_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
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
            window_low: Years before t=0 (positive int).
            window_high: Years after t=0 (positive int).
            window_by_age: Optional per-age (window_low, window_high); keys as in age groups.

        """
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
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

        # Filter to agents with valid first death period
        # and trim to window
        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
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
        agents_1_year, agents_2_year, agents_3_year, agents_4_year, _ = (
            identify_agents_by_caregiving_before_death_at_least(
                merged,
                distance_col="distance_to_mother_death",
                add_five_year=False,
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
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication_reverse_employment_total_caregiving
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving")
    def task_plot_employment_rate_by_distance_to_mother_death_total(  # noqa: PLR0912, PLR0915
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "reverse_employment"
        / (
            f"employment_rate_by_distance_to_mother_death_total_caregiving_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Plot employment rate by distance to mother's death (total care years).

        Creates an event study plot comparing baseline vs no-care-demand employment
        rates, where t=0 is when mother dies (mother_dead == PARENT_RECENTLY_DEAD).
        Groups are based on TOTAL (cumulative) caregiving years before death:
        exactly 1, 2, 3, 4, or 5+ periods with care in the window before death.
        Not consecutive: periods can be anywhere in [-window, -1].

        Can be filtered by age at mother's death period.

        Steps:
          1) Restrict to alive agents.
          2) Ensure agent/period columns, employment outcomes, merge.
          3) Compute distance_to_mother_death, filter to window and age.
          4) Identify agents by total caregiving years before death (1, 2, 3, 4, 5+).
          5) Aggregate and plot.
        """
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
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

        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        care_codes_for_indicator = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols["current_caregiving"] = (
            o_cols["choice"].isin(care_codes_for_indicator).astype(int)
        )

        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work

        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

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

        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
        ]

        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_mother_death", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof = prof.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Identify agents by total (cumulative) caregiving years before death
        agents_1_year, agents_2_year, agents_3_year, agents_4_year, agents_5_year = (
            identify_agents_by_total_caregiving_before_death(
                merged,
                distance_col="distance_to_mother_death",
                window=w_low,
            )
        )

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

        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_5_year = prof_5_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        total_labels = (
            "Baseline (1 total care year before death)",
            "Baseline (2 total care years before death)",
            "Baseline (3 total care years before death)",
            "Baseline (4 total care years before death)",
            "Baseline (5+ total care years before death)",
        )
        plot_employment_rate_by_distance_to_mother_death(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            subgroup_labels=total_labels,
        )


# Sixth battery: first care demand timing before mother's death (5 groups: t=-3, -5, -7, -10, 11+)
# Version A: all agents
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    # @pytask.mark.publication_reverse_employment_care_demand_start
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_care_demand_start_all")
    def task_plot_employment_rate_by_distance_to_mother_death_care_demand_start_all(  # noqa: PLR0912, PLR0915
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
        / "counterfactual"
        / "reverse_employment"
        / "care_demand_start"
        / (
            f"employment_rate_by_distance_to_mother_death_care_demand_start_all_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Plot employment rate by distance to mother's death (first care demand timing, all agents).

        Five lines: first care demand at t=-3, -5, -7, -10, or 11+ years before death.
        Thin vertical dashed lines at -3, -5, -7, -10, -11. No caregiving_type filter.
        """
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )

        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period"]].copy()
        o_cols["work_o"] = o_work
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

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
        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_mother_death", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof = prof.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        first_death_by_agent = merged[["agent", "first_death_period"]].drop_duplicates()
        agents_3, agents_5, agents_7, agents_10, agents_11_plus = (
            identify_agents_by_first_care_demand_timing_before_death(
                df_o, first_death_by_agent
            )
        )

        merged_3 = merged[merged["agent"].isin(agents_3)]
        prof_1_year = (
            merged_3.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_1_year = prof_1_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_5 = merged[merged["agent"].isin(agents_5)]
        prof_2_year = (
            merged_5.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_2_year = prof_2_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_7 = merged[merged["agent"].isin(agents_7)]
        prof_3_year = (
            merged_7.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_3_year = prof_3_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_10 = merged[merged["agent"].isin(agents_10)]
        prof_4_year = (
            merged_10.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_11_plus = merged[merged["agent"].isin(agents_11_plus)]
        prof_5_year = (
            merged_11_plus.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_5_year = prof_5_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        start_labels = (
            "Baseline (First care demand 3 years before death)",
            "Baseline (First care demand 5 years before death)",
            "Baseline (First care demand 7 years before death)",
            "Baseline (First care demand 10 years before death)",
            "Baseline (First care demand 11+ years before death)",
        )
        plot_employment_rate_by_distance_to_mother_death(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            subgroup_labels=start_labels,
            vertical_lines_at=[-3, -5, -7, -10, -11],
        )


# Sixth battery: first care demand timing (Version B: caregiving_type == 1)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication_reverse_employment_care_demand_start
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_care_demand_start_type1")
    def task_plot_employment_rate_by_distance_to_mother_death_care_demand_start_type1(  # noqa: PLR0912, PLR0915
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
        / "counterfactual"
        / "reverse_employment"
        / "care_demand_start"
        / (
            f"employment_rate_by_distance_to_mother_death_care_demand_start_type1_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Plot employment rate by distance to mother's death (first care demand timing, type 1).

        Five lines: first care demand at t=-3, -5, -7, -10, or 11+ years before death.
        Restricted to caregiving_type == 1. Thin vertical dashed lines at -3, -5, -7, -10, -11.
        """
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )

        if "caregiving_type" not in df_o.columns:
            raise ValueError(
                "caregiving_type column not found. Cannot filter to type 1."
            )
        type_1_agents = df_o[df_o["caregiving_type"] == 1]["agent"].unique()
        df_o = df_o[df_o["agent"].isin(type_1_agents)].copy()
        df_c = df_c[df_c["agent"].isin(type_1_agents)].copy()

        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        o_cols = df_o[["agent", "period"]].copy()
        o_cols["work_o"] = o_work
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

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
        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_mother_death", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof = prof.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        first_death_by_agent = merged[["agent", "first_death_period"]].drop_duplicates()
        agents_3, agents_5, agents_7, agents_10, agents_11_plus = (
            identify_agents_by_first_care_demand_timing_before_death(
                df_o, first_death_by_agent
            )
        )

        merged_3 = merged[merged["agent"].isin(agents_3)]
        prof_1_year = (
            merged_3.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_1_year = prof_1_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_5 = merged[merged["agent"].isin(agents_5)]
        prof_2_year = (
            merged_5.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_2_year = prof_2_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_7 = merged[merged["agent"].isin(agents_7)]
        prof_3_year = (
            merged_7.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_3_year = prof_3_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_10 = merged[merged["agent"].isin(agents_10)]
        prof_4_year = (
            merged_10.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_11_plus = merged[merged["agent"].isin(agents_11_plus)]
        prof_5_year = (
            merged_11_plus.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_5_year = prof_5_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        start_labels = (
            "Baseline (First care demand 3 years before death)",
            "Baseline (First care demand 5 years before death)",
            "Baseline (First care demand 7 years before death)",
            "Baseline (First care demand 10 years before death)",
            "Baseline (First care demand 11+ years before death)",
        )
        plot_employment_rate_by_distance_to_mother_death(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            subgroup_labels=start_labels,
            vertical_lines_at=[-3, -5, -7, -10, -11],
        )


# First care demand <5y before death; 4 lines = exact 1,2,3,4 years caregiving in [-4,-1]
# No caregiving_type filter (same agents in baseline and counterfactual). One vertical line at -5.
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (45, 50, "ages_45_50"),
    (50, 55, "ages_50_55"),
    (55, 60, "ages_55_60"),
    (60, 65, "ages_60_65"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication_reverse_employment_care_demand_start
    @pytask.mark.publication_reverse_employment_care_demand_start_caregivers
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_first_care_demand_under_5_exact_caregiving"
    )
    def task_plot_employment_rate_by_distance_to_mother_death_first_caregiving_spell_type1(  # noqa: PLR0912, PLR0915
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
        / "counterfactual"
        / "reverse_employment"
        / "care_demand_start"
        / (
            f"employment_rate_by_distance_to_mother_death_first_care_demand_under_5_"
            f"exact_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Plot employment rate by distance to mother's death (first care demand <5y, exact caregiving).

        Sample: agents whose first care demand is 1-4 years before mother's death (distance in
        [-4,-1]). No caregiving_type filter: counterfactual uses same agents as baseline.
        Four lines: exact 1, 2, 3, or 4 years of (informal) caregiving in [-4,-1]. One vertical line at -5.
        """
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )

        # Restrict to agents whose first care demand is 1-4 years before death (<5 years before)
        first_care_demand = (
            df_o[df_o["care_demand"] > 0]
            .groupby("agent", observed=False)["period"]
            .min()
            .reset_index()
            .rename(columns={"period": "first_care_demand_period"})
        )
        df_o_dist = add_distance_to_mother_death(df_o)
        first_death = (
            df_o_dist.groupby("agent", observed=False)["first_death_period"]
            .first()
            .reset_index()
        )
        timing = first_care_demand.merge(
            first_death[["agent", "first_death_period"]], on="agent", how="inner"
        )
        timing["dist_at_first_care_demand"] = (
            timing["first_care_demand_period"] - timing["first_death_period"]
        )
        agents_sample = timing[
            (timing["dist_at_first_care_demand"] >= -4)
            & (timing["dist_at_first_care_demand"] <= -1)
        ]["agent"].unique()

        df_o = df_o[df_o["agent"].isin(agents_sample)].copy()
        df_c = df_c[df_c["agent"].isin(agents_sample)].copy()

        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols[["agent", "period", "work_o", "current_caregiving"]].merge(
            c_cols, on=["agent", "period"], how="inner"
        )
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )

        dist_map = first_death[first_death["agent"].isin(agents_sample)]
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_mother_death"] = (
            merged["period"] - merged["first_death_period"]
        )

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
        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_mother_death", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof = prof.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Exact 1, 2, 3, 4 years of caregiving in [-4, -1] (no consecutive requirement)
        agents_1, agents_2, agents_3, agents_4 = (
            identify_agents_by_exact_caregiving_years_in_window(
                merged, "distance_to_mother_death", window_start=-4, window_end=-1
            )
        )

        merged_1 = merged[merged["agent"].isin(agents_1)]
        prof_1_year = (
            merged_1.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_1_year = prof_1_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_2 = merged[merged["agent"].isin(agents_2)]
        prof_2_year = (
            merged_2.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_2_year = prof_2_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_3 = merged[merged["agent"].isin(agents_3)]
        prof_3_year = (
            merged_3.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_3_year = prof_3_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )
        merged_4 = merged[merged["agent"].isin(agents_4)]
        prof_4_year = (
            merged_4.groupby("distance_to_mother_death", observed=False)[["work_o"]]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_4_year = prof_4_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        exact_caregiving_labels = (
            "Baseline (1 year caregiving)",
            "Baseline (2 years caregiving)",
            "Baseline (3 years caregiving)",
            "Baseline (4 years caregiving)",
        )
        plot_employment_rate_by_distance_to_mother_death(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=None,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            subgroup_labels=exact_caregiving_labels,
            vertical_lines_at=[-5],
        )


# First care demand in [-9,-5] (5-9y before death); 5 lines = exact 1,2,3,4, 5+ years caregiving in [-9,-5]
# Age groups: all_ages, ages_60_70 only. Vertical lines at -10, -5.
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (50, 55, "ages_50_55"),
    (55, 60, "ages_55_60"),
    (60, 65, "ages_60_65"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication_reverse_employment_care_demand_start
    @pytask.mark.publication_reverse_employment_care_demand_start_caregivers
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_first_care_demand_5_to_9_exact_caregiving"
    )
    def task_plot_employment_rate_by_distance_to_mother_death_first_care_demand_5_to_9(  # noqa: PLR0912, PLR0915
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
        / "counterfactual"
        / "reverse_employment"
        / "care_demand_start"
        / (
            f"employment_rate_by_distance_to_mother_death_first_care_demand_5_to_9_"
            f"exact_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """First care demand 5-9y before death; exact 1,2,3,4, 5+ years caregiving in [-9,-5]. No type filter."""
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        first_care_demand = (
            df_o[df_o["care_demand"] > 0]
            .groupby("agent", observed=False)["period"]
            .min()
            .reset_index()
            .rename(columns={"period": "first_care_demand_period"})
        )
        df_o_dist = add_distance_to_mother_death(df_o)
        first_death = (
            df_o_dist.groupby("agent", observed=False)["first_death_period"]
            .first()
            .reset_index()
        )
        timing = first_care_demand.merge(
            first_death[["agent", "first_death_period"]], on="agent", how="inner"
        )
        timing["dist_at_first_care_demand"] = (
            timing["first_care_demand_period"] - timing["first_death_period"]
        )
        agents_sample = timing[
            (timing["dist_at_first_care_demand"] >= -9)
            & (timing["dist_at_first_care_demand"] <= -5)
        ]["agent"].unique()

        df_o = df_o[df_o["agent"].isin(agents_sample)].copy()
        df_c = df_c[df_c["agent"].isin(agents_sample)].copy()

        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols[["agent", "period", "work_o", "current_caregiving"]].merge(
            c_cols, on=["agent", "period"], how="inner"
        )
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = first_death[first_death["agent"].isin(agents_sample)]
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_mother_death"] = (
            merged["period"] - merged["first_death_period"]
        )
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
        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_mother_death", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof = prof.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        agents_1, agents_2, agents_3, agents_4, agents_5_plus = (
            identify_agents_by_exact_caregiving_years_in_window(
                merged,
                "distance_to_mother_death",
                window_start=-9,
                window_end=-5,
                include_5_plus=True,
            )
        )

        def _prof_for_agents(agents_arr):
            m = merged[merged["agent"].isin(agents_arr)]
            p = (
                m.groupby("distance_to_mother_death", observed=False)[["work_o"]]
                .mean()
                .reset_index()
                .sort_values("distance_to_mother_death")
            )
            return p.rename(
                columns={"distance_to_mother_death": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1)
        prof_2_year = _prof_for_agents(agents_2)
        prof_3_year = _prof_for_agents(agents_3)
        prof_4_year = _prof_for_agents(agents_4)
        prof_5_year = _prof_for_agents(agents_5_plus)

        labels_5 = (
            "Baseline (1 year caregiving)",
            "Baseline (2 years caregiving)",
            "Baseline (3 years caregiving)",
            "Baseline (4 years caregiving)",
            "Baseline (5+ years caregiving)",
        )
        plot_employment_rate_by_distance_to_mother_death(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            subgroup_labels=labels_5,
            vertical_lines_at=[-10, -5],
        )


# First care demand <= -10 (10+ years before death); 5 lines = exact 1,2,3,4, 5+ years caregiving in [-window,-1]
# Age groups: all_ages, ages_60_70 only. Vertical line at -10.
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (50, 55, "ages_50_55"),
    (55, 60, "ages_55_60"),
    (60, 65, "ages_60_65"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication_reverse_employment_care_demand_start
    @pytask.mark.publication_reverse_employment_care_demand_start_caregivers
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_first_care_demand_10_plus_exact_caregiving"
    )
    def task_plot_employment_rate_by_distance_to_mother_death_first_care_demand_10_plus(  # noqa: PLR0912, PLR0915
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
        / "counterfactual"
        / "reverse_employment"
        / "care_demand_start"
        / (
            f"employment_rate_by_distance_to_mother_death_first_care_demand_10_plus_"
            f"exact_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """First care demand 10+ years before death; exact 1,2,3,4, 5+ years caregiving in [-window,-1]. No type filter."""
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        first_care_demand = (
            df_o[df_o["care_demand"] > 0]
            .groupby("agent", observed=False)["period"]
            .min()
            .reset_index()
            .rename(columns={"period": "first_care_demand_period"})
        )
        df_o_dist = add_distance_to_mother_death(df_o)
        first_death = (
            df_o_dist.groupby("agent", observed=False)["first_death_period"]
            .first()
            .reset_index()
        )
        timing = first_care_demand.merge(
            first_death[["agent", "first_death_period"]], on="agent", how="inner"
        )
        timing["dist_at_first_care_demand"] = (
            timing["first_care_demand_period"] - timing["first_death_period"]
        )
        agents_sample = timing[timing["dist_at_first_care_demand"] <= -10][
            "agent"
        ].unique()

        df_o = df_o[df_o["agent"].isin(agents_sample)].copy()
        df_c = df_c[df_c["agent"].isin(agents_sample)].copy()

        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols[["agent", "period", "work_o", "current_caregiving"]].merge(
            c_cols, on=["agent", "period"], how="inner"
        )
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = first_death[first_death["agent"].isin(agents_sample)]
        merged = merged.merge(dist_map, on="agent", how="left")
        merged["distance_to_mother_death"] = (
            merged["period"] - merged["first_death_period"]
        )
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
        merged = merged[
            merged["first_death_period"].notna()
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof = (
            merged.groupby("distance_to_mother_death", observed=False)[
                ["work_o", "work_c"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof = prof.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        agents_1, agents_2, agents_3, agents_4, agents_5_plus = (
            identify_agents_by_exact_caregiving_years_in_window(
                merged,
                "distance_to_mother_death",
                window_start=-w_low,
                window_end=-1,
                include_5_plus=True,
            )
        )

        def _prof_for_agents(agents_arr):
            m = merged[merged["agent"].isin(agents_arr)]
            p = (
                m.groupby("distance_to_mother_death", observed=False)[["work_o"]]
                .mean()
                .reset_index()
                .sort_values("distance_to_mother_death")
            )
            return p.rename(
                columns={"distance_to_mother_death": "distance_to_first_care"}
            )

        prof_1_year = _prof_for_agents(agents_1)
        prof_2_year = _prof_for_agents(agents_2)
        prof_3_year = _prof_for_agents(agents_3)
        prof_4_year = _prof_for_agents(agents_4)
        prof_5_year = _prof_for_agents(agents_5_plus)

        labels_5 = (
            "Baseline (1 year caregiving)",
            "Baseline (2 years caregiving)",
            "Baseline (3 years caregiving)",
            "Baseline (4 years caregiving)",
            "Baseline (5+ years caregiving)",
        )
        plot_employment_rate_by_distance_to_mother_death(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            subgroup_labels=labels_5,
            vertical_lines_at=[-10],
        )


def plot_employment_rate_by_distance_to_mother_death(  # noqa: PLR0912, PLR0913
    prof,
    prof_1_year,
    prof_2_year,
    prof_3_year,
    prof_4_year,
    prof_5_year=None,
    window_low: int = 20,
    window_high: int = 20,
    path_to_plot: Optional[Path] = None,
    subgroup_labels: Optional[tuple[str, ...]] = None,
    vertical_lines_at: Optional[list[int]] = None,
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
        prof_5_year: Optional DataFrame for 5-year caregivers (before death)
        window_low: Years before t=0 (positive int).
        window_high: Years after t=0 (positive int).
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
        subgroup_labels: Optional tuple of 4 or 5 legend labels for the duration lines.
            If None, uses default labels (consecutive years before death).
        vertical_lines_at: Optional list of x positions for thin vertical dashed lines
            (e.g. [-3, -5, -7, -10, -11] for first care demand timing).
    """
    S = PUBLICATION_PLOT_STYLE
    plt.figure(figsize=S["figsize"])

    # Plot overall baseline employment rate (entire baseline sample) - dashed black line
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_o"],
        label="Baseline",
        color="black",
        linewidth=S["linewidth"],
        linestyle="--",
        marker=None,
    )

    # Plot no-care-demand employment rate - solid black line
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_c"],
        label="No Care Demand",
        color="black",
        linewidth=S["linewidth"],
        linestyle="-",
        marker=None,
    )

    labels = subgroup_labels or (
        "Baseline (1-Year Caregivers: t=-1)",
        "Baseline (2-Year Caregivers: t=-1, t=-2)",
        "Baseline (3-Year Caregivers: t=-1, t=-2, t=-3)",
        "Baseline (4-Year Caregivers: t=-1, t=-2, t=-3, t=-4)",
        "Baseline (5-Year Caregivers: t=-1, t=-2, t=-3, t=-4, t=-5)",
    )
    # Plot baseline employment rate for 1-year caregivers (care at t=-1, but NOT t=-2)
    if len(prof_1_year) > 0:
        plt.plot(
            prof_1_year["distance_to_first_care"],
            prof_1_year["work_o"],
            label=labels[0],
            color="0.8",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="8",  # Octagon
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )

    # Plot baseline employment rate for 2-year caregivers
    if len(prof_2_year) > 0:
        plt.plot(
            prof_2_year["distance_to_first_care"],
            prof_2_year["work_o"],
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

    # Plot baseline employment rate for 3-year caregivers
    if len(prof_3_year) > 0:
        plt.plot(
            prof_3_year["distance_to_first_care"],
            prof_3_year["work_o"],
            label=labels[2],
            color="0.4",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="D",  # Diamond
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )

    # Plot baseline employment rate for 4-year caregivers
    if len(prof_4_year) > 0:
        plt.plot(
            prof_4_year["distance_to_first_care"],
            prof_4_year["work_o"],
            label=labels[3],
            color="0.2",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="s",  # Hollow square
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )

    # Plot baseline employment rate for 5-year caregivers
    if prof_5_year is not None and len(prof_5_year) > 0 and len(labels) > 4:
        plt.plot(
            prof_5_year["distance_to_first_care"],
            prof_5_year["work_o"],
            label=labels[4],
            color="black",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="*",  # Star
            markersize=S["markersize_star"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth_star"],
        )

    # Optional thin vertical dashed lines (e.g. at first care demand timing)
    if vertical_lines_at:
        for x in vertical_lines_at:
            plt.axvline(
                x=x,
                color="gray",
                linestyle="--",
                linewidth=0.6,
                alpha=0.7,
            )

    # Add vertical line at t=0 (mother's death)
    plt.axvline(
        x=-0.5,
        color="k",
        linestyle=(0, (7, 7)),
        linewidth=S["axvline_linewidth"],
    )

    # Formatting
    plt.xlabel("Year relative to mother's death", fontsize=S["label_fontsize"])
    plt.ylabel("Employment Rate", fontsize=S["label_fontsize"])
    plt.xlim(-window_low - 0.5, window_high + 0.5)
    plt.ylim(-0.025, 1.0)
    plt.grid(
        True, axis="y", alpha=S["grid_alpha"], linewidth=S["grid_linewidth"]
    )
    plt.xticks(
        range(-window_low, window_high + 1, 5), fontsize=S["xtick_fontsize"]
    )
    plt.yticks(fontsize=S["ytick_fontsize"])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=S["tick_length"], width=S["tick_width"])

    plt.tight_layout()
    if path_to_plot:
        publication_savefig(path_to_plot)
    plt.close()


# Task functions for care demand duration (restricted to caregiving_type == 1)
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_care_demand_exact")
    def task_plot_employment_rate_by_distance_to_mother_death_care_demand_exact(  # noqa: PLR0912, PLR0915
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
        / "counterfactual"
        / "reverse_employment"
        / (
            f"employment_rate_by_distance_to_mother_death_care_demand_exact_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Plot employment rate by distance to mother's death (exact care demand).

        Creates an event study plot comparing baseline vs no-care-demand employment
        rates, where t=0 is when mother dies (mother_dead == PARENT_RECENTLY_DEAD).
        The analysis is "reverse" - we examine employment rates before and after
        mother's death.

        Restricted to caregiving_type == 1 (agents who can provide informal care).

        Homogeneous groups are based on EXACT care demand duration BEFORE death:
        - 1-year: care demand at t=-1, but NOT at t=-2
        - 2-year: care demand at t=-1 and t=-2, but NOT at t=-3
        - 3-year: care demand at t=-1, t=-2, t=-3, but NOT at t=-4
        - 4-year: care demand at t=-1, t=-2, t=-3, t=-4, but NOT at t=-5
        - 5-year: care demand at t=-1, t=-2, t=-3, t=-4, t=-5, but NOT at t=-6
          (exactly 5 years)

        Groups are mutually exclusive (no overlap).

        Can be filtered by age at mother's death period.

        Steps:
          1) Restrict to alive and caregiving_type == 1.
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
            window_low: Years before t=0 (positive int).
            window_high: Years after t=0 (positive int).
            window_by_age: Optional per-age (window_low, window_high); keys as in age groups.

        """
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
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
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
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

        # Identify agents by EXACT care demand duration BEFORE death
        # (1, 2, 3, 4, 5 years exactly)
        agents_1_year, agents_2_year, agents_3_year, agents_4_year, agents_5_year = (
            identify_agents_by_care_demand_before_death(
                merged,
                distance_col="distance_to_mother_death",
                add_five_year=True,
                last_group_at_least=False,  # 5-year is "exactly 5 years"
            )
        )

        # Create conditional series for each group
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

        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_5_year = prof_5_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Call plotting function with 5 groups
        plot_employment_rate_by_distance_to_mother_death_care_demand(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_reverse_employment
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_care_demand_at_least")
    def task_plot_employment_rate_by_distance_to_mother_death_care_demand_at_least(  # noqa: PLR0912, PLR0915
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
        / "counterfactual"
        / "reverse_employment"
        / (
            f"employment_rate_by_distance_to_mother_death_care_demand_at_least_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Plot employment rate by distance to mother's death (at least care demand).

        Creates an event study plot comparing baseline vs no-care-demand employment
        rates, where t=0 is when mother dies (mother_dead == PARENT_RECENTLY_DEAD).
        The analysis is "reverse" - we examine employment rates before and after
        mother's death.

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
            window_low: Years before t=0 (positive int).
            window_high: Years after t=0 (positive int).
            window_by_age: Optional per-age (window_low, window_high); keys as in age groups.

        """
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
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
            & (merged["distance_to_mother_death"] >= -w_low)
            & (merged["distance_to_mother_death"] <= w_high)
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

        merged_5_year = merged[merged["agent"].isin(agents_5_year)].copy()
        prof_5_year = (
            merged_5_year.groupby("distance_to_mother_death", observed=False)[
                ["work_o"]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        prof_5_year = prof_5_year.rename(
            columns={"distance_to_mother_death": "distance_to_first_care"}
        )

        # Call plotting function with 5 groups
        plot_employment_rate_by_distance_to_mother_death_care_demand(
            prof=prof,
            prof_1_year=prof_1_year,
            prof_2_year=prof_2_year,
            prof_3_year=prof_3_year,
            prof_4_year=prof_4_year,
            prof_5_year=prof_5_year,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
        )


def plot_employment_rate_by_distance_to_mother_death_care_demand(  # noqa: PLR0912, PLR0913
    prof,
    prof_1_year,
    prof_2_year,
    prof_3_year,
    prof_4_year,
    prof_5_year,
    window_low: int = 20,
    window_high: int = 20,
    path_to_plot: Optional[Path] = None,
) -> None:
    """Plot employment rate by distance to mother's death (care demand).

    Creates an event study plot comparing baseline vs no-care-demand employment
    rates, with separate lines for different care demand durations before death.

    Args:
        prof: DataFrame with columns 'distance_to_first_care', 'work_o', 'work_c'
        prof_1_year: DataFrame for 1-year care demand group
        prof_2_year: DataFrame for 2-year care demand group
        prof_3_year: DataFrame for 3-year care demand group
        prof_4_year: DataFrame for 4-year care demand group
        prof_5_year: DataFrame for 5-year care demand group
        window_low: Years before t=0 (positive int).
        window_high: Years after t=0 (positive int).
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
    """
    S = PUBLICATION_PLOT_STYLE
    plt.figure(figsize=S["figsize"])

    # Plot overall baseline employment rate (entire baseline sample) - dashed black line
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_o"],
        label="Baseline",
        color="black",
        linewidth=S["linewidth"],
        linestyle="--",
        marker=None,
    )

    # Plot no-care-demand employment rate - solid black line
    plt.plot(
        prof["distance_to_first_care"],
        prof["work_c"],
        label="No Care Demand",
        color="black",
        linewidth=S["linewidth"],
        linestyle="-",
        marker=None,
    )

    # Plot baseline employment rate for 1-year care demand group
    if len(prof_1_year) > 0:
        plt.plot(
            prof_1_year["distance_to_first_care"],
            prof_1_year["work_o"],
            label="Baseline (1-Year Care Demand)",
            color="0.8",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="8",  # Octagon
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )

    # Plot baseline employment rate for 2-year care demand group
    if len(prof_2_year) > 0:
        plt.plot(
            prof_2_year["distance_to_first_care"],
            prof_2_year["work_o"],
            label="Baseline (2-Year Care Demand)",
            color="0.6",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="^",
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )

    # Plot baseline employment rate for 3-year care demand group
    if len(prof_3_year) > 0:
        plt.plot(
            prof_3_year["distance_to_first_care"],
            prof_3_year["work_o"],
            label="Baseline (3-Year Care Demand)",
            color="0.4",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="D",  # Diamond
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )

    # Plot baseline employment rate for 4-year care demand group
    if len(prof_4_year) > 0:
        plt.plot(
            prof_4_year["distance_to_first_care"],
            prof_4_year["work_o"],
            label="Baseline (4-Year Care Demand)",
            color="0.2",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="s",  # Hollow square
            markersize=S["markersize"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth"],
        )

    # Plot baseline employment rate for 5-year care demand group
    if len(prof_5_year) > 0:
        plt.plot(
            prof_5_year["distance_to_first_care"],
            prof_5_year["work_o"],
            label="Baseline (5+ Year Care Demand)",
            color="black",
            linewidth=S["linewidth"],
            linestyle="-",
            marker="*",  # Star
            markersize=S["markersize_star"],
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=S["markeredgewidth_star"],
        )

    # Add vertical line at t=0 (mother's death)
    plt.axvline(
        x=-0.5,
        color="k",
        linestyle=(0, (7, 7)),
        linewidth=S["axvline_linewidth"],
    )

    # Formatting
    plt.xlabel("Year relative to mother's death", fontsize=S["label_fontsize"])
    plt.ylabel("Employment Rate", fontsize=S["label_fontsize"])
    plt.xlim(-window_low - 0.5, window_high + 0.5)
    plt.ylim(-0.025, 1.0)
    plt.grid(
        True, axis="y", alpha=S["grid_alpha"], linewidth=S["grid_linewidth"]
    )
    plt.xticks(
        range(-window_low, window_high + 1, 5), fontsize=S["xtick_fontsize"]
    )
    plt.yticks(fontsize=S["ytick_fontsize"])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=S["tick_length"], width=S["tick_width"])

    plt.tight_layout()
    if path_to_plot:
        publication_savefig(path_to_plot)
    plt.close()
