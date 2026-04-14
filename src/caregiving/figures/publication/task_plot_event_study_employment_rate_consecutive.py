"""Plot event study of employment rate differences by distance to first care.

This module creates event study plots showing the difference in employment rates
between baseline and no-care-demand counterfactual, aligned by distance to first
caregiving spell (t=0). Groups are defined by at least N years of caregiving.
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
    get_age_at_first_event,
    prepare_dataframes_simple,
)
from caregiving.figures.publication.plotting_helpers import (
    add_distance_to_first_care,
    identify_agents_by_consecutive_duration,
    identify_agents_by_duration_at_least,
)
from caregiving.model.shared import (
    INFORMAL_CARE,
)

for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication_event_study_at_least
    # @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_at_least")
    def task_plot_event_study_employment_rate(  # noqa: PLR0912, PLR0915
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
        / "event_study"
        / "employment"
        / (
            f"event_study_employment_rate_by_distance_to_first_care_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot event study of employment rate differences by distance to first care.

        Creates an event study plot showing the difference in employment rates
        between baseline and no-care-demand counterfactual, where t=0 is the start
        of the first caregiving spell.

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
          7) Calculate differences: (work_o - work_c) (raw employment rate difference).
          8) Aggregate differences by distance for each group.
          9) Plot differences on same graph.

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

        # Calculate difference: (work_o - work_c) for raw employment rate difference
        merged["diff"] = merged["work_o"] - merged["work_c"]

        # Aggregate differences by distance for overall baseline
        prof_diff = (
            merged.groupby("distance_to_first_care", observed=False)["diff"]
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
        prof_1_year_diff = (
            merged_1_year.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for at least 2-year caregivers
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year_diff = (
            merged_2_year.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for at least 3-year caregivers
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year_diff = (
            merged_3_year.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for at least 4-year caregivers
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year_diff = (
            merged_4_year.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Call plotting function
        plot_employment_rate_difference_by_distance(
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

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_event_study_consecutive
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_consecutive")
    def task_plot_event_study_employment_rate_consecutive(  # noqa: PLR0912, PLR0915
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
        / "event_study"
        / "employment"
        / (
            f"event_study_employment_rate_by_distance_to_first_care_spell_consecutive_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Plot event study of employment rate differences
        (consecutive N years then stop).

        Creates an event study plot showing the difference in employment rates
        between baseline and no-care-demand counterfactual, where t=0 is the start
        of the first caregiving spell.

        Homogeneous groups are based on CONSECUTIVE N years of caregiving, then stop:
        - 1-year consecutive: care at t=0, then NOT at t=1 (at least 1 year off)
        - 2-year consecutive: care at t=0 and t=1, then NOT at t=2 (at least 1 year off)
        - 3-year consecutive: care at t=0, t=1, t=2, then NOT at t=3
          (at least 1 year off)
        - 4-year consecutive: care at t=0, t=1, t=2, t=3, then NOT at t=4
          (at least 1 year off)

        Groups are mutually exclusive (no overlap).

        Can be filtered by age at first care period.

        Steps:
          1) Restrict to alive and (optionally) ever-caregivers/ever-care-demand.
          2) Ensure agent/period columns.
          3) Calculate employment outcomes (work indicator) for both scenarios.
          4) Merge on (agent, period) to ensure matched comparison.
          5) Compute distance_to_first_care from baseline, attach to merged.
          6) Filter by age at first care period (if age_min/age_max specified).
          7) Calculate differences: (work_o - work_c) (raw employment rate difference).
          8) Aggregate differences by distance for each group.
          9) Plot differences on same graph.

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

        # Calculate difference: (work_o - work_c) for raw employment rate difference
        merged["diff"] = merged["work_o"] - merged["work_c"]

        # Aggregate differences by distance for overall baseline
        prof_diff = (
            merged.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Identify agents by CONSECUTIVE N years of caregiving duration, then stop
        agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
            identify_agents_by_consecutive_duration(
                merged,
                distance_col="distance_to_first_care",
                duration_type="caregiving",
                last_group_at_least=True,  # Default: last group "at least" N years
            )
        )

        # Create conditional series for 1-year consecutive caregivers
        merged_1_year = merged[merged["agent"].isin(agents_1_year)].copy()
        prof_1_year_diff = (
            merged_1_year.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for 2-year consecutive caregivers
        merged_2_year = merged[merged["agent"].isin(agents_2_year)].copy()
        prof_2_year_diff = (
            merged_2_year.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for 3-year consecutive caregivers
        merged_3_year = merged[merged["agent"].isin(agents_3_year)].copy()
        prof_3_year_diff = (
            merged_3_year.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Create conditional series for 4-year consecutive caregivers
        merged_4_year = merged[merged["agent"].isin(agents_4_year)].copy()
        prof_4_year_diff = (
            merged_4_year.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

        # Call plotting function with consecutive labels
        plot_employment_rate_difference_by_distance_consecutive(
            prof_diff=prof_diff,
            prof_1_year_diff=prof_1_year_diff,
            prof_2_year_diff=prof_2_year_diff,
            prof_3_year_diff=prof_3_year_diff,
            prof_4_year_diff=prof_4_year_diff,
            window=window,
            path_to_plot=path_to_plot,
        )


# ---- Event study outcome differences: full-time, part-time, working hours, labor income (consecutive) ----
def _event_study_consecutive_merged_and_profiles(
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    outcome_o_series,
    outcome_c_series,
    window: int,
    age_min: int | None,
    age_max: int | None,
):
    """Build merged df with distance_to_first_care and diff=outcome_o - outcome_c; return prof_diff, prof_1..4.
    df_o, df_c must already be prepared (e.g. via prepare_dataframes_simple). Outcome series align to df_o/df_c.
    """
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    o_cols = df_o[["agent", "period", "choice"]].copy()
    o_cols["outcome_o"] = np.asarray(outcome_o_series).astype(float)
    o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    c_cols = df_c[["agent", "period"]].copy()
    c_cols["outcome_c"] = np.asarray(outcome_c_series).astype(float)
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
    merged["diff"] = merged["outcome_o"] - merged["outcome_c"]
    prof_diff = (
        merged.groupby("distance_to_first_care", observed=False)["diff"]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )
    agents_1_year, agents_2_year, agents_3_year, agents_4_year = (
        identify_agents_by_consecutive_duration(
            merged,
            distance_col="distance_to_first_care",
            duration_type="caregiving",
            last_group_at_least=True,
        )
    )

    def _profile(agents):
        sub = merged[merged["agent"].isin(agents)]
        return (
            sub.groupby("distance_to_first_care", observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values("distance_to_first_care")
        )

    prof_1 = _profile(agents_1_year)
    prof_2 = _profile(agents_2_year)
    prof_3 = _profile(agents_3_year)
    prof_4 = _profile(agents_4_year)
    return prof_diff, prof_1, prof_2, prof_3, prof_4


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_event_study_consecutive
    @pytask.mark.publication_full_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_consecutive_full_time")
    def task_plot_event_study_full_time_consecutive(  # noqa: PLR0912, PLR0915
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
        / "event_study"
        / "full_time"
        / (
            f"event_study_full_time_by_distance_to_first_care_spell_consecutive_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: full-time share difference by distance to first care spell (1–4 year exact consecutive)."""
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_ft, _ = calculate_simple_outcomes(df_o, "original")
        _, c_ft, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        prof_diff, prof_1, prof_2, prof_3, prof_4 = (
            _event_study_consecutive_merged_and_profiles(
                df_o,
                df_c,
                o_ft,
                c_ft,
                window,
                age_min,
                age_max,
            )
        )
        plot_outcome_difference_by_distance_consecutive(
            prof_diff=prof_diff,
            prof_1_year_diff=prof_1,
            prof_2_year_diff=prof_2,
            prof_3_year_diff=prof_3,
            prof_4_year_diff=prof_4,
            window=window,
            path_to_plot=path_to_plot,
            ylabel="Difference in full-time rate",
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_event_study_consecutive
    @pytask.mark.publication_part_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_consecutive_part_time")
    def task_plot_event_study_part_time_consecutive(  # noqa: PLR0912, PLR0915
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
        / "event_study"
        / "part_time"
        / (
            f"event_study_part_time_by_distance_to_first_care_spell_consecutive_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: part-time share difference by distance to first care spell (1–4 year exact consecutive)."""
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_pt = calculate_simple_outcomes(df_o, "original")
        _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
        prof_diff, prof_1, prof_2, prof_3, prof_4 = (
            _event_study_consecutive_merged_and_profiles(
                df_o,
                df_c,
                o_pt,
                c_pt,
                window,
                age_min,
                age_max,
            )
        )
        plot_outcome_difference_by_distance_consecutive(
            prof_diff=prof_diff,
            prof_1_year_diff=prof_1,
            prof_2_year_diff=prof_2,
            prof_3_year_diff=prof_3,
            prof_4_year_diff=prof_4,
            window=window,
            path_to_plot=path_to_plot,
            ylabel="Difference in part-time rate",
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_event_study_consecutive
    @pytask.mark.publication_working_hours
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_consecutive_working_hours")
    def task_plot_event_study_working_hours_consecutive(  # noqa: PLR0912, PLR0915
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
        / "event_study"
        / "working_hours"
        / (
            f"event_study_working_hours_weekly_by_distance_to_first_care_spell_consecutive_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: weekly working hours difference by distance to first care spell (1–4 year exact consecutive). Endogenous y scale."""
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
        prof_diff, prof_1, prof_2, prof_3, prof_4 = (
            _event_study_consecutive_merged_and_profiles(
                df_o,
                df_c,
                wh_o,
                wh_c,
                window,
                age_min,
                age_max,
            )
        )
        plot_outcome_difference_by_distance_consecutive(
            prof_diff=prof_diff,
            prof_1_year_diff=prof_1,
            prof_2_year_diff=prof_2,
            prof_3_year_diff=prof_3,
            prof_4_year_diff=prof_4,
            window=window,
            path_to_plot=path_to_plot,
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_event_study_consecutive
    @pytask.mark.publication_labor_income
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_consecutive_labor_income")
    def task_plot_event_study_labor_income_consecutive(  # noqa: PLR0912, PLR0915
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
        / "event_study"
        / "labor_income"
        / (
            f"event_study_monthly_gross_labor_income_by_distance_to_first_care_spell_consecutive_"
            f"{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: monthly gross labor income difference by distance to first care spell (1–4 year exact consecutive). Endogenous y scale."""
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
        prof_diff, prof_1, prof_2, prof_3, prof_4 = (
            _event_study_consecutive_merged_and_profiles(
                df_o,
                df_c,
                inc_o,
                inc_c,
                window,
                age_min,
                age_max,
            )
        )
        plot_outcome_difference_by_distance_consecutive(
            prof_diff=prof_diff,
            prof_1_year_diff=prof_1,
            prof_2_year_diff=prof_2,
            prof_3_year_diff=prof_3,
            prof_4_year_diff=prof_4,
            window=window,
            path_to_plot=path_to_plot,
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


def plot_employment_rate_difference_by_distance(  # noqa: PLR0913
    prof_diff,
    prof_1_year_diff,
    prof_2_year_diff,
    prof_3_year_diff,
    prof_4_year_diff,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
    xlabel: str = "Year relative to start of first care spell",
) -> None:
    """Plot employment rate difference by distance to first caregiving spell.

    Creates an event study plot showing the difference in employment rates
    between baseline and no-care-demand counterfactual, with separate lines
    for different caregiving durations (at least N years).

    Args:
        prof_diff: DataFrame with columns 'distance_to_first_care', 'diff'
            (raw difference in employment rate)
        prof_1_year_diff: DataFrame for at least 1-year caregivers
        prof_2_year_diff: DataFrame for at least 2-year caregivers
        prof_3_year_diff: DataFrame for at least 3-year caregivers
        prof_4_year_diff: DataFrame for at least 4-year caregivers
        window: Window size around event (e.g., 20 = -20 to +20 periods)
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
        xlabel: Label for x-axis (default: "Year relative to start of first care spell")
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
    plt.subplots_adjust(left=0.14, bottom=0.11)
    if path_to_plot:
        plt.savefig(path_to_plot, dpi=1200, bbox_inches="tight", pad_inches=0.25)
    plt.close()


def plot_employment_rate_difference_by_distance_consecutive(  # noqa: PLR0913
    prof_diff,
    prof_1_year_diff,
    prof_2_year_diff,
    prof_3_year_diff,
    prof_4_year_diff,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
    xlabel: str = "Year relative to start of first care spell",
) -> None:
    """Plot employment rate difference by distance (consecutive N years then stop).

    Creates an event study plot showing the difference in employment rates
    between baseline and no-care-demand counterfactual, with separate lines
    for different consecutive caregiving durations (N consecutive years then stop).

    Args:
        prof_diff: DataFrame with columns 'distance_to_first_care', 'diff'
            (raw difference in employment rate)
        prof_1_year_diff: DataFrame for 1-year consecutive caregivers
        prof_2_year_diff: DataFrame for 2-year consecutive caregivers
        prof_3_year_diff: DataFrame for 3-year consecutive caregivers
        prof_4_year_diff: DataFrame for 4-year consecutive caregivers
        window: Window size around event (e.g., 20 = -20 to +20 periods)
        path_to_plot: Optional path to save the plot. If None, plot is not saved.
        xlabel: Label for x-axis (default: "Year relative to start of first care spell")
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

    # Plot difference for 1-year exact consecutive caregiving spell
    if len(prof_1_year_diff) > 0:
        plt.plot(
            prof_1_year_diff["distance_to_first_care"],
            prof_1_year_diff["diff"],
            label="1-year exact consecutive caregiving spell",
            color="0.8",
            linewidth=2.0,
            linestyle="-",
            marker="8",  # Octagon
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 2-year exact consecutive caregiving spells
    if len(prof_2_year_diff) > 0:
        plt.plot(
            prof_2_year_diff["distance_to_first_care"],
            prof_2_year_diff["diff"],
            label="2-year exact consecutive caregiving spells",
            color="0.6",
            linewidth=2.0,
            linestyle="-",
            marker="^",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 3-year exact consecutive caregiving spells
    if len(prof_3_year_diff) > 0:
        plt.plot(
            prof_3_year_diff["distance_to_first_care"],
            prof_3_year_diff["diff"],
            label="3-year exact consecutive caregiving spells",
            color="0.4",
            linewidth=2.0,
            linestyle="-",
            marker="D",  # Diamond
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot difference for 4-year exact consecutive caregiving spells
    if len(prof_4_year_diff) > 0:
        plt.plot(
            prof_4_year_diff["distance_to_first_care"],
            prof_4_year_diff["diff"],
            label="4-year exact consecutive caregiving spells",
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
    plt.subplots_adjust(left=0.14, bottom=0.11)
    if path_to_plot:
        plt.savefig(path_to_plot, dpi=1200, bbox_inches="tight", pad_inches=0.25)
    plt.close()


def plot_outcome_difference_by_distance_consecutive(  # noqa: PLR0913
    prof_diff,
    prof_1_year_diff,
    prof_2_year_diff,
    prof_3_year_diff,
    prof_4_year_diff,
    window: int = 20,
    path_to_plot: Optional[Path] = None,
    xlabel: str = "Year relative to start of first care spell",
    ylabel: str = "Difference in outcome",
    endogenous_ylim: bool = False,
) -> None:
    """Plot outcome difference by distance (1–4 year exact consecutive caregiving).

    Same layout as employment consecutive: dashed baseline (average caregiver),
    plus 1–4 year exact consecutive caregiving spell(s) groups.
    When endogenous_ylim is True, y-axis scale is determined by data (no 0-centered symmetric range).
    """
    plt.figure(figsize=(14, 8))

    plt.plot(
        prof_diff["distance_to_first_care"],
        prof_diff["diff"],
        label="Baseline",
        color="black",
        linewidth=2.0,
        linestyle="--",
        marker=None,
    )
    plt.axhline(y=0, color="k", linestyle="-", linewidth=0.8, alpha=0.5)

    if len(prof_1_year_diff) > 0:
        plt.plot(
            prof_1_year_diff["distance_to_first_care"],
            prof_1_year_diff["diff"],
            label="1-year exact consecutive caregiving spell",
            color="0.8",
            linewidth=2.0,
            linestyle="-",
            marker="8",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )
    if len(prof_2_year_diff) > 0:
        plt.plot(
            prof_2_year_diff["distance_to_first_care"],
            prof_2_year_diff["diff"],
            label="2-year exact consecutive caregiving spells",
            color="0.6",
            linewidth=2.0,
            linestyle="-",
            marker="^",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )
    if len(prof_3_year_diff) > 0:
        plt.plot(
            prof_3_year_diff["distance_to_first_care"],
            prof_3_year_diff["diff"],
            label="3-year exact consecutive caregiving spells",
            color="0.4",
            linewidth=2.0,
            linestyle="-",
            marker="D",
            markersize=5,
            markevery=1,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )
    if len(prof_4_year_diff) > 0:
        plt.plot(
            prof_4_year_diff["distance_to_first_care"],
            prof_4_year_diff["diff"],
            label="4-year exact consecutive caregiving spells",
            color="0.2",
            linewidth=2.0,
            linestyle="-",
            marker="s",
            markersize=5,
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

    # Use only finite values for y-axis limits (NaN/Inf break matplotlib)
    finite_diffs = [x for x in all_diffs if np.isfinite(x)]
    if finite_diffs:
        if endogenous_ylim:
            y_min, y_max = min(finite_diffs), max(finite_diffs)
            pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
            plt.ylim(y_min - pad, y_max + pad)
        else:
            y_max = max(abs(min(finite_diffs)), abs(max(finite_diffs)))
            y_lim = (int(y_max * 1.1 / 0.05) + 1) * 0.05
            y_lim = max(y_lim, 0.05)
            plt.ylim(-y_lim, y_lim)
    else:
        plt.ylim(-0.1, 0.1)

    plt.grid(True, axis="y", alpha=0.3, linewidth=0.8)
    plt.xticks(range(-window, window + 1, 5), fontsize=12)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=8)
    plt.tight_layout()
    plt.subplots_adjust(left=0.14, bottom=0.11)
    if path_to_plot:
        plt.savefig(path_to_plot, dpi=1200, bbox_inches="tight", pad_inches=0.25)
    plt.close()
