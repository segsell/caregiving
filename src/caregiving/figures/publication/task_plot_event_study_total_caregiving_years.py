"""Plot event study (difference) by distance, grouped by total caregiving years (1–5+).

Event study plots in the style of task_plot_event_study_employment_rate_consecutive:
difference in outcome (baseline minus no-care-demand) by distance to event (t=0).
Setup from task_plot_employment_rate_by_distance_to_first_care tasks with
publication_employment_check and publication_other_check: total care years 1, 2, 3, 4, 5+
over lifecycle; event = first care demand or first caregiving spell; data = estimated_params
or back_to_Jan7. Outputs go into event_study/{outcome}/total_caregiving_years/ with
dataset and event in function names and filenames.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    AGE_GROUPS_EVENT_STUDY,
    add_distance_to_first_care,
    add_distance_to_first_care_demand,
    calculate_simple_outcomes,
    event_study_total_caregiving_merged_and_profiles,
    get_age_at_first_event,
    identify_agents_by_total_caregiving_over_lifecycle,
    job_offer_outcome_series,
    plot_outcome_difference_by_distance_total_caregiving,
    prepare_dataframes_simple,
)

# ---------------------------------------------------------------------------
# Employment event study: first care demand, standard data
# One loop over age groups; path_to_plot and args written in the function signature.
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.explore
    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_employment_first_care_demand_estimated_params")
    def task_plot_event_study_employment_rate_by_distance_to_first_care_demand_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_Feb16.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_Feb16.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "employment"
        / "total_caregiving_years"
        / (
            f"event_study_employment_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 15,
        window_high: int = 15,
        window_by_age: dict[str, tuple[int, int]] | None = (
            {"ages_40_49": (10, 15), "ages_60_70": (15, 10)}
        ),
    ) -> None:
        """Event study: employment rate difference by distance to first care demand (total care years 1–5+).

        window_by_age overrides window_low/window_high per age group; keys are AGE_GROUPS_EVENT_STUDY
        labels ("all_ages", "ages_40_49", "ages_50_59", "ages_60_70"); value is (window_low, window_high).
        """

        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)

        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_work,
                c_work,
                w_low,
                w_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Employment event study: first care demand, back_to_Jan7 data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_employment_first_care_demand_back_to_Jan7")
    def task_plot_event_study_employment_rate_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "employment"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_employment_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: employment rate difference by distance to first care demand (total care years 1–5+), back_to_Jan7 data."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_work,
                c_work,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Employment event study: first caregiving spell, standard data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_employment_first_caregiving_spell_estimated_params"
    )
    def task_plot_event_study_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "employment"
        / "total_caregiving_years"
        / (
            f"event_study_employment_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: employment rate difference by distance to first caregiving spell (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_work,
                c_work,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Employment event study: first caregiving spell, back_to_Jan7 data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_employment_first_caregiving_spell_back_to_Jan7")
    def task_plot_event_study_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "employment"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_employment_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: employment rate difference by distance to first caregiving spell (total care years 1–5+), back_to_Jan7 data."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_work,
                c_work,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Full-time event study: first care demand, standard data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_full_time_first_care_demand_estimated_params")
    def task_plot_event_study_full_time_by_distance_to_first_care_demand_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"event_study_full_time_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: full-time rate difference by distance to first care demand (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_out, _ = calculate_simple_outcomes(df_o, "original")
        _, c_out, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in full-time rate",
        )


# Full-time: first care demand, back_to_Jan7
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_full_time_first_care_demand_back_to_Jan7")
    def task_plot_event_study_full_time_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_full_time_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: full-time rate difference by distance to first care demand (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_out, _ = calculate_simple_outcomes(df_o, "original")
        _, c_out, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in full-time rate",
        )


# Full-time: first caregiving spell, standard
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_full_time_first_caregiving_spell_estimated_params"
    )
    def task_plot_event_study_full_time_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"event_study_full_time_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: full-time rate difference by distance to first caregiving spell (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_out, _ = calculate_simple_outcomes(df_o, "original")
        _, c_out, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in full-time rate",
        )


# Full-time: first caregiving spell, back_to_Jan7
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_full_time_first_caregiving_spell_back_to_Jan7")
    def task_plot_event_study_full_time_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_full_time_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: full-time rate difference by distance to first caregiving spell (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_out, _ = calculate_simple_outcomes(df_o, "original")
        _, c_out, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in full-time rate",
        )


# ---------------------------------------------------------------------------
# Part-time event study: first care demand (standard), back_to_Jan7, caregiving spell (standard), caregiving spell back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_part_time_first_care_demand_estimated_params")
    def task_plot_event_study_part_time_by_distance_to_first_care_demand_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "part_time"
        / "total_caregiving_years"
        / (
            f"event_study_part_time_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: part-time rate difference by distance to first care demand (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_out = calculate_simple_outcomes(df_o, "original")
        _, _, c_out = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in part-time rate",
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_part_time_first_care_demand_back_to_Jan7")
    def task_plot_event_study_part_time_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "part_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_part_time_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: part-time rate difference by distance to first care demand (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_out = calculate_simple_outcomes(df_o, "original")
        _, _, c_out = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in part-time rate",
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_part_time_first_caregiving_spell_estimated_params"
    )
    def task_plot_event_study_part_time_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "part_time"
        / "total_caregiving_years"
        / (
            f"event_study_part_time_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: part-time rate difference by distance to first caregiving spell (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_out = calculate_simple_outcomes(df_o, "original")
        _, _, c_out = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in part-time rate",
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_part_time_first_caregiving_spell_back_to_Jan7")
    def task_plot_event_study_part_time_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "part_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_part_time_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: part-time rate difference by distance to first caregiving spell (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_out = calculate_simple_outcomes(df_o, "original")
        _, _, c_out = calculate_simple_outcomes(df_c, "no_care_demand")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in part-time rate",
        )


# ---------------------------------------------------------------------------
# Working hours event study: 4 variants (care_demand standard/back_to_Jan7, caregiving_spell standard/back_to_Jan7)
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_working_hours_first_care_demand_estimated_params")
    def task_plot_event_study_working_hours_by_distance_to_first_care_demand_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"event_study_working_hours_weekly_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: weekly working hours difference by distance to first care demand (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = (
            df_o["working_hours"].astype(float) / 52.0
            if "working_hours" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        c_out = (
            df_c["working_hours"].astype(float) / 52.0
            if "working_hours" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_working_hours_first_care_demand_back_to_Jan7")
    def task_plot_event_study_working_hours_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_working_hours_weekly_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: weekly working hours difference by distance to first care demand (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = (
            df_o["working_hours"].astype(float) / 52.0
            if "working_hours" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        c_out = (
            df_c["working_hours"].astype(float) / 52.0
            if "working_hours" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_working_hours_first_caregiving_spell_estimated_params"
    )
    def task_plot_event_study_working_hours_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"event_study_working_hours_weekly_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: weekly working hours difference by distance to first caregiving spell (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = (
            df_o["working_hours"].astype(float) / 52.0
            if "working_hours" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        c_out = (
            df_c["working_hours"].astype(float) / 52.0
            if "working_hours" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_working_hours_first_caregiving_spell_back_to_Jan7"
    )
    def task_plot_event_study_working_hours_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_working_hours_weekly_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: weekly working hours difference by distance to first caregiving spell (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = (
            df_o["working_hours"].astype(float) / 52.0
            if "working_hours" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        c_out = (
            df_c["working_hours"].astype(float) / 52.0
            if "working_hours" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


# ---------------------------------------------------------------------------
# Labor income event study: 4 variants
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_labor_income_first_care_demand_estimated_params")
    def task_plot_event_study_labor_income_by_distance_to_first_care_demand_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"event_study_monthly_gross_labor_income_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: monthly gross labor income difference by distance to first care demand (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = (
            df_o["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        c_out = (
            df_c["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_labor_income_first_care_demand_back_to_Jan7")
    def task_plot_event_study_labor_income_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_monthly_gross_labor_income_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: monthly gross labor income difference by distance to first care demand (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = (
            df_o["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        c_out = (
            df_c["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_labor_income_first_caregiving_spell_estimated_params"
    )
    def task_plot_event_study_labor_income_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"event_study_monthly_gross_labor_income_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: monthly gross labor income difference by distance to first caregiving spell (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = (
            df_o["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        c_out = (
            df_c["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_labor_income_first_caregiving_spell_back_to_Jan7")
    def task_plot_event_study_labor_income_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_monthly_gross_labor_income_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: monthly gross labor income difference by distance to first caregiving spell (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = (
            df_o["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_o.columns
            else pd.Series(0.0, index=df_o.index)
        )
        c_out = (
            df_c["gross_labor_income"].astype(float) / 12.0
            if "gross_labor_income" in df_c.columns
            else pd.Series(0.0, index=df_c.index)
        )
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


# ---------------------------------------------------------------------------
# Job finding rate event study (job_offer | previously not working, not retired)
# ---------------------------------------------------------------------------
# Job finding: first care demand, standard data
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_job_finding_first_care_demand_estimated_params")
    def task_plot_event_study_job_finding_by_distance_to_first_care_demand_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "job_finding"
        / "total_caregiving_years"
        / (
            f"event_study_job_finding_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: job finding rate difference by distance to first care demand (total care years 1–5+). Conditional on previously not working, not retired."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = job_offer_outcome_series(df_o, "job_finding")
        c_out = job_offer_outcome_series(df_c, "job_finding")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in job finding rate",
        )


# Job finding: first care demand, back_to_Jan7
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_job_finding_first_care_demand_back_to_Jan7")
    def task_plot_event_study_job_finding_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "job_finding"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_job_finding_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: job finding rate difference by distance to first care demand (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = job_offer_outcome_series(df_o, "job_finding")
        c_out = job_offer_outcome_series(df_c, "job_finding")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in job finding rate",
        )


# Job finding: first caregiving spell, standard data
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_job_finding_first_caregiving_spell_estimated_params"
    )
    def task_plot_event_study_job_finding_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "job_finding"
        / "total_caregiving_years"
        / (
            f"event_study_job_finding_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: job finding rate difference by distance to first caregiving spell (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = job_offer_outcome_series(df_o, "job_finding")
        c_out = job_offer_outcome_series(df_c, "job_finding")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in job finding rate",
        )


# Job finding: first caregiving spell, back_to_Jan7
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_job_finding_first_caregiving_spell_back_to_Jan7")
    def task_plot_event_study_job_finding_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "job_finding"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_job_finding_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: job finding rate difference by distance to first caregiving spell (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = job_offer_outcome_series(df_o, "job_finding")
        c_out = job_offer_outcome_series(df_c, "job_finding")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in job finding rate",
        )


# ---------------------------------------------------------------------------
# Job retention rate event study (job_offer | previously working)
# ---------------------------------------------------------------------------
# Job retention: first care demand, standard data
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_job_retention_first_care_demand_estimated_params")
    def task_plot_event_study_job_retention_by_distance_to_first_care_demand_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "job_retention"
        / "total_caregiving_years"
        / (
            f"event_study_job_retention_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: job retention rate (1 - separation) difference by distance to first care demand (total care years 1–5+). Conditional on previously working."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = job_offer_outcome_series(df_o, "job_retention")
        c_out = job_offer_outcome_series(df_c, "job_retention")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in job retention rate",
        )


# Job retention: first care demand, back_to_Jan7
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_job_retention_first_care_demand_back_to_Jan7")
    def task_plot_event_study_job_retention_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "job_retention"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_job_retention_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: job retention rate difference by distance to first care demand (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = job_offer_outcome_series(df_o, "job_retention")
        c_out = job_offer_outcome_series(df_c, "job_retention")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in job retention rate",
        )


# Job retention: first caregiving spell, standard data
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_job_retention_first_caregiving_spell_estimated_params"
    )
    def task_plot_event_study_job_retention_by_distance_to_first_caregiving_spell_total_caregiving(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "job_retention"
        / "total_caregiving_years"
        / (
            f"event_study_job_retention_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: job retention rate difference by distance to first caregiving spell (total care years 1–5+)."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = job_offer_outcome_series(df_o, "job_retention")
        c_out = job_offer_outcome_series(df_c, "job_retention")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in job retention rate",
        )


# Job retention: first caregiving spell, back_to_Jan7
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_job_retention_first_caregiving_spell_back_to_Jan7"
    )
    def task_plot_event_study_job_retention_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "event_study"
        / "job_retention"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_job_retention_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
    ) -> None:
        """Event study: job retention rate difference by distance to first caregiving spell (total care years 1–5+), back_to_Jan7."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_out = job_offer_outcome_series(df_o, "job_retention")
        c_out = job_offer_outcome_series(df_c, "job_retention")
        _, prof_diff, p1, p2, p3, p4, p5 = (
            event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window_low,
                window_high,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=False,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window_low=window_low,
            window_high=window_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in job retention rate",
        )
