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
from typing import Annotated, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    add_distance_to_first_care,
    add_distance_to_first_care_demand,
    calculate_simple_outcomes,
    get_age_at_first_event,
    identify_agents_by_total_caregiving_over_lifecycle,
    prepare_dataframes_simple,
)
from caregiving.model.shared import INFORMAL_CARE

# Distance column name used in profile DataFrames (for plotting)
_DIST_COL = "distance_to_first_care"

# Age groups: (age_min, age_max, age_label) for task id and path_to_plot filename
_AGE_GROUPS = (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
)


# ---------------------------------------------------------------------------
# Employment event study: first care demand, standard data
# One loop over age groups; path_to_plot and args written in the function signature.
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_employment_first_care_demand_estimated_params")
    def task_plot_event_study_employment_rate_by_distance_to_first_care_demand_total_caregiving(  # noqa: E501
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
            f"event_study_employment_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: employment rate difference by distance to first care demand (total care years 1–5+)."""
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Employment event study: first care demand, back_to_Jan7 data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Employment event study: first caregiving spell, standard data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Employment event study: first caregiving spell, back_to_Jan7 data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Full-time event study: first care demand, standard data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in full-time rate",
        )


# Full-time: first care demand, back_to_Jan7
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in full-time rate",
        )


# Full-time: first caregiving spell, standard
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in full-time rate",
        )


# Full-time: first caregiving spell, back_to_Jan7
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in full-time rate",
        )


# ---------------------------------------------------------------------------
# Part-time event study: first care demand (standard), back_to_Jan7, caregiving spell (standard), caregiving spell back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in part-time rate",
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in part-time rate",
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in part-time rate",
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in part-time rate",
        )


# ---------------------------------------------------------------------------
# Working hours event study: 4 variants (care_demand standard/back_to_Jan7, caregiving_spell standard/back_to_Jan7)
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


# ---------------------------------------------------------------------------
# Labor income event study: 4 variants
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first care demand",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

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
        window: int = 20,
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
            )
        )
        plot_outcome_difference_by_distance_total_caregiving(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to start of first caregiving spell",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


def plot_outcome_difference_by_distance_total_caregiving(  # noqa: PLR0913
    prof_diff: pd.DataFrame,
    prof_1_year_diff: pd.DataFrame,
    prof_2_year_diff: pd.DataFrame,
    prof_3_year_diff: pd.DataFrame,
    prof_4_year_diff: pd.DataFrame,
    prof_5_year_diff: pd.DataFrame,
    window: int = 20,
    path_to_plot: Path | None = None,
    xlabel: str = "Year relative to start of first care spell",
    ylabel: str = "Difference in outcome",
    endogenous_ylim: bool = False,
) -> None:
    """Plot outcome difference by distance with 5 lines: total care years 1, 2, 3, 4, 5+.

    Same layout as event study consecutive: dashed black baseline, horizontal line at 0,
    vertical line at t=-0.5, five subgroup lines (1, 2, 3, 4, 5+ total care years).
    Profile DataFrames must have column _DIST_COL and 'diff'.
    """
    plt.figure(figsize=(14, 8))

    plt.plot(
        prof_diff[_DIST_COL],
        prof_diff["diff"],
        label="Baseline",
        color="black",
        linewidth=2.0,
        linestyle="--",
        marker=None,
    )
    plt.axhline(y=0, color="k", linestyle="-", linewidth=0.8, alpha=0.5)

    def _plot_prof(prof: pd.DataFrame, label: str, color: str, marker: str) -> None:
        if len(prof) > 0:
            plt.plot(
                prof[_DIST_COL],
                prof["diff"],
                label=label,
                color=color,
                linewidth=2.0,
                linestyle="-",
                marker=marker,
                markersize=5,
                markevery=1,
                markerfacecolor="none",
                markeredgewidth=1.5,
            )

    _plot_prof(prof_1_year_diff, "1 total care year", "0.9", "8")
    _plot_prof(prof_2_year_diff, "2 total care years", "0.7", "^")
    _plot_prof(prof_3_year_diff, "3 total care years", "0.5", "D")
    _plot_prof(prof_4_year_diff, "4 total care years", "0.3", "s")
    _plot_prof(prof_5_year_diff, "5+ total care years", "0.1", "v")

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
    for p in (
        prof_diff,
        prof_1_year_diff,
        prof_2_year_diff,
        prof_3_year_diff,
        prof_4_year_diff,
        prof_5_year_diff,
    ):
        if len(p) > 0 and "diff" in p.columns:
            all_diffs.extend(p["diff"].tolist())
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
    if path_to_plot:
        path_to_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_plot, dpi=1200, bbox_inches="tight")
    plt.close()


def event_study_total_caregiving_merged_and_profiles(
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    outcome_o_series: pd.Series,
    outcome_c_series: pd.Series,
    window: int,
    age_min: int | None,
    age_max: int | None,
    event_type: Literal["care_demand", "caregiving_spell"],
    start_age: int,
    end_age_caregiving: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build merged df and profile diffs for event study (total care years 1–5+).

    Returns (merged, prof_diff, prof_1_year_diff, ..., prof_5_year_diff).
    All profiles have columns _DIST_COL and 'diff'.
    """
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    o_cols = df_o[["agent", "period", "choice"]].copy()
    o_cols["outcome_o"] = np.asarray(outcome_o_series).astype(float)
    o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    c_cols = df_c[["agent", "period"]].copy()
    c_cols["outcome_c"] = np.asarray(outcome_c_series).astype(float)
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")

    if event_type == "care_demand":
        merged = merged.merge(
            df_o[["agent", "period", "care_demand"]],
            on=["agent", "period"],
            how="left",
        )
        df_o_dist = add_distance_to_first_care_demand(df_o)
        dist_col_raw = "first_care_demand_period"
        age_col = "age_at_first_care_demand"
        care_demand_mask = df_o["care_demand"] > 0
        first_event = get_age_at_first_event(
            df_o, care_demand_mask, "age_at_first_care_demand"
        )
    else:
        df_o_dist = add_distance_to_first_care(df_o)
        dist_col_raw = "first_care_period"
        age_col = "age_at_first_care"
        caregiving_mask = df_o["choice"].isin(care_codes)
        first_event = get_age_at_first_event(df_o, caregiving_mask, "age_at_first_care")

    dist_map = (
        df_o_dist.groupby("agent", observed=False)[dist_col_raw].first().reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_raw"] = merged["period"] - merged[dist_col_raw]
    merged = merged.merge(first_event, on="agent", how="left")

    merged = merged[
        merged[dist_col_raw].notna()
        & (merged["distance_raw"] >= -window)
        & (merged["distance_raw"] <= window)
    ]
    if age_min is not None:
        merged = merged[merged[age_col] >= age_min].copy()
    if age_max is not None:
        merged = merged[merged[age_col] <= age_max].copy()

    merged["diff"] = merged["outcome_o"] - merged["outcome_c"]
    merged[_DIST_COL] = merged["distance_raw"]

    prof_diff = (
        merged.groupby(_DIST_COL, observed=False)["diff"]
        .mean()
        .reset_index()
        .sort_values(_DIST_COL)
    )

    (
        agents_1_year,
        agents_2_year,
        agents_3_year,
        agents_4_year,
        agents_5_year,
    ) = identify_agents_by_total_caregiving_over_lifecycle(
        df_o, start_age, end_age_caregiving
    )

    def _prof_for_agents(agents: np.ndarray) -> pd.DataFrame:
        m = merged[merged["agent"].isin(agents)]
        if len(m) == 0:
            return pd.DataFrame(columns=[_DIST_COL, "diff"])
        p = (
            m.groupby(_DIST_COL, observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values(_DIST_COL)
        )
        return p

    prof_1 = _prof_for_agents(agents_1_year)
    prof_2 = _prof_for_agents(agents_2_year)
    prof_3 = _prof_for_agents(agents_3_year)
    prof_4 = _prof_for_agents(agents_4_year)
    prof_5 = _prof_for_agents(agents_5_year)

    return merged, prof_diff, prof_1, prof_2, prof_3, prof_4, prof_5
