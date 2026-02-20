"""Plot event study (baseline minus caregiving leave 65%) by distance, total caregiving years 1–5+.

Same structure as task_plot_event_study_total_caregiving_years but compares baseline
(estimated_params) vs caregiving leave with job retention counterfactual.
Outputs: event_study_caregiving_leave/{outcome}/total_caregiving_years/.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    calculate_simple_outcomes,
    prepare_dataframes_simple,
)
from caregiving.counterfactual.plotting_helpers import (
    AGE_GROUPS_EVENT_STUDY,
    event_study_total_caregiving_merged_and_profiles,
    job_offer_outcome_series,
    plot_outcome_difference_by_distance_total_caregiving,
)

_PATH_ORIGINAL = BLD / "solve_and_simulate" / "simulated_data_estimated_params.pkl"
_PATH_LEAVE = (
    BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl"
)
_PATH_SPECS = BLD / "model" / "specs" / "specs_full.pkl"
_PLOT_PREFIX = "event_study_caregiving_leave"
_TASK_ID_SUFFIX = "caregiving_leave"


def _path_plot(subdir: str, filename: str) -> Path:
    return (
        BLD
        / "figures"
        / "publication"
        / _PLOT_PREFIX
        / subdir
        / "total_caregiving_years"
        / filename
    )


# ---------------------------------------------------------------------------
# Employment
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study_caregiving_leave
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_employment_first_care_demand_{_TASK_ID_SUFFIX}")
    def task_plot_event_study_employment_by_distance_to_first_care_demand_caregiving_leave(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = _PATH_ORIGINAL,
        path_to_leave_data: Path = _PATH_LEAVE,
        path_to_specs: Path = _PATH_SPECS,
        path_to_plot: Annotated[Path, Product] = _path_plot(
            "employment",
            f"event_study_employment_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf",
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
        compare_against_baseline: bool = True,
    ) -> None:
        """Event study: employment rate diff (policy − baseline) by distance to first care demand."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_leave_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "job_retention")
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
                compare_against_baseline=compare_against_baseline,
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


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study_caregiving_leave
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_employment_first_caregiving_spell_{_TASK_ID_SUFFIX}"
    )
    def task_plot_event_study_employment_by_distance_to_first_caregiving_spell_caregiving_leave(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = _PATH_ORIGINAL,
        path_to_leave_data: Path = _PATH_LEAVE,
        path_to_specs: Path = _PATH_SPECS,
        path_to_plot: Annotated[Path, Product] = _path_plot(
            "employment",
            f"event_study_employment_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf",
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
        compare_against_baseline: bool = True,
    ) -> None:
        """Event study: employment rate diff (policy − baseline) by distance to first caregiving spell."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_leave_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "job_retention")
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
                compare_against_baseline=compare_against_baseline,
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


# # ---------------------------------------------------------------------------
# # Full-time
# # ---------------------------------------------------------------------------
# for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

#     @pytask.mark.publication_event_study_caregiving_leave
#     @pytask.mark.publication
#     @pytask.task(id=f"{age_label_val}_full_time_first_care_demand_{_TASK_ID_SUFFIX}")
#     def task_plot_event_study_full_time_by_distance_to_first_care_demand_caregiving_leave(  # noqa: E501
#         age_min: int | None = age_min_val,
#         age_max: int | None = age_max_val,
#         path_to_original_data: Path = _PATH_ORIGINAL,
#         path_to_leave_data: Path = _PATH_LEAVE,
#         path_to_specs: Path = _PATH_SPECS,
#         path_to_plot: Annotated[Path, Product] = _path_plot(
#             "full_time",
#             f"event_study_full_time_by_distance_to_first_care_demand_"
#             f"total_caregiving_{age_label_val}.pdf",
#         ),
#         ever_caregivers: bool = True,
#         ever_care_demand: bool = False,
#         window: int = 20,
#         compare_against_baseline: bool = True,
#     ) -> None:
#         """Event study: full-time rate diff (policy − baseline) by distance to first care demand."""
#         with path_to_specs.open("rb") as f:
#             specs = pickle.load(f)
#         start_age = int(specs["start_age"])
#         end_age_caregiving = int(specs["end_age_caregiving"])
#         df_o, df_c = prepare_dataframes_simple(
#             pd.read_pickle(path_to_original_data),
#             pd.read_pickle(path_to_leave_data),
#             ever_caregivers,
#             ever_care_demand,
#         )
#         _, o_out, _ = calculate_simple_outcomes(df_o, "original")
#         _, c_out, _ = calculate_simple_outcomes(df_c, "job_retention")
#         _, prof_diff, p1, p2, p3, p4, p5 = (
#             event_study_total_caregiving_merged_and_profiles(
#                 df_o,
#                 df_c,
#                 o_out,
#                 c_out,
#                 window,
#                 age_min,
#                 age_max,
#                 "care_demand",
#                 start_age,
#                 end_age_caregiving,
#                 compare_against_baseline=compare_against_baseline,
#             )
#         )
#         plot_outcome_difference_by_distance_total_caregiving(
#             prof_diff=prof_diff,
#             prof_1_year_diff=p1,
#             prof_2_year_diff=p2,
#             prof_3_year_diff=p3,
#             prof_4_year_diff=p4,
#             prof_5_year_diff=p5,
#             window=window,
#             path_to_plot=path_to_plot,
#             xlabel="Year relative to start of first care demand",
#             ylabel="Difference in full-time rate",
#         )


# for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

#     @pytask.mark.publication_event_study_caregiving_leave
#     @pytask.mark.publication
#     @pytask.task(
#         id=f"{age_label_val}_full_time_first_caregiving_spell_{_TASK_ID_SUFFIX}"
#     )
#     def task_plot_event_study_full_time_by_distance_to_first_caregiving_spell_caregiving_leave(  # noqa: E501
#         age_min: int | None = age_min_val,
#         age_max: int | None = age_max_val,
#         path_to_original_data: Path = _PATH_ORIGINAL,
#         path_to_leave_data: Path = _PATH_LEAVE,
#         path_to_specs: Path = _PATH_SPECS,
#         path_to_plot: Annotated[Path, Product] = _path_plot(
#             "full_time",
#             f"event_study_full_time_by_distance_to_first_caregiving_spell_"
#             f"total_caregiving_{age_label_val}.pdf",
#         ),
#         ever_caregivers: bool = True,
#         ever_care_demand: bool = False,
#         window: int = 20,
#         compare_against_baseline: bool = True,
#     ) -> None:
#         """Event study: full-time rate diff (policy − baseline) by distance to first caregiving spell."""
#         with path_to_specs.open("rb") as f:
#             specs = pickle.load(f)
#         start_age = int(specs["start_age"])
#         end_age_caregiving = int(specs["end_age_caregiving"])
#         df_o, df_c = prepare_dataframes_simple(
#             pd.read_pickle(path_to_original_data),
#             pd.read_pickle(path_to_leave_data),
#             ever_caregivers,
#             ever_care_demand,
#         )
#         _, o_out, _ = calculate_simple_outcomes(df_o, "original")
#         _, c_out, _ = calculate_simple_outcomes(df_c, "job_retention")
#         _, prof_diff, p1, p2, p3, p4, p5 = (
#             event_study_total_caregiving_merged_and_profiles(
#                 df_o,
#                 df_c,
#                 o_out,
#                 c_out,
#                 window,
#                 age_min,
#                 age_max,
#                 "caregiving_spell",
#                 start_age,
#                 end_age_caregiving,
#                 compare_against_baseline=compare_against_baseline,
#             )
#         )
#         plot_outcome_difference_by_distance_total_caregiving(
#             prof_diff=prof_diff,
#             prof_1_year_diff=p1,
#             prof_2_year_diff=p2,
#             prof_3_year_diff=p3,
#             prof_4_year_diff=p4,
#             prof_5_year_diff=p5,
#             window=window,
#             path_to_plot=path_to_plot,
#             xlabel="Year relative to start of first caregiving spell",
#             ylabel="Difference in full-time rate",
#         )


# # ---------------------------------------------------------------------------
# # Part-time
# # ---------------------------------------------------------------------------
# for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

#     @pytask.mark.publication_event_study_caregiving_leave
#     @pytask.mark.publication_counterfactual
#     @pytask.mark.publication
#     @pytask.task(id=f"{age_label_val}_part_time_first_care_demand_{_TASK_ID_SUFFIX}")
#     def task_plot_event_study_part_time_by_distance_to_first_care_demand_caregiving_leave(  # noqa: E501
#         age_min: int | None = age_min_val,
#         age_max: int | None = age_max_val,
#         path_to_original_data: Path = _PATH_ORIGINAL,
#         path_to_leave_data: Path = _PATH_LEAVE,
#         path_to_specs: Path = _PATH_SPECS,
#         path_to_plot: Annotated[Path, Product] = _path_plot(
#             "part_time",
#             f"event_study_part_time_by_distance_to_first_care_demand_"
#             f"total_caregiving_{age_label_val}.pdf",
#         ),
#         ever_caregivers: bool = True,
#         ever_care_demand: bool = False,
#         window: int = 20,
#         compare_against_baseline: bool = True,
#     ) -> None:
#         """Event study: part-time rate diff (policy − baseline) by distance to first care demand."""
#         with path_to_specs.open("rb") as f:
#             specs = pickle.load(f)
#         start_age = int(specs["start_age"])
#         end_age_caregiving = int(specs["end_age_caregiving"])
#         df_o, df_c = prepare_dataframes_simple(
#             pd.read_pickle(path_to_original_data),
#             pd.read_pickle(path_to_leave_data),
#             ever_caregivers,
#             ever_care_demand,
#         )
#         _, _, o_out = calculate_simple_outcomes(df_o, "original")
#         _, _, c_out = calculate_simple_outcomes(df_c, "job_retention")
#         _, prof_diff, p1, p2, p3, p4, p5 = (
#             event_study_total_caregiving_merged_and_profiles(
#                 df_o,
#                 df_c,
#                 o_out,
#                 c_out,
#                 window,
#                 age_min,
#                 age_max,
#                 "care_demand",
#                 start_age,
#                 end_age_caregiving,
#                 compare_against_baseline=compare_against_baseline,
#             )
#         )
#         plot_outcome_difference_by_distance_total_caregiving(
#             prof_diff=prof_diff,
#             prof_1_year_diff=p1,
#             prof_2_year_diff=p2,
#             prof_3_year_diff=p3,
#             prof_4_year_diff=p4,
#             prof_5_year_diff=p5,
#             window=window,
#             path_to_plot=path_to_plot,
#             xlabel="Year relative to start of first care demand",
#             ylabel="Difference in part-time rate",
#         )


# for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

#     @pytask.mark.publication_event_study_caregiving_leave
#     @pytask.mark.publication
#     @pytask.task(
#         id=f"{age_label_val}_part_time_first_caregiving_spell_{_TASK_ID_SUFFIX}"
#     )
#     def task_plot_event_study_part_time_by_distance_to_first_caregiving_spell_caregiving_leave(  # noqa: E501
#         age_min: int | None = age_min_val,
#         age_max: int | None = age_max_val,
#         path_to_original_data: Path = _PATH_ORIGINAL,
#         path_to_leave_data: Path = _PATH_LEAVE,
#         path_to_specs: Path = _PATH_SPECS,
#         path_to_plot: Annotated[Path, Product] = _path_plot(
#             "part_time",
#             f"event_study_part_time_by_distance_to_first_caregiving_spell_"
#             f"total_caregiving_{age_label_val}.pdf",
#         ),
#         ever_caregivers: bool = True,
#         ever_care_demand: bool = False,
#         window: int = 20,
#         compare_against_baseline: bool = True,
#     ) -> None:
#         """Event study: part-time rate diff (policy − baseline) by distance to first caregiving spell."""
#         with path_to_specs.open("rb") as f:
#             specs = pickle.load(f)
#         start_age = int(specs["start_age"])
#         end_age_caregiving = int(specs["end_age_caregiving"])
#         df_o, df_c = prepare_dataframes_simple(
#             pd.read_pickle(path_to_original_data),
#             pd.read_pickle(path_to_leave_data),
#             ever_caregivers,
#             ever_care_demand,
#         )
#         _, _, o_out = calculate_simple_outcomes(df_o, "original")
#         _, _, c_out = calculate_simple_outcomes(df_c, "job_retention")
#         _, prof_diff, p1, p2, p3, p4, p5 = (
#             event_study_total_caregiving_merged_and_profiles(
#                 df_o,
#                 df_c,
#                 o_out,
#                 c_out,
#                 window,
#                 age_min,
#                 age_max,
#                 "caregiving_spell",
#                 start_age,
#                 end_age_caregiving,
#                 compare_against_baseline=compare_against_baseline,
#             )
#         )
#         plot_outcome_difference_by_distance_total_caregiving(
#             prof_diff=prof_diff,
#             prof_1_year_diff=p1,
#             prof_2_year_diff=p2,
#             prof_3_year_diff=p3,
#             prof_4_year_diff=p4,
#             prof_5_year_diff=p5,
#             window=window,
#             path_to_plot=path_to_plot,
#             xlabel="Year relative to start of first caregiving spell",
#             ylabel="Difference in part-time rate",
#         )


# # ---------------------------------------------------------------------------
# # Working hours
# # ---------------------------------------------------------------------------
# for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

#     @pytask.mark.publication_event_study_caregiving_leave
#     @pytask.mark.publication
#     @pytask.task(
#         id=f"{age_label_val}_working_hours_first_care_demand_{_TASK_ID_SUFFIX}"
#     )
#     def task_plot_event_study_working_hours_by_distance_to_first_care_demand_caregiving_leave(  # noqa: E501
#         age_min: int | None = age_min_val,
#         age_max: int | None = age_max_val,
#         path_to_original_data: Path = _PATH_ORIGINAL,
#         path_to_leave_data: Path = _PATH_LEAVE,
#         path_to_specs: Path = _PATH_SPECS,
#         path_to_plot: Annotated[Path, Product] = _path_plot(
#             "working_hours",
#             f"event_study_working_hours_weekly_by_distance_to_first_care_demand_"
#             f"total_caregiving_{age_label_val}.pdf",
#         ),
#         ever_caregivers: bool = True,
#         ever_care_demand: bool = False,
#         window: int = 20,
#         compare_against_baseline: bool = True,
#     ) -> None:
#         """Event study: weekly working hours diff (policy − baseline) by distance to first care demand."""
#         with path_to_specs.open("rb") as f:
#             specs = pickle.load(f)
#         start_age = int(specs["start_age"])
#         end_age_caregiving = int(specs["end_age_caregiving"])
#         df_o, df_c = prepare_dataframes_simple(
#             pd.read_pickle(path_to_original_data),
#             pd.read_pickle(path_to_leave_data),
#             ever_caregivers,
#             ever_care_demand,
#         )
#         wh_o = (
#             df_o["working_hours"].astype(float) / 52.0
#             if "working_hours" in df_o.columns
#             else pd.Series(0.0, index=df_o.index)
#         )
#         wh_c = (
#             df_c["working_hours"].astype(float) / 52.0
#             if "working_hours" in df_c.columns
#             else pd.Series(0.0, index=df_c.index)
#         )
#         _, prof_diff, p1, p2, p3, p4, p5 = (
#             event_study_total_caregiving_merged_and_profiles(
#                 df_o,
#                 df_c,
#                 wh_o,
#                 wh_c,
#                 window,
#                 age_min,
#                 age_max,
#                 "care_demand",
#                 start_age,
#                 end_age_caregiving,
#                 compare_against_baseline=compare_against_baseline,
#             )
#         )
#         plot_outcome_difference_by_distance_total_caregiving(
#             prof_diff=prof_diff,
#             prof_1_year_diff=p1,
#             prof_2_year_diff=p2,
#             prof_3_year_diff=p3,
#             prof_4_year_diff=p4,
#             prof_5_year_diff=p5,
#             window=window,
#             path_to_plot=path_to_plot,
#             xlabel="Year relative to start of first care demand",
#             ylabel="Difference in weekly working hours",
#         )


# for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

#     @pytask.mark.publication_event_study_caregiving_leave
#     @pytask.mark.publication
#     @pytask.task(
#         id=f"{age_label_val}_working_hours_first_caregiving_spell_{_TASK_ID_SUFFIX}"
#     )
#     def task_plot_event_study_working_hours_by_distance_to_first_caregiving_spell_caregiving_leave(  # noqa: E501
#         age_min: int | None = age_min_val,
#         age_max: int | None = age_max_val,
#         path_to_original_data: Path = _PATH_ORIGINAL,
#         path_to_leave_data: Path = _PATH_LEAVE,
#         path_to_specs: Path = _PATH_SPECS,
#         path_to_plot: Annotated[Path, Product] = _path_plot(
#             "working_hours",
#             f"event_study_working_hours_weekly_by_distance_to_first_caregiving_spell_"
#             f"total_caregiving_{age_label_val}.pdf",
#         ),
#         ever_caregivers: bool = True,
#         ever_care_demand: bool = False,
#         window: int = 20,
#         compare_against_baseline: bool = True,
#     ) -> None:
#         """Event study: weekly working hours diff (policy − baseline) by distance to first caregiving spell."""
#         with path_to_specs.open("rb") as f:
#             specs = pickle.load(f)
#         start_age = int(specs["start_age"])
#         end_age_caregiving = int(specs["end_age_caregiving"])
#         df_o, df_c = prepare_dataframes_simple(
#             pd.read_pickle(path_to_original_data),
#             pd.read_pickle(path_to_leave_data),
#             ever_caregivers,
#             ever_care_demand,
#         )
#         wh_o = (
#             df_o["working_hours"].astype(float) / 52.0
#             if "working_hours" in df_o.columns
#             else pd.Series(0.0, index=df_o.index)
#         )
#         wh_c = (
#             df_c["working_hours"].astype(float) / 52.0
#             if "working_hours" in df_c.columns
#             else pd.Series(0.0, index=df_c.index)
#         )
#         _, prof_diff, p1, p2, p3, p4, p5 = (
#             event_study_total_caregiving_merged_and_profiles(
#                 df_o,
#                 df_c,
#                 wh_o,
#                 wh_c,
#                 window,
#                 age_min,
#                 age_max,
#                 "caregiving_spell",
#                 start_age,
#                 end_age_caregiving,
#                 compare_against_baseline=compare_against_baseline,
#             )
#         )
#         plot_outcome_difference_by_distance_total_caregiving(
#             prof_diff=prof_diff,
#             prof_1_year_diff=p1,
#             prof_2_year_diff=p2,
#             prof_3_year_diff=p3,
#             prof_4_year_diff=p4,
#             prof_5_year_diff=p5,
#             window=window,
#             path_to_plot=path_to_plot,
#             xlabel="Year relative to start of first caregiving spell",
#             ylabel="Difference in weekly working hours",
#         )


# # ---------------------------------------------------------------------------
# # Labor income
# # ---------------------------------------------------------------------------
# for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

#     @pytask.mark.publication_event_study_caregiving_leave
#     @pytask.mark.publication
#     @pytask.task(id=f"{age_label_val}_labor_income_first_care_demand_{_TASK_ID_SUFFIX}")
#     def task_plot_event_study_labor_income_by_distance_to_first_care_demand_caregiving_leave(  # noqa: E501
#         age_min: int | None = age_min_val,
#         age_max: int | None = age_max_val,
#         path_to_original_data: Path = _PATH_ORIGINAL,
#         path_to_leave_data: Path = _PATH_LEAVE,
#         path_to_specs: Path = _PATH_SPECS,
#         path_to_plot: Annotated[Path, Product] = _path_plot(
#             "labor_income",
#             f"event_study_monthly_gross_labor_income_by_distance_to_first_care_demand_"
#             f"total_caregiving_{age_label_val}.pdf",
#         ),
#         ever_caregivers: bool = True,
#         ever_care_demand: bool = False,
#         window: int = 20,
#         compare_against_baseline: bool = True,
#     ) -> None:
#         """Event study: monthly gross labor income diff (policy − baseline) by distance to first care demand."""
#         with path_to_specs.open("rb") as f:
#             specs = pickle.load(f)
#         start_age = int(specs["start_age"])
#         end_age_caregiving = int(specs["end_age_caregiving"])
#         df_o, df_c = prepare_dataframes_simple(
#             pd.read_pickle(path_to_original_data),
#             pd.read_pickle(path_to_leave_data),
#             ever_caregivers,
#             ever_care_demand,
#         )
#         inc_o = (
#             df_o["gross_labor_income"].astype(float) / 12.0
#             if "gross_labor_income" in df_o.columns
#             else pd.Series(0.0, index=df_o.index)
#         )
#         inc_c = (
#             df_c["gross_labor_income"].astype(float) / 12.0
#             if "gross_labor_income" in df_c.columns
#             else pd.Series(0.0, index=df_c.index)
#         )
#         _, prof_diff, p1, p2, p3, p4, p5 = (
#             event_study_total_caregiving_merged_and_profiles(
#                 df_o,
#                 df_c,
#                 inc_o,
#                 inc_c,
#                 window,
#                 age_min,
#                 age_max,
#                 "care_demand",
#                 start_age,
#                 end_age_caregiving,
#                 compare_against_baseline=compare_against_baseline,
#             )
#         )
#         plot_outcome_difference_by_distance_total_caregiving(
#             prof_diff=prof_diff,
#             prof_1_year_diff=p1,
#             prof_2_year_diff=p2,
#             prof_3_year_diff=p3,
#             prof_4_year_diff=p4,
#             prof_5_year_diff=p5,
#             window=window,
#             path_to_plot=path_to_plot,
#             xlabel="Year relative to start of first care demand",
#             ylabel="Difference in monthly gross labor income",
#             endogenous_ylim=True,
#         )


# for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

#     @pytask.mark.publication_event_study_caregiving_leave
#     @pytask.mark.publication
#     @pytask.task(
#         id=f"{age_label_val}_labor_income_first_caregiving_spell_{_TASK_ID_SUFFIX}"
#     )
#     def task_plot_event_study_labor_income_by_distance_to_first_caregiving_spell_caregiving_leave(  # noqa: E501
#         age_min: int | None = age_min_val,
#         age_max: int | None = age_max_val,
#         path_to_original_data: Path = _PATH_ORIGINAL,
#         path_to_leave_data: Path = _PATH_LEAVE,
#         path_to_specs: Path = _PATH_SPECS,
#         path_to_plot: Annotated[Path, Product] = _path_plot(
#             "labor_income",
#             f"event_study_monthly_gross_labor_income_by_distance_to_first_caregiving_spell_"
#             f"total_caregiving_{age_label_val}.pdf",
#         ),
#         ever_caregivers: bool = True,
#         ever_care_demand: bool = False,
#         window: int = 20,
#         compare_against_baseline: bool = True,
#     ) -> None:
#         """Event study: monthly gross labor income diff (policy − baseline) by distance to first caregiving spell."""
#         with path_to_specs.open("rb") as f:
#             specs = pickle.load(f)
#         start_age = int(specs["start_age"])
#         end_age_caregiving = int(specs["end_age_caregiving"])
#         df_o, df_c = prepare_dataframes_simple(
#             pd.read_pickle(path_to_original_data),
#             pd.read_pickle(path_to_leave_data),
#             ever_caregivers,
#             ever_care_demand,
#         )
#         inc_o = (
#             df_o["gross_labor_income"].astype(float) / 12.0
#             if "gross_labor_income" in df_o.columns
#             else pd.Series(0.0, index=df_o.index)
#         )
#         inc_c = (
#             df_c["gross_labor_income"].astype(float) / 12.0
#             if "gross_labor_income" in df_c.columns
#             else pd.Series(0.0, index=df_c.index)
#         )
#         _, prof_diff, p1, p2, p3, p4, p5 = (
#             event_study_total_caregiving_merged_and_profiles(
#                 df_o,
#                 df_c,
#                 inc_o,
#                 inc_c,
#                 window,
#                 age_min,
#                 age_max,
#                 "caregiving_spell",
#                 start_age,
#                 end_age_caregiving,
#                 compare_against_baseline=compare_against_baseline,
#             )
#         )
#         plot_outcome_difference_by_distance_total_caregiving(
#             prof_diff=prof_diff,
#             prof_1_year_diff=p1,
#             prof_2_year_diff=p2,
#             prof_3_year_diff=p3,
#             prof_4_year_diff=p4,
#             prof_5_year_diff=p5,
#             window=window,
#             path_to_plot=path_to_plot,
#             xlabel="Year relative to start of first caregiving spell",
#             ylabel="Difference in monthly gross labor income",
#             endogenous_ylim=True,
#         )


# ---------------------------------------------------------------------------
# Job finding
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study_caregiving_leave
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_job_finding_first_care_demand_{_TASK_ID_SUFFIX}")
    def task_plot_event_study_job_finding_by_distance_to_first_care_demand_caregiving_leave(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = _PATH_ORIGINAL,
        path_to_leave_data: Path = _PATH_LEAVE,
        path_to_specs: Path = _PATH_SPECS,
        path_to_plot: Annotated[Path, Product] = _path_plot(
            "job_finding",
            f"event_study_job_finding_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf",
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
        compare_against_baseline: bool = True,
    ) -> None:
        """Event study: job finding rate diff (policy − baseline) by distance to first care demand."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_leave_data),
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=compare_against_baseline,
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
            ylabel="Difference in job finding rate",
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study_caregiving_leave
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_job_finding_first_caregiving_spell_{_TASK_ID_SUFFIX}"
    )
    def task_plot_event_study_job_finding_by_distance_to_first_caregiving_spell_caregiving_leave(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = _PATH_ORIGINAL,
        path_to_leave_data: Path = _PATH_LEAVE,
        path_to_specs: Path = _PATH_SPECS,
        path_to_plot: Annotated[Path, Product] = _path_plot(
            "job_finding",
            f"event_study_job_finding_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf",
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
        compare_against_baseline: bool = True,
    ) -> None:
        """Event study: job finding rate diff (policy − baseline) by distance to first caregiving spell."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_leave_data),
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=compare_against_baseline,
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
            ylabel="Difference in job finding rate",
        )


# ---------------------------------------------------------------------------
# Job retention
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study_caregiving_leave
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_job_retention_first_care_demand_{_TASK_ID_SUFFIX}"
    )
    def task_plot_event_study_job_retention_by_distance_to_first_care_demand_caregiving_leave(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = _PATH_ORIGINAL,
        path_to_leave_data: Path = _PATH_LEAVE,
        path_to_specs: Path = _PATH_SPECS,
        path_to_plot: Annotated[Path, Product] = _path_plot(
            "job_retention",
            f"event_study_job_retention_rate_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf",
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
        compare_against_baseline: bool = True,
    ) -> None:
        """Event study: job retention rate diff (policy − baseline) by distance to first care demand."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_leave_data),
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
                window,
                age_min,
                age_max,
                "care_demand",
                start_age,
                end_age_caregiving,
                compare_against_baseline=compare_against_baseline,
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
            ylabel="Difference in job retention rate",
        )


for age_min_val, age_max_val, age_label_val in AGE_GROUPS_EVENT_STUDY:

    @pytask.mark.publication_event_study_caregiving_leave
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_job_retention_first_caregiving_spell_{_TASK_ID_SUFFIX}"
    )
    def task_plot_event_study_job_retention_by_distance_to_first_caregiving_spell_caregiving_leave(  # noqa: E501
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        path_to_original_data: Path = _PATH_ORIGINAL,
        path_to_leave_data: Path = _PATH_LEAVE,
        path_to_specs: Path = _PATH_SPECS,
        path_to_plot: Annotated[Path, Product] = _path_plot(
            "job_retention",
            f"event_study_job_retention_rate_by_distance_to_first_caregiving_spell_"
            f"total_caregiving_{age_label_val}.pdf",
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
        compare_against_baseline: bool = True,
    ) -> None:
        """Event study: job retention rate diff (policy − baseline) by distance to first caregiving spell."""
        with path_to_specs.open("rb") as f:
            specs = pickle.load(f)
        start_age = int(specs["start_age"])
        end_age_caregiving = int(specs["end_age_caregiving"])
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_leave_data),
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
                window,
                age_min,
                age_max,
                "caregiving_spell",
                start_age,
                end_age_caregiving,
                compare_against_baseline=compare_against_baseline,
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
            ylabel="Difference in job retention rate",
        )
