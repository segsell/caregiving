"""Plot event study (difference) by distance to mother's death, grouped by total caregiving years (1–5+).

Reverse event studies (t=0 = mother's death): difference in outcome (baseline minus
no-care-demand) by distance to mother's death. Grouping by total caregiving years
over the lifecycle (1, 2, 3, 4, 5+), same as task_plot_event_study_total_caregiving_years.
All outcomes (employment, full_time, part_time, working_hours, labor_income) and two
data pairs (estimated_params, back_to_Jan7). Outputs to
event_study_reverse/{outcome}/total_caregiving_years/.

Sample: ever_caregivers=True, ever_care_demand=False (aligned with forward module).
Pytask marks: publication_event_study_reverse, publication_counterfactual, publication.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    calculate_simple_outcomes,
    identify_agents_by_total_caregiving_over_lifecycle,
    prepare_dataframes_simple,
)
from caregiving.figures.publication.plotting_helpers_mother_death import (
    add_distance_to_mother_death,
)
from caregiving.model.shared import PARENT_RECENTLY_DEAD


def _get_plot_outcome_difference_by_distance_total_caregiving():
    """Lazy import to avoid loading the forward event-study module at collect time (prevents duplicate task ids)."""
    from caregiving.figures.publication.task_plot_event_study_total_caregiving_years import (
        plot_outcome_difference_by_distance_total_caregiving,
    )

    return plot_outcome_difference_by_distance_total_caregiving


# Distance column name in profile DataFrames (must match plot_outcome_difference_by_distance_total_caregiving)
_DIST_COL = "distance_to_first_care"

_AGE_GROUPS = (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
)


def reverse_event_study_total_caregiving_merged_and_profiles(
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    outcome_o_series: pd.Series,
    outcome_c_series: pd.Series,
    window: int,
    age_min: int | None,
    age_max: int | None,
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
    """Build merged df and profile diffs for reverse event study (mother death, total care years 1–5+).

    Returns (merged, prof_diff, prof_1_year_diff, ..., prof_5_year_diff).
    All profiles have columns _DIST_COL and 'diff' for plot_outcome_difference_by_distance_total_caregiving.
    """
    o_cols = df_o[["agent", "period", "choice"]].copy()
    o_cols["outcome_o"] = np.asarray(outcome_o_series).astype(float)
    c_cols = df_c[["agent", "period"]].copy()
    c_cols["outcome_c"] = np.asarray(outcome_c_series).astype(float)
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
    merged["distance_to_mother_death"] = merged["period"] - merged["first_death_period"]
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
        & (merged["distance_to_mother_death"] >= -window)
        & (merged["distance_to_mother_death"] <= window)
    ]
    if age_min is not None:
        merged = merged[merged["age_at_death"] >= age_min].copy()
    if age_max is not None:
        merged = merged[merged["age_at_death"] <= age_max].copy()

    merged["diff"] = merged["outcome_o"] - merged["outcome_c"]
    merged[_DIST_COL] = merged["distance_to_mother_death"]

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


# ---------------------------------------------------------------------------
# Employment: reverse event study (mother death), standard data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_employment_total_caregiving_estimated_params"
    )
    def task_plot_event_study_employment_rate_by_distance_to_mother_death_total_caregiving(  # noqa: E501
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
        / "event_study_reverse"
        / "employment"
        / "total_caregiving_years"
        / (
            f"event_study_employment_rate_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: employment rate difference by distance to mother's death (total care years 1–5+)."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_work,
                c_work,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Employment: reverse event study (mother death), back_to_Jan7 data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_employment_total_caregiving_back_to_Jan7"
    )
    def task_plot_event_study_employment_rate_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: E501
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
        / "event_study_reverse"
        / "employment"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_employment_rate_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: employment rate difference by distance to mother's death (total care years 1–5+), back_to_Jan7."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_work,
                c_work,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in employment rate",
        )


# ---------------------------------------------------------------------------
# Full-time: standard and back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_full_time_total_caregiving_estimated_params"
    )
    def task_plot_event_study_full_time_by_distance_to_mother_death_total_caregiving(  # noqa: E501
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
        / "event_study_reverse"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"event_study_full_time_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: full-time rate difference by distance to mother's death (total care years 1–5+)."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in full-time rate",
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_full_time_total_caregiving_back_to_Jan7"
    )
    def task_plot_event_study_full_time_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: E501
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
        / "event_study_reverse"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_full_time_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: full-time rate difference by distance to mother's death (total care years 1–5+), back_to_Jan7."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in full-time rate",
        )


# ---------------------------------------------------------------------------
# Part-time: standard and back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_part_time_total_caregiving_estimated_params"
    )
    def task_plot_event_study_part_time_by_distance_to_mother_death_total_caregiving(  # noqa: E501
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
        / "event_study_reverse"
        / "part_time"
        / "total_caregiving_years"
        / (
            f"event_study_part_time_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: part-time rate difference by distance to mother's death (total care years 1–5+)."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in part-time rate",
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_part_time_total_caregiving_back_to_Jan7"
    )
    def task_plot_event_study_part_time_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: E501
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
        / "event_study_reverse"
        / "part_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_part_time_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: part-time rate difference by distance to mother's death (total care years 1–5+), back_to_Jan7."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in part-time rate",
        )


# ---------------------------------------------------------------------------
# Working hours: standard and back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_working_hours_total_caregiving_estimated_params"
    )
    def task_plot_event_study_working_hours_by_distance_to_mother_death_total_caregiving(  # noqa: E501
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
        / "event_study_reverse"
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"event_study_working_hours_weekly_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: weekly working hours difference by distance to mother's death (total care years 1–5+)."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_working_hours_total_caregiving_back_to_Jan7"
    )
    def task_plot_event_study_working_hours_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: E501
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
        / "event_study_reverse"
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_working_hours_weekly_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: weekly working hours difference by distance to mother's death (total care years 1–5+), back_to_Jan7."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in weekly working hours",
            endogenous_ylim=True,
        )


# ---------------------------------------------------------------------------
# Labor income: standard and back_to_Jan7
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_labor_income_total_caregiving_estimated_params"
    )
    def task_plot_event_study_labor_income_by_distance_to_mother_death_total_caregiving(  # noqa: E501
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
        / "event_study_reverse"
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"event_study_monthly_gross_labor_income_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: monthly gross labor income difference by distance to mother's death (total care years 1–5+)."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )


for age_min_val, age_max_val, age_label_val in _AGE_GROUPS:

    @pytask.mark.publication_event_study_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_reverse_event_study_mother_death_labor_income_total_caregiving_back_to_Jan7"
    )
    def task_plot_event_study_labor_income_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: E501
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
        / "event_study_reverse"
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_event_study_monthly_gross_labor_income_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Event study: monthly gross labor income difference by distance to mother's death (total care years 1–5+), back_to_Jan7."""
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
            reverse_event_study_total_caregiving_merged_and_profiles(
                df_o,
                df_c,
                o_out,
                c_out,
                window,
                age_min,
                age_max,
                start_age,
                end_age_caregiving,
            )
        )
        _get_plot_outcome_difference_by_distance_total_caregiving()(
            prof_diff=prof_diff,
            prof_1_year_diff=p1,
            prof_2_year_diff=p2,
            prof_3_year_diff=p3,
            prof_4_year_diff=p4,
            prof_5_year_diff=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            ylabel="Difference in monthly gross labor income",
            endogenous_ylim=True,
        )
