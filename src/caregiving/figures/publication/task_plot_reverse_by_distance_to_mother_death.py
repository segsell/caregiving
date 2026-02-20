"""Plot outcomes by distance to mother's death, total caregiving years (1–5+) before death.

New module for reverse event-study plots (t=0 = mother's death) with grouping by
total caregiving years before death. All outcomes (employment, full-time, part-time,
working hours, labor income) and two data pairs (standard, back_to_Jan7). Outputs
go to reverse_employment/{outcome}/total_caregiving_years/ with naming aligned to
task_plot_employment_rate_by_distance_to_first_care total_caregiving_years tasks.

Pytask marks: publication_reverse, publication_counterfactual, publication.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    calculate_simple_outcomes,
    prepare_dataframes_simple,
)
from caregiving.figures.publication.plotting_helpers import (
    plot_employment_rate_by_distance,
)
from caregiving.figures.publication.plotting_helpers_mother_death import (
    add_distance_to_mother_death,
    identify_agents_by_total_caregiving_before_death,
)
from caregiving.model.shared import (
    INFORMAL_CARE,
    PARENT_RECENTLY_DEAD,
)

# Subgroup labels for total care years *before death* (1–5+)
TOTAL_LABELS_BEFORE_DEATH = (
    "Baseline (1 total care year before death)",
    "Baseline (2 total care years before death)",
    "Baseline (3 total care years before death)",
    "Baseline (4 total care years before death)",
    "Baseline (5+ total care years before death)",
)


def _build_profiles_total_caregiving_before_death(
    merged: pd.DataFrame,
    window: int,
    outcome_baseline_col: str,
    outcome_counterfactual_col: str,
):
    """Build prof and prof_1..5 for total caregiving years before death.

    merged must have: distance_to_mother_death, current_caregiving,
    outcome_baseline_col, outcome_counterfactual_col.
    """
    agents_1, agents_2, agents_3, agents_4, agents_5 = (
        identify_agents_by_total_caregiving_before_death(
            merged,
            distance_col="distance_to_mother_death",
            window=window,
        )
    )

    prof = (
        merged.groupby("distance_to_mother_death", observed=False)[
            [outcome_baseline_col, outcome_counterfactual_col]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_to_mother_death")
    )
    prof = prof.rename(columns={"distance_to_mother_death": "distance_to_first_care"})

    def _prof_for_agents(agents):
        m = merged[merged["agent"].isin(agents)].copy()
        p = (
            m.groupby("distance_to_mother_death", observed=False)[
                [outcome_baseline_col]
            ]
            .mean()
            .reset_index()
            .sort_values("distance_to_mother_death")
        )
        return p.rename(columns={"distance_to_mother_death": "distance_to_first_care"})

    prof_1_year = _prof_for_agents(agents_1)
    prof_2_year = _prof_for_agents(agents_2)
    prof_3_year = _prof_for_agents(agents_3)
    prof_4_year = _prof_for_agents(agents_4)
    prof_5_year = _prof_for_agents(agents_5)

    return prof, prof_1_year, prof_2_year, prof_3_year, prof_4_year, prof_5_year


# ---------------------------------------------------------------------------
# Standard data: estimated_params, no_care_demand
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving_employment")
    def task_plot_employment_rate_by_distance_to_mother_death_total_caregiving(  # noqa: PLR0913
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
        / "employment"
        / "total_caregiving_years"
        / (
            f"employment_rate_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Employment rate by distance to mother's death, total care years 1–5+ before death. Standard data."""
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, window, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving_full_time")
    def task_plot_full_time_share_by_distance_to_mother_death_total_caregiving(  # noqa: PLR0913
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
        / "full_time"
        / "total_caregiving_years"
        / (
            f"full_time_share_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Full-time share by distance to mother's death, total care years 1–5+ before death. Standard data."""
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_ft, _ = calculate_simple_outcomes(df_o, "original")
        _, c_ft, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["full_time_o"] = o_ft.astype(float)
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["full_time_c"] = c_ft.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, window, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving_part_time")
    def task_plot_part_time_share_by_distance_to_mother_death_total_caregiving(  # noqa: PLR0913
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
        / "part_time"
        / "total_caregiving_years"
        / (
            f"part_time_share_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Part-time share by distance to mother's death, total care years 1–5+ before death. Standard data."""
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_pt = calculate_simple_outcomes(df_o, "original")
        _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["part_time_o"] = o_pt.astype(float)
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["part_time_c"] = c_pt.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, window, "part_time_o", "part_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="part_time_o",
            outcome_counterfactual="part_time_c",
            ylabel="Part-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving_working_hours")
    def task_plot_working_hours_by_distance_to_mother_death_total_caregiving(  # noqa: PLR0913
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
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"working_hours_weekly_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Weekly working hours by distance to mother's death, total care years 1–5+ before death. Standard data."""
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
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["working_hours_weekly_o"] = wh_o.values
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["working_hours_weekly_c"] = wh_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="working_hours_weekly_o",
            outcome_counterfactual="working_hours_weekly_c",
            ylabel="Weekly Working Hours",
            ylim=None,
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving_labor_income")
    def task_plot_labor_income_by_distance_to_mother_death_total_caregiving(  # noqa: PLR0913
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
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"monthly_gross_labor_income_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Monthly gross labor income by distance to mother's death, total care years 1–5+ before death. Standard data."""
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
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["monthly_gross_labor_income_o"] = inc_o.values
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["monthly_gross_labor_income_c"] = inc_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged,
            window,
            "monthly_gross_labor_income_o",
            "monthly_gross_labor_income_c",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="monthly_gross_labor_income_o",
            outcome_counterfactual="monthly_gross_labor_income_c",
            ylabel="Monthly Gross Labor Income",
            ylim=None,
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )


# ---------------------------------------------------------------------------
# back_to_Jan7 data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_total_caregiving_back_to_Jan7_employment"
    )
    def task_plot_employment_rate_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: PLR0913
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "reverse_employment"
        / "employment"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_employment_rate_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Employment rate by distance to mother's death, total care years 1–5+ before death. back_to_Jan7 data."""
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        o_work, _, _ = calculate_simple_outcomes(df_o, "original")
        c_work, _, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["work_o"] = o_work
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["work_c"] = c_work
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, window, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_total_caregiving_back_to_Jan7_full_time"
    )
    def task_plot_full_time_share_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: PLR0913
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "reverse_employment"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_full_time_share_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Full-time share by distance to mother's death, total care years 1–5+ before death. back_to_Jan7 data."""
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, o_ft, _ = calculate_simple_outcomes(df_o, "original")
        _, c_ft, _ = calculate_simple_outcomes(df_c, "no_care_demand")
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["full_time_o"] = o_ft.astype(float)
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["full_time_c"] = c_ft.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, window, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_total_caregiving_back_to_Jan7_part_time"
    )
    def task_plot_part_time_share_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: PLR0913
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "reverse_employment"
        / "part_time"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_part_time_share_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Part-time share by distance to mother's death, total care years 1–5+ before death. back_to_Jan7 data."""
        df_o, df_c = prepare_dataframes_simple(
            pd.read_pickle(path_to_original_data),
            pd.read_pickle(path_to_no_care_demand_data),
            ever_caregivers,
            ever_care_demand,
        )
        _, _, o_pt = calculate_simple_outcomes(df_o, "original")
        _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["part_time_o"] = o_pt.astype(float)
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["part_time_c"] = c_pt.astype(float)
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, window, "part_time_o", "part_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="part_time_o",
            outcome_counterfactual="part_time_c",
            ylabel="Part-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_total_caregiving_back_to_Jan7_working_hours"
    )
    def task_plot_working_hours_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: PLR0913
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "reverse_employment"
        / "working_hours"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_working_hours_weekly_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Weekly working hours by distance to mother's death, total care years 1–5+ before death. back_to_Jan7 data."""
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
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["working_hours_weekly_o"] = wh_o.values
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["working_hours_weekly_c"] = wh_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="working_hours_weekly_o",
            outcome_counterfactual="working_hours_weekly_c",
            ylabel="Weekly Working Hours",
            ylim=None,
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_total_caregiving_back_to_Jan7_labor_income"
    )
    def task_plot_labor_income_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: PLR0913
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params_back_to_Jan7.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand_back_to_Jan7.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "reverse_employment"
        / "labor_income"
        / "total_caregiving_years"
        / (
            f"back_to_Jan7_monthly_gross_labor_income_by_distance_to_mother_death_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = False,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """Monthly gross labor income by distance to mother's death, total care years 1–5+ before death. back_to_Jan7 data."""
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
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        o_cols = df_o[["agent", "period", "choice"]].copy()
        o_cols["monthly_gross_labor_income_o"] = inc_o.values
        o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
        c_cols = df_c[["agent", "period"]].copy()
        c_cols["monthly_gross_labor_income_c"] = inc_c.values
        merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
        merged = merged.merge(
            df_o[["agent", "period", "mother_dead", "age"]],
            on=["agent", "period"],
            how="left",
        )
        dist_map = (
            add_distance_to_mother_death(df_o)
            .groupby("agent", observed=False)["first_death_period"]
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
            & (merged["distance_to_mother_death"] >= -window)
            & (merged["distance_to_mother_death"] <= window)
        ]
        if age_min is not None:
            merged = merged[merged["age_at_death"] >= age_min].copy()
        if age_max is not None:
            merged = merged[merged["age_at_death"] <= age_max].copy()

        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged,
            window,
            "monthly_gross_labor_income_o",
            "monthly_gross_labor_income_c",
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window=window,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="monthly_gross_labor_income_o",
            outcome_counterfactual="monthly_gross_labor_income_c",
            ylabel="Monthly Gross Labor Income",
            ylim=None,
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )
