"""Plot outcomes by distance to mother's death, total caregiving years (1–5+) before death.

New module for reverse event-study plots (t=0 = mother's death) with grouping by
total caregiving years before death. All specifications compare caregivers with
caregivers (ever_caregivers=True). All outcomes (employment, full-time, part-time,
working hours, labor income) and two data pairs (standard, back_to_Jan7). Outputs
go to reverse_employment/{outcome}/total_caregiving_years/ with naming aligned to
task_plot_employment_rate_by_distance_to_first_care total_caregiving_years tasks.

Pytask marks: publication_reverse, publication_counterfactual, publication.
"""

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    PATH_TO_BASELINE,
    PATH_TO_NO_CARE_DEMAND,
    calculate_simple_outcomes,
    get_publication_plot_style,
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


def _add_mother_death_distance_and_filter(
    merged: pd.DataFrame,
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    window_low: int,
    window_high: int,
    age_min: int | None,
    age_max: int | None,
) -> pd.DataFrame:
    """Add first_death_period, distance_to_mother_death, age_at_death; filter by window and optional age at death.

    window_low, window_high are positive integers (years before/after t=0).
    Distance is defined from the baseline only (no-care-demand has fewer stochastic
    draws per period so RNG diverges and mother_dead can differ; we still compare
    outcomes on the same (agent, period) grid using baseline's death timing).
    """
    dist_map = (
        add_distance_to_mother_death(df_o)
        .groupby("agent", observed=False)["first_death_period"]
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
        & (merged["distance_to_mother_death"] >= -window_low)
        & (merged["distance_to_mother_death"] <= window_high)
    ]
    if age_min is not None:
        merged = merged[merged["age_at_death"] >= age_min].copy()
    if age_max is not None:
        merged = merged[merged["age_at_death"] <= age_max].copy()
    return merged


def _first_care_demand_period_map(df_o: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with agent and first_care_demand_period (first period with care_demand > 0)."""
    care_demand_mask = df_o["care_demand"] > 0
    first = (
        df_o.loc[care_demand_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(columns={"period": "first_care_demand_period"})
    )
    return first[["agent", "first_care_demand_period"]]


def _filter_merged_by_first_care_demand_to_death(
    merged: pd.DataFrame,
    first_care_demand_map: pd.DataFrame,
    condition: Literal[">=10", "<10", "5_to_10", "<=5"],
) -> pd.DataFrame:
    """Filter merged to agents where years from first care demand to mother death satisfies condition.

    years_to_death = first_death_period - first_care_demand_period (positive when care demand before death).
    >=10: years_to_death >= 10; <10: years_to_death < 10; 5_to_10: 5 < years_to_death <= 10; <=5: years_to_death <= 5.
    """
    m = merged.merge(first_care_demand_map, on="agent", how="inner")
    m = m[m["first_care_demand_period"].notna()].copy()
    years = m["first_death_period"] - m["first_care_demand_period"]
    if condition == ">=10":
        m = m[years >= 10].drop_duplicates(subset=["agent", "period"])
    elif condition == "<10":
        m = m[years < 10].drop_duplicates(subset=["agent", "period"])
    elif condition == "5_to_10":
        m = m[(years > 5) & (years <= 10)].drop_duplicates(subset=["agent", "period"])
    else:  # <=5
        m = m[years <= 5].drop_duplicates(subset=["agent", "period"])
    return m


# ---------------------------------------------------------------------------
# Standard data: estimated_params, no_care_demand
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.mark.publication_selection
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving_employment")
    def task_plot_employment_rate_by_distance_to_mother_death_total_caregiving(  # noqa: PLR0913
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = PATH_TO_BASELINE,
        path_to_no_care_demand_data: Path = PATH_TO_NO_CARE_DEMAND,
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "selection"
        / "conditional_means_mother_death"
        / f"employment_rate_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 15,
        window_high: int = 15,
        window_by_age: dict[str, tuple[int, int]] | None = ({"ages_60_70": (15, 10)}),
        ylim: tuple[float, float] | None = (0, 1),
        yticks: list[float] | None = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        plot_caregivers_mean: bool = True,
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment rate",
            ylim=ylim,
            yticks=yticks,
            plot_caregivers_mean=plot_caregivers_mean,
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            style=get_publication_plot_style(age_label, use_subgroup_overrides=False),
            age_label=age_label,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.mark.publication_selection
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving_full_time")
    def task_plot_full_time_share_by_distance_to_mother_death_total_caregiving(  # noqa: PLR0913
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = PATH_TO_BASELINE,
        path_to_no_care_demand_data: Path = PATH_TO_NO_CARE_DEMAND,
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "selection"
        / "conditional_means_mother_death"
        / f"full_time_share_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 15,
        window_high: int = 15,
        window_by_age: dict[str, tuple[int, int]] | None = ({"ages_60_70": (15, 10)}),
        ylim: tuple[float, float] | None = (0, 0.5),
        yticks: list[float] | None = [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        plot_caregivers_mean: bool = True,
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-time share",
            ylim=ylim,
            yticks=yticks,
            plot_caregivers_mean=plot_caregivers_mean,
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            style=get_publication_plot_style(age_label, use_subgroup_overrides=False),
            age_label=age_label,
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.mark.publication_selection
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving_part_time")
    def task_plot_part_time_share_by_distance_to_mother_death_total_caregiving(  # noqa: PLR0913
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = PATH_TO_BASELINE,
        path_to_no_care_demand_data: Path = PATH_TO_NO_CARE_DEMAND,
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "selection"
        / "conditional_means_mother_death"
        / f"part_time_share_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 15,
        window_high: int = 15,
        window_by_age: dict[str, tuple[int, int]] | None = ({"ages_60_70": (15, 10)}),
        ylim: tuple[float, float] | None = (0, 0.5),
        yticks: list[float] | None = [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        plot_caregivers_mean: bool = True,
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "part_time_o", "part_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="part_time_o",
            outcome_counterfactual="part_time_c",
            ylabel="Part-time share",
            ylim=ylim,
            yticks=yticks,
            plot_caregivers_mean=plot_caregivers_mean,
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            style=get_publication_plot_style(age_label, use_subgroup_overrides=False),
            age_label=age_label,
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
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "working_hours_weekly_o", "working_hours_weekly_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
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
    @pytask.mark.publication_selection
    @pytask.task(id=f"{age_label_val}_mother_death_total_caregiving_labor_income")
    def task_plot_labor_income_by_distance_to_mother_death_total_caregiving(  # noqa: PLR0913
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = PATH_TO_BASELINE,
        path_to_no_care_demand_data: Path = PATH_TO_NO_CARE_DEMAND,
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "selection"
        / "conditional_means_mother_death"
        / f"monthly_gross_earnings_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 15,
        window_high: int = 15,
        window_by_age: dict[str, tuple[int, int]] | None = ({"ages_60_70": (15, 10)}),
        ylim: tuple[float, float] | None = (0, 1.8),
        yticks: list[float] | None = [0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75],
        plot_caregivers_mean: bool = True,
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged,
            w_low,
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
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="monthly_gross_labor_income_o",
            outcome_counterfactual="monthly_gross_labor_income_c",
            ylabel="Monthly gross earnings (1,000 euros)",
            ylim=ylim,
            yticks=yticks,
            plot_caregivers_mean=plot_caregivers_mean,
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            style=get_publication_plot_style(age_label, use_subgroup_overrides=False),
            age_label=age_label,
        )


# ---------------------------------------------------------------------------
# Standard data: condition 1 — first care demand >= 10 years before mother death
# ---------------------------------------------------------------------------
_CARE_CONDITION_1 = "care_demand_geq10_years"
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_1}_employment")
    def task_plot_employment_rate_by_distance_to_mother_death_care_geq10(  # noqa: PLR0913
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
        / _CARE_CONDITION_1
        / f"employment_rate_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, ">=10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10,),
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_1}_full_time")
    def task_plot_full_time_by_distance_to_mother_death_care_geq10(  # noqa: PLR0913
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
        / _CARE_CONDITION_1
        / f"full_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, ">=10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10,),
        )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_1}_part_time")
    # def task_plot_part_time_by_distance_to_mother_death_care_geq10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "part_time"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_1
    #     / f"part_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     _, _, o_pt = calculate_simple_outcomes(df_o, "original")
    #     _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["part_time_o"] = o_pt.astype(float)
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["part_time_c"] = c_pt.astype(float)
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, ">=10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "part_time_o", "part_time_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="part_time_o",
    #         outcome_counterfactual="part_time_c",
    #         ylabel="Part-Time Share",
    #         ylim=(-0.025, 1.0),
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_1}_working_hours")
    # def task_plot_working_hours_by_distance_to_mother_death_care_geq10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "working_hours"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_1
    #     / f"working_hours_weekly_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     wh_o = (
    #         df_o["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     wh_c = (
    #         df_c["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["working_hours_weekly_o"] = wh_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["working_hours_weekly_c"] = wh_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, ">=10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="working_hours_weekly_o",
    #         outcome_counterfactual="working_hours_weekly_c",
    #         ylabel="Weekly Working Hours",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_1}_labor_income")
    # def task_plot_labor_income_by_distance_to_mother_death_care_geq10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "labor_income"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_1
    #     / f"monthly_gross_labor_income_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     inc_o = (
    #         df_o["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     inc_c = (
    #         df_c["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["monthly_gross_labor_income_o"] = inc_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["monthly_gross_labor_income_c"] = inc_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, ">=10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged,
    #         window,
    #         "monthly_gross_labor_income_o",
    #         "monthly_gross_labor_income_c",
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="monthly_gross_labor_income_o",
    #         outcome_counterfactual="monthly_gross_labor_income_c",
    #         ylabel="Monthly Gross Labor Income",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10,),
    #     )


# ---------------------------------------------------------------------------
# Standard data: condition 2 — first care demand < 10 years before mother death
# ---------------------------------------------------------------------------
_CARE_CONDITION_2 = "care_demand_less10_years"
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_2}_employment")
    def task_plot_employment_rate_by_distance_to_mother_death_care_less10(  # noqa: PLR0913
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
        / _CARE_CONDITION_2
        / f"employment_rate_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "<10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10,),
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_2}_full_time")
    def task_plot_full_time_by_distance_to_mother_death_care_less10(  # noqa: PLR0913
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
        / _CARE_CONDITION_2
        / f"full_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "<10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10,),
        )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_2}_part_time")
    # def task_plot_part_time_by_distance_to_mother_death_care_less10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "part_time"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_2
    #     / f"part_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     _, _, o_pt = calculate_simple_outcomes(df_o, "original")
    #     _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["part_time_o"] = o_pt.astype(float)
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["part_time_c"] = c_pt.astype(float)
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "<10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "part_time_o", "part_time_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="part_time_o",
    #         outcome_counterfactual="part_time_c",
    #         ylabel="Part-Time Share",
    #         ylim=(-0.025, 1.0),
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_2}_working_hours")
    # def task_plot_working_hours_by_distance_to_mother_death_care_less10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "working_hours"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_2
    #     / f"working_hours_weekly_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     wh_o = (
    #         df_o["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     wh_c = (
    #         df_c["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["working_hours_weekly_o"] = wh_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["working_hours_weekly_c"] = wh_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "<10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="working_hours_weekly_o",
    #         outcome_counterfactual="working_hours_weekly_c",
    #         ylabel="Weekly Working Hours",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_2}_labor_income")
    # def task_plot_labor_income_by_distance_to_mother_death_care_less10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "labor_income"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_2
    #     / f"monthly_gross_labor_income_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     inc_o = (
    #         df_o["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     inc_c = (
    #         df_c["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["monthly_gross_labor_income_o"] = inc_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["monthly_gross_labor_income_c"] = inc_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "<10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged,
    #         window,
    #         "monthly_gross_labor_income_o",
    #         "monthly_gross_labor_income_c",
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="monthly_gross_labor_income_o",
    #         outcome_counterfactual="monthly_gross_labor_income_c",
    #         ylabel="Monthly Gross Labor Income",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10,),
    #     )


# ---------------------------------------------------------------------------
# Standard data: condition 3 — first care demand (5, 10] years before mother death
# ---------------------------------------------------------------------------
_CARE_CONDITION_3 = "care_demand_5_to_10_years"
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_3}_employment")
    def task_plot_employment_rate_by_distance_to_mother_death_care_5_to_10(  # noqa: PLR0913
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
        / _CARE_CONDITION_3
        / f"employment_rate_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "5_to_10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10, -5),
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_3}_full_time")
    def task_plot_full_time_by_distance_to_mother_death_care_5_to_10(  # noqa: PLR0913
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
        / _CARE_CONDITION_3
        / f"full_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "5_to_10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10, -5),
        )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_3}_part_time")
    # def task_plot_part_time_by_distance_to_mother_death_care_5_to_10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "part_time"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_3
    #     / f"part_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     _, _, o_pt = calculate_simple_outcomes(df_o, "original")
    #     _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["part_time_o"] = o_pt.astype(float)
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["part_time_c"] = c_pt.astype(float)
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "5_to_10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "part_time_o", "part_time_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="part_time_o",
    #         outcome_counterfactual="part_time_c",
    #         ylabel="Part-Time Share",
    #         ylim=(-0.025, 1.0),
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10, -5),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_3}_working_hours")
    # def task_plot_working_hours_by_distance_to_mother_death_care_5_to_10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "working_hours"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_3
    #     / f"working_hours_weekly_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     wh_o = (
    #         df_o["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     wh_c = (
    #         df_c["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["working_hours_weekly_o"] = wh_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["working_hours_weekly_c"] = wh_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "5_to_10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="working_hours_weekly_o",
    #         outcome_counterfactual="working_hours_weekly_c",
    #         ylabel="Weekly Working Hours",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10, -5),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_3}_labor_income")
    # def task_plot_labor_income_by_distance_to_mother_death_care_5_to_10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "labor_income"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_3
    #     / f"monthly_gross_labor_income_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     inc_o = (
    #         df_o["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     inc_c = (
    #         df_c["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["monthly_gross_labor_income_o"] = inc_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["monthly_gross_labor_income_c"] = inc_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "5_to_10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged,
    #         window,
    #         "monthly_gross_labor_income_o",
    #         "monthly_gross_labor_income_c",
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="monthly_gross_labor_income_o",
    #         outcome_counterfactual="monthly_gross_labor_income_c",
    #         ylabel="Monthly Gross Labor Income",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10, -5),
    #     )


# ---------------------------------------------------------------------------
# Standard data: condition 4 — first care demand <= 5 years before mother death
# ---------------------------------------------------------------------------
_CARE_CONDITION_4 = "care_demand_leq5_years"
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_4}_employment")
    def task_plot_employment_rate_by_distance_to_mother_death_care_leq5(  # noqa: PLR0913
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
        / _CARE_CONDITION_4
        / f"employment_rate_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "<=5"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-5,),
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_4}_full_time")
    def task_plot_full_time_by_distance_to_mother_death_care_leq5(  # noqa: PLR0913
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
        / _CARE_CONDITION_4
        / f"full_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "<=5"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-5,),
        )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_4}_part_time")
    # def task_plot_part_time_by_distance_to_mother_death_care_leq5(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "part_time"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_4
    #     / f"part_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     _, _, o_pt = calculate_simple_outcomes(df_o, "original")
    #     _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["part_time_o"] = o_pt.astype(float)
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["part_time_c"] = c_pt.astype(float)
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "<=5"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "part_time_o", "part_time_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="part_time_o",
    #         outcome_counterfactual="part_time_c",
    #         ylabel="Part-Time Share",
    #         ylim=(-0.025, 1.0),
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-5,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_4}_working_hours")
    # def task_plot_working_hours_by_distance_to_mother_death_care_leq5(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "working_hours"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_4
    #     / f"working_hours_weekly_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     wh_o = (
    #         df_o["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     wh_c = (
    #         df_c["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["working_hours_weekly_o"] = wh_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["working_hours_weekly_c"] = wh_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "<=5"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="working_hours_weekly_o",
    #         outcome_counterfactual="working_hours_weekly_c",
    #         ylabel="Weekly Working Hours",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-5,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_{_CARE_CONDITION_4}_labor_income")
    # def task_plot_labor_income_by_distance_to_mother_death_care_leq5(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "labor_income"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_4
    #     / f"monthly_gross_labor_income_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     inc_o = (
    #         df_o["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     inc_c = (
    #         df_c["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["monthly_gross_labor_income_o"] = inc_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["monthly_gross_labor_income_c"] = inc_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "<=5"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged,
    #         window,
    #         "monthly_gross_labor_income_o",
    #         "monthly_gross_labor_income_c",
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="monthly_gross_labor_income_o",
    #         outcome_counterfactual="monthly_gross_labor_income_c",
    #         ylabel="Monthly Gross Labor Income",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-5,),
    #     )


# ---------------------------------------------------------------------------
# back_to_Jan7 data
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
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
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
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
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
        )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_total_caregiving_back_to_Jan7_part_time"
    # )
    # def task_plot_part_time_share_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     age_label: str = age_label_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "part_time"
    #     / "total_caregiving_years"
    #     / (
    #         f"back_to_Jan7_part_time_share_by_distance_to_mother_death_"
    #         f"total_caregiving_{age_label_val}.pdf"
    #     ),
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     """Part-time share by distance to mother's death, total care years 1–5+ before death. back_to_Jan7 data."""
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     _, _, o_pt = calculate_simple_outcomes(df_o, "original")
    #     _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["part_time_o"] = o_pt.astype(float)
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["part_time_c"] = c_pt.astype(float)
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "part_time_o", "part_time_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="part_time_o",
    #         outcome_counterfactual="part_time_c",
    #         ylabel="Part-Time Share",
    #         ylim=(-0.025, 1.0),
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_total_caregiving_back_to_Jan7_working_hours"
    # )
    # def task_plot_working_hours_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     age_label: str = age_label_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "working_hours"
    #     / "total_caregiving_years"
    #     / (
    #         f"back_to_Jan7_working_hours_weekly_by_distance_to_mother_death_"
    #         f"total_caregiving_{age_label_val}.pdf"
    #     ),
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     """Weekly working hours by distance to mother's death, total care years 1–5+ before death. back_to_Jan7 data."""
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     wh_o = (
    #         df_o["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     wh_c = (
    #         df_c["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["working_hours_weekly_o"] = wh_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["working_hours_weekly_c"] = wh_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="working_hours_weekly_o",
    #         outcome_counterfactual="working_hours_weekly_c",
    #         ylabel="Weekly Working Hours",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_total_caregiving_back_to_Jan7_labor_income"
    # )
    # def task_plot_labor_income_by_distance_to_mother_death_total_caregiving_back_to_Jan7(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     age_label: str = age_label_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "labor_income"
    #     / "total_caregiving_years"
    #     / (
    #         f"back_to_Jan7_monthly_gross_labor_income_by_distance_to_mother_death_"
    #         f"total_caregiving_{age_label_val}.pdf"
    #     ),
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     """Monthly gross labor income by distance to mother's death, total care years 1–5+ before death. back_to_Jan7 data."""
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     inc_o = (
    #         df_o["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     inc_c = (
    #         df_c["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["monthly_gross_labor_income_o"] = inc_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["monthly_gross_labor_income_c"] = inc_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged,
    #         window,
    #         "monthly_gross_labor_income_o",
    #         "monthly_gross_labor_income_c",
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="monthly_gross_labor_income_o",
    #         outcome_counterfactual="monthly_gross_labor_income_c",
    #         ylabel="Monthly Gross Labor Income",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #     )


# ---------------------------------------------------------------------------
# back_to_Jan7: condition 1 — first care demand >= 10 years before mother death
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_1}_employment"
    )
    def task_plot_employment_rate_by_distance_to_mother_death_back_to_Jan7_care_geq10(  # noqa: PLR0913
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
        / _CARE_CONDITION_1
        / f"back_to_Jan7_employment_rate_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, ">=10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10,),
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_1}_full_time"
    )
    def task_plot_full_time_by_distance_to_mother_death_back_to_Jan7_care_geq10(  # noqa: PLR0913
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
        / _CARE_CONDITION_1
        / f"back_to_Jan7_full_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, ">=10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10,),
        )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_1}_part_time"
    # )
    # def task_plot_part_time_by_distance_to_mother_death_back_to_Jan7_care_geq10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "part_time"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_1
    #     / f"back_to_Jan7_part_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     _, _, o_pt = calculate_simple_outcomes(df_o, "original")
    #     _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["part_time_o"] = o_pt.astype(float)
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["part_time_c"] = c_pt.astype(float)
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, ">=10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "part_time_o", "part_time_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="part_time_o",
    #         outcome_counterfactual="part_time_c",
    #         ylabel="Part-Time Share",
    #         ylim=(-0.025, 1.0),
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_1}_working_hours"
    # )
    # def task_plot_working_hours_by_distance_to_mother_death_back_to_Jan7_care_geq10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "working_hours"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_1
    #     / f"back_to_Jan7_working_hours_weekly_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     wh_o = (
    #         df_o["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     wh_c = (
    #         df_c["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["working_hours_weekly_o"] = wh_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["working_hours_weekly_c"] = wh_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, ">=10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="working_hours_weekly_o",
    #         outcome_counterfactual="working_hours_weekly_c",
    #         ylabel="Weekly Working Hours",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_1}_labor_income"
    # )
    # def task_plot_labor_income_by_distance_to_mother_death_back_to_Jan7_care_geq10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "labor_income"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_1
    #     / f"back_to_Jan7_monthly_gross_labor_income_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     inc_o = (
    #         df_o["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     inc_c = (
    #         df_c["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["monthly_gross_labor_income_o"] = inc_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["monthly_gross_labor_income_c"] = inc_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, ">=10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged,
    #         window,
    #         "monthly_gross_labor_income_o",
    #         "monthly_gross_labor_income_c",
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="monthly_gross_labor_income_o",
    #         outcome_counterfactual="monthly_gross_labor_income_c",
    #         ylabel="Monthly Gross Labor Income",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10,),
    #     )


# ---------------------------------------------------------------------------
# back_to_Jan7: condition 2 — first care demand < 10 years before mother death
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_2}_employment"
    )
    def task_plot_employment_rate_by_distance_to_mother_death_back_to_Jan7_care_less10(  # noqa: PLR0913
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
        / _CARE_CONDITION_2
        / f"back_to_Jan7_employment_rate_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "<10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10,),
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_2}_full_time"
    )
    def task_plot_full_time_by_distance_to_mother_death_back_to_Jan7_care_less10(  # noqa: PLR0913
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
        / _CARE_CONDITION_2
        / f"back_to_Jan7_full_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "<10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10,),
        )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_2}_part_time")
    # def task_plot_part_time_by_distance_to_mother_death_back_to_Jan7_care_less10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD / "solve_and_simulate" / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD / "solve_and_simulate" / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD / "figures" / "publication" / "counterfactual" / "reverse_employment" / "part_time" / "total_caregiving_years" / _CARE_CONDITION_2 / f"back_to_Jan7_part_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(pd.read_pickle(path_to_original_data), pd.read_pickle(path_to_no_care_demand_data), ever_caregivers, ever_care_demand)
    #     _, _, o_pt = calculate_simple_outcomes(df_o, "original")
    #     _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["part_time_o"] = o_pt.astype(float)
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["part_time_c"] = c_pt.astype(float)
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(df_o[["agent", "period", "mother_dead", "age"]], on=["agent", "period"], how="left")
    #     merged = _add_mother_death_distance_and_filter(merged, df_o, df_c, window, age_min, age_max)
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(merged, first_care_map, "<10")
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(merged, window, "part_time_o", "part_time_c")
    #     plot_employment_rate_by_distance(prof=prof, prof_1_year=p1, prof_2_year=p2, prof_3_year=p3, prof_4_year=p4, prof_5_year=p5, window=window, path_to_plot=path_to_plot, xlabel="Year relative to mother's death", outcome_baseline="part_time_o", outcome_counterfactual="part_time_c", ylabel="Part-Time Share", ylim=(-0.025, 1.0), subgroup_labels=TOTAL_LABELS_BEFORE_DEATH, extra_vlines=(-10,))

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_2}_working_hours")
    # def task_plot_working_hours_by_distance_to_mother_death_back_to_Jan7_care_less10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD / "solve_and_simulate" / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD / "solve_and_simulate" / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD / "figures" / "publication" / "counterfactual" / "reverse_employment" / "working_hours" / "total_caregiving_years" / _CARE_CONDITION_2 / f"back_to_Jan7_working_hours_weekly_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(pd.read_pickle(path_to_original_data), pd.read_pickle(path_to_no_care_demand_data), ever_caregivers, ever_care_demand)
    #     wh_o = df_o["working_hours"].astype(float) / 52.0 if "working_hours" in df_o.columns else pd.Series(0.0, index=df_o.index)
    #     wh_c = df_c["working_hours"].astype(float) / 52.0 if "working_hours" in df_c.columns else pd.Series(0.0, index=df_c.index)
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["working_hours_weekly_o"] = wh_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["working_hours_weekly_c"] = wh_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(df_o[["agent", "period", "mother_dead", "age"]], on=["agent", "period"], how="left")
    #     merged = _add_mother_death_distance_and_filter(merged, df_o, df_c, window, age_min, age_max)
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(merged, first_care_map, "<10")
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(merged, window, "working_hours_weekly_o", "working_hours_weekly_c")
    #     plot_employment_rate_by_distance(prof=prof, prof_1_year=p1, prof_2_year=p2, prof_3_year=p3, prof_4_year=p4, prof_5_year=p5, window=window, path_to_plot=path_to_plot, xlabel="Year relative to mother's death", outcome_baseline="working_hours_weekly_o", outcome_counterfactual="working_hours_weekly_c", ylabel="Weekly Working Hours", ylim=None, subgroup_labels=TOTAL_LABELS_BEFORE_DEATH, extra_vlines=(-10,))

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_2}_labor_income")
    # def task_plot_labor_income_by_distance_to_mother_death_back_to_Jan7_care_less10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD / "solve_and_simulate" / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD / "solve_and_simulate" / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD / "figures" / "publication" / "counterfactual" / "reverse_employment" / "labor_income" / "total_caregiving_years" / _CARE_CONDITION_2 / f"back_to_Jan7_monthly_gross_labor_income_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(pd.read_pickle(path_to_original_data), pd.read_pickle(path_to_no_care_demand_data), ever_caregivers, ever_care_demand)
    #     inc_o = df_o["gross_labor_income"].astype(float) / 12.0 if "gross_labor_income" in df_o.columns else pd.Series(0.0, index=df_o.index)
    #     inc_c = df_c["gross_labor_income"].astype(float) / 12.0 if "gross_labor_income" in df_c.columns else pd.Series(0.0, index=df_c.index)
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["monthly_gross_labor_income_o"] = inc_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["monthly_gross_labor_income_c"] = inc_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(df_o[["agent", "period", "mother_dead", "age"]], on=["agent", "period"], how="left")
    #     merged = _add_mother_death_distance_and_filter(merged, df_o, df_c, window, age_min, age_max)
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(merged, first_care_map, "<10")
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(merged, window, "monthly_gross_labor_income_o", "monthly_gross_labor_income_c")
    #     plot_employment_rate_by_distance(prof=prof, prof_1_year=p1, prof_2_year=p2, prof_3_year=p3, prof_4_year=p4, prof_5_year=p5, window=window, path_to_plot=path_to_plot, xlabel="Year relative to mother's death", outcome_baseline="monthly_gross_labor_income_o", outcome_counterfactual="monthly_gross_labor_income_c", ylabel="Monthly Gross Labor Income", ylim=None, subgroup_labels=TOTAL_LABELS_BEFORE_DEATH, extra_vlines=(-10,))


# ---------------------------------------------------------------------------
# back_to_Jan7: condition 3 — first care demand (5, 10] years before mother death
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_3}_employment"
    )
    def task_plot_employment_rate_by_distance_to_mother_death_back_to_Jan7_care_5_to_10(  # noqa: PLR0913
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
        / _CARE_CONDITION_3
        / f"back_to_Jan7_employment_rate_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "5_to_10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10, -5),
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_3}_full_time"
    )
    def task_plot_full_time_by_distance_to_mother_death_back_to_Jan7_care_5_to_10(  # noqa: PLR0913
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
        / _CARE_CONDITION_3
        / f"back_to_Jan7_full_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "5_to_10"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-10, -5),
        )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_3}_part_time"
    # )
    # def task_plot_part_time_by_distance_to_mother_death_back_to_Jan7_care_5_to_10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "part_time"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_3
    #     / f"back_to_Jan7_part_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     _, _, o_pt = calculate_simple_outcomes(df_o, "original")
    #     _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["part_time_o"] = o_pt.astype(float)
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["part_time_c"] = c_pt.astype(float)
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "5_to_10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "part_time_o", "part_time_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="part_time_o",
    #         outcome_counterfactual="part_time_c",
    #         ylabel="Part-Time Share",
    #         ylim=(-0.025, 1.0),
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10, -5),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_3}_working_hours"
    # )
    # def task_plot_working_hours_by_distance_to_mother_death_back_to_Jan7_care_5_to_10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "working_hours"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_3
    #     / f"back_to_Jan7_working_hours_weekly_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     wh_o = (
    #         df_o["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     wh_c = (
    #         df_c["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["working_hours_weekly_o"] = wh_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["working_hours_weekly_c"] = wh_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "5_to_10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="working_hours_weekly_o",
    #         outcome_counterfactual="working_hours_weekly_c",
    #         ylabel="Weekly Working Hours",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10, -5),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_3}_labor_income"
    # )
    # def task_plot_labor_income_by_distance_to_mother_death_back_to_Jan7_care_5_to_10(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "labor_income"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_3
    #     / f"back_to_Jan7_monthly_gross_labor_income_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     inc_o = (
    #         df_o["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     inc_c = (
    #         df_c["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["monthly_gross_labor_income_o"] = inc_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["monthly_gross_labor_income_c"] = inc_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "5_to_10"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged,
    #         window,
    #         "monthly_gross_labor_income_o",
    #         "monthly_gross_labor_income_c",
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="monthly_gross_labor_income_o",
    #         outcome_counterfactual="monthly_gross_labor_income_c",
    #         ylabel="Monthly Gross Labor Income",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-10, -5),
    #     )


# ---------------------------------------------------------------------------
# back_to_Jan7: condition 4 — first care demand <= 5 years before mother death
# ---------------------------------------------------------------------------
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_4}_employment"
    )
    def task_plot_employment_rate_by_distance_to_mother_death_back_to_Jan7_care_leq5(  # noqa: PLR0913
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
        / _CARE_CONDITION_4
        / f"back_to_Jan7_employment_rate_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "<=5"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "work_o", "work_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="work_o",
            outcome_counterfactual="work_c",
            ylabel="Employment Rate",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-5,),
        )

    @pytask.mark.publication_reverse
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication
    @pytask.task(
        id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_4}_full_time"
    )
    def task_plot_full_time_by_distance_to_mother_death_back_to_Jan7_care_leq5(  # noqa: PLR0913
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
        / _CARE_CONDITION_4
        / f"back_to_Jan7_full_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window_low: int = 20,
        window_high: int = 20,
        window_by_age: dict[str, tuple[int, int]] | None = None,
    ) -> None:
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
        if window_by_age is not None and age_label in window_by_age:
            w_low, w_high = window_by_age[age_label]
        else:
            w_low, w_high = window_low, window_high
        merged = _add_mother_death_distance_and_filter(
            merged, df_o, df_c, w_low, w_high, age_min, age_max
        )
        first_care_map = _first_care_demand_period_map(df_o)
        merged = _filter_merged_by_first_care_demand_to_death(
            merged, first_care_map, "<=5"
        )
        prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
            merged, w_low, "full_time_o", "full_time_c"
        )
        plot_employment_rate_by_distance(
            prof=prof,
            prof_1_year=p1,
            prof_2_year=p2,
            prof_3_year=p3,
            prof_4_year=p4,
            prof_5_year=p5,
            window_low=w_low,
            window_high=w_high,
            path_to_plot=path_to_plot,
            xlabel="Year relative to mother's death",
            outcome_baseline="full_time_o",
            outcome_counterfactual="full_time_c",
            ylabel="Full-Time Share",
            ylim=(-0.025, 1.0),
            subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
            extra_vlines=(-5,),
        )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_4}_part_time"
    # )
    # def task_plot_part_time_by_distance_to_mother_death_back_to_Jan7_care_leq5(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "part_time"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_4
    #     / f"back_to_Jan7_part_time_share_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     _, _, o_pt = calculate_simple_outcomes(df_o, "original")
    #     _, _, c_pt = calculate_simple_outcomes(df_c, "no_care_demand")
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["part_time_o"] = o_pt.astype(float)
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["part_time_c"] = c_pt.astype(float)
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "<=5"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "part_time_o", "part_time_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="part_time_o",
    #         outcome_counterfactual="part_time_c",
    #         ylabel="Part-Time Share",
    #         ylim=(-0.025, 1.0),
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-5,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_4}_working_hours"
    # )
    # def task_plot_working_hours_by_distance_to_mother_death_back_to_Jan7_care_leq5(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "working_hours"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_4
    #     / f"back_to_Jan7_working_hours_weekly_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     wh_o = (
    #         df_o["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     wh_c = (
    #         df_c["working_hours"].astype(float) / 52.0
    #         if "working_hours" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["working_hours_weekly_o"] = wh_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["working_hours_weekly_c"] = wh_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "<=5"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged, window, "working_hours_weekly_o", "working_hours_weekly_c"
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="working_hours_weekly_o",
    #         outcome_counterfactual="working_hours_weekly_c",
    #         ylabel="Weekly Working Hours",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-5,),
    #     )

    # @pytask.mark.publication_reverse
    # @pytask.mark.publication_counterfactual
    # @pytask.mark.publication
    # @pytask.task(
    #     id=f"{age_label_val}_mother_death_back_to_Jan7_{_CARE_CONDITION_4}_labor_income"
    # )
    # def task_plot_labor_income_by_distance_to_mother_death_back_to_Jan7_care_leq5(  # noqa: PLR0913
    #     age_min: int | None = age_min_val,
    #     age_max: int | None = age_max_val,
    #     path_to_original_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_estimated_params_back_to_Jan7.pkl",
    #     path_to_no_care_demand_data: Path = BLD
    #     / "solve_and_simulate"
    #     / "simulated_data_no_care_demand_back_to_Jan7.pkl",
    #     path_to_plot: Annotated[Path, Product] = BLD
    #     / "figures"
    #     / "publication"
    #     / "counterfactual"
    #     / "reverse_employment"
    #     / "labor_income"
    #     / "total_caregiving_years"
    #     / _CARE_CONDITION_4
    #     / f"back_to_Jan7_monthly_gross_labor_income_by_distance_to_mother_death_total_caregiving_{age_label_val}.pdf",
    #     ever_caregivers: bool = True,
    #     ever_care_demand: bool = False,
    #     window: int = 20,
    # ) -> None:
    #     df_o, df_c = prepare_dataframes_simple(
    #         pd.read_pickle(path_to_original_data),
    #         pd.read_pickle(path_to_no_care_demand_data),
    #         ever_caregivers,
    #         ever_care_demand,
    #     )
    #     inc_o = (
    #         df_o["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_o.columns
    #         else pd.Series(0.0, index=df_o.index)
    #     )
    #     inc_c = (
    #         df_c["gross_labor_income"].astype(float) / 12.0
    #         if "gross_labor_income" in df_c.columns
    #         else pd.Series(0.0, index=df_c.index)
    #     )
    #     care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    #     o_cols = df_o[["agent", "period", "choice"]].copy()
    #     o_cols["monthly_gross_labor_income_o"] = inc_o.values
    #     o_cols["current_caregiving"] = o_cols["choice"].isin(care_codes).astype(int)
    #     c_cols = df_c[["agent", "period"]].copy()
    #     c_cols["monthly_gross_labor_income_c"] = inc_c.values
    #     merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    #     merged = merged.merge(
    #         df_o[["agent", "period", "mother_dead", "age"]],
    #         on=["agent", "period"],
    #         how="left",
    #     )
    #     merged = _add_mother_death_distance_and_filter(
    #         merged, df_o, df_c, window, age_min, age_max
    #     )
    #     first_care_map = _first_care_demand_period_map(df_o)
    #     merged = _filter_merged_by_first_care_demand_to_death(
    #         merged, first_care_map, "<=5"
    #     )
    #     prof, p1, p2, p3, p4, p5 = _build_profiles_total_caregiving_before_death(
    #         merged,
    #         window,
    #         "monthly_gross_labor_income_o",
    #         "monthly_gross_labor_income_c",
    #     )
    #     plot_employment_rate_by_distance(
    #         prof=prof,
    #         prof_1_year=p1,
    #         prof_2_year=p2,
    #         prof_3_year=p3,
    #         prof_4_year=p4,
    #         prof_5_year=p5,
    #         window=window,
    #         path_to_plot=path_to_plot,
    #         xlabel="Year relative to mother's death",
    #         outcome_baseline="monthly_gross_labor_income_o",
    #         outcome_counterfactual="monthly_gross_labor_income_c",
    #         ylabel="Monthly Gross Labor Income",
    #         ylim=None,
    #         subgroup_labels=TOTAL_LABELS_BEFORE_DEATH,
    #         extra_vlines=(-5,),
    #     )
