"""Conditional means of labor outcomes by distance to first care demand.

Conditional on caregiving_type == 1 (agents who can provide informal care).
Groups 1–4: exact 1, 2, 3, 4 years consecutive care demand; group 5: at least 5 years.
Uses Jan7 simulated data. Working hours are weekly.
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
    publication_savefig,
)
from caregiving.counterfactual.plotting_utils import ensure_agent_period
from caregiving.figures.publication.plotting_helpers import (
    add_distance_to_first_care_demand,
    identify_agents_by_duration,
)
from caregiving.model.shared import DEAD, INFORMAL_CARE

# Style for 5 duration groups: 1–4 exact consecutive years, 5+ at least
_GROUP_STYLE = [
    {
        "color": "0.8",
        "marker": "8",
        "markersize": 5,
        "label": "1-Year Care Demand (exact, consecutive)",
    },
    {
        "color": "0.6",
        "marker": "^",
        "markersize": 5,
        "label": "2-Year Care Demand (exact, consecutive)",
    },
    {
        "color": "0.4",
        "marker": "D",
        "markersize": 5,
        "label": "3-Year Care Demand (exact, consecutive)",
    },
    {
        "color": "0.2",
        "marker": "s",
        "markersize": 5,
        "label": "4-Year Care Demand (exact, consecutive)",
    },
    {
        "color": "black",
        "marker": "*",
        "markersize": 6,
        "label": "5+ Year Care Demand (at least)",
    },
]

# Base filename suffix for all final plots
_PLOT_NAME_SUFFIX = (
    "by_distance_to_first_care_demand_caregiving_type_1_"
    "exact_consec_1_2_3_4_years_at_least_5"
)


def _identify_agents_at_least_5_years_care_demand(
    merged: pd.DataFrame,
    distance_col: str = "distance_to_first_care_demand",
) -> np.ndarray:
    """Agents with care_demand > 0 at t=0,1,2,3,4 (at least 5 years)."""
    merged = merged.copy()
    merged["care_status"] = (merged["care_demand"] > 0).astype(int)
    agent_care_matrix = merged[merged[distance_col] >= 0].pivot_table(
        index="agent",
        columns=distance_col,
        values="care_status",
        aggfunc="first",
    )
    agents_5_year = []
    for agent in agent_care_matrix.index:
        care_at = []
        for t in range(5):
            if t in agent_care_matrix.columns and pd.notna(
                agent_care_matrix.loc[agent, t]
            ):
                care_at.append(agent_care_matrix.loc[agent, t] == 1)
            else:
                care_at.append(False)
        if all(care_at):
            agents_5_year.append(agent)
    return np.array(agents_5_year)


def _prepare_merged_type1(
    path_to_data: Path,
    window_low: int,
    window_high: int,
    ever_caregivers: bool = False,
    ever_care_demand: bool = False,
) -> pd.DataFrame:
    """Load baseline data, restrict to caregiving_type==1, add distance_to_first_care_demand.

    If ever_caregivers (ever_care_demand) is True, restrict to agents who ever
    provided informal care (ever experienced care demand).
    """
    df = pd.read_pickle(path_to_data)
    df = df[df["health"] != DEAD].copy()
    df = ensure_agent_period(df)
    if "caregiving_type" not in df.columns:
        raise ValueError("caregiving_type column not found. Cannot restrict to type 1.")
    type_1_agents = df[df["caregiving_type"] == 1]["agent"].unique()
    df = df[df["agent"].isin(type_1_agents)].copy()

    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df.loc[df["choice"].isin(care_codes), "agent"].unique()
        df = df[df["agent"].isin(caregiver_ids)].copy()
    if ever_care_demand:
        care_demand_ids = df.loc[df["care_demand"] > 0, "agent"].unique()
        df = df[df["agent"].isin(care_demand_ids)].copy()

    df_dist = add_distance_to_first_care_demand(df)
    dist_map = (
        df_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )
    merged = df.merge(dist_map, on="agent", how="inner")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged[
        merged["first_care_demand_period"].notna()
        & (merged["distance_to_first_care_demand"] >= -window_low)
        & (merged["distance_to_first_care_demand"] <= window_high)
    ]
    return merged


def _compute_no_care_demand_profile(
    path_to_original_data: Path,
    path_to_no_care_demand_data: Path,
    window_low: int,
    window_high: int,
    outcome_col: str,
    ever_caregivers: bool = False,
    ever_care_demand: bool = False,
) -> pd.DataFrame:
    """Mean outcome by distance in no-care-demand data (same agents, aligned by original first care demand)."""
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)
    df_o = df_o[df_o["health"] != DEAD].copy()
    df_c = df_c[df_c["health"] != DEAD].copy()
    df_o = ensure_agent_period(df_o)
    df_c = ensure_agent_period(df_c)
    if "caregiving_type" in df_o.columns:
        type_1 = df_o[df_o["caregiving_type"] == 1]["agent"].unique()
        df_o = df_o[df_o["agent"].isin(type_1)].copy()
        df_c = df_c[df_c["agent"].isin(type_1)].copy()
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        agent_ids = df_o.loc[df_o["choice"].isin(care_codes), "agent"].unique()
        df_o = df_o[df_o["agent"].isin(agent_ids)].copy()
        df_c = df_c[df_c["agent"].isin(agent_ids)].copy()
    if ever_care_demand:
        agent_ids = df_o.loc[df_o["care_demand"] > 0, "agent"].unique()
        df_o = df_o[df_o["agent"].isin(agent_ids)].copy()
        df_c = df_c[df_c["agent"].isin(agent_ids)].copy()

    df_o_dist = add_distance_to_first_care_demand(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )
    merged_ncd = df_c.merge(dist_map, on="agent", how="inner")
    merged_ncd["distance_to_first_care_demand"] = (
        merged_ncd["period"] - merged_ncd["first_care_demand_period"]
    )
    merged_ncd = merged_ncd[
        merged_ncd["first_care_demand_period"].notna()
        & (merged_ncd["distance_to_first_care_demand"] >= -window_low)
        & (merged_ncd["distance_to_first_care_demand"] <= window_high)
    ]
    merged_ncd = _add_outcome_columns_no_care_demand(merged_ncd)
    if outcome_col not in merged_ncd.columns:
        return pd.DataFrame(columns=["distance_to_first_care_demand", "value"])
    return (
        merged_ncd.groupby("distance_to_first_care_demand", observed=False)[outcome_col]
        .mean()
        .reset_index()
        .rename(columns={outcome_col: "value"})
        .sort_values("distance_to_first_care_demand")
    )


def _compute_ever_caregiver_profile(
    merged: pd.DataFrame,
    outcome_col: str,
) -> pd.DataFrame:
    """Mean outcome by distance for ever-caregiver agents in original data."""
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    ever_cg = (
        merged.groupby("agent", observed=False)["choice"]
        .apply(lambda s: s.isin(care_codes).any())
        .pipe(lambda x: x[x].index)
    )
    sub = merged[merged["agent"].isin(ever_cg)]
    if len(sub) == 0 or outcome_col not in sub.columns:
        return pd.DataFrame(columns=["distance_to_first_care_demand", "value"])
    return (
        sub.groupby("distance_to_first_care_demand", observed=False)[outcome_col]
        .mean()
        .reset_index()
        .rename(columns={outcome_col: "value"})
        .sort_values("distance_to_first_care_demand")
    )


def _add_outcome_columns(merged: pd.DataFrame) -> pd.DataFrame:
    """Add work, pt, ft, working_hours_weekly (annual/52), monthly_gross_labor_income."""
    work, ft, pt = calculate_simple_outcomes(merged, "original")
    merged = merged.copy()
    merged["work"] = work.astype(float)
    merged["part_time"] = pt.astype(float)
    merged["full_time"] = ft.astype(float)

    if "working_hours" in merged.columns:
        wh_annual = merged["working_hours"].astype(float)
    else:
        wh_annual = 0.0
    merged["working_hours_weekly"] = wh_annual / 52.0

    if "gross_labor_income" in merged.columns:
        merged["monthly_gross_labor_income"] = merged["gross_labor_income"] / 12
    else:
        merged["monthly_gross_labor_income"] = 0.0

    return merged


def _add_outcome_columns_no_care_demand(merged: pd.DataFrame) -> pd.DataFrame:
    """Add work, pt, ft, working_hours_weekly, monthly_gross_labor_income from no_care_demand."""
    work, ft, pt = calculate_simple_outcomes(merged, "no_care_demand")
    merged = merged.copy()
    merged["work"] = work.astype(float)
    merged["part_time"] = pt.astype(float)
    merged["full_time"] = ft.astype(float)
    if "working_hours" in merged.columns:
        wh_annual = merged["working_hours"].astype(float)
    else:
        wh_annual = 0.0
    merged["working_hours_weekly"] = wh_annual / 52.0
    if "gross_labor_income" in merged.columns:
        merged["monthly_gross_labor_income"] = merged["gross_labor_income"] / 12
    else:
        merged["monthly_gross_labor_income"] = 0.0
    return merged


def _build_profiles_by_group(
    merged: pd.DataFrame,
    outcome_col: str,
    agents_1: np.ndarray,
    agents_2: np.ndarray,
    agents_3: np.ndarray,
    agents_4: np.ndarray,
    agents_5: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build mean outcome by distance for each of the 5 groups."""
    dist_col = "distance_to_first_care_demand"

    def profile(agents: np.ndarray) -> pd.DataFrame:
        sub = merged[merged["agent"].isin(agents)]
        if len(sub) == 0:
            return pd.DataFrame(columns=[dist_col, "value"])
        return (
            sub.groupby(dist_col, observed=False)[outcome_col]
            .mean()
            .reset_index()
            .rename(columns={outcome_col: "value"})
            .sort_values(dist_col)
        )

    return (
        profile(agents_1),
        profile(agents_2),
        profile(agents_3),
        profile(agents_4),
        profile(agents_5),
    )


def plot_conditional_means_by_distance(
    prof_1_year: pd.DataFrame,
    prof_2_year: pd.DataFrame,
    prof_3_year: pd.DataFrame,
    prof_4_year: pd.DataFrame,
    prof_5_year: pd.DataFrame,
    ylabel: str,
    window_low: int = 20,
    window_high: int = 20,
    path_to_plot: Optional[Path] = None,
    prof_no_care_demand: Optional[pd.DataFrame] = None,
    prof_ever_caregiver: Optional[pd.DataFrame] = None,
) -> None:
    """Plot conditional means (5 groups) along distance to first care demand.

    window_low and window_high are positive integers (years before/after t=0);
    internally window_low is negated for the axis range.
    Uses same marker/color style as publication employment-by-distance plots.
    Optionally add: no-care-demand mean by distance (black solid), ever-caregiver
    mean by distance in original (dashed).
    """
    w_low = -window_low
    S = PUBLICATION_PLOT_STYLE
    plt.figure(figsize=S["figsize"])
    xlabel = "Year relative to start of first care demand"
    dist_col_plot = "distance_to_first_care"

    profiles = [
        prof_1_year.rename(columns={"distance_to_first_care_demand": dist_col_plot}),
        prof_2_year.rename(columns={"distance_to_first_care_demand": dist_col_plot}),
        prof_3_year.rename(columns={"distance_to_first_care_demand": dist_col_plot}),
        prof_4_year.rename(columns={"distance_to_first_care_demand": dist_col_plot}),
        prof_5_year.rename(columns={"distance_to_first_care_demand": dist_col_plot}),
    ]

    for prof, style in zip(profiles, _GROUP_STYLE):
        if len(prof) > 0:
            plt.plot(
                prof[dist_col_plot],
                prof["value"],
                label=style["label"],
                color=style["color"],
                linewidth=S["linewidth"],
                linestyle="-",
                marker=style["marker"],
                markersize=style["markersize"],
                markevery=1,
                markerfacecolor="none",
                markeredgewidth=style.get("markeredgewidth", S["markeredgewidth"]),
            )

    if prof_no_care_demand is not None and len(prof_no_care_demand) > 0:
        p = prof_no_care_demand.rename(
            columns={"distance_to_first_care_demand": dist_col_plot}
        )
        plt.plot(
            p[dist_col_plot],
            p["value"],
            label="No care demand (mean)",
            color="black",
            linewidth=S["linewidth"],
            linestyle="-",
        )
    if prof_ever_caregiver is not None and len(prof_ever_caregiver) > 0:
        p = prof_ever_caregiver.rename(
            columns={"distance_to_first_care_demand": dist_col_plot}
        )
        plt.plot(
            p[dist_col_plot],
            p["value"],
            label="Ever caregiver (original)",
            color="black",
            linewidth=S["linewidth"],
            linestyle="--",
        )

    plt.axvline(
        x=-0.5,
        color="k",
        linestyle=(0, (7, 7)),
        linewidth=S["axvline_linewidth"],
    )
    plt.xlabel(xlabel, fontsize=S["label_fontsize"])
    plt.ylabel(ylabel, fontsize=S["label_fontsize"])
    plt.xlim(w_low - 0.5, window_high + 0.5)
    plt.grid(True, axis="y", alpha=S["grid_alpha"], linewidth=S["grid_linewidth"])
    plt.xticks(range(w_low, window_high + 1, 5), fontsize=S["xtick_fontsize"])
    plt.yticks(fontsize=S["ytick_fontsize"])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=S["tick_length"], width=S["tick_width"])
    plt.legend(loc="best", prop={"size": 10}, framealpha=0.9)
    plt.tight_layout()
    if path_to_plot:
        publication_savefig(path_to_plot)
    plt.close()


# ---------------------------------------------------------------------------
# 5 tasks: employment rate, part time rate, full time rate, working hours
# (weekly), monthly gross labor income. Jan7 data; plot names include
# distance_to_first_care_demand, caregiving_type_1, exact consec 1-4, at least 5.
# ---------------------------------------------------------------------------


@pytask.mark.publication_final
@pytask.task(id="employment_rate")
def task_conditional_means_employment_rate(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_Jan7.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "final"
    / f"conditional_means_employment_rate_{_PLOT_NAME_SUFFIX}.pdf",
    window_low: int = 20,
    window_high: int = 20,
    ever_caregivers: bool = False,
    ever_care_demand: bool = False,
) -> None:
    """Conditional mean employment rate by distance to first care demand (caregiving_type==1)."""
    merged = _prepare_merged_type1(
        path_to_original_data,
        window_low,
        window_high,
        ever_caregivers,
        ever_care_demand,
    )
    merged = _add_outcome_columns(merged)
    agents_1, agents_2, agents_3, agents_4 = identify_agents_by_duration(
        merged,
        distance_col="distance_to_first_care_demand",
        duration_type="care_demand",
    )
    agents_5 = _identify_agents_at_least_5_years_care_demand(merged)
    prof_1, prof_2, prof_3, prof_4, prof_5 = _build_profiles_by_group(
        merged, "work", agents_1, agents_2, agents_3, agents_4, agents_5
    )
    prof_no_care = _compute_no_care_demand_profile(
        path_to_original_data,
        path_to_no_care_demand_data,
        window_low,
        window_high,
        "work",
        ever_caregivers,
        ever_care_demand,
    )
    prof_ever_cg = _compute_ever_caregiver_profile(merged, "work")
    plot_conditional_means_by_distance(
        prof_1,
        prof_2,
        prof_3,
        prof_4,
        prof_5,
        ylabel="Employment Rate",
        window_low=window_low,
        window_high=window_high,
        path_to_plot=path_to_plot,
        prof_no_care_demand=prof_no_care,
        prof_ever_caregiver=prof_ever_cg,
    )


@pytask.mark.publication_final
@pytask.task(id="part_time_rate")
def task_conditional_means_part_time_rate(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_Jan7.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "final"
    / f"conditional_means_part_time_rate_{_PLOT_NAME_SUFFIX}.pdf",
    window_low: int = 20,
    window_high: int = 20,
    ever_caregivers: bool = False,
    ever_care_demand: bool = False,
) -> None:
    """Conditional mean part-time rate by distance to first care demand (caregiving_type==1)."""
    merged = _prepare_merged_type1(
        path_to_original_data,
        window_low,
        window_high,
        ever_caregivers,
        ever_care_demand,
    )
    merged = _add_outcome_columns(merged)
    agents_1, agents_2, agents_3, agents_4 = identify_agents_by_duration(
        merged,
        distance_col="distance_to_first_care_demand",
        duration_type="care_demand",
    )
    agents_5 = _identify_agents_at_least_5_years_care_demand(merged)
    prof_1, prof_2, prof_3, prof_4, prof_5 = _build_profiles_by_group(
        merged, "part_time", agents_1, agents_2, agents_3, agents_4, agents_5
    )
    prof_no_care = _compute_no_care_demand_profile(
        path_to_original_data,
        path_to_no_care_demand_data,
        window_low,
        window_high,
        "part_time",
        ever_caregivers,
        ever_care_demand,
    )
    prof_ever_cg = _compute_ever_caregiver_profile(merged, "part_time")
    plot_conditional_means_by_distance(
        prof_1,
        prof_2,
        prof_3,
        prof_4,
        prof_5,
        ylabel="Part-Time Rate",
        window_low=window_low,
        window_high=window_high,
        path_to_plot=path_to_plot,
        prof_no_care_demand=prof_no_care,
        prof_ever_caregiver=prof_ever_cg,
    )


@pytask.mark.publication_final
@pytask.task(id="full_time_rate")
def task_conditional_means_full_time_rate(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_Jan7.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "final"
    / f"conditional_means_full_time_rate_{_PLOT_NAME_SUFFIX}.pdf",
    window_low: int = 20,
    window_high: int = 20,
    ever_caregivers: bool = False,
    ever_care_demand: bool = False,
) -> None:
    """Conditional mean full-time rate by distance to first care demand (caregiving_type==1)."""
    merged = _prepare_merged_type1(
        path_to_original_data,
        window_low,
        window_high,
        ever_caregivers,
        ever_care_demand,
    )
    merged = _add_outcome_columns(merged)
    agents_1, agents_2, agents_3, agents_4 = identify_agents_by_duration(
        merged,
        distance_col="distance_to_first_care_demand",
        duration_type="care_demand",
    )
    agents_5 = _identify_agents_at_least_5_years_care_demand(merged)
    prof_1, prof_2, prof_3, prof_4, prof_5 = _build_profiles_by_group(
        merged, "full_time", agents_1, agents_2, agents_3, agents_4, agents_5
    )
    prof_no_care = _compute_no_care_demand_profile(
        path_to_original_data,
        path_to_no_care_demand_data,
        window_low,
        window_high,
        "full_time",
        ever_caregivers,
        ever_care_demand,
    )
    prof_ever_cg = _compute_ever_caregiver_profile(merged, "full_time")
    plot_conditional_means_by_distance(
        prof_1,
        prof_2,
        prof_3,
        prof_4,
        prof_5,
        ylabel="Full-Time Rate",
        window_low=window_low,
        window_high=window_high,
        path_to_plot=path_to_plot,
        prof_no_care_demand=prof_no_care,
        prof_ever_caregiver=prof_ever_cg,
    )


@pytask.mark.publication_final
@pytask.task(id="working_hours_weekly")
def task_conditional_means_working_hours_weekly(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_Jan7.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "final"
    / f"conditional_means_working_hours_weekly_{_PLOT_NAME_SUFFIX}.pdf",
    window_low: int = 20,
    window_high: int = 20,
    ever_caregivers: bool = False,
    ever_care_demand: bool = False,
) -> None:
    """Conditional mean working hours (weekly) by distance to first care demand (caregiving_type==1)."""
    merged = _prepare_merged_type1(
        path_to_original_data,
        window_low,
        window_high,
        ever_caregivers,
        ever_care_demand,
    )
    merged = _add_outcome_columns(merged)
    agents_1, agents_2, agents_3, agents_4 = identify_agents_by_duration(
        merged,
        distance_col="distance_to_first_care_demand",
        duration_type="care_demand",
    )
    agents_5 = _identify_agents_at_least_5_years_care_demand(merged)
    prof_1, prof_2, prof_3, prof_4, prof_5 = _build_profiles_by_group(
        merged, "working_hours_weekly", agents_1, agents_2, agents_3, agents_4, agents_5
    )
    prof_no_care = _compute_no_care_demand_profile(
        path_to_original_data,
        path_to_no_care_demand_data,
        window_low,
        window_high,
        "working_hours_weekly",
        ever_caregivers,
        ever_care_demand,
    )
    prof_ever_cg = _compute_ever_caregiver_profile(merged, "working_hours_weekly")
    plot_conditional_means_by_distance(
        prof_1,
        prof_2,
        prof_3,
        prof_4,
        prof_5,
        ylabel="Working Hours (weekly)",
        window_low=window_low,
        window_high=window_high,
        path_to_plot=path_to_plot,
        prof_no_care_demand=prof_no_care,
        prof_ever_caregiver=prof_ever_cg,
    )


@pytask.mark.publication_final
@pytask.task(id="monthly_gross_labor_income")
def task_conditional_means_monthly_gross_labor_income(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_Jan7.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "final"
    / f"conditional_means_monthly_gross_labor_income_{_PLOT_NAME_SUFFIX}.pdf",
    window_low: int = 20,
    window_high: int = 20,
    ever_caregivers: bool = False,
    ever_care_demand: bool = False,
) -> None:
    """Conditional mean monthly gross labor income by distance to first care demand (caregiving_type==1)."""
    merged = _prepare_merged_type1(
        path_to_original_data,
        window_low,
        window_high,
        ever_caregivers,
        ever_care_demand,
    )
    merged = _add_outcome_columns(merged)
    agents_1, agents_2, agents_3, agents_4 = identify_agents_by_duration(
        merged,
        distance_col="distance_to_first_care_demand",
        duration_type="care_demand",
    )
    agents_5 = _identify_agents_at_least_5_years_care_demand(merged)
    prof_1, prof_2, prof_3, prof_4, prof_5 = _build_profiles_by_group(
        merged,
        "monthly_gross_labor_income",
        agents_1,
        agents_2,
        agents_3,
        agents_4,
        agents_5,
    )
    prof_no_care = _compute_no_care_demand_profile(
        path_to_original_data,
        path_to_no_care_demand_data,
        window_low,
        window_high,
        "monthly_gross_labor_income",
        ever_caregivers,
        ever_care_demand,
    )
    prof_ever_cg = _compute_ever_caregiver_profile(merged, "monthly_gross_labor_income")
    plot_conditional_means_by_distance(
        prof_1,
        prof_2,
        prof_3,
        prof_4,
        prof_5,
        ylabel="Monthly Gross Labor Income",
        window_low=window_low,
        window_high=window_high,
        path_to_plot=path_to_plot,
        prof_no_care_demand=prof_no_care,
        prof_ever_caregiver=prof_ever_cg,
    )
