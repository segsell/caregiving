"""Helper functions for plotting counterfactual differences.

This module provides reusable functions to reduce code duplication in plotting tasks.
"""

from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.config import JET_COLOR_MAP
from caregiving.counterfactual.plotting_utils import ensure_agent_period
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    PART_TIME,
    RETIREMENT,
    WORK,
)
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)

# ============================================================================
# Helper functions for data preparation and outcome calculation
# ============================================================================


def calculate_simple_outcomes(
    df: pd.DataFrame, choice_set_type: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate simple work, ft, pt outcomes from choice column.

    Args:
        df: DataFrame with 'choice' column
        choice_set_type: 'original', 'no_care_demand', or 'job_retention'

    Returns:
        Tuple of (work, ft, pt) Series
    """
    if choice_set_type == "original":
        work_values = np.asarray(WORK).ravel().tolist()
        ft_values = np.asarray(FULL_TIME).ravel().tolist()
        pt_values = np.asarray(PART_TIME).ravel().tolist()
    elif choice_set_type == "no_care_demand":
        work_values = np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist()
        ft_values = np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist()
        pt_values = np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist()
    elif choice_set_type == "job_retention":
        # Job retention uses same choice structure as original
        work_values = np.asarray(WORK).ravel().tolist()
        ft_values = np.asarray(FULL_TIME).ravel().tolist()
        pt_values = np.asarray(PART_TIME).ravel().tolist()
    else:
        raise ValueError(
            f"choice_set_type must be 'original', 'no_care_demand', "
            f"or 'job_retention', got {choice_set_type}"
        )

    work = df["choice"].isin(work_values).astype(float)
    ft = df["choice"].isin(ft_values).astype(float)
    pt = df["choice"].isin(pt_values).astype(float)

    return work, ft, pt


def prepare_dataframes_simple(
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    ever_caregivers: bool,
    ever_care_demand: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare DataFrames with simple outcome calculation pattern.

    This is used by functions that calculate simple work/ft/pt outcomes
    rather than using the full outcome calculation utilities.

    Args:
        df_o: Original DataFrame
        df_c: Counterfactual DataFrame
        ever_caregivers: If True, filter both df_o and df_c to agents who ever
            provided informal care (choice in INFORMAL_CARE) in df_o
        ever_care_demand: If True, filter both df_o and df_c to agents who ever
            experienced care demand in df_o

    Returns:
        Tuple of prepared DataFrames
    """
    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()
    df_c = df_c[df_c["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = ensure_agent_period(df_o)
    df_c = ensure_agent_period(df_c)

    # Fully flatten any residual index levels named 'agent' or 'period'
    if isinstance(df_o.index, pd.MultiIndex):
        idx_names_o = {n for n in df_o.index.names if n is not None}
        if ("agent" in idx_names_o) or ("period" in idx_names_o):
            df_o = df_o.reset_index()
    if isinstance(df_c.index, pd.MultiIndex):
        idx_names_c = {n for n in df_c.index.names if n is not None}
        if ("agent" in idx_names_c) or ("period" in idx_names_c):
            df_c = df_c.reset_index()

    # Ensure no index name collisions remain (fully flatten)
    df_o = df_o.reset_index(drop=True)
    df_c = df_c.reset_index(drop=True)

    # Ever-caregiver restriction
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_o.loc[df_o["choice"].isin(care_codes), "agent"].unique()
        df_o = df_o[df_o["agent"].isin(caregiver_ids)].copy()
        df_c = df_c[df_c["agent"].isin(caregiver_ids)].copy()

    # Ever-care-demand restriction
    if ever_care_demand:
        care_demand_ids = df_o.loc[df_o["care_demand"] > 0, "agent"].unique()
        df_o = df_o[df_o["agent"].isin(care_demand_ids)].copy()
        df_c = df_c[df_c["agent"].isin(care_demand_ids)].copy()

    return df_o, df_c


def get_age_at_first_event(
    df: pd.DataFrame,
    event_mask: pd.Series,
    age_col_name: str = "age_at_first_care",
) -> pd.DataFrame:
    """Get age at first event for each agent.

    Args:
        df: DataFrame with 'agent', 'period', and 'age' columns
        event_mask: Boolean mask indicating event occurrence
        age_col_name: Name for the age column in output

    Returns:
        DataFrame with 'agent' and age_col_name columns
    """

    first_event = (
        df.loc[event_mask, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(columns={"age": age_col_name})
    )

    return first_event[["agent", age_col_name]]


def add_distance_to_first_care(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care column; 0 is first period providing informal care."""
    df = df_original.reset_index(drop=True)
    df = ensure_agent_period(df)
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df["choice"].isin(care_codes)
    first_care = (
        df.loc[caregiving_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_care_period"})
    )
    out = df.merge(first_care, on="agent", how="left")
    out["distance_to_first_care"] = out["period"] - out["first_care_period"]
    return out


def add_distance_to_first_care_demand(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add first_care_demand_period and distance; 0 is first period with care_demand > 0."""
    df = df_original.reset_index(drop=True)
    df = ensure_agent_period(df)
    care_demand_mask = df["care_demand"] > 0
    first_care_demand = (
        df.loc[care_demand_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_care_demand_period"})
    )
    out = df.merge(first_care_demand, on="agent", how="left")
    out["distance_to_first_care_demand"] = (
        out["period"] - out["first_care_demand_period"]
    )
    return out


def identify_agents_by_total_caregiving_over_lifecycle(
    df_full: pd.DataFrame,
    start_age: int,
    end_age_caregiving: int,
    informal_care_choices: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify agents by total caregiving years over lifecycle up to end_age_caregiving.

    Counts periods with informal caregiving from start until age <= end_age_caregiving.
    Groups: exactly 1, 2, 3, 4, or 5+ total years (not necessarily consecutive).

    Args:
        df_full: DataFrame with agent, period, choice
        start_age: Agent age at period 0 (age = period + start_age)
        end_age_caregiving: Max age (inclusive) for caregiving; from specs.
        informal_care_choices: Choice codes for informal care; default INFORMAL_CARE

    Returns:
        Tuple of (agents_1_year, agents_2_year, agents_3_year, agents_4_year, agents_5_year)
    """
    if informal_care_choices is None:
        informal_care_choices = np.asarray(INFORMAL_CARE).ravel().tolist()
    df = df_full.copy()
    df["age"] = df["period"] + start_age
    df = df[df["age"] <= end_age_caregiving]
    df["current_caregiving"] = df["choice"].isin(informal_care_choices).astype(int)
    total_care = (
        df.groupby("agent", observed=False)["current_caregiving"].sum().astype(int)
    )
    agents_1_year = total_care[total_care == 1].index.to_numpy()
    agents_2_year = total_care[total_care == 2].index.to_numpy()
    agents_3_year = total_care[total_care == 3].index.to_numpy()
    agents_4_year = total_care[total_care == 4].index.to_numpy()
    agents_5_year = total_care[total_care >= 5].index.to_numpy()
    return (agents_1_year, agents_2_year, agents_3_year, agents_4_year, agents_5_year)


# ============================================================================
# Event study (total caregiving years): shared constants and helpers
# ============================================================================

# Distance column name used in event-study profile DataFrames
EVENT_STUDY_DIST_COL = "distance_to_first_care"

# Age groups for event-study tasks: (age_min, age_max, age_label)
AGE_GROUPS_EVENT_STUDY = (
    (None, None, "all_ages"),
    # (40, 49, "ages_40_49"),
    # (50, 59, "ages_50_59"),
    # (60, 70, "ages_60_70"),
)

_WORK_SET = set(np.asarray(WORK).ravel().tolist())
_RETIREMENT_SET = set(np.asarray(RETIREMENT).ravel().tolist())


def job_offer_outcome_series(
    df: pd.DataFrame,
    kind: Literal["job_finding", "job_retention"],
) -> pd.Series:
    """Build outcome series for job finding or job retention (conditional on lagged status).

    job_finding: mean(job_offer) among (agent, period) where previous period was
        not working and not retired (unemployed). job_retention: mean(job_offer)
        among (agent, period) where previous period was working (1 - separation).
    Returns a series with same index as df; NaN where condition not met (excluded from mean).
    """
    if "job_offer" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    df_prev = df[["agent", "period", "choice"]].copy()
    df_prev["period"] = df_prev["period"] + 1
    df_prev = df_prev.rename(columns={"choice": "lagged_choice"})
    merged = df.merge(df_prev, on=["agent", "period"], how="left")
    # Use right-side column after merge (suffix _y if df already had 'lagged_choice')
    lagged_col = (
        "lagged_choice_y" if "lagged_choice_y" in merged.columns else "lagged_choice"
    )
    merged["lagged_working"] = merged[lagged_col].isin(_WORK_SET)
    merged["lagged_retired"] = merged[lagged_col].isin(_RETIREMENT_SET)
    if kind == "job_finding":
        mask = (
            (~merged["lagged_working"])
            & (~merged["lagged_retired"])
            & merged[lagged_col].notna()
        )
    else:
        mask = merged["lagged_working"].fillna(False)
    outcome = merged["job_offer"].astype(float).where(mask)
    outcome.index = df.index
    return outcome


def _plot_outcome_difference_by_distance_total_caregiving_impl(  # noqa: PLR0913
    prof_diff: pd.DataFrame,
    prof_1_year_diff: pd.DataFrame,
    prof_2_year_diff: pd.DataFrame,
    prof_3_year_diff: pd.DataFrame,
    prof_4_year_diff: pd.DataFrame,
    prof_5_year_diff: pd.DataFrame,
    window_low: int,
    window_high: int,
    path_to_plot: Optional[Path],
    xlabel: str,
    ylabel: str,
    endogenous_ylim: bool,
    label_fontsize: float,
    xtick_fontsize: float,
    ytick_fontsize: float,
    tick_length: float,
    tick_width: float,
    font_family: Optional[str] = None,
) -> None:
    """Implementation: plot outcome difference by distance (total care years 1–5+)."""
    if font_family is not None:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [font_family]
    plt.figure(figsize=(14, 12))

    plt.plot(
        prof_diff[EVENT_STUDY_DIST_COL],
        prof_diff["diff"],
        label="Baseline",
        color="black",
        linewidth=3.7,
        linestyle="--",
        marker=None,
    )
    plt.axhline(y=0, color="k", linestyle="-", linewidth=1.6)

    def _plot_prof(
        prof: pd.DataFrame,
        label: str,
        color: str,
        marker: str,
        markersize: float = 9,
        markeredgewidth: float = 3.0,
    ) -> None:
        if len(prof) > 0:
            plt.plot(
                prof[EVENT_STUDY_DIST_COL],
                prof["diff"],
                label=label,
                color=color,
                linewidth=3.7,
                linestyle="-",
                marker=marker,
                markersize=markersize,
                markevery=1,
                markerfacecolor="none",
                markeredgewidth=markeredgewidth,
            )

    _plot_prof(prof_1_year_diff, "1 total care year", "0.85", "8")
    _plot_prof(prof_2_year_diff, "2 total care years", "0.7", "^")
    _plot_prof(prof_3_year_diff, "3 total care years", "0.5", "D")
    _plot_prof(prof_4_year_diff, "4 total care years", "0.3", "s")
    _plot_prof(
        prof_5_year_diff,
        "5+ total care years",
        "0.1",
        "*",
        markersize=12,
        markeredgewidth=3.5,
    )

    plt.axvline(
        x=-0.5,
        color="k",
        linestyle=(0, (7, 7)),
        linewidth=1.85,
    )
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xlim(window_low - 0.5, window_high + 0.5)

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
            # Add small margin so bottom y-tick is not clamped to x-axis
            y_margin = 0.03 * (2 * y_lim)
            plt.ylim(-y_lim - y_margin, y_lim + y_margin)
    else:
        plt.ylim(-0.1, 0.1)

    tick_step = 5
    plt.grid(True, axis="y", alpha=0.18, linewidth=1.15)
    plt.xticks(range(window_low, window_high + 1, tick_step), fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=tick_length, width=tick_width)
    # Explicitly set font on all text so Times New Roman (or requested font) is used
    if font_family is not None:
        ax.xaxis.get_label().set_fontfamily(font_family)
        ax.yaxis.get_label().set_fontfamily(font_family)
        for label in ax.get_xticklabels():
            label.set_fontfamily(font_family)
        for label in ax.get_yticklabels():
            label.set_fontfamily(font_family)
    plt.tight_layout()
    # Ensure gap between bottom y-tick label and x-axis so layout is not clamped
    plt.subplots_adjust(bottom=0.11)
    if path_to_plot:
        path_to_plot.parent.mkdir(parents=True, exist_ok=True)
        # Use pad_inches so labels/numbers are not cut off; embed fonts in PDF for crisp text
        save_kwargs = {
            "dpi": 1200,
            "bbox_inches": "tight",
            "pad_inches": 0.25,
        }
        is_pdf = path_to_plot.suffix.lower() == ".pdf"
        if is_pdf:
            _pdf_fonttype = plt.rcParams["pdf.fonttype"]
            plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF for sharp text
        plt.savefig(path_to_plot, **save_kwargs)
        if is_pdf:
            plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close()


def plot_outcome_difference_by_distance_total_caregiving(  # noqa: PLR0913
    prof_diff: pd.DataFrame,
    prof_1_year_diff: pd.DataFrame,
    prof_2_year_diff: pd.DataFrame,
    prof_3_year_diff: pd.DataFrame,
    prof_4_year_diff: pd.DataFrame,
    prof_5_year_diff: pd.DataFrame,
    window_low: int = 20,
    window_high: int = 20,
    path_to_plot: Optional[Path] = None,
    xlabel: str = "Year relative to start of first care spell",
    ylabel: str = "Difference in outcome",
    endogenous_ylim: bool = False,
    font_family: Optional[str] = "Times New Roman",
    label_fontsize: float = 28,
    xtick_fontsize: float = 25,
    ytick_fontsize: float = 25,
    tick_length: float = 16,
    tick_width: float = 1.85,
) -> None:
    """Plot outcome difference by distance with 5 lines: total care years 1, 2, 3, 4, 5+.

    Same layout as event study consecutive: dashed black baseline, horizontal line at 0,
    vertical line at t=-0.5, five subgroup lines (1, 2, 3, 4, 5+ total care years).
    Profile DataFrames must have column EVENT_STUDY_DIST_COL and 'diff'.

    window_low and window_high are positive integers (years before/after t=0);
    internally window_low is negated for the axis range.
    """
    w_low = -window_low
    if font_family is not None:
        with plt.rc_context({"font.family": "serif", "font.serif": [font_family]}):
            _plot_outcome_difference_by_distance_total_caregiving_impl(
                prof_diff,
                prof_1_year_diff,
                prof_2_year_diff,
                prof_3_year_diff,
                prof_4_year_diff,
                prof_5_year_diff,
                w_low,
                window_high,
                path_to_plot,
                xlabel,
                ylabel,
                endogenous_ylim,
                label_fontsize,
                xtick_fontsize,
                ytick_fontsize,
                tick_length,
                tick_width,
                font_family,
            )
    else:
        _plot_outcome_difference_by_distance_total_caregiving_impl(
            prof_diff,
            prof_1_year_diff,
            prof_2_year_diff,
            prof_3_year_diff,
            prof_4_year_diff,
            prof_5_year_diff,
            w_low,
            window_high,
            path_to_plot,
            xlabel,
            ylabel,
            endogenous_ylim,
            label_fontsize,
            xtick_fontsize,
            ytick_fontsize,
            tick_length,
            tick_width,
            None,
        )


def event_study_total_caregiving_merged_and_profiles(
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    outcome_o_series: pd.Series,
    outcome_c_series: pd.Series,
    window_low: int,
    window_high: int,
    age_min: int | None,
    age_max: int | None,
    event_type: Literal["care_demand", "caregiving_spell"],
    start_age: int,
    end_age_caregiving: int,
    compare_against_baseline: bool = True,
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
    All profiles have columns EVENT_STUDY_DIST_COL and 'diff'.

    When compare_against_baseline is True, diff = policy − baseline (outcome_c − outcome_o).
    When False, diff = baseline − policy (outcome_o − outcome_c).

    window_low and window_high are positive integers (years before/after t=0);
    internally window_low is negated for the distance filter.
    """
    w_low = -window_low
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
        & (merged["distance_raw"] >= w_low)
        & (merged["distance_raw"] <= window_high)
    ]
    if age_min is not None:
        merged = merged[merged[age_col] >= age_min].copy()
    if age_max is not None:
        merged = merged[merged[age_col] <= age_max].copy()

    if compare_against_baseline:
        merged["diff"] = merged["outcome_c"] - merged["outcome_o"]
    else:
        merged["diff"] = merged["outcome_o"] - merged["outcome_c"]
    merged[EVENT_STUDY_DIST_COL] = merged["distance_raw"]

    prof_diff = (
        merged.groupby(EVENT_STUDY_DIST_COL, observed=False)["diff"]
        .mean()
        .reset_index()
        .sort_values(EVENT_STUDY_DIST_COL)
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
            return pd.DataFrame(columns=[EVENT_STUDY_DIST_COL, "diff"])
        p = (
            m.groupby(EVENT_STUDY_DIST_COL, observed=False)["diff"]
            .mean()
            .reset_index()
            .sort_values(EVENT_STUDY_DIST_COL)
        )
        return p

    prof_1 = _prof_for_agents(agents_1_year)
    prof_2 = _prof_for_agents(agents_2_year)
    prof_3 = _prof_for_agents(agents_3_year)
    prof_4 = _prof_for_agents(agents_4_year)
    prof_5 = _prof_for_agents(agents_5_year)

    return merged, prof_diff, prof_1, prof_2, prof_3, prof_4, prof_5


def get_distinct_colors(n: int) -> list[str]:
    """Get distinct colors for plotting.

    Args:
        n: Number of colors needed

    Returns:
        List of color strings
    """
    distinct_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]
    return [distinct_colors[i % len(distinct_colors)] for i in range(n)]


# ============================================================================
# Helper functions for plotting
# ============================================================================


def plot_three_line_differences(
    prof: pd.DataFrame,
    x_col: str,
    path_to_plot: Path,
    xlabel: str,
    window: int,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot three-line difference plot (work, ft, pt).

    Args:
        prof: Profile DataFrame with x_col and diff_work, diff_ft, diff_pt columns
        x_col: Name of x-axis column
        path_to_plot: Path to save plot
        xlabel: X-axis label
        window: Window size for xlim
        ylim: Y-axis limits. If None, automatically calculated from data with padding.
    """
    plt.figure(figsize=(12, 7))

    plt.plot(
        prof[x_col],
        prof["diff_work"],
        label="Working",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        prof[x_col],
        prof["diff_ft"],
        label="Full Time",
        color=JET_COLOR_MAP[1],
        linewidth=2,
    )
    plt.plot(
        prof[x_col],
        prof["diff_pt"],
        label="Part Time",
        color=JET_COLOR_MAP[0],
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Proportion Working\nDeviation from Counterfactual", fontsize=16)
    plt.xlim(-window, window)

    # Calculate ylim from data if not provided
    if ylim is None:
        # Filter out NaN and Inf values
        work_vals = prof["diff_work"].replace([np.inf, -np.inf], np.nan).dropna()
        ft_vals = prof["diff_ft"].replace([np.inf, -np.inf], np.nan).dropna()
        pt_vals = prof["diff_pt"].replace([np.inf, -np.inf], np.nan).dropna()

        if len(work_vals) == len(ft_vals) == len(pt_vals) == 0:
            # No valid data, use default range
            ylim = (-0.1, 0.1)
        else:
            y_min = min(
                work_vals.min() if len(work_vals) > 0 else 0,
                ft_vals.min() if len(ft_vals) > 0 else 0,
                pt_vals.min() if len(pt_vals) > 0 else 0,
            )
            y_max = max(
                work_vals.max() if len(work_vals) > 0 else 0,
                ft_vals.max() if len(ft_vals) > 0 else 0,
                pt_vals.max() if len(pt_vals) > 0 else 0,
            )
            # Add 10% padding on both sides
            y_range = y_max - y_min
            if y_range == 0:
                # All values are the same, add small padding
                ylim = (y_min - 0.01, y_max + 0.01)
            else:
                padding = max(y_range * 0.1, 0.01)  # At least 0.01 padding
                ylim = (y_min - padding, y_max + padding)

    plt.ylim(ylim)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        ["Working", "Full Time", "Part Time"],
        loc="lower left",
        bbox_to_anchor=(0.05, 0.05),
        prop={"size": 16},
    )
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multi_line_differences_by_group(
    prof: pd.DataFrame,
    x_col: str,
    group_col: str,
    diff_col: str,
    groups: list,
    colors: list[str],
    path_to_plot: Path,
    xlabel: str,
    ylabel: str,
    window: int,
    legend_title: str,
) -> None:
    """Plot multi-line differences grouped by a categorical variable.

    Args:
        prof: Profile DataFrame with x_col, group_col, and diff_col
        x_col: Name of x-axis column
        group_col: Name of grouping column
        diff_col: Name of difference column to plot
        groups: List of group values to plot
        colors: List of colors (one per group)
        path_to_plot: Path to save plot
        xlabel: X-axis label
        ylabel: Y-axis label
        window: Window size for xlim
        legend_title: Title for legend
    """
    plt.figure(figsize=(12, 7))
    for i, group_val in enumerate(sorted(groups)):
        prof_group = prof[prof[group_col] == group_val].sort_values(x_col)
        plt.plot(
            prof_group[x_col],
            prof_group[diff_col],
            label=f"Age {group_val}",
            color=colors[i],
            linewidth=2,
        )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        loc="best",
        prop={"size": 14},
        title=legend_title,
        title_fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_outcomes_by_group(
    prof: pd.DataFrame,
    x_col: str,
    group_col: str,
    groups: list,
    colors: list[str],
    plot_configs: dict[str, dict],
    window: int,
    legend_title: str,
    default_xlabel: str = "Year relative to start of first care spell",
) -> None:
    """Plot multiple outcomes in separate figures, grouped by a categorical variable.

    Args:
        prof: Profile DataFrame with x_col, group_col, and diff_* columns
        x_col: Name of x-axis column
        group_col: Name of grouping column
        groups: List of group values to plot
        colors: List of colors (one per group)
        plot_configs: Dict mapping outcome name to config dict with keys:
            'path', 'ylabel', 'diff_col', optionally 'xlabel'
        window: Window size for xlim
        legend_title: Title for legend
        default_xlabel: Default xlabel if not specified in config
    """
    for config in plot_configs.values():
        # Skip configurations where the diff_col doesn't exist in the dataframe
        diff_col = config["diff_col"]
        if diff_col not in prof.columns:
            print(f"Skipping plot for {diff_col} - column not found in dataframe")
            continue

        plot_multi_line_differences_by_group(
            prof=prof,
            x_col=x_col,
            group_col=group_col,
            diff_col=diff_col,
            groups=groups,
            colors=colors,
            path_to_plot=config["path"],
            xlabel=config.get("xlabel", default_xlabel),
            ylabel=config["ylabel"],
            window=window,
            legend_title=legend_title,
        )


def plot_all_outcomes_by_age(
    prof: pd.DataFrame,
    plot_configs: dict[str, dict],
    age_min: int = 30,
    age_max: int = 100,
) -> None:
    """Plot all outcomes by age using the configurations provided.

    Args:
        prof: DataFrame with aggregated differences by age
        plot_configs: Dictionary with plot configurations for each outcome.
            Each config can optionally include an "age_max" key to override
            the default age_max for that specific outcome.
        age_min: Minimum age for x-axis (default for all outcomes)
        age_max: Maximum age for x-axis (default, can be overridden per outcome)
    """
    for config in plot_configs.values():
        # Skip configurations where the diff_col doesn't exist in the dataframe
        diff_col = config["diff_col"]
        if diff_col not in prof.columns:
            print(f"Skipping plot for {diff_col} - column not found in dataframe")
            continue

        # Use outcome-specific age_max if provided, otherwise use default
        outcome_age_max = config.get("age_max", age_max)

        plot_single_line_differences_by_age(
            prof=prof,
            path_to_plot=config["path"],
            ylabel=config["ylabel"],
            diff_col=diff_col,
            age_min=age_min,
            age_max=outcome_age_max,
        )


def plot_single_line_differences_by_age(
    prof: pd.DataFrame,
    path_to_plot: Path,
    ylabel: str,
    diff_col: str,
    age_min: int = 30,
    age_max: int = 100,
) -> None:
    """Plot a single outcome difference by age.

    Args:
        prof: DataFrame with age and difference columns
        path_to_plot: Path to save the plot
        ylabel: Y-axis label
        diff_col: Column name for the difference values
        age_min: Minimum age for x-axis
        age_max: Maximum age for x-axis
    """
    plt.figure(figsize=(12, 7))

    # Sort by age for proper line plotting
    prof_sorted = prof.sort_values("age")

    plt.plot(
        prof_sorted["age"],
        prof_sorted[diff_col],
        color="black",
        linewidth=2,
    )

    plt.axhline(y=0, color="k", linestyle=":", alpha=0.5, linewidth=1)
    plt.xlabel("Age", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlim(age_min, age_max)

    # Calculate ylim from data endogenously (with padding)
    # Ensure zero line is always visible
    filtered_data = prof_sorted[
        (prof_sorted["age"] >= age_min) & (prof_sorted["age"] <= age_max)
    ][diff_col]
    if len(filtered_data) > 0 and not filtered_data.isna().all():
        y_min = filtered_data.min()
        y_max = filtered_data.max()

        # Ensure zero is always included in the range
        y_min = min(y_min, 0)
        y_max = max(y_max, 0)

        # Add 10% padding on both sides
        y_range = y_max - y_min
        padding = max(y_range * 0.1, 0.01)  # At least 0.01 padding
        plt.ylim(y_min - padding, y_max + padding)
    else:
        # If no data, set a default range that includes zero
        plt.ylim(-0.1, 0.1)

    plt.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    plt.tick_params(labelsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
