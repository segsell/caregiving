"""Helper functions for plotting counterfactual differences.

This module provides reusable functions to reduce code duplication in plotting tasks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.config import JET_COLOR_MAP
from caregiving.counterfactual.plotting_utils import _ensure_agent_period
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    PART_TIME,
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare DataFrames with simple outcome calculation pattern.

    This is used by functions that calculate simple work/ft/pt outcomes
    rather than using the full outcome calculation utilities.

    Args:
        df_o: Original DataFrame
        df_c: Counterfactual DataFrame
        ever_caregivers: If True, filter to ever-caregivers

    Returns:
        Tuple of prepared DataFrames
    """
    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()
    df_c = df_c[df_c["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = _ensure_agent_period(df_o)
    df_c = _ensure_agent_period(df_c)

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
    if "age" not in df.columns:
        raise ValueError("Age column required but not found in data")

    first_event = (
        df.loc[event_mask, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(columns={"age": age_col_name})
    )

    return first_event[["agent", age_col_name]]


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
        y_min = min(
            prof["diff_work"].min(),
            prof["diff_ft"].min(),
            prof["diff_pt"].min(),
        )
        y_max = max(
            prof["diff_work"].max(),
            prof["diff_ft"].max(),
            prof["diff_pt"].max(),
        )
        # Add 10% padding on both sides
        y_range = y_max - y_min
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
        plot_multi_line_differences_by_group(
            prof=prof,
            x_col=x_col,
            group_col=group_col,
            diff_col=config["diff_col"],
            groups=groups,
            colors=colors,
            path_to_plot=config["path"],
            xlabel=config.get("xlabel", default_xlabel),
            ylabel=config["ylabel"],
            window=window,
            legend_title=legend_title,
        )
