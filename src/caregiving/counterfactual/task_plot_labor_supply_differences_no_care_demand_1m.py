"""Plot differences in labor supply by distance for no care demand counterfactual.

Original and no-care-demand scenario.
For simulations with 1,000,000 agents.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.counterfactual.task_plot_labor_supply_differences import (
    _add_distance_to_first_care,
    _ensure_agent_period,
)
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


@pytask.mark.counterfactual_differences_no_care_demand_1m
def task_plot_matched_differences_by_distance(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_1m.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_1m.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_by_distance_no_care_demand_1m.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (orig - no-care-demand), then average by distance.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care from original, attach to merged.
      6) Average diffs by distance and plot three series.

    """

    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

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

    # Outcomes per period
    o_work = df_o["choice"].isin(np.asarray(WORK).ravel().tolist()).astype(float)
    o_ft = df_o["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    o_pt = df_o["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    # No care demand uses reduced choice structure (4 choices)
    c_work = (
        df_c["choice"]
        .isin(np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_ft = (
        df_c["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_pt = (
        df_c["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    o_cols = df_o[["agent", "period"]].copy()
    o_cols["work_o"] = o_work
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["work_c"] = c_work
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt

    # Merge on (agent, period) to get matched differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_o"] - merged["work_c"]
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]

    # Compute distance in original and attach
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care", observed=False)[
            ["diff_work", "diff_ft", "diff_pt"]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Plot
    plt.figure(figsize=(12, 7))

    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_work"],
        label="Working",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_ft"],
        label="Full Time",
        color=JET_COLOR_MAP[1],
        linewidth=2,
    )
    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_pt"],
        label="Part Time",
        color=JET_COLOR_MAP[0],
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel("Proportion Working\nDeviation from Counterfactual", fontsize=16)
    plt.xlim(-window, window)
    plt.ylim(-0.125, 0.025)
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend with increased font size, positioned closer to (0,0) point
    plt.legend(
        ["Working", "Full Time", "Part Time"],
        loc="lower left",
        bbox_to_anchor=(0.05, 0.05),
        prop={"size": 16},
    )
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_no_care_demand_1m
def task_plot_matched_differences_by_age_at_first_care(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_1m.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_1m.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_part_time_by_age_at_first_care_1m.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_full_time_by_age_at_first_care_1m.png",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care: list[int] = [50, 54, 58, 62],
) -> None:
    """Compute matched period differences by age at first care spell.

    Creates separate plots for part-time and full-time work, with separate lines
    for each age at which caregiving started.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care and age_at_first_care from original.
      6) Filter to specific ages at first care.
      7) Average diffs by distance and age_at_first_care.
      8) Plot separate figures for PT and FT with one line per starting age.

    """
    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

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

    # Outcomes per period
    o_ft = df_o["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    o_pt = df_o["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    # No care demand uses reduced choice structure (4 choices)
    c_ft = (
        df_c["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_pt = (
        df_c["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    o_cols = df_o[["agent", "period"]].copy()
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt

    # Merge on (agent, period) to get matched differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]

    # Compute distance and age at first care from original
    df_o_dist = _add_distance_to_first_care(df_o)

    # Get first care period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    # Merge first_care_period with original data to get age at that period
    if "age" not in df_o.columns:
        raise ValueError("Age column required but not found in original data")

    first_care_with_age = (
        df_o[["agent", "period", "age"]]
        .merge(dist_map, on="agent", how="left")
        .query("period == first_care_period")
        .groupby("agent", observed=False)["age"]
        .first()
        .reset_index()
        .rename(columns={"age": "age_at_first_care"})
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    # Filter to specific ages at first care
    merged = merged[merged["age_at_first_care"].isin(ages_at_first_care)]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance and age_at_first_care
    prof = (
        merged.groupby(["distance_to_first_care", "age_at_first_care"], observed=False)[
            ["diff_ft", "diff_pt"]
        ]
        .mean()
        .reset_index()
        .sort_values(["age_at_first_care", "distance_to_first_care"])
    )

    # Define distinct colors for each age
    # Using a palette of clearly distinguishable colors
    distinct_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]
    # Map colors to ages (cycling if more ages than colors)
    n_ages = len(ages_at_first_care)
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(n_ages)]

    # Plot Part-Time
    plt.figure(figsize=(12, 7))
    for i, age in enumerate(sorted(ages_at_first_care)):
        prof_age = prof[prof["age_at_first_care"] == age].sort_values(
            "distance_to_first_care"
        )
        plt.plot(
            prof_age["distance_to_first_care"],
            prof_age["diff_pt"],
            label=f"Age {age}",
            color=colors[i],
            linewidth=2,
        )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel(
        "Proportion Part-Time Working\nDeviation from Counterfactual", fontsize=16
    )
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        loc="best",
        prop={"size": 14},
        title="Age at first care",
        title_fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_pt, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Full-Time
    plt.figure(figsize=(12, 7))
    for i, age in enumerate(sorted(ages_at_first_care)):
        prof_age = prof[prof["age_at_first_care"] == age].sort_values(
            "distance_to_first_care"
        )
        plt.plot(
            prof_age["distance_to_first_care"],
            prof_age["diff_ft"],
            label=f"Age {age}",
            color=colors[i],
            linewidth=2,
        )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel(
        "Proportion Full-Time Working\nDeviation from Counterfactual", fontsize=16
    )
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        loc="best",
        prop={"size": 14},
        title="Age at first care",
        title_fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_ft, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_no_care_demand_1m
def task_plot_matched_differences_by_age_bins_at_first_care(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_1m.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_1m.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_part_time_by_age_bins_at_first_care_1m.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_full_time_by_age_bins_at_first_care_1m.png",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care spell.

    Creates separate plots for part-time and full-time work, with separate lines
    for each age bin at which caregiving started (e.g., 50-53, 54-57, etc.).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care and age_at_first_care from original.
      6) Group ages into bins.
      7) Average diffs by distance and age_bin_at_first_care.
      8) Plot separate figures for PT and FT with one line per age bin.

    """
    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

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

    # Outcomes per period
    o_ft = df_o["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    o_pt = df_o["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    # No care demand uses reduced choice structure (4 choices)
    c_ft = (
        df_c["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_pt = (
        df_c["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    o_cols = df_o[["agent", "period"]].copy()
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt

    # Merge on (agent, period) to get matched differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]

    # Compute distance and age at first care from original
    df_o_dist = _add_distance_to_first_care(df_o)

    # Get first care period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )

    # Get age at first care period
    if "age" not in df_o.columns:
        raise ValueError("Age column required but not found in original data")

    first_care_with_age = (
        df_o[["agent", "period", "age"]]
        .merge(dist_map, on="agent", how="left")
        .query("period == first_care_period")
        .groupby("agent", observed=False)["age"]
        .first()
        .reset_index()
        .rename(columns={"age": "age_at_first_care"})
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]
    merged = merged.merge(first_care_with_age, on="agent", how="left")

    # Filter to age range
    merged = merged[
        (merged["age_at_first_care"] >= min_age)
        & (merged["age_at_first_care"] <= max_age)
    ]

    # Create age bins
    merged["age_bin_start"] = (
        (merged["age_at_first_care"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance and age_bin
    prof = (
        merged.groupby(["distance_to_first_care", "age_bin_label"], observed=False)[
            ["diff_ft", "diff_pt"]
        ]
        .mean()
        .reset_index()
    )

    # Add age_bin_start back for sorting
    prof["age_bin_start"] = prof["age_bin_label"].str.split("-").str[0].astype(int)
    prof = prof.sort_values(["age_bin_start", "distance_to_first_care"])

    # Get unique age bins in order (sorted by bin start)
    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )

    # Define distinct colors for each age bin
    distinct_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]
    n_bins = len(unique_bins)
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(n_bins)]

    # Plot Part-Time
    plt.figure(figsize=(12, 7))
    for i, age_bin in enumerate(unique_bins):
        prof_bin = prof[prof["age_bin_label"] == age_bin].sort_values(
            "distance_to_first_care"
        )
        plt.plot(
            prof_bin["distance_to_first_care"],
            prof_bin["diff_pt"],
            label=f"Age {age_bin}",
            color=colors[i],
            linewidth=2,
        )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel(
        "Proportion Part-Time Working\nDeviation from Counterfactual", fontsize=16
    )
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        loc="best",
        prop={"size": 14},
        title="Age at first care",
        title_fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_pt, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Full-Time
    plt.figure(figsize=(12, 7))
    for i, age_bin in enumerate(unique_bins):
        prof_bin = prof[prof["age_bin_label"] == age_bin].sort_values(
            "distance_to_first_care"
        )
        plt.plot(
            prof_bin["distance_to_first_care"],
            prof_bin["diff_ft"],
            label=f"Age {age_bin}",
            color=colors[i],
            linewidth=2,
        )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to start of first care spell", fontsize=16)
    plt.ylabel(
        "Proportion Full-Time Working\nDeviation from Counterfactual", fontsize=16
    )
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        loc="best",
        prop={"size": 14},
        title="Age at first care",
        title_fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_ft, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_no_care_demand_1m
def task_plot_first_care_start_by_age(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_1m.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "first_care_start_by_age_1m.png",
    path_to_csv: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "first_care_start_by_age_1m.csv",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Count how many people start giving care for the first time at each age.

    Creates a plot showing the distribution of first care start ages and saves
    the counts to a CSV file.

    Args:
        path_to_original_data: Path to original simulated data
        path_to_plot: Path to save the plot
        path_to_csv: Path to save the CSV with counts by age
        min_age: Minimum age to include in analysis
        max_age: Maximum age to include in analysis
    """
    # Load data
    df_o = pd.read_pickle(path_to_original_data)

    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = _ensure_agent_period(df_o)

    # Fully flatten any residual index levels
    if isinstance(df_o.index, pd.MultiIndex):
        idx_names_o = {n for n in df_o.index.names if n is not None}
        if ("agent" in idx_names_o) or ("period" in idx_names_o):
            df_o = df_o.reset_index()

    # Ensure no index name collisions remain
    df_o = df_o.reset_index(drop=True)

    # Check for age column
    if "age" not in df_o.columns:
        raise ValueError("Age column required but not found in data")

    # Find first care period for each agent
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df_o["choice"].isin(care_codes)

    first_care = (
        df_o.loc[caregiving_mask, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(columns={"period": "first_care_period", "age": "age_at_first_care"})
    )

    # Filter to age range
    first_care = first_care[
        (first_care["age_at_first_care"] >= min_age)
        & (first_care["age_at_first_care"] <= max_age)
    ]

    # Count by age
    counts_by_age = (
        first_care["age_at_first_care"]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"age_at_first_care": "age"})
    )

    # Ensure all ages in range are represented (fill missing with 0)
    all_ages = pd.DataFrame({"age": range(min_age, max_age + 1)})
    counts_by_age = all_ages.merge(counts_by_age, on="age", how="left").fillna(0)
    counts_by_age["count"] = counts_by_age["count"].astype(int)

    # Save to CSV
    counts_by_age.to_csv(path_to_csv, index=False)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.bar(
        counts_by_age["age"],
        counts_by_age["count"],
        color=JET_COLOR_MAP[1],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Age at first care spell", fontsize=16)
    plt.ylabel("Number of people", fontsize=16)
    plt.title("Distribution of First Care Start by Age", fontsize=18)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


def _add_distance_to_first_care_demand(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care_demand column where 0 is first time care_demand > 0."""
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = _ensure_agent_period(df)
    # Find first period where care_demand > 0
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


@pytask.mark.counterfactual_differences_no_care_demand_1m
def task_plot_matched_differences_by_distance_by_care_demand(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_1m.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_1m.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_by_distance_by_care_demand_1m.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (orig - no-care-demand), then average by distance.

    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand from original, attach to merged.
      6) Average diffs by distance and plot three series.

    """

    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

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

    # Outcomes per period
    o_work = df_o["choice"].isin(np.asarray(WORK).ravel().tolist()).astype(float)
    o_ft = df_o["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    o_pt = df_o["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    # No care demand uses reduced choice structure (4 choices)
    c_work = (
        df_c["choice"]
        .isin(np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_ft = (
        df_c["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_pt = (
        df_c["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    o_cols = df_o[["agent", "period"]].copy()
    o_cols["work_o"] = o_work
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["work_c"] = c_work
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt

    # Merge on (agent, period) to get matched differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_o"] - merged["work_c"]
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]

    # Compute distance to first care demand in original and attach
    df_o_dist = _add_distance_to_first_care_demand(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care_demand", observed=False)[
            ["diff_work", "diff_ft", "diff_pt"]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care_demand")
    )

    # Plot
    plt.figure(figsize=(12, 7))

    plt.plot(
        prof["distance_to_first_care_demand"],
        prof["diff_work"],
        label="Working",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        prof["distance_to_first_care_demand"],
        prof["diff_ft"],
        label="Full Time",
        color=JET_COLOR_MAP[1],
        linewidth=2,
    )
    plt.plot(
        prof["distance_to_first_care_demand"],
        prof["diff_pt"],
        label="Part Time",
        color=JET_COLOR_MAP[0],
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to first care demand", fontsize=16)
    plt.ylabel("Proportion Working\nDeviation from Counterfactual", fontsize=16)
    plt.xlim(-window, window)
    plt.ylim(-0.125, 0.025)
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend with increased font size, positioned closer to (0,0) point
    plt.legend(
        ["Working", "Full Time", "Part Time"],
        loc="lower left",
        bbox_to_anchor=(0.05, 0.05),
        prop={"size": 16},
    )
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_no_care_demand_1m
def task_plot_matched_differences_by_age_at_first_care_demand(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_1m.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_1m.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_part_time_by_age_at_first_care_demand_1m.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_full_time_by_age_at_first_care_demand_1m.png",
    ever_caregivers: bool = True,
    window: int = 20,
    ages_at_first_care_demand: list[int] = [50, 54, 58, 62],
) -> None:
    """Compute matched period differences by age at first care demand.

    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).
    Creates separate plots for part-time and full-time work, with separate lines
    for each age at which care demand first appeared.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand and age_at_first_care_demand from original.
      6) Filter to specific ages at first care demand.
      7) Average diffs by distance and age_at_first_care_demand.
      8) Plot separate figures for PT and FT with one line per starting age.

    """
    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

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

    # Outcomes per period
    o_ft = df_o["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    o_pt = df_o["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    # No care demand uses reduced choice structure (4 choices)
    c_ft = (
        df_c["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_pt = (
        df_c["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    o_cols = df_o[["agent", "period"]].copy()
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt

    # Merge on (agent, period) to get matched differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]

    # Compute distance and age at first care demand from original
    df_o_dist = _add_distance_to_first_care_demand(df_o)

    # Get first care demand period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    if "age" not in df_o.columns:
        raise ValueError("Age column required but not found in original data")

    first_care_demand_with_age = (
        df_o[["agent", "period", "age"]]
        .merge(dist_map, on="agent", how="left")
        .query("period == first_care_demand_period")
        .groupby("agent", observed=False)["age"]
        .first()
        .reset_index()
        .rename(columns={"age": "age_at_first_care_demand"})
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    # Filter to specific ages at first care demand
    merged = merged[merged["age_at_first_care_demand"].isin(ages_at_first_care_demand)]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance and age_at_first_care_demand
    prof = (
        merged.groupby(
            ["distance_to_first_care_demand", "age_at_first_care_demand"],
            observed=False,
        )[["diff_ft", "diff_pt"]]
        .mean()
        .reset_index()
        .sort_values(["age_at_first_care_demand", "distance_to_first_care_demand"])
    )

    # Define distinct colors for each age
    distinct_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]
    n_ages = len(ages_at_first_care_demand)
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(n_ages)]

    # Plot Part-Time
    plt.figure(figsize=(12, 7))
    for i, age in enumerate(sorted(ages_at_first_care_demand)):
        prof_age = prof[prof["age_at_first_care_demand"] == age].sort_values(
            "distance_to_first_care_demand"
        )
        plt.plot(
            prof_age["distance_to_first_care_demand"],
            prof_age["diff_pt"],
            label=f"Age {age}",
            color=colors[i],
            linewidth=2,
        )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to first care demand", fontsize=16)
    plt.ylabel(
        "Proportion Part-Time Working\nDeviation from Counterfactual", fontsize=16
    )
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        loc="best",
        prop={"size": 14},
        title="Age at first care demand",
        title_fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_pt, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Full-Time
    plt.figure(figsize=(12, 7))
    for i, age in enumerate(sorted(ages_at_first_care_demand)):
        prof_age = prof[prof["age_at_first_care_demand"] == age].sort_values(
            "distance_to_first_care_demand"
        )
        plt.plot(
            prof_age["distance_to_first_care_demand"],
            prof_age["diff_ft"],
            label=f"Age {age}",
            color=colors[i],
            linewidth=2,
        )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to first care demand", fontsize=16)
    plt.ylabel(
        "Proportion Full-Time Working\nDeviation from Counterfactual", fontsize=16
    )
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        loc="best",
        prop={"size": 14},
        title="Age at first care demand",
        title_fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_ft, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_no_care_demand_1m
def task_plot_matched_differences_by_age_bins_at_first_care_demand(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_1m.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_1m.pkl",
    path_to_plot_pt: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_part_time_by_age_bins_at_first_care_demand_1m.png",
    path_to_plot_ft: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_full_time_by_age_bins_at_first_care_demand_1m.png",
    ever_caregivers: bool = True,
    window: int = 20,
    min_age: int = 50,
    max_age: int = 62,
    bin_width: int = 3,
) -> None:
    """Compute matched period differences by age bins at first care demand.

    Uses t=0 as first time care_demand > 0 (instead of first caregiving spell).
    Creates separate plots for part-time and full-time work, with separate lines
    for each age bin at which care demand first appeared (e.g., 50-53, 54-57, etc.).

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care_demand and age_at_first_care_demand from original.
      6) Group ages into bins.
      7) Average diffs by distance and age_bin_at_first_care_demand.
      8) Plot separate figures for PT and FT with one line per age bin.

    """
    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

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

    # Outcomes per period
    o_ft = df_o["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    o_pt = df_o["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    # No care demand uses reduced choice structure (4 choices)
    c_ft = (
        df_c["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_pt = (
        df_c["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    o_cols = df_o[["agent", "period"]].copy()
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt

    # Merge on (agent, period) to get matched differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]

    # Compute distance and age at first care demand from original
    df_o_dist = _add_distance_to_first_care_demand(df_o)

    # Get first care demand period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    if "age" not in df_o.columns:
        raise ValueError("Age column required but not found in original data")

    first_care_demand_with_age = (
        df_o[["agent", "period", "age"]]
        .merge(dist_map, on="agent", how="left")
        .query("period == first_care_demand_period")
        .groupby("agent", observed=False)["age"]
        .first()
        .reset_index()
        .rename(columns={"age": "age_at_first_care_demand"})
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    # Filter to age range
    merged = merged[
        (merged["age_at_first_care_demand"] >= min_age)
        & (merged["age_at_first_care_demand"] <= max_age)
    ]

    # Create age bins
    merged["age_bin_start"] = (
        (merged["age_at_first_care_demand"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance and age_bin
    prof = (
        merged.groupby(
            ["distance_to_first_care_demand", "age_bin_label"], observed=False
        )[["diff_ft", "diff_pt"]]
        .mean()
        .reset_index()
    )

    # Add age_bin_start back for sorting
    prof["age_bin_start"] = prof["age_bin_label"].str.split("-").str[0].astype(int)
    prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])

    # Get unique age bins in order (sorted by bin start)
    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )

    # Define distinct colors for each age bin
    distinct_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]
    n_bins = len(unique_bins)
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(n_bins)]

    # Plot Part-Time
    plt.figure(figsize=(12, 7))
    for i, age_bin in enumerate(unique_bins):
        prof_bin = prof[prof["age_bin_label"] == age_bin].sort_values(
            "distance_to_first_care_demand"
        )
        plt.plot(
            prof_bin["distance_to_first_care_demand"],
            prof_bin["diff_pt"],
            label=f"Age {age_bin}",
            color=colors[i],
            linewidth=2,
        )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to first care demand", fontsize=16)
    plt.ylabel(
        "Proportion Part-Time Working\nDeviation from Counterfactual", fontsize=16
    )
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        loc="best",
        prop={"size": 14},
        title="Age at first care demand",
        title_fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_pt, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Full-Time
    plt.figure(figsize=(12, 7))
    for i, age_bin in enumerate(unique_bins):
        prof_bin = prof[prof["age_bin_label"] == age_bin].sort_values(
            "distance_to_first_care_demand"
        )
        plt.plot(
            prof_bin["distance_to_first_care_demand"],
            prof_bin["diff_ft"],
            label=f"Age {age_bin}",
            color=colors[i],
            linewidth=2,
        )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Year relative to first care demand", fontsize=16)
    plt.ylabel(
        "Proportion Full-Time Working\nDeviation from Counterfactual", fontsize=16
    )
    plt.xlim(-window, window)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        loc="best",
        prop={"size": 14},
        title="Age at first care demand",
        title_fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(path_to_plot_ft, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_no_care_demand_1m
def task_plot_first_care_demand_start_by_age(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_1m.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "first_care_demand_start_by_age_1m.png",
    path_to_csv: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "first_care_demand_start_by_age_1m.csv",
    min_age: int = 40,
    max_age: int = 69,
) -> None:
    """Count how many people first experience care_demand > 0 at each age.

    Creates a plot showing the distribution of first care demand start ages and saves
    the counts to a CSV file.

    Args:
        path_to_original_data: Path to original simulated data
        path_to_plot: Path to save the plot
        path_to_csv: Path to save the CSV with counts by age
        min_age: Minimum age to include in analysis
        max_age: Maximum age to include in analysis
    """
    # Load data
    df_o = pd.read_pickle(path_to_original_data)

    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = _ensure_agent_period(df_o)

    # Fully flatten any residual index levels
    if isinstance(df_o.index, pd.MultiIndex):
        idx_names_o = {n for n in df_o.index.names if n is not None}
        if ("agent" in idx_names_o) or ("period" in idx_names_o):
            df_o = df_o.reset_index()

    # Ensure no index name collisions remain
    df_o = df_o.reset_index(drop=True)

    # Check for age column
    if "age" not in df_o.columns:
        raise ValueError("Age column required but not found in data")

    # Find first period where care_demand > 0 for each agent
    care_demand_mask = df_o["care_demand"] > 0

    first_care_demand = (
        df_o.loc[care_demand_mask, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(
            columns={
                "period": "first_care_demand_period",
                "age": "age_at_first_care_demand",
            }
        )
    )

    # Filter to age range
    first_care_demand = first_care_demand[
        (first_care_demand["age_at_first_care_demand"] >= min_age)
        & (first_care_demand["age_at_first_care_demand"] <= max_age)
    ]

    # Count by age
    counts_by_age = (
        first_care_demand["age_at_first_care_demand"]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"age_at_first_care_demand": "age"})
    )

    # Ensure all ages in range are represented (fill missing with 0)
    all_ages = pd.DataFrame({"age": range(min_age, max_age + 1)})
    counts_by_age = all_ages.merge(counts_by_age, on="age", how="left").fillna(0)
    counts_by_age["count"] = counts_by_age["count"].astype(int)

    # Save to CSV
    counts_by_age.to_csv(path_to_csv, index=False)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.bar(
        counts_by_age["age"],
        counts_by_age["count"],
        color=JET_COLOR_MAP[1],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Age at first care demand", fontsize=16)
    plt.ylabel("Number of people", fontsize=16)
    plt.title("Distribution of First Care Demand Start by Age", fontsize=18)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_no_care_demand_1m
def task_plot_matched_differences_forced_care_demand_at_50(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_1m.pkl",
    path_to_forced_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_forced_care_demand_at_50_1m.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "no_care_demand"
    / "matched_differences_forced_care_demand_at_50_1m.png",
    ever_caregivers: bool = True,
    window: int = 20,
    forced_age: int = 50,
) -> None:
    """Compute matched period differences (baseline - forced care demand at 50).

    Uses t=0 as age 50 (where everyone experiences care demand in counterfactual).
    Since everyone experiences care demand at age 50 in the counterfactual,
    age 50 is at the center of the x-axis with a vertical line.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance from age 50 (age - forced_age) for x-axis.
      6) Average diffs by distance and plot three series.

    """
    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_forced_care_demand_data)

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

    # Outcomes per period - both use same 8-choice structure
    o_work = df_o["choice"].isin(np.asarray(WORK).ravel().tolist()).astype(float)
    o_ft = df_o["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    o_pt = df_o["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    # Forced care demand uses same choice structure as baseline
    c_work = df_c["choice"].isin(np.asarray(WORK).ravel().tolist()).astype(float)
    c_ft = df_c["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    c_pt = df_c["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    o_cols = df_o[["agent", "period", "age"]].copy()
    o_cols["work_o"] = o_work
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["work_c"] = c_work
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt

    # Merge on (agent, period) to get matched differences
    # Only include age from original to avoid duplicate column
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_o"] - merged["work_c"]
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]

    # Compute distance from forced_age (age 50)
    # Age should be the same for same agent and period in both dataframes
    merged["distance_from_forced_age"] = merged["age"] - forced_age

    # Trim to window around forced_age
    merged = merged[
        (merged["distance_from_forced_age"] >= -window)
        & (merged["distance_from_forced_age"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_from_forced_age", observed=False)[
            ["diff_work", "diff_ft", "diff_pt"]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_from_forced_age")
    )

    # Plot
    plt.figure(figsize=(12, 7))

    plt.plot(
        prof["distance_from_forced_age"],
        prof["diff_work"],
        label="Working",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        prof["distance_from_forced_age"],
        prof["diff_ft"],
        label="Full Time",
        color=JET_COLOR_MAP[1],
        linewidth=2,
    )
    plt.plot(
        prof["distance_from_forced_age"],
        prof["diff_pt"],
        label="Part Time",
        color=JET_COLOR_MAP[0],
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel(f"Year relative to age {forced_age}", fontsize=16)
    plt.ylabel("Proportion Working\nDeviation from Counterfactual", fontsize=16)
    plt.xlim(-window, window)
    plt.ylim(-0.125, 0.025)
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend with increased font size, positioned closer to (0,0) point
    plt.legend(
        ["Working", "Full Time", "Part Time"],
        loc="lower left",
        bbox_to_anchor=(0.05, 0.05),
        prop={"size": 16},
    )
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()
