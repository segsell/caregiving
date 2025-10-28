"""Plot differences in labor supply by age between scenarios.

Original and no-care-demand scenario.

"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from linearmodels.panel import PanelOLS
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    PART_TIME,
    RETIREMENT,
    SEX,
    UNEMPLOYED,
    WORK,
)
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)


@pytask.mark.counterfactual_differences
def task_plot_labor_supply_differences(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "labor_supply_differences_by_age.png",
    ever_caregivers: bool = True,
) -> None:
    """Plot differences in labor supply by age between scenarios."""

    # Load data
    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure an 'agent' column exists (source data often indexed by
    # MultiIndex agent, period)
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Optional sample restriction to ever-caregivers in the original scenario
    if ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    # Compute labor supply shares by age for both scenarios
    original_shares = compute_labor_supply_shares_by_age(df_original, is_original=True)
    no_care_demand_shares = compute_labor_supply_shares_by_age(
        df_no_care_demand, is_original=False
    )

    # Compute differences
    differences = compute_labor_supply_differences(
        original_shares, no_care_demand_shares
    )
    # Create plot
    create_labor_supply_difference_plot(differences, path_to_plot)


def compute_labor_supply_shares_by_age(
    df: pd.DataFrame, is_original: bool = True
) -> pd.DataFrame:
    """Compute labor supply shares by age using value_counts approach.

    Mirrors the approach in the model-fit plotting: map raw choices into
    4 aggregate groups (retired, unemployed, part-time, full-time), then
    compute age-specific normalized counts.
    """

    # Ensure required columns
    df_local = df[["choice", "age"]].copy()

    if is_original:
        choice_groups = {
            0: np.asarray(RETIREMENT).ravel().tolist(),
            1: np.asarray(UNEMPLOYED).ravel().tolist(),
            2: np.asarray(PART_TIME).ravel().tolist(),
            3: np.asarray(FULL_TIME).ravel().tolist(),
        }
    else:
        choice_groups = {
            0: np.asarray(RETIREMENT_NO_CARE_DEMAND).ravel().tolist(),
            1: np.asarray(UNEMPLOYED_NO_CARE_DEMAND).ravel().tolist(),
            2: np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist(),
            3: np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist(),
        }

    # Build a flat map from raw codes to aggregate group id
    code_to_group = {}
    for group_id, codes in choice_groups.items():
        for code in codes:
            code_to_group[code] = group_id

    # Map raw choice to group id
    df_local["choice_group"] = (
        df_local["choice"].map(code_to_group).fillna(0).astype(int)
    )

    # Compute normalized shares per age
    shares_by_age = (
        df_local.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .rename(columns={0: "retired", 1: "unemployed", 2: "part_time", 3: "full_time"})
        .reset_index()
    )

    # Construct output with desired columns
    out = pd.DataFrame(
        {
            "age": shares_by_age["age"],
            "is_full_time": shares_by_age.get("full_time", 0),
            "is_part_time": shares_by_age.get("part_time", 0),
        }
    )
    out["is_working"] = out["is_full_time"] + out["is_part_time"]
    out["is_not_working"] = 1 - out["is_working"]

    return out


def compute_labor_supply_differences(
    original_shares: pd.DataFrame, no_care_demand_shares: pd.DataFrame
) -> pd.DataFrame:
    """Compute differences in labor supply shares between scenarios."""

    # Merge on age only (assuming single sex model)
    merged = pd.merge(
        original_shares,
        no_care_demand_shares,
        on=["age"],
        suffixes=("_original", "_no_care_demand"),
    )

    # Compute differences (no_care_demand - original)
    merged["working_diff"] = (
        merged["is_working_no_care_demand"] - merged["is_working_original"]
    )
    merged["part_time_diff"] = (
        merged["is_part_time_no_care_demand"] - merged["is_part_time_original"]
    )
    merged["full_time_diff"] = (
        merged["is_full_time_no_care_demand"] - merged["is_full_time_original"]
    )

    return merged


def create_labor_supply_difference_plot(
    differences: pd.DataFrame, path_to_plot: Path
) -> None:
    """Create plot showing labor supply differences by age."""

    # Create single plot with 3 lines
    plt.figure(figsize=(10, 6))

    # Plot Working (black, dashed)
    plt.plot(
        differences["age"],
        differences["working_diff"],
        color="black",
        linewidth=2,
        linestyle="--",
        label="Working",
    )

    # Plot Full-time (JET_COLOR_MAP[1])
    plt.plot(
        differences["age"],
        differences["full_time_diff"],
        color=JET_COLOR_MAP[1],
        linewidth=2,
        label="Full Time",
    )

    # Plot Part-time (JET_COLOR_MAP[2])
    plt.plot(
        differences["age"],
        differences["part_time_diff"],
        color=JET_COLOR_MAP[0],
        linewidth=2,
        label="Part Time",
    )

    # Add horizontal line at zero
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # Set labels and formatting with increased font sizes
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Proportion Working\nDeviation from Baseline", fontsize=16)
    plt.xlim(30, 70)
    plt.ylim(None, 0.05)  # Set upper y-limit to 0.05
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend in specified order with increased font size
    plt.legend(["Working", "Full Time", "Part Time"], prop={"size": 16})

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Labor supply difference plot saved to: {path_to_plot}")


def compute_wealth_savings_by_age(df: pd.DataFrame, outcome_var: str) -> pd.DataFrame:
    """Compute mean wealth or savings by age."""
    df_local = df[["age", outcome_var]].copy()

    # Compute mean by age
    means_by_age = (
        df_local.groupby("age", observed=False)[outcome_var].mean().reset_index()
    )

    return means_by_age


def compute_wealth_savings_differences(
    original_data: pd.DataFrame, no_care_demand_data: pd.DataFrame, outcome_var: str
) -> pd.DataFrame:
    """Compute differences in wealth or savings between scenarios."""

    # Compute means by age for both scenarios
    original_means = compute_wealth_savings_by_age(original_data, outcome_var)
    no_care_means = compute_wealth_savings_by_age(no_care_demand_data, outcome_var)

    # Merge on age
    merged = pd.merge(
        original_means,
        no_care_means,
        on=["age"],
        suffixes=("_original", "_no_care_demand"),
    )

    # Compute differences (original - no_care_demand)
    merged[f"{outcome_var}_diff"] = (
        merged[f"{outcome_var}_original"] - merged[f"{outcome_var}_no_care_demand"]
    )

    return merged


def create_wealth_difference_plot(
    differences: pd.DataFrame, path_to_plot: Path
) -> None:
    """Create plot showing wealth differences by age."""

    plt.figure(figsize=(10, 6))

    # Plot wealth difference
    plt.plot(
        differences["age"],
        differences["wealth_at_beginning_diff"],
        color="black",
        linewidth=2,
        label="Wealth Difference",
    )

    # Add horizontal line at zero
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # Set labels and formatting with increased font sizes
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Wealth Difference\n(Original - No Care Demand)", fontsize=16)
    plt.xlim(30, 70)
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend with increased font size
    plt.legend(prop={"size": 16})

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Wealth difference plot saved to: {path_to_plot}")


def create_savings_difference_plot(
    differences: pd.DataFrame, path_to_plot: Path
) -> None:
    """Create plot showing savings differences by age."""

    plt.figure(figsize=(10, 6))

    # Plot savings difference
    plt.plot(
        differences["age"],
        differences["savings_dec_diff"],
        color="black",
        linewidth=2,
        label="Savings Difference",
    )

    # Add horizontal line at zero
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # Set labels and formatting with increased font sizes
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Savings Difference\n(Original - No Care Demand)", fontsize=16)
    plt.xlim(30, 70)
    plt.grid(True, alpha=0.3)

    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend with increased font size
    plt.legend(prop={"size": 16})

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Savings difference plot saved to: {path_to_plot}")


def create_labor_supply_age_profile_plot(
    df_original: pd.DataFrame,
    df_no_care_demand: pd.DataFrame,
    path_to_plot: Path,
) -> None:
    """Plot FT, PT and Not Working age profiles for both scenarios in one plot."""

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    shares_original = compute_labor_supply_shares_by_age(df_original, is_original=True)
    shares_ncd = compute_labor_supply_shares_by_age(
        df_no_care_demand, is_original=False
    )

    merged = pd.merge(
        shares_original,
        shares_ncd,
        on=["age"],
        suffixes=("_original", "_no_care_demand"),
    )

    plt.figure(figsize=(12, 7))
    plt.title("Labor Supply Age Profiles: Original vs No Care Demand")

    # Full-time
    plt.plot(
        merged["age"],
        merged["is_full_time_original"],
        label="Full-time (Original)",
        color="tab:red",
        linewidth=2,
    )
    plt.plot(
        merged["age"],
        merged["is_full_time_no_care_demand"],
        label="Full-time (No Care)",
        color="tab:red",
        linestyle="--",
        linewidth=2,
    )

    # Part-time
    plt.plot(
        merged["age"],
        merged["is_part_time_original"],
        label="Part-time (Original)",
        color="tab:green",
        linewidth=2,
    )
    plt.plot(
        merged["age"],
        merged["is_part_time_no_care_demand"],
        label="Part-time (No Care)",
        color="tab:green",
        linestyle="--",
        linewidth=2,
    )

    # Not working (unemployed or retired)
    plt.plot(
        merged["age"],
        merged["is_not_working_original"],
        label="Not working (Original)",
        color="tab:blue",
        linewidth=2,
    )
    plt.plot(
        merged["age"],
        merged["is_not_working_no_care_demand"],
        label="Not working (No Care)",
        color="tab:blue",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Age")
    plt.ylabel("Share")
    plt.xlim(30, 70)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences
def task_plot_labor_supply_age_profiles(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "labor_supply_age_profiles.png",
    ever_caregivers: bool = True,
) -> None:
    """Task: plot FT, PT, Not Working age profiles for both scenarios."""

    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Ensure 'agent' exists
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Optional sample restriction to ever-caregivers in the original scenario
    if ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    create_labor_supply_age_profile_plot(
        df_original=df_original,
        df_no_care_demand=df_no_care_demand,
        path_to_plot=path_to_plot,
    )


# ===================================================================================
# Distance to first care spell profiles
# ===================================================================================


def _ensure_agent_period(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that 'agent' and 'period' are columns (not index levels)."""
    if "agent" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("agent" in df.index.names):
            df = df.reset_index(level=["agent"])  # keep period indexed if present
        else:
            df = df.reset_index()
    if "period" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("period" in df.index.names):
            df = df.reset_index(level=["period"])  # keep any other levels
        else:
            if "age" in df.columns:
                df["period"] = df.groupby("agent")["age"].transform(
                    lambda s: s - s.min()
                )
            else:
                df["period"] = df.groupby("agent").cumcount()
    return df


def _add_distance_to_first_care(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care column to original data where 0 is first care."""
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = _ensure_agent_period(df)
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


def create_outcome_profiles_by_distance_plot(
    df_original_with_dist: pd.DataFrame,
    df_no_care_demand_with_dist: pd.DataFrame,
    path_to_plot: Path,
    x_window: int = 16,
) -> None:
    """Plot mean work, ft, pt by distance_to_first_care for both scenarios (6 lines)."""

    # Build indicators
    work_o = (
        df_original_with_dist["choice"]
        .isin(np.asarray(WORK).ravel().tolist())
        .astype(float)
    )
    ft_o = (
        df_original_with_dist["choice"]
        .isin(np.asarray(FULL_TIME).ravel().tolist())
        .astype(float)
    )
    pt_o = (
        df_original_with_dist["choice"]
        .isin(np.asarray(PART_TIME).ravel().tolist())
        .astype(float)
    )

    work_c = (
        df_no_care_demand_with_dist["choice"]
        .isin(np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    ft_c = (
        df_no_care_demand_with_dist["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    pt_c = (
        df_no_care_demand_with_dist["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    df_o = pd.DataFrame(
        {
            "distance": df_original_with_dist["distance_to_first_care"],
            "work": work_o,
            "ft": ft_o,
            "pt": pt_o,
        }
    )
    df_c = pd.DataFrame(
        {
            "distance": df_no_care_demand_with_dist["distance_to_first_care"],
            "work": work_c,
            "ft": ft_c,
            "pt": pt_c,
        }
    )

    # Restrict window if requested
    df_o = df_o[(df_o["distance"] >= -x_window) & (df_o["distance"] <= x_window)]
    df_c = df_c[(df_c["distance"] >= -x_window) & (df_c["distance"] <= x_window)]

    prof_o = df_o.groupby("distance", observed=False).mean().reset_index()
    prof_c = df_c.groupby("distance", observed=False).mean().reset_index()

    plt.figure(figsize=(12, 7))
    plt.title("Outcomes by Distance to First Care Spell (Original vs No Care)")

    # six lines
    plt.plot(
        prof_o["distance"],
        prof_o["work"],
        label="Work (Original)",
        color="tab:blue",
        linewidth=2,
    )
    plt.plot(
        prof_c["distance"],
        prof_c["work"],
        label="Work (No Care)",
        color="tab:blue",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        prof_o["distance"],
        prof_o["ft"],
        label="FT (Original)",
        color="tab:red",
        linewidth=2,
    )
    plt.plot(
        prof_c["distance"],
        prof_c["ft"],
        label="FT (No Care)",
        color="tab:red",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        prof_o["distance"],
        prof_o["pt"],
        label="PT (Original)",
        color="tab:green",
        linewidth=2,
    )
    plt.plot(
        prof_c["distance"],
        prof_c["pt"],
        label="PT (No Care)",
        color="tab:green",
        linestyle="--",
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Distance to first care spell (periods)")
    plt.ylabel("Mean outcome")
    plt.xlim(-x_window, x_window)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences
def task_plot_outcomes_by_distance_to_first_care(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "outcomes_by_distance_to_first_care.png",
    ever_caregivers: bool = True,
    window: int = 16,
) -> None:
    """Create distance_to_first_care and plot mean outcomes by distance (6 lines)."""

    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Restrict to alive periods
    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure agent/period exist
    df_original = _ensure_agent_period(df_original)
    df_no_care_demand = _ensure_agent_period(df_no_care_demand)

    # Optional restriction to ever-caregivers
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(care_codes), "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    # Compute distance in original and copy to counterfactual
    df_original = _add_distance_to_first_care(df_original)
    # Merge the per-agent first_care_period and compute distance in counterfactual
    dist_map = (
        df_original.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    df_no_care_demand = df_no_care_demand.merge(dist_map, on="agent", how="left")
    df_no_care_demand["distance_to_first_care"] = (
        df_no_care_demand["period"] - df_no_care_demand["first_care_period"]
    )

    # Plot
    create_outcome_profiles_by_distance_plot(
        df_original_with_dist=df_original,
        df_no_care_demand_with_dist=df_no_care_demand,
        path_to_plot=path_to_plot,
        x_window=window,
    )


@pytask.mark.counterfactual_differences
def task_plot_matched_differences_by_distance(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_by_distance.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (orig - no-care), then average by distance.

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
    plt.ylabel("Proportion Working\nDeviation from Baseline", fontsize=16)
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


@pytask.mark.counterfactual_differences
def task_plot_wealth_differences(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "wealth_differences_by_age.png",
    ever_caregivers: bool = True,
) -> None:
    """Plot differences in wealth by age between scenarios."""

    # Load data
    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure an 'agent' column exists
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Optional sample restriction to ever-caregivers in the original scenario
    if ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    # Compute wealth differences
    differences = compute_wealth_savings_differences(
        df_original, df_no_care_demand, "wealth_at_beginning"
    )

    # Create plot
    create_wealth_difference_plot(differences, path_to_plot)


@pytask.mark.counterfactual_differences
def task_plot_savings_differences(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "savings_differences_by_age.png",
    ever_caregivers: bool = True,
) -> None:
    """Plot differences in savings by age between scenarios."""

    # Load data
    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure an 'agent' column exists
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Optional sample restriction to ever-caregivers in the original scenario
    if ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    # Compute savings differences
    differences = compute_wealth_savings_differences(
        df_original, df_no_care_demand, "savings_dec"
    )

    # Create plot
    create_savings_difference_plot(differences, path_to_plot)
