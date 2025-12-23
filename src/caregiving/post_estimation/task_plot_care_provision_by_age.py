"""Plot care provision methods by age from estimated model simulation."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    CARE_DEMAND_AND_OTHER_SUPPLY,
    DEAD,
    INFORMAL_CARE,
    NO_CARE,
)

# Age range constants for analysis
AGE_40 = 40
AGE_50 = 50
AGE_60 = 60


@pytask.mark.post_estimation
def task_plot_care_provision_by_age(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "post_estimation"
    / "care_provision_by_age.png",
) -> None:
    """Plot care provision methods by age.

    Creates a plot showing:
    1. Overall care demand share (care_demand > 0) as the outer boundary
    2. Within that, the breakdown of how care is provided:
       - Informal caregiving (agent provides care)
       - Formal care (external care when agent doesn't provide)
       - Other family care (other family member provides care)

    Conditions on agents being alive (health != DEAD).
    """
    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Filter to alive agents only
    df_alive = df[df["health"] != DEAD].copy()

    # Calculate care provision by age
    provision_df = calculate_care_provision_by_age(df_alive)

    # Create the plots
    create_care_provision_plots(provision_df, path_to_save_plot)

    # Print summary statistics
    print_care_provision_summary(provision_df, path_to_save_plot)


def calculate_care_provision_by_age(df_alive: pd.DataFrame) -> pd.DataFrame:
    """Calculate care provision statistics by age."""
    care_provision_data = []

    for age in sorted(df_alive["age"].unique()):
        age_data = df_alive[df_alive["age"] == age]
        n_agents = len(age_data)

        if n_agents > 0:
            # Overall care demand
            has_care_demand = age_data["care_demand"] > 0
            share_care_demand = has_care_demand.mean()

            # Among those with care demand, determine provision method
            care_demand_subset = age_data[has_care_demand]

            if len(care_demand_subset) > 0:
                shares = _calculate_provision_shares(
                    care_demand_subset, share_care_demand
                )
            else:
                shares = _get_zero_shares()

            care_provision_data.append(
                {
                    "age": age,
                    "share_care_demand": share_care_demand,
                    "n_agents": n_agents,
                    "n_with_care_demand": len(care_demand_subset),
                }
                | shares
            )

    return pd.DataFrame(care_provision_data)


def _calculate_provision_shares(
    care_demand_subset: pd.DataFrame, share_care_demand: float
) -> dict:
    """Calculate care provision shares for agents with care demand."""
    # Informal care: agent chooses to provide care (odd choice numbers)
    provides_informal_care = care_demand_subset["choice"].isin(
        np.asarray(INFORMAL_CARE).tolist()
    )

    # Formal care: care demand exists, agent doesn't provide care,
    # and care_demand == 2 (no other family supply)
    needs_formal_care = ~provides_informal_care & (
        care_demand_subset["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY
    )

    # Other family care: care demand exists, agent doesn't provide care,
    # and care_demand == 1 (other family members provide care)
    other_family_care = ~provides_informal_care & (
        care_demand_subset["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY
    )

    # Calculate shares within care demand population
    share_informal_within_demand = provides_informal_care.mean()
    share_formal_within_demand = needs_formal_care.mean()
    share_other_family_within_demand = other_family_care.mean()

    # Calculate absolute shares (relative to total population)
    return {
        "share_informal_absolute": share_care_demand * share_informal_within_demand,
        "share_formal_absolute": share_care_demand * share_formal_within_demand,
        "share_other_family_absolute": share_care_demand
        * share_other_family_within_demand,
        "share_informal_within_demand": share_informal_within_demand,
        "share_formal_within_demand": share_formal_within_demand,
        "share_other_family_within_demand": share_other_family_within_demand,
    }


def _get_zero_shares() -> dict:
    """Return zero shares when no agents have care demand."""
    return {
        "share_informal_absolute": 0,
        "share_formal_absolute": 0,
        "share_other_family_absolute": 0,
        "share_informal_within_demand": 0,
        "share_formal_within_demand": 0,
        "share_other_family_within_demand": 0,
    }


def create_care_provision_plots(
    provision_df: pd.DataFrame, path_to_save_plot: Path
) -> None:
    """Create the care provision plots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot 1: Absolute shares (stacked area plot)
    _create_absolute_shares_plot(ax1, provision_df)

    # Plot 2: Conditional shares (among those with care demand)
    _create_conditional_shares_plot(ax2, provision_df)

    plt.tight_layout()

    # Save the plot
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_plot, dpi=300, bbox_inches="tight")
    plt.close()


def _create_absolute_shares_plot(ax, provision_df: pd.DataFrame) -> None:
    """Create the absolute shares stacked area plot."""
    ax.fill_between(
        provision_df["age"],
        0,
        provision_df["share_care_demand"],
        alpha=0.3,
        color="lightgray",
        label="Total care demand",
    )

    # Stack the care provision methods
    ax.fill_between(
        provision_df["age"],
        0,
        provision_df["share_informal_absolute"],
        alpha=0.8,
        color="#2ca02c",
        label="Informal caregiving",
    )

    ax.fill_between(
        provision_df["age"],
        provision_df["share_informal_absolute"],
        provision_df["share_informal_absolute"] + provision_df["share_formal_absolute"],
        alpha=0.8,
        color="#ff7f0e",
        label="Formal care",
    )

    ax.fill_between(
        provision_df["age"],
        provision_df["share_informal_absolute"] + provision_df["share_formal_absolute"],
        provision_df["share_care_demand"],
        alpha=0.8,
        color="#d62728",
        label="Other family care",
    )

    # Customize plot
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Share of All Agents", fontsize=12)
    ax.set_title(
        "Care Demand and Provision Methods by Age (Absolute Shares)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(provision_df["age"].min() - 1, 75)
    ax.set_ylim(0, 0.20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))


def _create_conditional_shares_plot(ax, provision_df: pd.DataFrame) -> None:
    """Create the conditional shares line plot."""
    ax.plot(
        provision_df["age"],
        provision_df["share_informal_within_demand"],
        label="Informal caregiving",
        linewidth=2.5,
        color="#2ca02c",
        marker="o",
        markersize=4,
    )

    ax.plot(
        provision_df["age"],
        provision_df["share_formal_within_demand"],
        label="Formal care",
        linewidth=2.5,
        color="#ff7f0e",
        marker="s",
        markersize=4,
    )

    ax.plot(
        provision_df["age"],
        provision_df["share_other_family_within_demand"],
        label="Other family care",
        linewidth=2.5,
        color="#d62728",
        marker="^",
        markersize=4,
    )

    # Customize plot
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Share of Agents with Care Demand", fontsize=12)
    ax.set_title(
        "Care Provision Methods by Age (Conditional on Having Care Demand)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(provision_df["age"].min() - 1, 75)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))


def print_care_provision_summary(
    provision_df: pd.DataFrame, path_to_save_plot: Path
) -> None:
    """Print summary statistics for care provision."""
    print("Care Provision by Age - Summary Statistics")
    print("=" * 50)
    print(f"Age range: {provision_df['age'].min()} - {provision_df['age'].max()}")
    print(f"Total observations: {provision_df['n_agents'].sum():,}")

    # Overall averages
    avg_care_demand = provision_df["share_care_demand"].mean()
    avg_informal_abs = provision_df["share_informal_absolute"].mean()
    avg_formal_abs = provision_df["share_formal_absolute"].mean()
    avg_other_family_abs = provision_df["share_other_family_absolute"].mean()

    print(f"Average share with care demand: {avg_care_demand:.1%}")
    print(f"Average share providing informal care: {avg_informal_abs:.1%}")
    print(f"Average share using formal care: {avg_formal_abs:.1%}")
    print(f"Average share with other family care: {avg_other_family_abs:.1%}")

    # Conditional averages (among those with care demand)
    weights = provision_df["n_with_care_demand"]
    total_with_demand = weights.sum()

    if total_with_demand > 0:
        weighted_informal = (
            provision_df["share_informal_within_demand"] * weights
        ).sum() / total_with_demand
        weighted_formal = (
            provision_df["share_formal_within_demand"] * weights
        ).sum() / total_with_demand
        weighted_other_family = (
            provision_df["share_other_family_within_demand"] * weights
        ).sum() / total_with_demand

        print("\nAmong those with care demand:")
        print(f"Share providing informal care: {weighted_informal:.1%}")
        print(f"Share using formal care: {weighted_formal:.1%}")
        print(f"Share with other family care: {weighted_other_family:.1%}")

    # Save summary data
    summary_path = path_to_save_plot.parent / "care_provision_by_age_data.csv"
    provision_df.to_csv(summary_path, index=False)
    print(f"\nSummary data saved to: {summary_path}")


@pytask.mark.post_estimation
def task_plot_informal_care_by_age(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "post_estimation"
    / "informal_care_by_age.png",
) -> None:
    """Plot share of agents providing informal care by age.

    Creates a focused plot showing only the share of agents who provide
    informal caregiving, conditioned on agents being alive (health != DEAD).
    """
    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Filter to alive agents only
    df_alive = df[df["health"] != DEAD].copy()

    # Calculate informal care provision by age
    informal_care_data = []

    for age in sorted(df_alive["age"].unique()):
        age_data = df_alive[df_alive["age"] == age]
        n_agents = len(age_data)

        if n_agents > 0:
            # Calculate share providing informal care
            provides_informal_care = age_data["choice"].isin(
                np.asarray(INFORMAL_CARE).tolist()
            )
            share_informal_care = provides_informal_care.mean()

            informal_care_data.append(
                {
                    "age": age,
                    "share_informal_care": share_informal_care,
                    "n_agents": n_agents,
                }
            )

    informal_care_df = pd.DataFrame(informal_care_data)

    # Create the plot
    create_informal_care_plot(informal_care_df, path_to_save_plot)

    # Print summary statistics
    print_informal_care_summary(informal_care_df, path_to_save_plot)


def create_informal_care_plot(
    informal_care_df: pd.DataFrame, path_to_save_plot: Path
) -> None:
    """Create the informal care by age plot."""
    plt.figure(figsize=(10, 6))

    # Create line plot with markers
    plt.plot(
        informal_care_df["age"],
        informal_care_df["share_informal_care"],
        linewidth=3,
        color="#2ca02c",
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#2ca02c",
        markeredgewidth=2,
        label="Share providing informal care",
    )

    # Customize plot
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Share of All Agents", fontsize=12)
    plt.title(
        "Share of Agents Providing Informal Care by Age",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.xlim(informal_care_df["age"].min() - 1, 75)

    # Set y-axis limits with a minimum range to avoid matplotlib warnings
    max_share = informal_care_df["share_informal_care"].max()
    y_max = max(max_share * 1.1, 0.01)  # Ensure at least 1% range
    plt.ylim(0, y_max)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    # Add legend
    plt.legend(fontsize=11, loc="upper right")

    # Save the plot
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_plot, dpi=300, bbox_inches="tight")
    plt.close()


def print_informal_care_summary(
    informal_care_df: pd.DataFrame, path_to_save_plot: Path
) -> None:
    """Print summary statistics for informal care provision."""
    print("\nInformal Care by Age - Summary Statistics")
    print("=" * 50)
    print(
        f"Age range: {informal_care_df['age'].min()} - {informal_care_df['age'].max()}"
    )
    print(f"Total observations: {informal_care_df['n_agents'].sum():,}")

    # Overall statistics
    avg_informal_care = informal_care_df["share_informal_care"].mean()
    max_informal_care = informal_care_df["share_informal_care"].max()
    peak_age = informal_care_df.loc[
        informal_care_df["share_informal_care"].idxmax(), "age"
    ]

    print(f"Average share providing informal care: {avg_informal_care:.1%}")
    print(
        f"Peak share providing informal care: {max_informal_care:.1%} at age {peak_age}"
    )

    # Age-specific insights
    ages_40_50 = informal_care_df[
        (informal_care_df["age"] >= AGE_40) & (informal_care_df["age"] <= AGE_50)
    ]
    ages_50_60 = informal_care_df[
        (informal_care_df["age"] >= AGE_50) & (informal_care_df["age"] <= AGE_60)
    ]

    if len(ages_40_50) > 0:
        avg_40_50 = ages_40_50["share_informal_care"].mean()
        print(f"Average share (ages 40-50): {avg_40_50:.1%}")

    if len(ages_50_60) > 0:
        avg_50_60 = ages_50_60["share_informal_care"].mean()
        print(f"Average share (ages 50-60): {avg_50_60:.1%}")

    # Save summary data
    summary_path = path_to_save_plot.parent / "informal_care_by_age_data.csv"
    informal_care_df.to_csv(summary_path, index=False)
    print(f"\nInformal care data saved to: {summary_path}")


@pytask.mark.post_estimation
def task_plot_care_mix_by_age_bins(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "post_estimation"
    / "care_mix_by_age_bins.png",
) -> None:
    """Plot care mix by age bins using stacked bar charts.

    Creates stacked bar charts showing the distribution of care provision methods
    (informal care, formal care, other family care) across age bins for agents
    with care demand, in the style of model fit plots.
    """
    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Filter to alive agents only
    df_alive = df[df["health"] != DEAD].copy()

    # Create care mix by age bins
    create_care_mix_stacked_bars(df_alive, path_to_save_plot)

    print(f"Care mix by age bins plot saved to: {path_to_save_plot}")


def create_care_mix_stacked_bars(  # noqa: PLR0915
    df_alive: pd.DataFrame, path_to_save_plot: Path, bin_width: int = 5
) -> None:
    """Create stacked bar chart of care mix by age bins."""
    # Define age bins
    age_min = AGE_40
    age_max = 75
    edges = list(range(age_min, age_max + 1, bin_width))
    if edges[-1] <= age_max:
        edges.append(age_max + 1)
    bin_starts = edges[:-1]

    # Calculate care mix for each age bin
    care_mix_data = []

    for start, end in zip(edges[:-1], edges[1:], strict=False):
        # Get ALL agents in this age bin (alive)
        all_agents_in_bin = df_alive[
            (df_alive["age"] >= start) & (df_alive["age"] < end)
        ]

        # Get agents with care demand in this age bin
        agents_with_care_demand = all_agents_in_bin[
            all_agents_in_bin["care_demand"] > 0
        ]

        if len(all_agents_in_bin) > 0:
            if len(agents_with_care_demand) > 0:
                # Calculate care provision among those with care demand
                provides_informal = agents_with_care_demand["choice"].isin(
                    np.asarray(INFORMAL_CARE).tolist()
                )

                # Formal care: care demand exists, agent doesn't provide care,
                # and care_demand == 2 (no other family supply)
                needs_formal = ~provides_informal & (
                    agents_with_care_demand["care_demand"]
                    == CARE_DEMAND_AND_NO_OTHER_SUPPLY
                )

                # Other family care: care demand exists, agent doesn't provide care,
                # and care_demand == 1 (other family members provide care)
                other_family = ~provides_informal & (
                    agents_with_care_demand["care_demand"]
                    == CARE_DEMAND_AND_OTHER_SUPPLY
                )

                # Calculate shares as proportion of ALL alive agents in bin
                total_agents = len(all_agents_in_bin)
                informal_share = provides_informal.sum() / total_agents
                formal_share = needs_formal.sum() / total_agents
                other_family_share = other_family.sum() / total_agents
            else:
                # No care demand in this bin
                informal_share = 0.0
                formal_share = 0.0
                other_family_share = 0.0

            care_mix_data.append(
                {
                    "age_bin": start,
                    "informal_care": informal_share,
                    "formal_care": formal_share,
                    "other_family_care": other_family_share,
                    "n_agents_total": len(all_agents_in_bin),
                    "n_agents_with_care_demand": len(agents_with_care_demand),
                }
            )
        else:
            # No agents in this bin
            care_mix_data.append(
                {
                    "age_bin": start,
                    "informal_care": 0.0,
                    "formal_care": 0.0,
                    "other_family_care": 0.0,
                    "n_agents_total": 0,
                    "n_agents_with_care_demand": 0,
                }
            )

    care_mix_df = pd.DataFrame(care_mix_data)

    # Follow the exact plotting style from plot_caregiver_shares_by_age_bins
    fig, ax = plt.subplots(figsize=(8, 4))  # Match reference figure size

    # Filter out the 75+ bin to match reference style
    keep = [bs < 75 for bs in bin_starts]  # noqa: PLR2004
    bin_starts_filtered = np.asarray(bin_starts)[keep]
    care_mix_df_filtered = care_mix_df[
        care_mix_df["age_bin"] < 75  # noqa: PLR2004
    ].copy()

    # Bar geometry matching the reference exactly
    bar_w = 1.5  # width of each bar (match reference)
    gap = 0.10  # empty space between bars (match reference)
    offset = bar_w / 2 + gap / 2

    # Define colors using mcolors.to_rgba like the reference
    colors = {
        "informal_care": mcolors.to_rgba("green", 1),
        "other_family_care": mcolors.to_rgba("red", 1),
        "formal_care": mcolors.to_rgba("orange", 1),
    }

    # Create stacked bars (reordered: informal -> other family -> formal)
    informal_bottom = np.zeros(len(care_mix_df_filtered))
    other_family_bottom = care_mix_df_filtered["informal_care"].values
    formal_bottom = (
        care_mix_df_filtered["informal_care"]
        + care_mix_df_filtered["other_family_care"]
    ).values

    # Plot stacked bars (bottom to top: informal -> other family -> formal)
    ax.bar(
        care_mix_df_filtered["age_bin"],
        care_mix_df_filtered["informal_care"],
        width=bar_w,
        bottom=informal_bottom,
        color=colors["informal_care"],
        label="Informal caregiving",
    )

    ax.bar(
        care_mix_df_filtered["age_bin"],
        care_mix_df_filtered["other_family_care"],
        width=bar_w,
        bottom=other_family_bottom,
        color=colors["other_family_care"],
        label="Other family care",
    )

    ax.bar(
        care_mix_df_filtered["age_bin"],
        care_mix_df_filtered["formal_care"],
        width=bar_w,
        bottom=formal_bottom,
        color=colors["formal_care"],
        label="Formal care",
    )

    # X-axis ticks and labels exactly like the reference
    xticks = bin_starts_filtered
    xticklabels = [
        f"{start}\u2013{start + bin_width - 1}" for start in bin_starts_filtered
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Cosmetics & extra whitespace exactly like reference
    pad = 2  # one age-unit padding on each side (match reference)
    ax.set_xlabel("Age Bin")  # Exact match to reference
    ax.set_ylabel("Share")  # Exact match to reference
    ax.set_xlim(
        bin_starts_filtered[0] - offset - pad, bin_starts_filtered[-1] + offset + pad
    )
    ax.set_ylim(0, 0.1)  # Match reference y-limit
    ax.legend()

    # Add sample size annotations (total alive agents in each age bin)
    for _i, row in care_mix_df_filtered.iterrows():
        if row["n_agents_total"] > 0:
            total_care_height = (
                row["informal_care"] + row["other_family_care"] + row["formal_care"]
            )
            ax.text(
                row["age_bin"],
                total_care_height + 0.005,  # Small offset above the bar
                f"n={row['n_agents_total']:,}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray",
            )

    # Use tight_layout and save exactly like reference
    plt.tight_layout()
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_save_plot:
        plt.savefig(
            path_to_save_plot, dpi=300, transparent=False
        )  # Match reference save settings

    # Print summary statistics
    print("\nCare Mix by Age Bins - Summary")
    print("=" * 40)
    print(f"Age range: {age_min} - {age_max} (bin width: {bin_width} years)")
    print("\nSample Size Interpretation:")
    print("- The 'n=' values shown above each bar represent the TOTAL number")
    print("  of alive agents in that age bin (the denominator for all percentages)")
    print("- Bar heights show what % of these total alive agents have care needs")
    print("- Stacking shows how care needs are met (bottom to top):")
    print("  • Green (bottom): Informal caregiving by the agent themselves")
    print("  • Red (middle): Other family member provides care")
    print("  • Orange (top): Formal care services")

    total_agents = care_mix_df["n_agents_total"].sum()
    total_with_care_demand = care_mix_df["n_agents_with_care_demand"].sum()
    print(f"\nTotal alive agents across all age bins: {total_agents:,}")
    print(f"Total agents with care demand: {total_with_care_demand:,}")

    if total_with_care_demand > 0:
        care_demand_rate = total_with_care_demand / total_agents
        print(f"Overall care demand rate: {care_demand_rate:.1%}")

    if total_agents > 0:
        # Weighted averages (relative to total alive population)
        weights = care_mix_df["n_agents_total"]
        avg_informal = (care_mix_df["informal_care"] * weights).sum() / total_agents
        avg_formal = (care_mix_df["formal_care"] * weights).sum() / total_agents
        avg_other_family = (
            care_mix_df["other_family_care"] * weights
        ).sum() / total_agents

        print("\nOverall care mix (as % of total alive population):")
        print(f"  Informal caregiving: {avg_informal:.1%}")
        print(f"  Other family care: {avg_other_family:.1%}")
        print(f"  Formal care: {avg_formal:.1%}")
        total_care_provision = avg_informal + avg_formal + avg_other_family
        print(f"  Total care provision: {total_care_provision:.1%}")

    # Save data
    data_path = path_to_save_plot.parent / "care_mix_by_age_bins_data.csv"
    care_mix_df.to_csv(data_path, index=False)
    print(f"\nCare mix data saved to: {data_path}")
