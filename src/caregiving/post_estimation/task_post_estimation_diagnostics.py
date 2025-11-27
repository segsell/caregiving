"""Comprehensive post-estimation diagnostics and summary statistics."""

import pickle
from pathlib import Path
from typing import Annotated

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
)


@pytask.mark.post_estimation
def task_post_estimation_diagnostics(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "post_estimation"
    / "summary_statistics.csv",
    path_to_save_plots_dir: Annotated[Path, Product] = BLD
    / "post_estimation"
    / "plots",
) -> None:
    """Generate comprehensive post-estimation diagnostics.

    Creates summary statistics and diagnostic plots for:
    - Care demand patterns by age and demographics
    - Labor supply patterns
    - Wealth accumulation
    - Health transitions
    - Choice distributions
    """

    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Create output directories
    path_to_save_plots_dir.mkdir(parents=True, exist_ok=True)

    # Filter to alive agents
    df_alive = df[df["health"] != DEAD].copy()

    print(f"Loaded simulation data with {len(df):,} total observations")
    print(f"Alive agents: {len(df_alive):,} observations")
    print(f"Age range: {df_alive['age'].min()} - {df_alive['age'].max()}")
    print(f"Number of agents: {df_alive.index.get_level_values('agent').nunique():,}")

    # Generate summary statistics
    summary_stats = generate_summary_statistics(df_alive)
    summary_stats.to_csv(path_to_save_summary, index=True)

    # Generate diagnostic plots
    create_diagnostic_plots(df_alive, path_to_save_plots_dir)

    print(f"Summary statistics saved to: {path_to_save_summary}")
    print(f"Diagnostic plots saved to: {path_to_save_plots_dir}")


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive summary statistics."""

    stats = {}

    # Overall statistics
    stats["n_observations"] = len(df)
    stats["n_agents"] = df.index.get_level_values("agent").nunique()
    stats["n_periods"] = df.index.get_level_values("period").nunique()

    # Age statistics
    stats["min_age"] = df["age"].min()
    stats["max_age"] = df["age"].max()
    stats["mean_age"] = df["age"].mean()

    # Care demand statistics
    stats["share_any_care_demand"] = (df["care_demand"] > 0).mean()
    stats["share_moderate_care_demand"] = (
        df["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY
    ).mean()
    stats["share_high_care_demand"] = (
        df["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY
    ).mean()
    stats["mean_care_demand"] = df["care_demand"].mean()

    # Choice statistics (assuming 0=retired, 1=unemployed, 2=part-time, 3=full-time)
    for choice in sorted(df["choice"].unique()):
        stats[f"share_choice_{choice}"] = (df["choice"] == choice).mean()

    # Health statistics
    for health in sorted(df["health"].unique()):
        stats[f"share_health_{health}"] = (df["health"] == health).mean()

    # Education statistics
    for edu in sorted(df["education"].unique()):
        stats[f"share_education_{edu}"] = (df["education"] == edu).mean()

    # Partner statistics
    for partner in sorted(df["partner_state"].unique()):
        stats[f"share_partner_state_{partner}"] = (
            df["partner_state"] == partner
        ).mean()

    # Wealth statistics
    if "savings" in df.columns:
        stats["mean_savings"] = df["savings"].mean()
        stats["median_savings"] = df["savings"].median()
        stats["std_savings"] = df["savings"].std()

    if "consumption" in df.columns:
        stats["mean_consumption"] = df["consumption"].mean()
        stats["median_consumption"] = df["consumption"].median()
        stats["std_consumption"] = df["consumption"].std()

    # Experience statistics
    if "experience" in df.columns:
        stats["mean_experience"] = df["experience"].mean()
        stats["median_experience"] = df["experience"].median()
        stats["std_experience"] = df["experience"].std()

    # Convert to DataFrame
    summary_df = pd.DataFrame(list(stats.items()), columns=["statistic", "value"])

    return summary_df


def create_diagnostic_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive diagnostic plots."""

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Care demand by age (detailed)
    create_care_demand_by_age_plot(df, output_dir)

    # 2. Choice distribution by age
    create_choice_distribution_by_age_plot(df, output_dir)

    # 3. Wealth accumulation by age
    create_wealth_by_age_plot(df, output_dir)

    # 4. Care demand by demographics
    create_care_demand_by_demographics_plot(df, output_dir)

    # 5. Health transition patterns
    create_health_patterns_plot(df, output_dir)

    # 6. Experience accumulation
    create_experience_patterns_plot(df, output_dir)


def create_care_demand_by_age_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create detailed care demand by age plot."""

    # Calculate shares by age
    age_stats = (
        df.groupby("age")
        .agg(
            {
                "care_demand": [
                    lambda x: (x > 0).mean(),  # any care
                    lambda x: (
                        x == CARE_DEMAND_AND_OTHER_SUPPLY
                    ).mean(),  # moderate care
                    lambda x: (
                        x == CARE_DEMAND_AND_NO_OTHER_SUPPLY
                    ).mean(),  # high care
                    "count",
                ]
            }
        )
        .round(4)
    )

    age_stats.columns = [
        "share_any_care",
        "share_moderate_care",
        "share_high_care",
        "n_obs",
    ]
    age_stats = age_stats.reset_index()

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Main plot
    ax1.plot(
        age_stats["age"],
        age_stats["share_any_care"],
        label="Any care demand (> 0)",
        linewidth=2.5,
        marker="o",
    )
    ax1.plot(
        age_stats["age"],
        age_stats["share_moderate_care"],
        label="Moderate care demand (= 1)",
        linewidth=2.5,
        marker="s",
    )
    ax1.plot(
        age_stats["age"],
        age_stats["share_high_care"],
        label="High care demand (= 2)",
        linewidth=2.5,
        marker="^",
    )

    ax1.set_xlabel("Age")
    ax1.set_ylabel("Share of Agents")
    ax1.set_title("Care Demand Shares by Age", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    # Sample size plot
    ax2.bar(age_stats["age"], age_stats["n_obs"], alpha=0.7, color="gray")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Number of Observations")
    ax2.set_title("Sample Size by Age")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "care_demand_by_age_detailed.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_choice_distribution_by_age_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create choice distribution by age plot."""

    # Calculate choice shares by age
    choice_by_age = df.groupby(["age", "choice"]).size().unstack(fill_value=0)
    choice_shares = choice_by_age.div(choice_by_age.sum(axis=1), axis=0)

    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(12, 8))

    choice_labels = ["Retired", "Unemployed", "Part-time", "Full-time"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    ax.stackplot(
        choice_shares.index,
        *[choice_shares[col] for col in choice_shares.columns],
        labels=[
            choice_labels[i] if i < len(choice_labels) else f"Choice {i}"
            for i in choice_shares.columns
        ],
        colors=colors[: len(choice_shares.columns)],
        alpha=0.8,
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Share of Agents")
    ax.set_title("Labor Market Choice Distribution by Age", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(
        output_dir / "choice_distribution_by_age.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_wealth_by_age_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create wealth accumulation by age plot."""

    if "savings" not in df.columns or "consumption" not in df.columns:
        print("Skipping wealth plot - savings/consumption columns not found")
        return

    # Calculate wealth statistics by age
    df["wealth"] = df["savings"] + df["consumption"]
    wealth_stats = (
        df.groupby("age")["wealth"].agg(["mean", "median", "std"]).reset_index()
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        wealth_stats["age"],
        wealth_stats["mean"],
        label="Mean wealth",
        linewidth=2.5,
        marker="o",
    )
    ax.plot(
        wealth_stats["age"],
        wealth_stats["median"],
        label="Median wealth",
        linewidth=2.5,
        marker="s",
    )

    # Add confidence bands
    ax.fill_between(
        wealth_stats["age"],
        wealth_stats["mean"] - wealth_stats["std"],
        wealth_stats["mean"] + wealth_stats["std"],
        alpha=0.2,
        label="±1 std dev",
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Wealth")
    ax.set_title("Wealth Accumulation by Age", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "wealth_by_age.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_care_demand_by_demographics_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create care demand by demographics plot."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # By education
    edu_care = df.groupby("education")["care_demand"].apply(lambda x: (x > 0).mean())
    axes[0, 0].bar(edu_care.index, edu_care.values, alpha=0.7)
    axes[0, 0].set_title("Care Demand by Education")
    axes[0, 0].set_xlabel("Education Level")
    axes[0, 0].set_ylabel("Share with Care Demand")
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    # By health status
    health_care = df.groupby("health")["care_demand"].apply(lambda x: (x > 0).mean())
    axes[0, 1].bar(health_care.index, health_care.values, alpha=0.7)
    axes[0, 1].set_title("Care Demand by Health Status")
    axes[0, 1].set_xlabel("Health Status")
    axes[0, 1].set_ylabel("Share with Care Demand")
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    # By partner status
    partner_care = df.groupby("partner_state")["care_demand"].apply(
        lambda x: (x > 0).mean()
    )
    axes[1, 0].bar(partner_care.index, partner_care.values, alpha=0.7)
    axes[1, 0].set_title("Care Demand by Partner Status")
    axes[1, 0].set_xlabel("Partner State")
    axes[1, 0].set_ylabel("Share with Care Demand")
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    # By age groups
    df["age_group"] = pd.cut(
        df["age"], bins=[39, 45, 55, 65, 100], labels=["40-45", "46-55", "56-65", "66+"]
    )
    age_group_care = df.groupby("age_group")["care_demand"].apply(
        lambda x: (x > 0).mean()
    )
    axes[1, 1].bar(range(len(age_group_care)), age_group_care.values, alpha=0.7)
    axes[1, 1].set_title("Care Demand by Age Group")
    axes[1, 1].set_xlabel("Age Group")
    axes[1, 1].set_ylabel("Share with Care Demand")
    axes[1, 1].set_xticks(range(len(age_group_care)))
    axes[1, 1].set_xticklabels(age_group_care.index)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    plt.tight_layout()
    plt.savefig(
        output_dir / "care_demand_by_demographics.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_health_patterns_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create health transition patterns plot."""

    # Health distribution by age
    health_by_age = df.groupby(["age", "health"]).size().unstack(fill_value=0)
    health_shares = health_by_age.div(health_by_age.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 8))

    health_labels = ["Bad Health", "Good Health", "Dead"]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]

    for _i, health_state in enumerate(health_shares.columns):
        if health_state < len(health_labels):
            label = health_labels[health_state]
            color = colors[health_state] if health_state < len(colors) else None
        else:
            label = f"Health State {health_state}"
            color = None

        ax.plot(
            health_shares.index,
            health_shares[health_state],
            label=label,
            linewidth=2.5,
            marker="o",
            color=color,
        )

    ax.set_xlabel("Age")
    ax.set_ylabel("Share of Agents")
    ax.set_title("Health Status Distribution by Age", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(output_dir / "health_patterns_by_age.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_experience_patterns_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create experience accumulation patterns plot."""

    if "experience" not in df.columns:
        print("Skipping experience plot - experience column not found")
        return

    # Experience statistics by age
    exp_stats = (
        df.groupby("age")["experience"].agg(["mean", "median", "std"]).reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        exp_stats["age"],
        exp_stats["mean"],
        label="Mean experience",
        linewidth=2.5,
        marker="o",
    )
    ax.plot(
        exp_stats["age"],
        exp_stats["median"],
        label="Median experience",
        linewidth=2.5,
        marker="s",
    )

    # Add confidence bands
    ax.fill_between(
        exp_stats["age"],
        exp_stats["mean"] - exp_stats["std"],
        exp_stats["mean"] + exp_stats["std"],
        alpha=0.2,
        label="±1 std dev",
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Experience Level")
    ax.set_title("Experience Accumulation by Age", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "experience_patterns_by_age.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
