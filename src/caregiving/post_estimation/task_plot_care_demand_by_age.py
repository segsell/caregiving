"""Plot care demand shares by age from estimated model simulation."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
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
def task_plot_care_demand_by_age(
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "post_estimation"
    / "care_demand_shares_by_age.png",
) -> None:
    """Plot care demand shares by age.

    Creates a plot showing the share of agents with:
    - care_demand > 0 (any care demand)
    - care_demand == 1 (moderate care demand)
    - care_demand == 2 (high care demand)

    Conditions on agents being alive (health != DEAD).
    """

    # Load simulated data
    df = pd.read_pickle(path_to_simulated_data)

    # Filter to alive agents only
    df_alive = df[df["health"] != DEAD].copy()

    # Calculate shares by age
    care_demand_shares = []

    for age in sorted(df_alive["age"].unique()):
        age_data = df_alive[df_alive["age"] == age]
        n_agents = len(age_data)

        if n_agents > 0:
            share_any_care = (age_data["care_demand"] > 0).mean()
            share_moderate_care = (
                age_data["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY
            ).mean()
            share_high_care = (
                age_data["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY
            ).mean()

            care_demand_shares.append(
                {
                    "age": age,
                    "share_any_care": share_any_care,
                    "share_moderate_care": share_moderate_care,
                    "share_high_care": share_high_care,
                    "n_agents": n_agents,
                }
            )

    # Convert to DataFrame
    shares_df = pd.DataFrame(care_demand_shares)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot the three lines
    plt.plot(
        shares_df["age"],
        shares_df["share_any_care"],
        label="Any care demand (> 0)",
        linewidth=2.5,
        color="#1f77b4",
        marker="o",
        markersize=4,
    )

    plt.plot(
        shares_df["age"],
        shares_df["share_moderate_care"],
        label="Moderate care demand (= 1)",
        linewidth=2.5,
        color="#ff7f0e",
        marker="s",
        markersize=4,
    )

    plt.plot(
        shares_df["age"],
        shares_df["share_high_care"],
        label="High care demand (= 2)",
        linewidth=2.5,
        color="#2ca02c",
        marker="^",
        markersize=4,
    )

    # Customize the plot
    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Share of Agents", fontsize=14)
    plt.title(
        "Care Demand Shares by Age (Alive Agents Only)", fontsize=16, fontweight="bold"
    )
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, alpha=0.3)

    # Set axis limits and ticks
    plt.xlim(shares_df["age"].min() - 1, 75)
    plt.ylim(0, 0.20)

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    # Add sample size annotation
    min_n = shares_df["n_agents"].min()
    max_n = shares_df["n_agents"].max()
    plt.figtext(
        0.02,
        0.02,
        f"Sample size per age: {min_n:,} - {max_n:,} agents",
        fontsize=10,
        style="italic",
    )

    plt.tight_layout()

    # Save the plot
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_plot, dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary statistics
    print("Care Demand Shares by Age - Summary Statistics")
    print("=" * 50)
    print(f"Age range: {shares_df['age'].min()} - {shares_df['age'].max()}")
    print(f"Total observations: {shares_df['n_agents'].sum():,}")
    print(
        f"Average share with any care demand: {shares_df['share_any_care'].mean():.1%}"
    )
    moderate_share = shares_df["share_moderate_care"].mean()
    print(f"Average share with moderate care demand: {moderate_share:.1%}")
    high_share = shares_df["share_high_care"].mean()
    print(f"Average share with high care demand: {high_share:.1%}")
    peak_age = shares_df.loc[shares_df["share_any_care"].idxmax(), "age"]
    print(f"Peak age for any care demand: {peak_age}")

    # Save summary data
    summary_path = path_to_save_plot.parent / "care_demand_shares_by_age_data.csv"
    shares_df.to_csv(summary_path, index=False)
    print(f"Summary data saved to: {summary_path}")


if __name__ == "__main__":
    task_plot_care_demand_by_age()
