"""Plot sibling statistics by age for different caregiver samples."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.moments.task_create_soep_moments import (
    create_df_caregivers,
    create_df_non_caregivers,
    create_df_with_caregivers,
)
from caregiving.specs.task_write_specs import read_and_derive_specs


@pytask.mark.pre_estimation
def task_plot_siblings(  # noqa: PLR0915
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_main_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_caregivers_sample: Path = BLD
    / "data"
    / "soep_structural_caregivers_sample.csv",
    path_to_save_avg_siblings: Annotated[Path, Product] = BLD
    / "plots"
    / "pre_estimation"
    / "average_siblings_by_age.png",
    path_to_save_share_zero_siblings: Annotated[Path, Product] = BLD
    / "plots"
    / "pre_estimation"
    / "share_zero_siblings_by_age.png",
    path_to_save_avg_has_sister: Annotated[Path, Product] = BLD
    / "plots"
    / "pre_estimation"
    / "average_has_sister_by_age.png",
) -> None:
    """Plot average number of siblings, share with 0 siblings, and has_sister by age.

    Creates three plots:
    1. Average number of siblings by age for three samples:
       - Non-caregivers
       - With caregivers (all)
       - Caregivers only
    2. Share of individuals with 0 siblings by age for the same three samples.
    3. Average has_sister by age for the same three samples.

    Parameters
    ----------
    path_to_specs : Path
        Path to specs.yaml file
    path_to_main_sample : Path
        Path to main structural estimation sample CSV
    path_to_caregivers_sample : Path
        Path to caregivers sample CSV
    path_to_save_avg_siblings : Path
        Path to save average siblings plot
    path_to_save_share_zero_siblings : Path
        Path to save share zero siblings plot
    path_to_save_avg_has_sister : Path
        Path to save average has_sister plot
    """
    specs = read_and_derive_specs(path_to_specs)

    start_age = specs["start_age"]
    end_age = specs["end_age_msm"]
    start_year = 2001
    end_year = 2019

    # Load data
    df_full = pd.read_csv(path_to_main_sample, index_col=[0])
    df_caregivers_full = pd.read_csv(path_to_caregivers_sample, index_col=[0])

    # Create standardized subsamples using shared functions
    df_non_caregivers = create_df_non_caregivers(
        df_full=df_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age,
    )
    df_with_caregivers = create_df_with_caregivers(
        df_full=df_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age,
    )
    df_caregivers = create_df_caregivers(
        df_caregivers_full=df_caregivers_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age,
    )

    # Filter to age range
    age_range = range(start_age, end_age + 1)
    df_non_caregivers = df_non_caregivers[
        df_non_caregivers["age"].isin(age_range)
    ].copy()
    df_with_caregivers = df_with_caregivers[
        df_with_caregivers["age"].isin(age_range)
    ].copy()
    df_caregivers = df_caregivers[df_caregivers["age"].isin(age_range)].copy()

    # Compute statistics by age for each sample
    def compute_avg_siblings_by_age(df):
        """Compute average number of siblings by age."""
        return df.groupby("age", observed=False)["n_siblings"].mean()

    def compute_share_zero_siblings_by_age(df):
        """Compute share of individuals with 0 siblings by age."""
        return df.groupby("age", observed=False)["n_siblings"].apply(
            lambda x: (x == 0).sum() / len(x)
        )

    def compute_avg_has_sister_by_age(df):
        """Compute average has_sister by age."""
        return df.groupby("age", observed=False)["has_sister"].mean()

    avg_siblings_non_caregivers = compute_avg_siblings_by_age(df_non_caregivers)
    avg_siblings_with_caregivers = compute_avg_siblings_by_age(df_with_caregivers)
    avg_siblings_caregivers = compute_avg_siblings_by_age(df_caregivers)

    share_zero_non_caregivers = compute_share_zero_siblings_by_age(df_non_caregivers)
    share_zero_with_caregivers = compute_share_zero_siblings_by_age(df_with_caregivers)
    share_zero_caregivers = compute_share_zero_siblings_by_age(df_caregivers)

    avg_has_sister_non_caregivers = compute_avg_has_sister_by_age(df_non_caregivers)
    avg_has_sister_with_caregivers = compute_avg_has_sister_by_age(df_with_caregivers)
    avg_has_sister_caregivers = compute_avg_has_sister_by_age(df_caregivers)

    # Reindex to ensure all ages in range are included
    age_index = pd.Index(age_range, name="age")
    avg_siblings_non_caregivers = avg_siblings_non_caregivers.reindex(
        age_index, fill_value=np.nan
    )
    avg_siblings_with_caregivers = avg_siblings_with_caregivers.reindex(
        age_index, fill_value=np.nan
    )
    avg_siblings_caregivers = avg_siblings_caregivers.reindex(
        age_index, fill_value=np.nan
    )

    share_zero_non_caregivers = share_zero_non_caregivers.reindex(
        age_index, fill_value=np.nan
    )
    share_zero_with_caregivers = share_zero_with_caregivers.reindex(
        age_index, fill_value=np.nan
    )
    share_zero_caregivers = share_zero_caregivers.reindex(age_index, fill_value=np.nan)

    avg_has_sister_non_caregivers = avg_has_sister_non_caregivers.reindex(
        age_index, fill_value=np.nan
    )
    avg_has_sister_with_caregivers = avg_has_sister_with_caregivers.reindex(
        age_index, fill_value=np.nan
    )
    avg_has_sister_caregivers = avg_has_sister_caregivers.reindex(
        age_index, fill_value=np.nan
    )

    # Plot 1: Average number of siblings by age
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        age_range,
        avg_siblings_non_caregivers,
        label="Non-caregivers",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax1.plot(
        age_range,
        avg_siblings_with_caregivers,
        label="With caregivers (all)",
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax1.plot(
        age_range,
        avg_siblings_caregivers,
        label="Caregivers only",
        linewidth=2,
        marker="^",
        markersize=4,
    )

    ax1.set_xlabel("Age", fontsize=12)
    ax1.set_ylabel("Average Number of Siblings", fontsize=12)
    ax1.set_title("Average Number of Siblings by Age", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([start_age, end_age])

    plt.tight_layout()
    path_to_save_avg_siblings.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_avg_siblings, dpi=300)
    plt.close(fig1)

    # Plot 2: Share with 0 siblings by age
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(
        age_range,
        share_zero_non_caregivers,
        label="Non-caregivers",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax2.plot(
        age_range,
        share_zero_with_caregivers,
        label="With caregivers (all)",
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax2.plot(
        age_range,
        share_zero_caregivers,
        label="Caregivers only",
        linewidth=2,
        marker="^",
        markersize=4,
    )

    ax2.set_xlabel("Age", fontsize=12)
    ax2.set_ylabel("Share with 0 Siblings", fontsize=12)
    ax2.set_title("Share of Individuals with 0 Siblings by Age", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([start_age, end_age])
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    path_to_save_share_zero_siblings.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_share_zero_siblings, dpi=300)
    plt.close(fig2)

    # Plot 3: Average has_sister by age
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    ax3.plot(
        age_range,
        avg_has_sister_non_caregivers,
        label="Non-caregivers",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax3.plot(
        age_range,
        avg_has_sister_with_caregivers,
        label="With caregivers (all)",
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax3.plot(
        age_range,
        avg_has_sister_caregivers,
        label="Caregivers only",
        linewidth=2,
        marker="^",
        markersize=4,
    )

    ax3.set_xlabel("Age", fontsize=12)
    ax3.set_ylabel("Average has_sister", fontsize=12)
    ax3.set_title("Average has_sister by Age", fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([start_age, end_age])
    ax3.set_ylim([0, 1])

    plt.tight_layout()
    path_to_save_avg_has_sister.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_avg_has_sister, dpi=300)
    plt.close(fig3)

    print(f"Average siblings plot saved to {path_to_save_avg_siblings}")
    print(f"Share zero siblings plot saved to {path_to_save_share_zero_siblings}")
    print(f"Average has_sister plot saved to {path_to_save_avg_has_sister}")
