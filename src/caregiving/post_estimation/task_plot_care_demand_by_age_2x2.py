"""Plot care demand by age in a 2x2 grid for post-estimation analysis."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.estimation.prepare_estimation import (
    load_and_setup_full_model_for_solution,
)
from caregiving.model.shared import (
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    CARE_DEMAND_AND_OTHER_SUPPLY,
    DEAD,
    INFORMAL_CARE,
    NO_CARE,
    SEX,
)


@pytask.mark.post_estimation
@pytask.mark.care_demand_post_estimation
def task_plot_care_demand_by_age_2x2(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_estimated_params: Path = BLD
    / "model"
    / "params"
    / "estimated_params_model.yaml",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "care_demand_by_age_2x2.png",
) -> None:
    """Plot care demand by age in a 2x2 grid (education × has_sister)."""

    options = pickle.load(path_to_options.open("rb"))

    model_full = load_and_setup_full_model_for_solution(
        options, path_to_model=path_to_solution_model
    )
    specs = model_full["options"]["model_params"]

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    # Test that care mix sums to care demand
    test_care_mix_sums_to_care_demand(
        df_sim=df_sim, specs=specs, age_min=40, age_max=80
    )

    plot_simulated_care_demand_by_age_2x2(
        df_sim=df_sim,
        specs=specs,
        age_min=40,
        age_max=80,
        path_to_save_plot=path_to_save_plot,
    )


# ============================================================================
# Auxiliary functions
# ============================================================================


def plot_simulated_care_demand_by_age_2x2(
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the yearly share with care_demand == 1 in a 2x2 grid,
    with one subplot for each combination of:
    • education (0 = low, 1 = high) → rows
    • has_sister (0 / 1)            → columns

    Layout:
    - Top left: Low education, No sister
    - Top right: Low education, Has sister
    - Bottom left: High education, No sister
    - Bottom right: High education, Has sister
    """

    # ---- 1. Setup
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = 100

    ages = np.arange(age_min, age_max + 1)

    df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()
    if "sex" in df_sim.columns:
        df_sim = df_sim.loc[df_sim["sex"] == SEX].copy()

    # ---- 2. Calculate care mix shares
    # Convert JAX arrays to numpy arrays for pandas compatibility
    informal_care_choices = np.asarray(INFORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Create care type indicators
    df_sim["informal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(informal_care_choices))
    ).astype(int)

    df_sim["other_informal_care"] = (
        df_sim["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY
    ).astype(int)

    df_sim["formal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(no_care_choices))
    ).astype(int)

    # Calculate shares for care demand (any care demand)
    care_demand_shares = (
        df_sim.groupby(["age", "education", "has_sister"], observed=False)[
            "care_demand"
        ]
        .apply(lambda x: (x > 0).mean())
        .reindex(
            pd.MultiIndex.from_product(
                [ages, [0, 1], [0, 1]], names=["age", "education", "has_sister"]
            ),
            fill_value=0,
        )
    )

    # Calculate care mix shares (conditional on care demand > 0)
    care_mix_shares = {}
    for care_type in ("informal_care", "other_informal_care", "formal_care"):
        shares = (
            df_sim.groupby(["age", "education", "has_sister"], observed=False)[
                care_type
            ]
            .mean()
            .reindex(
                pd.MultiIndex.from_product(
                    [ages, [0, 1], [0, 1]], names=["age", "education", "has_sister"]
                ),
                fill_value=0,
            )
        )
        care_mix_shares[care_type] = shares

    # ---- 3. Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Define subplot positions: (education, has_sister) -> index
    # Top left (0): Low edu (0), No sister (0)
    # Top right (1): Low edu (0), Has sister (1)
    # Bottom left (2): High edu (1), No sister (0)
    # Bottom right (3): High edu (1), Has sister (1)
    subplot_map = {
        (0, 0): 0,  # Low edu, No sister
        (0, 1): 1,  # Low edu, Has sister
        (1, 0): 2,  # High edu, No sister
        (1, 1): 3,  # High edu, Has sister
    }

    # Labels for titles
    edu_labels = {0: "Low education", 1: "High education"}
    sister_labels = {0: "No sister", 1: "Has sister"}

    # Colors for care mix (stacked from bottom to top)
    care_colors = {
        "informal_care": "#2E86AB",  # Blue
        "other_informal_care": "#A23B72",  # Purple
        "formal_care": "#F18F01",  # Orange
    }

    # ---- 4. Plot each combination
    for edu in (0, 1):
        for has_sister in (0, 1):
            idx = subplot_map[(edu, has_sister)]
            ax = axes[idx]

            # Get care demand share
            care_demand_series = care_demand_shares.xs(
                (edu, has_sister), level=("education", "has_sister")
            )

            # Get care mix shares
            informal_series = care_mix_shares["informal_care"].xs(
                (edu, has_sister), level=("education", "has_sister")
            )
            other_informal_series = care_mix_shares["other_informal_care"].xs(
                (edu, has_sister), level=("education", "has_sister")
            )
            formal_series = care_mix_shares["formal_care"].xs(
                (edu, has_sister), level=("education", "has_sister")
            )

            # Plot care demand curve
            ax.plot(
                ages,
                care_demand_series,
                color="black",
                linewidth=2,
                label="Care demand",
            )

            # Plot stacked area for care mix (below the curve)
            # Stack from bottom to top: informal, other informal, formal
            ax.fill_between(
                ages,
                0,
                informal_series,
                color=care_colors["informal_care"],
                alpha=0.6,
                label="Informal care",
            )
            ax.fill_between(
                ages,
                informal_series,
                informal_series + other_informal_series,
                color=care_colors["other_informal_care"],
                alpha=0.6,
                label="Other informal care",
            )
            ax.fill_between(
                ages,
                informal_series + other_informal_series,
                informal_series + other_informal_series + formal_series,
                color=care_colors["formal_care"],
                alpha=0.6,
                label="Formal care",
            )

            # Cosmetics
            pad = 1
            ax.set_xlabel("Age")
            ax.set_ylabel("Share")
            ax.set_xlim(age_min - pad, age_max + pad)
            ax.set_ylim(0, None)  # Let y-axis adjust automatically
            ax.set_title(f"{edu_labels[edu]}, {sister_labels[has_sister]}")
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


def test_care_mix_sums_to_care_demand(df_sim, specs, age_min=None, age_max=None):
    """
    Test that the three care modes (informal, other informal, formal) sum up
    to the number of agents facing care demand at each given age.

    This function asserts that the care mix shares sum to 1 (i.e., the absolute
    counts of the three care modes sum to the number of agents with care demand).

    Parameters
    ----------
    df_sim : pd.DataFrame
        Simulated data with columns: age, education, has_sister, care_demand, choice
    specs : dict
        Model specifications
    age_min : int, optional
        Minimum age to test. If None, uses specs["start_age"]
    age_max : int, optional
        Maximum age to test. If None, uses 100

    Raises
    ------
    AssertionError
        If the care mix does not sum to care demand within tolerance.
    """
    # Setup
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = 100

    # Filter data
    df_test = df_sim.loc[df_sim["health"] != DEAD].copy()
    if "sex" in df_test.columns:
        df_test = df_test.loc[df_test["sex"] == SEX].copy()

    # Convert JAX arrays to numpy arrays for pandas compatibility
    informal_care_choices = np.asarray(INFORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Create care type indicators
    df_test["informal_care"] = (
        (df_test["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_test["choice"].isin(informal_care_choices))
    ).astype(int)

    df_test["other_informal_care"] = (
        df_test["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY
    ).astype(int)

    df_test["formal_care"] = (
        (df_test["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_test["choice"].isin(no_care_choices))
    ).astype(int)

    # Calculate absolute counts by age, education, has_sister
    group_cols = ["age", "education", "has_sister"]
    counts = df_test.groupby(group_cols, observed=False).agg(
        {
            "care_demand": lambda x: (x > 0).sum(),  # Count with care demand
            "informal_care": "sum",
            "other_informal_care": "sum",
            "formal_care": "sum",
        }
    )

    # Calculate sum of three care modes
    counts["care_mix_sum"] = (
        counts["informal_care"] + counts["other_informal_care"] + counts["formal_care"]
    )

    # Calculate differences
    counts["absolute_diff"] = np.abs(counts["care_demand"] - counts["care_mix_sum"])
    counts["relative_diff"] = np.where(
        counts["care_demand"] > 0,
        counts["absolute_diff"] / counts["care_demand"],
        0,
    )

    # Find maximum differences
    max_absolute_diff = counts["absolute_diff"].max()
    max_relative_diff = counts["relative_diff"].max()

    # Assert that the test passed
    tolerance = 1e-10
    assert max_absolute_diff < tolerance, (
        f"Care mix does not sum to care demand. "
        f"Max absolute difference: {max_absolute_diff}, "
        f"Max relative difference: {max_relative_diff}"
    )
