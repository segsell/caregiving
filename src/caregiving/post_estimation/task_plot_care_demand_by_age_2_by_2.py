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
    PARENT_BAD_HEALTH,
    PARENT_DEAD,
    PARENT_GOOD_HEALTH,
    PARENT_MEDIUM_HEALTH,
    SEX,
)


@pytask.mark.baseline_model
@pytask.mark.post_estimation
@pytask.mark.care_demand_post_estimation
def task_plot_care_demand_by_age_2_by_2(
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
    / "care_demand_by_age_2_by_2.png",
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

    plot_simulated_care_demand_by_age_2_by_2(
        df_sim=df_sim,
        specs=specs,
        age_min=40,
        age_max=80,
        path_to_save_plot=path_to_save_plot,
    )


@pytask.mark.baseline_model
@pytask.mark.post_estimation
@pytask.mark.care_demand_post_estimation
def task_plot_care_demand_by_age_pooled(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
    # path_to_estimated_params: Path = BLD
    # / "model"
    # / "params"
    # / "estimated_params_model.yaml",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "care_demand_by_age_pooled.png",
) -> None:
    """Plot care demand by age pooled across all education and sister specifications."""

    options = pickle.load(path_to_options.open("rb"))
    model_full = load_and_setup_full_model_for_solution(
        options, path_to_model=path_to_solution_model
    )
    specs = model_full["options"]["model_params"]

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    plot_simulated_care_demand_by_age_pooled(
        df_sim=df_sim,
        specs=specs,
        age_min=40,
        age_max=75,
        path_to_save_plot=path_to_save_plot,
    )


@pytask.mark.baseline_model
@pytask.mark.post_estimation
@pytask.mark.care_demand_post_estimation
def task_plot_care_demand_by_age_2_by_2_combined(
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
    / "care_demand_by_age_2_by_2_combined.png",
) -> None:
    """Plot care demand by age in a 2x2 grid with combined informal care categories."""

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

    plot_simulated_care_demand_by_age_2_by_2_combined(
        df_sim=df_sim,
        specs=specs,
        age_min=40,
        age_max=80,
        path_to_save_plot=path_to_save_plot,
    )


@pytask.mark.baseline_model
@pytask.mark.post_estimation
@pytask.mark.care_demand_post_estimation
def task_plot_care_demand_by_age_pooled_combined(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
    # path_to_estimated_params: Path = BLD
    # / "model"
    # / "params"
    # / "estimated_params_model.yaml",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "care_demand_by_age_pooled_combined.png",
) -> None:
    """Plot care demand by age pooled with combined informal care categories."""

    options = pickle.load(path_to_options.open("rb"))
    model_full = load_and_setup_full_model_for_solution(
        options, path_to_model=path_to_solution_model
    )
    specs = model_full["options"]["model_params"]

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    plot_simulated_care_demand_by_age_pooled_combined(
        df_sim=df_sim,
        specs=specs,
        age_min=40,
        age_max=75,
        path_to_save_plot=path_to_save_plot,
    )


@pytask.mark.post_estimation
@pytask.mark.care_demand_post_estimation
def task_plot_mother_health_shares_by_age(
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
    / "mother_health_shares_by_age.png",
) -> None:
    """Plot the share of mother health states (good, medium, bad, dead) by age."""

    options = pickle.load(path_to_options.open("rb"))

    model_full = load_and_setup_full_model_for_solution(
        options, path_to_model=path_to_solution_model
    )
    specs = model_full["options"]["model_params"]

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    plot_mother_health_shares_by_age(
        df_sim=df_sim,
        specs=specs,
        age_min=50,
        age_max=100,
        path_to_save_plot=path_to_save_plot,
    )


# ============================================================================
# Auxiliary functions
# ============================================================================


def plot_simulated_care_demand_by_age_2_by_2(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the yearly share with care_demand > 0 in a 2x2 grid,
    with one subplot for each combination of:
    • education (0 = low, 1 = high) → rows
    • has_sister (0 / 1)            → columns

    Shows all four types of care choices upon positive care demand:
    1. Solo informal care (care_demand == 2, agent chooses informal care)
    2. Formal care (care_demand == 2, agent chooses no care)
    3. Joint informal care (care_demand == 1, agent chooses informal care)
    4. Other family member only (care_demand == 1, agent chooses no informal care)

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

    # ---- 2. Calculate care type indicators for all four scenarios
    # Convert JAX arrays to numpy arrays for pandas compatibility
    informal_care_choices = np.asarray(INFORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Four types of care choices upon positive care demand:
    # 1. Solo informal care: care_demand == 2 AND agent chooses informal care
    df_sim["solo_informal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(informal_care_choices))
    ).astype(int)

    # 2. Formal care: care_demand == 2 AND agent chooses no care
    df_sim["formal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(no_care_choices))
    ).astype(int)

    # 3. Joint informal care: care_demand == 1 AND agent chooses informal care
    df_sim["joint_informal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
        & (df_sim["choice"].isin(informal_care_choices))
    ).astype(int)

    # 4. Other family member only: care_demand == 1 AND agent chooses no informal care
    df_sim["other_family_only"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
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

    # Calculate care mix shares for all four types
    care_mix_shares = {}
    for care_type in (
        "solo_informal_care",
        "formal_care",
        "joint_informal_care",
        "other_family_only",
    ):
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
        "solo_informal_care": "#2E86AB",  # Blue
        "formal_care": "#A23B72",  # Purple
        "joint_informal_care": "#F18F01",  # Orange
        "other_family_only": "#D3D3D3",  # Light grey
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
            solo_informal_series = care_mix_shares["solo_informal_care"].xs(
                (edu, has_sister), level=("education", "has_sister")
            )
            formal_series = care_mix_shares["formal_care"].xs(
                (edu, has_sister), level=("education", "has_sister")
            )
            joint_informal_series = care_mix_shares["joint_informal_care"].xs(
                (edu, has_sister), level=("education", "has_sister")
            )
            other_family_series = care_mix_shares["other_family_only"].xs(
                (edu, has_sister), level=("education", "has_sister")
            )

            # Plot stacked area for care mix (below the curve)
            # Stack from bottom to top: solo informal, joint informal, formal,
            # other family only
            bottom = 0
            ax.fill_between(
                ages,
                bottom,
                bottom + solo_informal_series,
                color=care_colors["solo_informal_care"],
                alpha=0.6,
                label="Solo informal care",
            )
            bottom += solo_informal_series
            ax.fill_between(
                ages,
                bottom,
                bottom + joint_informal_series,
                color=care_colors["joint_informal_care"],
                alpha=0.6,
                label="Joint informal care",
            )
            bottom += joint_informal_series
            ax.fill_between(
                ages,
                bottom,
                bottom + formal_series,
                color=care_colors["formal_care"],
                alpha=0.6,
                label="Formal care",
            )
            bottom += formal_series
            ax.fill_between(
                ages,
                bottom,
                bottom + other_family_series,
                color=care_colors["other_family_only"],
                alpha=0.6,
                label="Other family care",
            )

            # Plot care demand curve (on top)
            ax.plot(
                ages,
                care_demand_series,
                color="black",
                linewidth=2,
                label="Care demand",
            )

            # Cosmetics
            pad = 1
            ax.set_xlabel("Age")
            ax.set_ylabel("Share")
            ax.set_xlim(age_min - pad, 75 + pad)  # Cut x-axis at 75
            ax.set_ylim(0, None)  # Let y-axis adjust automatically
            ax.set_title(f"{edu_labels[edu]}, {sister_labels[has_sister]}")

            # Get handles and labels, then reorder to show from bottom to top
            # Legend order: Care demand at top, then care types from top to bottom
            handles, labels = ax.get_legend_handles_labels()
            # Separate care demand from care types
            care_demand_idx = labels.index("Care demand")
            care_handles = [h for i, h in enumerate(handles) if i != care_demand_idx]
            care_labels = [
                label for i, label in enumerate(labels) if i != care_demand_idx
            ]
            # Replace "Other family only" with "Other family care"
            care_labels = [
                "Other family care" if label == "Other family only" else label
                for label in care_labels
            ]
            # Reverse care types so legend shows from bottom to top
            # (other family only at top)
            care_handles_reversed = care_handles[::-1]
            care_labels_reversed = care_labels[::-1]
            # Combine: care demand first, then reversed care types
            final_handles = [handles[care_demand_idx]] + care_handles_reversed
            final_labels = [labels[care_demand_idx]] + care_labels_reversed
            ax.legend(final_handles, final_labels, loc="upper left", fontsize=8)

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


def plot_simulated_care_demand_by_age_pooled(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the yearly share with care_demand > 0.

    Pooled across all education and sister groups.

    Shows all four types of care choices upon positive care demand:
    1. Solo informal care (care_demand == 2, agent chooses informal care)
    2. Formal care (care_demand == 2, agent chooses no care)
    3. Joint informal care (care_demand == 1, agent chooses informal care)
    4. Other family care (care_demand == 1, agent chooses no informal care)
    """

    # ---- 1. Setup
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = 75

    ages = np.arange(age_min, age_max + 1)

    df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()
    if "sex" in df_sim.columns:
        df_sim = df_sim.loc[df_sim["sex"] == SEX].copy()

    # ---- 2. Calculate care type indicators
    # Convert JAX arrays to numpy arrays for pandas compatibility
    informal_care_choices = np.asarray(INFORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Four types of care choices upon positive care demand:
    # 1. Solo informal care: care_demand == 2 AND agent chooses informal care
    df_sim["solo_informal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(informal_care_choices))
    ).astype(int)

    # 2. Formal care: care_demand == 2 AND agent chooses no care
    df_sim["formal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(no_care_choices))
    ).astype(int)

    # 3. Joint informal care: care_demand == 1 AND agent chooses informal care
    df_sim["joint_informal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
        & (df_sim["choice"].isin(informal_care_choices))
    ).astype(int)

    # 4. Other family care: care_demand == 1 AND agent chooses no informal care
    df_sim["other_family_only"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
        & (df_sim["choice"].isin(no_care_choices))
    ).astype(int)

    # Calculate shares for care demand (any care demand)
    # Pooled across education and sister
    care_demand_shares = (
        df_sim.groupby("age", observed=False)["care_demand"]
        .apply(lambda x: (x > 0).mean())
        .reindex(ages, fill_value=0)
    )

    # Calculate care mix shares for all four types - pooled across education and sister
    care_mix_shares = {}
    for care_type in (
        "solo_informal_care",
        "formal_care",
        "joint_informal_care",
        "other_family_only",
    ):
        shares = (
            df_sim.groupby("age", observed=False)[care_type]
            .mean()
            .reindex(ages, fill_value=0)
        )
        care_mix_shares[care_type] = shares

    # ---- 3. Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for care mix (stacked from bottom to top)
    care_colors = {
        "solo_informal_care": "#2E86AB",  # Blue
        "formal_care": "#A23B72",  # Purple
        "joint_informal_care": "#F18F01",  # Orange
        "other_family_only": "#D3D3D3",  # Light grey
    }

    # Get care mix shares
    solo_informal_series = care_mix_shares["solo_informal_care"]
    formal_series = care_mix_shares["formal_care"]
    joint_informal_series = care_mix_shares["joint_informal_care"]
    other_family_series = care_mix_shares["other_family_only"]

    # Plot stacked area for care mix (below the curve)
    # Stack from bottom to top: solo informal, joint informal, formal, other family care
    bottom = 0
    ax.fill_between(
        ages,
        bottom,
        bottom + solo_informal_series,
        color=care_colors["solo_informal_care"],
        alpha=0.6,
        label="Solo informal care",
    )
    bottom += solo_informal_series
    ax.fill_between(
        ages,
        bottom,
        bottom + joint_informal_series,
        color=care_colors["joint_informal_care"],
        alpha=0.6,
        label="Joint informal care",
    )
    bottom += joint_informal_series
    ax.fill_between(
        ages,
        bottom,
        bottom + formal_series,
        color=care_colors["formal_care"],
        alpha=0.6,
        label="Formal care",
    )
    bottom += formal_series
    ax.fill_between(
        ages,
        bottom,
        bottom + other_family_series,
        color=care_colors["other_family_only"],
        alpha=0.6,
        label="Other family care",
    )

    # Plot care demand curve (on top)
    ax.plot(
        ages,
        care_demand_shares,
        color="black",
        linewidth=2,
        label="Care demand",
    )

    # Cosmetics
    pad = 1
    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Share", fontsize=16)

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlim(age_min - pad, age_max + pad)
    ax.set_ylim(0, None)  # Let y-axis adjust automatically
    # ax.set_title("Care Demand by Age (Pooled)")

    # Get handles and labels, then reorder to show from bottom to top
    # Legend order: Care demand at top, then care types from top to bottom
    handles, labels = ax.get_legend_handles_labels()
    # Separate care demand from care types
    care_demand_idx = labels.index("Care demand")
    care_handles = [h for i, h in enumerate(handles) if i != care_demand_idx]
    care_labels = [label for i, label in enumerate(labels) if i != care_demand_idx]
    # Reverse care types so legend shows from bottom to top
    # (other family care at top)
    care_handles_reversed = care_handles[::-1]
    care_labels_reversed = care_labels[::-1]
    # Combine: care demand first, then reversed care types
    final_handles = [handles[care_demand_idx]] + care_handles_reversed
    final_labels = [labels[care_demand_idx]] + care_labels_reversed
    ax.legend(final_handles, final_labels, loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


def plot_simulated_care_demand_by_age_2_by_2_combined(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the yearly share with care_demand > 0 in a 2x2 grid with combined categories.

    Combines solo and joint informal care into "Informal care" (green).
    Other family care is "Other informal care" (yellow).
    Stacking order from bottom to top: Informal care, Other informal care, Formal care.
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

    # ---- 2. Calculate care type indicators
    informal_care_choices = np.asarray(INFORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Calculate individual categories
    df_sim["solo_informal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(informal_care_choices))
    ).astype(int)

    df_sim["formal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(no_care_choices))
    ).astype(int)

    df_sim["joint_informal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
        & (df_sim["choice"].isin(informal_care_choices))
    ).astype(int)

    df_sim["other_family_only"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
        & (df_sim["choice"].isin(no_care_choices))
    ).astype(int)

    # Combine solo and joint into informal care
    df_sim["informal_care"] = (
        df_sim["solo_informal_care"] + df_sim["joint_informal_care"]
    )

    # Calculate shares for care demand
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

    # Calculate care mix shares for combined categories
    care_mix_shares = {}
    for care_type in ("informal_care", "other_family_only", "formal_care"):
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

    subplot_map = {
        (0, 0): 0,  # Low edu, No sister
        (0, 1): 1,  # Low edu, Has sister
        (1, 0): 2,  # High edu, No sister
        (1, 1): 3,  # High edu, Has sister
    }

    edu_labels = {0: "Low education", 1: "High education"}
    sister_labels = {0: "No sister", 1: "Has sister"}

    # Colors for care mix (stacked from bottom to top)
    care_colors = {
        "informal_care": "#2E86AB",  # Greenish/turquoise
        # (original solo informal care color)
        "other_family_only": "#F9A825",  # Yellow
        "formal_care": "#A23B72",  # Purple
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
            other_family_series = care_mix_shares["other_family_only"].xs(
                (edu, has_sister), level=("education", "has_sister")
            )
            formal_series = care_mix_shares["formal_care"].xs(
                (edu, has_sister), level=("education", "has_sister")
            )

            # Plot stacked area for care mix
            # Stack from bottom to top: informal care, other informal care, formal care
            bottom = 0
            ax.fill_between(
                ages,
                bottom,
                bottom + informal_series,
                color=care_colors["informal_care"],
                alpha=0.6,
                label="Informal care",
            )
            bottom += informal_series
            ax.fill_between(
                ages,
                bottom,
                bottom + other_family_series,
                color=care_colors["other_family_only"],
                alpha=0.6,
                label="Other informal care",
            )
            bottom += other_family_series
            ax.fill_between(
                ages,
                bottom,
                bottom + formal_series,
                color=care_colors["formal_care"],
                alpha=0.6,
                label="Formal care",
            )

            # Plot care demand curve (on top)
            ax.plot(
                ages,
                care_demand_series,
                color="black",
                linewidth=2,
                label="Care demand",
            )

            # Cosmetics
            pad = 1
            ax.set_xlabel("Age")
            ax.set_ylabel("Share")
            ax.set_xlim(age_min - pad, 75 + pad)  # Cut x-axis at 75
            ax.set_ylim(0, None)
            ax.set_title(f"{edu_labels[edu]}, {sister_labels[has_sister]}")

            # Get handles and labels, then reorder to show from bottom to top
            handles, labels = ax.get_legend_handles_labels()
            care_demand_idx = labels.index("Care demand")
            care_handles = [h for i, h in enumerate(handles) if i != care_demand_idx]
            care_labels = [
                label for i, label in enumerate(labels) if i != care_demand_idx
            ]
            # Reverse care types so legend shows from bottom to top
            care_handles_reversed = care_handles[::-1]
            care_labels_reversed = care_labels[::-1]
            # Combine: care demand first, then reversed care types
            final_handles = [handles[care_demand_idx]] + care_handles_reversed
            final_labels = [labels[care_demand_idx]] + care_labels_reversed
            ax.legend(final_handles, final_labels, loc="upper left", fontsize=8)

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


def plot_simulated_care_demand_by_age_pooled_combined(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the yearly share with care_demand > 0 pooled with combined categories.

    Combines solo and joint informal care into "Informal care" (green).
    Other family care is "Other informal care" (yellow).
    Stacking order from bottom to top: Informal care, Other informal care, Formal care.
    """
    # ---- 1. Setup
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = 75

    ages = np.arange(age_min, age_max + 1)

    df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()
    if "sex" in df_sim.columns:
        df_sim = df_sim.loc[df_sim["sex"] == SEX].copy()

    # ---- 2. Calculate care type indicators
    informal_care_choices = np.asarray(INFORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Calculate individual categories
    df_sim["solo_informal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(informal_care_choices))
    ).astype(int)

    df_sim["formal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_sim["choice"].isin(no_care_choices))
    ).astype(int)

    df_sim["joint_informal_care"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
        & (df_sim["choice"].isin(informal_care_choices))
    ).astype(int)

    df_sim["other_family_only"] = (
        (df_sim["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
        & (df_sim["choice"].isin(no_care_choices))
    ).astype(int)

    # Combine solo and joint into informal care
    df_sim["informal_care"] = (
        df_sim["solo_informal_care"] + df_sim["joint_informal_care"]
    )

    # Calculate shares for care demand - pooled across education and sister
    care_demand_shares = (
        df_sim.groupby("age", observed=False)["care_demand"]
        .apply(lambda x: (x > 0).mean())
        .reindex(ages, fill_value=0)
    )

    # Calculate care mix shares for combined categories - pooled
    care_mix_shares = {}
    for care_type in ("informal_care", "other_family_only", "formal_care"):
        shares = (
            df_sim.groupby("age", observed=False)[care_type]
            .mean()
            .reindex(ages, fill_value=0)
        )
        care_mix_shares[care_type] = shares

    # ---- 3. Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for care mix (stacked from bottom to top)
    care_colors = {
        "informal_care": "#2E86AB",  # Greenish/turquoise
        # (original solo informal care color)
        "other_family_only": "#F9A825",  # Yellow
        "formal_care": "#A23B72",  # Purple
    }

    # Get care mix shares
    informal_series = care_mix_shares["informal_care"]
    other_family_series = care_mix_shares["other_family_only"]
    formal_series = care_mix_shares["formal_care"]

    # Plot stacked area for care mix
    # Stack from bottom to top: informal care, other informal care, formal care
    bottom = 0
    ax.fill_between(
        ages,
        bottom,
        bottom + informal_series,
        color=care_colors["informal_care"],
        alpha=0.6,
        label="Informal care",
    )
    bottom += informal_series
    ax.fill_between(
        ages,
        bottom,
        bottom + other_family_series,
        color=care_colors["other_family_only"],
        alpha=0.6,
        label="Other informal care",
    )
    bottom += other_family_series
    ax.fill_between(
        ages,
        bottom,
        bottom + formal_series,
        color=care_colors["formal_care"],
        alpha=0.6,
        label="Formal care",
    )

    # Plot care demand curve (on top)
    ax.plot(
        ages,
        care_demand_shares,
        color="black",
        linewidth=2,
        label="Care demand",
    )

    # Cosmetics
    pad = 1
    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Share", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlim(age_min - pad, age_max + pad)
    ax.set_ylim(0, None)

    # Get handles and labels, then reorder to show from bottom to top
    handles, labels = ax.get_legend_handles_labels()
    care_demand_idx = labels.index("Care demand")
    care_handles = [h for i, h in enumerate(handles) if i != care_demand_idx]
    care_labels = [label for i, label in enumerate(labels) if i != care_demand_idx]
    # Reverse care types so legend shows from bottom to top
    care_handles_reversed = care_handles[::-1]
    care_labels_reversed = care_labels[::-1]
    # Combine: care demand first, then reversed care types
    final_handles = [handles[care_demand_idx]] + care_handles_reversed
    final_labels = [labels[care_demand_idx]] + care_labels_reversed
    ax.legend(final_handles, final_labels, loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


def plot_mother_health_shares_by_age(
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the share of mother health states (good, medium, bad, dead) by age.

    Parameters
    ----------
    df_sim : pd.DataFrame
        Simulated data with columns: mother_health, mother_age
        (or age + mother_age_diff)
    specs : dict
        Model specifications
    age_min : int, optional
        Minimum mother age to plot. If None, uses 50
    age_max : int, optional
        Maximum mother age to plot. If None, uses 100
    path_to_save_plot : str | Path | None, optional
        If provided, the figure is written to this file (PNG, 300 dpi).
    """
    # ---- 1. Setup
    if age_min is None:
        age_min = 50
    if age_max is None:
        age_max = 100

    # Compute mother_age if not already present
    if "mother_age" not in df_sim.columns:
        df_sim["mother_age"] = (
            df_sim["age"].to_numpy()
            + specs["mother_age_diff"][
                df_sim["has_sister"].to_numpy(), df_sim["education"].to_numpy()
            ]
        )

    # Filter to relevant age range
    df_plot = df_sim[
        (df_sim["mother_age"] >= age_min) & (df_sim["mother_age"] <= age_max)
    ].copy()

    # Create age range
    mother_ages = np.arange(age_min, age_max + 1)

    # ---- 2. Calculate shares by mother age
    # Create health state indicators
    df_plot["health_good"] = (df_plot["mother_health"] == PARENT_GOOD_HEALTH).astype(
        int
    )
    df_plot["health_medium"] = (
        df_plot["mother_health"] == PARENT_MEDIUM_HEALTH
    ).astype(int)
    df_plot["health_bad"] = (df_plot["mother_health"] == PARENT_BAD_HEALTH).astype(int)
    df_plot["health_dead"] = (df_plot["mother_health"] == PARENT_DEAD).astype(int)

    # Calculate shares by mother age
    health_shares = {}
    for health_type in ("health_good", "health_medium", "health_bad", "health_dead"):
        shares = (
            df_plot.groupby("mother_age", observed=False)[health_type]
            .mean()
            .reindex(mother_ages, fill_value=0)
        )
        health_shares[health_type] = shares

    # ---- 3. Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for health states
    health_colors = {
        "health_good": "#2E7D32",  # Green
        "health_medium": "#F9A825",  # Yellow/Orange
        "health_bad": "#C62828",  # Red
        "health_dead": "#424242",  # Dark gray
    }

    health_labels = {
        "health_good": "Good",
        "health_medium": "Medium",
        "health_bad": "Bad",
        "health_dead": "Dead",
    }

    # Plot stacked area chart (from bottom to top: good, medium, bad, dead)
    bottom = np.zeros(len(mother_ages))
    for health_type in ("health_good", "health_medium", "health_bad", "health_dead"):
        ax.fill_between(
            mother_ages,
            bottom,
            bottom + health_shares[health_type],
            color=health_colors[health_type],
            alpha=0.7,
            label=health_labels[health_type],
        )
        bottom += health_shares[health_type]

    # Cosmetics
    pad = 1
    ax.set_xlabel("Mother Age")
    ax.set_ylabel("Share")
    ax.set_xlim(age_min - pad, age_max + pad)
    ax.set_ylim(0, 1)
    ax.set_title("Share of Mother Health States by Age")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


def test_care_mix_sums_to_care_demand(df_sim, specs, age_min=None, age_max=None):
    """
    Test that the four care modes sum up to the number of agents facing care demand
    at each given age.

    The four care modes are:
    1. Solo informal care (care_demand == 2, agent chooses informal care)
    2. Formal care (care_demand == 2, agent chooses no care)
    3. Joint informal care (care_demand == 1, agent chooses informal care)
    4. Other family member only (care_demand == 1, agent chooses no informal care)

    This function asserts that the absolute counts of the four care modes sum to
    the number of agents with care demand.

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

    # Create care type indicators for all four scenarios
    df_test["solo_informal_care"] = (
        (df_test["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_test["choice"].isin(informal_care_choices))
    ).astype(int)

    df_test["formal_care"] = (
        (df_test["care_demand"] == CARE_DEMAND_AND_NO_OTHER_SUPPLY)
        & (df_test["choice"].isin(no_care_choices))
    ).astype(int)

    df_test["joint_informal_care"] = (
        (df_test["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
        & (df_test["choice"].isin(informal_care_choices))
    ).astype(int)

    df_test["other_family_only"] = (
        (df_test["care_demand"] == CARE_DEMAND_AND_OTHER_SUPPLY)
        & (df_test["choice"].isin(no_care_choices))
    ).astype(int)

    # Calculate absolute counts by age, education, has_sister
    group_cols = ["age", "education", "has_sister"]
    counts = df_test.groupby(group_cols, observed=False).agg(
        {
            "care_demand": lambda x: (x > 0).sum(),  # Count with care demand
            "solo_informal_care": "sum",
            "formal_care": "sum",
            "joint_informal_care": "sum",
            "other_family_only": "sum",
        }
    )

    # Calculate sum of four care modes
    counts["care_mix_sum"] = (
        counts["solo_informal_care"]
        + counts["formal_care"]
        + counts["joint_informal_care"]
        + counts["other_family_only"]
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
