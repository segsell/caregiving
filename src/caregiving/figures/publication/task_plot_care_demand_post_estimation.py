"""Publication plots: post-estimation care demand by age."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    CARE_DEMAND_INTENSIVE,
    CARE_DEMAND_LIGHT,
    DEAD,
    FORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_CARE,
    SEX,
)

CARE_MIX_TOLERANCE = 1e-10


# ============================================================================
# Tasks
# ============================================================================


@pytask.mark.publication
@pytask.mark.publication_post_estimation
def task_plot_care_demand_by_age_pooled(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "care_demand_by_age_pooled.pdf",
) -> None:
    """Plot care demand by age pooled across all education and sister specifications."""

    specs = pickle.load(path_to_specs.open("rb"))

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    # Create age variable from start_age + period
    df_sim["age"] = df_sim["period"] + specs["start_age"]

    plot_simulated_care_demand_by_age_pooled(
        df_sim=df_sim,
        specs=specs,
        age_min=40,
        age_max=75,
        path_to_save_plot=path_to_save_plot,
    )


@pytask.mark.publication
@pytask.mark.publication_post_estimation
def task_plot_care_demand_by_age_pooled_bar_charts(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "care_demand_by_age_pooled_bar.pdf",
) -> None:
    """Plot care demand by age (pooled) as stacked bar chart in 5-year age bins."""

    specs = pickle.load(path_to_specs.open("rb"))

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    # Create age variable from start_age + period
    df_sim["age"] = df_sim["period"] + specs["start_age"]

    plot_simulated_care_demand_by_age_pooled_bar_charts(
        df_sim=df_sim,
        specs=specs,
        age_min=40,
        age_max=75,
        path_to_save_plot=path_to_save_plot,
    )


# @pytask.mark.publication
# def task_plot_care_demand_by_age_pooled_light_intensive(
#     path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
#     path_to_simulated_data: Path = BLD
#     / "solve_and_simulate"
#     / "simulated_data_estimated_params.pkl",
#     path_to_save_plot: Annotated[Path, Product] = BLD
#     / "figures"
#     / "publication"
#     / "post_estimation"
#     / "care_demand_by_age_pooled_light_intensive.png",
# ) -> None:
#     """Plot light vs intensive care demand by age (pooled).

#     Under each curve, display the care mix across care types.
#     """

#     specs = pickle.load(path_to_specs.open("rb"))

#     df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
#     df_sim["sex"] = SEX

#     # Create age variable from start_age + period
#     df_sim["age"] = df_sim["period"] + specs["start_age"]

#     plot_simulated_care_demand_by_age_pooled_light_intensive(
#         df_sim=df_sim,
#         specs=specs,
#         age_min=40,
#         age_max=75,
#         path_to_save_plot=path_to_save_plot,
#     )


# @pytask.mark.publication
# def task_plot_care_demand_by_age_by_education(
#     path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
#     path_to_simulated_data: Path = BLD
#     / "solve_and_simulate"
#     / "simulated_data_estimated_params.pkl",
#     path_to_save_plot: Annotated[Path, Product] = BLD
#     / "figures"
#     / "publication"
#     / "post_estimation"
#     / "care_demand_by_age_by_education.png",
# ) -> None:
#     """Plot care demand by age in a 1x2 grid by education type (pooled across care demand and caregiving type)."""  # noqa: E501

#     specs = pickle.load(path_to_specs.open("rb"))

#     df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
#     df_sim["sex"] = SEX

#     # Create age variable from start_age + period
#     df_sim["age"] = df_sim["period"] + specs["start_age"]

#     # Test that care mix sums to care demand (only when mother is alive)
#     test_care_mix_sums_to_care_demand(
#         df_sim=df_sim, specs=specs, age_min=40, age_max=80
#     )

#     plot_simulated_care_demand_by_age_by_education(
#         df_sim=df_sim,
#         specs=specs,
#         age_min=40,
#         age_max=80,
#         path_to_save_plot=path_to_save_plot,
#     )


# ============================================================================
# Plotting functions
# ============================================================================


def plot_simulated_care_demand_by_age_pooled(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the yearly share with care_demand > 0.

    Pooled across all education and sister groups.

    Shows all four types of care choices upon positive care demand
    (care_demand in {CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE}):
    1. No care (NO_CARE)
    2. Light informal care (LIGHT_INFORMAL_CARE)
    3. Intensive informal care (INTENSIVE_INFORMAL_CARE)
    4. Formal care (FORMAL_CARE)

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
    light_informal_care_choices = np.asarray(LIGHT_INFORMAL_CARE)
    intensive_informal_care_choices = np.asarray(INTENSIVE_INFORMAL_CARE)
    formal_care_choices = np.asarray(FORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Four types of care choices upon "true" care demand:
    # care_demand in {CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE} AND mother alive.
    positive_demand = df_sim["care_demand"].isin(
        [CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE]
    ) & (df_sim["mother_dead"] == 0)

    df_sim["no_care_choice"] = (
        positive_demand & df_sim["choice"].isin(no_care_choices)
    ).astype(int)
    df_sim["light_informal_care"] = (
        positive_demand & df_sim["choice"].isin(light_informal_care_choices)
    ).astype(int)
    df_sim["intensive_informal_care"] = (
        positive_demand & df_sim["choice"].isin(intensive_informal_care_choices)
    ).astype(int)
    df_sim["formal_care"] = (
        positive_demand & df_sim["choice"].isin(formal_care_choices)
    ).astype(int)

    # Calculate shares for care demand (any positive care demand, mother alive)
    # Pooled across education and sister.
    def _true_care_demand_share(group):
        mask = group["care_demand"].isin([CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE]) & (
            group["mother_dead"] == 0
        )
        return mask.mean()

    care_demand_shares = (
        df_sim.groupby("age", observed=False)
        .apply(_true_care_demand_share)
        .reindex(ages, fill_value=0)
    )

    # Calculate care mix shares for all four types - pooled across education and sister
    care_mix_shares = {}
    for care_type in (
        "no_care_choice",
        "light_informal_care",
        "intensive_informal_care",
        "formal_care",
    ):
        shares = (
            df_sim.groupby("age", observed=False)[care_type]
            .mean()
            .reindex(ages, fill_value=0)
        )
        care_mix_shares[care_type] = shares

    # ---- 3. Create single plot (layout matches task_plot_pre_estimation)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Grayscale shades for care mix segments (stacked from bottom to top)
    # 0 = black, 1 = white; lighter to darker: 0.8, 0.6, 0.4, 0.2
    care_colors = {
        "no_care_choice": "0.8",  # Other sibling provides care (lightest)
        "light_informal_care": "0.6",
        "intensive_informal_care": "0.4",
        "formal_care": "0.2",  # darkest
    }

    # Get care mix shares
    no_care_series = care_mix_shares["no_care_choice"]
    light_informal_series = care_mix_shares["light_informal_care"]
    intensive_informal_series = care_mix_shares["intensive_informal_care"]
    formal_series = care_mix_shares["formal_care"]

    # Plot stacked area for care mix (below the curve)
    # Stack from bottom to top: no care, light informal, intensive informal, formal
    bottom = 0
    ax.fill_between(
        ages,
        bottom,
        bottom + no_care_series,
        color=care_colors["no_care_choice"],
        alpha=1.0,
        label="Other sibling provides care",
    )
    bottom += no_care_series
    ax.fill_between(
        ages,
        bottom,
        bottom + light_informal_series,
        color=care_colors["light_informal_care"],
        alpha=1.0,
        label="Light informal care",
    )
    bottom += light_informal_series
    ax.fill_between(
        ages,
        bottom,
        bottom + intensive_informal_series,
        color=care_colors["intensive_informal_care"],
        alpha=1.0,
        label="Intensive informal care",
    )
    bottom += intensive_informal_series
    ax.fill_between(
        ages,
        bottom,
        bottom + formal_series,
        color=care_colors["formal_care"],
        alpha=1.0,
        label="Formal care",
    )

    # Solid line for any care demand (no light/intensive separate lines)
    linewidth = 2.0
    ax.plot(
        ages,
        care_demand_shares,
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    # Layout: match pre-estimation (margins, axes, grid, spines)
    age_min_plot, age_max_plot = age_min, age_max
    ax.set_xlim(age_min_plot - 0.5, age_max_plot + 0.5)
    ax.set_ylim(-0.005, 0.2)
    ax.set_yticks(np.arange(0, 0.21, 0.05))

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Share", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8)

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    if path_to_save_plot:
        path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save_plot, dpi=1200, bbox_inches="tight")
    plt.close(fig)


def plot_simulated_care_demand_by_age_pooled_bar_charts(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot care demand by age as stacked bar chart in 5-year age bins.

    Same information as plot_simulated_care_demand_by_age_pooled but with
    bars for [40, 45), [45, 50), etc. Care mode shares stacked in each bar.
    """

    # ---- 1. Setup
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = 75

    df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()
    if "sex" in df_sim.columns:
        df_sim = df_sim.loc[df_sim["sex"] == SEX].copy()

    # Filter to age range
    df_sim = df_sim[(df_sim["age"] >= age_min) & (df_sim["age"] <= age_max)].copy()

    # ---- 2. Create 5-year age bins up to [70, 75) only
    bins = np.arange(age_min, age_max, 5)  # [40, 45, 50, 55, 60, 65, 70]
    bin_labels = [f"{b}-{b+4}" for b in bins]
    df_sim["age_bin"] = pd.cut(
        df_sim["age"],
        bins=np.append(bins, bins[-1] + 5),
        right=False,
        labels=bin_labels,
    )

    # ---- 3. Calculate care type indicators
    light_informal_care_choices = np.asarray(LIGHT_INFORMAL_CARE)
    intensive_informal_care_choices = np.asarray(INTENSIVE_INFORMAL_CARE)
    formal_care_choices = np.asarray(FORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    positive_demand = df_sim["care_demand"].isin(
        [CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE]
    ) & (df_sim["mother_dead"] == 0)

    df_sim["no_care_choice"] = (
        positive_demand & df_sim["choice"].isin(no_care_choices)
    ).astype(int)
    df_sim["light_informal_care"] = (
        positive_demand & df_sim["choice"].isin(light_informal_care_choices)
    ).astype(int)
    df_sim["intensive_informal_care"] = (
        positive_demand & df_sim["choice"].isin(intensive_informal_care_choices)
    ).astype(int)
    df_sim["formal_care"] = (
        positive_demand & df_sim["choice"].isin(formal_care_choices)
    ).astype(int)

    # ---- 4. Mean shares by age bin (same order as bins)
    care_mix_by_bin = {}
    for care_type in (
        "no_care_choice",
        "light_informal_care",
        "intensive_informal_care",
        "formal_care",
    ):
        means = (
            df_sim.groupby("age_bin", observed=False)[care_type]
            .mean()
            .reindex(bin_labels, fill_value=0)
        )
        care_mix_by_bin[care_type] = means.values

    # ---- 5. Create stacked bar chart (grayscale)
    fig, ax = plt.subplots(figsize=(10, 8))

    care_colors = {
        "no_care_choice": "0.8",
        "light_informal_care": "0.6",
        "intensive_informal_care": "0.4",
        "formal_care": "0.2",
    }

    x = np.arange(len(bin_labels))
    width = 0.75

    bottom = np.zeros(len(bin_labels))
    for care_type, label in (
        ("no_care_choice", "Other sibling provides care"),
        ("light_informal_care", "Light informal care"),
        ("intensive_informal_care", "Intensive informal care"),
        ("formal_care", "Formal care"),
    ):
        heights = care_mix_by_bin[care_type]
        ax.bar(
            x,
            heights,
            width,
            bottom=bottom,
            color=care_colors[care_type],
            alpha=1.0,
            label=label,
        )
        bottom += heights

    # Layout: match pooled area plot
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=14, rotation=25)
    ax.set_xlim(-0.5, len(bin_labels) - 0.5)
    ax.set_ylim(-0.005, 0.2)
    ax.set_yticks(np.arange(0, 0.21, 0.05))

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Share", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8)

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    if path_to_save_plot:
        path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save_plot, dpi=1200, bbox_inches="tight")
    plt.close(fig)


def plot_simulated_care_demand_by_age_pooled_light_intensive(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot two pooled panels:

    - Left: share with light care demand
      (care_demand == CARE_DEMAND_LIGHT) by age.
    - Right: share with intensive care demand
      (care_demand == CARE_DEMAND_INTENSIVE) by age.

    Under each demand curve, stack (shares in total population):
    - No care (NO_CARE)
    - Light informal care (LIGHT_INFORMAL_CARE)
    - Intensive informal care (INTENSIVE_INFORMAL_CARE)
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

    # ---- 2. Care type indicators conditional on positive demand type
    light_informal_care_choices = np.asarray(LIGHT_INFORMAL_CARE)
    intensive_informal_care_choices = np.asarray(INTENSIVE_INFORMAL_CARE)
    formal_care_choices = np.asarray(FORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Light and intensive demand indicators (only when mother is alive)
    light_demand = (df_sim["care_demand"] == CARE_DEMAND_LIGHT) & (
        df_sim["mother_dead"] == 0
    )
    intensive_demand = (df_sim["care_demand"] == CARE_DEMAND_INTENSIVE) & (
        df_sim["mother_dead"] == 0
    )

    # For light demand panel
    df_sim["no_care_light"] = (
        light_demand & df_sim["choice"].isin(no_care_choices)
    ).astype(int)
    df_sim["light_informal_given_light"] = (
        light_demand & df_sim["choice"].isin(light_informal_care_choices)
    ).astype(int)
    df_sim["intensive_informal_given_light"] = (
        light_demand & df_sim["choice"].isin(intensive_informal_care_choices)
    ).astype(int)
    df_sim["formal_given_light"] = (
        light_demand & df_sim["choice"].isin(formal_care_choices)
    ).astype(int)

    # For intensive demand panel
    df_sim["no_care_intensive"] = (
        intensive_demand & df_sim["choice"].isin(no_care_choices)
    ).astype(int)
    df_sim["light_informal_given_intensive"] = (
        intensive_demand & df_sim["choice"].isin(light_informal_care_choices)
    ).astype(int)
    df_sim["intensive_informal_given_intensive"] = (
        intensive_demand & df_sim["choice"].isin(intensive_informal_care_choices)
    ).astype(int)
    df_sim["formal_given_intensive"] = (
        intensive_demand & df_sim["choice"].isin(formal_care_choices)
    ).astype(int)

    # ---- 3. Demand shares (in total population, mother alive)
    def _light_demand_share(group):
        mask = (group["care_demand"] == CARE_DEMAND_LIGHT) & (group["mother_dead"] == 0)
        return mask.mean()

    def _intensive_demand_share(group):
        mask = (group["care_demand"] == CARE_DEMAND_INTENSIVE) & (
            group["mother_dead"] == 0
        )
        return mask.mean()

    light_demand_shares = (
        df_sim.groupby("age", observed=False)
        .apply(_light_demand_share)
        .reindex(ages, fill_value=0)
    )
    intensive_demand_shares = (
        df_sim.groupby("age", observed=False)
        .apply(_intensive_demand_share)
        .reindex(ages, fill_value=0)
    )

    # ---- 4. Care mix shares (each panel, in total population)
    def _mean_by_age(col):
        series = (
            df_sim.groupby("age", observed=False)[col]
            .mean()
            .reindex(ages, fill_value=0)
        )
        return series

    mixes = {
        "no_care_light": _mean_by_age("no_care_light"),
        "light_informal_given_light": _mean_by_age("light_informal_given_light"),
        "intensive_informal_given_light": _mean_by_age(
            "intensive_informal_given_light"
        ),
        "formal_given_light": _mean_by_age("formal_given_light"),
        "no_care_intensive": _mean_by_age("no_care_intensive"),
        "light_informal_given_intensive": _mean_by_age(
            "light_informal_given_intensive"
        ),
        "intensive_informal_given_intensive": _mean_by_age(
            "intensive_informal_given_intensive"
        ),
        "formal_given_intensive": _mean_by_age("formal_given_intensive"),
    }

    # ---- 5. Plot 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    care_colors = {
        "no_care": "#D3D3D3",  # Light grey
        "light": "#2E86AB",  # Blue
        "intensive": "#F18F01",  # Orange
        "formal": "#A23B72",  # Purple
    }

    panels = (
        {
            "ax": axes[0],
            "title": "Light care demand",
            "demand": light_demand_shares,
            "no_care": mixes["no_care_light"],
            "light": mixes["light_informal_given_light"],
            "intensive": mixes["intensive_informal_given_light"],
            "formal": mixes["formal_given_light"],
        },
        {
            "ax": axes[1],
            "title": "Intensive care demand",
            "demand": intensive_demand_shares,
            "no_care": mixes["no_care_intensive"],
            "light": mixes["light_informal_given_intensive"],
            "intensive": mixes["intensive_informal_given_intensive"],
            "formal": mixes["formal_given_intensive"],
        },
    )

    for panel in panels:
        ax = panel["ax"]

        # Stacked bands: no care, light informal, intensive informal, formal
        bottom = 0
        no_care_series = panel["no_care"]
        light_series = panel["light"]
        intensive_series = panel["intensive"]
        formal_series = panel["formal"]

        ax.fill_between(
            ages,
            bottom,
            bottom + no_care_series,
            color=care_colors["no_care"],
            alpha=0.6,
            label="Other sibling provides care",
        )
        bottom += no_care_series

        ax.fill_between(
            ages,
            bottom,
            bottom + light_series,
            color=care_colors["light"],
            alpha=0.6,
            label="Light informal care",
        )
        bottom += light_series

        ax.fill_between(
            ages,
            bottom,
            bottom + intensive_series,
            color=care_colors["intensive"],
            alpha=0.6,
            label="Intensive informal care",
        )
        bottom += intensive_series

        ax.fill_between(
            ages,
            bottom,
            bottom + formal_series,
            color=care_colors["formal"],
            alpha=0.6,
            label="Formal care",
        )

        # Demand curve
        ax.plot(
            ages,
            panel["demand"],
            color="black",
            linewidth=2,
            label="Care demand",
        )

        pad = 1
        ax.set_xlim(age_min - pad, age_max + pad)
        ax.set_xlabel("Age", fontsize=14)
        ax.set_title(panel["title"], fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Build legend: demand first, then stacked components (top-to-bottom order)
        handles, labels = ax.get_legend_handles_labels()
        demand_idx = labels.index("Care demand")
        comp_handles = [h for i, h in enumerate(handles) if i != demand_idx]
        comp_labels = [lab for i, lab in enumerate(labels) if i != demand_idx]
        comp_handles_rev = comp_handles[::-1]
        comp_labels_rev = comp_labels[::-1]
        final_handles = [handles[demand_idx]] + comp_handles_rev
        final_labels = [labels[demand_idx]] + comp_labels_rev
        ax.legend(final_handles, final_labels, loc="upper left", fontsize=10)

    axes[0].set_ylabel("Share", fontsize=14)

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


def plot_simulated_care_demand_by_age_by_education(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the yearly share with care_demand > 0 in a 1x2 grid by education type.

    Left panel: Low education (education == 0)
    Right panel: High education (education == 1)

    Pooled across care demand type and caregiving type.

    Shows care choices upon positive care demand
    (care_demand in {CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE}):
    1. No care:
       has_care_demand(care_demand) AND agent chooses NO_CARE.
    2. Light informal care:
       has_care_demand(care_demand) AND agent chooses LIGHT_INFORMAL_CARE.
    3. Intensive informal care:
       has_care_demand(care_demand) AND agent chooses INTENSIVE_INFORMAL_CARE.
    4. Formal care:
       has_care_demand(care_demand) AND agent chooses FORMAL_CARE.

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
    # Convert JAX arrays to numpy arrays for pandas compatibility
    light_informal_care_choices = np.asarray(LIGHT_INFORMAL_CARE)
    intensive_informal_care_choices = np.asarray(INTENSIVE_INFORMAL_CARE)
    formal_care_choices = np.asarray(FORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Four types of care choices upon "true" care demand:
    # care_demand in {CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE} AND mother alive.
    positive_demand = df_sim["care_demand"].isin(
        [CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE]
    ) & (df_sim["mother_dead"] == 0)

    # 1. No care
    df_sim["no_care_choice"] = (
        positive_demand & df_sim["choice"].isin(no_care_choices)
    ).astype(int)

    # 2. Light informal care
    df_sim["light_informal_care"] = (
        positive_demand & df_sim["choice"].isin(light_informal_care_choices)
    ).astype(int)

    # 3. Intensive informal care
    df_sim["intensive_informal_care"] = (
        positive_demand & df_sim["choice"].isin(intensive_informal_care_choices)
    ).astype(int)

    # 4. Formal care
    df_sim["formal_care"] = (
        positive_demand & df_sim["choice"].isin(formal_care_choices)
    ).astype(int)

    # Calculate shares for care demand (any positive care demand, mother alive)
    def _true_care_demand_share(group):
        mask = group["care_demand"].isin([CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE]) & (
            group["mother_dead"] == 0
        )
        return mask.mean()

    care_demand_shares = (
        df_sim.groupby(["age", "education"], observed=False)
        .apply(_true_care_demand_share)
        .reindex(
            pd.MultiIndex.from_product([ages, [0, 1]], names=["age", "education"]),
            fill_value=0,
        )
    )

    # Calculate care mix shares for all four types
    care_mix_shares = {}
    for care_type in (
        "no_care_choice",
        "light_informal_care",
        "intensive_informal_care",
        "formal_care",
    ):
        shares = (
            df_sim.groupby(["age", "education"], observed=False)[care_type]
            .mean()
            .reindex(
                pd.MultiIndex.from_product([ages, [0, 1]], names=["age", "education"]),
                fill_value=0,
            )
        )
        care_mix_shares[care_type] = shares

    # ---- 3. Create 1x2 subplot grid
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    # Labels for titles
    edu_labels = {0: "Low education", 1: "High education"}

    # Colors for care mix (stacked from bottom to top)
    care_colors = {
        "no_care_choice": "#D3D3D3",  # Light grey
        "light_informal_care": "#2E86AB",  # Blue
        "intensive_informal_care": "#F18F01",  # Orange
        "formal_care": "#A23B72",  # Purple
    }

    # ---- 4. Plot each education level
    for edu in (0, 1):
        ax = axes[edu]

        # Get care demand share
        care_demand_series = care_demand_shares.xs(edu, level="education")

        # Get care mix shares
        no_care_series = care_mix_shares["no_care_choice"].xs(edu, level="education")
        light_informal_series = care_mix_shares["light_informal_care"].xs(
            edu, level="education"
        )
        intensive_informal_series = care_mix_shares["intensive_informal_care"].xs(
            edu, level="education"
        )
        formal_series = care_mix_shares["formal_care"].xs(edu, level="education")

        # Plot stacked area for care mix (below the curve)
        # Stack from bottom to top:
        #   no care, light informal, intensive informal, formal care
        bottom = 0
        ax.fill_between(
            ages,
            bottom,
            bottom + no_care_series,
            color=care_colors["no_care_choice"],
            alpha=0.6,
            label="Other sibling provides care",
        )
        bottom += no_care_series
        ax.fill_between(
            ages,
            bottom,
            bottom + light_informal_series,
            color=care_colors["light_informal_care"],
            alpha=0.6,
            label="Light informal care",
        )
        bottom += light_informal_series
        ax.fill_between(
            ages,
            bottom,
            bottom + intensive_informal_series,
            color=care_colors["intensive_informal_care"],
            alpha=0.6,
            label="Intensive informal care",
        )
        bottom += intensive_informal_series
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
        ax.set_xlabel("Age", fontsize=14)
        if edu == 0:
            ax.set_ylabel("Share", fontsize=14)
        ax.set_xlim(age_min - pad, 75 + pad)  # Cut x-axis at 75
        ax.set_ylim(0, None)  # Let y-axis adjust automatically
        ax.set_title(edu_labels[edu], fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Get handles and labels, then reorder to show from bottom to top
        # Legend order: Care demand at top, then care types from top to bottom
        handles, labels = ax.get_legend_handles_labels()
        # Separate care demand from care types
        care_demand_idx = labels.index("Care demand")
        care_handles = [h for i, h in enumerate(handles) if i != care_demand_idx]
        care_labels = [label for i, label in enumerate(labels) if i != care_demand_idx]
        # Reverse care types so legend shows from bottom to top
        care_handles_reversed = care_handles[::-1]
        care_labels_reversed = care_labels[::-1]
        # Combine: care demand first, then reversed care types
        final_handles = [handles[care_demand_idx]] + care_handles_reversed
        final_labels = [labels[care_demand_idx]] + care_labels_reversed
        ax.legend(final_handles, final_labels, loc="upper left", fontsize=10)

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


def test_care_mix_sums_to_care_demand(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None
):
    """
    Test that the four care modes sum to the number of agents with care demand.

    The four care modes (conditional on "true" care demand, i.e.
    care_demand in {CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE} AND mother_dead == 0)
    are defined purely by the agent's choice:
    1. No care: choice in NO_CARE
    2. Light informal care: choice in LIGHT_INFORMAL_CARE
    3. Intensive informal care: choice in INTENSIVE_INFORMAL_CARE
    4. Formal care: choice in FORMAL_CARE

    This function asserts that the absolute counts of the four care modes sum to
    the number of agents with positive care demand.

    Parameters
    ----------
    df_sim : pd.DataFrame
        Simulated data with columns:
        age, education, caregiving_type, care_demand, choice.
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
    light_informal_care_choices = np.asarray(LIGHT_INFORMAL_CARE)
    intensive_informal_care_choices = np.asarray(INTENSIVE_INFORMAL_CARE)
    formal_care_choices = np.asarray(FORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # "True" care demand:
    # care_demand in {CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE} AND mother_dead == 0.
    positive_demand = df_test["care_demand"].isin(
        [CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE]
    ) & (df_test["mother_dead"] == 0)

    df_test["true_positive_demand"] = positive_demand.astype(int)

    df_test["no_care_choice"] = (
        positive_demand & df_test["choice"].isin(no_care_choices)
    ).astype(int)
    df_test["light_informal_care"] = (
        positive_demand & df_test["choice"].isin(light_informal_care_choices)
    ).astype(int)
    df_test["intensive_informal_care"] = (
        positive_demand & df_test["choice"].isin(intensive_informal_care_choices)
    ).astype(int)
    df_test["formal_care"] = (
        positive_demand & df_test["choice"].isin(formal_care_choices)
    ).astype(int)

    # Debug: Check for uncategorized agents with "true" positive care demand
    care_demand_1_mask = positive_demand
    categorized_mask = (
        df_test["no_care_choice"]
        + df_test["light_informal_care"]
        + df_test["intensive_informal_care"]
        + df_test["formal_care"]
    ) > 0
    uncategorized = df_test[care_demand_1_mask & ~categorized_mask]
    if len(uncategorized) > 0:
        print(
            f"\nWARNING: {len(uncategorized)} agents with positive care demand "
            f"are not categorized!"
        )
        print(
            "Sample uncategorized choices:",
            uncategorized["choice"].value_counts().head(10),
        )
        print(
            "Sample uncategorized caregiving_type:",
            uncategorized["caregiving_type"].value_counts(),
        )

    # Calculate absolute counts by age, education, caregiving_type
    group_cols = ["age", "education", "caregiving_type"]
    counts = df_test.groupby(group_cols, observed=False).agg(
        {
            # Count of agents with "true" positive care demand in this group.
            "true_positive_demand": "sum",
            "no_care_choice": "sum",
            "light_informal_care": "sum",
            "intensive_informal_care": "sum",
            "formal_care": "sum",
        }
    )

    # Calculate sum of four care modes
    counts["care_mix_sum"] = (
        counts["no_care_choice"]
        + counts["light_informal_care"]
        + counts["intensive_informal_care"]
        + counts["formal_care"]
    )

    # Calculate differences
    counts["absolute_diff"] = np.abs(
        counts["true_positive_demand"] - counts["care_mix_sum"]
    )
    counts["relative_diff"] = np.where(
        counts["true_positive_demand"] > 0,
        counts["absolute_diff"] / counts["true_positive_demand"],
        0,
    )

    # Find maximum differences
    max_absolute_diff = counts["absolute_diff"].max()
    max_relative_diff = counts["relative_diff"].max()

    # Assert that the test passed
    tolerance = CARE_MIX_TOLERANCE
    assert max_absolute_diff < tolerance, (
        f"Care mix does not sum to care demand. "
        f"Max absolute difference: {max_absolute_diff}, "
        f"Max relative difference: {max_relative_diff}"
    )
