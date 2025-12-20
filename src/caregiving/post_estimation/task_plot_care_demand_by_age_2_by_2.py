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
import dcegm
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.shared import (
    DEAD,
    FORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_CARE,
    PARENT_BAD_HEALTH,
    PARENT_DEAD,
    PARENT_GOOD_HEALTH,
    PARENT_MEDIUM_HEALTH,
    SEX,
)

CARE_MIX_TOLERANCE = 1e-10


@pytask.mark.baseline_model
@pytask.mark.post_estimation
@pytask.mark.care_demand_post_estimation
def task_plot_care_demand_by_age_2_by_2(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_solution_model: Path = BLD / "model" / "model.pkl",
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
    """Plot care demand by age in a 2x2 grid (education × caregiving_type)."""

    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_solution_model,
    )

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    # Create age variable from start_age + period
    df_sim["age"] = df_sim["period"] + specs["start_age"]

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
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_solution_model: Path = BLD / "model" / "model.pkl",
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

    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_solution_model,
    )

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


@pytask.mark.baseline_model
@pytask.mark.post_estimation
@pytask.mark.care_demand_post_estimation
def task_plot_care_demand_by_age_pooled_light_intensive(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_solution_model: Path = BLD / "model" / "model.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "care_demand_by_age_pooled_light_intensive.png",
) -> None:
    """Plot light vs intensive care demand by age (pooled), with care mix under curves."""

    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_solution_model,
    )

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    # Create age variable from start_age + period
    df_sim["age"] = df_sim["period"] + specs["start_age"]

    plot_simulated_care_demand_by_age_pooled_light_intensive(
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
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_solution_model: Path = BLD / "model" / "model.pkl",
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

    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_solution_model,
    )

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    # Create age variable from start_age + period
    df_sim["age"] = df_sim["period"] + specs["start_age"]

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
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_solution_model: Path = BLD / "model" / "model.pkl",
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

    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_solution_model,
    )

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    # Create age variable from start_age + period
    df_sim["age"] = df_sim["period"] + specs["start_age"]

    plot_simulated_care_demand_by_age_pooled_combined(
        df_sim=df_sim,
        specs=specs,
        age_min=40,
        age_max=75,
        path_to_save_plot=path_to_save_plot,
    )


@pytask.mark.skip(reason="mother health no longer in state space")
@pytask.mark.post_estimation
@pytask.mark.care_demand_post_estimation
def task_plot_mother_health_shares_by_age(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_solution_model: Path = BLD / "model" / "model.pkl",
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

    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_solution_model,
    )

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    # Create age variable from start_age + period
    df_sim["age"] = df_sim["period"] + specs["start_age"]

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
    • caregiving_type (0 / 1)       → columns

    Shows care choices upon positive care demand (care_demand in {1, 2}):
    1. No care:
       care_demand > 0 AND agent chooses NO_CARE.
    2. Light informal care:
       care_demand > 0 AND agent chooses LIGHT_INFORMAL_CARE.
    3. Intensive informal care:
       care_demand > 0 AND agent chooses INTENSIVE_INFORMAL_CARE.
    4. Formal care:
       care_demand > 0 AND agent chooses FORMAL_CARE.

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

    # ================================================================================
    # # Drop cases where care_demand > 0 and mother_dead == 1
    # df_sim = df_sim.loc[
    #     ~((df_sim["care_demand"] > 0) & (df_sim["mother_dead"] == 1))
    # ].copy()
    # ================================================================================

    # ---- 2. Calculate care type indicators for all four scenarios
    # Convert JAX arrays to numpy arrays for pandas compatibility
    light_informal_care_choices = np.asarray(LIGHT_INFORMAL_CARE)
    intensive_informal_care_choices = np.asarray(INTENSIVE_INFORMAL_CARE)
    formal_care_choices = np.asarray(FORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    # Four types of care choices upon positive care demand (care_demand in {1, 2}).
    positive_demand = df_sim["care_demand"] > 0

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

    # Calculate shares for care demand (any positive care demand)
    care_demand_shares = (
        df_sim.groupby(["age", "education", "caregiving_type"], observed=False)[
            "care_demand"
        ]
        .apply(lambda x: (x > 0).mean())
        .reindex(
            pd.MultiIndex.from_product(
                [ages, [0, 1], [0, 1]], names=["age", "education", "caregiving_type"]
            ),
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
            df_sim.groupby(["age", "education", "caregiving_type"], observed=False)[
                care_type
            ]
            .mean()
            .reindex(
                pd.MultiIndex.from_product(
                    [ages, [0, 1], [0, 1]],
                    names=["age", "education", "caregiving_type"],
                ),
                fill_value=0,
            )
        )
        care_mix_shares[care_type] = shares

    # ---- 3. Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Define subplot positions: (education, caregiving_type) -> index
    # Top left (0): Low edu (0), Other provides informal care (0)
    # Top right (1): Low edu (0), Agent provides informal care (1)
    # Bottom left (2): High edu (1), Other provides informal care (0)
    # Bottom right (3): High edu (1), Agent provides informal care (1)
    subplot_map = {
        (0, 0): 0,  # Low edu, Other provides informal care
        (0, 1): 1,  # Low edu, Agent provides informal care
        (1, 0): 2,  # High edu, Other provides informal care
        (1, 1): 3,  # High edu, Agent provides informal care
    }

    # Labels for titles
    edu_labels = {0: "Low education", 1: "High education"}
    caregiving_type_labels = {
        0: "Other provides informal care",
        1: "Agent provides informal care",
    }

    # Colors for care mix (stacked from bottom to top)
    care_colors = {
        "no_care_choice": "#D3D3D3",  # Light grey
        "light_informal_care": "#2E86AB",  # Blue
        "intensive_informal_care": "#F18F01",  # Orange
        "formal_care": "#A23B72",  # Purple
    }

    # ---- 4. Plot each combination
    for edu in (0, 1):
        for caregiving_type in (0, 1):
            idx = subplot_map[(edu, caregiving_type)]
            ax = axes[idx]

            # Get care demand share
            care_demand_series = care_demand_shares.xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )

            # Get care mix shares
            no_care_series = care_mix_shares["no_care_choice"].xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )
            light_informal_series = care_mix_shares["light_informal_care"].xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )
            intensive_informal_series = care_mix_shares["intensive_informal_care"].xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )
            formal_series = care_mix_shares["formal_care"].xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )

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
                label="No care",
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
            ax.set_xlabel("Age")
            ax.set_ylabel("Share")
            ax.set_xlim(age_min - pad, 75 + pad)  # Cut x-axis at 75
            ax.set_ylim(0, None)  # Let y-axis adjust automatically
            ax.set_title(
                f"{edu_labels[edu]}, {caregiving_type_labels[caregiving_type]}"
            )

            # Get handles and labels, then reorder to show from bottom to top
            # Legend order: Care demand at top, then care types from top to bottom
            handles, labels = ax.get_legend_handles_labels()
            # Separate care demand from care types
            care_demand_idx = labels.index("Care demand")
            care_handles = [h for i, h in enumerate(handles) if i != care_demand_idx]
            care_labels = [
                label for i, label in enumerate(labels) if i != care_demand_idx
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

    Shows all four types of care choices upon positive care demand
    (care_demand in {1, 2}):
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

    # Four types of care choices upon positive care demand (care_demand in {1, 2}).
    positive_demand = df_sim["care_demand"] > 0

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

    # Calculate shares for care demand (any positive care demand)
    # Pooled across education and sister
    care_demand_shares = (
        df_sim.groupby("age", observed=False)["care_demand"]
        .apply(lambda x: (x > 0).mean())
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

    # ---- 3. Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for care mix (stacked from bottom to top)
    care_colors = {
        "no_care_choice": "#D3D3D3",  # Light grey
        "light_informal_care": "#2E86AB",  # Blue
        "intensive_informal_care": "#F18F01",  # Orange
        "formal_care": "#A23B72",  # Purple
    }

    # Get care mix shares
    no_care_series = care_mix_shares["no_care_choice"]
    light_informal_series = care_mix_shares["light_informal_care"]
    intensive_informal_series = care_mix_shares["intensive_informal_care"]
    formal_series = care_mix_shares["formal_care"]

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
        label="No care",
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


def plot_simulated_care_demand_by_age_pooled_light_intensive(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot two pooled panels:

    - Left: share with light care demand (care_demand == 1) by age.
    - Right: share with intensive care demand (care_demand == 2) by age.

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

    # Light and intensive demand indicators
    light_demand = df_sim["care_demand"] == 1
    intensive_demand = df_sim["care_demand"] == 2

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

    # ---- 3. Demand shares (in total population)
    light_demand_shares = (
        df_sim.groupby("age", observed=False)["care_demand"]
        .apply(lambda x: (x == 1).mean())
        .reindex(ages, fill_value=0)
    )
    intensive_demand_shares = (
        df_sim.groupby("age", observed=False)["care_demand"]
        .apply(lambda x: (x == 2).mean())
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
            label="No care",
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
        ax.grid(True, alpha=0.3)

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


def plot_simulated_care_demand_by_age_2_by_2_combined(  # noqa: PLR0915
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the yearly share with care_demand > 0 in a 2x2 grid with **combined** categories.

    For each (education, caregiving_type) cell we show:

    - Informal care (light + intensive)
    - No care
    - Formal care

    stacked underneath the care_demand curve.
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
    light_informal_care_choices = np.asarray(LIGHT_INFORMAL_CARE)
    intensive_informal_care_choices = np.asarray(INTENSIVE_INFORMAL_CARE)
    formal_care_choices = np.asarray(FORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    positive_demand = df_sim["care_demand"] > 0

    # Base indicators by choice
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

    # Combined informal band: light + intensive
    df_sim["informal_care"] = (
        df_sim["light_informal_care"] + df_sim["intensive_informal_care"]
    )

    # Calculate shares for care demand
    care_demand_shares = (
        df_sim.groupby(["age", "education", "caregiving_type"], observed=False)[
            "care_demand"
        ]
        .apply(lambda x: (x > 0).mean())
        .reindex(
            pd.MultiIndex.from_product(
                [ages, [0, 1], [0, 1]], names=["age", "education", "caregiving_type"]
            ),
            fill_value=0,
        )
    )

    # Calculate care mix shares for three combined categories
    care_mix_shares = {}
    for care_type in (
        "informal_care",
        "no_care_choice",
        "formal_care",
    ):
        shares = (
            df_sim.groupby(["age", "education", "caregiving_type"], observed=False)[
                care_type
            ]
            .mean()
            .reindex(
                pd.MultiIndex.from_product(
                    [ages, [0, 1], [0, 1]],
                    names=["age", "education", "caregiving_type"],
                ),
                fill_value=0,
            )
        )
        care_mix_shares[care_type] = shares

    # ---- 3. Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    subplot_map = {
        (0, 0): 0,  # Low edu, Other provides informal care
        (0, 1): 1,  # Low edu, Agent provides informal care
        (1, 0): 2,  # High edu, Other provides informal care
        (1, 1): 3,  # High edu, Agent provides informal care
    }

    edu_labels = {0: "Low education", 1: "High education"}
    caregiving_type_labels = {
        0: "Other provides informal care",
        1: "Agent provides informal care",
    }

    # Colors for care mix (stacked from bottom to top)
    care_colors = {
        "informal_care": "#2E86AB",  # Blue (all informal care)
        "no_care_choice": "#D3D3D3",  # Light grey
        "formal_care": "#A23B72",  # Purple
    }

    # ---- 4. Plot each combination
    for edu in (0, 1):
        for caregiving_type in (0, 1):
            idx = subplot_map[(edu, caregiving_type)]
            ax = axes[idx]

            # Get care demand share
            care_demand_series = care_demand_shares.xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )

            # Get care mix shares
            informal_series = care_mix_shares["informal_care"].xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )
            no_care_series = care_mix_shares["no_care_choice"].xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )
            formal_series = care_mix_shares["formal_care"].xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )

            # Plot stacked area for care mix
            # Stack from bottom to top:
            #   informal care, no care, formal care
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
                bottom + no_care_series,
                color=care_colors["no_care_choice"],
                alpha=0.6,
                label="No care",
            )
            bottom += no_care_series
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
            ax.set_title(
                f"{edu_labels[edu]}, {caregiving_type_labels[caregiving_type]}"
            )

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
    Plot the yearly share with care_demand > 0 pooled with **combined** categories.

    We show three stacked types:
    - Informal care (light + intensive)
    - No care
    - Formal care
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
    light_informal_care_choices = np.asarray(LIGHT_INFORMAL_CARE)
    intensive_informal_care_choices = np.asarray(INTENSIVE_INFORMAL_CARE)
    formal_care_choices = np.asarray(FORMAL_CARE)
    no_care_choices = np.asarray(NO_CARE)

    positive_demand = df_sim["care_demand"] > 0

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

    # Combined informal band: light + intensive
    df_sim["informal_care"] = (
        df_sim["light_informal_care"] + df_sim["intensive_informal_care"]
    )

    # Calculate shares for care demand - pooled across education and sister
    care_demand_shares = (
        df_sim.groupby("age", observed=False)["care_demand"]
        .apply(lambda x: (x > 0).mean())
        .reindex(ages, fill_value=0)
    )

    # Calculate care mix shares for three categories - pooled
    care_mix_shares = {}
    for care_type in ("informal_care", "no_care_choice", "formal_care"):
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
        "informal_care": "#2E86AB",  # Blue
        "no_care_choice": "#D3D3D3",  # Light grey
        "formal_care": "#A23B72",  # Purple
    }

    # Get care mix shares
    informal_series = care_mix_shares["informal_care"]
    no_care_series = care_mix_shares["no_care_choice"]
    formal_series = care_mix_shares["formal_care"]

    # Plot stacked area for care mix
    # Stack from bottom to top:
    #   informal care, no care, formal care
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
        bottom + no_care_series,
        color=care_colors["no_care_choice"],
        alpha=0.6,
        label="No care",
    )
    bottom += no_care_series
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
            + specs["mother_age_diff"][df_sim["education"].to_numpy()]
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

    The four care modes (conditional on positive care demand, care_demand in {1, 2})
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

    # Create care type indicators for all four scenarios (care_demand in {1, 2})
    positive_demand = df_test["care_demand"] > 0

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

    # Debug: Check for uncategorized agents with care_demand == 1
    care_demand_1_mask = df_test["care_demand"] > 0
    categorized_mask = (
        df_test["no_care_choice"]
        + df_test["light_informal_care"]
        + df_test["intensive_informal_care"]
        + df_test["formal_care"]
    ) > 0
    uncategorized = df_test[care_demand_1_mask & ~categorized_mask]
    if len(uncategorized) > 0:
        print(
            f"\nWARNING: {len(uncategorized)} agents with care_demand == 1 "
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
            "care_demand": lambda x: (x > 0).sum(),  # Count with care_demand > 0
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
    counts["absolute_diff"] = np.abs(counts["care_demand"] - counts["care_mix_sum"])
    counts["relative_diff"] = np.where(
        counts["care_demand"] > 0,
        counts["absolute_diff"] / counts["care_demand"],
        0,
    )

    # Find maximum differences
    max_absolute_diff = counts["absolute_diff"].max()
    max_relative_diff = counts["relative_diff"].max()

    # Debug: Print problematic rows if test fails
    if max_absolute_diff >= CARE_MIX_TOLERANCE:
        problematic = counts[counts["absolute_diff"] >= CARE_MIX_TOLERANCE].copy()
        print("\nProblematic groups where care mix doesn't sum to care demand:")
        print(problematic.head(20))
        print(f"\nTotal problematic groups: {len(problematic)}")
        print("\nSample of problematic data:")
        # Show a sample of the raw data for one problematic group
        if len(problematic) > 0:
            sample_idx = problematic.index[0]
            sample_data = df_test[
                (df_test["age"] == sample_idx[0])
                & (df_test["education"] == sample_idx[1])
                & (df_test["caregiving_type"] == sample_idx[2])
            ]
            print(
                f"\nSample group: age={sample_idx[0]}, "
                f"education={sample_idx[1]}, caregiving_type={sample_idx[2]}"
            )
            print(f"care_demand == 1 count: {(sample_data['care_demand'] == 1).sum()}")
            print(f"solo_informal_care sum: {sample_data['solo_informal_care'].sum()}")
            print(f"formal_care sum: {sample_data['formal_care'].sum()}")
            print(f"other_family_only sum: {sample_data['other_family_only'].sum()}")
            care_mix_sum_sample = (
                sample_data["solo_informal_care"].sum()
                + sample_data["formal_care"].sum()
                + sample_data["other_family_only"].sum()
            )
            print(f"care_mix_sum: {care_mix_sum_sample}")
            print("\nChoices when care_demand == 1:")
            care_demand_1 = sample_data[sample_data["care_demand"] == 1]
            print(care_demand_1["choice"].value_counts().sort_index())

    # Assert that the test passed
    tolerance = CARE_MIX_TOLERANCE
    assert max_absolute_diff < tolerance, (
        f"Care mix does not sum to care demand. "
        f"Max absolute difference: {max_absolute_diff}, "
        f"Max relative difference: {max_relative_diff}"
    )


@pytask.mark.debug
@pytask.mark.baseline_model
@pytask.mark.post_estimation
def task_check_no_care_with_positive_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
) -> None:
    """
    Quick consistency check:

    Verify that within the caregiving age window, whenever care_demand > 0 and
    caregiving_type == 1, the model never chooses NO_CARE (choices 0–3).
    """

    # Load model specs to infer caregiving age window
    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=BLD / "model" / "model.pkl",
    )

    start_age = specs["start_age"]
    end_age_msm = specs["end_age_msm"]
    start_age_caregiving = specs["start_age_caregiving"]
    end_age_caregiving = end_age_msm

    # Load simulated data
    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()

    # Create age variable from start_age + period
    df_sim["age"] = df_sim["period"] + start_age

    # NO_CARE choices are 0–3
    no_care_choices = set(NO_CARE.tolist())

    mask = (
        (df_sim["care_demand"] > 0)
        & (df_sim["health"] != DEAD)
        # & (df_sim["mother_dead"] == 0)
        & (df_sim["caregiving_type"] == 1)
        & (df_sim["choice"].isin(no_care_choices))
        & (df_sim["age"].between(start_age_caregiving, end_age_caregiving))
    )

    n_violations = int(mask.sum())

    print(
        f"\nConsistency check: caregiving window ages "
        f"[{start_age_caregiving}, {end_age_caregiving}]."
    )
    print(
        "Rows with care_demand>0, caregiving_type==1, choice in NO_CARE within "
        f"this window: {n_violations}"
    )

    if n_violations > 0:
        sample = df_sim.loc[
            mask,
            [
                "age",
                "education",
                "care_demand",
                "caregiving_type",
                "choice",
            ],
        ].head(20)
        print("\nSample violating rows (first 20):")
        print(sample.to_string(index=False))

    assert n_violations == 0, (
        "Found choices in NO_CARE for caregiving_type==1 and care_demand>0 "
        f"in ages [{start_age_caregiving}, {end_age_caregiving}]. "
        f"Count: {n_violations}."
    )
    # breakpoint()
