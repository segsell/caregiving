"""Initial conditions plotting for wealth distributions."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import yaml
from dcegm.asset_correction import adjust_observed_assets
from pytask import Product

from caregiving.config import BLD
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.task_generate_initial_conditions import (
    draw_start_wealth_dist,
)
from dcegm import setup_model


@pytask.mark.initial_wealth
def task_plot_initial_wealth(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_model: Path = BLD / "model" / "model.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "initial_conditions"
    / "wealth_distributions.png",
) -> None:
    """Plot initial wealth distributions and model-implied wealth samples."""

    observed_data = pd.read_csv(path_to_sample, index_col=[0])

    specs = pickle.load(path_to_specs.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model_class = setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_model,
    )
    model_specs = model_class.model_specs
    model_structure = model_class.model_structure

    n_agents_edu = model_specs["n_agents"]
    seed = model_specs["seed"]

    np.random.seed(seed)

    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()
    start_period_data = start_period_data[start_period_data["wealth"].notnull()].copy()

    states_dict = {
        name: start_period_data[name].values
        for name in model_structure["discrete_states_names"]
        if name
        not in (
            "mother_health",
            "mother_adl",
            "mother_dead",
            "care_demand",
            "care_supply",
            "caregiving_type",
            "mother_alive",
            "father_alive",
        )
    }

    states_dict["care_demand"] = np.zeros_like(start_period_data["wealth"])
    states_dict["experience"] = start_period_data["experience"].values
    states_dict["assets_begin_of_period"] = (
        start_period_data["wealth"].values / model_specs["wealth_unit"]
    )

    start_period_data.loc[:, "adjusted_wealth"] = adjust_observed_assets(
        observed_states_dict=states_dict,
        params=params,
        model_class=model_class,
    )

    print(start_period_data["adjusted_wealth"].describe())

    # Plotting
    methods = ["uniform", "lognormal", "kde"]
    n_agents_edu = 1_000

    xmin, xmax = 0, 1_000
    bins = np.linspace(xmin, xmax, 101)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, edu in enumerate(model_specs["education_labels"]):
        # Filter data for this education level
        start_period_data_edu = start_period_data[start_period_data["education"] == idx]
        wealth_data = start_period_data_edu["adjusted_wealth"]
        print(f"Education {edu}:\n", wealth_data.describe(), "\n")

        ax = axes[idx]  # Current subplot

        for method in methods:
            samples = draw_start_wealth_dist(
                start_period_data_edu, n_agents_edu, method
            )
            ax.hist(samples, bins=bins, density=True, alpha=0.6, label=method)

        ax.set_title(str(edu))
        ax.set_xlim([xmin, xmax])
        ax.set_xlabel("Wealth")
        ax.grid(True)
        if idx == 0:
            ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()

    plt.savefig(path_to_save, dpi=300)
    plt.close(fig)


@pytask.mark.initial_conditions
def task_plot_education_shares(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "initial_conditions"
    / "education_shares.png",
) -> None:
    """Plot bar chart of education shares (low and high education)."""
    observed_data = pd.read_csv(path_to_sample, index_col=[0])
    specs = pickle.load(path_to_specs.open("rb"))

    # Define start data
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()

    # Calculate shares
    education_counts = start_period_data["education"].value_counts().sort_index()
    total = len(start_period_data)
    shares = education_counts / total

    # Create labels
    labels = [specs["education_labels"][idx] for idx in shares.index]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, shares.values, color=["#1f77b4", "#ff7f0e"], alpha=0.7)
    ax.set_ylabel("Share", fontsize=12)
    ax.set_title("Education Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, share in zip(bars, shares.values, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{share:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300)
    plt.close(fig)


@pytask.mark.initial_conditions
def task_plot_job_offer_shares(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "initial_conditions"
    / "job_offer_shares.png",
) -> None:
    """Plot bar chart of positive job offer shares for all, low educ, and high educ."""
    observed_data = pd.read_csv(path_to_sample, index_col=[0])
    specs = pickle.load(path_to_specs.open("rb"))

    # Define start data
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()

    # Filter out missing job_offer values (if any)
    data = start_period_data[start_period_data["job_offer"].notna()].copy()

    # Calculate shares
    # All
    share_all = (data["job_offer"] == 1).sum() / len(data)

    # Low education
    data_low = data[data["education"] == 0]
    share_low = (
        (data_low["job_offer"] == 1).sum() / len(data_low) if len(data_low) > 0 else 0.0
    )

    # High education
    data_high = data[data["education"] == 1]
    share_high = (
        (data_high["job_offer"] == 1).sum() / len(data_high)
        if len(data_high) > 0
        else 0.0
    )

    # Create labels
    labels = [
        "All",
        specs["education_labels"][0],
        specs["education_labels"][1],
    ]
    shares = [share_all, share_low, share_high]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, shares, color=["#2ca02c", "#1f77b4", "#ff7f0e"], alpha=0.7)
    ax.set_ylabel("Share with Positive Job Offer", fontsize=12)
    ax.set_title("Job Offer Distribution by Education", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, share in zip(bars, shares, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{share:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300)
    plt.close(fig)
