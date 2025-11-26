"""Initial conditions."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dcegm.pre_processing.setup_model import load_model_dict
from dcegm.wealth_correction import adjust_observed_wealth
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.task_generate_initial_conditions import (
    draw_start_wealth_dist,
)


def task_plot_initial_wealth(
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "initial_conditions"
    / "wealth_distributions.png",
):
    """Plot initial wealth for different underlying distributions."""

    observed_data = pd.read_csv(path_to_sample, index_col=[0])

    options = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))

    model = load_model_dict(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_model,
        sim_model=False,
    )

    specs = options["model_params"]
    n_agents_edu = specs["n_agents"]
    seed = specs["seed"]

    np.random.seed(seed)

    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()
    start_period_data = start_period_data[start_period_data["wealth"].notnull()].copy()

    states_dict = {
        name: start_period_data[name].values
        for name in model["model_structure"]["discrete_states_names"]
        if name not in ("mother_health", "care_demand", "care_supply")
    }

    states_dict["care_demand"] = np.zeros_like(start_period_data["wealth"])
    states_dict["wealth"] = start_period_data["wealth"].values / specs["wealth_unit"]
    states_dict["experience"] = start_period_data["experience"].values
    start_period_data.loc[:, "adjusted_wealth"] = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=params,
        model=model,
    )

    print(start_period_data["adjusted_wealth"].describe())

    # Plotting
    methods = ["uniform", "lognormal", "kde"]
    n_agents_edu = 1_000

    xmin, xmax = 0, 1_000
    bins = np.linspace(xmin, xmax, 101)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, edu in enumerate(specs["education_labels"]):
        # Filter data for this education level
        start_period_data_edu = start_period_data[start_period_data["education"] == idx]
        wealth_data = start_period_data_edu["adjusted_wealth"]
        print(f"Education {edu}:\n", wealth_data.describe(), "\n")

        ax = axes[idx]  # Current subplot

        for method in methods:
            samples = draw_start_wealth_dist(
                start_period_data_edu, n_agents_edu, method
            )
            # samples = np.clip(samples, xmin, xmax)  # Clip for visibility
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
