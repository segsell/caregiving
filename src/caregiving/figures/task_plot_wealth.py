"""Wealth plots."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint


def task_plot_budget_of_unemployed(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "wealth_and_budget"
    / "budget_of_unemployed.png",
):
    """Plot the budget constraint (=wealth) for different levels of end-of period
    savings of an unemployed person.

    Special emphasis on the area around eligibility for unemployment benefits.

    """

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    params = {
        "interest_rate": specs["interest_rate"],
    }

    savings = np.linspace(0, 1_000, 100)

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            wealth = budget_constraint(
                period=70,
                education=edu_var,
                lagged_choice=0,
                experience=0.01,
                # sex=sex_var,
                partner_state=np.array([1]),
                has_sister=np.array([0]),
                care_demand=np.array([0]),
                savings_end_of_previous_period=savings,
                income_shock_previous_period=0,
                params=params,
                options=specs,
            )
            ax.plot(savings, wealth, label=edu_label)
            ax.set_xlabel("Savings")
            ax.set_ylabel("Wealth")
            ax.legend()
            ax.set_title(f"Unemployment benefits {sex_label}; {edu_label}")

    plt.tight_layout()
    plt.savefig(path_to_save)
