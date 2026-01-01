"""Plot inheritance receipt probability for no care demand model.

Visualizes the inheritance receipt probability from the inheritance_transition_no_care_demand
function by age and education.
"""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.stochastic_processes.inheritance_transition_no_care_demand import (
    inheritance_transition_no_care_demand,
)


@pytask.mark.pre_estimation_no_care_demand
@pytask.mark.figures
def task_plot_inheritance_probability_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "inheritance_probability_no_care_demand.png",
):
    """Plot inheritance receipt probability by age and education for no care demand model.

    Uses the inheritance_transition_no_care_demand function to compute probabilities
    and plots them by age with separate lines for low and high education.

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_save_plot : Path
        Path to save the plot

    """
    # Load specs
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    # Age range
    start_age = specs["start_age"]
    ages = np.arange(40, 81)  # 40 to 80 inclusive
    periods = ages - start_age

    # Colors for education levels
    edu_colors = [plt.cm.tab10(i) for i in range(len(specs["education_labels"]))]

    # =================================================================================
    # Plot: Inheritance receipt probability by age and education
    # =================================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for edu_var, edu_label in enumerate(specs["education_labels"]):
        color = edu_colors[edu_var]

        probabilities = []

        for period, age in zip(periods, ages, strict=True):
            if period < 0 or period >= specs["n_periods"]:
                probabilities.append(np.nan)
                continue

            # Get probability vector from transition function
            prob_vector = inheritance_transition_no_care_demand(
                period=period,
                education=edu_var,
                model_specs=specs,
            )

            # Extract probability of receiving inheritance (state 1)
            prob_inheritance = float(prob_vector[1])
            probabilities.append(prob_inheritance)

        probabilities = np.array(probabilities)
        mask = ~np.isnan(probabilities)

        if mask.sum() > 0:
            ax.plot(
                ages[mask],
                probabilities[mask],
                linewidth=2,
                color=color,
                label=edu_label,
                alpha=0.8,
            )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Probability of Receiving Inheritance", fontsize=12)
    ax.set_title(
        "Inheritance Receipt Probability by Age and Education (No Care Demand Model)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)

    print(f"Inheritance probability plot saved to {path_to_save_plot}")
