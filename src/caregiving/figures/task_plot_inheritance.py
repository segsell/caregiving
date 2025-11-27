"""Inheritance amount plots."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP


@pytask.mark.inheritance
def task_plot_inheritance_by_age_and_education(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "inheritance_by_age_and_education.png",
):
    """Plot predicted inheritance amounts by age and education.

    Based on regression results from Table C.9 (inheritance conditional on positive
    inheritance). The plot shows the predicted ln(inheritance) exponentiated to get
    the inheritance amount in euros.

    Shows four scenarios:
    - Low education with 4 years intensive care (solid blue)
    - High education with 4 years intensive care (solid orange)
    - Low education with 0 years intensive care (dashed blue)
    - High education with 0 years intensive care (dashed orange)

    Assumptions:
    - Parent died in t-1 (death_t_minus_1 = 1)
    - Region = West (region_east = 0)
    - Light care years = 0

    """
    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    # Regression coefficients from Table C.9
    coef_constant = -1.211
    coef_age = 0.368
    coef_age_squared = -0.294  # divided by 100 in the formula
    coef_region_east = -1.237
    coef_care_experience = 0.087  # (1/3) years light care + years intensive care
    coef_death_t_minus_1 = -0.215
    coef_interaction = -0.059  # death * care experience
    coef_education_high = 0.248

    # Fixed conditions
    region_east = 0  # West Germany
    death_t_minus_1 = 1  # Parent died in previous period
    light_care_years = 0

    # Age range
    ages = np.arange(40, 81)  # 40 to 80 inclusive

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for both education levels and both care scenarios
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        education_high = edu_var
        color = JET_COLOR_MAP[edu_var]

        # Scenario 1: 4 years intensive care (solid line)
        intensive_care_years = 6
        care_experience = (1 / 3) * light_care_years + intensive_care_years
        interaction_term = death_t_minus_1 * care_experience

        ln_inheritance_pred = (
            coef_constant
            + coef_age * ages
            + coef_age_squared * (ages**2) / 100
            + coef_region_east * region_east
            + coef_care_experience * care_experience
            + coef_death_t_minus_1 * death_t_minus_1
            + coef_interaction * interaction_term
            + coef_education_high * education_high
        )
        inheritance_pred = np.exp(ln_inheritance_pred)

        ax.plot(
            ages,
            inheritance_pred,
            linewidth=2,
            color=color,
            linestyle="-",
            label=f"{edu_label}, {intensive_care_years} years intensive care",
        )

        # Scenario 2: 0 years intensive care (dashed line)
        intensive_care_years = 0
        care_experience = (1 / 3) * light_care_years + intensive_care_years
        interaction_term = death_t_minus_1 * care_experience

        ln_inheritance_pred = (
            coef_constant
            + coef_age * ages
            + coef_age_squared * (ages**2) / 100
            + coef_region_east * region_east
            + coef_care_experience * care_experience
            + coef_death_t_minus_1 * death_t_minus_1
            + coef_interaction * interaction_term
            + coef_education_high * education_high
        )
        inheritance_pred = np.exp(ln_inheritance_pred)

        ax.plot(
            ages,
            inheritance_pred,
            linewidth=2,
            color=color,
            linestyle="--",
            label=f"{edu_label}, 0 years intensive care",
        )

    # Formatting
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Expected Inheritance (€)", fontsize=12)
    ax.set_title(
        "Predicted Inheritance Amount by Age, Education, and Care Experience\n"
        "(Parent died t-1, West Germany)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 80)
    ax.legend(loc="best", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close(fig)


@pytask.mark.inheritance
def task_plot_inheritance_probability_by_age_and_education(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "inheritance_probability_by_age_and_education.png",
):
    """Plot probability of positive inheritance by age and education.

    Based on logit regression results from Table C.8 (probability of positive
    inheritance). The plot shows the predicted probability using the logistic
    function: P(inheritance>0) = 1 / (1 + exp(-X*beta)).

    Shows four scenarios:
    - Low education with 6 years intensive care (solid blue)
    - High education with 6 years intensive care (solid orange)
    - Low education with 0 years intensive care (dashed blue)
    - High education with 0 years intensive care (dashed orange)

    Assumptions:
    - Parent died in t-1 (death_t_minus_1 = 1)
    - Region = West (region_east = 0)
    - Light care years = 0

    """
    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    # Logit regression coefficients from Table C.8
    coef_constant = -18.693
    coef_age = 0.524
    coef_age_squared = -0.463  # divided by 100 in the formula
    coef_region_east = -0.537
    coef_care_experience = 0.141  # (1/3) years light care + years intensive care
    coef_death_t_minus_1 = 1.134
    coef_interaction = 0.080  # death * care experience
    coef_education_high = 0.582

    # Fixed conditions
    region_east = 0  # West Germany
    death_t_minus_1 = 1  # Parent died in previous period
    light_care_years = 0

    # Age range
    ages = np.arange(40, 81)  # 40 to 80 inclusive

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for both education levels and both care scenarios
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        education_high = edu_var
        color = JET_COLOR_MAP[edu_var]

        # Scenario 1: 6 years intensive care (solid line)
        intensive_care_years = 6
        care_experience = (1 / 3) * light_care_years + intensive_care_years
        interaction_term = death_t_minus_1 * care_experience

        # Calculate linear predictor (X*beta)
        linear_pred = (
            coef_constant
            + coef_age * ages
            + coef_age_squared * (ages**2) / 100
            + coef_region_east * region_east
            + coef_care_experience * care_experience
            + coef_death_t_minus_1 * death_t_minus_1
            + coef_interaction * interaction_term
            + coef_education_high * education_high
        )

        # Apply logistic function to get probability
        prob_positive_inheritance = 1 / (1 + np.exp(-linear_pred))

        ax.plot(
            ages,
            prob_positive_inheritance,
            linewidth=2,
            color=color,
            linestyle="-",
            label=f"{edu_label}, {intensive_care_years} years intensive care",
        )

        # Scenario 2: 0 years intensive care (dashed line)
        intensive_care_years = 0
        care_experience = (1 / 3) * light_care_years + intensive_care_years
        interaction_term = death_t_minus_1 * care_experience

        # Calculate linear predictor (X*beta)
        linear_pred = (
            coef_constant
            + coef_age * ages
            + coef_age_squared * (ages**2) / 100
            + coef_region_east * region_east
            + coef_care_experience * care_experience
            + coef_death_t_minus_1 * death_t_minus_1
            + coef_interaction * interaction_term
            + coef_education_high * education_high
        )

        # Apply logistic function to get probability
        prob_positive_inheritance = 1 / (1 + np.exp(-linear_pred))

        ax.plot(
            ages,
            prob_positive_inheritance,
            linewidth=2,
            color=color,
            linestyle="--",
            label=f"{edu_label}, 0 years intensive care",
        )

    # Formatting
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Probability of Positive Inheritance", fontsize=12)
    ax.set_title(
        "Probability of Positive Inheritance by Age, Education, and Care Experience\n"
        "(Parent died t-1, West Germany)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 80)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close(fig)
