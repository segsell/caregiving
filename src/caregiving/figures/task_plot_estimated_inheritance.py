"""Plot estimated inheritance probability and amounts from SOEP regressions."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs


@pytask.mark.inheritance
def task_plot_estimated_inheritance_probability(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_logit_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_logit_params.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_inheritance_probability_by_age_education.png",
):
    """Plot estimated probability of inheritance by age and education from logit model.

    Uses estimated parameters from task_estimate_inheritance.
    Shows four scenarios per sex:
    - Low/High education with care (solid lines)
    - Low/High education without care (dashed lines)

    Assumptions:
    - parent_died_recent=1 (parent died in t or t-1)

    Caregiving scenario: any_care_recent=1
    No care scenario: any_care_recent=0

    """
    specs = read_and_derive_specs(path_to_specs)

    # Load estimated parameters
    logit_params = pd.read_csv(path_to_logit_params, index_col=0)

    # Age range for prediction
    ages = np.arange(40, 81)
    age_sq = ages**2

    # Create figure with subplots for each sex
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axes[sex_var]

        # Get parameters for this sex
        try:
            params = logit_params.loc[sex_label]

            # Plot for each education level and care scenario
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                color = JET_COLOR_MAP[edu_var]

                # Scenario 1: With care, parent died recently (solid line)
                linear_pred_care = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["any_care_recent"] * 1
                    + params["formal_care_costs_dummy_recent"] * 0
                    + params["parent_died_recent"] * 1
                    + params["education"] * edu_var
                )
                prob_care = 1 / (1 + np.exp(-linear_pred_care))

                ax.plot(
                    ages,
                    prob_care,
                    linewidth=2.5,
                    color=color,
                    linestyle="-",
                    label=f"{edu_label}, with care",
                )

                # Scenario 2: No care, parent died recently (dashed line)
                linear_pred_nocare = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["any_care_recent"] * 0
                    + params["formal_care_costs_dummy_recent"] * 0
                    + params["parent_died_recent"] * 1
                    + params["education"] * edu_var
                )
                prob_nocare = 1 / (1 + np.exp(-linear_pred_nocare))

                ax.plot(
                    ages,
                    prob_nocare,
                    linewidth=2.5,
                    color=color,
                    linestyle="--",
                    label=f"{edu_label}, no care",
                )

                # Scenario 3: Formal care, parent died recently (dash-dot line)
                linear_pred_formal = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["any_care_recent"] * 0
                    + params["formal_care_costs_dummy_recent"] * 1
                    + params["parent_died_recent"] * 1
                    + params["education"] * edu_var
                )
                prob_formal = 1 / (1 + np.exp(-linear_pred_formal))

                ax.plot(
                    ages,
                    prob_formal,
                    linewidth=2.5,
                    color=color,
                    linestyle=(0, (3, 1, 1, 1)),
                    label=f"{edu_label}, formal care",
                )

        except (KeyError, TypeError):
            print(f"Warning: No parameters found for {sex_label}")
            continue

        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Probability of Inheritance", fontsize=12)
        ax.set_title(str(sex_label), fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(40, 80)
        ax.set_ylim(0, None)

    plt.suptitle(
        "Estimated Probability of Inheritance by Age and Education\n"
        "(Conditional on Parent Death in t or t-1)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.inheritance
def task_plot_estimated_inheritance_amount(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_amount_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_amount_params.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_inheritance_amount_by_age_education.png",
):
    """Plot estimated inheritance amount by age and education from OLS model.

    Uses estimated parameters from task_estimate_inheritance.
    Note: The OLS regression has ln(inheritance_amount) as the outcome,
    which is exponentiated to get the actual amount in euros.

    Shows four scenarios per sex:
    - Low/High education with intensive care (solid lines)
    - Low/High education without care (dashed lines)

    Caregiving scenario: lagged_intensive_care=1, lagged_light_care=0
    No care scenario: lagged_intensive_care=0, lagged_light_care=0

    """
    specs = read_and_derive_specs(path_to_specs)

    # Load estimated parameters
    amount_params = pd.read_csv(path_to_amount_params, index_col=0)

    # Age range for prediction
    ages = np.arange(40, 81)
    age_sq = ages**2

    # Create figure with subplots for each sex
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axes[sex_var]

        # Get parameters for this sex
        try:
            params = amount_params.loc[sex_label]

            # Plot for each education level and care scenario
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                color = JET_COLOR_MAP[edu_var]

                # Scenario 1: With intensive care (solid line)
                ln_amount_care = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["light_care_recent"] * 0
                    + params["intensive_care_recent"] * 1
                    + params["formal_care_costs_dummy_recent"] * 0
                    + params["education"] * edu_var
                )
                amount_care = np.exp(ln_amount_care)

                ax.plot(
                    ages,
                    amount_care,
                    linewidth=2.5,
                    color=color,
                    linestyle="-",
                    label=f"{edu_label}, intensive care",
                )

                # Scenario 2: No care (dashed line)
                ln_amount_nocare = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["light_care_recent"] * 0
                    + params["intensive_care_recent"] * 0
                    + params["formal_care_costs_dummy_recent"] * 0
                    + params["education"] * edu_var
                )
                amount_nocare = np.exp(ln_amount_nocare)

                ax.plot(
                    ages,
                    amount_nocare,
                    linewidth=2.5,
                    color=color,
                    linestyle="--",
                    label=f"{edu_label}, no care",
                )

                # Scenario 3: Formal care (dash-dot line)
                ln_amount_formal = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["light_care_recent"] * 0
                    + params["intensive_care_recent"] * 0
                    + params["formal_care_costs_dummy_recent"] * 1
                    + params["education"] * edu_var
                )
                amount_formal = np.exp(ln_amount_formal)

                ax.plot(
                    ages,
                    amount_formal,
                    linewidth=2.5,
                    color=color,
                    linestyle=(0, (3, 1, 1, 1)),
                    label=f"{edu_label}, formal care",
                )

        except (KeyError, TypeError):
            print(f"Warning: No parameters found for {sex_label}")
            continue

        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Inheritance Amount (€)", fontsize=12)
        ax.set_title(str(sex_label), fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(40, 80)
        ax.set_ylim(0, None)

        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"€{x:,.0f}"))

    plt.suptitle(
        "Estimated Inheritance Amount by Age and Education\n"
        "(Conditional on Positive Inheritance, Parent Death in t or t-1)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.inheritance
def task_plot_estimated_inheritance_amount_three_scenarios(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_amount_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_amount_params.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_inheritance_amount_by_care_type.png",
):
    """Plot estimated inheritance amount with three caregiving scenarios.

    Uses estimated parameters from task_estimate_inheritance.
    Shows six scenarios per sex (3 care types × 2 education levels):
    - Intensive care: light_care_recent=0, intensive_care_recent=1 (solid lines)
    - Light care: light_care_recent=1, intensive_care_recent=0 (dotted lines)
    - No care: light_care_recent=0, intensive_care_recent=0 (dashed lines)

    """
    specs = read_and_derive_specs(path_to_specs)

    # Load estimated parameters
    amount_params = pd.read_csv(path_to_amount_params, index_col=0)

    # Age range for prediction
    ages = np.arange(40, 81)
    age_sq = ages**2

    # Create figure with subplots for each sex
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axes[sex_var]

        # Get parameters for this sex
        try:
            params = amount_params.loc[sex_label]

            # Plot for each education level and three care scenarios
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                color = JET_COLOR_MAP[edu_var]

                # Scenario 1: Intensive care (solid line)
                ln_amount_intensive = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["light_care_recent"] * 0
                    + params["intensive_care_recent"] * 1
                    + params["formal_care_costs_dummy_recent"] * 0
                    + params["education"] * edu_var
                )
                amount_intensive = np.exp(ln_amount_intensive)

                ax.plot(
                    ages,
                    amount_intensive,
                    linewidth=2.5,
                    color=color,
                    linestyle="-",
                    label=f"{edu_label}, intensive care",
                )

                # Scenario 2: Light care (dotted line)
                ln_amount_light = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["light_care_recent"] * 1
                    + params["intensive_care_recent"] * 0
                    + params["formal_care_costs_dummy_recent"] * 0
                    + params["education"] * edu_var
                )
                amount_light = np.exp(ln_amount_light)

                ax.plot(
                    ages,
                    amount_light,
                    linewidth=2.5,
                    color=color,
                    linestyle=":",
                    label=f"{edu_label}, light care",
                )

                # Scenario 3: No care (dashed line)
                ln_amount_nocare = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["light_care_recent"] * 0
                    + params["intensive_care_recent"] * 0
                    + params["formal_care_costs_dummy_recent"] * 0
                    + params["education"] * edu_var
                )
                amount_nocare = np.exp(ln_amount_nocare)

                ax.plot(
                    ages,
                    amount_nocare,
                    linewidth=2.5,
                    color=color,
                    linestyle="--",
                    label=f"{edu_label}, no care",
                )

                # Scenario 4: Formal care (dash-dot line)
                ln_amount_formal = (
                    params["const"]
                    + params["age"] * ages
                    + params["age_sq"] * age_sq
                    + params["light_care_recent"] * 0
                    + params["intensive_care_recent"] * 0
                    + params["formal_care_costs_dummy_recent"] * 1
                    + params["education"] * edu_var
                )
                amount_formal = np.exp(ln_amount_formal)

                ax.plot(
                    ages,
                    amount_formal,
                    linewidth=2.5,
                    color=color,
                    linestyle=(0, (3, 1, 1, 1)),
                    label=f"{edu_label}, formal care",
                )

        except (KeyError, TypeError):
            print(f"Warning: No parameters found for {sex_label}")
            continue

        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Inheritance Amount (€)", fontsize=12)
        ax.set_title(str(sex_label), fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(40, 80)
        ax.set_ylim(0, None)

        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"€{x:,.0f}"))

    plt.suptitle(
        "Estimated Inheritance Amount by Age, Education, and Care Type\n"
        "(Conditional on Positive Inheritance, Parent Death in t or t-1)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close()
