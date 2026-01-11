"""Plot inheritance probability for different model specifications."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs

# Specification definitions matching task_estimate_inheritance_soep.py
SPECIFICATIONS = [
    {
        "name": "spec1_any_care_parent_this_year",
        "care_var": "any_care",
        "parent_var": "parent_died_this_year",
        "filter": None,
        "title": "Any Care This Year + Parent Died This Year",
        "care_value": 1,  # For plotting "with care" scenario
    },
    {
        "name": "spec2_any_care_recent_parent_recent",
        "care_var": "any_care_recent",
        "parent_var": "parent_died_recent",
        "filter": None,
        "title": "Any Care Recent + Parent Died Recent",
        "care_value": 1,
    },
    {
        "name": "spec3_any_care_last_year_parent_last_year",
        "care_var": "any_care_last_year",
        "parent_var": "parent_died_last_year",
        "filter": None,
        "title": "Any Care Last Year + Parent Died Last Year",
        "care_value": 1,
    },
    {
        "name": "spec4_any_care_last_year_filter_parent_this_year",
        "care_var": "any_care_last_year",
        "parent_var": None,
        "filter": "parent_died_this_year == 1",
        "title": "Any Care Last Year (Filter: Parent Died This Year)",
        "care_value": 1,
    },
    {
        "name": "spec7_any_care_this_year_filter_parent_this_year",
        "care_var": "any_care",
        "parent_var": None,
        "filter": "parent_died_this_year == 1",
        "title": "Any Care This Year (Filter: Parent Died This Year)",
        "care_value": 1,
    },
    {
        "name": "spec10_any_care_recent_filter_parent_this_year",
        "care_var": "any_care_recent",
        "parent_var": None,
        "filter": "parent_died_this_year == 1",
        "title": "Any Care Recent (Filter: Parent Died This Year)",
        "care_value": 1,
    },
    {
        "name": "spec5_any_care_last_year_filter_parent_last_year",
        "care_var": "any_care_last_year",
        "parent_var": None,
        "filter": "parent_died_last_year == 1",
        "title": "Any Care Last Year (Filter: Parent Died Last Year)",
        "care_value": 1,
    },
    {
        "name": "spec8_any_care_this_year_filter_parent_last_year",
        "care_var": "any_care",
        "parent_var": None,
        "filter": "parent_died_last_year == 1",
        "title": "Any Care This Year (Filter: Parent Died Last Year)",
        "care_value": 1,
    },
    {
        "name": "spec11_any_care_recent_filter_parent_last_year",
        "care_var": "any_care_recent",
        "parent_var": None,
        "filter": "parent_died_last_year == 1",
        "title": "Any Care Recent (Filter: Parent Died Last Year)",
        "care_value": 1,
    },
    {
        "name": "spec6_any_care_last_year_filter_parent_recent",
        "care_var": "any_care_last_year",
        "parent_var": None,
        "filter": "parent_died_recent == 1",
        "title": "Any Care Last Year (Filter: Parent Died Recent)",
        "care_value": 1,
    },
    {
        "name": "spec9_any_care_this_year_filter_parent_recent",
        "care_var": "any_care",
        "parent_var": None,
        "filter": "parent_died_recent == 1",
        "title": "Any Care This Year (Filter: Parent Died Recent)",
        "care_value": 1,
    },
    {
        "name": "spec12_any_care_recent_filter_parent_recent",
        "care_var": "any_care_recent",
        "parent_var": None,
        "filter": "parent_died_recent == 1",
        "title": "Any Care Recent (Filter: Parent Died Recent)",
        "care_value": 1,
    },
]


def plot_single_specification(
    spec_info, spec_params, specs, ages, ax, sex_var, sex_label
):
    """Plot a single specification's predictions."""
    try:
        params = spec_params.loc[sex_label]

        # Build base prediction
        age_sq = ages**2

        # Calculate for care and no-care scenarios
        for care_scenario in (0, 1):
            linear_pred = (
                params["const"]
                + params["age"] * ages
                + params["age_sq"] * age_sq
                + params[spec_info["care_var"]] * care_scenario
                + params["education"] * 1  # High education
            )

            # Add parent variable if it exists
            if spec_info["parent_var"] is not None:
                linear_pred += params[spec_info["parent_var"]] * 1

            prob = 1 / (1 + np.exp(-linear_pred))

            linestyle = "-" if care_scenario == 1 else "--"
            label = "With care" if care_scenario == 1 else "No care"

            ax.plot(
                ages,
                prob,
                linewidth=2,
                linestyle=linestyle,
                label=label,
                alpha=0.8,
            )

        return True
    except (KeyError, TypeError) as e:
        print(f"Warning: Could not plot {spec_info['name']} for {sex_label}: {e}")
        return False


# Create individual plotting tasks for each specification
for spec in SPECIFICATIONS:

    @pytask.mark.inheritance
    @pytask.task(id=spec["name"])
    def task_plot_specification(
        spec_info=spec,
        path_to_specs: Path = SRC / "specs.yaml",
        path_to_summary: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_specs"
        / "_specs_summary.txt",
        path_to_params: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_specs"
        / f"{spec['name']}_params.csv",
        path_to_save: Annotated[Path, Product] = BLD
        / "plots"
        / "stochastic_processes"
        / "inheritance_specs"
        / f"{spec['name']}.png",
    ):
        """Plot inheritance probability for a specific specification."""
        specs_dict = read_and_derive_specs(path_to_specs)

        # Load parameters
        try:
            spec_params = pd.read_csv(path_to_params, index_col=0)
        except FileNotFoundError:
            print(f"Parameters not found for {spec_info['name']}, skipping plot")
            return

        # Age range
        ages = np.arange(40, 81)

        # Create figure with subplots for each sex
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for sex_var, sex_label in enumerate(specs_dict["sex_labels"]):
            ax = axes[sex_var]

            # Plot for each education level
            for edu_var, edu_label in enumerate(specs_dict["education_labels"]):
                try:
                    params = spec_params.loc[sex_label]
                    age_sq = ages**2

                    # Scenario 1: With care (solid)
                    linear_pred = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["care_var"]] * 1
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        linear_pred += params[spec_info["parent_var"]] * 1

                    prob_care = 1 / (1 + np.exp(-linear_pred))

                    ax.plot(
                        ages,
                        prob_care,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="-",
                        label=f"{edu_label}, with care",
                    )

                    # Scenario 2: No care (dashed)
                    linear_pred_no = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["care_var"]] * 0
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        linear_pred_no += params[spec_info["parent_var"]] * 1

                    prob_no = 1 / (1 + np.exp(-linear_pred_no))

                    ax.plot(
                        ages,
                        prob_no,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="--",
                        label=f"{edu_label}, no care",
                    )

                except (KeyError, TypeError):
                    continue

            ax.set_xlabel("Age", fontsize=12)
            ax.set_ylabel("Probability of Inheritance", fontsize=12)
            ax.set_title(str(sex_label), fontsize=13, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim(40, 80)
            ax.set_ylim(0, None)

        # Add specification info to title
        filter_text = f"\n{spec_info['filter']}" if spec_info["filter"] else ""
        plt.suptitle(
            f"{spec_info['title']}{filter_text}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
        plt.close()


# Create plotting tasks for amount specifications
for spec in SPECIFICATIONS:

    @pytask.mark.inheritance
    @pytask.task(id=f"{spec['name']}_amount")
    def task_plot_amount_specification(
        spec_info=spec,
        path_to_specs: Path = SRC / "specs.yaml",
        path_to_summary: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_amount_specs"
        / "_specs_summary.txt",
        path_to_params: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_amount_specs"
        / f"{spec['name']}_params.csv",
        path_to_save: Annotated[Path, Product] = BLD
        / "plots"
        / "stochastic_processes"
        / "inheritance_amount_specs"
        / f"{spec['name']}.png",
    ):
        """Plot inheritance amount for a specific specification."""
        specs_dict = read_and_derive_specs(path_to_specs)

        # Load parameters
        try:
            spec_params = pd.read_csv(path_to_params, index_col=0)
        except FileNotFoundError:
            print(f"Parameters not found for {spec_info['name']}, skipping plot")
            return

        # Age range
        ages = np.arange(40, 81)

        # Create figure with subplots for each sex
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for sex_var, sex_label in enumerate(specs_dict["sex_labels"]):
            ax = axes[sex_var]

            # Plot for each education level
            for edu_var, edu_label in enumerate(specs_dict["education_labels"]):
                try:
                    params = spec_params.loc[sex_label]
                    age_sq = ages**2

                    # Scenario 1: With care (solid)
                    ln_amount_pred = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["care_var"]] * 1
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        ln_amount_pred += params[spec_info["parent_var"]] * 1

                    # Exponentiate to get actual amount
                    amount_care = np.exp(ln_amount_pred)

                    ax.plot(
                        ages,
                        amount_care,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="-",
                        label=f"{edu_label}, with care",
                    )

                    # Scenario 2: No care (dashed)
                    ln_amount_no = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["care_var"]] * 0
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        ln_amount_no += params[spec_info["parent_var"]] * 1

                    # Exponentiate to get actual amount
                    amount_no = np.exp(ln_amount_no)

                    ax.plot(
                        ages,
                        amount_no,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="--",
                        label=f"{edu_label}, no care",
                    )

                except (KeyError, TypeError):
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

        # Add specification info to title
        filter_text = f"\n{spec_info['filter']}" if spec_info["filter"] else ""
        plt.suptitle(
            f"Inheritance Amount: {spec_info['title']}{filter_text}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
        plt.close()
