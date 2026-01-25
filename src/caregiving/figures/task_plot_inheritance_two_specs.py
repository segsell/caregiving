"""Plot inheritance specifications with separate light/intensive care variables."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs

# Specification definitions matching two-care estimation tasks
SPECIFICATIONS_TWO_CARE = [
    {
        "name": "spec1_care_parent_this_year",
        "light_var": "light_care",
        "intensive_var": "intensive_care",
        "formal_care_var": "formal_care_costs_dummy",
        "parent_var": "parent_died_this_year",
        "filter": None,
        "title": "Light + Intensive Care This Year + Parent Died This Year",
    },
    {
        "name": "spec2_care_recent_parent_recent",
        "light_var": "light_care_recent",
        "intensive_var": "intensive_care_recent",
        "formal_care_var": "formal_care_costs_dummy_recent",
        "parent_var": "parent_died_recent",
        "filter": None,
        "title": "Light + Intensive Care Recent + Parent Died Recent",
    },
    {
        "name": "spec3_care_last_year_parent_last_year",
        "light_var": "light_care_last_year",
        "intensive_var": "intensive_care_last_year",
        "formal_care_var": "formal_care_costs_dummy_last_year",
        "parent_var": "parent_died_last_year",
        "filter": None,
        "title": "Light + Intensive Care Last Year + Parent Died Last Year",
    },
    {
        "name": "spec4_care_last_year_filter_parent_this_year",
        "light_var": "light_care_last_year",
        "intensive_var": "intensive_care_last_year",
        "formal_care_var": "formal_care_costs_dummy_last_year",
        "parent_var": None,
        "filter": "parent_died_this_year == 1",
        "title": "Light + Intensive Care Last Year (Filter: Parent Died This Year)",
    },
    {
        "name": "spec7_care_this_year_filter_parent_this_year",
        "light_var": "light_care",
        "intensive_var": "intensive_care",
        "formal_care_var": "formal_care_costs_dummy",
        "parent_var": None,
        "filter": "parent_died_this_year == 1",
        "title": "Light + Intensive Care This Year (Filter: Parent Died This Year)",
    },
    {
        "name": "spec10_care_recent_filter_parent_this_year",
        "light_var": "light_care_recent",
        "intensive_var": "intensive_care_recent",
        "formal_care_var": "formal_care_costs_dummy_recent",
        "parent_var": None,
        "filter": "parent_died_this_year == 1",
        "title": "Light + Intensive Care Recent (Filter: Parent Died This Year)",
    },
    {
        "name": "spec5_care_last_year_filter_parent_last_year",
        "light_var": "light_care_last_year",
        "intensive_var": "intensive_care_last_year",
        "formal_care_var": "formal_care_costs_dummy_last_year",
        "parent_var": None,
        "filter": "parent_died_last_year == 1",
        "title": "Light + Intensive Care Last Year (Filter: Parent Died Last Year)",
    },
    {
        "name": "spec8_care_this_year_filter_parent_last_year",
        "light_var": "light_care",
        "intensive_var": "intensive_care",
        "formal_care_var": "formal_care_costs_dummy",
        "parent_var": None,
        "filter": "parent_died_last_year == 1",
        "title": "Light + Intensive Care This Year (Filter: Parent Died Last Year)",
    },
    {
        "name": "spec11_care_recent_filter_parent_last_year",
        "light_var": "light_care_recent",
        "intensive_var": "intensive_care_recent",
        "formal_care_var": "formal_care_costs_dummy_recent",
        "parent_var": None,
        "filter": "parent_died_last_year == 1",
        "title": "Light + Intensive Care Recent (Filter: Parent Died Last Year)",
    },
    {
        "name": "spec6_care_last_year_filter_parent_recent",
        "light_var": "light_care_last_year",
        "intensive_var": "intensive_care_last_year",
        "formal_care_var": "formal_care_costs_dummy_last_year",
        "parent_var": None,
        "filter": "parent_died_recent == 1",
        "title": "Light + Intensive Care Last Year (Filter: Parent Died Recent)",
    },
    {
        "name": "spec9_care_this_year_filter_parent_recent",
        "light_var": "light_care",
        "intensive_var": "intensive_care",
        "formal_care_var": "formal_care_costs_dummy",
        "parent_var": None,
        "filter": "parent_died_recent == 1",
        "title": "Light + Intensive Care This Year (Filter: Parent Died Recent)",
    },
    {
        "name": "spec12_care_recent_filter_parent_recent",
        "light_var": "light_care_recent",
        "intensive_var": "intensive_care_recent",
        "formal_care_var": "formal_care_costs_dummy_recent",
        "parent_var": None,
        "filter": "parent_died_recent == 1",
        "title": "Light + Intensive Care Recent (Filter: Parent Died Recent)",
    },
]


# Create plotting tasks for probability specifications (two-care)
for spec in SPECIFICATIONS_TWO_CARE:

    @pytask.mark.inheritance
    @pytask.task(id=f"{spec['name']}_prob_two_care")
    def task_plot_prob_two_care_specification(
        spec_info=spec,
        path_to_specs: Path = SRC / "specs.yaml",
        path_to_summary: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_specs_two_care"
        / "_specs_summary.txt",
        path_to_params: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_specs_two_care"
        / f"{spec['name']}_params.csv",
        path_to_save: Annotated[Path, Product] = BLD
        / "plots"
        / "stochastic_processes"
        / "inheritance_specs_two_care"
        / f"{spec['name']}.png",
    ):
        """Plot inheritance probability for two-care specification."""
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

                    # Scenario 1: Intensive care only (solid)
                    linear_pred = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["light_var"]] * 0
                        + params[spec_info["intensive_var"]] * 1
                        + params[spec_info["formal_care_var"]] * 0
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        linear_pred += params[spec_info["parent_var"]] * 1

                    prob_intensive = 1 / (1 + np.exp(-linear_pred))

                    ax.plot(
                        ages,
                        prob_intensive,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="-",
                        label=f"{edu_label}, intensive",
                    )

                    # Scenario 2: Light care only (dotted)
                    linear_pred_light = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["light_var"]] * 1
                        + params[spec_info["intensive_var"]] * 0
                        + params[spec_info["formal_care_var"]] * 0
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        linear_pred_light += params[spec_info["parent_var"]] * 1

                    prob_light = 1 / (1 + np.exp(-linear_pred_light))

                    ax.plot(
                        ages,
                        prob_light,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle=":",
                        label=f"{edu_label}, light",
                    )

                    # Scenario 3: No care (dashed)
                    linear_pred_no = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["light_var"]] * 0
                        + params[spec_info["intensive_var"]] * 0
                        + params[spec_info["formal_care_var"]] * 0
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

                    # Scenario 4: Formal care (dash-dot)
                    linear_pred_formal = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["light_var"]] * 0
                        + params[spec_info["intensive_var"]] * 0
                        + params[spec_info["formal_care_var"]] * 1
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        linear_pred_formal += params[spec_info["parent_var"]] * 1

                    prob_formal = 1 / (1 + np.exp(-linear_pred_formal))

                    ax.plot(
                        ages,
                        prob_formal,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle=(0, (3, 1, 1, 1)),
                        label=f"{edu_label}, formal care",
                    )

                except (KeyError, TypeError):
                    continue

            ax.set_xlabel("Age", fontsize=12)
            ax.set_ylabel("Probability of Inheritance", fontsize=12)
            ax.set_title(str(sex_label), fontsize=13, fontweight="bold")
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xlim(40, 80)
            ax.set_ylim(0, None)

        # Add specification info to title
        filter_text = f"\n{spec_info['filter']}" if spec_info["filter"] else ""
        plt.suptitle(
            f"Probability: {spec_info['title']}{filter_text}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
        plt.close()


# Create plotting tasks for amount specifications (two-care)
for spec in SPECIFICATIONS_TWO_CARE:

    @pytask.mark.inheritance
    @pytask.task(id=f"{spec['name']}_amount_two_care")
    def task_plot_amount_two_care_specification(
        spec_info=spec,
        path_to_specs: Path = SRC / "specs.yaml",
        path_to_summary: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_amount_specs_two_care"
        / "_specs_summary.txt",
        path_to_params: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_amount_specs_two_care"
        / f"{spec['name']}_params.csv",
        path_to_save: Annotated[Path, Product] = BLD
        / "plots"
        / "stochastic_processes"
        / "inheritance_amount_specs_two_care"
        / f"{spec['name']}.png",
    ):
        """Plot inheritance amount for two-care specification.

        With formal care costs dummy included.

        """

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

                    # Scenario 1: Intensive care only (solid)
                    ln_amount = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["light_var"]] * 0
                        + params[spec_info["intensive_var"]] * 1
                        + params[spec_info["formal_care_var"]] * 0
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        ln_amount += params[spec_info["parent_var"]] * 1

                    # Exponentiate to get actual amount
                    amount_intensive = np.exp(ln_amount)

                    ax.plot(
                        ages,
                        amount_intensive,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="-",
                        label=f"{edu_label}, intensive",
                    )

                    # Scenario 2: Light care only (dotted)
                    ln_amount_light = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["light_var"]] * 1
                        + params[spec_info["intensive_var"]] * 0
                        + params[spec_info["formal_care_var"]] * 0
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        ln_amount_light += params[spec_info["parent_var"]] * 1

                    amount_light = np.exp(ln_amount_light)

                    ax.plot(
                        ages,
                        amount_light,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle=":",
                        label=f"{edu_label}, light",
                    )

                    # Scenario 3: No care (dashed)
                    ln_amount_no = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["light_var"]] * 0
                        + params[spec_info["intensive_var"]] * 0
                        + params[spec_info["formal_care_var"]] * 0
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        ln_amount_no += params[spec_info["parent_var"]] * 1

                    amount_no = np.exp(ln_amount_no)

                    ax.plot(
                        ages,
                        amount_no,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="--",
                        label=f"{edu_label}, no care",
                    )

                    # Scenario 4: Formal care (dash-dot)
                    ln_amount_formal = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params[spec_info["light_var"]] * 0
                        + params[spec_info["intensive_var"]] * 0
                        + params[spec_info["formal_care_var"]] * 1
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        ln_amount_formal += params[spec_info["parent_var"]] * 1

                    amount_formal = np.exp(ln_amount_formal)

                    ax.plot(
                        ages,
                        amount_formal,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle=(0, (3, 1, 1, 1)),
                        label=f"{edu_label}, formal care",
                    )

                except (KeyError, TypeError):
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

        # Add specification info to title
        filter_text = f"\n{spec_info['filter']}" if spec_info["filter"] else ""
        plt.suptitle(
            f"Amount: {spec_info['title']}{filter_text}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
        plt.close()
