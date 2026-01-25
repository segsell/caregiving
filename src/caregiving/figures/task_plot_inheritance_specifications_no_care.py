"""Plot inheritance probability specifications without care variables."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs

# Specification definitions matching task_estimate_inheritance_specifications_no_care
SPECIFICATIONS = [
    {
        "name": "spec1_no_care_parent_this_year",
        "parent_var": "parent_died_this_year",
        "filter": None,
        "title": "Parent Died This Year",
    },
    {
        "name": "spec2_no_care_parent_recent",
        "parent_var": "parent_died_recent",
        "filter": None,
        "title": "Parent Died Recent",
    },
    {
        "name": "spec3_no_care_parent_last_year",
        "parent_var": "parent_died_last_year",
        "filter": None,
        "title": "Parent Died Last Year",
    },
    {
        "name": "spec4_no_care_filter_parent_this_year",
        "parent_var": None,
        "filter": "parent_died_this_year == 1",
        "title": "Filter: Parent Died This Year",
    },
    {
        "name": "spec5_no_care_filter_parent_last_year",
        "parent_var": None,
        "filter": "parent_died_last_year == 1",
        "title": "Filter: Parent Died Last Year",
    },
    {
        "name": "spec6_no_care_filter_parent_recent",
        "parent_var": None,
        "filter": "parent_died_recent == 1",
        "title": "Filter: Parent Died Recent",
    },
]


# Create individual plotting tasks for each specification
for spec in SPECIFICATIONS:

    @pytask.mark.inheritance
    @pytask.task(id=f"{spec['name']}_no_care")
    def task_plot_specification_no_care(
        spec_info=spec,
        path_to_specs: Path = SRC / "specs.yaml",
        path_to_summary: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_specs_no_care"
        / "_specs_summary.txt",
        path_to_params: Path = BLD
        / "estimation"
        / "stochastic_processes"
        / "inheritance_specs_no_care"
        / f"{spec['name']}_params.csv",
        path_to_save: Annotated[Path, Product] = BLD
        / "plots"
        / "stochastic_processes"
        / "inheritance_specs_no_care"
        / f"{spec['name']}.png",
    ):
        """Plot inheritance probability for no-care specification."""
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

            # Plot for each education level (low and high)
            for edu_var, edu_label in enumerate(specs_dict["education_labels"]):
                try:
                    params = spec_params.loc[sex_label]
                    age_sq = ages**2

                    # Calculate probability (no care variables, only education)
                    linear_pred = (
                        params["const"]
                        + params["age"] * ages
                        + params["age_sq"] * age_sq
                        + params["education"] * edu_var
                    )
                    if spec_info["parent_var"] is not None:
                        linear_pred += params[spec_info["parent_var"]] * 1

                    prob = 1 / (1 + np.exp(-linear_pred))

                    ax.plot(
                        ages,
                        prob,
                        linewidth=2.5,
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="-",
                        label=edu_label,
                    )

                except (KeyError, TypeError):
                    continue

            ax.set_xlabel("Age", fontsize=12)
            ax.set_ylabel("Probability of Inheritance", fontsize=12)
            ax.set_title(str(sex_label), fontsize=13, fontweight="bold")
            ax.legend(loc="best", fontsize=10)
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
