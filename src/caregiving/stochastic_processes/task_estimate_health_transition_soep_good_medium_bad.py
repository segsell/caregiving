"""SOEP health transitions: good, medium, bad."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import statsmodels.formula.api as smf
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs


def task_estimate_health_transitions_parametric(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_health_sample: Path = BLD
    / "data"
    / "health_transition_estimation_sample_good_medium_bad.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix_good_medium_bad.csv",
):
    """Estimate the health state transition with logit regression model."""

    specs = read_and_derive_specs(path_to_specs)

    transition_data = pd.read_pickle(path_to_health_sample)

    # Parameters
    ages = np.arange(specs["start_age"], specs["end_age"] + 1)

    alive_health_vars = specs["alive_health_vars_three"]
    alive_health_labels = [specs["health_labels_three"][i] for i in alive_health_vars]

    index = pd.MultiIndex.from_product(
        [
            specs["sex_labels"],
            ages - specs["start_age"],
            alive_health_labels,
            alive_health_labels,
        ],
        names=["sex", "period", "health", "lead_health"],
    )

    # Compute transition probabilities
    health_transition_matrix = pd.DataFrame(
        index=index, data=None, columns=["transition_prob"]
    )
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for alive_health_var in alive_health_vars:
            alive_health_label = specs["health_labels_three"][alive_health_var]

            # Filter the data
            data = transition_data[
                (transition_data["sex"] == sex_var)
                & (transition_data["health"] == alive_health_var)
            ]

            # Fit the logit model
            y_var = "lead_health"
            x_vars = ["age"]
            formula = y_var + " ~ " + " + ".join(x_vars)
            model = smf.mnlogit(formula=formula, data=data)
            result = model.fit()

            # Compute the transition probabilities
            transition_probabilities = result.predict(
                pd.DataFrame({"age": ages})
            ).values

            # Take the transition probabilities
            medium_label = specs["health_labels_three"][1]
            health_transition_matrix.loc[
                (
                    sex_label,
                    slice(None),
                    alive_health_label,
                    medium_label,
                ),
                "transition_prob",
            ] = transition_probabilities[:, 1]

            good_label = specs["health_labels_three"][2]
            health_transition_matrix.loc[
                (
                    sex_label,
                    slice(None),
                    alive_health_label,
                    good_label,
                ),
                "transition_prob",
            ] = transition_probabilities[:, 2]

            bad_label = specs["health_labels_three"][0]
            health_transition_matrix.loc[
                (
                    sex_label,
                    slice(None),
                    alive_health_label,
                    bad_label,
                ),
                "transition_prob",
            ] = transition_probabilities[:, 0]

    health_transition_matrix.to_csv(path_to_save)


# =====================================================================================
# Plotting
# =====================================================================================


def task_plot_health_transitions(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_health_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix_good_medium_bad.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_health_transition_probabilities_good_medium_bad.png",
):
    """Plot transition probabilities for good, medium and bad health by gender."""

    # 1. Load specifications
    specs = read_and_derive_specs(path_to_specs)
    sex_labels = specs["sex_labels"]

    start_age = specs["start_age"]
    end_age = specs["end_age"]
    health_labels = specs["health_labels_three"]

    alive_health_states = [h for h in health_labels if h != "Death"]

    # 2. Load transition data
    df = pd.read_csv(path_to_health_transition_matrix)

    # Create age column if needed
    if "age" not in df.columns:
        df["age"] = df["period"] + start_age

    # 3. Setup mappings
    color_map = {
        "Bad Health": JET_COLOR_MAP[1],
        "Medium Health": JET_COLOR_MAP[0],
        "Good Health": JET_COLOR_MAP[2],
    }
    linestyle_map = {
        "Bad Health": ":",
        "Medium Health": "--",
        "Good Health": "-",
    }

    # 4. Create plot
    fig, axes = plt.subplots(ncols=len(sex_labels), figsize=(14, 6), sharey=True)

    for sex_idx, sex in enumerate(sex_labels):
        ax = axes[sex_idx] if len(sex_labels) > 1 else axes

        df_sex = df[df["sex"] == sex]

        for prev_health in alive_health_states:
            df_prev = df_sex[df_sex["health"] == prev_health]

            for next_health in alive_health_states:
                df_transition = df_prev[
                    df_prev["lead_health"] == next_health
                ].sort_values("age")

                if df_transition.empty:
                    continue

                ax.plot(
                    df_transition["age"],
                    df_transition["transition_prob"],
                    color=color_map[next_health],
                    linestyle=linestyle_map[prev_health],
                    linewidth=2,
                    label=f"{prev_health} → {next_health}",
                )

        ax.set_title(f"Health Transitions ({sex})")
        ax.set_xlabel("Age")
        ax.set_ylabel("Transition Probability")
        ax.set_xlim(start_age, end_age)
        ax.set_ylim(0, 1)

        if sex_idx == 0:
            ax.legend(title="Transitions", fontsize=9, title_fontsize=10)

    plt.tight_layout()
    fig.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)
