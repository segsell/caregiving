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
    / "health_transition_estimation_sample_three_states.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix_three_states.csv",
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
            specs["education_labels"],
            ages - specs["start_age"],
            alive_health_labels,
            alive_health_labels,
        ],
        names=["sex", "education", "period", "health", "lead_health"],
    )

    # Compute transition probabilities
    health_transition_matrix = pd.DataFrame(
        index=index, data=None, columns=["transition_prob"]
    )
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            for alive_health_var in alive_health_vars:
                alive_health_label = specs["health_labels_three"][alive_health_var]

                # Filter the data
                data = transition_data[
                    (transition_data["sex"] == sex_var)
                    & (transition_data["education"] == edu_var)
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
                        edu_label,
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
                        edu_label,
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
                        edu_label,
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
    / "health_transition_matrix.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_health_transition_probabilities_good_medium_bad.png",
):
    """
    Plot all health → health transitions, with:
      - different colors if the previous state = Bad Health,
      - dashed lines if the next state = Bad Health,
      - neutral color if the previous state = Good Health.

    This version includes transitions:
      Bad Health → Bad Health
      Bad Health → Good Health
      Good Health → Bad Health
      Good Health → Good Health

    The function assumes that `health_transition_matrix.csv` has columns:
      - sex
      - education
      - period
      - health
      - lead_health
      - transition_prob
      (Optionally: 'age' = period + start_age, or we can create it.)
    """

    # 1. Read specs
    specs = read_and_derive_specs(path_to_specs)
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    sex_labels = specs["sex_labels"]
    edu_labels = specs["education_labels"]
    health_labels = specs["health_labels"]

    # 2. Identify alive health states (not "Death")
    alive_health_states = [h for h in health_labels if h != "Death"]

    # 3. Load the data
    df = pd.read_csv(path_to_health_transition_matrix)
    # Ensure we have an 'age' column (if not already present)
    if "age" not in df.columns:
        df["age"] = df["period"] + start_age

    # 4. Set up figure: one column per sex
    fig, axes = plt.subplots(ncols=len(sex_labels), figsize=(12, 6), sharey=True)

    # 5. Loop over sexes
    for sex_idx, sex_label in enumerate(sex_labels):
        ax = axes[sex_idx] if len(sex_labels) > 1 else axes

        # Subset for this sex
        df_sex = df[df["sex"] == sex_label]

        # Nested loops over previous health, education, next health
        for prev_health in alive_health_states:
            df_prev = df_sex[df_sex["health"] == prev_health]

            for edu_idx, edu_label in enumerate(edu_labels):
                df_edu = df_prev[df_prev["education"] == edu_label]

                for next_health in alive_health_states:
                    df_next = df_edu[df_edu["lead_health"] == next_health].sort_values(
                        "age"
                    )
                    if df_next.empty:
                        continue

                    # --  COLOR LOGIC --
                    # "Use different colors for transitions FROM Bad Health"
                    # so if prev_health == "Bad Health",
                    # pick color = JET_COLOR_MAP[edu_idx]
                    # otherwise, pick a neutral color, e.g., gray.
                    if prev_health == "Bad Health":
                        color = JET_COLOR_MAP[edu_idx]  # color by education
                    else:
                        color = "gray"

                    # --  LINESTYLE LOGIC --
                    # "Keep dashed for transitions TO Bad Health"
                    # so if next_health == "Bad Health", dashed, else solid
                    linestyle = "--" if next_health == "Bad Health" else "-"

                    ax.plot(
                        df_next["age"],
                        df_next["transition_prob"],
                        color=color,
                        linestyle=linestyle,
                        label=f"{edu_label}; {prev_health}→{next_health}",
                    )

        ax.set_xlabel("Age")
        ax.set_xlim(start_age, end_age)
        ax.set_ylabel("Transition Probability")
        ax.set_ylim(0, 1)
        ax.set_title(f"Health Transition Probability for {sex_label}")

        # Show legend only on the first panel or unify outside:
        if sex_idx == 0:
            ax.legend(loc="best", ncol=1)

    fig.tight_layout()
    fig.savefig(path_to_save_plot)
    plt.show()
    plt.close(fig)
