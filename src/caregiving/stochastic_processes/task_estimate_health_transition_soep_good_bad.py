"""SOEP health transitions."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import statsmodels.formula.api as smf
from pytask import Product
from scipy.stats import norm  # Import norm from scipy.stats for the Gaussian kernel

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs


@pytask.mark.skip(reason="Estimate parametrically (see below)")
def task_estimate_health_transitions_nonparametric(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_health_sample: Path = BLD
    / "data"
    / "health_transition_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix_nonparametric.csv",
):
    """Estimate the health state transition matrix non-parametrically."""

    specs = read_and_derive_specs(path_to_specs)

    transition_data = pd.read_pickle(path_to_health_sample)

    # Define the Epanechnikov kernel function
    def epanechnikov_kernel(distance, bandwidth):
        u = distance / bandwidth
        return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

    # Define the Gaussian kernel function
    def gaussian_kernel(distance, bandwidth):
        # Scale the distance by the bandwidth and use the Gaussian pdf
        return norm.pdf(distance / bandwidth)

    # Function to calculate the weighted mean using a specified kernel
    def kernel_weighted_mean(df, target_age, bandwidth, kernel_type):
        age_distances = np.abs(df["age"] - target_age)
        if kernel_type == "epanechnikov":
            weights = epanechnikov_kernel(age_distances, bandwidth)
        elif kernel_type == "gaussian":
            weights = gaussian_kernel(age_distances, bandwidth)
        else:
            raise ValueError("Invalid kernel type. Use 'epanechnikov' or 'gaussian'.")

        return np.sum(weights * df["lead_health"]) / np.sum(weights)

    # Parameters
    kernel_type = specs.get(
        "health_kernel_type", "epanechnikov"
    )  # Default to Epanechnikov
    bandwidth = specs["health_smoothing_bandwidth"]

    # Adjust bandwidth for Gaussian kernel to ensure the desired probability mass
    if kernel_type == "gaussian":
        # Compute the bandwidth such that the Gaussian CDF from -infinity to -5
        # is approximately 1%
        bandwidth = bandwidth / norm.ppf(0.99)

    ages = np.arange(specs["start_age"], specs["end_age"] + 1)

    # Calculate the smoothed probabilities for each education level and
    # health transition to transition to the lead_health
    def calculate_smoothed_probabilities(education, health):
        smoothed_values = [
            kernel_weighted_mean(
                transition_data[
                    (transition_data["education"] == education)
                    & (transition_data["health"] == health)
                ],
                age,
                bandwidth,
                kernel_type,
            )
            for age in ages
        ]
        return pd.Series(smoothed_values, index=ages)

    # Compute transition probabilities
    transition_probabilities = {
        "hgg_h": calculate_smoothed_probabilities(education=1, health=1),
        "hgg_l": calculate_smoothed_probabilities(education=0, health=1),
        "hbg_h": calculate_smoothed_probabilities(education=1, health=0),
        "hbg_l": calculate_smoothed_probabilities(education=0, health=0),
    }

    # Complementary probabilities
    transition_probabilities.update(
        {
            "hgb_h": 1 - transition_probabilities["hgg_h"],
            "hgb_l": 1 - transition_probabilities["hgg_l"],
            "hbb_h": 1 - transition_probabilities["hbg_h"],
            "hbb_l": 1 - transition_probabilities["hbg_l"],
        }
    )

    # Construct the health transition matrix
    rows = []
    for education in (1, 0):
        for health in (1, 0):
            for lead_health, prob_key in zip(
                [1, 0], ["hgg", "hgb"] if health else ["hbg", "hbb"], strict=False
            ):
                key = f"{prob_key}_{'h' if education == 1 else 'l'}"
                rows.append(
                    {
                        "education": education,
                        "period": ages - specs["start_age"],
                        "health": health,
                        "lead_health": lead_health,
                        "transition_prob": transition_probabilities[key],
                    }
                )

    health_transition_matrix = pd.concat(
        [pd.DataFrame(row) for row in rows], ignore_index=True
    )
    health_transition_matrix.to_csv(path_to_save)


def task_estimate_health_transitions_parametric(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_health_sample: Path = BLD
    / "data"
    / "health_transition_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix.csv",
):
    """Estimate the health state transition with logit regression model."""

    specs = read_and_derive_specs(path_to_specs)

    transition_data = pd.read_pickle(path_to_health_sample)

    # Parameters
    ages = np.arange(specs["start_age"], specs["end_age"] + 1)

    alive_health_vars = specs["alive_health_vars"]
    alive_health_labels = [specs["health_labels"][i] for i in alive_health_vars]

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
                alive_health_label = specs["health_labels"][alive_health_var]

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
                model = smf.logit(formula=formula, data=data)
                result = model.fit()

                # Compute the transition probabilities
                transition_probabilities = result.predict(
                    pd.DataFrame({"age": ages})
                ).values

                # For transition to medium healthy, we take the transition probabilities
                good_label = specs["health_labels"][1]

                health_transition_matrix.loc[
                    (
                        sex_label,
                        edu_label,
                        slice(None),
                        alive_health_label,
                        good_label,
                    ),
                    "transition_prob",
                ] = transition_probabilities

                # For transition to unhealthy, we simply take the complement
                # of the transition to healthy
                bad_label = specs["health_labels"][0]

                health_transition_matrix.loc[
                    (
                        sex_label,
                        edu_label,
                        slice(None),
                        alive_health_label,
                        bad_label,
                    ),
                    "transition_prob",
                ] = (
                    1 - transition_probabilities
                )

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
    / "estimated_health_transition_probabilities.png",
):
    """
    Plot all health → health transitions, colored by education level and with different
    line styles for each (prev_health → next_health) combination:

      - Good Health → Good Health: solid
      - Good Health → Bad Health: dashed
      - Bad Health → Good Health: dash-dot
      - Bad Health → Bad Health: dotted

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

    # Define a mapping from (prev_health, next_health) to line styles
    linestyle_map = {
        ("Good Health", "Good Health"): "-",
        ("Good Health", "Bad Health"): "--",
        ("Bad Health", "Good Health"): "-.",
        ("Bad Health", "Bad Health"): ":",
    }

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

                    # -- COLOR: determined by education level
                    color = JET_COLOR_MAP[edu_idx]

                    # -- LINESTYLE: determined by (prev_health, next_health)
                    if (prev_health, next_health) in linestyle_map:
                        linestyle = linestyle_map[(prev_health, next_health)]
                    else:
                        linestyle = "-"

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
    # plt.show()
    plt.close(fig)
