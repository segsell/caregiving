"""SOEP jealth transitions."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import optimagic as om
import pandas as pd
import pytask
import statsmodels.formula.api as smf
from pytask import Product
from scipy.stats import norm  # Import norm from scipy.stats for the Gaussian kernel

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs


@pytask.mark.skip(reason="Estimate parametrically (see below)")
def task_estimate_health_transitions(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_health_sample: Path = BLD
    / "data"
    / "health_transition_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix_nonparametric.csv",
):
    """Estimate the health state transition matrix."""

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

    # Save the results to a CSV file
    health_transition_matrix.to_csv(path_to_save)

    return health_transition_matrix


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

                # For transition to healthy, we take the transition probabilities
                healthy_label = specs["health_labels"][1]

                health_transition_matrix.loc[
                    (
                        sex_label,
                        edu_label,
                        slice(None),
                        alive_health_label,
                        healthy_label,
                    ),
                    "transition_prob",
                ] = transition_probabilities

                # For transition to unhealthy, we simply take the complement
                # of the transition to healthy
                unhealthy_label = specs["health_labels"][0]

                health_transition_matrix.loc[
                    (
                        sex_label,
                        edu_label,
                        slice(None),
                        alive_health_label,
                        unhealthy_label,
                    ),
                    "transition_prob",
                ] = (
                    1 - transition_probabilities
                )

    # Save the results to a CSV file
    health_transition_matrix.to_csv(path_to_save)
