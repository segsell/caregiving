"""Estimate the mortality matrix given two health states."""

import itertools
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs


def task_estimate_mortality_logit_good_bad(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_lifetable: Path = SRC
    / "data"
    / "statistical_office"
    / "mortality_table_for_pandas.csv",
    path_to_mortatility_sample: Path = BLD
    / "data"
    / "mortality_transition_estimation_sample_duplicated.pkl",
    path_to_save_mortality_params_men: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_params_men_logit.csv",
    path_to_save_mortality_params_women: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_params_women_logit.csv",
    path_to_save_mortality_transition_matrix: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix_logit.csv",
    path_to_save_lifetable: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "lifetable_logit.csv",
):
    """Estimate the mortality matrix for two health states and two education levels."""

    specs = read_and_derive_specs(path_to_specs)

    # Load life table data and expand/duplicate it to include all
    # possible combinations of health, education and sex
    lifetable_df = pd.read_csv(
        path_to_lifetable,
        sep=";",
    )

    mortality_df = pd.DataFrame(
        [
            {
                "age": row["age"],
                "health": combo[0],
                "education": combo[1],
                "sex": combo[2],
                "death_prob": (
                    row["death_prob_male"]
                    if combo[2] == 0
                    else row["death_prob_female"]
                ),  # male (0) or female (1) death prob
            }
            for _, row in lifetable_df.iterrows()
            for combo in list(
                itertools.product([0, 1], repeat=3)
            )  # (health, education, sex)
        ]
    )
    mortality_df.reset_index(drop=True, inplace=True)

    # Plain life table data
    lifetable_df = mortality_df[["age", "sex", "death_prob"]].copy()
    lifetable_df.drop_duplicates(inplace=True)

    # Estimation sample - as in Kroll Lampert 2008 / Haan Schaller et al. 2024
    df = pd.read_pickle(path_to_mortatility_sample)

    # Df initial values i.e. first observations (+ sex column)
    start_df = df[
        [col for col in df.columns if col.startswith("start")] + ["sex"]
    ].copy()
    str_cols = start_df.columns.str.replace("start ", "")
    start_df.columns = str_cols

    # Add a intercept column to the df and start_df
    df["intercept"] = 1
    start_df["intercept"] = 1

    for sex, sex_label in enumerate(specs["sex_labels"]):

        if sex_label.lower() == "men":
            path_to_save_params = path_to_save_mortality_params_men
        else:
            path_to_save_params = path_to_save_mortality_params_women

        # Filter data by sex
        filtered_df = df[df["sex"] == sex]

        exog_cols = [
            "intercept",
            "age",
            f"{specs['health_labels'][0]} {specs['education_labels'][0]}",
            f"{specs['health_labels'][0]} {specs['education_labels'][1]}",
            f"{specs['health_labels'][1]} {specs['education_labels'][0]}",
            f"{specs['health_labels'][1]} {specs['education_labels'][1]}",
        ]
        endog = filtered_df["death event"]
        exog = filtered_df[exog_cols]

        model = sm.Logit(endog, exog)
        res = model.fit(disp=True)  # disp=True prints iteration messages

        print(res)
        print(res.params)

        to_csv_summary = res.params.copy()
        to_csv_summary.to_csv(path_to_save_params)

        # update mortality_df with the estimated parameters
        for health, health_label in enumerate(
            specs["health_labels"][:-1]
        ):  # exclude the last health label (death)
            for education, education_label in enumerate(specs["education_labels"]):
                param = f"{health_label} {education_label}"
                mortality_df.loc[
                    (mortality_df["sex"] == sex)
                    & (mortality_df["health"] == health)
                    & (mortality_df["education"] == education),
                    "death_prob",
                ] *= np.exp(res.params.loc[param])

    # Export the estimated mortality table and the original life table as csv
    lifetable_df = lifetable_df[
        (lifetable_df["age"] >= specs["start_age_mortality"])
        & (lifetable_df["age"] <= specs["end_age_mortality"])
    ]
    mortality_df = mortality_df[
        (mortality_df["age"] >= specs["start_age_mortality"])
        & (mortality_df["age"] <= specs["end_age_mortality"])
    ]
    mortality_df = mortality_df[["age", "sex", "health", "education", "death_prob"]]
    lifetable_df = lifetable_df[["age", "sex", "death_prob"]]
    mortality_df = mortality_df.astype(
        {
            "age": "int",
            "sex": "int",
            "health": "int",
            "education": "int",
            "death_prob": "float",
        }
    )
    lifetable_df = lifetable_df.astype(
        {
            "age": "int",
            "sex": "int",
            "death_prob": "float",
        }
    )
    mortality_df.to_csv(
        path_to_save_mortality_transition_matrix,
        sep=",",
        index=False,
    )
    lifetable_df.to_csv(
        path_to_save_lifetable,
        sep=",",
        index=False,
    )


def task_estimate_mortality_logit_good_medium_bad(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_lifetable: Path = SRC
    / "data"
    / "statistical_office"
    / "mortality_table_for_pandas.csv",
    path_to_mortatility_sample: Path = BLD
    / "data"
    / "mortality_transition_estimation_sample_good_medium_bad_duplicated.pkl",
    path_to_save_mortality_params_men: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_params_men_logit_good_medium_bad.csv",
    path_to_save_mortality_params_women: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_params_women_logit_good_medium_bad.csv",
    path_to_save_mortality_transition_matrix: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix_logit_good_medium_bad.csv",
    path_to_save_lifetable: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "lifetable_logit_good_medium_bad.csv",
):
    """Estimate the mortality matrix for three health states and no education."""

    specs = read_and_derive_specs(path_to_specs)

    lifetable_df = pd.read_csv(path_to_lifetable, sep=";")

    mortality_df = pd.DataFrame(
        [
            {
                "age": row["age"],
                "health": health,
                "sex": sex,
                "death_prob": (
                    row["death_prob_male"] if sex == 0 else row["death_prob_female"]
                ),
            }
            for _, row in lifetable_df.iterrows()
            for health in (0, 1, 2)  # 0=bad, 1=medium, 2=good
            for sex in (0, 1)
        ]
    )

    mortality_df.reset_index(drop=True, inplace=True)

    lifetable_df = mortality_df[["age", "sex", "death_prob"]].drop_duplicates()

    df = pd.read_pickle(path_to_mortatility_sample)
    df["intercept"] = 1

    health_labels = specs["health_labels_three"]

    for sex, sex_label in enumerate(specs["sex_labels"]):

        path_to_save_params = (
            path_to_save_mortality_params_men
            if sex_label.lower() == "men"
            else path_to_save_mortality_params_women
        )

        filtered_df = df[df["sex"] == sex]

        exog_cols = ["intercept", "age"] + health_labels[
            :-1
        ]  # exclude the last health label (death)

        endog = filtered_df["death event"]
        exog = filtered_df[exog_cols]

        model = sm.Logit(endog, exog)
        res = model.fit(disp=True)

        print(res.summary())

        res.params.to_csv(path_to_save_params)

        for health_idx, health_label in enumerate(health_labels[:-1]):
            param = health_label
            mortality_df.loc[
                (mortality_df["sex"] == sex) & (mortality_df["health"] == health_idx),
                "death_prob",
            ] *= np.exp(res.params[param])

    lifetable_df = lifetable_df[
        (lifetable_df["age"] >= specs["start_age_mortality"])
        & (lifetable_df["age"] <= specs["end_age_mortality"])
    ]

    mortality_df = mortality_df[
        (mortality_df["age"] >= specs["start_age_mortality"])
        & (mortality_df["age"] <= specs["end_age_mortality"])
    ]

    mortality_df.to_csv(path_to_save_mortality_transition_matrix, index=False)
    lifetable_df.to_csv(path_to_save_lifetable, index=False)


# =====================================================================================
# Plotting
# =====================================================================================


def task_plot_mortality_good_bad(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_mortality_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix_logit.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_survival_probabilities_logit.png",
):
    """Plot mortality characteristics."""

    specs = read_and_derive_specs(path_to_specs)

    # Load the data
    estimated_mortality = pd.read_csv(path_to_mortality_transition_matrix)

    _n_edu_types = len(specs["education_labels"])
    _n_ages = specs["end_age_mortality"] - specs["start_age_mortality"] + 1
    _n_alive_health_states = len(specs["health_labels"]) - 1
    alive_health_states = np.where(np.array(specs["health_labels"]) != "Death")[0]
    _health_states = [specs["health_labels"][i] for i in alive_health_states]
    _education_states = specs["education_labels"]
    _age_range = np.arange(specs["start_age_mortality"], specs["end_age_mortality"] + 1)

    estimated_mortality["survival_prob_year"] = np.nan
    estimated_mortality["survival_prob"] = np.nan

    for sex_var, _sex_label in enumerate(specs["sex_labels"]):
        for health in alive_health_states:
            for edu_var, _edu_label in enumerate(specs["education_labels"]):
                mask = (
                    (estimated_mortality["sex"] == sex_var)
                    & (estimated_mortality["health"] == health)
                    & (estimated_mortality["education"] == edu_var)
                )

                filtered_data = estimated_mortality.loc[
                    mask,
                    ["death_prob", "age"],
                ].sort_values(by="age")

                filtered_data["survival_prob_year"] = 1 - filtered_data["death_prob"]
                filtered_data["survival_prob"] = filtered_data[
                    "survival_prob_year"
                ].cumprod()
                filtered_data["survival_prob"] = filtered_data["survival_prob"].shift(1)
                filtered_data.loc[0, "survival_prob"] = 1

                estimated_mortality.update(filtered_data)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 8))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axes[sex_var]

        for edu_var, edu_label in enumerate(specs["education_labels"]):
            for health in alive_health_states:
                mask = (
                    (estimated_mortality["sex"] == sex_var)
                    & (estimated_mortality["health"] == health)
                    & (estimated_mortality["education"] == edu_var)
                )
                health_label = specs["health_labels"][health]
                linestyle = "--" if health == 0 else "-"

                ax.plot(
                    estimated_mortality.loc[mask, "age"],
                    estimated_mortality.loc[mask, "survival_prob"],
                    color=JET_COLOR_MAP[edu_var],
                    label=f"{edu_label}; {health_label}",
                    linestyle=linestyle,
                )

        ax.set_xlabel("Age")
        ax.set_xlim(specs["start_age"], specs["end_age"] + 1)
        ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1)
        ax.set_title(f"Estimated Survival Probability for {sex_label}")

    axes[0].legend(loc="lower left")
    fig.savefig(path_to_save_plot)
    plt.close(fig)


def task_plot_mortality_good_medium_bad(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_mortality_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix_logit_good_medium_bad.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_survival_probabilities_logit_good_medium_bad.png",
):
    """Plot mortality characteristics."""

    specs = read_and_derive_specs(path_to_specs)
    health_labels = "health_labels_three"

    # Load the data
    estimated_mortality = pd.read_csv(path_to_mortality_transition_matrix)

    _n_edu_types = len(specs["education_labels"])
    _n_ages = specs["end_age_mortality"] - specs["start_age_mortality"] + 1
    _n_alive_health_states = len(specs[health_labels]) - 1
    alive_health_states = np.where(np.array(specs[health_labels]) != "Death")[0]
    _health_states = [specs[health_labels][i] for i in alive_health_states]
    _education_states = specs["education_labels"]
    _age_range = np.arange(specs["start_age_mortality"], specs["end_age_mortality"] + 1)

    estimated_mortality["survival_prob_year"] = np.nan
    estimated_mortality["survival_prob"] = np.nan

    for sex_var, _sex_label in enumerate(specs["sex_labels"]):
        for health in alive_health_states:
            mask = (estimated_mortality["sex"] == sex_var) & (
                estimated_mortality["health"] == health
            )

            filtered_data = estimated_mortality.loc[
                mask,
                ["death_prob", "age"],
            ].sort_values(by="age")

            filtered_data["survival_prob_year"] = 1 - filtered_data["death_prob"]
            filtered_data["survival_prob"] = filtered_data[
                "survival_prob_year"
            ].cumprod()
            filtered_data["survival_prob"] = filtered_data["survival_prob"].shift(1)
            filtered_data.loc[0, "survival_prob"] = 1

            estimated_mortality.update(filtered_data)

    # Setup mappings
    color_map = {
        "Bad Health": JET_COLOR_MAP[1],
        "Medium Health": JET_COLOR_MAP[0],
        "Good Health": JET_COLOR_MAP[2],
    }

    fig, axes = plt.subplots(ncols=2, figsize=(12, 8))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axes[sex_var]
        for health in alive_health_states:
            mask = (estimated_mortality["sex"] == sex_var) & (
                estimated_mortality["health"] == health
            )
            health_label = specs[health_labels][health]

            ax.plot(
                estimated_mortality.loc[mask, "age"],
                estimated_mortality.loc[mask, "survival_prob"],
                label=str(health_label),
                color=color_map[health_label],
                linestyle="-",
            )

        ax.set_xlabel("Age")
        ax.set_xlim(specs["start_age"], specs["end_age"] + 1)
        ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1)
        ax.set_title(f"Estimated Survival Probability for {sex_label}")

    axes[0].legend(loc="lower left")
    fig.savefig(path_to_save_plot)
    plt.close(fig)


def _survival_function(age, health_factors, params):
    """
    Calculates the survival function: Exp(-(integral of the hazard function
    as a function of age from 0 to age)).

    Parameters:
        age (float or array-like): The age(s) at which to calculate the function.
        health_factors (dict): Keys are health-education categories
            (e.g., 'health1_edu1') and values are their indicator variables (0 or 1).
        params (DataFrame): Model parameters, with 'intercept' and coefficient names
                            in the index and their values in a column named 'value'.

    Returns:

        float or array-like: The value(s) of the survival function.
    """
    coefficients = params["value"]
    intercept = coefficients["intercept"]
    age_coef = coefficients["age"]

    lambda_ = np.exp(
        intercept
        + sum(
            coefficients[key] * value
            for key, value in health_factors.items()
            if key != "intercept"
        )
    )

    age_contrib = np.exp(age_coef * age) - 1

    return np.exp(-lambda_ / age_coef * age_contrib)
