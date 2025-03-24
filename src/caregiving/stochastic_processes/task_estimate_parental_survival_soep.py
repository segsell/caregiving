import itertools
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import optimagic as om
import pandas as pd
from linearmodels.panel.model import PanelOLS
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs
from caregiving.stochastic_processes.auxiliary import loglike


def task_estimate_mortality(
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
    / "mortality_params_men.csv",
    path_to_save_mortality_params_women: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_params_women.csv",
    path_to_save_mortality_transition_matrix: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix.csv",
    path_to_save_lifetable: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "lifetable.csv",
):
    """Estimate the mortality matrix."""

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
        filtered_start_df = start_df[start_df["sex"] == sex]

        # Initial parameters
        initial_params_data = {
            "intercept": {
                "value": -13,
                "lower_bound": -np.inf,
                "upper_bound": np.inf,
                "soft_lower_bound": -15.0,
                "soft_upper_bound": 15.0,
            },
            "age": {
                "value": 0.1,
                "lower_bound": 1e-8,
                "upper_bound": np.inf,
                "soft_lower_bound": 0.0001,
                "soft_upper_bound": 1.0,
            },
            f"{specs['health_labels'][1]} {specs['education_labels'][1]}": {
                "value": -0.4,
                "lower_bound": -np.inf,
                "upper_bound": np.inf,
                "soft_lower_bound": -2.5,
                "soft_upper_bound": 2.5,
            },
            f"{specs['health_labels'][1]} {specs['education_labels'][0]}": {
                "value": -0.3,
                "lower_bound": -np.inf,
                "upper_bound": np.inf,
                "soft_lower_bound": -2.5,
                "soft_upper_bound": 2.5,
            },
            f"{specs['health_labels'][0]} {specs['education_labels'][1]}": {
                "value": 0.0,
                "lower_bound": -np.inf,
                "upper_bound": np.inf,
                "soft_lower_bound": -2.5,
                "soft_upper_bound": 2.5,
            },
            f"{specs['health_labels'][0]} {specs['education_labels'][0]}": {
                "value": 0.2,
                "lower_bound": -np.inf,
                "upper_bound": np.inf,
                "soft_lower_bound": -2.5,
                "soft_upper_bound": 2.5,
            },
        }
        initial_params = pd.DataFrame.from_dict(initial_params_data, orient="index")

        # Estimate parameters
        res = om.maximize(
            fun=loglike,
            params=initial_params,
            algorithm="scipy_lbfgsb",
            fun_kwargs={"data": filtered_df, "start_data": filtered_start_df},
            numdiff_options=om.NumdiffOptions(n_cores=4),
            multistart=om.MultistartOptions(n_samples=100, seed=0, n_cores=4),
        )

        # terminal log the results
        print(res)
        print(res.params)

        # save the results
        to_csv_summary = res.params.copy()
        to_csv_summary["hazard_ratio"] = np.exp(to_csv_summary["value"])
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
                ] *= np.exp(res.params.loc[param, "value"])

    # export the estimated mortality table and the original life table as csv
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


# def loglike(params, data, start_data):
#     """Log-likelihood calculation.

#     params: pd.DataFrame
#         DataFrame with the parameters.
#     data: pd.DataFrame
#         DataFrame with the data.
#     data: pd.DataFrame
#         DataFrame with the start data.

#     """

#     event = data["death event"]
#     death = event.astype(bool)

#     return (
#         _log_density_function(data[death], params).sum()
#         + _log_survival_function(data[~death], params).sum()
#         - _log_survival_function(start_data, params).sum()
#     )


# def _log_density_function(data, params):
#     """Calculate the log-density function.

#     Log of the density function:

#        (log of d[-S(age)]/d(age) = log of - dS(age)/d(age))

#     """
#     age_coef = params.loc["age", "value"]
#     age_contrib = np.exp(age_coef * data["age"]) - 1

#     log_lambda_ = sum(
#         [params.loc[x, "value"] * data[x] for x in params.index if x != "age"]
#     )
#     lambda_ = np.exp(log_lambda_)

#     return log_lambda_ + age_coef * data["age"] - ((lambda_ * age_contrib) / age_coef)


# def _log_survival_function(data, params):
#     """Calculate the log-survival function."""
#     age_coef = params.loc["age", "value"]
#     age_contrib = np.exp(age_coef * data["age"]) - 1

#     lambda_ = np.exp(
#         sum([params.loc[x, "value"] * data[x] for x in params.index if x != "age"])
#     )

#     return -(lambda_ * age_contrib) / age_coef
