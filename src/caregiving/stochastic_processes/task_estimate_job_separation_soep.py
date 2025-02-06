"""Estimate the job separation probability and parameters."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.specs.derive_specs import read_and_derive_specs


def task_estimate_job_separation(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_job_separation_data.csv",
    path_to_save_params: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_sep_params.csv",
    path_to_save_probs: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_sep_probs.pkl",
) -> None:
    """Estimate job separation probability by age and education level."""

    specs = read_and_derive_specs(path_to_specs)
    df_job = pd.read_csv(path_to_data, index_col=0)

    # Estimate job separation probabilities
    job_sep_probs, job_sep_params = estimate_logit_by_sample(df_job, specs)

    # Save results
    job_sep_params.to_csv(path_to_save_params)
    pkl.dump(job_sep_probs, path_to_save_probs.open("wb"))

    # load pickle file
    # pkl_path = BLD / "estimation" / "stochastic_processes" / "job_sep_probs.pkl"
    # with open(pkl_path, "rb") as fp:
    #     job_pkl = pkl.load(fp)


def estimate_logit_by_sample(df_job, specs):
    """Estimate Logit parameters and probabilities."""
    df_job["age_sq"] = df_job["age"] ** 2

    index = pd.MultiIndex.from_product(
        [specs["sex_labels"], specs["education_labels"]],
        names=["sex", "education"],
    )
    # Create solution containers
    job_sep_params = pd.DataFrame(index=index, columns=["age", "age_sq", "const"])

    # Estimate job separation probabilities until max retirement age
    n_working_age = specs["max_ret_age"] - specs["start_age"] + 1
    job_sep_probs = np.zeros(
        (specs["n_sexes"], specs["n_education_types"], n_working_age), dtype=float
    )

    max_age_labor = specs["max_est_age_labor"]
    max_period_labor = max_age_labor - specs["start_age"]
    df_job = df_job[df_job["age"] <= max_age_labor]

    # Loop over sexes
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        # Loop over education levels

        for edu_var, edu_label in enumerate(specs["education_labels"]):
            # Filter data and estimate with OLS
            df_job_edu = df_job[
                (df_job["sex"] == sex_var) & (df_job["education"] == edu_var)
            ]
            # Estimate a logit model with age and age squared
            model = sm.Logit(
                endog=df_job_edu["job_sep"].astype(float),
                exog=sm.add_constant(df_job_edu[["age", "age_sq"]]),
            )
            results = model.fit()

            # Save params
            job_sep_params.loc[(sex_label, edu_label), :] = results.params

            # Calculate job sep for each age
            ages = np.sort(df_job_edu["age"].unique())
            exp_factor = (
                job_sep_params.loc[(sex_label, edu_label), "const"]
                + job_sep_params.loc[(sex_label, edu_label), "age"] * ages
                + job_sep_params.loc[(sex_label, edu_label), "age_sq"] * ages**2
            )

            job_sep_probs_group = 1 / (1 + np.exp(-exp_factor))
            job_sep_probs[sex_var, edu_var, : max_period_labor + 1] = (
                job_sep_probs_group
            )
            job_sep_probs[sex_var, edu_var, max_period_labor + 1 :] = (
                job_sep_probs_group[-1]
            )

    return job_sep_probs, job_sep_params
