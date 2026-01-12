"""Estimate the job separation probability and parameters."""

import pickle as pkl
from pathlib import Path
from typing import Annotated
import pytask

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.specs.derive_specs import read_and_derive_specs
from caregiving.model.shared import GOOD_HEALTH


@pytask.mark.job_separation
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

    # job_sep_probs_df = pd.DataFrame(
    #     job_sep_probs.reshape(specs["n_sexes"], specs["n_education_types"], -1)
    # )
    # job_sep_probs_df.to_csv(path_to_save_probs, index=False)
    # breakpoint()

    # load pickle file
    # pkl_path = BLD / "estimation" / "stochastic_processes" / "job_sep_probs.pkl"
    # with open(pkl_path, "rb") as fp:
    #     job_pkl = pkl.load(fp)


@pytask.mark.job_separation
def task_estimate_job_separation_age_dummies(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_job_separation_data.csv",
    path_to_save_params: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_sep_params_age_dummies.csv",
    path_to_save_probs: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_sep_probs_age_dummies.pkl",
) -> None:
    """Estimate job separation probability by age and education level."""

    specs = read_and_derive_specs(path_to_specs)
    df_job = pd.read_csv(path_to_data, index_col=0)

    # Estimate job separation probabilities
    job_sep_probs, job_sep_params = est_job_for_sample_age_dummies(df_job, specs)

    # Save results
    job_sep_params.to_csv(path_to_save_params)
    pkl.dump(job_sep_probs, path_to_save_probs.open("wb"))


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

    max_age_labor = specs["max_est_age_labor"] + 5
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


def est_job_for_sample_age_dummies(df_job, specs):

    # Estimate job separation probabilities until max retirement age.

    df_job = df_job[df_job["age"] <= specs["max_est_age_labor"]].copy()
    df_job["good_health"] = df_job["lagged_health"] == GOOD_HEALTH
    df_job["above_50"] = df_job["age"] >= 50
    df_job["above_55"] = df_job["age"] >= 55
    df_job["above_60"] = df_job["age"] >= 60
    df_job["high_educ"] = df_job["education"] == 1
    df_job = sm.add_constant(df_job)

    logit_cols = [
        "const",
        "high_educ",
        "good_health",
        "above_50",
        "above_55",
        "above_60",
    ]

    # Create solution containers - now including standard errors
    param_cols = logit_cols + [f"{col}_ser" for col in logit_cols]
    job_sep_params = pd.DataFrame(index=specs["sex_labels"], columns=param_cols)

    # Loop over sexes
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        sub_mask = df_job["sex"] == sex_var
        # Filter data and estimate with OLS
        df_job_subset = df_job[sub_mask]
        # Estimate a logit model with age and age squared
        exog = df_job_subset[logit_cols].astype(float)
        model = sm.Logit(endog=df_job_subset["job_sep"].astype(float), exog=exog)
        results = model.fit()
        # Save params
        job_sep_params.loc[sex_label, logit_cols] = results.params
        # Save standard errors
        for col in logit_cols:
            job_sep_params.loc[sex_label, f"{col}_ser"] = results.bse[col]
        # Calculate job sep for each age
        df_job.loc[sub_mask, "predicted_probs"] = results.predict(exog)

    # We will generate an array starting with age 0 to be able to use age as index

    all_ages = np.arange(0, specs["max_ret_age"] + 1)

    job_sep_probs = np.zeros(
        (specs["n_sexes"], specs["n_education_types"], 2, len(all_ages)), dtype=float
    )
    predicted_ages = np.arange(specs["start_age"], specs["max_ret_age"] + 1)
    above_50 = predicted_ages >= 50
    above_55 = predicted_ages >= 55
    above_60 = predicted_ages >= 60

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        params = job_sep_params.loc[sex_label, logit_cols]
        for edu_var in range(specs["n_education_types"]):
            for good_health in [0, 1]:
                exp_factor = (
                    params.loc["const"]
                    + params.loc["high_educ"] * edu_var
                    + params.loc["good_health"] * good_health
                    + params.loc["above_50"] * above_50
                    + params.loc["above_55"] * above_55
                    + params.loc["above_60"] * above_60
                )
                job_sep_probs_group = 1 / (1 + np.exp(-exp_factor))
                job_sep_probs[sex_var, edu_var, good_health, predicted_ages] = (
                    job_sep_probs_group
                )
                first_predicted_age = predicted_ages[0]
                # Fill in for ages below prediction with first predicted age
                job_sep_probs[sex_var, edu_var, good_health, :first_predicted_age] = (
                    job_sep_probs_group[0]
                )
                job_sep_probs[
                    sex_var, edu_var, good_health, specs["max_est_age_labor"] + 1 :
                ] = job_sep_probs_group[-1]

    return job_sep_probs, job_sep_params
