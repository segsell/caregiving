"""Estimate job offer probability."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.shared import SEX, UNEMPLOYED_CHOICES, WORK_CHOICES


def task_estimate_job_offer(
    path_to_load_specs: Path = SRC / "specs.yaml",
    # path_to_start_params: Path = SRC
    # / "start_params_and_bounds"
    # / "start_params.yaml",
    path_to_struct_estimation_sample: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_save_job_offer_params: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_offer_params.csv",
):
    """Estimate job offer parameters via Logit."""

    specs = yaml.safe_load(path_to_load_specs.open())
    # start_params_all = yaml.safe_load(open(path_to_start_params, "rb"))

    # Check if job_offer_start_year exists in specs
    if "job_offer_start_year" not in specs or "job_offer_end_year" not in specs:
        start_year = specs["start_year"]
        end_year = specs["end_year"]
    else:
        start_year = specs["job_offer_start_year"]
        end_year = specs["job_offer_end_year"]

    struct_est_sample = pd.read_csv(path_to_struct_estimation_sample, index_col=0)
    struct_est_sample = struct_est_sample[
        (struct_est_sample["syear"] >= start_year)
        & (struct_est_sample["syear"] <= end_year)
    ].copy()

    job_offer_params = estimate_logit_job_offer_params(struct_est_sample, specs)

    # Update start values
    # start_params_all.update(job_offer_params)

    # Convert to DataFrame
    job_offer_params_df = pd.DataFrame.from_dict(
        job_offer_params, orient="index"
    ).reset_index()
    job_offer_params_df.columns = ["param", "value"]

    # Save as csv
    job_offer_params_df.to_csv(path_to_save_job_offer_params, index=False)


def estimate_logit_job_offer_params(df, specs):
    """Estimate job offer logit parameters."""

    unemployed_values = np.asarray(UNEMPLOYED_CHOICES).ravel().tolist()
    work_values = np.asarray(WORK_CHOICES).ravel().tolist()
    sex_var = SEX

    # Filter for unemployed, because we only estimate job offer probs on them
    df_unemployed = df[df["lagged_choice"].isin(unemployed_values)].copy()
    df_unemployed["sex"] = sex_var

    # Create work start indicator
    df_unemployed.loc[:, "work_start"] = (
        df_unemployed["choice"].isin(work_values).astype(int)
    )

    # Filter for relevant columns
    logit_df = df_unemployed[["sex", "period", "education", "work_start"]].copy()
    logit_df["age"] = logit_df["period"] + specs["start_age"]
    logit_df["age_squared"] = logit_df["age"] ** 2
    logit_df["age_cubed"] = logit_df["age"] ** 3

    logit_df = logit_df[logit_df["age"] < specs["min_SRA"] + 5]  # 65
    logit_df["intercept"] = 1

    logit_vars = [
        "intercept",
        "age",
        "education",
        "age_squared",
        "age_cubed",
    ]

    # sex_append = ["men", "women"]
    job_offer_params = {}

    suffix = "women"

    # for sex_var, suffix in enumerate(sex_append):
    logit_df_gender = logit_df[logit_df["sex"] == sex_var]
    logit_model = sm.Logit(logit_df_gender["work_start"], logit_df_gender[logit_vars])
    logit_fitted = logit_model.fit()

    params = logit_fitted.params

    gender_params = {
        f"job_finding_logit_const_{suffix}": params["intercept"],
        f"job_finding_logit_age_{suffix}": params["age"],
        f"job_finding_logit_age_squared_{suffix}": params["age_squared"],
        f"job_finding_logit_age_cubed_{suffix}": params["age_cubed"],
        # f"job_finding_logit_below_49_{suffix}": params["below_49"],
        # f"job_finding_logit_above_49_{suffix}": params["above_49"],
        f"job_finding_logit_high_educ_{suffix}": params["education"],
    }
    job_offer_params = {**job_offer_params, **gender_params}

    return job_offer_params
