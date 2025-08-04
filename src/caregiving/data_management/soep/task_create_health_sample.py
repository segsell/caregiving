"""Health transition sample."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import (
    filter_above_age,
    filter_below_age,
    filter_years,
    recode_sex,
    span_dataframe,
)
from caregiving.data_management.soep.variables import (
    clean_health_create_states,
    create_education_type,
    create_health_var_good_bad,
    create_health_var_good_medium_bad,
    create_nursing_home,
)
from caregiving.specs.task_write_specs import read_and_derive_specs
from caregiving.utils import table

AGE_LOW = 65
AGE_HIGH = 105


def task_create_health_transition_sample_good_bad(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw_data: Path = BLD / "data" / "soep_health_data_raw.csv",
    path_to_save: Annotated[Path, Product] = (
        BLD / "data" / "health_transition_estimation_sample.pkl"
    ),
):

    specs = read_and_derive_specs(path_to_specs)
    specs["start_year"] = 2010
    specs["end_year"] = 2017

    df = pd.read_csv(path_to_raw_data, index_col=["pid", "syear"])

    # Pre-Filter estimation years
    df = filter_years(df, specs["start_year"] - 1, specs["end_year"] + 1)

    # Pre-Filter age and sex
    df = filter_below_age(df, specs["start_age"] - specs["health_smoothing_bandwidth"])
    df = filter_above_age(df, specs["end_age"] + specs["health_smoothing_bandwidth"])
    df = recode_sex(df)

    # Create education type
    df = create_education_type(df)

    # create health states
    df = create_health_var_good_bad(df)

    df = span_dataframe(df, specs["start_year"] - 1, specs["end_year"] + 1)
    df = clean_health_create_states(df)

    df = df[["age", "education", "health", "lead_health", "sex"]]

    print(
        str(len(df))
        + " observations in the final health transition sample.  \n ----------------"
    )

    df.to_pickle(path_to_save)


def task_create_health_transition_sample_good_medium_bad(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw_data: Path = BLD / "data" / "soep_health_data_raw.csv",
    path_to_save: Annotated[Path, Product] = (
        BLD / "data" / "health_transition_estimation_sample_good_medium_bad.pkl"
    ),
):

    specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_raw_data, index_col=["pid", "syear"])

    # Pre-Filter estimation years
    df = filter_years(df, specs["start_year"] - 1, specs["end_year"] + 1)

    # Pre-Filter age and sex
    df = filter_below_age(df, specs["start_age"] - specs["health_smoothing_bandwidth"])
    df = filter_above_age(df, specs["end_age"] + specs["health_smoothing_bandwidth"])
    df = recode_sex(df)

    # Create education type
    df = create_education_type(df)

    # create health states
    df = create_health_var_good_medium_bad(df)

    df = span_dataframe(df, specs["start_year"] - 1, specs["end_year"] + 1)
    df = clean_health_create_states(df)

    df = df[["age", "education", "health", "lead_health", "sex"]]

    print(
        str(len(df))
        + " observations in the final health transition sample.  \n ----------------"
    )

    df.to_pickle(path_to_save)


def task_create_nursing_home_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw_data: Path = BLD / "data" / "soep_health_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "nursing_home_sample.pkl",
):

    _specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_raw_data, index_col=["pid", "syear"])

    df = recode_sex(df)

    # # Create education type
    df = create_education_type(df, drop_missing=False)

    # # create health states
    df = create_health_var_good_bad(df, drop_missing=False)
    df = create_nursing_home(df)

    df = df[["age", "education", "health", "nursing_home", "sex"]]

    df.to_pickle(path_to_save)

    # First model: full sample (age 65–110)
    filtered_data = df[
        (df["age"] >= AGE_LOW) & (df["age"] <= AGE_HIGH) & (df["sex"] == 1)
    ].copy()
    filtered_data = filtered_data[["age", "nursing_home"]].dropna()
    filtered_data["age_squared"] = filtered_data["age"] ** 2
    X = sm.add_constant(filtered_data[["age", "age_squared"]])
    y = filtered_data["nursing_home"]
    logit_model = sm.Logit(y, X).fit()
    print("=== Full Sample ===")
    print(logit_model.summary())

    # Predict over age range
    age_range = np.arange(AGE_LOW, filtered_data["age"].max() + 1)
    X_pred = pd.DataFrame({"const": 1, "age": age_range, "age_squared": age_range**2})
    predicted_probs_full = logit_model.predict(X_pred)

    # Second model: bad health only
    bad_health_data = df[
        (df["age"] >= AGE_LOW) & (df["age"] <= AGE_HIGH + 5) & (df["health"] == 0)
    ].copy()
    bad_health_data = bad_health_data[["age", "nursing_home"]].dropna()
    bad_health_data["age_squared"] = bad_health_data["age"] ** 2
    X2 = sm.add_constant(bad_health_data[["age", "age_squared"]])
    y2 = bad_health_data["nursing_home"]
    logit_model_bad = sm.Logit(y2, X2).fit()
    print("=== Bad Health Subsample ===")
    print(logit_model_bad.summary())

    # Predict for bad health sample
    X_pred_bad = pd.DataFrame(
        {"const": 1, "age": age_range, "age_squared": age_range**2}
    )
    predicted_probs_bad = logit_model_bad.predict(X_pred_bad)

    # Plot both prediction curves with raw data (full sample)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        filtered_data["age"],
        filtered_data["nursing_home"],
        alpha=0.2,
        label="Raw Data (All)",
    )
    plt.plot(
        age_range,
        predicted_probs_full,
        color="red",
        linewidth=2,
        label="Prediction (All)",
    )
    plt.plot(
        age_range,
        predicted_probs_bad,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Prediction (Bad Health)",
    )
    plt.xlabel("Age")
    plt.ylabel("Probability of Nursing Home")
    plt.title("Logit Predictions: All vs. Bad Health")
    plt.grid(True)
    plt.legend()
    plt.show()
