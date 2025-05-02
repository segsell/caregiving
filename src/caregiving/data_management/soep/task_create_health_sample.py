"""Health transition sample."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
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
)
from caregiving.specs.task_write_specs import read_and_derive_specs


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
