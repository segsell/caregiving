"""Merge SOEP modules."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import (
    create_lagged_and_lead_variables,
    enforce_model_choice_restriction,
    filter_data,
)
from caregiving.data_management.soep.variables import (
    create_choice_variable,
    create_education_type,
    create_experience_variable,
    create_kidage_youngest,
    create_partner_state,
    determine_observed_job_offers,
    generate_job_separation_var,
)
from caregiving.model.shared import PART_TIME, WORK
from caregiving.specs.task_write_specs import read_and_derive_specs


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_create_structural_estimation_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw: Path = BLD / "data" / "soep_estimation_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
) -> None:

    specs = read_and_derive_specs(path_to_specs)

    # merged_data = pd.read_csv(path_to_raw, index_col=[0, 1])
    df = pd.read_csv(path_to_raw)

    df = create_partner_state(df, filter_missing=True)
    df = create_kidage_youngest(df)

    df = create_choice_variable(df)

    # filter data. Leave additional years in for lagging and leading.
    df = filter_data(df, specs)

    df = generate_job_separation_var(df)
    df = create_lagged_and_lead_variables(df, specs, lead_job_sep=True)

    # df = add_wealth_interpolate_and_deflate(df, specs)

    df["period"] = df["age"] - specs["start_age"]
    df = create_experience_variable(df)
    df = create_education_type(df)

    # enforce choice restrictions based on model setup
    df = enforce_model_choice_restriction(df, specs)

    # Construct job offer state
    was_fired_last_period = df["job_sep_this_year"] == 1
    df = determine_observed_job_offers(
        df, working_choices=WORK, was_fired_last_period=was_fired_last_period
    )

    # Filter out part-time men
    part_time_values = np.asarray(PART_TIME).ravel().tolist()
    mask = df["sex"] == 0
    # df = df[~(mask & (df["choice"].isin(part_time_values)))]
    df = df.loc[~(mask & df["choice"].isin(part_time_values))]
    df = df.loc[~(mask & df["lagged_choice"].isin(part_time_values))]

    # Keep relevant columns (i.e. state variables) and set their minimal datatype
    type_dict = {
        "age": "int8",
        "period": "int8",
        "choice": "int8",
        "lagged_choice": "int8",
        "partner_state": "int8",
        "job_offer": "int8",
        "experience": "int8",
        # "wealth": "float32",
        "education": "int8",
        "sex": "int8",
        "children": "int8",
        "kidage_youngest": "int8",
    }
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)

    # print_data_description(df)

    # Anonymize and save data
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path_to_save)
