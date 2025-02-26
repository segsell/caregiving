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
from caregiving.model.shared import PART_TIME, RETIREMENT, WORK
from caregiving.specs.task_write_specs import read_and_derive_specs


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_create_structural_estimation_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw: Path = BLD / "data" / "soep_estimation_data_raw.csv",
    path_to_wealth: Path = BLD / "data" / "soep_wealth_data.csv",
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

    df = create_alreay_retired_variable(df)
    # df = df.reset_index()
    # df = df.sort_values(["pid", "syear"])

    # retired_values = np.asarray(RETIREMENT).ravel().tolist()
    # df["retire_flag"] = (
    #     df["lagged_choice"].isin(retired_values) & df["choice"].isin(retired_values)
    # ).astype(int)
    # df["already_retired"] = df.groupby("pid")["retire_flag"].cummax()

    # df.drop(columns=["retire_flag"], inplace=True)
    # df.set_index(["pid", "syear"], inplace=True)

    wealth = pd.read_csv(path_to_wealth, index_col=[0])
    df = add_wealth_data(df, wealth, drop_missing=False)

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
    df = df.loc[~(mask & df["choice"].isin(part_time_values))]
    df = df.loc[~(mask & df["lagged_choice"].isin(part_time_values))]

    # Keep relevant columns (i.e. state variables) and set their minimal datatype
    type_dict = {
        "age": "int8",
        "period": "int8",
        "choice": "int8",
        "already_retired": "int8",
        "lagged_choice": "int8",
        "partner_state": "int8",
        "job_offer": "int8",
        "experience": "int8",
        "wealth": "float32",
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


def add_wealth_data(data, wealth, drop_missing=False):
    """Add wealth data to the estimation sample."""

    data = data.reset_index()
    data = data.merge(wealth, on=["hid", "syear"], how="left")

    data.set_index(["pid", "syear"], inplace=True)

    if drop_missing:
        data = data[(data["wealth"].notna())]

    print(str(len(data)) + " left after dropping people with missing wealth.")

    return data


def create_alreay_retired_variable(data):
    """Create already retired variable."""
    data = data.reset_index()
    data = data.sort_values(["pid", "syear"])

    retired_values = np.asarray(RETIREMENT).ravel().tolist()
    data["retire_flag"] = (
        data["lagged_choice"].isin(retired_values) & data["choice"].isin(retired_values)
    ).astype(int)
    data["already_retired"] = data.groupby("pid")["retire_flag"].cummax()

    # # Done in model enforcements
    # switch_back_condition = data["lagged_choice"].isin(retired_values) & ~data[
    #     "choice"
    # ].isin(retired_values)
    # assert not switch_back_condition.any()

    data.drop(columns=["retire_flag"], inplace=True)
    data.set_index(["pid", "syear"], inplace=True)

    return data
