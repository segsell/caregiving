"""Create sample for wage estimation."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import filter_data
from caregiving.data_management.soep.variables import (
    create_choice_variable,
    create_education_type,
    generate_working_hours,
    sum_experience_variables,
)
from caregiving.model.shared import N_MONTHS, N_WEEKS_IN_YEAR, WORK
from caregiving.specs.derive_specs import read_and_derive_specs


def task_create_wage_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw: Path = BLD / "data" / "soep_wage_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_wage_data.csv",
) -> None:

    specs = read_and_derive_specs(path_to_specs)
    specs["start_year"] = 2010
    specs["end_year"] = 2017

    df = pd.read_csv(path_to_raw, index_col=["pid", "syear"])

    # Filter data (age, sex, estimation period)
    df = filter_data(df, specs, lag_and_lead_buffer_years=False)

    # Create labor choice, keep only working (2: part-time, 3: full-time)
    df = create_choice_variable(df)

    # Weekly working hours
    df = generate_working_hours(df)

    # Experience, where we use the sum of part and full time (note: unlike in
    # structural estimation, we do not round or enforce a cap on experience here)
    df = sum_experience_variables(df)

    # Gross monthly wage
    df.rename(columns={"pglabgro": "monthly_wage"}, inplace=True)
    df = df[df["monthly_wage"] > 0]
    print(str(len(df)) + " observations after dropping invalid wage values.")

    # Drop retirees
    df = df[df["choice"].isin(WORK.tolist())]
    print(str(len(df)) + " observations after dropping non-working individuals.")

    # Hourly wage
    df["monthly_hours"] = df["working_hours"] * N_WEEKS_IN_YEAR / N_MONTHS
    df["hourly_wage"] = df["monthly_wage"] / df["monthly_hours"]

    df = create_education_type(df)

    df = df.sort_values(by=["pid", "syear"])
    df = df.reset_index()

    print(str(len(df)) + " observations in final wage estimation dataset.")

    type_dict = {
        "pid": np.int32,
        "syear": np.int32,
        "age": np.int32,
        "choice": np.int32,
        "experience": np.int32,
        "monthly_wage": np.float64,
        "hourly_wage": np.float64,
        "monthly_hours": np.float64,
        "working_hours": np.float64,
        "education": np.int32,
        "sex": np.int8,
    }

    # Keep relevant columns
    df = df[type_dict.keys()]
    df = df.astype(type_dict)
    # df = df.set_index(["pid", "syear"])

    # save data
    df.to_csv(path_to_save)
