"""Create sample for estimation of job separation probability."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import (
    create_lagged_and_lead_variables,
    filter_data,
)
from caregiving.data_management.soep.variables import (
    create_choice_variable,
    create_education_type,
    create_health_var_good_bad,
    generate_job_separation_var,
)
from caregiving.model.shared import PART_TIME_CHOICES, WORK_CHOICES
from caregiving.specs.derive_specs import read_and_derive_specs


def task_create_job_separation_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw: Path = BLD / "data" / "soep_job_separation_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_job_separation_data.csv",
) -> None:

    specs = read_and_derive_specs(path_to_specs)
    specs["start_year"] = 2010
    specs["end_year"] = 2017

    df = pd.read_csv(path_to_raw, index_col=["pid", "syear"])

    # filter data (age, sex, estimation period)
    df = filter_data(df, specs)

    # create choice and lagged choice variable
    df = create_choice_variable(df)

    # Create lagged health variable
    # lagged choice
    df = create_lagged_and_lead_variables(df, specs)

    # We create the health variable and correct it
    df = create_health_var_good_bad(df, drop_missing=False)
    df = correct_health_state(df)

    # Job separation
    df = generate_job_separation_var(df)

    # Overwrite job separation when individuals choose working
    work_values = np.asarray(WORK_CHOICES).ravel().tolist()
    df.loc[df["choice"].isin(work_values), "job_sep"] = 0

    # education
    df = create_education_type(df)

    # Now restrict sample to all who worked last period or did loose their job
    df = df[(df["lagged_choice"].isin(work_values)) | (df["plb0282_h"] == 1)]

    # Kick out men that worked part-time last period
    part_time_values = np.asarray(PART_TIME_CHOICES).ravel().tolist()
    df = df[~((df["lagged_choice"].isin(part_time_values)) & (df["sex"] == 0))]

    # Create age at which one got fired and rename job separation column
    df["age_fired"] = df["age"] - 1
    df.reset_index(inplace=True)

    # Relevant columns and datatype
    columns = {
        "age_fired": np.int32,
        "education": np.uint8,
        "sex": np.uint8,
        "lagged_health": np.float32,
        "job_sep": np.uint8,
    }

    df_sub = df[df["sex"] >= 0]
    df_sub = df_sub[columns.keys()]
    df_sub = df_sub.astype(columns)
    # Rename age fired to age
    df_sub.rename(columns={"age_fired": "age"}, inplace=True)
    # Limit age range to start age and maximum retirement age
    df_sub = df_sub[
        (df_sub["age"] >= specs["start_age"]) & (df_sub["age"] <= specs["max_ret_age"])
    ]
    print(f"{len(df_sub)} observations in job separation sample.")

    # Save data
    df_sub.to_csv(path_to_save)


def correct_health_state(data, filter_missings=False):
    """This function creates a lagged health state variable in the soep-PEQUIV dataset.

    The function replaces the health variable with 1 if both the previous and next
    health are 1.

    """

    # replace health with 1 if both previous and next health are 1
    data["lagged_health"] = data.groupby(["pid"])["health"].shift(1)
    data["lead_health"] = data.groupby(["pid"])["health"].shift(-1)

    # one year bad health in between two years of good health is still considered good health
    data.loc[
        (data["lagged_health"] == 0) & (data["lead_health"] == 0),
        "health",
    ] = 0

    # update lead_health
    data["lead_health"] = data.groupby(["pid"])["health"].shift(-1)
    # update lag_health
    data["lagged_health"] = data.groupby(["pid"])["health"].shift(1)

    if filter_missings:
        data = data[data["health"].notna()]
        print(
            str(len(data))
            + " observations left after dropping people with missing health data."
        )

    # drop people with missing lead health data
    # print(str(len(data)) + " observations after spanning the dataframe before dropping people with missing health data.")
    # data = data[data["lead_health"].notna()] # need to do this here because spanning the dataframe creates new missing values
    # print(str(len(data)) + " observations left after dropping people with missing lead health data.")
    # data = data[data["lag_health"].notna()] # need to do this here because spanning the dataframe creates new missing values
    # print(str(len(data)) + " observations left after dropping people with missing lagged health data.")
    # data = data[data["health"].notna()] # need to do this again here because spanning the dataframe creates new missing values
    # print(str(len(data)) + " observations left after dropping people with missing health data.")

    return data
