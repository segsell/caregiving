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
    generate_job_separation_var,
)
from caregiving.model.shared import PART_TIME, WORK
from caregiving.specs.derive_specs import read_and_derive_specs


def task_create_job_separation_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw: Path = BLD / "data" / "soep_job_separation_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_job_separation_data.csv",
) -> None:

    specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_raw, index_col=["pid", "syear"])

    # filter data (age, sex, estimation period)
    df = filter_data(df, specs)

    # create choice and lagged choice variable
    df = create_choice_variable(df)
    # lagged choice
    df = create_lagged_and_lead_variables(df, specs)

    # Job separation
    df = generate_job_separation_var(df)

    # Overwrite job separation when individuals choose working
    work_values = np.asarray(WORK).ravel().tolist()
    df.loc[df["choice"].isin(work_values), "job_sep"] = 0

    # education
    df = create_education_type(df)

    # Now restrict sample to all who worked last period or did loose their job
    df = df[(df["lagged_choice"].isin(work_values)) | (df["plb0282_h"] == 1)]

    # Kick out men that worked part-time last period
    part_time_values = np.asarray(PART_TIME).ravel().tolist()
    df = df[~((df["lagged_choice"].isin(part_time_values)) & (df["sex"] == 0))]

    # Create age at which one got fired and rename job separation column
    df["age_fired"] = df["age"] - 1
    df.reset_index(inplace=True)

    # Relevant columns and datatype
    columns = {
        "age_fired": np.int32,
        "education": np.uint8,
        "sex": np.uint8,
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
