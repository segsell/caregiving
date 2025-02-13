"""Create variables for event study."""

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
    create_health_var,
    create_partner_state,
    determine_observed_job_offers,
    generate_job_separation_var,
)
from caregiving.model.shared import PART_TIME, WORK
from caregiving.specs.task_write_specs import read_and_derive_specs


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_create_event_study_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    path_to_raw: Path = BLD / "data" / "soep_event_study_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_event_study_sample.csv",
) -> None:
    """Create variables and prepare sample for event study."""

    specs = read_and_derive_specs(path_to_specs)
    cpi = pd.read_csv(path_to_cpi, index_col=0)

    df = pd.read_csv(path_to_raw)

    df = create_partner_state(df, filter_missing=True)

    df = create_choice_variable(df)

    # filter data. Leave additional years in for lagging and leading.
    df = filter_data(df, specs)

    df = generate_job_separation_var(df)
    df = create_lagged_and_lead_variables(df, specs, lead_job_sep=True)

    df = deflate_gross_labor_income(df, cpi_data=cpi, specs=specs)

    df = create_experience_variable(df)
    df = create_education_type(df)
    df = create_health_var(df)

    # enforce choice restrictions based on model setup
    df = enforce_model_choice_restriction(df, specs)

    # Construct job offer state
    was_fired_last_period = df["job_sep_this_year"] == 1
    df = determine_observed_job_offers(
        df, working_choices=WORK, was_fired_last_period=was_fired_last_period
    )

    # Keep relevant columns and set their minimal datatype
    type_dict = {
        "choice": "int8",
        "lagged_choice": "int8",
        "partner_state": "int8",
        "job_offer": "int8",
        "experience": "int8",
        "education": "int8",
        "sex": "int8",
        "children": "int8",
        "health": "int8",
        "pglabgro_deflated": "float32",
    }
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)

    # print_data_description(df)

    # Anonymize and save data
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path_to_save)


# =====================================================================================
# Deflate
# =====================================================================================


def deflate_gross_labor_income(df, cpi_data, specs):
    """Deflate gross labor income."""

    df["pglabgro"] = df["pglabgro"].replace({-2: 0, -5: np.nan, -7: np.nan})

    _base_year = specs["reference_year_event_study"]

    cpi_data = cpi_data.rename(columns={"int_year": "syear"})
    base_year_cpi = cpi_data.loc[cpi_data["syear"] == _base_year, "cpi"].iloc[0]

    cpi_data["cpi_normalized"] = cpi_data["cpi"] / base_year_cpi

    df = pd.merge(df, cpi_data, on="syear", how="left")
    df["pglabgro_deflated"] = df["pglabgro"] / df["cpi_normalized"]

    return df
