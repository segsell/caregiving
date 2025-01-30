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
    create_partner_state,
    determine_observed_job_offers,
    generate_job_separation_var,
)
from caregiving.model.shared import is_part_time
from caregiving.specs.derive_specs import read_and_derive_specs


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_load_and_merge_soep(
    soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
    soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
    soep_c38_pl: Path = SRC / "data" / "soep" / "pl.dta",
    soep_c38_hl: Path = SRC / "data" / "soep" / "hl.dta",
    soep_c38_pequiv: Path = SRC / "data" / "soep" / "pequiv.dta",
    soep_c38_pflege: Path = SRC / "data" / "soep" / "pflege.dta",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_estimation_data_raw.csv",
) -> None:
    """Merge SOEP modules.

    https://paneldata.org/soep-core/datasets/pequiv/
    """

    # Load SOEP core data
    pgen_data = pd.read_stata(
        soep_c38_pgen,
        columns=[
            "syear",
            "pid",
            "hid",
            "pgemplst",
            "pgexpft",
            "pgexppt",
            "pgstib",
            "pgpartz",
            "pglabgro",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )

    ppathl_data = pd.read_stata(
        soep_c38_ppathl,
        columns=["pid", "hid", "syear", "sex", "gebjahr", "parid", "rv_id"],
        convert_categoricals=False,
    )
    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )

    # Add pl data
    pl_data_reader = pd.read_stata(
        soep_c38_pl,
        columns=["pid", "hid", "syear", "plb0304_h"],
        chunksize=100000,
        convert_categoricals=False,
    )
    pl_data = pd.DataFrame()
    for itm in pl_data_reader:
        pl_data = pd.concat([pl_data, itm])
    merged_data = pd.merge(
        merged_data, pl_data, on=["pid", "hid", "syear"], how="inner"
    )

    # get household level data
    hl_data = pd.read_stata(
        soep_c38_hl,
        columns=["hid", "syear", "hlc0043"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, hl_data, on=["hid", "syear"], how="left")
    pequiv_data = pd.read_stata(
        # d11107: number of children in household
        # d11101: age of individual
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        # m111050-m11123: Activities of Daily Living
        soep_c38_pequiv,
        columns=["pid", "syear", "d11107", "d11101", "m11126", "m11124"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)

    # Pflege
    pflege = pd.read_stata(soep_c38_pflege, convert_categoricals=False)

    pflege = pflege[pflege["pnrcare"] >= 0]
    merged_data_with_pflege = pd.merge(
        merged_data,
        pflege,
        left_on=["pid", "syear"],
        right_on=["pnrcare", "syear"],
        how="left",
    )
    merged_data_with_pflege.rename(columns={"pid_x": "pid"}, inplace=True)

    # Set index
    merged_data["age"] = merged_data["d11101"].astype(int)
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")

    merged_data.to_csv(path_to_save)


def task_create_estimation_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw: Path = BLD / "data" / "soep_estimation_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_estimation_data.csv",
):

    specs = read_and_derive_specs(path_to_specs)

    # merged_data = pd.read_csv(path_to_raw, index_col=[0, 1])
    df = pd.read_csv(path_to_raw)

    df = create_partner_state(df, filter_missing=True)
    df = create_choice_variable(df)

    # filter data. Leave additional years in for lagging and leading.
    df = filter_data(df, specs)

    df = generate_job_separation_var(df)
    df = create_lagged_and_lead_variables(df, specs, lead_job_sep=True)

    # df = add_wealth_interpolate_and_deflate(df, paths, specs)

    df["period"] = df["age"] - specs["start_age"]
    df = create_experience_variable(df)
    df = create_education_type(df)

    # enforce choice restrictions based on model setup
    df = enforce_model_choice_restriction(df, specs)

    # Construct job offer state
    was_fired_last_period = df["job_sep_this_year"] == 1
    df = determine_observed_job_offers(
        df, working_choices=[2, 3], was_fired_last_period=was_fired_last_period
    )

    # Filter out part-time men
    mask = df["sex"] == 0
    df = df[~(mask & is_part_time(df["choice"]))]
    df = df[~(mask & is_part_time(df["lagged_choice"]))]

    # Keep relevant columns (i.e. state variables) and set their minimal datatype
    type_dict = {
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
        # "health": "int8",
    }
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)

    # df = df[list(type_dict.keys())]
    # df = df.astype(type_dict)

    # print_data_description(df)

    # Anonymize and save data
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(path_to_save)
