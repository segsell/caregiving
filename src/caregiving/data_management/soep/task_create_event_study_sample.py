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
    generate_working_hours,
)
from caregiving.model.shared import N_MONTHS, N_WEEKS_IN_YEAR, PART_TIME, WORK
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

    df = create_choice_variable(df)
    df = generate_working_hours(df, include_actual_hours=True, drop_missing=False)
    df = create_education_type(df)
    df = create_health_var(df, drop_missing=True)
    df = create_caregiving(df, filter_missing=False)

    df = deflate_gross_labor_income(df, cpi_data=cpi, specs=specs)
    df = create_hourly_wage(df)

    df = create_partner_state(df, filter_missing=True)

    df = create_parent_info(df, filter_missing=False)
    df = create_sibling_info(df, filter_missing=False)

    # filter data. Leave additional years in for lagging and leading.
    df = filter_data(df, specs, event_study=True)

    df = generate_job_separation_var(df)
    df = create_lagged_and_lead_variables(
        df, specs, lead_job_sep=False, event_study=True
    )

    df = create_experience_variable(df)

    # enforce choice restrictions based on model setup
    df = enforce_model_choice_restriction(df, specs)

    # # Construct job offer state
    # was_fired_last_period = df["job_sep_this_year"] == 1
    # df = determine_observed_job_offers(
    #     df, working_choices=WORK, was_fired_last_period=was_fired_last_period
    # )

    # Keep relevant columns and set their minimal datatype
    type_dict = {
        # own
        "pid": "int32",
        "syear": "int16",
        "age": "int8",
        "sex": "int8",
        "choice": "int8",
        "partner_state": "int8",
        "experience": "int8",
        "education": "int8",
        "children": "int8",
        "health": "float16",
        "working_hours": "float32",
        "working_hours_actual": "float32",
        "pglabgro_deflated": "float32",
        "hourly_wage": "float32",
        "pli0046": "int8",
        "any_care": "float32",
        "light_care": "float32",
        "intensive_care": "float32",
        "n_sisters": "float32",
        "n_brothers": "float32",
        # parent information
        "mother_age": "float32",
        "father_age": "float32",
        "mother_alive": "float32",
        "father_alive": "float32",
        # partner
        "parid": "int32",
        "age_p": "float16",
        "sex_p": "float16",
        "choice_p": "float16",
        "education_p": "float16",
        "health_p": "float16",
        "working_hours_p": "float32",
        "working_hours_actual_p": "float32",
        "pglabgro_deflated_p": "float32",
        "hourly_wage_p": "float32",
        "any_care_p": "float32",
    }

    df.reset_index(drop=False, inplace=True)
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)

    df = df[df["syear"] <= specs["end_year_event_study"]]

    # print_data_description(df)
    # breakpoint()

    df.to_csv(path_to_save)


# =====================================================================================
# Variables
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


def create_hourly_wage(df):
    df["monthly_wage"] = np.where(
        df["pglabgro_deflated"] > 0, df["pglabgro_deflated"], 0
    )

    df["monthly_hours"] = df["working_hours"] * N_MONTHS / N_WEEKS_IN_YEAR
    df["hourly_wage"] = np.where(
        df["working_hours"] == 0, 0, df["pglabgro_deflated"] / df["monthly_hours"]
    )

    return df


def create_parent_info(df, filter_missing=True):
    """Create parent age and alive status."""

    df = df.reset_index()

    df.loc[df["mybirth"] < 0] = np.nan
    df.loc[df["fybirth"] < 0] = np.nan

    df.loc[df["mydeath"] < 0] = np.nan
    df.loc[df["fydeath"] < 0] = np.nan

    for parent_var in ("mybirth", "fybirth"):
        dup_byear = df.dropna(subset=[parent_var]).groupby("pid")[parent_var].nunique()
        conflicts = dup_byear[dup_byear > 1]
        if not conflicts.empty:
            print(
                "Warning: Conflicting birth years detected for some pids:",
                conflicts.index.tolist(),
            )

    # Doesn't change anything
    df = df.sort_values(["pid", "syear"])
    df["mybirth"] = df.groupby("pid")["mybirth"].transform(lambda x: x.ffill().bfill())
    df["fybirth"] = df.groupby("pid")["fybirth"].transform(lambda x: x.ffill().bfill())

    # Drop observations with missing parent information
    if filter_missing:
        df = df[df["mybirth"] > 0]
        df = df[df["fybirth"] > 0]

    df["mother_age"] = df["syear"] - df["mybirth"]
    df["father_age"] = df["syear"] - df["fybirth"]

    _cond = (
        (df["mybirth"].notna())
        & (df["mybirth"] <= df["syear"])
        & (df["mydeath"].isna() | (df["syear"] < df["mydeath"]))
    )
    df["mother_alive"] = np.where(_cond, 1, 0)

    _cond = (
        (df["fybirth"].notna())
        & (df["fybirth"] <= df["syear"])
        & (df["fydeath"].isna() | (df["syear"] < df["fydeath"]))
    )
    df["father_alive"] = np.where(_cond, 1, 0)

    df_age = df.copy()

    # If mother_alive == 0, set mother_age to NaN
    df_age["mother_age"] = np.where(
        df_age["mother_alive"] == 1, df_age["mother_age"], np.nan
    )

    # If father_alive == 0, set father_age to NaN
    df_age["father_age"] = np.where(
        df_age["father_alive"] == 1, df_age["father_age"], np.nan
    )

    df_age.set_index(["pid", "syear"], inplace=True)
    return df_age


def create_sibling_info(df, filter_missing=False):

    df.loc[df["pld0030"] < 0, "pld0030"] = np.nan
    df.loc[df["pld0032"] < 0, "pld0032"] = np.nan

    out = df.sort_values(["pid", "syear"]).copy()
    out["pld0030"] = out.groupby("pid")["pld0030"].transform(
        lambda x: x.ffill().bfill()
    )
    out["pld0032"] = out.groupby("pid")["pld0032"].transform(
        lambda x: x.ffill().bfill()
    )

    out = out.rename(columns={"pld0030": "n_sisters", "pld0032": "n_brothers"})

    if filter_missing:
        out = out[out["n_sisters"] >= 0]
        out = out[out["n_brothers"] >= 0]

    return out


def create_caregiving(df, filter_missing=False):

    # any care, light care, intensive care

    _cond = (
        df["pli0046"].isna(),
        df["pli0046"] == 0,
        df["pli0046"] > 0,
    )
    _val = [np.nan, 0, 1]
    df["any_care"] = np.select(_cond, _val, default=np.nan)

    if filter_missing:
        df = df[df["any_care"].notna()]

    _cond = (
        df["pli0046"].isna(),
        (df["pli0046"] == 0) | (df["pli0046"] > 1),
        df["pli0046"] == 1,
    )
    _val = [np.nan, 0, 1]
    df["light_care"] = np.select(_cond, _val, default=np.nan)

    _cond = (
        df["pli0046"].isna(),
        (df["pli0046"] == 0) | (df["pli0046"] == 1),
        df["pli0046"] > 1,
    )
    _val = [np.nan, 0, 1]
    df["intensive_care"] = np.select(_cond, _val, default=np.nan)

    return df
