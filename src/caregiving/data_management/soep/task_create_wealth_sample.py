"""Load and process wealth data."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import (
    create_lagged_and_lead_variables,
    enforce_model_choice_restriction,
)
from caregiving.data_management.soep.soep_variables.experience import (
    create_experience_variable_with_cap,
)
from caregiving.data_management.soep.variables import (
    create_choice_variable,
    create_education_type,
    create_policy_state,
)
from caregiving.specs.task_write_specs import read_and_derive_specs


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


@pytask.mark.wealth_sample
def task_create_soep_wealth_data(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    soep_c40_hwealth: Path = SRC / "data" / "soep" / "hwealth.dta",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_wealth_data.csv",
) -> None:
    """Create simple wealth data with trim, rename, and deflate only.

    No interpolation, extrapolation, or merging with individual data.
    """
    specs = read_and_derive_specs(path_to_specs)
    cpi = pd.read_csv(path_to_cpi, index_col=0)

    # Load household wealth data
    wealth_data = pd.read_stata(
        soep_c40_hwealth,
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)

    # Trim and rename
    wealth_data = trim_and_rename(wealth_data)

    # Deflate wealth
    wealth_data = deflate_wealth(wealth_data, cpi_data=cpi, specs=specs)

    # Ensure non-negative wealth
    wealth_data.loc[wealth_data["wealth"] < 0, "wealth"] = 0

    print(str(len(wealth_data)) + " observations in simple wealth data.")

    wealth_data.to_csv(path_to_save, index=False)


@pytask.mark.wealth_sample
def task_create_household_wealth_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    path_to_raw: Path = BLD / "data" / "soep_estimation_data_raw.csv",
    soep_c40_hwealth: Path = SRC / "data" / "soep" / "hwealth.dta",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_wealth_data_full.csv",
) -> None:
    """Create sample for wealth estimation with individual-level information."""

    specs = read_and_derive_specs(path_to_specs)
    specs["start_year"] = 2001
    specs["end_year"] = 2023

    cpi = pd.read_csv(path_to_cpi, index_col=0)

    # Load individual-level data
    df_raw = pd.read_csv(path_to_raw, index_col=[0, 1])
    # Keep index as MultiIndex (pid, syear) for create_lagged_and_lead_variables

    # Load household wealth data
    wealth_data = pd.read_stata(
        soep_c40_hwealth,
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)

    wealth_data = trim_and_rename(wealth_data)
    wealth_data_full = interpolate_and_extrapolate_wealth(wealth_data, specs)
    wealth_data_full = deflate_wealth(wealth_data_full, cpi_data=cpi, specs=specs)
    wealth_data_full.loc[wealth_data_full["wealth"] < 0, "wealth"] = 0

    # Create necessary variables
    df_raw = create_choice_variable(df_raw)
    df_raw = create_lagged_and_lead_variables(
        df_raw, specs, lead_job_sep=False, drop_missing_lagged_choice=False
    )
    # Reset index to move pid and syear back to columns for merging
    df_raw = df_raw.reset_index()
    df_raw["period"] = df_raw["age"] - specs["start_age"]
    df_raw = create_policy_state(df_raw, specs)
    df_raw = create_experience_variable_with_cap(
        df_raw, exp_cap=specs["start_age"] - 14
    )
    df_raw = create_education_type(df_raw)

    df_raw = enforce_model_choice_restriction(df_raw, specs)

    # Merge wealth data with individual data on hid and syear
    df = df_raw.merge(wealth_data_full, on=["hid", "syear"], how="inner")

    # Keep only required columns
    type_dict = {
        "pid": "int32",
        "hid": "int32",
        "syear": "int16",
        "sex": "int8",
        "gebjahr": "int16",
        "age": "int8",
        "education": "int8",
        "lagged_choice": "float32",  # can be NA
        "policy_state": "int8",
        "policy_state_value": "int8",
        "experience": "int8",
        "wealth": "float32",
    }

    # Select and type columns
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)

    print(str(len(df)) + " observations after merging wealth with individual data.")

    df.to_csv(path_to_save, index=True)


def trim_and_rename(wealth_data):
    """Trim wealth data (drop missing, negative set to 0) and rename."""
    wealth_data = wealth_data[wealth_data["w011ha"].notna()]
    wealth_data.loc[wealth_data["w011ha"] < 0, "w011ha"] = 0
    wealth_data.rename(columns={"w011ha": "wealth"}, inplace=True)

    return wealth_data


def interpolate_and_extrapolate_wealth(wealth_data, options):
    """Interpolate and extrapolate wealth data."""
    wealth_data_full = span_full_wealth_panel(wealth_data, options)

    # Interpolate between existing points
    wealth_data_full["wealth"] = wealth_data_full.groupby("hid")["wealth"].transform(
        lambda group: group.interpolate(method="linear")
    )

    # Extrapolate until the first and last observation
    wealth_data_full["wealth"] = wealth_data_full.groupby("hid").apply(
        extrapolate_wealth
    )["wealth"]

    return wealth_data_full.reset_index()


def deflate_wealth(df, cpi_data, specs, var_name="wealth"):
    """Deflate wealth using consumer price index."""
    # Copy cpi_data to avoid modifying the original
    cpi_data = cpi_data.copy()
    cpi_data = cpi_data.rename(columns={"int_year": "syear"})

    df = df.copy()
    df[f"{var_name}_raw"] = df[var_name].copy()

    _base_year = specs["reference_year"]
    base_year_cpi = cpi_data.loc[cpi_data["syear"] == _base_year, "cpi"].iloc[0]

    cpi_data["cpi_normalized"] = cpi_data["cpi"] / base_year_cpi

    # Only merge the columns we need
    data_merged = df.merge(
        cpi_data[["syear", "cpi_normalized"]], on="syear", how="left"
    )
    data_merged[var_name] = data_merged[var_name] / data_merged["cpi_normalized"]

    data_merged = data_merged.drop(columns=["cpi_normalized"])

    return data_merged


# =====================================================================================
# Auxiliary
# =====================================================================================


def span_full_wealth_panel(wealth_data, options):
    """Create extra rows for each household and year between start_year and end_year.

    Every household without any wealth data is dropped.

    """
    start_year = options["start_year"]
    end_year = options["end_year"]
    wealth_data.set_index(["hid", "syear"], inplace=True)

    all_combinations = pd.concat(
        [
            pd.DataFrame({"hid": hid, "syear": range(start_year, end_year + 1)})
            for hid in wealth_data.index.get_level_values("hid").unique()
        ]
    )
    wealth_data_full = pd.merge(
        all_combinations, wealth_data, on=["hid", "syear"], how="left"
    )
    wealth_data_full.set_index(["hid", "syear"], inplace=True)
    wealth_data_full = wealth_data_full.groupby(level="hid").filter(
        lambda x: x["wealth"].notna().any()
    )

    return wealth_data_full


def extrapolate_wealth(household):
    """Linearly extrapolate wealth at the beginning and end of each group."""

    household = household.reset_index()
    wealth = household["wealth"].copy()

    # Extrapolate at the start
    if pd.isnull(wealth.iloc[0]):
        valid = wealth.dropna()
        if len(valid) >= 2:  # noqa: PLR2004
            x = valid.index[:2]
            y = valid.iloc[:2]
            slope = (y.iloc[1] - y.iloc[0]) / (x[1] - x[0])
            missing_start = wealth.index[wealth.index < x[0]]
            wealth.loc[missing_start] = y.iloc[0] - slope * (x[0] - missing_start)

    # Extrapolate at the end
    if pd.isnull(wealth.iloc[-1]):
        valid = wealth.dropna()
        if len(valid) >= 2:  # noqa: PLR2004
            x = valid.index[-2:]
            y = valid.iloc[-2:]
            slope = (y.iloc[1] - y.iloc[0]) / (x[1] - x[0])
            missing_end = wealth.index[wealth.index > x[1]]
            wealth.loc[missing_end] = y.iloc[1] + slope * (missing_end - x[1])

    household["wealth"] = wealth
    household.set_index("syear", inplace=True)
    # household.set_index(["hid", "syear"], inplace=True)

    return household


def interpolate_wealth(wealth_data):
    """Interpolate missing values in wealth data."""

    # for each household, create a row for each year between min and max syear
    min_max_syear = wealth_data.groupby("hid")["syear"].agg(["min", "max"])
    all_combinations = pd.concat(
        [
            pd.DataFrame({"hid": hid, "syear": range(row["min"], row["max"] + 1)})
            for hid, row in min_max_syear.iterrows()
        ]
    )
    wealth_data_full = pd.merge(
        all_combinations, wealth_data, on=["hid", "syear"], how="left"
    )

    # Set 'hid' and 'syear' as the index
    wealth_data_full.set_index(["hid", "syear"], inplace=True)
    wealth_data_full.sort_index(inplace=True)

    # Interpolate the missing values for each household
    wealth_data_full["wealth"] = wealth_data_full.groupby("hid")["wealth"].transform(
        lambda group: group.interpolate(method="linear")
    )
    return wealth_data_full
