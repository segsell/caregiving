"""Load and process wealth data."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.specs.task_write_specs import read_and_derive_specs

from caregiving.utils import table


def task_create_household_wealth_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    soep_c38_hwealth: Path = SRC / "data" / "soep" / "hwealth.dta",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_wealth_data_old.csv",
) -> None:
    """Create sample for wealth estimation."""

    specs = read_and_derive_specs(path_to_specs)
    specs["start_year"] = 2010
    specs["end_year"] = 2017

    cpi = pd.read_csv(path_to_cpi, index_col=0)

    wealth_data = pd.read_stata(
        soep_c38_hwealth,
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)

    wealth_data = trim_and_rename(wealth_data)
    wealth_data_full = interpolate_and_extrapolate_wealth(wealth_data, specs)

    # data = data.merge(wealth_data_full, on=["hid", "syear"], how="left")
    data = deflate_wealth(wealth_data_full, cpi_data=cpi, specs=specs)

    data.loc[data["wealth"] < 0, "wealth"] = 0
    # data.set_index(["pid", "syear"], inplace=True)
    # data = data[(data["wealth"].notna())]

    print(str(len(data)) + " left after dropping people with missing wealth.")

    data.to_csv(path_to_save)


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


def deflate_wealth(df, cpi_data, specs):
    """Deflate wealth using consumer price index."""
    cpi_data = cpi_data.rename(columns={"int_year": "syear"})

    _base_year = specs["reference_year"]
    base_year_cpi = cpi_data.loc[cpi_data["syear"] == _base_year, "cpi"].iloc[0]

    cpi_data["cpi_normalized"] = cpi_data["cpi"] / base_year_cpi

    data_merged = df.merge(cpi_data, on="syear", how="left")
    data_merged["wealth"] = data_merged["wealth"] / data_merged["cpi_normalized"]

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
