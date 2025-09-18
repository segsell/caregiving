"""Load and process wealth data."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.specs.task_write_specs import read_and_derive_specs

from caregiving.utils import table


def deflate_wealth(df, cpi_data, specs):
    """Deflate wealth using consumer price index."""
    cpi_data = cpi_data.rename(columns={"int_year": "syear"})

    _base_year = specs["reference_year"]
    base_year_cpi = cpi_data.loc[cpi_data["syear"] == _base_year, "cpi"].iloc[0]

    cpi_data["cpi_normalized"] = cpi_data["cpi"] / base_year_cpi

    data_merged = df.merge(cpi_data, on="syear", how="left")
    data_merged["wealth"] = data_merged["wealth"] / data_merged["cpi_normalized"]

    return data_merged


def calc_age_at_interview(df, drop_missing_month=True):
    """
    Calculate the age at interview date. Both float_interview and float_birth
    are created with missing nans for invalid data. So age will be invalid if
    one of them is invalid.

    Vars needed: gebjahr, gebmonat, syear, pmonin, ptagin
    """
    # Create birth and interview date
    df = create_float_interview_date(df)
    df = _create_float_birth_date(df, drop_missing_month=drop_missing_month)

    # Calculate the age at interview date
    df["float_age"] = df["float_interview"] - df["float_birth_date"]
    df["age"] = np.floor(df["float_age"])
    return df


def _create_float_birth_date(df, drop_missing_month=True):
    """This functions creates a float birthdate, assuming to be born on the 1st of the month.

    It uses the variables gebjahr and gebmonat from ppathl or ppath.
    It allows to specify to drop missing month or otherwise use June/mid year
    instead.

    """
    # Make sure, all pids have same birth year everywhere. It might be that some
    # are missing for particular years
    df["gebjahr"] = df.groupby("pid")["gebjahr"].transform("max")
    df["gebmonat"] = df.groupby("pid")["gebmonat"].transform("max")

    invalid_year = df["gebjahr"] < 0
    invalid_month = df["gebmonat"] < 0

    df["float_birth_date"] = np.nan

    if drop_missing_month:
        valid_data = ~invalid_year & ~invalid_month
        df.loc[valid_data, "float_birth_date"] = df["gebjahr"] + df["gebmonat"] / 12
    else:
        month = df["gebmonat"].copy()
        month[invalid_month] = 6
        df.loc[~invalid_year, "float_birth_date"] = df["gebjahr"] + month / 12
    return df


def create_float_interview_date(df):
    """
    Create a float variable for the interview date in the format YYYY.MM.
    """
    months_array = df["pmonin"].values
    months_array[np.isnan(months_array)] = -1
    ivalid_mask = months_array == 0
    months_array[ivalid_mask] = -1
    months_array = months_array.astype(int)

    days_up_to_month = _create_n_days_for_month(months_array)
    total_days = days_up_to_month + df["ptagin"]

    total_invalid_mask = (df["ptagin"].values == 0) | ivalid_mask
    total_days[total_invalid_mask] = np.nan
    df["float_interview"] = df.index.get_level_values("syear").values + total_days / 365
    return df


def _create_n_days_for_month(current_months):
    """
    Create a variable for sum of number of days excluding current month.
    Code missing months as -1 days

    Args:

        -- current_months (int): Array of months with negative values for missing
    """
    # Create a list of the number of days excluding the current month
    n_days = np.array(
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=int
    )
    days_up_to_month = n_days[current_months - 1]
    days_up_to_month[current_months < 0] = -1

    return days_up_to_month


# =====================================================================================


def task_create_household_wealth_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    soep_c40_hwealth: Path = SRC / "data" / "soep_c40" / "hwealth.dta",
    soep_c40_ppathl: Path = SRC / "data" / "soep_c40" / "ppathl.dta",
    soep_c40_pl: Path = SRC / "data" / "soep_c40" / "pl.dta",
    soep_c40_pequiv: Path = SRC / "data" / "soep_c40" / "pequiv.dta",
    path_to_save_wealth: Annotated[Path, Product] = BLD
    / "data"
    / "soep_wealth_data.csv",
    path_to_save_wealth_and_personal: Annotated[Path, Product] = BLD
    / "data"
    / "soep_wealth_and_personal_data.csv",
) -> None:

    #     # def add_wealth_interpolate_and_deflate(
    #     data,
    #     path_dict,
    #     specs,
    #     filter_missings=True,
    #     load_wealth=False,
    #     use_processed_pl=True,
    # ):
    """Load household wealth, span to a full (hid, year) panel, deflate by CPI,
    set negatives to zero, and merge onto the person-year data. No interpolation or extrapolation.
    """
    specs = read_and_derive_specs(path_to_specs)
    cpi = pd.read_csv(path_to_cpi, index_col=0)

    # 1) Load raw household wealth (hwealth.dta)
    wealth_data = pd.read_stata(
        soep_c40_hwealth,
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)
    wealth_data.set_index(["hid", "syear"], inplace=True)
    wealth_data.rename(columns={"w011ha": "wealth"}, inplace=True)

    # 2) Span to complete (hid × syear) within analysis window
    wealth_data_full = span_full_wealth_panel(wealth_data, specs)

    # 3) Attach personal data (age, partner flag, children, etc.) on (hid, syear)
    wealth_data_full = add_personal_data(
        soep_c40_ppathl, soep_c40_pl, soep_c40_pequiv, specs, wealth_data_full
    )

    # 4) Deflate wealth (CPI)
    wealth_data_full = deflate_wealth(wealth_data_full, cpi_data=cpi, specs=specs)

    # 5) No negatives
    wealth_data_full.loc[wealth_data_full["wealth"] < 0, "wealth"] = 0

    # 6) Keep one row per (hid, syear)
    wealth_data_full = (
        wealth_data_full[["wealth"]]
        .reset_index()
        .drop_duplicates(subset=["hid", "syear"])
    )

    wealth_data_full.to_csv(path_to_save_wealth)
    # wealth_data_full.to_pickle(file_name)

    # Merge onto person-year data
    data = data.reset_index()
    data = data.merge(wealth_data_full, on=["hid", "syear"], how="left")
    data.set_index(["pid", "syear"], inplace=True)

    # if filter_missings:
    data = data[data["wealth"].notna()]
    print(str(len(data)) + " left after dropping people with missing wealth.")

    data.to_csv(path_to_save_wealth_and_personal)
    breakpoint()


def span_full_wealth_panel(wealth_data, specs):
    """Create rows for each household for each year between start_year and end_year.
    Households not present in the wealth file are not added."""
    hid_values = wealth_data.index.get_level_values("hid").unique()
    min_year_wealth = wealth_data.index.get_level_values("syear").min()
    max_year_wealth = wealth_data.index.get_level_values("syear").max()
    start_year = np.minimum(min_year_wealth, specs["start_year"])
    end_year = np.maximum(max_year_wealth, specs["end_year"])

    all_index = pd.MultiIndex.from_product(
        [hid_values, range(start_year, end_year + 1)], names=["hid", "syear"]
    )
    wealth_data_full = wealth_data.reindex(all_index, fill_value=np.nan, copy=True)
    return wealth_data_full


def load_wealth_data(soep_c38_path):
    """Load household wealth from SOEP hwealth.dta."""
    wealth_data = pd.read_stata(
        f"{soep_c38_path}/hwealth.dta",
        columns=["hid", "syear", "w011ha"],  # "w011hb", "w011hc", "w011hd", "w020h0"
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)
    wealth_data.set_index(["hid", "syear"], inplace=True)
    wealth_data.rename(columns={"w011ha": "wealth"}, inplace=True)
    return wealth_data


def add_personal_data(
    soep_c40_ppathl,
    soep_c40_pl,
    soep_c40_pequiv,
    specs,
    wealth_data_full,
    # use_processed_pl=False,
):
    """(Unchanged) Load ppathl/pl and attach person-year info by (hid, syear).
    Not used by the simplified pipeline, but kept in case other modules rely on it."""
    ppathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=["pid", "hid", "syear", "sex", "parid", "gebjahr", "gebmonat"],
        convert_categoricals=False,
    )
    ppathl_data.dropna(inplace=True)
    ppathl_data["hid"] = ppathl_data["hid"].astype(int)

    pl_data_reader = pd.read_stata(
        soep_c40_pl,
        columns=["pid", "hid", "syear", "pmonin", "ptagin"],
        chunksize=100000,
        convert_categoricals=False,
    )
    pl_data = pd.concat([chunk for chunk in pl_data_reader], ignore_index=True)
    pl_data["hid"] = pl_data["hid"].astype(int)
    # pl_data.to_pickle(pl_intermediate_file)

    merged_data = pd.merge(ppathl_data, pl_data, on=["pid", "syear", "hid"], how="left")
    merged_data.set_index(["pid", "syear"], inplace=True)
    merged_data = calc_age_at_interview(merged_data)
    merged_data = merged_data[merged_data["age"] >= specs["start_age"]]
    merged_data["is_par"] = np.where(merged_data["parid"] == -2, 0, 1)
    merged_data.reset_index(inplace=True)

    pequiv_data = pd.read_stata(
        soep_c40_pequiv,
        columns=["pid", "syear", "d11107"],
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="left")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)
    merged_data.reset_index(inplace=True)
    merged_data.set_index(["hid", "syear"], inplace=True)

    return wealth_data_full.merge(
        merged_data, right_index=True, left_index=True, how="left"
    )
