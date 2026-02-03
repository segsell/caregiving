"""Load and process wealth data."""

from pathlib import Path
from typing import Annotated

import numpy as np
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

# Constants for magic values
NO_PARTNER_ID = -2
MIN_MONTH = 1
MAX_MONTH = 12
MAX_DAY = 31
MIN_HH_SIZE_FOR_PARTNER_CHECK = 2
MAX_HH_SIZE = 2
MIN_VALID_OBSERVATIONS = 2


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
    soep_c40_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
    soep_c40_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
    soep_c40_pl: Path = SRC / "data" / "soep" / "pl.dta",
    soep_c40_pequiv: Path = SRC / "data" / "soep" / "pequiv.dta",
    path_to_pl_intermediate: Annotated[Path, Product] = BLD
    / "data"
    / "pl_structural_w.pkl",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_wealth_data_full.csv",
) -> None:
    """Create sample for wealth estimation with individual-level information."""

    filter_missings = True

    specs = read_and_derive_specs(path_to_specs)
    specs["start_year"] = 2001
    specs["end_year"] = 2023

    cpi = pd.read_csv(path_to_cpi, index_col=0)

    # Load individual-level data
    df_raw = pd.read_csv(path_to_raw, index_col=[0, 1])
    # Keep index as MultiIndex (pid, syear) for create_lagged_and_lead_variables

    print("Processing wealth data. This might take a while...")
    wealth_data = load_wealth_data(soep_c40_hwealth)
    wealth_data_full = _span_full_wealth_panel(wealth_data, specs)

    # Merge wealth data with pid/syear information
    wealth_data_full = add_personal_data(
        specs,
        wealth_data_full,
        soep_c40_ppathl=soep_c40_ppathl,
        soep_c40_pgen=soep_c40_pgen,
        soep_c40_pl=soep_c40_pl,
        soep_c40_pequiv=soep_c40_pequiv,
        path_to_pl_intermediate=path_to_pl_intermediate,
        use_processed_pl=True,
    )

    wealth_data_full = create_education_type(wealth_data_full, drop_missing=False)

    # Interpolate wealth for each household (consistent hh size)
    wealth_data_full = _interpolate_and_extrapolate_wealth_new(wealth_data_full, specs)
    # Reset index before deflating (deflate_wealth needs syear as column for merge)
    if isinstance(wealth_data_full.index, pd.MultiIndex):
        wealth_data_full = wealth_data_full.reset_index()
    # Deflate wealth
    wealth_data_full = deflate_wealth(wealth_data_full, cpi_data=cpi, specs=specs)
    # We do not allow for negative wealth values
    wealth_data_full.loc[wealth_data_full["wealth"] < 0, "wealth"] = 0
    # We only keep one wealth obs per household and year
    # Select only wealth column and drop duplicates
    wealth_data_full = wealth_data_full[["hid", "syear", "wealth"]].drop_duplicates(
        subset=["hid", "syear"]
    )

    # # Now merge with existing dataset on hid and syear
    # df_raw = create_choice_variable(df_raw)
    # df_raw = create_lagged_and_lead_variables(
    #     df_raw, specs, lead_job_sep=False, drop_missing_lagged_choice=False
    # )
    # Reset index to move pid and syear back to columns for merging
    df_raw = df_raw.reset_index()
    # df_raw["period"] = df_raw["age"] - specs["start_age"]
    # df_raw = create_policy_state(df_raw, specs)
    # df_raw = create_experience_variable_with_cap(
    #     df_raw, exp_cap=specs["start_age"] - 14
    # )
    # df_raw = create_education_type(df_raw)
    # df_raw = enforce_model_choice_restriction(df_raw, specs)

    df = df_raw.merge(wealth_data_full, on=["hid", "syear"], how="inner")

    # # Keep only required columns
    # type_dict = {
    #     "pid": "int32",
    #     "hid": "int32",
    #     "syear": "int16",
    #     "sex": "int8",
    #     "gebjahr": "int16",
    #     "age": "int8",
    #     "education": "int8",
    #     "lagged_choice": "float32",  # can be NA
    #     "policy_state": "int8",
    #     "policy_state_value": "int8",
    #     "experience": "int8",
    #     "wealth": "float32",
    # }
    # # Select and type columns
    # df = df[list(type_dict.keys())]
    # df = df.astype(type_dict)

    df.set_index(["pid", "syear"], inplace=True)
    print(str(len(df)) + " observations after merging wealth with individual data.")

    if filter_missings:
        before = len(df)
        df = df[df["wealth"].notna()]
        print_filter(before, len(df), "left after dropping people with missing wealth")

    # df.to_csv(path_to_save, index=True)
    wealth_data_full.to_csv(path_to_save, index=True)


def print_filter(before, after, msg):
    pct = (after - before) / before * 100 if before > 0 else 0
    print(f"{after} {msg} ({pct:+.2f}%)")


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


# =====================================================================================
# New wealth processing functions (adapted from friend's approach)
# =====================================================================================


def load_wealth_data(soep_c40_hwealth_path):
    """Load SOEP wealth data."""
    wealth_data = pd.read_stata(
        soep_c40_hwealth_path,
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)
    wealth_data.set_index(["hid", "syear"], inplace=True)
    wealth_data.rename(columns={"w011ha": "wealth"}, inplace=True)
    return wealth_data


def _span_full_wealth_panel(wealth_data, specs):
    """Creates additional rows for each household for each year between start_year and
    end_year. Every household without any wealth data is dropped."""
    hid_values = wealth_data.index.get_level_values("hid").unique()
    min_year_wealth = wealth_data.index.get_level_values("syear").min()
    max_year_wealth = wealth_data.index.get_level_values("syear").max()
    start_year = int(np.minimum(min_year_wealth, specs["start_year"]))
    end_year = int(np.maximum(max_year_wealth, specs["end_year"]))

    all_index = pd.MultiIndex.from_product(
        [hid_values, range(start_year, end_year + 1)], names=["hid", "syear"]
    )
    wealth_data_full = wealth_data.reindex(all_index, fill_value=np.nan, copy=True)
    return wealth_data_full


def add_personal_data(
    specs,
    wealth_data_full,
    soep_c40_ppathl,
    soep_c40_pgen,
    soep_c40_pl,
    soep_c40_pequiv,
    path_to_pl_intermediate,
    use_processed_pl=True,
):
    """Load ppathl data from SOEP."""
    # Start with ppathl. Everyone is in there even if not individually surveyed
    ppathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=[
            "pid",
            "hid",
            "syear",
            "sex",
            "parid",
            "gebjahr",
            "gebmonat",
        ],
        convert_categoricals=False,
    )
    ppathl_data.dropna(inplace=True)  # drop if most basic data is missing
    ppathl_data["hid"] = ppathl_data["hid"].astype(int)

    # Load SOEP core data
    pgen_data = pd.read_stata(
        soep_c40_pgen,
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
    # Merge pgen data with ppathl data
    merged_data = pd.merge(
        ppathl_data, pgen_data, on=["pid", "hid", "syear"], how="left"
    )

    if use_processed_pl and path_to_pl_intermediate.exists():
        pl_data = pd.read_pickle(path_to_pl_intermediate)
    else:
        # Add pl data
        pl_data_reader = pd.read_stata(
            soep_c40_pl,
            columns=[
                "pid",
                "hid",
                "syear",
                "pmonin",
                "ptagin",
            ],
            chunksize=100000,
            convert_categoricals=False,
        )
        pl_data = pd.DataFrame()
        for itm in pl_data_reader:
            pl_data = pd.concat([pl_data, itm])

        pl_data["hid"] = pl_data["hid"].astype(int)
        if use_processed_pl:
            path_to_pl_intermediate.parent.mkdir(parents=True, exist_ok=True)
            pl_data.to_pickle(path_to_pl_intermediate)

    merged_data = pd.merge(merged_data, pl_data, on=["pid", "syear", "hid"], how="left")

    # Set index to pid and syear, create age and filter by age
    merged_data.set_index(["pid", "syear"], inplace=True)
    merged_data = calc_age_at_interview(merged_data)
    merged_data = merged_data[merged_data["age"] >= specs["start_age"]]
    merged_data["is_par"] = np.where(merged_data["parid"] == NO_PARTNER_ID, 0, 1)
    merged_data.reset_index(inplace=True)

    # Add number of children in household to merged_data
    pequiv_data = pd.read_stata(
        soep_c40_pequiv,
        columns=["pid", "syear", "d11107"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="left")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)
    merged_data.reset_index(inplace=True, drop=True)
    merged_data.set_index(["hid", "syear"], inplace=True)

    # Merge wealth data with hid/syear information and return
    return wealth_data_full.merge(
        merged_data, right_index=True, left_index=True, how="left"
    )


def calc_age_at_interview(df, impute_missing_month=True, impute_missing_interview=True):
    """Calculate the age at interview date."""
    # Create birth and interview date
    df = create_float_interview_date(
        df, impute_missing_interview=impute_missing_interview
    )
    df = create_float_birth_date(df, impute_missing_month=impute_missing_month)

    # Calculate the age at interview date
    df["float_age"] = df["float_interview"] - df["float_birth_date"]
    # Convert to int, handling NaN values properly
    # Keep as float to preserve NaN, will be filtered later
    df["age"] = np.floor(df["float_age"])
    # Convert valid values to int (NaN will remain as NaN)
    valid_mask = df["age"].notna() & np.isfinite(df["age"])
    if valid_mask.any():
        df.loc[valid_mask, "age"] = df.loc[valid_mask, "age"].astype(int)
    return df


def create_float_birth_date(df, impute_missing_month=False):
    """Create float birthdate, assuming to be born on the 1st of the month."""
    # Make sure all pids have same birth year everywhere
    df["gebjahr"] = df.groupby("pid")["gebjahr"].transform("max")
    if "gebmonat" in df.columns:
        df["gebmonat"] = df.groupby("pid")["gebmonat"].transform("max")

    invalid_year = df["gebjahr"] < 0
    if "gebmonat" in df.columns:
        invalid_month = df["gebmonat"] < 0
    else:
        invalid_month = pd.Series(False, index=df.index)

    df["float_birth_date"] = np.nan

    if not impute_missing_month:
        valid_data = ~invalid_year & ~invalid_month
        if "gebmonat" in df.columns:
            df.loc[valid_data, "float_birth_date"] = df["gebjahr"] + df["gebmonat"] / 12
        else:
            df.loc[~invalid_year, "float_birth_date"] = df["gebjahr"] + 6 / 12
    else:
        if "gebmonat" in df.columns:
            month = df["gebmonat"].copy()
            month[invalid_month] = 6
            df.loc[~invalid_year, "float_birth_date"] = df["gebjahr"] + month / 12
        else:
            df.loc[~invalid_year, "float_birth_date"] = df["gebjahr"] + 6 / 12
    return df


def create_float_interview_date(df, impute_missing_interview=False):
    """Create float interview date from syear, pmonin, ptagin."""
    if "pmonin" not in df.columns or "ptagin" not in df.columns:
        # If interview date columns not available, use mid-year
        if "syear" in df.index.names:
            df["float_interview"] = df.index.get_level_values("syear") + 0.5
        elif "syear" in df.columns:
            df["float_interview"] = df["syear"] + 0.5
        else:
            raise ValueError("syear not found in columns or index")
        return df

    month = df["pmonin"].copy()
    day = df["ptagin"].copy()

    if impute_missing_interview:
        month, day = _impute_missing_interview_dates(df, month, day)

    # Check for invalid data
    invalid_month = month.isna() | (month < MIN_MONTH) | (month > MAX_MONTH)
    invalid_day = day.isna() | (day < MIN_MONTH) | (day > MAX_DAY)
    any_invalid = invalid_month | invalid_day

    # Calculate days from year start
    total_days = pd.Series(np.nan, index=df.index)
    valid_mask = ~any_invalid

    if valid_mask.any():
        days_up_to_month = _cumulative_days_before_month(month[valid_mask])
        total_days[valid_mask] = days_up_to_month + day[valid_mask]

    # Create float date
    if "syear" in df.columns:
        year_values = df["syear"]
    elif "syear" in df.index.names:
        year_values = df.index.get_level_values("syear")
    else:
        raise ValueError("syear not found in columns or index")

    df["float_interview"] = year_values + total_days / 365
    return df


def _cumulative_days_before_month(months):
    """Get cumulative days before each month (Jan=0, Feb=31, Mar=59, etc.)"""
    days_before = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    month_indices = months.astype(int) - 1
    return pd.Series(days_before[month_indices], index=months.index)


def _impute_missing_interview_dates(df, month, day):
    """Impute missing interview dates using mode of valid interviews per year"""
    valid_month_mask = (
        df["pmonin"].notna() & (df["pmonin"] >= MIN_MONTH) & (df["pmonin"] <= MAX_MONTH)
    )

    if valid_month_mask.any():
        if "syear" in df.columns:
            syear_values = df.loc[valid_month_mask, "syear"]
        elif "syear" in df.index.names:
            syear_values = df.index.get_level_values("syear")[valid_month_mask]
        else:
            raise ValueError("syear not found in columns or index")

        valid_months_df = pd.DataFrame(
            {
                "pmonin": df.loc[valid_month_mask, "pmonin"].values,
                "syear": syear_values.values,
            }
        )

        mode_month_by_year = valid_months_df.groupby("syear")["pmonin"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
        )

        if len(mode_month_by_year) > 0:
            overall_mode = valid_months_df["pmonin"].mode()
            fallback_month = overall_mode.iloc[0] if not overall_mode.empty else 6
        else:
            fallback_month = 6
    else:
        mode_month_by_year = pd.Series(dtype=float)
        fallback_month = 6

    if "syear" in df.columns:
        year_values = df["syear"]
    elif "syear" in df.index.names:
        year_values = df.index.get_level_values("syear")
    else:
        raise ValueError("syear not found in columns or index")

    imputed_months = year_values.map(mode_month_by_year).fillna(fallback_month)

    missing_month = month.isna() | (month < MIN_MONTH) | (month > MAX_MONTH)
    month = month.where(~missing_month, imputed_months)

    missing_day = day.isna() | (day <= 0)
    day = day.where(~missing_day, 15)

    return month, day


def _interpolate_and_extrapolate_wealth_new(wealth_data_full, specs):
    """Interpolate wealth for each household (consistent hh size) and extrapolate."""
    # Reset index to work with columns
    wealth_data_full = wealth_data_full.reset_index()

    # Interpolate between existing points with consistent household size
    wealth_data_full["hh_size_adjusted"] = wealth_data_full.groupby(["hid", "syear"])[
        "pid"
    ].transform("count")
    # Only keep households with data for at least one person in a given year
    wealth_data_full = wealth_data_full[wealth_data_full["hh_size_adjusted"] > 0]

    # In households with 2+ people above start age, drop people with no partner
    wealth_data_full = wealth_data_full[
        ~(
            (wealth_data_full["hh_size_adjusted"] >= MIN_HH_SIZE_FOR_PARTNER_CHECK)
            & (wealth_data_full["is_par"] == 0)
        )
    ]

    # Recalculate household size
    wealth_data_full["hh_size_adjusted"] = wealth_data_full.groupby(["hid", "syear"])[
        "pid"
    ].transform("count")

    # Drop households with more than 2 people above start age
    wealth_data_full = wealth_data_full[
        wealth_data_full["hh_size_adjusted"] <= MAX_HH_SIZE
    ]

    # Interpolate wealth between observations
    wealth_data_full["wealth"] = wealth_data_full.groupby(
        ["hid", "hh_size_adjusted", "pid"]
    )["wealth"].transform(
        lambda group: group.interpolate(method="linear", limit_area="inside")
    )

    # Keep hh if the hh (by size and pid) has at least one non-NaN wealth value
    mask = wealth_data_full.groupby(["hid", "hh_size_adjusted", "pid"])[
        "wealth"
    ].transform(lambda x: x.notna().any())
    mask &= wealth_data_full["is_par"].notna()
    wealth_data_full = wealth_data_full[mask]

    # Extrapolate wealth for each household at the start and end
    extrapolated = wealth_data_full.groupby(["hid", "hh_size_adjusted", "pid"]).apply(
        _extrapolate_wealth_linear
    )
    # Merge the extrapolated wealth back
    extrapolated = extrapolated.drop(
        columns=["hid", "hh_size_adjusted", "pid"], errors="ignore"
    )
    extrapolated = extrapolated.reset_index()
    wealth_data_full = (
        wealth_data_full.reset_index()
        .drop(columns="wealth", errors="ignore")
        .merge(
            extrapolated[["hid", "syear", "wealth"]], on=["hid", "syear"], how="left"
        )
        .set_index(["hid", "syear", "pid"])
    )

    # Drop duplicates caused by extrapolation in hh with 2 people
    wealth_data_full = wealth_data_full[
        ~wealth_data_full.index.duplicated(keep="first")
    ]

    # Find mean hh age for each household (rounded to nearest 5 years)
    wealth_data_full["mean_hh_age_rounded"] = wealth_data_full.groupby(
        ["hid", "syear"]
    )["float_age"].transform(lambda x: (x.mean() / 5.0).round(0) * 5)

    # Create dataframe with percentiles (deciles) of wealth
    wealth_data_unique = wealth_data_full.reset_index().drop_duplicates(
        subset=["hid", "syear"]
    )

    # Compute deciles for each group
    deciles_list = []
    for (syear, age, size), group in wealth_data_unique.groupby(
        ["syear", "mean_hh_age_rounded", "hh_size_adjusted"]
    ):
        deciles = _compute_deciles(group)
        if deciles is not None and not deciles.isna().all():
            for i, val in enumerate(deciles):
                deciles_list.append(
                    {
                        "syear": syear,
                        "mean_hh_age_rounded": age,
                        "hh_size_adjusted": size,
                        "decile_idx": i,
                        "value": val,
                    }
                )

    if deciles_list:
        deciles_df = pd.DataFrame(deciles_list)
        # Pivot to get deciles as columns
        wealth_deciles_df = deciles_df.pivot_table(
            index=["syear", "mean_hh_age_rounded", "hh_size_adjusted"],
            columns="decile_idx",
            values="value",
        )
    else:
        wealth_deciles_df = pd.DataFrame()

    # Impute wealth for households with only one valid observation
    imputed = wealth_data_full.groupby(["hid", "hh_size_adjusted", "pid"]).apply(
        lambda group: _impute_wealth_from_deciles(group, wealth_deciles_df)
    )

    # Merge the imputed wealth back
    imputed = imputed.drop(columns=["hid", "hh_size_adjusted", "pid"], errors="ignore")
    imputed = imputed.reset_index()
    wealth_data_full = (
        wealth_data_full.reset_index()
        .drop(columns="wealth", errors="ignore")
        .merge(imputed[["hid", "syear", "wealth"]], on=["hid", "syear"], how="left")
        .set_index(["hid", "syear", "pid"])
    )

    # Drop duplicates caused by imputation in hh with 2 people
    wealth_data_full = wealth_data_full[
        ~wealth_data_full.index.duplicated(keep="first")
    ]

    return wealth_data_full


def _extrapolate_wealth_linear(household):
    """Linearly extrapolate wealth if at least 2 valid observations exist."""
    household_int = household.reset_index()
    wealth = household_int["wealth"].copy()
    syear = household_int["syear"].copy()
    valid = wealth.dropna()

    if len(valid) >= MIN_VALID_OBSERVATIONS:
        # Linear extrapolation at start
        if pd.isnull(wealth.iloc[0]):
            first_valid_index = valid.index[0]
            missing_start_idxs = wealth.index[wealth.index < first_valid_index]
            y = valid.iloc[:2].values
            x = syear.iloc[valid.iloc[:2].index].values
            slope = (y[1] - y[0]) / (x[1] - x[0]) if (x[1] - x[0]) != 0 else 0
            wealth.loc[missing_start_idxs] = y[0] - slope * (
                first_valid_index - missing_start_idxs
            )

        # Linear extrapolation at end
        if pd.isnull(wealth.iloc[-1]):
            last_valid_index = valid.index[-1]
            missing_end_idxs = wealth.index[wealth.index > last_valid_index]
            y = valid.iloc[-2:].values
            x = syear.iloc[valid.iloc[-2:].index].values
            slope = (y[1] - y[0]) / (x[1] - x[0]) if (x[1] - x[0]) != 0 else 0
            wealth.loc[missing_end_idxs] = y[1] + slope * (
                missing_end_idxs - last_valid_index
            )

    household_int["wealth"] = wealth
    return household_int.set_index("syear")


def _impute_wealth_from_deciles(household, wealth_deciles_df):  # noqa: PLR0912, PLR0915
    """Impute wealth for a household with only one valid observation using deciles."""
    household_int = household.reset_index()
    valid = household_int["wealth"].dropna()

    if len(valid) != 1:
        return household_int.set_index("syear")

    valid_index = valid.index[0]
    row_valid = household_int.loc[valid_index]
    valid_value = row_valid["wealth"]
    syear_valid = row_valid["syear"]
    age_valid = row_valid["mean_hh_age_rounded"]
    size_valid = row_valid["hh_size_adjusted"]

    # Find percentile approx via linear interpolation between deciles
    if len(wealth_deciles_df) == 0:
        return household_int.set_index("syear")

    try:
        deciles = wealth_deciles_df.loc[(syear_valid, age_valid, size_valid)]
        if isinstance(deciles, pd.Series):
            deciles = deciles.values
        elif isinstance(deciles, pd.DataFrame):
            deciles = deciles.iloc[0].values
    except (KeyError, IndexError):
        # If no deciles exist, leave wealth as NaN
        return household_int.set_index("syear")

    decile_bounds = np.arange(10, 100, 10) / 100  # 0.1 to 0.9
    f_x_j = 0.1  # since deciles are 10% intervals

    # Find first class j, such that F(x_{j_o}) >= p
    below_10th = True
    above_90th = False
    p = 0.5  # default
    F_x_j_u = 0.1  # default

    for i in range(len(deciles) - 1):
        if deciles[i] <= valid_value <= deciles[i + 1]:
            below_10th = False
            x_j_u = deciles[i]
            x_j_o = deciles[i + 1]
            F_x_j_u = decile_bounds[i]
            # Linear interpolation formula for p of valid_value
            p = (
                F_x_j_u + f_x_j * (valid_value - x_j_u) / (x_j_o - x_j_u)
                if x_j_u != x_j_o
                else F_x_j_u
            )
            break
    else:  # valid wealth above 9th decile
        above_90th = True

    # Impute missing values
    for idx, row in household_int.iterrows():
        if pd.notnull(row["wealth"]):
            continue
        else:
            target_key = (
                row["syear"],
                row["mean_hh_age_rounded"],
                row["hh_size_adjusted"],
            )
            try:
                target_deciles = wealth_deciles_df.loc[target_key]
                if isinstance(target_deciles, pd.Series):
                    target_deciles = target_deciles.values
                elif isinstance(target_deciles, pd.DataFrame):
                    target_deciles = target_deciles.iloc[0].values
            except (KeyError, IndexError):
                # If no deciles exist for that year, leave wealth as NaN
                continue

            if below_10th:  # If valid wealth was below 10th decile
                wealth_guess = target_deciles[0]
            elif above_90th:  # If valid wealth was above 90th decile
                wealth_guess = target_deciles[-1]
            else:
                # p and F_x_j_u from previous step
                x_j_u = target_deciles[int(F_x_j_u * 10) - 1]
                x_j_o = target_deciles[int(F_x_j_u * 10)]
                # Impute wealth using linear interpolation formula
                wealth_guess = x_j_u + ((p - F_x_j_u) / f_x_j) * (x_j_o - x_j_u)

            household_int.loc[idx, "wealth"] = wealth_guess

    return household_int.set_index("syear")


def _compute_deciles(group):
    """Compute deciles (10% to 90%) for a group. Returns Series of 9 values."""
    if group["wealth"].isna().all():
        return pd.Series(np.full(9, np.nan), index=list(range(9)))
    else:
        percentiles = np.nanpercentile(group["wealth"], np.arange(10, 100, 10))
        return pd.Series(percentiles, index=list(range(9)))
