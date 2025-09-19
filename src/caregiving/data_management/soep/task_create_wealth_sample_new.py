"""Load and process wealth data."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import recode_sex
from caregiving.data_management.soep.variables import create_education_type
from caregiving.specs.task_write_specs import read_and_derive_specs
from caregiving.utils import table


def deflate_wealth(df, cpi_data, specs):
    """Deflate wealth using consumer price index."""

    # base_year_cpi = cpi_data.loc[
    #     cpi_data["syear"] == specs["reference_year"], "cpi"
    # ].iloc[0]
    # We need to set the index to the year
    cpi_data /= cpi_data.loc[specs["reference_year"]]

    # cpi_data["cpi_normalized"] = cpi_data["cpi"] / base_year_cpi

    data_merged = df.merge(cpi_data, left_on="syear", right_index=True)
    data_merged["wealth"] = data_merged["wealth"] / data_merged["cpi"]

    return data_merged


def calc_age_at_interview(df, drop_missing_month=True):
    """
    Calculate the age at interview date. Both float_interview and float_birth
    are created with missing nans for invalid data. So age will be invalid if
    one of them is invalid.

    Vars needed: gebjahr, gebmonat, syear, pmonin, ptagin
    """
    # Create birth and interview date
    # df = create_float_interview_date(df)
    total_days = 365 / 2
    df["float_interview"] = df.index.get_level_values("syear").values + total_days / 365
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
    soep_c40_pgen: Path = SRC / "data" / "soep_c40" / "pgen.dta",
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

    https://paneldata.org/soep-core/datasets/hwealth/w011ha

    Drop negative wealth values or set to zero?

    """
    specs = read_and_derive_specs(path_to_specs)
    cpi = pd.read_csv(path_to_cpi, index_col=0)
    cpi.rename(columns={"int_year": "syear"}, inplace=True)
    cpi.set_index("syear", inplace=True)
    # cpi.index.name = "syear    cpi.index.name = "syear""

    # 1) Load raw household wealth (hwealth.dta)
    wealth_data = pd.read_stata(
        soep_c40_hwealth,
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)
    wealth_data.set_index(["hid", "syear"], inplace=True)
    wealth_data.rename(columns={"w011ha": "wealth"}, inplace=True)

    print(
        "Number of wealth observations per survey year:"
        f" {wealth_data.groupby('syear')['wealth'].count()}"
    )

    # 4) Deflate wealth (CPI)
    wealth_data = deflate_wealth(wealth_data, cpi_data=cpi, specs=specs)

    # 2) Span to complete (hid × syear) within analysis window
    wealth_data_full = span_full_wealth_panel(wealth_data, specs)

    # 3) Attach personal data (age, partner flag, children, etc.) on (hid, syear)
    personal_data = perpare_personal_data(
        soep_c40_pgen, soep_c40_ppathl, soep_c40_pl, soep_c40_pequiv, specs=specs
    )
    wealth_and_personal_data = wealth_data_full.merge(
        personal_data, right_index=True, left_index=True, how="left"
    )

    # 5) No negatives
    wealth_and_personal_data.loc[wealth_and_personal_data["wealth"] < 0, "wealth"] = 0

    wealth_and_personal_data = recode_sex(wealth_and_personal_data)
    wealth_and_personal_data = create_education_type(wealth_and_personal_data)
    wealth_and_personal_data = wealth_and_personal_data[
        wealth_and_personal_data["sex"] >= 0
    ]

    # 6) Keep one row per (hid, syear)
    # wealth_and_personal_data = (
    #     wealth_and_personal_data[["wealth"]].reset_index()
    #     .drop_duplicates(subset=["hid", "pid", "syear"])
    # )
    wealth_and_personal_data = wealth_and_personal_data.reset_index().drop_duplicates(
        subset=["hid", "syear", "pid"]
    )

    wealth_and_personal_data = wealth_and_personal_data[
        wealth_and_personal_data["wealth"].notna()
    ]
    wealth_and_personal_data.to_csv(path_to_save_wealth_and_personal)
    # wealth_and_personal_data.to_pickle(file_name)

    # # Merge onto person-year data
    # data = data.reset_index()
    # data = data.merge(wealth_data_full, on=["hid", "syear"], how="left")
    # data.set_index(["pid", "syear"], inplace=True)

    # # if filter_missings:
    # data = data[data["wealth"].notna()]
    # print(str(len(data)) + " left after dropping people with missing wealth.")

    # data.to_csv(path_to_save_wealth_and_personal)

    # ==================================================================================

    # Plotting
    # plot_wealth_by_age_bins_and_age(
    #     df=wealth_and_personal_data, specs=specs, bin_width=5, edu_col="education"
    # )


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


def perpare_personal_data(
    soep_c40_pgen,
    soep_c40_ppathl,
    soep_c40_pl,
    soep_c40_pequiv,
    specs,
):
    """(Unchanged) Load ppathl/pl and attach person-year info by (hid, syear).
    Not used by the simplified pipeline, but kept in case other modules rely on it."""

    # Load SOEP core data
    pgen_data = pd.read_stata(
        soep_c40_pgen,
        columns=[
            "syear",
            "pid",
            "hid",
            # "pgemplst",
            # "pgexpft",
            # "pgexppt",
            # "pgstib",
            # "pgpartz",
            # "pglabgro",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )
    pgen_data.dropna(inplace=True)
    pgen_data["hid"] = pgen_data["hid"].astype(int)

    ppathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=["pid", "hid", "syear", "sex", "parid", "gebjahr", "gebmonat"],
        convert_categoricals=False,
    )
    ppathl_data.dropna(inplace=True)
    ppathl_data["hid"] = ppathl_data["hid"].astype(int)

    # 1st Merge
    merged_data = pd.merge(
        ppathl_data, pgen_data, on=["pid", "hid", "syear"], how="left"
    )

    pl_data_reader = pd.read_stata(
        soep_c40_pl,
        columns=[
            "pid",
            "hid",
            "syear",
            #  "pmonin", "ptagin"
        ],
        chunksize=100000,
        convert_categoricals=False,
    )
    pl_data = pd.concat([chunk for chunk in pl_data_reader], ignore_index=True)
    pl_data["hid"] = pl_data["hid"].astype(int)
    # pl_data.to_pickle(pl_intermediate_file)

    # 2nd Merge
    merged_data = pd.merge(merged_data, pl_data, on=["pid", "syear", "hid"], how="left")
    # merged_data = pd.merge(ppathl_data, pl_data, on=["pid", "syear", "hid"], how="left")

    # merged_data = calc_age_at_interview(merged_data)
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]

    merged_data.set_index(["pid", "syear"], inplace=True)
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

    return merged_data


def plot_wealth_by_age_bins_and_age(df, specs, bin_width=5, edu_col="education"):
    # keep only rows with needed columns
    df = df.copy()
    df = df[df["age"] < specs["end_age"]]

    needed = ["age", "wealth", edu_col]

    df = df.dropna(subset=needed)
    df = df[df[edu_col].isin([0, 1])]

    def make_age_bins(s, width):
        # left-closed bins [a, a+width)
        bins = range(
            int(np.floor(s.min() // width) * width),
            int(np.ceil((s.max() + 1) / width) * width) + 1,
            width,
        )
        return pd.cut(s, bins=bins, right=False)

    # split by education
    d0 = df[df[edu_col] == 0].copy()
    d1 = df[df[edu_col] == 1].copy()

    # compute for each subset
    for d in (d0, d1):
        d["age_bin"] = make_age_bins(d["age"], bin_width)

    # means by 5-year bin
    bin0 = d0.groupby("age_bin")["wealth"].mean()
    bin1 = d1.groupby("age_bin")["wealth"].mean()

    # means by exact age
    age0 = d0.groupby("age")["wealth"].mean().sort_index()
    age1 = d1.groupby("age")["wealth"].mean().sort_index()

    # nice bin labels like "30–34"
    def bin_labels(idx):
        labels = []
        for iv in idx:
            a = int(iv.left)
            b = int(iv.right) - 1
            labels.append(f"{a}–{b}")
        return labels

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

    # Top-left: education 0, 5-year bins
    labels0 = bin_labels(bin0.index)
    axes[0, 0].bar(range(len(bin0)), bin0.values)
    axes[0, 0].set_xticks(range(len(bin0)))
    axes[0, 0].set_xticklabels(labels0, rotation=45, ha="right")
    axes[0, 0].set_title("Avg Wealth by 5-Year Age Bin (education = 0)")
    axes[0, 0].set_ylabel("Average Wealth")

    # Top-right: education 0, exact age
    axes[0, 1].plot(age0.index.values, age0.values)
    axes[0, 1].set_title("Avg Wealth by Exact Age (education = 0)")
    axes[0, 1].set_xlabel("Age")

    # Bottom-left: education 1, 5-year bins
    labels1 = bin_labels(bin1.index)
    axes[1, 0].bar(range(len(bin1)), bin1.values)
    axes[1, 0].set_xticks(range(len(bin1)))
    axes[1, 0].set_xticklabels(labels1, rotation=45, ha="right")
    axes[1, 0].set_title("Avg Wealth by 5-Year Age Bin (education = 1)")
    axes[1, 0].set_ylabel("Average Wealth")
    axes[1, 0].set_xlabel("Age bin")

    # Bottom-right: education 1, exact age
    axes[1, 1].plot(age1.index.values, age1.values)
    axes[1, 1].set_title("Avg Wealth by Exact Age (education = 1)")
    axes[1, 1].set_xlabel("Age")

    # global y-label
    fig.supylabel("Average Wealth", x=0.03)
    plt.tight_layout()
    plt.show()
