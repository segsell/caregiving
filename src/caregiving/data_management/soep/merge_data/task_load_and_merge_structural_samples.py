"""Merge SOEP modules."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, SRC


def table(df_col):
    """Return a table with the count of unique values in a column."""
    return pd.crosstab(df_col, columns="Count")["Count"]


def _get_cols_biobirth():
    """Get children's birth year columns for biobirth data."""

    cols_biobirth = [
        "pid",
        "sumkids",
        # "biokids",
        # "bioinfo",
        # "bioage",
    ]
    _child_columns = []
    prefix = "kidgeb"
    # for prefix in ["kidpnr", "kidgeb", "kidsex", "kidmon"]:
    _child_columns += [
        f"{prefix}0{i}" for i in range(1, 10)
    ]  # kidpnr01 to kidpnr09, etc.
    _child_columns += [
        f"{prefix}{i}" for i in range(10, 20)
    ]  # kidpnr10 to kidpnr19, etc.
    cols_biobirth.extend(_child_columns)

    return cols_biobirth


# =====================================================================================
# Estimation sample
# =====================================================================================
def task_load_and_merge_estimation_sample(
    soep_c40_pgen: Path = SRC / "data" / "soep_c40" / "pgen.dta",
    soep_c40_ppathl: Path = SRC / "data" / "soep_c40" / "ppathl.dta",
    soep_c40_pl: Path = SRC / "data" / "soep_c40" / "pl.dta",
    soep_c40_hl: Path = SRC / "data" / "soep_c40" / "hl.dta",
    soep_c40_pequiv: Path = SRC / "data" / "soep_c40" / "pequiv.dta",
    soep_c40_pflege: Path = SRC / "data" / "soep_c40" / "pflege.dta",
    soep_c40_bioparen: Path = SRC / "data" / "soep_c40" / "bioparen.dta",
    soep_c40_biobirth: Path = SRC / "data" / "soep_c40" / "biobirth.dta",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_estimation_data_raw.csv",
) -> None:
    """Merge main estimation sample.

    https://paneldata.org/soep-core/datasets/pequiv/
    """

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

    ppathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=["pid", "hid", "syear", "sex", "gebjahr", "parid", "rv_id"],
        convert_categoricals=False,
    )
    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )

    # Add pl data
    pl_data_reader = pd.read_stata(
        soep_c40_pl,
        columns=[
            "pid",
            "hid",
            "syear",
            "plb0304_h",  # why job ended
            # caregiving model
            "pli0046",  # how many hours per weekday support persons in care
            "pld0030",  # number of sisters
            "pld0032",  # number of brothers
            "plj0118_h",  # distance to mother
            "plj0119_h",  # distance to father
            # inheritance
            "plc0375_v1",  # Erbschaft (jemals)
            "plc0375_v2",  # Erbschaft (letzte 15 Jahre)
            "plc0383_h",  # Erbschaft/Schenkung Betrag (Euro)
            "plc0376_v1",  # Jahr Erbschaft 1. Person
            "plc0386_v1",  # Jahr Erbschaft 2. Person
            "plc0396_v1",  # Jahr Erbschaft 3. Person
        ],
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
        soep_c40_hl,
        columns=[
            "hid",
            "syear",
            "hlc0043",  # Kindergeld für wie viele Kinder
            # "hlc0005_h",  # monthly net household income
            # "hlc0120_h",  # monthly amount of savings
            "hlf0155_h",  # Unterkunftsart (Wohn)heim
            "hlf0291",
        ],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, hl_data, on=["hid", "syear"], how="left")
    pequiv_data = pd.read_stata(
        # d11107: number of children in household
        # d11101: age of individual
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        # m111050-m11123: Activities of Daily Living
        soep_c40_pequiv,
        columns=["pid", "syear", "d11107", "d11101", "m11126", "m11124"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)

    # Parent information
    biparen = pd.read_stata(
        soep_c40_bioparen,
        columns=[
            "pid",
            "mybirth",
            "fybirth",
            "mydeath",
            "fydeath",
            "locchild1",  #  lives now in same area where grew up
        ],
        convert_categoricals=False,
    )
    # all unique pid, left joint
    merged_data = pd.merge(merged_data, biparen, on="pid", how="left")

    # Pflege
    # pflege = pd.read_stata(soep_c40_pflege, convert_categoricals=False)
    # pflege = pflege[pflege["pnrcare"] >= 0]
    # merged_data_with_pflege = pd.merge(
    #     merged_data,
    #     pflege,
    #     left_on=["pid", "syear"],
    #     right_on=["pnrcare", "syear"],
    #     how="left",
    # )
    # merged_data_with_pflege.rename(columns={"pid_x": "pid"}, inplace=True)

    # Age of youngest child
    cols_biobirth = _get_cols_biobirth()
    biobirth = pd.read_stata(
        soep_c40_biobirth, columns=cols_biobirth, convert_categoricals=False
    )
    merged_data = pd.merge(merged_data, biobirth, on="pid", how="left")

    # Set index
    # merged_data["age"] = merged_data["d11101"].astype(int)
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C40 core.")

    merged_data.to_csv(path_to_save)


@pytask.mark.skip(reason="Labor-only model")
def _task_load_and_merge_estimation_sample(
    soep_c40_pgen: Path = SRC / "data" / "soep_c40" / "pgen.dta",
    soep_c40_ppathl: Path = SRC / "data" / "soep_c40" / "ppathl.dta",
    soep_c40_pl: Path = SRC / "data" / "soep_c40" / "pl.dta",
    soep_c40_hl: Path = SRC / "data" / "soep_c40" / "hl.dta",
    soep_c40_pequiv: Path = SRC / "data" / "soep_c40" / "pequiv.dta",
    soep_c40_pflege: Path = SRC / "data" / "soep_c40" / "pflege.dta",
    soep_c40_biobirth: Path = SRC / "data" / "soep_c40" / "biobirth.dta",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_estimation_data_raw.csv",
) -> None:
    """Merge main estimation sample.

    https://paneldata.org/soep-core/datasets/pequiv/
    """

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

    ppathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=["pid", "hid", "syear", "sex", "gebjahr", "parid", "rv_id"],
        convert_categoricals=False,
    )
    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )

    # Add pl data
    pl_data_reader = pd.read_stata(
        soep_c40_pl,
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
        soep_c40_hl,
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
        soep_c40_pequiv,
        columns=["pid", "syear", "d11107", "d11101", "m11126", "m11124"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)

    # Pflege
    pflege = pd.read_stata(soep_c40_pflege, convert_categoricals=False)
    pflege = pflege[pflege["pnrcare"] >= 0]
    merged_data_with_pflege = pd.merge(
        merged_data,
        pflege,
        left_on=["pid", "syear"],
        right_on=["pnrcare", "syear"],
        how="left",
    )
    merged_data_with_pflege.rename(columns={"pid_x": "pid"}, inplace=True)

    # Age of youngest child
    cols_biobirth = _get_cols_biobirth()
    biobirth = pd.read_stata(
        soep_c40_biobirth, columns=cols_biobirth, convert_categoricals=False
    )
    merged_data = pd.merge(merged_data, biobirth, on="pid", how="left")

    # Set index
    merged_data["age"] = merged_data["d11101"].astype(int)
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C40 core.")

    merged_data.to_csv(path_to_save)


# =====================================================================================
# Partner transition sample
# =====================================================================================


def task_load_and_merge_partner_transition_sample(
    soep_c40_pgen: Path = SRC / "data" / "soep_c40" / "pgen.dta",
    soep_c40_ppathl: Path = SRC / "data" / "soep_c40" / "ppathl.dta",
    soep_c40_pequiv: Path = SRC / "data" / "soep_c40" / "pequiv.dta",
    soep_c40_biobirth: Path = SRC / "data" / "soep_c40" / "biobirth.dta",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_partner_transition_data_raw.csv",
) -> None:
    """Merge exogenous partner transition sample."""

    # Load SOEP core data
    pgen_data = pd.read_stata(
        soep_c40_pgen,
        columns=[
            "syear",
            "pid",
            "hid",
            "pgemplst",
            "pgpsbil",
            "pgstib",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=["syear", "pid", "hid", "sex", "parid", "gebjahr"],
        convert_categoricals=False,
    )
    pequiv_data = pd.read_stata(
        soep_c40_pequiv,
        # d11107: number of children in household
        columns=["pid", "syear", "d11107"],
    )
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")

    # Age of youngest child
    cols_biobirth = _get_cols_biobirth()
    biobirth = pd.read_stata(
        soep_c40_biobirth, columns=cols_biobirth, convert_categoricals=False
    )
    merged_data = pd.merge(merged_data, biobirth, on="pid", how="left")

    merged_data.rename(columns={"d11107": "children"}, inplace=True)
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C40 core.")

    merged_data.to_csv(path_to_save)


# =====================================================================================
# Wage sample
# =====================================================================================


def task_load_and_merge_wage_sample(
    soep_c40_pgen: Path = SRC / "data" / "soep_c40" / "pgen.dta",
    soep_c40_ppathl: Path = SRC / "data" / "soep_c40" / "ppathl.dta",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_wage_data_raw.csv",
) -> None:
    """Merge exogenous wage sample."""

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
            "pglabgro",
            "pgpsbil",
            "pgvebzeit",
        ],
        convert_categoricals=False,
    )
    pathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=["pid", "hid", "syear", "sex", "gebjahr"],
        convert_categoricals=False,
    )

    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    )

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    del pgen_data, pathl_data
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C40 core.")

    merged_data.to_csv(path_to_save)


# =====================================================================================
# Job separation sample
# =====================================================================================


def task_load_and_merge_job_separation_sample(
    soep_c40_pgen: Path = SRC / "data" / "soep_c40" / "pgen.dta",
    soep_c40_ppathl: Path = SRC / "data" / "soep_c40" / "ppathl.dta",
    soep_c40_pl: Path = SRC / "data" / "soep_c40" / "pl.dta",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_job_separation_data_raw.csv",
) -> None:
    """Merge stochastic job transition sample."""
    # Load SOEP core data
    pgen_data = pd.read_stata(
        soep_c40_pgen,
        columns=[
            "syear",
            "pid",
            "hid",
            "pgemplst",
            "pgstib",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )
    pathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=["pid", "hid", "syear", "sex", "gebjahr"],
        convert_categoricals=False,
    )

    pl_data_reader = pd.read_stata(
        soep_c40_pl,
        columns=["pid", "hid", "syear", "plb0304_h", "plb0282_h"],
        chunksize=100000,
        convert_categoricals=False,
    )
    pl_data = pd.DataFrame()

    for itm in pl_data_reader:
        pl_data = pd.concat([pl_data, itm])

    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        merged_data, pl_data, on=["pid", "hid", "syear"], how="inner"
    )

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    del pgen_data, pathl_data
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C40 core.")

    merged_data.to_csv(path_to_save)


# =====================================================================================
# Partner wage sample
# =====================================================================================


def task_load_and_merge_partner_wage_sample(
    soep_c40_pgen: Path = SRC / "data" / "soep_c40" / "pgen.dta",
    soep_c40_ppathl: Path = SRC / "data" / "soep_c40" / "ppathl.dta",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_partner_wage_data_raw.csv",
) -> None:
    """Merge exogenous partner wage sample."""

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
            "pglabgro",
            "pgpsbil",
            "pgvebzeit",
        ],
        convert_categoricals=False,
    )
    pathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=["pid", "hid", "parid", "syear", "sex", "gebjahr"],
        convert_categoricals=False,
    )

    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    )

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    del pgen_data, pathl_data
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C40 core.")

    merged_data.to_csv(path_to_save)


# =====================================================================================
# Health sample
# =====================================================================================


def task_load_and_merge_health_sample(
    soep_c40_pgen: Path = SRC / "data" / "soep_c40" / "pgen.dta",
    soep_c40_ppathl: Path = SRC / "data" / "soep_c40" / "ppathl.dta",
    soep_c40_pequiv: Path = SRC / "data" / "soep_c40" / "pequiv.dta",
    soep_c40_hl: Path = SRC / "data" / "soep_c40" / "hl.dta",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_health_data_raw.csv",
):
    """Merge stochastic health transition sample."""

    # Load SOEP core data
    pgen_data = pd.read_stata(
        soep_c40_pgen,
        columns=[
            "syear",
            "pid",
            "hid",
            "pgemplst",
            "pgpsbil",
            "pgstib",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=["syear", "pid", "hid", "sex", "parid", "gebjahr"],
        convert_categoricals=False,
    )
    pequiv_data = pd.read_stata(
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        soep_c40_pequiv,
        columns=["pid", "syear", "m11126", "m11124"],
        convert_categoricals=False,
    )
    # get household level data
    hl_data = pd.read_stata(
        soep_c40_hl,
        columns=[
            "hid",
            "syear",
            "hlc0043",  # Kindergeld für wie viele Kinder
            # "hlc0005_h",  # monthly net household income
            # "hlc0120_h",  # monthly amount of savings
            "hlf0155_h",  # Unterkunftsart (Wohn)heim
        ],
        convert_categoricals=False,
    )

    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data = pd.merge(merged_data, hl_data, on=["hid", "syear"], how="left")

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C40 core.")

    merged_data.to_csv(path_to_save)
