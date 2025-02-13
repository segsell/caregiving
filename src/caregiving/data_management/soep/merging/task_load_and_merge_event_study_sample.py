"""Merge SOEP modules."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_load_and_merge_event_study_sample(
    soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
    soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
    soep_c38_pl: Path = SRC / "data" / "soep" / "pl.dta",
    soep_c38_hl: Path = SRC / "data" / "soep" / "hl.dta",
    soep_c38_pequiv: Path = SRC / "data" / "soep" / "pequiv.dta",
    soep_c38_pflege: Path = SRC / "data" / "soep" / "pflege.dta",
    soep_c38_bioparen: Path = SRC / "data" / "soep" / "bioparen.dta",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_event_study_raw.csv",
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
            "pgtatzeit",
            "pgvebzeit",
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
        columns=[
            "pid",
            "hid",
            "syear",
            "plb0304_h",  # why job ended
            "pli0046",  # how many hours per weekday support persons in care
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

    # Parent information
    biparen = pd.read_stata(
        soep_c38_bioparen,
        columns=[
            "pid",
            "mybirth",
            "fybirth",
            "mydeath",
            "fydeath",
            "sibl",
            "nums",
            "numb",
        ],
        convert_categoricals=False,
    )
    # all unique pid, left joint
    merged_data = pd.merge(merged_data, biparen, on="pid", how="left")

    # Set index
    merged_data["age"] = merged_data["d11101"].astype(int)
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")

    merged_data.to_csv(path_to_save)


# =====================================================================================
# Partner transition sample
# =====================================================================================


def load_and_merge_partner_transition_sample(
    soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
    soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
    soep_c38_pequiv: Path = SRC / "data" / "soep" / "pequiv.dta",
    soep_c38_biobirth: Path = SRC / "data" / "soep" / "biobirth.dta",
    # path_to_save: Annotated[Path, Product] = BLD
    # / "data"
    # / "soep_partner_transition_data_raw.csv",
) -> None:
    """Load partner sample"""

    # Load SOEP core data
    pgen_data = pd.read_stata(
        soep_c38_pgen,
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
        soep_c38_ppathl,
        columns=["syear", "pid", "hid", "sex", "parid", "gebjahr"],
        convert_categoricals=False,
    )
    pequiv_data = pd.read_stata(
        soep_c38_pequiv,
        # d11107: number of children in household
        columns=["pid", "syear", "d11107"],
    )
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")

    merged_data.rename(columns={"d11107": "children"}, inplace=True)
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")

    # merged_data.to_csv(path_to_save)
