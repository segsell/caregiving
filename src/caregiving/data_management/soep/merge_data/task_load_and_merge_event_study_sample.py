"""Merge SOEP modules."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.utils import table


def task_load_and_merge_event_study_sample(
    soep_c40_pgen: Path = SRC / "data" / "soep_c40" / "pgen.dta",
    soep_c40_ppathl: Path = SRC / "data" / "soep_c40" / "ppathl.dta",
    soep_c40_pl: Path = SRC / "data" / "soep_c40" / "pl.dta",
    soep_c40_hl: Path = SRC / "data" / "soep_c40" / "hl.dta",
    soep_c40_pequiv: Path = SRC / "data" / "soep_c40" / "pequiv.dta",
    soep_c40_pflege: Path = SRC / "data" / "soep_c40" / "pflege.dta",
    soep_c40_bioparen: Path = SRC / "data" / "soep_c40" / "bioparen.dta",
    soep_c40_bioagel: Path = SRC / "data" / "soep_c40" / "bioagel.dta",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_event_study_raw.csv",
    use_pequiv_age_var: bool = False,
) -> None:
    """Merge SOEP modules.

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
            "pgtatzeit",
            "pgvebzeit",
        ],
        convert_categoricals=False,
    )

    ppathl_data = pd.read_stata(
        soep_c40_ppathl,
        columns=[
            "pid",
            "hid",
            "syear",
            "sex",
            "gebjahr",
            "parid",
            "rv_id",
            "birthregion_ew",
            "migback",
        ],
        convert_categoricals=False,
    )
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
            "pli0046",  # how many hours per weekday support persons in care
            "pld0030",  # number of sisters
            "pld0032",  # number of brothers
            "plj0118_h",  # distance to mother
            "plj0119_h",  # distance to father
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
            "hlc0005_h",  # monthly net household income
            "hlc0120_h",  # monthly amount of savings
        ],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, hl_data, on=["hid", "syear"], how="left")

    # Data from 2003 on (children's birth years from 2000)
    # bioagel_data = pd.read_stata(
    #     soep_c40_bioagel,
    #     columns=[
    #         "pid",  # child id
    #         "pide",  # parent id
    #         "hid",
    #         "syear",
    #         "birthy",  # Geburtsjahr des Kindes
    #         "maincare",  # Mutter Hauptbetreuungsperson (1=ja; 2=Vater; 3=andere)
    #         "ill0",  # Einschraenkungen (1=ja; 2=nein)j
    #         "disord",  # Anhaltspunkte fuer Stoerungen (1,2=ja; 3=nein)
    #     ],
    #     convert_categoricals=False,
    # )

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

    if use_pequiv_age_var:
        merged_data["age"] = merged_data["d11101"].astype(int)
    else:
        merged_data["age"] = (merged_data["syear"] - merged_data["gebjahr"]).astype(int)

    # Set index
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C40 core.")

    merged_data.to_csv(path_to_save)
