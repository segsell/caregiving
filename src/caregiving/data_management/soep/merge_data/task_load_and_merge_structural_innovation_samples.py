"""Merge SOEP modules."""

import itertools
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.utils import table

SYEAR_IS = 2016
MALE = 1
FEMALE = 2

NO = 2

SPOUSE = 1
MOTHER_OR_FATHER = 2
MOTHER_OR_FATHER_IN_LAW = 3

PGSBIL_FACHHOCHSCHULREIFE = 3
PGSBIL_ABITUR = 4


# =====================================================================================
# SOEP-IS
# =====================================================================================


def task_load_and_merge_exog_care_sample(
    soep_is38_inno: Path = SRC / "data" / "soep_is" / "inno.dta",
    # soep_is38_pgen: Path = SRC / "data" / "soep_is" / "pgen.dta",
    soep_is38_ppfad: Path = SRC / "data" / "soep_is" / "ppfad.dta",
    # soep_is38_pl: Path = SRC / "data" / "soep_is" / "p.dta",
    soep_is38_biol: Path = SRC / "data" / "soep_is" / "bio.dta",
    # soep_is38_bioparen: Path = SRC / "data" / "soep_is" / "bioparen.dta",
    soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
    # soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_is_exog_care_data_raw.csv",
) -> None:
    """Merge SOEP-IS caregiving module.

    ip01: "Does a person within your circle of relatives, friends or close
    acquaintances need care or help because of age, disease or disability?
    This person can live in your household or outside."

    https://www.diw.de/documents/publikationen/73/diw_01.c.850217.de/diw_ssp1165.pdf

    """

    # Load SOEP-IS data
    inno_data = pd.read_stata(
        soep_is38_inno,
        columns=[
            "syear",
            "pid",
            "hid",
            "ip01",  # Informelle Pflege
            "ip02",  # Informelle Pflege Person
            "ip03",  # Informelle Pflege Geschlecht
            "ip05",  # Informelle Pflege Wohnort
            "ip06",  # Informelle Pflege persönlich
            "ip08",  # Informelle Pflege weitere Instituton / Person
            "ip07w",  # Informal Care hours of care on a typical workday
            # care from others: family and friends
            "ip08a1",  # Angehörige
            "ip08a4",  # Freunde/Bekannte/Nachbarn
            # care from others: professional
            "ip08a2",  # Wohlfahrtsverbände
            "ip08a3",  # priv. Pflegedienst
            "ip08a5",  # sonst. regelm. Pflegehilfe
            "ip10",  # Informelle Pflege weitere Person
        ],
        convert_categoricals=False,
    )
    ppfad_data = pd.read_stata(
        soep_is38_ppfad,
        columns=["pid", "sex", "gebjahr"],
        convert_categoricals=False,
    )
    # pl_data = pd.read_stata(
    #     soep_is38_pl,
    #     columns=["pid", "pld0029", "pld0030"],
    #     # columns=["pid", "nums"],
    #     convert_categoricals=False,
    # )
    # bioparen_data = pd.read_stata(
    #     soep_is38_bioparen,
    #     columns=["pid", "nums", "numb"],
    #     convert_categoricals=False,
    # )
    biol_data = pd.read_stata(
        soep_is38_biol,
        columns=["pid", "l0061", "l0062", "l0063"],
        convert_categoricals=False,
    )

    # Load SOEP core data
    pgen_data = pd.read_stata(
        soep_c38_pgen,
        columns=[
            "syear",
            "pid",
            "hid",
            # "pgemplst",
            "pgpsbil",
            # "pgstib",
        ],
        convert_categoricals=False,
    )

    # Filter to only include observations who have someone in need of care
    inno_data = inno_data[
        # (inno_data["ip01"] == 1)
        # & (inno_data["ip02"].isin([MOTHER_OR_FATHER, MOTHER_OR_FATHER_IN_LAW]))
        (inno_data["ip01"] == 1)
        & (inno_data["ip02"].isin([MOTHER_OR_FATHER]))
    ].copy()

    merged_data = pd.merge(inno_data, pgen_data, on=["pid", "hid"], how="outer")

    # Sort and fill pgpsbil by pid and year (syear_y)
    merged_data = merged_data.sort_values(by=["pid", "syear_y"])

    # Forward-fill then backward-fill pgpsbil within each pid group
    merged_data["pgpsbil"] = merged_data.groupby("pid")["pgpsbil"].ffill().bfill()

    # Drop syear from pgen

    # Trim to only include observations from inno_data in 2016
    merged_data = merged_data[merged_data["syear_x"] == SYEAR_IS].copy()
    merged_data.drop(columns=["syear_x", "syear_y"], inplace=True)
    merged_data.drop_duplicates(inplace=True)

    # Step 1: Define custom priority mapping
    pgpsbil_priority = {
        8: 1,
        6: 2,
        7: 3,
        1: 4,
        2: 5,
        3: 6,
        4: 7,
        5: 8,
    }

    # Any value not in the mapping (including negatives or NaN) gets lowest priority
    merged_data["pgpsbil_rank"] = merged_data["pgpsbil"].map(pgpsbil_priority).fillna(0)

    # Step 2: Sort by pid and pgpsbil_rank descending (highest rank = best)
    merged_data = merged_data.sort_values(
        by=["pid", "pgpsbil_rank"], ascending=[True, False]
    )

    # Step 3: Drop duplicates, keeping the highest-ranked row per pid
    merged_data = merged_data.drop_duplicates(subset="pid", keep="first")

    # Step 4: Drop helper column
    merged_data = merged_data.drop(columns="pgpsbil_rank")

    merged_data = merged_data[merged_data["pgpsbil"] > 0]
    merged_data["education"] = 0
    merged_data.loc[
        merged_data["pgpsbil"] == PGSBIL_FACHHOCHSCHULREIFE, "education"
    ] = 1
    merged_data.loc[merged_data["pgpsbil"] == PGSBIL_ABITUR, "education"] = 1

    merged_data = pd.merge(merged_data, ppfad_data, on=["pid"], how="left")
    merged_data["age"] = SYEAR_IS - merged_data["gebjahr"]
    merged_data["female"] = merged_data["sex"].map({MALE: 0, FEMALE: 1})

    merged_data = pd.merge(merged_data, biol_data, on=["pid"], how="left")

    del inno_data, pgen_data, ppfad_data, biol_data
    merged_data.set_index(["pid"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP IS 2016.")

    merged_data.to_csv(path_to_save)
