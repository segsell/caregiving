"""Merge SOEP modules."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pytask import Product

from caregiving.config import BLD, SRC

SYEAR_IS = 2016
MALE = 1
FEMALE = 2

SPOUSE = 1
MOTHER_OR_FATHER = 2
MOTHER_OR_FATHER_IN_LAW = 3

PGSBIL_FACHHOCHSCHULREIFE = 3
PGSBIL_ABITUR = 4


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def _get_cols_biobirth():
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


# # =====================================================================================
# # Estimation sample
# # =====================================================================================


# def task_load_and_merge_estimation_sample(
#     soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
#     soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
#     soep_c38_pl: Path = SRC / "data" / "soep" / "pl.dta",
#     soep_c38_hl: Path = SRC / "data" / "soep" / "hl.dta",
#     soep_c38_pequiv: Path = SRC / "data" / "soep" / "pequiv.dta",
#     soep_c38_pflege: Path = SRC / "data" / "soep" / "pflege.dta",
#     soep_c38_biobirth: Path = SRC / "data" / "soep" / "biobirth.dta",
#     path_to_save: Annotated[Path, Product] = BLD
#     / "data"
#     / "soep_estimation_data_raw.csv",
# ) -> None:
#     """Merge SOEP modules.

#     https://paneldata.org/soep-core/datasets/pequiv/
#     """

#     # Load SOEP core data
#     pgen_data = pd.read_stata(
#         soep_c38_pgen,
#         columns=[
#             "syear",
#             "pid",
#             "hid",
#             "pgemplst",
#             "pgexpft",
#             "pgexppt",
#             "pgstib",
#             "pgpartz",
#             "pglabgro",
#             "pgpsbil",
#         ],
#         convert_categoricals=False,
#     )

#     ppathl_data = pd.read_stata(
#         soep_c38_ppathl,
#         columns=["pid", "hid", "syear", "sex", "gebjahr", "parid", "rv_id"],
#         convert_categoricals=False,
#     )
#     # Merge pgen data with pathl data and hl data
#     merged_data = pd.merge(
#         pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
#     )

#     # Add pl data
#     pl_data_reader = pd.read_stata(
#         soep_c38_pl,
#         columns=["pid", "hid", "syear", "plb0304_h"],
#         chunksize=100000,
#         convert_categoricals=False,
#     )
#     pl_data = pd.DataFrame()
#     for itm in pl_data_reader:
#         pl_data = pd.concat([pl_data, itm])
#     merged_data = pd.merge(
#         merged_data, pl_data, on=["pid", "hid", "syear"], how="inner"
#     )

#     # get household level data
#     hl_data = pd.read_stata(
#         soep_c38_hl,
#         columns=["hid", "syear", "hlc0043"],
#         convert_categoricals=False,
#     )
#     merged_data = pd.merge(merged_data, hl_data, on=["hid", "syear"], how="left")
#     pequiv_data = pd.read_stata(
#         # d11107: number of children in household
#         # d11101: age of individual
#         # m11126: Self-Rated Health Status
#         # m11124: Disability Status of Individual
#         # m111050-m11123: Activities of Daily Living
#         soep_c38_pequiv,
#         columns=["pid", "syear", "d11107", "d11101", "m11126", "m11124"],
#         convert_categoricals=False,
#     )
#     merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
#     merged_data.rename(columns={"d11107": "children"}, inplace=True)

#     # Pflege
#     pflege = pd.read_stata(soep_c38_pflege, convert_categoricals=False)
#     pflege = pflege[pflege["pnrcare"] >= 0]
#     merged_data_with_pflege = pd.merge(
#         merged_data,
#         pflege,
#         left_on=["pid", "syear"],
#         right_on=["pnrcare", "syear"],
#         how="left",
#     )
#     merged_data_with_pflege.rename(columns={"pid_x": "pid"}, inplace=True)

#     # Age of youngest child
#     cols_biobirth = _get_cols_biobirth()
#     biobirth = pd.read_stata(
#         soep_c38_biobirth, columns=cols_biobirth, convert_categoricals=False
#     )
#     merged_data = pd.merge(merged_data, biobirth, on="pid", how="left")

#     # Set index
#     merged_data["age"] = merged_data["d11101"].astype(int)
#     merged_data.set_index(["pid", "syear"], inplace=True)
#     print(str(len(merged_data)) + " observations in SOEP C38 core.")

#     merged_data.to_csv(path_to_save)


# # =====================================================================================
# # Partner transition sample
# # =====================================================================================


# def task_load_and_merge_partner_transition_sample(
#     soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
#     soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
#     soep_c38_pequiv: Path = SRC / "data" / "soep" / "pequiv.dta",
#     soep_c38_biobirth: Path = SRC / "data" / "soep" / "biobirth.dta",
#     path_to_save: Annotated[Path, Product] = BLD
#     / "data"
#     / "soep_partner_transition_data_raw.csv",
# ) -> None:
#     # Load SOEP core data
#     pgen_data = pd.read_stata(
#         soep_c38_pgen,
#         columns=[
#             "syear",
#             "pid",
#             "hid",
#             "pgemplst",
#             "pgpsbil",
#             "pgstib",
#         ],
#         convert_categoricals=False,
#     )
#     ppathl_data = pd.read_stata(
#         soep_c38_ppathl,
#         columns=["syear", "pid", "hid", "sex", "parid", "gebjahr"],
#         convert_categoricals=False,
#     )
#     pequiv_data = pd.read_stata(
#         soep_c38_pequiv,
#         # d11107: number of children in household
#         columns=["pid", "syear", "d11107"],
#     )
#     merged_data = pd.merge(
#         pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
#     )
#     merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")

#     # Age of youngest child
#     cols_biobirth = _get_cols_biobirth()
#     biobirth = pd.read_stata(
#         soep_c38_biobirth, columns=cols_biobirth, convert_categoricals=False
#     )
#     merged_data = pd.merge(merged_data, biobirth, on="pid", how="left")

#     merged_data.rename(columns={"d11107": "children"}, inplace=True)
#     merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
#     merged_data.set_index(["pid", "syear"], inplace=True)
#     print(str(len(merged_data)) + " observations in SOEP C38 core.")

#     merged_data.to_csv(path_to_save)


# # =====================================================================================
# # Wage sample
# # =====================================================================================


# def task_load_and_merge_wage_sample(
#     soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
#     soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
#     path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_wage_data_raw.csv",
# ) -> None:
#     # Load SOEP core data
#     pgen_data = pd.read_stata(
#         soep_c38_pgen,
#         columns=[
#             "syear",
#             "pid",
#             "hid",
#             "pgemplst",
#             "pgexpft",
#             "pgexppt",
#             "pgstib",
#             "pglabgro",
#             "pgpsbil",
#             "pgvebzeit",
#         ],
#         convert_categoricals=False,
#     )
#     pathl_data = pd.read_stata(
#         soep_c38_ppathl,
#         columns=["pid", "hid", "syear", "sex", "gebjahr"],
#         convert_categoricals=False,
#     )

#     # Merge pgen data with pathl data and hl data
#     merged_data = pd.merge(
#         pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
#     )

#     merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
#     del pgen_data, pathl_data
#     merged_data.set_index(["pid", "syear"], inplace=True)
#     print(str(len(merged_data)) + " observations in SOEP C38 core.")

#     merged_data.to_csv(path_to_save)


# # =====================================================================================
# # Job separation sample
# # =====================================================================================


# def task_load_and_merge_job_separation_sample(
#     soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
#     soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
#     soep_c38_pl: Path = SRC / "data" / "soep" / "pl.dta",
#     path_to_save: Annotated[Path, Product] = BLD
#     / "data"
#     / "soep_job_separation_data_raw.csv",
# ) -> None:
#     # Load SOEP core data
#     pgen_data = pd.read_stata(
#         soep_c38_pgen,
#         columns=[
#             "syear",
#             "pid",
#             "hid",
#             "pgemplst",
#             "pgstib",
#             "pgpsbil",
#         ],
#         convert_categoricals=False,
#     )
#     pathl_data = pd.read_stata(
#         soep_c38_ppathl,
#         columns=["pid", "hid", "syear", "sex", "gebjahr"],
#         convert_categoricals=False,
#     )

#     pl_data_reader = pd.read_stata(
#         soep_c38_pl,
#         columns=["pid", "hid", "syear", "plb0304_h", "plb0282_h"],
#         chunksize=100000,
#         convert_categoricals=False,
#     )
#     pl_data = pd.DataFrame()

#     for itm in pl_data_reader:
#         pl_data = pd.concat([pl_data, itm])

#     # Merge pgen data with pathl data and hl data
#     merged_data = pd.merge(
#         pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
#     )
#     # Merge pgen data with pathl data and hl data
#     merged_data = pd.merge(
#         merged_data, pl_data, on=["pid", "hid", "syear"], how="inner"
#     )

#     merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
#     del pgen_data, pathl_data
#     merged_data.set_index(["pid", "syear"], inplace=True)
#     print(str(len(merged_data)) + " observations in SOEP C38 core.")

#     merged_data.to_csv(path_to_save)


# # =====================================================================================
# # Partner wage sample
# # =====================================================================================


# def task_load_and_merge_partner_wage_sample(
#     soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
#     soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
#     path_to_save: Annotated[Path, Product] = BLD
#     / "data"
#     / "soep_partner_wage_data_raw.csv",
# ) -> None:

#     # Load SOEP core data
#     pgen_data = pd.read_stata(
#         soep_c38_pgen,
#         columns=[
#             "syear",
#             "pid",
#             "hid",
#             "pgemplst",
#             "pgexpft",
#             "pgexppt",
#             "pgstib",
#             "pglabgro",
#             "pgpsbil",
#             "pgvebzeit",
#         ],
#         convert_categoricals=False,
#     )
#     pathl_data = pd.read_stata(
#         soep_c38_ppathl,
#         columns=["pid", "hid", "parid", "syear", "sex", "gebjahr"],
#         convert_categoricals=False,
#     )

#     # Merge pgen data with pathl data and hl data
#     merged_data = pd.merge(
#         pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
#     )

#     merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
#     del pgen_data, pathl_data
#     merged_data.set_index(["pid", "syear"], inplace=True)
#     print(str(len(merged_data)) + " observations in SOEP C38 core.")

#     merged_data.to_csv(path_to_save)


# # =====================================================================================
# # Health sample
# # =====================================================================================


# def task_load_and_merge_health_sample(
#     soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
#     soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
#     soep_c38_pequiv: Path = SRC / "data" / "soep" / "pequiv.dta",
#     path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_health_data_raw.csv",
# ):
#     # Load SOEP core data
#     pgen_data = pd.read_stata(
#         soep_c38_pgen,
#         columns=[
#             "syear",
#             "pid",
#             "hid",
#             "pgemplst",
#             "pgpsbil",
#             "pgstib",
#         ],
#         convert_categoricals=False,
#     )
#     ppathl_data = pd.read_stata(
#         soep_c38_ppathl,
#         columns=["syear", "pid", "hid", "sex", "parid", "gebjahr"],
#         convert_categoricals=False,
#     )
#     pequiv_data = pd.read_stata(
#         # m11126: Self-Rated Health Status
#         # m11124: Disability Status of Individual
#         soep_c38_pequiv,
#         columns=["pid", "syear", "m11126", "m11124"],
#         convert_categoricals=False,
#     )
#     merged_data = pd.merge(
#         pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
#     )
#     merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
#     merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
#     merged_data.set_index(["pid", "syear"], inplace=True)
#     print(str(len(merged_data)) + " observations in SOEP C38 core.")

#     merged_data.to_csv(path_to_save)


# =====================================================================================
# SOEP-IS
# =====================================================================================


def task_load_and_merge_exog_care_sample(
    soep_is38_inno: Path = SRC / "data" / "soep_is" / "inno.dta",
    soep_is38_pgen: Path = SRC / "data" / "soep_is" / "pgen.dta",
    soep_is38_ppfad: Path = SRC / "data" / "soep_is" / "ppfad.dta",
    soep_is38_pl: Path = SRC / "data" / "soep_is" / "p.dta",
    soep_is38_biol: Path = SRC / "data" / "soep_is" / "bio.dta",
    soep_is38_bioparen: Path = SRC / "data" / "soep_is" / "bioparen.dta",
    soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
    soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
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
            # "ip05", # Informelle Pflege Wohnort
            "ip06",  # Informelle Pflege persönlich
            "ip08",  # Informelle Pflege weitere Instituton / Person
            # care from others: family and friends
            "ip08a1",  # Angehörige
            "ip08a4",  # Freunde/Bekannte/Nachbarn
            # care from others: professional
            "ip08a2",  # Wohlfahrtsverbände
            "ip08a3",  # priv. Pflegedienst
            "ip08a5",  # sonst. regelm. Pflegehilfe
            "ip10",  # Informelle Pflege weitere Person
            # "bil3",  #  Schulabschluss BRD
            # "sp11",
            # "bilyear",
        ],
        convert_categoricals=False,
    )
    ppfad_data = pd.read_stata(
        soep_is38_ppfad,
        columns=["pid", "sex", "gebjahr"],
        convert_categoricals=False,
    )
    pl_data = pd.read_stata(
        soep_is38_pl,
        columns=["pid", "pld0029", "pld0030"],
        # columns=["pid", "nums"],
        convert_categoricals=False,
    )
    bioparen_data = pd.read_stata(
        soep_is38_bioparen,
        columns=["pid", "nums", "numb"],
        convert_categoricals=False,
    )
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
        (inno_data["ip01"] == 1)
        & (inno_data["ip02"].isin([MOTHER_OR_FATHER, MOTHER_OR_FATHER_IN_LAW]))
    ].copy()

    merged_data = pd.merge(inno_data, pgen_data, on=["pid", "hid"], how="outer")

    # Sort and fill pgpsbil by pid and year (syear_y)
    merged_data = merged_data.sort_values(by=["pid", "syear_y"])

    # Forward-fill then backward-fill pgpsbil within each pid group
    merged_data["pgpsbil"] = merged_data.groupby("pid")["pgpsbil"].ffill().bfill()

    # Drop syear from pgen

    # Trim to only include observations from inno_data in 2016
    merged_data = merged_data[merged_data["syear_x"] == 2016].copy()
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

    # Initialize dummies with 0
    merged_data["other_informal_care"] = 0
    merged_data["formal_care"] = 0

    # Assign 1 if conditions are met
    merged_data.loc[
        (merged_data["ip08a1"] == 1) | (merged_data["ip08a4"] == 1),
        "other_informal_care",
    ] = 1

    merged_data.loc[
        (merged_data["ip08a2"] == 1)
        | (merged_data["ip08a3"] == 1)
        | (merged_data["ip08a5"] == 1),
        "formal_care",
    ] = 1

    merged_data["only_own_informal_care"] = 0
    merged_data["only_other_informal_care"] = 0
    merged_data["only_formal_care"] = 0

    # Assign 1 where conditions are met
    merged_data.loc[
        (merged_data["ip06"] == 1) & (merged_data["other_informal_care"] == 0),
        # & (merged_data["formal_care"] == 0),
        "only_own_informal_care",
    ] = 1

    merged_data.loc[
        (merged_data["ip06"] == 2) & (merged_data["other_informal_care"] == 1),
        # & (merged_data["formal_care"] == 0),
        "only_other_informal_care",
    ] = 1

    # only_other_informal_care
    # 0    622
    # 1    320

    merged_data.loc[
        (merged_data["ip06"] == 2)  # No
        # & (merged_data["other_informal_care"] == 0)
        & (merged_data["formal_care"] == 1),
        "only_formal_care",
    ] = 1

    merged_data["has_sister"] = np.nan  # start with NaN (or use np.nan)
    merged_data.loc[merged_data["l0063"] > 0, "has_sister"] = 1
    merged_data.loc[merged_data["l0063"].isin([0, -2]), "has_sister"] = 0

    # Replace negative values with NaN for summing purposes
    l0062_clean = merged_data["l0062"].where(merged_data["l0062"] >= 0)
    l0063_clean = merged_data["l0063"].where(merged_data["l0063"] >= 0)

    # Calculate n_siblings as sum of non-negative values
    merged_data["n_siblings"] = l0062_clean + l0063_clean

    est_sample = merged_data[merged_data["female"] == 1]

    # First, drop any rows with missing values in the relevant variables
    reg_data = est_sample[
        ["other_informal_care", "age", "has_sister", "education"]
    ].dropna()

    # Run logistic regression
    model1 = smf.logit(
        "other_informal_care ~ age + has_sister + education", data=reg_data
    ).fit()
    print(model1.summary())

    # Define age bins: [40–44], [45–49], ..., [65–69]
    # bins = list(range(40, 70, 5))
    # labels = [f"age_{b}_{b+4}" for b in bins[:-1]]  # e.g. age_40_44, age_45_49, ...

    bins = list(range(40, 70, 5))  # [40, 45, 50, 55, 60, 65]
    bins + [70]  # → [40, 45, 50, 55, 60, 65, 70] → 7 edges = 6 bins
    labels = [f"age_{b}_{b+4}" for b in bins[:-1]]  # only 5 labels!

    # Assign bin labels (values outside range will be NaN)
    est_sample["age_bin"] = pd.cut(
        est_sample["age"], bins=bins + [70], right=False, labels=labels
    )

    # Create dummy variables from age_bin
    age_dummies = pd.get_dummies(
        est_sample["age_bin"], prefix="", prefix_sep="", drop_first=True
    )

    # Combine with other variables
    reg_data_bins = pd.concat(
        [est_sample[["other_informal_care", "has_sister", "eduation"]], age_dummies],
        axis=1,
    )
    reg_data_bins = reg_data_bins.dropna()  # drop missing values

    age_bin_terms = " + ".join(age_dummies.columns)

    formula = f"other_informal_care ~ has_sister + pgpsbil + {age_bin_terms}"

    # Run the model
    model2 = smf.logit(formula, data=reg_data_bins).fit()
    breakpoint()

    # Load SOEP core data
    pgen_is_data = pd.read_stata(
        soep_c38_pgen,
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
        soep_c38_ppathl,
        columns=["pid", "hid", "syear", "sex", "gebjahr"],
        convert_categoricals=False,
    )

    # # Merge pgen data with pathl data and hl data
    # merged_data = pd.merge(
    #     pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    # )

    # merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    # del pgen_data, pathl_data
    # merged_data.set_index(["pid", "syear"], inplace=True)
    # print(str(len(merged_data)) + " observations in SOEP C38 core.")

    # merged_data.to_csv(path_to_save)
