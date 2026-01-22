"""Merge SOEP modules."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, SRC

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


@pytask.mark.soep_is
def task_load_and_merge_innovation_sample(
    soep_is40_inno: Path = SRC / "data" / "soep_is" / "inno.dta",
    # soep_is38_pgen: Path = SRC / "data" / "soep_is" / "pgen.dta",
    soep_is40_ppfad: Path = SRC / "data" / "soep_is" / "ppfad.dta",
    # soep_is38_pl: Path = SRC / "data" / "soep_is" / "p.dta",
    soep_is40_biol: Path = SRC / "data" / "soep_is" / "bio.dta",
    # soep_is38_bioparen: Path = SRC / "data" / "soep_is" / "bioparen.dta",
    soep_c40_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
    # soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_is_care_raw.csv",
) -> None:
    """Merge SOEP-IS caregiving module.

    ip01: "Does a person within your circle of relatives, friends or close
    acquaintances need care or help because of age, disease or disability?
    This person can live in your household or outside."

    https://www.diw.de/documents/publikationen/73/diw_01.c.850217.de/diw_ssp1165.pdf

    """

    # Load SOEP-IS data
    inno_data = pd.read_stata(
        soep_is40_inno,
        columns=[
            "syear",
            "pid",
            "hid",
            "ip02",  # Wer ist die pflegebedürftige Person?
            "ip02_2",  # Wer ist die pflegebedürftige Person? 2
            "ip02_3",  # Wer ist die pflegebedürftige Person? 3
            "ip02_4",  # Wer ist die pflegebedürftige Person? 4
            "ip03",  # Geschlecht der pflegebedürftigen Person
            "ip03_2",  # Geschlecht der pflegebedürftigen Person 2
            "ip03_3",  # Geschlecht der pflegebedürftigen Person 3
            "ip03_4",  # Geschlecht der pflegebedürftigen Person 4
            "ip04",  # Leistung aus Pflegeversicherung?
            "ip04_2",  # Leistung aus Pflegeversicherung?
            "ip04_3",  # Leistung aus Pflegeversicherung?
            "ip04_4",  # Leistung aus Pflegeversicherung?
            "ip04a",  # Pflegestufe
            "ip04a_2",  # Pflegestufe
            "ip04a_3",  # Pflegestufe
            "ip04a_4",  # Pflegestufe
            "ip05",  # Wo lebt diese Person?
            "ip05_2",  # Wo lebt diese Person? 2
            "ip05_3",  # Wo lebt diese Person? 3
            "ip05_4",  # Wo lebt diese Person? 4
            "ip06",  # Pflege von Ihen?
            "ip06_2",  # Pflege von Ihen? 2
            "ip06_3",  # Pflege von Ihen? 3
            "ip06_4",  # Pflege von Ihen? 4
            "ip08",  # Weitere Pflege?
            "ip08_2",  # Weitere Pflege? 2
            "ip08_3",  # Weitere Pflege? 3
            "ip08_4",  # Weitere Pflege? 4
            "ip08a2",  # Weitere Pflege? Wohlfahrtsverbände
            "ip08a2_2",  # Weitere Pflege? Wohlfahrtsverbände 2
            "ip08a2_3",  # Weitere Pflege? Wohlfahrtsverbände 3
            "ip08a2_4",  # Weitere Pflege? Wohlfahrtsverbände 4
            "ip08a3",  # Weitere Pflege? priv. Pflegedienst
            "ip08a3_2",  # Weitere Pflege? priv. Pflegedienst 2
            "ip08a3_3",  # Weitere Pflege? priv. Pflegedienst 3
            "ip08a3_4",  # Weitere Pflege? priv. Pflegedienst 4
            "ip08a5",  # Weitere Pflege? sonst. regelm. Pflegehilfe
            "ip08a5_2",  # Weitere Pflege? sonst. regelm. Pflegehilfe 2
            "ip08a5_3",  # Weitere Pflege? sonst. regelm. Pflegehilfe 3
            "ip08a5_4",  # Weitere Pflege? sonst. regelm. Pflegehilfe 4
            "ip08wn",  # Weiß nicht
            "ip08wn_2",  # Weiß nicht 2
            "ip08wn_3",  # Weiß nicht 3
            "ip08wn_4",  # Weiß nicht 4
            "ip01",  # Informelle Pflege
            # "ip02",  # Informelle Pflege Person
            # "ip03",  # Informelle Pflege Geschlecht
            # "ip05",  # Informelle Pflege Wohnort
            # "ip06",  # Informelle Pflege persönlich
            # "ip08",  # Informelle Pflege weitere Instituton / Person
            # "ip07w",  # Informal Care hours of care on a typical workday
            # # care from others: family and friends
            # "ip08a1",  # Angehörige
            # "ip08a4",  # Freunde/Bekannte/Nachbarn
            # # care from others: professional
            # "ip08a2",  # Wohlfahrtsverbände
            # "ip08a3",  # priv. Pflegedienst
            # "ip08a5",  # sonst. regelm. Pflegehilfe
            # "ip10",  # Informelle Pflege weitere Person
            # "ip02_2",  # Informelle Pflege Person 2
            # "ip03_2",  # Informelle Pflege Geschlecht 2
            # "ip05_2",  # Informelle Pflege Wohnort 2
            # "ip06_2",  # Informelle Pflege persönlich 2
            # "ip08a1_2",  # Angehörige 2
            # "ip08a4_2",  # Freunde/Bekannte/Nachbarn 2
            # "ip02_3",  # Informelle Pflege Person 3
            # "ip03_3",  # Informelle Pflege Geschlecht 3
            # "ip05_3",  # Informelle Pflege Wohnort 3
            # "ip06_3",  # Informelle Pflege persönlich 3
            # "ip08a1_3",  # Angehörige 3
            # "ip08a4_3",  # Freunde/Bekannte/Nachbarn 3
            # "ip02_4",  # Informelle Pflege Person 4
            # "ip03_4",  # Informelle Pflege Geschlecht 4
            # "ip05_4",  # Informelle Pflege Wohnort 4
            # "ip06_4",  # Informelle Pflege persönlich 4
            # "ip08a1_4",  # Angehörige 4
            # "ip08a4_4",  # Freunde/Bekannte/Nachbarn 4
        ],
        convert_categoricals=False,
    )
    ppfad_data = pd.read_stata(
        soep_is40_ppfad,
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
        soep_is40_biol,
        columns=["pid", "l0061", "l0062", "l0063"],
        convert_categoricals=False,
    )

    # Load SOEP core data
    pgen_data = pd.read_stata(
        soep_c40_pgen,
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
            "ip02_2",  # Informelle Pflege Person 2
            "ip03_2",  # Informelle Pflege Geschlecht 2
            "ip05_2",  # Informelle Pflege Wohnort 2
            "ip06_2",  # Informelle Pflege persönlich 2
            "ip08a1_2",  # Angehörige 2
            "ip08a4_2",  # Freunde/Bekannte/Nachbarn 2
            "ip02_3",  # Informelle Pflege Person 3
            "ip03_3",  # Informelle Pflege Geschlecht 3
            "ip05_3",  # Informelle Pflege Wohnort 3
            "ip06_3",  # Informelle Pflege persönlich 3
            "ip08a1_3",  # Angehörige 3
            "ip08a4_3",  # Freunde/Bekannte/Nachbarn 3
            "ip02_4",  # Informelle Pflege Person 4
            "ip03_4",  # Informelle Pflege Geschlecht 4
            "ip05_4",  # Informelle Pflege Wohnort 4
            "ip06_4",  # Informelle Pflege persönlich 4
            "ip08a1_4",  # Angehörige 4
            "ip08a4_4",  # Freunde/Bekannte/Nachbarn 4
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


def plot_logit_prediction_by_age(model, age_range=(40, 80), step=1):
    """
    Plot predicted probability vs. age for all combinations of has_sister and education.

    Parameters:
    - model: fitted statsmodels Logit model (with age and age_squared)
    - age_range: tuple, (min_age, max_age)
    - step: step size for age increments

    """

    # Age range
    ages = np.arange(age_range[0], age_range[1] + 1, step)

    # All 4 combinations of has_sister and education (0 or 1)
    combos = [(hs, ed) for hs in (0, 1) for ed in (0, 1)]

    plt.figure(figsize=(10, 6))

    for has_sister, education in combos:
        # Build prediction DataFrame
        pred_df = pd.DataFrame(
            {
                "age": ages,
                "age_squared": ages**2,
                "has_sister": has_sister,
                "education": education,
            }
        )

        # Predict probabilities
        pred_df["predicted_prob"] = model.predict(pred_df)

        # Line label
        label = f"has_sister={has_sister}, education={education}"
        plt.plot(pred_df["age"], pred_df["predicted_prob"], label=label)

    # Plot settings
    plt.xlabel("Age")
    plt.ylabel("Predicted Probability of Other Informal Care")
    plt.title("Predicted Probability vs Age (by has_sister & education)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
