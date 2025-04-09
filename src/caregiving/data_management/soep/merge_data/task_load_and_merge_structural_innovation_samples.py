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

SYEAR_IS = 2016
MALE = 1
FEMALE = 2

NO = 2

SPOUSE = 1
MOTHER_OR_FATHER = 2
MOTHER_OR_FATHER_IN_LAW = 3

PGSBIL_FACHHOCHSCHULREIFE = 3
PGSBIL_ABITUR = 4


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


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
    # path_to_save: Annotated[Path, Product] = BLD
    # / "data"
    # / "soep_is_exog_care_data_raw.csv",
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

    # Initialize dummies with 0
    merged_data["other_informal_care"] = 0
    merged_data["formal_care"] = 0

    # Assign 1 if conditions are met
    merged_data.loc[
        (merged_data["ip05"].isin([1, 2]))  # lives in private household
        & ((merged_data["ip08a1"] == 1) | (merged_data["ip08a4"] == 1)),
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
        (merged_data["ip06"] == 1)
        & (merged_data["ip05"].isin([1, 2]))  # lives in private household
        # & (merged_data["ip07w"] >= 1)
        & (merged_data["other_informal_care"] == 0),
        # & (merged_data["formal_care"] == 0),
        "only_own_informal_care",
    ] = 1

    merged_data.loc[
        # ((merged_data["ip06"] == 2) | (merged_data["ip07w"] < 1))
        (merged_data["ip06"] == NO) & (merged_data["other_informal_care"] == 1),
        # & (merged_data["formal_care"] == 0),
        "only_other_informal_care",
    ] = 1

    # only_other_informal_care
    # 0    622
    # 1    320

    merged_data.loc[
        (merged_data["ip06"] == NO)  # No
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

    # Create subsamples for mother and father
    est_sample = merged_data[merged_data["female"] == 1]
    est_sample_mothers = est_sample[est_sample["ip03"] == FEMALE]

    # First, drop any rows with missing values in the relevant variables
    reg_data = est_sample_mothers[
        ["other_informal_care", "age", "has_sister", "education"]
    ].dropna()
    reg_data["age_squared"] = reg_data["age"] ** 2

    # Run logistic regression
    model1 = smf.logit(
        "other_informal_care ~ age + age_squared + has_sister + education",
        data=reg_data,
    ).fit()
    print(model1.summary())

    plot_logit_prediction_vs_age(model1)

    # # Step 1: Define bin edges and labels
    # bins = list(range(40, 70, 5)) + [70]  # [40, 45, 50, 55, 60, 65, 70]
    # labels = [f"age_bin_{b}_{b+4}" for b in bins[:-1]]  # e.g. age_bin_40_44, ...

    # # Step 2: Assign age bin categories
    # est_sample_mothers["age_bin"] = pd.cut(
    #     est_sample_mothers["age"],
    #     bins=bins,
    #     right=False,  # [40, 45) instead of (40, 45]
    #     labels=labels,
    # )

    # # Step 3: Create dummy variables (one-hot encoding, include all dummies)
    # age_bin_dummies = pd.get_dummies(
    #     est_sample_mothers["age_bin"], prefix="", prefix_sep=""
    # )

    # # Step 4: Append dummies to est_sample_mothers
    # est_sample_mothers = pd.concat([est_sample_mothers, age_bin_dummies], axis=1)

    # # ### Plotting ###
    # # # Count number of observations in each age bin
    # # age_bin_counts = est_sample_mothers["age_bin"].value_counts().sort_index()

    # # # Plot
    # # plt.figure(figsize=(8, 5))
    # # age_bin_counts.plot(kind="bar")

    # # plt.xlabel("Age Bin")
    # # plt.ylabel("Number of Observations")
    # # plt.title("Number of Observations per Age Bin")
    # # plt.grid(axis="y")
    # # plt.tight_layout()
    # # plt.show()

    # # Choose your reference age bin (drop it from regression)
    # ref_bin = "age_bin_40_44"
    # age_bin_vars = [
    #     col
    #     for col in est_sample_mothers.columns
    #     if col.startswith("age_bin_") and col != ref_bin
    # ]

    # # Build formula dynamically
    # formula = (
    #     f"other_informal_care ~ has_sister + education + {' + '.join(age_bin_vars)}"
    # )

    # # Drop rows with missing values in relevant columns
    # reg_data_bins = est_sample_mothers[
    #     ["other_informal_care", "has_sister", "education"] + age_bin_vars
    # ].dropna()

    # # Fit logistic regression
    # model2 = smf.logit(formula, data=reg_data_bins).fit()
    # print(model2.summary())

    # plot_logit_prediction_by_age_bin(model2, age_bin_vars, ref_bin="age_bin_40_44")

    # # Merge pgen data with pathl data and hl data
    # merged_data = pd.merge(
    #     pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    # )

    # merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    # del pgen_data, pathl_data
    # merged_data.set_index(["pid", "syear"], inplace=True)
    # print(str(len(merged_data)) + " observations in SOEP C38 core.")

    # merged_data.to_csv(path_to_save)


def plot_logit_prediction_vs_age(model, age_range=(40, 80), step=1):
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


def plot_logit_prediction_by_age_bin(model, age_bins, ref_bin="age_bin_40_44"):
    """Plot predicted probabilities across age bins by_sister and education.

    Parameters:
    - model: fitted statsmodels Logit model
    - age_bins: list of all age_bin_* dummy column names
    - ref_bin: the reference age bin omitted from the model
    """
    # All combinations of has_sister and education (0/1)
    combos = list(itertools.product([0, 1], [0, 1]))

    plt.figure(figsize=(10, 6))

    for has_sister, education in combos:
        pred_rows = []

        for age_bin in age_bins + [ref_bin]:  # include ref bin for plotting
            # Initialize all bins as 0
            row = {bin_col: 0 for bin_col in age_bins}
            if age_bin != ref_bin:
                row[age_bin] = 1  # set 1 for the active bin

            row.update({"has_sister": has_sister, "education": education})

            # Predict
            pred_df = pd.DataFrame([row])
            prob = model.predict(pred_df)[0]

            pred_rows.append((age_bin, prob))

        # Sort by bin order
        pred_rows = sorted(pred_rows, key=lambda x: x[0])  # noqa: FURB118
        labels, probs = zip(*pred_rows, strict=False)

        label = f"has_sister={has_sister}, education={education}"
        plt.plot(labels, probs, marker="o", label=label)

    plt.xlabel("Age Bin")
    plt.ylabel("Predicted Probability of Other Informal Care")
    plt.title("Predicted Probabilities by Age Bin (Logit Model)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
