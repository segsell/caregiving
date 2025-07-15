"""Create SOEP RV sample."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.utils import table

MONTHS_WORK_THRESHOLD = 6
MONTHS_CARE_THRESHOLD = 6


def task_create_soep_rv_sample(
    path_to_rv_data: Path = BLD / "data" / "rv_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_rv.csv",
) -> None:
    """Create SOEP RV sample."""

    # Load RV data
    rv_data = pd.read_csv(path_to_rv_data, index_col=[0])

    # Create a variable for the age of the individual
    rv_data["syear"] = rv_data["JAHR"].astype(int)
    rv_data["smonth"] = rv_data["MONAT"].astype(int)
    rv_data["age"] = rv_data["JAHR"] - rv_data["GBJAVS"]

    rv_data["sex"] = np.where(rv_data["GEVS"] == 2, 1, 0)  # noqa: PLR2004

    rv_data.sort_values(by=["rv_id", "syear", "smonth"], inplace=True)

    # # ------------------------------------------------------------------
    # # 1.  baseline: regular employment (STATUS 1) + insured minijob (STATUS 5)
    # # ------------------------------------------------------------------
    # rv_data["MEGPT_main"] = rv_data[["STATUS_1_EGPT", "STATUS_5_EGPT"]].sum(
    #     axis=1, skipna=True
    # )

    # # ------------------------------------------------------------------
    # # 2.  add a Nebenjob that is stored in STATUS 2 or 3
    # # ------------------------------------------------------------------
    # rv_data["MEGPT_all"] = rv_data["MEGPT_main"].copy()

    # mask_njb2 = (rv_data["STATUS_2"] == "NJB") & rv_data["STATUS_2_EGPT"].notna()
    # mask_njb3 = (rv_data["STATUS_3"] == "NJB") & rv_data["STATUS_3_EGPT"].notna()

    # rv_data.loc[mask_njb2, "MEGPT_all"] = (
    #     rv_data.loc[mask_njb2, "MEGPT_all"].fillna(0)
    #     + rv_data.loc[mask_njb2, "STATUS_2_EGPT"]
    # )
    # rv_data.loc[mask_njb3, "MEGPT_all"] = (
    #     rv_data.loc[mask_njb3, "MEGPT_all"].fillna(0)
    #     + rv_data.loc[mask_njb3, "STATUS_3_EGPT"]
    # )

    # # ------------------------------------------------------------------
    # # 3.  NEW: add caregiving EP (STATUS code "PFL" in SLOT 2 or 3)
    # # ------------------------------------------------------------------
    # mask_pfl2 = (rv_data["STATUS_2"] == "PFL") & rv_data["STATUS_2_EGPT"].notna()
    # mask_pfl3 = (rv_data["STATUS_3"] == "PFL") & rv_data["STATUS_3_EGPT"].notna()

    # rv_data.loc[mask_pfl2, "MEGPT_all"] = (
    #     rv_data.loc[mask_pfl2, "MEGPT_all"].fillna(0)
    #     + rv_data.loc[mask_pfl2, "STATUS_2_EGPT"]
    # )
    # rv_data.loc[mask_pfl3, "MEGPT_all"] = (
    #     rv_data.loc[mask_pfl3, "MEGPT_all"].fillna(0)
    #     + rv_data.loc[mask_pfl3, "STATUS_3_EGPT"]
    # )

    # ----------------------------------------------------------
    # 0.  convenience masks
    # ----------------------------------------------------------
    mask_njb2 = (rv_data["STATUS_2"] == "NJB") & rv_data["STATUS_2_EGPT"].notna()
    mask_njb3 = (rv_data["STATUS_3"] == "NJB") & rv_data["STATUS_3_EGPT"].notna()

    mask_pfl2 = (rv_data["STATUS_2"] == "PFL") & rv_data["STATUS_2_EGPT"].notna()
    mask_pfl3 = (rv_data["STATUS_3"] == "PFL") & rv_data["STATUS_3_EGPT"].notna()

    # ----------------------------------------------------------
    # 1.  main employment + insured minijob        →  EGP_main
    # ----------------------------------------------------------
    rv_data["EGP_main"] = rv_data[["STATUS_1_EGPT", "STATUS_5_EGPT"]].sum(
        axis=1, skipna=True
    )

    # ----------------------------------------------------------
    # 2.  main + side job(s) (NJB)                 →  EGP_main_and_side
    # ----------------------------------------------------------
    rv_data["EGP_main_and_side"] = rv_data["EGP_main"].copy()

    rv_data.loc[mask_njb2, "EGP_main_and_side"] += rv_data.loc[
        mask_njb2, "STATUS_2_EGPT"
    ].fillna(0)
    rv_data.loc[mask_njb3, "EGP_main_and_side"] += rv_data.loc[
        mask_njb3, "STATUS_3_EGPT"
    ].fillna(0)

    # ----------------------------------------------------------
    # 3.  main + caregiving (PFL)                  →  EGP_main_and_care
    # ----------------------------------------------------------
    rv_data["EGP_main_and_care"] = rv_data["EGP_main"].copy()

    rv_data.loc[mask_pfl2, "EGP_main_and_care"] += rv_data.loc[
        mask_pfl2, "STATUS_2_EGPT"
    ].fillna(0)
    rv_data.loc[mask_pfl3, "EGP_main_and_care"] += rv_data.loc[
        mask_pfl3, "STATUS_3_EGPT"
    ].fillna(0)

    # ----------------------------------------------------------
    # 4.  main + side job(s) + caregiving          →  EGP_all
    # ----------------------------------------------------------
    rv_data["EGP_all"] = rv_data["EGP_main"].copy()

    # add side-job points
    rv_data.loc[mask_njb2, "EGP_all"] += rv_data.loc[mask_njb2, "STATUS_2_EGPT"].fillna(
        0
    )
    rv_data.loc[mask_njb3, "EGP_all"] += rv_data.loc[mask_njb3, "STATUS_3_EGPT"].fillna(
        0
    )

    # add care points
    rv_data.loc[mask_pfl2, "EGP_all"] += rv_data.loc[mask_pfl2, "STATUS_2_EGPT"].fillna(
        0
    )
    rv_data.loc[mask_pfl3, "EGP_all"] += rv_data.loc[mask_pfl3, "STATUS_3_EGPT"].fillna(
        0
    )

    # ----------------------------------------------------------
    # 5.  yearly aggregates
    # ----------------------------------------------------------
    for col in ("EGP_main", "EGP_main_and_side", "EGP_main_and_care", "EGP_all"):
        rv_data[f"{col}_yearly"] = rv_data.groupby(["rv_id", "syear"])[col].transform(
            "sum"
        )

    # rv_data["MEGPT_all_positive"] = np.where(
    #     rv_data["MEGPT_all"] > 0, rv_data["MEGPT_all"], np.nan
    # )

    # Create new variable that contains aggregated yearly pension points
    # using the monthly pension points variable MEGPT_all
    # rv_data["MEGPT_main_yearly"] = rv_data.groupby(["rv_id", "syear"])[
    #     "MEGPT_main"
    # ].transform("sum")
    # rv_data["MEGPT_all_yearly"] = rv_data.groupby(["rv_id", "syear"])[
    #     "MEGPT_all"
    # ].transform("sum")

    # Create a variable that indicates whether the individual has worked
    # at least 6 months in the year
    rv_data["working"] = rv_data.groupby(["rv_id", "syear"])["EGP_main"].transform(
        lambda x: (x > 0).sum() >= MONTHS_WORK_THRESHOLD
    )

    rv_data["EGP_main_yearly"].describe()

    rv_data["rv_care"] = np.where(
        (rv_data["STATUS_2"] == "PFL") | (rv_data["STATUS_3"] == "PFL"), 1, 0
    )
    rv_data["rv_care_yearly"] = (
        rv_data.groupby(["rv_id", "syear"])["rv_care"]
        .transform(lambda x: (x == 1).sum() >= MONTHS_CARE_THRESHOLD)
        .astype(int)
    )

    # Step 1: Keep only December observations
    rv_year = rv_data[rv_data["smonth"] == 12]  # noqa: PLR2004

    assert (rv_year.groupby(["rv_id", "syear"]).size() == 1).all()

    # # B) “last-observation” per (rv_id, JAHR)
    # idx_last = rv_year.groupby(["rv_id", "syear"])["smonth"].idxmax()
    # df_last = rv_year.loc[idx_last].copy()

    # # Step 1: Calculate months of care per person per year
    months_care_per_year = (
        rv_data.groupby(["rv_id", "syear"])["rv_care"]
        .sum()
        .reset_index(name="n_months_with_care")
    )

    # # Step 2: Average across all years for each person
    _avg_care_per_person = (
        months_care_per_year.groupby("rv_id")["n_months_with_care"]
        .mean()
        .reset_index(name="avg_yearly_care")
    )

    # # Filter to only years with at least 1 month of care
    care_years = months_care_per_year[months_care_per_year["n_months_with_care"] >= 1]

    # Compute conditional average per person
    cond_avg_care_per_person = (
        care_years.groupby("rv_id")["n_months_with_care"]
        .mean()
        .reset_index(name="avg_yearly_care_conditional")
    )
    print(cond_avg_care_per_person)

    # Group by person and year, check if rv_care==1 at least once in that year
    caregiver_status = (
        rv_data.groupby(["rv_id", "syear"])["rv_care"]
        .max()
        .reset_index(name="is_caregiver")
    )

    # Number of caregivers per year
    _caregivers_per_year = caregiver_status.groupby("syear")["is_caregiver"].sum()

    # Number of non-caregivers per year
    _non_caregivers_per_year = caregiver_status.groupby("syear")["is_caregiver"].apply(
        lambda x: (x == 0).sum()
    )

    # Save the filtered data
    rv_year.to_csv(path_to_save)
