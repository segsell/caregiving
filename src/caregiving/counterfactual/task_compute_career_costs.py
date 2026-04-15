"""Compute career costs by comparing NPV of incomes between scenarios.

Baseline and no-care demand counterfactual.

Note: own_income_after_ssc in the model includes labor income when working and
retirement/pension income when retired (see budget_equation.py).
"""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    BETA_NPV,
    DEAD,
    INFORMAL_CARE,
    NPV_END_AGE,
    NPV_START_AGE,
)

NPV_BASELINE_TOL = 1e-10

# Columns needed for NPV computation and filtering. Restrict loaded data to these to save RAM.

REQUIRED_COLUMNS_NPV = [
    "agent",
    "age",
    "health",
    "choice",
    "care_demand",
    "own_income_after_ssc",
    "household_unemployment_benefits",
    "child_benefits",
    "partner_state",
    "care_benefits_and_costs",
    "caregiving_leave_top_up",
    "bequest_from_parent",
]

# NPV specification: (only_labor_pension, only_own_income, include_bequest_from_parent)
# only_labor_pension=True => labor + pension only (excludes caregiving_leave_top_up from own_income)
# only_own_income=True => model's own_income_after_ssc (in leave models this includes top-up)
# only_own_income=False => full income (own + unemployment + child + care benefits + top-up)
NPV_SPECS = [
    (True, True, True),  # labor and pension only, incl. bequest
    (True, True, False),  # labor and pension only, no bequest
    (
        False,
        True,
        True,
    ),  # own income only (may include top-up in leave models) + bequest
    (False, True, False),  # own income only, no bequest
    (False, False, True),  # full income + bequest
]
NPV_START_AGES = [30, 40]  # 40 is shared.NPV_START_AGE; 30 for extended analysis


def _restrict_to_npv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns needed for NPV computation; save RAM."""
    keep = [c for c in REQUIRED_COLUMNS_NPV if c in df.columns]
    return df[keep].copy()


def _npv_spec_label(
    only_labor_pension: bool, only_own: bool, bequest: bool, start_age: int
) -> str:
    """Human-readable label for NPV spec and npv_start_age."""
    if only_labor_pension:
        inc = "labor and pension only"
    else:
        inc = "own income only" if only_own else "full income"
    beq = "incl. bequest" if bequest else "no bequest"
    return f"{inc}, {beq}, start age {start_age}"


@pytask.mark.career_costs
@pytask.mark.career_costs_baseline
def task_compute_career_costs(
    # Baseline
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    # path_to_original_data: Path = BLD
    # / "solve_and_simulate"
    # / "simulated_data_estimated_params_career_costs.pkl",
    # path_to_no_care_demand_data: Path = BLD
    # / "solve_and_simulate"
    # / "simulated_data_no_care_demand_career_costs.pkl",
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_no_care_demand.csv",
    path_to_baseline_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_baseline.csv",
    path_to_npv_care_ratios: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios.csv",
    path_to_npv_summary_tex: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary.tex",
    restrict_to_ever_caregivers: bool = True,
    restrict_to_ever_care_demand: bool = False,
) -> None:
    """Compute career costs as NPV difference between baseline and counterfactual.

    Writes one comprehensive LaTeX table (npv_summary.tex) with all NPV
    specifications (only_own ± bequest, full income + bequest) and start ages (30, 40).
    """
    _specs = pkl.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    if "agent" not in df_original.columns:
        if (
            isinstance(df_original.index, pd.MultiIndex)
            and "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()
    if "agent" not in df_no_care_demand.columns:
        if (
            isinstance(df_no_care_demand.index, pd.MultiIndex)
            and "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    if restrict_to_ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()
    elif restrict_to_ever_care_demand:
        care_demand_ids = df_original.loc[
            df_original["care_demand"] > 0, "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(care_demand_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(care_demand_ids)
        ].copy()
    else:
        df_counterfactual = df_no_care_demand.copy()

    df_original = _restrict_to_npv_columns(df_original)
    df_counterfactual = _restrict_to_npv_columns(df_counterfactual)

    summary = compute_npv_summary_table(df_original, df_counterfactual)
    path_to_npv_summary_tex.parent.mkdir(parents=True, exist_ok=True)
    path_to_npv_summary_tex.write_text(
        _summary_table_to_latex(summary), encoding="utf-8"
    )
    print(f"NPV summary (all specs) saved to {path_to_npv_summary_tex}")

    # Default spec (only_own=True, bequest=False, start_age=40): keep CSVs for compatibility
    original_npv = compute_career_npv(
        df_original,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    no_care_demand_npv = compute_career_npv(
        df_counterfactual,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    original_npv.to_csv(path_to_baseline_npv, index=False)
    no_care_demand_npv.to_csv(path_to_no_care_demand_npv, index=False)
    merged = pd.merge(
        original_npv,
        no_care_demand_npv,
        on="agent",
        suffixes=("_baseline", "_no_care_demand"),
    )
    merged = merged[merged["career_npv_baseline"] > NPV_BASELINE_TOL].copy()
    npv_care = 1 - (merged["career_npv_no_care_demand"] / merged["career_npv_baseline"])
    pd.DataFrame({"agent": merged["agent"], "npv_care_ratio": npv_care}).to_csv(
        path_to_npv_care_ratios, index=False
    )


@pytask.mark.career_costs
@pytask.mark.career_costs_caregiving_leave_with_job_retention
def task_compute_career_costs_caregiving_leave_with_job_retention(
    # Baseline
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_caregiving_leave_with_job_retention.csv",
    path_to_baseline_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_caregiving_leave_with_job_retention.csv",
    path_to_npv_care_ratios: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios_caregiving_leave_with_job_retention.csv",
    path_to_npv_summary_tex: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary_caregiving_leave_with_job_retention.tex",
    restrict_to_ever_caregivers: bool = True,
    restrict_to_ever_care_demand: bool = False,
) -> None:
    """Career costs for caregiving leave with job retention; one LaTeX summary table."""

    _specs = pkl.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_caregiving_leave_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    if "agent" not in df_original.columns:
        if (
            isinstance(df_original.index, pd.MultiIndex)
            and "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()
    if "agent" not in df_no_care_demand.columns:
        if (
            isinstance(df_no_care_demand.index, pd.MultiIndex)
            and "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    if restrict_to_ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()
    elif restrict_to_ever_care_demand:
        care_demand_ids = df_original.loc[
            df_original["care_demand"] > 0, "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(care_demand_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(care_demand_ids)
        ].copy()
    else:
        df_counterfactual = df_no_care_demand.copy()

    df_original = _restrict_to_npv_columns(df_original)
    df_counterfactual = _restrict_to_npv_columns(df_counterfactual)

    summary = compute_npv_summary_table(df_original, df_counterfactual)
    path_to_npv_summary_tex.parent.mkdir(parents=True, exist_ok=True)
    path_to_npv_summary_tex.write_text(
        _summary_table_to_latex(summary), encoding="utf-8"
    )
    print(f"NPV summary (caregiving leave) saved to {path_to_npv_summary_tex}")

    original_npv = compute_career_npv(
        df_original,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    no_care_demand_npv = compute_career_npv(
        df_counterfactual,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    original_npv.to_csv(path_to_baseline_npv, index=False)
    no_care_demand_npv.to_csv(path_to_no_care_demand_npv, index=False)
    merged = pd.merge(
        original_npv,
        no_care_demand_npv,
        on="agent",
        suffixes=("_baseline", "_no_care_demand"),
    )
    merged = merged[merged["career_npv_baseline"] > NPV_BASELINE_TOL].copy()
    npv_care = 1 - (merged["career_npv_no_care_demand"] / merged["career_npv_baseline"])
    pd.DataFrame({"agent": merged["agent"], "npv_care_ratio": npv_care}).to_csv(
        path_to_npv_care_ratios, index=False
    )


@pytask.mark.career_costs
@pytask.mark.career_costs_full_caregiving_leave_with_job_retention
def task_compute_career_costs_full_caregiving_leave_with_job_retention(
    # Baseline
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_full_caregiving_leave_with_job_retention.csv",
    path_to_baseline_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_full_caregiving_leave_with_job_retention.csv",
    path_to_npv_care_ratios: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios_full_caregiving_leave_with_job_retention.csv",
    path_to_npv_summary_tex: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary_full_caregiving_leave_with_job_retention.tex",
    restrict_to_ever_caregivers: bool = True,
    restrict_to_ever_care_demand: bool = False,
) -> None:
    """Career costs for full caregiving leave with job retention; one LaTeX summary."""

    _specs = pkl.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_caregiving_leave_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    if "agent" not in df_original.columns:
        if (
            isinstance(df_original.index, pd.MultiIndex)
            and "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()
    if "agent" not in df_no_care_demand.columns:
        if (
            isinstance(df_no_care_demand.index, pd.MultiIndex)
            and "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    if restrict_to_ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()
    elif restrict_to_ever_care_demand:
        care_demand_ids = df_original.loc[
            df_original["care_demand"] > 0, "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(care_demand_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(care_demand_ids)
        ].copy()
    else:
        df_counterfactual = df_no_care_demand.copy()

    df_original = _restrict_to_npv_columns(df_original)
    df_counterfactual = _restrict_to_npv_columns(df_counterfactual)

    summary = compute_npv_summary_table(df_original, df_counterfactual)
    path_to_npv_summary_tex.parent.mkdir(parents=True, exist_ok=True)
    path_to_npv_summary_tex.write_text(
        _summary_table_to_latex(summary), encoding="utf-8"
    )
    print(f"NPV summary (full caregiving leave) saved to {path_to_npv_summary_tex}")

    original_npv = compute_career_npv(
        df_original,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    no_care_demand_npv = compute_career_npv(
        df_counterfactual,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    original_npv.to_csv(path_to_baseline_npv, index=False)
    no_care_demand_npv.to_csv(path_to_no_care_demand_npv, index=False)
    merged = pd.merge(
        original_npv,
        no_care_demand_npv,
        on="agent",
        suffixes=("_baseline", "_no_care_demand"),
    )
    merged = merged[merged["career_npv_baseline"] > NPV_BASELINE_TOL].copy()
    npv_care = 1 - (merged["career_npv_no_care_demand"] / merged["career_npv_baseline"])
    pd.DataFrame({"agent": merged["agent"], "npv_care_ratio": npv_care}).to_csv(
        path_to_npv_care_ratios, index=False
    )


@pytask.mark.career_costs
@pytask.mark.career_costs_full_leave_no_pflegegeld
def task_compute_career_costs_full_leave_no_pflegegeld(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_policy_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params_no_pflegegeld.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_full_leave_no_pflegegeld_no_care_demand.csv",
    path_to_baseline_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_full_leave_no_pflegegeld.csv",
    path_to_npv_care_ratios: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios_full_leave_no_pflegegeld.csv",
    path_to_npv_summary_tex: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary_full_leave_no_pflegegeld.tex",
    restrict_to_ever_caregivers: bool = True,
    restrict_to_ever_care_demand: bool = False,
) -> None:
    """Career costs for full caregiving leave without Pflegegeld; one LaTeX summary."""
    _specs = pkl.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_policy_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    if "agent" not in df_original.columns:
        if (
            isinstance(df_original.index, pd.MultiIndex)
            and "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()
    if "agent" not in df_no_care_demand.columns:
        if (
            isinstance(df_no_care_demand.index, pd.MultiIndex)
            and "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    if restrict_to_ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()
    elif restrict_to_ever_care_demand:
        care_demand_ids = df_original.loc[
            df_original["care_demand"] > 0, "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(care_demand_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(care_demand_ids)
        ].copy()
    else:
        df_counterfactual = df_no_care_demand.copy()

    df_original = _restrict_to_npv_columns(df_original)
    df_counterfactual = _restrict_to_npv_columns(df_counterfactual)

    summary = compute_npv_summary_table(df_original, df_counterfactual)
    path_to_npv_summary_tex.parent.mkdir(parents=True, exist_ok=True)
    path_to_npv_summary_tex.write_text(
        _summary_table_to_latex(summary), encoding="utf-8"
    )
    print(f"NPV summary (full leave no Pflegegeld) saved to {path_to_npv_summary_tex}")

    original_npv = compute_career_npv(
        df_original,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    no_care_demand_npv = compute_career_npv(
        df_counterfactual,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    original_npv.to_csv(path_to_baseline_npv, index=False)
    no_care_demand_npv.to_csv(path_to_no_care_demand_npv, index=False)
    merged = pd.merge(
        original_npv,
        no_care_demand_npv,
        on="agent",
        suffixes=("_baseline", "_no_care_demand"),
    )
    merged = merged[merged["career_npv_baseline"] > NPV_BASELINE_TOL].copy()
    npv_care = 1 - (merged["career_npv_no_care_demand"] / merged["career_npv_baseline"])
    pd.DataFrame({"agent": merged["agent"], "npv_care_ratio": npv_care}).to_csv(
        path_to_npv_care_ratios, index=False
    )


@pytask.mark.career_costs
@pytask.mark.career_costs_full_beirat
def task_compute_career_costs_full_beirat(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_policy_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_full_beirat_no_care_demand.csv",
    path_to_baseline_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_full_beirat.csv",
    path_to_npv_care_ratios: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios_full_beirat.csv",
    path_to_npv_summary_tex: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary_full_beirat.tex",
    restrict_to_ever_caregivers: bool = True,
    restrict_to_ever_care_demand: bool = False,
) -> None:
    """Career costs for full Beirat caregiving leave; one LaTeX summary."""
    _specs = pkl.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_policy_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    if "agent" not in df_original.columns:
        if (
            isinstance(df_original.index, pd.MultiIndex)
            and "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()
    if "agent" not in df_no_care_demand.columns:
        if (
            isinstance(df_no_care_demand.index, pd.MultiIndex)
            and "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    if restrict_to_ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()
    elif restrict_to_ever_care_demand:
        care_demand_ids = df_original.loc[
            df_original["care_demand"] > 0, "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(care_demand_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(care_demand_ids)
        ].copy()
    else:
        df_counterfactual = df_no_care_demand.copy()

    df_original = _restrict_to_npv_columns(df_original)
    df_counterfactual = _restrict_to_npv_columns(df_counterfactual)

    summary = compute_npv_summary_table(df_original, df_counterfactual)
    path_to_npv_summary_tex.parent.mkdir(parents=True, exist_ok=True)
    path_to_npv_summary_tex.write_text(
        _summary_table_to_latex(summary), encoding="utf-8"
    )
    print(f"NPV summary (full Beirat) saved to {path_to_npv_summary_tex}")

    original_npv = compute_career_npv(
        df_original,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    no_care_demand_npv = compute_career_npv(
        df_counterfactual,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    original_npv.to_csv(path_to_baseline_npv, index=False)
    no_care_demand_npv.to_csv(path_to_no_care_demand_npv, index=False)
    merged = pd.merge(
        original_npv,
        no_care_demand_npv,
        on="agent",
        suffixes=("_baseline", "_no_care_demand"),
    )
    merged = merged[merged["career_npv_baseline"] > NPV_BASELINE_TOL].copy()
    npv_care = 1 - (merged["career_npv_no_care_demand"] / merged["career_npv_baseline"])
    pd.DataFrame({"agent": merged["agent"], "npv_care_ratio": npv_care}).to_csv(
        path_to_npv_care_ratios, index=False
    )


@pytask.mark.career_costs
@pytask.mark.career_costs_beirat
def task_compute_career_costs_beirat(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_policy_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_beirat_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_beirat_no_care_demand.csv",
    path_to_baseline_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_beirat.csv",
    path_to_npv_care_ratios: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios_beirat.csv",
    path_to_npv_summary_tex: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary_beirat.tex",
    restrict_to_ever_caregivers: bool = True,
    restrict_to_ever_care_demand: bool = False,
) -> None:
    """Career costs for Beirat partial caregiving leave; one LaTeX summary."""
    _specs = pkl.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_policy_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    if "agent" not in df_original.columns:
        if (
            isinstance(df_original.index, pd.MultiIndex)
            and "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(level=["agent"])
        else:
            df_original = df_original.reset_index()
    if "agent" not in df_no_care_demand.columns:
        if (
            isinstance(df_no_care_demand.index, pd.MultiIndex)
            and "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(level=["agent"])
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    if restrict_to_ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()
    elif restrict_to_ever_care_demand:
        care_demand_ids = df_original.loc[
            df_original["care_demand"] > 0, "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(care_demand_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(care_demand_ids)
        ].copy()
    else:
        df_counterfactual = df_no_care_demand.copy()

    df_original = _restrict_to_npv_columns(df_original)
    df_counterfactual = _restrict_to_npv_columns(df_counterfactual)

    summary = compute_npv_summary_table(df_original, df_counterfactual)
    path_to_npv_summary_tex.parent.mkdir(parents=True, exist_ok=True)
    path_to_npv_summary_tex.write_text(
        _summary_table_to_latex(summary), encoding="utf-8"
    )
    print(f"NPV summary (Beirat partial) saved to {path_to_npv_summary_tex}")

    original_npv = compute_career_npv(
        df_original,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    no_care_demand_npv = compute_career_npv(
        df_counterfactual,
        BETA_NPV,
        include_bequest_from_parent=False,
        only_own_income=True,
        npv_start_age=NPV_START_AGE,
    )
    original_npv.to_csv(path_to_baseline_npv, index=False)
    no_care_demand_npv.to_csv(path_to_no_care_demand_npv, index=False)
    merged = pd.merge(
        original_npv,
        no_care_demand_npv,
        on="agent",
        suffixes=("_baseline", "_no_care_demand"),
    )
    merged = merged[merged["career_npv_baseline"] > NPV_BASELINE_TOL].copy()
    npv_care = 1 - (merged["career_npv_no_care_demand"] / merged["career_npv_baseline"])
    pd.DataFrame({"agent": merged["agent"], "npv_care_ratio": npv_care}).to_csv(
        path_to_npv_care_ratios, index=False
    )


def compute_career_npv(
    df: pd.DataFrame,
    beta: float,
    include_bequest_from_parent: bool = False,
    only_own_income: bool = True,
    only_labor_pension: bool = False,
    npv_start_age: int | None = None,
    npv_end_age: int | None = None,
) -> pd.DataFrame:
    """Compute net present value of total income.

    - only_labor_pension=True: labor + pension only (own_income_after_ssc minus
      caregiving_leave_top_up when present). Excludes leave top-up so career costs
      reflect direct labor-supply (and pension) effect only.
    - only_own_income=True and only_labor_pension=False: model's own_income_after_ssc
      (in leave models this already includes caregiving_leave_top_up).
    - only_own_income=False: full income = own_income_after_ssc + unemployment +
      child_benefits + care_benefits (top-up not added again; it is already in
      own_income_after_ssc in leave models).
    """
    start = npv_start_age if npv_start_age is not None else NPV_START_AGE
    end = npv_end_age if npv_end_age is not None else NPV_END_AGE

    df_filtered = df[(df["age"] >= start) & (df["age"] <= end)].copy()

    # Labor + pension only: exclude caregiving_leave_top_up (model may bake it into own_income_after_ssc)
    top_up = (
        df_filtered["caregiving_leave_top_up"].fillna(0)
        if "caregiving_leave_top_up" in df_filtered.columns
        else 0.0
    )

    # Divide household_unemployment_benefits by 2 if has partner (partner_state > 0)
    household_unemployment = df_filtered["household_unemployment_benefits"].copy()
    if "partner_state" in df_filtered.columns:
        household_unemployment = np.where(
            df_filtered["partner_state"] > 0,
            household_unemployment / 2,
            household_unemployment,
        )

    if only_labor_pension:
        # Labor + pension only (no leave top-up)
        base_sum = df_filtered["own_income_after_ssc"].copy() - np.asarray(top_up)
        if include_bequest_from_parent and "bequest_from_parent" in df_filtered.columns:
            base_sum = base_sum + df_filtered["bequest_from_parent"].fillna(0)
    elif only_own_income:
        base_sum = df_filtered["own_income_after_ssc"].copy()
        if include_bequest_from_parent and "bequest_from_parent" in df_filtered.columns:
            base_sum = base_sum + df_filtered["bequest_from_parent"].fillna(0)
    else:
        # Full income: own_income_after_ssc (already includes top-up in leave models) + transfers
        base_sum = (
            df_filtered["own_income_after_ssc"]
            + household_unemployment
            + df_filtered["child_benefits"]
        )
        if "care_benefits_and_costs" in df_filtered.columns:
            base_sum = base_sum + df_filtered["care_benefits_and_costs"].fillna(0)
        # Do not add caregiving_leave_top_up here; it is already in own_income_after_ssc in leave runs
        if include_bequest_from_parent and "bequest_from_parent" in df_filtered.columns:
            base_sum = base_sum + df_filtered["bequest_from_parent"].fillna(0)

    df_filtered["total_gross_income"] = base_sum

    df_filtered["total_npv_income"] = np.where(
        df_filtered["health"] != DEAD,
        np.maximum(df_filtered["total_gross_income"], 0),
        0,
    )

    # Discount factor: beta^(age - npv_start_age)
    df_filtered["discount_factor"] = beta ** (df_filtered["age"] - start)
    df_filtered["discounted_income"] = (
        df_filtered["total_npv_income"] * df_filtered["discount_factor"]
    )

    npv_by_agent = df_filtered.groupby("agent")["discounted_income"].sum().reset_index()
    npv_by_agent.columns = ["agent", "career_npv"]
    return npv_by_agent


def compute_npv_summary_table(
    df_baseline: pd.DataFrame,
    df_no_care_demand: pd.DataFrame,
) -> pd.DataFrame:
    """Compute npv_care_mean and npv_mean for all NPV_SPECS × NPV_START_AGES.

    Returns a DataFrame with columns: specification, npv_start_age, npv_care_mean,
    npv_mean (and optionally only_own_income, include_bequest).
    """
    rows = []
    for only_labor_pension, only_own, bequest in NPV_SPECS:
        for start_age in NPV_START_AGES:
            orig_npv = compute_career_npv(
                df_baseline,
                BETA_NPV,
                include_bequest_from_parent=bequest,
                only_own_income=only_own,
                only_labor_pension=only_labor_pension,
                npv_start_age=start_age,
            )
            no_care_npv = compute_career_npv(
                df_no_care_demand,
                BETA_NPV,
                include_bequest_from_parent=bequest,
                only_own_income=only_own,
                only_labor_pension=only_labor_pension,
                npv_start_age=start_age,
            )
            merged = pd.merge(
                orig_npv,
                no_care_npv,
                on="agent",
                suffixes=("_baseline", "_no_care_demand"),
            )
            merged = merged[merged["career_npv_baseline"] > NPV_BASELINE_TOL].copy()
            npv_care = 1 - (
                merged["career_npv_no_care_demand"] / merged["career_npv_baseline"]
            )
            npv_care_mean = float(npv_care.mean())
            npv_mean = 1 - (
                merged["career_npv_no_care_demand"].mean()
                / merged["career_npv_baseline"].mean()
            )
            rows.append(
                {
                    "specification": _npv_spec_label(
                        only_labor_pension, only_own, bequest, start_age
                    ),
                    "npv_start_age": start_age,
                    "only_labor_pension": only_labor_pension,
                    "only_own_income": only_own,
                    "include_bequest": bequest,
                    "npv_care_mean": npv_care_mean,
                    "npv_mean": npv_mean,
                }
            )
    return pd.DataFrame(rows)


def _summary_table_to_latex(summary: pd.DataFrame) -> str:
    """Format NPV summary DataFrame as a LaTeX table."""
    cols = ["specification", "npv_start_age", "npv_care_mean", "npv_mean"]
    sub = summary[cols].copy()
    sub["npv_care_mean"] = sub["npv_care_mean"].map("{:.4f}".format)
    sub["npv_mean"] = sub["npv_mean"].map("{:.4f}".format)
    sub = sub.rename(
        columns={
            "specification": "Specification",
            "npv_start_age": "NPV start age",
            "npv_care_mean": "NPV care mean",
            "npv_mean": "NPV mean (ratio of means)",
        }
    )
    latex = sub.to_latex(
        index=False,
        escape=False,
        column_format="llrr",
        caption="Career costs NPV summary: mean of 1 - (NPV no care / NPV baseline) and ratio-of-means. Negative = career cost of caregiving. 'Labor and pension only' excludes leave top-up; 'own income only' uses model own_income_after_ssc (includes top-up in leave models).",
    )
    return latex


def compute_career_costs(
    original_npv: pd.DataFrame, no_care_demand_npv: pd.DataFrame
) -> pd.DataFrame:
    """Compute career costs as difference in NPV between scenarios."""

    # Merge the two NPV datasets
    merged = pd.merge(
        original_npv,
        no_care_demand_npv,
        on=["agent"],
        suffixes=("_original", "_no_care_demand"),
    )

    merged["career_costs"] = (
        merged["career_npv_no_care_demand"] - merged["career_npv_original"]
    )

    # Select relevant columns
    result = merged[
        [
            "agent",
            "career_npv_original",
            "career_npv_no_care_demand",
            "career_costs",
        ]
    ].copy()

    return result


def create_care_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create care ever and sum care variables."""
    # Caregiving
    df["informal_care"] = df["choice"].isin(np.asarray(INFORMAL_CARE).ravel().tolist())

    # Care ever - use agent column for grouping
    df["care_ever"] = df.groupby("agent")["informal_care"].transform(
        lambda x: x.cumsum().clip(upper=1)
    )

    # # Sum care
    # df["sum_informal_care"] = df.groupby("agent")["informal_care"].transform(
    #     lambda x: x.cumsum()
    # )

    # # Care demand ever
    # df["care_demand_ever"] = df.groupby("agent")["care_demand"].transform(
    #     lambda x: x.cumsum().clip(upper=1)
    # )

    return df


@pytask.mark.government_budget
def task_compute_government_budget_caregiving_leave(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_output: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "government_budget_caregiving_leave.csv",
) -> None:
    """Compute sum of government budget variables for caregiving leave scenario.

    Sums net_government_budget, caregiving_leave_top_up, and income_tax_single
    over all agents and periods up to and including max_ret_age.
    Each period-specific outcome is divided by the number of agents in that period
    before summing to account for deaths.
    """
    specs = pkl.load(path_to_specs.open("rb"))
    max_ret_age = specs["max_ret_age"]

    df = pd.read_pickle(path_to_caregiving_leave_data)
    _df_baseline = pd.read_pickle(path_to_baseline_data)  # Unused (commented code)
    # net = (
    #     (df["income_tax_single"].sum() - df_baseline["income_tax_single"].sum())
    #     + (
    #         df["caregiving_leave_top_up"].sum()
    #         - df_baseline["caregiving_leave_top_up"].sum()
    #     )
    #     + (
    #         df["net_government_budget"].sum()
    #         - df_baseline["net_government_budget"].sum()
    #     )
    # )
    # breakpoint()

    # Ensure agent and period columns exist
    if "agent" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("agent" in df.index.names):
            df = df.reset_index(level=["agent"])
        else:
            df = df.reset_index()

    # Ensure age column exists
    if "age" not in df.columns:
        if "period" in df.columns:
            df["age"] = df["period"] + specs["start_age"]
        else:
            raise ValueError("Neither 'age' nor 'period' column found in data")

    # Filter to periods up to and including max_ret_age
    df_filtered = df[df["age"] <= max_ret_age].copy()

    # Ensure required columns exist, set to zero if missing
    if "net_government_budget" not in df_filtered.columns:
        df_filtered["net_government_budget"] = 0
    if "caregiving_leave_top_up" not in df_filtered.columns:
        df_filtered["caregiving_leave_top_up"] = 0
    if "income_tax_single" not in df_filtered.columns:
        df_filtered["income_tax_single"] = 0

    # Fill NaN values with 0
    df_filtered["net_government_budget"] = df_filtered["net_government_budget"].fillna(
        0
    )
    df_filtered["caregiving_leave_top_up"] = df_filtered[
        "caregiving_leave_top_up"
    ].fillna(0)
    df_filtered["income_tax_single"] = df_filtered["income_tax_single"].fillna(0)

    # Group by period and compute per-agent averages (dividing by number of agents)
    period_stats = df_filtered.groupby("period").agg(
        {
            "net_government_budget": ["sum", "count"],
            "caregiving_leave_top_up": "sum",
            "income_tax_single": "sum",
        }
    )

    # Compute per-agent values for each period (divide by number of agents)
    period_stats["net_government_budget_per_agent"] = (
        period_stats[("net_government_budget", "sum")]
        / period_stats[("net_government_budget", "count")]
    )
    period_stats["caregiving_leave_top_up_per_agent"] = (
        period_stats[("caregiving_leave_top_up", "sum")]
        / period_stats[("net_government_budget", "count")]
    )
    period_stats["income_tax_single_per_agent"] = (
        period_stats[("income_tax_single", "sum")]
        / period_stats[("net_government_budget", "count")]
    )

    # Sum across all periods
    total_net_government_budget = period_stats["net_government_budget_per_agent"].sum()
    total_caregiving_leave_top_up = period_stats[
        "caregiving_leave_top_up_per_agent"
    ].sum()
    total_income_tax_single = period_stats["income_tax_single_per_agent"].sum()

    # Save to CSV
    result = pd.DataFrame(
        {
            "variable": [
                "net_government_budget",
                "caregiving_leave_top_up",
                "income_tax_single",
            ],
            "total_sum": [
                total_net_government_budget,
                total_caregiving_leave_top_up,
                total_income_tax_single,
            ],
        }
    )
    result.to_csv(path_to_output, index=False)

    print(f"Government budget sums saved to {path_to_output}")
    print(f"Net government budget: {total_net_government_budget:.2f}")
    print(f"Caregiving leave top-up: {total_caregiving_leave_top_up:.2f}")
    print(f"Income tax single: {total_income_tax_single:.2f}")


@pytask.mark.government_budget
def task_compute_government_budget_full_caregiving_leave(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_full_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_output: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "government_budget_full_caregiving_leave.csv",
) -> None:
    """Compute sum of government budget variables for full caregiving leave scenario.

    Sums net_government_budget, caregiving_leave_top_up, and income_tax_single
    over all agents and periods up to and including max_ret_age.
    Each period-specific outcome is divided by the number of agents in that period
    before summing to account for deaths.
    """
    specs = pkl.load(path_to_specs.open("rb"))
    max_ret_age = specs["max_ret_age"]

    df = pd.read_pickle(path_to_full_caregiving_leave_data)

    # Ensure agent and period columns exist
    if "agent" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("agent" in df.index.names):
            df = df.reset_index(level=["agent"])
        else:
            df = df.reset_index()

    # Ensure age column exists
    if "age" not in df.columns:
        if "period" in df.columns:
            df["age"] = df["period"] + specs["start_age"]
        else:
            raise ValueError("Neither 'age' nor 'period' column found in data")

    # Filter to periods up to and including max_ret_age
    df_filtered = df[df["age"] <= max_ret_age].copy()

    # Ensure required columns exist, set to zero if missing
    if "net_government_budget" not in df_filtered.columns:
        df_filtered["net_government_budget"] = 0
    if "caregiving_leave_top_up" not in df_filtered.columns:
        df_filtered["caregiving_leave_top_up"] = 0
    if "income_tax_single" not in df_filtered.columns:
        df_filtered["income_tax_single"] = 0

    # Fill NaN values with 0
    df_filtered["net_government_budget"] = df_filtered["net_government_budget"].fillna(
        0
    )
    df_filtered["caregiving_leave_top_up"] = df_filtered[
        "caregiving_leave_top_up"
    ].fillna(0)
    df_filtered["income_tax_single"] = df_filtered["income_tax_single"].fillna(0)

    # Group by period and compute per-agent averages (dividing by number of agents)
    period_stats = df_filtered.groupby("period").agg(
        {
            "net_government_budget": ["sum", "count"],
            "caregiving_leave_top_up": "sum",
            "income_tax_single": "sum",
        }
    )

    # Compute per-agent values for each period (divide by number of agents)
    period_stats["net_government_budget_per_agent"] = (
        period_stats[("net_government_budget", "sum")]
        / period_stats[("net_government_budget", "count")]
    )
    period_stats["caregiving_leave_top_up_per_agent"] = (
        period_stats[("caregiving_leave_top_up", "sum")]
        / period_stats[("net_government_budget", "count")]
    )
    period_stats["income_tax_single_per_agent"] = (
        period_stats[("income_tax_single", "sum")]
        / period_stats[("net_government_budget", "count")]
    )

    # Sum across all periods
    total_net_government_budget = period_stats["net_government_budget_per_agent"].sum()
    total_caregiving_leave_top_up = period_stats[
        "caregiving_leave_top_up_per_agent"
    ].sum()
    total_income_tax_single = period_stats["income_tax_single_per_agent"].sum()

    # Save to CSV
    result = pd.DataFrame(
        {
            "variable": [
                "net_government_budget",
                "caregiving_leave_top_up",
                "income_tax_single",
            ],
            "total_sum": [
                total_net_government_budget,
                total_caregiving_leave_top_up,
                total_income_tax_single,
            ],
        }
    )
    result.to_csv(path_to_output, index=False)

    print(f"Government budget sums saved to {path_to_output}")
    print(f"Net government budget: {total_net_government_budget:.2f}")
    print(f"Caregiving leave top-up: {total_caregiving_leave_top_up:.2f}")
    print(f"Income tax single: {total_income_tax_single:.2f}")
