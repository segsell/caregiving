"""Publication: fiscal costs of caregiving policies (LaTeX table)."""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import INFORMAL_CARE
from caregiving.tables.publication.task_government_budget_caregiving_leave_top_up import (  # noqa: E501
    compute_net_caregiving_leave_top_up_cost,
)

# Caregiving is possible up to and including this age (from specs: end_age_caregiving).
# Costs accrue only for periods with age <= END_AGE_CAREGIVING.
END_AGE_CAREGIVING = 100

# Outcome columns for which we report avg per caregiver (currency). In df: wealth_unit.
OUTCOME_COLUMNS_AVG_PER_CAREGIVER = [
    "income_tax",
    "income_tax_single",
    "joint_gross_labor_income",
    "joint_gross_retirement_income",
    "household_unemployment_benefits",
    "total_tax_revenue",
    "net_government_budget",
]
# LaTeX column headers for the table (one per outcome).
OUTCOME_COLUMN_LABELS = [
    "Avg. income tax per caregiver (currency)",
    "Avg. income tax (single) per caregiver (currency)",
    "Avg. joint gross labor income per caregiver (currency)",
    "Avg. joint gross retirement income per caregiver (currency)",
    "Avg. household unemployment benefits per caregiver (currency)",
    "Avg. total tax revenue per caregiver (currency)",
    "Avg. net government budget per caregiver (currency)",
]


@pytask.mark.tables
@pytask.mark.fiscal_costs
@pytask.mark.publication
def task_create_fiscal_costs(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "fiscal_costs_caregiving_policies.tex",
) -> None:
    """Create LaTeX table of fiscal costs of caregiving policies.

    Three policies:
    1) Baseline: cash benefits for informal care (care_benefits_and_costs).
    2) Normal caregiving leave with job retention.
    3) Full caregiving leave with job retention.

    For each policy we compute:
    - Total cost over the life cycle (only periods with age <= end_age_caregiving,
      i.e. 70; costs accrue only while caregiving is possible).
    - Total number of unique caregivers (agents who ever choose informal care
      in the caregiving window).
    - Average cost per caregiver = total cost / n_unique_caregivers.
    - Avg. monthly cost per caregiver per caregiving month = (avg cost per
      caregiver) / 12 / (avg caregiving years).

    Interpretation: average lifetime fiscal cost per person who ever provides
    informal care under that policy. Alternative metrics could be: cost per
    caregiver-year (total cost / total caregiver-periods), total cost per capita
    (over all agents), or present value per caregiver; the chosen metric is
    standard for "cost per beneficiary" and is comparable across policies.

    """
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = float(specs["wealth_unit"])
    start_age = int(specs.get("start_age", 30))
    end_age_caregiving = int(specs.get("end_age_caregiving", END_AGE_CAREGIVING))

    baseline_df = pd.read_pickle(path_to_baseline_sim)
    normal_df = pd.read_pickle(path_to_normal_leave_sim)
    full_df = pd.read_pickle(path_to_full_leave_sim)

    if "age" not in baseline_df.columns:
        baseline_df = baseline_df.copy()
        baseline_df["age"] = start_age + baseline_df["period"]
    if "age" not in normal_df.columns:
        normal_df = normal_df.copy()
        normal_df["age"] = start_age + normal_df["period"]
    if "age" not in full_df.columns:
        full_df = full_df.copy()
        full_df["age"] = start_age + full_df["period"]

    cost_baseline, n_baseline, periods_baseline = _total_cost_baseline(
        baseline_df, wealth_unit, start_age=start_age
    )
    cost_normal, n_normal, periods_normal = _total_cost_leave(normal_df, specs)
    cost_full, n_full, periods_full = _total_cost_leave(full_df, specs)

    avg_baseline = cost_baseline / n_baseline if n_baseline else np.nan
    avg_normal = cost_normal / n_normal if n_normal else np.nan
    avg_full = cost_full / n_full if n_full else np.nan

    avg_years_baseline = periods_baseline / n_baseline if n_baseline else np.nan
    avg_years_normal = periods_normal / n_normal if n_normal else np.nan
    avg_years_full = periods_full / n_full if n_full else np.nan

    # Avg. monthly per caregiver per caregiving month
    # = (avg cost per caregiver)/12/(avg years)
    def _avg_monthly_per_caregiving_month(avg_cost, avg_years):
        if np.isnan(avg_years) or avg_years <= 0:
            return np.nan
        return avg_cost / 12.0 / avg_years

    avg_monthly_baseline = _avg_monthly_per_caregiving_month(
        avg_baseline, avg_years_baseline
    )
    avg_monthly_normal = _avg_monthly_per_caregiving_month(avg_normal, avg_years_normal)
    avg_monthly_full = _avg_monthly_per_caregiving_month(avg_full, avg_years_full)

    # Avg per caregiver for outcomes (same rows as cost: lagged_choice in INFORMAL_CARE)
    outcomes_baseline = _avg_outcomes_per_caregiver_baseline(
        baseline_df, wealth_unit, start_age, OUTCOME_COLUMNS_AVG_PER_CAREGIVER
    )
    outcomes_normal = _avg_outcomes_per_caregiver_leave(
        normal_df, specs, OUTCOME_COLUMNS_AVG_PER_CAREGIVER
    )
    outcomes_full = _avg_outcomes_per_caregiver_leave(
        full_df, specs, OUTCOME_COLUMNS_AVG_PER_CAREGIVER
    )

    table_dict = {
        "Policy": [
            "Baseline (cash benefits for informal care)",
            "Normal caregiving leave with job retention",
            "Full caregiving leave with job retention",
        ],
        "Total cost (currency)": [cost_baseline, cost_normal, cost_full],
        "N unique caregivers": [n_baseline, n_normal, n_full],
        "Avg. caregiving years (cond. on caregivers)": [
            avg_years_baseline,
            avg_years_normal,
            avg_years_full,
        ],
        "Avg cost per caregiver (currency)": [avg_baseline, avg_normal, avg_full],
        "Avg. monthly cost per caregiver per caregiving month (currency)": [
            avg_monthly_baseline,
            avg_monthly_normal,
            avg_monthly_full,
        ],
    }
    for col, label in zip(
        OUTCOME_COLUMNS_AVG_PER_CAREGIVER, OUTCOME_COLUMN_LABELS, strict=True
    ):
        table_dict[label] = [
            outcomes_baseline[col],
            outcomes_normal[col],
            outcomes_full[col],
        ]
    table = pd.DataFrame(table_dict)

    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)
    n_cols = 6 + len(OUTCOME_COLUMNS_AVG_PER_CAREGIVER)
    latex_str = table.to_latex(
        index=False,
        float_format="%.2f",
        column_format="l" + "r" * (n_cols - 1),
        caption="Fiscal costs of caregiving policies (life cycle until age "
        f"{end_age_caregiving}).",
        label="tab:fiscal_costs_caregiving",
    )
    path_to_save_table.write_text(latex_str)


def _total_cost_baseline(
    df: pd.DataFrame, wealth_unit: float, start_age: int = 30
) -> tuple[float, int, int]:
    """Total government cost, unique caregivers, and total caregiver-periods (baseline).

    In the model, care_benefits_and_costs in period t is computed from *lagged_choice*
    (d_{t-1}): the benefit *paid* in period t is for having been in informal care in
    period t-1 (see transfers.calc_care_benefits_and_costs(period, lagged_choice, ...)).
    So we sum over rows where *lagged_choice* in INFORMAL_CARE (and age <= 70) to get
    total benefits paid. Unique caregivers and caregiver-periods use *choice* (actual
    periods in care) for the denominator and avg. caregiving years.
    Returns (cost, n_caregivers, n_caregiver_periods).
    """
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df = df.copy()
    if "age" not in df.columns and "period" in df.columns:
        df["age"] = start_age + df["period"]
    df = df[df["age"] <= END_AGE_CAREGIVING].copy()
    if "care_benefits_and_costs" not in df.columns:
        return 0.0, 0, 0
    if "lagged_choice" not in df.columns:
        return 0.0, 0, 0
    care_choices = np.asarray(INFORMAL_CARE).ravel()
    # Total cost: sum benefits *paid* (each row's value is for lagged_choice)
    rows_with_benefit = df[df["lagged_choice"].isin(care_choices)]
    if rows_with_benefit.empty:
        return 0.0, 0, 0
    benefits = np.maximum(rows_with_benefit["care_benefits_and_costs"].values, 0.0)
    cost = (benefits * wealth_unit).sum()
    # Unique caregivers and periods in care: use current choice
    caregivers = df[df["choice"].isin(care_choices)]
    if caregivers.empty:
        return float(cost), 0, 0
    n_caregivers = caregivers["agent"].nunique()
    n_caregiver_periods = len(caregivers)
    return float(cost), int(n_caregivers), n_caregiver_periods


def _total_cost_leave(df: pd.DataFrame, model_specs: dict) -> tuple[float, int, int]:
    """Total net caregiving leave top-up cost, unique caregivers, caregiver-periods.

    In the model, caregiving_leave_top_up in period t is computed from *lagged_choice*
    (d_{t-1}): the top-up *paid* in period t is for having been in informal care in
    period t-1 (see caregiving_leave_top_up.calc_*_caregiving_leave_top_up(
    lagged_choice=...)). So we sum net top-up over rows where *lagged_choice* in
    INFORMAL_CARE (and age <= 70). Unique caregivers and caregiver-periods use
    *choice* (actual periods in care) for the denominator and avg. caregiving years.
    Returns (cost, n_caregivers, n_caregiver_periods).

    """
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df = df.copy()
    start_age = int(model_specs.get("start_age", 30))
    if "age" not in df.columns:
        df["age"] = start_age + df["period"]
    df = df[df["age"] <= END_AGE_CAREGIVING].copy()
    required = [
        "caregiving_leave_top_up",
        "own_income_after_ssc",
        "income_tax_single",
        "lagged_choice",
    ]
    if not all(c in df.columns for c in required):
        return 0.0, 0, 0
    care_choices = np.asarray(INFORMAL_CARE).ravel()
    # Total cost: sum net top-up over rows where benefit is *paid* (lagged_choice)
    rows_with_top_up = df[df["lagged_choice"].isin(care_choices)].copy()
    if rows_with_top_up.empty:
        return 0.0, 0, 0
    net_cost = compute_net_caregiving_leave_top_up_cost(
        caregiving_leave_top_up=rows_with_top_up["caregiving_leave_top_up"].values,
        own_income_after_ssc=rows_with_top_up["own_income_after_ssc"].values,
        income_tax_single=rows_with_top_up["income_tax_single"].values,
        model_specs=model_specs,
    )
    cost = float(np.asarray(net_cost).sum())
    # Unique caregivers and periods in care: use current choice
    caregivers = df[df["choice"].isin(care_choices)]
    if caregivers.empty:
        return cost, 0, 0
    n_caregivers = int(caregivers["agent"].nunique())
    n_caregiver_periods = len(caregivers)
    return cost, n_caregivers, n_caregiver_periods


def _avg_outcomes_per_caregiver_baseline(
    df: pd.DataFrame,
    wealth_unit: float,
    start_age: int,
    outcome_columns: list[str],
) -> dict[str, float]:
    """Average of each outcome (in currency) per unique caregiver, baseline.

    Sums each outcome over rows where lagged_choice in INFORMAL_CARE (and age <= 70),
    multiplies by wealth_unit, divides by n_caregivers (choice in INFORMAL_CARE).
    Missing columns yield np.nan for that outcome.
    """
    result = {col: np.nan for col in outcome_columns}
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df = df.copy()
    if "age" not in df.columns and "period" in df.columns:
        df["age"] = start_age + df["period"]
    df = df[df["age"] <= END_AGE_CAREGIVING].copy()
    if "lagged_choice" not in df.columns:
        return result
    care_choices = np.asarray(INFORMAL_CARE).ravel()
    rows_with_benefit = df[df["lagged_choice"].isin(care_choices)]
    caregivers = df[df["choice"].isin(care_choices)]
    if rows_with_benefit.empty or caregivers.empty:
        return result
    n_caregivers = caregivers["agent"].nunique()
    for col in outcome_columns:
        if col not in df.columns:
            continue
        total = (rows_with_benefit[col].values * wealth_unit).sum()
        result[col] = float(total / n_caregivers)
    return result


def _avg_outcomes_per_caregiver_leave(
    df: pd.DataFrame, model_specs: dict, outcome_columns: list[str]
) -> dict[str, float]:
    """Average of each outcome (in currency) per unique caregiver, leave policies.

    Same logic as baseline: sum over lagged_choice in INFORMAL_CARE, * wealth_unit,
    / n_caregivers. Uses model_specs['wealth_unit'] and start_age.
    """
    result = {col: np.nan for col in outcome_columns}
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df = df.copy()
    start_age = int(model_specs.get("start_age", 30))
    wealth_unit = float(model_specs.get("wealth_unit", 1.0))
    if "age" not in df.columns:
        df["age"] = start_age + df["period"]
    df = df[df["age"] <= END_AGE_CAREGIVING].copy()
    if "lagged_choice" not in df.columns:
        return result
    care_choices = np.asarray(INFORMAL_CARE).ravel()
    rows_with_benefit = df[df["lagged_choice"].isin(care_choices)]
    caregivers = df[df["choice"].isin(care_choices)]
    if rows_with_benefit.empty or caregivers.empty:
        return result
    n_caregivers = caregivers["agent"].nunique()
    for col in outcome_columns:
        if col not in df.columns:
            continue
        total = (rows_with_benefit[col].values * wealth_unit).sum()
        result[col] = float(total / n_caregivers)
    return result
