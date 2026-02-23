"""Publication: fiscal costs of caregiving policies (LaTeX table)."""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    CARE_DEMAND_INTENSIVE,
    CARE_DEMAND_LIGHT,
    FORMAL_CARE,
    INFORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
)
from caregiving.tables.publication.task_government_budget_caregiving_leave_top_up import (  # noqa: E501
    compute_net_caregiving_leave_top_up_cost,
)

# ---------------------------------------------------------------------------
# Government cost of formal care (social planner perspective)
# ---------------------------------------------------------------------------
# Monthly nursing home cost by care demand level (EUR, 2010 prices).
GOV_COST_NURSING_HOME_LIGHT_2010 = 1023
GOV_COST_NURSING_HOME_INTENSIVE_2010 = 1394.5

# Monthly pure formal home care cost by care demand level (EUR, 2010 prices).
PURE_FORMAL_HOME_CARE_COST_LIGHT_2010 = 440
PURE_FORMAL_HOME_CARE_COST_INTENSIVE_2010 = 1275

# Monthly combination care cost (intensive demand only; 50% of Sachleistungen).
COMBINATION_CARE_COST_INTENSIVE_2010 = PURE_FORMAL_HOME_CARE_COST_INTENSIVE_2010 * 0.5

# Share of intensive informal caregivers who also use in-kind services (combo care).
IN_KIND_BENEFITS_UTILIZATION_RATE = 0.7

# ---------------------------------------------------------------------------
# Share breakdown within the "formal care" choice (FORMAL_CARE = [4,5,6,7]).
# The formal care choice in the model encompasses three real-world care types:
#   1. Nursing home care (stationaere Pflege)
#   2. Pure formal home care (ambulante Sachleistungen)
#   3. 24-hour live-in care (privately borne → zero government cost)
# ---------------------------------------------------------------------------
SHARE_LIVE_IN_HOME_CARE_2021 = 0.1275
SHARE_PURE_FORMAL_HOME_CARE_2021 = 0.1619
SHARE_NURSING_HOME = (
    1.0 - SHARE_LIVE_IN_HOME_CARE_2021 - SHARE_PURE_FORMAL_HOME_CARE_2021
)

# ---------------------------------------------------------------------------
# Derived annual government costs per formal-care period, by demand level.
# Live-in care has zero government cost, so only nursing home + home care enter.
# ---------------------------------------------------------------------------
GOV_ANNUAL_FC_COST_LIGHT = (
    SHARE_NURSING_HOME * GOV_COST_NURSING_HOME_LIGHT_2010 * 12
    + SHARE_PURE_FORMAL_HOME_CARE_2021 * PURE_FORMAL_HOME_CARE_COST_LIGHT_2010 * 12
)
GOV_ANNUAL_FC_COST_INTENSIVE = (
    SHARE_NURSING_HOME * GOV_COST_NURSING_HOME_INTENSIVE_2010 * 12
    + SHARE_PURE_FORMAL_HOME_CARE_2021 * PURE_FORMAL_HOME_CARE_COST_INTENSIVE_2010 * 12
)

# Annual expected government cost per combination-care period (intensive only).
GOV_ANNUAL_COMBINATION_CARE_COST = COMBINATION_CARE_COST_INTENSIVE_2010 * 12

# ---------------------------------------------------------------------------
# Columns needed from the simulation DataFrames. Loading only these saves RAM.
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS = [
    "agent",
    "period",
    "choice",
    "lagged_choice",
    "care_demand",
    # --- Revenue side ---
    "income_tax",
    "income_tax_single",
    "own_ssc",
    "partner_ssc",
    "total_tax_revenue",
    # --- Expenditure side ---
    "government_expenditures",
    "care_benefits_and_costs",
    "caregiving_leave_top_up",
    "formal_care_costs",
    "household_unemployment_benefits",
    "unemployment_transfer_paid",
    # --- Net ---
    "net_government_budget",
    # --- Income ---
    "own_income_after_ssc",
    "joint_gross_labor_income",
    "joint_gross_retirement_income",
    # --- Leave-specific decomposition ---
    "normal_leave_net_cost",
    "tax_increase_from_progression",
    "full_leave_net_cost",
    "full_leave_net_cost_incl_transfer",
    "tax_attributable_to_full_leave",
    "delta_transfer_savings",
    # --- Decomposition & auxiliary ---
    "labor_income_after_ssc",
    "retirement_income_after_ssc",
    "partner_income_after_ssc",
    "gross_labor_income",
    "gross_retirement_income",
    "child_benefits",
]

# Outcome columns for which we report avg per caregiver (currency). In df: wealth_unit.
# Missing columns yield NaN (e.g. baseline has no caregiving_leave_top_up).
OUTCOME_COLUMNS_AVG_PER_CAREGIVER = [
    # --- Revenue side ---
    "income_tax",
    "income_tax_single",
    "own_ssc",
    "partner_ssc",
    "total_tax_revenue",
    # --- Expenditure side ---
    "government_expenditures",
    "care_benefits_and_costs",
    "caregiving_leave_top_up",
    "formal_care_costs",
    "household_unemployment_benefits",
    "unemployment_transfer_paid",
    # --- Net ---
    "net_government_budget",
    # --- Income ---
    "own_income_after_ssc",
    "joint_gross_labor_income",
    "joint_gross_retirement_income",
    # --- Leave-specific decomposition ---
    "normal_leave_net_cost",
    "tax_increase_from_progression",
    "full_leave_net_cost",
    "full_leave_net_cost_incl_transfer",
    "tax_attributable_to_full_leave",
    "delta_transfer_savings",
]
OUTCOME_COLUMN_LABELS = [
    # --- Revenue side ---
    "Avg. income tax",
    "Avg. income tax (single)",
    "Avg. own SSC",
    "Avg. partner SSC",
    "Avg. total tax revenue",
    # --- Expenditure side ---
    "Avg. gov. expenditures",
    "Avg. care benefits and costs",
    "Avg. leave top-up (gross)",
    "Avg. formal care costs (agent)",
    "Avg. HH unemployment benefits",
    "Avg. unemployment transfer paid",
    # --- Net ---
    "Avg. net gov. budget",
    # --- Income ---
    "Avg. own income after SSC",
    "Avg. joint gross labor income",
    "Avg. joint gross retirement income",
    # --- Leave-specific decomposition ---
    "Avg. normal leave net cost",
    "Avg. tax increase (progression)",
    "Avg. full leave net cost",
    "Avg. full leave net cost (incl. transfer)",
    "Avg. tax attributable to full leave",
    "Avg. transfer savings (delta)",
]


def _load_sim_df(path: Path) -> pd.DataFrame:
    """Load a simulation pickle, keeping only the columns needed for fiscal analysis.

    This avoids holding the full wide DataFrame in memory.
    """
    df = pd.read_pickle(path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    keep = [c for c in _REQUIRED_COLUMNS if c in df.columns]
    return df[keep].copy()


def _add_lagged_care_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged_care_demand column (care_demand from t-1, matching lagged_choice).

    Since lagged_choice at period t is d_{t-1}, the care demand that was active
    when formal care was chosen is care_demand_{t-1}, not the current period's.
    """
    if "care_demand" not in df.columns:
        return df
    df = df.sort_values(["agent", "period"])
    df["lagged_care_demand"] = df.groupby("agent")["care_demand"].shift(1)
    return df


def _n_total_agents(df: pd.DataFrame) -> int:
    """Count total unique agents in the DataFrame (already age-filtered)."""
    return int(df["agent"].nunique()) if "agent" in df.columns else 0


def _total_sum_column(df: pd.DataFrame, column: str, wealth_unit: float) -> float:
    """Sum column * wealth_unit over rows where lagged_choice in INFORMAL_CARE.

    Expects df already age-filtered.
    """
    if column not in df.columns or "lagged_choice" not in df.columns:
        return np.nan
    care_choices = np.asarray(INFORMAL_CARE).ravel()
    rows = df[df["lagged_choice"].isin(care_choices)]
    if rows.empty:
        return np.nan
    return float((rows[column].values * wealth_unit).sum())


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

    Life-cycle scope: The fiscal comparison covers the full model life cycle
    (ages start_age to end_age). This is important because retirement income
    effects of caregiving policies persist well beyond the caregiving window
    (end_age_caregiving, typically 70). The max statutory retirement age is 67,
    but pension income accrues for all subsequent periods.
    """
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = float(specs["wealth_unit"])
    start_age = int(specs.get("start_age", 30))
    end_age = int(specs.get("end_age", 100))

    # Load slim DataFrames (only relevant columns)
    baseline_df = _load_sim_df(path_to_baseline_sim)
    normal_df = _load_sim_df(path_to_normal_leave_sim)
    full_df = _load_sim_df(path_to_full_leave_sim)

    # Ensure age column exists
    for df in (baseline_df, normal_df, full_df):
        if "age" not in df.columns and "period" in df.columns:
            df["age"] = start_age + df["period"]

    # Verify period and age evolve identically (age = start_age + period)
    for label, df in [
        ("baseline", baseline_df),
        ("normal", normal_df),
        ("full", full_df),
    ]:
        if "period" in df.columns and "age" in df.columns:
            assert (
                df["age"] == start_age + df["period"]
            ).all(), f"{label}: age != start_age + period"

    # Compute lagged_care_demand (care_demand from t-1, matching lagged_choice)
    baseline_df = _add_lagged_care_demand(baseline_df)
    normal_df = _add_lagged_care_demand(normal_df)
    full_df = _add_lagged_care_demand(full_df)

    cost_baseline, n_baseline, periods_baseline = _total_cost_baseline(
        baseline_df, wealth_unit
    )
    cost_normal, n_normal, periods_normal = _total_cost_leave(normal_df, specs)
    cost_full, n_full, periods_full = _total_cost_leave(full_df, specs)

    avg_baseline = cost_baseline / n_baseline if n_baseline else np.nan
    avg_normal = cost_normal / n_normal if n_normal else np.nan
    avg_full = cost_full / n_full if n_full else np.nan

    avg_years_baseline = periods_baseline / n_baseline if n_baseline else np.nan
    avg_years_normal = periods_normal / n_normal if n_normal else np.nan
    avg_years_full = periods_full / n_full if n_full else np.nan

    def _avg_monthly_per_caregiving_month(avg_cost, avg_years):
        if np.isnan(avg_years) or avg_years <= 0:
            return np.nan
        return avg_cost / 12.0 / avg_years

    avg_monthly_baseline = _avg_monthly_per_caregiving_month(
        avg_baseline, avg_years_baseline
    )
    avg_monthly_normal = _avg_monthly_per_caregiving_month(avg_normal, avg_years_normal)
    avg_monthly_full = _avg_monthly_per_caregiving_month(avg_full, avg_years_full)

    # --- N total agents, share ever caregiving, per-capita cost ---
    n_total_baseline = _n_total_agents(baseline_df)
    n_total_normal = _n_total_agents(normal_df)
    n_total_full = _n_total_agents(full_df)

    def _safe_div(num, den):
        return num / den if den else np.nan

    share_cg_baseline = _safe_div(n_baseline, n_total_baseline)
    share_cg_normal = _safe_div(n_normal, n_total_normal)
    share_cg_full = _safe_div(n_full, n_total_full)

    percap_baseline = _safe_div(cost_baseline, n_total_baseline)
    percap_normal = _safe_div(cost_normal, n_total_normal)
    percap_full = _safe_div(cost_full, n_total_full)

    # --- Total gross benefit (leave_top_up for leave; gross care_benefits for baseline)
    gross_benefit_baseline = cost_baseline
    gross_benefit_normal = _total_sum_column(
        normal_df, "caregiving_leave_top_up", wealth_unit
    )
    gross_benefit_full = _total_sum_column(
        full_df, "caregiving_leave_top_up", wealth_unit
    )

    # --- Total net cost from model aux (NaN if column missing in old sim data) ---
    net_cost_aux_baseline = cost_baseline
    net_cost_aux_normal = _total_sum_column(
        normal_df, "normal_leave_net_cost", wealth_unit
    )
    net_cost_aux_full = _total_sum_column(full_df, "full_leave_net_cost", wealth_unit)
    net_cost_incl_transfer_full = _total_sum_column(
        full_df, "full_leave_net_cost_incl_transfer", wealth_unit
    )

    # --- Government cost of formal care (social planner perspective) ---
    gov_fc_baseline, n_fc_baseline, avg_fc_yrs_baseline = _formal_care_stats(
        baseline_df
    )
    gov_fc_normal, n_fc_normal, avg_fc_yrs_normal = _formal_care_stats(normal_df)
    gov_fc_full, n_fc_full, avg_fc_yrs_full = _formal_care_stats(full_df)

    # --- Combination care (intensive informal + formal home care services) ---
    combo_baseline = _combination_care_stats(baseline_df)
    combo_normal = _combination_care_stats(normal_df)
    combo_full = _combination_care_stats(full_df)

    # Total government care cost = formal care + combination care
    gov_total_care_baseline = gov_fc_baseline + combo_baseline["total_cost"]
    gov_total_care_normal = gov_fc_normal + combo_normal["total_cost"]
    gov_total_care_full = gov_fc_full + combo_full["total_cost"]

    # Avg per caregiver for outcomes (same rows as cost: lagged_choice in INFORMAL_CARE)
    outcomes_baseline = _avg_outcomes_per_caregiver(
        baseline_df, wealth_unit, OUTCOME_COLUMNS_AVG_PER_CAREGIVER
    )
    outcomes_normal = _avg_outcomes_per_caregiver(
        normal_df, wealth_unit, OUTCOME_COLUMNS_AVG_PER_CAREGIVER
    )
    outcomes_full = _avg_outcomes_per_caregiver(
        full_df, wealth_unit, OUTCOME_COLUMNS_AVG_PER_CAREGIVER
    )

    # =====================================================================
    # Build table
    # =====================================================================
    table_dict = {
        "Policy": [
            "Baseline (cash benefits)",
            r"Normal leave (65\%)",
            r"Full leave (100\%)",
        ],
        "Total cost (net)": [cost_baseline, cost_normal, cost_full],
        "Total gross benefit": [
            gross_benefit_baseline,
            gross_benefit_normal,
            gross_benefit_full,
        ],
        "Total net cost (model)": [
            net_cost_aux_baseline,
            net_cost_aux_normal,
            net_cost_aux_full,
        ],
        "Gov. formal care cost (total)": [gov_fc_baseline, gov_fc_normal, gov_fc_full],
        "Avg. gov. formal care cost per caregiver": [
            _safe_div(gov_fc_baseline, n_baseline),
            _safe_div(gov_fc_normal, n_normal),
            _safe_div(gov_fc_full, n_full),
        ],
        "Avg. gov. formal care cost per FC user": [
            _safe_div(gov_fc_baseline, n_fc_baseline),
            _safe_div(gov_fc_normal, n_fc_normal),
            _safe_div(gov_fc_full, n_fc_full),
        ],
        "Avg. gov. FC cost per CG (pct of baseline)": [
            np.nan,
            _safe_div(
                _safe_div(gov_fc_normal, n_normal),
                _safe_div(gov_fc_baseline, n_baseline),
            )
            * 100,
            _safe_div(
                _safe_div(gov_fc_full, n_full),
                _safe_div(gov_fc_baseline, n_baseline),
            )
            * 100,
        ],
        "Avg. gov. FC cost per FC user (pct of baseline)": [
            np.nan,
            _safe_div(
                _safe_div(gov_fc_normal, n_fc_normal),
                _safe_div(gov_fc_baseline, n_fc_baseline),
            )
            * 100,
            _safe_div(
                _safe_div(gov_fc_full, n_fc_full),
                _safe_div(gov_fc_baseline, n_fc_baseline),
            )
            * 100,
        ],
        "Gov. formal care cost per capita": [
            _safe_div(gov_fc_baseline, n_total_baseline),
            _safe_div(gov_fc_normal, n_total_normal),
            _safe_div(gov_fc_full, n_total_full),
        ],
        "Avg. formal care years": [
            avg_fc_yrs_baseline,
            avg_fc_yrs_normal,
            avg_fc_yrs_full,
        ],
        "N formal care users": [n_fc_baseline, n_fc_normal, n_fc_full],
        # --- Combination care ---
        "Gov. combination care cost (total)": [
            combo_baseline["total_cost"],
            combo_normal["total_cost"],
            combo_full["total_cost"],
        ],
        "Avg. gov. combo care cost per caregiver": [
            _safe_div(combo_baseline["total_cost"], n_baseline),
            _safe_div(combo_normal["total_cost"], n_normal),
            _safe_div(combo_full["total_cost"], n_full),
        ],
        "Avg. gov. combo care cost per eligible": [
            combo_baseline["avg_cost_per_eligible"],
            combo_normal["avg_cost_per_eligible"],
            combo_full["avg_cost_per_eligible"],
        ],
        "Avg. gov. combo care cost per combo user": [
            combo_baseline["avg_cost_per_combo_user"],
            combo_normal["avg_cost_per_combo_user"],
            combo_full["avg_cost_per_combo_user"],
        ],
        "Avg. combination care years (eligible)": [
            combo_baseline["avg_combo_years"],
            combo_normal["avg_combo_years"],
            combo_full["avg_combo_years"],
        ],
        "Avg. combination care years (per user)": [
            combo_baseline["avg_combo_years_per_user"],
            combo_normal["avg_combo_years_per_user"],
            combo_full["avg_combo_years_per_user"],
        ],
        "N eligible for combo care": [
            combo_baseline["n_eligible_agents"],
            combo_normal["n_eligible_agents"],
            combo_full["n_eligible_agents"],
        ],
        "N combo care users (expected)": [
            combo_baseline["n_expected_combo_users"],
            combo_normal["n_expected_combo_users"],
            combo_full["n_expected_combo_users"],
        ],
        # --- Total government care cost (formal + combination) ---
        "Gov. total care cost": [
            gov_total_care_baseline,
            gov_total_care_normal,
            gov_total_care_full,
        ],
        "Gov. total care cost per caregiver": [
            _safe_div(gov_total_care_baseline, n_baseline),
            _safe_div(gov_total_care_normal, n_normal),
            _safe_div(gov_total_care_full, n_full),
        ],
        "Gov. total care cost per capita": [
            _safe_div(gov_total_care_baseline, n_total_baseline),
            _safe_div(gov_total_care_normal, n_total_normal),
            _safe_div(gov_total_care_full, n_total_full),
        ],
        "N caregivers": [n_baseline, n_normal, n_full],
        "N total agents": [n_total_baseline, n_total_normal, n_total_full],
        "Share ever caregiving": [share_cg_baseline, share_cg_normal, share_cg_full],
        "Avg. caregiving years": [
            avg_years_baseline,
            avg_years_normal,
            avg_years_full,
        ],
        "Avg cost per caregiver": [avg_baseline, avg_normal, avg_full],
        "Per-capita cost (all agents)": [percap_baseline, percap_normal, percap_full],
        "Avg. monthly cost per CG month": [
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
    table_wide = pd.DataFrame(table_dict)

    # Transpose: metrics as rows, policies as columns.
    table = table_wide.set_index("Policy").T
    table.index.name = "Metric"

    # Compute delta columns
    baseline_col = table.columns[0]
    for leave_col, delta_label in [
        (table.columns[1], "Delta Normal - Baseline"),
        (table.columns[2], "Delta Full - Baseline"),
    ]:
        table[delta_label] = pd.to_numeric(
            table[leave_col], errors="coerce"
        ) - pd.to_numeric(table[baseline_col], errors="coerce")

    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)
    n_data_cols = len(table.columns)
    latex_str = table.to_latex(
        float_format="%.2f",
        column_format="l" + "r" * n_data_cols,
        caption=(
            f"Fiscal costs of caregiving policies (life cycle ages {start_age}"
            f"--{end_age})."
        ),
        label="tab:fiscal_costs_caregiving",
        na_rep="--",
    )
    path_to_save_table.write_text(latex_str)


def _total_cost_baseline(
    df: pd.DataFrame, wealth_unit: float
) -> tuple[float, int, int]:
    """Total government cost, unique caregivers, and total caregiver-periods (baseline).

    care_benefits_and_costs in period t is computed from *lagged_choice* (d_{t-1}):
    the benefit paid in period t is for having been in informal care in t-1.
    Unique caregivers and caregiver-periods use *choice* (actual periods in care).
    Returns (cost, n_caregivers, n_caregiver_periods).
    """
    if "care_benefits_and_costs" not in df.columns:
        return 0.0, 0, 0
    if "lagged_choice" not in df.columns:
        return 0.0, 0, 0
    care_choices = np.asarray(INFORMAL_CARE).ravel()
    rows_with_benefit = df[df["lagged_choice"].isin(care_choices)]
    if rows_with_benefit.empty:
        return 0.0, 0, 0
    benefits = np.maximum(rows_with_benefit["care_benefits_and_costs"].values, 0.0)
    cost = (benefits * wealth_unit).sum()
    caregivers = df[df["choice"].isin(care_choices)]
    if caregivers.empty:
        return float(cost), 0, 0
    n_caregivers = caregivers["agent"].nunique()
    n_caregiver_periods = len(caregivers)
    return float(cost), int(n_caregivers), n_caregiver_periods


def _total_cost_leave(df: pd.DataFrame, model_specs: dict) -> tuple[float, int, int]:
    """Total net caregiving leave top-up cost, unique caregivers, caregiver-periods.

    caregiving_leave_top_up in period t is computed from *lagged_choice* (d_{t-1}).
    Returns (cost, n_caregivers, n_caregiver_periods).
    """
    required = [
        "caregiving_leave_top_up",
        "own_income_after_ssc",
        "income_tax_single",
        "lagged_choice",
    ]
    if not all(c in df.columns for c in required):
        return 0.0, 0, 0
    care_choices = np.asarray(INFORMAL_CARE).ravel()
    rows_with_top_up = df[df["lagged_choice"].isin(care_choices)]
    if rows_with_top_up.empty:
        return 0.0, 0, 0
    net_cost = compute_net_caregiving_leave_top_up_cost(
        caregiving_leave_top_up=rows_with_top_up["caregiving_leave_top_up"].values,
        own_income_after_ssc=rows_with_top_up["own_income_after_ssc"].values,
        income_tax_single=rows_with_top_up["income_tax_single"].values,
        model_specs=model_specs,
    )
    cost = float(np.asarray(net_cost).sum())
    caregivers = df[df["choice"].isin(care_choices)]
    if caregivers.empty:
        return cost, 0, 0
    n_caregivers = int(caregivers["agent"].nunique())
    n_caregiver_periods = len(caregivers)
    return cost, n_caregivers, n_caregiver_periods


def _formal_care_stats(
    df: pd.DataFrame,
) -> tuple[float, int, float]:
    """Government formal care cost, N formal care users, and avg formal care years.

    The formal care choice in the model encompasses three real-world care types:
      1. Nursing home (share = SHARE_NURSING_HOME): demand-level cost
      2. Pure formal home care (share = SHARE_PURE_FORMAL_HOME_CARE_2021): demand-level
      3. 24h live-in care (share = SHARE_LIVE_IN_HOME_CARE_2021): zero gov cost

    Costs are differentiated by care demand level (light=1 vs intensive=2) using
    lagged_care_demand (the care demand from t-1, matching lagged_choice).

    Returns (total_gov_cost, n_formal_care_users, avg_formal_care_years).
    """
    if "lagged_choice" not in df.columns:
        return np.nan, 0, np.nan
    formal_choices = np.asarray(FORMAL_CARE).ravel()
    fc_mask = df["lagged_choice"].isin(formal_choices)
    n_formal_care_periods = int(fc_mask.sum())
    if n_formal_care_periods == 0:
        return 0.0, 0, 0.0

    fc_rows = df.loc[fc_mask]
    if "lagged_care_demand" in fc_rows.columns:
        n_light = int((fc_rows["lagged_care_demand"] == CARE_DEMAND_LIGHT).sum())
        n_intensive = int(
            (fc_rows["lagged_care_demand"] == CARE_DEMAND_INTENSIVE).sum()
        )
        n_unknown = n_formal_care_periods - n_light - n_intensive
    else:
        n_light = 0
        n_intensive = 0
        n_unknown = n_formal_care_periods

    # For rows with unknown demand (e.g. first period NaN), use average of the two.
    avg_cost_fallback = (GOV_ANNUAL_FC_COST_LIGHT + GOV_ANNUAL_FC_COST_INTENSIVE) / 2.0
    total_cost = float(
        n_light * GOV_ANNUAL_FC_COST_LIGHT
        + n_intensive * GOV_ANNUAL_FC_COST_INTENSIVE
        + n_unknown * avg_cost_fallback
    )
    n_fc_users = int(df.loc[fc_mask, "agent"].nunique())
    avg_fc_years = n_formal_care_periods / n_fc_users if n_fc_users else np.nan
    return total_cost, n_fc_users, avg_fc_years


def _combination_care_stats(df: pd.DataFrame) -> dict[str, float]:
    """Government cost of combination care (intensive informal + formal home services).

    Combination care (Kombinationsleistungen) applies when:
      - lagged_choice in INTENSIVE_INFORMAL_CARE (choices 12-15)
      - This implies care_demand == intensive (verified by model choice structure)
    A fraction IN_KIND_BENEFITS_UTILIZATION_RATE of these agents also use formal
    home care services on top of their Pflegegeld. The government cost per
    combination-care period is COMBINATION_CARE_COST_INTENSIVE_2010 * 12.

    Returns dict with: total_cost, n_eligible_agents, n_expected_combo_periods,
    avg_combo_years, avg_cost_per_eligible.
    """
    result = {
        "total_cost": 0.0,
        "n_eligible_agents": 0,
        "n_expected_combo_periods": 0.0,
        "avg_combo_years": np.nan,
        "avg_cost_per_eligible": np.nan,
    }
    if "lagged_choice" not in df.columns:
        return result
    intensive_choices = np.asarray(INTENSIVE_INFORMAL_CARE).ravel()
    eligible_mask = df["lagged_choice"].isin(intensive_choices)
    n_eligible_periods = int(eligible_mask.sum())
    if n_eligible_periods == 0:
        return result

    n_eligible_agents = int(df.loc[eligible_mask, "agent"].nunique())
    n_expected_combo_periods = n_eligible_periods * IN_KIND_BENEFITS_UTILIZATION_RATE
    total_cost = n_expected_combo_periods * GOV_ANNUAL_COMBINATION_CARE_COST

    avg_combo_years = n_expected_combo_periods / n_eligible_agents
    avg_cost_per_eligible = total_cost / n_eligible_agents

    n_expected_combo_users = n_eligible_agents * IN_KIND_BENEFITS_UTILIZATION_RATE
    avg_combo_years_per_user = (
        n_expected_combo_periods / n_expected_combo_users
        if n_expected_combo_users > 0
        else np.nan
    )
    avg_cost_per_combo_user = (
        total_cost / n_expected_combo_users if n_expected_combo_users > 0 else np.nan
    )

    result["total_cost"] = float(total_cost)
    result["n_eligible_agents"] = n_eligible_agents
    result["n_expected_combo_users"] = float(n_expected_combo_users)
    result["n_expected_combo_periods"] = float(n_expected_combo_periods)
    result["avg_combo_years"] = float(avg_combo_years)
    result["avg_cost_per_eligible"] = float(avg_cost_per_eligible)
    result["avg_combo_years_per_user"] = float(avg_combo_years_per_user)
    result["avg_cost_per_combo_user"] = float(avg_cost_per_combo_user)
    return result


def _avg_outcomes_per_caregiver(
    df: pd.DataFrame,
    wealth_unit: float,
    outcome_columns: list[str],
) -> dict[str, float]:
    """Average of each outcome (in currency) per unique caregiver.

    Sums each outcome over rows where lagged_choice in INFORMAL_CARE,
    multiplies by wealth_unit, divides by n_caregivers (choice in INFORMAL_CARE).
    Missing columns yield np.nan for that outcome.
    Works for both baseline and leave policies (unified).
    """
    result = {col: np.nan for col in outcome_columns}
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
