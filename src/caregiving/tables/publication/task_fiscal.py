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


# Policy scenario order: baseline, full Beirat, partial Beirat, normal leave, Norwegian with PG, Norwegian no PG.
FISCAL_POLICY_LABELS = [
    "Baseline (cash benefits)",
    r"Full Beirat (65\%, 1y full)",
    r"Partial Beirat (65\%)",
    r"Normal leave (65\%, no cap)",
    r"Norwegian (100\%, with PG)",
    r"Norwegian (100\%, no PG)",
]


@pytask.mark.tables
@pytask.mark.fiscal_costs
@pytask.mark.publication
def task_create_fiscal_costs(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_full_beirat_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_estimated_params.pkl",
    path_to_partial_beirat_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_beirat_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_norwegian_pg_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_norwegian_no_pg_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params_no_pflegegeld.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "fiscal_costs_caregiving_policies.tex",
) -> None:
    """Create LaTeX table of fiscal costs of caregiving policies (six scenarios).

    Policies: (1) Baseline (cash benefits); (2) Full Beirat; (3) Partial Beirat;
    (4) Normal leave (65%, no cap); (5) Norwegian leave with Pflegegeld;
    (6) Norwegian leave without Pflegegeld.
    Each leave scenario is compared to baseline via delta columns.

    Life-cycle scope: The fiscal comparison covers the full model life cycle
    (ages start_age to end_age).
    """
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = float(specs["wealth_unit"])
    start_age = int(specs.get("start_age", 30))
    end_age = int(specs.get("end_age", 100))

    # Load slim DataFrames: baseline + 5 leave scenarios
    baseline_df = _load_sim_df(path_to_baseline_sim)
    full_beirat_df = _load_sim_df(path_to_full_beirat_sim)
    partial_beirat_df = _load_sim_df(path_to_partial_beirat_sim)
    normal_leave_df = _load_sim_df(path_to_normal_leave_sim)
    norwegian_pg_df = _load_sim_df(path_to_norwegian_pg_sim)
    norwegian_no_pg_df = _load_sim_df(path_to_norwegian_no_pg_sim)
    policy_dfs = [
        baseline_df,
        full_beirat_df,
        partial_beirat_df,
        normal_leave_df,
        norwegian_pg_df,
        norwegian_no_pg_df,
    ]

    # Ensure age column exists
    for df in policy_dfs:
        if "age" not in df.columns and "period" in df.columns:
            df["age"] = start_age + df["period"]

    # Verify period and age
    for i, (label, df) in enumerate(zip(FISCAL_POLICY_LABELS, policy_dfs, strict=True)):
        if "period" in df.columns and "age" in df.columns:
            assert (
                df["age"] == start_age + df["period"]
            ).all(), f"{label}: age != start_age + period"

    # Compute lagged_care_demand
    for i, df in enumerate(policy_dfs):
        policy_dfs[i] = _add_lagged_care_demand(df)

    (
        baseline_df,
        full_beirat_df,
        partial_beirat_df,
        normal_leave_df,
        norwegian_pg_df,
        norwegian_no_pg_df,
    ) = policy_dfs

    def _safe_div(num, den):
        return num / den if den else np.nan

    # --- Costs and caregiver counts (per scenario) ---
    cost_baseline, n_baseline, periods_baseline = _total_cost_baseline(
        baseline_df, wealth_unit
    )
    cost_fb, n_fb, periods_fb = _total_cost_leave(full_beirat_df, specs)
    cost_pb, n_pb, periods_pb = _total_cost_leave(partial_beirat_df, specs)
    cost_nl, n_nl, periods_nl = _total_cost_leave(normal_leave_df, specs)
    cost_npg, n_npg, periods_npg = _total_cost_leave(norwegian_pg_df, specs)
    cost_nnpg, n_nnpg, periods_nnpg = _total_cost_leave(norwegian_no_pg_df, specs)
    costs = [cost_baseline, cost_fb, cost_pb, cost_nl, cost_npg, cost_nnpg]
    n_cgs = [n_baseline, n_fb, n_pb, n_nl, n_npg, n_nnpg]
    periods_list = [
        periods_baseline,
        periods_fb,
        periods_pb,
        periods_nl,
        periods_npg,
        periods_nnpg,
    ]

    avg_costs = [c / n if n else np.nan for c, n in zip(costs, n_cgs, strict=True)]
    avg_years_list = [
        p / n if n else np.nan for p, n in zip(periods_list, n_cgs, strict=True)
    ]

    def _avg_monthly_per_caregiving_month(avg_cost, avg_years):
        if np.isnan(avg_years) or avg_years <= 0:
            return np.nan
        return avg_cost / 12.0 / avg_years

    avg_monthly = [
        _avg_monthly_per_caregiving_month(ac, ay)
        for ac, ay in zip(avg_costs, avg_years_list, strict=True)
    ]

    # --- N total agents, share ever caregiving, per-capita cost ---
    n_totals = [_n_total_agents(df) for df in policy_dfs]
    share_cg = [
        _safe_div(n_cg, n_tot) for n_cg, n_tot in zip(n_cgs, n_totals, strict=True)
    ]
    percap = [_safe_div(c, n_tot) for c, n_tot in zip(costs, n_totals, strict=True)]

    # --- Total gross benefit ---
    gross_benefit_baseline = cost_baseline
    gross_benefits_leave = [
        _total_sum_column(df, "caregiving_leave_top_up", wealth_unit)
        for df in (
            full_beirat_df,
            partial_beirat_df,
            normal_leave_df,
            norwegian_pg_df,
            norwegian_no_pg_df,
        )
    ]
    gross_benefits = [gross_benefit_baseline] + gross_benefits_leave

    # --- Total net cost from model aux ---
    net_cost_aux_baseline = cost_baseline
    net_cost_aux_leave = [
        _total_leave_net_cost_aux(df, wealth_unit)
        for df in (
            full_beirat_df,
            partial_beirat_df,
            normal_leave_df,
            norwegian_pg_df,
            norwegian_no_pg_df,
        )
    ]
    net_costs_aux = [net_cost_aux_baseline] + net_cost_aux_leave

    # --- Government formal care stats ---
    fc_stats = [_formal_care_stats(df) for df in policy_dfs]
    gov_fc_list = [s[0] for s in fc_stats]
    n_fc_list = [s[1] for s in fc_stats]
    avg_fc_yrs_list = [s[2] for s in fc_stats]

    # --- Combination care ---
    combo_list = [_combination_care_stats(df) for df in policy_dfs]
    gov_total_care_list = [
        gov_fc + combo["total_cost"]
        for gov_fc, combo in zip(gov_fc_list, combo_list, strict=True)
    ]

    # --- Avg outcomes per caregiver ---
    outcomes_list = [
        _avg_outcomes_per_caregiver(df, wealth_unit, OUTCOME_COLUMNS_AVG_PER_CAREGIVER)
        for df in policy_dfs
    ]

    # =====================================================================
    # Build table (each key -> list of 6 values)
    # =====================================================================
    table_dict: dict[str, list] = {
        "Policy": list(FISCAL_POLICY_LABELS),
        "Total cost (net)": costs,
        "Total gross benefit": gross_benefits,
        "Total net cost (model)": net_costs_aux,
        "Gov. formal care cost (total)": gov_fc_list,
        "Avg. gov. formal care cost per caregiver": [
            _safe_div(gov_fc, n_cg)
            for gov_fc, n_cg in zip(gov_fc_list, n_cgs, strict=True)
        ],
        "Avg. gov. formal care cost per FC user": [
            _safe_div(gov_fc, n_fc)
            for gov_fc, n_fc in zip(gov_fc_list, n_fc_list, strict=True)
        ],
        "Avg. gov. FC cost per CG (pct of baseline)": [
            np.nan,
            *[
                _safe_div(
                    _safe_div(gov_fc, n_cg),
                    _safe_div(gov_fc_list[0], n_cgs[0]),
                )
                * 100
                for gov_fc, n_cg in zip(gov_fc_list[1:], n_cgs[1:], strict=True)
            ],
        ],
        "Avg. gov. FC cost per FC user (pct of baseline)": [
            np.nan,
            *[
                _safe_div(
                    _safe_div(gov_fc, n_fc),
                    _safe_div(gov_fc_list[0], n_fc_list[0]),
                )
                * 100
                for gov_fc, n_fc in zip(gov_fc_list[1:], n_fc_list[1:], strict=True)
            ],
        ],
        "Gov. formal care cost per capita": [
            _safe_div(gov_fc, n_tot)
            for gov_fc, n_tot in zip(gov_fc_list, n_totals, strict=True)
        ],
        "Avg. formal care years": avg_fc_yrs_list,
        "N formal care users": n_fc_list,
        "Gov. combination care cost (total)": [c["total_cost"] for c in combo_list],
        "Avg. gov. combo care cost per caregiver": [
            _safe_div(c["total_cost"], n_cg)
            for c, n_cg in zip(combo_list, n_cgs, strict=True)
        ],
        "Avg. gov. combo care cost per eligible": [
            c["avg_cost_per_eligible"] for c in combo_list
        ],
        "Avg. gov. combo care cost per combo user": [
            c["avg_cost_per_combo_user"] for c in combo_list
        ],
        "Avg. combination care years (eligible)": [
            c["avg_combo_years"] for c in combo_list
        ],
        "Avg. combination care years (per user)": [
            c["avg_combo_years_per_user"] for c in combo_list
        ],
        "N eligible for combo care": [c["n_eligible_agents"] for c in combo_list],
        "N combo care users (expected)": [
            c["n_expected_combo_users"] for c in combo_list
        ],
        "Gov. total care cost": gov_total_care_list,
        "Gov. total care cost per caregiver": [
            _safe_div(gtc, n_cg)
            for gtc, n_cg in zip(gov_total_care_list, n_cgs, strict=True)
        ],
        "Gov. total care cost per capita": [
            _safe_div(gtc, n_tot)
            for gtc, n_tot in zip(gov_total_care_list, n_totals, strict=True)
        ],
        "N caregivers": n_cgs,
        "N total agents": n_totals,
        "Share ever caregiving": share_cg,
        "Avg. caregiving years": avg_years_list,
        "Avg cost per caregiver": avg_costs,
        "Per-capita cost (all agents)": percap,
        "Avg. monthly cost per CG month": avg_monthly,
    }
    for col, label in zip(
        OUTCOME_COLUMNS_AVG_PER_CAREGIVER, OUTCOME_COLUMN_LABELS, strict=True
    ):
        table_dict[label] = [outcomes_list[j].get(col, np.nan) for j in range(len(FISCAL_POLICY_LABELS))]

    table_wide = pd.DataFrame(table_dict)
    table = table_wide.set_index("Policy").T
    table.index.name = "Metric"

    # Delta columns: each leave scenario minus baseline
    baseline_col = table.columns[0]
    delta_labels = [
        "Delta Full Beirat - Baseline",
        "Delta Partial Beirat - Baseline",
        "Delta Normal leave - Baseline",
        "Delta Norwegian (PG) - Baseline",
        "Delta Norwegian (no PG) - Baseline",
    ]
    for leave_col, delta_label in zip(table.columns[1:], delta_labels, strict=True):
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


def _total_leave_net_cost_aux(df: pd.DataFrame, wealth_unit: float) -> float:
    """Total net cost from model aux: full_leave_net_cost if present, else normal_leave_net_cost."""
    for col in ("full_leave_net_cost", "normal_leave_net_cost"):
        val = _total_sum_column(df, col, wealth_unit)
        if val == val:  # false for NaN
            return float(val)
    return np.nan


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
