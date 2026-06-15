"""Per-(agent, period) fiscal ledger for the self-financing heterogeneity tables.

Loads the seven policy pickles produced by ``solve_and_simulate`` and computes
per-(agent, period) fiscal components (A within-spell tax, decomposed SSC pieces,
C unemployment transfers paid, D imputed FC cost, E gov pension paid,
F Pflegegeld outlay,
F-tilde imputed Pflegegeld regardless of policy rules, G leave top-up, and
partner-only flows for the exogeneity assertion). All flows are returned in
EUR (already scaled by ``wealth_unit``) as positive magnitudes (revenue or
expenditure on the gov side; the writer assigns signs when forming
deltas/savings).

This module is consumed by ``task_self_financing_heterogeneity.py``. It is
deliberately Phase-0-aligned: the same column names, mask conventions and
SSC re-derivations as ``scripts/check_phase0_self_financing.py``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.model.shared import (
    CARE_DEMAND_INTENSIVE,
    CARE_DEMAND_LIGHT,
    FORMAL_CARE,
    INFORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    PARTNER_RETIRED,
    PARTNER_WORKING,
    RETIREMENT,
)
from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_health_ltc_contr,
    calc_pension_unempl_contr,
)
from caregiving.tables.publication.task_fiscal import (
    GOV_ANNUAL_FC_COST_INTENSIVE,
    GOV_ANNUAL_FC_COST_LIGHT,
)

R_GOV_ANNUAL: float = 0.01
DISCOUNT: float = 1.0 / (1.0 + R_GOV_ANNUAL)

PENSION_FRAC: float = 0.0995 / 0.1135
UE_FRAC: float = 0.0140 / 0.1135
LTC_FRAC: float = 0.0195 / 0.0940
HEALTH_FRAC: float = 0.0745 / 0.0940

WORK_CODES = (2, 3, 6, 7, 10, 11, 14, 15)

POLICY_KEYS: tuple[str, ...] = (
    "baseline",
    "full_beirat",
    "full_beirat_no_total_cap",
    "partial_beirat",
    "normal_leave",
    "norwegian_pg",
    "norwegian_no_pg",
)

POLICY_LABELS_SHORT: dict[str, str] = {
    "baseline": "Baseline",
    "full_beirat": "Reform",
    "full_beirat_no_total_cap": "Reform (no cap)",
    "partial_beirat": "Reform (PT)",
    "normal_leave": "Uncapped",
    "norwegian_pg": "Full Wage",
    "norwegian_no_pg": "Full Wage no-CA",
}

HAS_PFLEGEGELD: dict[str, bool] = {
    "baseline": True,
    "full_beirat": True,
    "full_beirat_no_total_cap": True,
    "partial_beirat": True,
    "normal_leave": False,
    "norwegian_pg": True,
    "norwegian_no_pg": False,
}

HAS_NORWEGIAN_TAX_COL: dict[str, bool] = {
    "baseline": False,
    "full_beirat": False,
    "full_beirat_no_total_cap": False,
    "partial_beirat": False,
    "normal_leave": False,
    "norwegian_pg": True,
    "norwegian_no_pg": True,
}

PATHS_BY_KEY: dict[str, Path] = {
    "baseline": BLD / "solve_and_simulate" / "simulated_data_estimated_params.pkl",
    "full_beirat": BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_estimated_params.pkl",
    "full_beirat_no_total_cap": BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_no_total_cap_estimated_params.pkl",
    "partial_beirat": BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_beirat_estimated_params.pkl",
    "normal_leave": BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    "norwegian_pg": BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    "norwegian_no_pg": BLD
    / "solve_and_simulate"
    / (
        "simulated_data_full_caregiving_leave_with_job_retention_"
        "estimated_params_no_pflegegeld.pkl"
    ),
}

PATH_TO_SPECS: Path = BLD / "model" / "specs" / "specs_full.pkl"

AGE_BIN_EDGES: tuple[tuple[int, int, str], ...] = (
    (40, 49, "40-49"),
    (50, 59, "50-59"),
    (60, 69, "60-69"),
)
NEVER_LABEL: str = "never"
EDU_LABELS: dict[int, str] = {0: "low", 1: "high"}

LEDGER_KEEP_COLUMNS: tuple[str, ...] = (
    "agent",
    "period",
    "education",
    "lagged_choice",
    "choice",
    "care_demand",
    "partner_state",
    "gross_partner_income",
    "gross_partner_pension",
    "income_tax",
    "own_ssc",
    "partner_ssc",
    "total_tax_revenue",
    "gross_labor_income",
    "gross_retirement_income",
    "unemployment_transfer_paid",
)
LEDGER_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "caregiving_leave_top_up",
    "tax_increase_from_progression",
    "tax_attributable_to_full_leave",
)


@dataclass
class LedgerInputs:
    """Container for inputs needed to construct a per-(a, t) ledger."""

    wealth_unit: float
    start_age: int
    pflegegeld_light_eur_year: float
    pflegegeld_intensive_eur_year: float


def load_specs(path: Path = PATH_TO_SPECS) -> LedgerInputs:
    """Load model specs and return ledger inputs."""
    with path.open("rb") as f:
        specs = pickle.load(f)
    wealth_unit = float(specs["wealth_unit"])
    start_age = int(specs.get("start_age", 30))
    light_month = float(specs["informal_care_cash_benefits_light"])
    intensive_month = float(specs["informal_care_cash_benefits_intensive"])
    return LedgerInputs(
        wealth_unit=wealth_unit,
        start_age=start_age,
        pflegegeld_light_eur_year=light_month * 12.0,
        pflegegeld_intensive_eur_year=intensive_month * 12.0,
    )


def load_slim(path: Path) -> pd.DataFrame:
    """Load a policy pickle and keep only the columns needed for the ledger.

    Backfills the optional columns that vary across pickles so downstream
    consumers can assume a uniform schema:

    - ``caregiving_leave_top_up`` (missing in baseline) -> 0.
    - ``tax_increase_from_progression`` (missing for the Norwegian and
      baseline pickles) -> 0.
    - ``tax_attributable_to_full_leave`` (only in Norwegian pickles) -> 0.

    Adds ``lagged_care_demand`` (per-agent shift of ``care_demand``); used
    by the imputed government formal-care cost.
    """
    df = pd.read_pickle(path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    keep = [c for c in LEDGER_KEEP_COLUMNS if c in df.columns]
    optional = [c for c in LEDGER_OPTIONAL_COLUMNS if c in df.columns]
    sub = df[keep + optional].copy()
    if "caregiving_leave_top_up" not in sub.columns:
        sub["caregiving_leave_top_up"] = 0.0
    if "tax_increase_from_progression" not in sub.columns:
        sub["tax_increase_from_progression"] = 0.0
    if "tax_attributable_to_full_leave" not in sub.columns:
        sub["tax_attributable_to_full_leave"] = 0.0
    sub = sub.sort_values(["agent", "period"]).reset_index(drop=True)
    sub["lagged_care_demand"] = sub.groupby("agent")["care_demand"].shift(1)
    return sub


def _bin_age(age: float) -> str:
    if pd.isna(age):
        return NEVER_LABEL
    for lo, hi, label in AGE_BIN_EDGES:
        if lo <= age <= hi:
            return label
    return NEVER_LABEL


def build_subgroup_table(baseline_df: pd.DataFrame, start_age: int) -> pd.DataFrame:
    """Per-agent education + first-care-demand age-bin labels.

    Built ONCE from baseline (care_demand is policy-invariant; verified by
    Phase 0 Section D). Returned DataFrame is indexed by agent.
    """
    edu_by_agent = baseline_df.groupby("agent")["education"].first()

    care_demand = baseline_df[baseline_df["care_demand"] > 0]
    if care_demand.empty:
        first_period = pd.Series(dtype=float, name="period")
    else:
        first_period = care_demand.groupby("agent")["period"].min()

    first_demand_age = first_period.add(start_age).reindex(edu_by_agent.index)
    age_bins = [_bin_age(a) for a in first_demand_age.values]
    edu_arr = edu_by_agent.values

    sg = pd.DataFrame(
        {
            "education": edu_arr.astype(int),
            "education_label": [EDU_LABELS.get(int(e), str(int(e))) for e in edu_arr],
            "first_demand_age": first_demand_age.values,
            "age_bin": age_bins,
            "ever_caregiver": [b != NEVER_LABEL for b in age_bins],
        },
        index=edu_by_agent.index,
    )
    return sg


def _ssc_pieces_eur(
    df: pd.DataFrame, wealth_unit: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-derive SSC bundles per (a, t) in EUR, NaN-sanitised.

    Returns four arrays (length = len(df), EUR/year):

    - ``pen_ue_own``: own pension+UE bundle (workers only).  # codespell:ignore
    - ``pen_ue_par``: partner pension+UE bundle  # codespell:ignore
      (working partner only).
    - ``hl_own``: own health+LTC bundle (workers + retirees on retirement
      income).
    - ``hl_par``: partner health+LTC bundle (working + retired partner).

    Caller splits each bundle into the two constituent rates using
    ``PENSION_FRAC``/``UE_FRAC`` and ``LTC_FRAC``/``HEALTH_FRAC``.
    """
    gl_eur = np.nan_to_num(df["gross_labor_income"].values, nan=0.0) * wealth_unit
    gr_eur = np.nan_to_num(df["gross_retirement_income"].values, nan=0.0) * wealth_unit
    gp_w_eur = np.nan_to_num(df["gross_partner_income"].values, nan=0.0) * wealth_unit
    gp_r_eur = np.nan_to_num(df["gross_partner_pension"].values, nan=0.0) * wealth_unit

    work_codes = np.asarray(WORK_CODES)
    retire_codes = np.asarray(RETIREMENT).ravel()
    lag = df["lagged_choice"].values

    is_worker_lag = np.isin(lag, work_codes)
    is_retired_lag = np.isin(lag, retire_codes)

    partner_state = df["partner_state"].values
    is_partner_working = partner_state == int(PARTNER_WORKING)
    is_partner_retired = partner_state == int(PARTNER_RETIRED)

    pen_ue_own = np.where(
        is_worker_lag, np.asarray(calc_pension_unempl_contr(gl_eur)), 0.0
    )
    pen_ue_par = np.where(
        is_partner_working, np.asarray(calc_pension_unempl_contr(gp_w_eur)), 0.0
    )

    hl_own = np.where(
        is_worker_lag, np.asarray(calc_health_ltc_contr(gl_eur)), 0.0
    ) + np.where(is_retired_lag, np.asarray(calc_health_ltc_contr(gr_eur)), 0.0)
    hl_par = np.where(
        is_partner_working, np.asarray(calc_health_ltc_contr(gp_w_eur)), 0.0
    ) + np.where(is_partner_retired, np.asarray(calc_health_ltc_contr(gp_r_eur)), 0.0)

    return pen_ue_own, pen_ue_par, hl_own, hl_par


def build_ledger(  # noqa: PLR0915
    df: pd.DataFrame,
    inputs: LedgerInputs,
    policy_key: str,
    sg: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the per-(agent, period) fiscal ledger for one policy.

    All EUR magnitudes are stored as positive numbers (raw revenue or
    expenditure on the government side). The aggregator and writer assign
    signs when forming deltas/savings.

    Returns a DataFrame with columns:

    - keys: ``agent``, ``period``, ``policy``;
    - state: ``lagged_choice``, ``partner_state``, ``care_demand``;
    - subgroup labels (broadcast from ``sg``): ``education_label``,
      ``age_bin``, ``ever_caregiver``;
    - revenue side (EUR, positive): ``A_at_eur``, ``B_income_tax_at_eur``,
      ``B_pension_ssc_at_eur``, ``B_ue_ssc_at_eur``, ``B_ltc_ssc_at_eur``,
      ``B_health_ssc_at_eur``;
    - expenditure side (EUR, positive magnitude): ``C_at_eur``,
      ``D_at_eur``, ``E_at_eur``, ``F_at_eur``, ``G_at_eur``;
    - discount factor: ``discount_factor`` = ``DISCOUNT ** period``;
    - perspective masks: ``mask_care_demand``, ``mask_active_caregiving``,
      ``mask_all`` (all booleans).
    """
    wealth_unit = inputs.wealth_unit

    lag = df["lagged_choice"].values
    informal_codes = np.asarray(INFORMAL_CARE).ravel()
    light_codes = np.asarray(LIGHT_INFORMAL_CARE).ravel()
    intensive_codes = np.asarray(INTENSIVE_INFORMAL_CARE).ravel()
    formal_codes = np.asarray(FORMAL_CARE).ravel()
    retire_codes = np.asarray(RETIREMENT).ravel()

    is_informal_lag = np.isin(lag, informal_codes)
    is_light_lag = np.isin(lag, light_codes)
    is_intensive_lag = np.isin(lag, intensive_codes)
    is_formal_lag = np.isin(lag, formal_codes)
    is_retired_lag = np.isin(lag, retire_codes)

    partner_state = df["partner_state"].values
    is_partner_retired = partner_state == int(PARTNER_RETIRED)

    pen_ue_own, pen_ue_par, hl_own, hl_par = _ssc_pieces_eur(df, wealth_unit)

    pension_at = PENSION_FRAC * (pen_ue_own + pen_ue_par)
    ue_ssc_at = UE_FRAC * (pen_ue_own + pen_ue_par)
    ltc_at = LTC_FRAC * (hl_own + hl_par)
    health_at = HEALTH_FRAC * (hl_own + hl_par)

    # Partner-only flows, kept separately for the policy-invariance assertion
    # in the aggregator (partner behaviour is exogenous; totals must match
    # across policies up to tolerance).
    partner_ssc_at = pen_ue_par + hl_par

    income_tax_eur = np.nan_to_num(df["income_tax"].values, nan=0.0) * wealth_unit

    if HAS_NORWEGIAN_TAX_COL[policy_key]:
        a_col = df["tax_attributable_to_full_leave"].values
    else:
        a_col = df["tax_increase_from_progression"].values
    A_at_eur = np.nan_to_num(a_col, nan=0.0) * wealth_unit

    C_at_eur = (
        np.nan_to_num(df["unemployment_transfer_paid"].values, nan=0.0) * wealth_unit
    )

    lag_dem = df["lagged_care_demand"].values
    # Loud guard: every formal-care-lagged row must have a known lagged care
    # demand (light/intensive). Verified empirically to hold (0 violations);
    # if a future model change routes rows through the midpoint fallback,
    # fail instead of silently averaging.
    n_fallback = int(
        np.count_nonzero(
            is_formal_lag
            & ~np.isin(lag_dem, [int(CARE_DEMAND_LIGHT), int(CARE_DEMAND_INTENSIVE)])
        )
    )
    if n_fallback > 0:
        raise RuntimeError(
            f"[ledger:{policy_key}] {n_fallback} formal-care-lagged rows have "
            "lagged care_demand outside {light, intensive}. The imputed "
            "formal-care cost (D) would silently use the midpoint fallback. "
            "Investigate the simulation output before building SF tables."
        )
    fc_cost_per_period = np.where(
        lag_dem == int(CARE_DEMAND_LIGHT),
        GOV_ANNUAL_FC_COST_LIGHT,
        np.where(
            lag_dem == int(CARE_DEMAND_INTENSIVE),
            GOV_ANNUAL_FC_COST_INTENSIVE,
            (GOV_ANNUAL_FC_COST_LIGHT + GOV_ANNUAL_FC_COST_INTENSIVE) / 2.0,
        ),
    )
    D_at_eur = is_formal_lag.astype(float) * fc_cost_per_period

    gr_eur = np.nan_to_num(df["gross_retirement_income"].values, nan=0.0) * wealth_unit
    gp_r_eur = np.nan_to_num(df["gross_partner_pension"].values, nan=0.0) * wealth_unit
    partner_pension_at = gp_r_eur * is_partner_retired.astype(float)
    E_at_eur = gr_eur * is_retired_lag.astype(float) + partner_pension_at

    # F-tilde: Pflegegeld imputed from lagged choices for EVERY policy,
    # regardless of whether the policy actually pays it. Enables the
    # mechanical (rule change) vs behavioral (induced take-up) split of the
    # Pflegegeld saving for the PG-abolishing variants.
    Ftilde_at_eur = (
        is_light_lag.astype(float) * inputs.pflegegeld_light_eur_year
        + is_intensive_lag.astype(float) * inputs.pflegegeld_intensive_eur_year
    )
    if HAS_PFLEGEGELD[policy_key]:
        F_at_eur = Ftilde_at_eur
    else:
        F_at_eur = np.zeros(len(df))

    G_at_eur = (
        np.nan_to_num(df["caregiving_leave_top_up"].values, nan=0.0) * wealth_unit
    )

    period = df["period"].values.astype(int)
    discount_factor = np.power(DISCOUNT, period)

    mask_care_demand = df["care_demand"].values > 0
    mask_active_caregiving = is_informal_lag
    mask_all = np.ones(len(df), dtype=bool)

    sg_view = sg.reindex(df["agent"].values)
    edu_label = sg_view["education_label"].values
    age_bin = sg_view["age_bin"].values
    ever_cg = sg_view["ever_caregiver"].values

    return pd.DataFrame(
        {
            "agent": df["agent"].values,
            "period": period,
            "policy": policy_key,
            "lagged_choice": lag,
            "choice": df["choice"].values,
            "partner_state": partner_state,
            "care_demand": df["care_demand"].values,
            "education_label": edu_label,
            "age_bin": age_bin,
            "ever_caregiver": ever_cg,
            "A_at_eur": A_at_eur,
            "B_income_tax_at_eur": income_tax_eur,
            "B_pension_ssc_at_eur": pension_at,
            "B_ue_ssc_at_eur": ue_ssc_at,
            "B_ltc_ssc_at_eur": ltc_at,
            "B_health_ssc_at_eur": health_at,
            "C_at_eur": C_at_eur,
            "D_at_eur": D_at_eur,
            "E_at_eur": E_at_eur,
            "F_at_eur": F_at_eur,
            "Ftilde_at_eur": Ftilde_at_eur,
            "G_at_eur": G_at_eur,
            "partner_ssc_at_eur": partner_ssc_at,
            "partner_pension_at_eur": partner_pension_at,
            # Direct model column (income tax + own SSC + partner SSC), used
            # by Config (v) to replicate the legacy "Avg. total tax revenue"
            # exactly instead of via the re-derived SSC decomposition.
            "T_total_at_eur": np.nan_to_num(df["total_tax_revenue"].values, nan=0.0)
            * wealth_unit,
            "discount_factor": discount_factor,
            "mask_care_demand": mask_care_demand,
            "mask_active_caregiving": mask_active_caregiving,
            "mask_all": mask_all,
        }
    )


def build_all_ledgers(
    inputs: LedgerInputs,
    paths: dict[str, Path] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all policy pickles and return the per-policy ledgers + subgroup table.

    The subgroup table is built from the baseline ledger only (care_demand
    is policy-invariant; verified in Phase 0 Section D). The same table is
    broadcast to all policies so that subgroup membership is uniform.

    Memory: pickles are ~3.9 GB each and there are seven policies. Each slim
    frame is loaded, converted to a ledger, and freed immediately before the
    next pickle is touched (this machine has OOM history with these files).
    """
    import gc

    paths = paths if paths is not None else PATHS_BY_KEY

    baseline_slim = load_slim(paths["baseline"])
    sg = build_subgroup_table(baseline_slim, inputs.start_age)

    ledgers: dict[str, pd.DataFrame] = {}
    for key in POLICY_KEYS:
        slim = baseline_slim if key == "baseline" else load_slim(paths[key])
        ledgers[key] = build_ledger(slim, inputs, key, sg)
        del slim
        if key == "baseline":
            del baseline_slim
        gc.collect()
    return ledgers, sg
