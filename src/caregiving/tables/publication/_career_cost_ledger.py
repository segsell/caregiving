"""Per-agent career-cost NPVs for the career-cost heterogeneity tables.

Career cost (Adda-style) of caregiving = ``1 - NPV_no_care / NPV_policy`` (negative
= cost): the fractional drop in a lifetime-income NPV that an agent suffers because
the parent needs care, relative to a world where the parent never needs care
(``simulated_data_no_care_demand.pkl``, policy-invariant).

Two income measures (see the budget equation for column definitions):

- ``labpen`` (labor + pension): ``own_income_after_ssc - caregiving_leave_top_up
  + bequest_from_parent``. Individual own earnings/pension net of SSC (gross of
  income tax), plus parental bequest -- this matches the EXISTING headline
  computation (the -37.1% baseline is "labor+pension, incl. bequest"). Note: the
  bequest is not strictly "labor or pension"; it is kept only to reproduce the
  established headline. NOT the same as "own income" (which keeps the leave top-up);
  they coincide only for the baseline.
- ``full`` (disposable, model-consistent): ``hh_net_income_wo_interest +
  care_benefits_and_costs + bequest_from_parent``. Household disposable income
  excluding capital interest -- already after income tax, already applies the
  unemployment safety-net floor via ``max(.)``, already includes child benefits.
  This REPLACES the previous buggy full-income measure (which additively added the
  means-tested unemployment *floor* ``household_unemployment_benefits`` -- computed
  for every agent, not what is actually received -- on top of own income).

Discounting: ``beta ** (age - NPV_START_AGE)`` over ages
``[NPV_START_AGE, NPV_END_AGE]``; income floored at 0; dead periods contribute 0.
A small beta sweep is reported (primary = the model's discount factor).

Consumed by ``task_career_costs_heterogeneity.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.model.shared import DEAD, INFORMAL_CARE, NPV_END_AGE, NPV_START_AGE

# Beta sweep for the agent-side NPV. Primary = 0.97, the established career-cost
# convention that reproduces the existing headline (the user asked to keep the
# current headline computation); 0.98 is the model's own discount factor and 0.95 a
# lower-bound sensitivity, both reported in the sweep.
BETAS: dict[str, float] = {"b098": 0.98, "b097": 0.97, "b095": 0.95}
PRIMARY_BETA: str = "b097"
LEGACY_BETA: str = "b097"
BETA_LABELS: dict[str, str] = {
    "b098": "0.98 (model)",
    "b097": "0.97",
    "b095": "0.95",
}

MEASURES: tuple[str, ...] = ("labpen", "full")
MEASURE_LABELS: dict[str, str] = {
    "labpen": "Labor + pension",
    "full": "Full disposable income",
}

# Policy keys mirror the SF pipeline; ``no_care_demand`` is the shared reference.
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

_SS = BLD / "solve_and_simulate"
PATHS_BY_KEY: dict[str, Path] = {
    "baseline": _SS / "simulated_data_estimated_params.pkl",
    "full_beirat": _SS
    / "simulated_data_caregiving_leave_full_beirat_estimated_params.pkl",
    "full_beirat_no_total_cap": _SS
    / "simulated_data_caregiving_leave_full_beirat_no_total_cap_estimated_params.pkl",
    "partial_beirat": _SS
    / "simulated_data_caregiving_leave_beirat_estimated_params.pkl",
    "normal_leave": _SS
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    "norwegian_pg": _SS
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    "norwegian_no_pg": _SS
    / (
        "simulated_data_full_caregiving_leave_with_job_retention_"
        "estimated_params_no_pflegegeld.pkl"
    ),
}
PATH_NO_CARE_DEMAND: Path = _SS / "simulated_data_no_care_demand.pkl"

# Columns needed for the NPV (a subset, to save RAM). Optional columns are
# backfilled to 0 (absent in baseline / no-care worlds, where they are 0).
_CC_KEEP = ("agent", "age", "health", "choice", "education")
_CC_INCOME_REQUIRED = ("own_income_after_ssc", "hh_net_income_wo_interest")
_CC_INCOME_OPTIONAL = (
    "caregiving_leave_top_up",
    "care_benefits_and_costs",
    "bequest_from_parent",
)


def load_cc_slim(path: Path) -> pd.DataFrame:
    """Load a policy/no-care pickle, keep NPV columns, backfill optional ones to 0."""
    df = pd.read_pickle(path)
    if "agent" not in df.columns:
        df = df.reset_index()
    keep = [c for c in (*_CC_KEEP, *_CC_INCOME_REQUIRED) if c in df.columns]
    optional = [c for c in _CC_INCOME_OPTIONAL if c in df.columns]
    sub = df[keep + optional].copy()
    for col in _CC_INCOME_OPTIONAL:
        if col not in sub.columns:
            sub[col] = 0.0
        else:
            sub[col] = sub[col].fillna(0.0)
    return sub


def _measure_income(df: pd.DataFrame, measure: str) -> np.ndarray:
    """Per-(agent, age) income flow for the requested measure (pre-discounting)."""
    if measure == "labpen":
        income = (
            df["own_income_after_ssc"].to_numpy()
            - df["caregiving_leave_top_up"].to_numpy()
            + df["bequest_from_parent"].to_numpy()
        )
    elif measure == "full":
        income = (
            df["hh_net_income_wo_interest"].to_numpy()
            + df["care_benefits_and_costs"].to_numpy()
            + df["bequest_from_parent"].to_numpy()
        )
    else:  # pragma: no cover - defended by callers
        raise ValueError(f"Unknown measure: {measure}")
    return income


def compute_agent_npvs(
    df: pd.DataFrame,
    *,
    npv_start_age: int = NPV_START_AGE,
    npv_end_age: int = NPV_END_AGE,
) -> pd.DataFrame:
    """Per-agent discounted-income NPVs for every (measure, beta).

    Returns a frame indexed by ``agent`` with columns ``npv_<measure>_<beta>``.
    Income is floored at 0 and dead periods contribute 0 (matches the legacy
    ``compute_career_npv``).
    """
    d = df[(df["age"] >= npv_start_age) & (df["age"] <= npv_end_age)].copy()
    age_offset = d["age"].to_numpy() - npv_start_age
    alive = (d["health"].to_numpy() != int(DEAD)).astype(float)
    agent = d["agent"].to_numpy()

    measure_income = {
        m: np.maximum(_measure_income(d, m), 0.0) * alive for m in MEASURES
    }
    base = pd.DataFrame({"agent": agent})
    for beta_key, beta in BETAS.items():
        discount = np.power(beta, age_offset)
        for measure in MEASURES:
            base[f"npv_{measure}_{beta_key}"] = measure_income[measure] * discount

    value_cols = [c for c in base.columns if c != "agent"]
    npv = base.groupby("agent", sort=True)[value_cols].sum()
    return npv.reset_index()


def ever_caregiver_ids(df: pd.DataFrame) -> np.ndarray:
    """Agent IDs that ever choose informal care in ``df`` (the policy world)."""
    codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    return df.loc[df["choice"].isin(codes), "agent"].unique()
