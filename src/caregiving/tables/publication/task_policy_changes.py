"""Publication: behavioral changes from caregiving policies (LaTeX table).

Compares three policy regimes across twenty-four panels:
  A) Labor supply shares by age group
  B) Caregiving shares by age group (conditional on care demand > 0)
  C) Caregiving shares — Low Education
  D) Caregiving shares — High Education
  E) Labor composition of current informal caregivers
  F) Labor composition of ever-caregivers
  G) Average caregiving benefits / top-ups by labor state while caregiving
  H) Economic outcomes (age < 63)
  I) Retirement outcomes (conditional on being retired)
  J) Experience years at retirement (all agents vs ever-caregivers)
  K) Labor state at first CG year (by age at first CG)
  L) Ever-CG labor supply excl. first CG year
  M) Ever-CG labor by total CG duration (1yr, 2-3yr, 4+yr)
  N) CG labor composition — Low Education
  O) CG labor composition — High Education
  P) CG benefits/top-ups — Low Education
  Q) CG benefits/top-ups — High Education
  R) Leave eligibility by education (job_before_caregiving composition)
  S) Duration heterogeneity — Low Education (condensed)
  T) Duration heterogeneity — High Education (condensed)
  U) Pension/experience at retirement — Low Education
  V) Pension/experience at retirement — High Education
  W) Lifecycle outcomes for ever-CG — Low Education
  X) Lifecycle outcomes for ever-CG — High Education
"""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    FORMAL_CARE,
    FULL_TIME,
    INFORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_CARE_DEMAND,
    PART_TIME,
    RETIREMENT,
    UNEMPLOYED,
)
from caregiving.tables.publication.task_fiscal import IN_KIND_BENEFITS_UTILIZATION_RATE

AGE_GROUPS_LABOR = [
    ("30--39", 30, 39),
    ("40--49", 40, 49),
    ("50--54", 50, 54),
    ("55--59", 55, 59),
    ("50--59", 50, 59),
    ("60--67", 60, 67),
    ("All (30--67)", 30, 67),
]

AGE_GROUPS_CAREGIVING = [
    ("40--49", 40, 49),
    ("50--54", 50, 54),
    ("55--59", 55, 59),
    ("50--59", 50, 59),
    ("60--67", 60, 67),
    ("All (40--67)", 40, 67),
]

_REQUIRED_COLUMNS = [
    "agent",
    "period",
    "choice",
    "lagged_choice",
    "care_demand",
    "education",
    "consumption",
    "savings",
    "savings_dec",
    "assets_begin_of_period",
    "working_hours",
    "gross_labor_income",
    "gross_retirement_income",
    "care_benefits_and_costs",
    "caregiving_leave_top_up",
    "caregiving_type",
    "experience",
    "job_before_caregiving",
]

_CG_FT = [11, 15]
_CG_PT = [10, 14]
_CG_EMPLOYED = [10, 11, 14, 15]
_CG_UNEMPLOYED = [9, 13]
_CG_RETIRED = [8, 12]

_EMPLOYED = np.concatenate(
    [np.asarray(FULL_TIME).ravel(), np.asarray(PART_TIME).ravel()]
)

AGE_GROUPS_BROAD = [
    ("40--49", 40, 49),
    ("50--59", 50, 59),
    ("60--67", 60, 67),
    ("All (40--67)", 40, 67),
]

AGE_GROUPS_CG_HETERO = [
    ("30--39", 30, 39),
    ("40--49", 40, 49),
    ("50--54", 50, 54),
    ("55--59", 55, 59),
    ("50--59", 50, 59),
    ("60--67", 60, 67),
    ("All (30--67)", 30, 67),
]

AGE_GROUPS_BROAD_HETERO = [
    ("30--39", 30, 39),
    ("40--49", 40, 49),
    ("50--59", 50, 59),
    ("55--65", 55, 65),
    ("60--67", 60, 67),
    ("All (30--67)", 30, 67),
]

_RETIRED_AGE_BINS_K = [
    ("63--67", 63, 67),
    ("All (30--67)", 30, 67),
]

CG_DURATION_GROUPS = [
    ("1 CG yr", 1, 1),
    ("2--3 CG yrs", 2, 3),
    ("4+ CG yrs", 4, 9999),
]


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------


def _build_policy_changes_table(
    path_to_specs: Path,
    path_to_baseline_sim: Path,
    path_to_normal_leave_sim: Path,
    path_to_full_leave_sim: Path,
    path_to_save_table: Path,
    table_label: str = "tab:policy_behavioral_changes",
) -> None:
    """Shared logic for building the behavioral-changes LaTeX table."""
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = float(specs["wealth_unit"])
    start_age = int(specs.get("start_age", 30))
    end_age = int(specs.get("end_age", 100))
    end_age_cg = int(specs.get("end_age_caregiving", 70))
    max_exps = np.asarray(specs["max_exps_period_working"])

    baseline_df = _load_sim_df(path_to_baseline_sim)
    normal_df = _load_sim_df(path_to_normal_leave_sim)
    full_df = _load_sim_df(path_to_full_leave_sim)

    for df in (baseline_df, normal_df, full_df):
        if "age" not in df.columns and "period" in df.columns:
            df["age"] = start_age + df["period"]
        _add_cg_metadata(df, end_age_cg)

    panels = [
        ("A: Labor supply", _panel_a_labor, {}),
        ("B: Caregiving", _panel_b_caregiving, {}),
        ("C: Caregiving (Low Edu)", _panel_caregiving_by_edu, {"edu_level": 0}),
        ("D: Caregiving (High Edu)", _panel_caregiving_by_edu, {"edu_level": 1}),
        ("E: CG labor composition", _panel_c_cg_labor, {}),
        ("F: Ever-CG labor composition", _panel_d_ever_cg_labor, {}),
        ("G: CG benefits/top-ups", _panel_e_cg_benefits, {"wealth_unit": wealth_unit}),
        ("H: Economic (< 63)", _panel_f_economic, {"wealth_unit": wealth_unit}),
        ("I: Retirement", _panel_g_retirement, {"wealth_unit": wealth_unit}),
        (
            "J: Experience at retirement",
            _panel_h_experience_at_retirement,
            {"max_exps_period_working": max_exps, "start_age": start_age},
        ),
        ("K: Labor at first CG year", _panel_i_labor_at_first_cg, {}),
        ("L: Ever-CG labor excl. 1st CG yr", _panel_j_ever_cg_excl_first, {}),
        (
            "M: Ever-CG by CG duration",
            _panel_k_ever_cg_by_duration,
            {},
        ),
        ("N: CG labor (Low Edu)", _panel_cg_labor_by_edu, {"edu_level": 0}),
        ("O: CG labor (High Edu)", _panel_cg_labor_by_edu, {"edu_level": 1}),
        (
            "P: CG benefits (Low Edu)",
            _panel_cg_benefits_by_edu,
            {"edu_level": 0, "wealth_unit": wealth_unit},
        ),
        (
            "Q: CG benefits (High Edu)",
            _panel_cg_benefits_by_edu,
            {"edu_level": 1, "wealth_unit": wealth_unit},
        ),
        ("R: Leave eligibility by edu", _panel_leave_eligibility, {}),
        ("S: Duration (Low Edu)", _panel_duration_by_edu, {"edu_level": 0}),
        ("T: Duration (High Edu)", _panel_duration_by_edu, {"edu_level": 1}),
        (
            "U: Pension/exp (Low Edu)",
            _panel_pension_by_edu,
            {
                "edu_level": 0,
                "wealth_unit": wealth_unit,
                "max_exps_period_working": max_exps,
                "start_age": start_age,
            },
        ),
        (
            "V: Pension/exp (High Edu)",
            _panel_pension_by_edu,
            {
                "edu_level": 1,
                "wealth_unit": wealth_unit,
                "max_exps_period_working": max_exps,
                "start_age": start_age,
            },
        ),
        (
            "W: Lifecycle (Low Edu)",
            _panel_lifecycle_by_edu,
            {"edu_level": 0, "wealth_unit": wealth_unit},
        ),
        (
            "X: Lifecycle (High Edu)",
            _panel_lifecycle_by_edu,
            {"edu_level": 1, "wealth_unit": wealth_unit},
        ),
    ]

    all_metrics_baseline = {}
    all_metrics_normal = {}
    all_metrics_full = {}

    for panel_name, panel_fn, extra_kwargs in panels:
        bl = panel_fn(baseline_df, **extra_kwargs)
        nl = panel_fn(normal_df, **extra_kwargs)
        fl = panel_fn(full_df, **extra_kwargs)

        header_key = f"--- {panel_name} ---"
        all_metrics_baseline[header_key] = ""
        all_metrics_normal[header_key] = ""
        all_metrics_full[header_key] = ""

        all_metrics_baseline.update(bl)
        all_metrics_normal.update(nl)
        all_metrics_full.update(fl)

    metrics = list(all_metrics_baseline.keys())
    table_dict = {
        "Metric": metrics,
        "Baseline": [all_metrics_baseline[m] for m in metrics],
        "Normal leave": [all_metrics_normal[m] for m in metrics],
        "Full leave": [all_metrics_full[m] for m in metrics],
    }

    pct_normal = []
    pct_full = []
    for m in metrics:
        bv = all_metrics_baseline[m]
        nv = all_metrics_normal[m]
        fv = all_metrics_full[m]
        if isinstance(bv, str) or isinstance(nv, str):
            pct_normal.append("")
            pct_full.append("")
        else:
            pct_normal.append(_pct_change(bv, nv))
            pct_full.append(_pct_change(bv, fv))

    table_dict["Pct chg Normal"] = pct_normal
    table_dict["Pct chg Full"] = pct_full

    table = pd.DataFrame(table_dict).set_index("Metric")

    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)
    n_data_cols = len(table.columns)
    latex_str = table.to_latex(
        float_format="%.4f",
        column_format="l" + "r" * n_data_cols,
        caption=(
            f"Behavioral changes from caregiving policies (ages {start_age}"
            f"--{end_age})."
        ),
        label=table_label,
        na_rep="--",
    )
    path_to_save_table.write_text(latex_str)


@pytask.mark.tables
@pytask.mark.policy_changes
@pytask.mark.publication
def task_create_policy_changes_table(
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
    / "policy_behavioral_changes.tex",
) -> None:
    """Create LaTeX table of behavioral changes from caregiving policies."""
    _build_policy_changes_table(
        path_to_specs,
        path_to_baseline_sim,
        path_to_normal_leave_sim,
        path_to_full_leave_sim,
        path_to_save_table,
    )


@pytask.mark.tables
@pytask.mark.policy_changes
@pytask.mark.publication
def task_create_policy_changes_table_back_to_jan7(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_back_to_Jan7.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params_back_to_Jan7.pkl",
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params_back_to_Jan7.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "policy_behavioral_changes_back_to_Jan7.tex",
) -> None:
    """Same table as above but using back-to-Jan7 simulation data."""
    _build_policy_changes_table(
        path_to_specs,
        path_to_baseline_sim,
        path_to_normal_leave_sim,
        path_to_full_leave_sim,
        path_to_save_table,
        table_label="tab:policy_behavioral_changes_jan7",
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_sim_df(path: Path) -> pd.DataFrame:
    """Load simulation pickle keeping only columns needed for behavioral analysis."""
    df = pd.read_pickle(path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    keep = [c for c in _REQUIRED_COLUMNS if c in df.columns]
    return df[keep].copy()


def _pct_change(baseline_val: float, leave_val: float) -> float:
    if baseline_val == 0 or np.isnan(baseline_val) or np.isnan(leave_val):
        return np.nan
    return (leave_val - baseline_val) / abs(baseline_val) * 100.0


def _share(df: pd.DataFrame, choices) -> float:
    if df.empty:
        return np.nan
    return float(df["choice"].isin(np.asarray(choices).ravel()).mean())


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.mean())


def _add_cg_metadata(df: pd.DataFrame, end_age_caregiving: int) -> pd.DataFrame:
    """Add per-agent CG metadata columns: first_cg_age, total_cg_years."""
    informal = np.asarray(INFORMAL_CARE).ravel()
    is_cg = df["choice"].isin(informal)

    first_cg_period = df.loc[is_cg].groupby("agent")["period"].min()
    df["first_cg_period"] = df["agent"].map(first_cg_period)

    cg_in_range = is_cg & (df["age"] <= end_age_caregiving)
    total_cg = cg_in_range.groupby(df["agent"]).sum()
    df["total_cg_years"] = df["agent"].map(total_cg).fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Panel builders — each returns an OrderedDict-like dict {metric_name: value}
# ---------------------------------------------------------------------------


def _panel_a_labor(df: pd.DataFrame) -> dict[str, float]:
    """Share FT / PT / employed / unemployed / retired — outcome first, then age groups."""
    rows = {}
    outcomes = [
        ("Share FT", FULL_TIME),
        ("Share PT", PART_TIME),
        ("Empl. rate", _EMPLOYED),
        ("Share unemp.", UNEMPLOYED),
        ("Share retired", RETIREMENT),
    ]
    for outcome_label, choice_set in outcomes:
        for age_label, lo, hi in AGE_GROUPS_LABOR:
            sub = df[(df["age"] >= lo) & (df["age"] <= hi)]
            rows[f"{outcome_label} ({age_label})"] = _share(sub, choice_set)
    return rows


def _panel_b_caregiving(df: pd.DataFrame) -> dict[str, float]:
    """Caregiving shares — outcome first, then age groups. Cond. on care_demand > 0."""
    informal = np.asarray(INFORMAL_CARE).ravel()
    light = np.asarray(LIGHT_INFORMAL_CARE).ravel()
    intensive = np.asarray(INTENSIVE_INFORMAL_CARE).ravel()
    formal = np.asarray(FORMAL_CARE).ravel()

    def _care_share(with_demand, choice_set):
        n = len(with_demand)
        if n == 0:
            return np.nan
        return float(with_demand["choice"].isin(choice_set).sum()) / n

    def _combo_share(with_demand):
        n = len(with_demand)
        if n == 0:
            return np.nan
        n_intensive = int(with_demand["choice"].isin(intensive).sum())
        return float(n_intensive * IN_KIND_BENEFITS_UTILIZATION_RATE) / n

    outcomes = [
        ("Share informal CG", lambda wd: _care_share(wd, informal)),
        ("Share light CG", lambda wd: _care_share(wd, light)),
        ("Share intensive CG", lambda wd: _care_share(wd, intensive)),
        ("Share combination care", lambda wd: _combo_share(wd)),
        ("Share formal care", lambda wd: _care_share(wd, formal)),
    ]

    rows = {}
    for outcome_label, compute_fn in outcomes:
        for age_label, lo, hi in AGE_GROUPS_CAREGIVING:
            sub = df[(df["age"] >= lo) & (df["age"] <= hi)]
            with_demand = sub[sub["care_demand"] != NO_CARE_DEMAND]
            rows[f"{outcome_label} ({age_label})"] = compute_fn(with_demand)

    if "caregiving_type" in df.columns:
        all_lo, all_hi = 40, 67
        sub_all = df[(df["age"] >= all_lo) & (df["age"] <= all_hi)]
        cg1 = sub_all[sub_all["caregiving_type"] == 1]
        n_cg1 = len(cg1)
        if n_cg1 > 0:
            for outcome_label, compute_fn in outcomes:
                rows[f"{outcome_label} (CG type=1, All)"] = compute_fn(cg1)
        else:
            for outcome_label, _ in outcomes:
                rows[f"{outcome_label} (CG type=1, All)"] = np.nan

    return rows


_EDU_LABELS = {0: "Low Edu", 1: "High Edu"}


def _panel_caregiving_by_edu(df: pd.DataFrame, edu_level: int) -> dict[str, float]:
    """Caregiving shares for a specific education group (reuses Panel B logic)."""
    raw = _panel_b_caregiving(df[df["education"] == edu_level])
    label = _EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    return {f"{k} [{label}]": v for k, v in raw.items()}


def _panel_c_cg_labor(df: pd.DataFrame) -> dict[str, float]:
    """Labor state distribution among current informal caregivers (working age)."""
    informal = np.asarray(INFORMAL_CARE).ravel()

    labor_states = [
        ("Share FT among CG", _CG_FT),
        ("Share PT among CG", _CG_PT),
        ("Empl. rate among CG", _CG_EMPLOYED),
        ("Share unemp. among CG", _CG_UNEMPLOYED),
        ("Share retired among CG", _CG_RETIRED),
    ]

    rows: dict[str, float] = {}
    for outcome_label, cg_choices in labor_states:
        for age_label, lo, hi in AGE_GROUPS_CAREGIVING:
            sub = df[(df["age"] >= lo) & (df["age"] <= hi)]
            cg = sub[sub["choice"].isin(informal)]
            if cg.empty:
                rows[f"{outcome_label} ({age_label})"] = np.nan
            else:
                rows[f"{outcome_label} ({age_label})"] = float(
                    cg["choice"].isin(cg_choices).mean()
                )
    return rows


def _panel_d_ever_cg_labor(df: pd.DataFrame) -> dict[str, float]:
    """Labor state distribution among ever-caregivers (may or may not currently CG)."""
    informal = np.asarray(INFORMAL_CARE).ravel()
    ever_cg_agents = set(df.loc[df["choice"].isin(informal), "agent"].unique())

    labor_states = [
        ("Share FT (ever CG)", FULL_TIME),
        ("Share PT (ever CG)", PART_TIME),
        ("Empl. rate (ever CG)", _EMPLOYED),
        ("Share unemp. (ever CG)", UNEMPLOYED),
        ("Share retired (ever CG)", RETIREMENT),
    ]

    rows: dict[str, float] = {}
    for outcome_label, choice_set in labor_states:
        choices_arr = np.asarray(choice_set).ravel()
        for age_label, lo, hi in AGE_GROUPS_CG_HETERO:
            sub = df[(df["age"] >= lo) & (df["age"] <= hi)]
            ecg = sub[sub["agent"].isin(ever_cg_agents)]
            if ecg.empty:
                rows[f"{outcome_label} ({age_label})"] = np.nan
            else:
                rows[f"{outcome_label} ({age_label})"] = float(
                    ecg["choice"].isin(choices_arr).mean()
                )
    return rows


def _panel_e_cg_benefits(df: pd.DataFrame, wealth_unit: float) -> dict[str, float]:
    """Average benefit/top-up while caregiving, conditional on labor state.

    Baseline sim data contains ``care_benefits_and_costs`` (Pflegegeld);
    leave sim data contains ``caregiving_leave_top_up`` instead.
    Benefits are stored divided by wealth_unit in the sim data.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()

    if "caregiving_leave_top_up" in df.columns:
        benefit_col = "caregiving_leave_top_up"
    elif "care_benefits_and_costs" in df.columns:
        benefit_col = "care_benefits_and_costs"
    else:
        benefit_col = None

    labor_states = [
        ("Avg. benefit FT CG", _CG_FT),
        ("Avg. benefit PT CG", _CG_PT),
        ("Avg. benefit unemp. CG", _CG_UNEMPLOYED),
        ("Avg. benefit retired CG", _CG_RETIRED),
        ("Avg. benefit all CG", list(informal)),
    ]

    rows: dict[str, float] = {}
    for outcome_label, lag_choices in labor_states:
        for age_label, lo, hi in AGE_GROUPS_CAREGIVING:
            sub = df[(df["age"] >= lo) & (df["age"] <= hi)]
            cg = sub[sub["lagged_choice"].isin(lag_choices)]
            if cg.empty or benefit_col is None or benefit_col not in cg.columns:
                rows[f"{outcome_label} ({age_label})"] = np.nan
            else:
                rows[f"{outcome_label} ({age_label})"] = (
                    float(cg[benefit_col].mean()) * wealth_unit
                )
    return rows


def _panel_f_economic(df: pd.DataFrame, wealth_unit: float) -> dict[str, float]:
    """Economic outcomes for agents with age < 63."""
    sub = df[df["age"] < 63]
    rows = {}

    sav_col = "savings_dec" if "savings_dec" in sub.columns else "savings"
    rows["Avg. savings dec (< 63)"] = (
        _safe_mean(sub[sav_col]) * wealth_unit if sav_col in sub.columns else np.nan
    )
    rows["Avg. wealth (< 63)"] = (
        _safe_mean(sub["assets_begin_of_period"]) * wealth_unit
        if "assets_begin_of_period" in sub.columns
        else np.nan
    )
    rows["Avg. consumption (< 63)"] = (
        _safe_mean(sub["consumption"]) * wealth_unit
        if "consumption" in sub.columns
        else np.nan
    )
    rows["Avg. working hours (< 63)"] = (
        _safe_mean(sub["working_hours"]) if "working_hours" in sub.columns else np.nan
    )
    rows["Avg. gross labor income (< 63)"] = (
        _safe_mean(sub["gross_labor_income"]) * wealth_unit
        if "gross_labor_income" in sub.columns
        else np.nan
    )
    return rows


def _panel_g_retirement(df: pd.DataFrame, wealth_unit: float) -> dict[str, float]:
    """Retirement outcomes, conditional on agent being retired (choice in RETIREMENT)."""
    retired = np.asarray(RETIREMENT).ravel()
    sub = df[df["choice"].isin(retired)]
    rows = {}

    sav_col = "savings_dec" if "savings_dec" in sub.columns else "savings"
    rows["Avg. savings dec (ret)"] = (
        _safe_mean(sub[sav_col]) * wealth_unit if sav_col in sub.columns else np.nan
    )
    rows["Avg. wealth (ret)"] = (
        _safe_mean(sub["assets_begin_of_period"]) * wealth_unit
        if "assets_begin_of_period" in sub.columns
        else np.nan
    )
    rows["Avg. gross pension income"] = (
        _safe_mean(sub["gross_retirement_income"]) * wealth_unit
        if "gross_retirement_income" in sub.columns
        else np.nan
    )
    return rows


def _panel_h_experience_at_retirement(
    df: pd.DataFrame,
    max_exps_period_working: np.ndarray,
    start_age: int,
) -> dict[str, float]:
    """Average experience years at retirement for all agents and ever-caregivers.

    At the first retirement period the ``experience`` state still holds the
    working-period scaled float (not yet converted to pension points).  We
    recover years via ``experience * max_exps_period_working[period]``.
    """
    retired_choices = np.asarray(RETIREMENT).ravel()
    informal_choices = np.asarray(INFORMAL_CARE).ravel()

    ret_mask = df["choice"].isin(retired_choices)
    if not ret_mask.any() or "experience" not in df.columns:
        return {
            "Avg. retirement age (all)": np.nan,
            "Avg. retirement age (ever CG)": np.nan,
            "Avg. exp. years at ret. (all)": np.nan,
            "Avg. exp. years at ret. (ever CG)": np.nan,
        }

    first_ret = (
        df.loc[ret_mask]
        .groupby("agent")["period"]
        .min()
        .reset_index()
        .rename(columns={"period": "first_ret_period"})
    )

    merged = first_ret.merge(
        df[["agent", "period", "experience"]],
        left_on=["agent", "first_ret_period"],
        right_on=["agent", "period"],
        how="left",
    )

    periods = merged["first_ret_period"].values.astype(int)
    clipped = np.clip(periods, 0, len(max_exps_period_working) - 1)
    merged["exp_years_at_ret"] = (
        merged["experience"].values * max_exps_period_working[clipped]
    )
    merged["ret_age"] = start_age + merged["first_ret_period"]

    ever_cg = set(df.loc[df["choice"].isin(informal_choices), "agent"].unique())
    cg_mask = merged["agent"].isin(ever_cg)

    rows: dict[str, float] = {}
    rows["Avg. retirement age (all)"] = float(merged["ret_age"].mean())
    rows["Avg. retirement age (ever CG)"] = (
        float(merged.loc[cg_mask, "ret_age"].mean()) if cg_mask.any() else np.nan
    )
    rows["Avg. exp. years at ret. (all)"] = float(merged["exp_years_at_ret"].mean())
    rows["Avg. exp. years at ret. (ever CG)"] = (
        float(merged.loc[cg_mask, "exp_years_at_ret"].mean())
        if cg_mask.any()
        else np.nan
    )
    return rows


# ---------------------------------------------------------------------------
# Panels I–K: CG heterogeneity (first CG year, after, by duration)
# ---------------------------------------------------------------------------


def _panel_i_labor_at_first_cg(df: pd.DataFrame) -> dict[str, float]:
    """Labor state at the first caregiving period, binned by age at first CG.

    At the first CG period the agent IS caregiving (choice in INFORMAL_CARE),
    so the labour component is read from the CG-specific sub-choice constants.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()
    if "first_cg_period" not in df.columns:
        return {}

    first_cg_rows = df[df["period"] == df["first_cg_period"]].copy()
    first_cg_rows = first_cg_rows[first_cg_rows["choice"].isin(informal)]
    if first_cg_rows.empty:
        return {}

    outcomes = [
        ("Share FT at 1st CG", _CG_FT),
        ("Share PT at 1st CG", _CG_PT),
        ("Empl. rate at 1st CG", _CG_EMPLOYED),
    ]

    rows: dict[str, float] = {}
    for outcome_label, cg_choices in outcomes:
        for age_label, lo, hi in AGE_GROUPS_CG_HETERO:
            sub = first_cg_rows[
                (first_cg_rows["age"] >= lo) & (first_cg_rows["age"] <= hi)
            ]
            if sub.empty:
                rows[f"{outcome_label} ({age_label})"] = np.nan
            else:
                rows[f"{outcome_label} ({age_label})"] = float(
                    sub["choice"].isin(cg_choices).mean()
                )
    return rows


def _panel_j_ever_cg_excl_first(df: pd.DataFrame) -> dict[str, float]:
    """Labor supply for ever-CG agents, omitting the first caregiving year."""
    informal = np.asarray(INFORMAL_CARE).ravel()
    if "first_cg_period" not in df.columns:
        return {}

    ever_cg_agents = set(df.loc[df["choice"].isin(informal), "agent"].unique())
    excl = df[
        df["agent"].isin(ever_cg_agents) & (df["period"] != df["first_cg_period"])
    ]
    if excl.empty:
        return {}

    outcomes = [
        ("Share FT (ever CG, excl 1st)", FULL_TIME),
        ("Share PT (ever CG, excl 1st)", PART_TIME),
        ("Empl. rate (ever CG, excl 1st)", _EMPLOYED),
        ("Share unemp. (ever CG, excl 1st)", UNEMPLOYED),
        ("Share retired (ever CG, excl 1st)", RETIREMENT),
    ]

    rows: dict[str, float] = {}
    for outcome_label, choice_set in outcomes:
        choices_arr = np.asarray(choice_set).ravel()
        for age_label, lo, hi in AGE_GROUPS_CG_HETERO:
            sub = excl[(excl["age"] >= lo) & (excl["age"] <= hi)]
            if sub.empty:
                rows[f"{outcome_label} ({age_label})"] = np.nan
            else:
                rows[f"{outcome_label} ({age_label})"] = float(
                    sub["choice"].isin(choices_arr).mean()
                )
    return rows


def _panel_k_ever_cg_by_duration(df: pd.DataFrame) -> dict[str, float]:
    """Panel-D-style labour shares for ever-CG agents split by total CG years.

    Duration groups: 1 year, 2–3 years, >=4 years.
    Uses broad 10-year age bins (30–39, 40–49, 50–59, 60–67, All 30–67).
    """
    informal = np.asarray(INFORMAL_CARE).ravel()
    if "total_cg_years" not in df.columns:
        return {}

    ever_cg_agents = set(df.loc[df["choice"].isin(informal), "agent"].unique())
    ever_cg_df = df[df["agent"].isin(ever_cg_agents)]
    if ever_cg_df.empty:
        return {}

    agent_cg_years = ever_cg_df.groupby("agent")["total_cg_years"].first()

    outcomes = [
        ("Share FT", FULL_TIME),
        ("Share PT", PART_TIME),
        ("Empl. rate", _EMPLOYED),
        ("Share unemp.", UNEMPLOYED),
        ("Share retired", RETIREMENT),
    ]

    rows: dict[str, float] = {}
    for dur_label, dur_lo, dur_hi in CG_DURATION_GROUPS:
        agents_in_group = set(
            agent_cg_years[
                (agent_cg_years >= dur_lo) & (agent_cg_years <= dur_hi)
            ].index
        )
        dur_df = ever_cg_df[ever_cg_df["agent"].isin(agents_in_group)]

        for outcome_label, choice_set in outcomes:
            choices_arr = np.asarray(choice_set).ravel()
            is_retired = choice_set is RETIREMENT
            age_bins = _RETIRED_AGE_BINS_K if is_retired else AGE_GROUPS_BROAD_HETERO
            for age_label, lo, hi in age_bins:
                sub = dur_df[(dur_df["age"] >= lo) & (dur_df["age"] <= hi)]
                key = f"{outcome_label} ({dur_label}, {age_label})"
                if sub.empty:
                    rows[key] = np.nan
                else:
                    rows[key] = float(sub["choice"].isin(choices_arr).mean())
    return rows


# ---------------------------------------------------------------------------
# Panels N–X: education-stratified distributional analysis
# ---------------------------------------------------------------------------


def _panel_cg_labor_by_edu(df: pd.DataFrame, edu_level: int) -> dict[str, float]:
    """CG labor composition for a specific education group (reuses Panel E)."""
    raw = _panel_c_cg_labor(df[df["education"] == edu_level])
    label = _EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    return {f"{k} [{label}]": v for k, v in raw.items()}


def _panel_cg_benefits_by_edu(
    df: pd.DataFrame, edu_level: int, wealth_unit: float
) -> dict[str, float]:
    """CG benefits/top-ups for a specific education group (reuses Panel G)."""
    raw = _panel_e_cg_benefits(
        df[df["education"] == edu_level], wealth_unit=wealth_unit
    )
    label = _EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    return {f"{k} [{label}]": v for k, v in raw.items()}


def _panel_leave_eligibility(df: pd.DataFrame) -> dict[str, float]:
    """Prior-job composition among caregivers, by education.

    job_before_caregiving: 0 = no prior job (leave-ineligible), 1 = PT, 2 = FT.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()
    rows: dict[str, float] = {}

    if "job_before_caregiving" not in df.columns:
        for label in _EDU_LABELS.values():
            rows[f"Share no prior job (CG) [{label}]"] = np.nan
            rows[f"Share prior PT (CG) [{label}]"] = np.nan
            rows[f"Share prior FT (CG) [{label}]"] = np.nan
        rows["Share no prior job (CG) [All]"] = np.nan
        return rows

    cg_df = df[df["choice"].isin(informal)]

    for edu_level, label in _EDU_LABELS.items():
        edu_cg = cg_df[cg_df["education"] == edu_level]
        if edu_cg.empty:
            rows[f"Share no prior job (CG) [{label}]"] = np.nan
            rows[f"Share prior PT (CG) [{label}]"] = np.nan
            rows[f"Share prior FT (CG) [{label}]"] = np.nan
        else:
            jbc = edu_cg["job_before_caregiving"]
            rows[f"Share no prior job (CG) [{label}]"] = float((jbc == 0).mean())
            rows[f"Share prior PT (CG) [{label}]"] = float((jbc == 1).mean())
            rows[f"Share prior FT (CG) [{label}]"] = float((jbc == 2).mean())

    if cg_df.empty:
        rows["Share no prior job (CG) [All]"] = np.nan
    else:
        rows["Share no prior job (CG) [All]"] = float(
            (cg_df["job_before_caregiving"] == 0).mean()
        )

    return rows


def _panel_duration_by_edu(df: pd.DataFrame, edu_level: int) -> dict[str, float]:
    """Condensed duration heterogeneity for one education group.

    Focuses on empl. rate (All 30--67 and 60--67) and retirement (63--67)
    to show whether the job retention channel benefits this group.
    """
    label = _EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    informal = np.asarray(INFORMAL_CARE).ravel()

    if "total_cg_years" not in df.columns:
        return {}

    edu_df = df[df["education"] == edu_level]
    ever_cg_agents = set(edu_df.loc[edu_df["choice"].isin(informal), "agent"].unique())
    ever_cg_df = edu_df[edu_df["agent"].isin(ever_cg_agents)]
    if ever_cg_df.empty:
        return {}

    agent_cg_years = ever_cg_df.groupby("agent")["total_cg_years"].first()

    outcomes = [
        ("Empl. rate", _EMPLOYED, [("All (30--67)", 30, 67), ("60--67", 60, 67)]),
        ("Share retired", RETIREMENT, [("63--67", 63, 67)]),
    ]

    rows: dict[str, float] = {}
    for dur_label, dur_lo, dur_hi in CG_DURATION_GROUPS:
        agents_in_group = set(
            agent_cg_years[
                (agent_cg_years >= dur_lo) & (agent_cg_years <= dur_hi)
            ].index
        )
        dur_df = ever_cg_df[ever_cg_df["agent"].isin(agents_in_group)]

        for outcome_label, choice_set, age_bins in outcomes:
            choices_arr = np.asarray(choice_set).ravel()
            for age_label, lo, hi in age_bins:
                sub = dur_df[(dur_df["age"] >= lo) & (dur_df["age"] <= hi)]
                key = f"{outcome_label} ({dur_label}, {age_label}) [{label}]"
                rows[key] = (
                    float(sub["choice"].isin(choices_arr).mean())
                    if not sub.empty
                    else np.nan
                )
    return rows


def _panel_pension_by_edu(
    df: pd.DataFrame,
    edu_level: int,
    wealth_unit: float,
    max_exps_period_working: np.ndarray,
    start_age: int,
) -> dict[str, float]:
    """Pension and experience at retirement for one education group."""
    label = _EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    edu_df = df[df["education"] == edu_level]

    ret_raw = _panel_g_retirement(edu_df, wealth_unit=wealth_unit)
    exp_raw = _panel_h_experience_at_retirement(
        edu_df,
        max_exps_period_working=max_exps_period_working,
        start_age=start_age,
    )

    rows: dict[str, float] = {}
    for k, v in ret_raw.items():
        rows[f"{k} [{label}]"] = v
    for k, v in exp_raw.items():
        rows[f"{k} [{label}]"] = v
    return rows


def _panel_lifecycle_by_edu(
    df: pd.DataFrame, edu_level: int, wealth_unit: float
) -> dict[str, float]:
    """Lifecycle outcomes for ever-caregivers in one education group.

    Reports avg. gross labor income (30--67), avg. gross pension income
    (retired periods), and avg. benefit per CG period.  The pct-change
    columns then directly show the policy impact on each.
    """
    label = _EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    informal = np.asarray(INFORMAL_CARE).ravel()
    retired_choices = np.asarray(RETIREMENT).ravel()

    edu_df = df[df["education"] == edu_level]
    ever_cg_agents = set(edu_df.loc[edu_df["choice"].isin(informal), "agent"].unique())
    ever_cg = edu_df[edu_df["agent"].isin(ever_cg_agents)]

    rows: dict[str, float] = {}

    if ever_cg.empty:
        rows[f"Avg. gross labor inc. (ever CG, 30--67) [{label}]"] = np.nan
        rows[f"Avg. gross pension inc. (ever CG, ret) [{label}]"] = np.nan
        rows[f"Avg. benefit per CG period [{label}]"] = np.nan
        rows[f"Avg. consumption (ever CG, <63) [{label}]"] = np.nan
        return rows

    working_age = ever_cg[(ever_cg["age"] >= 30) & (ever_cg["age"] <= 67)]
    rows[f"Avg. gross labor inc. (ever CG, 30--67) [{label}]"] = (
        _safe_mean(working_age["gross_labor_income"]) * wealth_unit
        if "gross_labor_income" in working_age.columns
        else np.nan
    )

    ret_periods = ever_cg[ever_cg["choice"].isin(retired_choices)]
    rows[f"Avg. gross pension inc. (ever CG, ret) [{label}]"] = (
        _safe_mean(ret_periods["gross_retirement_income"]) * wealth_unit
        if "gross_retirement_income" in ret_periods.columns and not ret_periods.empty
        else np.nan
    )

    if "caregiving_leave_top_up" in ever_cg.columns:
        benefit_col = "caregiving_leave_top_up"
    elif "care_benefits_and_costs" in ever_cg.columns:
        benefit_col = "care_benefits_and_costs"
    else:
        benefit_col = None

    cg_periods = ever_cg[ever_cg["lagged_choice"].isin(informal)]
    if benefit_col and not cg_periods.empty:
        rows[f"Avg. benefit per CG period [{label}]"] = (
            _safe_mean(cg_periods[benefit_col]) * wealth_unit
        )
    else:
        rows[f"Avg. benefit per CG period [{label}]"] = np.nan

    under63 = ever_cg[ever_cg["age"] < 63]
    rows[f"Avg. consumption (ever CG, <63) [{label}]"] = (
        _safe_mean(under63["consumption"]) * wealth_unit
        if "consumption" in under63.columns
        else np.nan
    )

    return rows
