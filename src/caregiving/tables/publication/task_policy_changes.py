"""Publication: behavioral changes from caregiving policies (LaTeX table).

Compares three policy regimes across twenty-seven panels:
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
  Y) Leave take-up decomposition (eligibility, take-up, recipient composition)
  Z) Prior labor state at first caregiving entry, by age bin and education
 AA) Care demand & CG duration by prior job status (had-job vs no-prior-job)
 BB) Care arrangement choices by demand intensity & prior job (combination care)
 CC) Care demand onset heterogeneity across all CG types (0 and 1)
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
    CARE_DEMAND_INTENSIVE,
    CARE_DEMAND_LIGHT,
    FORMAL_CARE,
    FULL_TIME,
    INFORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_CARE,
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

CG_ENTRY_AGE_BINS = [
    ("40--49", 40, 49),
    ("50--54", 50, 54),
    ("55--59", 55, 59),
    ("50--59", 50, 59),
    ("60--62", 60, 62),
    ("63--67", 63, 67),
    ("All (40--67)", 40, 67),
]

REQUIRED_COLUMNS = [
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
    "years_leave_used_total",
    "full_leave_year_used",
]

CG_FT = [11, 15]
CG_PT = [10, 14]
CG_EMPLOYED = [10, 11, 14, 15]
CG_UNEMPLOYED = [9, 13]
CG_RETIRED = [8, 12]

EMPLOYED = np.concatenate(
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

RETIRED_AGE_BINS_K = [
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
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params_no_pflegegeld.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "policy_behavioral_changes.tex",
) -> None:
    """Create LaTeX table of behavioral changes from caregiving policies."""
    build_policy_changes_table(
        path_to_specs,
        path_to_baseline_sim,
        path_to_normal_leave_sim,
        path_to_full_leave_sim,
        path_to_save_table,
    )


# @pytask.mark.tables
# @pytask.mark.policy_changes
# @pytask.mark.publication
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
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params_no_pflegegeld_back_to_Jan7.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "policy_behavioral_changes_back_to_Jan7.tex",
) -> None:
    """Same table as above but using back-to-Jan7 simulation data."""
    build_policy_changes_table(
        path_to_specs,
        path_to_baseline_sim,
        path_to_normal_leave_sim,
        path_to_full_leave_sim,
        path_to_save_table,
        table_label="tab:policy_behavioral_changes_jan7",
    )


@pytask.mark.tables
@pytask.mark.policy_changes
@pytask.mark.policy_changes_new
@pytask.mark.publication
def task_create_policy_changes_table_full_beirat(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_estimated_params.pkl",
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "policy_behavioral_changes_full_beirat.tex",
) -> None:
    """Policy changes table: full Beirat leave vs full CG leave with job retention."""
    build_policy_changes_table(
        path_to_specs,
        path_to_baseline_sim,
        path_to_normal_leave_sim,
        path_to_full_leave_sim,
        path_to_save_table,
        table_label="tab:policy_behavioral_changes_full_beirat",
    )


@pytask.mark.tables
@pytask.mark.policy_changes
@pytask.mark.policy_changes_new
@pytask.mark.publication
def task_create_policy_changes_table_full_beirat_no_pflegegeld(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_estimated_params.pkl",
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params_no_pflegegeld.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "policy_behavioral_changes_full_beirat_no_pflegegeld.tex",
) -> None:
    """Policy changes table: full Beirat leave vs full CG leave (without Pflegegeld)."""
    build_policy_changes_table(
        path_to_specs,
        path_to_baseline_sim,
        path_to_normal_leave_sim,
        path_to_full_leave_sim,
        path_to_save_table,
        table_label="tab:policy_behavioral_changes_full_beirat_no_pflegegeld",
    )


# @pytask.mark.tables
# @pytask.mark.policy_changes
# @pytask.mark.policy_changes_new
# @pytask.mark.publication
def task_create_policy_changes_table_full_beirat_parquet(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_estimated_params.parquet",
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "policy_behavioral_changes_full_beirat_parquet.tex",
) -> None:
    """Same as full_beirat but loading normal leave from parquet (integrity check)."""
    build_policy_changes_table(
        path_to_specs,
        path_to_baseline_sim,
        path_to_normal_leave_sim,
        path_to_full_leave_sim,
        path_to_save_table,
        table_label="tab:policy_behavioral_changes_full_beirat_parquet",
    )


@pytask.mark.tables
@pytask.mark.policy_changes
@pytask.mark.policy_changes_new
@pytask.mark.publication
def task_create_policy_changes_table_full_leave_pflegegeld_comparison(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params_no_pflegegeld.pkl",
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "policy_behavioral_changes_full_leave_pflegegeld_comparison.tex",
) -> None:
    """Full CG leave without vs with Pflegegeld (both against baseline)."""
    build_policy_changes_table(
        path_to_specs,
        path_to_baseline_sim,
        path_to_normal_leave_sim,
        path_to_full_leave_sim,
        path_to_save_table,
        table_label="tab:policy_behavioral_changes_full_leave_pflegegeld",
    )


@pytask.mark.tables
@pytask.mark.policy_changes
@pytask.mark.policy_changes_neww
@pytask.mark.publication
def task_create_policy_changes_table_beirat_comparison(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_beirat_estimated_params.pkl",
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_estimated_params.pkl",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "policy_behavioral_changes_beirat_comparison.tex",
) -> None:
    """Beirat partial-leave-only vs full Beirat (both against baseline).

    Both variants retain baseline Pflegegeld; they differ in the leave
    top-up: the partial-only variant pays 65% only to PT CG with prior FT,
    while the full variant also covers up to 1 year for unemployed CG.
    """
    build_policy_changes_table(
        path_to_specs,
        path_to_baseline_sim,
        path_to_normal_leave_sim,
        path_to_full_leave_sim,
        path_to_save_table,
        table_label="tab:policy_behavioral_changes_beirat_comparison",
    )


def build_policy_changes_table(
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

    baseline_df = load_sim_df(path_to_baseline_sim)
    normal_df = load_sim_df(path_to_normal_leave_sim)
    full_df = load_sim_df(path_to_full_leave_sim)

    for df in (baseline_df, normal_df, full_df):
        if "age" not in df.columns and "period" in df.columns:
            df["age"] = start_age + df["period"]
        add_cg_metadata(df, end_age_cg)

    panels = [
        ("A: Labor supply", panel_a_labor, {}),
        ("B: Caregiving", panel_b_caregiving, {}),
        ("C: Caregiving (Low Edu)", panel_caregiving_by_edu, {"edu_level": 0}),
        ("D: Caregiving (High Edu)", panel_caregiving_by_edu, {"edu_level": 1}),
        ("E: CG labor composition", panel_c_cg_labor, {}),
        ("F: Ever-CG labor composition", panel_d_ever_cg_labor, {}),
        ("G: CG benefits/top-ups", panel_e_cg_benefits, {"wealth_unit": wealth_unit}),
        ("H: Economic (< 63)", panel_f_economic, {"wealth_unit": wealth_unit}),
        ("I: Retirement", panel_g_retirement, {"wealth_unit": wealth_unit}),
        (
            "J: Experience at retirement",
            panel_h_experience_at_retirement,
            {"max_exps_period_working": max_exps, "start_age": start_age},
        ),
        ("K: Labor at first CG year", panel_i_labor_at_first_cg, {}),
        ("L: Ever-CG labor excl. 1st CG yr", panel_j_ever_cg_excl_first, {}),
        (
            "M: Ever-CG by CG duration",
            panel_k_ever_cg_by_duration,
            {},
        ),
        ("N: CG labor (Low Edu)", panel_cg_labor_by_edu, {"edu_level": 0}),
        ("O: CG labor (High Edu)", panel_cg_labor_by_edu, {"edu_level": 1}),
        (
            "P: CG benefits (Low Edu)",
            panel_cg_benefits_by_edu,
            {"edu_level": 0, "wealth_unit": wealth_unit},
        ),
        (
            "Q: CG benefits (High Edu)",
            panel_cg_benefits_by_edu,
            {"edu_level": 1, "wealth_unit": wealth_unit},
        ),
        ("R: Leave eligibility by edu", panel_leave_eligibility, {}),
        (
            "Y: Leave take-up decomposition",
            panel_leave_takeup,
            {"wealth_unit": wealth_unit},
        ),
        ("Z: Prior state at CG entry", panel_prior_state_at_cg_entry, {}),
        ("AA: Care demand/duration by prior job", panel_care_demand_duration, {}),
        ("BB: Care arrangements by demand intensity", panel_care_arrangements, {}),
        ("CC: Care demand onset (all CG types)", panel_care_demand_onset, {}),
        ("S: Duration (Low Edu)", panel_duration_by_edu, {"edu_level": 0}),
        ("T: Duration (High Edu)", panel_duration_by_edu, {"edu_level": 1}),
        (
            "U: Pension/exp (Low Edu)",
            panel_pension_by_edu,
            {
                "edu_level": 0,
                "wealth_unit": wealth_unit,
                "max_exps_period_working": max_exps,
                "start_age": start_age,
            },
        ),
        (
            "V: Pension/exp (High Edu)",
            panel_pension_by_edu,
            {
                "edu_level": 1,
                "wealth_unit": wealth_unit,
                "max_exps_period_working": max_exps,
                "start_age": start_age,
            },
        ),
        (
            "W: Lifecycle (Low Edu)",
            panel_lifecycle_by_edu,
            {"edu_level": 0, "wealth_unit": wealth_unit},
        ),
        (
            "X: Lifecycle (High Edu)",
            panel_lifecycle_by_edu,
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
    for k in all_metrics_normal:
        if k not in all_metrics_baseline:
            metrics.append(k)
    for k in all_metrics_full:
        if k not in all_metrics_baseline and k not in all_metrics_normal:
            metrics.append(k)

    table_dict = {
        "Metric": metrics,
        "Baseline": [all_metrics_baseline.get(m, np.nan) for m in metrics],
        "Normal leave": [all_metrics_normal.get(m, np.nan) for m in metrics],
        "Full leave": [all_metrics_full.get(m, np.nan) for m in metrics],
    }

    pct_normal = []
    pct_full = []
    for m in metrics:
        bv = all_metrics_baseline.get(m, np.nan)
        nv = all_metrics_normal.get(m, np.nan)
        fv = all_metrics_full.get(m, np.nan)
        if isinstance(bv, str) or isinstance(nv, str):
            pct_normal.append("")
            pct_full.append("")
        else:
            pct_normal.append(pct_change(bv, nv))
            pct_full.append(pct_change(bv, fv))

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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_sim_df(path: Path) -> pd.DataFrame:
    """Load simulation data (pickle or parquet) keeping only needed columns."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    keep = [c for c in REQUIRED_COLUMNS if c in df.columns]
    return df[keep].copy()


def pct_change(baseline_val: float, leave_val: float) -> float:
    if baseline_val == 0 or np.isnan(baseline_val) or np.isnan(leave_val):
        return np.nan
    return (leave_val - baseline_val) / abs(baseline_val) * 100.0


def share(df: pd.DataFrame, choices) -> float:
    if df.empty:
        return np.nan
    return float(df["choice"].isin(np.asarray(choices).ravel()).mean())


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.mean())


def add_cg_metadata(df: pd.DataFrame, end_age_caregiving: int) -> pd.DataFrame:
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


def panel_a_labor(df: pd.DataFrame) -> dict[str, float]:
    """Share FT / PT / employed / unemployed / retired — outcome first, then age groups."""
    rows = {}
    outcomes = [
        ("Share FT", FULL_TIME),
        ("Share PT", PART_TIME),
        ("Empl. rate", EMPLOYED),
        ("Share unemp.", UNEMPLOYED),
        ("Share retired", RETIREMENT),
    ]
    for outcome_label, choice_set in outcomes:
        for age_label, lo, hi in AGE_GROUPS_LABOR:
            sub = df[(df["age"] >= lo) & (df["age"] <= hi)]
            rows[f"{outcome_label} ({age_label})"] = share(sub, choice_set)
    return rows


def panel_b_caregiving(df: pd.DataFrame) -> dict[str, float]:
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


EDU_LABELS = {0: "Low Edu", 1: "High Edu"}


def panel_caregiving_by_edu(df: pd.DataFrame, edu_level: int) -> dict[str, float]:
    """Caregiving shares for a specific education group (reuses Panel B logic)."""
    raw = panel_b_caregiving(df[df["education"] == edu_level])
    label = EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    return {f"{k} [{label}]": v for k, v in raw.items()}


def panel_c_cg_labor(df: pd.DataFrame) -> dict[str, float]:
    """Labor state distribution among current informal caregivers (working age)."""
    informal = np.asarray(INFORMAL_CARE).ravel()

    labor_states = [
        ("Share FT among CG", CG_FT),
        ("Share PT among CG", CG_PT),
        ("Empl. rate among CG", CG_EMPLOYED),
        ("Share unemp. among CG", CG_UNEMPLOYED),
        ("Share retired among CG", CG_RETIRED),
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


def panel_d_ever_cg_labor(df: pd.DataFrame) -> dict[str, float]:
    """Labor state distribution among ever-caregivers (may or may not currently CG)."""
    informal = np.asarray(INFORMAL_CARE).ravel()
    ever_cg_agents = set(df.loc[df["choice"].isin(informal), "agent"].unique())

    labor_states = [
        ("Share FT (ever CG)", FULL_TIME),
        ("Share PT (ever CG)", PART_TIME),
        ("Empl. rate (ever CG)", EMPLOYED),
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


def panel_e_cg_benefits(df: pd.DataFrame, wealth_unit: float) -> dict[str, float]:
    """Average benefit/top-up while caregiving, conditional on labor state.

    For the baseline, reports ``care_benefits_and_costs`` (Pflegegeld).
    For leave models, reports ``caregiving_leave_top_up`` as the primary
    benefit. When a leave model also retains Pflegegeld (both columns
    present), additional rows report Pflegegeld and total transfers.
    Benefits are stored divided by wealth_unit in the sim data.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()

    has_leave = "caregiving_leave_top_up" in df.columns
    has_pflegegeld = "care_benefits_and_costs" in df.columns

    if has_leave:
        benefit_col = "caregiving_leave_top_up"
    elif has_pflegegeld:
        benefit_col = "care_benefits_and_costs"
    else:
        benefit_col = None

    labor_states = [
        ("Avg. benefit FT CG", CG_FT),
        ("Avg. benefit PT CG", CG_PT),
        ("Avg. benefit unemp. CG", CG_UNEMPLOYED),
        ("Avg. benefit retired CG", CG_RETIRED),
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

    if has_leave and has_pflegegeld:
        pflegegeld_states = [
            ("Avg. Pflegegeld FT CG", CG_FT),
            ("Avg. Pflegegeld PT CG", CG_PT),
            ("Avg. Pflegegeld unemp. CG", CG_UNEMPLOYED),
            ("Avg. Pflegegeld retired CG", CG_RETIRED),
            ("Avg. Pflegegeld all CG", list(informal)),
        ]
        for outcome_label, lag_choices in pflegegeld_states:
            for age_label, lo, hi in AGE_GROUPS_CAREGIVING:
                sub = df[(df["age"] >= lo) & (df["age"] <= hi)]
                cg = sub[sub["lagged_choice"].isin(lag_choices)]
                if cg.empty or "care_benefits_and_costs" not in cg.columns:
                    rows[f"{outcome_label} ({age_label})"] = np.nan
                else:
                    rows[f"{outcome_label} ({age_label})"] = (
                        float(cg["care_benefits_and_costs"].mean()) * wealth_unit
                    )

        total_states = [
            ("Avg. total transfer FT CG", CG_FT),
            ("Avg. total transfer PT CG", CG_PT),
            ("Avg. total transfer unemp. CG", CG_UNEMPLOYED),
            ("Avg. total transfer retired CG", CG_RETIRED),
            ("Avg. total transfer all CG", list(informal)),
        ]
        for outcome_label, lag_choices in total_states:
            for age_label, lo, hi in AGE_GROUPS_CAREGIVING:
                sub = df[(df["age"] >= lo) & (df["age"] <= hi)]
                cg = sub[sub["lagged_choice"].isin(lag_choices)]
                if cg.empty:
                    rows[f"{outcome_label} ({age_label})"] = np.nan
                else:
                    leave_val = (
                        float(cg["caregiving_leave_top_up"].mean()) * wealth_unit
                        if "caregiving_leave_top_up" in cg.columns
                        else 0.0
                    )
                    pg_val = (
                        float(cg["care_benefits_and_costs"].mean()) * wealth_unit
                        if "care_benefits_and_costs" in cg.columns
                        else 0.0
                    )
                    rows[f"{outcome_label} ({age_label})"] = leave_val + pg_val

    return rows


def panel_f_economic(df: pd.DataFrame, wealth_unit: float) -> dict[str, float]:
    """Economic outcomes for agents with age < 63."""
    sub = df[df["age"] < 63]
    rows = {}

    sav_col = "savings_dec" if "savings_dec" in sub.columns else "savings"
    rows["Avg. savings dec (< 63)"] = (
        safe_mean(sub[sav_col]) * wealth_unit if sav_col in sub.columns else np.nan
    )
    rows["Avg. wealth (< 63)"] = (
        safe_mean(sub["assets_begin_of_period"]) * wealth_unit
        if "assets_begin_of_period" in sub.columns
        else np.nan
    )
    rows["Avg. consumption (< 63)"] = (
        safe_mean(sub["consumption"]) * wealth_unit
        if "consumption" in sub.columns
        else np.nan
    )

    informal = np.asarray(INFORMAL_CARE).ravel()
    active_cg = sub[sub["lagged_choice"].isin(informal)]
    rows["Avg. consumption (active CG, < 63)"] = (
        safe_mean(active_cg["consumption"]) * wealth_unit
        if "consumption" in active_cg.columns and not active_cg.empty
        else np.nan
    )

    rows["Avg. working hours (< 63)"] = (
        safe_mean(sub["working_hours"]) if "working_hours" in sub.columns else np.nan
    )
    rows["Avg. gross labor income (< 63)"] = (
        safe_mean(sub["gross_labor_income"]) * wealth_unit
        if "gross_labor_income" in sub.columns
        else np.nan
    )
    return rows


def panel_g_retirement(df: pd.DataFrame, wealth_unit: float) -> dict[str, float]:
    """Retirement outcomes, conditional on agent being retired (choice in RETIREMENT)."""
    retired = np.asarray(RETIREMENT).ravel()
    sub = df[df["choice"].isin(retired)]
    rows = {}

    sav_col = "savings_dec" if "savings_dec" in sub.columns else "savings"
    rows["Avg. savings dec (ret)"] = (
        safe_mean(sub[sav_col]) * wealth_unit if sav_col in sub.columns else np.nan
    )
    rows["Avg. wealth (ret)"] = (
        safe_mean(sub["assets_begin_of_period"]) * wealth_unit
        if "assets_begin_of_period" in sub.columns
        else np.nan
    )
    rows["Avg. gross pension income"] = (
        safe_mean(sub["gross_retirement_income"]) * wealth_unit
        if "gross_retirement_income" in sub.columns
        else np.nan
    )
    return rows


def panel_h_experience_at_retirement(
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


def panel_i_labor_at_first_cg(df: pd.DataFrame) -> dict[str, float]:
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
        ("Share FT at 1st CG", CG_FT),
        ("Share PT at 1st CG", CG_PT),
        ("Empl. rate at 1st CG", CG_EMPLOYED),
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


def panel_j_ever_cg_excl_first(df: pd.DataFrame) -> dict[str, float]:
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
        ("Empl. rate (ever CG, excl 1st)", EMPLOYED),
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


def panel_k_ever_cg_by_duration(df: pd.DataFrame) -> dict[str, float]:
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
        ("Empl. rate", EMPLOYED),
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
            age_bins = RETIRED_AGE_BINS_K if is_retired else AGE_GROUPS_BROAD_HETERO
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


def panel_cg_labor_by_edu(df: pd.DataFrame, edu_level: int) -> dict[str, float]:
    """CG labor composition for a specific education group (reuses Panel E)."""
    raw = panel_c_cg_labor(df[df["education"] == edu_level])
    label = EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    return {f"{k} [{label}]": v for k, v in raw.items()}


def panel_cg_benefits_by_edu(
    df: pd.DataFrame, edu_level: int, wealth_unit: float
) -> dict[str, float]:
    """CG benefits/top-ups for a specific education group (reuses Panel G)."""
    raw = panel_e_cg_benefits(df[df["education"] == edu_level], wealth_unit=wealth_unit)
    label = EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    return {f"{k} [{label}]": v for k, v in raw.items()}


def panel_leave_eligibility(df: pd.DataFrame) -> dict[str, float]:
    """Prior-job composition among caregivers, by education.

    job_before_caregiving: 0 = no prior job (leave-ineligible), 1 = PT, 2 = FT.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()
    rows: dict[str, float] = {}

    if "job_before_caregiving" not in df.columns:
        for label in EDU_LABELS.values():
            rows[f"Share no prior job (CG) [{label}]"] = np.nan
            rows[f"Share prior PT (CG) [{label}]"] = np.nan
            rows[f"Share prior FT (CG) [{label}]"] = np.nan
        rows["Share no prior job (CG) [All]"] = np.nan
        return rows

    cg_df = df[df["choice"].isin(informal)]

    for edu_level, label in EDU_LABELS.items():
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


def panel_leave_takeup(df: pd.DataFrame, wealth_unit: float) -> dict[str, float]:
    """Leave take-up decomposition: eligibility, take-up rates, recipient composition.

    Answers: who is eligible, who actually receives the leave top-up,
    what is their education / labor state / prior job, and what is the
    average benefit conditional on receipt.

    Uses ``lagged_choice`` for the CG labor state that triggers the
    benefit (the budget equation evaluates eligibility on lagged_choice)
    and ``choice`` for the current-period labor state of caregivers.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()
    rows: dict[str, float] = {}

    has_leave = "caregiving_leave_top_up" in df.columns
    has_jbc = "job_before_caregiving" in df.columns
    has_ylu = "years_leave_used_total" in df.columns

    cg = df[df["choice"].isin(informal)]
    n_cg = len(cg)

    if n_cg == 0:
        return rows

    # -- A) Current labor state of all caregivers (choice) --
    rows["Share FT CG (all CG)"] = float(cg["choice"].isin(CG_FT).mean())
    rows["Share PT CG (all CG)"] = float(cg["choice"].isin(CG_PT).mean())
    rows["Share Unemp CG (all CG)"] = float(
        cg["choice"].isin(CG_UNEMPLOYED).mean()
    )
    rows["Share Retired CG (all CG)"] = float(
        cg["choice"].isin(CG_RETIRED).mean()
    )

    # -- B) Prior-job eligibility structure --
    if has_jbc:
        jbc = cg["job_before_caregiving"]
        rows["Share no prior job (all CG)"] = float((jbc == 0).mean())
        rows["Share prior PT (all CG)"] = float((jbc == 1).mean())
        rows["Share prior FT (all CG)"] = float((jbc == 2).mean())
        rows["Share any prior job (all CG)"] = float((jbc > 0).mean())

        # Eligibility uses lagged_choice (budget equation trigger)
        lag_pt = cg["lagged_choice"].isin(CG_PT)
        lag_unemp = cg["lagged_choice"].isin(CG_UNEMPLOYED)
        prior_ft = jbc == 2
        prior_any = jbc > 0

        n_pt_prior_ft = int((lag_pt & prior_ft).sum())
        n_unemp_prior_job = int((lag_unemp & prior_any).sum())
        rows["Share PT CG w/ prior FT (all CG)"] = n_pt_prior_ft / n_cg
        rows["Share Unemp CG w/ prior job (all CG)"] = (
            n_unemp_prior_job / n_cg
        )
        rows["Share leave-eligible (all CG)"] = (
            n_pt_prior_ft + n_unemp_prior_job
        ) / n_cg

        for edu_level, label in EDU_LABELS.items():
            edu_cg = cg[cg["education"] == edu_level]
            ne = len(edu_cg)
            if ne == 0:
                for k in [
                    f"Share no prior job [{label}]",
                    f"Share prior FT [{label}]",
                    f"Share leave-eligible [{label}]",
                ]:
                    rows[k] = np.nan
                continue
            ejbc = edu_cg["job_before_caregiving"]
            rows[f"Share no prior job [{label}]"] = float((ejbc == 0).mean())
            rows[f"Share prior FT [{label}]"] = float((ejbc == 2).mean())
            ept = edu_cg["lagged_choice"].isin(CG_PT) & (ejbc == 2)
            eun = edu_cg["lagged_choice"].isin(CG_UNEMPLOYED) & (ejbc > 0)
            rows[f"Share leave-eligible [{label}]"] = float(
                (ept | eun).sum() / ne
            )

    # -- C) Take-up rates --
    if has_leave:
        topup = cg["caregiving_leave_top_up"]
        receiving = topup > 0
        n_recv = int(receiving.sum())
        rows["Take-up rate (all CG)"] = n_recv / n_cg
        rows["N CG person-periods"] = float(n_cg)
        rows["N receiving leave top-up"] = float(n_recv)

        if has_jbc:
            lag_pt = cg["lagged_choice"].isin(CG_PT)
            lag_unemp = cg["lagged_choice"].isin(CG_UNEMPLOYED)
            prior_ft = cg["job_before_caregiving"] == 2
            prior_any = cg["job_before_caregiving"] > 0
            eligible_mask = (lag_pt & prior_ft) | (lag_unemp & prior_any)
            n_eligible = int(eligible_mask.sum())
            rows["N effectively eligible"] = float(n_eligible)
            rows["Take-up rate | eligible"] = (
                n_recv / n_eligible if n_eligible > 0 else np.nan
            )

        # Intensive margin
        if n_recv > 0:
            recv_df = cg[receiving]
            rows["Avg. top-up | receipt (EUR/yr)"] = (
                float(recv_df["caregiving_leave_top_up"].mean()) * wealth_unit
            )
            rows["Avg. top-up | all CG (EUR/yr)"] = (
                float(topup.mean()) * wealth_unit
            )
        else:
            rows["Avg. top-up | receipt (EUR/yr)"] = 0.0
            rows["Avg. top-up | all CG (EUR/yr)"] = 0.0

        # By education
        for edu_level, label in EDU_LABELS.items():
            edu_cg = cg[cg["education"] == edu_level]
            if edu_cg.empty:
                rows[f"Take-up rate [{label}]"] = np.nan
                rows[f"Avg. top-up | receipt [{label}]"] = np.nan
                continue
            edu_recv = edu_cg["caregiving_leave_top_up"] > 0
            rows[f"Take-up rate [{label}]"] = float(edu_recv.mean())
            edu_receivers = edu_cg[edu_recv]
            if edu_receivers.empty:
                rows[f"Avg. top-up | receipt [{label}]"] = 0.0
            else:
                rows[f"Avg. top-up | receipt [{label}]"] = (
                    float(edu_receivers["caregiving_leave_top_up"].mean())
                    * wealth_unit
                )

        # -- D) Composition of recipients (lagged_choice = triggering state) --
        if n_recv > 0:
            recv_df = cg[receiving]
            rows["Recv: lag. PT CG (trigger)"] = float(
                recv_df["lagged_choice"].isin(CG_PT).mean()
            )
            rows["Recv: lag. Unemp CG (trigger)"] = float(
                recv_df["lagged_choice"].isin(CG_UNEMPLOYED).mean()
            )
            rows["Recv: lag. FT CG"] = float(
                recv_df["lagged_choice"].isin(CG_FT).mean()
            )
            rows["Recv: lag. Retired CG"] = float(
                recv_df["lagged_choice"].isin(CG_RETIRED).mean()
            )

            # Current choice of recipients (what they do this period)
            rows["Recv: curr. FT CG"] = float(
                recv_df["choice"].isin(CG_FT).mean()
            )
            rows["Recv: curr. PT CG"] = float(
                recv_df["choice"].isin(CG_PT).mean()
            )
            rows["Recv: curr. Unemp CG"] = float(
                recv_df["choice"].isin(CG_UNEMPLOYED).mean()
            )
            rows["Recv: curr. Retired CG"] = float(
                recv_df["choice"].isin(CG_RETIRED).mean()
            )

            if has_jbc:
                rjbc = recv_df["job_before_caregiving"]
                rows["Recv: Share prior FT"] = float((rjbc == 2).mean())
                rows["Recv: Share prior PT"] = float((rjbc == 1).mean())
                rows["Recv: Share no prior job"] = float((rjbc == 0).mean())

            for edu_level, label in EDU_LABELS.items():
                rows[f"Recv: Share [{label}]"] = float(
                    (recv_df["education"] == edu_level).mean()
                )

    # -- E) Beirat cap usage (if available) --
    if has_ylu:
        ylu = cg["years_leave_used_total"]
        rows["Share 0 leave years used"] = float((ylu == 0).mean())
        rows["Share 1 leave year used"] = float((ylu == 1).mean())
        rows["Share 2 leave years used"] = float((ylu == 2).mean())
        rows["Share 3 leave years used (capped)"] = float(
            (ylu >= 3).mean()
        )

    return rows


def panel_prior_state_at_cg_entry(df: pd.DataFrame) -> dict[str, float]:
    """Prior labor state at first caregiving entry, by age at entry.

    For each agent, identifies the first period of informal caregiving
    and records ``lagged_choice`` (previous-period state) as the labor
    state immediately before caregiving began.  The retirement share is
    only reported for age bins that include ages 63+.

    Also derives leave-eligibility from the prior labor state:
    agents whose ``lagged_choice`` was FT or PT had a job before
    caregiving and would be eligible for earnings-related leave.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()
    ft_arr = np.asarray(FULL_TIME).ravel()
    pt_arr = np.asarray(PART_TIME).ravel()
    unemp_arr = np.asarray(UNEMPLOYED).ravel()
    ret_arr = np.asarray(RETIREMENT).ravel()
    employed_arr = np.concatenate([ft_arr, pt_arr])

    rows: dict[str, float] = {}

    cg_mask = df["choice"].isin(informal)
    if not cg_mask.any():
        return rows

    first_cg_per = df.loc[cg_mask].groupby("agent")["period"].min()
    df_first = df.set_index(["agent", "period"]).loc[
        list(zip(first_cg_per.index, first_cg_per.values))
    ].reset_index()

    if df_first.empty:
        return rows

    def _entry_rows(
        sub: pd.DataFrame, prefix: str, age_label: str, hi: int,
    ) -> None:
        n = len(sub)
        if n == 0:
            rows[f"{prefix}Share prior FT ({age_label})"] = np.nan
            rows[f"{prefix}Share prior PT ({age_label})"] = np.nan
            rows[f"{prefix}Share prior unemp ({age_label})"] = np.nan
            if hi >= 63:
                rows[f"{prefix}Share prior retired ({age_label})"] = np.nan
            rows[f"{prefix}Share had job ({age_label})"] = np.nan
            rows[f"{prefix}Share no prior job ({age_label})"] = np.nan
            rows[f"{prefix}N entrants ({age_label})"] = 0.0
            return

        lag = sub["lagged_choice"]
        rows[f"{prefix}N entrants ({age_label})"] = float(n)
        rows[f"{prefix}Share prior FT ({age_label})"] = float(
            lag.isin(ft_arr).mean()
        )
        rows[f"{prefix}Share prior PT ({age_label})"] = float(
            lag.isin(pt_arr).mean()
        )
        rows[f"{prefix}Share prior unemp ({age_label})"] = float(
            lag.isin(unemp_arr).mean()
        )
        if hi >= 63:
            rows[f"{prefix}Share prior retired ({age_label})"] = float(
                lag.isin(ret_arr).mean()
            )
        had_job = lag.isin(employed_arr)
        rows[f"{prefix}Share had job ({age_label})"] = float(had_job.mean())
        rows[f"{prefix}Share no prior job ({age_label})"] = float(
            (~had_job).mean()
        )

    # -- Overall --
    for age_label, lo, hi in CG_ENTRY_AGE_BINS:
        sub = df_first[(df_first["age"] >= lo) & (df_first["age"] <= hi)]
        _entry_rows(sub, "Entry: ", age_label, hi)

    # -- By education --
    for edu_level, label in EDU_LABELS.items():
        edu_df = df_first[df_first["education"] == edu_level]
        for age_label, lo, hi in CG_ENTRY_AGE_BINS:
            sub = edu_df[(edu_df["age"] >= lo) & (edu_df["age"] <= hi)]
            _entry_rows(sub, f"Entry [{label}]: ", age_label, hi)

    return rows


# ---------------------------------------------------------------------------
# Panels AA, BB, CC — Care demand heterogeneity & arrangement substitution
# ---------------------------------------------------------------------------

CARE_DEMAND_AGE_BINS = [
    ("40--49", 40, 49),
    ("50--54", 50, 54),
    ("55--59", 55, 59),
    ("60--67", 60, 67),
    ("All (40--67)", 40, 67),
]


def _classify_entrants(df: pd.DataFrame):
    """Identify first-time informal CG entrants and classify prior job status.

    Returns (df_first, had_job_agents, no_job_agents) where df_first contains
    one row per CG entrant at their first informal-care period.
    ``had_job_agents`` / ``no_job_agents`` are sets of agent ids.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()
    ft_arr = np.asarray(FULL_TIME).ravel()
    pt_arr = np.asarray(PART_TIME).ravel()
    employed_arr = np.concatenate([ft_arr, pt_arr])

    cg_mask = df["choice"].isin(informal)
    if not cg_mask.any():
        return pd.DataFrame(), set(), set()

    first_cg_per = df.loc[cg_mask].groupby("agent")["period"].min()
    df_first = (
        df.set_index(["agent", "period"])
        .loc[list(zip(first_cg_per.index, first_cg_per.values))]
        .reset_index()
    )
    if df_first.empty:
        return df_first, set(), set()

    had_job_mask = df_first["lagged_choice"].isin(employed_arr)
    had_job_agents = set(df_first.loc[had_job_mask, "agent"])
    no_job_agents = set(df_first.loc[~had_job_mask, "agent"])
    return df_first, had_job_agents, no_job_agents


def panel_care_demand_duration(df: pd.DataFrame) -> dict[str, float]:
    """Care demand and caregiving duration profile by prior job status.

    For CG type 1 entrants (first-time informal caregivers), compares agents
    who had a job before caregiving vs those who did not.  Reports care demand
    onset, duration, informal CG duration, formal care usage, combination care,
    and the informal-care coverage ratio.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()
    light_inf = np.asarray(LIGHT_INFORMAL_CARE).ravel()
    intensive_inf = np.asarray(INTENSIVE_INFORMAL_CARE).ravel()
    formal = np.asarray(FORMAL_CARE).ravel()

    rows: dict[str, float] = {}

    type1 = df[df["caregiving_type"] == 1] if "caregiving_type" in df.columns else df
    df_first, had_job_agents, no_job_agents = _classify_entrants(type1)
    if df_first.empty:
        return rows

    all_entrant_agents = had_job_agents | no_job_agents
    age_offset = int(type1["age"].iloc[0] - type1["period"].iloc[0])

    first_demand_per = (
        type1.loc[type1["care_demand"] > 0].groupby("agent")["period"].min()
    )
    first_demand_age = first_demand_per + age_offset

    demand_years = (type1["care_demand"] > 0).groupby(type1["agent"]).sum()
    cg_years = type1["choice"].isin(informal).groupby(type1["agent"]).sum()

    demand_periods = type1[type1["care_demand"] > 0]
    formal_years = (
        demand_periods["choice"].isin(formal).groupby(demand_periods["agent"]).sum()
    )
    combo_mask = demand_periods["choice"].isin(light_inf) & (
        demand_periods["care_demand"] == CARE_DEMAND_INTENSIVE
    )
    combo_years = combo_mask.groupby(demand_periods["agent"]).sum()

    first_cg_per_agent = df_first.set_index("agent")["period"]

    groups = [
        ("Had job", had_job_agents),
        ("No prior job", no_job_agents),
        ("All entrants", all_entrant_agents),
    ]

    def _emit(prefix: str, agents: set) -> None:
        agent_list = list(agents)
        n = len(agent_list)
        rows[f"{prefix}N agents"] = float(n)
        if n == 0:
            return

        fda = first_demand_age.reindex(agent_list).dropna()
        rows[f"{prefix}Mean age 1st care demand"] = (
            float(fda.mean()) if not fda.empty else np.nan
        )

        entrant_sub = df_first[df_first["agent"].isin(agents)]
        rows[f"{prefix}Mean age 1st CG"] = (
            float(entrant_sub["age"].mean()) if not entrant_sub.empty else np.nan
        )

        gap = (
            first_cg_per_agent.reindex(agent_list)
            - first_demand_per.reindex(agent_list)
        ).dropna()
        rows[f"{prefix}Mean gap demand-to-CG (yrs)"] = (
            float(gap.mean()) if not gap.empty else np.nan
        )

        d_yrs = demand_years.reindex(agent_list).fillna(0)
        c_yrs = cg_years.reindex(agent_list).fillna(0)
        f_yrs = formal_years.reindex(agent_list).fillna(0)
        cb_yrs = combo_years.reindex(agent_list).fillna(0)

        rows[f"{prefix}Mean care demand yrs"] = float(d_yrs.mean())
        rows[f"{prefix}Mean informal CG yrs"] = float(c_yrs.mean())
        rows[f"{prefix}Mean formal care yrs"] = float(f_yrs.mean())
        rows[f"{prefix}Mean combo care yrs"] = float(cb_yrs.mean())

        coverage = c_yrs / d_yrs.replace(0, np.nan)
        rows[f"{prefix}Coverage (CG/demand)"] = float(coverage.mean())

        n_sub = len(entrant_sub)
        if n_sub > 0:
            rows[f"{prefix}Share light demand at 1st CG"] = float(
                (entrant_sub["care_demand"] == CARE_DEMAND_LIGHT).mean()
            )
            rows[f"{prefix}Share intens. demand at 1st CG"] = float(
                (entrant_sub["care_demand"] == CARE_DEMAND_INTENSIVE).mean()
            )

    for label, agent_set in groups:
        _emit(f"AA [{label}]: ", agent_set)

    for edu_level, edu_label in EDU_LABELS.items():
        edu_first = df_first[df_first["education"] == edu_level]
        edu_had = had_job_agents & set(edu_first["agent"])
        edu_no = no_job_agents & set(edu_first["agent"])
        for label, agent_set in [("Had job", edu_had), ("No prior job", edu_no)]:
            _emit(f"AA [{edu_label}, {label}]: ", agent_set)

    for age_label, lo, hi in CARE_DEMAND_AGE_BINS:
        demand_in_bin = first_demand_age[
            (first_demand_age >= lo) & (first_demand_age <= hi)
        ]
        bin_agents = set(demand_in_bin.index) & all_entrant_agents
        bin_had = bin_agents & had_job_agents
        bin_no = bin_agents & no_job_agents
        for label, agent_set in [
            ("All", bin_agents),
            ("Had job", bin_had),
            ("No prior job", bin_no),
        ]:
            _emit(f"AA [1st demand {age_label}, {label}]: ", agent_set)

    return rows


def panel_care_arrangements(df: pd.DataFrame) -> dict[str, float]:
    """Care arrangement choices by demand intensity and prior job status.

    For CG type 1 agents with care_demand > 0, reports the share choosing
    each care arrangement.  When care_demand == 2 (intensive), LIGHT_INFORMAL
    is reported as combination care.
    """
    light_inf = np.asarray(LIGHT_INFORMAL_CARE).ravel()
    intensive_inf = np.asarray(INTENSIVE_INFORMAL_CARE).ravel()
    formal = np.asarray(FORMAL_CARE).ravel()

    rows: dict[str, float] = {}

    type1 = df[df["caregiving_type"] == 1] if "caregiving_type" in df.columns else df
    _, had_job_agents, no_job_agents = _classify_entrants(type1)
    all_entrant_agents = had_job_agents | no_job_agents

    with_demand = type1[type1["care_demand"] > 0]
    if with_demand.empty:
        return rows

    entrant_demand = with_demand[with_demand["agent"].isin(all_entrant_agents)]

    def _care_shares(sub: pd.DataFrame, prefix: str) -> None:
        n = len(sub)
        if n == 0:
            return

        light_demand = sub[sub["care_demand"] == CARE_DEMAND_LIGHT]
        intens_demand = sub[sub["care_demand"] == CARE_DEMAND_INTENSIVE]

        rows[f"{prefix}N person-periods (demand>0)"] = float(n)
        rows[f"{prefix}Share light demand"] = float(len(light_demand) / n)
        rows[f"{prefix}Share intens. demand"] = float(len(intens_demand) / n)

        nl = len(light_demand)
        if nl > 0:
            rows[f"{prefix}Light: Sh. light informal"] = float(
                light_demand["choice"].isin(light_inf).mean()
            )
            rows[f"{prefix}Light: Sh. formal"] = float(
                light_demand["choice"].isin(formal).mean()
            )

        ni = len(intens_demand)
        if ni > 0:
            rows[f"{prefix}Intens: Sh. intensive informal"] = float(
                intens_demand["choice"].isin(intensive_inf).mean()
            )
            rows[f"{prefix}Intens: Sh. combo care (light inf)"] = float(
                intens_demand["choice"].isin(light_inf).mean()
            )
            rows[f"{prefix}Intens: Sh. formal"] = float(
                intens_demand["choice"].isin(formal).mean()
            )

    groups = [
        ("All entrants", all_entrant_agents),
        ("Had job", had_job_agents),
        ("No prior job", no_job_agents),
    ]
    for label, agent_set in groups:
        sub = entrant_demand[entrant_demand["agent"].isin(agent_set)]
        _care_shares(sub, f"BB [{label}]: ")

    for edu_level, edu_label in EDU_LABELS.items():
        for label, agent_set in [("Had job", had_job_agents), ("No prior job", no_job_agents)]:
            edu_sub = entrant_demand[
                entrant_demand["agent"].isin(agent_set)
                & (entrant_demand["education"] == edu_level)
            ]
            _care_shares(edu_sub, f"BB [{edu_label}, {label}]: ")

    for age_label, lo, hi in CARE_DEMAND_AGE_BINS:
        age_sub = entrant_demand[
            (entrant_demand["age"] >= lo) & (entrant_demand["age"] <= hi)
        ]
        for label, agent_set in [
            ("All entrants", all_entrant_agents),
            ("Had job", had_job_agents),
            ("No prior job", no_job_agents),
        ]:
            sub = age_sub[age_sub["agent"].isin(agent_set)]
            _care_shares(sub, f"BB [{age_label}, {label}]: ")

    return rows


def panel_care_demand_onset(df: pd.DataFrame) -> dict[str, float]:
    """Care demand onset heterogeneity across all caregiving types.

    Reports the composition of agents experiencing care demand (CG types 0
    and 1), and for type 1, whether they ever become informal caregivers.
    """
    informal = np.asarray(INFORMAL_CARE).ravel()
    rows: dict[str, float] = {}

    ever_demand = df.loc[df["care_demand"] > 0, "agent"].unique()
    n_ever_demand = len(ever_demand)
    if n_ever_demand == 0:
        return rows

    rows["CC: N agents ever care demand"] = float(n_ever_demand)

    has_cg_type = "caregiving_type" in df.columns
    if has_cg_type:
        type_of = df.groupby("agent")["caregiving_type"].first()
        demand_types = type_of.reindex(ever_demand)
        rows["CC: Share CG type 1"] = float((demand_types == 1).mean())
        rows["CC: Share CG type 0"] = float((demand_types == 0).mean())

        type1_demand = set(demand_types[demand_types == 1].index)
        ever_cg = set(df.loc[df["choice"].isin(informal), "agent"].unique())
        type1_became_cg = type1_demand & ever_cg
        n_type1 = len(type1_demand)
        if n_type1 > 0:
            rows["CC: Type 1 — share became CG"] = float(
                len(type1_became_cg) / n_type1
            )
            rows["CC: Type 1 — share only formal"] = float(
                1.0 - len(type1_became_cg) / n_type1
            )

    first_demand_per = df.loc[df["care_demand"] > 0].groupby("agent")["period"].min()
    if "age" in df.columns and not first_demand_per.empty:
        age_offset = int(df["age"].iloc[0] - df["period"].iloc[0])
        first_demand_ages = first_demand_per + age_offset
        demand_agent_ages = first_demand_ages.reindex(ever_demand).dropna()
        rows["CC: Mean age 1st care demand"] = float(demand_agent_ages.mean())

    demand_years = (df["care_demand"] > 0).groupby(df["agent"]).sum()
    rows["CC: Mean total demand yrs"] = float(
        demand_years.reindex(ever_demand).fillna(0).mean()
    )

    for edu_level, label in EDU_LABELS.items():
        edu_agents = set(
            df.loc[df["education"] == edu_level, "agent"].unique()
        ) & set(ever_demand)
        n_edu = len(edu_agents)
        rows[f"CC [{label}]: N agents"] = float(n_edu)
        if n_edu > 0:
            rows[f"CC [{label}]: Mean demand yrs"] = float(
                demand_years.reindex(list(edu_agents)).fillna(0).mean()
            )
            if has_cg_type:
                edu_types = type_of.reindex(list(edu_agents))
                rows[f"CC [{label}]: Share CG type 1"] = float(
                    (edu_types == 1).mean()
                )
                type1_edu = set(edu_types[edu_types == 1].index)
                type1_edu_cg = type1_edu & ever_cg
                if type1_edu:
                    rows[f"CC [{label}]: Type 1 became CG"] = float(
                        len(type1_edu_cg) / len(type1_edu)
                    )

    return rows


def panel_duration_by_edu(df: pd.DataFrame, edu_level: int) -> dict[str, float]:
    """Condensed duration heterogeneity for one education group.

    Focuses on empl. rate (All 30--67 and 60--67) and retirement (63--67)
    to show whether the job retention channel benefits this group.
    """
    label = EDU_LABELS.get(edu_level, f"Edu {edu_level}")
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
        ("Empl. rate", EMPLOYED, [("All (30--67)", 30, 67), ("60--67", 60, 67)]),
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


def panel_pension_by_edu(
    df: pd.DataFrame,
    edu_level: int,
    wealth_unit: float,
    max_exps_period_working: np.ndarray,
    start_age: int,
) -> dict[str, float]:
    """Pension and experience at retirement for one education group."""
    label = EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    edu_df = df[df["education"] == edu_level]

    ret_raw = panel_g_retirement(edu_df, wealth_unit=wealth_unit)
    exp_raw = panel_h_experience_at_retirement(
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


def panel_lifecycle_by_edu(
    df: pd.DataFrame, edu_level: int, wealth_unit: float
) -> dict[str, float]:
    """Lifecycle outcomes for ever-caregivers in one education group.

    Reports avg. gross labor income (30--67), avg. gross pension income
    (retired periods), and avg. benefit per CG period.  The pct-change
    columns then directly show the policy impact on each.
    """
    label = EDU_LABELS.get(edu_level, f"Edu {edu_level}")
    informal = np.asarray(INFORMAL_CARE).ravel()
    retired_choices = np.asarray(RETIREMENT).ravel()

    edu_df = df[df["education"] == edu_level]
    ever_cg_agents = set(edu_df.loc[edu_df["choice"].isin(informal), "agent"].unique())
    ever_cg = edu_df[edu_df["agent"].isin(ever_cg_agents)]

    rows: dict[str, float] = {}

    has_cons = "consumption" in df.columns

    nan_keys = [
        f"Avg. gross labor inc. (ever CG, 30--67) [{label}]",
        f"Avg. gross pension inc. (ever CG, ret) [{label}]",
        f"Avg. benefit per CG period [{label}]",
        f"Avg. consumption (ever CG, <63) [{label}]",
        f"Avg. consumption (ever CG, active CG) [{label}]",
        f"Avg. consumption (ever CG, 30--39) [{label}]",
        f"Avg. consumption (ever CG, 40--49) [{label}]",
        f"Avg. consumption (ever CG, 50--59) [{label}]",
        f"Avg. consumption (ever CG, retired) [{label}]",
    ]
    if ever_cg.empty:
        for k in nan_keys:
            rows[k] = np.nan
        return rows

    working_age = ever_cg[(ever_cg["age"] >= 30) & (ever_cg["age"] <= 67)]
    rows[f"Avg. gross labor inc. (ever CG, 30--67) [{label}]"] = (
        safe_mean(working_age["gross_labor_income"]) * wealth_unit
        if "gross_labor_income" in working_age.columns
        else np.nan
    )

    ret_periods = ever_cg[ever_cg["choice"].isin(retired_choices)]
    rows[f"Avg. gross pension inc. (ever CG, ret) [{label}]"] = (
        safe_mean(ret_periods["gross_retirement_income"]) * wealth_unit
        if "gross_retirement_income" in ret_periods.columns and not ret_periods.empty
        else np.nan
    )

    has_leave = "caregiving_leave_top_up" in ever_cg.columns
    has_pflegegeld = "care_benefits_and_costs" in ever_cg.columns

    if has_leave:
        benefit_col = "caregiving_leave_top_up"
    elif has_pflegegeld:
        benefit_col = "care_benefits_and_costs"
    else:
        benefit_col = None

    cg_periods = ever_cg[ever_cg["lagged_choice"].isin(informal)]
    if benefit_col and not cg_periods.empty:
        rows[f"Avg. benefit per CG period [{label}]"] = (
            safe_mean(cg_periods[benefit_col]) * wealth_unit
        )
    else:
        rows[f"Avg. benefit per CG period [{label}]"] = np.nan

    if has_leave and has_pflegegeld and not cg_periods.empty:
        pg_val = safe_mean(cg_periods["care_benefits_and_costs"]) * wealth_unit
        leave_val = safe_mean(cg_periods["caregiving_leave_top_up"]) * wealth_unit
        rows[f"Avg. Pflegegeld per CG period [{label}]"] = pg_val
        rows[f"Avg. total transfer per CG period [{label}]"] = leave_val + pg_val
    elif has_leave and has_pflegegeld:
        rows[f"Avg. Pflegegeld per CG period [{label}]"] = np.nan
        rows[f"Avg. total transfer per CG period [{label}]"] = np.nan

    under63 = ever_cg[ever_cg["age"] < 63]
    rows[f"Avg. consumption (ever CG, <63) [{label}]"] = (
        safe_mean(under63["consumption"]) * wealth_unit if has_cons else np.nan
    )

    rows[f"Avg. consumption (ever CG, active CG) [{label}]"] = (
        safe_mean(cg_periods["consumption"]) * wealth_unit
        if has_cons and not cg_periods.empty
        else np.nan
    )

    for bin_label, lo, hi in [
        ("30--39", 30, 39),
        ("40--49", 40, 49),
        ("50--59", 50, 59),
    ]:
        age_bin = ever_cg[(ever_cg["age"] >= lo) & (ever_cg["age"] <= hi)]
        rows[f"Avg. consumption (ever CG, {bin_label}) [{label}]"] = (
            safe_mean(age_bin["consumption"]) * wealth_unit
            if has_cons and not age_bin.empty
            else np.nan
        )

    ret_ever_cg = ever_cg[ever_cg["choice"].isin(retired_choices)]
    rows[f"Avg. consumption (ever CG, retired) [{label}]"] = (
        safe_mean(ret_ever_cg["consumption"]) * wealth_unit
        if has_cons and not ret_ever_cg.empty
        else np.nan
    )

    return rows
