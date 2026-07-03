"""Career-cost heterogeneity tables (agent side of the self-financing story).

Builds, per policy variant, the Adda-style career cost of caregiving
(``1 - NPV_no_care / NPV_policy``, negative = cost) on two income measures
(labor+pension, full disposable income), a small discount-factor (beta) sweep, two
caregiver populations (baseline-fixed vs per-policy), and the SF heterogeneity grid
(education x first-care-demand age bin). Emits a compact summary, a heterogeneity
table, and a discount-sensitivity table, compiled into a dedicated "Agent side"
section of ``sf_report.pdf``.

Pipeline (mirrors the self-financing pipeline):
1. ``task_aggregate_career_costs_heterogeneity`` -> long-format pickle.
2. ``task_write_career_cost_tables`` -> LaTeX tables.
3. The report compile step picks up the new section.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.tables.publication._career_cost_ledger import (
    BETA_LABELS,
    BETAS,
    LEGACY_BETA,
    MEASURE_LABELS,
    MEASURES,
    PATH_NO_CARE_DEMAND,
    PATHS_BY_KEY,
    POLICY_KEYS,
    POLICY_LABELS_SHORT,
    PRIMARY_BETA,
    compute_agent_npvs,
    ever_caregiver_ids,
    load_cc_slim,
)
from caregiving.tables.publication._self_financing_ledger import (
    build_subgroup_table,
    load_slim,
    load_specs,
)

COUNTERFACTUALS: tuple[str, ...] = tuple(p for p in POLICY_KEYS if p != "baseline")
POPULATIONS: tuple[str, ...] = ("baseline_fixed", "per_policy")
POPULATION_LABELS: dict[str, str] = {
    "baseline_fixed": "Baseline caregivers (fixed)",
    "per_policy": "Per-policy caregivers",
}

# (education, age_bin, column label) -- identical grid to the SF tables.
SUBGROUP_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("all", "all", "Total"),
    ("low", "all", "Low: Total"),
    ("low", "40-49", "Low: 40-49"),
    ("low", "50-59", "Low: 50-59"),
    ("low", "60-69", "Low: 60-69"),
    ("high", "all", "High: Total"),
    ("high", "40-49", "High: 40-49"),
    ("high", "50-59", "High: 50-59"),
    ("high", "60-69", "High: 60-69"),
)
MIN_CELL_N: int = 100
NPV_TOL: float = 1e-10

CC_DIR: Path = BLD / "tables" / "publication" / "self_financing_heterogeneity"
AGG_PATH: Path = BLD / "derived" / "career_costs_heterogeneity_aggregates.pkl"

# Regression anchor: baseline labor+pension career cost at beta=0.97, Total.
EXISTING_BASELINE_LABPEN_PCT: float = -37.1
ANCHOR_TOL_PCT: float = 1.5


# =============================================================================
# Aggregator
# =============================================================================


def _subgroup_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = [("all", "all")]
    for edu in ("low", "high"):
        keys.append((edu, "all"))
        for ab in ("40-49", "50-59", "60-69"):
            keys.append((edu, ab))
    return keys


def _subgroup_filter(
    edu_label: np.ndarray, age_bin: np.ndarray, education: str, age_bin_key: str
) -> np.ndarray:
    sel = np.ones(len(edu_label), dtype=bool)
    if education != "all":
        sel = sel & (edu_label == education)
    if age_bin_key != "all":
        sel = sel & (age_bin == age_bin_key)
    return sel


def _aggregate_policy(
    policy: str,
    policy_npv: pd.DataFrame,
    nocare_npv: pd.DataFrame,
    policy_cg_ids: np.ndarray,
    baseline_cg_ids: np.ndarray,
    sg: pd.DataFrame,
) -> list[dict]:
    """All (population, subgroup, measure, beta) career-cost rows for one policy."""
    merged = policy_npv.merge(nocare_npv, on="agent", suffixes=("_pol", "_nc"))
    sg_view = sg.reindex(merged["agent"].to_numpy())
    edu_label = sg_view["education_label"].to_numpy()
    age_bin = sg_view["age_bin"].to_numpy()
    agents = merged["agent"].to_numpy()

    records: list[dict] = []
    for population, cg_ids in (
        ("baseline_fixed", baseline_cg_ids),
        ("per_policy", policy_cg_ids),
    ):
        in_pop = np.isin(agents, cg_ids)
        for education, age_bin_key in _subgroup_keys():
            sel = in_pop & _subgroup_filter(edu_label, age_bin, education, age_bin_key)
            n = int(sel.sum())
            for measure in MEASURES:
                for beta_key in BETAS:
                    col_pol = f"npv_{measure}_{beta_key}_pol"
                    col_nc = f"npv_{measure}_{beta_key}_nc"
                    pol_vals = merged.loc[sel, col_pol].to_numpy()
                    nc_vals = merged.loc[sel, col_nc].to_numpy()
                    mean_pol = float(np.mean(pol_vals)) if n else np.nan
                    mean_nc = float(np.mean(nc_vals)) if n else np.nan
                    cc_rom = (
                        1.0 - mean_nc / mean_pol
                        if (n and mean_pol and abs(mean_pol) > NPV_TOL)
                        else np.nan
                    )
                    pos = pol_vals > NPV_TOL
                    per_agent = 1.0 - nc_vals[pos] / pol_vals[pos]
                    cc_mor = float(np.mean(per_agent)) if pos.any() else np.nan
                    cc_med = float(np.median(per_agent)) if pos.any() else np.nan
                    records.append(
                        {
                            "policy": policy,
                            "population": population,
                            "measure": measure,
                            "beta": beta_key,
                            "education": education,
                            "age_bin": age_bin_key,
                            "n_agents": n,
                            "mean_npv_policy": mean_pol,
                            "mean_npv_no_care": mean_nc,
                            "cc_ratio_of_means": cc_rom,
                            "cc_mean_of_ratios": cc_mor,
                            "cc_median_of_ratios": cc_med,
                        }
                    )
    return records


@pytask.task(id="aggregate_career_costs_heterogeneity")
def task_aggregate_career_costs_heterogeneity(
    path_to_aggregates: Annotated[Path, Product] = AGG_PATH,
) -> None:
    """Load the 7 policy pickles + the no-care reference (one at a time) and write
    the long-format career-cost aggregate keyed by
    ``(policy, population, measure, beta, education, age_bin)``."""
    inputs = load_specs()
    # Subgroup labels + baseline caregiver set, from the baseline sim.
    baseline_slim = load_slim(PATHS_BY_KEY["baseline"])
    sg = build_subgroup_table(baseline_slim, inputs.start_age)
    del baseline_slim
    gc.collect()

    nocare_npv = compute_agent_npvs(load_cc_slim(PATH_NO_CARE_DEMAND))
    gc.collect()

    baseline_df = load_cc_slim(PATHS_BY_KEY["baseline"])
    baseline_npv = compute_agent_npvs(baseline_df)
    baseline_cg_ids = ever_caregiver_ids(baseline_df)
    del baseline_df
    gc.collect()

    rows: list[dict] = []
    for policy in POLICY_KEYS:
        if policy == "baseline":
            policy_npv, policy_cg_ids = baseline_npv, baseline_cg_ids
        else:
            df = load_cc_slim(PATHS_BY_KEY[policy])
            policy_npv = compute_agent_npvs(df)
            policy_cg_ids = ever_caregiver_ids(df)
            del df
            gc.collect()
        rows.extend(
            _aggregate_policy(
                policy, policy_npv, nocare_npv, policy_cg_ids, baseline_cg_ids, sg
            )
        )

    agg = pd.DataFrame(rows)
    path_to_aggregates.parent.mkdir(parents=True, exist_ok=True)
    agg.to_pickle(path_to_aggregates)

    # Anchor print (baseline labor+pension, beta 0.97, Total, per-policy == fixed).
    anchor = agg[
        (agg.policy == "baseline")
        & (agg.measure == "labpen")
        & (agg.beta == LEGACY_BETA)
        & (agg.population == "baseline_fixed")
        & (agg.education == "all")
        & (agg.age_bin == "all")
    ]
    if not anchor.empty:
        pct = float(anchor["cc_mean_of_ratios"].iloc[0]) * 100
        diff = pct - EXISTING_BASELINE_LABPEN_PCT
        status = "OK" if abs(diff) <= ANCHOR_TOL_PCT else "DRIFT"
        print(
            f"[career costs] baseline labpen mean-of-ratios @ beta0.97 = {pct:.2f}% "
            f"(legacy {EXISTING_BASELINE_LABPEN_PCT:.1f}%, diff {diff:+.2f} pp -- "
            f"{status})",
            flush=True,
        )


# =============================================================================
# Writers
# =============================================================================


def _lookup(
    df: pd.DataFrame,
    *,
    policy: str,
    population: str,
    measure: str,
    beta: str,
    education: str,
    age_bin: str,
) -> pd.Series | None:
    sel = (
        (df["policy"] == policy)
        & (df["population"] == population)
        & (df["measure"] == measure)
        & (df["beta"] == beta)
        & (df["education"] == education)
        & (df["age_bin"] == age_bin)
    )
    sub = df[sel]
    return None if sub.empty else sub.iloc[0]


def _cc_level(row: pd.Series | None) -> float:
    """Headline career-cost statistic = per-agent MEAN-OF-RATIOS (reproduces the
    existing -37.1% headline). See ``_cc_diag`` for the robustness statistics."""
    if row is None or int(row.get("n_agents", 0)) < MIN_CELL_N:
        return float("nan")
    return float(row.get("cc_mean_of_ratios", float("nan")))


def _cc_diag(row: pd.Series | None) -> tuple[float, float]:
    """Robustness diagnostics: (ratio-of-means, median-of-ratios)."""
    if row is None or int(row.get("n_agents", 0)) < MIN_CELL_N:
        return float("nan"), float("nan")
    return (
        float(row.get("cc_ratio_of_means", float("nan"))),
        float(row.get("cc_median_of_ratios", float("nan"))),
    )


def _fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value * 100:.1f}\\%"


def _fmt_pp(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value * 100:+.1f}"


def _two_level_header() -> tuple[str, str]:
    top = (
        r"\multirow{2}{*}{Policy} & "
        r"\multicolumn{1}{c}{} & "
        r"\multicolumn{4}{c}{Low education} & "
        r"\multicolumn{4}{c}{High education} \\"
    )
    cols = " & ".join(
        ["Total"]
        + ["Total", "40-49", "50-59", "60-69"]
        + ["Total", "40-49", "50-59", "60-69"]
    )
    return top, rf" & {cols} \\"


# --- compact summary --------------------------------------------------------


def _compact_cc_body(df: pd.DataFrame) -> str:
    pop, beta, edu, ab = "per_policy", PRIMARY_BETA, "all", "all"
    base = {
        m: _cc_level(
            _lookup(
                df,
                policy="baseline",
                population=pop,
                measure=m,
                beta=beta,
                education=edu,
                age_bin=ab,
            )
        )
        for m in MEASURES
    }
    lines = [
        r"% Auto-generated by task_career_costs_heterogeneity.py. Do not edit.",
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Career cost of caregiving -- compact overview "
        r"(Total population, per-policy caregivers, $\beta=0.97$, per-agent "
        r"mean-of-ratios -- the existing headline). Negative = career cost; "
        r"$\Delta$ = pp change vs Baseline (positive = smaller cost).}",
        r"  \label{tab:cc_compact}",
        r"  \small",
        r"  \begin{tabular}{lrrrr}",
        r"  \toprule",
        r"  Policy & Lab+pen & $\Delta$ & Full & $\Delta$ \\",
        r"  \midrule",
    ]
    for policy in POLICY_KEYS:
        cells = []
        for m in MEASURES:
            lvl = _cc_level(
                _lookup(
                    df,
                    policy=policy,
                    population=pop,
                    measure=m,
                    beta=beta,
                    education=edu,
                    age_bin=ab,
                )
            )
            delta = (
                lvl - base[m]
                if not (math.isnan(lvl) or math.isnan(base[m]))
                else float("nan")
            )
            cells.extend(
                [_fmt_pct(lvl), "--" if policy == "baseline" else _fmt_pp(delta)]
            )
        lines.append(f"  {POLICY_LABELS_SHORT[policy]} & " + " & ".join(cells) + r" \\")
    lines.extend(
        (
            r"  \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{0.5em}",
            r"  \begin{minipage}{\textwidth}",
            r"  \scriptsize",
            r"  \textbf{Notes.} Career cost $=1-\mathrm{NPV_{no\ care}}/"
            r"\mathrm{NPV_{policy}}$, per-agent mean over each policy's "
            r"ever-caregivers, relative to a world where the parent never needs "
            r"care. \emph{Lab+pen} = own income after SSC minus the leave top-up, "
            r"plus bequest (net of SSC, gross of income tax; matches the existing "
            r"headline). \emph{Full} = model household disposable income excl.\ "
            r"capital interest (after tax, safety-net floor applied) + care cash "
            r"benefits + bequest. $\beta=0.97$; see the sensitivity table for "
            r"$\beta\in\{0.98\,(\text{model}),0.97,0.95\}$.",
            "",
            _cc_caveat_note(df),
            r"  \end{minipage}",
            r"\end{table}",
            "",
        )
    )
    return "\n".join(lines)


def _cc_caveat_note(df: pd.DataFrame) -> str:
    """One-paragraph robustness caveat: the no-care sim is not per-agent
    shock-aligned with the policy sims, so the per-agent mean is tail-sensitive;
    report the aggregate ratio-of-means and median as robustness."""
    base = _lookup(
        df,
        policy="baseline",
        population="per_policy",
        measure="labpen",
        beta=PRIMARY_BETA,
        education="all",
        age_bin="all",
    )
    rom, med = _cc_diag(base)
    return (
        r"  \textbf{Caveat (robustness).} The no-care counterfactual is not "
        r"per-agent shock-aligned with the policy simulations (different RNG "
        r"draw; incomes differ across worlds before care onset), so the "
        r"\emph{per-agent} ratio is dispersed and tail-/denominator-sensitive. "
        r"The headline per-agent mean is kept for continuity, but for the "
        r"baseline lab+pen cell the robust aggregate ratio-of-means is "
        rf"{_fmt_pct(rom)} and the median per-agent cost is {_fmt_pct(med)} "
        r"(vs the per-agent mean headline). Read the per-agent means as "
        r"upper-bound magnitudes, not point estimates."
    )


# --- heterogeneity ----------------------------------------------------------


def _cc_heterogeneity_body(df: pd.DataFrame, *, population: str, beta: str) -> str:
    pop_label = POPULATION_LABELS[population]
    beta_label = BETA_LABELS[beta]
    lines = [
        r"% Auto-generated by task_career_costs_heterogeneity.py. Do not edit.",
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{Career cost of caregiving by education $\times$ age at first "
        rf"care demand -- {pop_label}, $\beta={beta_label}$, per-agent "
        rf"mean-of-ratios. Each measure in its own panel; rows are the policy "
        rf"variants.}}",
        rf"  \label{{tab:cc_het_{population}_{beta}}}",
        r"  \scriptsize",
    ]
    top, bottom = _two_level_header()
    for measure in MEASURES:
        lines.extend(
            (
                rf"\par\medskip\noindent\textbf{{{MEASURE_LABELS[measure]}}}"
                r"\par\nopagebreak\smallskip",
                r"\begin{tabular}{l r r r r r r r r r}",
                r"\toprule",
                top,
                bottom,
                r"\midrule",
            )
        )
        for policy in POLICY_KEYS:
            cells = []
            for education, age_bin, _lab in SUBGROUP_COLUMNS:
                lvl = _cc_level(
                    _lookup(
                        df,
                        policy=policy,
                        population=population,
                        measure=measure,
                        beta=beta,
                        education=education,
                        age_bin=age_bin,
                    )
                )
                cells.append(_fmt_pct(lvl))
            lines.append(
                f"    {POLICY_LABELS_SHORT[policy]} & " + " & ".join(cells) + r" \\"
            )
        lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    lines.extend(
        (
            r"\vspace{0.5em}",
            r"\begin{minipage}{\textwidth}",
            r"\scriptsize",
            r"\textbf{Notes.} Career cost (negative = cost) by subgroup; per-agent "
            r"mean-of-ratios over the stated caregiver population. Cells with fewer "
            rf"than {MIN_CELL_N} caregivers print ``--''. Subgroups: education and age "
            r"at first parental care demand, fixed from the baseline simulation. "
            r"Measures as in the compact table.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        )
    )
    return "\n".join(lines)


# --- discount sensitivity ---------------------------------------------------


def _cc_discount_sensitivity_body(df: pd.DataFrame) -> str:
    pop, edu, ab = "per_policy", "all", "all"
    beta_order = ("b098", "b097", "b095")
    lines = [
        r"% Auto-generated by task_career_costs_heterogeneity.py. Do not edit.",
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Career cost (labor + pension) -- discount-factor sensitivity "
        r"(Total population, per-policy caregivers, per-agent mean-of-ratios).}",
        r"  \label{tab:cc_discount_sensitivity}",
        r"  \small",
        r"  \begin{tabular}{l" + "r" * len(beta_order) + "}",
        r"  \toprule",
        r"  Policy & " + " & ".join(BETA_LABELS[b] for b in beta_order) + r" \\",
        r"  \midrule",
    ]
    for policy in POLICY_KEYS:
        cells = [
            _fmt_pct(
                _cc_level(
                    _lookup(
                        df,
                        policy=policy,
                        population=pop,
                        measure="labpen",
                        beta=b,
                        education=edu,
                        age_bin=ab,
                    )
                )
            )
            for b in beta_order
        ]
        lines.append(f"  {POLICY_LABELS_SHORT[policy]} & " + " & ".join(cells) + r" \\")
    lines.extend(
        (
            r"  \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{0.5em}",
            r"  \begin{minipage}{\textwidth}",
            r"  \scriptsize",
            r"  \textbf{Notes.} Career cost (labor + pension) at discount factors "
            r"$\beta\in\{0.98,0.97,0.95\}$ (per-period weight $\beta^{age-40}$). "
            r"$\beta=0.98$ is the model's own discount factor (primary).",
            r"  \end{minipage}",
            r"\end{table}",
            "",
        )
    )
    return "\n".join(lines)


CC_COMPACT_TEX: Path = CC_DIR / "cc_compact.tex"
CC_HET_TEX: Path = CC_DIR / "cc_heterogeneity.tex"
CC_HET_BASEFIXED_TEX: Path = CC_DIR / "cc_heterogeneity_baseline_fixed.tex"
CC_DISCOUNT_TEX: Path = CC_DIR / "cc_discount_sensitivity.tex"


@pytask.task(id="write_career_cost_tables")
def task_write_career_cost_tables(
    path_to_aggregates: Path = AGG_PATH,
    path_to_compact: Annotated[Path, Product] = CC_COMPACT_TEX,
    path_to_het: Annotated[Path, Product] = CC_HET_TEX,
    path_to_het_per_policy: Annotated[Path, Product] = CC_HET_BASEFIXED_TEX,
    path_to_discount: Annotated[Path, Product] = CC_DISCOUNT_TEX,
) -> None:
    """Write the career-cost LaTeX tables from the aggregate pickle."""
    df = pd.read_pickle(path_to_aggregates)
    path_to_compact.parent.mkdir(parents=True, exist_ok=True)
    path_to_compact.write_text(_compact_cc_body(df), encoding="utf-8")
    # Headline heterogeneity = per-policy caregivers (existing convention);
    # baseline-fixed population is the robustness twin (removes composition).
    path_to_het.write_text(
        _cc_heterogeneity_body(df, population="per_policy", beta=PRIMARY_BETA),
        encoding="utf-8",
    )
    path_to_het_per_policy.write_text(
        _cc_heterogeneity_body(df, population="baseline_fixed", beta=PRIMARY_BETA),
        encoding="utf-8",
    )
    path_to_discount.write_text(_cc_discount_sensitivity_body(df), encoding="utf-8")
