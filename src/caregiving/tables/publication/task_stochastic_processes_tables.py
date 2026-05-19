"""Publication LaTeX tables for stochastic process estimation results.

Reads existing estimation outputs (CSV/pkl) from BLD/estimation/stochastic_processes/
and produces one clean LaTeX table per process. Tables include coefficients,
standard errors where available, R²/pseudo R² where available, and notes.
All tables are women-only where the process is estimated by sex.
"""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD

MIN_PARTNER_STATE_COLS = 6
N_CATEGORICAL_INDEX_COLS = 2

_EST = BLD / "estimation" / "stochastic_processes"
_OUT = BLD / "tables" / "publication" / "stochastic_processes"
_SAMPLE_SIZES_PATH = _EST / "sample_sizes.json"

# Paths used in task_write_specs (inheritance)
_PATH_INHERITANCE_PROB_SPEC7 = (
    _EST
    / "inheritance_specs"
    / "spec7_any_care_this_year_filter_parent_this_year_params.csv"
)
_PATH_INHERITANCE_AMOUNT_SPEC12 = (
    _EST
    / "inheritance_amount_specs_two_care"
    / "spec12_care_recent_filter_parent_recent_params.csv"
)


def _load_sample_sizes() -> dict:
    if _SAMPLE_SIZES_PATH.exists():
        with _SAMPLE_SIZES_PATH.open() as f:
            return json.load(f)
    return {}


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters in table content."""
    for c, r in (
        ("_", "\\_"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
    ):
        s = s.replace(c, r)
    return s


def _format_n(n: int | None, n_cols: int = 2) -> str:
    """Format a sample size row for LaTeX, spanning n_cols data columns."""
    if n is None:
        return ""
    return f"$N$ & \\multicolumn{{{n_cols}}}{{c}}{{{n:,}}} \\\\"


def _notes_common():
    return (
        "\\\\\n\\multicolumn{3}{l}{\\footnotesize Notes: SOEP; "
        "structural estimation sample. Women only unless otherwise noted.}"
    )


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_partner_transition_women(
    path_to_partner_transition: Path = _EST / "partner_transition_matrix.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "partner_transition_women.tex",
) -> None:
    """LaTeX table: partner state transition probabilities.

    Women, by education and age bin.
    """
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path_to_partner_transition)
    # CSV may have index columns: sex, education, age_bin,
    # lagged_partner_state, lead_partner_state
    # and column proportion (or last column)
    if "sex" not in df.columns and len(df.columns) >= MIN_PARTNER_STATE_COLS:
        df.columns = [
            "sex",
            "education",
            "age_bin",
            "lagged_partner_state",
            "lead_partner_state",
            "proportion",
        ]
    if "sex" in df.columns:
        df = df.loc[df["sex"] == 1].copy()
    pivot_col = "proportion" if "proportion" in df.columns else df.columns[-1]
    lead_col = (
        "lead_partner_state" if "lead_partner_state" in df.columns else "partner_state"
    )
    if lead_col not in df.columns:
        lead_col = [
            c
            for c in df.columns
            if c not in ("education", "age_bin", "lagged_partner_state", pivot_col)
        ]
        lead_col = lead_col[0] if lead_col else "lead_partner_state"
    df_wide = df.pivot_table(
        index=["education", "age_bin", "lagged_partner_state"],
        columns=lead_col,
        values=pivot_col,
        aggfunc="first",
    ).reset_index()
    df_wide["education"] = df_wide["education"].map({0: "Low", 1: "High"})
    df_wide["lagged_partner_state"] = df_wide["lagged_partner_state"].map(
        {0: "No partner", 1: "Working", 2: "Retired"}
    )
    to_cols = [
        c
        for c in df_wide.columns
        if c not in ("education", "age_bin", "lagged_partner_state")
    ]
    n_to = len(to_cols)
    header = (
        "\\begin{tabular}{lll" + "c" * n_to + "}\n\\toprule\n"
        "Education & Age bin & From state & "
        + " & ".join(f"To {int(c)}" for c in to_cols)
        + " \\\\\n\\midrule\n"
    )
    rows = []
    for _, r in df_wide.iterrows():
        vals = [
            str(r["education"]),
            str(int(r["age_bin"])),
            str(r["lagged_partner_state"]),
        ]
        for c in to_cols:
            v = r[c]
            vals.append(f"{v:.3f}" if pd.notna(v) else "--")
        rows.append(" & ".join(vals) + " \\\\")
    body = "\n".join(rows)
    ss = _load_sample_sizes()
    n_val = ss.get("partner_transition_women_total")
    n_row = ""
    if n_val:
        n_row = f"\n\\midrule\n$N$ & \\multicolumn{{{2 + n_to}}}{{c}}{{{n_val:,}}} \\\\"
    notes = (
        f"{n_row}\n\\bottomrule\n\\end{{tabular}}\n"
        f"\\\\\n\\multicolumn{{{3 + n_to}}}{{l}}"
        "{\\footnotesize Notes: Non-parametric transition "
        "P(partner state next year | lagged state). "
        "Age bins in 10-year intervals. Women only. SOEP.}}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_partner_wage_women(
    path_to_params: Path = _EST / "partner_wage_eq_params_women.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "partner_wage_women.tex",
) -> None:
    """LaTeX table: partner (male) wage equation coefficients.

    OLS by education. Women's partners.
    """
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path_to_params, index_col=0)
    df.index = df.index.map(lambda x: _latex_escape(str(x)))
    coef_cols = [
        c for c in df.columns if c in ("constant", "period", "period_sq", "const")
    ]
    if not coef_cols:
        coef_cols = list(df.columns)
    header = (
        "\\begin{tabular}{lc}\n\\toprule\nParameter & Coefficient \\\\\n\\midrule\n"
    )
    rows = []
    for edu in df.index:
        rows.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{edu}}}}} \\\\")
        for c in coef_cols:
            v = df.loc[edu, c]
            rows.append(f"\\quad {_latex_escape(c)} & {v:.4f} \\\\")
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])  # drop last addlinespace
    ss = _load_sample_sizes()
    n_val = ss.get("partner_wage_women_total")
    n_row = ""
    if n_val:
        n_row = f"\n\\midrule\n$N$ & {n_val:,} \\\\"
    notes = (
        f"{n_row}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\\\\n\\multicolumn{2}{l}{\\footnotesize Notes: "
        "OLS monthly gross wage (partner). "
        "Agents are women; coefficients for male partners. "
        "By education. "
        "Specification: wage = const + period + period\\_sq. SOEP.}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_number_of_children_women(
    path_to_params: Path = _EST / "nb_children_estimates.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "number_of_children_women.tex",
) -> None:
    """LaTeX table: number of children in household.

    OLS by education and partner status. Women only.
    """
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path_to_params, index_col=[0, 1, 2])
    df.index.names = ["sex", "education", "has_partner"]
    df = df.loc[(1, slice(None), slice(None)), :].droplevel(0)  # women only
    edu_map = {0: "Low", 1: "High"}
    partner_map = {0: "Single", 1: "Partnered"}
    header = (
        "\\begin{tabular}{llccc}\n\\toprule\n"
        "Education & Partner & Const & Period & Period²"
        " \\\\\n\\midrule\n"
    )
    rows = []
    for (edu, has_partner), r in df.iterrows():
        edu_lab = edu_map.get(edu, str(edu))
        part_lab = partner_map.get(has_partner, str(has_partner))
        rows.append(
            f"{edu_lab} & {part_lab} & {r.get('const', r.iloc[0]):.4f} & "
            f"{r.get('period', r.iloc[1]):.4f} & "
            f"{r.get('period_sq', r.iloc[2]):.4f} \\\\"
        )
    body = "\n".join(rows)
    ss = _load_sample_sizes()
    n_total = sum(
        ss.get(f"children_women_{e}_{p}", 0)
        for e in ("low", "high")
        for p in ("single", "partnered")
    )
    n_row = ""
    if n_total > 0:
        n_row = f"\n\\midrule\n$N$ & \\multicolumn{{4}}{{c}}{{{n_total:,}}} \\\\"
    notes = (
        f"{n_row}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\\\\n\\multicolumn{5}{l}{\\footnotesize Notes: "
        "OLS: children = const + period + "
        "period\\textsuperscript{2}. "
        "By education and partner status. Women only. SOEP.}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_job_separation_women(
    path_to_params: Path = _EST / "job_sep_params.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "job_separation_women.tex",
) -> None:
    """LaTeX table: job separation logit coefficients (women, by education)."""
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path_to_params, index_col=[0, 1])
    df = df.loc[("Women", slice(None)), :].droplevel(0)
    header = (
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Education & Const & Age & Age² \\\\\n\\midrule\n"
    )
    rows = [
        f"{_latex_escape(str(edu))} & {r.get('const', r.iloc[0]):.4f} & "
        f"{r.get('age', r.iloc[1]):.4f} & {r.get('age_sq', r.iloc[2]):.4f} \\\\"
        for edu, r in df.iterrows()
    ]
    body = "\n".join(rows)
    ss = _load_sample_sizes()
    n_val = ss.get("job_sep_women_total")
    n_row = ""
    if n_val:
        n_row = f"\n\\midrule\n$N$ & \\multicolumn{{3}}{{c}}{{{n_val:,}}} \\\\"
    notes = (
        f"{n_row}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\\\\n\\multicolumn{4}{l}{\\footnotesize Notes: Logit P(job separation). "
        "Women only. SOEP. Standard errors from estimation.}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_health_transition_women(
    path_to_health: Path = _EST / "health_transition_matrix.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "health_transition_women.tex",
) -> None:
    """LaTeX table: health transition probabilities.

    Women, by education and age. Sample of ages.
    """
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    if not path_to_health.exists():
        path_to_save.write_text(
            "% Health transition matrix not found\n"
            "\\begin{tabular}{l}\n\\toprule\n"
            "(not found)\n\\bottomrule\n"
            "\\end{tabular}\n",
            encoding="utf-8",
        )
        return
    df = pd.read_csv(path_to_health)
    # Filter women (sex == 1 or "Women")
    if "sex" in df.columns:
        df = df.loc[
            df["sex"].astype(str).str.contains("Women|1", na=False, regex=True)
        ].copy()
    # Select key columns: education, period/age, health, lead_health, transition_prob
    prob_col = "transition_prob" if "transition_prob" in df.columns else df.columns[-1]
    header = (
        "\\begin{tabular}{lllcc}\n\\toprule\n"
        "Education & Age & From health & To health "
        "& Probability \\\\\n\\midrule\n"
    )
    rows = []
    for _, r in df.head(80).iterrows():  # limit rows for readability
        edu = r.get("education", r.iloc[0])
        period = r.get("period", r.get("age", ""))
        h = r.get("health", "")
        lh = r.get("lead_health", "")
        p = r.get(prob_col, np.nan)
        if pd.notna(p):
            rows.append(f"{edu} & {period} & {h} & {lh} & {p:.3f} \\\\")
    body = "\n".join(rows) if rows else "\\multicolumn{5}{l}{(no data)} \\\\"
    ss = _load_sample_sizes()
    n_val = ss.get("health_women_total")
    n_row = ""
    if n_val:
        n_row = f"\n\\midrule\n$N$ & \\multicolumn{{4}}{{c}}{{{n_val:,}}} \\\\"
    notes = (
        f"{n_row}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\\\\n\\multicolumn{5}{l}{\\footnotesize Notes: "
        "P(health next period | current health). "
        "Women only. Sample of ages. SOEP.}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_mortality_women(
    path_to_params: Path = _EST / "mortality_params_women_logit.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "mortality_women.tex",
) -> None:
    """LaTeX table: mortality (death) logit coefficients. Women only."""
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path_to_params, index_col=0)
    if df.shape[1] == 0:
        df = pd.read_csv(path_to_params)
    col = df.columns[0]
    header = (
        "\\begin{tabular}{lc}\n\\toprule\nParameter & Coefficient \\\\\n\\midrule\n"
    )
    rows = [f"{_latex_escape(str(i))} & {v:.4f} \\\\" for i, v in df[col].items()]
    body = "\n".join(rows)
    ss = _load_sample_sizes()
    n_val = ss.get("mortality_women_total")
    n_row = ""
    if n_val:
        n_row = f"\n\\midrule\n$N$ & {n_val:,} \\\\"
    notes = (
        f"{n_row}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\\\\n\\multicolumn{2}{l}{\\footnotesize Notes: Logit P(death). "
        "Baseline life table scaled by health and education dummies. Women only. SOEP.}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_own_wage_women(
    path_to_params: Path = _EST / "wage_eq_params.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "own_wage_women.tex",
) -> None:
    """LaTeX table: own wage equation (Panel OLS). Women only; coefficients and SE."""
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path_to_params)
    # Columns may be: education, sex, parameter, value (or index 0,1,2, value)
    if (
        df.shape[1] == N_CATEGORICAL_INDEX_COLS
        and df.columns[0] in ("0", "1", "2")
        or "Unnamed" in str(df.columns[0])
    ):
        df = pd.read_csv(path_to_params, index_col=[0, 1, 2])
        df = df.reset_index()
        df.columns = ["education", "sex", "parameter", "value"]
    val_col = "value" if "value" in df.columns else df.columns[-1]
    sex_col = "sex" if "sex" in df.columns else df.columns[1]
    df = df.loc[df[sex_col].astype(str).str.contains("Women", na=False)].copy()
    param_col = "parameter" if "parameter" in df.columns else df.columns[2]
    header = (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Education & Parameter & Coef. & (s.e.)"
        " \\\\\n\\midrule\n"
    )
    rows = []
    for edu in df["education"].unique():
        sub = df.loc[df["education"] == edu]
        edu_lab = (
            "Low" if "Low" in str(edu) else "High" if "High" in str(edu) else str(edu)
        )
        params = [
            p
            for p in sub[param_col].unique()
            if p and "_ser" not in str(p) and str(p) != "ser"  # codespell:ignore ser
        ]
        for pname in params:
            r = sub.loc[sub[param_col] == pname]
            val = r[val_col].iloc[0] if len(r) else np.nan
            ser_r = sub.loc[sub[param_col] == str(pname) + "_ser", val_col]
            se_val = ser_r.iloc[0] if len(ser_r) else np.nan
            se_str = f"{se_val:.4f}" if pd.notna(se_val) else "--"
            rows.append(
                f"{edu_lab} & {_latex_escape(str(pname))} & {val:.4f} & {se_str} \\\\"
            )
    body = "\n".join(rows) if rows else "\\multicolumn{4}{l}{(no data)} \\\\"
    ss = _load_sample_sizes()
    n_val = ss.get("wage_women_total")
    n_row = ""
    if n_val:
        n_row = f"\n\\midrule\n$N$ & \\multicolumn{{2}}{{c}}{{{n_val:,}}} \\\\"
    notes = (
        f"{n_row}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\\\\n\\multicolumn{4}{l}{\\footnotesize Notes: Panel OLS ln(hourly wage). "
        "Entity and year fixed effects. Women only. SOEP.}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_formal_care_costs(
    path_to_params: Path = _EST / "formal_care_costs_params_pooled.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "formal_care_costs.tex",
) -> None:
    """LaTeX table: formal care costs OLS (pooled). Coefficient, SE, R², N, notes."""
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    if not path_to_params.exists():
        path_to_save.write_text(
            "% Formal care costs params not found\n\\begin{tabular}{lcc}\n\\toprule\n"
            "Parameter & Coefficient & (s.e.) \\\\\n"
            "\\midrule\n(not found)\n\\bottomrule\n"
            "\\end{tabular}\n",
            encoding="utf-8",
        )
        return
    df = pd.read_csv(path_to_params, index_col=0)
    coef_row = df.loc["coefficient"] if "coefficient" in df.index else df.iloc[0]
    se_row = df.loc["coefficient_se"] if "coefficient_se" in df.index else None
    r2_val = None
    if "rsquared" in df.index:
        r2_val = (
            df.loc["rsquared", "const"]
            if pd.notna(df.loc["rsquared", "const"])
            else None
        )
    param_cols = [c for c in coef_row.index if c != "N" and pd.notna(coef_row.get(c))]
    param_labels = {
        "const": "Const",
        "age": "Age",
        "age_sq": "Age\\textsuperscript{2}",
    }
    header = (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Parameter & Coefficient & (s.e.)"
        " \\\\\n\\midrule\n"
    )
    rows = []
    for c in param_cols:
        coef = coef_row.get(c, np.nan)
        se = se_row.get(c, np.nan) if se_row is not None else np.nan
        se_str = f"{se:.4f}" if pd.notna(se) else "--"
        label = param_labels.get(str(c), _latex_escape(str(c)))
        rows.append(f"{label} & {coef:.4f} & {se_str} \\\\")
    n_val = coef_row.get("N", np.nan)
    if pd.notna(n_val):
        rows.append(f"N & \\multicolumn{{2}}{{c}}{{ {int(n_val)} }} \\\\")
    if r2_val is not None:
        rows.append(
            f"R\\textsuperscript{{2}} & \\multicolumn{{2}}{{c}}{{ {r2_val:.4f} }} \\\\"
        )
    body = "\n".join(rows)
    notes = (
        "\\\\\n\\bottomrule\n\\end{tabular}\n"
        "\\\\\n\\multicolumn{3}{l}{\\footnotesize Notes: "
        "OLS formal care costs (monthly). "
        "Specification: formal\\_care\\_costs "
        "$\\sim$ age + age². Pooled (no education). "
        "SOEP. Sample age $\\leq$ 70.}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


def _table_from_inheritance_spec(
    path_csv: Path, path_save: Path, title: str, spec_type: str
) -> None:
    """Build LaTeX table from inheritance spec CSV.

    Params + _se + _rsq rows. Women only.
    """
    path_save.parent.mkdir(parents=True, exist_ok=True)
    if not path_csv.exists():
        path_save.write_text(
            f"% Table source not found: {path_csv}\n"
            "\\begin{tabular}{lcc}\n\\toprule\n"
            "Parameter & Coefficient & (s.e.) \\\\\n\\midrule\n"
            "\\multicolumn{3}{l}{(data not found)} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n",
            encoding="utf-8",
        )
        return
    df = pd.read_csv(path_csv, index_col=0)
    idx_str = df.index.astype(str)
    women_row = None
    women_se_row = None
    women_rsq = None
    for i, idx in enumerate(idx_str):
        if idx == "Women" and "Women_se" not in idx:
            women_row = df.iloc[i]
        elif "Women_se" in idx or idx == "Women_se":
            women_se_row = df.iloc[i]
        elif "Women_rsq" in idx or idx == "Women_rsq":
            women_rsq = df.iloc[i].dropna()
    if women_row is None:
        women_row = df.iloc[0]
    param_cols = [
        c
        for c in women_row.index
        if c != "N" and pd.notna(women_row.get(c)) and str(c) != "nan"
    ]
    header = (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Parameter & Coefficient & (s.e.) \\\\\n\\midrule\n"
    )
    rows = []
    for c in param_cols:
        coef = women_row.get(c, np.nan)
        if pd.isna(coef):
            continue
        se = women_se_row.get(c, np.nan) if women_se_row is not None else np.nan
        se_str = f"{se:.4f}" if pd.notna(se) else "--"
        rows.append(f"{_latex_escape(str(c))} & {coef:.4f} & {se_str} \\\\")
    if women_rsq is not None and len(women_rsq):
        r2 = women_rsq.iloc[0]
        rows.append(
            "Pseudo R\\textsuperscript{2} & "
            f"\\multicolumn{{2}}{{c}}{{ {r2:.4f} }} \\\\"
        )
    n_val = women_row.get("N", np.nan)
    if pd.notna(n_val):
        rows.append(f"$N$ & \\multicolumn{{2}}{{c}}{{{int(n_val):,}}} \\\\")
    body = "\n".join(rows)
    notes = (
        "\\\\\n\\bottomrule\n\\end{tabular}\n"
        f"\\\\\n\\multicolumn{{3}}{{l}}{{\\footnotesize Notes: {title} "
        f"{spec_type}. Women only. SOEP.}}"
    )
    path_save.write_text(header + body + notes, encoding="utf-8")


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_inheritance_prob_women(
    path_to_params: Path = _PATH_INHERITANCE_PROB_SPEC7,
    path_to_save: Annotated[Path, Product] = _OUT / "inheritance_probability_women.tex",
) -> None:
    """LaTeX table: inheritance probability (logit) spec 7. Women only."""
    _table_from_inheritance_spec(
        path_to_params,
        path_to_save,
        "Logit P(positive inheritance). Spec 7: any care this year, "
        "filter parent this year.",
        "Probability",
    )


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_inheritance_amount_women(
    path_to_params: Path = _PATH_INHERITANCE_AMOUNT_SPEC12,
    path_to_save: Annotated[Path, Product] = _OUT / "inheritance_amount_women.tex",
) -> None:
    """LaTeX table: inheritance amount (OLS ln(amount)) spec 12. Women only."""
    _table_from_inheritance_spec(
        path_to_params,
        path_to_save,
        "OLS ln(inheritance amount). Spec 12: care recent, filter parent recent.",
        "Amount",
    )


# ---------------------------------------------------------------------------
# Mother mortality (life table)
# ---------------------------------------------------------------------------


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_mother_mortality_women(
    path_to_lifetable: Path = _EST / "death_transition_mat.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "mother_mortality_women.tex",
) -> None:
    """LaTeX table: mother's death probabilities from life table.

    Women (mothers) only.
    """
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path_to_lifetable)
    if "sex" in df.columns:
        df = df.loc[df["sex"].astype(str).str.contains("Women|1", regex=True)].copy()
    if "age" not in df.columns:
        df.columns = ["sex", "age", "death_prob"]
        df = df.loc[df["sex"].astype(str).str.contains("Women|1", regex=True)].copy()
    df["age"] = df["age"].astype(int)
    sample_ages = list(range(50, 101, 5))
    df_sample = df.loc[df["age"].isin(sample_ages)].sort_values("age")
    header = (
        "\\begin{tabular}{lc}\n\\toprule\n"
        "Mother's age & $p_{\\text{death}}$ \\\\\n\\midrule\n"
    )
    rows = [
        f"{int(r['age'])} & {r['death_prob']:.6f} \\\\" for _, r in df_sample.iterrows()
    ]
    body = "\n".join(rows) if rows else "\\multicolumn{2}{l}{(no data)} \\\\"
    notes = (
        "\\\\\n\\bottomrule\n\\end{tabular}\n"
        "\\\\\n\\multicolumn{2}{l}{\\footnotesize Notes: Annual death "
        "probabilities from "
        "Federal Statistical Office life tables.}"
        "\n\\multicolumn{2}{l}{\\footnotesize Selected ages shown. "
        "Applied to the mother "
        "via the mother--daughter age difference.}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


# ---------------------------------------------------------------------------
# Mother ADL transition (multinomial logit on SHARE data)
# ---------------------------------------------------------------------------


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_mother_adl_transition_women(
    path_to_params: Path = _EST / "adl_state_params.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "mother_adl_transition_women.tex",
) -> None:
    """LaTeX table: multinomial logit for mother's ADL transitions.

    Women (mothers) only.
    """
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path_to_params)
    df = df.loc[df["sex"].astype(str).str.contains("Women", case=False)].copy()

    cat_labels = {1: "Light ADL", 2: "Intensive ADL", 3: "Very Intensive ADL"}
    param_labels = {
        "const": "Constant",
        "age": "Age",
        "age_sq": "Age$^2$",
        "age_cubed": "Age$^3$",
        "adl_cat_1": "Lagged ADL = 1",
        "adl_cat_2": "Lagged ADL = 2",
        "adl_cat_3": "Lagged ADL = 3",
    }
    param_order = [
        "const",
        "age",
        "age_sq",
        "age_cubed",
        "adl_cat_1",
        "adl_cat_2",
        "adl_cat_3",
    ]

    n_cats = len(df)
    col_spec = "l" + "c" * n_cats
    cat_header = " & ".join(
        cat_labels.get(int(row["adl_cat"]), f"Cat.~{int(row['adl_cat'])}")
        for _, row in df.iterrows()
    )
    header = (
        f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n& {cat_header} \\\\\n\\midrule\n"
    )
    rows = []
    for p in param_order:
        if p not in df.columns:
            continue
        label = param_labels.get(p, _latex_escape(p))
        vals = []
        for _, row in df.iterrows():
            v = row.get(p, np.nan)
            vals.append(f"{v:.4f}" if pd.notna(v) else "--")
        rows.append(f"{label} & " + " & ".join(vals) + " \\\\")
    body = "\n".join(rows) if rows else "\\multicolumn{4}{l}{(no data)} \\\\"
    ss = _load_sample_sizes()
    n_val = ss.get("adl_women_total")
    n_row = ""
    if n_val:
        n_row = f"\n\\midrule\n$N$ & \\multicolumn{{{n_cats}}}{{c}}{{{n_val:,}}} \\\\"
    notes = (
        f"{n_row}\n\\bottomrule\n\\end{{tabular}}\n"
        f"\\\\\n\\multicolumn{{{1 + n_cats}}}{{l}}"
        "{\\footnotesize Notes: Multinomial logit. "
        "Dependent variable: mother's ADL category.}}\n"
        f"\\multicolumn{{{1 + n_cats}}}{{l}}"
        "{\\footnotesize Reference category: No ADL "
        "(category 0). Estimated on SHARE parent--child data, women (mothers).}}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")


# ---------------------------------------------------------------------------
# Exogenous care supply (logit, re-estimated from raw data)
# ---------------------------------------------------------------------------


@pytask.mark.tables
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_table_exog_care_supply_women(
    path_to_sample: Path = BLD / "data" / "exog_care_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = _OUT / "exog_care_supply_women.tex",
) -> None:
    """LaTeX table: exogenous care supply logit parameters. Women only."""
    import statsmodels.formula.api as smf

    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(path_to_sample)
    df = df[df["female"] == 1].copy()
    reg_data = df.loc[
        df["parent_care_demand"] == 1,
        ["other_informal_care", "age", "has_sister", "education"],
    ].dropna()
    reg_data["age_squared"] = reg_data["age"] ** 2

    model = smf.logit(
        "other_informal_care ~ age + age_squared + has_sister + education",
        data=reg_data,
    ).fit(disp=False)

    param_labels = {
        "Intercept": "Constant",
        "age": "Age",
        "age_squared": "Age$^2$",
        "has_sister": "Has sister",
        "education": "Education (high)",
    }

    header = (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Parameter & Coefficient & (s.e.) \\\\\n\\midrule\n"
    )
    rows = []
    for pname in model.params.index:
        coef = model.params[pname]
        se = model.bse[pname]
        label = param_labels.get(pname, _latex_escape(pname))
        rows.append(f"{label} & {coef:.4f} & ({se:.4f}) \\\\")

    pseudo_r2 = model.prsquared
    n_obs = int(model.nobs)
    rows.extend(
        (
            f"Pseudo R$^2$ & \\multicolumn{{2}}{{c}}{{ {pseudo_r2:.4f} }} \\\\",
            f"N & \\multicolumn{{2}}{{c}}{{ {n_obs} }} \\\\",
        )
    )

    body = "\n".join(rows)
    notes = (
        "\\\\\n\\bottomrule\n\\end{tabular}\n"
        "\\\\\n\\multicolumn{3}{l}{\\footnotesize Notes: Logit. Dependent variable: "
        "other family member provides informal care.}\n"
        "\\multicolumn{3}{l}{\\footnotesize SOEP, women with parental care demand. "
        "Standard errors in parentheses.}"
    )
    path_to_save.write_text(header + body + notes, encoding="utf-8")
