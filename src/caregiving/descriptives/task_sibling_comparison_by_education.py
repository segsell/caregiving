"""Sibling comparison by education: low vs high education women aged 25-35.

Compares number of siblings, brothers, sisters, and birth order. Outputs a LaTeX
table to BLD/descriptives. Uses SOEP sibling comparison sample (biosib + ppathl + pgen).
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product
from scipy import stats
from statsmodels.discrete.discrete_model import Logit

from caregiving.config import BLD
from caregiving.data_management.soep.variables import create_education_type

# For LPM (OLS with robust SEs)
try:
    from statsmodels.regression.linear_model import OLS
except ImportError:
    OLS = None  # type: ignore[misc, assignment]
# For multinomial logit (no care / light / intensive)
try:
    from statsmodels.discrete.discrete_model import MNLogit
except ImportError:
    MNLogit = None  # type: ignore[misc, assignment]

# SOEP Core: 1 = Male, 2 = Female
SOEP_FEMALE = 2
SOEP_MALE = 1

# Valid birth year range for counting siblings (exclude SOEP missing codes)
_MIN_GEBJAHR = 1900
_MAX_GEBJAHR = 2015


def _gebsib_cols(df: pd.DataFrame) -> list[str]:
    """Return gebsib* column names present in df."""
    return [c for c in df.columns if c.startswith("gebsib") and c[6:].isdigit()]


def _sexsib_cols(df: pd.DataFrame) -> list[str]:
    """Return sexsib* column names present in df."""
    return [c for c in df.columns if c.startswith("sexsib") and c[6:].isdigit()]


def _valid_birth_year(ser: pd.Series) -> pd.Series:
    """True where birth year is valid (not SOEP missing)."""
    return ser.notna() & (ser >= _MIN_GEBJAHR) & (ser <= _MAX_GEBJAHR)


def _n_siblings(row: pd.Series, gebsib_cols: list[str]) -> int:
    """Count siblings with valid birth year."""
    return int(_valid_birth_year(row[gebsib_cols]).sum())


def _n_brothers(row: pd.Series, sexsib_cols: list[str]) -> int:
    """Count brothers (sexsib == SOEP_MALE)."""
    return int((row[sexsib_cols] == SOEP_MALE).sum())


def _n_sisters(row: pd.Series, sexsib_cols: list[str]) -> int:
    """Count sisters (sexsib == SOEP_FEMALE)."""
    return int((row[sexsib_cols] == SOEP_FEMALE).sum())


def _birth_order(row: pd.Series, gebjahr_col: str, gebsib_cols: list[str]) -> float:
    """Birth order (1 = eldest) from respondent gebjahr and sibling gebsib*."""
    y_self = row[gebjahr_col]
    if pd.isna(y_self) or not (_MIN_GEBJAHR <= y_self <= _MAX_GEBJAHR):
        return np.nan
    y_self = int(y_self)
    years = [y_self]
    for c in gebsib_cols:
        y = row[c]
        if pd.notna(y) and _MIN_GEBJAHR <= y <= _MAX_GEBJAHR:
            years.append(int(y))
    years = sorted(years)
    rank = years.index(y_self) + 1
    return float(rank)


def _gebsib_sexsib_pairs(
    gebsib_cols: list[str], sexsib_cols: list[str]
) -> list[tuple[str, str]]:
    """Return (gebsib_col, sexsib_col) pairs matched by sibling index (e.g. gebsib1, sexsib1)."""
    sexsib_by_idx = {c[6:]: c for c in sexsib_cols if c[6:].isdigit()}
    pairs = []
    for g in gebsib_cols:
        idx = g[6:]
        if idx.isdigit() and idx in sexsib_by_idx:
            pairs.append((g, sexsib_by_idx[idx]))
    return sorted(pairs, key=lambda p: int(p[0][6:]))


def _birth_order_daughters(
    row: pd.Series,
    gebjahr_col: str,
    pairs: list[tuple[str, str]],
) -> float:
    """Birth order among daughters only (1 = oldest daughter). Returns np.nan if invalid."""
    y_self = row[gebjahr_col]
    if pd.isna(y_self) or not (_MIN_GEBJAHR <= y_self <= _MAX_GEBJAHR):
        return np.nan
    y_self = int(y_self)
    female_birth_years = [y_self]
    for geb_col, sex_col in pairs:
        if row[sex_col] != SOEP_FEMALE:
            continue
        y = row[geb_col]
        if pd.notna(y) and _MIN_GEBJAHR <= y <= _MAX_GEBJAHR:
            female_birth_years.append(int(y))
    female_birth_years = sorted(female_birth_years)
    rank = female_birth_years.index(y_self) + 1
    return float(rank)


def _is_youngest_daughter(
    row: pd.Series,
    gebjahr_col: str,
    pairs: list[tuple[str, str]],
) -> float:
    """1 if respondent is the youngest among all daughters (self + sisters), 0 otherwise."""
    y_self = row[gebjahr_col]
    if pd.isna(y_self) or not (_MIN_GEBJAHR <= y_self <= _MAX_GEBJAHR):
        return np.nan
    y_self = int(y_self)
    female_birth_years = [y_self]
    for geb_col, sex_col in pairs:
        if row[sex_col] != SOEP_FEMALE:
            continue
        y = row[geb_col]
        if pd.notna(y) and _MIN_GEBJAHR <= y <= _MAX_GEBJAHR:
            female_birth_years.append(int(y))
    return 1.0 if max(female_birth_years) == y_self else 0.0


@pytask.mark.descriptives
@pytask.mark.sibling_comparison
def task_sibling_comparison_by_education(
    path_to_sibling_sample: Path = BLD
    / "data"
    / "soep_sibling_comparison_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "descriptives"
    / "sibling_comparison_by_education.tex",
) -> None:
    """Compare low vs high education women aged 25-35: siblings, brothers, sisters, birth order.

    Reads sibling comparison sample, filters to women with age in [25, 35], keeps one
    observation per person (first syear in range). Builds summary table by education
    and writes LaTeX to bld/descriptives.
    """
    df = pd.read_csv(path_to_sibling_sample)
    if "pid" not in df.columns and df.index.names and "pid" in (df.index.names or []):
        df = df.reset_index()
    df["age"] = df["syear"] - df["gebjahr"]
    df = df[(df["sex"] == SOEP_FEMALE) & (df["age"] >= 25) & (df["age"] <= 35)].copy()
    df = df.sort_values(["pid", "syear"]).drop_duplicates(subset="pid", keep="first")

    df = create_education_type(df, drop_missing=True)

    gebsib_cols = _gebsib_cols(df)
    sexsib_cols = _sexsib_cols(df)
    if not gebsib_cols:
        df["n_siblings"] = 0
    else:
        df["n_siblings"] = df.apply(lambda r: _n_siblings(r, gebsib_cols), axis=1)
    if not sexsib_cols:
        df["n_brothers"] = 0
        df["n_sisters"] = 0
    else:
        df["n_brothers"] = df.apply(lambda r: _n_brothers(r, sexsib_cols), axis=1)
        df["n_sisters"] = df.apply(lambda r: _n_sisters(r, sexsib_cols), axis=1)

    gebjahr_col = "gebjahr"
    if gebjahr_col in df.columns and gebsib_cols:
        df["birth_order"] = df.apply(
            lambda r: _birth_order(r, gebjahr_col, gebsib_cols), axis=1
        )
    else:
        df["birth_order"] = 1.0

    # Birth order detail: 1 = firstborn (eldest), last-born = rank (n_siblings + 1)
    df["firstborn"] = (df["birth_order"] == 1).astype(float)
    df["last_born"] = (df["birth_order"] == df["n_siblings"] + 1).astype(float)

    # Youngest daughter: among respondent and all sisters, respondent has latest birth year
    pairs = _gebsib_sexsib_pairs(gebsib_cols, sexsib_cols)
    if pairs:
        df["youngest_daughter"] = df.apply(
            lambda r: _is_youngest_daughter(r, gebjahr_col, pairs), axis=1
        )
    else:
        df["youngest_daughter"] = np.nan

    path_to_save.parent.mkdir(parents=True, exist_ok=True)

    low = df[df["education"] == 0]
    high = df[df["education"] == 1]

    # Panel A: outcomes with Mean, SD, N for low and high education
    outcomes_panel_a = [
        ("Number of siblings", "n_siblings"),
        ("Number of brothers", "n_brothers"),
        ("Number of sisters", "n_sisters"),
        (
            "Birth order (1 = firstborn, higher = younger)",
            "birth_order",
        ),
        ("Share firstborn", "firstborn"),
        ("Share last-born", "last_born"),
        ("Share youngest daughter", "youngest_daughter"),
    ]
    rows_a = []
    for label, col in outcomes_panel_a:
        mean_low = low[col].mean()
        sd_low = low[col].std()
        mean_high = high[col].mean()
        sd_high = high[col].std()
        n_low = len(low)
        n_high = len(high)
        rows_a.append((label, mean_low, sd_low, n_low, mean_high, sd_high, n_high))

    # Panel B: difference (high - low) and p-value from two-sample Welch t-test
    rows_b = []
    for _label, col in outcomes_panel_a:
        high_clean = high[col].dropna()
        low_clean = low[col].dropna()
        if len(high_clean) > 0 and len(low_clean) > 0:
            _stat, pval = stats.ttest_ind(high_clean, low_clean, equal_var=False)
        else:
            pval = np.nan
        diff = high[col].mean() - low[col].mean()
        rows_b.append((diff, pval))

    caption = (
        "Sibling composition by education: women aged 25--35. "
        "Birth order is the respondent's rank among her siblings by birth year "
        "(1 = firstborn, 2 = second-born, etc.). "
        "Share firstborn (last-born) is the fraction who are the oldest (youngest) "
        "among their siblings. Share youngest daughter is the fraction who are the "
        "youngest among all daughters in the family (respondent and her sisters). "
        "Panel B: two-sample Welch $t$-test for difference in means."
    )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:sibling_comparison_by_education}",
        "\\begin{tabular}{l c c c c c c c}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Low education} & \\multicolumn{3}{c}{High education} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        "Outcome & Mean & SD & N & Mean & SD & N \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\textit{Panel A: Descriptive statistics}} \\\\",
        "\\addlinespace[0.5em]",
    ]
    for label, mean_low, sd_low, n_low, mean_high, sd_high, n_high in rows_a:
        lines.append(
            f"{label} & {mean_low:.2f} & {sd_low:.2f} & {n_low} & {mean_high:.2f} & {sd_high:.2f} & {n_high} \\\\"
        )
    lines.extend(
        [
            "\\addlinespace[1em]",
            "\\midrule",
            "\\multicolumn{7}{l}{\\textit{Panel B: Difference (High $-$ Low) and significance}} \\\\",
            "\\addlinespace[0.5em]",
            "Outcome & Difference & $p$-value & & & & \\\\",
            "\\midrule",
        ]
    )
    for i in range(len(outcomes_panel_a)):
        diff, pval = rows_b[i]
        if np.isnan(pval):
            p_str = "n.a."
        elif pval >= 0.001:
            p_str = f"{pval:.3f}"
        else:
            p_str = "<0.001"
        diff_str = f"{diff:+.2f}" if not np.isnan(diff) else "n.a."
        lines.append(f"{outcomes_panel_a[i][0]} & {diff_str} & {p_str} & & & & \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path_to_save.write_text("\n".join(lines), encoding="utf-8")


def _stars(p: float) -> str:
    """Return significance stars for p-value (* 10%, ** 5%, *** 1%)."""
    if np.isnan(p) or p > 0.10:
        return ""
    if p <= 0.01:
        return "***"
    if p <= 0.05:
        return "**"
    return "*"


def _get_caregiving_sibling_regression_data(
    path_to_sibling_sample: Path,
) -> pd.DataFrame:
    """Load and prepare sibling sample for caregiving regressions (women 30--70, all vars)."""
    df = pd.read_csv(path_to_sibling_sample)
    if "pid" not in df.columns and df.index.names and "pid" in (df.index.names or []):
        df = df.reset_index()
    df["age"] = df["syear"] - df["gebjahr"]
    df = df[(df["sex"] == SOEP_FEMALE) & (df["age"] >= 30) & (df["age"] <= 70)].copy()
    df = create_education_type(df, drop_missing=True)
    gebsib_cols = _gebsib_cols(df)
    sexsib_cols = _sexsib_cols(df)
    gebjahr_col = "gebjahr"
    if gebsib_cols:
        df["n_siblings"] = df.apply(lambda r: _n_siblings(r, gebsib_cols), axis=1)
        df["birth_order"] = df.apply(
            lambda r: _birth_order(r, gebjahr_col, gebsib_cols), axis=1
        )
    else:
        df["n_siblings"] = 0
        df["birth_order"] = 1.0
    if sexsib_cols:
        df["n_brothers"] = df.apply(lambda r: _n_brothers(r, sexsib_cols), axis=1)
        df["n_sisters"] = df.apply(lambda r: _n_sisters(r, sexsib_cols), axis=1)
    else:
        df["n_brothers"] = 0
        df["n_sisters"] = 0
    df["firstborn"] = np.where(
        df["birth_order"].notna(), (df["birth_order"] == 1).astype(float), np.nan
    )
    pairs = _gebsib_sexsib_pairs(gebsib_cols, sexsib_cols)
    if pairs:
        df["youngest_daughter"] = df.apply(
            lambda r: _is_youngest_daughter(r, gebjahr_col, pairs), axis=1
        )
        df["birth_order_daughters"] = df.apply(
            lambda r: _birth_order_daughters(r, gebjahr_col, pairs), axis=1
        )
    else:
        df["youngest_daughter"] = np.nan
        df["birth_order_daughters"] = np.nan
    df["n_daughters"] = 1 + df["n_sisters"]
    only_daughter = (df["n_sisters"] == 0).astype(float)
    more_daughters = (df["n_sisters"] >= 1).astype(float)
    oldest_daughter = (df["birth_order_daughters"] == 1).astype(float)
    second_youngest_daughter = (
        (df["birth_order_daughters"] == df["n_daughters"] - 1)
        & (df["n_daughters"] >= 2)
    ).astype(float)
    df["youngest_only_daughter"] = only_daughter
    df["youngest_more_daughters"] = (
        more_daughters.astype(bool) & (df["youngest_daughter"] == 1)
    ).astype(float)
    df["second_youngest_only_daughter"] = 0.0
    df["second_youngest_more_daughters"] = (
        more_daughters.astype(bool) & (second_youngest_daughter == 1)
    ).astype(float)
    df["oldest_only_daughter"] = 0.0
    df["oldest_more_daughters"] = (
        more_daughters.astype(bool) & (oldest_daughter == 1)
    ).astype(float)
    df["age_sq"] = df["age"].astype(float) ** 2
    return df


@pytask.mark.descriptives
@pytask.mark.sibling_comparison
def task_logit_caregiving_youngest_daughter(
    path_to_sibling_sample: Path = BLD
    / "data"
    / "soep_sibling_comparison_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "descriptives"
    / "logit_caregiving_youngest_daughter.tex",
) -> None:
    """Logit of any_care, light_care, intensive_care on sibling composition and covariates.

    Sample: women, age 30--70 (panel). All specs include sibling battery (youngest daughter,
    firstborn, number of sisters, number of brothers). Spec 1: sibling composition only;
    Spec 2: + education; Spec 3: + age, age squared. Writes LaTeX table with coef (SE) and stars.
    """
    df = _get_caregiving_sibling_regression_data(path_to_sibling_sample)

    # Sibling battery (in all specs); Spec 2/3 add education and age as robustness
    sibling_battery = ["youngest_daughter", "firstborn", "n_sisters", "n_brothers"]
    # Only dummies with variation: second_youngest_only and oldest_only are identically 0
    position_type_dummies = [
        "youngest_only_daughter",
        "youngest_more_daughters",
        "second_youngest_more_daughters",
        "oldest_more_daughters",
    ]
    outcomes = [
        ("Any care", "any_care"),
        ("Light care", "light_care"),
        ("Intensive care", "intensive_care"),
    ]
    path_to_save.parent.mkdir(parents=True, exist_ok=True)

    spec_cols_list = [
        sibling_battery,
        sibling_battery + ["education"],
        sibling_battery + ["education", "age", "age_sq"],
        position_type_dummies,
    ]
    all_results: list[tuple[str, str, list[tuple[str, float, float, float]], int]] = []
    for out_label, out_col in outcomes:
        for spec_label, cols in [
            ("Spec 1: Sibling composition", spec_cols_list[0]),
            ("Spec 2: + Education", spec_cols_list[1]),
            ("Spec 3: + Age, age$^2$", spec_cols_list[2]),
            ("Spec 4: Position $\\times$ type", spec_cols_list[3]),
        ]:
            use = df[[out_col] + cols].dropna()
            n_obs = len(use)
            y = use[out_col].astype(int)
            X = use[cols].copy()
            X.insert(0, "const", 1.0)
            try:
                logit = Logit(y, X).fit(disp=0)
                params = logit.params
                bse = logit.bse
                pvalues = logit.pvalues
            except Exception:
                params = pd.Series(index=X.columns, data=np.nan)
                bse = pd.Series(index=X.columns, data=np.nan)
                pvalues = pd.Series(index=X.columns, data=np.nan)
            row_list = []
            table_vars = [
                "const",
                "youngest_daughter",
                "firstborn",
                "n_sisters",
                "n_brothers",
                "youngest_only_daughter",
                "youngest_more_daughters",
                "second_youngest_only_daughter",
                "second_youngest_more_daughters",
                "oldest_only_daughter",
                "oldest_more_daughters",
                "education",
                "age",
                "age_sq",
            ]
            for var in table_vars:
                if var not in params.index:
                    continue
                coef = params[var]
                se = bse[var]
                p = pvalues[var]
                row_list.append((var, coef, se, p))
            all_results.append((out_label, spec_label, row_list, n_obs))

    # Build one table: sections by outcome, columns by spec (1--4), rows by variable
    var_labels = {
        "const": "Constant",
        "youngest_daughter": "Youngest daughter (1 = yes)",
        "firstborn": "Firstborn (1 = yes)",
        "n_sisters": "Number of sisters",
        "n_brothers": "Number of brothers",
        "youngest_only_daughter": "Youngest, only daughter",
        "youngest_more_daughters": "Youngest, more daughters",
        "second_youngest_only_daughter": "Second-youngest, only daughter",
        "second_youngest_more_daughters": "Second-youngest, more daughters",
        "oldest_only_daughter": "Oldest, only daughter",
        "oldest_more_daughters": "Oldest, more daughters",
        "education": "Education (1 = high)",
        "age": "Age",
        "age_sq": "Age squared",
    }
    n_specs = 4
    n_outcomes = 3
    # Table: 4 specs x 3 outcomes = 12 coefficient columns
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Logit of caregiving on sibling composition and covariates. "
        "Sample: women, age 30--70 (person-year panel). "
        "Spec 1--3: sibling composition (youngest daughter, firstborn, n. sisters, n. brothers); "
        "Spec 2: + education; Spec 3: + age, age squared. "
        "Spec 4: position $\\times$ type (youngest/second-youngest/oldest $\\times$ only/more daughters); ref.\\ = middle. "
        "Standard errors in parentheses; $^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$.}",
        "\\label{tab:logit_caregiving_youngest_daughter}",
        "\\begin{tabular}{l" + " c c c c" * n_outcomes + "}",
        "\\toprule",
    ]
    # Header: Outcome blocks (4 cols each)
    header = (
        " & ".join([" & \\multicolumn{4}{c}{" + label + "}" for label, _ in outcomes])
        + " \\\\"
    )
    lines.append(header)
    lines.append("\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13}")
    spec_headers = (
        " & ".join(["Spec 1 & Spec 2 & Spec 3 & Spec 4"] * n_outcomes) + " \\\\"
    )
    lines.append(spec_headers)
    lines.append("\\midrule")

    # Row order for table: Constant, sibling battery, position×type (Spec 4), education/age
    row_order = [
        "const",
        "youngest_daughter",
        "firstborn",
        "n_sisters",
        "n_brothers",
        "youngest_only_daughter",
        "youngest_more_daughters",
        "second_youngest_only_daughter",
        "second_youngest_more_daughters",
        "oldest_only_daughter",
        "oldest_more_daughters",
        "education",
        "age",
        "age_sq",
    ]
    for var in row_order:
        label = var_labels[var]
        cells = []
        for out_idx in range(n_outcomes):
            for spec_idx in range(n_specs):
                idx = out_idx * n_specs + spec_idx
                _, _spec, row_list, _n_obs = all_results[idx]
                entry = next((r for r in row_list if r[0] == var), None)
                if entry is None:
                    cells.append("")
                else:
                    _v, coef, se, p = entry
                    if np.isnan(coef) or np.isnan(se):
                        cells.append("")
                    else:
                        cells.append(f"{coef:.3f}{_stars(p)} ({se:.3f})")
        lines.append(" & ".join([label] + cells) + " \\\\")

    # N observations row (stored in all_results)
    n_obs_list = [all_results[i][3] for i in range(len(all_results))]
    lines.append("\\midrule")
    lines.append("N & " + " & ".join(str(n) for n in n_obs_list) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path_to_save.write_text("\n".join(lines), encoding="utf-8")


# --- Shared table layout for LPM (same vars/row_order as logit) ---
_CAREGIVING_TABLE_VAR_LABELS = {
    "const": "Constant",
    "youngest_daughter": "Youngest daughter (1 = yes)",
    "firstborn": "Firstborn (1 = yes)",
    "n_sisters": "Number of sisters",
    "n_brothers": "Number of brothers",
    "youngest_only_daughter": "Youngest, only daughter",
    "youngest_more_daughters": "Youngest, more daughters",
    "second_youngest_only_daughter": "Second-youngest, only daughter",
    "second_youngest_more_daughters": "Second-youngest, more daughters",
    "oldest_only_daughter": "Oldest, only daughter",
    "oldest_more_daughters": "Oldest, more daughters",
    "education": "Education (1 = high)",
    "age": "Age",
    "age_sq": "Age squared",
}
_CAREGIVING_TABLE_ROW_ORDER = [
    "const",
    "youngest_daughter",
    "firstborn",
    "n_sisters",
    "n_brothers",
    "youngest_only_daughter",
    "youngest_more_daughters",
    "second_youngest_only_daughter",
    "second_youngest_more_daughters",
    "oldest_only_daughter",
    "oldest_more_daughters",
    "education",
    "age",
    "age_sq",
]


@pytask.mark.descriptives
@pytask.mark.sibling_comparison
def task_caregiving_sibling_composition_lpm(
    path_to_sibling_sample: Path = BLD
    / "data"
    / "soep_sibling_comparison_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "descriptives"
    / "caregiving_sibling_composition_lpm.tex",
) -> None:
    """Linear probability model (OLS, robust SEs): caregiving on sibling composition.

    Same specs and outcomes as logit. Coefficients = change in probability (pp).
    Robust (HC1) standard errors. Complements logit with directly interpretable margins.
    """
    if OLS is None:
        raise RuntimeError("statsmodels OLS required for LPM task")
    df = _get_caregiving_sibling_regression_data(path_to_sibling_sample)
    sibling_battery = ["youngest_daughter", "firstborn", "n_sisters", "n_brothers"]
    position_type_dummies = [
        "youngest_only_daughter",
        "youngest_more_daughters",
        "second_youngest_more_daughters",
        "oldest_more_daughters",
    ]
    outcomes = [
        ("Any care", "any_care"),
        ("Light care", "light_care"),
        ("Intensive care", "intensive_care"),
    ]
    spec_cols_list = [
        sibling_battery,
        sibling_battery + ["education"],
        sibling_battery + ["education", "age", "age_sq"],
        position_type_dummies,
    ]
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    table_vars = [
        "const",
        "youngest_daughter",
        "firstborn",
        "n_sisters",
        "n_brothers",
        "youngest_only_daughter",
        "youngest_more_daughters",
        "second_youngest_only_daughter",
        "second_youngest_more_daughters",
        "oldest_only_daughter",
        "oldest_more_daughters",
        "education",
        "age",
        "age_sq",
    ]
    all_results: list[tuple[str, str, list[tuple[str, float, float, float]], int]] = []
    for out_label, out_col in outcomes:
        for spec_label, cols in [
            ("Spec 1: Sibling composition", spec_cols_list[0]),
            ("Spec 2: + Education", spec_cols_list[1]),
            ("Spec 3: + Age, age$^2$", spec_cols_list[2]),
            ("Spec 4: Position $\\times$ type", spec_cols_list[3]),
        ]:
            use = df[[out_col] + cols].dropna()
            n_obs = len(use)
            y = use[out_col].astype(float)
            X = use[cols].copy()
            X.insert(0, "const", 1.0)
            try:
                res = OLS(y, X).fit(cov_type="HC1", use_t=True)
                params, bse, pvalues = res.params, res.bse, res.pvalues
            except Exception:
                params = pd.Series(index=X.columns, data=np.nan)
                bse = pd.Series(index=X.columns, data=np.nan)
                pvalues = pd.Series(index=X.columns, data=np.nan)
            row_list = []
            for var in table_vars:
                if var not in params.index:
                    continue
                row_list.append(
                    (var, float(params[var]), float(bse[var]), float(pvalues[var]))
                )
            all_results.append((out_label, spec_label, row_list, n_obs))
    n_specs, n_outcomes = 4, 3
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Linear probability model (OLS, robust SEs): caregiving on sibling composition. "
        "Sample: women, age 30--70 (person-year panel). Coefficients = change in probability (pp). "
        "Specs as in Table~\\ref{tab:logit_caregiving_youngest_daughter}. "
        "Standard errors in parentheses (heteroskedasticity-robust); $^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$.}",
        "\\label{tab:caregiving_sibling_composition_lpm}",
        "\\begin{tabular}{l" + " c c c c" * n_outcomes + "}",
        "\\toprule",
    ]
    header = (
        " & ".join([" & \\multicolumn{4}{c}{" + label + "}" for label, _ in outcomes])
        + " \\\\"
    )
    lines.append(header)
    lines.append("\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13}")
    lines.append(
        " & ".join(["Spec 1 & Spec 2 & Spec 3 & Spec 4"] * n_outcomes) + " \\\\"
    )
    lines.append("\\midrule")
    for var in _CAREGIVING_TABLE_ROW_ORDER:
        label = _CAREGIVING_TABLE_VAR_LABELS[var]
        cells = []
        for out_idx in range(n_outcomes):
            for spec_idx in range(n_specs):
                idx = out_idx * n_specs + spec_idx
                _, _, row_list, _ = all_results[idx]
                entry = next((r for r in row_list if r[0] == var), None)
                if entry is None:
                    cells.append("")
                else:
                    _, coef, se, p = entry
                    if np.isnan(coef) or np.isnan(se):
                        cells.append("")
                    else:
                        cells.append(f"{coef:.3f}{_stars(p)} ({se:.3f})")
        lines.append(" & ".join([label] + cells) + " \\\\")
    n_obs_list = [all_results[i][3] for i in range(len(all_results))]
    lines.append("\\midrule")
    lines.append("N & " + " & ".join(str(n) for n in n_obs_list) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path_to_save.write_text("\n".join(lines), encoding="utf-8")


@pytask.mark.descriptives
@pytask.mark.sibling_comparison
def task_caregiving_by_sibling_position_descriptives(
    path_to_sibling_sample: Path = BLD
    / "data"
    / "soep_sibling_comparison_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "descriptives"
    / "caregiving_by_sibling_position_descriptives.tex",
) -> None:
    """Descriptive table: mean caregiving (any, light, intensive) by sibling position x type.

    Sample: women 30--70. Rows = position type. Complements regression with raw rates.
    """
    df = _get_caregiving_sibling_regression_data(path_to_sibling_sample)

    def _position_type(row: pd.Series) -> str:
        if row["youngest_only_daughter"] == 1:
            return "Youngest, only daughter"
        if row["youngest_more_daughters"] == 1:
            return "Youngest, more daughters"
        if row["second_youngest_more_daughters"] == 1:
            return "Second-youngest, more daughters"
        if row["oldest_more_daughters"] == 1:
            return "Oldest, more daughters"
        return "Middle"

    df["position_type"] = df.apply(_position_type, axis=1)
    order = [
        "Youngest, only daughter",
        "Youngest, more daughters",
        "Second-youngest, more daughters",
        "Oldest, more daughters",
        "Middle",
    ]
    agg = (
        df.groupby("position_type", observed=True)
        .agg(
            any_care=("any_care", "mean"),
            light_care=("light_care", "mean"),
            intensive_care=("intensive_care", "mean"),
            N=("any_care", "count"),
        )
        .reindex(order)
    )
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Mean caregiving by sibling position (women, age 30--70, person-year).}",
        "\\label{tab:caregiving_by_sibling_position_descriptives}",
        "\\begin{tabular}{l c c c c}",
        "\\toprule",
        "Position & Any care & Light care & Intensive care & N \\\\",
        "\\midrule",
    ]
    for idx in order:
        if idx not in agg.index:
            continue
        r = agg.loc[idx]
        lines.append(
            f"{idx} & {r['any_care']:.3f} & {r['light_care']:.3f} & {r['intensive_care']:.3f} & {int(r['N'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path_to_save.write_text("\n".join(lines), encoding="utf-8")


@pytask.mark.descriptives
@pytask.mark.sibling_comparison
def task_caregiving_sibling_multinomial_logit(
    path_to_sibling_sample: Path = BLD
    / "data"
    / "soep_sibling_comparison_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "descriptives"
    / "caregiving_sibling_multinomial_logit.tex",
) -> None:
    """Multinomial logit: outcome = no care (0) / light (1) / intensive (2). Base = no care."""
    if MNLogit is None:
        raise RuntimeError("statsmodels MNLogit required")
    df = _get_caregiving_sibling_regression_data(path_to_sibling_sample)
    df["care_level"] = np.where(
        df["intensive_care"].astype(bool),
        2,
        np.where(df["light_care"].astype(bool), 1, 0),
    )
    sibling_battery = ["youngest_daughter", "firstborn", "n_sisters", "n_brothers"]
    position_type_dummies = [
        "youngest_only_daughter",
        "youngest_more_daughters",
        "second_youngest_more_daughters",
        "oldest_more_daughters",
    ]
    spec_cols_list = [
        sibling_battery,
        sibling_battery + ["education"],
        sibling_battery + ["education", "age", "age_sq"],
        position_type_dummies,
    ]
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[tuple[str, dict, int]] = []
    for spec_label, cols in [
        ("Spec 1", spec_cols_list[0]),
        ("Spec 2", spec_cols_list[1]),
        ("Spec 3", spec_cols_list[2]),
        ("Spec 4", spec_cols_list[3]),
    ]:
        use = df[["care_level"] + cols].dropna()
        use = use[use["care_level"].isin([0, 1, 2])]
        n_obs = len(use)
        y = use["care_level"].astype(int)
        X = use[cols].copy()
        X.insert(0, "const", 1.0)
        try:
            res = MNLogit(y, X).fit(disp=0)
            params, bse, pvalues = res.params, res.bse, res.pvalues
        except Exception:
            params = pd.Series(dtype=float)
            bse = pd.Series(dtype=float)
            pvalues = pd.Series(dtype=float)
        coefs: dict = {}
        if isinstance(params, pd.DataFrame):
            # Two columns: first = light vs base, second = intensive vs base. Map by position to 1, 2.
            col_list = list(params.columns)
            for var in params.index:
                for j, col in enumerate(col_list):
                    outcome = j + 1  # 1 = light vs no care, 2 = intensive vs no care
                    coefs[(var, outcome)] = (
                        float(params.loc[var, col]),
                        float(bse.loc[var, col]),
                        float(pvalues.loc[var, col]),
                    )
        else:
            for idx in params.index:
                if isinstance(idx, tuple) and len(idx) >= 2:
                    var, cat = idx[0], idx[1]
                    c = (
                        int(cat)
                        if cat in (1, 2)
                        else (int(cat) + 1 if int(cat) == 0 else int(cat))
                    )
                    coefs[(var, c)] = (
                        float(params[idx]),
                        float(bse[idx]),
                        float(pvalues[idx]),
                    )
                else:
                    s = str(idx)
                    if s.endswith("_1") or s.endswith("_2"):
                        parts = s.rsplit("_", 1)
                        var_base, cat = parts[0], int(parts[1])
                        coefs[(var_base, cat)] = (
                            float(params[idx]),
                            float(bse[idx]),
                            float(pvalues[idx]),
                        )
        all_results.append((spec_label, coefs, n_obs))
    # Binary logit any_care (any vs no care) with same specs for the third block
    any_results: list[tuple[str, dict[str, tuple[float, float, float]], int]] = []
    for spec_label, cols in [
        ("Spec 1", spec_cols_list[0]),
        ("Spec 2", spec_cols_list[1]),
        ("Spec 3", spec_cols_list[2]),
        ("Spec 4", spec_cols_list[3]),
    ]:
        use = df[["any_care"] + cols].dropna()
        use = use[use["any_care"].isin([0, 1])]
        n_obs = len(use)
        y = use["any_care"].astype(int)
        X = use[cols].copy()
        X.insert(0, "const", 1.0)
        try:
            res = Logit(y, X).fit(disp=0)
            params, bse, pvalues = res.params, res.bse, res.pvalues
        except Exception:
            params = pd.Series(index=X.columns, data=np.nan)
            bse = pd.Series(index=X.columns, data=np.nan)
            pvalues = pd.Series(index=X.columns, data=np.nan)
        coefs_any: dict[str, tuple[float, float, float]] = {}
        for var in params.index:
            coefs_any[var] = (float(params[var]), float(bse[var]), float(pvalues[var]))
        any_results.append((spec_label, coefs_any, n_obs))
    var_order = [v for v in _CAREGIVING_TABLE_ROW_ORDER if v != "const"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Multinomial logit: care level (no/light/intensive) on sibling composition; Any vs no care from binary logit. Base: no care. Women 30--70.}",
        "\\label{tab:caregiving_sibling_multinomial_logit}",
        "\\begin{tabular}{l l c c c c}",
        "\\toprule",
        "Variable & Contrast & Spec 1 & Spec 2 & Spec 3 & Spec 4 \\\\",
        "\\midrule",
    ]
    contrasts = [
        (1, "Light vs no care"),
        (2, "Intensive vs no care"),
        (0, "Any vs no care"),
    ]
    for var in var_order:
        for cat_val, cat_label in contrasts:
            label = _CAREGIVING_TABLE_VAR_LABELS.get(var, var)
            cells = []
            for spec_idx in range(4):
                if cat_val == 0:
                    _, coefs_any, _ = any_results[spec_idx]
                    if var in coefs_any:
                        coef, se, p = coefs_any[var]
                        cells.append(
                            f"{coef:.3f}{_stars(p)} ({se:.3f})"
                            if not (np.isnan(coef) or np.isnan(se))
                            else ""
                        )
                    else:
                        cells.append("")
                else:
                    _, coefs, _ = all_results[spec_idx]
                    key = (var, cat_val)
                    if key not in coefs:
                        key = next(
                            (k for k in coefs if k[0] == var and k[1] == cat_val), None
                        )
                    if key is not None and key in coefs:
                        coef, se, p = coefs[key]
                        cells.append(
                            f"{coef:.3f}{_stars(p)} ({se:.3f})"
                            if not (np.isnan(coef) or np.isnan(se))
                            else ""
                        )
                    else:
                        cells.append("")
            lines.append(" & ".join([label, cat_label] + cells) + " \\\\")
    lines.append("\\midrule")
    lines.append(
        "N & & " + " & ".join(str(all_results[i][2]) for i in range(4)) + " \\\\"
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path_to_save.write_text("\n".join(lines), encoding="utf-8")


# Sibling composition subsamples for youngest-daughter effect (see task below)
_COMPOSITION_SPECS = [
    (
        "Two siblings, other female",
        lambda df: (df["n_siblings"] == 1)
        & (df["n_sisters"] == 1)
        & (df["n_brothers"] == 0),
    ),
    (
        "Two siblings, other male",
        lambda df: (df["n_siblings"] == 1)
        & (df["n_brothers"] == 1)
        & (df["n_sisters"] == 0),
    ),
    (
        "Three siblings, 1M 1F",
        lambda df: (df["n_siblings"] == 2)
        & (df["n_sisters"] == 1)
        & (df["n_brothers"] == 1),
    ),
    (
        "Three siblings, 2M",
        lambda df: (df["n_siblings"] == 2)
        & (df["n_brothers"] == 2)
        & (df["n_sisters"] == 0),
    ),
    (
        "Three siblings, 2F",
        lambda df: (df["n_siblings"] == 2)
        & (df["n_sisters"] == 2)
        & (df["n_brothers"] == 0),
    ),
    ("Four or more siblings", lambda df: df["n_siblings"] >= 3),
]


@pytask.mark.descriptives
@pytask.mark.sibling_comparison
def task_youngest_daughter_by_sibling_composition(
    path_to_sibling_sample: Path = BLD
    / "data"
    / "soep_sibling_comparison_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "descriptives"
    / "youngest_daughter_by_sibling_composition.tex",
) -> None:
    """Logit of any/light/intensive care on youngest daughter within sibling-composition subsamples.

    Six compositions: (1) two sib, other female; (2) two sib, other male; (3) three sib, 1M 1F;
    (4) three sib, 2M; (5) three sib, 2F; (6) four or more siblings. Reports coef (SE) for
    youngest_daughter and N. Table note reports mean and median of sibship size (1 + n_siblings).
    """
    df = _get_caregiving_sibling_regression_data(path_to_sibling_sample)
    df["total_children"] = 1 + df["n_siblings"]
    # Mean and median sibship size (person-year level)
    mean_sibship = float(df["total_children"].mean())
    median_sibship = float(df["total_children"].median())
    outcomes = [
        ("Any care", "any_care"),
        ("Light care", "light_care"),
        ("Intensive care", "intensive_care"),
    ]
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, list[tuple[float, float, float, int]]]] = []
    for label, mask_fn in _COMPOSITION_SPECS:
        sub = df.loc[mask_fn(df)].copy()
        sub = sub.dropna(
            subset=["youngest_daughter", "any_care", "light_care", "intensive_care"]
        )
        cell_list: list[tuple[float, float, float, int]] = []
        for _out_label, out_col in outcomes:
            use = sub[[out_col, "youngest_daughter"]].dropna()
            n_obs = len(use)
            y = use[out_col].astype(int)
            X = use[["youngest_daughter"]].copy()
            X.insert(0, "const", 1.0)
            try:
                res = Logit(y, X).fit(disp=0)
                coef = float(res.params["youngest_daughter"])
                se = float(res.bse["youngest_daughter"])
                p = float(res.pvalues["youngest_daughter"])
            except Exception:
                coef = se = p = np.nan
            cell_list.append((coef, se, p, n_obs))
        rows.append((label, cell_list))
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Logit of caregiving on youngest daughter (1 = yes) by sibling composition. "
        "Sample: women 30--70 (person-year). Each column restricts to the stated composition; "
        "only youngest daughter is included as regressor. "
        f"Mean (median) sibship size (respondent + siblings) in full sample: {mean_sibship:.2f} ({median_sibship:.1f}).}}",
        "\\label{tab:youngest_daughter_by_sibling_composition}",
        "\\begin{tabular}{l c c c c c c}",
        "\\toprule",
        "Composition & Any care & & Light care & & Intensive care & \\\\",
        " & Coef. (SE) & N & Coef. (SE) & N & Coef. (SE) & N \\\\",
        "\\midrule",
    ]
    for label, cell_list in rows:
        parts = []
        for coef, se, p, n_obs in cell_list:
            if np.isnan(coef) or np.isnan(se):
                parts.append("--")
                parts.append("")
            else:
                parts.append(f"{coef:.3f}{_stars(p)} ({se:.3f})")
                parts.append(str(n_obs))
        lines.append(" & ".join([label] + parts) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path_to_save.write_text("\n".join(lines), encoding="utf-8")
