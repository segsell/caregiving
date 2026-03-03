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

from caregiving.config import BLD
from caregiving.data_management.soep.variables import create_education_type

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
