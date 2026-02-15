"""Create LaTeX table: share of domestic care arrangements by parent age and sex.

Uses SHARE parent-child data. Panel A: general care columns (pure_informal_care_
general, combination_care_general, pure_home_care_general). Panel B: daily care
columns (pure_informal_care_daily, combination_care_daily, pure_home_care_daily).
By age bin of parent (care recipient) and by sex (All, Men, Women).
Saves to BLD/descriptives.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD

# Age bin boundaries for parent (care recipient)
_AGE_50 = 50
_AGE_60 = 60
_AGE_70 = 70
_AGE_80 = 80
_AGE_85 = 85
_AGE_90 = 90


@pytask.mark.descriptives
def task_care_arrangements_by_age_latex(
    path_to_parent_child_data: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_save_latex_all: Annotated[Path, Product] = BLD
    / "descriptives"
    / "care_arrangements_by_age_all.tex",
    path_to_save_latex_men: Annotated[Path, Product] = BLD
    / "descriptives"
    / "care_arrangements_by_age_men.tex",
    path_to_save_latex_women: Annotated[Path, Product] = BLD
    / "descriptives"
    / "care_arrangements_by_age_women.tex",
) -> None:
    """Create LaTeX tables: care arrangement shares by parent age and sex.

    Uses parent-child data. Restricts to domestic care (pure informal,
    combination, or pure formal home care). Age bins: <50, 60-69, 70-79,
    80-89, 90+, and 70+, 70-84, 85+. Saves three tables: All, Men, Women.
    """
    df_full = pd.read_csv(path_to_parent_child_data)
    main_bins = ["<50", "60--69", "70--79", "80--89", "90+"]
    path_to_save_latex_all.parent.mkdir(parents=True, exist_ok=True)

    for sex_filter, path, label in (
        ("all", path_to_save_latex_all, "All"),
        ("men", path_to_save_latex_men, "Men"),
        ("women", path_to_save_latex_women, "Women"),
    ):
        if sex_filter == "all":
            data = df_full.copy()
        elif sex_filter == "men":
            data = df_full[df_full["sex"] == 0].copy()
        else:
            data = df_full[df_full["sex"] == 1].copy()

        # Panel A: general care
        data_g = data[_domestic_care_mask_general(data)].copy()
        data_g["pure_informal"] = (
            data_g["pure_informal_care_general"].fillna(0) == 1
        ).astype(int)
        data_g["combination"] = (
            data_g["combination_care_general"].fillna(0) == 1
        ).astype(int)
        data_g["pure_formal"] = (
            data_g["pure_home_care_general"].fillna(0) == 1
        ).astype(int)
        rows_a = _compute_panel_rows(data_g, main_bins)

        # Panel B: daily care
        data_d = data[_domestic_care_mask_daily(data)].copy()
        data_d["pure_informal"] = (
            data_d["pure_informal_care_daily"].fillna(0) == 1
        ).astype(int)
        data_d["combination"] = (
            data_d["combination_care_daily"].fillna(0) == 1
        ).astype(int)
        data_d["pure_formal"] = (data_d["pure_home_care_daily"].fillna(0) == 1).astype(
            int
        )
        rows_b = _compute_panel_rows(data_d, main_bins)

        if not rows_a and not rows_b:
            continue
        cap = f"Share of domestic care arrangements by parent age ({label})."
        header = (
            "Age (parent) & N & Pure informal (\\%) & Combination (\\%) & "
            "Pure formal (\\%) \\\\"
        )
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{cap}}}",
            f"\\label{{tab:care_arrangements_age_{label.lower()}}}",
            "\\begin{tabular}{l r r r r}",
            "\\toprule",
            "\\multicolumn{5}{l}{\\textbf{Panel A: General care}} \\\\",
            "\\midrule",
            header,
            "\\midrule",
        ]
        for b, n, p_inf, p_comb, p_form in rows_a:
            lines.append(f"{b} & {n} & {p_inf:.1f} & {p_comb:.1f} & {p_form:.1f} \\\\")
        lines.extend(
            [
                "\\midrule",
                "\\multicolumn{5}{l}{\\textbf{Panel B: Daily care}} \\\\",
                "\\midrule",
                header,
                "\\midrule",
            ]
        )
        for b, n, p_inf, p_comb, p_form in rows_b:
            lines.append(f"{b} & {n} & {p_inf:.1f} & {p_comb:.1f} & {p_form:.1f} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        path.write_text("\n".join(lines), encoding="utf-8")


def _domestic_care_mask_general(df: pd.DataFrame) -> pd.Series:
    """Rows with at least one of: pure informal, combination, pure formal (general)."""
    return (
        (df["pure_informal_care_general"].fillna(0) == 1)
        | (df["combination_care_general"].fillna(0) == 1)
        | (df["pure_home_care_general"].fillna(0) == 1)
    )


def _domestic_care_mask_daily(df: pd.DataFrame) -> pd.Series:
    """Rows with at least one of: pure informal, combination, pure formal (daily)."""
    return (
        (df["pure_informal_care_daily"].fillna(0) == 1)
        | (df["combination_care_daily"].fillna(0) == 1)
        | (df["pure_home_care_daily"].fillna(0) == 1)
    )


def _compute_panel_rows(
    data: pd.DataFrame, main_bins: list[str]
) -> list[tuple[str, int, float, float, float]]:
    """Return (bin, N, p_inf, p_comb, p_form) for main_bins + 70+, 70-84, 85+."""
    rows = []
    for b in main_bins:
        sub = _main_bin_subset(data, b)
        if len(sub) == 0:
            continue
        n = len(sub)
        p_inf = 100 * sub["pure_informal"].mean()
        p_comb = 100 * sub["combination"].mean()
        p_form = 100 * sub["pure_formal"].mean()
        rows.append((b, n, p_inf, p_comb, p_form))
    sub70 = data[data["age"] >= _AGE_70]
    if len(sub70) > 0:
        rows.append(
            (
                "70+",
                len(sub70),
                100 * sub70["pure_informal"].mean(),
                100 * sub70["combination"].mean(),
                100 * sub70["pure_formal"].mean(),
            )
        )
    sub7084 = data[(data["age"] >= _AGE_70) & (data["age"] < _AGE_85)]
    if len(sub7084) > 0:
        rows.append(
            (
                "70--84",
                len(sub7084),
                100 * sub7084["pure_informal"].mean(),
                100 * sub7084["combination"].mean(),
                100 * sub7084["pure_formal"].mean(),
            )
        )
    sub85 = data[data["age"] >= _AGE_85]
    if len(sub85) > 0:
        rows.append(
            (
                "85+",
                len(sub85),
                100 * sub85["pure_informal"].mean(),
                100 * sub85["combination"].mean(),
                100 * sub85["pure_formal"].mean(),
            )
        )
    return rows


def _main_bin_subset(data: pd.DataFrame, bin_name: str):
    """Return subset of data for the given main age bin (parent age)."""
    age = data["age"]
    if bin_name == "<50":
        return data[age < _AGE_50]
    if bin_name == "60--69":
        return data[(age >= _AGE_60) & (age < _AGE_70)]
    if bin_name == "70--79":
        return data[(age >= _AGE_70) & (age < _AGE_80)]
    if bin_name == "80--89":
        return data[(age >= _AGE_80) & (age < _AGE_90)]
    if bin_name == "90+":
        return data[age >= _AGE_90]
    return data.iloc[0:0]
