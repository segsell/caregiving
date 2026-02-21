"""Create LaTeX table summarising the estimation moments.

Reads the moments CSV, groups moment names by type, counts them, extracts
age ranges, and produces a publication-ready LaTeX table.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD

# ---------------------------------------------------------------------------
# Moment group definitions
# ---------------------------------------------------------------------------

_LS = r"(retired|unemployed|part_time|full_time)"

_GROUPS: list[dict] = [
    # ---- Panel A: Care Arrangements ----
    {
        "panel": "Panel A: Care Arrangements",
        "label": "Share providing informal care by 5-year age bin",
        "dataset": "GSOEP/SHARE",
        "patterns": [
            r"^share_informal_care_(any|light|intensive)_age_bin_\d+_\d+$",
        ],
    },
    {
        "label": ("Share providing informal care by 5-year age bin and education"),
        "dataset": "GSOEP/SHARE",
        "patterns": [
            r"^share_informal_care_(any|light|intensive)_(low|high)_educ_age_bin_\d+_\d+$",
        ],
    },
    {
        "label": "Share pure formal care by own health and education",
        "dataset": "GSOEP-IS",
        "ages_override": "40--70",
        "patterns": [
            r"^pure_formal_care_(low|high)_education_(light|intensive)_care_demand$",
        ],
    },
    {
        "label": "Share of caregivers with high education",
        "dataset": "GSOEP/SHARE",
        "ages_override": "40--70",
        "patterns": [r"^share_informal_care_high_educ$"],
    },
    # ---- Panel B: Labor Supply Shares ----
    {
        "panel": "Panel B: Labor Supply Shares",
        "label": "Labor supply shares by age",
        "dataset": "GSOEP",
        "patterns": [rf"^share_{_LS}_age_\d+$"],
    },
    {
        "label": "Labor supply shares by age and education",
        "dataset": "GSOEP",
        "patterns": [rf"^share_{_LS}_(low|high)_education_age_\d+$"],
    },
    # ---- Panel C: Labor Supply of Caregivers ----
    {
        "panel": "Panel C: Labor Supply of Caregivers",
        "label": "Caregiver labor supply by 3-year age bin",
        "dataset": "GSOEP",
        "patterns": [rf"^share_{_LS}_caregivers_age_bin_\d+_\d+$"],
    },
    {
        "label": "Caregiver labor supply by 3-year age bin and education",
        "dataset": "GSOEP",
        "patterns": [
            rf"^share_{_LS}_caregivers_(low|high)_education_age_bin_\d+_\d+$",
        ],
    },
    {
        "label": "Light caregiver labor supply by 3-year age bin",
        "dataset": "GSOEP",
        "patterns": [rf"^share_{_LS}_light_caregivers_age_bin_\d+_\d+$"],
    },
    {
        "label": ("Light caregiver labor supply by 3-year age bin and education"),
        "dataset": "GSOEP",
        "patterns": [
            rf"^share_{_LS}_light_caregivers_(low|high)_education_age_bin_\d+_\d+$",
        ],
    },
    {
        "label": "Intensive caregiver labor supply by 3-year age bin",
        "dataset": "GSOEP",
        "patterns": [rf"^share_{_LS}_intensive_caregivers_age_bin_\d+_\d+$"],
    },
    {
        "label": (
            "Intensive caregiver labor supply by 3-year age bin" " and education"
        ),
        "dataset": "GSOEP",
        "patterns": [
            rf"^share_{_LS}_intensive_caregivers_(low|high)_education_age_bin_\d+_\d+$",
        ],
    },
    # ---- Panel D: Wealth ----
    {
        "panel": "Panel D: Wealth",
        "label": "Mean wealth by 5-year age bin and education",
        "dataset": "GSOEP",
        "patterns": [
            r"^mean_wealth_(low|high)_education_adjusted_wealth_age_bin_\d+_\d+$",
        ],
    },
    # ---- Panel E: Transitions ----
    {
        "panel": "Panel E: Transitions",
        "label": (
            "Year-to-year labor supply transitions" " by 5-year age bin and education"
        ),
        "dataset": "GSOEP",
        "patterns": [
            r"^trans_(not_working_to_not_working|working_to_working)_(low|high)_education_age_\d+_\d+$",
        ],
    },
    {
        "label": "Year-to-year caregiving transitions by education",
        "dataset": "GSOEP",
        "patterns": [
            r"^trans_caregiving_to_caregiving_(low|high)_education_age_\d+_\d+$",
        ],
    },
]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@pytask.mark.publication_moments_table
def task_create_moments_table(
    path_to_moments: Path = BLD / "moments" / "moments_full_with_mean_wealth.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "moments_overview.tex",
):
    df = pd.read_csv(path_to_moments)
    moments = df["moment"].tolist()

    rows = _build_rows(moments)
    latex = _build_latex(rows)

    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    path_to_save.write_text(latex)


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------


def _count_and_ages(
    moments: list[str],
    patterns: list[str],
) -> tuple[int, str]:
    """Count moments matching any *pattern* and extract the age range.

    For bin-style moments (``_age_bin_`` or short-span ``_age_XX_YY``), the
    upper bound is incremented by one so that e.g. bins ending at 69 are
    reported as 70.
    """
    compiled = [re.compile(p) for p in patterns]
    matched = [m for m in moments if any(c.match(m) for c in compiled)]
    if not matched:
        return 0, "--"

    ages: list[int] = []
    is_bin = False
    for name in matched:
        if "_age_bin_" in name:
            is_bin = True
        tail = name.split("age")[-1]
        nums = re.findall(r"\d+", tail)
        for pair_idx in range(0, len(nums), 2):
            if pair_idx + 1 < len(nums):
                lo, hi = int(nums[pair_idx]), int(nums[pair_idx + 1])
                ages.extend([lo, hi])
                if hi - lo < 10:
                    is_bin = True
            else:
                ages.append(int(nums[pair_idx]))

    if not ages:
        return len(matched), "--"

    min_age = min(ages)
    max_age = max(ages) + 1 if is_bin else max(ages)
    return len(matched), f"{min_age}--{max_age}"


def _build_rows(moments: list[str]) -> list[dict]:
    """For each group, compute count and age range."""
    rows: list[dict] = []
    grand_total = 0

    for grp in _GROUPS:
        n, ages = _count_and_ages(moments, grp["patterns"])
        if "ages_override" in grp:
            ages = grp["ages_override"]
        grand_total += n
        rows.append(
            {
                "label": grp["label"],
                "dataset": grp.get("dataset", "GSOEP"),
                "panel": grp.get("panel"),
                "n": n,
                "ages": ages,
            }
        )

    rows.append({"label": "__total__", "grand_total": grand_total})
    return rows


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def _build_latex(rows: list[dict]) -> str:
    L: list[str] = []

    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\caption{Overview of Estimation Moments}")
    L.append(r"\label{tab:moments_overview}")
    L.append(r"\begin{tabular}{llcc}")
    L.append(r"\toprule")
    L.append(r"Moments & Data Set & Ages & $N$ \\")
    L.append(r"\midrule\midrule")

    for row in rows:
        if row["label"] == "__total__":
            continue
        if row.get("panel"):
            L.append(
                rf"\multicolumn{{4}}{{l}}{{\textit{{{row['panel']}}}}} \\",
            )
            L.append(r"\midrule")

        L.append(
            f"{row['label']} & {row['dataset']}" f" & {row['ages']} & {row['n']} \\\\",
        )

    gt = rows[-1]["grand_total"]
    L.append(r"\midrule")
    L.append(f"Total & & & {gt} \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")
    return "\n".join(L)
