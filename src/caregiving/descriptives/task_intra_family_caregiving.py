"""Intra-family caregiving statistics from SHARE (LaTeX table).

Statistic 1: Share of female caregivers caring for both parents simultaneously
(estimation data). Two denominators: (a) all female caregivers, (b) those with
both parents alive who give any care. General and daily (intensive) care.

Statistic 2: Among individuals with partner alive (parent-child sample), share of
informal care from spouse/partner and from (at least one) child. Two denominators:
(a) among those who receive any informal care, (b) among all with partner alive.
General and daily. Two panels: spouse inside HH only, and spouse inside + outside HH.

Statistic 3: Among families with at least one informal caregiver and at least two
children, share with multiple caregiving children. General and daily.

Weight_type: "hh", "design", "individual", or None. None and "unweighted" are
equivalent (no weights); "unweighted" is the string name for the no-weights option
when invoking the task.
"""

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD

# Weight type for computation: None = unweighted
WeightType = Literal["hh", "design", "individual"] | None

_WEIGHT_COL_ESTIMATION = {
    "hh": "hh_weight",
    "design": "design_weight",
    "individual": "ind_weight",
}


def _share_numerator_denominator(
    num: pd.Series,
    den: pd.Series,
    weights: pd.Series | None,
) -> float:
    """Share = weighted sum(num) / weighted sum(den); or mean(num/den) if no weights."""
    mask = den.fillna(0).astype(bool)
    if mask.sum() == 0:
        return float("nan")
    if weights is None or weights.isna().all():
        return float(num.loc[mask].mean())
    w = weights.loc[mask]
    n = num.loc[mask].astype(float).fillna(0)
    if w.sum() == 0:
        return float("nan")
    return float((n * w).sum() / w.sum())


def _get_weights(
    est: pd.DataFrame,
    pc: pd.DataFrame,
    weight_type: WeightType,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Return (w_est, w_pc) for the given weight type. None means unweighted."""
    w_est: pd.Series | None = None
    w_pc: pd.Series | None = None
    if weight_type is not None:
        col = _WEIGHT_COL_ESTIMATION.get(weight_type)
        if col and col in est.columns:
            w_est = est[col]
        if col and col in pc.columns:
            w_pc = pc[col]
    return (w_est, w_pc)


def _fmt(x: float) -> str:
    return f"{100 * x:.1f}" if not np.isnan(x) else "---"


def _compute_statistic_1_rows(
    est: pd.DataFrame,
    w_est: pd.Series | None,
) -> list[tuple[str, str, str, str]]:
    """Rows for statistic 1: share of female caregivers caring for both parents."""
    care_any = (est["care_to_mother"] == 1) | (est["care_to_father"] == 1)
    both_gen = est["care_to_both_parents"] == 1
    share1a_gen = _share_numerator_denominator(both_gen, care_any, w_est)
    both_int = est["care_to_both_parents_intensive"] == 1
    care_any_int = (est["care_to_mother_intensive"] == 1) | (
        est["care_to_father_intensive"] == 1
    )
    share1a_daily = _share_numerator_denominator(both_int, care_any_int, w_est)
    both_alive = (est["mother_alive"] == 1) & (est["father_alive"] == 1)
    care_any_b = care_any & both_alive
    share1b_gen = _share_numerator_denominator(both_gen, care_any_b, w_est)
    care_any_int_b = care_any_int & both_alive
    share1b_daily = _share_numerator_denominator(both_int, care_any_int_b, w_est)
    return [
        (
            "1",
            "Share of female caregivers caring for both parents (simultaneously)",
            "",
            "",
        ),
        (
            "",
            "\\quad Denom. A: all female caregivers",
            _fmt(share1a_gen),
            _fmt(share1a_daily),
        ),
        (
            "",
            "\\quad Denom. B: both parents alive, any care to parent",
            _fmt(share1b_gen),
            _fmt(share1b_daily),
        ),
    ]


def _compute_statistic_2_rows(
    pc: pd.DataFrame,
    w_pc: pd.Series | None,
) -> list[tuple[str, str, str, str]]:
    """Rows for statistic 2: partner alive — share from spouse vs child (two panels)."""
    married = pc["married"] == 1
    rec_care_gen = pc["informal_care_general"] == 1
    rec_care_daily = pc["informal_care_daily"] == 1
    sub_a_gen = married & rec_care_gen
    sub_a_daily = married & rec_care_daily

    def share_a_daily(num_gen: pd.Series, num_daily: pd.Series) -> tuple[float, float]:
        g = (
            _share_numerator_denominator(
                num_gen.astype(float), sub_a_gen.astype(float), w_pc
            )
            if sub_a_gen.any()
            else float("nan")
        )
        d = (
            _share_numerator_denominator(
                num_daily.astype(float), sub_a_daily.astype(float), w_pc
            )
            if sub_a_daily.any()
            else float("nan")
        )
        return (g, d)

    share_spouse_a_gen, share_spouse_a_daily = share_a_daily(
        (pc["care_from_spouse_partner"] == 1),
        (pc["care_from_spouse_partner_daily"] == 1),
    )
    share_child_a_gen, share_child_a_daily = share_a_daily(
        (pc["informal_care_child"] == 1),
        (pc["informal_care_daily_child"] == 1),
    )
    share_spouse_b_gen = _share_numerator_denominator(
        (pc["care_from_spouse_partner"] == 1).astype(float),
        married.astype(float),
        w_pc,
    )
    share_spouse_b_daily = _share_numerator_denominator(
        (pc["care_from_spouse_partner_daily"] == 1).astype(float),
        married.astype(float),
        w_pc,
    )
    share_child_b_gen = _share_numerator_denominator(
        (pc["informal_care_child"] == 1).astype(float),
        married.astype(float),
        w_pc,
    )
    share_child_b_daily = _share_numerator_denominator(
        (pc["informal_care_daily_child"] == 1).astype(float),
        married.astype(float),
        w_pc,
    )
    share_spouse_inside_a_gen, share_spouse_inside_a_daily = share_a_daily(
        (pc["care_from_spouse_partner_inside_only"] == 1),
        (pc["care_from_spouse_partner_inside_only_daily"] == 1),
    )
    share_spouse_inside_b_gen = _share_numerator_denominator(
        (pc["care_from_spouse_partner_inside_only"] == 1).astype(float),
        married.astype(float),
        w_pc,
    )
    share_spouse_inside_b_daily = _share_numerator_denominator(
        (pc["care_from_spouse_partner_inside_only_daily"] == 1).astype(float),
        married.astype(float),
        w_pc,
    )

    return [
        (
            "2",
            "Among partner alive: share from spouse/partner; share from (≥1) child",
            "",
            "",
        ),
        ("", "\\quad Panel A: Spouse/partner (inside HH only)", "", ""),
        (
            "",
            "\\quad \\quad Denom. A: among those who receive any informal care",
            "",
            "",
        ),
        (
            "",
            "\\quad \\quad \\quad Share from spouse/partner",
            _fmt(share_spouse_inside_a_gen),
            _fmt(share_spouse_inside_a_daily),
        ),
        (
            "",
            "\\quad \\quad \\quad Share from child",
            _fmt(share_child_a_gen),
            _fmt(share_child_a_daily),
        ),
        ("", "\\quad \\quad Denom. B: among all with partner alive", "", ""),
        (
            "",
            "\\quad \\quad \\quad Share from spouse/partner",
            _fmt(share_spouse_inside_b_gen),
            _fmt(share_spouse_inside_b_daily),
        ),
        (
            "",
            "\\quad \\quad \\quad Share from child",
            _fmt(share_child_b_gen),
            _fmt(share_child_b_daily),
        ),
        ("", "\\quad Panel B: Spouse/partner (inside + outside HH)", "", ""),
        (
            "",
            "\\quad \\quad Denom. A: among those who receive any informal care",
            "",
            "",
        ),
        (
            "",
            "\\quad \\quad \\quad Share from spouse/partner",
            _fmt(share_spouse_a_gen),
            _fmt(share_spouse_a_daily),
        ),
        (
            "",
            "\\quad \\quad \\quad Share from child",
            _fmt(share_child_a_gen),
            _fmt(share_child_a_daily),
        ),
        ("", "\\quad \\quad Denom. B: among all with partner alive", "", ""),
        (
            "",
            "\\quad \\quad \\quad Share from spouse/partner",
            _fmt(share_spouse_b_gen),
            _fmt(share_spouse_b_daily),
        ),
        (
            "",
            "\\quad \\quad \\quad Share from child",
            _fmt(share_child_b_gen),
            _fmt(share_child_b_daily),
        ),
    ]


def _compute_statistic_3_rows(
    pc: pd.DataFrame,
    w_pc: pd.Series | None,
) -> list[tuple[str, str, str, str]]:
    """Rows for stat 3: share with ≥2 caregiving children (≥1 care., ≥2 children)."""

    rec_care_gen = pc["informal_care_general"] == 1
    rec_care_daily = pc["informal_care_daily"] == 1
    two_children = pc["has_two_children"] == 1
    sub3_gen = two_children & rec_care_gen
    sub3_daily = two_children & rec_care_daily
    share3_gen = (
        _share_numerator_denominator(
            (pc["informal_care_two_children"] == 1).astype(float),
            sub3_gen.astype(float),
            w_pc,
        )
        if sub3_gen.any()
        else float("nan")
    )
    share3_daily = (
        _share_numerator_denominator(
            (pc["informal_care_daily_two_children"] == 1).astype(float),
            sub3_daily.astype(float),
            w_pc,
        )
        if sub3_daily.any()
        else float("nan")
    )
    return [
        (
            "3",
            "Among ≥1 caregiver, ≥2 children: share with ≥2 caregiving children",  # noqa: E501
            _fmt(share3_gen),
            _fmt(share3_daily),
        ),
    ]


def compute_intra_family_table(
    est: pd.DataFrame,
    pc: pd.DataFrame,
    weight_type: WeightType,
) -> list[tuple[str, str, str, str]]:
    """Compute all table rows for the given weight type. None = unweighted."""
    w_est, w_pc = _get_weights(est, pc, weight_type)
    rows: list[tuple[str, str, str, str]] = []
    rows.extend(_compute_statistic_1_rows(est, w_est))
    rows.extend(_compute_statistic_2_rows(pc, w_pc))
    rows.extend(_compute_statistic_3_rows(pc, w_pc))
    return rows


def build_latex_table(rows: list[tuple[str, str, str, str]]) -> str:
    """Build full LaTeX table from rows (list of 4-tuples)."""
    header = (
        "\\begin{tabular}{llcc}\n"
        "\\toprule\n"
        " & & General & Daily \\\\\n"
        "\\midrule\n"
    )
    body = [" & ".join(x for x in row) + " \\\\" for row in rows]
    body.append("\\bottomrule\n\\end{tabular}")
    return header + "\n".join(body) + "\n"


def _run_intra_family_statistics(
    path_to_estimation_data: Path,
    path_to_parent_child_data: Path,
    path_to_save_table: Path,
    weight_type: WeightType,
) -> None:
    """Load data, compute table for weight_type, write LaTeX to path_to_save_table."""
    est = pd.read_csv(path_to_estimation_data)
    pc = pd.read_csv(path_to_parent_child_data)
    rows = compute_intra_family_table(est, pc, weight_type)
    latex = build_latex_table(rows)
    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)
    path_to_save_table.write_text(latex, encoding="utf-8")


# -----------------------------------------------------------------------------
# Tasks: one table per weight type (hh, design, individual, unweighted)
# -----------------------------------------------------------------------------

_TABLE_DIR = BLD / "descriptives" / "intra_family_caregiving"


@pytask.mark.descriptives
@pytask.mark.family_statistics
def task_intra_family_statistics_hh(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_parent_child_data: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_save_table: Annotated[Path, Product] = _TABLE_DIR
    / "intra_family_caregiving_statistics_hh.tex",
) -> None:
    """Intra-family caregiving statistics (SHARE), household weights."""
    _run_intra_family_statistics(
        path_to_estimation_data,
        path_to_parent_child_data,
        path_to_save_table,
        weight_type="hh",
    )


@pytask.mark.descriptives
@pytask.mark.family_statistics
def task_intra_family_statistics_design(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_parent_child_data: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_save_table: Annotated[Path, Product] = _TABLE_DIR
    / "intra_family_caregiving_statistics_design.tex",
) -> None:
    """Intra-family caregiving statistics (SHARE), design weights."""
    _run_intra_family_statistics(
        path_to_estimation_data,
        path_to_parent_child_data,
        path_to_save_table,
        weight_type="design",
    )


@pytask.mark.descriptives
@pytask.mark.family_statistics
def task_intra_family_statistics_individual(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_parent_child_data: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_save_table: Annotated[Path, Product] = _TABLE_DIR
    / "intra_family_caregiving_statistics_individual.tex",
) -> None:
    """Intra-family caregiving statistics (SHARE), individual weights."""
    _run_intra_family_statistics(
        path_to_estimation_data,
        path_to_parent_child_data,
        path_to_save_table,
        weight_type="individual",
    )


@pytask.mark.descriptives
@pytask.mark.family_statistics
def task_intra_family_statistics_unweighted(
    path_to_estimation_data: Path = BLD / "data" / "share_estimation_data.csv",
    path_to_parent_child_data: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_save_table: Annotated[Path, Product] = _TABLE_DIR
    / "intra_family_caregiving_statistics_unweighted.tex",
) -> None:
    """Intra-family caregiving statistics (SHARE), unweighted (no weights)."""
    _run_intra_family_statistics(
        path_to_estimation_data,
        path_to_parent_child_data,
        path_to_save_table,
        weight_type=None,
    )
