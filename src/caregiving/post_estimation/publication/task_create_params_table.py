"""Create LaTeX table of estimated disutility parameters with standard errors.

Produces one table float with two tabular environments:
- Panels A & B (4 columns): Non-caregiver work/unemployment and children
- Panels C & D (8 columns): Caregiver disutilities with care-intensity splits

For the caregiver panel, the reported disutility for full-time work and
unemployment combines the base informal-care param with the light/intensive
level shift:  reported = base_informal_care - level_shift
(negative shifts add to disutility; positive shifts reduce it).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
import yaml
from pytask import Product

from caregiving.config import BLD

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@pytask.mark.publication_params_table
def task_create_params_table(
    path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
    path_to_save: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "disutility_of_work_params.tex",
):
    _build_and_save(path_to_params, path_to_save, se_dict=_load_latest_se())


@pytask.mark.publication_params_table
def task_create_params_table_model_fit(
    path_to_params: Path = BLD / "model" / "params" / "estimated_params_model_fit.yaml",
    path_to_save: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "disutility_of_work_params_model_fit.tex",
):
    _build_and_save(path_to_params, path_to_save, se_dict={})


def _build_and_save(path_to_params: Path, path_to_save: Path, se_dict: dict):
    with path_to_params.open() as f:
        params = yaml.safe_load(f)

    lines: list[str] = []
    lines.extend((r"\begin{table}[htbp]", r"\centering"))
    lines.extend(
        (r"\caption{Estimated Disutility Parameters}", r"\label{tab:disutil_params}")
    )
    lines.append("")
    lines.extend(_non_caregiver_tabular(params, se_dict))
    lines.extend(("", r"\vspace{0.5cm}"))
    lines.append("")
    lines.extend(_caregiver_tabular(params, se_dict))
    lines.extend(("", r"\end{table}"))
    lines.append("")

    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    path_to_save.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_latest_se() -> dict:
    """Load the most recent ``standard_errors_*.csv`` by modification time."""
    se_dir = BLD / "solve_and_simulate"
    files = sorted(
        se_dir.glob("standard_errors_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return {}
    df = pd.read_csv(files[-1])
    return dict(zip(df["parameter"], df["standard_error"], strict=False))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_NAN = float("nan")


def _v(val) -> str:
    """Format a parameter value to 4 decimal places."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "--"
    return f"{val:.4f}"


def _s(se_val) -> str:
    """Format an SE in parentheses to 4 decimal places."""
    if se_val is None or (isinstance(se_val, float) and math.isnan(se_val)):
        return "--"
    return f"({se_val:.4f})"


def _get_se(se_dict: dict, key: str) -> float:
    """Retrieve SE for *key*, returning NaN if absent / empty."""
    val = se_dict.get(key)
    if val is None:
        return _NAN
    try:
        f = float(val)
        return f if not math.isnan(f) else _NAN
    except (ValueError, TypeError):
        return _NAN


def _combined_se(se_base: float, se_inc: float) -> float:
    """SE of (base - increment) assuming independence."""
    if math.isnan(se_base) or math.isnan(se_inc):
        return _NAN
    return math.sqrt(se_base**2 + se_inc**2)


def _mc(span: int, content: str) -> str:
    r"""Wrap *content* in ``\multicolumn{span}{c}{content}``."""
    return f"\\multicolumn{{{span}}}{{c}}{{{content}}}"


def _val_row(label: str, vals: list) -> str:
    cells = " & ".join(_v(v) for v in vals)
    return f"{label} & {cells} \\\\"


def _se_row(ses: list, *, last: bool = False) -> str:
    cells = " & ".join(_s(s) for s in ses)
    suffix = "" if last else "[2pt]"
    return f" & {cells} \\\\{suffix}"


# ---------------------------------------------------------------------------
# Table 1: Non-caregiver (4 data columns)
# ---------------------------------------------------------------------------

_HEADER_4COL = [
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r" & \multicolumn{2}{c}{Low Education}" r" & \multicolumn{2}{c}{High Education} \\",
    r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
    r" & Bad Health & Good Health & Bad Health & Good Health \\",
    r" & (1) & (2) & (3) & (4) \\",
    r"\midrule\midrule",
]


def _non_caregiver_tabular(p: dict, se: dict) -> list[str]:
    G = _get_se
    L: list[str] = _HEADER_4COL.copy()

    # --- Panel A: Work and Unemployment ---
    L.extend(
        (
            r"\multicolumn{5}{l}{\textit{Panel A:"
            r" Disutility of Work and Unemployment}} \\",
            r"\midrule",
        )
    )

    # Full-time work
    keys = [
        "disutil_ft_work_low_bad",
        "disutil_ft_work_low_good",
        "disutil_ft_work_high_bad",
        "disutil_ft_work_high_good",
    ]
    L.extend(
        (
            _val_row("Full-time work", [p[k] for k in keys]),
            _se_row([G(se, k) for k in keys]),
        )
    )

    # Part-time work
    keys = [
        "disutil_pt_work_low_bad",
        "disutil_pt_work_low_good",
        "disutil_pt_work_high_bad",
        "disutil_pt_work_high_good",
    ]
    L.extend(
        (
            _val_row("Part-time work", [p[k] for k in keys]),
            _se_row([G(se, k) for k in keys]),
        )
    )

    # Unemployed (no health split)
    vl = p["disutil_unemployed_low_women"]
    vh = p["disutil_unemployed_high_women"]
    sl = G(se, "disutil_unemployed_low_women")
    sh = G(se, "disutil_unemployed_high_women")
    L.extend(
        (
            f"Unemployed & {_mc(2, _v(vl))} & {_mc(2, _v(vh))} \\\\",
            f" & {_mc(2, _s(sl))} & {_mc(2, _s(sh))} \\\\",
        )
    )

    L.append(r"\midrule")

    # --- Panel B: Children (non-caregiver, sign-flipped, no health split) ---
    L.extend(
        (
            r"\multicolumn{5}{l}{\textit{Panel B:"
            r" Interaction with Number of Children}} \\",
            r"\midrule",
        )
    )

    # FT × children
    vl = -p["disutil_children_ft_work_low"]
    vh = -p["disutil_children_ft_work_high"]
    sl = G(se, "disutil_children_ft_work_low")
    sh = G(se, "disutil_children_ft_work_high")
    L.extend(
        (
            f"Full-time work $\\times$ children"
            f" & {_mc(2, _v(vl))} & {_mc(2, _v(vh))} \\\\",
            f" & {_mc(2, _s(sl))} & {_mc(2, _s(sh))} \\\\[2pt]",
        )
    )

    # PT × children
    vl = -p["disutil_children_pt_work_low"]
    vh = -p["disutil_children_pt_work_high"]
    sl = G(se, "disutil_children_pt_work_low")
    sh = G(se, "disutil_children_pt_work_high")
    L.extend(
        (
            f"Part-time work $\\times$ children"
            f" & {_mc(2, _v(vl))} & {_mc(2, _v(vh))} \\\\",
            f" & {_mc(2, _s(sl))} & {_mc(2, _s(sh))} \\\\",
        )
    )

    L.extend((r"\bottomrule", r"\end{tabular}"))
    return L


# ---------------------------------------------------------------------------
# Table 2: Caregiver (8 data columns)
# ---------------------------------------------------------------------------

_HEADER_8COL = [
    r"\begin{tabular}{lcccccccc}",
    r"\toprule",
    r" & \multicolumn{4}{c}{Low Education}" r" & \multicolumn{4}{c}{High Education} \\",
    r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
    r" & \multicolumn{2}{c}{Bad Health}"
    r" & \multicolumn{2}{c}{Good Health}"
    r" & \multicolumn{2}{c}{Bad Health}"
    r" & \multicolumn{2}{c}{Good Health} \\",
    r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}" r" \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
    r" & Light & Intensive & Light & Intensive"
    r" & Light & Intensive & Light & Intensive \\",
    r" & (1) & (2) & (3) & (4)" r" & (5) & (6) & (7) & (8) \\",
    r"\midrule\midrule",
]

# Column order for the 8-col caregiver panel:
#   Low/Bad/Light, Low/Bad/Int, Low/Good/Light, Low/Good/Int,
#   High/Bad/Light, High/Bad/Int, High/Good/Light, High/Good/Int
_COL = [
    ("low", "bad", "light"),
    ("low", "bad", "intensive"),
    ("low", "good", "light"),
    ("low", "good", "intensive"),
    ("high", "bad", "light"),
    ("high", "bad", "intensive"),
    ("high", "good", "light"),
    ("high", "good", "intensive"),
]


def _caregiver_tabular(p: dict, se: dict) -> list[str]:  # noqa: PLR0914
    G = _get_se
    L: list[str] = _HEADER_8COL.copy()

    # --- Panel C: Work and Unemployment While Caregiving ---
    L.extend(
        (
            r"\multicolumn{9}{l}{\textit{Panel C:"
            r" Disutility of Work and Unemployment While Caregiving}} \\",
            r"\midrule",
        )
    )

    # -- Full-time work: combined = base(educ,health) - shift(educ,intensity) --
    ft_base = {
        ("low", "bad"): "disutil_ft_work_low_bad_informal_care",
        ("low", "good"): "disutil_ft_work_low_good_informal_care",
        ("high", "bad"): "disutil_ft_work_high_bad_informal_care",
        ("high", "good"): "disutil_ft_work_high_good_informal_care",
    }
    ft_inc = {
        ("low", "light"): "disutil_ft_work_light_informal_care_low",
        ("low", "intensive"): "disutil_ft_work_intensive_informal_care_low",
        ("high", "light"): "disutil_ft_work_light_informal_care_high",
        ("high", "intensive"): "disutil_ft_work_intensive_informal_care_high",
    }
    vals, ses = [], []
    for educ, health, intensity in _COL:
        bk, ik = ft_base[(educ, health)], ft_inc[(educ, intensity)]
        vals.append(p[bk] - p[ik])
        ses.append(_combined_se(G(se, bk), G(se, ik)))
    L.extend((_val_row("Full-time work", vals), _se_row(ses)))

    # -- Part-time work: no light/intensive split --
    pt_keys = [
        ("low", "bad", "disutil_pt_work_low_bad_informal_care"),
        ("low", "good", "disutil_pt_work_low_good_informal_care"),
        ("high", "bad", "disutil_pt_work_high_bad_informal_care"),
        ("high", "good", "disutil_pt_work_high_good_informal_care"),
    ]
    v_cells = " & ".join(_mc(2, _v(p[k])) for _, _, k in pt_keys)
    s_cells = " & ".join(_mc(2, _s(G(se, k))) for _, _, k in pt_keys)
    L.extend((f"Part-time work & {v_cells} \\\\", f" & {s_cells} \\\\[2pt]"))

    # -- Unemployed: combined = base(educ) - shift(educ, intensity) --
    # No health split, so bad/good columns within each (educ, intensity) pair
    # show identical values.
    ue_base = {
        "low": "disutil_unemployed_low_women_informal_care",
        "high": "disutil_unemployed_high_women_informal_care",
    }
    ue_shift = {
        ("low", "light"): "disutil_unemployed_light_informal_care_low",
        ("low", "intensive"): "disutil_unemployed_intensive_informal_care_low",
        ("high", "light"): "disutil_unemployed_light_informal_care_high",
        ("high", "intensive"): "disutil_unemployed_intensive_informal_care_high",
    }
    ue_vals, ue_ses = [], []
    for educ, _health, intensity in _COL:
        bk = ue_base[educ]
        sk = ue_shift[(educ, intensity)]
        ue_vals.append(p[bk] - p[sk])
        ue_ses.append(_combined_se(G(se, bk), G(se, sk)))
    L.extend((_val_row("Unemployed", ue_vals), _se_row(ue_ses, last=True)))

    L.append(r"\midrule")

    # --- Panel D: Children while caregiving ---
    # (sign-flipped, no health/intensity split)
    L.extend(
        (
            r"\multicolumn{9}{l}{\textit{Panel D:"
            r" Interaction with Number of Children While Caregiving}} \\",
            r"\midrule",
        )
    )

    # FT × children (caregiver variant)
    vl = -p["disutil_children_ft_work_low_informal_care"]
    vh = -p["disutil_children_ft_work_high_informal_care"]
    sl = G(se, "disutil_children_ft_work_low_informal_care")
    sh = G(se, "disutil_children_ft_work_high_informal_care")
    L.extend(
        (
            f"Full-time work $\\times$ children"
            f" & {_mc(4, _v(vl))} & {_mc(4, _v(vh))} \\\\",
            f" & {_mc(4, _s(sl))} & {_mc(4, _s(sh))} \\\\[2pt]",
        )
    )

    # PT × children (caregiver variant)
    vl = -p["disutil_children_pt_work_low_informal_care"]
    vh = -p["disutil_children_pt_work_high_informal_care"]
    sl = G(se, "disutil_children_pt_work_low_informal_care")
    sh = G(se, "disutil_children_pt_work_high_informal_care")
    L.extend(
        (
            f"Part-time work $\\times$ children"
            f" & {_mc(4, _v(vl))} & {_mc(4, _v(vh))} \\\\",
            f" & {_mc(4, _s(sl))} & {_mc(4, _s(sh))} \\\\",
        )
    )

    L.extend((r"\bottomrule", r"\end{tabular}"))
    return L
