"""Create LaTeX table of estimated job-finding parameters with standard errors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
import yaml
from pytask import Product

from caregiving.config import BLD

_NAN = float("nan")

_PARAMS = [
    ("Constant", "job_finding_logit_const_women"),
    ("Age / 10", "job_finding_logit_age_women"),
    ("(Age / 10)$^2$", "job_finding_logit_age_squared_women"),
    ("(Age / 10)$^3$", "job_finding_logit_age_cubed_women"),
    ("High education", "job_finding_logit_high_educ_women"),
]


def _v(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "--"
    return f"{val:.4f}"


def _s(se_val) -> str:
    if se_val is None or (isinstance(se_val, float) and math.isnan(se_val)):
        return "--"
    return f"({se_val:.4f})"


def _load_latest_se() -> dict:
    se_dir = BLD / "solve_and_simulate"
    files = sorted(
        se_dir.glob("standard_errors_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return {}
    df = pd.read_csv(files[-1])
    return dict(zip(df["parameter"], df["standard_error"]))


def _get_se(se_dict: dict, key: str) -> float:
    val = se_dict.get(key)
    if val is None:
        return _NAN
    try:
        f = float(val)
        return f if not math.isnan(f) else _NAN
    except (ValueError, TypeError):
        return _NAN


@pytask.mark.publication_params_table
def task_create_job_offer_params_table(
    path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
    path_to_save: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "job_offer_params.tex",
):
    with open(path_to_params) as f:
        params = yaml.safe_load(f)

    se_dict = _load_latest_se()

    L: list[str] = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\caption{Estimated Job-Finding Parameters}")
    L.append(r"\label{tab:job_offer_params}")
    L.append(r"\begin{tabular}{lc}")
    L.append(r"\toprule")
    L.append(r" & Estimate \\")
    L.append(r"\midrule\midrule")

    for label, key in _PARAMS:
        val = params[key]
        se = _get_se(se_dict, key)
        L.append(f"{label} & {_v(val)} \\\\")
        last = key == _PARAMS[-1][1]
        suffix = "" if last else "[2pt]"
        L.append(f" & {_s(se)} \\\\{suffix}")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")

    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    path_to_save.write_text("\n".join(L))
