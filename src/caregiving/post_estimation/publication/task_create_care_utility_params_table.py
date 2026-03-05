"""Create LaTeX table of estimated caregiving-mode utility parameters with SEs."""

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


_COL_ORDER = [
    ("low", "bad"),
    ("low", "good"),
    ("high", "bad"),
    ("high", "good"),
]

_ROWS = [
    ("Light informal care", "util_light_informal_care"),
    ("Intensive informal care", "util_intensive_informal_care"),
    ("Formal care", "util_formal_care"),
]


@pytask.mark.publication_params_table
def task_create_care_utility_params_table(
    path_to_params: Path = BLD / "model" / "params" / "estimated_params_model.yaml",
    path_to_save: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "care_utility_params.tex",
):
    with open(path_to_params) as f:
        params = yaml.safe_load(f)

    se_dict = _load_latest_se()

    L: list[str] = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\caption{Estimated Utility of Caregiving Arrangement}")
    L.append(r"\label{tab:care_utility_params}")
    L.append(r"\begin{tabular}{lcccc}")
    L.append(r"\toprule")
    L.append(
        r" & \multicolumn{2}{c}{Low Education}"
        r" & \multicolumn{2}{c}{High Education} \\"
    )
    L.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    L.append(r" & Bad Health & Good Health & Bad Health & Good Health \\")
    L.append(r" & (1) & (2) & (3) & (4) \\")
    L.append(r"\midrule\midrule")

    for row_idx, (label, prefix) in enumerate(_ROWS):
        keys = [f"{prefix}_{edu}_{health}" for edu, health in _COL_ORDER]
        vals = [params[k] for k in keys]
        ses = [_get_se(se_dict, k) for k in keys]

        cells_v = " & ".join(_v(v) for v in vals)
        L.append(f"{label} & {cells_v} \\\\")

        cells_s = " & ".join(_s(s) for s in ses)
        last = row_idx == len(_ROWS) - 1
        suffix = "" if last else "[2pt]"
        L.append(f" & {cells_s} \\\\{suffix}")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")

    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    path_to_save.write_text("\n".join(L))
