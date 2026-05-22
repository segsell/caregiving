"""Self-financing decomposition of the caregiving leave policies.

Conceptual reference: Stuermer-Heiber (2024 PhD thesis, ch. 1.7, Tables 1.7 /
1.8 and Figure 1.7). Adapts his self-financing-degree framework to the
caregiving-leave setting.

For a leave policy P with gross per-caregiver outlay g(P) and behaviourally
induced fiscal flows Delta tax(P), Delta unemployment(P), Delta formal_care(P)
(each policy minus baseline, signed so positive values reduce the net cost):

    sf(P) = (Delta tax(P) + Delta unemployment(P) + Delta formal_care(P)) / g(P).

Two perspectives:

  * **Spell period** (Stuermer-Heiber's "Impact period") -- only the within-spell
    tax effects of the leave benefit being either taxable (full-wage-replacement
    variants) or subject to a progressive-tax adjustment (Reform variants).
  * **All periods** (lifecycle aggregate) -- adds the dynamic experience channel
    (preserved employment over the spell raises tax revenue in later periods),
    formal-care expenditure savings, and unemployment-transfer savings.

Reads the master fiscal table at
``bld/tables/publication/fiscal_costs_caregiving_policies.tex`` (produced by
``task_create_fiscal_costs``) and emits the self-financing tables and figure.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD

POLICY_ORDER: tuple[str, ...] = (
    "full_beirat",
    "partial_beirat",
    "normal_leave",
    "norwegian_with_pg",
    "norwegian_no_pg",
)

POLICY_LABELS: dict[str, str] = {
    "full_beirat": "Earnings-Related Caregiving Leave (Reform)",
    "partial_beirat": "Reform (part-time eligibility)",
    "normal_leave": "Uncapped Earnings-Related Caregiving Leave",
    "norwegian_with_pg": "Caregiving leave with full wage replacement",
    "norwegian_no_pg": (
        "Caregiving leave with full wage replacement and without care allowance"
    ),
}

POLICY_LABELS_SHORT: dict[str, str] = {
    "full_beirat": "Reform",
    "partial_beirat": "Reform (PT)",
    "normal_leave": "Uncapped",
    "norwegian_with_pg": "Full Replacement",
    "norwegian_no_pg": "Full Replacement (no CA)",
}

POLICY_COLUMN_KEY: dict[str, str] = {
    "full_beirat": "Full Beirat (65%, 1y full)",
    "partial_beirat": "Partial Beirat (65%)",
    "normal_leave": "Normal leave (65%, no cap)",
    "norwegian_with_pg": "Norwegian (100%, with PG)",
    "norwegian_no_pg": "Norwegian (100%, no PG)",
}


@dataclass
class FiscalRow:
    label: str
    values: dict[str, float]


def _parse_master_fiscal_table(path: Path) -> dict[str, FiscalRow]:
    text = path.read_text(encoding="utf-8")
    # Find the line that starts with "Policy &" (header row of the LaTeX
    # tabular), strip the trailing "\\\\" and split on "&".
    header_line: str | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith(("Policy &", "Policy\t&")):
            header_line = stripped
            break
    if header_line is None:
        raise RuntimeError(f"Header row not found in {path}")
    header_line = header_line.rstrip()
    if header_line.endswith(r"\\"):
        header_line = header_line[:-2].rstrip()
    raw_cols = header_line.split("&")[1:]
    header_cols = [c.strip().replace(r"\%", "%").replace(r"\&", "&") for c in raw_cols]
    rows: dict[str, FiscalRow] = {}
    body = text.split(r"\midrule")[1] if r"\midrule" in text else text
    body = body.split(r"\bottomrule")[0]
    for raw in body.splitlines():
        line = raw.strip()
        if not line or "&" not in line or "Metric" in line:
            continue
        if line.startswith((r"\multicolumn", r"\addlinespace")):
            continue
        if line.endswith(r"\\"):
            line = line[:-2].rstrip()
        cols = [c.strip() for c in line.split("&")]
        metric = cols[0].replace(r"\%", "%").replace(r"\&", "&")
        values: dict[str, float] = {}
        for h, v in zip(header_cols, cols[1:], strict=False):
            if v in {"--", ""}:
                values[h] = float("nan")
            else:
                try:
                    values[h] = float(v)
                except ValueError:
                    values[h] = float("nan")
        rows[metric] = FiscalRow(label=metric, values=values)
    return rows


def _val(rows: dict[str, FiscalRow], metric: str, header: str) -> float:
    if metric not in rows:
        return 0.0
    v = rows[metric].values.get(header, float("nan"))
    return 0.0 if math.isnan(v) else v


@dataclass
class SelfFinancingResult:
    policy: str
    gross_outlay: float
    spell_tax_surplus: float
    lifecycle_tax_surplus: float
    lifecycle_unemployment_savings: float
    lifecycle_formal_care_savings: float

    @property
    def spell_self_financing(self) -> float:
        if self.gross_outlay <= 0:
            return float("nan")
        return self.spell_tax_surplus / self.gross_outlay

    @property
    def all_periods_self_financing(self) -> float:
        if self.gross_outlay <= 0:
            return float("nan")
        total = (
            self.spell_tax_surplus
            + self.lifecycle_tax_surplus
            + self.lifecycle_unemployment_savings
            + self.lifecycle_formal_care_savings
        )
        return total / self.gross_outlay


def compute_self_financing(
    rows: dict[str, FiscalRow],
) -> dict[str, SelfFinancingResult]:
    results: dict[str, SelfFinancingResult] = {}
    for policy_id in POLICY_ORDER:
        col = POLICY_COLUMN_KEY[policy_id]
        delta_col = (
            f"Delta {col} - Baseline".replace(
                "Full Beirat (65%, 1y full)", "Full Beirat"
            )
            .replace("Partial Beirat (65%)", "Partial Beirat")
            .replace("Normal leave (65%, no cap)", "Normal leave")
            .replace("Norwegian (100%, with PG)", "Norwegian (PG)")
            .replace("Norwegian (100%, no PG)", "Norwegian (no PG)")
        )

        gross_outlay = _val(rows, "Avg. leave top-up (gross)", col)
        spell_tax = _val(rows, "Avg. tax increase (progression)", col) + _val(
            rows, "Avg. tax attributable to full leave", col
        )
        delta_total_tax = _val(rows, "Avg. total tax revenue", delta_col)
        lifecycle_tax = max(delta_total_tax - spell_tax, 0.0)
        delta_ue = _val(rows, "Avg. unemployment transfer paid", delta_col)
        ue_savings = -delta_ue
        delta_fc = _val(rows, "Avg. gov. formal care cost per caregiver", delta_col)
        fc_savings = -delta_fc

        results[policy_id] = SelfFinancingResult(
            policy=policy_id,
            gross_outlay=gross_outlay,
            spell_tax_surplus=spell_tax,
            lifecycle_tax_surplus=lifecycle_tax,
            lifecycle_unemployment_savings=ue_savings,
            lifecycle_formal_care_savings=fc_savings,
        )
    return results


def _format_pct(x: float) -> str:
    if math.isnan(x):
        return "--"
    return f"{x * 100:.1f}\\%"


def write_self_financing_degree_table(
    results: dict[str, SelfFinancingResult], path: Path
) -> None:
    """Stuermer-Heiber Table 1.7 analogue."""
    lines: list[str] = []
    lines.extend((r"\begin{table}[htbp]", r"  \centering"))
    lines.append(
        r"  \caption{Self-financing degree of the leave policies. Spell period: "
        r"taxes recovered on the leave benefit itself within the caregiving "
        r"spell. All periods: lifecycle aggregate adding tax revenue from "
        r"preserved employment, unemployment-transfer savings, and reduced "
        r"government formal-care expenditure. Computed in the spirit of "
        r"\citet{stuermerheiber_2024} Table 1.7.}"
    )
    lines.extend((r"  \label{tab:self_financing_degree}", r"  \small"))
    cols = " r" * len(POLICY_ORDER)
    lines.extend((r"  \begin{tabular}{l" + cols + r"}", r"    \toprule"))
    header = " & ".join(POLICY_LABELS_SHORT[p] for p in POLICY_ORDER)
    lines.extend((rf"    Perspective & {header} \\", r"    \midrule"))
    spell = " & ".join(
        _format_pct(results[p].spell_self_financing) for p in POLICY_ORDER
    )
    lines.append(rf"    Spell period & {spell} \\")
    allp = " & ".join(
        _format_pct(results[p].all_periods_self_financing) for p in POLICY_ORDER
    )
    lines.extend(
        (
            rf"    All periods  & {allp} \\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_self_financing_decomposition_table(
    results: dict[str, SelfFinancingResult], path: Path
) -> None:
    """Stuermer-Heiber Table 1.8 analogue: per-Euro channel decomposition."""
    lines: list[str] = []
    lines.extend((r"\begin{table}[htbp]", r"  \centering"))
    lines.append(
        r"  \caption{Decomposition of the All-periods self-financing degree "
        r"into the four contributing channels (per Euro of gross leave outlay). "
        r"\emph{Within-spell tax} is the tax recovered on the leave benefit "
        r"itself (progressive-tax adjustment for the Reform variants; direct "
        r"taxation for the full-wage-replacement variants). \emph{Lifecycle tax} "
        r"is the residual tax revenue from preserved employment over the "
        r"working life. \emph{Unemployment-transfer savings} are the reduced "
        r"means-tested transfers paid because protected employment lifts "
        r"agents above the safety-net floor. \emph{Formal-care expenditure "
        r"savings} are the reduced government outlay on subsidised "
        r"formal-care arrangements as caregivers shift toward informal care. "
        r"Following \citet{stuermerheiber_2024} Table 1.8.}"
    )
    lines.extend((r"  \label{tab:self_financing_decomposition}", r"  \small"))
    cols = " r" * len(POLICY_ORDER)
    lines.extend((r"  \begin{tabular}{l" + cols + r"}", r"    \toprule"))
    header = " & ".join(POLICY_LABELS_SHORT[p] for p in POLICY_ORDER)
    lines.extend(
        (
            rf"    Channel (per EUR\,1 gross outlay) & {header} \\",
            r"    \midrule",
        )
    )

    def per_euro(field: str) -> str:
        cells = []
        for p in POLICY_ORDER:
            r = results[p]
            denom = r.gross_outlay if r.gross_outlay > 0 else float("nan")
            v = {
                "spell_tax": r.spell_tax_surplus,
                "lifecycle_tax": r.lifecycle_tax_surplus,
                "ue_savings": r.lifecycle_unemployment_savings,
                "fc_savings": r.lifecycle_formal_care_savings,
            }[field]
            ratio = v / denom if denom > 0 else float("nan")
            cells.append("--" if math.isnan(ratio) else f"{ratio:.3f}")
        return " & ".join(cells)

    lines.extend(
        (
            rf"    Within-spell tax recovery & {per_euro('spell_tax')} \\",
            rf"    Lifecycle tax surplus     & {per_euro('lifecycle_tax')} \\",
            rf"    Unemployment savings      & {per_euro('ue_savings')} \\",
            rf"    Formal-care savings       & {per_euro('fc_savings')} \\",
            r"    \midrule",
        )
    )
    total = " & ".join(
        _format_pct(results[p].all_periods_self_financing) for p in POLICY_ORDER
    )
    lines.extend(
        (
            rf"    \textbf{{Total self-financing}} & {total} \\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_self_financing_decomposition(
    results: dict[str, SelfFinancingResult], path: Path
) -> None:
    """Stuermer-Heiber Figure 1.7 analogue: stacked-bar of self-financing channels."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    spell = np.array(
        [
            results[p].spell_tax_surplus / results[p].gross_outlay * 100
            for p in POLICY_ORDER
        ]
    )
    lcyc = np.array(
        [
            results[p].lifecycle_tax_surplus / results[p].gross_outlay * 100
            for p in POLICY_ORDER
        ]
    )
    ues = np.array(  # codespell:ignore
        [
            results[p].lifecycle_unemployment_savings / results[p].gross_outlay * 100
            for p in POLICY_ORDER
        ]
    )
    fcs = np.array(
        [
            results[p].lifecycle_formal_care_savings / results[p].gross_outlay * 100
            for p in POLICY_ORDER
        ]
    )

    x = np.arange(len(POLICY_ORDER))
    width = 0.55
    bottoms = np.zeros_like(spell)
    for vals, label, hatch in zip(
        [spell, lcyc, ues, fcs],  # codespell:ignore
        [
            "Within-spell tax recovery",
            "Lifecycle tax surplus",
            "Unemployment savings",
            "Formal-care savings",
        ],
        ["", "//", "..", "xx"],
        strict=True,
    ):
        ax.bar(
            x,
            vals,
            width,
            bottom=bottoms,
            edgecolor="black",
            linewidth=1.2,
            facecolor="white",
            hatch=hatch,
            label=label,
        )
        bottoms = bottoms + vals

    ax.axhline(100, color="0.30", linewidth=1.2, linestyle="--")
    ax.text(
        len(POLICY_ORDER) - 0.15,
        103,
        "100% self-financing",
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.30",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [POLICY_LABELS_SHORT[p] for p in POLICY_ORDER],
        rotation=15,
        ha="right",
        fontsize=12,
    )
    ax.set_ylabel(
        "Self-financing degree (% of gross outlay)",
        fontsize=14,
        labelpad=8,
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.7, alpha=0.18)
    ax.legend(loc="upper right", fontsize=11, frameon=False, ncol=2)

    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


@pytask.task(id="self_financing")
def task_self_financing(
    path_to_master_fiscal: Path = BLD
    / "tables"
    / "publication"
    / "fiscal_costs_caregiving_policies.tex",
    path_to_table_degree: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "self_financing_degree.tex",
    path_to_table_decomposition: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "self_financing_decomposition.tex",
    path_to_figure: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "self_financing_decomposition.pdf",
) -> None:
    """Produce self-financing tables and figure from the master fiscal table."""
    rows = _parse_master_fiscal_table(path_to_master_fiscal)
    results = compute_self_financing(rows)
    write_self_financing_degree_table(results, path_to_table_degree)
    write_self_financing_decomposition_table(results, path_to_table_decomposition)
    plot_self_financing_decomposition(results, path_to_figure)
