"""Cost-effectiveness scatter for the Policy Experiments chapter.

Produces a single publication-quality scatter plot used in the Policy
Experiments chapter (see `docs/POLICY_EXPERIMENTS_CHAPTER_BLUEPRINT.md`) to
visualise, in one image, the headline finding F1: an earnings-related
caregiving leave at a moderate replacement rate (Familienpflegegeld) achieves
most of the long-run labour-market gains of a 100% upper-bound benchmark
(Full Wage Replacement) at roughly one-tenth of the per-caregiver fiscal
cost.

Axes:
    x: per-caregiver fiscal cost of the policy (EUR), linear scale.
    y: NPV career-cost reduction (percentage points relative to the Status
       Quo baseline) on the labour-plus-pension NPV measure. Positive y
       means the policy reduces the -37% baseline career cost.

One labelled marker per policy. Two thin reference rays from the origin
indicate iso-cost-effectiveness slopes (EUR per percentage-point gained);
the slope from the origin to a policy's marker is its cost-effectiveness
ratio.

Layout policy
-------------
This module deliberately does NOT inherit ``PUBLICATION_PLOT_STYLE`` from
``caregiving.counterfactual.plotting_helpers``. That style is calibrated
for the dcegm event-study plots, which are saved at native size and
embedded close to full page. The scatter here is embedded inside a
text-block at most one ``\\linewidth`` wide and benefits from a more
compact landscape layout with smaller native fonts. ``_LOCAL_STYLE``
below is sized so that the saved PDF reads cleanly when included via
``\\includegraphics[width=\\linewidth]{...}``.

Data source policy
------------------
The numerical values for each policy currently come from the stub dictionary
``STUB_POLICY_RESULTS`` below, sourced from
`docs/CAREER_COSTS_NPV_SUMMARY_COMPARISON.md` and the per-capita fiscal-cost
figures cited in the blueprint Findings (S5). They are explicitly marked as
stubs because the consolidated master tables (Master 1/2/3) have not yet
been produced. Once the prerequisite session generates those tables and
emits a corresponding numbers file, replace ``STUB_POLICY_RESULTS`` with a
loader that reads the canonical inputs.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import publication_savefig

_LOCAL_STYLE: dict[str, float | tuple[float, float] | str] = {
    "figsize": (10.5, 6.0),
    "font_family": "DejaVu Sans",
    "label_fontsize": 14,
    "tick_fontsize": 12,
    "label_text_fontsize": 10.5,
    "annotation_fontsize": 10.5,
    "slope_label_fontsize": 9.5,
    "marker_size": 140,
    "marker_edgewidth": 1.6,
    "ray_linewidth": 1.4,
    "axis_linewidth": 1.2,
    "grid_alpha": 0.18,
    "grid_linewidth": 0.8,
}


STUB_POLICY_RESULTS: dict[str, dict[str, float | str]] = {
    "status_quo": {
        "label": "Status quo (Care Allowance only)",
        "fiscal_cost_per_caregiver_eur": 0.0,
        "npv_reduction_pp": 0.0,
        "marker": "o",
        "filled": True,
    },
    "uncapped_leave": {
        "label": "Uncapped Earnings-Related Leave",
        "fiscal_cost_per_caregiver_eur": 10_273.0,
        "npv_reduction_pp": 3.3,
        "marker": "o",
        "filled": True,
    },
    "reform": {
        "label": "Earnings-Related Leave (Reform)",
        "fiscal_cost_per_caregiver_eur": 2_695.0,
        "npv_reduction_pp": 1.9,
        "marker": "s",
        "filled": True,
    },
    "reform_pt": {
        "label": "Reform (part-time eligibility)",
        "fiscal_cost_per_caregiver_eur": 394.0,
        "npv_reduction_pp": 2.1,
        "marker": "s",
        "filled": False,
    },
    "full_wage_replacement_no_ca": {
        "label": "Full Wage Replacement (no Care Allowance)",
        "fiscal_cost_per_caregiver_eur": 28_309.0,
        "npv_reduction_pp": 2.7,
        "marker": "D",
        "filled": False,
    },
    "full_wage_replacement": {
        "label": "Full Wage Replacement",
        "fiscal_cost_per_caregiver_eur": 28_938.0,
        "npv_reduction_pp": 3.5,
        "marker": "D",
        "filled": True,
    },
}


# Per-marker label placement table to eliminate collisions. Tuples are
# (dx_data, dy_data, ha, va) where dx is in EUR and dy in percentage points.
_LABEL_OFFSETS: dict[str, tuple[float, float, str, str]] = {
    "status_quo": (300.0, 0.18, "left", "bottom"),
    "reform_pt": (350.0, -0.30, "left", "top"),
    "reform": (350.0, 0.22, "left", "bottom"),
    "uncapped_leave": (350.0, 0.22, "left", "bottom"),
    "full_wage_replacement_no_ca": (0.0, -0.38, "center", "top"),
    "full_wage_replacement": (-350.0, 0.22, "right", "bottom"),
}


COST_EFFECTIVENESS_REFERENCE_SLOPES_EUR_PER_PP = (1_500.0, 7_500.0)
"""Reference rays from the origin shown as iso-cost-effectiveness lines.

EUR per percentage-point of NPV career-cost reduction. The 1,500 ray is
roughly the slope of the headline reform; the 7,500 ray is roughly the
slope of the 100%-replacement benchmark. The visual ratio between the two
slopes is the chapter's "10x cost for marginal extra gain" headline.
"""


def _apply_local_style(ax: plt.Axes) -> None:
    style = _LOCAL_STYLE
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=style["tick_fontsize"],
        length=5,
        width=1.0,
    )
    ax.grid(
        which="major",
        axis="both",
        linewidth=style["grid_linewidth"],
        alpha=style["grid_alpha"],
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(style["axis_linewidth"])


def plot_cost_effectiveness_scatter(out_path: Path) -> None:
    """Produce the cost-effectiveness scatter PDF at ``out_path``."""

    style = _LOCAL_STYLE
    plt.rcParams["font.family"] = style["font_family"]

    fig, ax = plt.subplots(figsize=style["figsize"])

    # Fixed visible window. y_max gives label headroom above the highest marker
    # and accommodates the steep slope-ray label at the top of the visible axes.
    x_max = 26_000.0
    y_min = -0.4
    y_max = 5.5

    # --- reference rays from origin -------------------------------------
    for slope_eur_per_pp in COST_EFFECTIVENESS_REFERENCE_SLOPES_EUR_PER_PP:
        # Trace each ray only over its visible extent (clip at the
        # nearest of x_max or y_max).
        x_at_top = y_max * slope_eur_per_pp
        x_end = min(x_max, x_at_top)
        x_ray = np.linspace(0.0, x_end, 50)
        y_ray = x_ray / slope_eur_per_pp
        ax.plot(
            x_ray,
            y_ray,
            linestyle=":",
            linewidth=style["ray_linewidth"],
            color="0.45",
            zorder=1,
        )

        label_text = (f"EUR\u202f{slope_eur_per_pp:,.0f}\u202fper\u202fpp").replace(
            ",", "\u202f"
        )
        slope_bbox = {
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        }

        if x_at_top <= x_max:
            # Steep ray exits at top of axes -> label just inside the top.
            ax.text(
                x_at_top - 200.0,
                y_max - 0.10,
                label_text,
                fontsize=style["slope_label_fontsize"],
                color="0.30",
                ha="right",
                va="top",
                bbox=slope_bbox,
                zorder=2,
            )
        else:
            # Shallow ray exits at right of axes -> label just above the
            # rightmost point of the visible portion of the ray.
            ax.text(
                x_max - 250.0,
                x_max / slope_eur_per_pp + 0.18,
                label_text,
                fontsize=style["slope_label_fontsize"],
                color="0.30",
                ha="right",
                va="bottom",
                bbox=slope_bbox,
                zorder=2,
            )

    # --- markers and per-marker labels ----------------------------------
    for key, policy in STUB_POLICY_RESULTS.items():
        x = float(policy["fiscal_cost_per_caregiver_eur"])
        y = float(policy["npv_reduction_pp"])
        marker = policy["marker"]
        filled = policy["filled"]

        face = "black" if filled else "white"
        ax.scatter(
            x,
            y,
            s=style["marker_size"],
            marker=marker,
            facecolor=face,
            edgecolor="black",
            linewidth=style["marker_edgewidth"],
            zorder=3,
            label=None,
        )

        dx, dy, ha, va = _LABEL_OFFSETS[key]
        ax.text(
            x + dx,
            y + dy,
            policy["label"],
            fontsize=style["label_text_fontsize"],
            ha=ha,
            va=va,
            zorder=4,
        )

    # --- F1 annotation moved to lower-right ----------------------------
    annot = (
        "F1: roughly 10x the per-caregiver fiscal cost\n"
        "for ~1.6 pp of additional long-run labour-market gain"
    )
    ax.text(
        x_max * 0.985,
        y_min + 0.45,
        annot,
        fontsize=style["annotation_fontsize"],
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.40",
            "linewidth": 1.0,
        },
        zorder=5,
    )

    # --- axes, formatting ----------------------------------------------
    ax.set_xlim(-x_max * 0.02, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(
        "Per-caregiver fiscal cost (EUR)",
        fontsize=style["label_fontsize"],
        labelpad=8,
    )
    ax.set_ylabel(
        "NPV career-cost reduction (pp, labour + pension)",
        fontsize=style["label_fontsize"],
        labelpad=8,
    )

    def _euro_fmt(value: float, _pos: int) -> str:
        if value == 0:
            return "0"
        return f"{value/1000:.0f}k".replace(".0", "")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_euro_fmt))
    ax.axhline(0.0, color="0.30", linewidth=style["axis_linewidth"], zorder=2)
    ax.axvline(0.0, color="0.30", linewidth=style["axis_linewidth"], zorder=2)

    _apply_local_style(ax)

    fig.tight_layout(pad=0.6)
    publication_savefig(out_path)
    plt.close(fig)


@pytask.task(id="cost_effectiveness_scatter")
def task_plot_cost_effectiveness_scatter(
    out_path: Annotated[
        Path,
        Product,
    ] = BLD
    / "figures"
    / "publication"
    / "cost_effectiveness_scatter.pdf",
) -> None:
    """Render the cost-effectiveness scatter and save as PDF.

    The produced PDF is referenced from ``docs/policy_experiments_chapter/``
    via ``\\includegraphics{cost_effectiveness_scatter.pdf}`` inside the
    main policy comparison subsection of every chapter version.
    """
    plot_cost_effectiveness_scatter(out_path)
