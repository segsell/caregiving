"""Publication-style model fit plots.

One PDF per labor outcome per sample per education.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    PUBLICATION_PLOT_STYLE,
    publication_savefig,
)
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    FULL_TIME_CHOICES,
    INFORMAL_CARE,
    PART_TIME,
    PART_TIME_CHOICES,
    RETIREMENT,
    RETIREMENT_CHOICES,
    SEX,
    UNEMPLOYED,
    UNEMPLOYED_CHOICES,
)
from caregiving.moments.task_create_soep_moments import (
    create_df_caregivers,
    create_df_non_caregivers,
    create_df_with_caregivers,
)

_Y_LABELS = {
    0: "Share retired",
    1: "Share non-employed",
    2: "Share working part-time",
    3: "Share full-time",
}

_OUTCOMES = ["retired", "non_employed", "part_time", "full_time"]
_SAMPLES = ["all", "non_caregiver", "caregiver"]
_EDUCATIONS = ["low", "high"]

_S = PUBLICATION_PLOT_STYLE
_X_PAD = 1


def _add_choice_group(data_sim: pd.DataFrame, data_emp: pd.DataFrame) -> None:
    choice_groups_sim = {0: RETIREMENT, 1: UNEMPLOYED, 2: PART_TIME, 3: FULL_TIME}
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }
    data_sim["choice_group"] = np.nan
    data_emp["choice_group"] = np.nan
    for agg_code, raw_codes in choice_groups_sim.items():
        data_sim.loc[
            data_sim["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code
    for agg_code, raw_codes in choice_groups_emp.items():
        data_emp.loc[
            data_emp["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code
    data_sim["choice_group"] = data_sim["choice_group"].astype(int)
    data_emp["choice_group"] = data_emp["choice_group"].astype(int)


def _shares_by_age(
    data_emp: pd.DataFrame,
    data_sim: pd.DataFrame,
    choice_var: int,
    age_min: int,
    age_max: int,
    education: int | None = None,
):
    ages = list(range(age_min, age_max + 1))
    emp_sex = data_emp[data_emp["sex"] == SEX]

    if education is not None:
        emp_sex = emp_sex[emp_sex["education"] == education]
        sim_edu = data_sim[data_sim["education"] == education]
    else:
        sim_edu = data_sim

    sim_shares = (
        sim_edu.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    emp_shares = (
        emp_sex.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    sim_shares = sim_shares.reindex(columns=[0, 1, 2, 3], fill_value=0)
    emp_shares = emp_shares.reindex(columns=[0, 1, 2, 3], fill_value=0)
    vals_sim = sim_shares.reindex(ages, fill_value=0)[choice_var]
    vals_emp = emp_shares.reindex(ages, fill_value=0)[choice_var]
    return ages, vals_sim, vals_emp


def _apply_style(ax):
    ax.grid(True, axis="y", alpha=_S["grid_alpha"], linewidth=_S["grid_linewidth"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=_S["tick_length"], width=_S["tick_width"])
    for label in ax.get_xticklabels():
        label.set_fontsize(_S["xtick_fontsize"])
        label.set_fontfamily(_S["font_family"])
    for label in ax.get_yticklabels():
        label.set_fontsize(_S["ytick_fontsize"])
        label.set_fontfamily(_S["font_family"])
    ax.xaxis.get_label().set_fontsize(_S["label_fontsize"])
    ax.xaxis.get_label().set_fontfamily(_S["font_family"])
    ax.yaxis.get_label().set_fontsize(_S["label_fontsize"])
    ax.yaxis.get_label().set_fontfamily(_S["font_family"])
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(_S["label_fontsize"])
            text.set_fontfamily(_S["font_family"])


def _finalize(ax, path, ymin=None, ymax=None, xmin=None, xmax=None):
    _apply_style(ax)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin - _X_PAD, xmax + _X_PAD)
    if ymin is None or ymax is None:
        ymin, ymax = ax.get_ylim()
    pad = _S["y_axis_margin_factor"] * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax)
    plt.tight_layout()
    plt.subplots_adjust(bottom=_S["subplots_adjust_bottom"])
    publication_savefig(path)
    plt.close()


# ---------------------------------------------------------------------------
# Build product paths: 4 outcomes x 3 samples x 2 educations = 24 plots
# ---------------------------------------------------------------------------

_MODEL_FIT_DIR = BLD / "figures" / "publication" / "model_fit"

_PRODUCT_KEYS: list[tuple[int, str, str, str]] = []
_PRODUCT_PATHS: dict[str, Path] = {}

for _oc_idx, _oc_name in enumerate(_OUTCOMES):
    for _sample in _SAMPLES:
        for _edu in _EDUCATIONS:
            _key = f"path_to_share_{_oc_name}_{_sample}_{_edu}"
            _PRODUCT_KEYS.append((_oc_idx, _oc_name, _sample, _edu))
            _PRODUCT_PATHS[_key] = (
                _MODEL_FIT_DIR / f"share_{_oc_name}_{_sample}_{_edu}.pdf"
            )


@pytask.mark.publication_model_fit
def task_plot_model_fit_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_empirical_data: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_caregivers_sample: Path = BLD
    / "data"
    / "soep_structural_caregivers_sample.csv",
    products: Annotated[dict[str, Path], Product] = _PRODUCT_PATHS,
) -> None:
    """Produce 24 publication PDFs: one per (outcome, sample, education)."""
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    start_year = 2001
    end_year = 2019
    end_age_msm = specs["end_age_msm"]
    start_age = specs["start_age"]
    start_age_caregivers = specs["start_age_caregiving"]
    end_age_caregiver = 69

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    df_emp_full = pd.read_csv(path_to_empirical_data, index_col=[0])
    df_caregivers_full = pd.read_csv(path_to_caregivers_sample, index_col=[0])

    df_emp_all = create_df_with_caregivers(
        df_full=df_emp_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age_msm,
    )
    df_emp_non_caregiver = create_df_non_caregivers(
        df_full=df_emp_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age_msm,
    )
    df_emp_caregiver = create_df_caregivers(
        df_caregivers_full=df_caregivers_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age_msm,
    )

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX
    df_sim["age"] = df_sim["period"] + start_age
    df_sim = df_sim[df_sim["health"] != DEAD].copy()

    df_sim_all = df_sim.copy()
    df_sim_non_caregiver = df_sim.loc[
        ~df_sim["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
    ].copy()
    df_sim_caregiver = df_sim.loc[
        df_sim["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
    ].copy()

    _add_choice_group(df_sim_all, df_emp_all)
    _add_choice_group(df_sim_non_caregiver, df_emp_non_caregiver)
    _add_choice_group(df_sim_caregiver, df_emp_caregiver)

    emp_by_sample = {
        "all": df_emp_all,
        "non_caregiver": df_emp_non_caregiver,
        "caregiver": df_emp_caregiver,
    }
    sim_by_sample = {
        "all": df_sim_all,
        "non_caregiver": df_sim_non_caregiver,
        "caregiver": df_sim_caregiver,
    }
    age_range_by_sample = {
        "all": (start_age, end_age_msm),
        "non_caregiver": (start_age, end_age_msm),
        "caregiver": (start_age_caregivers, end_age_caregiver),
    }

    edu_map = {"low": 0, "high": 1}

    for choice_var, outcome_name, sample_name, edu_label in _PRODUCT_KEYS:
        age_min, age_max = age_range_by_sample[sample_name]
        data_emp = emp_by_sample[sample_name]
        data_sim = sim_by_sample[sample_name]
        edu_idx = edu_map[edu_label]

        ages, vals_sim, vals_emp = _shares_by_age(
            data_emp, data_sim, choice_var, age_min, age_max, education=edu_idx
        )

        fig, ax = plt.subplots(figsize=(_S["figsize"][0], _S["figsize"][1]))
        ax.plot(ages, vals_sim, color="0", linestyle="-", linewidth=_S["linewidth"])
        ax.plot(ages, vals_emp, color="0.5", linestyle="--", linewidth=_S["linewidth"])
        ax.set_xlabel("Age", labelpad=_S["labelpad"])
        ax.set_ylabel(_Y_LABELS[choice_var], labelpad=_S["labelpad"])

        key = f"path_to_share_{outcome_name}_{sample_name}_{edu_label}"
        _finalize(ax, products[key], ymin=0, ymax=1, xmin=age_min, xmax=age_max)


# ---------------------------------------------------------------------------
# Mean wealth model fit (by 5-year age bins and education)
# ---------------------------------------------------------------------------


@pytask.mark.publication_model_fit
def task_plot_mean_wealth_model_fit(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_moments: Path = BLD / "moments" / "moments_full_with_mean_wealth.csv",
    path_to_save_low: Annotated[Path, Product] = _MODEL_FIT_DIR / "mean_wealth_low.pdf",
    path_to_save_high: Annotated[Path, Product] = _MODEL_FIT_DIR
    / "mean_wealth_high.pdf",
) -> None:
    """Mean wealth model fit.

    Simulated vs empirical, by 5-year age bins, per education.
    """
    from caregiving.model.shared import WEALTH_MOMENTS_SCALE

    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    start_age = specs["start_age"]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    moments_df = pd.read_csv(path_to_moments)
    emp_moments = dict(zip(moments_df["moment"], moments_df["value"], strict=False))

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["age"] = df_sim["period"] + start_age
    df_sim = df_sim[df_sim["health"] != DEAD].copy()

    edu_configs = [
        (0, "low", path_to_save_low),
        (1, "high", path_to_save_high),
    ]

    # First pass: compute empirical/simulated values per education
    # and collect all values to determine a common y-range.
    all_values = []
    results_per_edu = []

    for edu_idx, edu_str, path_to_save in edu_configs:
        prefix = f"mean_wealth_{edu_str}_education_adjusted_wealth_age_bin_"
        emp_bins = {}
        for key, val in emp_moments.items():
            if key.startswith(prefix):
                rest = key[len(prefix) :]
                parts = rest.split("_")
                lo, hi = int(parts[0]), int(parts[1])
                emp_bins[lo] = val / WEALTH_MOMENTS_SCALE

        bin_starts = sorted(emp_bins.keys())
        emp_vals = [emp_bins[b] for b in bin_starts]

        sim_edu = df_sim[df_sim["education"] == edu_idx]
        sim_vals = []
        for b in bin_starts:
            hi = b + 5
            mask = (sim_edu["age"] >= b) & (sim_edu["age"] < hi)
            sub = sim_edu.loc[mask, "assets_begin_of_period"]
            sim_vals.append(sub.mean() if len(sub) > 0 else np.nan)

        # Collect values for global y-range (ignore NaNs)
        for v in emp_vals + sim_vals:
            if not np.isnan(v):
                all_values.append(v)

        bin_labels = [f"{b}\u2013{b + 4}" for b in bin_starts]
        results_per_edu.append(
            (edu_idx, edu_str, path_to_save, bin_starts, emp_vals, sim_vals, bin_labels)
        )

    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        # Add a small margin so lines do not touch the plot borders.
        span = y_max - y_min
        if span <= 0:
            span = max(abs(y_max), 1.0)
        margin = 0.05 * span
        ymin = min(0, y_min - margin)
        ymax = y_max + margin
    else:
        ymin, ymax = 0, 1

    # Second pass: create one figure per education using the common y-range.
    for (
        _edu_idx,
        _edu_str,
        path_to_save,
        bin_starts,
        emp_vals,
        sim_vals,
        bin_labels,
    ) in results_per_edu:
        fig, ax = plt.subplots(figsize=(_S["figsize"][0], _S["figsize"][1]))
        ax.plot(
            bin_starts, sim_vals, color="0", linestyle="-", linewidth=_S["linewidth"]
        )
        ax.plot(
            bin_starts,
            emp_vals,
            color="0.5",
            linestyle="--",
            linewidth=_S["linewidth"],
        )
        ax.set_xticks(bin_starts)
        ax.set_xticklabels(bin_labels, rotation=25, ha="right")
        ax.set_xlabel("Age bin", labelpad=_S["labelpad"])
        ax.set_ylabel("Mean wealth (1000 EUR)", labelpad=_S["labelpad"])

        xmin = bin_starts[0]
        xmax = bin_starts[-1]

        _finalize(ax, path_to_save, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
