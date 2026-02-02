"""Functions to plot model fit between empirical and simulated data."""

from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from caregiving.config import JET_COLOR_MAP
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    FULL_TIME_CHOICES,
    PART_TIME,
    PART_TIME_CHOICES,
    RETIREMENT,
    RETIREMENT_CHOICES,
    SEX,
    UNEMPLOYED,
    UNEMPLOYED_CHOICES,
)

# ====================================================================================
# Wealth
# ====================================================================================


def plot_wealth_by_age_and_education(
    data_emp: pd.DataFrame,
    data_sim: pd.DataFrame,
    specs: dict,
    *,
    wealth_var_emp: str,
    wealth_var_sim: str,
    median: bool = False,
    age_min: int | None = None,
    age_max: int | None = None,
    path_to_save_plot: str | None = None,
):
    """
    Plot average/median wealth by age and education (Observed vs Simulated),
    with education groups side by side, shared y-axis, y-labels on left for both
    panels, and internal x-padding near the vertical axes.

    """
    # ---------- 0. Setup ----------
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    ages = range(age_min, age_max + 1)

    n_edu = len(specs["education_labels"])
    fig, axs = plt.subplots(1, n_edu, figsize=(5 * n_edu, 4), sharex=True, sharey=True)
    if n_edu == 1:
        axs = np.array([axs])

    agg = (
        (lambda s: s.median(skipna=True)) if median else (lambda s: s.mean(skipna=True))
    )

    # ---------- 1. Loop over education groups ----------
    all_values = []
    for edu_idx, edu_label in enumerate(specs["education_labels"]):
        ax = axs[edu_idx]

        emp_edu = data_emp[data_emp["education"] == edu_idx]
        sim_edu = data_sim[data_sim["education"] == edu_idx]

        emp_series = (
            emp_edu.groupby("age", observed=False)[wealth_var_emp]
            .apply(agg)
            .reindex(ages, fill_value=np.nan)
        )
        sim_series = (
            sim_edu.groupby("age", observed=False)[wealth_var_sim]
            .apply(agg)
            .reindex(ages, fill_value=np.nan)
        )

        all_values.extend(emp_series.dropna().tolist())
        all_values.extend(sim_series.dropna().tolist())

        ax.plot(ages, sim_series.values, label="Simulated")
        ax.plot(ages, emp_series.values, ls="--", label="Observed")

        # Add internal x padding (5% of the range)
        xrange = age_max - age_min
        pad = int(0.05 * xrange)
        ax.set_xlim(age_min - pad, age_max + pad)

        ax.set_xlabel("Age")
        ax.set_title(edu_label)
        ax.grid(True, alpha=0.2)

        if edu_idx == 0:
            ax.set_ylabel("Wealth (in 1000 €)")
            ax.legend()
        else:
            # remove default left ticks
            ax.tick_params(labelleft=False)

    # ---------- 2. Common y-limits ----------
    if all_values:
        ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
        pad = 0.05 * (ymax - ymin)
        for ax in axs:
            ax.set_ylim(ymin - pad, ymax + pad)

    # ---------- 3. Add left y-axis to right panel too ----------
    axs[-1].yaxis.set_ticks_position("left")
    axs[-1].yaxis.set_label_position("left")
    axs[-1].set_ylabel("Wealth (in 1000 €)")

    plt.tight_layout()

    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_wealth_by_age_bins_and_education(  # noqa: PLR0912, PLR0915
    data_emp: pd.DataFrame,
    data_sim: pd.DataFrame,
    specs: dict,
    *,
    wealth_var_emp: str,
    wealth_var_sim: str,
    median: bool = False,
    age_min: int | None = None,
    age_max: int | None = None,
    bin_width: int = 5,
    age_bin_ticks: bool = False,
    path_to_save_plot: str | None = None,
):
    """
    Plot average/median wealth by age bins and education (Observed vs Simulated),
    with education groups side by side, shared y-axis, y-labels on left for both
    panels, and internal x-padding near the vertical axes.

    Parameters
    ----------
    age_bin_ticks : bool, default False
        If True: X-axis labels are range-style (e.g., "55-59"), rotated 45°,
        and shown at the bottom of every subplot with label "Age bin".
        If False: X-axis ticks every 5 years (e.g., 40, 45, 50...) with label "Age".
    """
    # ---------- 0. Setup ----------
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]

    # Build age bins & labels
    edges = list(range(age_min, age_max + 1, bin_width))
    if edges[-1] <= age_max:
        edges.append(age_max + 1)

    # Left-edge x positions and labels like "55-59"
    bin_starts = edges[:-1]
    bin_labels = [
        f"{start}-{end - 1}" for start, end in zip(edges[:-1], edges[1:], strict=False)
    ]

    agg = (
        (lambda s: s.median(skipna=True)) if median else (lambda s: s.mean(skipna=True))
    )

    n_edu = len(specs["education_labels"])
    fig, axs = plt.subplots(1, n_edu, figsize=(5 * n_edu, 4), sharex=True, sharey=True)
    if n_edu == 1:
        axs = np.array([axs])

    # ---------- 1. Loop over education groups ----------
    all_values = []
    for edu_idx, edu_label in enumerate(specs["education_labels"]):
        ax = axs[edu_idx]

        emp_edu = data_emp[data_emp["education"] == edu_idx]
        sim_edu = data_sim[data_sim["education"] == edu_idx]

        emp_rates = []
        sim_rates = []

        for start, end in zip(edges[:-1], edges[1:], strict=False):
            # Empirical bin
            emp_bin = emp_edu[(emp_edu["age"] >= start) & (emp_edu["age"] < end)]
            emp_rates.append(agg(emp_bin[wealth_var_emp]) if len(emp_bin) else np.nan)
            # Simulated bin
            sim_bin = sim_edu[(sim_edu["age"] >= start) & (sim_edu["age"] < end)]
            sim_rates.append(agg(sim_bin[wealth_var_sim]) if len(sim_bin) else np.nan)

        all_values.extend([v for v in emp_rates if not np.isnan(v)])
        all_values.extend([v for v in sim_rates if not np.isnan(v)])

        ax.plot(bin_starts, sim_rates, label="Simulated")
        ax.plot(bin_starts, emp_rates, ls="--", label="Observed")

        # Add internal x padding (similar to original function)
        if len(bin_starts) > 1:
            xrange = bin_starts[-1] - bin_starts[0]
            pad = 0.05 * xrange
            ax.set_xlim(bin_starts[0] - pad, bin_starts[-1] + pad)
        else:
            # Single bin: give symmetric whitespace
            pad = 0.5 * bin_width
            ax.set_xlim(bin_starts[0] - pad, bin_starts[0] + pad)

        # --- Conditional x-axis tick behavior ---
        if age_bin_ticks:
            # Range-style x-axis labels (e.g., "55-59"), rotated 45°
            ax.set_xticks(bin_starts)
            ax.set_xticklabels(bin_labels, rotation=45, ha="right")
            xlabel = "Age bin"
        else:
            # X-axis ticks every 5 years (e.g., 40, 45, 50...)
            tick_positions = list(range(age_min, age_max + 1, 5))
            ax.set_xticks(tick_positions)
            xlabel = "Age"

        ax.set_xlabel(xlabel)
        ax.set_title(edu_label)
        ax.grid(True, alpha=0.2)

        if edu_idx == 0:
            ax.set_ylabel("Wealth (in 1000 €)")
            ax.legend()
        else:
            # remove default left ticks
            ax.tick_params(labelleft=False)

    # ---------- 2. Common y-limits ----------
    if all_values:
        ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
        pad = 0.05 * (ymax - ymin)
        for ax in axs:
            ax.set_ylim(ymin - pad, ymax + pad)

    # ---------- 3. Add left y-axis to right panel too ----------
    axs[-1].yaxis.set_ticks_position("left")
    axs[-1].yaxis.set_label_position("left")
    axs[-1].set_ylabel("Wealth (in 1000 €)")

    plt.tight_layout()

    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_average_wealth(
    data_emp,
    data_sim,
    specs,
    path_to_save_plot,
):
    """Plot average wealth by age and education."""

    # data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_sim.loc[:, "age"] = data_sim["period"] + specs["start_age"]

    sex = SEX
    for edu in range(2):
        mask_sim = (data_sim["sex"] == sex) & (data_sim["education"] == edu)
        data_sim_edu = data_sim[mask_sim]
        mask_obs = (data_emp["sex"] == sex) & (data_emp["education"] == edu)
        data_decision_edu = data_emp[mask_obs]

        ages = np.arange(specs["start_age"] + 1, 90)
        average_wealth_sim = (
            data_sim_edu.groupby("age")["assets_begin_of_period"].median().loc[ages]
        )
        average_wealth_obs = (
            data_decision_edu.groupby("age")["adjusted_wealth"].median().loc[ages]
        )

        fig, ax = plt.subplots()

        ax.plot(ages, average_wealth_sim, label="Simulated")
        ax.plot(
            ages,
            average_wealth_obs,
            label="Median observed wealth by age",
            ls="--",
        )
        ax.legend()
        fig.savefig(path_to_save_plot, transparent=True, dpi=300)


# =====================================================================================
# Choices
# =====================================================================================


def plot_choice_shares_by_education(
    data_emp,
    data_sim,
    specs,
    age_min=None,
    age_max=None,
    choice_groups_sim=None,
    path_to_save_plot=None,
):
    """Plot choice-specific shares by age and education, only over the
    age range [start_age, end_age_msm]."""

    # ---------- 1. Map raw codes → 4-way choice ----------------------------
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }
    if choice_groups_sim is None:
        choice_groups_sim = {
            0: RETIREMENT,
            1: UNEMPLOYED,
            2: PART_TIME,
            3: FULL_TIME,
        }

    data_sim = data_sim.loc[data_sim["health"] != DEAD].copy()
    data_emp = data_emp.copy()

    for agg_code, raw_codes in choice_groups_sim.items():
        data_sim.loc[
            data_sim["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code

    for agg_code, raw_codes in choice_groups_emp.items():
        data_emp.loc[
            data_emp["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code

    data_sim["choice_group"] = data_sim["choice_group"].fillna(0).astype(int)
    # data_sim["choice_group"] = data_sim["choice_group"].astype(int)
    data_emp["choice_group"] = data_emp["choice_group"].astype(int)

    # ---------- 2. Plotting setup ------------------------------------------
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    sex = SEX  # assumed scalar {0,1}

    n_edu = len(specs["education_labels"])
    # Ensure hashable elements when deriving number of aggregated choices
    # choice_groups_sim.values() may contain JAX arrays; convert to tuples of ints
    # n_choices = sum(arr.size for arr in choice_groups_sim.values())
    n_choices = len(specs["choice_labels"])
    fig, axs = plt.subplots(n_edu, n_choices, figsize=(16, 6), sharex=True, sharey=True)

    # ---------- 3. Loop over education groups ------------------------------
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        emp_edu = data_emp[
            (data_emp["sex"] == sex) & (data_emp["education"] == edu_var)
        ]
        sim_edu = data_sim[data_sim["education"] == edu_var]

        # shares by age × aggregated choice
        sim_shares = (
            sim_edu.groupby("age", observed=False)["choice_group"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        emp_shares = (
            emp_edu.groupby("age", observed=False)["choice_group"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

        # ---------- 4. Plot each aggregated choice -------------------------
        for choice_var in range(n_choices):
            ax = axs[edu_var, choice_var]

            ages = range(age_min, age_max + 1)
            vals_sim = sim_shares.reindex(ages, fill_value=0)[choice_var]
            vals_emp = emp_shares.reindex(ages, fill_value=0)[choice_var]

            ax.plot(ages, vals_sim, label="Simulated")
            ax.plot(ages, vals_emp, ls="--", label="Observed")

            ax.set_title(specs["choice_labels"][choice_var], fontsize=14)
            ax.set_ylim(0, 1)
            ax.set_xlim(age_min, age_max)

            # if edu_var == n_edu - 1:
            ax.set_xlabel("Age", fontsize=12)
            ax.tick_params(labelbottom=True, labelsize=11)
            if choice_var == 0:
                ax.set_ylabel(f"{edu_label}\nShare", fontsize=12)
                ax.legend(prop={"size": 11})

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_choice_shares_overall(
    data_emp, data_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot choice-specific shares by age, unconditional on education level.

    Parameters
    ----------
    data_emp : pandas.DataFrame
        Observed micro data. Must contain columns: 'age', 'sex', 'education', 'choice'.
    data_sim : pandas.DataFrame
        Simulated data with the same columns as `data_emp`.
    specs : dict
        Model specification dictionary.
    path_to_save_plot : str | pathlib.Path | None, optional
        If given, the figure is saved to this location.
    """

    # ---------- 1. Map raw codes → 4-way aggregated choice -----------------
    choice_groups_sim = {
        0: RETIREMENT,
        1: UNEMPLOYED,
        2: PART_TIME,
        3: FULL_TIME,
    }
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }

    data_sim = data_sim.loc[data_sim["health"] != DEAD].copy()
    data_emp = data_emp.copy()

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

    # ---------- 2. Plotting setup ------------------------------------------
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    sex = SEX  # assumed scalar {0,1}

    n_choices = 4  # after aggregation
    fig, axs = plt.subplots(1, n_choices, figsize=(16, 4), sharex=True, sharey=True)

    # ---------- 3. Subset data (ignore education) ---------------------------
    emp_all = data_emp[data_emp["sex"] == sex]
    sim_all = data_sim  # simulated data already restricted to relevant sex earlier

    # Pre-compute age × choice shares
    sim_shares = (
        sim_all.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    emp_shares = (
        emp_all.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # ---------- 4. Plot each aggregated choice ------------------------------
    ages = range(age_min, age_max + 1)
    for choice_var in range(n_choices):
        ax = axs[choice_var]

        vals_sim = sim_shares.reindex(ages, fill_value=0)[choice_var]
        vals_emp = emp_shares.reindex(ages, fill_value=0)[choice_var]

        ax.plot(ages, vals_sim, label="Simulated")
        ax.plot(ages, vals_emp, ls="--", label="Observed")

        ax.set_title(specs["choice_labels"][choice_var])
        ax.set_ylim(0, 1)
        ax.set_xlim(age_min, age_max)

        ax.set_xlabel("Age")
        if choice_var == 0:
            ax.set_ylabel("Share")
            ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_choice_shares_by_education_age_bins(  # noqa: PLR0912, PLR0915
    data_emp,
    data_sim,  # <— added
    specs,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    bin_width: int = 5,
    age_bin_ticks: bool = False,
    path_to_save_plot: str | None = None,
):
    """
    Plot observed and simulated choice shares by age bins and education.
    Each panel contains one aggregated choice; rows are education groups.

    Parameters
    ----------
    age_bin_ticks : bool, default False
        If True: X-axis labels are range-style (e.g., "55-59"), rotated 45°,
        and shown at the bottom of every subplot with label "Age bin".
        If False: X-axis ticks every 5 years (e.g., 40, 45, 50...) with label "Age".
    """

    # ── 1. Map raw choice codes → 4 aggregated groups ────────────────────────
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }
    # added: mapping for simulated data (single codes except full-time group)
    choice_groups_sim = {
        0: RETIREMENT,
        1: UNEMPLOYED,
        2: PART_TIME,
        3: FULL_TIME,
    }

    emp = data_emp.copy()
    for g, raw in choice_groups_emp.items():
        emp.loc[emp["choice"].isin(np.atleast_1d(raw)), "choice_group"] = g
    emp["choice_group"] = emp["choice_group"].astype(int)

    # added: prepare simulated data (exclude DEAD and map to choice_group)
    sim = data_sim.loc[data_sim["health"] != DEAD].copy()
    for g, raw in choice_groups_sim.items():
        sim.loc[sim["choice"].isin(np.atleast_1d(raw)), "choice_group"] = g
    sim["choice_group"] = sim["choice_group"].astype(int)

    # ── 2. Build age bins & labels ───────────────────────────────────────────
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]

    # Bin edges: [start, next_start, ..., age_max+1]
    edges = list(range(age_min, age_max + 1, bin_width))
    if edges[-1] <= age_max:
        edges.append(age_max + 1)

    # Left-edge x positions and labels like "55-59"
    bin_starts = edges[:-1]
    bin_labels = [
        f"{start}-{end - 1}" for start, end in zip(edges[:-1], edges[1:], strict=False)
    ]

    sex = SEX
    edu_labels = specs["education_labels"]
    choice_labels = specs["choice_labels"]
    n_edu, n_choices = len(edu_labels), len(choice_labels)

    # sharey only—no sharex so we can force labels on each row cleanly
    fig, axs = plt.subplots(
        n_edu, n_choices, figsize=(4 * n_choices, 3 * n_edu), sharey=True
    )

    # Ensure axs is 2D with shape (n_edu, n_choices)
    if n_edu == n_choices == 1:
        axs = np.array([[axs]])
    elif n_edu == 1:
        axs = axs[np.newaxis, :]
    elif n_choices == 1:
        axs = axs[:, np.newaxis]

    # --- Padding amounts ---
    y_pad = 0.03  # adds whitespace above 1 and below 0
    # compute a representative bin step for x padding
    if len(bin_starts) > 1:
        step = float(np.median(np.diff(bin_starts)))
    else:
        step = float(bin_width) if bin_width is not None else 1.0
    x_pad = 0.35 * step  # "little" left/right whitespace (~35% of one bin)

    # ── 3. Loop over education × choice bins ─────────────────────────────────
    for i, edu_label in enumerate(edu_labels):
        emp_sub = emp[(emp["sex"] == sex) & (emp["education"] == i)]
        # added: simulated subset (by education only to mirror your non-binned function)
        sim_sub = sim[sim["education"] == i]

        for j in range(n_choices):
            ax = axs[i, j]
            emp_rates = []
            sim_rates = []  # added

            for start, end in zip(edges[:-1], edges[1:], strict=False):
                # empirical bin
                emp_bin = emp_sub[(emp_sub["age"] >= start) & (emp_sub["age"] < end)]
                emp_rates.append(
                    (emp_bin["choice_group"] == j).mean() if len(emp_bin) else np.nan
                )
                # simulated bin (added)
                sim_bin = sim_sub[(sim_sub["age"] >= start) & (sim_sub["age"] < end)]
                sim_rates.append(
                    (sim_bin["choice_group"] == j).mean() if len(sim_bin) else np.nan
                )

            # plot simulated + observed (observed dashed as before)
            ax.plot(bin_starts, sim_rates, label="Simulated")  # added
            ax.plot(bin_starts, emp_rates, ls="--", label="Observed")

            ax.set_title(choice_labels[j], fontsize=14)

            # --- Uniform axes + whitespace on all sides ---
            ax.set_ylim(-y_pad, 1 + y_pad)  # bottom/top whitespace
            if len(bin_starts) > 1:
                ax.set_xlim(
                    bin_starts[0] - x_pad, bin_starts[-1] + x_pad
                )  # left/right whitespace
            else:
                # single bin: give symmetric whitespace
                ax.set_xlim(bin_starts[0] - x_pad, bin_starts[0] + x_pad)

            # --- Conditional x-axis tick behavior ---
            if age_bin_ticks:
                # Range-style x-axis labels (e.g., "55-59"), rotated 45°
                ax.set_xticks(bin_starts)
                ax.set_xticklabels(bin_labels)
                ax.tick_params(
                    axis="x",
                    labelbottom=True,
                    labeltop=False,
                    bottom=True,
                    top=False,
                    labelsize=11,
                )
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(45)
                    lbl.set_ha("right")
                    lbl.set_rotation_mode("anchor")
                xlabel = "Age bin"
            else:
                # X-axis ticks every 5 years (e.g., 40, 45, 50...)
                tick_positions = list(range(age_min, age_max + 1, 5))
                ax.set_xticks(tick_positions)
                ax.tick_params(
                    axis="x",
                    labelbottom=True,
                    labeltop=False,
                    bottom=True,
                    top=False,
                    labelsize=11,
                )
                xlabel = "Age"

            # Set y-axis tick font size
            ax.tick_params(axis="y", labelsize=11)

            if i == n_edu - 1:
                ax.set_xlabel(xlabel, fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{edu_label}\nShare", fontsize=12)
                ax.legend(prop={"size": 11})

    plt.tight_layout()

    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300)


def _plot_choice_shares_by_education_age_bins(
    data_emp,
    data_sim,
    specs,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    bin_width: int = 5,
    path_to_save_plot: str | None = None,
):
    """
    Plot aggregated-choice shares by *age bins* and education.
    Each panel contains one choice; rows are education groups.

    New argument
    ------------
    bin_width : int, default 5
        Width of the age bins (in years).  Bin [a,a+bin_width) is labelled by *a*.
    """
    choice_groups_sim = {
        0: RETIREMENT,
        1: UNEMPLOYED,
        2: PART_TIME,
        3: FULL_TIME,
    }
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }

    sim = data_sim.loc[data_sim["health"] != DEAD].copy()
    emp = data_emp.copy()

    for g, raw in choice_groups_sim.items():
        sim.loc[sim["choice"].isin(np.atleast_1d(raw)), "choice_group"] = g
    for g, raw in choice_groups_emp.items():
        emp.loc[emp["choice"].isin(np.atleast_1d(raw)), "choice_group"] = g

    sim["choice_group"] = sim["choice_group"].astype(int)
    emp["choice_group"] = emp["choice_group"].astype(int)

    # ── 2. Build age bins & plotting grid ─────────────────────────────────────
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]

    edges = list(range(age_min, age_max + 1, bin_width))
    if edges[-1] <= age_max:
        edges.append(age_max + 1)
    bin_starts = edges[:-1]  # x-axis

    sex = SEX
    edu_labels = specs["education_labels"]
    choice_labels = specs["choice_labels"]
    n_edu, n_choices = len(edu_labels), len(choice_labels)

    fig, axs = plt.subplots(
        n_edu, n_choices, figsize=(4 * n_choices, 3 * n_edu), sharex=True, sharey=True
    )
    axs = np.atleast_2d(axs)

    # ── 3. Loop over education × choice bins ──────────────────────────────────
    for i, edu_label in enumerate(edu_labels):
        emp_sub = emp[(emp["sex"] == sex) & (emp["education"] == i)]
        sim_sub = sim[sim["education"] == i]

        for j in range(n_choices):
            ax = axs[i, j]
            emp_rates, sim_rates = [], []

            for start, end in zip(edges[:-1], edges[1:], strict=False):
                emp_bin = emp_sub[(emp_sub["age"] >= start) & (emp_sub["age"] < end)]
                sim_bin = sim_sub[(sim_sub["age"] >= start) & (sim_sub["age"] < end)]

                emp_rates.append(
                    (emp_bin["choice_group"] == j).mean() if len(emp_bin) else np.nan
                )
                sim_rates.append(
                    (sim_bin["choice_group"] == j).mean() if len(sim_bin) else np.nan
                )

            ax.plot(bin_starts, sim_rates, label="Simulated")
            ax.plot(bin_starts, emp_rates, ls="--", label="Observed")
            ax.set_title(choice_labels[j])
            ax.set_ylim(0, 1)
            ax.set_xticks(bin_starts)
            ax.tick_params(labelbottom=True)

            if i == n_edu - 1:
                ax.set_xlabel("Age Bin (start year)")
            if j == 0:
                ax.set_ylabel(f"{edu_label}\nShare")
                ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300)


def plot_choice_shares_overall_age_bins(
    data_emp,
    data_sim,
    specs,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    bin_width: int = 5,
    path_to_save_plot: str | None = None,
):
    """
    Same as `plot_choice_shares_overall`, but aggregates on *age bins*
    instead of single-year ages.

    Parameters
    ----------
    bin_width : int, default 5
        Width of the age bins (in years).  Bin [a,a+bin_width) is labelled by *a*.
    """
    choice_groups_sim = {
        0: RETIREMENT,
        1: UNEMPLOYED,
        2: PART_TIME,
        3: FULL_TIME,
    }
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }

    sim = data_sim.loc[data_sim["health"] != DEAD].copy()
    emp = data_emp.copy()

    for g, raw in choice_groups_sim.items():
        sim.loc[sim["choice"].isin(np.atleast_1d(raw)), "choice_group"] = g
    for g, raw in choice_groups_emp.items():
        emp.loc[emp["choice"].isin(np.atleast_1d(raw)), "choice_group"] = g

    sim["choice_group"] = sim["choice_group"].astype(int)
    emp["choice_group"] = emp["choice_group"].astype(int)

    # ── 2. Build age bins & plotting grid ─────────────────────────────────────
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]

    edges = list(range(age_min, age_max + 1, bin_width))
    if edges[-1] <= age_max:
        edges.append(age_max + 1)
    bin_starts = edges[:-1]

    sex = SEX
    n_choices = 4
    fig, axs = plt.subplots(1, n_choices, figsize=(4 * n_choices, 3), sharey=True)

    emp_sub = emp[emp["sex"] == sex]
    sim_sub = sim  # sex already restricted earlier

    # ── 3. Compute & plot shares for each choice group ────────────────────────
    for j in range(n_choices):
        ax = axs[j]
        emp_rates, sim_rates = [], []

        for start, end in zip(edges[:-1], edges[1:], strict=False):
            emp_bin = emp_sub[(emp_sub["age"] >= start) & (emp_sub["age"] < end)]
            sim_bin = sim_sub[(sim_sub["age"] >= start) & (sim_sub["age"] < end)]

            emp_rates.append(
                (emp_bin["choice_group"] == j).mean() if len(emp_bin) else np.nan
            )
            sim_rates.append(
                (sim_bin["choice_group"] == j).mean() if len(sim_bin) else np.nan
            )

        ax.plot(bin_starts, sim_rates, label="Simulated")
        ax.plot(bin_starts, emp_rates, ls="--", label="Observed")
        ax.set_title(specs["choice_labels"][j])
        ax.set_ylim(0, 1)
        ax.set_xticks(bin_starts)
        ax.tick_params(labelbottom=True)

        ax.set_xlabel("Age Bin (start year)")
        if j == 0:
            ax.set_ylabel("Share")
            ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300)


# =====================================================================================
# Caregiver shares and demand
# =====================================================================================


def plot_caregiver_shares_by_age(
    df_emp,
    df_sim,
    specs,
    choice_set,
    age_min=None,
    age_max=None,
    path_to_save_plot=None,
):
    """
    Plot the share of informal caregivers by single year of age.

    * Empirical share  = (# obs with any_care == 1) / (# obs with any_care in {0,1})
    * Simulated share  = (# obs with choice ∈ INFORMAL_CARE) / (total obs at that age)

    Parameters
    ----------
    df_emp, df_sim : pandas.DataFrame
        DataFrames that contain at least
            'age', 'choice', 'any_care', 'sex'  (sex optional but recommended).
    specs : dict
        Needs the usual keys:
            'start_age'   : int  (lower bound of plot, inclusive)
            'end_age_msm' : int  (upper bound of plot, inclusive)
    path_to_save_plot : str | pathlib.Path | None, optional
        If provided, the figure is saved there (PNG, 300 dpi).
    """

    # -------- 1. Basic setup ------------------------------------------------
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]

    ages = np.arange(age_min, age_max + 1)

    # Keep only alive individuals in the simulated data
    df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()

    sex = SEX  # scalar {0,1}; keep behaviour consistent with earlier functions
    if "sex" in df_emp:
        df_emp = df_emp.loc[df_emp["sex"] == sex].copy()
    if "sex" in df_sim:
        df_sim = df_sim.loc[df_sim["sex"] == sex].copy()

    # -------- 2. Compute empirical caregiver share -------------------------
    # pandas' mean() ignores NaNs → exactly the desired denominator logic
    emp_share = (
        df_emp.groupby("age", observed=False)["any_care"]
        .mean()  # share of ones among {0,1}
        .reindex(ages)  # ensure full age range
    )

    # -------- 3. Compute simulated caregiver share -------------------------
    care_codes = np.asarray(choice_set).tolist()
    sim_share = (
        df_sim.assign(is_care=df_sim["choice"].isin(care_codes))
        .groupby("age", observed=False)["is_care"]
        .mean()  # mean of Boolean = share
        .reindex(ages)
    )

    # -------- 4. Plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(ages, sim_share, label="Simulated")
    ax.plot(ages, emp_share, ls="--", label="Observed")

    ax.set_xlabel("Age")
    ax.set_ylabel("Share of informal caregivers")
    ax.set_xlim(age_min, age_max)
    ax.set_ylim(0, 0.15)
    ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_caregiver_shares_by_age_bins(
    # df_emp: pd.DataFrame,
    emp_moments: pd.DataFrame,
    df_sim: pd.DataFrame,
    specs: dict,
    choice_set,
    *,
    age_min: int | None = None,
    age_max: int | None = None,
    bin_width: int = 5,
    scale: float = 1.0,
    moment_prefix: str = "share_informal_care_age_bin_",
    path_to_save_plot: str | Path | None = None,
):
    """
    Plot the share of informal caregivers by *age bins*.

    Bin [a, a+bin_width) is labelled by its lower bound *a*.

    Parameters
    ----------
    df_emp, df_sim : pandas.DataFrame
        Must contain at least 'age', 'choice', 'any_care' and ideally 'sex'.
    specs : dict
        Needs 'start_age' and 'end_age_msm'.
    choice_set : iterable
        Codes in df_sim["choice"] that count as (informal) care for the
        simulated data.
    bin_width : int, default 5
        Width of the age bins.
    moment_prefix : str, default \"share_informal_care_age_bin_\"
        Prefix used for looking up empirical moments in ``emp_moments``.
        The full key is constructed as
        ``f\"{moment_prefix}{start}_{end-1}\"``.
    path_to_save_plot : str | pathlib.Path | None
        If given, the figure is stored as PNG (300 dpi).
    """

    # # ── 1. Basic setup ──────────────────────────────────────────────────
    # if age_min is None:
    #     age_min = specs["start_age"]
    # if age_max is None:
    #     age_max = specs["end_age_msm"]

    # # define bin edges  [a, b, c, …]  with last edge > age_max
    # edges = list(range(age_min, age_max + 1, bin_width))
    # if edges[-1] <= age_max:
    #     edges.append(age_max + 1)  # right-open interval
    # bin_starts = edges[:-1]  # x-axis ticks / labels

    # # restrict to alive individuals in the simulated data
    # df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()

    # # choose sex if the column is present (mimics earlier behaviour)
    # sex = SEX  # scalar {0,1}
    # if "sex" in df_emp:
    #     df_emp = df_emp.loc[df_emp["sex"] == sex].copy()
    # if "sex" in df_sim:
    #     df_sim = df_sim.loc[df_sim["sex"] == sex].copy()

    # # ── 2. Compute bin-level caregiver shares ───────────────────────────
    # care_codes = np.asarray(choice_set).tolist()

    # emp_rates, sim_rates = [], []
    # for start, end in zip(edges[:-1], edges[1:], strict=False):
    #     # --- empirical --------------------------------------------------
    #     emp_bin = df_emp[(df_emp["age"] >= start) & (df_emp["age"] < end)]
    #     # mean() on 0/1/NaN implements (#1) / (#0 + #1)
    #     emp_rates.append(emp_bin["any_care"].mean() if len(emp_bin) else np.nan)

    #     # --- simulated --------------------------------------------------
    #     sim_bin = df_sim[(df_sim["age"] >= start) & (df_sim["age"] < end)]
    #     if len(sim_bin):
    #         is_care = sim_bin["choice"].isin(care_codes)
    #         sim_rates.append(is_care.mean())
    #     else:
    #         sim_rates.append(np.nan)

    # # ── 3. Plot ─────────────────────────────────────────────────────────
    # # fig, ax = plt.subplots(figsize=(8, 4))

    # # ax.plot(bin_starts, sim_rates, label="Simulated")
    # # ax.plot(bin_starts, emp_rates, ls="--", label="Observed")

    # # ax.set_xlabel("Age bin (start year)")
    # # ax.set_ylabel("Share of informal caregivers")
    # # ax.set_xticks(bin_starts)
    # # ax.set_xlim(bin_starts[0], bin_starts[-1])
    # # ax.set_ylim(0, 0.15)
    # # ax.legend()

    # # plt.tight_layout()
    # # if path_to_save_plot:
    # #     plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    # # plt.close(fig)

    # fig, ax = plt.subplots(figsize=(8, 4))

    # bar_w = 0.4  # width of each bar
    # x_emp = np.asarray(bin_starts) - bar_w / 2  # left bar (empirical)
    # x_sim = np.asarray(bin_starts) + bar_w / 2  # right bar (simulated)

    # ax.bar(x_emp, emp_rates, width=bar_w, label="Observed")
    # ax.bar(x_sim, sim_rates, width=bar_w, label="Simulated")

    # ax.set_xlabel("Age bin (start year)")
    # ax.set_ylabel("Share of informal caregivers")
    # ax.set_xticks(bin_starts)
    # ax.set_xlim(bin_starts[0] - bar_w, bin_starts[-1] + bar_w)
    # ax.set_ylim(0, 0.15)
    # ax.legend()

    # plt.tight_layout()
    # if path_to_save_plot:
    #     plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    # plt.close(fig)

    # ── 1. Age grid & bin edges ─────────────────────────────────────────
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]

    edges = list(range(age_min, age_max + 1, bin_width))
    if edges[-1] <= age_max:
        edges.append(age_max + 1)  # right-open
    bin_starts = edges[:-1]

    # ── 2. Simulated data (unchanged) ───────────────────────────────────
    df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()
    if "sex" in df_sim:
        df_sim = df_sim.loc[df_sim["sex"] == SEX].copy()

    care_codes = np.asarray(choice_set).tolist()

    # ── 3. Empirical lookup from *Series* ───────────────────────────────
    emp_lookup = emp_moments.to_dict()  # key → value

    # ── 4. Build vectors of rates ───────────────────────────────────────
    emp_rates, sim_rates = [], []

    for start, end in zip(edges[:-1], edges[1:], strict=False):
        key = f"{moment_prefix}{start}_{end-1}"
        emp_rates.append(emp_lookup.get(key, np.nan) / scale)

        sim_bin = df_sim[(df_sim["age"] >= start) & (df_sim["age"] < end)]
        sim_rates.append(
            sim_bin["choice"].isin(care_codes).mean() if len(sim_bin) else np.nan
        )

    fig, ax = plt.subplots(figsize=(8, 4))

    # --- 5a.  Filter out the 75–79 bin -----------------------------------
    keep = [bs < 75 for bs in bin_starts]  # noqa: PLR2004
    bin_starts = np.asarray(bin_starts)[keep]
    emp_rates = np.asarray(emp_rates)[keep]
    sim_rates = np.asarray(sim_rates)[keep]

    # --- 5b.  Bar geometry & colours -------------------------------------
    bar_w = 1.5  # width of each bar
    gap = 0.10  # empty space BETWEEN the two bars
    offset = bar_w / 2 + gap / 2

    # x_emp = bin_starts - offset
    # x_sim = bin_starts + offset

    # ax.bar(x_emp, emp_rates, width=bar_w, color="tab:orange", label="Observed")
    # ax.bar(x_sim, sim_rates, width=bar_w, color="tab:blue", label="Simulated")

    # # --- 5c.  X-axis ticks & labels --------------------------------------
    # xticks = bin_starts
    # xticklabels = [
    #     f"{start}\u2013{start + bin_width - 1}" for start in bin_starts
    # ]  # u2013 = en-dash
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels)

    # # --- 5d.  Remaining cosmetics ----------------------------------------
    # ax.set_xlabel("Age group")
    # ax.set_ylabel("Share of informal caregivers")
    # ax.set_xlim(bin_starts[0] - bar_w - gap, bin_starts[-1] + bar_w + gap)
    # ax.set_ylim(0, 0.12)
    # ax.legend()

    # plt.tight_layout()
    # if path_to_save_plot:
    #     plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    # plt.close(fig)
    x_emp = bin_starts - offset

    x_sim = bin_starts + offset

    ax.bar(
        x_emp,
        emp_rates,
        width=bar_w,
        color=mcolors.to_rgba("orange", 1),
        label="Observed",
    )
    ax.bar(
        x_sim,
        sim_rates,
        width=bar_w,
        color=mcolors.to_rgba("blue", 1),
        label="Simulated",
    )  # lighter blue

    # --- 5c.  X-axis ticks & labels ---------------------------------------
    xticks = bin_starts
    xticklabels = [f"{start}\u2013{start + bin_width - 1}" for start in bin_starts]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # --- 5d.  Cosmetics & extra whitespace --------------------------------
    pad = 2  # one age-unit padding on each side
    ax.set_xlabel("Age Bin")
    ax.set_ylabel("Share")
    ax.set_xlim(bin_starts[0] - offset - pad, bin_starts[-1] + offset + pad)
    ax.set_ylim(0, 0.1)
    ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


# def plot_simulated_care_demand_by_age(
#     df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
# ):
#     """
#     Plot the yearly share of individuals with care_demand == 1 in the simulated data.

#     Parameters
#     ----------
#     df_sim : pandas.DataFrame
#         Simulated micro data containing at least the columns
#         'age', 'care_demand', and (optionally) 'sex'.
#     specs : dict
#         Should include:
#             'start_age'   : int - lower bound of age range (inclusive)
#             'end_age_msm' : int - upper bound of age range (inclusive)
#     path_to_save_plot : str | pathlib.Path | None, optional
#         If provided, the figure is written to this file (PNG, 300 dpi).
#     """

#     # ---- 1. Setup ---------------------------------------------------------
#     if age_min is None:
#         age_min = specs["start_age"]
#     if age_max is None:
#         age_max = 100

#     ages = np.arange(age_min, age_max + 1)

#     # Keep only alive individuals in the simulated data
#     df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()

#     # Keep only the model-relevant sex if that convention is used elsewhere
#     if "sex" in df_sim.columns:
#         df_sim = df_sim.loc[df_sim["sex"] == SEX].copy()

#     # # ---- 2. Compute share of care demand per age --------------------------
#     # share_care = (
#     #     df_sim.groupby("age")["care_demand"]
#     #     .mean()  # mean of {0,1} → share
#     #     .reindex(ages)  # keep full age range, NaN if missing
#     # )

#     # # ---- 3. Plot ----------------------------------------------------------
#     # fig, ax = plt.subplots(figsize=(8, 4))
#     # ax.plot(ages, share_care, label="Simulated", color="blue")

#     # ax.set_xlabel("Age")
#     # ax.set_ylabel("Share with care demand")
#     # ax.set_xlim(age_min, age_max)
#     # ax.set_ylim(0, 0.15)
#     # ax.set_title("Prevalence of Care Demand by Age (Simulated)")
#     # ax.legend()

#     # plt.tight_layout()
#     # if path_to_save_plot:
#     #     plt.savefig(path_to_save_plot, dpi=300, transparent=False)

#     # ---- 2. Compute share of care demand per age & education --------------
#     share_care_low = (
#         df_sim.loc[df_sim["education"] == 0]
#         .groupby("age")["care_demand"]
#         .mean()
#         .reindex(ages)  # full grid, NaN if missing
#     )

#     share_care_high = (
#         df_sim.loc[df_sim["education"] == 1]
#         .groupby("age")["care_demand"]
#         .mean()
#         .reindex(ages)
#     )

#     # ---- 3. Plot -----------------------------------------------------------
#     fig, ax = plt.subplots()
#     # fig, ax = plt.subplots(figsize=(8, 4))

#     ax.plot(ages, share_care_low, label="Low education", color="blue", lw=2)
#     ax.plot(ages, share_care_high, label="High education", color="orange", lw=2)

#     pad = 1
#     ax.set_xlabel("Age")
#     ax.set_ylabel("Share")
#     ax.set_xlim(age_min - pad, age_max + pad)
#     ax.set_ylim(0, 0.15)
#     # ax.set_title("Prevalence of Care Demand by Age (Simulated)")
#     ax.legend()

#     plt.tight_layout()
#     if path_to_save_plot:
#         plt.savefig(path_to_save_plot, dpi=300, transparent=False)
#     plt.close(fig)


def plot_simulated_care_demand_by_age(
    df_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot the yearly share with care_demand == 1, broken out by
    • education (0 = low, 1 = high) → colour
    • caregiving_type (0 / 1)       → dashed / solid line
    """

    # ---- 1. Setup
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = 100

    ages = np.arange(age_min, age_max + 1)

    df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()
    if "sex" in df_sim.columns:
        df_sim = df_sim.loc[df_sim["sex"] == SEX].copy()

    # ---- 2. Share by (age, education, caregiving_type)
    shares = (
        df_sim.groupby(["age", "education", "caregiving_type"], observed=False)[
            "care_demand"
        ]
        .mean()
        .reindex(ages, level="age")  # keep full age grid on level 0
    )

    # ---- 3. Plot four lines (2 edu × 2 caregiving_type)
    fig, ax = plt.subplots()

    for caregiving_type in (0, 1):
        linestyle = "--" if caregiving_type == 0 else "-"
        type_lbl = (
            "Other provides informal care"
            if caregiving_type == 0
            else "Agent provides informal care"
        )

        for edu in (0, 1):
            colour = JET_COLOR_MAP[edu]
            edu_lbl = "Low education" if edu == 0 else "High education"

            share_series = shares.xs(
                (edu, caregiving_type), level=("education", "caregiving_type")
            )

            label = f"{type_lbl}, {edu_lbl}"
            ax.plot(
                ages,
                share_series,
                label=label,
                color=colour,
                linestyle=linestyle,
                # linewidth=2,
            )

    # ---- 4. Cosmetics
    pad = 1
    ax.set_xlabel("Age")
    ax.set_ylabel("Share")
    ax.set_xlim(age_min - pad, age_max + pad)
    # ax.set_ylim(0, 0.17)
    ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_choice_shares_single(data_emp, data_sim, specs, path_to_save_plot):
    """Plot choice-specific shares by age and education."""

    # data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    # data_sim.loc[:, "age"] = data_sim["period"] + specs["start_age"]

    sex = SEX

    fig, axes = plt.subplots(2, specs["n_choices"])
    for edu_var, edu_label in enumerate(specs["education_labels"]):

        mask_sim = (data_sim["sex"] == sex) & (data_sim["education"] == edu_var)
        data_sim_restr = data_sim[mask_sim]
        mask_obs = (data_emp["sex"] == sex) & (data_emp["education"] == edu_var)
        data_decision_restr = data_emp[mask_obs]

        choice_shares_sim = (
            data_sim_restr.groupby(["age"], observed=False)["choice"]
            .value_counts(normalize=True)
            .unstack()
        )
        choice_shares_obs = (
            data_decision_restr.groupby(["age"], observed=False)["choice"]
            .value_counts(normalize=True)
            .unstack()
        )
        # if sex == 0:
        #     choice_range = all but part-time
        choice_range = range(len(specs["choice_labels"]))

        for choice in choice_range:
            ax = axes[edu_var, choice]
            choice_share_sim = choice_shares_sim[choice]
            choice_share_obs = choice_shares_obs[choice]
            ax.plot(choice_share_sim, label="Simulated")
            ax.plot(choice_share_obs, label="Observed", ls="--")
            choice_label = specs["choice_labels"][choice]
            ax.set_title(f"{edu_label}; Choice {choice_label}")
            ax.set_ylim([0, 1])
            ax.legend()

            # fig.savefig(
            #     f"{dir_to_save_plots}_edu_{edu_label}_choice_{choice}",
            #     transparent=True,
            #     dpi=300,
            # )

    fig.tight_layout()

    fig.savefig(
        path_to_save_plot,
        transparent=False,
        dpi=300,
    )


# =====================================================================================
# Transitions
# =====================================================================================


def _plot_transitions_by_age(
    data_emp,
    data_sim,
    specs,
    states,
    state_labels,
    age_min=None,
    age_max=None,
    one_way=False,
    path_to_save_plot=None,
):
    """
    Plot age-specific transition shares (“lagged_choice → choice”) by education:
    top row low education, bottom row high education.

    Transitions are only computed for ages in [start_age, end_age_msm].
    Axes, titles, and legends follow the new scheme.
    """
    # add age
    emp = data_emp.copy()
    sim = data_sim.loc[data_sim["health"] != DEAD].copy()
    # emp["age"] = emp["period"] + specs["start_age"]
    # sim["age"] = sim["period"] + specs["start_age"]

    sex = SEX
    edu_labels = specs["education_labels"]
    from_states = list(states.keys())
    to_states = list(states.keys())

    # choose transitions
    if one_way:
        transitions = [(s, s) for s in from_states]
    else:
        transitions = list(product(from_states, to_states))

    # age grid
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    all_ages = list(range(age_min, age_max + 1))

    fig, axes = plt.subplots(
        len(edu_labels),
        len(transitions),
        figsize=(4 * len(transitions), 4 * len(edu_labels)),
        sharey=True,
    )

    for i, edu_label in enumerate(edu_labels):
        # filter by sex, education, and age‐cap
        emp_sub = emp[
            (emp["sex"] == sex) & (emp["education"] == i) & (emp["age"] <= age_max)
        ]
        sim_sub = sim[
            (sim["sex"] == sex) & (sim["education"] == i) & (sim["age"] <= age_max)
        ]

        for j, (from_label, to_label) in enumerate(transitions):
            ax = axes[i, j] if len(edu_labels) > 1 else axes[j]

            # compute transition rates for each age in [start, end_age_msm]
            emp_rates = []
            sim_rates = []
            for age in all_ages:
                emp_from = emp_sub[
                    (emp_sub["age"] == age)
                    & (emp_sub["lagged_choice"].isin(np.atleast_1d(states[from_label])))
                ]
                sim_from = sim_sub[
                    (sim_sub["age"] == age)
                    & (sim_sub["lagged_choice"].isin(np.atleast_1d(states[from_label])))
                ]
                emp_rates.append(
                    emp_from["choice"].isin(np.atleast_1d(states[to_label])).mean()
                    if len(emp_from)
                    else np.nan
                )
                sim_rates.append(
                    sim_from["choice"].isin(np.atleast_1d(states[to_label])).mean()
                    if len(sim_from)
                    else np.nan
                )

            # filter out NaNs so each line only spans ages with data
            emp_ages = [
                a for a, r in zip(all_ages, emp_rates, strict=False) if not np.isnan(r)
            ]
            emp_vals = [r for r in emp_rates if not np.isnan(r)]
            sim_ages = [
                a for a, r in zip(all_ages, sim_rates, strict=False) if not np.isnan(r)
            ]
            sim_vals = [r for r in sim_rates if not np.isnan(r)]

            # Plot
            ax.plot(sim_ages, sim_vals, label="Simulated")
            ax.plot(emp_ages, emp_vals, ls="--", label="Observed")

            # Titles & labels
            ax.set_title(f"{state_labels[from_label]} → {state_labels[to_label]}")
            ax.tick_params(labelbottom=True)
            if emp_ages or sim_ages:
                xmin = min(emp_ages + sim_ages)
                xmax = max(emp_ages + sim_ages)
                ax.set_xlim(xmin, xmax)
            # ax.set_xlim(age_min, age_max)

            ax.set_xlabel("Age")
            ax.set_ylim([0, 1])

            # only on the first “from→to” cell in each row do we put y‐label + legend
            if (from_label == from_states[0]) and (to_label == to_states[0]):
                ax.set_ylabel(f"{edu_label}\nTransition Rate")
                ax.legend()

    fig.tight_layout()
    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300, transparent=False)


def _plot_transitions_by_age_bins(
    data_emp,
    data_sim,
    specs,
    states,
    state_labels=None,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    one_way: bool = False,
    bin_width: int = 5,
    path_to_save_plot: str = None,
):
    """
    Plot transition shares by age bins.

    For each education level (rows) and each transition (cols),
    compute the share of observations in each age bin (width=bin_width)
    that move from lagged_choice → choice, separately for observed vs simulated.

    If one_way=True, only plots self-transitions (s → s) for each state.

    Args:
        data_emp, data_sim: DataFrames with columns
            ['period','sex','education','lagged_choice','choice'].
        specs: dict with
            'start_age' (int), 'end_age_msm' (int),
            'education_labels' (list of str)
        states: dict of state_key → state_value(s)
        state_labels: optional dict of state_key → pretty name;
            if None, uses state_key.replace("_"," ").capitalize()
        one_way: bool, if True only plot (s→s)
        bin_width: int, width of each age bin
        path_to_save_plot: optional filename to save figure

    """
    # copy & compute age
    emp = data_emp.copy()
    sim = data_sim.loc[data_sim["health"] != DEAD].copy()

    emp["age"] = emp["period"] + specs["start_age"]
    sim["age"] = sim["period"] + specs["start_age"]

    if state_labels is None:
        state_labels = {s: s.replace("_", " ").capitalize() for s in states}

    sex = SEX
    edu_labels = specs["education_labels"]
    from_states = list(states.keys())

    # choose transitions
    if one_way:
        transitions = [(s, s) for s in from_states]
    else:
        to_states = from_states
        transitions = list(product(from_states, to_states))

    # build age bins
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]

    # edges: [start, start+bin_width, ..., last_edge]
    edges = list(range(age_min, age_max + 1, bin_width))
    if edges[-1] <= age_max:
        edges.append(age_max + 1)

    bin_starts = edges[:-1]  # x-axis values

    # set up plot grid
    n_edu = len(edu_labels)
    n_trans = len(transitions)
    fig, axs = plt.subplots(
        n_edu, n_trans, figsize=(4 * n_trans, 3 * n_edu), sharey=True, sharex=True
    )
    axs = np.atleast_2d(axs)

    for i, edu_label in enumerate(edu_labels):
        # filter by sex, education & max age
        emp_sub = emp[
            (emp["sex"] == sex) & (emp["education"] == i) & (emp["age"] <= age_max)
        ]
        sim_sub = sim[
            (sim["sex"] == sex) & (sim["education"] == i) & (sim["age"] <= age_max)
        ]

        for j, (from_label, to_label) in enumerate(transitions):
            ax = axs[i, j]

            emp_rates, sim_rates = [], []
            for start, end in zip(edges[:-1], edges[1:], strict=False):
                # define bin [start, end-1]
                emp_bin = emp_sub[
                    (emp_sub["age"] >= start)
                    & (emp_sub["age"] < end)
                    & (emp_sub["lagged_choice"].isin(np.atleast_1d(states[from_label])))
                ]
                sim_bin = sim_sub[
                    (sim_sub["age"] >= start)
                    & (sim_sub["age"] < end)
                    & (sim_sub["lagged_choice"].isin(np.atleast_1d(states[from_label])))
                ]
                # compute share transitioning into to_label
                emp_rates.append(
                    emp_bin["choice"].isin(np.atleast_1d(states[to_label])).mean()
                    if len(emp_bin)
                    else np.nan
                )
                sim_rates.append(
                    sim_bin["choice"].isin(np.atleast_1d(states[to_label])).mean()
                    if len(sim_bin)
                    else np.nan
                )

            # plot
            ax.plot(bin_starts, sim_rates, label="Simulated")
            ax.plot(bin_starts, emp_rates, ls="--", label="Observed")

            # Ensure ticks appear on  all rows, not just bottom
            ax.tick_params(labelbottom=True)

            # titles & labels
            ax.set_title(f"{state_labels[from_label]} → {state_labels[to_label]}")
            ax.set_xlabel("Age Bin")
            ax.set_ylim(0, 1)

            # Explicitly set every 5-year tick (e.g. 30, 35, 40…)
            ax.set_xticks(bin_starts)

            # only first column gets y-label + legend
            if j == 0:
                yl = f"{edu_label}\nTransition Rate" if edu_label else "Transition Rate"
                ax.set_ylabel(yl)
                ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300)


def plot_transitions_by_age(  # noqa: PLR0912, PLR0915
    data_emp,
    data_sim,
    specs,
    state_labels,
    states_emp,
    states_sim,
    *,
    age_min=None,
    age_max=None,
    one_way=False,
    path_to_save_plot=None,
):
    """
    Plot age-specific transition shares (“lagged_choice → choice”) by education.
    Supports different code mappings for empirical vs simulated data via
    `states_emp` and `states_sim`. If those are not provided, `states` is used
    for both (backward-compatible).
    """

    if set(states_emp.keys()) != set(states_sim.keys()):
        raise ValueError(
            "states_emp and states_sim must have the same keys (state labels)."
        )

    label_order = list(state_labels.keys())
    if set(label_order) != set(states_emp.keys()):
        label_order = list(states_emp.keys())

    emp = data_emp.copy()
    sim = data_sim.loc[data_sim["health"] != DEAD].copy()

    sex = SEX
    edu_labels = specs["education_labels"]
    from_states = label_order
    to_states = label_order

    transitions = (
        [(s, s) for s in from_states]
        if one_way
        else list(product(from_states, to_states))
    )

    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    all_ages = list(range(age_min, age_max + 1))

    fig, axes = plt.subplots(
        len(edu_labels),
        len(transitions),
        figsize=(4 * len(transitions), 4 * len(edu_labels)),
        sharey=True,
    )

    # --- NEW: global padding knobs ---
    y_pad = 0.03  # little whitespace below 0 and above 1
    x_pad_default = 0.5  # fallback if we can't infer a step from data

    if len(edu_labels) == len(transitions) == 1:
        axes = np.array([[axes]])
    elif len(edu_labels) == 1:
        axes = axes[np.newaxis, :]
    elif len(transitions) == 1:
        axes = axes[:, np.newaxis]

    for i, edu_label in enumerate(edu_labels):
        emp_sub = emp[
            (emp["sex"] == sex) & (emp["education"] == i) & (emp["age"] <= age_max)
        ]
        sim_sub = sim[
            (sim["sex"] == sex) & (sim["education"] == i) & (sim["age"] <= age_max)
        ]

        for j, (from_label, to_label) in enumerate(transitions):
            ax = axes[i, j]

            emp_rates, sim_rates = [], []
            for age in all_ages:
                emp_from = emp_sub[
                    (emp_sub["age"] == age)
                    & (
                        emp_sub["lagged_choice"].isin(
                            np.atleast_1d(states_emp[from_label])
                        )
                    )
                ]
                sim_from = sim_sub[
                    (sim_sub["age"] == age)
                    & (
                        sim_sub["lagged_choice"].isin(
                            np.atleast_1d(states_sim[from_label])
                        )
                    )
                ]
                emp_rates.append(
                    emp_from["choice"].isin(np.atleast_1d(states_emp[to_label])).mean()
                    if len(emp_from)
                    else np.nan
                )
                sim_rates.append(
                    sim_from["choice"].isin(np.atleast_1d(states_sim[to_label])).mean()
                    if len(sim_from)
                    else np.nan
                )

            emp_ages = [
                a for a, r in zip(all_ages, emp_rates, strict=False) if not np.isnan(r)
            ]
            emp_vals = [r for r in emp_rates if not np.isnan(r)]
            sim_ages = [
                a for a, r in zip(all_ages, sim_rates, strict=False) if not np.isnan(r)
            ]
            sim_vals = [r for r in sim_rates if not np.isnan(r)]

            ax.plot(sim_ages, sim_vals, label="Simulated")
            ax.plot(emp_ages, emp_vals, ls="--", label="Observed")

            ax.set_title(f"{state_labels[from_label]} → {state_labels[to_label]}")
            ax.tick_params(labelbottom=True)

            # --- NEW: left/right padding (x) ---
            if emp_ages or sim_ages:
                ages_all = sorted(set(emp_ages + sim_ages))
                if len(ages_all) > 1:
                    diffs = np.diff(ages_all)
                    step = float(np.median(diffs)) if len(diffs) > 0 else 1.0
                else:
                    step = 1.0
                x_pad = 0.35 * step
                xmin, xmax = min(ages_all), max(ages_all)
                ax.set_xlim(xmin - x_pad, xmax + x_pad)
            else:
                ax.set_xlim(age_min - x_pad_default, age_max + x_pad_default)

            ax.set_xlabel("Age")

            # --- NEW: bottom/top padding (y) ---
            ax.set_ylim(-y_pad, 1 + y_pad)

            if (from_label == from_states[0]) and (to_label == to_states[0]):
                ax.set_ylabel(f"{edu_label}\nTransition Rate")
                ax.legend()

    fig.tight_layout()
    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_transition_counts_by_age(  # noqa: PLR0912, PLR0915
    data_emp,
    data_sim,
    specs,
    state_labels,
    states_emp,
    states_sim,
    *,
    age_min=None,
    age_max=None,
    one_way=False,
    path_to_save_plot=None,
):
    """
    Plot age-specific transition counts (absolute numbers) by education.

    Layout: 4 rows x 2 columns
    - Rows alternate between empirical and simulated data
    - Columns represent different transitions
    - Each education level gets 2 rows (empirical, simulated)

    Similar to plot_transitions_by_age, but plots the absolute number of
    observations that transition from lagged_choice → choice at each age,
    rather than the transition rate.

    Supports different code mappings for empirical vs simulated data via
    `states_emp` and `states_sim`.
    """

    if set(states_emp.keys()) != set(states_sim.keys()):
        raise ValueError(
            "states_emp and states_sim must have the same keys (state labels)."
        )

    label_order = list(state_labels.keys())
    if set(label_order) != set(states_emp.keys()):
        label_order = list(states_emp.keys())

    emp = data_emp.copy()
    sim = data_sim.loc[data_sim["health"] != DEAD].copy()

    sex = SEX
    edu_labels = specs["education_labels"]
    from_states = label_order
    to_states = label_order

    transitions = (
        [(s, s) for s in from_states]
        if one_way
        else list(product(from_states, to_states))
    )

    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    all_ages = list(range(age_min, age_max + 1))

    # Layout: 4 rows (2 education levels × 2 data types) x 2 columns (transitions)
    n_rows = len(edu_labels) * 2  # Each education gets 2 rows (empirical, simulated)
    n_cols = len(transitions)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 4 * n_rows),  # Wider to accommodate rotated age labels
        sharey=False,  # Don't share y-axis since sample sizes differ
    )

    # Padding knobs
    y_pad_frac = 0.05  # 5% padding above and below
    x_pad_default = 0.5  # fallback if we can't infer a step from data

    if n_rows == n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Store all y-values to set consistent ylim per panel type
    all_emp_vals = []
    all_sim_vals = []

    for i, edu_label in enumerate(edu_labels):
        emp_sub = emp[
            (emp["sex"] == sex) & (emp["education"] == i) & (emp["age"] <= age_max)
        ]
        sim_sub = sim[
            (sim["sex"] == sex) & (sim["education"] == i) & (sim["age"] <= age_max)
        ]

        for j, (from_label, to_label) in enumerate(transitions):
            # Row for empirical data: i * 2
            ax_emp = axes[i * 2, j]
            # Row for simulated data: i * 2 + 1
            ax_sim = axes[i * 2 + 1, j]

            emp_counts, sim_counts = [], []
            for age in all_ages:
                emp_from = emp_sub[
                    (emp_sub["age"] == age)
                    & (
                        emp_sub["lagged_choice"].isin(
                            np.atleast_1d(states_emp[from_label])
                        )
                    )
                ]
                sim_from = sim_sub[
                    (sim_sub["age"] == age)
                    & (
                        sim_sub["lagged_choice"].isin(
                            np.atleast_1d(states_sim[from_label])
                        )
                    )
                ]
                # Count transitions (sum of boolean) instead of mean
                emp_counts.append(
                    emp_from["choice"].isin(np.atleast_1d(states_emp[to_label])).sum()
                    if len(emp_from)
                    else np.nan
                )
                sim_counts.append(
                    sim_from["choice"].isin(np.atleast_1d(states_sim[to_label])).sum()
                    if len(sim_from)
                    else np.nan
                )

            emp_ages = [
                a for a, c in zip(all_ages, emp_counts, strict=False) if not np.isnan(c)
            ]
            emp_vals = [c for c in emp_counts if not np.isnan(c)]
            sim_ages = [
                a for a, c in zip(all_ages, sim_counts, strict=False) if not np.isnan(c)
            ]
            sim_vals = [c for c in sim_counts if not np.isnan(c)]

            # Plot empirical data
            if emp_vals:
                ax_emp.plot(emp_ages, emp_vals, ls="--", color="blue", label="Observed")
                all_emp_vals.extend(emp_vals)

            # Plot simulated data
            if sim_vals:
                ax_sim.plot(sim_ages, sim_vals, color="orange", label="Simulated")
                all_sim_vals.extend(sim_vals)

            # Set titles
            title = f"{state_labels[from_label]} → {state_labels[to_label]}"
            ax_emp.set_title(f"{title} - Observed ({edu_label})")
            ax_sim.set_title(f"{title} - Simulated ({edu_label})")

            ax_emp.tick_params(labelbottom=True)
            ax_sim.tick_params(labelbottom=True)

            # Left/right padding (x) - same for both panels
            if emp_ages or sim_ages:
                ages_all = sorted(set(emp_ages + sim_ages))
                if len(ages_all) > 1:
                    diffs = np.diff(ages_all)
                    step = float(np.median(diffs)) if len(diffs) > 0 else 1.0
                else:
                    step = 1.0
                x_pad = 0.35 * step
                xmin, xmax = min(ages_all), max(ages_all)
                ax_emp.set_xlim(xmin - x_pad, xmax + x_pad)
                ax_sim.set_xlim(xmin - x_pad, xmax + x_pad)
            else:
                ax_emp.set_xlim(age_min - x_pad_default, age_max + x_pad_default)
                ax_sim.set_xlim(age_min - x_pad_default, age_max + x_pad_default)

            ax_emp.set_xlabel("Age")
            ax_sim.set_xlabel("Age")

            # Set x-axis ticks to all ages and rotate labels
            ax_emp.set_xticks(all_ages)
            ax_emp.set_xticklabels(all_ages, rotation=90)
            ax_sim.set_xticks(all_ages)
            ax_sim.set_xticklabels(all_ages, rotation=90)

            # Set ylim endogenously for each panel
            # Empirical panel
            if emp_vals:
                y_min_emp = min(emp_vals)
                y_max_emp = max(emp_vals)
                y_range_emp = (
                    y_max_emp - y_min_emp if y_max_emp > y_min_emp else y_max_emp
                )
                ax_emp.set_ylim(
                    max(0, y_min_emp - y_pad_frac * y_range_emp),
                    y_max_emp + y_pad_frac * y_range_emp,
                )
            else:
                ax_emp.set_ylim(0, 1)

            # Simulated panel
            if sim_vals:
                y_min_sim = min(sim_vals)
                y_max_sim = max(sim_vals)
                y_range_sim = (
                    y_max_sim - y_min_sim if y_max_sim > y_min_sim else y_max_sim
                )
                ax_sim.set_ylim(
                    max(0, y_min_sim - y_pad_frac * y_range_sim),
                    y_max_sim + y_pad_frac * y_range_sim,
                )
            else:
                ax_sim.set_ylim(0, 1)

            # Set ylabel and legend only for first column
            if j == 0:
                ax_emp.set_ylabel(f"{edu_label}\nNumber of Transitions")
                ax_sim.set_ylabel(f"{edu_label}\nNumber of Transitions")
                ax_emp.legend()
                ax_sim.legend()

    fig.tight_layout()
    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_transitions_by_age_bins(  # noqa: PLR0912, PLR0915
    data_emp,
    data_sim,
    specs,
    state_labels,
    states_emp,
    states_sim,
    *,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    one_way: bool = False,
    bin_width: int = 5,
    path_to_save_plot: str = None,
):
    """
    Plot transition shares by age bins.

    For each education level (rows) and each transition (cols),
    compute the share of observations in each age bin (width=bin_width)
    that move from lagged_choice → choice, separately for observed vs simulated.

    Supports different code mappings for empirical vs simulated data via
    `states_emp` and `states_sim`. If those are not provided, falls back to `states`.
    """

    if set(states_emp.keys()) != set(states_sim.keys()):
        raise ValueError(
            "states_emp and states_sim must have the same keys (state labels)."
        )

    # Preserve user label order if provided; otherwise stable order from mapping
    label_order = list(state_labels.keys())
    if set(label_order) != set(states_emp.keys()):
        label_order = list(states_emp.keys())

    # ── Data prep ───────────────────────────────────────────────────────────
    emp = data_emp.copy()
    sim = data_sim.loc[data_sim["health"] != DEAD].copy()

    # Age from period (inclusive at start_age)
    emp["age"] = emp["period"] + specs["start_age"]
    sim["age"] = sim["period"] + specs["start_age"]

    sex = SEX
    edu_labels = specs["education_labels"]
    from_states = label_order
    to_states = label_order

    # choose transitions
    transitions = (
        [(s, s) for s in from_states]
        if one_way
        else list(product(from_states, to_states))
    )

    # ── Build age bins & labels ─────────────────────────────────────────────
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]

    # max_ret_age = specs["max_ret_age"]

    # edges = list(range(age_min, age_max + 1, bin_width))
    # # Adjust the last bin to end at max_ret_age (exclusive, open interval)
    # if edges[-1] < max_ret_age:
    #     # If the last edge is before max_ret_age, add max_ret_age as the final edge
    #     edges.append(max_ret_age)
    # else:
    #     # If the last edge is at or beyond max_ret_age, replace it with max_ret_age
    #     edges[-1] = max_ret_age

    # bin_starts = edges[:-1]

    edges = list(range(age_min, age_max + 1, bin_width))
    if edges[-1] <= age_max:
        edges.append(age_max + 1)

    bin_starts = edges[:-1]

    bin_labels = [
        f"{start}-{end - 1}" for start, end in zip(edges[:-1], edges[1:], strict=False)
    ]

    # Drop the final single-year bin (e.g., 70–70) if it exists
    if len(edges) >= 2 and (edges[-1] - edges[-2] == 1):  # noqa: PLR2004
        # remove the last (1-year) edge so the last bin disappears
        edges = edges[:-1]
        # and remove the corresponding last start/label
        if bin_starts:
            bin_starts = bin_starts[:-1]
        if bin_labels:
            bin_labels = bin_labels[:-1]

    # ── Figure setup (sharey only so we can pad x per panel) ────────────────
    n_edu, n_trans = len(edu_labels), len(transitions)
    fig, axs = plt.subplots(
        n_edu, n_trans, figsize=(4 * n_trans, 3 * n_edu), sharey=True
    )
    if n_edu == n_trans == 1:
        axs = np.array([[axs]])
    elif n_edu == 1:
        axs = axs[np.newaxis, :]
    elif n_trans == 1:
        axs = axs[:, np.newaxis]

    # Padding knobs
    y_pad = 0.03
    if len(bin_starts) > 1:
        step = float(np.median(np.diff(bin_starts)))
    else:
        step = float(bin_width) if bin_width is not None else 1.0
    x_pad = 0.35 * step

    # ── Loop: education × transitions ───────────────────────────────────────
    for i, edu_label in enumerate(edu_labels):
        emp_sub = emp[
            (emp["sex"] == sex) & (emp["education"] == i) & (emp["age"] <= age_max)
        ]
        sim_sub = sim[
            (sim["sex"] == sex) & (sim["education"] == i) & (sim["age"] <= age_max)
        ]

        for j, (from_label, to_label) in enumerate(transitions):
            ax = axs[i, j]

            emp_rates, sim_rates = [], []
            for start, end in zip(edges[:-1], edges[1:], strict=False):
                # Define bin [start, end)
                emp_bin = emp_sub[
                    (emp_sub["age"] >= start)
                    & (emp_sub["age"] < end)
                    & (
                        emp_sub["lagged_choice"].isin(
                            np.atleast_1d(states_emp[from_label])
                        )
                    )
                ]
                sim_bin = sim_sub[
                    (sim_sub["age"] >= start)
                    & (sim_sub["age"] < end)
                    & (
                        sim_sub["lagged_choice"].isin(
                            np.atleast_1d(states_sim[from_label])
                        )
                    )
                ]
                emp_rates.append(
                    emp_bin["choice"].isin(np.atleast_1d(states_emp[to_label])).mean()
                    if len(emp_bin)
                    else np.nan
                )
                sim_rates.append(
                    sim_bin["choice"].isin(np.atleast_1d(states_sim[to_label])).mean()
                    if len(sim_bin)
                    else np.nan
                )

            # Plot simulated and observed
            ax.plot(bin_starts, sim_rates, label="Simulated")
            ax.plot(bin_starts, emp_rates, ls="--", label="Observed")

            # Titles & labels
            ax.set_title(f"{state_labels[from_label]} → {state_labels[to_label]}")
            ax.set_xlabel("Age (bin range)")

            # Range-style ticks (rotated) on every panel
            ax.set_xticks(bin_starts)
            ax.set_xticklabels(bin_labels)
            ax.tick_params(
                axis="x", labelbottom=True, labeltop=False, bottom=True, top=False
            )
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")
                lbl.set_rotation_mode("anchor")

            # Uniform axes + whitespace all around
            ax.set_ylim(-y_pad, 1 + y_pad)
            if len(bin_starts) > 1:
                ax.set_xlim(bin_starts[0] - x_pad, bin_starts[-1] + x_pad)
            else:
                ax.set_xlim(bin_starts[0] - x_pad, bin_starts[0] + x_pad)

            # Only first column gets y-label + legend
            if j == 0:
                ax.set_ylabel(f"{edu_label}\nTransition Rate")
                ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300)


def plot_choice_shares(data_emp, data_sim, specs):
    """Plot stacked choice shares."""

    data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_sim.loc[:, "age"] = data_sim["period"] + specs["start_age"]

    data_sim.groupby(["age"], observed=False)["choice"].value_counts(
        normalize=True
    ).unstack().plot(title="Simulated choice shares by age", kind="bar", stacked=True)

    data_emp.groupby(["age"], observed=False)["choice"].value_counts(
        normalize=True
    ).unstack().plot(title="Observed choice shares by age", kind="bar", stacked=True)


def plot_states(data_emp, data_sim, discrete_state_names, specs):
    """Plot discrete states by age."""

    data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_sim.loc[:, "age"] = data_sim["period"] + specs["start_age"]

    for state_name in discrete_state_names:
        data_emp.groupby(["age"], observed=False)[state_name].value_counts(
            normalize=True
        ).unstack().plot()
        data_sim.groupby(["age"], observed=False)[state_name].value_counts(
            normalize=True
        ).unstack().plot()
        plt.show()


def plot_average_savings_decision(data_sim, path_to_save_plot, end_age=100):
    """Plot average savings decision."""
    # import matplotlib.pyplot as plt
    #
    # # %%
    # # Plot choice shares by age
    # df.groupby(["age"]).choice.value_counts(normalize=True).unstack().plot(
    #     title="Choice shares by age"
    # )
    #
    # # %%fig_1 = (
    #
    # # plot average income by age and choice
    # df.groupby(["age", "choice"])["labor_income"].mean().unstack().plot(
    #     title="Average income by age and choice"
    # )
    # # plot average income by age and choice
    # df.groupby(["age", "choice"])["total_income"].mean().unstack().plot(
    #     title="Average total income by age and choice"
    # )
    # %%
    # plot average consumption by age and choice
    # df.groupby(["age", "choice"])["consumption"].mean().unstack().plot(
    #     title="Average consumption by age and choice"
    # )
    # %%

    fig, ax = plt.subplots()
    # plot average periodic savings by age and choice
    data_sim_filtered = data_sim[data_sim["age"] < end_age].copy()
    data_sim_filtered.groupby("age", observed=False)["savings_dec"].mean().plot(
        ax=ax, label="Average savings"
    )
    ax.set_title("Average savings by age")
    ax.legend()
    fig.savefig(path_to_save_plot, transparent=True, dpi=300)


# =====================================================================================
# Debugging
# =====================================================================================


def plot_job_offer_share_by_age(df, min_age=30, max_age=75, path_to_save_plot=None):
    """Plot the share of positive job offers by age within a specified age range.

    Parameters:
    - df: pandas.DataFrame containing at least 'age' and 'job_offer' columns
    - min_age: int, minimum age to include (inclusive)
    - max_age: int, maximum age to include (inclusive)

    Usage:
    plot_job_offer_share_by_age(df_sim)
    """
    # Filter the DataFrame for the desired age range
    df_age = df[(df["age"] >= min_age) & (df["age"] <= max_age)]

    # Calculate the share of positive job offers by age
    share_by_age = df_age.groupby("age", observed=False)["job_offer"].mean()

    # Debugging
    # df_age_working = df_age[df_age["choice"] >= 2]
    # df_age_ret = df_age[df_age["choice"] == 0]
    # df_age_unemp = df_age[df_age["choice"] == 1]

    # w = df_age_working.groupby("age")["job_offer"].mean()
    # r = df_age_ret.groupby("age")["job_offer"].mean()
    # u = df_age_unemp.groupby("age")["job_offer"].mean()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(share_by_age.index, share_by_age.values, marker="o")
    plt.xlabel("Age")
    plt.ylabel("Share of Positive Job Offers")
    plt.title(f"Share of Positive Job Offers by Age ({min_age}-{max_age})")
    plt.xticks(range(min_age, max_age + 1, 5))
    plt.grid(True)
    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, transparent=False, dpi=300)
