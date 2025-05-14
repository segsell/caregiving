"""Functions to plot model fit between empirical and simulated data."""

from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.model.shared import (
    DEAD,
    FILL_VALUE,
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
from caregiving.utils import table


def plot_average_wealth(
    data_emp,
    data_sim,
    specs,
    path_to_save_plot,
):
    """Plot average wealth by age and education."""

    data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_sim.loc[:, "age"] = data_sim["period"] + specs["start_age"]

    sex = SEX
    for edu in range(2):
        mask_sim = (data_sim["sex"] == sex) & (data_sim["education"] == edu)
        data_sim_edu = data_sim[mask_sim]
        mask_obs = (data_emp["sex"] == sex) & (data_emp["education"] == edu)
        data_decision_edu = data_emp[mask_obs]

        ages = np.arange(specs["start_age"] + 1, 90)
        average_wealth_sim = (
            data_sim_edu.groupby("age")["wealth_at_beginning"].median().loc[ages]
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


def plot_choice_shares_by_education(
    data_emp, data_sim, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """Plot choice-specific shares by age and education, only over the
    age range [start_age, end_age_msm]."""

    # ---------- 1. Map raw codes → 4-way choice ----------------------------
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

    n_edu = len(specs["education_labels"])
    n_choices = len(specs["choice_labels"])  # after aggregation
    fig, axs = plt.subplots(n_edu, n_choices, figsize=(16, 6), sharex=True, sharey=True)

    # ---------- 3. Loop over education groups ------------------------------
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        emp_edu = data_emp[
            (data_emp["sex"] == sex) & (data_emp["education"] == edu_var)
        ]
        sim_edu = data_sim[data_sim["education"] == edu_var]

        # shares by age × aggregated choice
        sim_shares = (
            sim_edu.groupby("age")["choice_group"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        emp_shares = (
            emp_edu.groupby("age")["choice_group"]
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

            ax.set_title(specs["choice_labels"][choice_var])
            ax.set_ylim(0, 1)
            ax.set_xlim(age_min, age_max)

            # if edu_var == n_edu - 1:
            ax.set_xlabel("Age")
            ax.tick_params(labelbottom=True)
            if choice_var == 0:
                ax.set_ylabel(f"{edu_label}\nShare")
                ax.legend()

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
        sim_all.groupby("age")["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    emp_shares = (
        emp_all.groupby("age")["choice_group"]
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


def plot_choice_shares_by_education_age_bins(
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
    choice_groups_sim = {0: RETIREMENT, 1: UNEMPLOYED, 2: PART_TIME, 3: FULL_TIME}
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
    plt.close(fig)


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
    choice_groups_sim = {0: RETIREMENT, 1: UNEMPLOYED, 2: PART_TIME, 3: FULL_TIME}
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
    plt.close(fig)


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
        df_emp.groupby("age")["any_care"]
        .mean()  # share of ones among {0,1}
        .reindex(ages)  # ensure full age range
    )

    # -------- 3. Compute simulated caregiver share -------------------------
    care_codes = np.asarray(choice_set).tolist()
    sim_share = (
        df_sim.assign(is_care=df_sim["choice"].isin(care_codes))
        .groupby("age")["is_care"]
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
        Codes in df_sim["choice"] that count as informal care.
    bin_width : int, default 5
        Width of the age bins.
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
        key = f"share_informal_care_age_bin_{start}_{end}"
        emp_rates.append(emp_lookup.get(key, np.nan))

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
    ax.set_ylabel("Share (percent)")
    ax.set_xlim(bin_starts[0] - offset - pad, bin_starts[-1] + offset + pad)
    ax.set_ylim(0, 0.1)
    ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


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
    • has_sister (0 / 1)            → dashed / solid line
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

    # ---- 2. Share by (age, education, has_sister)
    shares = (
        df_sim.groupby(["age", "education", "has_sister"])["care_demand"]
        .mean()
        .reindex(ages, level="age")  # keep full age grid on level 0
    )

    # ---- 3. Plot four lines (2 edu × 2 sister)
    fig, ax = plt.subplots()

    for has_sister in (0, 1):
        linestyle = "--" if has_sister == 0 else "-"
        sister_lbl = "No sister" if has_sister == 0 else "Has sister"

        for edu in (0, 1):
            colour = JET_COLOR_MAP[edu]
            edu_lbl = "Low education" if edu == 0 else "High education"

            share_series = shares.xs(
                (edu, has_sister), level=("education", "has_sister")
            )

            label = f"{sister_lbl}, {edu_lbl}"
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
    ax.set_ylim(0, 0.17)
    ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)
    plt.close(fig)


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
            data_sim_restr.groupby(["age"])["choice"]
            .value_counts(normalize=True)
            .unstack()
        )
        choice_shares_obs = (
            data_decision_restr.groupby(["age"])["choice"]
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


def plot_transitions_by_age(
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
    plt.close(fig)


def plot_transitions_by_age_bins(
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
    plt.close(fig)


def plot_choice_shares(data_emp, data_sim, specs):
    """Plot stacked choice shares."""

    data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_sim.loc[:, "age"] = data_sim["period"] + specs["start_age"]

    data_sim.groupby(["age"])["choice"].value_counts(normalize=True).unstack().plot(
        title="Simulated choice shares by age", kind="bar", stacked=True
    )

    data_emp.groupby(["age"])["choice"].value_counts(normalize=True).unstack().plot(
        title="Observed choice shares by age", kind="bar", stacked=True
    )


def plot_states(data_emp, data_sim, discrete_state_names, specs):
    """Plot discrete states by age."""

    data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_sim.loc[:, "age"] = data_sim["period"] + specs["start_age"]

    for state_name in discrete_state_names:
        data_emp.groupby(["age"])[state_name].value_counts(
            normalize=True
        ).unstack().plot()
        data_sim.groupby(["age"])[state_name].value_counts(
            normalize=True
        ).unstack().plot()
        plt.show()


def plot_average_savings_decision(data_sim, path_to_save_plot):
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
    data_sim.groupby("age")["savings_dec"].mean().plot(ax=ax, label="Average savings")
    ax.set_title("Average savings by age")
    ax.legend()
    fig.savefig(path_to_save_plot, transparent=True, dpi=300)


# =====================================================================================
# Debugging
# =====================================================================================


def plot_job_offer_share_by_age(df, min_age=30, max_age=75, path_to_save_plot=None):
    """
    Plots the share of positive job offers (job_offer == 1) by age within a specified range.

    Parameters:
    - df: pandas.DataFrame containing at least 'age' and 'job_offer' columns
    - min_age: int, minimum age to include (inclusive)
    - max_age: int, maximum age to include (inclusive)

    Usage:
    >>> plot_job_offer_share_by_age(df_sim)
    """
    # Filter the DataFrame for the desired age range
    df_age = df[(df["age"] >= min_age) & (df["age"] <= max_age)]

    # Calculate the share of positive job offers by age
    share_by_age = df_age.groupby("age")["job_offer"].mean()

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
    plt.close()
