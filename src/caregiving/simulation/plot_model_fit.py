"""Functions to plot model fit between empirical and simulated data."""

from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.model.shared import (
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


def plot_choice_shares_by_education(data_emp, data_sim, specs, path_to_save_plot=None):
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

    data_sim = data_sim.copy()
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
    age_min = specs["start_age"]
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

            if edu_var == n_edu - 1:
                ax.set_xlabel("Age")
            if choice_var == 0:
                ax.set_ylabel(f"{edu_label}\nShare")
                ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_choice_shares_overall(data_emp, data_sim, specs, path_to_save_plot=None):
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

    data_sim = data_sim.copy()
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
    age_min, age_max = specs["start_age"], specs["end_age_msm"]
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


def plot_caregiver_shares_by_age(
    df_emp, df_sim, specs, choice_set, path_to_save_plot=None
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
    age_min, age_max = specs["start_age"], specs["end_age_msm"]
    ages = np.arange(age_min, age_max + 1)

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
    ax.set_ylim(0, 0.2)
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


def plot_transitions_by_age(
    data_emp,
    data_sim,
    specs,
    states,
    state_labels,
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
    sim = data_sim.copy()
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
    age_min = specs["start_age"]
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
    sim = data_sim.copy()
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
    age_min = specs["start_age"]
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
