"""Functions to plot model fit between empirical and simulated data."""

from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.model.shared import ALL, SEX


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

    # Copy and compute age columns
    # data_emp = data_emp.copy()
    # data_sim = data_sim.copy()
    # data_emp["age"] = data_emp["period"] + specs["start_age"]
    # data_sim["age"] = data_sim["period"] + specs["start_age"]

    # Define age bounds
    age_min = specs["start_age"]
    age_max = specs["end_age_msm"]

    sex = SEX

    # Prepare figure grid
    n_edu = len(specs["education_labels"])
    n_choices = specs["n_choices"]
    fig, axs = plt.subplots(n_edu, n_choices, figsize=(16, 6), sharex=True, sharey=True)

    # Loop over education groups and choices
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        # Filter by education
        emp_edu = data_emp[
            (data_emp["sex"] == sex) & (data_emp["education"] == edu_var)
        ]
        sim_edu = data_sim[(data_sim["education"] == edu_var)]

        # Compute choice‐by‐age shares
        sim_shares = (
            sim_edu.groupby("age")["choice"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        emp_shares = (
            emp_edu.groupby("age")["choice"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

        # Loop through each choice
        for choice_var in range(n_choices):
            ax = axs[edu_var, choice_var]

            # Select only ages within bounds
            ages = range(age_min, age_max + 1)
            vals_sim = sim_shares.reindex(ages, fill_value=0)[choice_var]
            vals_emp = emp_shares.reindex(ages, fill_value=0)[choice_var]

            # Plot
            ax.plot(ages, vals_sim, label="Simulated")
            ax.plot(ages, vals_emp, ls="--", label="Observed")

            # Styling
            ax.set_title(specs["choice_labels"][choice_var])
            ax.set_ylim(0, 1)
            ax.set_xlim(age_min, age_max)
            if edu_var == n_edu - 1:
                ax.set_xlabel("Age")
            else:
                ax.set_xlabel("")
            if choice_var == 0:
                ax.set_ylabel(f"{edu_label}\nShare")
                ax.legend()
            else:
                ax.set_ylabel("")

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
        choice_range = range(len(ALL))

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
