"""Functions to plot model fit between empirical and simulated data."""

from pathlib import Path

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
