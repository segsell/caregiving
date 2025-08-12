"""Family transition plots."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.model.shared import MINIMUM_CHILDBEARING_AGE


def task_plot_number_of_children(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "children.png",
    male: bool = False,
):
    """Plot the number of children by age.

    Calculate the number of children in the household for each individual conditional
    on sex, education and age bin.

    """

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])

    start_age = specs["start_age"]
    end_age = specs["end_age"]
    df = df[df["age"] <= end_age]

    df["has_partner"] = (df["partner_state"] > 0).astype(int)

    # calculate average hours worked by partner by age, sex and education
    cov_list = ["sex", "education", "has_partner", "age"]
    nb_children_data = df.groupby(cov_list)["children"].mean()

    nb_children_est = specs["children_by_state"]
    ages = np.arange(start_age, end_age + 1)

    # fig, axs = plt.subplots(ncols=4, figsize=(12, 8))
    # i = 0
    # -----------------------------------------------------------------
    partner_labels = ["Single", "Partnered"]
    sexes_to_plot = [1] if not male else [0, 1]
    sex_labels = ["Men", "Women"]

    ncols = len(sexes_to_plot) * len(partner_labels)
    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 6), squeeze=False)
    axs = axs[0]  # flatten

    plot_idx = 0

    # sex_labels = ["Men", "Women"]
    # partner_labels = ["Single", "Partnered"]
    # for sex, sex_label in enumerate(sex_labels):
    #     for has_partner, partner_label in enumerate(partner_labels):
    #         ax = axs[i]
    #         i += 1
    for sex in sexes_to_plot:
        for has_partner, partner_label in enumerate(partner_labels):
            ax = axs[plot_idx]
            plot_idx += 1
            for edu, edu_label in enumerate(specs["education_labels"]):
                nb_children_data_edu = nb_children_data.loc[
                    (sex, edu, has_partner, slice(None))
                ]
                nb_children_container = pd.Series(data=0, index=ages, dtype=float)
                nb_children_container.update(nb_children_data_edu)

                nb_children_est_edu = nb_children_est[sex, edu, has_partner, :]
                ax.plot(
                    ages,
                    nb_children_container,
                    color=JET_COLOR_MAP[edu],
                    linestyle="--",
                    label=f"Obs. {edu_label}",
                )
                ax.plot(
                    ages,
                    nb_children_est_edu,
                    color=JET_COLOR_MAP[edu],
                    label=f"Est. {edu_label}",
                )

            ax.set_ylim([0, 2.5])
            # if male:
            #     ax.set_title(f"{sex}, {partner_label}")
            # else:
            #     ax.set_title(f"{partner_label}")
            title = partner_label if not male else f"{sex_labels[sex]}, {partner_label}"
            ax.set_title(title)

    axs[0].legend()

    for ax in axs:
        ax.set_xlim([start_age, end_age])
        ax.set_xticks(np.arange(start_age, end_age + 1, 10))

    fig.tight_layout()
    fig.savefig(path_to_save, dpi=300)
    plt.close(fig)


# def task_plot_age_youngest_child(
#     path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
#     path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
#     path_to_save: Annotated[Path, Product] = BLD
#     / "plots"
#     / "stochastic_processes"
#     / "age_youngest_child.png",
#     male: bool = False,
# ):
#     """Plot age of the youngest child.

#     Split by sex x education x presence of partner.

#     """

#     # --- Load specs + data
#     with path_to_full_specs.open("rb") as file:
#         specs = pkl.load(file)

#     df = pd.read_csv(path_to_data, index_col=["pid", "syear"])

#     start_age = specs["start_age"]
#     # end_age = specs["end_age"]
#     end_age = 50
#     ages = np.arange(start_age, end_age + 1)

#     df = df[df["age"] <= end_age].copy()
#     df["has_partner"] = (df["partner_state"] > 0).astype(int)

#     # Filter data. Keep only observations with children
#     df = df.loc[df["kidage_youngest"] >= 0].copy()

#     # Drop observations with a positive age of the youngest child
#     # but no children in the household
#     df = df[~((df["kidage_youngest"] >= 0) & (df["children"] == 0))]
#     df = df.loc[
#         (df["kidage_youngest"] <= df["age"] - MINIMUM_CHILDBEARING_AGE)
#         # & (df["kidage_youngest"] + df["age"] >= MAXIMUM_CHILDBEARING_AGE)
#     ]

#     # --- Observed means by (sex, education, has_partner, age)
#     cov_list = ["sex", "education", "has_partner", "age"]
#     kidage_data = df.groupby(cov_list)["kidage_youngest"].mean()

#     kidage_est = np.asarray(specs["child_age_youngest_by_state"])

#     partner_labels = ["Single", "Partnered"]
#     sex_labels = ["Men", "Women"]
#     sexes_to_plot = [1] if not male else [0, 1]  # default: only Women unless male=True

#     edu_labels = specs["education_labels"]

#     ncols = len(sexes_to_plot) * len(partner_labels)
#     fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 6), squeeze=False)
#     axs = axs[0]

#     plot_idx = 0
#     for sex in sexes_to_plot:
#         for has_partner, partner_label in enumerate(partner_labels):
#             ax = axs[plot_idx]
#             plot_idx += 1

#             # track ymax to set axis range per panel
#             panel_ymax = 0.0

#             for edu, edu_label in enumerate(edu_labels):
#                 # Observed series on the full age grid (leave gaps as NaN)
#                 obs_sel = (
#                     kidage_data.loc[(sex, edu, has_partner, slice(None))]
#                     if (sex, edu, has_partner)
#                     in kidage_data.index.droplevel("age").unique()
#                     else pd.Series(dtype=float)
#                 )

#                 obs_series = pd.Series(index=ages, dtype=float)  # NaN baseline
#                 if len(obs_sel) > 0:
#                     obs_series.update(obs_sel)

#                 # Estimated series (convert jnp -> np if needed)
#                 est_series = np.asarray(
#                     kidage_est[sex, edu, has_partner, :], dtype=float
#                 )
#                 breakpoint()

#                 ax.plot(
#                     ages,
#                     obs_series.values,
#                     linestyle="--",
#                     label=f"Obs. {edu_label}",
#                     color=JET_COLOR_MAP[edu],
#                 )
#                 ax.plot(
#                     ages,
#                     est_series,
#                     label=f"Est. {edu_label}",
#                     color=JET_COLOR_MAP[edu],
#                 )

#                 # Update ymax, ignoring NaNs
#                 panel_ymax = np.nanmax(
#                     [panel_ymax, np.nanmax(obs_series.values), np.nanmax(est_series)]
#                 )

#             # Axis cosmetics
#             title = partner_label if not male else f"{sex_labels[sex]}, {partner_label}"
#             ax.set_title(title)
#             ax.set_xlim([start_age, end_age])
#             ax.set_xticks(np.arange(start_age, end_age + 1, 10))

#             # y-limits: start at 0, go to next integer above max
#             if np.isfinite(panel_ymax):
#                 y_top = int(np.ceil(panel_ymax))
#                 y_top = max(y_top, 1)  # at least 1
#             else:
#                 y_top = 1
#             ax.set_ylim([0, y_top])

#             # Integer y-ticks
#             ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

#             ax.set_xlabel("Age of agent")
#             ax.set_ylabel("Age of youngest child")

#     # Legend on first axis
#     axs[0].legend(ncol=1, frameon=True)

#     fig.tight_layout()
#     fig.savefig(path_to_save, dpi=300, bbox_inches="tight")
#     breakpoint()
#     plt.close(fig)


def task_plot_partner_transitions(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "partner.png",
    male: bool = False,
):
    """Illustrate the partnership rates by age.

    Age-specific probabilities of
    - being single
    - having working age partner
    - having retired partner

    """

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])

    start_age = specs["start_age"]
    end_age = specs["end_age"]
    partner_states = specs["partner_labels"]
    n_partner_states = specs["n_partner_states"]

    grouped_shares = df.groupby(["sex", "education", "age"])[
        "partner_state"
    ].value_counts(normalize=True)
    partner_shares_obs = grouped_shares.loc[
        (slice(None), slice(None), slice(None), slice(None))
    ]

    ages = np.arange(start_age, end_age + 1 - 10)
    initial_dist = np.zeros(n_partner_states)

    # fig, axs = plt.subplots(nrows=2, ncols=specs["n_partner_states"], figsize=(12, 8))
    # for partner_state, partner_label in enumerate(specs["partner_labels"]):
    #     for sex_var, sex_label in enumerate(specs["sex_labels"]):
    #         ax = axs[sex_var, partner_state]
    #         for edu, edu_label in enumerate(specs["education_labels"]):
    #             edu_shares_obs = partner_shares_obs.loc[
    #                 (sex_var, edu, slice(None), partner_state)
    #             ]
    # -----------------------------------------------------------------
    sexes_to_plot = [1] if not male else [0, 1]
    nrows = len(sexes_to_plot)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=n_partner_states,
        figsize=(4 * n_partner_states, 3.5 * nrows),
        squeeze=False,
    )

    for row_idx, sex_var in enumerate(sexes_to_plot):
        sex_label = specs["sex_labels"][sex_var]

        for partner_state in range(n_partner_states):
            ax = axs[row_idx, partner_state]

            for edu, edu_label in enumerate(specs["education_labels"]):
                # ------- observed ----------------------------------------------------
                edu_shares_obs = partner_shares_obs.loc[
                    (sex_var, edu, slice(None), partner_state)
                ]  # index collapses to 'age'
                share_data_container = pd.Series(data=np.nan, index=ages, dtype=float)
                share_data_container.update(edu_shares_obs)

                # Assign only single and married shares at start
                initial_dist[0] = partner_shares_obs.loc[(sex_var, edu, 30, 0)]
                initial_dist[1] = 1 - initial_dist[0]
                shares_over_time = _markov_simulator(
                    initial_dist,
                    specs["partner_trans_mat"][sex_var, edu, :, :, :],
                    n_periods=len(ages),
                )
                relev_share = shares_over_time[:, partner_state]

                # Use fifty percent as default if not available in the data
                # (just for plotting).
                share_data_container = pd.Series(data=0.0, index=ages, dtype=float)
                share_data_container.update(edu_shares_obs)

                ax.plot(
                    ages,
                    relev_share,
                    color=JET_COLOR_MAP[edu],
                    label=f"Est. {edu_label}",
                )
                ax.plot(
                    ages,
                    share_data_container,
                    color=JET_COLOR_MAP[edu],
                    linestyle="--",
                    label=f"Obs. {edu_label}",
                )
                ax.set_ylim([0, 1])
                ax.set_xlabel("Age")
                ax.set_ylabel("Number")

            if male:
                ax.set_title(f"{sex_label}, {partner_states[partner_state]}")
            else:
                ax.set_title(str(partner_states[partner_state]))

    axs[0, 0].legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(path_to_save, dpi=300)
    plt.close(fig)


def _markov_simulator(initial_dist, trans_probs, n_periods=None):
    """Simulate a Markov process."""
    if n_periods is None:
        n_periods = trans_probs.shape[0]
    else:
        # Check if n_periods is integer
        if not isinstance(n_periods, int):
            raise ValueError("n_periods must be an integer.")

    n_states = initial_dist.shape[0]
    final_dist = np.zeros((n_periods, n_states))
    final_dist[0, :] = initial_dist

    for t in range(n_periods - 1):
        current_dist = final_dist[t, :]
        for state in range(n_states - 1):
            final_dist[t + 1, state] = current_dist @ trans_probs[t, :, state]

        final_dist[t + 1, -1] = 1 - final_dist[t + 1, :-1].sum()

    return final_dist
