"""Expected health plots."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP


@pytask.mark.health_transition
def task_plot_healthy_unhealthy(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "health_transition_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "health_transition.png",
    male: bool = False,
):
    """Illustrate the health rates by age.

    (actual vs. estimated by markov chain)

    Conditional on being alive, what is the probability of being healthy?

    """

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    # Load the health transition sample
    df = pd.read_pickle(path_to_data)

    # Define age range
    start_age = specs["start_age"]
    end_age = specs["end_age"]

    max_age_est_physical = 90
    max_period_physical = max_age_est_physical - start_age
    est_ages = np.arange(start_age, end_age + 1)

    # Calculate the smoothed shares for healthy individuals
    edu_shares_data = (
        df.groupby(["sex", "education", "age"])["health"]
        .mean()
        .loc[slice(None), slice(None), slice(start_age, end_age + 1)]
    )
    full_index = pd.MultiIndex.from_product(
        [range(specs["n_sexes"]), range(specs["n_education_types"]), est_ages],
        names=["sex", "education", "age"],
    )
    edu_shares_healthy = pd.DataFrame(index=full_index, columns=["health"], data=0.0)
    edu_shares_healthy.update(edu_shares_data)

    # Which sex(es) to plot?  0 = men, 1 = women
    sexes_to_plot = [1] if not male else [0, 1]

    # Create the figure
    fig, axs = plt.subplots(
        ncols=len(sexes_to_plot),
        # figsize=(6 * len(sexes_to_plot), 8),
        squeeze=False,
    )
    axs = axs[0]  # flatten first (only one row)

    # Initialize the distribution
    initial_dist = np.zeros(specs["n_health_states"])

    # Create the plot
    # fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
    # for sex_var, sex_label in enumerate(specs["sex_labels"]):
    #     ax = axs[sex_var]
    for col_idx, sex_var in enumerate(sexes_to_plot):
        ax = axs[col_idx]
        sex_label = specs["sex_labels"][sex_var]
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            # Set the initial distribution for the Markov simulation
            # and assume nobody is dead
            initial_dist[1] = edu_shares_healthy.loc[(sex_var, edu_var, start_age)]
            initial_dist[0] = 1 - initial_dist[1]

            # Simulate the Markov process and get health probabilities
            shares_over_time = _markov_simulator(
                initial_dist, specs["health_trans_mat"][sex_var, edu_var, :, :, :]
            )
            # Construct health probabilities if alive (Use death probs as column vector)
            alive_share = 1 - shares_over_time[:, 2:]
            alive_health_shares = shares_over_time[:, :2] / alive_share
            health_prob_edu_est = alive_health_shares[:, 1]
            health_prob_edu_data = edu_shares_healthy.loc[
                (sex_var, edu_var, slice(None))
            ].values

            # Plot the estimates and the data
            ax.plot(
                est_ages[:max_period_physical],
                health_prob_edu_est[:max_period_physical],
                color=JET_COLOR_MAP[edu_var],
                label=f"Est. {edu_label}",
            )
            # ax.plot(
            #     ages,
            #     health_prob_edu_data,
            #     linestyle="--",
            #     color=JET_COLOR_MAP[edu_var],
            #     label=f"{edu_label}; {sex_label} Data, RM w.
            # BW={specs['health_smoothing_bandwidth']}",
            # )
            ax.plot(
                est_ages[:max_period_physical],
                health_prob_edu_data[:max_period_physical],
                color=JET_COLOR_MAP[edu_var],
                linestyle="--",
                label=f"Obs. {edu_label}",
                # alpha=0.5,
                # s=8,
            )

            # Adjust the x-axis ticks and labels
            x_ticks = np.arange(start_age, max_age_est_physical + 1, 10)
            ax.set_xticks(x_ticks)
            # Set font size for x-axis labels
            ax.set_xticklabels(x_ticks)

        # Set y-axis limits and ticks
        ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        # Set yticks labels and fontsize
        ax.set_yticklabels([f"{i:.0%}" for i in np.arange(0.0, 1.1, 0.1)])

        # Add title and legend
        if male:
            ax.set_title(str(sex_label))
        ax.set_xlabel("Age")
        ax.set_ylabel("Probability of being Healthy")
    axs[0].legend()

    # Show the plot
    fig.tight_layout()
    fig.savefig(path_to_save, dpi=300)
    plt.close(fig)


@pytask.mark.health_transition
def task_plot_healthy_unhealthy_with_caregiving(  # noqa: PLR0912, PLR0915
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "health_transition_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "health_transition_with_caregiving.png",
    male: bool = False,
):
    """Illustrate the health rates by age conditional on lagged intensive caregiving.

    (actual vs. estimated by markov chain)

    Conditional on being alive, what is the probability of being healthy?
    Creates two separate plots: one for no lagged intensive care,
    one for lagged intensive care.
    Each plot shows 4 lines: 2 education levels × 2 types
    (estimated solid, observed dashed)

    """
    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    # Load the health transition sample
    df = pd.read_pickle(path_to_data)

    # Define age range
    start_age = specs["start_age"]
    end_age = specs["end_age"]

    max_age_est_physical = 90
    max_period_physical = max_age_est_physical - start_age
    est_ages = np.arange(start_age, end_age + 1)

    # Which sex(es) to plot?  0 = men, 1 = women
    sexes_to_plot = [1] if not male else [0, 1]

    # Caregiving status mapping
    caregiving_status_map = {
        0: "no_lagged_intensive_care",
        1: "lagged_intensive_care",
    }
    caregiving_labels_plot = {
        0: "No Lagged Intensive Care",
        1: "Lagged Intensive Care",
    }

    # Create two separate plots, one for each caregiving status
    for care_idx, care_status in caregiving_status_map.items():
        # Create the figure for this caregiving status
        fig, axs = plt.subplots(
            ncols=len(sexes_to_plot),
            squeeze=False,
        )
        axs = axs[0]  # flatten first (only one row)

        # Initialize the distribution
        initial_dist = np.zeros(specs["n_health_states"])

        for col_idx, sex_var in enumerate(sexes_to_plot):
            ax = axs[col_idx]
            sex_label = specs["sex_labels"][sex_var]

            # Calculate observed shares for each education level
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                # Filter data by lagged intensive care status
                if care_idx == 0:  # no_lagged_intensive_care
                    df_filtered = df[
                        (df["sex"] == sex_var)
                        & (df["education"] == edu_var)
                        & (
                            (df["lagged_intensive_care"] == 0)
                            | (df["lagged_intensive_care"].isna())
                        )
                    ].copy()
                else:  # lagged_intensive_care
                    df_filtered = df[
                        (df["sex"] == sex_var)
                        & (df["education"] == edu_var)
                        & (df["lagged_intensive_care"] == 1)
                    ].copy()

                # Calculate observed shares for this combination
                if len(df_filtered) > 0:
                    edu_care_shares_data = (
                        df_filtered.groupby("age")["health"]
                        .mean()
                        .loc[slice(start_age, end_age + 1)]
                    )
                else:
                    # If no data, create empty series
                    edu_care_shares_data = pd.Series(index=est_ages, dtype=float)

                # Set the initial distribution for the Markov simulation
                if start_age in edu_care_shares_data.index:
                    initial_dist[1] = edu_care_shares_data.loc[start_age]
                else:
                    # Use overall average if missing
                    initial_dist[1] = (
                        df[
                            (df["sex"] == sex_var)
                            & (df["education"] == edu_var)
                            & (df["age"] == start_age)
                        ]["health"].mean()
                        if len(
                            df[
                                (df["sex"] == sex_var)
                                & (df["education"] == edu_var)
                                & (df["age"] == start_age)
                            ]
                        )
                        > 0
                        else 0.5
                    )
                initial_dist[0] = 1 - initial_dist[1]

                # Extract transition matrix for this sex, education,
                # and caregiving status
                # The caregiving matrix only covers the caregiving age range
                # Period 0 in caregiving matrix = start_age_caregiving
                start_age_caregiving = specs["start_age_caregiving"]
                end_age_caregiving = specs["end_age_caregiving"]
                start_period_caregiving = start_age_caregiving - start_age
                end_period_caregiving = end_age_caregiving - start_age

                # Start with regular health transition matrix
                trans_probs = np.array(
                    specs["health_trans_mat"][sex_var, edu_var, :, :, :]
                )

                # Override with caregiving-specific transitions
                # for caregiving age range
                # Map periods: period in full matrix =
                # period in caregiving matrix + start_period_caregiving
                for period_full in range(
                    start_period_caregiving, end_period_caregiving + 1
                ):
                    trans_probs[period_full, :, :] = np.array(
                        specs["health_trans_mat_with_caregiving"][
                            sex_var, edu_var, period_full, :, care_idx, :
                        ]
                    )

                # Simulate the Markov process and get health probabilities
                shares_over_time = _markov_simulator(initial_dist, trans_probs)
                # Construct health probabilities if alive
                alive_share = 1 - shares_over_time[:, 2:]
                alive_health_shares = shares_over_time[:, :2] / alive_share
                health_prob_est = alive_health_shares[:, 1]

                # Prepare observed data for plotting
                full_index_ages = pd.Index(est_ages, name="age")
                health_prob_data = pd.Series(
                    index=full_index_ages, dtype=float, data=0.0
                )
                health_prob_data.update(edu_care_shares_data)

                # Choose color: different colors for education
                color = JET_COLOR_MAP[edu_var]

                # Plot estimated probabilities (solid line)
                ax.plot(
                    est_ages[:max_period_physical],
                    health_prob_est[:max_period_physical],
                    color=color,
                    linestyle="-",
                    linewidth=2,
                    label=f"Est. {edu_label}",
                )

                # Plot observed data (dashed line) if we have data
                if len(df_filtered) > 0:
                    ax.plot(
                        est_ages[:max_period_physical],
                        health_prob_data.values[:max_period_physical],
                        color=color,
                        linestyle="--",
                        linewidth=2,
                        label=f"Obs. {edu_label}",
                    )

            # Adjust the x-axis ticks and labels
            x_ticks = np.arange(start_age, max_age_est_physical + 1, 10)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)

            # Set y-axis limits and ticks
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0.0, 1.1, 0.1))
            ax.set_yticklabels([f"{i:.0%}" for i in np.arange(0.0, 1.1, 0.1)])

            # Add title and legend
            if male:
                ax.set_title(f"{sex_label} - {caregiving_labels_plot[care_idx]}")
            else:
                ax.set_title(caregiving_labels_plot[care_idx])
            ax.set_xlabel("Age")
            ax.set_ylabel("Probability of being Healthy")
            ax.legend(loc="best")

        # Save the plot for this caregiving status
        fig.tight_layout()
        # Modify the path to include caregiving status in filename
        path_save_care = path_to_save.parent / (
            path_to_save.stem + f"_{care_status}" + path_to_save.suffix
        )
        fig.savefig(path_save_care, dpi=300)
        plt.close(fig)


def _markov_simulator(initial_dist, trans_probs):
    """Simulate a Markov process."""
    n_periods = trans_probs.shape[0]
    n_states = initial_dist.shape[0]
    final_dist = np.zeros((n_periods, n_states))
    final_dist[0, :] = initial_dist

    for t in range(n_periods - 1):
        current_dist = final_dist[t, :]
        for state in range(n_states - 1):
            final_dist[t + 1, state] = current_dist @ trans_probs[t, :, state]

        final_dist[t + 1, -1] = 1 - final_dist[t + 1, :-1].sum()

    return final_dist


# =====================================================================================
# Not used
# =====================================================================================


def task_plot_health_shock_prob(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
):
    """Plot kernel-smoothed probabilities of good or bad health shock."""

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    # Load the transition probabilities
    trans_probs = specs["health_trans_mat"]

    # Define the age range and periods
    n_periods = specs["n_periods"]
    start_age = specs["start_age"]
    periods = range(n_periods)

    # Calculate the tick labels for the x-axis
    age_ticks = [start_age + p for p in range(0, n_periods, 10)]
    tick_positions = list(range(0, n_periods, 10))

    # Define the bandwidth for the kernel density estimation
    bandwidth = specs["health_smoothing_bandwidth"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    for health_var in specs["alive_health_vars"]:
        color_id = 0
        ax = axs[health_var]
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                health_var_to = 1 - health_var
                ax.plot(
                    periods,
                    trans_probs[sex_var, edu_var, :, health_var, health_var_to],
                    color=JET_COLOR_MAP[color_id],
                    label=f"{sex_label}; {edu_label}",
                )
                # Shock is reverse from which is baseline health
                label = specs["health_labels"][health_var_to]
                ax.set_title(
                    f"Probability of {label} Shock (kernel-smoothed), bw={bandwidth}",
                )
                ax.set_ylabel("Probability")
                ax.legend(loc="upper right")
                ax.set_ylim(0, 0.4)
                ax.set_yticks(
                    [i * 0.05 for i in range(9)]
                )  # Y-axis ticks from 0 to 0.4 with steps of 0.05
                # # Set the x-axis ticks and labels
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(age_ticks)
                ax.grid(False)
                color_id += 1
