"""Create joint health and death transition matrix."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.config import JET_COLOR_MAP


def read_in_health_transition_specs(health_trans_probs_df, death_prob_df, specs):
    """Read in health and death transition specs."""

    alive_health_vars = specs["alive_health_vars"]
    death_health_var = specs["death_health_var"]

    # Transition probalities for health
    health_trans_mat = np.zeros(
        (
            specs["n_sexes"],
            specs["n_education_types"],
            specs["n_periods"],
            specs["n_health_states"],
            specs["n_health_states"],
        ),
        dtype=float,
    )
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            for period in range(specs["n_periods"]):
                for current_health_var in alive_health_vars:
                    for lead_health_var in alive_health_vars:
                        current_health_label = specs["health_labels"][
                            current_health_var
                        ]
                        next_health_label = specs["health_labels"][lead_health_var]
                        trans_prob = health_trans_probs_df.loc[
                            (health_trans_probs_df["sex"] == sex_label)
                            & (health_trans_probs_df["education"] == edu_label)
                            & (health_trans_probs_df["period"] == period)
                            & (health_trans_probs_df["health"] == current_health_label)
                            & (
                                health_trans_probs_df["lead_health"]
                                == next_health_label
                            ),
                            "transition_prob",
                        ].values[0]
                        health_trans_mat[
                            sex_var,
                            edu_var,
                            period,
                            current_health_var,
                            lead_health_var,
                        ] = trans_prob

                    current_age = period + specs["start_age"]

                    # This needs to become label based
                    death_prob = death_prob_df.loc[
                        (death_prob_df["age"] == current_age)
                        & (death_prob_df["sex"] == sex_var)
                        & (death_prob_df["health"] == current_health_var)
                        & (death_prob_df["education"] == edu_var),
                        "death_prob",
                    ].values[0]

                    # Death state. Condition health transitions on surviving and
                    # then assign death probability to death state
                    health_trans_mat[
                        sex_var, edu_var, period, current_health_var, :
                    ] *= (1 - death_prob)
                    health_trans_mat[
                        sex_var, edu_var, period, current_health_var, death_health_var
                    ] = death_prob

    # Death as absorbing state. There are only zeros in the last row of the
    # transition matrix and a 1 on the diagonal element
    health_trans_mat[:, :, :, death_health_var, death_health_var] = 1

    return jnp.asarray(health_trans_mat)


# =====================================================================================
# DataFrame version of the health transition matrix
# =====================================================================================


def read_in_health_transition_specs_df(health_trans_probs_df, death_prob_df, specs):
    """
    Read in health and death transition specs and return a pandas DataFrame
    where the dimensions (sex, education, period, current_health, lead_health)
    become columns, and each row represents a transition with its probability.
    """
    records = []

    alive_health_vars = specs["alive_health_vars"]
    death_health_var = specs["death_health_var"]
    health_labels = specs["health_labels"]
    sex_labels = specs["sex_labels"]
    education_labels = specs["education_labels"]
    start_age = specs["start_age"]

    n_periods = specs["n_periods"]

    # Loop over sex, education, and period
    for sex_idx, sex_label in enumerate(sex_labels):
        for edu_idx, edu_label in enumerate(education_labels):
            for period in range(n_periods):
                current_age = period + start_age

                # Process alive states as current state
                for current_health in alive_health_vars:
                    current_health_label = health_labels[current_health]

                    # Get the death probability for the current alive state
                    death_prob = death_prob_df.loc[
                        (death_prob_df["age"] == current_age)
                        & (death_prob_df["sex"] == sex_idx)
                        & (death_prob_df["health"] == current_health)
                        & (death_prob_df["education"] == edu_idx),
                        "death_prob",
                    ].values[0]

                    # For each possible lead state among alive states
                    for lead_health in alive_health_vars:
                        lead_health_label = health_labels[lead_health]
                        trans_prob = health_trans_probs_df.loc[
                            (health_trans_probs_df["sex"] == sex_label)
                            & (health_trans_probs_df["education"] == edu_label)
                            & (health_trans_probs_df["period"] == period)
                            & (health_trans_probs_df["health"] == current_health_label)
                            & (
                                health_trans_probs_df["lead_health"]
                                == lead_health_label
                            ),
                            "transition_prob",
                        ].values[0]

                        # Adjust transition probability by the survival probability
                        adjusted_prob = trans_prob * (1 - death_prob)

                        records.append(
                            {
                                "sex": sex_label,
                                "education": edu_label,
                                "period": period,
                                "health": current_health_label,
                                "lead_health": lead_health_label,
                                "transition_prob": adjusted_prob,
                            }
                        )

                    # Add the row for transitioning to the death state
                    death_state_label = health_labels[death_health_var]
                    records.append(
                        {
                            "sex": sex_label,
                            "education": edu_label,
                            "period": period,
                            "health": current_health_label,
                            "lead_health": death_state_label,
                            "transition_prob": death_prob,
                        }
                    )

                # Process the death state as the current state (absorbing state)
                death_state_label = health_labels[death_health_var]

                # Loop over all possible lead states in health_labels
                # (both alive and death)
                for lead_health in range(len(health_labels)):
                    lead_health_label = health_labels[lead_health]
                    # Only death-to-death transition has probability 1; all others 0.
                    prob = 1.0 if lead_health == death_health_var else 0.0
                    records.append(
                        {
                            "sex": sex_label,
                            "education": edu_label,
                            "period": period,
                            "health": death_state_label,
                            "lead_health": lead_health_label,
                            "transition_prob": prob,
                        }
                    )

    # Create a DataFrame from the list of records
    df = pd.DataFrame(records)

    return df


# =====================================================================================
# Plotting
# =====================================================================================


def plot_health_transitions(specs, df, path_to_save_plot):
    """Plot health and death transition probabilities by education level and gender.

    The plot is arranged in a 2x2 grid:
      - Rows: education level (first row: Low Education, second row: High Education)
      - Columns: gender (left: men, right: women)

    Transitions from 'Death' to any other state are not plotted.

    """
    sex_labels = specs["sex_labels"]
    edu_labels = specs["education_labels"]
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    health_states = specs["health_labels"]

    # Create age column if needed.
    if "age" not in df.columns:
        df["age"] = df["period"] + start_age

    # Setup color and line style mappings.
    color_map = {
        "Death": JET_COLOR_MAP[0],
        "Bad Health": JET_COLOR_MAP[1],
        "Good Health": JET_COLOR_MAP[2],
    }
    # Define linestyles for transitions originating from alive states.
    linestyle_map = {
        "Bad Health": "--",
        "Good Health": "-",
    }

    # Create a 2x2 grid: rows = education levels, columns = genders.
    fig, axes = plt.subplots(
        nrows=len(edu_labels), ncols=len(sex_labels), figsize=(14, 10), sharey=True
    )

    for edu_idx, edu_label in enumerate(edu_labels):
        for sex_idx, sex in enumerate(sex_labels):
            # Get the appropriate subplot.
            ax = axes[edu_idx, sex_idx]

            # Subset for the given gender and education.
            df_subset = df[(df["sex"] == sex) & (df["education"] == edu_label)]

            # Loop over transitions for non-death originating states.
            for prev_health in health_states:
                if prev_health == "Death":
                    continue  # Skip transitions originating from Death.
                df_prev = df_subset[df_subset["health"] == prev_health]
                for next_health in health_states:
                    df_transition = df_prev[
                        df_prev["lead_health"] == next_health
                    ].sort_values("age")
                    if df_transition.empty:
                        continue

                    # Use the linestyle based on the originating (prev) health state.
                    ls = linestyle_map.get(prev_health, "-")
                    # Color is chosen based on the destination (next) health state.
                    color = color_map.get(next_health, "black")

                    ax.plot(
                        df_transition["age"],
                        df_transition["transition_prob"],
                        color=color,
                        linestyle=ls,
                        linewidth=2,
                        label=f"{prev_health} → {next_health}",
                    )

            # Set subplot labels and titles.
            ax.set_title(f"{edu_label}, {sex}")
            ax.set_xlabel("Age")
            ax.set_xlim(start_age, end_age)
            ax.set_ylim(0, 1)
            if sex_idx == 0:
                ax.set_ylabel("Transition Probability")
            ax.legend(title="Transitions", fontsize=8, title_fontsize=9)

    plt.tight_layout()
    fig.savefig(path_to_save_plot, dpi=300)
