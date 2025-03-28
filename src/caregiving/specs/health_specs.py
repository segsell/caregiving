"""Create joint health and death transition matrix."""

import jax.numpy as jnp
import numpy as np


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
