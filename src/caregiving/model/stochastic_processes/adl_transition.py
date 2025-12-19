"""Transition functions for limitations with ADL states."""

import jax.numpy as jnp

from caregiving.model.shared import MOTHER

PARENT_AGE_OFFSET = 3


def limitations_with_adl_transition(mother_adl, period, education, model_specs):
    """Transition probability for next period ADL state given current ADL.

    Parameters
    ----------
    mother_adl : int
        Current ADL state (0=No ADL, 1=ADL 1, 2=ADL 2 or ADL 3)
    period : int
        Current period
    education : int
        Education level (0=Low, 1=High)
    options : dict
        Options dictionary containing:
        - adl_state_transition_mat_light_intensive: ADL transition matrix
          [sex, period, adl_lag, adl_next]
        - mother_age_diff: age difference matrix

    Returns
    -------
    jnp.ndarray
        Probability vector over ADL states [No ADL, ADL 1, ADL 2 or ADL 3]
    """
    # Calculate mother age index
    mother_age = (
        period
        - model_specs["agent_to_parent_mat_age_offset"]
        + model_specs["mother_age_diff"][education]
        + PARENT_AGE_OFFSET
    )

    adl_trans_mat = model_specs["adl_state_transition_mat_light_intensive"]
    # Shape: [sex, period, adl_lag, adl_next]
    prob_vector = adl_trans_mat[MOTHER, mother_age, mother_adl, :]

    return prob_vector


def death_transition(period, mother_dead, education, model_specs):
    """Death transition probability for next period.

    Uses death transition matrix indexed by sex and age. If mother was already
    dead (mother_dead=1), she remains dead with certainty.

    Parameters
    ----------
    period : int
        Current period
    mother_dead : int
        Lagged mother death status (1=dead, 0=alive)
    education : int
        Education level (0=Low, 1=High)
    model_specs : dict
        Model specifications dictionary containing:
        - death_transition_mat: death probability matrix [sex, age_index]
        - survival_min_age: minimum age in death/survival matrices
          (age_index 0 = age 16)
        - agent_to_parent_mat_age_offset: age offset between agent and parent matrices
        - mother_age_diff: age difference matrix

    Returns
    -------
    jnp.ndarray
        Probability vector [1 - death_prob, death_prob] where:
        - If mother_dead=1: [0, 1] (dead with certainty)
        - If mother_dead=0: [1 - death_prob, death_prob] from transition matrix
    """
    # Calculate mother's actual age from period using correct mother_age_diff
    mother_age = (
        period
        - model_specs["agent_to_parent_mat_age_offset"]
        + model_specs["mother_age_diff"][education]
        + PARENT_AGE_OFFSET
    )

    # Convert to age_index for death_transition_mat
    # death_transition_mat[sex, age_index] where age_index = age - survival_min_age
    # age_index = mother_age + 34
    age_index = mother_age + model_specs["parent_to_survival_mat_age_offset"]

    # Get death transition matrix
    death_mat = model_specs["death_transition_mat"]
    # Shape: [sex, age_index] where sex=0 is Men, sex=1 is Women (MOTHER=1)

    # Clip age_index to valid bounds and get death probability (vectorized)
    max_idx = death_mat.shape[1] - 1
    clipped_idx = jnp.clip(age_index, 0, max_idx)
    in_bounds = (age_index >= 0) & (age_index <= max_idx)
    death_prob = jnp.where(
        in_bounds, death_mat[MOTHER, clipped_idx], 1.0
    )  # If out of bounds, assume dead
    # death_prob = death_mat[MOTHER, age_index]

    # If mother was already dead, she stays dead with certainty
    # alive_prob = jnp.where(mother_dead == 1, 0.0, 1.0 - death_prob)
    # dead_prob = jnp.where(mother_dead == 1, 1.0, death_prob)
    alive_prob = (1 - death_prob) * (1 - mother_dead)

    return jnp.array([alive_prob, 1 - alive_prob])
