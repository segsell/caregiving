"""Transition functions for limitations with ADL states."""

import jax.numpy as jnp

from caregiving.model.shared import MOTHER

PARENT_AGE_OFFSET = 3


def limitations_with_adl_transition(mother_adl, period, has_sister, education, options):
    """Transition probability for next period ADL state given current ADL.

    Parameters
    ----------
    mother_adl : int
        Current ADL state (0=No ADL, 1=ADL 1, 2=ADL 2 or ADL 3)
    period : int
        Current period
    has_sister : int
        Whether caregiver has a sister (0 or 1)
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
        - options["agent_to_parent_mat_age_offset"]
        + options["mother_age_diff"][has_sister, education]
        + PARENT_AGE_OFFSET
    )

    adl_trans_mat = options["adl_state_transition_mat_light_intensive"]
    # Shape: [sex, period, adl_lag, adl_next]
    prob_vector = adl_trans_mat[MOTHER, mother_age, mother_adl, :]

    return prob_vector


def adl_transition_weighted_by_survival(
    mother_adl, parent_alive, period, has_sister, education, options
):
    """ADL transition probabilities weighted by parent survival status.

    Uses ADL state transition matrix (light/intensive) and weights by
    parent_alive state variable. ADL probabilities only apply to alive population.

    Parameters
    ----------
    mother_adl : int
        Current ADL state (0=No ADL, 1=ADL 1, 2=ADL 2 or ADL 3)
    parent_alive : int
        Parent alive status (1=alive, 0=dead)
    period : int
        Current period
    has_sister : int
        Whether caregiver has a sister (0 or 1)
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
        Weighted probability vector over ADL states [No ADL, ADL 1, ADL 2 or ADL 3]
        where ADL 0 includes both alive with no ADL and dead
    """
    # Calculate mother age index
    mother_age = (
        period
        - options["agent_to_parent_mat_age_offset"]
        + options["mother_age_diff"][has_sister, education]
        + PARENT_AGE_OFFSET
    )

    # Get ADL transition matrix (light/intensive: 3 categories)
    adl_trans_mat = options["adl_state_transition_mat_light_intensive"]
    # Shape: [sex, period, adl_lag, adl_next]
    # adl_lag/adl_next: 0=No ADL, 1=ADL 1, 2=ADL 2 or ADL 3

    # Get ADL transition probabilities for next period
    # Shape: [3] for transitions to [No ADL, ADL 1, ADL 2 or ADL 3]
    adl_trans_probs = adl_trans_mat[MOTHER, mother_age, mother_adl, :]

    # Weight ADL probabilities by parent_alive
    # ADL 0 (No ADL) includes both alive with no ADL and dead
    # ADL 1, ADL 2 or ADL 3 only apply to alive population
    adl_0_weighted = (1 - parent_alive) + parent_alive * adl_trans_probs[0]
    adl_1_weighted = parent_alive * adl_trans_probs[1]
    adl_2_or_3_weighted = parent_alive * adl_trans_probs[2]

    return jnp.array([adl_0_weighted, adl_1_weighted, adl_2_or_3_weighted])


def death_transition(period, mother_dead, has_sister, education, options):
    """Death transition probability for next period.

    Uses death transition matrix indexed by sex and age. If mother was already
    dead (mother_dead=1), she remains dead with certainty.

    Parameters
    ----------
    period : int
        Current period
    mother_dead : int
        Lagged mother death status (1=dead, 0=alive)
    has_sister : int
        Whether caregiver has a sister (0 or 1)
    education : int
        Education level (0=Low, 1=High)
    options : dict
        Options dictionary containing:
        - death_transition_mat: death probability matrix [sex, age_index]
        - survival_min_age: minimum age in death/survival matrices (age_index 0 = age 16)
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
        - options["agent_to_parent_mat_age_offset"]
        + options["mother_age_diff"][has_sister, education]
        + PARENT_AGE_OFFSET
    )

    # Convert to age_index for death_transition_mat
    # death_transition_mat[sex, age_index] where age_index = age - survival_min_age
    # survival_min_age = 16, so age_index 0 = age 16
    survival_min_age = options["survival_min_age"]
    age_index = mother_age - survival_min_age

    # Get death transition matrix
    death_mat = options["death_transition_mat"]
    # Shape: [sex, age_index] where sex=0 is Men, sex=1 is Women (MOTHER=1)

    # Clip age_index to valid bounds and get death probability (vectorized)
    max_idx = death_mat.shape[1] - 1
    clipped_idx = jnp.clip(age_index, 0, max_idx)
    in_bounds = (age_index >= 0) & (age_index <= max_idx)
    death_prob = jnp.where(
        in_bounds, death_mat[MOTHER, clipped_idx], 1.0
    )  # If out of bounds, assume dead

    # If mother was already dead, she stays dead with certainty
    # alive_prob = jnp.where(mother_dead == 1, 0.0, 1.0 - death_prob)
    # dead_prob = jnp.where(mother_dead == 1, 1.0, death_prob)

    alive_prob = (1 - death_prob) * (1 - mother_dead)

    return jnp.array([alive_prob, 1 - alive_prob])
