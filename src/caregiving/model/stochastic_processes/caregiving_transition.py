import jax.numpy as jnp

from caregiving.model.shared import MOTHER, SEX


def limitations_with_adl_transition(health, period, options):
    """Transition probability for next period care demand."""
    trans_mat = options["limitations_with_adl_mat"]
    age = period + options["mother_age_difference"]
    prob_vector = trans_mat[MOTHER, age, health, :]

    return prob_vector


def exog_care_transition(has_sister, education, period, options):
    """Transition probability for next period family care supply."""
    trans_mat = options["exog_care_trans_mat"]
    age = period + options["start_age"]
    other_care_supply = trans_mat[age, has_sister, education, :]

    return jnp.array([1 - other_care_supply, other_care_supply])
