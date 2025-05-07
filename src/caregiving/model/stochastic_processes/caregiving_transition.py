import jax.numpy as jnp
import numpy as np

from caregiving.model.shared import MOTHER


def health_transition_good_medium_bad(
    mother_health, education, has_sister, period, options
):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat_three"]
    mother_age = period + options["mother_age_diff"][has_sister, education]

    prob_vector = trans_mat[MOTHER, mother_age, mother_health, :]

    return prob_vector


def care_demand_transition(mother_health, period, has_sister, education, options):
    """Transition probability for next period care demand."""
    adl_mat = options["limitations_with_adl_mat"]
    mother_age = period - 20 + options["mother_age_diff"][has_sister, education]
    limitations_with_adl = adl_mat[MOTHER, mother_age, mother_health, :]

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care = exog_care_supply_mat[period, has_sister, education]

    care_demand = (1 - prob_other_care) * limitations_with_adl[1]
    # no_care_demand = prob_other_care * limitations_with_adl[0]

    return jnp.array([1 - care_demand, care_demand])


def exog_care_transition(has_sister, education, period, options):
    """Transition probability for next period family care supply."""
    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = exog_care_supply_mat[period, has_sister, education]

    return jnp.array([1 - prob_other_care_supply, prob_other_care_supply])
