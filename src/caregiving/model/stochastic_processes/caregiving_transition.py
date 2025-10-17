import jax.numpy as jnp

from caregiving.model.shared import (
    MOTHER,
    PARENT_DEAD,
    SHARE_CARE_TO_MOTHER,
    START_PERIOD_CAREGIVING,
)

AGE_OFFSET = 3
SCALE_DOWN_DEMAND = 0.9


def health_transition_good_medium_bad(
    mother_health, education, has_sister, period, options
):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat_three"]
    mother_age = period + options["mother_age_diff"][has_sister, education] + AGE_OFFSET

    prob_vector = trans_mat[MOTHER, mother_age, mother_health, :]

    return prob_vector


def care_demand_and_supply_transition(
    mother_health, period, has_sister, education, options
):
    """Transition probability for next period care demand."""
    mother_age = (
        period - 20 + options["mother_age_diff"][has_sister, education] + AGE_OFFSET
    )
    end_period_caregiving = options["end_age_msm"] - options["start_age"]

    adl_mat = options["limitations_with_adl_mat"]
    limitations_with_adl = adl_mat[MOTHER, mother_age, mother_health, :]

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = exog_care_supply_mat[period, has_sister, education]

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        (
            limitations_with_adl[1]
            * (mother_health != PARENT_DEAD)
            * (period >= START_PERIOD_CAREGIVING - 1)
            * (period < end_period_caregiving)
        )
        * SCALE_DOWN_DEMAND
        * (period >= START_PERIOD_CAREGIVING - 1)
    )

    prob_vector = jnp.array(
        [
            1 - care_demand,  # no care demand
            care_demand * prob_other_care_supply,  # care demand and others supply care
            care_demand * (1 - prob_other_care_supply),  # care demand and not other
        ]
    )

    return prob_vector


def care_demand_transition(mother_health, period, has_sister, education, options):
    """Transition probability for next period care demand."""
    adl_mat = options["limitations_with_adl_mat"]
    mother_age = period - 20 + options["mother_age_diff"][has_sister, education]

    limitations_with_adl = adl_mat[MOTHER, mother_age, mother_health, :]

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        limitations_with_adl[1]
        * (mother_health != PARENT_DEAD)
        * (period >= START_PERIOD_CAREGIVING - 1)
    )

    return jnp.array([1 - care_demand, care_demand])


def exog_care_supply_transition(mother_health, has_sister, education, period, options):
    """Transition probability for next period family care supply."""
    exog_care_supply_mat = options["exog_care_supply"]
    exog_care_supply = exog_care_supply_mat[period, has_sister, education]

    care_supply = (
        exog_care_supply
        * (mother_health != PARENT_DEAD)
        * (period >= START_PERIOD_CAREGIVING)
    )
    # SHARE_CARE_TO_MOTHER *

    return jnp.array([1 - care_supply, care_supply])


def _care_demand_with_exog_supply_transition(
    mother_health, period, has_sister, education, options
):
    """Transition probability for next period care demand."""
    adl_mat = options["limitations_with_adl_mat"]
    mother_age = period - 20 + options["mother_age_diff"][has_sister, education]
    limitations_with_adl = adl_mat[MOTHER, mother_age, mother_health, :]

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = exog_care_supply_mat[period, has_sister, education]

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        (1 - (prob_other_care_supply * SHARE_CARE_TO_MOTHER))
        * limitations_with_adl[1]
        * (mother_health != PARENT_DEAD)
    )

    return jnp.array([1 - care_demand, care_demand])
