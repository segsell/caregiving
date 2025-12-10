import jax.numpy as jnp

from caregiving.model.shared import (
    MOTHER,
    PARENT_DEAD,
    SHARE_CARE_TO_MOTHER,
    START_PERIOD_CAREGIVING,
)
from caregiving.model.stochastic_processes.adl_transition import (
    adl_transition_weighted_by_survival,
)

PARENT_AGE_OFFSET = 3


def health_transition_good_medium_bad(
    mother_health, education, has_sister, period, options
):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat_three"]
    mother_age = (
        period + options["mother_age_diff"][has_sister, education] + PARENT_AGE_OFFSET
    )

    prob_vector = trans_mat[MOTHER, mother_age, mother_health, :]

    return prob_vector


def care_demand_and_supply_transition(
    mother_health, period, has_sister, education, options
):
    """Transition probability for next period care demand."""
    mother_age = (
        period
        - options["agent_to_parent_mat_age_offset"]
        + options["mother_age_diff"][has_sister, education]
        + PARENT_AGE_OFFSET
    )
    end_age_caregiving = options["end_age_msm"] - options["start_age"]

    adl_mat = options["limitations_with_adl_mat"]
    limitations_with_adl = adl_mat[MOTHER, mother_age, mother_health, :]

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = exog_care_supply_mat[period, has_sister, education]

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        limitations_with_adl[1]
        * (mother_health != PARENT_DEAD)
        * (period >= START_PERIOD_CAREGIVING - 1)
        * (period < end_age_caregiving)
        # * SCALE_DOWN_DEMAND
    )

    prob_vector = jnp.array(
        [
            1 - care_demand,  # no care demand
            care_demand
            * prob_other_care_supply
            * SHARE_CARE_TO_MOTHER,  # care demand and others supply care
            care_demand * (1 - prob_other_care_supply),  # care demand and not other
        ]
    )

    return prob_vector


def care_demand_and_supply_transition_adl(
    mother_adl, parent_alive, period, has_sister, education, options
):
    """Transition probability for next period care demand based on ADL.

    Uses ADL state transition matrix (light/intensive) weighted by parent_alive.
    Care demand is based on any ADL (categories 1 or 2).

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
        - mother_age_diff: age difference matrix
        - exog_care_supply: exogenous care supply matrix
        - end_age_msm, start_age: age bounds

    Returns
    -------
    jnp.ndarray
        Probability vector [no_care_demand, care_demand_and_others_supply,
        care_demand_and_not_other]
    """
    end_age_caregiving = options["end_age_msm"] - options["start_age"]

    # Get weighted ADL transition probabilities
    adl_weighted = adl_transition_weighted_by_survival(
        mother_adl, parent_alive, period, has_sister, education, options
    )

    # Care demand is probability of any ADL (ADL 1 OR ADL 2 or ADL 3)
    care_demand = (
        (adl_weighted[1] + adl_weighted[2])
        * (period >= START_PERIOD_CAREGIVING - 1)
        * (period < end_age_caregiving)
    )

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = exog_care_supply_mat[period, has_sister, education]

    prob_vector = jnp.array(
        [
            1 - care_demand,  # no care demand
            care_demand
            * prob_other_care_supply
            * SHARE_CARE_TO_MOTHER,  # care demand and others supply care
            care_demand * (1 - prob_other_care_supply),  # care demand and not other
        ]
    )

    return prob_vector


def care_demand_transition(mother_health, period, has_sister, education, options):
    """Transition probability for next period care demand."""
    adl_mat = options["limitations_with_adl_mat"]
    mother_age = (
        period
        - options["agent_to_parent_mat_age_offset"]
        + options["mother_age_diff"][has_sister, education]
    )

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
        SHARE_CARE_TO_MOTHER  # scale down exog care supply to mother
        * exog_care_supply
        * (mother_health != PARENT_DEAD)
        * (period >= START_PERIOD_CAREGIVING)
    )

    return jnp.array([1 - care_supply, care_supply])


def _care_demand_with_exog_supply_transition(
    mother_health, period, has_sister, education, options
):
    """Transition probability for next period care demand."""
    adl_mat = options["limitations_with_adl_mat"]
    mother_age = (
        period
        - options["agent_to_parent_mat_age_offset"]
        + options["mother_age_diff"][has_sister, education]
    )
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
