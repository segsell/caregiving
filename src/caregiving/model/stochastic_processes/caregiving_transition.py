import jax.numpy as jnp

from caregiving.model.shared import (
    MOTHER,
    PARENT_DEAD,
    SHARE_CARE_TO_MOTHER,
    START_PERIOD_CAREGIVING,
)
from caregiving.model.stochastic_processes.adl_transition import (
    limitations_with_adl_transition,
)

PARENT_AGE_OFFSET = 3


def care_demand_and_supply_transition_adl(
    mother_adl, mother_dead, period, education, options
):
    """Transition probability for next period care demand based on ADL.

    Uses ADL state transition matrix (light/intensive) conditional on mother_dead.
    Care demand is based on any ADL (categories 1 or 2).

    Parameters
    ----------
    mother_adl : int
        Current ADL state (0=No ADL, 1=ADL 1, 2=ADL 2 or ADL 3)
    mother_dead : int
        Mother death status (1=dead, 0=alive)
    period : int
        Current period
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

    # No ADL, ADL 1, ADL 2 or ADL 3 (three states)
    prob_adl = limitations_with_adl_transition(mother_adl, period, education, options)

    # Care demand is probability of any ADL (ADL 1 OR ADL 2 or ADL 3)
    # Who provides care depends on caregiving_type (handled in choice set)
    care_demand = (
        (prob_adl[1] + prob_adl[2])
        * (1 - mother_dead)
        * (period >= START_PERIOD_CAREGIVING - 1)
        * (period < end_age_caregiving)
    )

    return jnp.array([1 - care_demand, care_demand])


def health_transition_good_medium_bad(mother_health, education, period, options):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat_three"]
    mother_age = period + options["mother_age_diff"][education] + PARENT_AGE_OFFSET

    prob_vector = trans_mat[MOTHER, mother_age, mother_health, :]

    return prob_vector


def _care_demand_and_supply_transition_adl(
    mother_adl, mother_dead, period, education, options
):
    """Transition probability for next period care demand based on ADL.

    Uses ADL state transition matrix (light/intensive) conditional on mother_dead.
    Care demand is based on any ADL (categories 1 or 2).

    Parameters
    ----------
    mother_adl : int
        Current ADL state (0=No ADL, 1=ADL 1, 2=ADL 2 or ADL 3)
    mother_dead : int
        Mother death status (1=dead, 0=alive)
    period : int
        Current period
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

    # No ADL, ADL 1, ADL 2 or ADL 3 (three states)
    prob_adl = limitations_with_adl_transition(mother_adl, period, education, options)

    # Care demand is probability of any ADL (ADL 1 OR ADL 2 or ADL 3)
    care_demand = (
        (prob_adl[1] + prob_adl[2])
        * (1 - mother_dead)
        * (period >= START_PERIOD_CAREGIVING - 1)
        * (period < end_age_caregiving)
    )

    # Pre-compute care supply probability
    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = (
        exog_care_supply_mat[period, education] * SHARE_CARE_TO_MOTHER
    )

    prob_vector = jnp.array(
        [
            1 - care_demand,  # no care demand
            care_demand * prob_other_care_supply,  # care demand and others supply care
            care_demand * (1 - prob_other_care_supply),  # care demand and not other
        ]
    )

    return prob_vector


def care_demand_and_supply_transition(mother_health, period, education, options):
    """Transition probability for next period care demand."""
    mother_age = (
        period
        - options["agent_to_parent_mat_age_offset"]
        + options["mother_age_diff"][education]
        + PARENT_AGE_OFFSET
    )
    end_age_caregiving = options["end_age_msm"] - options["start_age"]

    adl_mat = options["limitations_with_adl_mat"]
    limitations_with_adl = adl_mat[MOTHER, mother_age, mother_health, :]

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = (
        exog_care_supply_mat[period, education] * SHARE_CARE_TO_MOTHER
    )

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
            care_demand * prob_other_care_supply,  # care demand and others supply care
            care_demand * (1 - prob_other_care_supply),  # care demand and not other
        ]
    )

    return prob_vector


def _care_demand_transition(mother_health, period, education, options):
    """Transition probability for next period care demand."""
    adl_mat = options["limitations_with_adl_mat"]
    mother_age = (
        period
        - options["agent_to_parent_mat_age_offset"]
        + options["mother_age_diff"][education]
    )

    limitations_with_adl = adl_mat[MOTHER, mother_age, mother_health, :]

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        limitations_with_adl[1]
        * (mother_health != PARENT_DEAD)
        * (period >= START_PERIOD_CAREGIVING - 1)
    )

    return jnp.array([1 - care_demand, care_demand])


def _exog_care_supply_transition(mother_health, education, period, options):
    """Transition probability for next period family care supply."""
    exog_care_supply_mat = options["exog_care_supply"]
    exog_care_supply = exog_care_supply_mat[period, education]

    care_supply = (
        SHARE_CARE_TO_MOTHER  # scale down exog care supply to mother
        * exog_care_supply
        * (mother_health != PARENT_DEAD)
        * (period >= START_PERIOD_CAREGIVING)
    )

    return jnp.array([1 - care_supply, care_supply])
