import jax.numpy as jnp

from caregiving.model.shared import (
    MOTHER,
    PARENT_DEAD,
    SHARE_CARE_TO_MOTHER,
)
from caregiving.model.stochastic_processes.adl_transition import (
    limitations_with_adl_transition,
)

PARENT_AGE_OFFSET = 3


def care_demand_transition_adl_light_intensive(
    mother_adl, mother_dead, period, education, model_specs
):
    """Transition probability for next period care demand based on ADL.

    Uses ADL state transition matrix (light/intensive) conditional on mother_dead.
    Care demand has three states:
    0 = no care demand (ADL 0),
    1 = light care demand (ADL 1),
    2 = intensive care demand (ADL 2 or higher).

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
        - end_age_caregiving, start_age: age bounds (or end_age_msm as fallback)

    Returns
    -------
    jnp.ndarray
        Probability vector over care_demand states [0, 1, 2], where:
        - 0: no care demand (no ADL limitations),
        - 1: light care (ADL 1),
        - 2: intensive care (ADL 2 or ADL 3).
    """
    end_period_caregiving = model_specs["end_age_caregiving"] - model_specs["start_age"]
    start_period_caregiving = model_specs["start_period_caregiving"]

    # No ADL, ADL 1, ADL 2 or ADL 3 (three states)
    prob_adl = limitations_with_adl_transition(
        mother_adl, period, education, model_specs
    )

    # Restrict care demand to the caregiving window and to living mothers.
    # Outside the caregiving window or if mother is dead: no care demand (state 0).
    # Note: If mother_dead == 1, then in_caregiving_window = 0, ensuring care_demand = 0.
    in_caregiving_window = (
        (1 - mother_dead)
        * (period >= start_period_caregiving - 1)
        * (period < end_period_caregiving)
    )

    # Map ADL states to care_demand:
    # - ADL 0 -> care_demand 0 (no demand)
    # - ADL 1 -> care_demand 1 (light)
    # - ADL 2/3 -> care_demand 2 (intensive)
    #
    # When mother_dead == 1: in_caregiving_window = 0, so:
    # - p_no_care = prob_adl[0] + prob_adl[1] + prob_adl[2] = 1.0
    # - p_light = 0, p_intensive = 0
    # This guarantees care_demand == 0 whenever mother_dead == 1.
    p_no_care = prob_adl[0] + (1 - in_caregiving_window) * (prob_adl[1] + prob_adl[2])
    p_light = prob_adl[1] * in_caregiving_window
    p_intensive = prob_adl[2] * in_caregiving_window

    # Ensure probabilities sum to 1 (sanity check)
    probs = jnp.array([p_no_care, p_light, p_intensive])
    # Note: In JAX, we can't use assertions that would break compilation,
    # but the math guarantees this sums to 1.0

    return probs


def care_demand_and_supply_transition_adl(
    mother_adl, mother_dead, period, education, model_specs
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
        - end_age_caregiving, start_age: age bounds (or end_age_msm as fallback)

    Returns
    -------
    jnp.ndarray
        Probability vector [no_care_demand, care_demand_and_others_supply,
        care_demand_and_not_other]
    """
    end_period_caregiving = model_specs["end_age_caregiving"] - model_specs["start_age"]
    start_period_caregiving = model_specs["start_period_caregiving"]

    # No ADL, ADL 1, ADL 2 or ADL 3 (three states)
    prob_adl = limitations_with_adl_transition(
        mother_adl, period, education, model_specs
    )

    # Care demand is probability of any ADL (ADL 1 OR ADL 2 or ADL 3)
    # Who provides care depends on caregiving_type (handled in choice set)
    care_demand = (
        (prob_adl[1] + prob_adl[2])
        * (1 - mother_dead)
        * (period >= start_period_caregiving - 1)
        * (period < end_period_caregiving)
    )

    return jnp.array([1 - care_demand, care_demand])


def health_transition_good_medium_bad(mother_health, education, period, model_specs):
    """Transition probability for next period health state."""
    trans_mat = model_specs["health_trans_mat_three"]
    mother_age = period + model_specs["mother_age_diff"][education] + PARENT_AGE_OFFSET

    prob_vector = trans_mat[MOTHER, mother_age, mother_health, :]

    return prob_vector


def _care_demand_and_supply_transition_adl(
    mother_adl, mother_dead, period, education, model_specs
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
    model_specs : dict
        Model specifications dictionary containing:
        - adl_state_transition_mat_light_intensive: ADL transition matrix
        - mother_age_diff: age difference matrix
        - exog_care_supply: exogenous care supply matrix
        - end_age_caregiving, start_age: age bounds (or end_age_msm as fallback)

    Returns
    -------
    jnp.ndarray
        Probability vector [no_care_demand, care_demand_and_others_supply,
        care_demand_and_not_other]
    """

    end_period_caregiving = model_specs["end_age_caregiving"] - model_specs["start_age"]
    start_period_caregiving = model_specs["start_period_caregiving"]

    # No ADL, ADL 1, ADL 2 or ADL 3 (three states)
    prob_adl = limitations_with_adl_transition(
        mother_adl, period, education, model_specs
    )

    # Care demand is probability of any ADL (ADL 1 OR ADL 2 or ADL 3)
    care_demand = (
        (prob_adl[1] + prob_adl[2])
        * (1 - mother_dead)
        * (period >= start_period_caregiving - 1)
        * (period < end_period_caregiving)
    )

    # Pre-compute care supply probability
    exog_care_supply_mat = model_specs["exog_care_supply"]
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


def care_demand_and_supply_transition(mother_health, period, education, model_specs):
    """Transition probability for next period care demand."""
    mother_age = (
        period
        - model_specs["agent_to_parent_mat_age_offset"]
        + model_specs["mother_age_diff"][education]
        + PARENT_AGE_OFFSET
    )
    end_period_caregiving = model_specs["end_age_caregiving"] - model_specs["start_age"]

    adl_mat = model_specs["limitations_with_adl_mat"]
    limitations_with_adl = adl_mat[MOTHER, mother_age, mother_health, :]

    exog_care_supply_mat = model_specs["exog_care_supply"]
    prob_other_care_supply = (
        exog_care_supply_mat[period, education] * SHARE_CARE_TO_MOTHER
    )

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        limitations_with_adl[1]
        * (mother_health != PARENT_DEAD)
        * (period >= start_period_caregiving - 1)
        * (period < end_period_caregiving)
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


def _care_demand_transition(mother_health, period, education, model_specs):
    """Transition probability for next period care demand."""
    adl_mat = model_specs["limitations_with_adl_mat"]
    mother_age = (
        period
        - model_specs["agent_to_parent_mat_age_offset"]
        + model_specs["mother_age_diff"][education]
    )

    limitations_with_adl = adl_mat[MOTHER, mother_age, mother_health, :]

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        limitations_with_adl[1]
        * (mother_health != PARENT_DEAD)
        * (period >= model_specs["start_period_caregiving"] - 1)
    )

    return jnp.array([1 - care_demand, care_demand])


def _exog_care_supply_transition(mother_health, education, period, model_specs):
    """Transition probability for next period family care supply."""
    exog_care_supply_mat = model_specs["exog_care_supply"]
    exog_care_supply = exog_care_supply_mat[period, education]

    care_supply = (
        SHARE_CARE_TO_MOTHER  # scale down exog care supply to mother
        * exog_care_supply
        * (mother_health != PARENT_DEAD)
        * (period >= model_specs["start_period_caregiving"])
    )

    return jnp.array([1 - care_supply, care_supply])
