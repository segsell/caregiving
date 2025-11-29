import jax.numpy as jnp

from caregiving.model.shared import (
    MOTHER,
    PARENT_DEAD,
    SHARE_CARE_TO_MOTHER,
    START_PERIOD_CAREGIVING,
)

_AGE_OFFSET = 3


def health_transition_good_medium_bad(
    mother_health, education, has_sister, period, options
):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat_three"]
    # health_trans_mat_three is indexed by periods 0-70 for ages 30-100
    # mother_period = (child_age + mother_age_diff) - 30
    # where child_age = start_age + period
    mother_period = (
        period
        + options["start_age"]
        + options["mother_age_diff"][has_sister, education]
        - 30
    )
    # Clamp to valid range: ages 30-100 correspond to periods 0-70
    mother_period = jnp.clip(mother_period, 0, 70)

    prob_vector = trans_mat[MOTHER, mother_period, mother_health, :]

    return prob_vector


def care_demand_and_supply_transition(
    mother_health, period, has_sister, education, options
):
    """Transition probability for next period care demand.

    This function matches the logic from plot_care_demand_from_hdeath_matrix:
    1. Get health transition probabilities for next period
    2. Compute care demand as expected value over next period's health distribution

    Parameters
    ----------
    mother_health : int
        Current mother's health state (0=Bad, 1=Medium, 2=Good, 3=Death)
    period : int
        Current period (agent's age - start_age)
    has_sister : int
        Whether agent has a sister (0 or 1)
    education : int
        Agent's education level (0 or 1)
    options : dict
        Model options containing transition matrices and parameters

    Returns
    -------
    jnp.ndarray
        Probability vector: [no care demand, care demand + other supply,
        care demand + no other supply]
    """
    # Compute mother's period for ADL matrix indexing
    # limitations_with_adl_mat is indexed by periods 0-50 for ages 50-100
    # mother_period = (child_age + mother_age_diff) - start_age_parents
    # where child_age = start_age + period
    mother_period = (
        period
        + (options["start_age"] - options["start_age_parents"])
        + options["mother_age_diff"][has_sister, education]
    )
    end_age_caregiving = options["end_age_msm"] - options["start_age"]

    health_trans_mat = options["health_trans_mat_three"]
    # Get health transition probabilities for next period
    # This gives us P(next health | current health, age)
    # health_trans_probs = health_transition_good_medium_bad(
    #     mother_health, education, has_sister, period, options
    # )
    health_trans_probs = health_trans_mat[MOTHER, mother_period, mother_health, :]
    # health_trans_probs shape: (4,) - probabilities for [Bad, Medium, Good, Death]

    # Get ADL transition matrix (indexed by period and health)
    adl_mat = options["limitations_with_adl_mat"]
    # ADL matrix shape: (sex, period, health, adl_cat)
    # Clamp mother_period to valid range
    # mother_period_clipped = jnp.clip(mother_period, 0, adl_mat.shape[1] - 1)

    # Compute expected ADL probability over next period's health distribution
    # For each possible next health state (0=Bad, 1=Medium, 2=Good), get ADL probability
    # and weight by health transition probability
    next_health_states = jnp.arange(3)  # 0, 1, 2 (exclude Death=3)

    # Get ADL probabilities for each next health state: shape (3, 4)
    adl_probs_by_health = adl_mat[MOTHER, mother_period, next_health_states, :]

    # Sum probabilities for ADL 1, 2, 3 (any ADL) for each health state: shape (3,)
    prob_adl_given_health = (
        adl_probs_by_health[:, 1]
        + adl_probs_by_health[:, 2]
        + adl_probs_by_health[:, 3]
    )

    # Weight by health transition probabilities (first 3 states: Bad, Medium, Good)
    health_weights = health_trans_probs[:3]
    prob_any_adl = jnp.sum(health_weights * prob_adl_given_health)

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = exog_care_supply_mat[period, has_sister, education]

    # Care demand only if mother is not dead
    care_demand = (
        prob_any_adl
        * (mother_health != PARENT_DEAD)
        * (period >= START_PERIOD_CAREGIVING - 1)
        * (period < end_age_caregiving)
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
    # limitations_with_adl_mat is indexed by periods 0-50 for ages 50-100
    # mother_period = (child_age + mother_age_diff) - start_age_parents
    # where child_age = start_age + period
    adl_mat = options["limitations_with_adl_mat"]
    mother_period = (
        period
        + (options["start_age"] - options["start_age_parents"])
        + options["mother_age_diff"][has_sister, education]
    )

    limitations_with_adl = adl_mat[MOTHER, mother_period, mother_health, :]

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
    # limitations_with_adl_mat is indexed by periods 0-50 for ages 50-100
    # mother_period = (child_age + mother_age_diff) - start_age_parents
    # where child_age = start_age + period
    adl_mat = options["limitations_with_adl_mat"]
    mother_period = (
        period
        + (options["start_age"] - options["start_age_parents"])
        + options["mother_age_diff"][has_sister, education]
    )
    limitations_with_adl = adl_mat[MOTHER, mother_period, mother_health, :]

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = exog_care_supply_mat[period, has_sister, education]

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        (1 - (prob_other_care_supply * SHARE_CARE_TO_MOTHER))
        * limitations_with_adl[1]
        * (mother_health != PARENT_DEAD)
    )

    return jnp.array([1 - care_demand, care_demand])
