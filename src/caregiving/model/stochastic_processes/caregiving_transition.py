import jax.numpy as jnp

from caregiving.model.shared import (
    ADL_DEAD,
    MOTHER,
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
    mother_adl, period, has_sister, education, options
):
    """Transition probability for next period care demand."""
    # adl_state_transition_mat_with_death is indexed by periods 0-50 for ages 50-100
    # mother_period = (child_age + mother_age_diff) - start_age_parents
    # where child_age = start_age + period
    mother_period = (
        period
        + (options["start_age"] - options["start_age_parents"])
        + options["mother_age_diff"][has_sister, education]
    )
    end_age_caregiving = options["end_age_msm"] - options["start_age"]

    # Get ADL transition probabilities: [sex, period, adl_lag_state, adl_next_state]
    # adl_next_state: 0=No ADL, 1=ADL 1, 2=ADL 2, 3=ADL 3, 4=Death
    adl_trans_mat = options["adl_state_transition_mat_with_death"]
    adl_trans_probs = adl_trans_mat[MOTHER, mother_period, mother_adl, :]

    # Care demand occurs if next ADL state is 1, 2, or 3 (any ADL, not death)
    prob_any_adl = (
        adl_trans_probs[1] + adl_trans_probs[2] + adl_trans_probs[3]
    )  # P(ADL 1, 2, or 3)

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = exog_care_supply_mat[period, has_sister, education]

    # Care demand only if mother is not dead
    care_demand = (
        prob_any_adl
        * (mother_adl != ADL_DEAD)
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


def care_demand_transition(mother_adl, period, has_sister, education, options):
    """Transition probability for next period care demand."""
    # adl_state_transition_mat_with_death is indexed by periods 0-50 for ages 50-100
    # mother_period = (child_age + mother_age_diff) - start_age_parents
    # where child_age = start_age + period
    mother_period = (
        period
        + (options["start_age"] - options["start_age_parents"])
        + options["mother_age_diff"][has_sister, education]
    )

    # Get ADL transition probabilities: [sex, period, adl_lag_state, adl_next_state]
    adl_trans_mat = options["adl_state_transition_mat_with_death"]
    adl_trans_probs = adl_trans_mat[MOTHER, mother_period, mother_adl, :]

    # Care demand occurs if next ADL state is 1, 2, or 3 (any ADL, not death)
    prob_any_adl = (
        adl_trans_probs[1] + adl_trans_probs[2] + adl_trans_probs[3]
    )  # P(ADL 1, 2, or 3)

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        prob_any_adl
        * (mother_adl != ADL_DEAD)
        * (period >= START_PERIOD_CAREGIVING - 1)
    )

    return jnp.array([1 - care_demand, care_demand])


def exog_care_supply_transition(mother_adl, has_sister, education, period, options):
    """Transition probability for next period family care supply."""
    exog_care_supply_mat = options["exog_care_supply"]
    exog_care_supply = exog_care_supply_mat[period, has_sister, education]

    care_supply = (
        exog_care_supply
        * (mother_adl != ADL_DEAD)
        * (period >= START_PERIOD_CAREGIVING)
    )
    # SHARE_CARE_TO_MOTHER *

    return jnp.array([1 - care_supply, care_supply])


def _care_demand_with_exog_supply_transition(
    mother_adl, period, has_sister, education, options
):
    """Transition probability for next period care demand."""
    # adl_state_transition_mat_with_death is indexed by periods 0-50 for ages 50-100
    # mother_period = (child_age + mother_age_diff) - start_age_parents
    # where child_age = start_age + period
    mother_period = (
        period
        + (options["start_age"] - options["start_age_parents"])
        + options["mother_age_diff"][has_sister, education]
    )

    # Get ADL transition probabilities: [sex, period, adl_lag_state, adl_next_state]
    adl_trans_mat = options["adl_state_transition_mat_with_death"]
    adl_trans_probs = adl_trans_mat[MOTHER, mother_period, mother_adl, :]

    # Care demand occurs if next ADL state is 1, 2, or 3 (any ADL, not death)
    prob_any_adl = (
        adl_trans_probs[1] + adl_trans_probs[2] + adl_trans_probs[3]
    )  # P(ADL 1, 2, or 3)

    exog_care_supply_mat = options["exog_care_supply"]
    prob_other_care_supply = exog_care_supply_mat[period, has_sister, education]

    # no_care_demand = prob_other_care * limitations_with_adl[0]
    care_demand = (
        (1 - (prob_other_care_supply * SHARE_CARE_TO_MOTHER))
        * prob_any_adl
        * (mother_adl != ADL_DEAD)
    )

    return jnp.array([1 - care_demand, care_demand])
