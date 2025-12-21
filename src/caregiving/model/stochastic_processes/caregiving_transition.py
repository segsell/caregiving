import jax.numpy as jnp

from caregiving.model.shared import (
    MOTHER,
    PARENT_DEAD,
    SHARE_CARE_TO_MOTHER,
    # NO_CARE_DEMAND_DEAD,
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


# def care_demand_death_transition_light_intensive(
#     mother_adl, period, education, model_specs
# ):
#     """Combined transition probability for care demand and death.

#     Combines death transition and care demand transition into a single 4-state process:
#     - 0: NO_CARE_DEMAND_DEAD (mother is dead)
#     - 1: NO_CARE_DEMAND_ALIVE (mother is alive, no ADL, outside caregiving window)
#     - 2: CARE_DEMAND_LIGHT (mother is alive, ADL 1, in caregiving window)
#     - 3: CARE_DEMAND_INTENSIVE (mother is alive, ADL 2 or 3, in caregiving window)

#     Once mother is dead (mother_adl == 0, "No ADL Dead"), she remains dead forever
#     and care_demand is NO_CARE_DEMAND_DEAD (0) forever.

#     The ADL transition probabilities from limitations_with_adl_transition are
#     conditional on being alive, so we can combine them with death probabilities.

#     Parameters
#     ----------
#     mother_adl : int
#         Current ADL state:
#         - 0: "No ADL Dead" (mother is dead)
#         - 1: "No ADL Alive" (mother is alive, no ADL)
#         - 2: "ADL 1" (mother is alive, ADL 1)
#         - 3: "ADL 2 or ADL 3" (mother is alive, ADL 2 or 3)
#     period : int
#         Current period
#     education : int
#         Education level (0=Low, 1=High)
#     model_specs : dict
#         Model specifications dictionary containing:
#         - adl_state_transition_mat_light_intensive: ADL transition matrix
#         - death_transition_mat: death probability matrix [sex, age_index]
#         - mother_age_diff: age difference matrix
#         - agent_to_parent_mat_age_offset: age offset between agent and parent matrices
#         - parent_to_survival_mat_age_offset: age offset for survival matrices
#         - end_age_caregiving, start_age: age bounds
#         - start_period_caregiving: start period for caregiving

#     Returns
#     -------
#     jnp.ndarray
#         Probability vector over care_demand states [0, 1, 2, 3], where:
#         - 0: NO_CARE_DEMAND_DEAD (mother is dead)
#         - 1: NO_CARE_DEMAND_ALIVE (mother is alive, no ADL, outside caregiving window)
#         - 2: CARE_DEMAND_LIGHT (mother is alive, ADL 1, in caregiving window)
#         - 3: CARE_DEMAND_INTENSIVE (mother is alive, ADL 2 or 3, in caregiving window)
#     """
#     # Infer mother_dead from mother_adl: mother_adl == 0 means "No ADL Dead"
#     # mother_adl states: 0="No ADL Dead", 1="No ADL Alive", 2="ADL 1", 3="ADL 2 or ADL 3"
#     already_dead = mother_adl == NO_CARE_DEMAND_DEAD

#     # Calculate death probability for next period (if mother is currently alive)
#     # Calculate mother's actual age from period
#     mother_age = (
#         period
#         - model_specs["agent_to_parent_mat_age_offset"]
#         + model_specs["mother_age_diff"][education]
#         + PARENT_AGE_OFFSET
#     )

#     # Convert to age_index for death_transition_mat
#     age_index = mother_age + model_specs["parent_to_survival_mat_age_offset"]

#     # Get death transition matrix
#     death_mat = model_specs["death_transition_mat"]
#     # Shape: [sex, age_index] where sex=0 is Men, sex=1 is Women (MOTHER=1)

#     # Clip age_index to valid bounds and get death probability
#     max_idx = death_mat.shape[1] - 1
#     clipped_idx = jnp.clip(age_index, 0, max_idx)
#     in_bounds = (age_index >= 0) & (age_index <= max_idx)
#     death_prob = jnp.where(
#         in_bounds, death_mat[MOTHER, clipped_idx], 1.0
#     )  # If out of bounds, assume dead

#     # If mother was already dead (mother_adl == 0), she stays dead forever
#     # Return [1.0, 0.0, 0.0, 0.0] (state 0: NO_CARE_DEMAND_DEAD)
#     # Use JAX-compatible conditional logic
#     dead_probs = jnp.array([1.0, 0.0, 0.0, 0.0])

#     # If mother was already dead, death_prob should be 1.0
#     death_prob = jnp.where(already_dead, 1.0, death_prob)
#     alive_prob = 1.0 - death_prob

#     # Get ADL transition probabilities (conditional on being alive)
#     # mother_adl states when alive: 1="No ADL Alive", 2="ADL 1", 3="ADL 2 or ADL 3"
#     # Map to internal ADL representation for limitations_with_adl_transition:
#     # - mother_adl == 1 (No ADL Alive) -> adl_state = 0 (No ADL)
#     # - mother_adl == 2 (ADL 1) -> adl_state = 1 (ADL 1)
#     # - mother_adl == 3 (ADL 2 or 3) -> adl_state = 2 (ADL 2/3)
#     # Note: mother_adl == 0 (dead) is handled above, so we subtract 1 to map [1,2,3] -> [0,1,2]
#     adl_state_for_transition = jnp.where(already_dead, 0, mother_adl - 1)

#     prob_adl = limitations_with_adl_transition(
#         adl_state_for_transition, period, education, model_specs
#     )

#     # Caregiving window restrictions
#     end_period_caregiving = model_specs["end_age_caregiving"] - model_specs["start_age"]
#     start_period_caregiving = model_specs["start_period_caregiving"]

#     # Restrict care demand to the caregiving window (boolean indicator)
#     in_caregiving_window = (period >= start_period_caregiving - 1) * (
#         period < end_period_caregiving
#     )

#     # Calculate probabilities for each state:
#     # State 0: NO_CARE_DEMAND_DEAD (mother dies or is already dead)
#     p_dead = death_prob

#     # State 1: NO_CARE_DEMAND_ALIVE (mother is alive, no ADL, outside caregiving window)
#     # This includes: alive * (ADL 0 OR (ADL 1/2/3 but outside caregiving window))
#     p_no_care_alive = alive_prob * (
#         prob_adl[0] + (1 - in_caregiving_window) * (prob_adl[1] + prob_adl[2])
#     )

#     # State 2: CARE_DEMAND_LIGHT (mother is alive, ADL 1, in caregiving window)
#     p_light = alive_prob * prob_adl[1] * in_caregiving_window

#     # State 3: CARE_DEMAND_INTENSIVE (mother is alive, ADL 2 or 3, in caregiving window)
#     p_intensive = alive_prob * prob_adl[2] * in_caregiving_window

#     # Ensure probabilities sum to 1
#     # Note: p_dead + p_no_care_alive + p_light + p_intensive should equal 1.0
#     # = death_prob + alive_prob * (prob_adl[0] + (1 - in_caregiving_window) * (prob_adl[1] + prob_adl[2]) + prob_adl[1] * in_caregiving_window + prob_adl[2] * in_caregiving_window)
#     # = death_prob + alive_prob * (prob_adl[0] + prob_adl[1] + prob_adl[2])
#     # = death_prob + alive_prob * 1.0
#     # = death_prob + (1 - death_prob) = 1.0

#     probs = jnp.array([p_dead, p_no_care_alive, p_light, p_intensive])

#     # If mother was already dead, return dead state with certainty
#     return jnp.where(already_dead, dead_probs, probs)


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
