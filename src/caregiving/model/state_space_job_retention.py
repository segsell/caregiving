"""State space for the job retention counterfactual model."""

import jax.numpy as jnp

from caregiving.model.shared import (
    PARENT_DEAD,
    is_alive,
    is_dead,
    is_informal_care,
    is_retired,
    is_unemployed,
    is_working,
)
from caregiving.model.state_space import (
    get_next_period_experience,
    state_specific_choice_set_with_caregiving,
)


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set_with_caregiving,
        "next_period_endogenous_state": next_period_endogenous_state_with_job_retention,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition_with_job_retention,
    }


# =====================================================================================
# State transitions
# =====================================================================================


def next_period_endogenous_state_with_job_retention(
    period,
    choice,
    lagged_choice,
    already_retired,
    job_before_caregiving,
):
    """Update endogenous states including job_before_caregiving tracking."""

    # Standard retirement tracking
    is_already_retired = is_retired(lagged_choice) & is_retired(choice)

    current_caregiver = is_informal_care(choice)
    previous_caregiver = is_informal_care(lagged_choice)

    had_job_before_caregiving = jnp.where(
        # If not caregiving, reset to 0
        ~current_caregiver,
        0,
        jnp.where(
            # If just started caregiving and was working, set to 1 (had job before caregiving)
            ~previous_caregiver & current_caregiver & is_working(lagged_choice),
            1,
            # If continuing caregiving, keep the same job_before_caregiving state
            job_before_caregiving,
        ),
    )

    states_already_retired = {
        "period": period + 1,
        "lagged_choice": choice,
        "already_retired": jnp.ones_like(already_retired),
        "job_before_caregiving": had_job_before_caregiving,
    }
    states_not_yet_retired = {
        "period": period + 1,
        "lagged_choice": choice,
        "already_retired": jnp.zeros_like(already_retired),
        "job_before_caregiving": had_job_before_caregiving,
    }

    return {
        "period": jnp.where(
            is_already_retired,
            states_already_retired["period"],
            states_not_yet_retired["period"],
        ),
        "lagged_choice": jnp.where(
            is_already_retired,
            states_already_retired["lagged_choice"],
            states_not_yet_retired["lagged_choice"],
        ),
        "already_retired": jnp.where(
            is_already_retired,
            states_already_retired["already_retired"],
            states_not_yet_retired["already_retired"],
        ),
        "job_before_caregiving": had_job_before_caregiving,
    }


def sparsity_condition_with_job_retention(  # noqa: PLR0911, PLR0912
    period,
    lagged_choice,
    already_retired,
    education,
    has_sister,
    health,
    partner_state,
    mother_health,
    care_demand,
    job_before_caregiving,
    job_offer,
    options,
):
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    min_ret_age_state_space = options["min_ret_age"]

    SRA_pol_state = options["min_SRA"]  # + policy_state

    # Generate last period, because only here are death states
    last_period = options["n_periods"] - 1

    age = start_age + period

    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age_state_space) & (is_retired(lagged_choice)):
        return False
    elif (age <= min_ret_age_state_space + 1) & (already_retired == 1):
        return False
    # elif (age >= options["min_SRA_baseline"] + 1) & (is_unemployed(lagged_choice)):
    elif (age > SRA_pol_state) & (is_unemployed(lagged_choice)):
        return False
    elif (not is_retired(lagged_choice)) & (already_retired == 1):
        return False
    # ================================================================================
    elif (age > options["end_age_msm"] + 1) & is_informal_care(lagged_choice):
        return False
    # # ================================================================================
    # elif (not care_demand == 0) & (job_before_caregiving == 1):
    #     return False
    # # ================================================================================
    # elif (not is_informal_care(lagged_choice)) & (job_before_caregiving == 1):
    #     return False
    # # ================================================================================
    # After the maximum retirement age, you must be retired.
    elif (age > max_ret_age) & (not is_retired(lagged_choice)) & (is_alive(health)):
        return False
    elif (age > max_ret_age + 1) & (already_retired != 1):
        return False
    else:
        # Now turn to the states, where it is decided by the value of an exogenous
        # state if it is valid or not. For invalid states we provide a proxy child state
        if is_dead(health):
            # Lead all states with death to last period death states
            # with job offer 0 (not relevant for bequest). You could be in principle
            # die upon retirement for which we need informed and policy state
            if period == last_period:
                return True

            state_proxy = {
                "period": last_period,
                "lagged_choice": 0,
                "already_retired": 1,
                "education": education,
                "has_sister": has_sister,
                "health": health,
                "partner_state": partner_state,
                "mother_health": PARENT_DEAD,
                "care_demand": 0,
                "job_offer": 0,
                "job_before_caregiving": job_before_caregiving,
            }
            return state_proxy
        elif mother_health == PARENT_DEAD:
            # If mother is dead, no care demand and supply
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "education": education,
                "has_sister": has_sister,
                "health": health,
                "partner_state": partner_state,
                "mother_health": mother_health,
                "care_demand": 0,
                "job_offer": job_offer,
                "job_before_caregiving": job_before_caregiving,
            }
            return state_proxy
        elif (age <= max_ret_age + 1) and is_retired(lagged_choice):
            # If retirement is already chosen we proxy all states to job offer 0.
            # Until age max_ret_age + 1 the individual could also be freshly retired
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "education": education,
                "has_sister": has_sister,
                "health": health,
                "partner_state": partner_state,
                "mother_health": mother_health,
                "care_demand": care_demand,
                "job_offer": 0,
                "job_before_caregiving": job_before_caregiving,
            }
            return state_proxy
        elif age > max_ret_age + 1:
            # If age is larger than max_ret_age + 1, the individual can only be
            # longer retired.
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "education": education,
                "has_sister": has_sister,
                "health": health,
                "partner_state": partner_state,
                "mother_health": PARENT_DEAD,
                "care_demand": 0,
                "job_offer": 0,
                "job_before_caregiving": job_before_caregiving,
            }
            return state_proxy
        # elif period < 10:
        #     # If agent before age 40, no care demand and supply
        #     state_proxy = {
        #         "period": period,
        #         "lagged_choice": lagged_choice,
        #         "already_retired": already_retired,
        #         "education": education,
        #         "has_sister": has_sister,
        #         "health": health,
        #         "partner_state": partner_state,
        #         "mother_health": mother_health,
        #         "care_demand": 0,
        #         "job_offer": job_offer,
        #     }
        #     return state_proxy

        else:
            return True
