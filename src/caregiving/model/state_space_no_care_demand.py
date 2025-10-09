"""State space for the model without care demand."""

import jax
import jax.numpy as jnp
import numpy as np

from caregiving.model.shared import (
    SEX,
    is_alive,
    is_dead,
)
from caregiving.model.shared_no_care_demand import (
    NOT_WORKING_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
    is_full_time,
    is_part_time,
    is_retired,
    is_unemployed,
)
from caregiving.model.state_space import (
    apply_retirement_constraint_for_SRA,
    calc_experience_years_for_pension_adjustment,
    construct_experience_years,
)


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_endogenous_state": next_period_endogenous_state,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
    }


# =====================================================================================
# State transitions (no care demand and no caregiving choices)
# =====================================================================================


def next_period_endogenous_state(
    period,
    choice,
    lagged_choice,
    already_retired,
):
    is_already_retired = is_retired(lagged_choice) & is_retired(choice)

    states_already_retired = {
        "period": period + 1,
        "lagged_choice": choice,
        "already_retired": jnp.ones_like(already_retired),
    }
    states_not_yet_retired = {
        "period": period + 1,
        "lagged_choice": choice,
        "already_retired": jnp.zeros_like(already_retired),
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
    }


def sparsity_condition(  # noqa: PLR0911, PLR0912
    period,
    lagged_choice,
    already_retired,
    education,
    has_sister,
    health,
    partner_state,
    job_offer,
    options,
):
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    min_ret_age_state_space = options["min_ret_age"]

    SRA_pol_state = options["min_SRA"]

    last_period = options["n_periods"] - 1

    age = start_age + period

    # Basic feasibility constraints without caregiving
    if (age <= min_ret_age_state_space) & (is_retired(lagged_choice)):
        return False
    elif (age <= min_ret_age_state_space + 1) & (already_retired == 1):
        return False
    # Cannot be unemployed after SRA
    elif (age > SRA_pol_state) & (is_unemployed(lagged_choice)):
        return False
    elif (not is_retired(lagged_choice)) & (already_retired == 1):
        return False
    # After the maximum retirement age, you must be retired.
    elif (age > max_ret_age) & (not is_retired(lagged_choice)) & (is_alive(health)):
        return False
    elif (age > max_ret_age + 1) & (already_retired != 1):
        return False
    else:
        # Lead death states to terminal/death states in last period with job offer 0
        if is_dead(health):
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
                "job_offer": 0,
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
                "job_offer": 0,
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
                "job_offer": 0,
            }
            return state_proxy
        else:
            return True


def state_specific_choice_set(  # noqa: PLR0911, PLR0912
    period, lagged_choice, job_offer, health, options
):
    age = period + options["start_age"]
    SRA_pol_state = options["min_SRA"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA_pol_state, options)

    if is_dead(health):
        return RETIREMENT_NO_CARE_DEMAND
    # Retirement is absorbing
    elif is_retired(lagged_choice):
        return RETIREMENT_NO_CARE_DEMAND
    # Check if the person is not in the voluntary retirement range.
    elif age < min_ret_age_pol_state:
        if job_offer == 0:
            return UNEMPLOYED_NO_CARE_DEMAND
        else:
            return WORK_NO_CARE_DEMAND
    # Person must be retired
    elif age >= options["max_ret_age"]:
        return RETIREMENT_NO_CARE_DEMAND
    # Person is in the voluntary retirement range.
    else:
        if age >= SRA_pol_state:
            if job_offer == 0:
                return RETIREMENT_NO_CARE_DEMAND
            else:
                return WORK_NO_CARE_DEMAND
        else:
            if job_offer == 0:
                # Choose unemployment or retirement
                return NOT_WORKING_NO_CARE_DEMAND
            else:
                return WORK_NO_CARE_DEMAND


def get_next_period_experience(
    period, lagged_choice, already_retired, education, experience, options
):
    """Update experience based on lagged choice and period."""
    sex = SEX

    exp_years_last_period = construct_experience_years(
        experience=experience,
        period=period - 1,
        max_exp_diffs_per_period=options["max_exp_diffs_per_period"],
    )

    # Update if working part or full time
    exp_update = (
        is_full_time(lagged_choice)
        + is_part_time(lagged_choice) * options["exp_increase_part_time"]
    )
    exp_new_period = exp_years_last_period + exp_update

    # If retired, then we update experience according to the deduction function
    fresh_retired = (already_retired == 0) & is_retired(lagged_choice)

    # Calculate experience with early retirement penalty
    experience_years_with_penalty = calc_experience_years_for_pension_adjustment(
        period=period,
        experience_years=exp_years_last_period,
        sex=sex,
        education=education,
        options=options,
    )
    # Update if fresh retired
    exp_new_period = jax.lax.select(
        fresh_retired, experience_years_with_penalty, exp_new_period
    )
    return (1 / (period + options["max_exp_diffs_per_period"][period])) * exp_new_period
