import jax
import jax.numpy as jnp
import numpy as np

from caregiving.model.shared import (  # BAD_HEALTH,; CARE_AND_NO_CARE,; FORMAL_CARE,; FORMAL_CARE_AND_NO_CARE,; NO_CARE,; is_formal_care,
    AGE_50,
    ALL,
    FULL_TIME_AND_NO_WORK,
    NO_RETIREMENT,
    NOT_WORKING,
    PART_TIME_AND_NO_WORK,
    RETIREMENT,
    SEX,
    UNEMPLOYED,
    WORK_AND_NO_WORK,
    WORK_AND_UNEMPLOYED,
    is_alive,
    is_dead,
    is_full_time,
    is_part_time,
    is_retired,
)
from caregiving.model.wealth_and_budget.pensions import (
    calc_experience_for_total_pension_points,
    calc_total_pension_points,
)


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_endogenous_state": next_period_endogenous_state,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
    }


# =====================================================================================
# State transitions
# =====================================================================================


def next_period_endogenous_state(period, choice, lagged_choice, already_retired):
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

    # return jax.lax.select(
    #     is_already_retired,
    #     states_already_retired,
    #     states_not_yet_retired,
    # )

    # return (
    #     is_already_retired * states_already_retired
    #     + (1 - is_already_retired) * states_not_yet_retired
    # )
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


def sparsity_condition(  # noqa: PLR0911
    period,
    lagged_choice,
    already_retired,
    education,
    health,
    partner_state,
    options,
):
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    min_ret_age_state_space = options["min_ret_age"]

    # Generate last period, because only here are death states
    last_period = options["n_periods"] - 1

    age = start_age + period

    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age_state_space) & (is_retired(lagged_choice)):
        return False
    elif (age <= min_ret_age_state_space + 1) & (already_retired == 1):
        return False
    elif (not is_retired(lagged_choice)) & (already_retired == 1):
        return False
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
            state_proxy = {
                "period": last_period,
                "lagged_choice": 0,
                "already_retired": 1,
                "education": education,
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
                "health": health,
                "partner_state": partner_state,
                "job_offer": 0,
            }
            return state_proxy
        else:
            return True


def state_specific_choice_set(  # noqa: PLR0911
    period, lagged_choice, job_offer, health, options
):
    age = period + options["start_age"]

    if is_dead(health):
        return np.array([0])
    # Retirement is absorbing
    elif lagged_choice == 0:
        return RETIREMENT
    # Check if the person is not in the voluntary retirement range.
    elif age < options["min_ret_age"]:
        if job_offer == 0:
            return UNEMPLOYED
        else:
            return WORK_AND_UNEMPLOYED
    # Persom must retire
    elif age >= options["max_ret_age"]:
        return RETIREMENT
    # Person is in the voluntary retirement range.
    else:
        # if age >= SRA_pol_state:
        #     if job_offer == 0:
        #         return RETIREMENT
        #     else:
        #         return WORK_AND_RETIREMENT
        # else:
        if job_offer == 0:
            # Choose unemployment or retirement
            return NOT_WORKING
        else:
            return ALL


# def get_next_period_experience(period, lagged_choice, experience, options):
#     """Update experience based on lagged choice and period."""
#     exp_years_last_period = construct_experience_years(
#         experience=experience,
#         period=period - 1,
#         max_exp_diffs_per_period=options["max_exp_diffs_per_period"],
#     )

#     # Update if working part or full time
#     exp_update = (
#         is_full_time(lagged_choice)
#         + is_part_time(lagged_choice) * options["exp_increase_part_time"]
#     )
#     exp_new_period = exp_years_last_period + exp_update

#     # # If retired, then we update experience according to the deduction function
#     # fresh_retired = is_retired(lagged_choice)

#     # # Calculate experience with early retirement penalty
#     # experience_years_with_penalty = calc_experience_years_for_pension_adjustment(
#     #     period=period,
#     #     experience_years=exp_years_last_period,
#     #     sex=sex,
#     #     education=education,
#     #     policy_state=policy_state,
#     #     informed=informed,
#     #     options=options,
#     # )

#     # # Update if fresh retired
#     # exp_new_period = jax.lax.select(
#     #     fresh_retired, experience_years_with_penalty, exp_new_period
#     # )

#     return (1 / (period + options["max_exp_diffs_per_period"][period])) * exp_new_period


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


# def calc_experience_years_for_pension_adjustment(
#     period, sex, experience_years, education, options
# ):
#     """Calculate the reduced experience with early retirement penalty."""
#     # retirement age is last periods age
#     age = options["start_age"] + period - 1

#     total_pension_points = calc_total_pension_points(
#         education=education,
#         experience_years=experience_years,
#         sex=sex,
#         options=options,
#     )

#     # Select penalty depending on age difference
#     pension_factor = (
#         1 - (age - options["min_SRA"]) * options["early_retirement_penalty"]
#     )
#     adjusted_pension_points = pension_factor * total_pension_points

#     reduced_experience_years = calc_experience_for_total_pension_points(
#         total_pension_points=adjusted_pension_points,
#         sex=sex,
#         education=education,
#         options=options,
#     )
#     return reduced_experience_years


def calc_experience_years_for_pension_adjustment(
    period, sex, experience_years, education, options
):
    """Calculate the reduced experience with early retirement penalty."""
    total_pension_points = calc_total_pension_points(
        education=education,
        experience_years=experience_years,
        sex=sex,
        options=options,
    )
    # retirement age is last periods age
    actual_retirement_age = options["start_age"] + period - 1
    # SRA at retirement, difference to actual retirement age and boolean for early retirement
    SRA_at_retirement = options["min_SRA"]
    retirement_age_difference = jnp.abs(SRA_at_retirement - actual_retirement_age)
    early_retired_bool = actual_retirement_age < SRA_at_retirement

    # deduction factor for early  retirement
    early_retirement_penalty_informed = options["early_retirement_penalty"]
    early_retirement_penalty = (
        1 - early_retirement_penalty_informed * retirement_age_difference
    )

    # Total bonus for late retirement
    late_retirement_bonus = 1 + (
        options["late_retirement_bonus"] * retirement_age_difference
    )

    # Select bonus or penalty depending on age difference
    pension_factor = jax.lax.select(
        early_retired_bool, early_retirement_penalty, late_retirement_bonus
    )

    adjusted_pension_points = pension_factor * total_pension_points
    reduced_experience_years = calc_experience_for_total_pension_points(
        total_pension_points=adjusted_pension_points,
        sex=sex,
        education=education,
        options=options,
    )
    return reduced_experience_years


def construct_experience_years(experience, period, max_exp_diffs_per_period):
    """Experience and period can also be arrays."""
    return experience * (period + max_exp_diffs_per_period[period])
