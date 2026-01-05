"""State space for the caregiving leave-with-job-retention counterfactual model.

For now, this is identical to the job retention state space, but kept in a
separate module so we can later add caregiving-leave-specific logic without
touching the original job-retention specification.
"""

import jax
import jax.numpy as jnp

from caregiving.model.shared import (
    NO_CARE_DEMAND,
    PARENT_LONGER_DEAD,
    PARENT_RECENTLY_DEAD,
    SEX,
    is_alive,
    is_dead,
    is_formal_care,
    is_full_time,
    is_informal_care,
    is_part_time,
    is_retired,
    is_unemployed,
    is_working,
)
from caregiving.model.state_space import (
    state_specific_choice_set_with_caregiving,
)
from caregiving.model.experience_baseline_model import get_next_period_experience


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set_with_caregiving,
        "next_period_deterministic_state": (
            next_period_deterministic_state_with_job_retention
        ),
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition_with_job_retention,
    }


# =====================================================================================
# State transitions
# =====================================================================================


def next_period_deterministic_state_with_job_retention(
    period,
    choice,
    lagged_choice,
    already_retired,
    job_before_caregiving,
):
    """Update endogenous states including job_before_caregiving tracking.

    job_before_caregiving takes three values:
    0: no job before caregiving
    1: part-time job before caregiving
    2: full-time job before caregiving
    """

    # Standard retirement tracking
    is_already_retired = is_retired(lagged_choice) & is_retired(choice)

    current_caregiver = is_informal_care(choice)
    previous_caregiver = is_informal_care(lagged_choice)

    # Detect first period of a caregiving spell
    just_started_care = ~previous_caregiver & current_caregiver

    # Update job_before_caregiving with three states:
    # 0 = no job before caregiving, 1 = PT before caregiving, 2 = FT before caregiving.
    had_job_before_caregiving = jnp.where(
        # If not caregiving, reset to 0 so that a future spell is re-evaluated
        ~current_caregiver,
        0,
        jnp.where(
            # If just started caregiving and was full-time, set to 2
            just_started_care & is_working(lagged_choice),
            jnp.where(
                # Distinguish PT vs FT among workers
                is_part_time(lagged_choice),
                1,
                2,
            ),
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
    health,
    partner_state,
    mother_adl,
    mother_dead,
    care_demand,
    job_before_caregiving,
    job_offer,
    caregiving_type,
    # gets_inheritance,
    model_specs,
):
    """Sparsity condition for the job retention counterfactual model."""
    start_age = model_specs["start_age"]
    max_ret_age = model_specs["max_ret_age"]
    min_ret_age_state_space = model_specs["min_ret_age"]

    start_age_caregiving = model_specs["start_age_caregiving"]
    end_age_caregiving = model_specs["end_age_caregiving"]

    SRA_pol_state = model_specs["min_SRA"]

    # Generate last period, because only here are death states
    last_period = model_specs["n_periods"] - 1

    age = start_age + period

    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age_state_space) & (is_retired(lagged_choice)):
        return False
    elif (age <= min_ret_age_state_space + 1) & (already_retired == 1):
        return False
    # elif (age >= model_specs["min_SRA_baseline"] + 1) & (is_unemployed(lagged_choice)):
    elif (age > SRA_pol_state) & (is_unemployed(lagged_choice)):
        return False
    elif (not is_retired(lagged_choice)) & (already_retired == 1):
        return False
    # Note: We allow (is_retired(lagged_choice)) & (already_retired == 0) because
    # this represents the period immediately after someone first retires
    # elif (is_retired(lagged_choice)) & (already_retired == 0):
    #     return False
    # ================================================================================
    # After the maximum retirement age, you must be retired.
    elif (age > max_ret_age) & (not is_retired(lagged_choice)) & (is_alive(health)):
        return False
    elif (age > max_ret_age + 1) & (already_retired != 1):
        return False
    # ================================================================================
    elif (age > end_age_caregiving + 1) & is_informal_care(lagged_choice):
        return False
    elif (age > end_age_caregiving + 1) & is_formal_care(lagged_choice):
        return False
    # ================================================================================
    elif (age <= start_age_caregiving) & (is_informal_care(lagged_choice)):
        return False
    elif (age <= start_age_caregiving) & (is_formal_care(lagged_choice)):
        return False
    # ================================================================================
    elif (caregiving_type == 0) & (is_informal_care(lagged_choice)):
        return False
    # ================================================================================
    elif (not is_informal_care(lagged_choice)) & (job_before_caregiving != 0):
        return False
    # ================================================================================
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
                "caregiving_type": caregiving_type,
                "health": health,
                "partner_state": partner_state,
                "mother_adl": 0,
                "mother_dead": PARENT_LONGER_DEAD,
                "care_demand": NO_CARE_DEMAND,
                # "gets_inheritance": 0,
                "job_offer": 0,
                "job_before_caregiving": 0,
            }
            return state_proxy
        # elif mother_dead == PARENT_RECENTLY_DEAD:
        #     # If mother recently died, no care demand and supply
        #     # Preserve mother_dead == 1 so inheritance can be calculated
        #     state_proxy = {
        #         "period": period,
        #         "lagged_choice": lagged_choice,
        #         "already_retired": already_retired,
        #         "education": education,
        #         "caregiving_type": caregiving_type,
        #         "health": health,
        #         "partner_state": partner_state,
        #         "mother_adl": 0,
        #         "mother_dead": PARENT_RECENTLY_DEAD,  # Preserve for inheritance calculation
        #         "care_demand": NO_CARE_DEMAND,
        #         "job_offer": job_offer,
        #         "job_before_caregiving": job_before_caregiving,
        #     }
        #     return state_proxy
        elif mother_dead == PARENT_LONGER_DEAD:
            # If mother is longer dead, no care demand and supply
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "education": education,
                "caregiving_type": caregiving_type,
                "health": health,
                "partner_state": partner_state,
                "mother_adl": 0,
                "mother_dead": PARENT_LONGER_DEAD,
                "care_demand": NO_CARE_DEMAND,
                # "gets_inheritance": 0,
                "job_offer": job_offer,
                # "job_before_caregiving": job_before_caregiving,  # No job before caregiving when no care
                "job_before_caregiving": 0,  # No job before caregiving when no care
            }
            return state_proxy
        # ================================================================================
        elif age > max_ret_age + 1:
            # If age is larger than max_ret_age + 1, the individual can only be
            # longer retired.
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": 1,
                "education": education,
                "caregiving_type": caregiving_type,
                "health": health,
                "partner_state": partner_state,
                "mother_adl": mother_adl,
                "mother_dead": mother_dead,
                "care_demand": care_demand,  # Outside caregiving window, no care demand
                # "gets_inheritance": gets_inheritance,
                "job_offer": 0,
                "job_before_caregiving": 0,
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
                "caregiving_type": caregiving_type,
                "health": health,
                "partner_state": partner_state,
                "mother_adl": mother_adl,
                "mother_dead": mother_dead,
                "care_demand": care_demand,
                # "gets_inheritance": gets_inheritance,
                "job_offer": 0,
                "job_before_caregiving": job_before_caregiving,
            }
            return state_proxy
        # elif age > model_specs["end_age_msm"] + 1:
        #     # If age is larger than max_ret_age + 1, the individual can only be
        #     # longer retired.
        #     state_proxy = {
        #         "period": period,
        #         "lagged_choice": lagged_choice,
        #         "already_retired": already_retired,
        #         "education": education,
        #         "health": health,
        #         "partner_state": partner_state,
        #         "mother_health": PARENT_DEAD,
        #         "care_demand": 0,
        #         "job_offer": 0,
        #     }
        #     return state_proxy
        # elif period < 10:
        #     # If agent before age 40, no care demand and supply
        #     state_proxy = {
        #         "period": period,
        #         "lagged_choice": lagged_choice,
        #         "already_retired": already_retired,
        #         "education": education,
        #         "health": health,
        #         "partner_state": partner_state,
        #         "mother_health": mother_health,
        #         "care_demand": 0,
        #         "job_offer": job_offer,
        #     }
        #     return state_proxy
        # Care demand cannot be light/intensive before the start period for caregiving
        elif age > end_age_caregiving + 1:
            # Proxy to state with no care demand (alive, outside window)
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "education": education,
                "caregiving_type": caregiving_type,
                "health": health,
                "partner_state": partner_state,
                "mother_adl": mother_adl,
                "mother_dead": mother_dead,
                "care_demand": NO_CARE_DEMAND,
                "job_offer": job_offer,
                # "gets_inheritance": gets_inheritance,
                "job_before_caregiving": job_before_caregiving,
            }
            return state_proxy
        elif age < start_age_caregiving:
            # Proxy to state with no care demand (alive, outside window)
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "education": education,
                "caregiving_type": caregiving_type,
                "health": health,
                "partner_state": partner_state,
                "mother_adl": mother_adl,
                "mother_dead": mother_dead,
                "care_demand": NO_CARE_DEMAND,
                "job_offer": job_offer,
                # "gets_inheritance": gets_inheritance,
                "job_before_caregiving": job_before_caregiving,
            }
            return state_proxy

        else:
            return True


# def get_next_period_experience(
#     period, lagged_choice, already_retired, education, experience, model_specs
# ):
#     """Update experience based on lagged choice and period."""
#     sex = SEX
#     informal_care = is_informal_care(lagged_choice)

#     exp_years_last_period = construct_experience_years(
#         experience=experience,
#         period=period - 1,
#         max_exp_diffs_per_period=model_specs["max_exp_diffs_per_period"],
#     )

#     # Update if working part or full time
#     # Full pension point for caregivers that work part time (vs half if not caring)
#     # exp_update = (
#     #     is_full_time(lagged_choice)
#     #     + is_part_time(lagged_choice)
#     #     * model_specs["exp_increase_part_time"]
#     #     * (1 - informal_care)
#     #     + is_part_time(lagged_choice) * informal_care
#     # )
#     exp_update = is_full_time(lagged_choice) + is_part_time(lagged_choice) * (
#         model_specs["exp_increase_part_time"] * (1 - informal_care) + informal_care
#     )
#     exp_new_period = exp_years_last_period + exp_update

#     # If retired, then we update experience according to the deduction function
#     fresh_retired = (already_retired == 0) & is_retired(lagged_choice)

#     # Calculate experience with early retirement penalty
#     experience_years_with_penalty = calc_experience_years_for_pension_adjustment(
#         period=period,
#         experience_years=exp_years_last_period,
#         sex=sex,
#         education=education,
#         model_specs=model_specs,
#     )
#     # Update if fresh retired
#     exp_new_period = jax.lax.select(
#         fresh_retired, experience_years_with_penalty, exp_new_period
#     )
#     return (
#         1 / (period + model_specs["max_exp_diffs_per_period"][period])
#     ) * exp_new_period


# # def calc_experience_years_for_pension_adjustment(
# #     period, sex, experience_years, education, options
# # ):
# #     """Calculate the reduced experience with early retirement penalty."""
# #     # retirement age is last periods age
# #     age = options["start_age"] + period - 1

# #     total_pension_points = calc_total_pension_points(
# #         education=education,
# #         experience_years=experience_years,
# #         sex=sex,
# #         options=options,
# #     )

# #     # Select penalty depending on age difference
# #     pension_factor = (
# #         1 - (age - options["min_SRA"]) * options["early_retirement_penalty"]
# #     )
# #     adjusted_pension_points = pension_factor * total_pension_points

# #     reduced_experience_years = calc_experience_for_total_pension_points(
# #         total_pension_points=adjusted_pension_points,
# #         sex=sex,
# #         education=education,
# #         options=options,
# #     )
# #     return reduced_experience_years


# def calc_experience_years_for_pension_adjustment(
#     period, sex, experience_years, education, model_specs
# ):
#     """Calculate the reduced experience with early retirement penalty."""
#     total_pension_points = calc_total_pension_points(
#         education=education,
#         experience_years=experience_years,
#         sex=sex,
#         model_specs=model_specs,
#     )
#     # retirement age is last periods age
#     actual_retirement_age = model_specs["start_age"] + period - 1
#     # SRA at retirement, difference to actual retirement age
#     # and boolean for early retirement
#     SRA_at_retirement = model_specs["min_SRA"]
#     retirement_age_difference = jnp.abs(SRA_at_retirement - actual_retirement_age)
#     early_retired_bool = actual_retirement_age < SRA_at_retirement

#     # deduction factor for early  retirement
#     early_retirement_penalty_informed = model_specs["early_retirement_penalty"]
#     early_retirement_penalty = (
#         1 - early_retirement_penalty_informed * retirement_age_difference
#     )

#     # Total bonus for late retirement
#     late_retirement_bonus = 1 + (
#         model_specs["late_retirement_bonus"] * retirement_age_difference
#     )

#     # Select bonus or penalty depending on age difference
#     pension_factor = jax.lax.select(
#         early_retired_bool, early_retirement_penalty, late_retirement_bonus
#     )
#     # pension_factor = late_retirement_bonus

#     adjusted_pension_points = pension_factor * total_pension_points
#     reduced_experience_years = calc_experience_for_total_pension_points(
#         total_pension_points=adjusted_pension_points,
#         sex=sex,
#         education=education,
#         model_specs=model_specs,
#     )
#     return reduced_experience_years


# def construct_experience_years(experience, period, max_exp_diffs_per_period):
#     """Experience and period can also be arrays."""
#     return experience * (period + max_exp_diffs_per_period[period])
