import jax
import jax.numpy as jnp
import numpy as np

from caregiving.model.shared import (  # BAD_HEALTH,; CARE_AND_NO_CARE,; FORMAL_CARE,; FORMAL_CARE_AND_NO_CARE,; NO_CARE,; is_formal_care,
    AGE_50,
    ALL,
    FULL_TIME_AND_NO_WORK,
    NO_RETIREMENT,
    NO_WORK,
    NOT_WORKING,
    PART_TIME_AND_NO_WORK,
    RETIREMENT,
    UNEMPLOYED,
    WORK_AND_NO_WORK,
    WORK_AND_UNEMPLOYED,
    is_full_time,
    is_part_time,
    is_retired,
)
from caregiving.model.wealth_and_budget.pensions import (
    calc_experience_for_total_pension_points,
    calc_total_pension_points,
)

# def create_state_space_functions():
#     return {
#         "get_next_period_state": update_endog_state,
#         "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
#     }


# ==============================================================================
# State transition
# ==============================================================================


# def get_state_specific_feasible_choice_set(
#     period,
#     lagged_choice,
#     part_time_offer,
#     full_time_offer,
#     mother_health,
#     options,
# ):
#     """Get feasible choice set for current parent state.

#     # if ((mother_alive == 1) & (mother_health in [MEDIUM_HEALTH, BAD_HEALTH])) | ( #
#     (father_alive == 1) & (father_health in [MEDIUM_HEALTH, BAD_HEALTH]) # ):

#     if experience == options["experience_cap"]:     feasible_choice_set = [i for i in
#     feasible_choice_set if i in NO_WORK]

#     # elif period + options["start_age"] > options["retirement_age"]: #
#     feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

#     # elif (age > EARLY_RETIREMENT_AGE) & lagged_choice in NO_WORK: #
#     feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

#     """
#     age = options["start_age"] + period

#     feasible_choice_set = np.arange(options["n_choices"])

#     _feasible_choice_set_all = list(np.arange(options["n_choices"]))

#     # Can only provide care if mother is alive and in bad health
#     if (mother_health == BAD_HEALTH) & (age >= AGE_50):
#         feasible_choice_set = [
#             i for i in _feasible_choice_set_all if i in CARE_AND_NO_CARE
#         ]

#         # Absorbing nursing home
#         if is_formal_care(lagged_choice):
#             feasible_choice_set = [i for i in feasible_choice_set if i in FORMAL_CARE]
#     else:
#         feasible_choice_set = [i for i in _feasible_choice_set_all if i in NO_CARE]

#     if age < options["min_ret_age"]:
#         feasible_choice_set = [i for i in feasible_choice_set if i in NO_RETIREMENT]

#     if age >= options["max_ret_age"]:
#         feasible_choice_set = RETIREMENT
#     elif is_retired(lagged_choice):
#         feasible_choice_set = [i for i in feasible_choice_set if i in RETIREMENT]
#     elif (full_time_offer == 0) & (part_time_offer == 1):
#         feasible_choice_set = [
#             i for i in feasible_choice_set if i in PART_TIME_AND_NO_WORK
#         ]
#     elif (full_time_offer == 1) & (part_time_offer == 0):
#         feasible_choice_set = [
#             i for i in feasible_choice_set if i in FULL_TIME_AND_NO_WORK
#         ]
#     elif (full_time_offer == 1) & (part_time_offer == 1):
#         feasible_choice_set = [i for i in feasible_choice_set if i in WORK_AND_NO_WORK]
#     else:
#         feasible_choice_set = [i for i in feasible_choice_set if i in NOT_WORKING]

#     return np.array(feasible_choice_set)


# def update_endog_state(
#     period,
#     choice,
#     experience,
#     # high_educ,
#     options,
# ):
#     """Update endogenous state variables.

#     next_state["mother_age"] = options["mother_min_age"] + mother_age + 1
#     next_state["father_age"] = options["father_min_age"] + father_age + 1


#     experience_cap: 15 # maximum of exp accumulated, see Adda et al (2017)
#     Returns to experience are flat after 15 years of experience.

#     below_exp_cap = experience < options["experience_cap"]
#     experience_current = below_exp_cap * is_working(choice)
#     next_state["experience"] = experience + experience_current

#     """
#     next_state = {}

#     next_state["period"] = period + 1
#     next_state["lagged_choice"] = choice
#     # next_state["high_educ"] = high_educ # noqa: ERA001

#     below_exp_cap_part = experience + 1 < options["experience_cap"]
#     below_exp_cap_full = experience + 2 < options["experience_cap"]
#     experience_part_time = 1 * below_exp_cap_part * is_part_time(choice)
#     experience_full_time = 2 * below_exp_cap_full * is_full_time(choice)
#     next_state["experience"] = experience + experience_part_time + experience_full_time

#     return next_state


# def sparsity_condition(
#     period,
#     lagged_choice,
#     experience,
#     options,
# ):
#     age = options["start_age"] + period

#     max_init_experience = options["max_init_experience"]

#     cond = True

#     # You cannot retire before the earliest retirement age
#     if (
#         (age <= options["min_ret_age"]) & is_retired(lagged_choice)
#         or (age > options["max_ret_age"]) & (is_retired(lagged_choice) is False)
#         or (
#             (is_full_time(lagged_choice) is False)
#             & (is_part_time(lagged_choice) is False)
#         )
#         & (period + max_init_experience == experience)
#         & (period > 0)
#         | (experience > options["experience_cap"])
#     ):
#         cond = False

#     if (age >= options["max_ret_age"] + 1) & (is_retired(lagged_choice) is False):
#         cond = False

#     return cond


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
    }


def sparsity_condition(
    period,
    lagged_choice,
    education,
    partner_state,
    options,
):
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    min_ret_age_state_space = options["min_ret_age"]
    # Generate last period, because only here are death states

    age = start_age + period

    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age_state_space) & (lagged_choice == 0):
        return False
    # After the maximum retirement age, you must be retired.
    elif (age > max_ret_age) & (lagged_choice != 0):
        return False
    else:
        # Now turn to the states, where it is decided by the value of an exogenous
        # state if it is valid or not. For invalid states we provide a proxy child state
        if (age <= max_ret_age + 1) and (lagged_choice == 0):
            # If retirement is already chosen we proxy all states to job offer 0.
            # Until age max_ret_age + 1 the individual could also be freshly retired
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "education": education,
                "partner_state": partner_state,
                "job_offer": 0,
            }
            return state_proxy
        elif age > max_ret_age + 1:
            # If age is larger than max_ret_age + 1, the individual can only be longer retired.
            # We can degenerate the policy state too
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "education": education,
                "partner_state": partner_state,
                "job_offer": 0,
            }
            return state_proxy
        else:
            return True


def state_specific_choice_set(period, lagged_choice, job_offer, options):
    age = period + options["start_age"]

    # Retirement is absorbing
    if lagged_choice == 0:
        return RETIREMENT
    # Check if the person is not in the voluntary retirement range.
    elif age < options["min_ret_age"]:
        if job_offer == 0:
            return UNEMPLOYED
        else:
            return WORK_AND_UNEMPLOYED
    elif age >= options["max_ret_age"]:
        return RETIREMENT
    else:
        # if age >= SRA_pol_state:
        #     if job_offer == 0:
        #         return RETIREMENT
        #     else:
        #         return WORK_AND_RETIREMENT
        # else:
        if job_offer == 0:
            return NO_WORK
        else:
            return ALL


def get_next_period_experience(
    period, lagged_choice, policy_state, sex, education, experience, informed, options
):
    """Update experience based on lagged choice and period."""
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

    # # If retired, then we update experience according to the deduction function
    # fresh_retired = is_retired(lagged_choice)

    # # Calculate experience with early retirement penalty
    # experience_years_with_penalty = calc_experience_years_for_pension_adjustment(
    #     period=period,
    #     experience_years=exp_years_last_period,
    #     sex=sex,
    #     education=education,
    #     policy_state=policy_state,
    #     informed=informed,
    #     options=options,
    # )

    # # Update if fresh retired
    # exp_new_period = jax.lax.select(
    #     fresh_retired, experience_years_with_penalty, exp_new_period
    # )

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


def construct_experience_years(experience, period, max_exp_diffs_per_period):
    """Experience and period can also be arrays."""
    return experience * (period + max_exp_diffs_per_period[period])
