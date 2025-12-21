"""State space for the model with care demand and caregiving."""

import jax
import jax.numpy as jnp
import numpy as np

from caregiving.model.shared import (
    ALL,
    ALL_CARE,
    ALL_INTENSIVE_INFORMAL_OR_FORMAL,
    ALL_LIGHT_INFORMAL_OR_FORMAL,
    ALL_NO_CARE,
    ALL_NO_CARE_OR_FORMAL,
    ALL_NO_FORMAL_CARE,
    ALL_NO_INFORMAL_CARE,
    CARE_DEMAND_INTENSIVE,
    CARE_DEMAND_LIGHT,
    NO_CARE_DEMAND,
    FORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NOT_WORKING,
    NOT_WORKING_CARE,
    NOT_WORKING_INTENSIVE_INFORMAL_OR_FORMAL,
    NOT_WORKING_LIGHT_INFORMAL_OR_FORMAL,
    NOT_WORKING_NO_CARE,
    NOT_WORKING_NO_CARE_OR_FORMAL,
    NOT_WORKING_NO_FORMAL_CARE,
    NOT_WORKING_NO_INFORMAL_CARE,
    PARENT_DEAD,
    RETIREMENT,
    RETIREMENT_CARE,
    RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL,
    RETIREMENT_LIGHT_INFORMAL_OR_FORMAL,
    RETIREMENT_NO_CARE,
    RETIREMENT_NO_CARE_OR_FORMAL,
    RETIREMENT_NO_FORMAL_CARE,
    RETIREMENT_NO_INFORMAL_CARE,
    SEX,
    UNEMPLOYED,
    UNEMPLOYED_CARE,
    UNEMPLOYED_INTENSIVE_INFORMAL_OR_FORMAL,
    UNEMPLOYED_LIGHT_INFORMAL_OR_FORMAL,
    UNEMPLOYED_NO_CARE,
    UNEMPLOYED_NO_CARE_OR_FORMAL,
    UNEMPLOYED_NO_FORMAL_CARE,
    UNEMPLOYED_NO_INFORMAL_CARE,
    WORK_AND_RETIREMENT,
    WORK_AND_RETIREMENT_CARE,
    WORK_AND_RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL,
    WORK_AND_RETIREMENT_LIGHT_INFORMAL_OR_FORMAL,
    WORK_AND_RETIREMENT_NO_CARE,
    WORK_AND_RETIREMENT_NO_CARE_OR_FORMAL,
    WORK_AND_RETIREMENT_NO_FORMAL_CARE,
    WORK_AND_RETIREMENT_NO_INFORMAL_CARE,
    WORK_AND_UNEMPLOYED,
    WORK_AND_UNEMPLOYED_CARE,
    WORK_AND_UNEMPLOYED_INTENSIVE_INFORMAL_OR_FORMAL,
    WORK_AND_UNEMPLOYED_LIGHT_INFORMAL_OR_FORMAL,
    WORK_AND_UNEMPLOYED_NO_CARE,
    WORK_AND_UNEMPLOYED_NO_CARE_OR_FORMAL,
    WORK_AND_UNEMPLOYED_NO_FORMAL_CARE,
    WORK_AND_UNEMPLOYED_NO_INFORMAL_CARE,
    is_alive,
    is_dead,
    is_formal_care,
    is_full_time,
    is_informal_care,
    is_no_care_demand,
    is_part_time,
    is_retired,
    is_unemployed,
)
from caregiving.model.wealth_and_budget.pensions import (
    calc_experience_for_total_pension_points,
    calc_total_pension_points,
)


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set_with_caregiving,
        "next_period_deterministic_state": next_period_deterministic_state,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
    }


def create_state_space_functions_counterfactual():
    return {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_deterministic_state": next_period_deterministic_state,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
    }


# =====================================================================================
# State transitions
# =====================================================================================


def next_period_deterministic_state(
    period,
    choice,
    lagged_choice,
    already_retired,
):
    # is_already_retired = is_retired(choice)
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


def sparsity_condition(  # noqa: PLR0911, PLR0912
    period,
    lagged_choice,
    already_retired,
    education,
    health,
    partner_state,
    mother_adl,
    mother_dead,
    care_demand,
    job_offer,
    caregiving_type,
    model_specs,
):
    start_age = model_specs["start_age"]
    max_ret_age = model_specs["max_ret_age"]
    min_ret_age_state_space = model_specs["min_ret_age"]

    start_age_caregiving = model_specs["start_age_caregiving"]
    end_age_caregiving = model_specs["end_age_caregiving"]

    SRA_pol_state = model_specs["min_SRA"]  # + policy_state

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
    # # elif (age <= start_age_caregiving) & (lagged_care_demand == 1):
    # #     return False
    # elif (is_informal_care(lagged_choice)) & (lagged_care_demand == 0):
    #     return False
    # elif (is_formal_care(lagged_choice)) & (lagged_care_demand == 0):
    #     return False
    # elif (
    #     (caregiving_type == 1)
    #     & (lagged_care_demand == 1)
    #     & (is_no_care(lagged_choice))
    #     & (age > start_age_caregiving)
    #     & (age <= end_age_caregiving + 1)
    # ):
    #     return False
    # # ================================================================================
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
                "mother_dead": 1,
                "care_demand": NO_CARE_DEMAND,
                "job_offer": 0,
            }
            return state_proxy
        elif mother_dead == 1:
            # If mother is dead, no care demand and supply
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "education": education,
                "caregiving_type": caregiving_type,
                "health": health,
                "partner_state": partner_state,
                "mother_adl": 0,
                "mother_dead": 1,
                "care_demand": NO_CARE_DEMAND,
                "job_offer": job_offer,
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
                "caregiving_type": caregiving_type,
                "health": health,
                "partner_state": partner_state,
                "mother_adl": mother_adl,
                "mother_dead": mother_dead,
                "care_demand": care_demand,
                "job_offer": 0,
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
                "mother_dead": 1,
                "care_demand": NO_CARE_DEMAND,
                "job_offer": job_offer,
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
            }
            return state_proxy

        else:
            return True


def state_specific_choice_set(  # noqa: PLR0911, PLR0912
    period, lagged_choice, job_offer, health, model_specs
):
    age = period + model_specs["start_age"]
    SRA_pol_state = model_specs[
        "min_SRA"
    ]  # + policy_state  # * model_specs["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(
        SRA_pol_state, model_specs
    )

    if is_dead(health):
        return RETIREMENT_NO_CARE
    # Retirement is absorbing
    elif is_retired(lagged_choice):
        return RETIREMENT_NO_CARE
    # Check if the person is not in the voluntary retirement range.
    # elif age < model_specs["min_ret_age"]:
    elif age < min_ret_age_pol_state:
        if job_offer == 0:
            return UNEMPLOYED_NO_CARE
        else:
            return WORK_AND_UNEMPLOYED_NO_CARE
    # Person must be retired
    elif age >= model_specs["max_ret_age"]:
        return RETIREMENT_NO_CARE
    # Person is in the voluntary retirement range.
    else:
        if age >= SRA_pol_state:
            if job_offer == 0:
                return RETIREMENT_NO_CARE
            else:
                return WORK_AND_RETIREMENT_NO_CARE
        else:
            if job_offer == 0:
                # Choose unemployment or retirement
                return NOT_WORKING_NO_CARE
            else:
                return ALL_NO_CARE


def apply_retirement_constraint_for_SRA(SRA, model_specs):
    return np.maximum(SRA - model_specs["ret_years_before_SRA"], 63)


# def state_specific_choice_set_with_caregiving(  # noqa: PLR0911, PLR0912, PLR0915
#     period, lagged_choice, job_offer, health, caregiving_type, care_demand, model_specs
# ):
#     age = period + model_specs["start_age"]
#     start_age_caregiving = model_specs["start_age_caregiving"]
#     end_age_caregiving = model_specs["end_age_caregiving"]

#     SRA_pol_state = model_specs["min_SRA"]  # + policy_state  # * model_specs["SRA_grid_size"]
#     min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA_pol_state, model_specs)

#     # Light care demand (care_demand == 1) with caregiving_type == 1
#     # Agent can choose: LIGHT_INFORMAL_CARE or FORMAL_CARE
#     if (
#         (care_demand == CARE_DEMAND_LIGHT)
#         & (caregiving_type == 1)
#         & (age >= start_age_caregiving)
#         & (age <= end_age_caregiving)
#     ):
#         if is_dead(health):
#             return RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
#         elif is_retired(lagged_choice):
#             return RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
#         elif age < min_ret_age_pol_state:
#             if job_offer == 0:
#                 return UNEMPLOYED_LIGHT_INFORMAL_OR_FORMAL
#             else:
#                 return WORK_AND_UNEMPLOYED_LIGHT_INFORMAL_OR_FORMAL
#         elif age >= model_specs["max_ret_age"]:
#             return RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
#         else:
#             if age >= SRA_pol_state:
#                 if job_offer == 0:
#                     return RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
#                 else:
#                     return WORK_AND_RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
#             else:
#                 if job_offer == 0:
#                     return NOT_WORKING_LIGHT_INFORMAL_OR_FORMAL
#                 else:
#                     return ALL_LIGHT_INFORMAL_OR_FORMAL
#     # Intensive care demand (care_demand == 2) with caregiving_type == 1
#     # Agent can choose: INTENSIVE_INFORMAL_CARE or FORMAL_CARE
#     elif (
#         (care_demand == CARE_DEMAND_INTENSIVE)
#         & (caregiving_type == 1)
#         & (age >= start_age_caregiving)
#         & (age <= end_age_caregiving)
#     ):
#         if is_dead(health):
#             return RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
#         elif is_retired(lagged_choice):
#             return RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
#         elif age < min_ret_age_pol_state:
#             if job_offer == 0:
#                 return UNEMPLOYED_INTENSIVE_INFORMAL_OR_FORMAL
#             else:
#                 return WORK_AND_UNEMPLOYED_INTENSIVE_INFORMAL_OR_FORMAL
#         elif age >= model_specs["max_ret_age"]:
#             return RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
#         else:
#             if age >= SRA_pol_state:
#                 if job_offer == 0:
#                     return RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
#                 else:
#                     return WORK_AND_RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
#             else:
#                 if job_offer == 0:
#                     return NOT_WORKING_INTENSIVE_INFORMAL_OR_FORMAL
#                 else:
#                     return ALL_INTENSIVE_INFORMAL_OR_FORMAL
#     # Care demand (light or intensive) with caregiving_type == 0
#     # Agent can choose: NO_CARE or FORMAL_CARE
#     # (Same choice set for both light and intensive care demand)
#     elif (
#         ((care_demand == CARE_DEMAND_LIGHT) | (care_demand == CARE_DEMAND_INTENSIVE))
#         # (care_demand > 0)
#         & (caregiving_type == 0)
#         & (age >= start_age_caregiving)
#         & (age <= end_age_caregiving)
#     ):
#         if is_dead(health):
#             return RETIREMENT_NO_CARE_OR_FORMAL
#         elif is_retired(lagged_choice):
#             return RETIREMENT_NO_CARE_OR_FORMAL
#         elif age < min_ret_age_pol_state:
#             if job_offer == 0:
#                 return UNEMPLOYED_NO_CARE_OR_FORMAL
#             else:
#                 return WORK_AND_UNEMPLOYED_NO_CARE_OR_FORMAL
#         elif age >= model_specs["max_ret_age"]:
#             return RETIREMENT_NO_CARE_OR_FORMAL
#         else:
#             if age >= SRA_pol_state:
#                 if job_offer == 0:
#                     return RETIREMENT_NO_CARE_OR_FORMAL
#                 else:
#                     return WORK_AND_RETIREMENT_NO_CARE_OR_FORMAL
#             else:
#                 if job_offer == 0:
#                     return NOT_WORKING_NO_CARE_OR_FORMAL
#                 else:
#                     return ALL_NO_CARE_OR_FORMAL
#     else:  # if (care_demand == 0) | (age outside caregiving window):
#         # When care_demand == 0: No care is needed (neither informal nor formal care)
#         if is_dead(health):
#             return RETIREMENT_NO_CARE
#         # Retirement is absorbing
#         elif is_retired(lagged_choice):
#             return RETIREMENT_NO_CARE
#         # Check if the person is not in the voluntary retirement range.
#         elif age < min_ret_age_pol_state:  # min_ret_age: 63
#             if job_offer == 0:
#                 return UNEMPLOYED_NO_CARE
#             else:
#                 return WORK_AND_UNEMPLOYED_NO_CARE
#         # Person must retire
#         elif age >= model_specs["max_ret_age"]:
#             return RETIREMENT_NO_CARE
#         # Person is in the voluntary retirement range
#         else:
#             if age >= SRA_pol_state:  # min_SRA: 65
#                 if job_offer == 0:
#                     return RETIREMENT_NO_CARE  # Cannot choose unemployment after 65
#                 else:
#                     return WORK_AND_RETIREMENT_NO_CARE
#             else:
#                 if job_offer == 0:
#                     # Choose unemployment or retirement
#                     return NOT_WORKING_NO_CARE
#                 else:
#                     return ALL_NO_CARE


def state_specific_choice_set_with_caregiving(  # noqa: PLR0911, PLR0912, PLR0915
    period,
    lagged_choice,
    job_offer,
    health,
    caregiving_type,
    care_demand,
    mother_dead,
    model_specs,
):
    age = period + model_specs["start_age"]
    start_age_caregiving = model_specs["start_age_caregiving"]
    end_age_caregiving = model_specs["end_age_caregiving"]

    SRA_pol_state = model_specs[
        "min_SRA"
    ]  # + policy_state  # * model_specs["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(
        SRA_pol_state, model_specs
    )

    # Light care demand (care_demand == 1) with caregiving_type == 1
    # Agent can choose: LIGHT_INFORMAL_CARE or FORMAL_CARE

    # If mother is dead, always return NO_CARE choices regardless of care_demand
    if mother_dead == 1:
        if is_dead(health):
            return RETIREMENT_NO_CARE
        # Retirement is absorbing
        elif is_retired(lagged_choice):
            return RETIREMENT_NO_CARE
        # Check if the person is not in the voluntary retirement range.
        elif age < min_ret_age_pol_state:  # min_ret_age: 63
            if job_offer == 0:
                return UNEMPLOYED_NO_CARE
            else:
                return WORK_AND_UNEMPLOYED_NO_CARE
        # Person must retire
        elif age >= model_specs["max_ret_age"]:
            return RETIREMENT_NO_CARE
        # Person is in the voluntary retirement range
        else:
            if age >= SRA_pol_state:  # min_SRA: 65
                if job_offer == 0:
                    return RETIREMENT_NO_CARE  # Cannot choose unemployment after 65
                else:
                    return WORK_AND_RETIREMENT_NO_CARE
            else:
                if job_offer == 0:
                    # Choose unemployment or retirement
                    return NOT_WORKING_NO_CARE
                else:
                    return ALL_NO_CARE
    if caregiving_type == 1:
        if (
            is_no_care_demand(care_demand)
            | (age < start_age_caregiving)
            | (age > end_age_caregiving)
        ):
            if is_dead(health):
                return RETIREMENT_NO_CARE
            # Retirement is absorbing
            elif is_retired(lagged_choice):
                return RETIREMENT_NO_CARE
            # Check if the person is not in the voluntary retirement range.
            elif age < min_ret_age_pol_state:  # min_ret_age: 63
                if job_offer == 0:
                    return UNEMPLOYED_NO_CARE
                else:
                    return WORK_AND_UNEMPLOYED_NO_CARE
            # Person must retire
            elif age >= model_specs["max_ret_age"]:
                return RETIREMENT_NO_CARE
            # Person is in the voluntary retirement range
            else:
                if age >= SRA_pol_state:  # min_SRA: 65
                    if job_offer == 0:
                        return RETIREMENT_NO_CARE  # Cannot choose unemployment after 65
                    else:
                        return WORK_AND_RETIREMENT_NO_CARE
                else:
                    if job_offer == 0:
                        # Choose unemployment or retirement
                        return NOT_WORKING_NO_CARE
                    else:
                        return ALL_NO_CARE
        else:
            if care_demand == CARE_DEMAND_LIGHT:
                if is_dead(health):
                    return RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
                elif is_retired(lagged_choice):
                    return RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
                elif age < min_ret_age_pol_state:
                    if job_offer == 0:
                        return UNEMPLOYED_LIGHT_INFORMAL_OR_FORMAL
                    else:
                        return WORK_AND_UNEMPLOYED_LIGHT_INFORMAL_OR_FORMAL
                elif age >= model_specs["max_ret_age"]:
                    return RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
                else:
                    if age >= SRA_pol_state:
                        if job_offer == 0:
                            return RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
                        else:
                            return WORK_AND_RETIREMENT_LIGHT_INFORMAL_OR_FORMAL
                    else:
                        if job_offer == 0:
                            return NOT_WORKING_LIGHT_INFORMAL_OR_FORMAL
                        else:
                            return ALL_LIGHT_INFORMAL_OR_FORMAL
            else:
                if is_dead(health):
                    return RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
                elif is_retired(lagged_choice):
                    return RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
                elif age < min_ret_age_pol_state:
                    if job_offer == 0:
                        return UNEMPLOYED_INTENSIVE_INFORMAL_OR_FORMAL
                    else:
                        return WORK_AND_UNEMPLOYED_INTENSIVE_INFORMAL_OR_FORMAL
                elif age >= model_specs["max_ret_age"]:
                    return RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
                else:
                    if age >= SRA_pol_state:
                        if job_offer == 0:
                            return RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
                        else:
                            return WORK_AND_RETIREMENT_INTENSIVE_INFORMAL_OR_FORMAL
                    else:
                        if job_offer == 0:
                            return NOT_WORKING_INTENSIVE_INFORMAL_OR_FORMAL
                        else:
                            return ALL_INTENSIVE_INFORMAL_OR_FORMAL

    elif caregiving_type == 0:
        if (
            is_no_care_demand(care_demand)
            | (age < start_age_caregiving)
            | (age > end_age_caregiving)
        ):
            if is_dead(health):
                return RETIREMENT_NO_CARE
            # Retirement is absorbing
            elif is_retired(lagged_choice):
                return RETIREMENT_NO_CARE
            # Check if the person is not in the voluntary retirement range.
            elif age < min_ret_age_pol_state:  # min_ret_age: 63
                if job_offer == 0:
                    return UNEMPLOYED_NO_CARE
                else:
                    return WORK_AND_UNEMPLOYED_NO_CARE
            # Person must retire
            elif age >= model_specs["max_ret_age"]:
                return RETIREMENT_NO_CARE
            # Person is in the voluntary retirement range
            else:
                if age >= SRA_pol_state:  # min_SRA: 65
                    if job_offer == 0:
                        return RETIREMENT_NO_CARE  # Cannot choose unemployment after 65
                    else:
                        return WORK_AND_RETIREMENT_NO_CARE
                else:
                    if job_offer == 0:
                        # Choose unemployment or retirement
                        return NOT_WORKING_NO_CARE
                    else:
                        return ALL_NO_CARE
        else:
            if is_dead(health):
                return RETIREMENT_NO_CARE_OR_FORMAL
            elif is_retired(lagged_choice):
                return RETIREMENT_NO_CARE_OR_FORMAL
            elif age < min_ret_age_pol_state:
                if job_offer == 0:
                    return UNEMPLOYED_NO_CARE_OR_FORMAL
                else:
                    return WORK_AND_UNEMPLOYED_NO_CARE_OR_FORMAL
            elif age >= model_specs["max_ret_age"]:
                return RETIREMENT_NO_CARE_OR_FORMAL
            else:
                if age >= SRA_pol_state:
                    if job_offer == 0:
                        return RETIREMENT_NO_CARE_OR_FORMAL
                    else:
                        return WORK_AND_RETIREMENT_NO_CARE_OR_FORMAL
                else:
                    if job_offer == 0:
                        return NOT_WORKING_NO_CARE_OR_FORMAL
                    else:

                        return ALL_NO_CARE_OR_FORMAL
    # else:
    #     if is_dead(health):
    #         return RETIREMENT_NO_CARE
    #     # Retirement is absorbing
    #     elif is_retired(lagged_choice):
    #         return RETIREMENT_NO_CARE
    #     # Check if the person is not in the voluntary retirement range.
    #     elif age < min_ret_age_pol_state:  # min_ret_age: 63
    #         if job_offer == 0:
    #             return UNEMPLOYED_NO_CARE
    #         else:
    #             return WORK_AND_UNEMPLOYED_NO_CARE
    #     # Person must retire
    #     elif age >= model_specs["max_ret_age"]:
    #         return RETIREMENT_NO_CARE
    #     # Person is in the voluntary retirement range
    #     else:
    #         if age >= SRA_pol_state:  # min_SRA: 65
    #             if job_offer == 0:
    #                 return RETIREMENT_NO_CARE  # Cannot choose unemployment after 65
    #             else:
    #                 return WORK_AND_RETIREMENT_NO_CARE
    #         else:
    #             if job_offer == 0:
    #                 # Choose unemployment or retirement
    #                 return NOT_WORKING_NO_CARE
    #             else:
    #                 return ALL_NO_CARE


# def get_next_period_experience(period, lagged_choice, experience, model_specs):
#     """Update experience based on lagged choice and period."""
#     exp_years_last_period = construct_experience_years(
#         experience=experience,
#         period=period - 1,
#         max_exp_diffs_per_period=model_specs["max_exp_diffs_per_period"],
#     )

#     # Update if working part or full time
#     exp_update = (
#         is_full_time(lagged_choice)
#         + is_part_time(lagged_choice) * model_specs["exp_increase_part_time"]
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
#     #     model_specs=model_specs,
#     # )

#     # # Update if fresh retired
#     # exp_new_period = jax.lax.select(
#     #     fresh_retired, experience_years_with_penalty, exp_new_period
#     # )

#     return (1 / (period + model_specs["max_exp_diffs_per_period"][period])) * exp_new_period


def get_next_period_experience(
    period, lagged_choice, already_retired, education, experience, model_specs
):
    """Update experience based on lagged choice and period."""
    sex = SEX

    exp_years_last_period = construct_experience_years(
        experience=experience,
        period=period - 1,
        max_exp_diffs_per_period=model_specs["max_exp_diffs_per_period"],
    )

    # Update if working part or full time
    exp_update = (
        is_full_time(lagged_choice)
        + is_part_time(lagged_choice) * model_specs["exp_increase_part_time"]
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
        model_specs=model_specs,
    )
    # Update if fresh retired
    exp_new_period = jax.lax.select(
        fresh_retired, experience_years_with_penalty, exp_new_period
    )
    return (
        1 / (period + model_specs["max_exp_diffs_per_period"][period])
    ) * exp_new_period


# def calc_experience_years_for_pension_adjustment(
#     period, sex, experience_years, education, model_specs
# ):
#     """Calculate the reduced experience with early retirement penalty."""
#     # retirement age is last periods age
#     age = model_specs["start_age"] + period - 1

#     total_pension_points = calc_total_pension_points(
#         education=education,
#         experience_years=experience_years,
#         sex=sex,
#         model_specs=model_specs,
#     )

#     # Select penalty depending on age difference
#     pension_factor = (
#         1 - (age - model_specs["min_SRA"]) * model_specs["early_retirement_penalty"]
#     )
#     adjusted_pension_points = pension_factor * total_pension_points

#     reduced_experience_years = calc_experience_for_total_pension_points(
#         total_pension_points=adjusted_pension_points,
#         sex=sex,
#         education=education,
#         model_specs=model_specs,
#     )
#     return reduced_experience_years


def calc_experience_years_for_pension_adjustment(
    period, sex, experience_years, education, model_specs
):
    """Calculate the reduced experience with early retirement penalty."""
    total_pension_points = calc_total_pension_points(
        education=education,
        experience_years=experience_years,
        sex=sex,
        model_specs=model_specs,
    )
    # retirement age is last periods age
    actual_retirement_age = model_specs["start_age"] + period - 1
    # SRA at retirement, difference to actual retirement age and boolean for early retirement
    # SRA_at_retirement = model_specs["min_SRA"]
    SRA_at_retirement = model_specs["min_SRA"]
    retirement_age_difference = jnp.abs(SRA_at_retirement - actual_retirement_age)
    early_retired_bool = actual_retirement_age < SRA_at_retirement

    # deduction factor for early  retirement
    early_retirement_penalty_informed = model_specs["early_retirement_penalty"]
    early_retirement_penalty = (
        1 - early_retirement_penalty_informed * retirement_age_difference
    )

    # Total bonus for late retirement
    late_retirement_bonus = 1 + (
        model_specs["late_retirement_bonus"] * retirement_age_difference
    )

    # Select bonus or penalty depending on age difference
    pension_factor = jax.lax.select(
        early_retired_bool, early_retirement_penalty, late_retirement_bonus
    )
    # pension_factor = late_retirement_bonus

    adjusted_pension_points = pension_factor * total_pension_points
    reduced_experience_years = calc_experience_for_total_pension_points(
        total_pension_points=adjusted_pension_points,
        sex=sex,
        education=education,
        model_specs=model_specs,
    )
    return reduced_experience_years


def construct_experience_years(experience, period, max_exp_diffs_per_period):
    """Experience and period can also be arrays."""
    return experience * (period + max_exp_diffs_per_period[period])
