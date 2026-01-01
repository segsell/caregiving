"""State space for the model without care demand."""

import jax
import jax.numpy as jnp
import numpy as np

from caregiving.model.shared import (
    PARENT_LONGER_DEAD,
    PARENT_RECENTLY_DEAD,
    SEX,
    is_alive,
    is_dead,
)
from caregiving.model.shared_no_care_demand import (
    ALL_NO_CARE_DEMAND,
    NOT_WORKING_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
    WORK_AND_RETIREMENT_NO_CARE_DEMAND,
    WORK_AND_UNEMPLOYED_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
    is_full_time,
    is_part_time,
    is_retired,
    is_unemployed,
)
from caregiving.model.state_space import (
    apply_retirement_constraint_for_SRA,
)
from caregiving.model.wealth_and_budget.pensions import (
    calc_experience_for_total_pension_points,
    calc_total_pension_points,
)


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_deterministic_state": next_period_deterministic_state,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
    }


# =====================================================================================
# State transitions (no care demand and no caregiving choices)
# =====================================================================================


def next_period_deterministic_state(
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
    health,
    partner_state,
    job_offer,
    mother_dead,
    caregiving_type,
    model_specs,
):
    """Sparsity condition for no care demand counterfactual.

    Excludes care_demand, mother_adl, and caregiving_type states.
    Includes mother_dead for inheritance calculation.
    """
    start_age = model_specs["start_age"]
    max_ret_age = model_specs["max_ret_age"]
    min_ret_age_state_space = model_specs["min_ret_age"]

    SRA_pol_state = model_specs["min_SRA"]

    last_period = model_specs["n_periods"] - 1

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
                "health": health,
                "partner_state": partner_state,
                "mother_dead": PARENT_LONGER_DEAD,
                "job_offer": 0,
                "caregiving_type": caregiving_type,
            }
            return state_proxy
        elif mother_dead == PARENT_RECENTLY_DEAD:
            # If mother recently died, no care demand and supply
            # Preserve mother_dead == 1 so inheritance can be calculated
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "education": education,
                "health": health,
                "partner_state": partner_state,
                "mother_dead": PARENT_RECENTLY_DEAD,  # Preserve for inheritance calculation
                "job_offer": job_offer,
                "caregiving_type": caregiving_type,
            }
            return state_proxy
        elif mother_dead == PARENT_LONGER_DEAD:
            # If mother is longer dead, no care demand and supply
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "education": education,
                "health": health,
                "partner_state": partner_state,
                "mother_dead": PARENT_LONGER_DEAD,
                "job_offer": job_offer,
                "caregiving_type": caregiving_type,
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
                "mother_dead": mother_dead,
                "job_offer": 0,
                "caregiving_type": caregiving_type,
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
                "mother_dead": mother_dead,
                "job_offer": 0,
                "caregiving_type": caregiving_type,
            }
            return state_proxy
        else:
            return True


def state_specific_choice_set(  # noqa: PLR0911, PLR0912
    period, lagged_choice, job_offer, health, model_specs
):
    age = period + model_specs["start_age"]
    SRA_pol_state = model_specs["min_SRA"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(
        SRA_pol_state, model_specs
    )

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
            return WORK_AND_UNEMPLOYED_NO_CARE_DEMAND
    # Person must be retired
    elif age >= model_specs["max_ret_age"]:
        return RETIREMENT_NO_CARE_DEMAND
    # Person is in the voluntary retirement range.
    else:
        if age >= SRA_pol_state:
            if job_offer == 0:
                return RETIREMENT_NO_CARE_DEMAND
            else:
                return WORK_AND_RETIREMENT_NO_CARE_DEMAND
        else:
            if job_offer == 0:
                # Choose unemployment or retirement
                return NOT_WORKING_NO_CARE_DEMAND
            else:
                return ALL_NO_CARE_DEMAND


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
    # SRA at retirement, difference to actual retirement age and boolean
    # for early retirement
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
