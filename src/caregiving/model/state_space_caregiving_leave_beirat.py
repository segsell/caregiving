"""State space for Beirat caregiving leave (max 3 years, partial leave only)."""

import jax.numpy as jnp

from caregiving.model.experience_caregiving_leave_model import (
    get_next_period_experience_caregiving_leave_beirat,
    get_next_period_experience_caregiving_leave_full_beirat,
    get_next_period_experience_caregiving_leave_full_beirat_no_total_cap,
)
from caregiving.model.shared import (
    LEAVE_CAP_YEARS,
    NO_CARE_DEMAND,
    PARENT_LONGER_DEAD,
    had_ft_job_before_caregiving,
    had_pt_job_before_caregiving,
    is_alive,
    is_dead,
    is_formal_care,
    is_informal_care,
    is_part_time,
    is_retired,
    is_unemployed,
    is_working,
)
from caregiving.model.state_space import (
    state_specific_choice_set_with_caregiving,
)
from caregiving.model.state_space_caregiving_leave_with_job_retention import (
    next_period_deterministic_state_with_job_retention,
)


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set_with_caregiving,
        "next_period_deterministic_state": next_period_deterministic_state_beirat,
        "next_period_experience": get_next_period_experience_caregiving_leave_beirat,
        "sparsity_condition": sparsity_condition_beirat,
    }


def create_state_space_functions_full_beirat():
    """State space for full Beirat (max 3 years, max 1 year full leave)."""
    return {
        "state_specific_choice_set": state_specific_choice_set_with_caregiving,
        "next_period_deterministic_state": next_period_deterministic_state_full_beirat,
        "next_period_experience": (
            get_next_period_experience_caregiving_leave_full_beirat
        ),
        "sparsity_condition": sparsity_condition_full_beirat,
    }


def create_state_space_functions_full_beirat_no_total_cap():
    """State space for full Beirat with the 3-year cumulative cap removed.

    Only the 1-year sub-cap on full leave is retained (via
    ``full_leave_year_used``). ``years_leave_used_total`` is dropped from
    deterministic states. Partial leave (PT with prior FT) is uncapped. Job
    retention is indefinite (delegated to the ``with_job_retention`` transition
    via the next_period_deterministic_state function).
    """
    return {
        "state_specific_choice_set": state_specific_choice_set_with_caregiving,
        "next_period_deterministic_state": (
            next_period_deterministic_state_full_beirat_no_total_cap
        ),
        "next_period_experience": (
            get_next_period_experience_caregiving_leave_full_beirat_no_total_cap
        ),
        "sparsity_condition": sparsity_condition_full_beirat_no_total_cap,
    }


# =====================================================================================
# State transitions
# =====================================================================================


def next_period_deterministic_state_beirat(
    period,
    choice,
    lagged_choice,
    already_retired,
    job_before_caregiving,
    years_leave_used_total,
):
    """Update deterministic states: job_before_caregiving + Beirat leave counter.

    Partial leave only: years_leave_used_total in {0,1,2,3} increments when on
    partial leave (PT while caregiving with prior FT job) and total < 3.
    job_before_caregiving transition is unchanged from the unlimited leave model;
    we add only the years_leave_used_total update.
    """
    base = next_period_deterministic_state_with_job_retention(
        period=period,
        choice=choice,
        lagged_choice=lagged_choice,
        already_retired=already_retired,
        job_before_caregiving=job_before_caregiving,
    )

    on_partial_leave = (
        is_informal_care(choice)
        * (1 - is_retired(choice))
        * is_part_time(choice)
        * had_ft_job_before_caregiving(job_before_caregiving)
    )
    still_eligible = years_leave_used_total < LEAVE_CAP_YEARS
    increment_leave = (on_partial_leave * still_eligible).astype(jnp.int32)
    years_leave_used_total_new = jnp.minimum(
        years_leave_used_total + increment_leave, 3
    )

    return {
        **base,
        "years_leave_used_total": years_leave_used_total_new,
    }


def next_period_deterministic_state_full_beirat(
    period,
    choice,
    lagged_choice,
    already_retired,
    job_before_caregiving,
    years_leave_used_total,
    full_leave_year_used,
):
    """Update deterministic states for full Beirat.

    Updates job_before_caregiving + leave counters.
    years_leave_used_total in {0,1,2,3}; full_leave_year_used in {0,1}.
    Invalid combination (0, 1) excluded by sparsity. Full leave = unemployed with
    prior job; partial leave = PT with prior FT. At most 1 year full leave,
    3 years total.
    """
    base = next_period_deterministic_state_with_job_retention(
        period=period,
        choice=choice,
        lagged_choice=lagged_choice,
        already_retired=already_retired,
        job_before_caregiving=job_before_caregiving,
    )

    current_caregiver = is_informal_care(choice)
    employed_before_caregiving = had_pt_job_before_caregiving(
        job_before_caregiving
    ) | had_ft_job_before_caregiving(job_before_caregiving)

    on_full_leave = (
        current_caregiver
        * (1 - is_retired(choice))
        * is_unemployed(choice)
        * employed_before_caregiving
    )
    on_partial_leave = (
        current_caregiver
        * (1 - is_retired(choice))
        * is_part_time(choice)
        * had_ft_job_before_caregiving(job_before_caregiving)
    )

    still_eligible_for_full = (years_leave_used_total < LEAVE_CAP_YEARS) * (
        full_leave_year_used == 0
    )
    still_eligible_for_partial = years_leave_used_total < LEAVE_CAP_YEARS

    actually_on_full_leave = on_full_leave * still_eligible_for_full
    actually_on_partial_leave = on_partial_leave * still_eligible_for_partial

    increment_leave = (actually_on_full_leave + actually_on_partial_leave).astype(
        jnp.int32
    )
    years_leave_used_total_new = jnp.minimum(
        years_leave_used_total + increment_leave, 3
    )
    full_leave_year_used_new = jnp.maximum(
        full_leave_year_used, actually_on_full_leave.astype(jnp.int32)
    )

    return {
        **base,
        "years_leave_used_total": years_leave_used_total_new,
        "full_leave_year_used": full_leave_year_used_new,
    }


def next_period_deterministic_state_full_beirat_no_total_cap(
    period,
    choice,
    lagged_choice,
    already_retired,
    job_before_caregiving,
    full_leave_year_used,
):
    """Update deterministic states for the Full-Beirat-no-total-cap variant.

    Same job_before_caregiving transition as ``with_job_retention``;
    ``full_leave_year_used`` (in {0,1}) flips to 1 when the agent goes on full
    leave (unemployed + caregiving + prior job) for the first time. The 3-year
    cumulative cap is removed (``years_leave_used_total`` is not a state).
    """
    base = next_period_deterministic_state_with_job_retention(
        period=period,
        choice=choice,
        lagged_choice=lagged_choice,
        already_retired=already_retired,
        job_before_caregiving=job_before_caregiving,
    )

    current_caregiver = is_informal_care(choice)
    employed_before_caregiving = had_pt_job_before_caregiving(
        job_before_caregiving
    ) | had_ft_job_before_caregiving(job_before_caregiving)

    on_full_leave = (
        current_caregiver
        * (1 - is_retired(choice))
        * is_unemployed(choice)
        * employed_before_caregiving
    )

    still_eligible_for_full = full_leave_year_used == 0
    actually_on_full_leave = on_full_leave * still_eligible_for_full

    full_leave_year_used_new = jnp.maximum(
        full_leave_year_used, actually_on_full_leave.astype(jnp.int32)
    )

    return {
        **base,
        "full_leave_year_used": full_leave_year_used_new,
    }


def sparsity_condition_beirat(  # noqa: PLR0911, PLR0912
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
    years_leave_used_total,
    job_offer,
    caregiving_type,
    model_specs,
):
    """Sparsity for Beirat model (partial leave only).

    Same as caregiving leave with job retention.
    """
    start_age = model_specs["start_age"]
    max_ret_age = model_specs["max_ret_age"]
    min_ret_age_state_space = model_specs["min_ret_age"]

    start_age_caregiving = model_specs["start_age_caregiving"]
    end_age_caregiving = model_specs["end_age_caregiving"]

    SRA_pol_state = model_specs["min_SRA"]

    last_period = model_specs["n_periods"] - 1

    age = start_age + period

    if (age <= min_ret_age_state_space) & (is_retired(lagged_choice)):
        return False
    elif (age <= min_ret_age_state_space + 1) & (already_retired == 1):
        return False
    elif (age > SRA_pol_state) & (is_unemployed(lagged_choice)):
        return False
    elif (not is_retired(lagged_choice)) & (already_retired == 1):
        return False
    elif (age > max_ret_age) & (not is_retired(lagged_choice)) & (is_alive(health)):
        return False
    elif (age > max_ret_age + 1) & (already_retired != 1):
        return False
    elif (age > end_age_caregiving + 1) & is_informal_care(lagged_choice):
        return False
    elif (age > end_age_caregiving + 1) & is_formal_care(lagged_choice):
        return False
    elif (age <= start_age_caregiving) & (is_informal_care(lagged_choice)):
        return False
    elif (age <= start_age_caregiving) & (is_formal_care(lagged_choice)):
        return False
    elif (caregiving_type == 0) & (is_informal_care(lagged_choice)):
        return False
    elif (not is_informal_care(lagged_choice)) & (job_before_caregiving != 0):
        return False
    # ================================================================================
    elif (caregiving_type == 0) & (years_leave_used_total > 0):
        return False
    # ================================================================================
    else:
        if is_dead(health):
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
                "job_offer": 0,
                "job_before_caregiving": 0,
                "years_leave_used_total": years_leave_used_total,
            }
            return state_proxy
        elif mother_dead == PARENT_LONGER_DEAD:
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
                "job_offer": job_offer,
                "job_before_caregiving": 0,
                "years_leave_used_total": years_leave_used_total,
            }
            return state_proxy
        elif age > max_ret_age + 1:
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
                "care_demand": care_demand,
                "job_offer": 0,
                "job_before_caregiving": 0,
                "years_leave_used_total": years_leave_used_total,
            }
            return state_proxy
        elif (age <= max_ret_age + 1) and is_retired(lagged_choice):
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
                "job_before_caregiving": job_before_caregiving,
                "years_leave_used_total": years_leave_used_total,
            }
            return state_proxy
        elif age > end_age_caregiving + 1:
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
                "job_before_caregiving": job_before_caregiving,
                "years_leave_used_total": years_leave_used_total,
            }
            return state_proxy
        elif age < start_age_caregiving:
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
                "job_before_caregiving": job_before_caregiving,
                "years_leave_used_total": 0,
            }
            return state_proxy

        else:
            return True


def sparsity_condition_full_beirat(  # noqa: PLR0911, PLR0912
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
    years_leave_used_total,
    full_leave_year_used,
    job_offer,
    caregiving_type,
    model_specs,
):
    """Sparsity for full Beirat model.

    Same as job retention + exclude (total=0, full=1).
    """
    start_age = model_specs["start_age"]
    max_ret_age = model_specs["max_ret_age"]
    min_ret_age_state_space = model_specs["min_ret_age"]

    start_age_caregiving = model_specs["start_age_caregiving"]
    end_age_caregiving = model_specs["end_age_caregiving"]

    SRA_pol_state = model_specs["min_SRA"]

    last_period = model_specs["n_periods"] - 1

    age = start_age + period

    # Invalid full Beirat state: cannot have used full leave with 0 total
    if (years_leave_used_total == 0) & (full_leave_year_used == 1):
        return False

    if (age <= min_ret_age_state_space) & (is_retired(lagged_choice)):
        return False
    elif (age <= min_ret_age_state_space + 1) & (already_retired == 1):
        return False
    elif (age > SRA_pol_state) & (is_unemployed(lagged_choice)):
        return False
    elif (not is_retired(lagged_choice)) & (already_retired == 1):
        return False
    elif (age > max_ret_age) & (not is_retired(lagged_choice)) & (is_alive(health)):
        return False
    elif (age > max_ret_age + 1) & (already_retired != 1):
        return False
    elif (age > end_age_caregiving + 1) & is_informal_care(lagged_choice):
        return False
    elif (age > end_age_caregiving + 1) & is_formal_care(lagged_choice):
        return False
    elif (age <= start_age_caregiving) & (is_informal_care(lagged_choice)):
        return False
    elif (age <= start_age_caregiving) & (is_formal_care(lagged_choice)):
        return False
    elif (caregiving_type == 0) & (is_informal_care(lagged_choice)):
        return False
    elif (not is_informal_care(lagged_choice)) & (job_before_caregiving != 0):
        return False
    # ================================================================================
    elif (caregiving_type == 0) & (years_leave_used_total > 0):
        return False
    # ================================================================================
    else:
        _proxy_full = {
            "years_leave_used_total": 0,
            "full_leave_year_used": 0,
        }
        if is_dead(health):
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
                "job_offer": 0,
                "job_before_caregiving": 0,
                "years_leave_used_total": years_leave_used_total,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif mother_dead == PARENT_LONGER_DEAD:
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
                "job_offer": job_offer,
                "job_before_caregiving": 0,
                "years_leave_used_total": years_leave_used_total,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif age > max_ret_age + 1:
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
                "care_demand": care_demand,
                "job_offer": 0,
                "job_before_caregiving": 0,
                "years_leave_used_total": years_leave_used_total,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif (age <= max_ret_age + 1) and is_retired(lagged_choice):
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
                "job_before_caregiving": job_before_caregiving,
                "years_leave_used_total": years_leave_used_total,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif age > end_age_caregiving + 1:
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
                "job_before_caregiving": job_before_caregiving,
                "years_leave_used_total": years_leave_used_total,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif age < start_age_caregiving:
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
                "job_before_caregiving": job_before_caregiving,
                "years_leave_used_total": 0,
                "full_leave_year_used": 0,
            }
            return state_proxy

        else:
            return True


def sparsity_condition_full_beirat_no_total_cap(  # noqa: PLR0911, PLR0912
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
    full_leave_year_used,
    job_offer,
    caregiving_type,
    model_specs,
):
    """Sparsity for Full-Beirat-no-total-cap variant.

    Same predicates as ``_full_beirat`` with all references to
    ``years_leave_used_total`` removed and the invalid-combination check
    ``(total==0 & full==1)`` dropped (only ``full_leave_year_used`` ∈ {0, 1}
    remains and both values are valid on their own). Non-caregivers cannot
    have used full leave, so ``(caregiving_type == 0) & (full_leave_year_used
    > 0)`` is excluded.
    """
    start_age = model_specs["start_age"]
    max_ret_age = model_specs["max_ret_age"]
    min_ret_age_state_space = model_specs["min_ret_age"]

    start_age_caregiving = model_specs["start_age_caregiving"]
    end_age_caregiving = model_specs["end_age_caregiving"]

    SRA_pol_state = model_specs["min_SRA"]

    last_period = model_specs["n_periods"] - 1

    age = start_age + period

    if (age <= min_ret_age_state_space) & (is_retired(lagged_choice)):
        return False
    elif (age <= min_ret_age_state_space + 1) & (already_retired == 1):
        return False
    elif (age > SRA_pol_state) & (is_unemployed(lagged_choice)):
        return False
    elif (not is_retired(lagged_choice)) & (already_retired == 1):
        return False
    elif (age > max_ret_age) & (not is_retired(lagged_choice)) & (is_alive(health)):
        return False
    elif (age > max_ret_age + 1) & (already_retired != 1):
        return False
    elif (age > end_age_caregiving + 1) & is_informal_care(lagged_choice):
        return False
    elif (age > end_age_caregiving + 1) & is_formal_care(lagged_choice):
        return False
    elif (age <= start_age_caregiving) & (is_informal_care(lagged_choice)):
        return False
    elif (age <= start_age_caregiving) & (is_formal_care(lagged_choice)):
        return False
    elif (caregiving_type == 0) & (is_informal_care(lagged_choice)):
        return False
    elif (not is_informal_care(lagged_choice)) & (job_before_caregiving != 0):
        return False
    # ================================================================================
    elif (caregiving_type == 0) & (full_leave_year_used > 0):
        return False
    # ================================================================================
    else:
        if is_dead(health):
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
                "job_offer": 0,
                "job_before_caregiving": 0,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif mother_dead == PARENT_LONGER_DEAD:
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
                "job_offer": job_offer,
                "job_before_caregiving": 0,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif age > max_ret_age + 1:
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
                "care_demand": care_demand,
                "job_offer": 0,
                "job_before_caregiving": 0,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif (age <= max_ret_age + 1) and is_retired(lagged_choice):
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
                "job_before_caregiving": job_before_caregiving,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif age > end_age_caregiving + 1:
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
                "job_before_caregiving": job_before_caregiving,
                "full_leave_year_used": full_leave_year_used,
            }
            return state_proxy
        elif age < start_age_caregiving:
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
                "job_before_caregiving": job_before_caregiving,
                "full_leave_year_used": 0,
            }
            return state_proxy

        else:
            return True
