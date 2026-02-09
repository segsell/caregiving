"""Experience accumulation for caregiving leave counterfactuals (full and 65%).

Used by both full (Norwegian-style) and normal (65%, German-style) caregiving leave
models. During periods with caregiving leave, experience is frozen at the pre-caregiving
growth path: the agent receives the same experience credit they would have received
if they had continued in their job_before_caregiving (0=none, 1=PT, 2=FT). Otherwise
uses the same logic as the baseline (FT=1.0, PT=exp_increase_part_time or 1.0 if
intensive informal care).
"""

import jax
import jax.numpy as jnp

from caregiving.model.experience_baseline_model import (
    construct_experience_years,
    scale_experience_years,
)
from caregiving.model.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)
from caregiving.model.shared import (
    JOB_RETENTION_FULL_TIME,
    JOB_RETENTION_PART_TIME,
    SEX,
    is_full_time,
    is_informal_care,
    is_intensive_informal_care,
    is_part_time,
    is_retired,
    is_unemployed,
)


def get_next_period_experience_caregiving_leave(
    period,
    lagged_choice,
    already_retired,
    partner_state,
    education,
    experience,
    job_before_caregiving,
    model_specs,
):
    """Update experience; during caregiving leave, freeze at pre-caregiving path.

    Used by both full (100%) and normal (65%) caregiving leave models. Caregiving
    leave = currently in informal care and (unemployed, or part-time with prior
    full-time job). In those periods, experience credit equals what the agent
    would have received in job_before_caregiving:
    - job_before_caregiving=0 (none): 0
    - job_before_caregiving=1 (PT): exp_increase_part_time
    - job_before_caregiving=2 (FT): 1.0

    Otherwise uses baseline rules: full-time=1.0, part-time=exp_increase_part_time
    or 1.0 if intensive informal care.

    Parameters
    ----------
    period : int
        Current period
    lagged_choice : int
        Choice from previous period (d_{t-1})
    already_retired : int
        Indicator if agent was already retired in previous period
    partner_state : int
        Partner state (0=no partner, 1=working, 2=retired)
    education : int
        Education level (0=low, 1=high)
    experience : float
        Current experience (scaled 0-1)
    job_before_caregiving : int
        0=none, 1=PT, 2=FT (job held when caregiving started)
    model_specs : dict
        Model specifications

    Returns
    -------
    float
        Updated experience (scaled 0-1)
    """
    sex = SEX

    retired_this_period = is_retired(lagged_choice)

    fresh_retired = (already_retired == 0) & retired_this_period

    last_period = period - 1
    last_period = last_period * (period != 0) + (period == 0) * (-1)

    exp_years_last_period = construct_experience_years(
        float_experience=experience,
        period=last_period,
        is_retired=already_retired,
        model_specs=model_specs,
    )

    # Caregiving leave = periods when the agent receives (or would receive) leave
    # benefit; we freeze experience at the pre-caregiving path. Same eligibility
    # as calc_full_caregiving_leave_top_up and calc_caregiving_leave_top_up (65%).
    #
    # INCLUDED (on_caregiving_leave = True):
    #   - Caregiver, not retired, UNEMPLOYED (any job_before_caregiving 0/1/2;
    #     prior none + unemp gives 0 benefit but we still freeze experience at 0).
    #   - Caregiver, not retired, PART-TIME with prior FULL-TIME (gap benefit).
    #
    # EXCLUDED (on_caregiving_leave = False):
    #   - Caregiver + FULL-TIME (any prior) — no benefit, experience = baseline.
    #   - Caregiver + PART-TIME with prior PT or prior none — no benefit.
    #   - Non-caregiver; retired.
    currently_caregiver = is_informal_care(lagged_choice)
    currently_unemployed = is_unemployed(lagged_choice)
    currently_pt = is_part_time(lagged_choice)

    prior_ft = job_before_caregiving == JOB_RETENTION_FULL_TIME
    prior_pt = job_before_caregiving == JOB_RETENTION_PART_TIME
    prior_none = job_before_caregiving == 0

    on_caregiving_leave = (
        currently_caregiver
        * (1 - retired_this_period)
        * (currently_unemployed | (currently_pt * prior_ft))
    )

    # Pre-caregiving growth path: credit as if they had kept job_before_caregiving
    exp_update_frozen = (
        prior_ft * 1.0
        + prior_pt * model_specs["exp_increase_part_time"]
        + prior_none * 0.0
    )

    # Baseline update when not on caregiving leave
    intensive_care = is_intensive_informal_care(lagged_choice)
    exp_update_baseline = is_full_time(lagged_choice) + is_part_time(lagged_choice) * (
        model_specs["exp_increase_part_time"] * (1 - intensive_care) + intensive_care
    )

    exp_update = jnp.where(on_caregiving_leave, exp_update_frozen, exp_update_baseline)
    exp_years_this_period = exp_years_last_period + exp_update

    pension_points = calc_pension_points_for_experience(
        period=period,
        experience_years=exp_years_last_period,
        sex=sex,
        partner_state=partner_state,
        education=education,
        model_specs=model_specs,
    )

    exp_years_this_period = jax.lax.select(
        fresh_retired, on_true=pension_points, on_false=exp_years_this_period
    )

    exp_scaled = scale_experience_years(
        experience_years=exp_years_this_period,
        period=period,
        is_retired=retired_this_period,
        model_specs=model_specs,
    )

    return exp_scaled
