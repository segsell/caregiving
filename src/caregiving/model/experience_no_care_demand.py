import jax
import jax.numpy as jnp

from caregiving.model.shared import (
    SEX,
)
from caregiving.model.shared_no_care_demand import (
    is_full_time,
    is_part_time,
    is_retired,
)
from caregiving.model.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)
from caregiving.model.experience_baseline_model import (
    construct_experience_years,
    scale_experience_years,
)

jax.config.update("jax_enable_x64", True)


def get_next_period_experience(
    period,
    lagged_choice,
    already_retired,
    partner_state,
    education,
    experience,
    model_specs,
):
    """Update experience based on lagged choice and period."""
    # Check if already longer retired. If it is not degenerated you could have not been
    # retired last period.
    sex = SEX
    retired_this_period = is_retired(lagged_choice)

    # If retired, then we update experience according to the deduction function
    fresh_retired = (already_retired == 0) & retired_this_period

    # If period is 0, then last period is also 0.
    last_period = period - 1
    last_period = last_period * (period != 0) + (period == 0) * (-1)

    exp_years_last_period = construct_experience_years(
        float_experience=experience,
        period=last_period,
        is_retired=already_retired,
        model_specs=model_specs,
    )

    # Update if working part or full time
    exp_update = is_full_time(lagged_choice) + is_part_time(lagged_choice) * (
        model_specs["exp_increase_part_time"]
    )
    exp_years_this_period = exp_years_last_period + exp_update

    # Calculate experience in the case of fresh retirement
    # We track all deductions and bonuses of the retirement decision through an adjusted
    # experience stock
    pension_points = calc_pension_points_for_experience(
        period=period,
        experience_years=exp_years_last_period,
        sex=sex,
        partner_state=partner_state,
        education=education,
        model_specs=model_specs,
    )

    # If fresh retired, the experience function returns pension points. Now the value and policy function
    # are calculated on a pension point grid. We do not need experience any more.
    exp_years_this_period = jax.lax.select(
        fresh_retired, on_true=pension_points, on_false=exp_years_this_period
    )

    # Now scale between 0 and 1
    exp_scaled = scale_experience_years(
        experience_years=exp_years_this_period,
        period=period,
        is_retired=retired_this_period,
        model_specs=model_specs,
    )

    return exp_scaled
