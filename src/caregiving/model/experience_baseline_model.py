import jax
import jax.numpy as jnp
import numpy as np

from caregiving.model.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)
from caregiving.model.shared import (
    SEX,
    is_full_time,
    is_intensive_informal_care,
    is_part_time,
    is_retired,
)


def define_experience_grid(specs):
    # Experience grid
    experience_grid = np.linspace(0, 1, 11)
    # Add very long insured threshold to experience grid and sort
    experience_grid = np.append(experience_grid, specs["very_long_insured_grid_points"])
    # Delete 0.5
    experience_grid = experience_grid[
        (~np.isclose(experience_grid, 0.5))
        & ~np.isclose(experience_grid, 0.6)
        & (~np.isclose(experience_grid, 0))
    ]

    experience_grid = np.sort(experience_grid)
    experience_grid[0] = 0
    experience_grid[1] = 0.15
    experience_grid[-3] = 0.85
    return jnp.asarray(experience_grid)


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
    # Full pension point (1.0) for part-time workers providing intensive informal care
    # Full-time workers are unaffected (always get 1.0)
    intensive_care = is_intensive_informal_care(lagged_choice)
    exp_update = is_full_time(lagged_choice) + is_part_time(lagged_choice) * (
        model_specs["exp_increase_part_time"] * (1 - intensive_care) + intensive_care
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

    # # If fresh retired, the experience function returns pension points. Now the
    #  value and policy function
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


def construct_experience_years(float_experience, period, is_retired, model_specs):
    """Experience and period can also be arrays. We have to distinguish  # noqa: E501
    between the phases where individuals are already
    longer retired or not."""
    # If period is past the last working period, then we take the maximum experience
    scale_not_retired = jnp.take(
        model_specs["max_exps_period_working"], period, mode="clip"
    )
    scale_retired = model_specs["max_pp_retirement"]
    scale = is_retired * scale_retired + (1 - is_retired) * scale_not_retired
    return float_experience * scale


def scale_experience_years(experience_years, period, is_retired, model_specs):
    """Scale experience between 0 and 1."""
    # If period is past the last working period, then we take the maximum experience
    scale_not_retired = jnp.take(
        model_specs["max_exps_period_working"], period, mode="clip"
    )
    scale_retired = model_specs["max_pp_retirement"]
    scale = is_retired * scale_retired + (1 - is_retired) * scale_not_retired
    return experience_years / scale
