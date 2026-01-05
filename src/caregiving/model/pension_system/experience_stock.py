import jax
from jax import numpy as jnp

from caregiving.model.pension_system.early_retirement_paths import (
    calc_early_retirement_pension_points,
)


def calc_pension_points_for_experience(
    period,
    sex,
    experience_years,
    education,
    partner_state,
    model_specs,
):
    """Calculate a new experience stock accounting for the pension adjustment.
    This function will only be used if the individual is fresh retired. So we
    can take this as a given here. Note, you can only retire if you are disabled or
    if you reached the long insured age.

    This function returns pension points adjusted for early or late retirement.
    """
    # Start by calculating the type specific pension points(Endgeltpunkte)
    total_pension_points = calc_pension_points_form_experience(
        education=education,
        experience_years=experience_years,
        sex=sex,
        model_specs=model_specs,
    )

    has_partner_int = (partner_state > 0).astype(int)
    mothers_pension = (
        sex * model_specs["max_children"][1, education, has_partner_int] * 3
    )
    total_pension_points += mothers_pension
    # retirement age is last periods age
    actual_retirement_age = jnp.minimum(
        model_specs["start_age"] + period - 1, model_specs["max_ret_age"]
    )
    # SRA at retirement, difference to actual retirement age and boolean for early retirement
    SRA_at_retirement = model_specs["min_SRA"]
    retirement_age_difference = jnp.abs(SRA_at_retirement - actual_retirement_age)
    early_retired_bool = actual_retirement_age < SRA_at_retirement

    # Pension points for early retirement
    pension_points_early_retired = calc_early_retirement_pension_points(
        total_pension_points=total_pension_points,
        retirement_age_difference=retirement_age_difference,
        experience_years=experience_years,
        sex=sex,
        model_specs=model_specs,
    )

    # Total bonus for late retirement
    pension_points_late_retirement = (
        1 + (model_specs["late_retirement_bonus"] * retirement_age_difference)
    ) * total_pension_points

    # Select bonus or penalty depending on age difference
    adjusted_pension_points = jax.lax.select(
        early_retired_bool,
        on_true=pension_points_early_retired,
        on_false=pension_points_late_retirement,
    )

    return adjusted_pension_points


def calc_pension_points_form_experience(education, sex, experience_years, model_specs):
    """Calculate the total pension point for the working live.

    We normalize by the mean wage of the whole population. The punishment for early
    retirement is already in the experience.

    """
    # mean_wage_all = model_specs["mean_hourly_ft_wage"][sex, education]
    # gamma_0 = model_specs["gamma_0"][sex, education]
    # gamma_1_plus_1 = model_specs["gamma_1"][sex, education] + 1
    # total_pens_points = (
    #     (jnp.exp(gamma_0) / gamma_1_plus_1)
    #     * ((experience_years + 1) ** gamma_1_plus_1 - 1)
    # ) / mean_wage_all
    exp_int = experience_years.astype(int)
    pp_exp_int = model_specs["pp_for_exp_by_sex_edu"][sex, education, exp_int]

    exp_frac = experience_years - exp_int
    pp_difference = (
        model_specs["pp_for_exp_by_sex_edu"][sex, education, exp_int + 1] - pp_exp_int
    )
    total_pension_points = pp_exp_int + exp_frac * pp_difference

    return total_pension_points


# def calc_experience_for_total_pension_points(
#     total_pension_points, sex, education, model_specs
# ):
#     """Calculate the experience for a given total pension points."""
#     mean_wage_all = model_specs["mean_hourly_ft_wage"][sex, education]
#     gamma_0 = model_specs["gamma_0"][sex, education]
#     gamma_1_plus_1 = model_specs["gamma_1"][sex, education] + 1
#     exp_for_pension_points = (
#         (total_pension_points * mean_wage_all * gamma_1_plus_1 / jnp.exp(gamma_0) + 1)
#         ** (1 / gamma_1_plus_1)
#     ) - 1
#     return exp_for_pension_points
