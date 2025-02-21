import jax.numpy as jnp

from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_after_ssc_income_pensioneer,
)


def calc_pensions_after_ssc(
    experience_years,
    sex,
    education,
    options,
):
    """Calculate the pension income after SSC contributions."""
    retirement_income_gross = calc_gross_pension_income(
        experience_years=experience_years,
        education=education,
        sex=sex,
        options=options,
    )
    retirement_income = calc_after_ssc_income_pensioneer(retirement_income_gross)

    return retirement_income


def calc_gross_pension_income(experience_years, sex, education, options):
    """Calculate the gross pension income."""

    # Pension point value by education and experience
    total_pension_points = calc_total_pension_points(
        education=education, sex=sex, experience_years=experience_years, options=options
    )
    retirement_income_gross = (
        options["annual_pension_point_value"] * total_pension_points
    )
    return retirement_income_gross


def calc_total_pension_points(education, sex, experience_years, options):
    """Calculate the total pension point for the working live.

    We normalize by the mean wage of the whole population. The punishment for early
    retirement is already in the experience.

    """
    mean_wage_all = options["mean_hourly_ft_wage"][sex, education]
    gamma_0 = options["gamma_0"][sex, education]
    gamma_1_plus_1 = options["gamma_1"][sex, education] + 1
    total_pens_points = (
        (jnp.exp(gamma_0) / gamma_1_plus_1)
        * ((experience_years + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all

    return total_pens_points


def calc_experience_for_total_pension_points(
    total_pension_points, sex, education, options
):
    """Calculate the experience for a given total pension points."""
    mean_wage_all = options["mean_hourly_ft_wage"][sex, education]
    gamma_0 = options["gamma_0"][sex, education]
    gamma_1_plus_1 = options["gamma_1"][sex, education] + 1

    return (
        (total_pension_points * mean_wage_all * gamma_1_plus_1 / jnp.exp(gamma_0) + 1)
        ** (1 / gamma_1_plus_1)
    ) - 1


def calc_experience_years_for_pension_adjustment(
    period, sex, experience_years, education, options
):
    """Calculate the reduced experience with early retirement penalty."""
    # retirement age is last periods age
    age = options["start_age"] + period - 1

    total_pension_points = calc_total_pension_points(
        education=education,
        experience_years=experience_years,
        sex=sex,
        options=options,
    )

    # Select penalty depending on age difference
    pension_factor = (
        1 - (age - options["min_SRA"]) * options["early_retirement_penalty"]
    )
    adjusted_pension_points = pension_factor * total_pension_points

    reduced_experience_years = calc_experience_for_total_pension_points(
        total_pension_points=adjusted_pension_points,
        sex=sex,
        education=education,
        options=options,
    )

    return reduced_experience_years
