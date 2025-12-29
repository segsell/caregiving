import jax.numpy as jnp

from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_after_ssc_income_pensioneer,
)


def calc_pensions_after_ssc(
    experience_years,
    sex,
    education,
    model_specs,
):
    """Calculate the pension income after SSC contributions."""
    retirement_income_gross = calc_gross_pension_income(
        experience_years=experience_years,
        education=education,
        sex=sex,
        model_specs=model_specs,
    )
    retirement_income = calc_after_ssc_income_pensioneer(retirement_income_gross)

    return retirement_income, retirement_income_gross


def calc_gross_pension_income(experience_years, sex, education, model_specs):
    """Calculate the gross pension income."""

    # Pension point value by education and experience
    total_pension_points = calc_total_pension_points(
        education=education,
        sex=sex,
        experience_years=experience_years,
        model_specs=model_specs,
    )
    retirement_income_gross = (
        model_specs["annual_pension_point_value"] * total_pension_points
    )
    return retirement_income_gross


def calc_total_pension_points(education, sex, experience_years, model_specs):
    """Calculate the total pension point for the working live.

    We normalize by the mean wage of the whole population. The punishment for early
    retirement is already in the experience.

    """
    mean_wage_all = model_specs["mean_hourly_ft_wage"][sex, education]
    gamma_0 = model_specs["gamma_0"][sex, education]
    gamma_1_plus_1 = model_specs["gamma_1"][sex, education] + 1
    total_pens_points = (
        (jnp.exp(gamma_0) / gamma_1_plus_1)
        * ((experience_years + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all

    return total_pens_points


def calc_experience_for_total_pension_points(
    total_pension_points, sex, education, model_specs
):
    """Calculate the experience for a given total pension points."""
    mean_wage_all = model_specs["mean_hourly_ft_wage"][sex, education]
    gamma_0 = model_specs["gamma_0"][sex, education]
    gamma_1_plus_1 = model_specs["gamma_1"][sex, education] + 1

    return (
        (total_pension_points * mean_wage_all * gamma_1_plus_1 / jnp.exp(gamma_0) + 1)
        ** (1 / gamma_1_plus_1)
    ) - 1
