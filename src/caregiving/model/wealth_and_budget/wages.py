from jax import numpy as jnp

from caregiving.model.shared import is_full_time, is_part_time
from caregiving.model.wealth_and_budget import calc_after_ssc_income_worker


def calc_labor_income_after_ssc(
    lagged_choice, experience_years, education, sex, income_shock, options
):
    # Gross labor income
    gross_labor_income = calculate_gross_labor_income(
        lagged_choice=lagged_choice,
        experience_years=experience_years,
        education=education,
        sex=sex,
        income_shock=income_shock,
        options=options,
    )
    labor_income_after_ssc = calc_after_ssc_income_worker(gross_labor_income)

    return labor_income_after_ssc


def calculate_gross_labor_income(
    lagged_choice, experience_years, education, sex, income_shock, options
):
    """Calculate the gross labor income.

    As we estimate the wage equation outside of the model, we fetch the experience
    returns from options.

    """
    gamma_0 = options["gamma_0"][sex, education]
    gamma_1 = options["gamma_1"][sex, education]
    hourly_wage = jnp.exp(
        gamma_0 + gamma_1 * jnp.log(experience_years + 1) + income_shock
    )

    # Part time choice
    pt_work = is_part_time(lagged_choice)
    ft_work = is_full_time(lagged_choice)

    average_hours = (
        options["av_annual_hours_pt"][sex, education] * pt_work
        + options["av_annual_hours_ft"][sex, education] * ft_work
    )
    labour_income = hourly_wage * average_hours

    # Minimum wage. Education specific as hours are different among educations.
    annual_min_wage_pt = options["annual_min_wage_pt"][sex, education]
    annual_min_wage_ft = options["annual_min_wage_ft"]
    annual_min_wage = annual_min_wage_pt * pt_work + annual_min_wage_ft * ft_work

    labor_income_min_checked = jnp.maximum(labour_income, annual_min_wage)

    return labor_income_min_checked
