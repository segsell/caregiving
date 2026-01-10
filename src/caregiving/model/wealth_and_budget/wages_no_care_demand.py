"""Wage calculation functions for the no-care-demand counterfactual.

This module provides wage calculation functions adapted for the reduced 4-state
choice space without informal caregiving model_specs.
"""

from jax import numpy as jnp

from caregiving.model.shared_no_care_demand import is_full_time, is_part_time
from caregiving.model.wealth_and_budget.tax_and_ssc import calc_after_ssc_income_worker
from caregiving.model.wealth_and_budget.wages import calc_hourly_wage


def calc_labor_income_after_ssc(
    lagged_choice, experience_years, education, sex, income_shock, model_specs
):
    """Calculate labor income after social security contributions.

    Args:
        lagged_choice: Previous period's choice:
            0=retirement, 1=unemployed, 2=part-time, 3=full-time
        experience_years: Years of work experience
        education: Education level (0=low, 1=high)
        sex: Sex indicator
        income_shock: Income shock realization
        model_specs: Model specifications containing wage parameters

    Returns:
        Tuple of (labor_income_after_ssc, gross_labor_income)
    """
    # Gross labor income
    gross_labor_income = calculate_gross_labor_income(
        lagged_choice=lagged_choice,
        experience_years=experience_years,
        education=education,
        sex=sex,
        income_shock=income_shock,
        model_specs=model_specs,
    )
    labor_income_after_ssc = calc_after_ssc_income_worker(gross_labor_income)

    return labor_income_after_ssc, gross_labor_income


def calculate_gross_labor_income(
    lagged_choice, experience_years, education, sex, income_shock, model_specs
):
    """Calculate the gross labor income.

    As we estimate the wage equation outside of the model, we fetch the experience
    returns from model_specs.

    Args:
        lagged_choice: Previous period's choice:
            0=retirement, 1=unemployed, 2=part-time, 3=full-time
        experience_years: Years of work experience
        education: Education level (0=low, 1=high)
        sex: Sex indicator
        income_shock: Income shock realization
        model_specs: Model specifications containing wage parameters

    Returns:
        Gross labor income with minimum wage floor applied
    """

    # Part time and full time choices using no-care-demand predicates
    pt_work = is_part_time(lagged_choice)
    ft_work = is_full_time(lagged_choice)

    average_hours = (
        model_specs["av_annual_hours_pt"][sex, education] * pt_work
        + model_specs["av_annual_hours_ft"][sex, education] * ft_work
    )

    hourly_wage = calc_hourly_wage(
        sex=sex,
        education=education,
        experience_years=experience_years,
        income_shock=income_shock,
        model_specs=model_specs,
    )
    labour_income = hourly_wage * average_hours

    # Minimum wage. Education specific as hours are different among educations.
    annual_min_wage_pt = model_specs["annual_min_wage_pt"][sex, education]
    annual_min_wage_ft = model_specs["annual_min_wage_ft"]
    annual_min_wage = annual_min_wage_pt * pt_work + annual_min_wage_ft * ft_work

    labor_income_min_checked = jnp.maximum(labour_income, annual_min_wage)

    return labor_income_min_checked
