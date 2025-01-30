"""Budget functions for the Elder Care model."""

from typing import Any

import jax.numpy as jnp
import numpy as np

from caregiving.model.shared import (
    RETIREMENT_AGE,
    is_full_time,
    is_part_time,
    is_retired,
    is_working,
)

TWO_YEARS = 2

# =====================================================================================
# Exogenous savings grid
# =====================================================================================


def create_savings_grid():
    """Create a saving grid with sections."""
    section_1 = np.arange(start=0, stop=10, step=0.5)  # 20
    section_2 = np.arange(start=10, stop=50, step=1)  # 40
    section_3 = np.arange(start=50, stop=100, step=5)  # 10
    section_4 = np.arange(start=100, stop=500, step=20)  # 20
    section_5 = np.arange(start=500, stop=1000, step=100)  # 5

    return (
        np.concatenate([section_1, section_2, section_3, section_4, section_5]) * 1_000
    )


# ==============================================================================
# Budget constraint
# ==============================================================================


def budget_constraint(
    period: int,
    lagged_choice: int,
    experience: int,
    # high_educ: int,
    savings_end_of_previous_period: float,
    income_shock_previous_period: float,
    options: dict[str, Any],
    params: dict[str, Any],
) -> float:
    """Budget constraint.

    + non_labor_income(age, high_educ, options)

    + spousal_income(period, high_educ, options) * married

    return jnp.maximum(
        wealth_beginning_of_period,
        options["consumption_floor"],
    )


    + options["unemployment_benefits"] * is_not_working(lagged_choice) * 12
    + options["informal_care_benefits"] * is_informal_care(lagged_choice) * 12
    - options["formal_care_costs"] * is_formal_care(lagged_choice) * 12

    net_labor_income = calc_net_income_working(labor_income_with_min)

    means_test = savings_end_of_previous_period < options["unemployment_wealth_thresh"]
    unemployment_benefits_yearly = means_test * options["unemployment_benefits"] * 12

    cash_benefits_informal_care = (
        (
            is_informal_care(lagged_choice)
            * is_medium_health(mother_health)
            * options["cash_benefits_medium"]
            + is_informal_care(lagged_choice)
            * is_bad_health(mother_health)
            * options["cash_benefits_bad"]
        )
        * 12
        * TWO_YEARS
    )

    + is_combination_care(lagged_choice)
    * (options["informal_care_benefits"] / 2)
    * 12

    """
    age = options["start_age"] + period

    working_hours = (
        is_part_time(lagged_choice) * 20 * 4.33 * 12  # week month year
        + is_full_time(lagged_choice) * 40 * 4.33 * 12  # week month year
    )

    wage_from_previous_period = get_exog_stochastic_wage(
        lagged_choice=lagged_choice,
        experience=experience,
        # high_educ=high_educ, # noqa: ERA001
        wage_shock=income_shock_previous_period,
        params=params,
    )
    wage = jnp.maximum(wage_from_previous_period, options["min_wage"])
    labor_income = wage * working_hours

    pension_factor = 1 - (age - RETIREMENT_AGE) * options["early_retirement_penalty"]
    retirement_income_gross_one_year = (
        options["pension_point_value"] * (experience / 2) * pension_factor * 12
    )
    retirement_income = calc_net_income_pensions(retirement_income_gross_one_year)

    means_test = savings_end_of_previous_period < options["unemployment_wealth_thresh"]
    unemployment_benefits = means_test * options["unemployment_benefits"] * 12

    # cash_benefits_informal_care = (
    #     is_pure_informal_care(lagged_choice) * options["informal_care_benefits"] * 12
    #     + is_combination_care(lagged_choice)
    #     * (options["informal_care_benefits"] / 2)
    #     * 12
    # )

    personal_income = jnp.maximum(
        is_working(lagged_choice) * labor_income
        + is_retired(lagged_choice) * retirement_income,
        unemployment_benefits,
    )

    log_spousal_income = get_exog_spousal_income(period, options)
    spousal_income = jnp.exp(log_spousal_income)

    # Assume everyone is married
    income_and_cash_benefits = (
        personal_income + spousal_income  # + cash_benefits_informal_care
    )

    return (
        1 + options["interest_rate"]
    ) * savings_end_of_previous_period + income_and_cash_benefits


def get_exog_stochastic_wage(
    lagged_choice: int,
    experience: int,
    # high_educ: int,
    wage_shock: float,
    params: dict[str, float],
) -> float:
    """Computes the current level of deterministic and stochastic income.

    # + params["wage_above_retirement_age"] * (period >= 25)

    Note that income is paid at the end of the current period, i.e. after
    the (potential) labor supply choice has been made. This is equivalent to
    allowing income to be dependent on a lagged choice of labor supply.
    The agent starts working in period t = 0.
    Relevant for the wage equation (deterministic income) are age-dependent
    coefficients of work experience:
    labor_income = constant + alpha_1 * age + alpha_2 * age**2
    They include a constant as well as two coefficients on age and age squared,
    respectively. Note that the last one (alpha_2) typically has a negative sign.


    Divide experience by 2 to get the number of years of experience.
    One year of part-time work experience is equivalent to
    0.5 years of full-time work experience. This is because the agent accumulates
    experience at a slower rate when working part-time.

    For computational reasons, we use the following transformation:
    exp = experience / 2
    exp_squared = (experience / 2) ** 2

    So one year of part-time experience is counted as 1 year of experience in the
    state space. But in the wage equation, it is counted as 0.5 years of experience.
    Analogously, one year of full-time experience is counted as 2 years of experience
    in the state space, but as 1 year of experience in the wage equation.


    Determinisctic component of income depending on experience:
    constant + alpha_1 * age + alpha_2 * age**2
    exp_coeffs = jnp.array([constant, exp, exp_squared])
    labor_income = exp_coeffs @ (age ** jnp.arange(len(exp_coeffs)))
    working_income = jnp.exp(labor_income + wage_shock)


    Args:
        period (int): Current period.
        state (jnp.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        lagged_choice (int): The lagged choice of the agent.
        experience (int): Work experience. Full-time work experience adds 1, part-time
            experience adds 0.5, and no work experience adds 0 years of experience
            per period.
        wage_shock (float): Stochastic shock on labor income;
            may or may not be normally distributed. This float represents one
            particular realization of the income_shock_draws carried over from
            the previous period.
        high_educ (int): Indicator for whether the agent has a high education.
            0 = low education, 1 = high education. High education means that
            the agent has completed at least 15 years of education.
        params (dict): Dictionary containing model parameters.
            Relevant here are the coefficients of the wage equation.
        options (dict): Options dictionary.

    Returns:
        stochastic_income (float): The potential end of period income. It consists of a
            deterministic component, i.e. age-dependent labor income,
            and a stochastic shock.

    """
    log_wage = (
        params["wage_constant"]
        + params["wage_experience"] * (experience / 2)
        + params["wage_experience_squared"] * (experience / 2) ** 2
        # + params["wage_high_education"] * high_educ # noqa: ERA001
        + params["wage_part_time"] * is_part_time(lagged_choice)
    )

    return jnp.exp(log_wage + wage_shock)


def get_exog_spousal_income(period, options):
    """Income from the spouse."""
    age = options["start_age"] + period

    return (
        options["spousal_income_constant"]
        + options["spousal_income_age"] * age
        + options["spousal_income_age_squared"] * age**2
        # + options["spousal_income_high_education"] * high_educ # noqa: ERA001
        + options["spousal_income_above_retirement_age"]
        * (age >= options["retirement_age"])
    )


# =====================================================================================
# From Max & Bruno's model
# =====================================================================================


def calc_net_income_pensions(gross_income):
    """Calculate net income after pension contributions."""
    ssc = calc_health_ltc_contr(gross_income)
    inc_tax = calc_inc_tax(gross_income - ssc)

    return gross_income - inc_tax - ssc


def calc_net_income_working(gross_income):
    """Calculate net income after pension and health contributions."""
    ssc = calc_pension_unempl_contr(gross_income) + calc_health_ltc_contr(gross_income)
    inc_tax = calc_inc_tax(gross_income - ssc)

    return gross_income - inc_tax - ssc


def calc_inc_tax(gross_income):
    """Parameters from 2010 gettsim params."""
    thresholds = [
        8004,
        13469,
        52881,
        250730,
    ]

    rates = [0.14, 0.2397, 0.42, 0.45]

    # In bracket 0 no taxes are paid
    poss_tax_bracket_0 = 0.0

    # In bracket 1 taxes are paid on income above the threshold
    poss_tax_bracket_1 = rates[0] * (gross_income - thresholds[0])
    tax_above_1 = (thresholds[1] - thresholds[0]) * rates[0]

    # In bracket 2 taxes are paid on income above the threshold and the tax paid in
    # bracket 1
    poss_tax_bracket_2 = (rates[1] * (gross_income - thresholds[1])) + tax_above_1
    tax_above_2 = (thresholds[2] - thresholds[1]) * rates[1]

    # In bracket 3 taxes are paid on income above the threshold and the tax paid in
    # brackets 1+2
    poss_tax_bracket_3 = (
        rates[2] * (gross_income - thresholds[2]) + tax_above_2 + tax_above_1
    )
    tax_above_3 = (thresholds[3] - thresholds[2]) * rates[2]

    # In bracket 4 taxes are paid on income above the threshold and the tax paid in
    # brackets 1+2+3
    poss_tax_bracket_4 = (
        rates[3] * (gross_income - thresholds[3])
        + tax_above_1
        + tax_above_2
        + tax_above_3
    )

    # Check in which bracket the income falls and calculate the tax
    in_bracket_0 = gross_income < thresholds[0]
    in_bracket_1 = (gross_income >= thresholds[0]) & (gross_income < thresholds[1])
    in_bracket_2 = (gross_income >= thresholds[1]) & (gross_income < thresholds[2])
    in_bracket_3 = (gross_income >= thresholds[2]) & (gross_income < thresholds[3])
    in_bracket_4 = gross_income >= thresholds[3]

    return (
        in_bracket_0 * poss_tax_bracket_0
        + in_bracket_1 * poss_tax_bracket_1
        + in_bracket_2 * poss_tax_bracket_2
        + in_bracket_3 * poss_tax_bracket_3
        + in_bracket_4 * poss_tax_bracket_4
    )


def calc_pension_unempl_contr(gross_income):
    """Calculate pension and unemployment contributions."""
    contribution_threshold = 5500 * 12
    rate = 0.113

    return jnp.minimum(gross_income, contribution_threshold) * rate


def calc_health_ltc_contr(gross_income):
    """Calculate health and long-term care contributions."""
    contribution_threshold = 3750 * 12
    rate = 0.08

    # social security contribution
    return jnp.minimum(gross_income, contribution_threshold) * rate
