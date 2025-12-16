from jax import numpy as jnp

from caregiving.model.shared import (
    SEX,
    is_formal_care,
    is_informal_care,
    is_retired,
    is_working,
)
from caregiving.model.wealth_and_budget.partner_income import (
    calc_partner_income_after_ssc,
)
from caregiving.model.wealth_and_budget.pensions import (
    calc_pensions_after_ssc,
)
from caregiving.model.wealth_and_budget.tax_and_ssc import calc_net_household_income
from caregiving.model.wealth_and_budget.transfers import (
    calc_child_benefits,
    calc_unemployment_benefits,
)
from caregiving.model.wealth_and_budget.wages import calc_labor_income_after_ssc


def calc_care_benefits_and_costs_lower_formal_care_costs(
    lagged_choice, education, care_demand, options
):
    """Care benefits and costs with lower formal care costs.

    - Informal care: same cash benefit as baseline.
    - Formal care: cost is reduced relative to the baseline.
    """
    informal_care_solo = is_informal_care(lagged_choice)
    formal_care = is_formal_care(lagged_choice)

    annual_care_benefits = options["informal_care_cash_benefits"] * 12
    annual_care_benefits_weighted = annual_care_benefits * informal_care_solo

    # Formal care costs contribution is zero or reduced (here: zero).
    annual_care_costs = options["formal_care_costs"] * 12 * 0.0
    annual_care_costs_weighted = annual_care_costs * formal_care

    return annual_care_benefits_weighted - annual_care_costs_weighted


def budget_constraint(
    period,
    education,
    lagged_choice,  # d_{t-1}
    experience,
    # sex,
    partner_state,
    care_demand,
    savings_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    options,
):
    sex_var = SEX

    savings_scaled = savings_end_of_previous_period * options["wealth_unit"]
    # Recalculate experience
    max_exp_period = period + options["max_exp_diffs_per_period"][period]
    experience_years = max_exp_period * experience

    # Calculate partner income
    partner_income_after_ssc = calc_partner_income_after_ssc(
        partner_state=partner_state,
        sex=sex_var,
        options=options,
        education=education,
        period=period,
    )

    # Income from lagged choice 0
    retirement_income_after_ssc = calc_pensions_after_ssc(
        experience_years=experience_years,
        sex=sex_var,
        education=education,
        options=options,
    )

    has_partner_int = (partner_state > 0).astype(int)

    # Income lagged choice 1
    unemployment_benefits = calc_unemployment_benefits(
        savings=savings_scaled,
        education=education,
        sex=sex_var,
        has_partner_int=has_partner_int,
        period=period,
        options=options,
    )

    # Income lagged choice 2
    labor_income_after_ssc = calc_labor_income_after_ssc(
        lagged_choice=lagged_choice,
        experience_years=experience_years,
        education=education,
        sex=sex_var,
        income_shock=income_shock_previous_period,
        options=options,
    )

    # Select relevant income
    # bools of last period decision: income is paid in following period!
    was_worker = is_working(lagged_choice)
    was_retired = is_retired(lagged_choice)

    # Aggregate over choice own income
    own_income_after_ssc = (
        was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc
    )

    # Calculate total houshold net income
    total_net_income = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        options=options,
    )

    child_benefits = calc_child_benefits(
        education=education,
        sex=sex_var,
        has_partner_int=has_partner_int,
        period=period,
        options=options,
    )
    care_benfits_and_costs = calc_care_benefits_and_costs_lower_formal_care_costs(
        lagged_choice=lagged_choice,
        education=education,
        care_demand=care_demand,
        options=options,
    )

    total_income = jnp.maximum(
        total_net_income + child_benefits + care_benfits_and_costs,
        unemployment_benefits,
    )
    # calculate beginning of period wealth M_t
    wealth = (1 + params["interest_rate"]) * savings_scaled + total_income

    return wealth / options["wealth_unit"]
