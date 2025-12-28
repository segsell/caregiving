from jax import numpy as jnp

from caregiving.model.shared import SEX
from caregiving.model.shared_no_care_demand import is_retired, is_working
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
from caregiving.model.wealth_and_budget.wages_no_care_demand import (
    calc_labor_income_after_ssc,
)


def budget_constraint(
    period,
    education,
    lagged_choice,  # d_{t-1}
    experience,
    partner_state,
    asset_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    model_specs,
):
    sex_var = SEX

    assets_scaled = asset_end_of_previous_period * model_specs["wealth_unit"]
    # Recalculate experience
    max_exp_period = period + model_specs["max_exp_diffs_per_period"][period]
    experience_years = max_exp_period * experience

    # Calculate partner income
    partner_income_after_ssc = calc_partner_income_after_ssc(
        partner_state=partner_state,
        sex=sex_var,
        model_specs=model_specs,
        education=education,
        period=period,
    )

    # Income from lagged choice 0
    retirement_income_after_ssc = calc_pensions_after_ssc(
        experience_years=experience_years,
        sex=sex_var,
        education=education,
        model_specs=model_specs,
    )

    has_partner_int = (partner_state > 0).astype(int)

    # Income lagged choice 1
    unemployment_benefits = calc_unemployment_benefits(
        assets=assets_scaled,
        education=education,
        sex=sex_var,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=model_specs,
    )

    # Income lagged choice 2
    labor_income_after_ssc = calc_labor_income_after_ssc(
        lagged_choice=lagged_choice,
        experience_years=experience_years,
        education=education,
        sex=sex_var,
        income_shock=income_shock_previous_period,
        model_specs=model_specs,
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
        model_specs=model_specs,
    )
    child_benefits = calc_child_benefits(
        education=education,
        sex=sex_var,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=model_specs,
    )

    # No care demand / caregiving transfers in counterfactual
    total_income = jnp.maximum(
        total_net_income + child_benefits,
        unemployment_benefits,
    )
    # calculate beginning of period wealth M_t
    wealth = (1 + model_specs["interest_rate"]) * assets_scaled + total_income

    return wealth / model_specs["wealth_unit"]
