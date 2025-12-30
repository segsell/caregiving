from jax import numpy as jnp

from caregiving.model.shared import SEX, is_retired, is_working
from caregiving.model.wealth_and_budget.government_budget import (
    calc_government_budget_components,
)
from caregiving.model.wealth_and_budget.partner_income import (
    calc_partner_income_after_ssc,
)
from caregiving.model.wealth_and_budget.pensions import (
    calc_pensions_after_ssc,
)
from caregiving.model.wealth_and_budget.tax_and_ssc import calc_net_household_income
from caregiving.model.wealth_and_budget.transfers import (
    calc_care_benefits_and_costs,
    calc_child_benefits,
    calc_inheritance,
    calc_unemployment_benefits,
)
from caregiving.model.wealth_and_budget.wages import calc_labor_income_after_ssc


def budget_constraint(
    period,
    education,
    lagged_choice,  # d_{t-1}
    experience,
    # sex,
    partner_state,
    care_demand,
    mother_dead,
    mother_longer_dead,
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
    partner_income_after_ssc, gross_partner_income, gross_partner_pension = (
        calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=sex_var,
            model_specs=model_specs,
            education=education,
            period=period,
        )
    )

    # Income from lagged choice 0
    retirement_income_after_ssc, gross_retirement_income = calc_pensions_after_ssc(
        experience_years=experience_years,
        sex=sex_var,
        education=education,
        model_specs=model_specs,
    )

    has_partner_int = (partner_state > 0).astype(int)

    # Income lagged choice 1
    household_unemployment_benefits, _own_unemployemnt_benefits = (
        calc_unemployment_benefits(
            assets=assets_scaled,
            education=education,
            sex=sex_var,
            has_partner_int=has_partner_int,
            period=period,
            model_specs=model_specs,
        )
    )

    # Income lagged choice 2
    labor_income_after_ssc, gross_labor_income = calc_labor_income_after_ssc(
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
    total_net_household_income, income_tax_total = calc_net_household_income(
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
    care_benfits_and_costs = calc_care_benefits_and_costs(
        lagged_choice=lagged_choice,
        education=education,
        care_demand=care_demand,
        model_specs=model_specs,
    )

    total_income = jnp.maximum(
        total_net_household_income + child_benefits + care_benfits_and_costs,
        household_unemployment_benefits,
    )

    bequest_from_parent = calc_inheritance(
        period=period,
        lagged_choice=lagged_choice,
        education=education,
        mother_dead=mother_dead,
        mother_longer_dead=mother_longer_dead,
        model_specs=model_specs,
    )

    # # calculate beginning of period wealth M_t
    # assets_begin_of_period = (
    #     (1 + model_specs["interest_rate"]) * assets_scaled
    #     + total_income
    #     + bequest_from_parent
    # )
    total_income = jnp.maximum(
        total_net_household_income, household_unemployment_benefits
    )
    interest_rate = model_specs["interest_rate"]
    interest = interest_rate * assets_scaled
    total_income_plus_interest = total_income + interest + bequest_from_parent

    # calculate beginning of period wealth M_t
    assets_begin_of_period = assets_scaled + total_income_plus_interest

    # Calculate government budget components (revenue and expenditures)
    (
        income_tax_total,
        own_ssc,
        partner_ssc,
        total_tax_revenue,
        government_expenditures,
        net_government_budget,
    ) = calc_government_budget_components(
        household_income_tax_total=income_tax_total,
        was_worker=was_worker,
        was_retired=was_retired,
        gross_labor_income=gross_labor_income,
        gross_retirement_income=gross_retirement_income,
        partner_state=partner_state,
        gross_partner_income=gross_partner_income,
        gross_partner_pension=gross_partner_pension,
        child_benefits=child_benefits,
        care_benefits_and_costs=care_benfits_and_costs,
        household_unemployment_benefits=household_unemployment_benefits,
        model_specs=model_specs,
    )

    aux = {
        "net_hh_income": total_income_plus_interest / model_specs["wealth_unit"],
        "hh_net_income_wo_interest": total_income / model_specs["wealth_unit"],
        "interest": interest / model_specs["wealth_unit"],
        "joint_gross_labor_income": (gross_labor_income + gross_partner_income)
        / model_specs["wealth_unit"],
        "joint_gross_retirement_income": (
            gross_partner_pension + gross_retirement_income
        )
        / model_specs["wealth_unit"],
        "gross_labor_income": gross_labor_income / model_specs["wealth_unit"],
        "gross_retirement_income": gross_retirement_income / model_specs["wealth_unit"],
        # Government budget components
        "income_tax": income_tax_total / model_specs["wealth_unit"],
        "own_ssc": own_ssc / model_specs["wealth_unit"],
        "partner_ssc": partner_ssc / model_specs["wealth_unit"],
        "total_tax_revenue": total_tax_revenue / model_specs["wealth_unit"],
        "government_expenditures": government_expenditures / model_specs["wealth_unit"],
        "net_government_budget": net_government_budget / model_specs["wealth_unit"],
    }

    return assets_begin_of_period / model_specs["wealth_unit"], aux
