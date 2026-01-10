from jax import numpy as jnp

from caregiving.model.experience_baseline_model import construct_experience_years
from caregiving.model.shared import (
    PARENT_RECENTLY_DEAD,
    SEX,
    is_formal_care,
    is_retired,
    is_working,
)
from caregiving.model.wealth_and_budget.caregiving_leave_top_up import (
    calc_caregiving_leave_top_up,
)
from caregiving.model.wealth_and_budget.government_budget_caregiving_leave_with_job_retention import (  # noqa: E501
    calc_government_budget_components_caregiving_leave_with_job_retention,
)
from caregiving.model.wealth_and_budget.partner_income import (
    calc_partner_income_after_ssc,
)
from caregiving.model.wealth_and_budget.pension_payments import (
    calc_pensions_after_ssc,
)
from caregiving.model.wealth_and_budget.tax_and_ssc import calc_net_household_income
from caregiving.model.wealth_and_budget.transfers import (
    calc_child_benefits,
    calc_inheritance_amount,
    calc_unemployment_benefits,
    draw_inheritance_outcome,
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
    job_before_caregiving,
    asset_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    model_specs,
):
    sex_var = SEX

    assets_scaled = asset_end_of_previous_period * model_specs["wealth_unit"]

    # Recalculate experience
    experience_years = construct_experience_years(
        float_experience=experience,
        period=period,
        is_retired=is_retired(lagged_choice),
        model_specs=model_specs,
    )

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
        pension_points=experience_years,
        model_specs=model_specs,
    )

    has_partner_int = (partner_state > 0).astype(int)

    # Income lagged choice 1
    household_unemployment_benefits, own_unemployment_benefits = (
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
    # For period 0, use mean income shock (0.0) since there's no previous period
    income_shock_for_labor = jnp.where(
        period == 0,
        model_specs["income_shock_mean"],
        income_shock_previous_period,
    )
    labor_income_after_ssc, gross_labor_income = calc_labor_income_after_ssc(
        lagged_choice=lagged_choice,
        experience_years=experience_years,
        education=education,
        sex=sex_var,
        income_shock=income_shock_for_labor,
        model_specs=model_specs,
    )

    # Additional caregiving-leave top-up (after SSC, before income tax)
    caregiving_leave_top_up = calc_caregiving_leave_top_up(
        lagged_choice=lagged_choice,
        education=education,
        job_before_caregiving=job_before_caregiving,
        experience_years=experience_years,
        income_shock_previous_period=income_shock_for_labor,
        sex=sex_var,
        labor_income_after_ssc=labor_income_after_ssc,
        household_unemployment_benefits=household_unemployment_benefits,
        model_specs=model_specs,
    )

    # Select relevant income
    # bools of last period decision: income is paid in following period!
    was_worker = is_working(lagged_choice)
    was_retired = is_retired(lagged_choice)

    # Aggregate over choice own income, including caregiving leave top-up
    own_income_after_ssc = (
        was_worker * labor_income_after_ssc
        + was_retired * retirement_income_after_ssc
        + caregiving_leave_top_up
    )

    # Calculate total household net income (taxes on earnings + wage replacement)
    total_net_household_income, income_tax_total, income_tax_single = (
        calc_net_household_income(
            own_income=own_income_after_ssc,
            partner_income=partner_income_after_ssc,
            has_partner_int=has_partner_int,
            model_specs=model_specs,
        )
    )

    child_benefits = calc_child_benefits(
        education=education,
        sex=sex_var,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=model_specs,
    )

    # # Formal care costs only (no informal care cash benefits, as caregiving lea
    # ve top-up replaces them)
    formal_care = is_formal_care(lagged_choice)
    annual_formal_care_costs = (
        -model_specs["formal_care_costs"] * formal_care * 12 * 0.5
    )

    total_income = jnp.maximum(
        total_net_household_income + child_benefits,
        household_unemployment_benefits,
    )

    # Only compute inheritance if mother recently died this period (state 1)
    mother_died_recently = mother_dead == PARENT_RECENTLY_DEAD
    inheritance_amount = calc_inheritance_amount(
        period=period,
        lagged_choice=lagged_choice,
        education=education,
        model_specs=model_specs,
    )
    gets_inheritance = draw_inheritance_outcome(
        period=period,
        lagged_choice=lagged_choice,
        education=education,
        asset_end_of_previous_period=asset_end_of_previous_period,
        model_specs=model_specs,
    )
    bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount

    interest_rate = model_specs["interest_rate"]
    interest = interest_rate * assets_scaled
    total_income_plus_interest = (
        total_income + interest + annual_formal_care_costs + bequest_from_parent
    )

    # Calculate beginning of period wealth M_t
    assets_begin_of_period = assets_scaled + total_income_plus_interest

    # Calculate government budget components (revenue and expenditures)
    (
        income_tax_total,
        own_ssc,
        partner_ssc,
        total_tax_revenue,
        government_expenditures,
        net_government_budget,
    ) = calc_government_budget_components_caregiving_leave_with_job_retention(
        # single_income_tax_total=income_tax_single,
        household_income_tax_total=income_tax_total,
        was_worker=was_worker,
        was_retired=was_retired,
        gross_labor_income=gross_labor_income,
        gross_retirement_income=gross_retirement_income,
        partner_state=partner_state,
        gross_partner_income=gross_partner_income,
        gross_partner_pension=gross_partner_pension,
        child_benefits=child_benefits,
        care_benefits_and_costs=annual_formal_care_costs,
        # Only formal care costs (no benefits
        household_unemployment_benefits=household_unemployment_benefits,
        # own_unemployment_benefits=own_unemployment_benefits,
        caregiving_leave_top_up=caregiving_leave_top_up,
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
        "gross_partner_income": gross_partner_income / model_specs["wealth_unit"],
        "gross_partner_pension": gross_partner_pension / model_specs["wealth_unit"],
        "gross_labor_income": gross_labor_income / model_specs["wealth_unit"],
        "gross_retirement_income": gross_retirement_income / model_specs["wealth_unit"],
        "bequest_from_parent": bequest_from_parent / model_specs["wealth_unit"],
        "gets_inheritance": gets_inheritance,
        "caregiving_leave_top_up": caregiving_leave_top_up / model_specs["wealth_unit"],
        "own_income_after_ssc": own_income_after_ssc / model_specs["wealth_unit"],
        # # "care_benefits_and_costs": care_benfits_and_costs / model_specs["wealth_u
        # nit"],
        "child_benefits": child_benefits / model_specs["wealth_unit"],
        "household_unemployment_benefits": household_unemployment_benefits
        / model_specs["wealth_unit"],
        # Government budget components
        "income_tax": income_tax_total / model_specs["wealth_unit"],
        "income_tax_single": income_tax_single / model_specs["wealth_unit"],
        "own_ssc": own_ssc / model_specs["wealth_unit"],
        "partner_ssc": partner_ssc / model_specs["wealth_unit"],
        "total_tax_revenue": total_tax_revenue / model_specs["wealth_unit"],
        "government_expenditures": government_expenditures / model_specs["wealth_unit"],
        "net_government_budget": net_government_budget / model_specs["wealth_unit"],
    }

    return assets_begin_of_period / model_specs["wealth_unit"], aux
