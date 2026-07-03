"""Full-Beirat-no-total-cap caregiving leave budget.

Same structure as the Full Beirat budget (Pflegegeld + 65% top-up) but with the
3-year cumulative leave cap removed. The 1-year full-leave sub-cap is retained
via ``full_leave_year_used``; ``years_leave_used_total`` is no longer a state.
"""

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
    calc_caregiving_leave_top_up_full_beirat_no_total_cap,
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
    calc_care_benefits_and_costs,
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
    partner_state,
    care_demand,
    mother_dead,
    job_before_caregiving,
    full_leave_year_used,
    asset_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    model_specs,
):
    """Full-Beirat-no-total-cap budget: Pflegegeld + gated 65% top-up.

    Same eligibility structure as Full Beirat with the 3-year cumulative cap
    removed. Partial leave is uncapped; full leave still capped at 1 year via
    ``full_leave_year_used``.
    """
    sex_var = SEX

    assets_scaled = asset_end_of_previous_period * model_specs["wealth_unit"]

    experience_years = construct_experience_years(
        float_experience=experience,
        period=period,
        is_retired=is_retired(lagged_choice),
        model_specs=model_specs,
    )

    partner_income_after_ssc, gross_partner_income, gross_partner_pension = (
        calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=sex_var,
            model_specs=model_specs,
            education=education,
            period=period,
        )
    )

    retirement_income_after_ssc, gross_retirement_income = calc_pensions_after_ssc(
        pension_points=experience_years,
        model_specs=model_specs,
    )

    has_partner_int = (partner_state > 0).astype(int)

    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=assets_scaled,
        education=education,
        sex=sex_var,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=model_specs,
    )

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

    caregiving_leave_top_up = calc_caregiving_leave_top_up_full_beirat_no_total_cap(
        lagged_choice=lagged_choice,
        education=education,
        job_before_caregiving=job_before_caregiving,
        experience_years=experience_years,
        income_shock_previous_period=income_shock_for_labor,
        sex=sex_var,
        labor_income_after_ssc=labor_income_after_ssc,
        full_leave_year_used=full_leave_year_used,
        model_specs=model_specs,
    )

    was_worker = is_working(lagged_choice)
    was_retired = is_retired(lagged_choice)

    own_income_for_tax = (
        was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc
    )
    total_net_household_income, income_tax_total, income_tax_single = (
        calc_net_household_income(
            own_income=own_income_for_tax,
            partner_income=partner_income_after_ssc,
            has_partner_int=has_partner_int,
            model_specs=model_specs,
            progression_income=caregiving_leave_top_up,
        )
    )

    _, tax_without_progression, _ = calc_net_household_income(
        own_income=own_income_for_tax,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=model_specs,
        progression_income=0.0,
    )
    tax_increase_from_progression = income_tax_total - tax_without_progression
    normal_leave_net_cost = caregiving_leave_top_up - tax_increase_from_progression

    own_income_after_ssc = own_income_for_tax + caregiving_leave_top_up

    child_benefits = calc_child_benefits(
        education=education,
        sex=sex_var,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=model_specs,
    )

    care_benefits_and_costs = calc_care_benefits_and_costs(
        period=period,
        lagged_choice=lagged_choice,
        model_specs=model_specs,
    )

    formal_care = is_formal_care(lagged_choice)
    annual_formal_care_costs_agent = (
        -model_specs["formal_care_costs"][period] * formal_care * 12
    )

    household_net_income_before_floor = total_net_household_income + child_benefits
    total_income = jnp.maximum(
        household_net_income_before_floor,
        household_unemployment_benefits,
    )

    unemployment_transfer_paid = jnp.maximum(
        0.0,
        household_unemployment_benefits - household_net_income_before_floor,
    )

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
        total_income + interest + care_benefits_and_costs + bequest_from_parent
    )

    assets_begin_of_period = assets_scaled + total_income_plus_interest

    (
        income_tax_total,
        own_ssc,
        partner_ssc,
        total_tax_revenue,
        government_expenditures,
        net_government_budget,
    ) = calc_government_budget_components_caregiving_leave_with_job_retention(
        household_income_tax_total=income_tax_total,
        was_worker=was_worker,
        was_retired=was_retired,
        gross_labor_income=gross_labor_income,
        gross_retirement_income=gross_retirement_income,
        partner_state=partner_state,
        gross_partner_income=gross_partner_income,
        gross_partner_pension=gross_partner_pension,
        child_benefits=child_benefits,
        care_benefits_and_costs=care_benefits_and_costs,
        unemployment_transfer_paid=unemployment_transfer_paid,
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
        "labor_income_after_ssc": labor_income_after_ssc / model_specs["wealth_unit"],
        "retirement_income_after_ssc": retirement_income_after_ssc
        / model_specs["wealth_unit"],
        "own_income_after_ssc": own_income_after_ssc / model_specs["wealth_unit"],
        "own_income_for_tax": own_income_for_tax / model_specs["wealth_unit"],
        "partner_income_after_ssc": partner_income_after_ssc
        / model_specs["wealth_unit"],
        "total_net_household_income": total_net_household_income
        / model_specs["wealth_unit"],
        "household_net_income_before_floor": household_net_income_before_floor
        / model_specs["wealth_unit"],
        "child_benefits": child_benefits / model_specs["wealth_unit"],
        "care_benefits_and_costs": care_benefits_and_costs / model_specs["wealth_unit"],
        "formal_care_costs": annual_formal_care_costs_agent
        / model_specs["wealth_unit"],
        "household_unemployment_benefits": household_unemployment_benefits
        / model_specs["wealth_unit"],
        "unemployment_transfer_paid": unemployment_transfer_paid
        / model_specs["wealth_unit"],
        "tax_increase_from_progression": tax_increase_from_progression
        / model_specs["wealth_unit"],
        "tax_without_progression": tax_without_progression / model_specs["wealth_unit"],
        "normal_leave_net_cost": normal_leave_net_cost / model_specs["wealth_unit"],
        "income_tax": income_tax_total / model_specs["wealth_unit"],
        "income_tax_single": income_tax_single / model_specs["wealth_unit"],
        "own_ssc": own_ssc / model_specs["wealth_unit"],
        "partner_ssc": partner_ssc / model_specs["wealth_unit"],
        "total_tax_revenue": total_tax_revenue / model_specs["wealth_unit"],
        "government_expenditures": government_expenditures / model_specs["wealth_unit"],
        "net_government_budget": net_government_budget / model_specs["wealth_unit"],
    }

    return assets_begin_of_period / model_specs["wealth_unit"], aux
