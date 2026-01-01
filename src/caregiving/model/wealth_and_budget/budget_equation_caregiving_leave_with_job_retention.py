from jax import numpy as jnp

from caregiving.model.shared import (
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    CARE_DEMAND_AND_OTHER_SUPPLY,
    PARENT_RECENTLY_DEAD,
    SEX,
    had_ft_job_before_caregiving,
    had_no_job_before_caregiving,
    had_pt_job_before_caregiving,
    is_informal_care,
    is_no_care,
    is_part_time,
    is_retired,
    is_unemployed,
    is_working,
)
from caregiving.model.wealth_and_budget.government_budget_caregiving_leave_with_job_retention import (
    calc_government_budget_components_caregiving_leave_with_job_retention,
)
from caregiving.model.wealth_and_budget.partner_income import (
    calc_partner_income_after_ssc,
)
from caregiving.model.wealth_and_budget.pensions import (
    calc_pensions_after_ssc,
)
from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_after_ssc_income_worker,
    calc_net_household_income,
)
from caregiving.model.wealth_and_budget.transfers import (
    calc_care_benefits_and_costs,
    calc_child_benefits,
    calc_inheritance_amount,
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
    gets_inheritance,
    job_before_caregiving,
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
    household_unemployment_benefits, _own_unemployment_benefits = (
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

    # Additional caregiving-leave top-up (after SSC, before income tax)
    caregiving_leave_top_up = calc_caregiving_leave_top_up(
        lagged_choice=lagged_choice,
        education=education,
        job_before_caregiving=job_before_caregiving,
        experience_years=experience_years,
        income_shock_previous_period=income_shock_previous_period,
        sex=sex_var,
        labor_income_after_ssc=labor_income_after_ssc,
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
    # Standard care benefits and costs (remain post-tax transfers)
    care_benefits_and_costs = calc_care_benefits_and_costs(
        lagged_choice=lagged_choice,
        education=education,
        care_demand=care_demand,
        model_specs=model_specs,
    )

    total_income = jnp.maximum(
        total_net_household_income + child_benefits + care_benefits_and_costs,
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
    bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount

    interest_rate = model_specs["interest_rate"]
    interest = interest_rate * assets_scaled
    total_income_plus_interest = total_income + interest + bequest_from_parent

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
        household_unemployment_benefits=household_unemployment_benefits,
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
        "gross_labor_income": gross_labor_income / model_specs["wealth_unit"],
        "gross_retirement_income": gross_retirement_income / model_specs["wealth_unit"],
        "bequest_from_parent": bequest_from_parent / model_specs["wealth_unit"],
        "caregiving_leave_top_up": caregiving_leave_top_up / model_specs["wealth_unit"],
        # Government budget components
        "income_tax": income_tax_total / model_specs["wealth_unit"],
        "own_ssc": own_ssc / model_specs["wealth_unit"],
        "partner_ssc": partner_ssc / model_specs["wealth_unit"],
        "total_tax_revenue": total_tax_revenue / model_specs["wealth_unit"],
        "government_expenditures": government_expenditures / model_specs["wealth_unit"],
        "net_government_budget": net_government_budget / model_specs["wealth_unit"],
    }

    return assets_begin_of_period / model_specs["wealth_unit"], aux


def calc_caregiving_leave_top_up(
    lagged_choice,
    education,
    job_before_caregiving,
    experience_years,
    income_shock_previous_period,
    sex,
    labor_income_after_ssc,
    model_specs,
):
    """Calculate additional wage replacement for caregiving leave.

    This adds on top of the baseline care benefits and costs.

    Policy:
    - job_before_caregiving: 0 = none, 1 = PT, 2 = FT.
    - Only caregivers (current informal care) who are NOT retired are eligible.
    - Previously non-working (0):
        * If currently unemployed: receive a lump-sum (monthly_unemployment_benefits).
        * If currently working (PT/FT): no top up.
    - Previously PT (1):
        * If currently PT: no top up.
        * If currently unemployed: top up to PT net wage.
        * If currently FT: no top up (working more than before).
    - Previously FT (2):
        * If currently FT: no top up.
        * If currently PT: top up so total income equals FT net wage.
        * If currently unemployed: top up to FT net wage.
    - Retired caregivers never receive wage replacement.
    """
    currently_caregiver = is_informal_care(lagged_choice)
    currently_part_time = is_part_time(lagged_choice)
    currently_unemployed = is_unemployed(lagged_choice)
    currently_retired = is_retired(lagged_choice)

    # Base eligibility: caregiver and not retired
    eligible_base = currently_caregiver * (1 - currently_retired)

    prior_none = had_no_job_before_caregiving(job_before_caregiving)
    prior_pt = had_pt_job_before_caregiving(job_before_caregiving)
    prior_ft = had_ft_job_before_caregiving(job_before_caregiving)

    # Construct full-time equivalent annual labor income (gross → after SSC),
    # based on the current experience and income shock.
    gamma_0 = model_specs["gamma_0"][sex, education]
    gamma_1 = model_specs["gamma_1"][sex, education]
    hourly_wage = jnp.exp(
        gamma_0 + gamma_1 * jnp.log(experience_years + 1) + income_shock_previous_period
    )

    # Full-time annual hours and minimum wage
    av_hours_ft = model_specs["av_annual_hours_ft"][sex, education]
    annual_min_wage_ft = model_specs["annual_min_wage_ft"]

    gross_ft_income = hourly_wage * av_hours_ft
    gross_ft_income_min_checked = jnp.maximum(gross_ft_income, annual_min_wage_ft)

    # Net-of-SSC full-time income
    net_ft_income = calc_after_ssc_income_worker(gross_ft_income_min_checked)

    # Net PT income (full PT schedule)
    av_hours_pt = model_specs["av_annual_hours_pt"][sex, education]
    annual_min_wage_pt = model_specs["annual_min_wage_pt"][sex, education]
    gross_pt_income = hourly_wage * av_hours_pt
    gross_pt_income_min_checked = jnp.maximum(gross_pt_income, annual_min_wage_pt)
    net_pt_income = calc_after_ssc_income_worker(gross_pt_income_min_checked)

    # Masks for specific cases
    mask_prior_none_unemp = eligible_base * prior_none * currently_unemployed
    mask_prior_pt_unemp = eligible_base * prior_pt * currently_unemployed
    mask_prior_ft_unemp = eligible_base * prior_ft * currently_unemployed
    mask_prior_ft_pt = eligible_base * prior_ft * currently_part_time

    # Top ups:
    # - Prior none, now unemployed: lump-sum monthly_unemployment_benefits.
    topup_prior_none = (
        model_specs["monthly_unemployment_benefits"] * 12.0 * mask_prior_none_unemp
    )

    # - Prior PT, now unemployed: top up to PT net wage.
    topup_prior_pt = net_pt_income * mask_prior_pt_unemp

    # - Prior FT, now unemployed: top up to FT net wage.
    topup_prior_ft_unemp = net_ft_income * mask_prior_ft_unemp

    # - Prior FT, now PT: top up so total income equals FT net wage.
    ft_gap = jnp.maximum(net_ft_income - labor_income_after_ssc, 0.0)
    topup_prior_ft_pt = ft_gap * mask_prior_ft_pt

    wage_replacement_annual = (
        topup_prior_none + topup_prior_pt + topup_prior_ft_unemp + topup_prior_ft_pt
    )

    return wage_replacement_annual
