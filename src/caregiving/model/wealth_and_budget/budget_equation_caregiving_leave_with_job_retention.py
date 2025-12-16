from jax import numpy as jnp

from caregiving.model.shared import (
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    CARE_DEMAND_AND_OTHER_SUPPLY,
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
    job_before_caregiving,
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

    # Additional caregiving-leave top-up (after SSC, before income tax)
    caregiving_leave_top_up = calc_caregiving_leave_top_up(
        lagged_choice=lagged_choice,
        education=education,
        job_before_caregiving=job_before_caregiving,
        experience_years=experience_years,
        income_shock_previous_period=income_shock_previous_period,
        sex=sex_var,
        labor_income_after_ssc=labor_income_after_ssc,
        options=options,
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
    # Standard care benefits and costs (remain post-tax transfers)
    care_benfits_and_costs = calc_care_benefits_and_costs(
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


def calc_caregiving_leave_top_up(
    lagged_choice,
    education,
    job_before_caregiving,
    experience_years,
    income_shock_previous_period,
    sex,
    labor_income_after_ssc,
    options,
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
    gamma_0 = options["gamma_0"][sex, education]
    gamma_1 = options["gamma_1"][sex, education]
    hourly_wage = jnp.exp(
        gamma_0 + gamma_1 * jnp.log(experience_years + 1) + income_shock_previous_period
    )

    # Full-time annual hours and minimum wage
    av_hours_ft = options["av_annual_hours_ft"][sex, education]
    annual_min_wage_ft = options["annual_min_wage_ft"]

    gross_ft_income = hourly_wage * av_hours_ft
    gross_ft_income_min_checked = jnp.maximum(gross_ft_income, annual_min_wage_ft)

    # Net-of-SSC full-time income
    net_ft_income = calc_after_ssc_income_worker(gross_ft_income_min_checked)

    # Net PT income (full PT schedule)
    av_hours_pt = options["av_annual_hours_pt"][sex, education]
    annual_min_wage_pt = options["annual_min_wage_pt"][sex, education]
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
        options["monthly_unemployment_benefits"] * 12.0 * mask_prior_none_unemp
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
