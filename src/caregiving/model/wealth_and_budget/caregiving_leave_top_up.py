"""Calculate caregiving leave top-up benefits."""

from jax import numpy as jnp

from caregiving.model.shared import (
    had_ft_job_before_caregiving,
    had_no_job_before_caregiving,
    had_pt_job_before_caregiving,
    is_informal_care,
    is_part_time,
    is_retired,
    is_unemployed,
)
from caregiving.model.wealth_and_budget.tax_and_ssc import calc_after_ssc_income_worker
from caregiving.model.wealth_and_budget.wages import calc_hourly_wage


def calc_full_caregiving_leave_top_up(
    lagged_choice,
    education,
    job_before_caregiving,
    experience_years,
    income_shock_previous_period,
    sex,
    labor_income_after_ssc,
    household_unemployment_benefits,
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
    hourly_wage = calc_hourly_wage(
        sex=sex,
        education=education,
        experience_years=experience_years,
        income_shock=income_shock_previous_period,
        model_specs=model_specs,
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
    # - Prior none, now unemployed: no top-up needed.
    # #   (Unemployment benefits are already handled via jnp.maximum() in budget
    # constraint,
    #   so adding them here would cause double-counting)
    topup_prior_none = 0.0 * mask_prior_none_unemp

    # - Prior PT, now unemployed: top up to PT net wage, MINUS unemployment benefits
    #   they're already receiving
    topup_prior_pt = (
        jnp.maximum(net_pt_income - household_unemployment_benefits, 0.0)
        * mask_prior_pt_unemp
    )

    # - Prior FT, now unemployed: top up to FT net wage, MINUS unemployment benefits
    #   they're already receiving
    topup_prior_ft_unemp = (
        jnp.maximum(net_ft_income - household_unemployment_benefits, 0.0)
        * mask_prior_ft_unemp
    )

    # - Prior FT, now PT: top up so total income equals FT net wage.
    #   (This case doesn't involve unemployment benefits, so no change needed)
    ft_gap = jnp.maximum(net_ft_income - labor_income_after_ssc, 0.0)
    topup_prior_ft_pt = ft_gap * mask_prior_ft_pt

    wage_replacement_annual = (
        topup_prior_none + topup_prior_pt + topup_prior_ft_unemp + topup_prior_ft_pt
    )

    return wage_replacement_annual


def calc_caregiving_leave_top_up(
    lagged_choice,
    education,
    job_before_caregiving,
    experience_years,
    income_shock_previous_period,
    sex,
    labor_income_after_ssc,
    household_unemployment_benefits,
    model_specs,
):
    """Calculate additional wage replacement for caregiving leave (65%  # noqa: E501
    replacement with bounds).

    This adds on top of the baseline care benefits and costs.

    Policy:
    - job_before_caregiving: 0 = none, 1 = PT, 2 = FT.
    - Only caregivers (current informal care) who are NOT retired are eligible.
    - Replacement rate: 65% of previous net wage
    - Lower bound: caregiving_leave_benefit_lower_bound (annualized from monthly)
    - Upper bound: caregiving_leave_benefit_upper_bound (annualized from monthly)
    - Previously non-working (0):
        * If currently unemployed: no top up (unemployment benefits handled elsewhere).
        * If currently working (PT/FT): no top up.
    - Previously PT (1):
        * If currently PT: no top up.
        * If currently unemployed: top up to 65% of PT net wage (with bounds),
            MINUS unemployment benefits.
        * If currently FT: no top up (working more than before).
    - Previously FT (2):
        * If currently FT: no top up.
        * If currently PT: top up to 65% of FT net wage (with bounds),
            MINUS current PT labor income.
        * If currently unemployed: top up to 65% of FT net wage (with bounds),
            MINUS unemployment benefits.
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

    # Get replacement rate and bounds from model_specs
    replacement_rate = model_specs["caregiving_leave_benefit_replacement_rate"]
    # Bounds are monthly in specs, convert to annual
    lower_bound_annual = model_specs["caregiving_leave_benefit_lower_bound"] * 12
    upper_bound_annual = model_specs["caregiving_leave_benefit_upper_bound"] * 12

    # Construct full-time equivalent annual labor income (gross → after SSC),
    # based on the current experience and income shock.
    hourly_wage = calc_hourly_wage(
        sex=sex,
        education=education,
        experience_years=experience_years,
        income_shock=income_shock_previous_period,
        model_specs=model_specs,
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

    # Calculate 65% replacement with bounds for PT and FT
    # # Apply bounds: benefit = min(max(replacement_rate * net_wage, lower_bound)
    # , upper_bound)
    benefit_pt = jnp.clip(
        replacement_rate * net_pt_income,
        lower_bound_annual,
        upper_bound_annual,
    )
    benefit_ft = jnp.clip(
        replacement_rate * net_ft_income,
        lower_bound_annual,
        upper_bound_annual,
    )

    # Masks for specific cases
    mask_prior_none_unemp = eligible_base * prior_none * currently_unemployed
    mask_prior_pt_unemp = eligible_base * prior_pt * currently_unemployed
    mask_prior_ft_unemp = eligible_base * prior_ft * currently_unemployed
    mask_prior_ft_pt = eligible_base * prior_ft * currently_part_time

    # Top ups:
    # - Prior none, now unemployed: no top-up needed.
    # #   (Unemployment benefits are already handled via jnp.maximum() in budget
    # constraint,
    #   so adding them here would cause double-counting)
    topup_prior_none = 0.0 * mask_prior_none_unemp

    # - Prior PT, now unemployed: top up to 65% of PT net wage (with bounds),
    #   MINUS unemployment benefits they're already receiving
    topup_prior_pt = (
        jnp.maximum(benefit_pt - household_unemployment_benefits, 0.0)
        * mask_prior_pt_unemp
    )

    # - Prior FT, now unemployed: top up to 65% of FT net wage (with bounds),
    #   MINUS unemployment benefits they're already receiving
    topup_prior_ft_unemp = (
        jnp.maximum(benefit_ft - household_unemployment_benefits, 0.0)
        * mask_prior_ft_unemp
    )

    # - Prior FT, now PT: top up to 65% of FT net wage (with bounds),
    #   MINUS current PT labor income
    topup_prior_ft_pt = (
        jnp.maximum(benefit_ft - labor_income_after_ssc, 0.0) * mask_prior_ft_pt
    )

    wage_replacement_annual = (
        topup_prior_none + topup_prior_pt + topup_prior_ft_unemp + topup_prior_ft_pt
    )

    return wage_replacement_annual
