"""Government budget for caregiving leave counterfactuals (job retention).

Two policies:
- Full leave (Norwegian-style): 100% wage replacement; benefit taxable, no SSC.
  Unemployment is an income floor (only top-up to floor is paid). Use
  calc_government_budget_components_full_caregiving_leave_with_job_retention.
- Partial leave (65%, German Elterngeld-style): 65% with bounds; benefit tax-free
  with Progressionsvorbehalt, no SSC. Unemployment is an income floor. Use
  calc_government_budget_components_caregiving_leave_with_job_retention.
"""

from jax import numpy as jnp

from caregiving.model.shared import PARTNER_RETIRED, PARTNER_WORKING
from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_health_ltc_contr,
    calc_pension_unempl_contr,
)


def calc_government_budget_components_caregiving_leave_with_job_retention(
    household_income_tax_total,
    was_worker,
    was_retired,
    gross_labor_income,
    gross_retirement_income,
    partner_state,
    gross_partner_income,
    gross_partner_pension,
    child_benefits,
    care_benefits_and_costs,
    unemployment_transfer_paid,
    caregiving_leave_top_up,
    model_specs,
):
    """Calculate government budget components for partial caregiving leave  # noqa: E501
    (65%, German Elterngeld-style).

    Tax is computed with Progressionsvorbehalt in the budget equation (benefit
    tax-free but raises average rate on other income). Unemployment is an income
    floor; only the top-up to the floor is paid (unemployment_transfer_paid).

    Returns tax revenue, government expenditures (benefits), and net budget.

    Parameters
    ----------
    household_income_tax_total : float
        Total income tax paid by household (from calc_net_household_income)
    was_worker : bool
        Indicator if agent was working
    was_retired : bool
        Indicator if agent was retired
    gross_labor_income : float
        Gross labor income
    gross_retirement_income : float
        Gross retirement income
    partner_state : int
        Partner state (0=no partner, 1=working, 2=retired)
    gross_partner_income : float
        Gross partner labor income
    gross_partner_pension : float
        Gross partner pension income
    child_benefits : float
        Child benefits paid by government
    care_benefits_and_costs : float
        Care benefits (positive) and costs (negative) paid by government
    unemployment_transfer_paid : float
        Actual transfer paid to top up household to the unemployment floor.
        Computed in the budget equation.
    caregiving_leave_top_up : float
        65% caregiving leave benefit (tax-free with Progressionsvorbehalt;
        not subject to SSC).
    model_specs : dict
        Model specifications

    Returns
    -------
    household_income_tax_total : float
        Total income tax paid by household
    own_ssc : float
        Own social security contributions
    partner_ssc : float
        Partner social security contributions
    total_tax_revenue : float
        Total tax revenue (income tax + SSC)
    government_expenditures : float
        Total government expenditures (benefits, including caregiving leave top-up)
    net_government_budget : float
        Net government budget (revenue - expenditures)
    """

    # 1. Calculate own SSC contributions
    own_gross_income = (
        was_worker * gross_labor_income + was_retired * gross_retirement_income
    )
    own_ssc = was_worker * (
        calc_pension_unempl_contr(own_gross_income)
        + calc_health_ltc_contr(own_gross_income)
    ) + was_retired * calc_health_ltc_contr(own_gross_income)

    # 2. Calculate partner SSC contributions
    partner_gross_income = (partner_state == PARTNER_WORKING) * gross_partner_income + (
        partner_state == PARTNER_RETIRED
    ) * gross_partner_pension
    partner_ssc = (partner_state == PARTNER_WORKING) * (
        calc_pension_unempl_contr(partner_gross_income)
        + calc_health_ltc_contr(partner_gross_income)
    ) + (partner_state == PARTNER_RETIRED) * calc_health_ltc_contr(partner_gross_income)

    # 3. Calculate total tax revenue
    total_tax_revenue = household_income_tax_total + own_ssc + partner_ssc

    # 4. Government expenditures. Unemployment is an income floor (only top-up paid).
    care_benefits = jnp.maximum(care_benefits_and_costs, 0.0)
    government_expenditures = (
        child_benefits
        + care_benefits
        + unemployment_transfer_paid
        + caregiving_leave_top_up
    )

    # 5. Calculate net government budget
    net_government_budget = total_tax_revenue - government_expenditures

    return (
        household_income_tax_total,
        own_ssc,
        partner_ssc,
        total_tax_revenue,
        government_expenditures,
        net_government_budget,
    )


def calc_government_budget_components_full_caregiving_leave_with_job_retention(
    household_income_tax_total,
    was_worker,
    was_retired,
    gross_labor_income,
    gross_retirement_income,
    partner_state,
    gross_partner_income,
    gross_partner_pension,
    child_benefits,
    care_benefits_and_costs,
    unemployment_transfer_paid,
    full_caregiving_leave_benefit,
    model_specs,
):
    """Calculate government budget for full caregiving leave (Norwegian-style).

    Full leave: 100% wage replacement. Benefit is taxable (included in
    household_income_tax_total via budget equation), not subject to SSC.

    Unemployment is an **income floor**: the government pays only the top-up
    to reach the floor (max(0, floor - income)). The budget equation computes
    that transfer and passes it as unemployment_transfer_paid. No benefit
    "replaces" another; when income (including leave) is above the floor,
    the transfer is zero.

    Government expenditures use the **gross** full leave benefit (actual cash
    paid) so that net_government_budget = revenue - expenditure is correct. The
    net cost of the leave (gross minus income tax attributable) is computed in
    the budget equation and stored in aux for reporting only.

    Returns tax revenue, government expenditures (benefits), and net budget.

    Parameters
    ----------
    household_income_tax_total : float
        Total income tax paid by household (from calc_net_household_income;
        includes tax on full leave benefit).
    was_worker : bool
        Indicator if agent was working
    was_retired : bool
        Indicator if agent was retired
    gross_labor_income : float
        Gross labor income
    gross_retirement_income : float
        Gross retirement income
    partner_state : int
        Partner state (0=no partner, 1=working, 2=retired)
    gross_partner_income : float
        Gross partner labor income
    gross_partner_pension : float
        Gross partner pension income
    child_benefits : float
        Child benefits paid by government
    care_benefits_and_costs : float
        Care benefits (positive) and costs (negative) paid by government
    unemployment_transfer_paid : float
        Actual transfer paid to top up household to the unemployment floor
        (max(0, floor - (total_net_household_income + child_benefits))).
        Computed in the budget equation.
    full_caregiving_leave_benefit : float
        Gross full caregiving leave benefit (Norwegian-style: 100% wage replacement;
        taxable, not subject to SSC). Used for government_expenditures (gross).
    full_leave_net_cost : float
        Net fiscal cost of full leave (gross minus tax attributable). Not used
        in this function; kept for API compatibility. Reported in aux by the
        budget equation.
    model_specs : dict
        Model specifications

    Returns
    -------
    household_income_tax_total : float
        Total income tax paid by household
    own_ssc : float
        Own social security contributions
    partner_ssc : float
        Partner social security contributions
    total_tax_revenue : float
        Total tax revenue (income tax + SSC)
    government_expenditures : float
        Total government expenditures (benefits; full leave at gross = cash paid)
    net_government_budget : float
        Net government budget (revenue - expenditures)
    """

    # 1. Calculate own SSC contributions
    own_gross_income = (
        was_worker * gross_labor_income + was_retired * gross_retirement_income
    )
    own_ssc = was_worker * (
        calc_pension_unempl_contr(own_gross_income)
        + calc_health_ltc_contr(own_gross_income)
    ) + was_retired * calc_health_ltc_contr(own_gross_income)

    # 2. Calculate partner SSC contributions
    partner_gross_income = (partner_state == PARTNER_WORKING) * gross_partner_income + (
        partner_state == PARTNER_RETIRED
    ) * gross_partner_pension
    partner_ssc = (partner_state == PARTNER_WORKING) * (
        calc_pension_unempl_contr(partner_gross_income)
        + calc_health_ltc_contr(partner_gross_income)
    ) + (partner_state == PARTNER_RETIRED) * calc_health_ltc_contr(partner_gross_income)

    # 3. Total tax revenue (income tax + SSC). household_income_tax_total
    # already includes tax on the full leave benefit.
    total_ssc_and_tax_revenue = household_income_tax_total + own_ssc + partner_ssc

    # 4. Government expenditures: actual cash paid out. Unemployment is an
    # income floor; only the top-up to the floor is paid (passed from budget
    # equation). Full leave at gross (no double counting of tax; net cost in aux).
    care_benefits = jnp.maximum(care_benefits_and_costs, 0.0)
    government_expenditures = (
        child_benefits
        + care_benefits
        + unemployment_transfer_paid
        + full_caregiving_leave_benefit
    )

    # 5. Net government budget (revenue - expenditure)
    net_government_budget = total_ssc_and_tax_revenue - government_expenditures

    return (
        household_income_tax_total,
        own_ssc,
        partner_ssc,
        total_ssc_and_tax_revenue,
        government_expenditures,
        net_government_budget,
    )
