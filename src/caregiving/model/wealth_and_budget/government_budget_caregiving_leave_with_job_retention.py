"""Government budget calculations for tax revenue and expenditures for caregiving leave counterfactual."""

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
    household_unemployment_benefits,
    caregiving_leave_top_up,
    model_specs,
):
    """Calculate government budget components (revenue and expenditures) for caregiving leave counterfactual.

    This function includes the caregiving leave top-up as an additional government expenditure.

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
    household_unemployment_benefits : float
        Unemployment benefits paid by government
    caregiving_leave_top_up : float
        Caregiving leave top-up payments paid by government to caregivers
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

    # 4. Calculate government expenditures (benefits)
    # Note: care_benefits_and_costs can be negative (costs) or positive (benefits)
    # We only count positive benefits as government expenditures
    care_benefits = jnp.maximum(care_benefits_and_costs, 0.0)
    government_expenditures = (
        child_benefits
        + care_benefits
        + household_unemployment_benefits
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
