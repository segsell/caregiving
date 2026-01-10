from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_after_ssc_income_pensioneer,
    calc_after_ssc_income_worker,
)

# def calc_partner_income_after_ssc(partner_state, sex, model_specs, education, period):
#     """Calculate the partner income after deduction of ssc."""
#     partner_wage_year = model_specs["annual_partner_wage"][sex, education, period]
#     partner_wage_after_ssc = calc_after_ssc_income_worker(
#         gross_wage=partner_wage_year
#     )
#     partner_pension_year = model_specs["annual_partner_pension"][sex, education]
#     partner_pension_after_ssc = calc_after_ssc_income_pensioneer(
#         gross_pesnion=partner_pension_year
#     )

#     working_age_partner = partner_state == 1
#     retired_partner = partner_state == 2  # noqa: PLR2004
#     partner_income = (
#         working_age_partner * partner_wage_after_ssc
#         + retired_partner * partner_pension_after_ssc
#     )
#     return partner_income


def calc_partner_income_after_ssc(partner_state, sex, model_specs, education, period):
    """Calculate the partner income after deduction of ssc."""
    # Calculate wage of partner
    gross_partner_wage = model_specs["annual_partner_wage"][sex, education, period]
    partner_wage_after_ssc = calc_after_ssc_income_worker(gross_wage=gross_partner_wage)

    # Calculate pension of partner
    gross_partner_pension = model_specs["annual_partner_pension"][sex, education]
    partner_pension_after_ssc = calc_after_ssc_income_pensioneer(
        gross_pesnion=gross_partner_pension
    )

    # Aggregate partner income based on state
    working_age_partner = partner_state == 1
    retired_partner = partner_state == 2  # noqa: PLR2004

    partner_income_after_ssc = (
        working_age_partner * partner_wage_after_ssc
        + retired_partner * partner_pension_after_ssc
    )

    return partner_income_after_ssc, gross_partner_wage, gross_partner_pension
