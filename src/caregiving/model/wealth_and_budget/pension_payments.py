from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_after_ssc_income_pensioneer,
)


def calc_pensions_after_ssc(
    pension_points,
    model_specs,
):
    """Calculate the pension income after SSC contributions."""
    retirement_income_gross = calc_gross_pension_income(
        pension_points=pension_points,
        model_specs=model_specs,
    )
    retirement_income = calc_after_ssc_income_pensioneer(retirement_income_gross)

    return retirement_income, retirement_income_gross


def calc_gross_pension_income(pension_points, model_specs):
    """Calculate the gross pension income."""
    retirement_income_gross = model_specs["annual_pension_point_value"] * pension_points
    return retirement_income_gross
