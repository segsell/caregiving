from caregiving.model.shared import (
    is_formal_care,
    is_informal_care,
    is_no_care,
)


def calc_child_benefits(sex, education, has_partner_int, period, model_specs):
    """Calculate the child benefits."""
    n_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]

    return n_children * model_specs["monthly_child_benefits"] * 12


def calc_unemployment_benefits(
    assets, sex, education, has_partner_int, period, model_specs
):
    # Unemployment benefits means test
    means_test = assets < model_specs["unemployment_wealth_thresh"]

    # Unemployment benefits for children living in the household
    n_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]
    unemployment_benefits_children = (
        n_children * model_specs["annual_child_unemployment_benefits"]
    )

    # Unemployment benefits for adults living in the household
    unemployment_benefits_adults = (1 + has_partner_int) * model_specs[
        "annual_unemployment_benefits"
    ]
    # For housing, second adult gets only half
    unemployment_benefits_housing = (1 + 0.5 * has_partner_int) * model_specs[
        "annual_unemployment_benefits_housing"
    ]

    # Total unemployment benefits
    total_unemployment_benefits = (
        unemployment_benefits_adults
        + unemployment_benefits_children
        + unemployment_benefits_housing
    )

    # reduced benefits for assets slightly above threshold
    reduced_benefits_threshhold = (
        model_specs["unemployment_wealth_thresh"] + total_unemployment_benefits
    )
    reduced_benefits_means_test = (1 - means_test) * (
        assets < reduced_benefits_threshhold
    )
    reduced_benefits = reduced_benefits_threshhold - assets

    unemployment_benefits = (
        means_test * total_unemployment_benefits
        + reduced_benefits_means_test * reduced_benefits
    )

    return unemployment_benefits


def calc_care_benefits_and_costs(lagged_choice, education, care_demand, model_specs):
    """Calculate the care benefits and costs."""

    informal_care_solo = is_informal_care(lagged_choice)
    formal_care = is_formal_care(lagged_choice)

    # # Care benefits
    # care_benefits = options["care_benefits"][education, has_sister]

    # # Care costs
    # care_costs = options["care_costs"][education, has_sister]

    annual_care_benefits = model_specs["informal_care_cash_benefits"] * 12
    annual_care_benefits_weighted = annual_care_benefits * informal_care_solo

    annual_care_costs_weighted = (
        model_specs["formal_care_costs"] * formal_care * 12 * 0.5
    )

    return annual_care_benefits_weighted - annual_care_costs_weighted
