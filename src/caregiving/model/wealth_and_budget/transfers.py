from caregiving.model.shared import (
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    CARE_DEMAND_AND_OTHER_SUPPLY,
    is_informal_care,
    is_no_care,
)


def calc_child_benefits(sex, education, has_partner_int, period, options):
    """Calculate the child benefits."""
    n_children = options["children_by_state"][sex, education, has_partner_int, period]

    return n_children * options["monthly_child_benefits"] * 12


def calc_unemployment_benefits(
    savings, sex, education, has_partner_int, period, options
):
    # Unemployment benefits means test
    means_test = savings < options["unemployment_wealth_thresh"]

    # Unemployment benefits for children living in the household
    n_children = options["children_by_state"][sex, education, has_partner_int, period]
    unemployment_benefits_children = (
        n_children * options["annual_child_unemployment_benefits"]
    )

    # Unemployment benefits for adults living in the household
    unemployment_benefits_adults = (1 + has_partner_int) * options[
        "annual_unemployment_benefits"
    ]
    # For housing, second adult gets only half
    unemployment_benefits_housing = (1 + 0.5 * has_partner_int) * options[
        "annual_unemployment_benefits_housing"
    ]

    # Total unemployment benefits
    total_unemployment_benefits = (
        unemployment_benefits_adults
        + unemployment_benefits_children
        + unemployment_benefits_housing
    )

    # reduced benefits for savings slightly above threshold
    reduced_benefits_threshhold = (
        options["unemployment_wealth_thresh"] + total_unemployment_benefits
    )
    reduced_benefits_means_test = (1 - means_test) * (
        savings < reduced_benefits_threshhold
    )
    reduced_benefits = reduced_benefits_threshhold - savings

    unemployment_benefits = (
        means_test * total_unemployment_benefits
        + reduced_benefits_means_test * reduced_benefits
    )

    return unemployment_benefits


def calc_care_benefits_and_costs(
    lagged_choice, education, has_sister, care_demand, options
):
    """Calculate the care benefits and costs."""

    informal_care_solo = is_informal_care(lagged_choice) * (
        care_demand == CARE_DEMAND_AND_NO_OTHER_SUPPLY
    )
    informal_care_joint = is_informal_care(lagged_choice) * (
        care_demand == CARE_DEMAND_AND_OTHER_SUPPLY
    )
    formal_care = is_no_care(lagged_choice) & (
        care_demand == CARE_DEMAND_AND_NO_OTHER_SUPPLY
    )
    # # Care benefits
    # care_benefits = options["care_benefits"][education, has_sister]

    # # Care costs
    # care_costs = options["care_costs"][education, has_sister]

    annual_care_benefits = options["informal_care_cash_benefits"] * 12
    annual_care_benefits_weighted = (
        annual_care_benefits * 0.5 * informal_care_joint
        + annual_care_benefits * informal_care_solo
    )

    annual_care_costs = options["formal_care_costs"] * 12
    annual_care_costs_weighted = (
        annual_care_costs * 0.5 * has_sister + annual_care_costs * (1 - has_sister)
    ) * formal_care

    return annual_care_benefits_weighted - annual_care_costs_weighted
