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
