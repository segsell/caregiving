import jax.numpy as jnp


def calc_net_household_income(own_income, partner_income, has_partner_int, options):
    """Calculate the income tax for a couple."""
    # Calculate the income tax for the couple
    family_income = own_income + partner_income

    # Calculate split factor. 1 if single, 2 if partnered
    split_factor = 1 + has_partner_int
    income_tax_split = calc_inc_tax_for_single_income(family_income / split_factor)

    # Readjust with split factor
    income_tax = income_tax_split * split_factor
    return family_income - income_tax


def calc_inc_tax_for_single_income(gross_income):
    """Parameters from 2010 gettsim params."""
    thresholds = [
        8004,
        13469,
        52881,
        250730,
    ]

    rates = [0.14, 0.2397, 0.42, 0.45]

    # In bracket 0 no taxes are paid
    poss_tax_bracket_0 = 0.0

    # In bracket 1 taxes are paid on income above the threshold
    poss_tax_bracket_1 = rates[0] * (gross_income - thresholds[0])
    tax_above_1 = (thresholds[1] - thresholds[0]) * rates[0]

    # In bracket 2 taxes are paid on income above the threshold and the tax paid in
    # bracket 1
    poss_tax_bracket_2 = (rates[1] * (gross_income - thresholds[1])) + tax_above_1
    tax_above_2 = (thresholds[2] - thresholds[1]) * rates[1]

    # In bracket 3 taxes are paid on income above the threshold and the tax paid in
    # brackets 1+2
    poss_tax_bracket_3 = (
        rates[2] * (gross_income - thresholds[2]) + tax_above_2 + tax_above_1
    )
    tax_above_3 = (thresholds[3] - thresholds[2]) * rates[2]

    # In bracket 4 taxes are paid on income above the threshold and the tax paid in
    # brackets 1+2+3
    poss_tax_bracket_4 = (
        rates[3] * (gross_income - thresholds[3])
        + tax_above_1
        + tax_above_2
        + tax_above_3
    )

    # Check in which bracket the income falls and calculate the tax
    in_bracket_0 = gross_income < thresholds[0]
    in_bracket_1 = (gross_income >= thresholds[0]) & (gross_income < thresholds[1])
    in_bracket_2 = (gross_income >= thresholds[1]) & (gross_income < thresholds[2])
    in_bracket_3 = (gross_income >= thresholds[2]) & (gross_income < thresholds[3])
    in_bracket_4 = gross_income >= thresholds[3]

    income_tax = (
        in_bracket_0 * poss_tax_bracket_0
        + in_bracket_1 * poss_tax_bracket_1
        + in_bracket_2 * poss_tax_bracket_2
        + in_bracket_3 * poss_tax_bracket_3
        + in_bracket_4 * poss_tax_bracket_4
    )
    return income_tax


def calc_after_ssc_income_worker(gross_wage):
    """Calculate the net income after social security contributions."""
    ssc = calc_pension_unempl_contr(gross_wage) + calc_health_ltc_contr(gross_wage)
    return gross_wage - ssc


def calc_after_ssc_income_pensioneer(gross_pesnion):
    """Calculate the net income after social security contributions."""
    ssc = calc_health_ltc_contr(gross_pesnion)
    return gross_pesnion - ssc


def calc_pension_unempl_contr(gross_income):
    """Calc pension and unemployment social security contribution."""
    contribution_threshold = 5500 * 12
    rate = 0.113
    # calculate pension contribution
    pension_contr = jnp.minimum(gross_income, contribution_threshold) * rate
    return pension_contr


def calc_health_ltc_contr(gross_income):
    """Calc health and ltc social security contribution."""
    contribution_threshold = 3750 * 12
    rate = 0.08
    # calculate social security contribution
    health_contr = jnp.minimum(gross_income, contribution_threshold) * rate
    return health_contr
