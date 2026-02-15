"""Tests for Progressionsvorbehalt in calc_net_household_income."""

import numpy as np
import pytest

from caregiving.model.wealth_and_budget.tax_and_ssc import calc_net_household_income

# Minimal model_specs for German income tax (2010)
# Used so tests do not depend on built specs pickle.
TAX_SPECS = {
    "income_tax_brackets": np.array([0.0, 8004, 13469, 52881, 250730]),
    "linear_income_tax_rates": np.array([0.0, 1400, 2397, 0.42, 0.45]),
    "quadratic_income_tax_rates": np.array([0.0, 912.17, 228.74]),
    "intercepts_income_tax": np.array([0.0, 0.0, 1038.00, -8172.00, -15694.00]),
}


def test_progression_income_tax_lower_than_fully_taxable():
    """With progression_income > 0, tax is lower than when same amount in own_income."""
    own_income = 20_000.0
    partner_income = 0.0
    has_partner_int = 0
    progression_income = 5_000.0

    # Tax when benefit is in tax base (included in own_income)
    _, tax_fully_taxable, _ = calc_net_household_income(
        own_income=own_income + progression_income,
        partner_income=partner_income,
        has_partner_int=has_partner_int,
        model_specs=TAX_SPECS,
        progression_income=0.0,
    )

    # Tax when benefit is progression only (not in tax base)
    _, tax_with_progression, _ = calc_net_household_income(
        own_income=own_income,
        partner_income=partner_income,
        has_partner_int=has_partner_int,
        model_specs=TAX_SPECS,
        progression_income=progression_income,
    )

    assert tax_with_progression < tax_fully_taxable


def test_progression_disposable_equals_taxable_minus_tax_plus_benefit():
    """Disposable = family_income - tax + progression_income when progression used."""
    own_income = 25_000.0
    partner_income = 10_000.0
    has_partner_int = 1
    progression_income = 8_000.0
    family_income = own_income + partner_income

    disposable, income_tax, _ = calc_net_household_income(
        own_income=own_income,
        partner_income=partner_income,
        has_partner_int=has_partner_int,
        model_specs=TAX_SPECS,
        progression_income=progression_income,
    )

    expected_disposable = family_income - income_tax + progression_income
    np.testing.assert_almost_equal(disposable, expected_disposable, decimal=10)


def test_progression_zero_unchanged_behavior():
    """progression_income=0 (default) gives same result as before (no progression)."""
    own_income = 30_000.0
    partner_income = 15_000.0
    has_partner_int = 1

    out_default = calc_net_household_income(
        own_income=own_income,
        partner_income=partner_income,
        has_partner_int=has_partner_int,
        model_specs=TAX_SPECS,
    )
    out_explicit_zero = calc_net_household_income(
        own_income=own_income,
        partner_income=partner_income,
        has_partner_int=has_partner_int,
        model_specs=TAX_SPECS,
        progression_income=0.0,
    )

    np.testing.assert_almost_equal(out_default[0], out_explicit_zero[0], decimal=10)
    np.testing.assert_almost_equal(out_default[1], out_explicit_zero[1], decimal=10)
    np.testing.assert_almost_equal(out_default[2], out_explicit_zero[2], decimal=10)
