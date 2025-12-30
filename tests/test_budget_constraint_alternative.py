"""Alternative test file for budget constraint functions."""

import copy
import pickle as pkl
from itertools import product

import jax
import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.shared import (
    FULL_TIME_NO_CARE,
    PART_TIME_NO_CARE,
    RETIREMENT_NO_CARE,
    UNEMPLOYED_NO_CARE,
)
from caregiving.model.state_space import get_next_period_experience
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.partner_income import (
    calc_partner_income_after_ssc,
)
from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_health_ltc_contr,
    calc_inc_tax_for_single_income,
    calc_net_household_income,
    calc_pension_unempl_contr,
)
from caregiving.model.wealth_and_budget.transfers import (
    calc_inheritance,
    calc_unemployment_benefits,
)

jax.config.update("jax_enable_x64", True)

SAVINGS_GRID_UNEMPLOYED = np.linspace(10, 25, 7)
PARTNER_STATES = np.array([0, 1, 2], dtype=int)
PERIOD_GRID = np.arange(0, 65, 15, dtype=int)
OLD_AGE_PERIOD_GRID = np.arange(25, 43, 8, dtype=int)
VERY_OLD_AGE_PERIOD_GRID = np.arange(35, 43, 3, dtype=int)
EDUCATION_GRID = [0, 1]
SEX_GRID = [1]
CARE_DEMAND_GRID = [0, 1, 2]


@pytest.fixture()
def load_specs():
    """Load specs from pickle file."""
    path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"

    with path_to_specs.open("rb") as file:
        specs = pkl.load(file)

    return specs


@pytest.mark.parametrize(
    "period, sex, partner_state, education, care_demand, savings",
    list(
        product(
            PERIOD_GRID,
            SEX_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            CARE_DEMAND_GRID,
            SAVINGS_GRID_UNEMPLOYED,
        )
    ),
)
def test_budget_unemployed(
    period,
    sex,
    partner_state,
    education,
    care_demand,
    savings,
    load_specs,
):
    specs = load_specs
    specs_internal = copy.deepcopy(specs)

    params = {"interest_rate": specs_internal["interest_rate"]}

    max_init_exp_period = period + specs_internal["max_exp_diffs_per_period"][period]
    exp_cont = 2 / max_init_exp_period

    wealth, _ = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=UNEMPLOYED_NO_CARE[0].item(),
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        asset_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    has_partner_int = int(partner_state > 0)

    # For unemployed: own_income_after_ssc = 0
    own_income_after_ssc = 0.0

    # Calculate partner income (already after SSC)
    partner_income_after_ssc, _, _ = calc_partner_income_after_ssc(
        partner_state=partner_state,
        sex=sex,
        model_specs=specs_internal,
        education=education,
        period=period,
    )

    # Calculate total net household income (without child benefits and care benefits)
    total_net_household_income, _ = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=specs_internal,
    )

    # Calculate household unemployment benefits
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses: total_income = max(total_net_household_income,
    # household_unemployment_benefits)
    # (without child benefits and care benefits - they are overwritten on line 135)
    total_income = np.maximum(
        total_net_household_income, household_unemployment_benefits
    )

    # Calculate interest and bequest
    interest_rate = specs_internal["interest_rate"]
    interest = interest_rate * savings_scaled

    bequest_from_parent = calc_inheritance(
        period=period,
        lagged_choice=UNEMPLOYED_NO_CARE[0].item(),
        education=education,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        model_specs=specs_internal,
    )

    # Budget equation: assets_begin_of_period = assets_scaled + total_income
    # + interest + bequest_from_parent
    expected_wealth = (
        savings_scaled + total_income + interest + bequest_from_parent
    ) / specs_internal["wealth_unit"]

    np.testing.assert_almost_equal(wealth, expected_wealth)


SAVINGS_GRID = np.linspace(8, 25, 4)
GAMMA_GRID = np.linspace(0.1, 0.9, 2)
EXP_GRID = np.linspace(10, 40, 10, dtype=int)
INCOME_SHOCK_GRID = np.linspace(-0.5, 0.5, 2)
WORKER_CHOICES = [PART_TIME_NO_CARE[0].item(), FULL_TIME_NO_CARE[0].item()]


@pytest.mark.parametrize(
    (
        "working_choice, sex, period, partner_state, education, gamma, "
        "income_shock, experience, savings, care_demand"
    ),
    list(
        product(
            WORKER_CHOICES,
            SEX_GRID,
            PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            GAMMA_GRID,
            INCOME_SHOCK_GRID,
            EXP_GRID,
            SAVINGS_GRID,
            CARE_DEMAND_GRID,
        )
    ),
)
def test_budget_worker(
    working_choice,
    sex,
    period,
    partner_state,
    education,
    gamma,
    income_shock,
    experience,
    savings,
    care_demand,
    load_specs,
):
    specs = load_specs
    specs_internal = copy.deepcopy(specs)
    gamma_array = np.array([[gamma, gamma - 0.01], [gamma / 2, gamma / 2 - 0.01]])
    specs_internal["gamma_0"] = gamma_array
    specs_internal["gamma_1"] = gamma_array

    params = {"interest_rate": specs_internal["interest_rate"]}

    max_init_exp_period = period + specs_internal["max_exp_diffs_per_period"][period]
    exp_cont = experience / max_init_exp_period

    wealth, _ = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=working_choice,
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        asset_end_of_previous_period=savings,
        income_shock_previous_period=income_shock,
        params=params,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]

    # Calculate experience years
    max_exp_period = period + specs_internal["max_exp_diffs_per_period"][period]
    experience_years = max_exp_period * exp_cont

    hourly_wage = np.exp(
        gamma_array[sex, education]
        + gamma_array[sex, education] * np.log(experience_years + 1)
        + income_shock
    )

    if working_choice == PART_TIME_NO_CARE[0].item():
        labor_income_year = (
            hourly_wage * specs_internal["av_annual_hours_pt"][sex, education]
        )
        min_wage_year = specs_internal["annual_min_wage_pt"][sex, education]
    else:
        labor_income_year = (
            hourly_wage * specs_internal["av_annual_hours_ft"][sex, education]
        )
        min_wage_year = specs_internal["annual_min_wage_ft"]

    # Check against min wage
    labor_income_year = max(labor_income_year, min_wage_year)

    income_scaled = labor_income_year
    sscs_worker = calc_health_ltc_contr(income_scaled) + calc_pension_unempl_contr(
        income_scaled
    )
    own_income_after_ssc = labor_income_year - sscs_worker

    has_partner_int = (partner_state > 0).astype(int)

    # Calculate partner income
    if partner_state == 0:
        partner_income_after_ssc = 0.0
    else:
        partner_income_after_ssc, _, _ = calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=sex,
            model_specs=specs_internal,
            education=education,
            period=period,
        )

    # Calculate total net household income (without child benefits and care benefits)
    total_net_household_income, _ = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=specs_internal,
    )

    # Calculate household unemployment benefits
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses: total_income = max(total_net_household_income,
    # household_unemployment_benefits)
    # (without child benefits and care benefits - they are overwritten on line 135)
    total_income = np.maximum(
        total_net_household_income, household_unemployment_benefits
    )

    # Calculate interest and bequest
    interest_rate = specs_internal["interest_rate"]
    interest = interest_rate * savings_scaled

    bequest_from_parent = calc_inheritance(
        period=period,
        lagged_choice=working_choice,
        education=education,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        model_specs=specs_internal,
    )

    # Budget equation: assets_begin_of_period = assets_scaled + total_income
    # + interest + bequest_from_parent
    expected_wealth = (
        savings_scaled + total_income + interest + bequest_from_parent
    ) / specs_internal["wealth_unit"]

    np.testing.assert_almost_equal(wealth, expected_wealth, decimal=7)


@pytest.mark.parametrize(
    "period, sex, partner_state, education, care_demand, savings, exp",
    list(
        product(
            VERY_OLD_AGE_PERIOD_GRID,
            SEX_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            CARE_DEMAND_GRID,
            SAVINGS_GRID,
            EXP_GRID,
        )
    ),
)
def test_retiree(
    period,
    sex,
    partner_state,
    education,
    care_demand,
    savings,
    exp,
    load_specs,
):
    specs_internal = load_specs

    params = {"interest_rate": specs_internal["interest_rate"]}

    # exp is the experience years here
    last_period = period - 1
    max_init_exp_prev_period = (
        last_period + specs_internal["max_exp_diffs_per_period"][last_period]
    )
    exp_cont_prev = exp / max_init_exp_prev_period

    exp_cont = get_next_period_experience(
        period=period,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        already_retired=1,
        education=education,
        experience=exp_cont_prev,
        model_specs=specs_internal,
    )

    # Check that experience does not get updated or added any penalty
    max_exp_this_period = period + specs_internal["max_exp_diffs_per_period"][period]
    np.testing.assert_allclose(exp_cont * max_exp_this_period, exp)

    wealth, _ = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        asset_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]

    # Calculate experience years
    max_exp_period = period + specs_internal["max_exp_diffs_per_period"][period]
    experience_years = max_exp_period * exp_cont

    mean_wage_all = specs_internal["mean_hourly_ft_wage"][sex, education]
    gamma_0 = specs_internal["gamma_0"][sex, education]
    gamma_1_plus_1 = specs_internal["gamma_1"][sex, education] + 1
    total_pens_points = (
        (np.exp(gamma_0) / gamma_1_plus_1)
        * ((experience_years + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all
    pension_year = specs_internal["annual_pension_point_value"] * total_pens_points
    own_income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)

    # Calculate partner income
    if partner_state == 0:
        partner_income_after_ssc = 0.0
    else:
        partner_income_after_ssc, _, _ = calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=sex,
            model_specs=specs_internal,
            education=education,
            period=period,
        )

    # Calculate total net household income (without child benefits and care benefits)
    total_net_household_income, _ = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=specs_internal,
    )

    # Calculate household unemployment benefits
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses: total_income = max(total_net_household_income,
    # household_unemployment_benefits)
    # (without child benefits and care benefits - they are overwritten on line 135)
    total_income = np.maximum(
        total_net_household_income, household_unemployment_benefits
    )

    # Calculate interest and bequest
    interest_rate = specs_internal["interest_rate"]
    interest = interest_rate * savings_scaled

    bequest_from_parent = calc_inheritance(
        period=period,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        education=education,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        model_specs=specs_internal,
    )

    # Budget equation: assets_begin_of_period = assets_scaled + total_income
    # + interest + bequest_from_parent
    expected_wealth = (
        savings_scaled + total_income + interest + bequest_from_parent
    ) / specs_internal["wealth_unit"]

    np.testing.assert_almost_equal(wealth, expected_wealth)


@pytest.mark.parametrize(
    "period, sex, education, care_demand, savings, exp",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
            SEX_GRID,
            EDUCATION_GRID,
            CARE_DEMAND_GRID,
            SAVINGS_GRID,
            EXP_GRID,
        )
    ),
)
def test_fresh_retiree(
    period,
    sex,
    education,
    care_demand,
    savings,
    exp,
    load_specs,
):
    """Test fresh retiree budget constraint.

    In this test we assume that disability and non disability retirement is
    always possible. Even though in the model there are choice set restrictions,
    which is tested in state space.
    """
    specs_internal = load_specs

    # In this test, we set all to married, as this does not matter for the
    # mechanic we test here, i.e. the experience adjustment
    partner_state = np.array(1, dtype=int)

    params = {"interest_rate": specs_internal["interest_rate"]}

    # Last the person was not retired. So exp is here really experience years
    last_period = period - 1
    max_init_exp_prev_period = (
        last_period + specs_internal["max_exp_diffs_per_period"][last_period]
    )
    exp_cont_prev = exp / max_init_exp_prev_period

    # This period the person retires, so lagged choice is RETIREMENT_NO_CARE
    exp_cont = get_next_period_experience(
        period=period,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        already_retired=0,
        education=education,
        experience=exp_cont_prev,
        model_specs=specs_internal,
    )

    wealth, _ = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        asset_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]

    # Calculate experience years
    max_exp_period = period + specs_internal["max_exp_diffs_per_period"][period]
    experience_years = max_exp_period * exp_cont

    mean_wage_all = specs_internal["mean_hourly_ft_wage"][sex, education]
    gamma_0 = specs_internal["gamma_0"][sex, education]
    gamma_1_plus_1 = specs_internal["gamma_1"][sex, education] + 1
    total_pens_points = (
        (np.exp(gamma_0) / gamma_1_plus_1)
        * ((experience_years + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all

    # Calculate partner income
    partner_income_after_ssc, _, _ = calc_partner_income_after_ssc(
        partner_state=partner_state,
        sex=sex,
        model_specs=specs_internal,
        education=education,
        period=period,
    )

    pension_year = specs_internal["annual_pension_point_value"] * total_pens_points
    own_income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)

    # Calculate total net household income (without child benefits and care benefits)
    total_net_household_income, _ = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=specs_internal,
    )

    # Calculate household unemployment benefits
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses: total_income = max(total_net_household_income,
    # household_unemployment_benefits)
    # (without child benefits and care benefits - they are overwritten on line 135)
    total_income = np.maximum(
        total_net_household_income, household_unemployment_benefits
    )

    # Calculate interest and bequest
    interest_rate = specs_internal["interest_rate"]
    interest = interest_rate * savings_scaled

    bequest_from_parent = calc_inheritance(
        period=period,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        education=education,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        model_specs=specs_internal,
    )

    # Budget equation: assets_begin_of_period = assets_scaled + total_income
    # + interest + bequest_from_parent
    expected_wealth = (
        savings_scaled + total_income + interest + bequest_from_parent
    ) / specs_internal["wealth_unit"]

    np.testing.assert_almost_equal(wealth, expected_wealth)
