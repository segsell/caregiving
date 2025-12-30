"""Test budget functions."""

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
from caregiving.model.wealth_and_budget.government_budget import (
    calc_government_budget_components,
)
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
    calc_care_benefits_and_costs,
    calc_child_benefits,
    calc_inheritance,
    calc_unemployment_benefits,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def load_specs():
    """Load specs from pickle file."""

    path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"
    # path_to_max_exp_diff = BLD / "model" / "specs" / "max_exp_diffs_per_period.txt"

    with path_to_specs.open("rb") as file:
        specs = pkl.load(file)

    # specs["max_exp_diffs_per_period"] = jnp.array(np.loadtxt(path_to_max_exp_diff))

    return specs


SAVINGS_GRID_UNEMPLOYED = np.linspace(10, 25, 3)
PARTNER_STATES = np.array([0, 1, 2], dtype=int)
PERIOD_GRID = np.arange(0, 65, 15, dtype=int)
OLD_AGE_PERIOD_GRID = np.arange(33, 43, 3, dtype=int)
EDUCATION_GRID = [0, 1]
SEX_GRID = [1]
CARE_DEMAND_GRID = [0, 1, 2]


@pytest.mark.parametrize(
    "period, sex, partner_state, education, care_demand, savings",
    list(
        product(
            PERIOD_GRID,
            SEX_GRID,
            PARTNER_STATES,
            CARE_DEMAND_GRID,
            EDUCATION_GRID,
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

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=UNEMPLOYED_NO_CARE[0].item(),
        experience=exp_cont,
        # sex=sex,
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


SAVINGS_GRID = np.linspace(8, 25, 3)
GAMMA_GRID = np.linspace(0.1, 0.9, 3)
EXP_GRID = np.linspace(10, 30, 3, dtype=int)
INCOME_SHOCK_GRID = np.linspace(-0.5, 0.5, 2)
WORKER_CHOICES = [PART_TIME_NO_CARE[0].item(), FULL_TIME_NO_CARE[0].item()]


@pytest.mark.parametrize(
    (
        "working_choice",
        "sex",
        "period",
        "partner_state",
        "education",
        "care_demand",
        "gamma",
        "income_shock",
        "experience",
        "savings",
    ),
    list(
        product(
            WORKER_CHOICES,
            SEX_GRID,
            PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            CARE_DEMAND_GRID,
            GAMMA_GRID,
            INCOME_SHOCK_GRID,
            EXP_GRID,
            SAVINGS_GRID,
        )
    ),
)
def test_budget_worker(
    working_choice,
    sex,
    period,
    partner_state,
    education,
    care_demand,
    gamma,
    income_shock,
    experience,
    savings,
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

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=working_choice,
        experience=exp_cont,
        # sex=sex,
        care_demand=care_demand,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        asset_end_of_previous_period=savings,
        income_shock_previous_period=income_shock,
        params=params,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    hourly_wage = np.exp(
        gamma_array[sex, education]
        + gamma_array[sex, education] * np.log(experience + 1)
        + income_shock
    )
    if working_choice == PART_TIME_NO_CARE[0].item():  # noqa: PLR2004
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
        if partner_state == 1:
            partner_income_year = specs_internal["annual_partner_wage"][
                sex, education, period
            ]
            sscs_partner = calc_health_ltc_contr(
                partner_income_year
            ) + calc_pension_unempl_contr(partner_income_year)
        else:
            partner_income_year = specs_internal["annual_partner_pension"][
                sex, education
            ]
            sscs_partner = calc_health_ltc_contr(partner_income_year)
        partner_income_after_ssc = partner_income_year - sscs_partner

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


EXP_GRID = np.linspace(10, 30, 3, dtype=int)
RET_AGE_GRID = np.linspace(0, 2, 3, dtype=int)


@pytest.mark.parametrize(
    "period, sex, partner_state ,education,care_demand, savings, exp",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
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
    last_period = period - 1
    max_exp_last_period = (
        specs_internal["max_exp_diffs_per_period"][last_period] + last_period
    )
    exp_cont_last_period = exp / max_exp_last_period

    exp_cont = get_next_period_experience(
        period=period,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        already_retired=1,
        # sex=sex,
        education=education,
        experience=exp_cont_last_period,
        model_specs=specs_internal,
    )
    # Check that experience does not get updated or added any penalty
    max_exp_this_period = period + specs_internal["max_exp_diffs_per_period"][period]
    np.testing.assert_allclose(exp_cont * max_exp_this_period, exp)

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        experience=exp_cont,
        # sex=sex,
        care_demand=care_demand,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        asset_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]

    mean_wage_all = specs_internal["mean_hourly_ft_wage"][sex, education]
    gamma_0 = specs_internal["gamma_0"][sex, education]
    gamma_1_plus_1 = specs_internal["gamma_1"][sex, education] + 1
    total_pens_points = (
        (np.exp(gamma_0) / gamma_1_plus_1) * ((exp + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all
    pension_year = specs_internal["annual_pension_point_value"] * total_pens_points
    own_income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)

    # Calculate partner income
    if partner_state == 0:
        partner_income_after_ssc = 0.0
    else:
        if partner_state == 1:
            partner_income_year = specs_internal["annual_partner_wage"][
                sex, education, period
            ]
            sscs_partner = calc_health_ltc_contr(
                partner_income_year
            ) + calc_pension_unempl_contr(partner_income_year)
        else:
            partner_income_year = specs_internal["annual_partner_pension"][
                sex, education
            ]
            sscs_partner = calc_health_ltc_contr(partner_income_year)
        partner_income_after_ssc = partner_income_year - sscs_partner

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
    "period, sex, partner_state ,education, care_demand, savings, exp",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
            SEX_GRID,
            PARTNER_STATES,
            CARE_DEMAND_GRID,
            EDUCATION_GRID,
            SAVINGS_GRID,
            EXP_GRID,
        )
    ),
)
def test_fresh_retiree(
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

    actual_retirement_age = specs_internal["start_age"] + period - 1

    params = {"interest_rate": specs_internal["interest_rate"]}
    last_period = period - 1
    max_init_exp_prev_period = (
        last_period + specs_internal["max_exp_diffs_per_period"][last_period]
    )
    exp_cont_prev = exp / max_init_exp_prev_period

    exp_cont = get_next_period_experience(
        period=period,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        already_retired=0,
        # sex=sex,
        education=education,
        experience=exp_cont_prev,
        model_specs=specs_internal,
    )

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=RETIREMENT_NO_CARE[0].item(),
        experience=exp_cont,
        # sex=sex,
        care_demand=care_demand,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        asset_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    SRA_at_resolution = specs_internal["min_SRA"]
    retirement_age_difference = SRA_at_resolution - actual_retirement_age

    if retirement_age_difference > 0:
        # informed
        ERP = specs_internal["early_retirement_penalty"]
        # uninformed
        # ERP = specs_internal["uninformed_early_retirement_penalty"][education]
        pension_factor = 1 - retirement_age_difference * ERP
    else:
        late_retirement_bonus = specs_internal["late_retirement_bonus"]
        pension_factor = 1 + np.abs(retirement_age_difference) * late_retirement_bonus

    mean_wage_all = specs_internal["mean_hourly_ft_wage"][sex, education]
    gamma_0 = specs_internal["gamma_0"][sex, education]
    gamma_1_plus_1 = specs_internal["gamma_1"][sex, education] + 1
    total_pens_points = (
        (np.exp(gamma_0) / gamma_1_plus_1) * ((exp + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all

    pension_year = (
        specs_internal["annual_pension_point_value"]
        * total_pens_points
        * pension_factor
    )
    own_income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)

    # Calculate partner income
    if partner_state == 0:
        partner_income_after_ssc = 0.0
    else:
        if partner_state == 1:
            partner_income_year = specs_internal["annual_partner_wage"][
                sex, education, period
            ]
            sscs_partner = calc_health_ltc_contr(
                partner_income_year
            ) + calc_pension_unempl_contr(partner_income_year)
        else:
            partner_income_year = specs_internal["annual_partner_pension"][
                sex, education
            ]
            sscs_partner = calc_health_ltc_contr(partner_income_year)
        partner_income_after_ssc = partner_income_year - sscs_partner

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
    "period, sex, partner_state, education, care_demand, savings, working_choice",
    list(
        product(
            PERIOD_GRID[:2],  # Test fewer periods for speed
            SEX_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            CARE_DEMAND_GRID[:2],  # Test fewer care demands
            SAVINGS_GRID[:2],  # Test fewer savings
            [PART_TIME_NO_CARE[0].item(), FULL_TIME_NO_CARE[0].item()],
        )
    ),
)
def test_government_budget_components(
    period,
    sex,
    partner_state,
    education,
    care_demand,
    savings,
    working_choice,
    load_specs,
):
    """Test government budget components in budget_aux."""
    specs = load_specs
    specs_internal = copy.deepcopy(specs)

    params = {"interest_rate": specs_internal["interest_rate"]}

    max_init_exp_period = period + specs_internal["max_exp_diffs_per_period"][period]
    exp_cont = 15 / max_init_exp_period  # Use fixed experience for testing

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=working_choice,
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=1,  # Dead (2-state system: 0=alive, 1=dead)
        mother_longer_dead=1,  # Longer dead (no inheritance)
        asset_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        model_specs=specs_internal,
    )

    # Check that budget_aux contains all expected government budget keys
    expected_keys = [
        "income_tax",
        "own_ssc",
        "partner_ssc",
        "total_tax_revenue",
        "government_expenditures",
        "net_government_budget",
    ]
    for key in expected_keys:
        assert key in budget_aux, f"Missing key '{key}' in budget_aux"

    # Manually calculate expected government budget components
    savings_scaled = savings * specs_internal["wealth_unit"]
    has_partner_int = int(partner_state > 0)

    # Calculate own income and SSC
    max_exp_period = period + specs_internal["max_exp_diffs_per_period"][period]
    experience_years = max_exp_period * exp_cont

    from caregiving.model.wealth_and_budget.wages import calc_labor_income_after_ssc

    labor_income_after_ssc, gross_labor_income = calc_labor_income_after_ssc(
        lagged_choice=working_choice,
        experience_years=experience_years,
        education=education,
        sex=sex,
        income_shock=0,
        model_specs=specs_internal,
    )

    was_worker = working_choice in (
        PART_TIME_NO_CARE[0].item(),
        FULL_TIME_NO_CARE[0].item(),
    )
    was_retired = False

    own_income_after_ssc = labor_income_after_ssc if was_worker else 0.0
    gross_retirement_income = 0.0

    # Calculate partner income
    partner_income_after_ssc, gross_partner_income, gross_partner_pension = (
        calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=sex,
            model_specs=specs_internal,
            education=education,
            period=period,
        )
    )

    # Calculate benefits
    child_benefits = calc_child_benefits(
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )
    care_benefits_and_costs = calc_care_benefits_and_costs(
        lagged_choice=working_choice,
        education=education,
        care_demand=care_demand,
        model_specs=specs_internal,
    )
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Calculate income tax using calc_net_household_income
    _, expected_income_tax = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=specs_internal,
    )

    # Calculate expected government budget components
    (
        expected_income_tax,
        expected_own_ssc,
        expected_partner_ssc,
        expected_total_tax_revenue,
        expected_government_expenditures,
        expected_net_government_budget,
    ) = calc_government_budget_components(
        household_income_tax_total=expected_income_tax,
        was_worker=was_worker,
        was_retired=was_retired,
        gross_labor_income=gross_labor_income,
        gross_retirement_income=gross_retirement_income,
        partner_state=partner_state,
        gross_partner_income=gross_partner_income,
        gross_partner_pension=gross_partner_pension,
        child_benefits=child_benefits,
        care_benefits_and_costs=care_benefits_and_costs,
        household_unemployment_benefits=household_unemployment_benefits,
        model_specs=specs_internal,
    )

    # Compare with budget_aux (scaled by wealth_unit)
    np.testing.assert_almost_equal(
        budget_aux["income_tax"],
        expected_income_tax / specs_internal["wealth_unit"],
        decimal=7,
    )
    np.testing.assert_almost_equal(
        budget_aux["own_ssc"],
        expected_own_ssc / specs_internal["wealth_unit"],
        decimal=7,
    )
    np.testing.assert_almost_equal(
        budget_aux["partner_ssc"],
        expected_partner_ssc / specs_internal["wealth_unit"],
        decimal=7,
    )
    np.testing.assert_almost_equal(
        budget_aux["total_tax_revenue"],
        expected_total_tax_revenue / specs_internal["wealth_unit"],
        decimal=7,
    )
    np.testing.assert_almost_equal(
        budget_aux["government_expenditures"],
        expected_government_expenditures / specs_internal["wealth_unit"],
        decimal=7,
    )
    np.testing.assert_almost_equal(
        budget_aux["net_government_budget"],
        expected_net_government_budget / specs_internal["wealth_unit"],
        decimal=7,
    )

    # Verify that total_tax_revenue = income_tax + own_ssc + partner_ssc
    np.testing.assert_almost_equal(
        budget_aux["total_tax_revenue"],
        (budget_aux["income_tax"] + budget_aux["own_ssc"] + budget_aux["partner_ssc"]),
        decimal=7,
    )

    # Verify that net_government_budget = total_tax_revenue - government_expenditures
    np.testing.assert_almost_equal(
        budget_aux["net_government_budget"],
        budget_aux["total_tax_revenue"] - budget_aux["government_expenditures"],
        decimal=7,
    )
