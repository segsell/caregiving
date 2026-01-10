"""Alternative test file for budget constraint functions."""

import copy
import pickle as pkl
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.experience_baseline_model import construct_experience_years
from caregiving.model.shared import (
    FULL_TIME_NO_CARE,
    PARENT_LONGER_DEAD,
    PARENT_RECENTLY_DEAD,
    PART_TIME_NO_CARE,
    RETIREMENT_NO_CARE,
    SEX,
    UNEMPLOYED_NO_CARE,
    is_retired,
    is_working,
)
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.partner_income import (
    calc_partner_income_after_ssc,
)
from caregiving.model.wealth_and_budget.pension_payments import calc_pensions_after_ssc
from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_health_ltc_contr,
    calc_net_household_income,
    calc_pension_unempl_contr,
)
from caregiving.model.wealth_and_budget.transfers import (
    calc_care_benefits_and_costs,
    calc_child_benefits,
    calc_inheritance_amount,
    calc_unemployment_benefits,
    draw_inheritance_outcome,
)
from caregiving.model.wealth_and_budget.wages import calc_labor_income_after_ssc

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

    # Convert experience using construct_experience_years
    # Use a fixed experience value for testing (2 years)
    experience_years_fixed = 2.0
    lagged_choice_val = UNEMPLOYED_NO_CARE[0].item()
    is_retired_val = is_retired(lagged_choice_val)
    max_exp_period = jnp.take(
        specs_internal["max_exps_period_working"], period, mode="clip"
    )
    scale_retired = specs_internal["max_pp_retirement"]
    scale = is_retired_val * scale_retired + (1 - is_retired_val) * max_exp_period
    exp_cont = experience_years_fixed / scale

    # Handle period 0 income shock (should use mean from specs, but defaults to 0)
    income_shock_prev = 0.0
    if period == 0:
        income_shock_prev = specs_internal.get("income_shock_mean", 0.0)

    # Convert partner_state to JAX array for budget_constraint
    partner_state_jax = jnp.asarray(partner_state)

    wealth, _ = budget_constraint(
        period=period,
        partner_state=partner_state_jax,
        education=education,
        lagged_choice=lagged_choice_val,
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=PARENT_LONGER_DEAD,
        asset_end_of_previous_period=savings,
        income_shock_previous_period=income_shock_prev,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    has_partner_int = int(partner_state > 0)

    # For unemployed: own_income_after_ssc = 0
    own_income_after_ssc = 0.0

    # Calculate partner income (already after SSC)
    partner_income_after_ssc, gross_partner_income, gross_partner_pension = (
        calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=SEX,
            model_specs=specs_internal,
            education=education,
            period=period,
        )
    )

    # Calculate total net household income (without child benefits and care benefits)
    total_net_household_income, _, _ = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=specs_internal,
    )

    # Calculate child benefits
    child_benefits = calc_child_benefits(
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Calculate household unemployment benefits
    # (NOTE: parameter order is sex before education)
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses:
    # total_income = max(total_net_household_income + child_benefits,
    #                    household_unemployment_benefits)
    total_income = np.maximum(
        total_net_household_income + child_benefits, household_unemployment_benefits
    )

    # Calculate care benefits and costs
    care_benefits_and_costs = calc_care_benefits_and_costs(
        lagged_choice=lagged_choice_val,
        education=education,
        care_demand=care_demand,
        model_specs=specs_internal,
    )

    # Calculate interest and bequest
    interest_rate = specs_internal["interest_rate"]
    interest = interest_rate * savings_scaled

    # Inheritance handling: only if mother recently died (PARENT_RECENTLY_DEAD)
    # Since we use PARENT_LONGER_DEAD in test, inheritance should be 0
    mother_died_recently = PARENT_LONGER_DEAD == PARENT_RECENTLY_DEAD
    inheritance_amount = calc_inheritance_amount(
        period=period,
        lagged_choice=lagged_choice_val,
        education=education,
        model_specs=specs_internal,
    )
    gets_inheritance = draw_inheritance_outcome(
        period=period,
        lagged_choice=lagged_choice_val,
        education=education,
        asset_end_of_previous_period=savings,
        model_specs=specs_internal,
    )
    bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount

    # Budget equation: assets_begin_of_period = assets_scaled + total_income
    # + interest + care_benefits_and_costs + bequest_from_parent
    expected_wealth = (
        savings_scaled
        + total_income
        + interest
        + care_benefits_and_costs
        + bequest_from_parent
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

    # Update gamma_0 and gamma_ln_exp for wage calculation (using gamma for both)
    gamma_array = np.array([[gamma, gamma - 0.01], [gamma / 2, gamma / 2 - 0.01]])
    specs_internal["gamma_0"] = gamma_array
    specs_internal["gamma_ln_exp"] = (
        gamma_array  # Use same gamma for ln_exp coefficient
    )

    # Convert experience: experience from grid is in years,
    # need to normalize for budget_constraint
    is_retired_val = is_retired(working_choice)
    max_exp_period = jnp.take(
        specs_internal["max_exps_period_working"], period, mode="clip"
    )
    scale_retired = specs_internal["max_pp_retirement"]
    scale = is_retired_val * scale_retired + (1 - is_retired_val) * max_exp_period
    # experience is already in years from the grid, so normalize it
    exp_cont = experience / scale
    # Recalculate experience_years using the same function as budget_constraint does
    experience_years = construct_experience_years(
        float_experience=exp_cont,
        period=period,
        is_retired=is_retired_val,
        model_specs=specs_internal,
    )

    # Handle period 0 income shock (should use mean from specs, but defaults to 0)
    income_shock_prev = income_shock
    if period == 0:
        income_shock_prev = specs_internal.get("income_shock_mean", 0.0)

    # Convert partner_state to JAX array for budget_constraint
    partner_state_jax = jnp.asarray(partner_state)

    wealth, _ = budget_constraint(
        period=period,
        partner_state=partner_state_jax,
        education=education,
        lagged_choice=working_choice,
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=PARENT_LONGER_DEAD,
        asset_end_of_previous_period=savings,
        income_shock_previous_period=income_shock_prev,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]

    # Calculate labor income using current function
    # Note: For period 0, income_shock_for_labor = income_shock_mean,
    #       otherwise = income_shock
    income_shock_for_labor = (
        specs_internal.get("income_shock_mean", 0.0) if period == 0 else income_shock
    )
    labor_income_after_ssc, gross_labor_income = calc_labor_income_after_ssc(
        lagged_choice=working_choice,
        experience_years=experience_years,
        education=education,
        sex=SEX,
        income_shock=income_shock_for_labor,
        model_specs=specs_internal,
    )
    own_income_after_ssc = labor_income_after_ssc

    has_partner_int = int(partner_state > 0)

    # Calculate partner income using current function
    partner_income_after_ssc, gross_partner_income, gross_partner_pension = (
        calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=SEX,
            model_specs=specs_internal,
            education=education,
            period=period,
        )
    )

    # Calculate total net household income (without child benefits and care benefits)
    total_net_household_income, _, _ = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=specs_internal,
    )

    # Calculate child benefits
    child_benefits = calc_child_benefits(
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Calculate household unemployment benefits
    # (NOTE: parameter order is sex before education)
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses:
    # total_income = max(total_net_household_income + child_benefits,
    #                    household_unemployment_benefits)
    total_income = np.maximum(
        total_net_household_income + child_benefits, household_unemployment_benefits
    )

    # Calculate care benefits and costs
    care_benefits_and_costs = calc_care_benefits_and_costs(
        lagged_choice=working_choice,
        education=education,
        care_demand=care_demand,
        model_specs=specs_internal,
    )

    # Calculate interest and bequest
    interest_rate = specs_internal["interest_rate"]
    interest = interest_rate * savings_scaled

    # Inheritance handling: only if mother recently died (PARENT_RECENTLY_DEAD)
    # Since we use PARENT_LONGER_DEAD in test, inheritance should be 0
    mother_died_recently = PARENT_LONGER_DEAD == PARENT_RECENTLY_DEAD
    inheritance_amount = calc_inheritance_amount(
        period=period,
        lagged_choice=working_choice,
        education=education,
        model_specs=specs_internal,
    )
    gets_inheritance = draw_inheritance_outcome(
        period=period,
        lagged_choice=working_choice,
        education=education,
        asset_end_of_previous_period=savings,
        model_specs=specs_internal,
    )
    bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount

    # Budget equation: assets_begin_of_period = assets_scaled + total_income
    # + interest + care_benefits_and_costs + bequest_from_parent
    expected_wealth = (
        savings_scaled
        + total_income
        + interest
        + care_benefits_and_costs
        + bequest_from_parent
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

    # For retirees, experience should be converted to pension points
    # using construct_experience_years
    lagged_choice_val = RETIREMENT_NO_CARE[0].item()
    is_retired_val = is_retired(lagged_choice_val)

    # Convert experience: exp is already in years, normalize it for budget_constraint
    scale_retired = specs_internal["max_pp_retirement"]
    exp_cont = exp / scale_retired
    # Recalculate experience_years using the same function as budget_constraint does
    experience_years = construct_experience_years(
        float_experience=exp_cont,
        period=period,
        is_retired=is_retired_val,
        model_specs=specs_internal,
    )

    # Handle period 0 income shock (should use mean from specs, but defaults to 0)
    income_shock_prev = 0.0
    if period == 0:
        income_shock_prev = specs_internal.get("income_shock_mean", 0.0)

    # Convert partner_state to JAX array for budget_constraint
    partner_state_jax = jnp.asarray(partner_state)

    wealth, _ = budget_constraint(
        period=period,
        partner_state=partner_state_jax,
        education=education,
        lagged_choice=lagged_choice_val,
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=PARENT_LONGER_DEAD,
        asset_end_of_previous_period=savings,
        income_shock_previous_period=income_shock_prev,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]

    # Calculate pension using current function
    # Note: calc_pensions_after_ssc takes pension_points
    # (which is experience_years for retirees)
    retirement_income_after_ssc, gross_retirement_income = calc_pensions_after_ssc(
        pension_points=experience_years,
        model_specs=specs_internal,
    )
    own_income_after_ssc = retirement_income_after_ssc

    has_partner_int = int(partner_state > 0)

    # Calculate partner income using current function
    partner_income_after_ssc, gross_partner_income, gross_partner_pension = (
        calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=SEX,
            model_specs=specs_internal,
            education=education,
            period=period,
        )
    )

    # Calculate total net household income (without child benefits and care benefits)
    total_net_household_income, _, _ = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=specs_internal,
    )

    # Calculate child benefits
    child_benefits = calc_child_benefits(
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Calculate household unemployment benefits
    # (NOTE: parameter order is sex before education)
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses:
    # total_income = max(total_net_household_income + child_benefits,
    #                    household_unemployment_benefits)
    total_income = np.maximum(
        total_net_household_income + child_benefits, household_unemployment_benefits
    )

    # Calculate care benefits and costs
    care_benefits_and_costs = calc_care_benefits_and_costs(
        lagged_choice=lagged_choice_val,
        education=education,
        care_demand=care_demand,
        model_specs=specs_internal,
    )

    # Calculate interest and bequest
    interest_rate = specs_internal["interest_rate"]
    interest = interest_rate * savings_scaled

    # Inheritance handling: only if mother recently died (PARENT_RECENTLY_DEAD)
    # Since we use PARENT_LONGER_DEAD in test, inheritance should be 0
    mother_died_recently = PARENT_LONGER_DEAD == PARENT_RECENTLY_DEAD
    inheritance_amount = calc_inheritance_amount(
        period=period,
        lagged_choice=lagged_choice_val,
        education=education,
        model_specs=specs_internal,
    )
    gets_inheritance = draw_inheritance_outcome(
        period=period,
        lagged_choice=lagged_choice_val,
        education=education,
        asset_end_of_previous_period=savings,
        model_specs=specs_internal,
    )
    bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount

    # Budget equation: assets_begin_of_period = assets_scaled + total_income
    # + interest + care_benefits_and_costs + bequest_from_parent
    expected_wealth = (
        savings_scaled
        + total_income
        + interest
        + care_benefits_and_costs
        + bequest_from_parent
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
    partner_state = 1  # PARTNER_WORKING

    # For fresh retirees, experience should be converted using
    # construct_experience_years
    # Note: The current budget_constraint implementation doesn't apply
    # early retirement penalties directly - those are handled through
    # experience years calculation elsewhere.
    # Here we match the current implementation behavior.
    lagged_choice_val = RETIREMENT_NO_CARE[0].item()
    is_retired_val = is_retired(lagged_choice_val)

    # Convert experience: exp is already in years, normalize it for budget_constraint
    scale_retired = specs_internal["max_pp_retirement"]
    exp_cont = exp / scale_retired
    # Recalculate experience_years using the same function as budget_constraint does
    experience_years = construct_experience_years(
        float_experience=exp_cont,
        period=period,
        is_retired=is_retired_val,
        model_specs=specs_internal,
    )

    # Handle period 0 income shock (should use mean from specs, but defaults to 0)
    income_shock_prev = 0.0
    if period == 0:
        income_shock_prev = specs_internal.get("income_shock_mean", 0.0)

    # Convert partner_state to JAX array for budget_constraint
    partner_state_jax = jnp.asarray(partner_state)

    wealth, _ = budget_constraint(
        period=period,
        partner_state=partner_state_jax,
        education=education,
        lagged_choice=lagged_choice_val,
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=PARENT_LONGER_DEAD,
        asset_end_of_previous_period=savings,
        income_shock_previous_period=income_shock_prev,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]

    # Calculate pension using current function
    # Note: The current implementation uses experience_years directly as pension_points
    # Early retirement penalties are not applied in budget_constraint itself
    retirement_income_after_ssc, gross_retirement_income = calc_pensions_after_ssc(
        pension_points=experience_years,
        model_specs=specs_internal,
    )
    own_income_after_ssc = retirement_income_after_ssc

    has_partner_int = int(partner_state > 0)

    # Calculate partner income using current function
    partner_income_after_ssc, gross_partner_income, gross_partner_pension = (
        calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=SEX,
            model_specs=specs_internal,
            education=education,
            period=period,
        )
    )

    # Calculate total net household income (without child benefits and care benefits)
    total_net_household_income, _, _ = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=specs_internal,
    )

    # Calculate child benefits
    child_benefits = calc_child_benefits(
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Calculate household unemployment benefits
    # (NOTE: parameter order is sex before education)
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses:
    # total_income = max(total_net_household_income + child_benefits,
    #                    household_unemployment_benefits)
    total_income = np.maximum(
        total_net_household_income + child_benefits, household_unemployment_benefits
    )

    # Calculate care benefits and costs
    care_benefits_and_costs = calc_care_benefits_and_costs(
        lagged_choice=lagged_choice_val,
        education=education,
        care_demand=care_demand,
        model_specs=specs_internal,
    )

    # Calculate interest and bequest
    interest_rate = specs_internal["interest_rate"]
    interest = interest_rate * savings_scaled

    # Inheritance handling: only if mother recently died (PARENT_RECENTLY_DEAD)
    # Since we use PARENT_LONGER_DEAD in test, inheritance should be 0
    mother_died_recently = PARENT_LONGER_DEAD == PARENT_RECENTLY_DEAD
    inheritance_amount = calc_inheritance_amount(
        period=period,
        lagged_choice=lagged_choice_val,
        education=education,
        model_specs=specs_internal,
    )
    gets_inheritance = draw_inheritance_outcome(
        period=period,
        lagged_choice=lagged_choice_val,
        education=education,
        asset_end_of_previous_period=savings,
        model_specs=specs_internal,
    )
    bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount

    # Budget equation: assets_begin_of_period = assets_scaled + total_income
    # + interest + care_benefits_and_costs + bequest_from_parent
    expected_wealth = (
        savings_scaled
        + total_income
        + interest
        + care_benefits_and_costs
        + bequest_from_parent
    ) / specs_internal["wealth_unit"]

    np.testing.assert_almost_equal(wealth, expected_wealth)
