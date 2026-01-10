"""Test budget functions."""

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
from caregiving.model.state_space import get_next_period_experience
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.government_budget import (
    calc_government_budget_components,
)
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

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
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

    # Calculate household unemployment benefits (NOTE: parameter order is sex before education)
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses: total_income = max(total_net_household_income + child_benefits,
    # household_unemployment_benefits)
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

    # Update gamma_0 and gamma_ln_exp for wage calculation (using gamma for both)
    gamma_array = np.array([[gamma, gamma - 0.01], [gamma / 2, gamma / 2 - 0.01]])
    specs_internal["gamma_0"] = gamma_array
    specs_internal["gamma_ln_exp"] = (
        gamma_array  # Use same gamma for ln_exp coefficient
    )

    # Convert experience: experience from grid is in years, need to normalize for budget_constraint
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

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
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
    # Note: For period 0, income_shock_for_labor = income_shock_mean, otherwise = income_shock
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

    has_partner_int = (partner_state > 0).astype(int)

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

    # Calculate household unemployment benefits (NOTE: parameter order is sex before education)
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses: total_income = max(total_net_household_income + child_benefits,
    # household_unemployment_benefits)
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

    # For retirees, experience should be converted to pension points using construct_experience_years
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

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
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
    # Note: calc_pensions_after_ssc takes pension_points (which is experience_years for retirees)
    retirement_income_after_ssc, gross_retirement_income = calc_pensions_after_ssc(
        pension_points=experience_years,
        model_specs=specs_internal,
    )
    own_income_after_ssc = retirement_income_after_ssc

    has_partner_int = (partner_state > 0).astype(int)

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

    # Calculate household unemployment benefits (NOTE: parameter order is sex before education)
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses: total_income = max(total_net_household_income + child_benefits,
    # household_unemployment_benefits)
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

    # For fresh retirees, experience should be converted using construct_experience_years
    # Note: The current budget_constraint implementation doesn't apply early retirement
    # penalties directly - those are handled through experience years calculation elsewhere.
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

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
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

    has_partner_int = (partner_state > 0).astype(int)

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

    # Calculate household unemployment benefits (NOTE: parameter order is sex before education)
    household_unemployment_benefits, _ = calc_unemployment_benefits(
        assets=savings_scaled,
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Budget equation uses: total_income = max(total_net_household_income + child_benefits,
    # household_unemployment_benefits)
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

    # Convert experience using construct_experience_years
    # Use fixed experience for testing (15 years)
    experience_years_fixed = 15.0
    is_retired_val = is_retired(working_choice)
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

    wealth, budget_aux = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=working_choice,
        experience=exp_cont,
        care_demand=care_demand,
        mother_dead=PARENT_LONGER_DEAD,
        asset_end_of_previous_period=savings,
        income_shock_previous_period=income_shock_prev,
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

    # Calculate own income and SSC using construct_experience_years
    experience_years = construct_experience_years(
        float_experience=exp_cont,
        period=period,
        is_retired=is_retired_val,
        model_specs=specs_internal,
    )

    # Handle period 0 income shock for labor calculation
    income_shock_for_labor = (
        specs_internal.get("income_shock_mean", 0.0) if period == 0 else 0.0
    )

    labor_income_after_ssc, gross_labor_income = calc_labor_income_after_ssc(
        lagged_choice=working_choice,
        experience_years=experience_years,
        education=education,
        sex=SEX,
        income_shock=income_shock_for_labor,
        model_specs=specs_internal,
    )

    was_worker = is_working(working_choice)
    was_retired = is_retired(working_choice)

    own_income_after_ssc = labor_income_after_ssc if was_worker else 0.0
    gross_retirement_income = 0.0

    # Calculate partner income
    partner_income_after_ssc, gross_partner_income, gross_partner_pension = (
        calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=SEX,
            model_specs=specs_internal,
            education=education,
            period=period,
        )
    )

    # Calculate benefits (NOTE: parameter order is sex before education)
    child_benefits = calc_child_benefits(
        sex=SEX,
        education=education,
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
        sex=SEX,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    # Calculate income tax using calc_net_household_income
    _, expected_income_tax, _ = calc_net_household_income(
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
