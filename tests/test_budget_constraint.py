"""Test budget functions."""

import copy
import pickle as pkl
from itertools import product

import jax
import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.state_space import get_next_period_experience
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.partner_income import (
    calc_partner_income_after_ssc,
)
from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_health_ltc_contr,
    calc_inc_tax_for_single_income,
    calc_pension_unempl_contr,
)
from caregiving.model.wealth_and_budget.transfers import calc_unemployment_benefits

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


@pytest.mark.parametrize(
    "period, sex, partner_state, education, savings",
    list(
        product(
            PERIOD_GRID,
            SEX_GRID,
            PARTNER_STATES,
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
    savings,
    load_specs,
):
    specs = load_specs
    specs_internal = copy.deepcopy(specs)

    params = {"interest_rate": specs_internal["interest_rate"]}

    max_init_exp_period = period + specs_internal["max_exp_diffs_per_period"][period]
    exp_cont = 2 / max_init_exp_period

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=1,
        experience=exp_cont,
        # sex=sex,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    has_partner = int(partner_state > 0)
    nb_children = specs["children_by_state"][sex, education, has_partner, period]
    income_partner = calc_partner_income_after_ssc(
        partner_state=partner_state,
        sex=sex,
        options=specs_internal,
        education=education,
        period=period,
    )
    split_factor = 1 + has_partner
    tax_partner = (
        calc_inc_tax_for_single_income(income_partner / split_factor) * split_factor
    )
    net_partner = income_partner - tax_partner
    net_partner_plus_child_benefits = (
        net_partner + nb_children * specs_internal["annual_child_benefits"]
    )

    unemployment_benefits = (1 + has_partner) * specs_internal[
        "annual_unemployment_benefits"
    ]
    unemployment_benefits_children = (
        specs_internal["annual_child_unemployment_benefits"] * nb_children
    )
    unemployment_benefits_housing = specs_internal[
        "annual_unemployment_benefits_housing"
    ] * (1 + 0.5 * has_partner)
    potential_unemployment_benefits = (
        unemployment_benefits
        + unemployment_benefits_children
        + unemployment_benefits_housing
    )

    means_test = savings_scaled < specs_internal["unemployment_wealth_thresh"]
    reduced_means_test_threshold = (
        specs_internal["unemployment_wealth_thresh"] + potential_unemployment_benefits
    )
    reduced_benefits_means_test = savings_scaled < reduced_means_test_threshold
    if means_test:
        income = np.maximum(
            potential_unemployment_benefits, net_partner_plus_child_benefits
        )
    elif ~means_test & reduced_benefits_means_test:
        reduced_unemployment_benefits = reduced_means_test_threshold - savings_scaled
        income = np.maximum(
            reduced_unemployment_benefits, net_partner_plus_child_benefits
        )
    else:
        income = net_partner_plus_child_benefits

    np.testing.assert_almost_equal(
        wealth,
        (savings_scaled * (1 + params["interest_rate"]) + income)
        / specs_internal["wealth_unit"],
    )


SAVINGS_GRID = np.linspace(8, 25, 3)
GAMMA_GRID = np.linspace(0.1, 0.9, 3)
EXP_GRID = np.linspace(10, 30, 3, dtype=int)
INCOME_SHOCK_GRID = np.linspace(-0.5, 0.5, 2)
WORKER_CHOICES = [2, 3]


@pytest.mark.parametrize(
    (
        "working_choice",
        "sex",
        "period",
        "partner_state",
        "education",
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

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=working_choice,
        experience=exp_cont,
        # sex=sex,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=income_shock,
        params=params,
        options=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    hourly_wage = np.exp(
        gamma_array[sex, education]
        + gamma_array[sex, education] * np.log(experience + 1)
        + income_shock
    )
    if working_choice == 2:  # noqa: PLR2004
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
    income_after_ssc = labor_income_year - sscs_worker

    has_partner_int = (partner_state > 0).astype(int)
    unemployment_benefits = calc_unemployment_benefits(
        savings=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        options=specs_internal,
    )

    nb_children = specs_internal["children_by_state"][
        sex, education, partner_state, period
    ]
    child_benefits = nb_children * specs_internal["monthly_child_benefits"] * 12
    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc)
        total_net_income = income_after_ssc - tax_total + child_benefits
        checked_income = np.maximum(total_net_income, unemployment_benefits)
        np.testing.assert_almost_equal(
            wealth,
            (savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income)
            / specs_internal["wealth_unit"],
            decimal=7,
        )
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

        income_partner = partner_income_year - sscs_partner
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = calc_inc_tax_for_single_income(total_income_after_ssc / 2) * 2
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        np.testing.assert_almost_equal(
            wealth,
            (savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income)
            / specs_internal["wealth_unit"],
            decimal=7,
        )


EXP_GRID = np.linspace(10, 30, 3, dtype=int)
RET_AGE_GRID = np.linspace(0, 2, 3, dtype=int)


@pytest.mark.parametrize(
    "period, sex, partner_state ,education, savings, exp",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
            SEX_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
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
        lagged_choice=0,
        already_retired=1,
        # sex=sex,
        education=education,
        experience=exp_cont_last_period,
        options=specs_internal,
    )
    # Check that experience does not get updated or added any penalty
    max_exp_this_period = period + specs_internal["max_exp_diffs_per_period"][period]
    np.testing.assert_allclose(exp_cont * max_exp_this_period, exp)

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=0,
        experience=exp_cont,
        # sex=sex,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]

    mean_wage_all = specs_internal["mean_hourly_ft_wage"][sex, education]
    gamma_0 = specs_internal["gamma_0"][sex, education]
    gamma_1_plus_1 = specs_internal["gamma_1"][sex, education] + 1
    total_pens_points = (
        (np.exp(gamma_0) / gamma_1_plus_1) * ((exp + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all
    pension_year = specs_internal["annual_pension_point_value"] * total_pens_points
    income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)
    unemployment_benefits = calc_unemployment_benefits(
        savings=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        options=specs_internal,
    )

    nb_children = specs_internal["children_by_state"][
        sex, education, partner_state, period
    ]
    child_benefits = nb_children * specs_internal["monthly_child_benefits"] * 12

    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc)
        total_net_income = income_after_ssc - tax_total + child_benefits
        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        np.testing.assert_almost_equal(
            wealth, scaled_wealth / specs_internal["wealth_unit"]
        )
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

        income_partner = partner_income_year - sscs_partner
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = calc_inc_tax_for_single_income(total_income_after_ssc / 2) * 2
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        np.testing.assert_almost_equal(
            wealth, scaled_wealth / specs_internal["wealth_unit"]
        )


@pytest.mark.parametrize(
    "period, sex, partner_state ,education, savings, exp",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
            SEX_GRID,
            PARTNER_STATES,
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
        lagged_choice=0,
        already_retired=0,
        # sex=sex,
        education=education,
        experience=exp_cont_prev,
        options=specs_internal,
    )

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=0,
        experience=exp_cont,
        # sex=sex,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=specs_internal,
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
    income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)
    unemployment_benefits = calc_unemployment_benefits(
        savings=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        options=specs_internal,
    )

    nb_children = specs_internal["children_by_state"][
        sex, education, partner_state, period
    ]
    child_benefits = nb_children * specs_internal["annual_child_benefits"]

    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc)
        total_net_income = income_after_ssc - tax_total + child_benefits
        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        np.testing.assert_almost_equal(
            wealth, scaled_wealth / specs_internal["wealth_unit"]
        )
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

        income_partner = partner_income_year - sscs_partner
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = calc_inc_tax_for_single_income(total_income_after_ssc / 2) * 2
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        np.testing.assert_almost_equal(
            wealth, scaled_wealth / specs_internal["wealth_unit"]
        )
