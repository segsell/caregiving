"""Tests for utility functions."""

import pickle as pkl
from itertools import product

import jax
import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.shared import (
    FULL_TIME_NO_CARE,
    PARENT_DEAD,
    PART_TIME_NO_CARE,
    UNEMPLOYED_NO_CARE,
)
from caregiving.model.utility.bequest_utility import (
    marginal_utility_final_consume_all,
    utility_final_consume_all,
)
from caregiving.model.utility.utility_functions import (
    consumption_scale,
    inverse_marginal,
    marginal_utility_function_alive,
    utility_func,
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


RHO_GRID = [1, 1.5]
EDUCATION_GRID = [0, 1]
CONSUMPTION_GRID = np.linspace(10, 100, 3)
_UTIL_UNEMPLOYED_GRID = np.linspace(-0.1, -0.9, 2)
_UTIL_WORK_GRID = np.linspace(-0.1, -0.9, 2)
DISUTIL_UNEMPLOYED_GRID = np.linspace(0.1, 0.9, 2)
DISUTIL_WORK_GRID = np.linspace(0.1, 0.9, 2)
BEQUEST_SCALE = np.linspace(1, 4, 2)
PARTNER_STATE_GRIRD = np.array([0, 1, 2], dtype=int)
NB_CHILDREN_GRID = np.arange(0, 2, 0.5, dtype=int)
EDUCATION_GRID = np.array([0, 1], dtype=int)
HEALTH_GRID = np.array([0, 1], dtype=int)
PERIOD_GRID = np.arange(0, 15, 5, dtype=int)
SEX_GRID = np.array([1], dtype=int)


@pytest.mark.parametrize(
    "partner_state, sex, education, period",
    list(product(PARTNER_STATE_GRIRD, SEX_GRID, EDUCATION_GRID, PERIOD_GRID)),
)
def test_consumption_scale(partner_state, sex, education, period, load_specs):
    """Test consumption scale function."""
    options = load_specs

    has_partner = int(partner_state > 0)
    n_children = options["children_by_state"][sex, education, has_partner, period]
    hh_size = 1 + has_partner + n_children

    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
        # has_partner=has_partner,
        # n_children=n_children,
    )

    np.testing.assert_almost_equal(cons_scale, np.sqrt(hh_size))


@pytest.mark.parametrize(
    "consumption, sex, partner_state, education, health, period, disutil_work, disutil_unemployed, rho",
    list(
        product(
            CONSUMPTION_GRID,
            SEX_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            HEALTH_GRID,
            PERIOD_GRID,
            DISUTIL_WORK_GRID,
            DISUTIL_UNEMPLOYED_GRID,
            RHO_GRID,
        )
    ),
)
def test_utility_func(
    consumption,
    sex,
    partner_state,
    education,
    health,
    period,
    disutil_work,
    disutil_unemployed,
    rho,
    load_specs,
):
    """Test utility function for unemployed, part and full-time."""
    options = load_specs
    params = {
        "rho": rho,
        "rho_bequest_low": rho,
        "rho_bequest_high": rho,
        "bequest_scale_low": 2,
        "bequest_scale_high": 2,
        # labor, no caregiving
        "disutil_pt_work_good_women": disutil_work,
        "disutil_pt_work_bad_women": disutil_work + 1,
        "disutil_ft_work_good_women": disutil_work,
        "disutil_ft_work_bad_women": disutil_work + 1,
        "disutil_unemployed_low_women": disutil_unemployed,
        "disutil_unemployed_high_women": disutil_unemployed,
        "disutil_children_pt_work_low": 0,
        "disutil_children_pt_work_high": 0,
        "disutil_children_ft_work_low": 0.1,
        "disutil_children_ft_work_high": 0.1,
        # labor and caregiving
        "disutil_pt_work_and_informal_care_good": disutil_work,
        "disutil_pt_work_and_informal_care_bad": disutil_work + 1,
        "disutil_ft_work_and_informal_care_good": disutil_work,
        "disutil_ft_work_and_informal_care_bad": disutil_work + 1,
        "disutil_unemployed_and_informal_care_low": disutil_unemployed,
        "disutil_unemployed_and_informal_care_high": disutil_unemployed,
        "disutil_children_pt_work_informal_care_low": 0,
        "disutil_children_pt_work_informal_care_high": 0,
        "disutil_children_ft_work_informal_care_low": 0.1,
        "disutil_children_ft_work_informal_care_high": 0.1,
    }

    # has_partner = int(partner_state > 0)
    # n_children = options["children_by_state"][sex, education, has_partner, period]
    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
        # has_partner=has_partner,
        # n_children=n_children,
    )

    # educ_str = "low" * (1 - education) + "high" * education
    health_str = "good" * health + "bad" * (1 - health)
    sex_str = "women"

    # disutil_unemployment = (
    # np.exp(params["util_cons_unemployed_low_educ"])
    # * (1 - education)
    # + np.exp(params["util_cons_unemployed_high_educ"]) * education
    # )
    disutil_unemployment = np.exp(-params[f"disutil_unemployed_low_{sex_str}"])

    # exp_factor_pt_work = params["util_cons_part_time_low_educ"] * (
    #     1 - education
    # ) + params["util_cons_children_part_time_low_educ"] * n_children * (1 - education)
    # exp_factor_pt_work += (
    #     params["util_cons_part_time_high_educ"] * education
    #     + params["util_cons_children_part_time_high_educ"] * n_children * education
    # )

    # exp_factor_ft_work = params["util_cons_full_time_low_educ"] * (
    #     1 - education
    # ) + params["util_cons_children_full_time_low_educ"] * n_children * (1 - education)
    # exp_factor_ft_work += (
    #     params["util_cons_full_time_high_educ"] * education
    #     + params["util_cons_children_full_time_high_educ"] * n_children * education
    # )

    exp_factor_pt_work = (
        params[f"disutil_pt_work_{health_str}_{sex_str}"]
        # + params[f"disutil_pt_work_{educ_str}_{sex_str}"]
    )
    exp_factor_ft_work = (
        params[f"disutil_ft_work_{health_str}_{sex_str}"]
        # + params[f"disutil_ft_work_{educ_str}_{sex_str}"]
    )

    # if sex == 1:
    has_partner_int = int(partner_state > 0)
    nb_children = options["children_by_state"][sex, education, has_partner_int, period]
    exp_factor_ft_work += (
        params["disutil_children_ft_work_high"] * nb_children * education
    )
    exp_factor_ft_work += (
        params["disutil_children_ft_work_low"] * nb_children * (1 - education)
    )

    disutil_pt_work = np.exp(-exp_factor_pt_work)
    disutil_ft_work = np.exp(-exp_factor_ft_work)

    # if rho == 1:
    #     utility_lambda = lambda util: np.log(  # noqa: E731
    #         consumption * util / cons_scale
    #     )
    # else:
    #     utility_lambda = (  # noqa: E731
    #         lambda util: ((consumption / cons_scale) ** (1 - rho) - 1)
    #         / (1 - rho)
    #         * util
    #     )
    if rho == 1:
        utility_lambda = lambda disutil: np.log(  # noqa: E731
            consumption * disutil / cons_scale
        )
    else:
        utility_lambda = lambda disutil: (  # noqa: E731
            (consumption * disutil / cons_scale) ** (1 - rho) - 1
        ) / (1 - rho)

    np.testing.assert_almost_equal(
        utility_func(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            care_demand=0,
            mother_health=PARENT_DEAD,
            period=period,
            choice=UNEMPLOYED_NO_CARE,
            params=params,
            options=options,
        ),
        utility_lambda(disutil_unemployment),
    )

    np.testing.assert_almost_equal(
        utility_func(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            care_demand=0,
            mother_health=PARENT_DEAD,
            period=period,
            choice=PART_TIME_NO_CARE,
            params=params,
            options=options,
        ),
        utility_lambda(disutil_pt_work),
    )

    np.testing.assert_almost_equal(
        utility_func(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            care_demand=0,
            mother_health=PARENT_DEAD,
            period=period,
            choice=FULL_TIME_NO_CARE,
            params=params,
            options=options,
        ),
        utility_lambda(disutil_ft_work),
    )


@pytest.mark.parametrize(
    "consumption, sex, partner_state, education, health, period, disutil_work, disutil_unemployed, rho",
    list(
        product(
            CONSUMPTION_GRID,
            SEX_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            HEALTH_GRID,
            PERIOD_GRID,
            DISUTIL_WORK_GRID,
            DISUTIL_UNEMPLOYED_GRID,
            RHO_GRID,
        )
    ),
)
def test_marginal_utility(
    consumption,
    sex,
    partner_state,
    education,
    health,
    period,
    disutil_work,
    disutil_unemployed,
    rho,
    load_specs,
):
    """Test marginal utility function."""

    options = load_specs
    params = {
        "rho": rho,
        "rho_bequest_low": rho,
        "rho_bequest_high": rho,
        "bequest_scale_low": 2,
        "bequest_scale_high": 2,
        # labor, no caregiving
        "disutil_pt_work_good_women": disutil_work,
        "disutil_pt_work_bad_women": disutil_work + 1,
        "disutil_ft_work_good_women": disutil_work,
        "disutil_ft_work_bad_women": disutil_work + 1,
        "disutil_unemployed_low_women": disutil_unemployed,
        "disutil_unemployed_high_women": disutil_unemployed,
        "disutil_children_pt_work_low": 0,
        "disutil_children_pt_work_high": 0,
        "disutil_children_ft_work_low": 0.1,
        "disutil_children_ft_work_high": 0.1,
        # labor and caregiving
        "disutil_pt_work_and_informal_care_good": disutil_work,
        "disutil_pt_work_and_informal_care_bad": disutil_work + 1,
        "disutil_ft_work_and_informal_care_good": disutil_work,
        "disutil_ft_work_and_informal_care_bad": disutil_work + 1,
        "disutil_unemployed_and_informal_care_low": disutil_unemployed,
        "disutil_unemployed_and_informal_care_high": disutil_unemployed,
        "disutil_children_pt_work_informal_care_low": 0,
        "disutil_children_pt_work_informal_care_high": 0,
        "disutil_children_ft_work_informal_care_low": 0.1,
        "disutil_children_ft_work_informal_care_high": 0.1,
    }

    random_choice = np.random.choice(np.array([0, 1, 2]))
    marg_util_jax = jax.jacfwd(utility_func, argnums=0)(
        consumption,
        random_choice,
        period,
        education,
        health,
        0,  # care_demand
        PARENT_DEAD,  # mother_health
        partner_state,
        params,
        options,
    )
    marg_util_model = marginal_utility_function_alive(
        consumption=consumption,
        choice=random_choice,
        period=period,
        education=education,
        health=health,
        care_demand=0,
        partner_state=partner_state,
        params=params,
        options=options,
    )
    np.testing.assert_almost_equal(marg_util_jax, marg_util_model)


@pytest.mark.parametrize(
    "consumption, sex, partner_state, education, health, period, disutil_work, disutil_unemployed, rho",
    list(
        product(
            CONSUMPTION_GRID,
            SEX_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            HEALTH_GRID,
            PERIOD_GRID,
            DISUTIL_WORK_GRID,
            DISUTIL_UNEMPLOYED_GRID,
            RHO_GRID,
        )
    ),
)
def test_inverse_marginal_utility(
    consumption,
    sex,
    partner_state,
    education,
    health,
    period,
    disutil_work,
    disutil_unemployed,
    rho,
    load_specs,
):
    options = load_specs
    params = {
        "rho": rho,
        "rho_bequest_low": rho,
        "rho_bequest_high": rho,
        "bequest_scale_low": 2,
        "bequest_scale_high": 2,
        # labor, no caregiving
        "disutil_pt_work_good_women": disutil_work,
        "disutil_pt_work_bad_women": disutil_work + 1,
        "disutil_ft_work_good_women": disutil_work,
        "disutil_ft_work_bad_women": disutil_work + 1,
        "disutil_unemployed_low_women": disutil_unemployed,
        "disutil_unemployed_high_women": disutil_unemployed,
        "disutil_children_pt_work_low": 0,
        "disutil_children_pt_work_high": 0,
        "disutil_children_ft_work_low": 0.1,
        "disutil_children_ft_work_high": 0.1,
        # labor and caregiving
        "disutil_pt_work_and_informal_care_good": disutil_work,
        "disutil_pt_work_and_informal_care_bad": disutil_work + 1,
        "disutil_ft_work_and_informal_care_good": disutil_work,
        "disutil_ft_work_and_informal_care_bad": disutil_work + 1,
        "disutil_unemployed_and_informal_care_low": disutil_unemployed,
        "disutil_unemployed_and_informal_care_high": disutil_unemployed,
        "disutil_children_pt_work_informal_care_low": 0,
        "disutil_children_pt_work_informal_care_high": 0,
        "disutil_children_ft_work_informal_care_low": 0.1,
        "disutil_children_ft_work_informal_care_high": 0.1,
    }

    random_choice = np.random.choice(np.array([0, 1, 2]))
    marg_util = marginal_utility_function_alive(
        consumption=consumption,
        choice=random_choice,
        period=period,
        education=education,
        health=health,
        care_demand=0,
        partner_state=partner_state,
        params=params,
        options=options,
    )
    np.testing.assert_almost_equal(
        inverse_marginal(
            marginal_utility=marg_util,
            choice=random_choice,
            period=period,
            education=education,
            health=health,
            care_demand=0,
            partner_state=partner_state,
            params=params,
            options=options,
        ),
        consumption,
    )


@pytest.mark.parametrize(
    "consumption, education, rho_educ, bequest_scale_educ",
    list(product(CONSUMPTION_GRID, EDUCATION_GRID, RHO_GRID, BEQUEST_SCALE)),
)
def test_bequest(consumption, education, rho_educ, bequest_scale_educ):
    """Test bequest utility function."""
    params = {
        "rho_bequest_low": rho_educ,
        "rho_bequest_high": rho_educ,
        "bequest_scale_low": bequest_scale_educ,
        "bequest_scale_high": bequest_scale_educ,
    }
    rho = (
        params["rho_bequest_low"] * (1 - education)
        + params["rho_bequest_high"] * education
    )
    bequest_scale = (
        params["bequest_scale_low"] * (1 - education)
        + params["bequest_scale_high"] * education
    )

    if rho == 1:
        bequest = bequest_scale * np.log(consumption)
    else:
        bequest = bequest_scale * (((consumption ** (1 - rho)) - 1) / (1 - rho))

    np.testing.assert_almost_equal(
        utility_final_consume_all(consumption, education=education, params=params),
        bequest,
    )


@pytest.mark.parametrize(
    "consumption, education, rho_educ, bequest_scale_educ",
    list(product(CONSUMPTION_GRID, EDUCATION_GRID, RHO_GRID, BEQUEST_SCALE)),
)
def test_bequest_marginal(consumption, education, rho_educ, bequest_scale_educ):
    """Test marginal utility of bequest function."""
    params = {
        "rho_bequest_low": rho_educ,
        "rho_bequest_high": rho_educ,
        "bequest_scale_low": bequest_scale_educ,
        "bequest_scale_high": bequest_scale_educ,
    }
    bequest = jax.jacfwd(utility_final_consume_all, argnums=0)(
        consumption, education, params
    )

    np.testing.assert_almost_equal(
        marginal_utility_final_consume_all(
            consumption, education=education, params=params
        ),
        bequest,
    )
