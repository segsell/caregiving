"""Tests for utility functions."""

import pickle as pkl
from itertools import product

import jax
import numpy as np
import pytest

from caregiving.config import SRC
from caregiving.model.utility.bequest_utility import (
    marginal_utility_final_consume_all,
    utility_final_consume_all,
)
from caregiving.model.utility.utility_functions import (
    consumption_scale,
    inverse_marginal_utility,
    marginal_utility,
    utility_func,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def load_specs():
    """Load specs from pickle file."""

    path_to_specs = SRC / "model" / "specs" / "specs.pkl"
    # path_to_max_exp_diff = BLD / "model" / "specs" / "max_exp_diffs_per_period.txt"

    with path_to_specs.open("rb") as file:
        specs = pkl.load(file)

    # specs["max_exp_diffs_per_period"] = jnp.array(np.loadtxt(path_to_max_exp_diff))

    return specs


RHO_GRID = [1, 1.5]
CONSUMPTION_GRID = np.linspace(10, 100, 3)
UTIL_UNEMPLOYED_GRID = np.linspace(-0.1, -0.9, 2)
UTIL_WORK_GRID = np.linspace(-0.1, -0.9, 2)
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
        # partner_state=partner_state,
        # sex=sex,
        # education=education,
        # period=period,
        # options=options,
        has_partner=has_partner,
        n_children=n_children,
    )

    np.testing.assert_almost_equal(cons_scale, np.sqrt(hh_size))


@pytest.mark.parametrize(
    "consumption, sex, partner_state, education, period, util_work, util_unemployed, rho",
    list(
        product(
            CONSUMPTION_GRID,
            SEX_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            PERIOD_GRID,
            UTIL_WORK_GRID,
            UTIL_UNEMPLOYED_GRID,
            RHO_GRID,
        )
    ),
)
def test_utility_func(
    consumption,
    sex,
    partner_state,
    education,
    period,
    util_work,
    util_unemployed,
    rho,
    load_specs,
):
    """Test utility function for unemployed, part and full-time."""
    params = {
        "rho": rho,
        "util_cons_unemployed": util_unemployed,
        "util_cons_part_time_low_educ": util_work,
        "util_cons_part_time_high_educ": util_work,
        "util_cons_full_time_low_educ": util_work,
        "util_cons_full_time_high_educ": util_work,
        "util_cons_children_part_time_low_educ": 0,
        "util_cons_children_part_time_high_educ": 0,
        "util_cons_children_full_time_low_educ": -0.1,
        "util_cons_children_full_time_high_educ": -0.1,
        "bequest_scale": 2,
    }
    options = load_specs

    has_partner = int(partner_state > 0)
    n_children = options["children_by_state"][sex, education, has_partner, period]
    cons_scale = consumption_scale(
        has_partner=has_partner,
        n_children=n_children,
    )

    disutil_unemployment = np.exp(params["util_cons_unemployed"])

    exp_factor_pt_work = params["util_cons_part_time_low_educ"] * (
        1 - education
    ) + params["util_cons_children_part_time_low_educ"] * n_children * (1 - education)
    exp_factor_pt_work += (
        params["util_cons_part_time_high_educ"] * education
        + params["util_cons_children_part_time_high_educ"] * n_children * education
    )

    exp_factor_ft_work = params["util_cons_full_time_low_educ"] * (
        1 - education
    ) + params["util_cons_children_full_time_low_educ"] * n_children * (1 - education)
    exp_factor_ft_work += (
        params["util_cons_full_time_high_educ"] * education
        + params["util_cons_children_full_time_high_educ"] * n_children * education
    )

    disutil_pt_work = np.exp(exp_factor_pt_work)
    disutil_ft_work = np.exp(exp_factor_ft_work)

    if rho == 1:
        utility_lambda = lambda util: np.log(  # noqa: E731
            consumption * util / cons_scale
        )
    else:
        utility_lambda = (  # noqa: E731
            lambda util: ((consumption / cons_scale) ** (1 - rho) - 1)
            / (1 - rho)
            * util
        )

    np.testing.assert_almost_equal(
        utility_func(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            # sex=sex,
            period=period,
            choice=1,
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
            # sex=sex,
            period=period,
            choice=2,
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
            # sex=sex,
            period=period,
            choice=3,
            params=params,
            options=options,
        ),
        utility_lambda(disutil_ft_work),
    )


# @pytest.mark.parametrize(
#     "consumption, sex, partner_state, education, health, period, disutil_work, disutil_unemployed, rho",
#     list(
#         product(
#             CONSUMPTION_GRID,
#             SEX_GRID,
#             PARTNER_STATE_GRIRD,
#             EDUCATION_GRID,
#             HEALTH_GRID,
#             PERIOD_GRID,
#             UTIL_WORK_GRID,
#             UTIL_UNEMPLOYED_GRID,
#             RHO_GRID,
#         )
#     ),
# )
# def test_marginal_utility(
#     consumption,
#     sex,
#     partner_state,
#     education,
#     health,
#     period,
#     disutil_work,
#     disutil_unemployed,
#     rho,
#     paths_and_specs,
# ):
#     options = paths_and_specs[1]
#     params = {
#         "rho": rho,
#         "disutil_pt_work_good_men": disutil_work + 1,
#         "disutil_pt_work_bad_men": disutil_work,
#         "disutil_ft_work_good_men": disutil_work + 1,
#         "disutil_ft_work_bad_men": disutil_work,
#         "disutil_unemployed_men": disutil_unemployed,
#         "disutil_pt_work_good_women": disutil_work + 1,
#         "disutil_pt_work_bad_women": disutil_work,
#         "disutil_ft_work_good_women": disutil_work + 1,
#         "disutil_ft_work_bad_women": disutil_work,
#         "disutil_unemployed_women": disutil_unemployed,
#         "disutil_children_ft_work_low": 0.1,
#         "disutil_children_ft_work_high": 0.1,
#         "bequest_scale": 2,
#     }

#     random_choice = np.random.choice(np.array([0, 1, 2]))
#     marg_util_jax = jax.jacfwd(utility_func, argnums=0)(
#         consumption,
#         sex,
#         partner_state,
#         education,
#         health,
#         period,
#         random_choice,
#         params,
#         options,
#     )
#     marg_util_model = marg_utility(
#         consumption=consumption,
#         partner_state=partner_state,
#         education=education,
#         health=health,
#         period=period,
#         sex=sex,
#         choice=random_choice,
#         params=params,
#         options=options,
#     )
#     np.testing.assert_almost_equal(marg_util_jax, marg_util_model)


# @pytest.mark.parametrize(
#     "consumption, sex, partner_state, education, health, period, disutil_work, disutil_unemployed, rho",
#     list(
#         product(
#             CONSUMPTION_GRID,
#             SEX_GRID,
#             PARTNER_STATE_GRIRD,
#             EDUCATION_GRID,
#             HEALTH_GRID,
#             PERIOD_GRID,
#             UTIL_WORK_GRID,
#             UTIL_UNEMPLOYED_GRID,
#             RHO_GRID,
#         )
#     ),
# )
# def test_inv_marginal_utility(
#     consumption,
#     sex,
#     partner_state,
#     education,
#     health,
#     period,
#     disutil_work,
#     disutil_unemployed,
#     rho,
#     paths_and_specs,
# ):
#     params = {
#         "rho": rho,
#         "disutil_pt_work_good_men": disutil_work + 1,
#         "disutil_pt_work_bad_men": disutil_work,
#         "disutil_ft_work_good_men": disutil_work + 1,
#         "disutil_ft_work_bad_men": disutil_work,
#         "disutil_unemployed_men": disutil_unemployed,
#         "disutil_pt_work_good_women": disutil_work + 1,
#         "disutil_pt_work_bad_women": disutil_work,
#         "disutil_ft_work_good_women": disutil_work + 1,
#         "disutil_ft_work_bad_women": disutil_work,
#         "disutil_unemployed_women": disutil_unemployed,
#         "disutil_children_ft_work_low": 0.1,
#         "disutil_children_ft_work_high": 0.1,
#         "bequest_scale": 2,
#     }

#     options = paths_and_specs[1]
#     random_choice = np.random.choice(np.array([0, 1, 2]))
#     marg_util = marg_utility(
#         consumption=consumption,
#         partner_state=partner_state,
#         education=education,
#         health=health,
#         sex=sex,
#         period=period,
#         choice=random_choice,
#         params=params,
#         options=options,
#     )
#     np.testing.assert_almost_equal(
#         inverse_marginal(
#             marginal_utility=marg_util,
#             partner_state=partner_state,
#             education=education,
#             health=health,
#             sex=sex,
#             period=period,
#             choice=random_choice,
#             params=params,
#             options=options,
#         ),
#         consumption,
#     )


@pytest.mark.parametrize(
    "consumption, rho, bequest_scale",
    list(product(CONSUMPTION_GRID, RHO_GRID, BEQUEST_SCALE)),
)
def test_bequest(consumption, rho, bequest_scale):
    params = {
        "rho": rho,
        "bequest_scale": bequest_scale,
    }
    if rho == 1:
        bequest = bequest_scale * np.log(consumption)
    else:
        bequest = bequest_scale * (((consumption ** (1 - rho)) - 1) / (1 - rho))
    np.testing.assert_almost_equal(
        utility_final_consume_all(consumption, params), bequest
    )


@pytest.mark.parametrize(
    "consumption, rho, bequest_scale",
    list(product(CONSUMPTION_GRID, RHO_GRID, BEQUEST_SCALE)),
)
def test_bequest_marginal(consumption, rho, bequest_scale):
    params = {
        "rho": rho,
        "bequest_scale": bequest_scale,
    }
    bequest = jax.jacfwd(utility_final_consume_all, argnums=0)(consumption, params)
    np.testing.assert_almost_equal(
        marginal_utility_final_consume_all(consumption, params),
        bequest,
    )
