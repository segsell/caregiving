"""Tests for utility functions."""

import pickle as pkl
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.shared import (
    AGE_40,
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    CARE_DEMAND_AND_OTHER_SUPPLY,
    FULL_TIME_NO_CARE,
    PART_TIME_NO_CARE,
    PARTNER_RETIRED,
    UNEMPLOYED_NO_CARE,
)
from caregiving.model.utility.bequest_utility import (
    marginal_utility_final_consume_all,
    utility_final_consume_all,
)
from caregiving.model.utility.utility_components import disutility_work
from caregiving.model.utility.utility_functions_additive import (
    consumption_scale,
    inverse_marginal_additive,
    marginal_utility_func_additive_alive,
    utility_func_additive,
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


# Common parameter dictionary for utility tests
def create_test_params(disutil_work, disutil_unemployed, rho):
    """Create a standardized parameter dictionary for utility tests."""
    return {
        "rho_low": rho,
        "rho_high": rho,
        "rho_bequest_low": rho,
        "rho_bequest_high": rho,
        "bequest_scale_low": 2,
        "bequest_scale_high": 2,
        # labor, no caregiving
        "disutil_pt_work_high_good": disutil_work,
        "disutil_pt_work_high_bad": disutil_work + 1,
        "disutil_ft_work_high_good": disutil_work,
        "disutil_ft_work_high_bad": disutil_work + 1,
        "disutil_pt_work_low_good": disutil_work,
        "disutil_pt_work_low_bad": disutil_work + 1,
        "disutil_ft_work_low_good": disutil_work,
        "disutil_ft_work_low_bad": disutil_work + 1,
        "disutil_unemployed_low_women": disutil_unemployed,
        "disutil_unemployed_high_women": disutil_unemployed,
        "disutil_partner_retired": 0,
        # Age-based disutility from children (age < 40 vs age >= 40)
        "disutil_children_pt_work_low_below_40": 0,
        "disutil_children_pt_work_low_above_40": 0,
        "disutil_children_pt_work_high_below_40": 0,
        "disutil_children_pt_work_high_above_40": 0,
        "disutil_children_ft_work_low_below_40": 0.1,
        "disutil_children_ft_work_low_above_40": 0.1,
        "disutil_children_ft_work_high_below_40": 0.2,
        "disutil_children_ft_work_high_above_40": 0.2,
        # labor and caregiving
        "disutil_pt_work_high_good_informal_care": disutil_work,
        "disutil_pt_work_high_bad_informal_care": disutil_work + 1,
        "disutil_ft_work_high_good_informal_care": disutil_work,
        "disutil_ft_work_high_bad_informal_care": disutil_work + 1,
        "disutil_pt_work_low_good_informal_care": disutil_work,
        "disutil_pt_work_low_bad_informal_care": disutil_work + 1,
        "disutil_ft_work_low_good_informal_care": disutil_work,
        "disutil_ft_work_low_bad_informal_care": disutil_work + 1,
        "disutil_unemployed_low_women_informal_care": disutil_unemployed,
        "disutil_unemployed_high_women_informal_care": disutil_unemployed,
        # No age differentiation for informal care
        "disutil_children_pt_work_low_informal_care": 0,
        "disutil_children_pt_work_high_informal_care": 0,
        "disutil_children_ft_work_low_informal_care": 0.1,
        "disutil_children_ft_work_high_informal_care": 0.2,
        # level-shift disutilities for light informal care (education × labor state)
        "disutil_unemployed_light_informal_care_high": 0,
        "disutil_unemployed_light_informal_care_low": 0,
        "disutil_ft_work_light_informal_care_high": 0,
        "disutil_ft_work_light_informal_care_low": 0,
        # level-shift disutilities for intensive informal care (education × labor state)
        "disutil_unemployed_intensive_informal_care_high": 0,
        "disutil_unemployed_intensive_informal_care_low": 0,
        "disutil_ft_work_intensive_informal_care_high": 0,
        "disutil_ft_work_intensive_informal_care_low": 0,
        # type of care (varies by health and education)
        # new light informal care utilities (education × health)
        "util_light_informal_care_high_good": 0,
        "util_light_informal_care_high_bad": 0,
        "util_light_informal_care_low_good": 0,
        "util_light_informal_care_low_bad": 0,
        # new intensive informal care utilities (education × health)
        "util_intensive_informal_care_high_good": 0,
        "util_intensive_informal_care_high_bad": 0,
        "util_intensive_informal_care_low_good": 0,
        "util_intensive_informal_care_low_bad": 0,
        # new formal-care utilities (education × health)
        "util_formal_care_high_good": 0,
        "util_formal_care_high_bad": 0,
        "util_formal_care_low_good": 0,
        "util_formal_care_low_bad": 0,
        # legacy formal/joint informal care utilities (no longer used in active code,
        # but kept so older interfaces remain valid)
        "util_formal_care_good": 0,
        "util_formal_care_bad": 0,
        "util_joint_informal_care_good": 0,
        "util_joint_informal_care_bad": 0,
    }


@pytest.fixture(scope="module")
def utility_params():
    """Shared parameters for utility/disutility tests.

    Uses a single, consistent parameter set across tests to avoid redefinition.
    """
    params = create_test_params(disutil_work=0.3, disutil_unemployed=0.15, rho=1.5)
    # Ensure partner-retired disutility effect is present for tests that expect it
    params.update(
        {
            "disutil_partner_retired": 0.1,
        }
    )
    return params


@pytest.mark.parametrize(
    "partner_state, sex, education, period",
    list(product(PARTNER_STATE_GRIRD, SEX_GRID, EDUCATION_GRID, PERIOD_GRID)),
)
def test_consumption_scale(partner_state, sex, education, period, load_specs):
    """Test consumption scale function."""
    model_specs = load_specs

    has_partner = int(partner_state > 0)
    n_children = model_specs["children_by_state"][sex, education, has_partner, period]
    hh_size = 1 + has_partner + n_children

    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        model_specs=model_specs,
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
    model_specs = load_specs
    params = create_test_params(disutil_work, disutil_unemployed, rho)

    # has_partner = int(partner_state > 0)
    # n_children = model_specs["children_by_state"][sex, education, has_partner, period]
    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        model_specs=model_specs,
        # has_partner=has_partner,
        # n_children=n_children,
    )

    educ_str = "low" * (1 - education) + "high" * education
    health_str = "good" * health + "bad" * (1 - health)
    sex_str = "women"

    # disutil_unemployment = (
    # np.exp(params["util_cons_unemployed_low_educ"])
    # * (1 - education)
    # + np.exp(params["util_cons_unemployed_high_educ"]) * education
    # )
    disutil_unemployment = -params[f"disutil_unemployed_low_{sex_str}"]

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
        params[f"disutil_pt_work_{educ_str}_{health_str}"]
        # + params[f"disutil_pt_work_{educ_str}_{sex_str}"]
    )
    exp_factor_ft_work = (
        params[f"disutil_ft_work_{educ_str}_{health_str}"]
        # + params[f"disutil_ft_work_{educ_str}_{sex_str}"]
    )

    # if sex == 1:
    has_partner_int = int(partner_state > 0)
    nb_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]
    # Calculate age for age-based parameters
    age = period + model_specs["start_age"]
    age_below_40 = int(age < AGE_40)
    age_above_40 = int(age >= AGE_40)

    # Age-based disutility from children for full-time work
    if education == 1:  # high education
        exp_factor_ft_work += (
            params["disutil_children_ft_work_high_below_40"] * age_below_40
            + params["disutil_children_ft_work_high_above_40"] * age_above_40
        ) * nb_children
    else:  # low education
        exp_factor_ft_work += (
            params["disutil_children_ft_work_low_below_40"] * age_below_40
            + params["disutil_children_ft_work_low_above_40"] * age_above_40
        ) * nb_children

    disutil_pt_work = -exp_factor_pt_work
    disutil_ft_work = -exp_factor_ft_work

    if rho == 1:
        utility_lambda = (  # noqa: E731
            lambda disutil: np.log(consumption / cons_scale) + disutil
        )
    else:
        utility_lambda = (  # noqa: E731
            lambda disutil: ((consumption / cons_scale) ** (1 - rho) - 1) / (1 - rho)
            + disutil
        )

        np.testing.assert_almost_equal(
            utility_func_additive(
                consumption=consumption,
                partner_state=partner_state,
                education=education,
                health=health,
                care_demand=0,
                period=period,
                choice=UNEMPLOYED_NO_CARE,
                params=params,
                model_specs=model_specs,
            ),
            utility_lambda(disutil_unemployment),
        )

    np.testing.assert_almost_equal(
        utility_func_additive(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            care_demand=0,
            period=period,
            choice=PART_TIME_NO_CARE,
            params=params,
            model_specs=model_specs,
        ),
        utility_lambda(disutil_pt_work),
    )

    np.testing.assert_almost_equal(
        utility_func_additive(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            care_demand=0,
            period=period,
            choice=FULL_TIME_NO_CARE,
            params=params,
            model_specs=model_specs,
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

    model_specs = load_specs
    params = create_test_params(disutil_work, disutil_unemployed, rho)

    random_choice = np.random.choice(np.array([0, 1, 2]))
    marg_util_jax = jax.jacfwd(utility_func_additive, argnums=0)(
        consumption,
        random_choice,
        period,
        education,
        health,
        0,  # care_demand
        partner_state,
        params,
        model_specs,
    )
    marg_util_model = marginal_utility_func_additive_alive(
        consumption=consumption,
        choice=random_choice,
        period=period,
        education=education,
        health=health,
        # care_demand=0,
        partner_state=partner_state,
        params=params,
        model_specs=model_specs,
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
    model_specs = load_specs
    params = create_test_params(disutil_work, disutil_unemployed, rho)

    random_choice = np.random.choice(np.array([0, 1, 2]))
    marg_util = marginal_utility_func_additive_alive(
        consumption=consumption,
        partner_state=partner_state,
        education=education,
        health=health,
        period=period,
        choice=random_choice,
        params=params,
        model_specs=model_specs,
    )
    np.testing.assert_almost_equal(
        inverse_marginal_additive(
            marginal_utility=marg_util,
            partner_state=partner_state,
            education=education,
            health=health,
            period=period,
            choice=random_choice,
            params=params,
            model_specs=model_specs,
        ),
        consumption,
    )


# ======================================================================================
# Bequest utility
# ======================================================================================


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


# ======================================================================================
# Disutility work tests
# ======================================================================================


@pytest.mark.parametrize(
    "period, choice, education, partner_state, health, care_demand",
    list(
        product(
            PERIOD_GRID,
            [UNEMPLOYED_NO_CARE, PART_TIME_NO_CARE, FULL_TIME_NO_CARE],
            EDUCATION_GRID,
            PARTNER_STATE_GRIRD,
            HEALTH_GRID,
            [0, CARE_DEMAND_AND_NO_OTHER_SUPPLY, CARE_DEMAND_AND_OTHER_SUPPLY],
        )
    ),
)
def test_disutility_work_no_caregiving(
    period,
    choice,
    education,
    partner_state,
    health,
    care_demand,
    load_specs,
    utility_params,
):
    """Test disutility_work function for no caregiving scenarios."""
    model_specs = load_specs
    params = utility_params

    disutil = disutility_work(
        period=period,
        choice=choice,
        education=education,
        partner_state=partner_state,
        health=health,
        care_demand=care_demand,
        params=params,
        model_specs=model_specs,
    )

    # Test that disutility is negative (as expected for disutility)
    assert disutil <= 0, f"Disutility should be negative, got {disutil}"

    # Test that the function returns a scalar (JAX array with shape ())
    assert (
        disutil.shape == ()
    ), f"Disutility should be scalar, got shape {disutil.shape}"


@pytest.mark.parametrize(
    "period, choice, education, partner_state, health, care_demand",
    list(
        product(
            PERIOD_GRID,
            [3, 5, 7],  # Informal care choices
            EDUCATION_GRID,
            PARTNER_STATE_GRIRD,
            HEALTH_GRID,
            [CARE_DEMAND_AND_NO_OTHER_SUPPLY, CARE_DEMAND_AND_OTHER_SUPPLY],
        )
    ),
)
def test_disutility_work_informal_care(
    period,
    choice,
    education,
    partner_state,
    health,
    care_demand,
    load_specs,
    utility_params,
):
    """Test disutility_work function for informal care scenarios."""
    model_specs = load_specs
    params = utility_params

    disutil = disutility_work(
        period=period,
        choice=choice,
        education=education,
        partner_state=partner_state,
        health=health,
        care_demand=care_demand,
        params=params,
        model_specs=model_specs,
    )

    # Test that disutility is negative (as expected for disutility)
    assert disutil <= 0, f"Disutility should be negative, got {disutil}"

    # Test that the function returns a scalar (JAX array with shape ())
    assert (
        disutil.shape == ()
    ), f"Disutility should be scalar, got shape {disutil.shape}"


def test_disutility_work_partner_retired_effect(load_specs, utility_params):
    """Test that partner retired status affects disutility correctly."""
    model_specs = load_specs
    params = utility_params

    # Test with partner not retired
    disutil_no_retired = disutility_work(
        period=jnp.array(0),
        choice=jnp.array(0),  # retired choice
        education=jnp.array(1),
        partner_state=jnp.array(1),  # partner not retired
        health=jnp.array(1),
        care_demand=jnp.array(0),
        params=params,
        model_specs=model_specs,
    )

    # Test with partner retired
    disutil_retired = disutility_work(
        period=jnp.array(0),
        choice=jnp.array(0),  # retired choice
        education=jnp.array(1),
        partner_state=jnp.array(PARTNER_RETIRED),  # partner retired
        health=jnp.array(1),
        care_demand=jnp.array(0),
        params=params,
        model_specs=model_specs,
    )

    # When partner is retired and individual is retired, there should be additional disutility
    assert disutil_retired < disutil_no_retired, (
        f"Disutility with retired partner should be more negative, "
        f"got {disutil_retired} vs {disutil_no_retired}"
    )


def test_disutility_work_education_health_effects(load_specs, utility_params):
    """Test that education and health status affect disutility correctly."""
    model_specs = load_specs
    params = utility_params

    # Test high education vs low education
    disutil_high_ed = disutility_work(
        period=jnp.array(0),
        choice=FULL_TIME_NO_CARE,
        education=jnp.array(1),  # high education
        partner_state=jnp.array(0),
        health=jnp.array(1),
        care_demand=jnp.array(0),
        params=params,
        model_specs=model_specs,
    )

    disutil_low_ed = disutility_work(
        period=jnp.array(0),
        choice=FULL_TIME_NO_CARE,
        education=jnp.array(0),  # low education
        partner_state=jnp.array(0),
        health=jnp.array(1),
        care_demand=jnp.array(0),
        params=params,
        model_specs=model_specs,
    )

    # Test bad health vs good health
    disutil_bad_health = disutility_work(
        period=jnp.array(0),
        choice=FULL_TIME_NO_CARE,
        education=jnp.array(1),
        partner_state=jnp.array(0),
        health=jnp.array(0),  # bad health
        care_demand=jnp.array(0),
        params=params,
        model_specs=model_specs,
    )

    disutil_good_health = disutility_work(
        period=jnp.array(0),
        choice=FULL_TIME_NO_CARE,
        education=jnp.array(1),
        partner_state=jnp.array(0),
        health=jnp.array(1),  # good health
        care_demand=jnp.array(0),
        params=params,
        model_specs=model_specs,
    )

    # With the given parameters, bad health should have higher disutility
    assert disutil_bad_health < disutil_good_health, (
        f"Bad health should have higher disutility, "
        f"got {disutil_bad_health} vs {disutil_good_health}"
    )

    # Test that function returns different values for different inputs
    assert (
        disutil_high_ed != disutil_low_ed
    ), "Different education levels should give different disutility"
    assert (
        disutil_bad_health != disutil_good_health
    ), "Different health levels should give different disutility"
