import pytest
from numpy.testing import assert_almost_equal as aaae

from caregiving.config import BLD, SRC
from caregiving.model.shared import BAD_HEALTH, GOOD_HEALTH
from caregiving.model.stochastic_processes import (
    exog_health_transition_mother_with_survival,
)
from caregiving.model.task_specify_model import get_options_dict
from caregiving.stochastic_processes.task_create_exog_processes import (
    exog_health_transition,
)
from caregiving.utils import load_dict_from_pickle

MEDIUM_HEALTH = -99


@pytest.mark.skip(reason="Outdated")
def test_exog_health_transition():
    params = {
        "medium_health": {
            "medium_health_age": 0.0304,
            "medium_health_age_squared": -1.31e-05,
            "medium_health_lagged_good_health": -1.155,
            "medium_health_lagged_medium_health": 0.736,
            "medium_health_lagged_bad_health": 1.434,
            "medium_health_constant": -1.550,
        },
        "bad_health": {
            "bad_health_age": 0.196,
            "bad_health_age_squared": -0.000885,
            "bad_health_lagged_good_health": -2.558,
            "bad_health_lagged_medium_health": -0.109,
            "bad_health_lagged_bad_health": 2.663,
            "bad_health_constant": -9.220,
        },
    }

    age = 50
    good_health, medium_health, bad_health = 1, 0, 0

    expected_prob_good, expected_prob_medium, expected_prob_bad = (
        0.76275676,
        0.22569606,
        0.01154716,
    )

    prob_good, prob_medium, prob_bad = exog_health_transition(
        age,
        good_health,
        medium_health,
        bad_health,
        params,
    )

    aaae(prob_good, expected_prob_good)
    aaae(prob_medium, expected_prob_medium)
    aaae(prob_bad, expected_prob_bad)
    aaae(prob_good + prob_medium + prob_bad, 1)


@pytest.mark.skip(reason="Outdated")
@pytest.mark.parametrize("health_state", [GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH])
def test_exog_health_transition_with_alive(health_state):
    path_to_specs = SRC / "model" / "specs.yaml"
    path_to_exog = BLD / "model" / "exog_processes.pkl"

    options = get_options_dict(path_to_specs=path_to_specs, path_to_exog=path_to_exog)

    path_to_survival_prob = BLD / "model" / "exog_survival_prob_female.pkl"

    model_params = {
        key: options["model_params"][key]
        for key in ["mother_medium_health", "mother_bad_health"]
    }
    _model_params = {
        "mother_medium_health": {
            "medium_health_age": 0.0304,
            "medium_health_age_squared": -1.31e-05,
            "medium_health_lagged_good_health": -1.155,
            "medium_health_lagged_medium_health": 0.736,
            "medium_health_lagged_bad_health": 1.434,
            "medium_health_constant": -1.550,
        },
        "mother_bad_health": {
            "bad_health_age": 0.196,
            "bad_health_age_squared": -0.000885,
            "bad_health_lagged_good_health": -2.558,
            "bad_health_lagged_medium_health": -0.109,
            "bad_health_lagged_bad_health": 2.663,
            "bad_health_constant": -9.220,
        },
    }
    survival_prob_params = load_dict_from_pickle(path_to_survival_prob)
    options = {"mother_start_age": 65} | model_params | survival_prob_params

    prob_good, prob_medium, prob_bad, prob_dead = (
        exog_health_transition_mother_with_survival(
            period=2,
            mother_health=health_state,
            options=options,
        )
    )

    aaae(prob_good + prob_medium + prob_bad + prob_dead, 1)


@pytest.mark.parametrize("health_state", [GOOD_HEALTH, BAD_HEALTH])
def test_exog_health_transition_with_alive_binary(health_state):
    path_to_specs = SRC / "model" / "specs.yaml"
    path_to_exog = BLD / "model" / "exog_processes.pkl"

    options = get_options_dict(path_to_specs=path_to_specs, path_to_exog=path_to_exog)

    path_to_survival_prob = BLD / "model" / "exog_survival_prob_female.pkl"

    model_params = {key: options["model_params"][key] for key in ["mother_bad_health"]}
    _model_params = {
        "mother_medium_health": {
            "medium_health_age": 0.0304,
            "medium_health_age_squared": -1.31e-05,
            "medium_health_lagged_good_health": -1.155,
            "medium_health_lagged_medium_health": 0.736,
            "medium_health_lagged_bad_health": 1.434,
            "medium_health_constant": -1.550,
        },
        "mother_bad_health": {
            "bad_health_age": 0.196,
            "bad_health_age_squared": -0.000885,
            "bad_health_lagged_good_health": -2.558,
            "bad_health_lagged_medium_health": -0.109,
            "bad_health_lagged_bad_health": 2.663,
            "bad_health_constant": -9.220,
        },
    }
    survival_prob_params = load_dict_from_pickle(path_to_survival_prob)
    options = {"mother_start_age": 65} | model_params | survival_prob_params

    prob_good, prob_bad, prob_dead = exog_health_transition_mother_with_survival(
        period=2,
        mother_health=health_state,
        options=options,
    )

    aaae(prob_good + prob_bad + prob_dead, 1)
