"""Tests for care demand transition based on ADL."""

import pickle

import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.shared import MOTHER
from caregiving.model.stochastic_processes.adl_transition import (
    PARENT_AGE_OFFSET,
    limitations_with_adl_transition,
)
from caregiving.model.stochastic_processes.caregiving_transition import (
    care_demand_transition_adl_light_intensive,
)


@pytest.mark.parametrize(
    "mother_adl, mother_dead, period, education",
    [
        # Alive, in caregiving window, different ADL states
        (0, 0, 10, 0),
        (1, 0, 10, 1),
        (2, 0, 10, 0),
        # Before caregiving window
        (1, 0, 5, 0),
        # After caregiving window
        (2, 0, 60, 1),
        # Mother already dead (regardless of window)
        (1, 1, 10, 0),
    ],
)
def test_care_demand_transition_adl_light_intensive_sums_to_one(
    mother_adl,
    mother_dead,
    period,
    education,
):
    """Check that care_demand probabilities behave as intended."""
    # Load options from BLD (created by task_specify_model)
    options = pickle.load((BLD / "model" / "options.pkl").open("rb"))
    # Extract model_params for easier access
    model_params = options["model_params"]

    probs = care_demand_transition_adl_light_intensive(
        mother_adl=mother_adl,
        mother_dead=mother_dead,
        period=period,
        education=education,
        options=model_params,
    )

    # 1) Probabilities must sum to 1
    assert np.isclose(probs.sum(), 1.0, atol=1e-8)

    # 2) Behavior inside vs. outside caregiving window (and for dead mothers)
    end_period_caregiving = (
        model_params["end_age_caregiving"] - model_params["start_age"]
    )
    start_period_caregiving = model_params["start_period_caregiving"]
    in_caregiving_window = (
        (1 - mother_dead)
        * (period >= start_period_caregiving - 1)
        * (period < end_period_caregiving)
    )

    if in_caregiving_window == 0:
        # Outside caregiving window or mother already dead:
        # all mass should be on "no care demand".
        assert np.isclose(probs[0], 1.0, atol=1e-8)
        assert np.isclose(probs[1], 0.0, atol=1e-8)
        assert np.isclose(probs[2], 0.0, atol=1e-8)
    else:
        # Inside caregiving window and mother alive:
        # mapping should follow limitations_with_adl_transition exactly.
        prob_adl = limitations_with_adl_transition(
            mother_adl=mother_adl,
            period=period,
            education=education,
            options=model_params,
        )
        assert np.allclose(probs, prob_adl, atol=1e-8)
