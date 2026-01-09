"""Tests for care demand transition based on ADL."""

import pickle

import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.shared import (
    PARENT_LONGER_DEAD,
    PARENT_RECENTLY_DEAD,
)
from caregiving.model.stochastic_processes.adl_transition import (
    death_transition,
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
        # Mother already dead (recently died, state 1)
        (1, 1, 10, 0),
        # Mother longer dead (state 2)
        (0, 2, 10, 0),
    ],
)
def test_care_demand_transition_adl_light_intensive_sums_to_one(
    mother_adl,
    mother_dead,
    period,
    education,
):
    """Check that care_demand probabilities behave as intended."""
    # Load model_specs from BLD (created by task_specify_model)
    model_specs = pickle.load((BLD / "model" / "specs" / "specs_full.pkl").open("rb"))

    probs = care_demand_transition_adl_light_intensive(
        mother_adl=mother_adl,
        mother_dead=mother_dead,
        period=period,
        education=education,
        model_specs=model_specs,
    )

    # 1) Probabilities must sum to 1
    assert np.isclose(probs.sum(), 1.0, atol=1e-8)

    # 2) Behavior inside vs. outside caregiving window (and for dead mothers)
    end_period_caregiving = model_specs["end_age_caregiving"] - model_specs["start_age"]
    start_period_caregiving = model_specs["start_period_caregiving"]
    is_mother_alive = mother_dead == 0
    in_caregiving_window = (
        is_mother_alive
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
            model_specs=model_specs,
        )
        assert np.allclose(probs, prob_adl, atol=1e-8)


# @pytest.mark.skip(reason="No longer implemented.")
# @pytest.mark.parametrize(
#     "mother_adl, period, education",
#     [
#         # Alive, in caregiving window, different ADL states
#         # mother_adl: 0=Dead, 1=No ADL Alive, 2=ADL 1, 3=ADL 2/3
#         (1, 10, 0),  # No ADL Alive, in window
#         (2, 10, 1),  # ADL 1, in window
#         (3, 10, 0),  # ADL 2/3, in window
#         # Before caregiving window
#         (1, 5, 0),  # No ADL Alive, before window
#         # After caregiving window
#         (3, 60, 1),  # ADL 2/3, after window
#         # Mother already dead (mother_adl == 0) - KEY TEST CASE
#         (0, 10, 0),  # Dead
#         (0, 5, 1),  # Dead, before window
#         (0, 60, 0),  # Dead, after window
#     ],
# )
# def test_care_demand_death_transition_light_intensive_sums_to_one(
#     mother_adl,
#     period,
#     education,
# ):
#     """Check that care_demand_death probabilities sum to 1 and have correct shape."""
#     # Load model_specs from BLD (created by task_specify_model)
#     model_specs = pickle.load((BLD / "model" / "specs" / "specs_full.pkl").open("rb"))

#     probs = care_demand_death_transition_light_intensive(
#         mother_adl=mother_adl,
#         period=period,
#         education=education,
#         model_specs=model_specs,
#     )

#     # 1) Probabilities must sum to 1
#     assert np.isclose(
#         probs.sum(), 1.0, atol=1e-8
#     ), f"Probabilities sum to {probs.sum()}"

#     # 2) Must have 4 states
#     assert probs.shape == (4,), f"Expected shape (4,), got {probs.shape}"

#     # 3) All probabilities must be non-negative
#     assert np.all(probs >= 0), f"Negative probabilities found: {probs}"


# @pytest.mark.skip(reason="No longer implemented.")
# @pytest.mark.parametrize(
#     "period, education",
#     [
#         # Test different periods for dead mothers (mother_adl == 0)
#         (10, 0),
#         (10, 1),
#         (5, 0),
#         (60, 1),
#     ],
# )
# def test_care_demand_death_transition_mother_already_dead(
#     period,
#     education,
# ):
#     """Mother dead (mother_adl == 0) stays in dead state (state 0)."""
#     # Load model_specs from BLD
#     model_specs = pickle.load((BLD / "model" / "specs" / "specs_full.pkl").open("rb"))

#     mother_adl = 0  # Mother is dead (mother_adl == 0 means "No ADL Dead")

#     probs = care_demand_death_transition_light_intensive(
#         mother_adl=mother_adl,
#         period=period,
#         education=education,
#         model_specs=model_specs,
#     )

#     # State 0 (NO_CARE_DEMAND_DEAD) should have probability 1.0
#     # All other states should have probability 0.0
#     assert np.isclose(
#         probs[0], 1.0, atol=1e-8
#     ), f"When mother is dead, state 0 should be 1.0, got {probs[0]}"
#     assert np.isclose(
#         probs[1], 0.0, atol=1e-8
#     ), f"When mother is dead, state 1 should be 0.0, got {probs[1]}"
#     assert np.isclose(
#         probs[2], 0.0, atol=1e-8
#     ), f"When mother is dead, state 2 should be 0.0, got {probs[2]}"
#     assert np.isclose(
#         probs[3], 0.0, atol=1e-8
#     ), f"When mother is dead, state 3 should be 0.0, got {probs[3]}"


# @pytest.mark.skip(reason="No longer implemented.")
# @pytest.mark.parametrize(
#     "mother_adl, period, education",
#     [
#         # Test different ADL states for alive mothers with no care demand
#         (0, 10, 0),  # No ADL, in caregiving window
#         (0, 5, 0),  # No ADL, before caregiving window
#         (0, 60, 1),  # No ADL, after caregiving window
#         (1, 5, 0),  # ADL 1, before caregiving window (no care demand)
#         (2, 5, 0),  # ADL 2, before caregiving window (no care demand)
#         (1, 60, 1),  # ADL 1, after caregiving window (no care demand)
#         (2, 60, 0),  # ADL 2, after caregiving window (no care demand)
#     ],
# )
# def test_care_demand_death_transition_no_care_demand_alive(
#     mother_adl,
#     period,
#     education,
# ):
#     """Test cases where mother is alive but has no care demand (state 1)."""
#     # Load model_specs from BLD
#     model_specs = pickle.load((BLD / "model" / "specs" / "specs_full.pkl").open("rb"))

#     # mother_adl: 1=No ADL Alive, 2=ADL 1, 3=ADL 2/3 (all alive states)
#     # mother_adl == 0 would be dead, which is not tested here

#     probs = care_demand_death_transition_light_intensive(
#         mother_adl=mother_adl,
#         period=period,
#         education=education,
#         model_specs=model_specs,
#     )

#     # Get caregiving window info
#     end_period_caregiving = model_specs["end_age_caregiving"] -
# model_specs["start_age"]
#     start_period_caregiving = model_specs["start_period_caregiving"]
#     in_caregiving_window = (period >= start_period_caregiving - 1) * (
#         period < end_period_caregiving
#     )

#     # Get ADL and death probabilities
#     # Map mother_adl to internal ADL representation: 1->0, 2->1, 3->2
#     adl_state_for_transition = mother_adl - 1
#     prob_adl = limitations_with_adl_transition(
#         mother_adl=adl_state_for_transition,
#         period=period,
#         education=education,
#         model_specs=model_specs,
#     )
#     # Infer mother_dead from mother_adl: mother_adl == 0 means dead
#     mother_dead = 1 if mother_adl == 0 else 0
#     death_prob_vector = death_transition(
#         period=period,
#         mother_dead=mother_dead,
#         education=education,
#         model_specs=model_specs,
#     )
#     alive_prob = death_prob_vector[0]  # First element is alive probability

#     # State 0 (NO_CARE_DEMAND_DEAD) should be death probability
#     expected_p_dead = 1.0 - alive_prob
#     assert np.isclose(probs[0], expected_p_dead, atol=1e-6), (
#         f"State 0 (dead) probability mismatch: expected {expected_p_dead}, "
#         f"got {probs[0]}"
#     )

#     # State 1 (NO_CARE_DEMAND_ALIVE) should be: alive * (ADL 0 OR
#     # (ADL 1/2/3 but outside window)).
#     expected_p_no_care_alive = alive_prob * (
#         prob_adl[0] + (1 - in_caregiving_window) * (prob_adl[1] + prob_adl[2])
#     )
#     assert np.isclose(probs[1], expected_p_no_care_alive, atol=1e-6), (
#         f"State 1 (no care alive) probability mismatch: "
#         f"expected {expected_p_no_care_alive}, got {probs[1]}"
#     )

#     # States 2 and 3 should be 0 if outside caregiving window
#     if not in_caregiving_window:
#         assert np.isclose(
#             probs[2], 0.0, atol=1e-8
#         ), f"Outside caregiving window: state 2 should be 0.0, got {probs[2]}"
#         assert np.isclose(
#             probs[3], 0.0, atol=1e-8
#         ), f"Outside caregiving window: state 3 should be 0.0, got {probs[3]}"


# @pytest.mark.skip(reason="No longer implemented.")
# @pytest.mark.parametrize(
#     "mother_adl, period, education",
#     [
#         # Test cases in caregiving window with different ADL states
#         # mother_adl: 1=No ADL Alive, 2=ADL 1, 3=ADL 2/3 (all alive states)
#         # Periods should be in caregiving window (typically [14, 40) for start_age=30)
#         (1, 20, 0),  # No ADL Alive, in window
#         (2, 20, 1),  # ADL 1, in window (should have light care demand)
#         (3, 20, 0),  # ADL 2/3, in window (should have intensive care demand)
#     ],
# )
# def test_care_demand_death_transition_in_caregiving_window(
#     mother_adl,
#     period,
#     education,
# ):
#     """Test cases where mother is alive and in caregiving window."""
#     # Load model_specs from BLD
#     model_specs = pickle.load((BLD / "model" / "specs" / "specs_full.pkl").open("rb"))

#     # mother_adl: 1=No ADL Alive, 2=ADL 1, 3=ADL 2/3 (all alive states)
#     # Verify we're in caregiving window
#     end_period_caregiving = model_specs["end_age_caregiving"] -
# model_specs["start_age"]
#     start_period_caregiving = model_specs["start_period_caregiving"]
#     in_caregiving_window = (period >= start_period_caregiving - 1) * (
#         period < end_period_caregiving
#     )
#     assert in_caregiving_window, (
#         f"Period {period} is not in caregiving window "
#         f"[{start_period_caregiving-1}, {end_period_caregiving})"
#     )

#     probs = care_demand_death_transition_light_intensive(
#         mother_adl=mother_adl,
#         period=period,
#         education=education,
#         model_specs=model_specs,
#     )

#     # Get ADL and death probabilities
#     # Map mother_adl to internal ADL representation: 1->0, 2->1, 3->2
#     adl_state_for_transition = mother_adl - 1
#     prob_adl = limitations_with_adl_transition(
#         mother_adl=adl_state_for_transition,
#         period=period,
#         education=education,
#         model_specs=model_specs,
#     )
#     # Infer mother_dead from mother_adl: mother_adl == 0 means dead
#     mother_dead = 1 if mother_adl == 0 else 0
#     death_prob_vector = death_transition(
#         period=period,
#         mother_dead=mother_dead,
#         education=education,
#         model_specs=model_specs,
#     )
#     alive_prob = death_prob_vector[0]

#     # Verify probabilities match expected values
#     expected_p_dead = 1.0 - alive_prob
#     assert np.isclose(probs[0], expected_p_dead, atol=1e-6)

#     expected_p_no_care_alive = alive_prob * prob_adl[0]
#     assert np.isclose(probs[1], expected_p_no_care_alive, atol=1e-6)

#     expected_p_light = alive_prob * prob_adl[1]
#     assert np.isclose(probs[2], expected_p_light, atol=1e-6)

#     expected_p_intensive = alive_prob * prob_adl[2]
#     assert np.isclose(probs[3], expected_p_intensive, atol=1e-6)


@pytest.mark.parametrize(
    "period, mother_dead, education",
    [
        # Test alive mother (state 0)
        (10, 0, 0),
        (20, 0, 1),
        (30, 0, 0),
        # Test recently died mother (state 1) - should transition to state 2
        (10, 1, 0),
        (20, 1, 1),
        # Test longer dead mother (state 2) - should stay in state 2
        (10, 2, 0),
        (20, 2, 1),
        (30, 2, 0),
    ],
)
def test_death_transition_three_states(period, mother_dead, education):
    """Test death_transition function with 3-state system.

    States:
    - 0: alive
    - 1: recently died (inheritance paid this period)
    - 2: longer dead (died in previous periods)

    Transition logic:
    - State 0 -> can stay 0 (alive) or transition to 1 (recently died)
    - State 1 -> transitions to 2 (longer dead) with certainty
    - State 2 -> stays at 2 (longer dead) with certainty
    """
    # Load model_specs from BLD
    model_specs = pickle.load((BLD / "model" / "specs" / "specs_full.pkl").open("rb"))

    probs = death_transition(
        period=period,
        mother_dead=mother_dead,
        education=education,
        model_specs=model_specs,
    )

    # 1) Probabilities must sum to 1
    assert np.isclose(
        probs.sum(), 1.0, atol=1e-8
    ), f"Probabilities sum to {probs.sum()}, expected 1.0"

    # 2) Must have 3 states: [alive_prob, recently_died_prob, longer_dead_prob]
    assert probs.shape == (3,), f"Expected shape (3,), got {probs.shape}"

    # 3) All probabilities must be non-negative
    assert np.all(probs >= 0), f"Negative probabilities found: {probs}"

    # 4) Test specific transition behavior
    if mother_dead == 0:
        # If alive: can stay alive or die (become recently dead)
        # Both probabilities should be between 0 and 1
        assert 0.0 <= probs[0] <= 1.0, f"Alive prob should be [0,1], got {probs[0]}"
        assert (
            0.0 <= probs[1] <= 1.0
        ), f"Recently died prob should be [0,1], got {probs[1]}"
        assert np.isclose(
            probs[2], 0.0, atol=1e-8
        ), f"If alive, longer dead prob should be 0, got {probs[2]}"
        # Alive and recently died should sum to 1
        assert np.isclose(
            probs[0] + probs[1], 1.0, atol=1e-8
        ), f"Alive + recently died should sum to 1, got {probs[0] + probs[1]}"
    elif mother_dead == PARENT_RECENTLY_DEAD:
        # If recently died: transitions to longer dead (state 2) with certainty
        assert np.isclose(
            probs[0], 0.0, atol=1e-8
        ), f"If recently died, alive prob should be 0, got {probs[0]}"
        assert np.isclose(
            probs[1], 0.0, atol=1e-8
        ), f"If recently died, recently died prob should be 0, got {probs[1]}"
        assert np.isclose(
            probs[2], 1.0, atol=1e-8
        ), f"If recently died, longer dead prob should be 1, got {probs[2]}"
    elif mother_dead == PARENT_LONGER_DEAD:
        # If longer dead: stays longer dead (state 2) with certainty
        assert np.isclose(
            probs[0], 0.0, atol=1e-8
        ), f"If longer dead, alive prob should be 0, got {probs[0]}"
        assert np.isclose(
            probs[1], 0.0, atol=1e-8
        ), f"If longer dead, recently died prob should be 0, got {probs[1]}"
        assert np.isclose(
            probs[2], 1.0, atol=1e-8
        ), f"If longer dead, longer dead prob should be 1, got {probs[2]}"
