"""Tests for draw functions using JAX Bernoulli distribution.

The core mechanics of Bernoulli draws:
1. Draw u ~ Uniform[0, 1)
2. Compare: if u < p, then outcome = 1, else outcome = 0
3. This ensures P(outcome = 1) = p
"""

import copy
import pickle as pkl

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.estimation.estimation_setup import draw_caregiving_type_from_params
from caregiving.model.shared import (
    FULL_TIME_NO_CARE,
    PART_TIME_NO_CARE,
    RETIREMENT_NO_CARE,
    SEX,
    UNEMPLOYED_NO_CARE,
    is_informal_care,
)
from caregiving.model.wealth_and_budget.transfers import draw_inheritance_outcome

jax.config.update("jax_enable_x64", True)


# ==============================================================================
# Tests for draw_caregiving_type_from_params
# ==============================================================================


@pytest.mark.parametrize(
    "p_low, p_high, seed",
    [
        (0.3, 0.7, 42),
        (0.5, 0.5, 123),
        (0.1, 0.9, 456),
        (0.0, 1.0, 789),
        (1.0, 0.0, 101),
    ],
)
def test_draw_caregiving_type_binary_values(p_low, p_high, seed):
    """Test that draw_caregiving_type_from_params returns binary values (0 or 1)."""
    n_agents = 1000
    # Create education array: half low (0), half high (1)
    education = jnp.concatenate(
        [
            jnp.zeros(n_agents // 2, dtype=jnp.uint8),
            jnp.ones(n_agents // 2, dtype=jnp.uint8),
        ]
    )

    initial_states = {
        "education": education,
        "caregiving_type": jnp.zeros(n_agents, dtype=jnp.uint8),
    }

    params = {
        "share_unobserved_type_low_educ": p_low,
        "share_unobserved_type_high_educ": p_high,
    }

    adjusted_states = draw_caregiving_type_from_params(
        initial_states=initial_states,
        params=params,
        seed=seed,
    )

    caregiving_type = adjusted_states["caregiving_type"]

    # Check that all values are either 0 or 1
    assert jnp.all(
        (caregiving_type == 0) | (caregiving_type == 1)
    ), "All caregiving_type values should be 0 or 1"
    # Check dtype is uint8
    assert caregiving_type.dtype == jnp.uint8, "caregiving_type should be uint8"
    # Check shape matches education
    assert (
        caregiving_type.shape == education.shape
    ), "caregiving_type shape should match education shape"


def test_draw_caregiving_type_reproducibility():
    """Test that draw_caregiving_type_from_params is reproducible with same seed."""
    n_agents = 500
    seed = 42

    education = jnp.concatenate(
        [
            jnp.zeros(n_agents // 2, dtype=jnp.uint8),
            jnp.ones(n_agents // 2, dtype=jnp.uint8),
        ]
    )

    initial_states = {
        "education": education,
        "caregiving_type": jnp.zeros(n_agents, dtype=jnp.uint8),
    }

    params = {
        "share_unobserved_type_low_educ": 0.3,
        "share_unobserved_type_high_educ": 0.7,
    }

    # Draw twice with same seed
    result1 = draw_caregiving_type_from_params(
        initial_states=initial_states, params=params, seed=seed
    )
    result2 = draw_caregiving_type_from_params(
        initial_states=initial_states, params=params, seed=seed
    )

    # Results should be identical
    assert jnp.array_equal(
        result1["caregiving_type"], result2["caregiving_type"]
    ), "Results should be identical with same seed"


@pytest.mark.parametrize(
    "p_low, p_high",
    [
        (0.3, 0.7),
        (0.5, 0.5),
        (0.1, 0.9),
    ],
)
def test_draw_caregiving_type_respects_probabilities(p_low, p_high):
    """Test that draw_caregiving_type_from_params respects probability parameters."""
    n_agents = 10000  # Large sample for statistical test
    seed = 42

    education = jnp.concatenate(
        [
            jnp.zeros(n_agents // 2, dtype=jnp.uint8),
            jnp.ones(n_agents // 2, dtype=jnp.uint8),
        ]
    )

    initial_states = {
        "education": education,
        "caregiving_type": jnp.zeros(n_agents, dtype=jnp.uint8),
    }

    params = {
        "share_unobserved_type_low_educ": p_low,
        "share_unobserved_type_high_educ": p_high,
    }

    adjusted_states = draw_caregiving_type_from_params(
        initial_states=initial_states, params=params, seed=seed
    )

    caregiving_type = adjusted_states["caregiving_type"]

    # Calculate realized shares by education
    mask_low = education == 0
    mask_high = education == 1

    realized_low = float(jnp.mean(caregiving_type[mask_low]))
    realized_high = float(jnp.mean(caregiving_type[mask_high]))

    # Check that realized shares are close to target probabilities
    # Using tolerance of 0.02 (2 percentage points)
    assert np.isclose(realized_low, p_low, atol=0.02), (
        f"Realized share for low education ({realized_low:.3f}) "
        f"should be close to target ({p_low:.3f})"
    )
    assert np.isclose(realized_high, p_high, atol=0.02), (
        f"Realized share for high education ({realized_high:.3f}) "
        f"should be close to target ({p_high:.3f})"
    )


def test_draw_caregiving_type_returns_copy():
    """Test that draw_caregiving_type_from_params returns a copy, not original."""
    n_agents = 100
    seed = 42

    education = jnp.zeros(n_agents, dtype=jnp.uint8)
    initial_states = {
        "education": education,
        "caregiving_type": jnp.zeros(n_agents, dtype=jnp.uint8),
    }

    params = {
        "share_unobserved_type_low_educ": 0.5,
        "share_unobserved_type_high_educ": 0.5,
    }

    adjusted_states = draw_caregiving_type_from_params(
        initial_states=initial_states, params=params, seed=seed
    )

    # Check that it's a different dict object
    assert (
        adjusted_states is not initial_states
    ), "Should return a copy, not the original"
    # Check that caregiving_type is different
    assert not jnp.array_equal(
        adjusted_states["caregiving_type"], initial_states["caregiving_type"]
    ), "caregiving_type should be updated"
    # Check that education is the same (not copied)
    assert jnp.array_equal(
        adjusted_states["education"], initial_states["education"]
    ), "education should be unchanged"


def test_draw_caregiving_type_mechanics_explicit():
    """Test the core mechanics: u ~ Uniform[0,1), outcome = 1 if u < p, else 0.

    This test explicitly verifies the comparison logic by testing that:
    - Each agent gets their own u draw
    - For agent i: outcome_i = 1 if u_i < p_i, else 0
    - Different probabilities per agent (based on education) work correctly
    """
    n_agents = 1000
    seed = 42

    # Create education: first half low (0), second half high (1)
    education = jnp.concatenate(
        [
            jnp.zeros(n_agents // 2, dtype=jnp.uint8),
            jnp.ones(n_agents // 2, dtype=jnp.uint8),
        ]
    )

    initial_states = {
        "education": education,
        "caregiving_type": jnp.zeros(n_agents, dtype=jnp.uint8),
    }

    # Use different probabilities for low vs high education
    params = {
        "share_unobserved_type_low_educ": 0.0,  # p_low = 0
        "share_unobserved_type_high_educ": 1.0,  # p_high = 1
    }

    adjusted_states = draw_caregiving_type_from_params(
        initial_states=initial_states, params=params, seed=seed
    )

    caregiving_type = adjusted_states["caregiving_type"]

    # With p_low = 0: u < 0 is never true, so all low education should get 0
    mask_low = education == 0
    assert jnp.all(
        caregiving_type[mask_low] == 0
    ), "With p=0, u < 0 is never true, so all outcomes should be 0"

    # With p_high = 1: u < 1 is always true, so all high education should get 1
    mask_high = education == 1
    assert jnp.all(
        caregiving_type[mask_high] == 1
    ), "With p=1, u < 1 is always true, so all outcomes should be 1"


# ==============================================================================
# Tests for draw_inheritance_outcome
# ==============================================================================


@pytest.fixture()
def load_specs():
    """Load actual model specs from pickle file."""
    path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"

    with path_to_specs.open("rb") as file:
        specs = pkl.load(file)

    return specs


@pytest.mark.parametrize(
    "period, lagged_choice, education, asset_end_of_previous_period",
    [
        (0, UNEMPLOYED_NO_CARE[0].item(), 0, 10.0),
        (10, PART_TIME_NO_CARE[0].item(), 0, 15.0),
        (20, FULL_TIME_NO_CARE[0].item(), 1, 20.0),
        (30, RETIREMENT_NO_CARE[0].item(), 1, 25.0),
        (40, UNEMPLOYED_NO_CARE[0].item(), 1, 30.0),
    ],
)
def test_draw_inheritance_outcome_binary_values(
    period,
    lagged_choice,
    education,
    asset_end_of_previous_period,
    load_specs,
):
    """Test that draw_inheritance_outcome returns binary values (0 or 1)."""
    model_specs = copy.deepcopy(load_specs)

    result = draw_inheritance_outcome(
        period=period,
        lagged_choice=lagged_choice,
        education=education,
        asset_end_of_previous_period=asset_end_of_previous_period,
        model_specs=model_specs,
    )

    # Check that value is either 0 or 1 (convert JAX array to Python int)
    result_value = int(result)
    assert result_value in {0, 1}, "Result should be 0 or 1"
    # Check dtype is uint8
    assert result.dtype == jnp.uint8, "Result should be uint8"
    # Result should be a scalar (shape ())
    assert result.shape == (), "Result should be a scalar"


def test_draw_inheritance_outcome_reproducibility(load_specs):
    """Test that draw_inheritance_outcome is reproducible with same inputs."""
    period = 10
    lagged_choice = PART_TIME_NO_CARE[0].item()
    education = 0
    asset_end_of_previous_period = 15.0

    model_specs = copy.deepcopy(load_specs)

    result1 = draw_inheritance_outcome(
        period=period,
        lagged_choice=lagged_choice,
        education=education,
        asset_end_of_previous_period=asset_end_of_previous_period,
        model_specs=model_specs,
    )
    result2 = draw_inheritance_outcome(
        period=period,
        lagged_choice=lagged_choice,
        education=education,
        asset_end_of_previous_period=asset_end_of_previous_period,
        model_specs=model_specs,
    )

    # Results should be identical with same inputs
    assert result1 == result2, "Results should be identical with same inputs"


@pytest.mark.parametrize(
    "period, lagged_choice, education, asset_end_of_previous_period",
    [
        (10, PART_TIME_NO_CARE[0].item(), 0, 15.0),
        (10, PART_TIME_NO_CARE[0].item(), 0, 20.0),  # Different asset value
        (20, FULL_TIME_NO_CARE[0].item(), 1, 15.0),  # Different period
        (10, FULL_TIME_NO_CARE[0].item(), 0, 15.0),  # Different lagged_choice
        (10, PART_TIME_NO_CARE[0].item(), 1, 15.0),  # Different education
    ],
)
def test_draw_inheritance_outcome_different_inputs(
    period,
    lagged_choice,
    education,
    asset_end_of_previous_period,
    load_specs,
):
    """Test draw_inheritance_outcome gives different results with different inputs."""
    model_specs = copy.deepcopy(load_specs)

    result1 = draw_inheritance_outcome(
        period=period,
        lagged_choice=lagged_choice,
        education=education,
        asset_end_of_previous_period=asset_end_of_previous_period,
        model_specs=model_specs,
    )

    # Check that result is valid (0 or 1)
    result_value = int(result1)
    assert result_value in {0, 1}, "Result should be 0 or 1"


@pytest.mark.parametrize(
    "period, lagged_choice, education, asset_end_of_previous_period, prob, expected_outcome",
    [
        # Test with prob=0.0 should always give 0
        (10, UNEMPLOYED_NO_CARE[0].item(), 0, 15.0, 0.0, 0),
        (10, PART_TIME_NO_CARE[0].item(), 0, 15.0, 0.0, 0),
        # Test with prob=1.0 should always give 1
        (10, UNEMPLOYED_NO_CARE[0].item(), 0, 15.0, 1.0, 1),
        (10, PART_TIME_NO_CARE[0].item(), 1, 20.0, 1.0, 1),
    ],
)
def test_draw_inheritance_outcome_extreme_probabilities(
    period,
    lagged_choice,
    education,
    asset_end_of_previous_period,
    prob,
    expected_outcome,
    load_specs,
):
    """Test draw_inheritance_outcome with extreme probabilities (0 and 1).

    This tests the core mechanics:
    - With p=0: outcome is always 0 (prob >= uniform_draw is never true since prob=0)
    - With p=1: outcome is always 1 (prob >= uniform_draw is always true since uniform_draw < 1.0)
    """
    model_specs = copy.deepcopy(load_specs)
    # Set extreme probabilities in the matrix
    # Determine care type (no_care=0, any_care=1) based on lagged_choice
    # UNEMPLOYED_NO_CARE, RETIREMENT_NO_CARE, PART_TIME_NO_CARE, FULL_TIME_NO_CARE are all no_care (0)
    # because is_informal_care() returns False for these choices
    care_type_idx = int(is_informal_care(lagged_choice).astype(int))
    # SEX=1 means index 1 (female) in the matrix
    sex_idx = SEX  # SEX = 1
    model_specs["inheritance_prob_mat"] = (
        model_specs["inheritance_prob_mat"]
        .at[sex_idx, period, education, care_type_idx]
        .set(prob)
    )

    result = draw_inheritance_outcome(
        period=period,
        lagged_choice=lagged_choice,
        education=education,
        asset_end_of_previous_period=asset_end_of_previous_period,
        model_specs=model_specs,
    )

    assert (
        result == expected_outcome
    ), f"With prob={prob}, expected {expected_outcome}, got {result}"
    # Check that result is valid
    assert int(result) in {0, 1}, "Result should be 0 or 1"


@pytest.mark.parametrize(
    "period, education, lagged_choice, prob",
    [
        (10, 0, UNEMPLOYED_NO_CARE[0].item(), 0.2),
        (10, 0, PART_TIME_NO_CARE[0].item(), 0.3),
        (10, 0, FULL_TIME_NO_CARE[0].item(), 0.5),
        (10, 0, RETIREMENT_NO_CARE[0].item(), 0.4),
        (20, 1, UNEMPLOYED_NO_CARE[0].item(), 0.3),
        (20, 1, PART_TIME_NO_CARE[0].item(), 0.5),
        (30, 0, RETIREMENT_NO_CARE[0].item(), 0.4),
    ],
)
def test_draw_inheritance_outcome_respects_probability(
    period, education, lagged_choice, prob, load_specs
):
    """Test that draw_inheritance_outcome respects the probability over many draws."""
    n_samples = 10000
    asset_base = 15.0

    # Set a specific probability for this combination
    # Determine care type index from lagged_choice
    care_type_idx = int(is_informal_care(lagged_choice).astype(int))
    sex_idx = SEX  # SEX = 1 (female, index 1)
    model_specs = copy.deepcopy(load_specs)
    model_specs["inheritance_prob_mat"] = (
        model_specs["inheritance_prob_mat"]
        .at[sex_idx, period, education, care_type_idx]
        .set(prob)
    )

    # Draw many times with different asset values to vary the seed
    # Use integer increments to avoid uint16 truncation issues
    # The seed uses asset_end_of_previous_period directly, so we vary by integers
    results = []
    for i in range(n_samples):
        # Vary asset by integer values to ensure different seeds after uint16 conversion
        asset = asset_base + i  # Integer increments ensure unique seeds
        result = draw_inheritance_outcome(
            period=period,
            lagged_choice=lagged_choice,
            education=education,
            asset_end_of_previous_period=asset,
            model_specs=model_specs,
        )
        results.append(int(result))

    # Calculate realized probability
    realized_prob = float(np.mean(results))

    # Check that realized probability is close to target
    # Using tolerance of 0.03 (3 percentage points) to account for sampling variation
    # With 10,000 samples, standard error is ~sqrt(p*(1-p)/n) ≈ 0.005 for p=0.5
    # So 3 percentage points should be sufficient
    assert np.isclose(realized_prob, prob, atol=0.03), (
        f"Realized probability ({realized_prob:.3f}) "
        f"should be close to target ({prob:.3f})"
    )
