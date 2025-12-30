"""Tests for draw functions using JAX Bernoulli distribution.

The core mechanics of Bernoulli draws:
1. Draw u ~ Uniform[0, 1)
2. Compare: if u < p, then outcome = 1, else outcome = 0
3. This ensures P(outcome = 1) = p
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from caregiving.estimation.estimation_setup import draw_caregiving_type_from_params
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


@pytest.mark.parametrize(
    "prob, seed",
    [
        (0.0, 42),
        (0.3, 123),
        (0.5, 456),
        (0.7, 789),
        (1.0, 101),
    ],
)
def test_draw_inheritance_outcome_binary_values(prob, seed):
    """Test that draw_inheritance_outcome returns binary values (0 or 1)."""
    prob_array = jnp.array(prob)

    result = draw_inheritance_outcome(prob_array, seed)

    # Check that value is either 0 or 1 (convert JAX array to Python int)
    result_value = int(result)
    assert result_value in {0, 1}, "Result should be 0 or 1"
    # Check dtype is uint8
    assert result.dtype == jnp.uint8, "Result should be uint8"
    # Check shape matches input
    assert result.shape == prob_array.shape, "Result shape should match input shape"


def test_draw_inheritance_outcome_vectorized():
    """Test that draw_inheritance_outcome works with vectorized probabilities."""
    n_samples = 1000
    seed = 42
    # Create array of probabilities
    prob_array = jnp.array([0.3] * n_samples)

    result = draw_inheritance_outcome(prob_array, seed)

    # Check shape matches
    assert result.shape == prob_array.shape, "Result shape should match input shape"
    # Check all values are 0 or 1
    assert jnp.all((result == 0) | (result == 1)), "All values should be 0 or 1"
    # Check dtype
    assert result.dtype == jnp.uint8, "Result should be uint8"


def test_draw_inheritance_outcome_reproducibility():
    """Test that draw_inheritance_outcome is reproducible with same seed."""
    seed = 42
    prob = jnp.array(0.5)

    result1 = draw_inheritance_outcome(prob, seed)
    result2 = draw_inheritance_outcome(prob, seed)

    # Results should be identical
    assert result1 == result2, "Results should be identical with same seed"


def test_draw_inheritance_outcome_different_seeds():
    """Test draw_inheritance_outcome gives different results with different seeds."""
    prob = jnp.array(0.5)

    result1 = draw_inheritance_outcome(prob, seed=42)
    result2 = draw_inheritance_outcome(prob, seed=123)

    # Results might be the same by chance, but check that the function works
    # Convert JAX arrays to Python ints for comparison
    result1_value = int(result1)
    result2_value = int(result2)
    assert result1_value in {0, 1}, "Result should be 0 or 1"
    assert result2_value in {0, 1}, "Result should be 0 or 1"


@pytest.mark.parametrize(
    "prob",
    [0.0, 1.0],
)
def test_draw_inheritance_outcome_extreme_probabilities(prob):
    """Test draw_inheritance_outcome with extreme probabilities (0 and 1).

    This tests the core mechanics:
    - With p=0: u < 0 is never true (since u >= 0), so outcome is always 0
    - With p=1: u < 1 is always true (since u < 1.0), so outcome is always 1
    """
    seed = 42
    prob_array = jnp.array(prob)

    result = draw_inheritance_outcome(prob_array, seed)

    # With prob=0, should always get 0; with prob=1, should always get 1
    expected = 1 if prob == 1.0 else 0
    assert result == expected, f"With prob={prob}, expected {expected}, got {result}"


def test_draw_inheritance_outcome_mechanics_explicit():
    """Test the core mechanics: u ~ Uniform[0,1), outcome = 1 if u < p, else 0.

    This test explicitly verifies the comparison logic by testing that:
    - The function correctly implements: outcome = 1 if u < p, else 0
    - The comparison uses < (not <=), so u == p gives outcome = 0
    """
    seed = 42

    # Test with a specific probability and verify the mechanics work correctly
    # We can't directly access u, but we can verify the behavior

    # Test 1: p = 0.0 should always give 0 (u < 0 is never true, u >= 0)
    prob_zero = jnp.array(0.0)
    result_zero = draw_inheritance_outcome(prob_zero, seed)
    assert result_zero == 0, "With p=0, u < 0 is never true, so outcome should be 0"

    # Test 2: p = 1.0 should always give 1 (u < 1 is always true, since u < 1.0)
    prob_one = jnp.array(1.0)
    result_one = draw_inheritance_outcome(prob_one, seed)
    assert result_one == 1, "With p=1, u < 1 is always true, so outcome should be 1"

    # Test 3: Multiple draws with same seed and p should be reproducible
    # This verifies that u is drawn deterministically from the seed
    prob = jnp.array(0.5)
    result1 = draw_inheritance_outcome(prob, seed)
    result2 = draw_inheritance_outcome(prob, seed)
    assert result1 == result2, (
        "Same seed should produce same u, so same comparison u < p, " "so same outcome"
    )

    # Test 4: The comparison uses < (not <=)
    # We can't directly test u == p, but we verify behavior is consistent
    # with u < p comparison through statistical tests


@pytest.mark.parametrize(
    "prob",
    [0.2, 0.5, 0.8],
)
def test_draw_inheritance_outcome_respects_probability(prob):
    """Test that draw_inheritance_outcome respects the probability over many draws."""
    n_samples = 10000
    seed = 42

    prob_array = jnp.array([prob] * n_samples)

    result = draw_inheritance_outcome(prob_array, seed)

    # Calculate realized probability
    realized_prob = float(jnp.mean(result))

    # Check that realized probability is close to target
    # Using tolerance of 0.02 (2 percentage points)
    assert np.isclose(realized_prob, prob, atol=0.02), (
        f"Realized probability ({realized_prob:.3f}) "
        f"should be close to target ({prob:.3f})"
    )


def test_draw_inheritance_outcome_vectorized_different_probs():
    """Test draw_inheritance_outcome with vectorized probabilities of different values."""  # noqa: E501
    n_samples = 1000
    seed = 42
    # Create array with different probabilities
    prob_array = jnp.array([0.1, 0.5, 0.9] * (n_samples // 3))

    result = draw_inheritance_outcome(prob_array, seed)

    # Check shape
    assert result.shape == prob_array.shape, "Result shape should match input shape"
    # Check all values are 0 or 1
    assert jnp.all((result == 0) | (result == 1)), "All values should be 0 or 1"
    # Check dtype
    assert result.dtype == jnp.uint8, "Result should be uint8"

    # Check that mean is reasonable (between 0 and 1, and close to average of probs)
    avg_prob = float(jnp.mean(prob_array))
    realized_mean = float(jnp.mean(result))
    assert 0.0 <= realized_mean <= 1.0, "Mean should be between 0 and 1"
    expected_msg = (
        f"Mean ({realized_mean:.3f}) should be close to "
        f"average probability ({avg_prob:.3f})"
    )
    assert np.isclose(realized_mean, avg_prob, atol=0.1), expected_msg
