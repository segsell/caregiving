"""Tests for drawing caregiving_type from unobserved-type share parameters."""

import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.estimation.estimation_setup import draw_caregiving_type_from_params


@pytest.fixture(scope="module")
def initial_states_and_seed():
    """Create initial states and seed, similar to task_generate_initial_conditions."""
    options = pickle.load((BLD / "model" / "options.pkl").open("rb"))
    seed = options["model_params"]["seed"]
    n_agents = options["model_params"]["n_agents"]

    # Create synthetic education: half low education (0), half high (1).
    education = jnp.concatenate(
        [jnp.zeros(n_agents, dtype=jnp.uint8), jnp.ones(n_agents, dtype=jnp.uint8)]
    )

    # Initialize caregiving_type to 50% type 0 and 50% type 1 within each
    # education group, mimicking the initial 50/50 setup.
    half = n_agents // 2
    caregiving_low = jnp.concatenate(
        [
            jnp.zeros(half, dtype=jnp.uint8),
            jnp.ones(n_agents - half, dtype=jnp.uint8),
        ]
    )
    caregiving_high = caregiving_low.copy()
    caregiving_type_init = jnp.concatenate([caregiving_low, caregiving_high])

    initial_states = {
        "education": education,
        "caregiving_type": caregiving_type_init,
    }

    return initial_states, education, seed


@pytest.mark.parametrize(
    "p_low,p_high",
    [
        # Shares are defined within each education group separately,
        # so they do not need to sum to one across education groups.
        (0.3, 0.7),
        (0.5, 0.5),
        (0.1, 0.9),
        (0.9, 0.1),
        (0.2, 0.4),
        (0.8, 0.8),
        (0.0, 0.0),
        (1.0, 1.0),
    ],
)
def test_draw_caregiving_type_from_params_shares(
    p_low: float, p_high: float, initial_states_and_seed
) -> None:
    """Check that drawing caregiving_type matches target shares by education."""
    initial_states, education, seed = initial_states_and_seed

    # Target shares for unobserved type 1.
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

    # Realized shares by education.
    mask_low = education == 0
    mask_high = education == 1

    realized_low = float(caregiving_type[mask_low].mean())
    realized_high = float(caregiving_type[mask_high].mean())

    assert np.allclose(realized_low, p_low, atol=0.02)
    assert np.allclose(realized_high, p_high, atol=0.02)
