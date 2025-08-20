"""Utility of bequest in the final period."""

from typing import Any

import jax
import jax.numpy as jnp


def create_final_period_utility_functions():
    """Create dict of utility functions for the final period."""
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


def utility_final_consume_all(
    wealth: jnp.array,
    params: dict[str, float],
    # options: dict[str, Any],
):
    """Compute the utility in the final period including bequest."""
    rho = params["rho"]
    bequest_scale = params["bequest_scale"]

    bequest_unscaled_with_rho_not_one = (wealth ** (1 - rho) - 1) / (1 - rho)

    bequest_unscaled = jax.lax.select(
        jnp.allclose(rho, 1),
        jnp.log(wealth),
        bequest_unscaled_with_rho_not_one,
    )

    return bequest_scale * bequest_unscaled


def marginal_utility_final_consume_all(
    wealth: jnp.array,
    params: dict[str, float],
    # options: dict[str, Any],
) -> jnp.array:
    """Compute marginal utility in the final period."""
    rho = params["rho"]
    bequest_scale = params["bequest_scale"]

    return bequest_scale * (wealth ** (-rho))
