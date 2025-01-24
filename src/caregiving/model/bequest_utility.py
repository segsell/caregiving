"""Utility of bequest in the final period."""

import jax.numpy as jnp


def create_final_period_utility_functions():
    """Create dict of utility functions for the final period."""
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


def utility_final_consume_all(
    resources: jnp.array,
    params: dict[str, float],
):
    """Compute the utility in the final period including bequest."""
    rho = params["rho"]
    bequest_scale = params["bequest_scale"]

    return bequest_scale * (resources ** (1 - rho) - 1) / (1 - rho)


def marginal_utility_final_consume_all(
    resources: jnp.array,
    params: dict[str, float],
) -> jnp.array:
    """Compute marginal utility in the final period."""
    bequest_scale = params["bequest_scale"]

    return bequest_scale * (resources ** (-params["rho"]))
