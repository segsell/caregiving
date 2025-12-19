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
    education: jnp.array,
    params: dict[str, float],
):
    """Compute the utility in the final period including bequest."""
    # rho = params["rho"]
    rho_bequest = (
        params["rho_bequest_low"] * (1 - education)
        + params["rho_bequest_high"] * education
    )
    bequest_scale = (
        params["bequest_scale_low"] * (1 - education)
        + params["bequest_scale_high"] * education
    )

    bequest_unscaled_with_rho_not_one = (wealth ** (1 - rho_bequest) - 1) / (
        1 - rho_bequest
    )

    bequest_unscaled = jax.lax.select(
        jnp.allclose(rho_bequest, 1),
        jnp.log(wealth),
        bequest_unscaled_with_rho_not_one,
    )

    return bequest_scale * bequest_unscaled


def marginal_utility_final_consume_all(
    wealth: jnp.array,
    education: jnp.array,
    params: dict[str, float],
) -> jnp.array:
    """Compute marginal utility in the final period."""
    # rho = params["rho"]
    # bequest_scale = params["bequest_scale"]
    rho_bequest = (
        params["rho_bequest_low"] * (1 - education)
        + params["rho_bequest_high"] * education
    )
    bequest_scale = (
        params["bequest_scale_low"] * (1 - education)
        + params["bequest_scale_high"] * education
    )

    return bequest_scale * (wealth ** (-rho_bequest))
