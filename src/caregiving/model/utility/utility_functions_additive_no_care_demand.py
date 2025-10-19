"""Counterfactual utility functions (additive) with no informal care choices.

This mirrors the API of the standard additive utility module but is intended
for the reduced state-choice space without informal caregiving.
"""

import jax
import jax.numpy as jnp

from caregiving.model.shared import (
    is_dead,
)
from caregiving.model.utility.bequest_utility import utility_final_consume_all
from caregiving.model.utility.utility_components_no_care_demand import (
    consumption_scale,
    disutility_work,
)


def create_utility_functions():
    """Create dict of utility functions for no-care-demand counterfactual."""
    return {
        "utility": utility_func_additive_no_care_demand,
        "marginal_utility": marginal_utility_func_additive_alive_no_care_demand,
        "inverse_marginal_utility": inverse_marginal_additive_no_care_demand,
    }


# =====================================================================================
# Per-period utility (no informal care in the choice set)
# =====================================================================================


def utility_func_additive_no_care_demand(
    consumption: jnp.array,
    choice: int,
    period: int,
    education: int,
    health: int,
    partner_state: int,
    params: dict,
    options: dict,
) -> jnp.array:
    """Per-period utility using CRRA with additive disutility of labor.

    Identical form to the standard additive utility, but provided as a
    separate module for the counterfactual model without informal care.
    """
    utility_alive = utility_func_alive_additive_no_care_demand(
        consumption=consumption,
        partner_state=partner_state,
        education=education,
        health=health,
        period=period,
        choice=choice,
        params=params,
        options=options,
    )
    utility_death = utility_final_consume_all(
        wealth=consumption,
        education=education,
        params=params,
    )
    death_bool = is_dead(health)
    utility = jax.lax.select(death_bool, utility_death, utility_alive)
    return utility


def utility_func_alive_additive_no_care_demand(
    consumption,
    partner_state,
    education,
    health,
    period,
    choice,
    params,
    options,
):
    """Alive-period utility with additive labor disutility (no care demand)."""
    rho = params["rho_low"] * (1 - education) + params["rho_high"] * education

    disutil_work = disutility_work(
        period=period,
        choice=choice,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    cons_scale, hh_size = consumption_scale(
        partner_state=partner_state,
        education=education,
        period=period,
        options=options,
    )

    scaled_consumption = consumption / cons_scale
    utility_rho_not_one = (scaled_consumption ** (1 - rho) - 1) / (1 - rho)
    utility = jax.lax.select(
        jnp.allclose(rho, 1),
        jnp.log(consumption / cons_scale),
        utility_rho_not_one,
    )
    return hh_size * utility + disutil_work


def marginal_utility_func_additive_alive_no_care_demand(
    consumption, partner_state, education, health, period, choice, params, options
):
    rho = params["rho_low"] * (1 - education) + params["rho_high"] * education
    cons_scale, hh_size = consumption_scale(
        partner_state=partner_state,
        education=education,
        period=period,
        options=options,
    )
    marg_util_rho_not_one = hh_size * (consumption / cons_scale) ** (-rho) / cons_scale
    marg_util = jax.lax.select(
        jnp.allclose(rho, 1),
        hh_size / consumption,
        marg_util_rho_not_one,
    )
    return marg_util


def inverse_marginal_additive_no_care_demand(
    marginal_utility,
    partner_state,
    education,
    health,
    period,
    choice,
    params,
    options,
):
    rho = params["rho_low"] * (1 - education) + params["rho_high"] * education
    cons_scale, hh_size = consumption_scale(
        partner_state=partner_state,
        education=education,
        period=period,
        options=options,
    )
    consumption_rho_not_one = cons_scale * (
        marginal_utility * cons_scale / hh_size
    ) ** (-1 / rho)
    consumption = jax.lax.select(
        jnp.allclose(rho, 1), hh_size / marginal_utility, consumption_rho_not_one
    )
    return consumption
