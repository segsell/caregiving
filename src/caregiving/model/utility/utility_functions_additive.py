"""Utility functions for the model with multiplicative disutility of labor."""

import jax
import jax.numpy as jnp

from caregiving.model.shared import (  # is_nursing_home_care,
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    CARE_DEMAND_AND_OTHER_SUPPLY,
    PARENT_BAD_HEALTH,
    PARENT_GOOD_HEALTH,
    PARENT_MEDIUM_HEALTH,
    PERIOD_SCALE,
    SEX,
    is_bad_health,
    is_child_age_0_to_3,
    is_child_age_4_to_6,
    is_child_age_7_to_9,
    is_dead,
    is_full_time,
    is_good_health,
    is_informal_care,
    is_intensive_informal_care,
    is_light_informal_care,
    is_no_care,
    is_part_time,
    is_unemployed,
)
from caregiving.model.utility.bequest_utility import utility_final_consume_all
from caregiving.model.utility.utility_components import (
    consumption_scale,
    disutility_work,
)


def create_utility_functions():
    """Create dict of utility functions."""
    return {
        "utility": utility_func_additive,
        "marginal_utility": marginal_utility_func_additive_alive,
        "inverse_marginal_utility": inverse_marginal_additive,
    }


def _create_utility_functions():
    """Create dict of utility functions."""
    return {
        "utility": utility_func_adda,
        "marginal_utility": marginal_utility_func_adda_alive,
        "inverse_marginal_utility": inverse_marginal_adda,
    }


# =====================================================================================
# Per-period utility
# =====================================================================================


def utility_func_adda(
    consumption: jnp.array,
    choice: int,
    period: int,
    education: int,
    health: int,
    care_demand: int,
    partner_state: int,
    params: dict,
    options: dict,
) -> jnp.array:
    """Compute the per-period utility based on a CRRA utility function.

    Args:
        period (int): Current period.
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        mother_alive (int): Indicator for whether the mother is alive.
            0 = mother is not alive, 1 = mother is alive.
        father_alive (int): Indicator for whether the father is alive.
            0 = father is not alive, 1 = father is alive.
        mother_health (int): Health status of the mother. One of 0, 1, 2.
            0 = good health, 1 = medium health, 2 = bad health.
        father_health (int): Health status of the father. One of 0, 1, 2.
            0 = good health, 1 = medium health, 2 = bad health.
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.
        options (dict): Dictionary containing model options.

    Returns:
        utility (jnp.array): Agent's utility. Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    utility_alive = utility_func_alive_adda(
        consumption=consumption,
        # sex=sex,
        partner_state=partner_state,
        education=education,
        health=health,
        care_demand=care_demand,
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


def utility_func_alive_adda(
    consumption,
    partner_state,
    education,
    health,
    care_demand,
    period,
    choice,
    params,
    options,
):
    """Calculate the choice specific cobb-douglas utility, i.e. u =
    ((c*eta/consumption_scale)^(1-rho))/(1-rho) ."""
    # gather params
    # rho = params["rho"]
    rho = params["rho_low"] * (1 - education) + params["rho_high"] * education

    disutil_work = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        care_demand=care_demand,
        params=params,
        options=options,
    )
    eta = jnp.exp(disutil_work)
    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
    )

    # zeta = utility_of_caregiving(
    #     period,
    #     choice,
    #     education,
    #     health=health,
    #     care_demand=care_demand,
    #     params=params,
    #     options=options,
    # )

    # compute utility
    scaled_consumption = consumption / cons_scale
    utility_rho_not_one = (scaled_consumption ** (1 - rho) - 1) / (1 - rho)

    utility = jax.lax.select(
        jnp.allclose(rho, 1),
        jnp.log(consumption / cons_scale),
        utility_rho_not_one,
    )
    return utility * eta  # + zeta * care_demand


def marginal_utility_func_adda_alive(
    consumption, partner_state, education, health, period, choice, params, options
):

    # rho = params["rho"]
    rho = params["rho_low"] * (1 - education) + params["rho_high"] * education

    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
    )
    disutil_work = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    eta = jnp.exp(disutil_work)

    marg_util_rho_not_one = eta * (consumption / cons_scale) ** (-rho) / cons_scale

    marg_util = jax.lax.select(
        jnp.allclose(rho, 1),
        eta / consumption,
        marg_util_rho_not_one,
    )

    return marg_util


def inverse_marginal_adda(
    marginal_utility,
    partner_state,
    education,
    health,
    period,
    choice,
    params,
    options,
):

    # rho = params["rho"]
    rho = params["rho_low"] * (1 - education) + params["rho_high"] * education

    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
    )
    disutil_work = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    eta = jnp.exp(disutil_work)

    consumption_rho_not_one = cons_scale * ((marginal_utility * cons_scale) / eta) ** (
        -1 / rho
    )
    consumption = jax.lax.select(
        jnp.allclose(rho, 1), eta / marginal_utility, consumption_rho_not_one
    )
    return consumption


# =====================================================================================
# Additive (dis)utility of labor
# =====================================================================================


def utility_func_additive(
    consumption: jnp.array,
    choice: int,
    period: int,
    education: int,
    health: int,
    care_demand: int,
    # mother_health: int,
    partner_state: int,
    caregiving_type: int,
    params: dict,
    options: dict,
) -> jnp.array:
    """Compute the per-period utility based on a CRRA utility function.

    Args:
        period (int): Current period.
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        mother_alive (int): Indicator for whether the mother is alive.
            0 = mother is not alive, 1 = mother is alive.
        father_alive (int): Indicator for whether the father is alive.
            0 = father is not alive, 1 = father is alive.
        mother_health (int): Health status of the mother. One of 0, 1, 2.
            0 = good health, 1 = medium health, 2 = bad health.
        father_health (int): Health status of the father. One of 0, 1, 2.
            0 = good health, 1 = medium health, 2 = bad health.
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.
        options (dict): Dictionary containing model options.

    Returns:
        utility (jnp.array): Agent's utility. Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    utility_alive = utility_func_alive_additive(
        consumption=consumption,
        # sex=sex,
        partner_state=partner_state,
        education=education,
        health=health,
        care_demand=care_demand,
        # mother_health=mother_health,
        caregiving_type=caregiving_type,
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


def utility_func_alive_additive(
    consumption,
    partner_state,
    education,
    health,
    care_demand,
    # mother_health,
    caregiving_type,
    period,
    choice,
    params,
    options,
):
    """Calculate the choice specific cobb-douglas utility, i.e. u =
    ((c*eta/consumption_scale)^(1-rho))/(1-rho) ."""
    # gather params
    # rho = params["rho"]
    rho = params["rho_low"] * (1 - education) + params["rho_high"] * education

    disutil_work = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        care_demand=care_demand,
        params=params,
        options=options,
    )
    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
    )

    # zeta = utility_of_caregiving(
    #     period,
    #     choice,
    #     education,
    #     health=health,
    #     care_demand=care_demand,
    #     params=params,
    #     options=options,
    # )

    # compute utility
    scaled_consumption = consumption / cons_scale
    utility_rho_not_one = (scaled_consumption ** (1 - rho) - 1) / (1 - rho)

    utility = jax.lax.select(
        jnp.allclose(rho, 1),
        jnp.log(consumption / cons_scale),
        utility_rho_not_one,
    )
    return utility + disutil_work  # + zeta * care_demand


def marginal_utility_func_additive_alive(
    consumption, partner_state, education, health, period, choice, params, options
):

    # rho = params["rho"]
    rho = params["rho_low"] * (1 - education) + params["rho_high"] * education

    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
    )

    marg_util_rho_not_one = (consumption / cons_scale) ** (-rho) / cons_scale
    marg_util = jax.lax.select(
        jnp.allclose(rho, 1),
        1.0 / consumption,
        marg_util_rho_not_one,
    )

    return marg_util


def inverse_marginal_additive(
    marginal_utility,
    partner_state,
    education,
    health,
    period,
    choice,
    params,
    options,
):

    # rho = params["rho"]
    rho = params["rho_low"] * (1 - education) + params["rho_high"] * education

    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
    )

    # Solve m = (c/cons_scale)^(-rho) / cons_scale
    # c = cons_scale * (m*cons_scale)^(-1/rho)
    consumption_rho_not_one = cons_scale * (marginal_utility * cons_scale) ** (-1 / rho)
    consumption = jax.lax.select(
        jnp.allclose(rho, 1), 1.0 / marginal_utility, consumption_rho_not_one
    )

    return consumption
