"""Utility functions for the model."""

import jax
import jax.numpy as jnp

from caregiving.model.shared import (
    SEX,
    is_bad_health,
    is_child_age_0_to_2,
    is_child_age_0_to_3,
    is_child_age_3_to_5,
    is_child_age_4_to_6,
    is_child_age_6_to_10,
    is_child_age_7_to_9,
    is_dead,
    is_full_time,
    is_good_health,
    is_informal_care,
    is_part_time,
    is_unemployed,
)
from caregiving.model.utility.bequest_utility import (
    marginal_utility_final_consume_all,
    utility_final_consume_all,
)

EPS = 0.1


def create_utility_functions():
    """Create dict of utility functions."""
    return {
        "utility": utility_func_additive,
        "marginal_utility": marg_utility_additive,
        "inverse_marginal_utility": inverse_marginal_additive,
    }


# =====================================================================================
# Per-period utility
# =====================================================================================


def utility_func_additive(
    consumption: jnp.array,
    choice: int,
    period: int,
    education: int,
    health: int,
    # care_demand: int,
    # care_supply: int,
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
        has_sibling (int): Indicator for whether the agent has a sibling.
            0 = no sibling, 1 = has sibling.
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.
        options (dict): Dictionary containing model options.

    Returns:
        utility (jnp.array): Agent's utility. Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    utility_alive = utility_func_alive_additive(
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


def utility_func_alive_additive(
    consumption,
    partner_state,
    education,
    health,
    period,
    choice,
    params,
    options,
):
    """Calculate the choice specific cobb-douglas utility, i.e. u =
    ((c*eta/consumption_scale)^(1-rho))/(1-rho) ."""
    # gather params
    rho = params["rho"]
    disutil_work = disutility_work(
        period=period,
        choice=choice,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    cons_scale = consumption_scale(
        partner_state=partner_state,
        education=education,
        period=period,
        options=options,
    )

    # disutil_children = disutility_of_children_and_work(
    #     period=period,
    #     choice=choice,
    #     education=education,
    #     partner_state=partner_state,
    #     health=health,
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
    return utility + disutil_work  # + disutil_children


def marg_utility_additive(
    consumption,
    partner_state,
    education,
    health,
    period,
    choice,
    params,
    options,
):
    """Calculate the choice specific marginal utility of consumption."""
    marginal_utility_alive = marginal_utility_function_alive(
        consumption=consumption,
        partner_state=partner_state,
        education=education,
        health=health,
        period=period,
        choice=choice,
        params=params,
        options=options,
    )
    marginal_utility_death = marginal_utility_final_consume_all(
        wealth=consumption,
        education=education,
        params=params,
    )
    death_bool = health == options["death_health_var"]
    marginal_utility = jax.lax.select(
        death_bool, marginal_utility_death, marginal_utility_alive
    )
    return marginal_utility


def marginal_utility_function_alive(
    consumption, partner_state, education, health, period, choice, params, options
):
    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
    )
    rho = params["rho"]

    marg_util_rho_not_one = ((1 / cons_scale) ** (1 - rho)) * (consumption ** (-rho))

    marg_util = jax.lax.select(
        jnp.allclose(rho, 1),
        1 / consumption,
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
    """Calculate the choice specific inverse marginal utility of consumption."""
    cons_scale = consumption_scale(
        partner_state=partner_state,
        education=education,
        period=period,
        options=options,
    )
    rho = params["rho"]

    consumption_rho_not_one = marginal_utility ** (-1 / rho) * (1 / cons_scale) ** (
        (1 - rho) / rho
    )

    consumption = jax.lax.select(
        jnp.allclose(rho, 1), 1 / marginal_utility, consumption_rho_not_one
    )

    return consumption


# =====================================================================================
# Disutility of labor
# =====================================================================================


def disutility_work(period, choice, education, partner_state, health, params, options):
    # choice booleans
    unemployed = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)
    # partner_retired = partner_state == 0
    bad_health = is_bad_health(health)
    good_health = is_good_health(health)

    disutil_ft_work_women = (
        params["disutil_ft_work_bad_women"] * (1 - good_health)
        + params["disutil_ft_work_good_women"] * good_health
    )
    disutil_pt_work_women = (
        params["disutil_pt_work_bad_women"] * (1 - good_health)
        + params["disutil_pt_work_good_women"] * good_health
    )

    disutil_children = params["disutil_children_ft_work_high"] * education + params[
        "disutil_children_ft_work_low"
    ] * (1 - education)

    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner, period]
    # has_children = (nb_children > 0).astype(int)

    # age_youngest_child = options["child_age_youngest_by_state"][
    #     SEX, education, has_partner, period
    # ]

    disutil_children = params["disutil_children_ft_work_high"] * education + params[
        "disutil_children_ft_work_low"
    ] * (1 - education)
    disutil_children_ft = disutil_children * _func_nb_children_in_hh(nb_children)

    disutil_unemployed_women = (
        params["disutil_unemployed_good_women"] * good_health
        + params["disutil_unemployed_bad_women"] * bad_health
    )

    disutil_sum_women = (
        disutil_unemployed_women * unemployed
        + disutil_pt_work_women * working_part_time
        + (disutil_ft_work_women + disutil_children_ft) * working_full_time
    )

    disutility = jnp.exp(-disutil_sum_women)

    return disutility


def _func_nb_children_in_hh(nb_children):
    # return jnp.sqrt(nb_children)
    return nb_children


# =====================================================================================
# Disutility of children labor
# =====================================================================================


def disutility_of_children_and_work(
    period, choice, education, partner_state, health, params, options
):

    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)

    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner, period]
    has_children = (nb_children > 0).astype(int)

    age_youngest_child = options["child_age_youngest_by_state"][
        SEX, education, has_partner, period
    ]
    # child_age_0_to_2 = is_child_age_0_to_2(age_youngest_child) * has_children
    # child_age_3_to_5 = is_child_age_3_to_5(age_youngest_child) * has_children
    # child_age_6_to_10 = is_child_age_6_to_10(age_youngest_child) * has_children

    # disutil_children_ft_low = params["disutil_children_ft_work_low"] * nb_children
    # disutil_children_ft_high = params["disutil_children_ft_work_high"] * nb_children

    # disutil_children_pt_low = params["disutil_children_pt_work_low"] * nb_children
    # disutil_children_pt_high = params["disutil_children_pt_work_high"] * nb_children

    # # Disutility of labor and age of youngest child
    # disutil_child_0_to_2_ft_low = (
    #     params["disutil_youngest_child_0_to_2_ft_work_low"] * child_age_0_to_2
    # )
    # disutil_child_3_to_5_ft_low = (
    #     params["disutil_youngest_child_3_to_5_ft_work_low"] * child_age_3_to_5
    # )
    # disutil_child_6_to_10_ft_low = (
    #     params["disutil_youngest_child_6_to_10_ft_work_low"] * child_age_6_to_10
    # )
    # disutil_child_0_to_2_pt_low = (
    #     params["disutil_youngest_child_0_to_2_pt_work_low"] * child_age_0_to_2
    # )
    # disutil_child_3_to_5_pt_low = (
    #     params["disutil_youngest_child_3_to_5_pt_work_low"] * child_age_3_to_5
    # )
    # disutil_child_6_to_10_pt_low = (
    #     params["disutil_youngest_child_6_to_10_pt_work_low"] * child_age_6_to_10
    # )

    # disutil_child_0_to_2_ft_high = (
    #     params["disutil_youngest_child_0_to_2_ft_work_high"] * child_age_0_to_2
    # )
    # disutil_child_3_to_5_ft_high = (
    #     params["disutil_youngest_child_3_to_5_ft_work_high"] * child_age_3_to_5
    # )
    # disutil_child_6_to_10_ft_high = (
    #     params["disutil_youngest_child_6_to_10_ft_work_high"] * child_age_6_to_10
    # )
    # disutil_child_0_to_2_pt_high = (
    #     params["disutil_youngest_child_0_to_2_pt_work_high"] * child_age_0_to_2
    # )
    # disutil_child_3_to_5_pt_high = (
    #     params["disutil_youngest_child_3_to_5_pt_work_high"] * child_age_3_to_5
    # )
    # disutil_child_6_to_10_pt_high = (
    #     params["disutil_youngest_child_6_to_10_pt_work_high"] * child_age_6_to_10
    # )

    # disutil_children_pt = (
    #     # disutil_children_pt_low
    #     disutil_child_0_to_2_pt_low
    #     + disutil_child_3_to_5_pt_low
    #     + disutil_child_6_to_10_pt_low
    # ) * (1 - education) + (
    #     # disutil_children_pt_high
    #     disutil_child_0_to_2_pt_high
    #     + disutil_child_3_to_5_pt_high
    #     + disutil_child_6_to_10_pt_high
    # ) * education
    # disutil_children_ft = (
    #     # disutil_children_ft_low
    #     disutil_child_0_to_2_ft_low
    #     + disutil_child_3_to_5_ft_low
    #     + disutil_child_6_to_10_ft_low
    # ) * (1 - education) + (
    #     # disutil_children_ft_high
    #     disutil_child_0_to_2_ft_high
    #     + disutil_child_3_to_5_ft_high
    #     + disutil_child_6_to_10_ft_high
    # ) * education

    disutil_age_youngest_child_pt_low = (
        # jnp.log(age_youngest_child + 1)
        _func_age_of_youngest_child(age_youngest_child, params)
        * has_children
        * params["disutil_age_youngest_child_pt_work_low"]
    )
    disutil_age_youngest_child_pt_high = (
        # jnp.log(age_youngest_child + 1)
        _func_age_of_youngest_child(age_youngest_child, params)
        * has_children
        * params["disutil_age_youngest_child_pt_work_high"]
    )
    disutil_age_youngest_child_ft_low = (
        # jnp.log(age_youngest_child + 1)
        _func_age_of_youngest_child(age_youngest_child, params)
        * has_children
        * params["disutil_age_youngest_child_ft_work_low"]
    )
    disutil_age_youngest_child_ft_high = (
        # jnp.log(age_youngest_child + 1)
        _func_age_of_youngest_child(age_youngest_child, params)
        * has_children
        * params["disutil_age_youngest_child_ft_work_high"]
    )

    disutil_children_pt = (
        disutil_age_youngest_child_pt_low * (1 - education)
        + (disutil_age_youngest_child_pt_high) * education
    )
    disutil_children_ft = (
        disutil_age_youngest_child_ft_low * (1 - education)
        + (disutil_age_youngest_child_ft_high) * education
    )

    exp_factor_women = (
        disutil_children_pt * working_part_time
        + disutil_children_ft * working_full_time
    )

    return jnp.exp(-exp_factor_women)


def _func_age_of_youngest_child(age_youngest_child, params):
    # return age_youngest_child
    # return jnp.sqrt(age_youngest_child)
    # return jnp.log(age_youngest_child + 1)
    # return 1 / jnp.sqrt(age_youngest_child + EPS)
    return jnp.log(age_youngest_child + 1)


def consumption_scale(partner_state, education, period, options):
    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
