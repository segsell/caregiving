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
    is_child_age_7_to_9,
    is_dead,
    is_full_time,
    is_good_health,
    is_informal_care,
    is_part_time,
    is_unemployed,
)
from caregiving.model.utility.bequest_utility import utility_final_consume_all


def create_utility_functions():
    """Create dict of utility functions."""
    return {
        "utility": utility_func,
        "marginal_utility": marg_utility,
        "inverse_marginal_utility": inverse_marginal,
    }


# =====================================================================================
# Per-period utility
# =====================================================================================


def utility_func(
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
    utility_alive = utility_func_alive(
        consumption=consumption,
        # sex=sex,
        partner_state=partner_state,
        education=education,
        health=health,
        # care_demand=care_demand,
        # care_supply=care_supply,
        period=period,
        choice=choice,
        params=params,
        options=options,
    )
    utility_death = utility_final_consume_all(
        wealth=consumption,
        params=params,
    )
    death_bool = is_dead(health)
    utility = jax.lax.select(death_bool, utility_death, utility_alive)

    return utility


def utility_func_alive(
    consumption,
    partner_state,
    education,
    health,
    # care_demand,
    # care_supply,
    period,
    choice,
    params,
    options,
):
    """Calculate the choice specific cobb-douglas utility, i.e. u =
    ((c*eta/consumption_scale)^(1-rho))/(1-rho) ."""
    # gather params
    rho = params["rho"]
    eta = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
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
    #     # care_supply=care_supply,
    #     params=params,
    #     options=options,
    # )

    # compute utility
    scaled_consumption = consumption * eta / cons_scale
    utility_rho_not_one = (scaled_consumption ** (1 - rho) - 1) / (1 - rho)

    utility = jax.lax.select(
        jnp.allclose(rho, 1),
        jnp.log(consumption * eta / cons_scale),
        utility_rho_not_one,
    )
    return utility  # + zeta * care_demand


# def _utility_func_alive(
#     params, consumption, choice, education, partner_state, health, period, options
# ):
#     rho = params["rho"]

#     has_partner = (partner_state > 0).astype(int)
#     n_children = options["children_by_state"][SEX, education, has_partner, period]
#     # age_youngest_child = options["age_youngest_child_by_state"][
#     #     SEX, education, has_partner, period
#     # ]

#     cons_scale = consumption_scale(has_partner, n_children)
#     utility_consumption = ((consumption / cons_scale) ** (1 - rho) - 1) / (1 - rho)

#     eta = _utility_of_labor_and_children(
#         choice=choice, education=education, n_children=n_children, params=params
#     )
#     # zeta = utility_of_labor_and_elder_care(
#     #     choice, params
#     # )

#     utility_with_rho_not_one = utility_consumption * jnp.exp(eta)  # + jnp.exp(zeta)

#     utility = jax.lax.select(
#         jnp.allclose(rho, 1),
#         jnp.log(consumption * jnp.exp(eta) / cons_scale),
#         utility_with_rho_not_one,
#     )

#     return utility


def utility_func_adda(
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
    utility_alive = utility_func_alive(
        consumption=consumption,
        # sex=sex,
        partner_state=partner_state,
        education=education,
        health=health,
        # care_demand=care_demand,
        # care_supply=care_supply,
        period=period,
        choice=choice,
        params=params,
        options=options,
    )
    utility_death = utility_final_consume_all(
        wealth=consumption,
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
    # care_demand,
    # care_supply,
    period,
    choice,
    params,
    options,
):
    """Calculate the choice specific cobb-douglas utility, i.e. u =
    ((c*eta/consumption_scale)^(1-rho))/(1-rho) ."""
    # gather params
    rho = params["rho"]
    eta = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
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
    #     # care_supply=care_supply,
    #     params=params,
    #     options=options,
    # )

    # compute utility
    scaled_consumption = consumption / cons_scale
    utility_rho_not_one = (scaled_consumption ** (1 - rho) - 1) / (1 - rho)

    utility = jax.lax.select(
        jnp.allclose(rho, 1),
        jnp.log(consumption * eta / cons_scale),
        utility_rho_not_one,
    )
    return utility * eta  # + zeta * care_demand


def marg_utility(
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
    eta = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    marg_util_rho_not_one = ((eta / cons_scale) ** (1 - rho)) * (consumption ** (-rho))

    marg_util = jax.lax.select(
        jnp.allclose(rho, 1),
        1 / consumption,
        marg_util_rho_not_one,
    )

    return marg_util


def marg_utility_adda(
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
    eta = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    marg_util_rho_not_one = eta * (consumption / cons_scale) ** (-rho) / cons_scale

    marg_util = jax.lax.select(
        jnp.allclose(rho, 1),
        1 / consumption,
        marg_util_rho_not_one,
    )

    return marg_util


def inverse_marginal(
    marginal_utility,
    partner_state,
    education,
    health,
    period,
    choice,
    params,
    options,
):
    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
    )
    rho = params["rho"]
    eta = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    consumption_rho_not_one = marginal_utility ** (-1 / rho) * (eta / cons_scale) ** (
        (1 - rho) / rho
    )
    consumption = jax.lax.select(
        jnp.allclose(rho, 1), 1 / marginal_utility, consumption_rho_not_one
    )
    return consumption


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
    cons_scale = consumption_scale(
        partner_state=partner_state,
        # sex=sex,
        education=education,
        period=period,
        options=options,
    )
    rho = params["rho"]
    eta = disutility_work(
        period=period,
        choice=choice,
        # sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    consumption_rho_not_one = cons_scale * ((marginal_utility * cons_scale) / eta) ** (
        -1 / rho
    )
    consumption = jax.lax.select(
        jnp.allclose(rho, 1), 1 / marginal_utility, consumption_rho_not_one
    )
    return consumption


# def marginal_utility(
#     consumption, choice, period, education, partner_state, params, options
# ):
#     """Computes the marginal utility of consumption and labor.

#     consumption ** (-params["rho"])
#     marginal utility = (consumption^(-rho)) * (cons_scale^(rho-1)).
#     """
#     rho = params["rho"]

#     has_partner = (partner_state > 0).astype(int)
#     n_children = options["children_by_state"][SEX, education, has_partner, period]

#     eta = utility_of_labor_and_children(
#         choice=choice, education=education, n_children=n_children, params=params
#     )

#     cons_scale = consumption_scale(has_partner, n_children)

#     marg_util_with_rho_not_one = (
#         consumption ** (-rho) * cons_scale ** (rho - 1)
#     ) * jnp.exp(eta)

#     marg_util = jax.lax.select(
#         jnp.allclose(rho, 1),
#         1 / consumption,
#         marg_util_with_rho_not_one,
#     )

#     return marg_util


# def inverse_marginal_utility(
#     marginal_utility, choice, period, education, partner_state, params, options
# ):
#     """Compute the inverse marginal utility of consumption and labor.

#     marginal_utility ** (-1 / params["rho"])
#     """
#     rho = params["rho"]

#     has_partner = (partner_state > 0).astype(int)
#     n_children = options["children_by_state"][SEX, education, has_partner, period]

#     eta = utility_of_labor_and_children(
#         choice=choice, education=education, n_children=n_children, params=params
#     )
#     cons_scale = consumption_scale(has_partner, n_children)

#     inv_marg_util_with_rho_not_one = (
#         marginal_utility ** (-1 / rho)
#         * (cons_scale ** ((rho - 1) / rho))
#         * (jnp.exp(eta) ** (1 / rho))
#     )

#     inv_marg_util = jax.lax.select(
#         jnp.allclose(rho, 1),
#         1 / marginal_utility,
#         inv_marg_util_with_rho_not_one,
#     )

#     return inv_marg_util


# =====================================================================================
# Auxiliary
# =====================================================================================


def disutility_work(period, choice, education, partner_state, health, params, options):
    # choice booleans
    unemployed = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)
    # partner_retired = partner_state == 0
    bad_health = is_bad_health(health)
    good_health = is_good_health(health)

    # # reading parameters
    # disutil_ft_work_men = (
    #     params["disutil_ft_work_bad_men"] * (1 - health)
    #     + params["disutil_ft_work_good_men"] * health
    # )
    # exp_factor_men = (
    #     params["disutil_unemployed_men"] * is_unemployed
    #     # + disutil_pt_work * is_working_part_time
    #     + disutil_ft_work_men * is_working_full_time
    #     # + partner_retired * disutil_only_partner_retired
    # )

    disutil_ft_work_women = (
        params["disutil_ft_work_bad_women"] * bad_health
        + params["disutil_ft_work_good_women"] * good_health
        # + params["disutil_ft_work_low_women"] * (1 - education)
        # + params["disutil_ft_work_high_women"] * education
    )
    disutil_pt_work_women = (
        params["disutil_pt_work_bad_women"] * bad_health
        + params["disutil_pt_work_good_women"] * good_health
        # + params["disutil_pt_work_low_women"] * (1 - education)
        # + params["disutil_pt_work_high_women"] * education
    )
    disutil_unemployed_women = (
        params["disutil_unemployed_low_women"] * (1 - education)
        + params["disutil_unemployed_high_women"] * education
    )

    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner, period]
    has_children = (nb_children > 0).astype(int)

    age_youngest_child = options["child_age_youngest_by_state"][
        SEX, education, has_partner, period
    ]
    child_age_0_to_2 = is_child_age_0_to_2(age_youngest_child) * has_children
    child_age_3_to_5 = is_child_age_3_to_5(age_youngest_child) * has_children

    disutil_children_ft_low = params["disutil_children_ft_work_low"] * nb_children
    disutil_children_ft_high = params["disutil_children_ft_work_high"] * nb_children

    disutil_children_pt_low = params["disutil_children_pt_work_low"] * nb_children
    disutil_children_pt_high = params["disutil_children_pt_work_high"] * nb_children

    # Disutility of labor and age of youngest child
    disutil_child_0_to_2_ft_low = (
        params["disutil_youngest_child_0_to_2_ft_work_low"] * child_age_0_to_2
    )
    disutil_child_3_to_5_ft_low = (
        params["disutil_youngest_child_3_to_5_ft_work_low"] * child_age_3_to_5
    )
    disutil_child_0_to_2_pt_low = (
        params["disutil_youngest_child_0_to_2_pt_work_low"] * child_age_0_to_2
    )
    disutil_child_3_to_5_pt_low = (
        params["disutil_youngest_child_3_to_5_pt_work_low"] * child_age_3_to_5
    )

    disutil_child_0_to_2_ft_high = (
        params["disutil_youngest_child_0_to_2_ft_work_high"] * child_age_0_to_2
    )
    disutil_child_3_to_5_ft_high = (
        params["disutil_youngest_child_3_to_5_ft_work_high"] * child_age_3_to_5
    )
    disutil_child_0_to_2_pt_high = (
        params["disutil_youngest_child_0_to_2_pt_work_high"] * child_age_0_to_2
    )
    disutil_child_3_to_5_pt_high = (
        params["disutil_youngest_child_3_to_5_pt_work_high"] * child_age_3_to_5
    )

    disutil_children_pt = (
        disutil_children_pt_low
        + disutil_child_0_to_2_pt_low
        + disutil_child_3_to_5_pt_low
    ) * (1 - education) + (
        disutil_children_pt_high
        + disutil_child_0_to_2_pt_high
        + disutil_child_3_to_5_pt_high
    ) * education
    disutil_children_ft = (
        disutil_children_ft_low
        + disutil_child_0_to_2_ft_low
        + disutil_child_3_to_5_ft_low
    ) * (1 - education) + (
        disutil_children_ft_high
        + disutil_child_0_to_2_ft_high
        + disutil_child_3_to_5_ft_high
    ) * education

    exp_factor_women = (
        disutil_unemployed_women * unemployed
        + (disutil_pt_work_women + disutil_children_pt) * working_part_time
        + (disutil_ft_work_women + disutil_children_ft) * working_full_time
    )

    # Compute eta
    disutility = jnp.exp(-exp_factor_women)

    return disutility


def utility_of_caregiving(
    period, choice, education, health, care_demand, params, options
):
    # choice booleans
    unemployed = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)
    informal_care = is_informal_care(choice)

    # bad_health = is_bad_health(health)
    # good_health = is_good_health(health)

    util_unemployed_and_care_women = params["util_unemployed_and_care_women"]
    util_pt_work_and_care_women = params["util_pt_work_and_care_women"]
    util_ft_work_and_care_women = params["util_ft_work_and_care_women"]
    util_no_informal_care = params["util_formal_care_women"]

    # util_joint_care = params["util_joint_informal_care_women"]  # * care_supply

    # util_informal_by_health = (
    #     params["util_informal_care_bad_women"] * bad_health
    #     + params["util_informal_care_good_women"] * good_health
    # )
    util_informal_by_education = params[
        "util_informal_care_high_women"
    ] * education + params["util_informal_care_low_women"] * (1 - education)
    util_informal_and_work = (
        util_unemployed_and_care_women * unemployed
        + util_pt_work_and_care_women * working_part_time
        + util_ft_work_and_care_women * working_full_time
    )

    util_informal = (
        util_informal_by_education + util_informal_and_work
    ) * informal_care
    _util_formal = util_no_informal_care * (1 - informal_care)

    utility = util_informal  # + _util_formal
    # + util_no_informal_care_educ * (1 - informal_care)

    return utility


def _utility_of_labor_and_children(choice, education, n_children, params):
    """Compute utility of labor.

    Interacted with utility of consumption above.

    Reference category is 'retired'.

    """

    unemployed = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)

    # util_unemployed = (
    #     params["util_cons_unemployed_low_educ"] * (1 - education)
    #     + params["util_cons_unemployed_high_educ"] * education
    # )
    # util_part_time = params["util_cons_part_time"]
    # util_full_time = (
    #     params["util_cons_full_time"]
    #     + params["util_cons_full_time_children"] * n_children
    # ) * education

    util_unemployed = (
        params["util_cons_unemployed_low_educ"] * (1 - education)
        + params["util_cons_unemployed_high_educ"] * education
    )
    util_part_time = (
        params["util_cons_part_time_low_educ"]
        + params["util_cons_children_part_time_low_educ"] * n_children
    ) * (1 - education) + (
        params["util_cons_part_time_high_educ"]
        + params["util_cons_children_part_time_high_educ"] * n_children
    ) * education
    util_full_time = (
        params["util_cons_full_time_low_educ"]
        + params["util_cons_children_full_time_low_educ"] * n_children
    ) * (1 - education) + (
        params["util_cons_full_time_high_educ"]
        + params["util_cons_children_full_time_high_educ"] * n_children
    ) * education

    return (
        util_unemployed * unemployed
        + util_part_time * working_part_time
        + util_full_time * working_full_time
    )


def _utility_of_labor_and_caregiving(choice, age_youngest_child, education, params):
    """Compute utility of labor and caregiving.

    Reference category is 'not working'.
    """
    # util_unemployed_low_educ = (
    #     params["util_unemployed_low_educ_constant"]
    #     + params["util_unemployed_low_educ_child_bin_one"]
    #     * is_child_age_0_to_3(age_youngest_child)
    #     + params["util_unemployed_low_educ_child_bin_two"]
    #     * is_child_age_4_to_6(age_youngest_child)
    #     + params["util_unemployed_low_educ_child_bin_three"]
    #     * is_child_age_7_to_9(age_youngest_child)
    # )
    util_part_time_low_educ = (
        params["util_part_time_low_educ_constant"]
        + params["util_part_time_low_educ_child_bin_one"]
        * is_child_age_0_to_3(age_youngest_child)
        + params["util_part_time_low_educ_child_bin_two"]
        * is_child_age_4_to_6(age_youngest_child)
        + params["util_part_time_low_educ_child_bin_three"]
        * is_child_age_7_to_9(age_youngest_child)
    )
    util_full_time_low_educ = (
        params["util_full_time_low_educ_constant"]
        + params["util_full_time_low_educ_child_bin_one"]
        * is_child_age_0_to_3(age_youngest_child)
        + params["util_full_time_low_educ_child_bin_two"]
        * is_child_age_4_to_6(age_youngest_child)
        + params["util_full_time_low_educ_child_bin_three"]
        * is_child_age_7_to_9(age_youngest_child)
    )

    # util_unemployed_high_educ = (
    #     params["util_unemployed_high_educ_constant"]
    #     + params["util_unemployed_high_educ_child_bin_one"]
    #     * is_child_age_0_to_3(age_youngest_child)
    #     + params["util_unemployed_high_educ_child_bin_two"]
    #     * is_child_age_4_to_6(age_youngest_child)
    #     + params["util_unemployed_high_educ_child_bin_three"]
    #     * is_child_age_7_to_9(age_youngest_child)
    # )
    util_part_time_high_educ = (
        params["util_part_time_high_educ_constant"]
        + params["util_part_time_high_educ_child_bin_one"]
        * is_child_age_0_to_3(age_youngest_child)
        + params["util_part_time_high_educ_child_bin_two"]
        * is_child_age_4_to_6(age_youngest_child)
        + params["util_part_time_high_educ_child_bin_three"]
        * is_child_age_7_to_9(age_youngest_child)
    )
    util_full_time_high_educ = (
        params["util_full_time_high_educ_constant"]
        + params["util_full_time_high_educ_child_bin_one"]
        * is_child_age_0_to_3(age_youngest_child)
        + params["util_full_time_high_educ_child_bin_two"]
        * is_child_age_4_to_6(age_youngest_child)
        + params["util_full_time_high_educ_child_bin_three"]
        * is_child_age_7_to_9(age_youngest_child)
    )

    # not_working = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)

    # util_not_working = (
    #     util_unemployed_low_educ * (1 - education)
    #     + util_unemployed_high_educ * education
    # )
    util_part_time = (
        util_part_time_low_educ * (1 - education) + util_part_time_high_educ * education
    )
    util_full_time = (
        util_full_time_low_educ * (1 - education) + util_full_time_high_educ * education
    )

    return (
        # util_not_working * not_working +
        util_part_time * working_part_time
        + util_full_time * working_full_time
    )


# def consumption_scale(has_partner, n_children):
#     """Adjust for number of people living in household."""
#     hh_size = 1 + has_partner + n_children
#     return jnp.sqrt(hh_size)


def consumption_scale(partner_state, education, period, options):
    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
