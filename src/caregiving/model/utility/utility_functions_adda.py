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


def create_utility_functions():
    """Create dict of utility functions."""
    return {
        "utility": utility_func_adda,
        "marginal_utility": marg_utility_adda,
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
    # care_demand: int,
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
    utility_alive = utility_func_alive_adda(
        consumption=consumption,
        # sex=sex,
        partner_state=partner_state,
        education=education,
        health=health,
        # care_demand=care_demand,
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

    informal_care = is_informal_care(choice)

    # t = period / PERIOD_SCALE

    disutil_ft_work_women = (
        params["disutil_ft_work_bad_women"] * bad_health
        + params["disutil_ft_work_good_women"] * good_health
        # + params["disutil_ft_work_low_women"] * (1 - education)
        # + params["disutil_ft_work_high_women"] * education
    )
    disutil_pt_work_women = (
        params["disutil_pt_work_bad_women"] * bad_health
        + params["disutil_pt_work_good_women"] * good_health
        # +params["disutil_pt_work_low_women"] * (1 - education)
        # + params["disutil_pt_work_high_women"] * education
    )
    disutil_unemployed_women = (
        params["disutil_unemployed_low_women"] * (1 - education)
        + params["disutil_unemployed_high_women"] * education
    )

    has_partner_int = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner_int, period]

    disutil_children_ft_low = params["disutil_children_ft_work_low"] * nb_children
    disutil_children_ft_high = params["disutil_children_ft_work_high"] * nb_children

    disutil_children_pt_low = params["disutil_children_pt_work_low"] * nb_children
    disutil_children_pt_high = params["disutil_children_pt_work_high"] * nb_children

    disutil_children_pt = (
        disutil_children_pt_low * (1 - education) + disutil_children_pt_high * education
    )
    disutil_children_ft = (
        disutil_children_ft_low * (1 - education) + disutil_children_ft_high * education
    )

    # disutil_pt_work_age = (
    #     params["disutil_pt_work_low_women"] * (1 - education)
    #     + params["disutil_pt_work_high_women"] * education
    #     + (
    #         params["disutil_pt_work_low_age"] * (1 - education)
    #         + params["disutil_pt_work_high_age"] * education
    #     )
    #     * period
    #     + (
    #         params["disutil_pt_work_low_age_squared"] * (1 - education)
    #         + params["disutil_pt_work_high_age_squared"] * education
    #     )
    #     * (period**2)
    # )

    # disutil_ft_work_age = (
    #     params["disutil_ft_work_low_women"] * (1 - education)
    #     + params["disutil_ft_work_high_women"] * education
    #     + params["disutil_ft_work_low_age"] * (1 - education)
    #     + params["disutil_ft_work_high_age"] * education
    # ) * period + (
    #     params["disutil_ft_work_low_age_squared"] * (1 - education)
    #     + params["disutil_ft_work_high_age_squared"] * education
    # ) * (
    #     period**2
    # )

    # pt_int = (
    #     params["disutil_pt_work_low_women"] * (1 - education)
    #     + params["disutil_pt_work_high_women"] * education
    # )
    # pt_lin = (
    #     params["disutil_pt_work_low_age"] * (1 - education)
    #     + params["disutil_pt_work_high_age"] * education
    # )
    # pt_quad = (
    #     params["disutil_pt_work_low_age_squared"] * (1 - education)
    #     + params["disutil_pt_work_high_age_squared"] * education
    # )

    # ft_int = (
    #     params["disutil_ft_work_low_women"] * (1 - education)
    #     + params["disutil_ft_work_high_women"] * education
    # )
    # ft_lin = (
    #     params["disutil_ft_work_low_age"] * (1 - education)
    #     + params["disutil_ft_work_high_age"] * education
    # )
    # ft_quad = (
    #     params["disutil_ft_work_low_age_squared"] * (1 - education)
    #     + params["disutil_ft_work_high_age_squared"] * education
    # )

    # disutil_pt_work = pt_int + pt_lin * t + pt_quad * t**2
    # disutil_ft_work = ft_int + ft_lin * t + ft_quad * t**2

    exp_factor_work = (
        disutil_unemployed_women * unemployed
        + (disutil_pt_work_women + disutil_children_pt) * working_part_time
        + (disutil_ft_work_women + disutil_children_ft) * working_full_time
    )

    # =================================================================================
    # Informal caregiving work interaction

    # util_unemployed_and_informal_care = params["util_unemployed_and_informal_care"]
    # util_pt_work_and_informal_care = params["util_pt_work_and_informal_care"]
    # util_ft_work_and_informal_care = params["util_ft_work_and_informal_care"]
    # util_pt_work_and_informal_care_good = params[
    # "util_pt_work_and_informal_care_good"]
    # util_pt_work_and_informal_care_bad = params["util_pt_work_and_informal_care_bad"]
    # util_ft_work_and_informal_care_good = params["
    # util_ft_work_and_informal_care_good"]
    # util_ft_work_and_informal_care_bad = params["util_ft_work_and_informal_care_bad"]

    # util_informal_care_and_work = (
    #     util_unemployed_and_informal_care * unemployed
    #     + util_pt_work_and_informal_care * working_part_time
    #     + util_ft_work_and_informal_care * working_full_time
    # )

    # exp_factor_work_and_care = util_informal_care_and_work * informal_care

    disutil_unemployed_and_informal_care = (
        params["disutil_unemployed_and_informal_care_low"] * (1 - education)
        + params["disutil_unemployed_and_informal_care_high"] * education
    )
    # util_pt_work_and_informal_care = (
    #     params["util_pt_work_and_informal_care_low"] * (1 - education)
    #     + params["util_pt_work_and_informal_care_high"] * education
    # )
    # util_ft_work_and_informal_care = (
    #     params["util_ft_work_and_informal_care_low"] * (1 - education)
    #     + params["util_ft_work_and_informal_care_high"] * education
    # )
    # util_unemployed_and_informal_care = (
    #     params["util_unemployed_and_informal_care_bad"] * bad_health
    #     + params["util_unemployed_and_informal_care_good"] * good_health
    # )
    disutil_pt_work_and_informal_care = (
        params["disutil_pt_work_and_informal_care_bad"] * bad_health
        + params["disutil_pt_work_and_informal_care_good"] * good_health
    )
    disutil_ft_work_and_informal_care = (
        params["disutil_ft_work_and_informal_care_bad"] * bad_health
        + params["disutil_ft_work_and_informal_care_good"] * good_health
    )

    disutil_children_ft_informal_care_low = (
        params["disutil_children_ft_work_informal_care_low"] * nb_children
    )
    disutil_children_ft_informal_care_high = (
        params["disutil_children_ft_work_informal_care_high"] * nb_children
    )

    disutil_children_pt_informal_care_low = (
        params["disutil_children_pt_work_informal_care_low"] * nb_children
    )
    disutil_children_pt_informal_care_high = (
        params["disutil_children_pt_work_informal_care_high"] * nb_children
    )

    disutil_children_pt_informal_care = (
        disutil_children_pt_informal_care_low * (1 - education)
        + disutil_children_pt_informal_care_high * education
    )
    disutil_children_ft_informal_care = (
        disutil_children_ft_informal_care_low * (1 - education)
        + disutil_children_ft_informal_care_high * education
    )

    exp_factor_work_and_care = (
        disutil_unemployed_and_informal_care * unemployed
        + (disutil_pt_work_and_informal_care + disutil_children_pt_informal_care)
        * working_part_time
        + (disutil_ft_work_and_informal_care + disutil_children_ft_informal_care)
        * working_full_time
    )

    # =================================================================================

    # Compute eta
    disutility = (
        -exp_factor_work * (1 - informal_care)
        - exp_factor_work_and_care * informal_care
    )

    return jnp.exp(disutility)


def utility_of_labor_and_caregiving(
    period, choice, education, health, care_demand, params, options
):

    # choice booleans
    unemployed = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)
    # partner_retired = partner_state == 0

    # bad_health = is_bad_health(health)
    # good_health = is_good_health(health)

    informal_care = is_informal_care(choice)

    util_unemployed_and_informal_care = (
        params["util_unemployed_and_informal_care_low"] * (1 - education)
        + params["util_unemployed_and_informal_care_high"] * education
    )
    util_pt_work_and_informal_care = params["util_pt_work_and_informal_care"]
    util_ft_work_and_informal_care = params["util_ft_work_and_informal_care"]

    util_informal_care_and_work = (
        util_unemployed_and_informal_care * unemployed
        + util_pt_work_and_informal_care * working_part_time
        + util_ft_work_and_informal_care * working_full_time
    )

    exp_factor_work_and_care = (
        util_informal_care_and_work * informal_care  # * (care_demand >= 1)
    )

    return exp_factor_work_and_care


def utility_of_caregiving(
    period, choice, education, health, care_demand, mother_health, params, options
):
    # choice booleans
    # unemployed = is_unemployed(choice)
    # working_part_time = is_part_time(choice)
    # working_full_time = is_full_time(choice)

    informal_care = is_informal_care(choice)

    # light_informal = is_light_informal_care(choice)
    # intensive_informal = is_intensive_informal_care(choice)
    # nursing_home = is_nursing_home_care(choice)

    # formal home care and nursing home
    formal_care = is_no_care(choice) & (care_demand == CARE_DEMAND_AND_NO_OTHER_SUPPLY)

    # bad_health = is_bad_health(health)
    # good_health = is_good_health(health)

    # mother_good_health = mother_health == PARENT_GOOD_HEALTH
    # mother_medium_health = mother_health == PARENT_MEDIUM_HEALTH
    # mother_bad_health = mother_health == PARENT_BAD_HEALTH

    # util_unemployed_and_informal_care = (
    #     params["util_unemployed_and_informal_care_good"] * good_health
    #     + params["util_unemployed_and_informal_care_bad"] * bad_health
    # )
    # util_pt_work_and_informal_care = (
    #     params["util_pt_work_and_informal_care_good"] * good_health
    #     + params["util_pt_work_and_informal_care_bad"] * bad_health
    # )
    # util_ft_work_and_informal_care = (
    #     params["util_ft_work_and_informal_care_good"] * good_health
    #     + params["util_ft_work_and_informal_care_bad"] * bad_health
    # )
    # util_no_informal_care = params["util_formal_care"]

    # disutil_unemployed_and_care = (
    # params["disutil_unemployed_informal_care_low"] * (1 - education) * informal_care
    # + params["disutil_unemployed_informal_care_high"] * education * informal_care
    # )

    # disutil_ft_work_informal_care_by_health = (
    # params["disutil_ft_work_light_informal_care_bad"] * bad_health * light_informal
    #     + params["disutil_ft_work_light_informal_care_good"]
    #     * good_health
    #     * light_informal
    #     + params["disutil_ft_work_intensive_informal_care_bad"]
    #     * bad_health
    #     * intensive_informal
    #     + params["disutil_ft_work_intensive_informal_care_good"]
    #     * good_health
    #     * intensive_informal
    # )

    # disutil_pt_work_informal_care_by_health = (
    # params["disutil_pt_work_light_informal_care_bad"] * bad_health * light_informal
    #     + params["disutil_pt_work_light_informal_care_good"]
    #     * good_health
    #     * light_informal
    #     + params["disutil_pt_work_intensive_informal_care_bad"]
    #     * bad_health
    #     * intensive_informal
    #     + params["disutil_pt_work_intensive_informal_care_good"]
    #     * good_health
    #     * intensive_informal
    # )

    # util_informal_by_education = (
    #     params["util_light_informal_care_high"] * education * light_informal
    #     + params["util_intensive_informal_care_high"] * education * intensive_informal
    #     + params["util_light_informal_care_low"] * (1 - education) * light_informal
    #     + params["util_intensive_informal_care_low"]
    #     * (1 - education)
    #     * intensive_informal
    # )

    util_informal_care_by_education = params[
        "util_informal_care_high"
    ] * education + params["util_informal_care_low"] * (1 - education)
    # util_informal_care_by_mother_health = (
    #     params["util_informal_care_mother_good"] * mother_good_health
    #     + params["util_informal_care_mother_medium"] * mother_medium_health
    #     + params["util_informal_care_mother_bad"] * mother_bad_health
    # )
    # util_solo_informal_care_by_health = (
    #     params["util_solo_informal_care_bad"] * bad_health
    #     + params["util_solo_informal_care_good"] * good_health
    # )
    # util_solo_informal_care_by_mother_health = (
    #     params["util_solo_informal_care_mother_good"] * mother_good_health
    #     + params["util_solo_informal_care_mother_medium"] * mother_medium_health
    #     + params["util_solo_informal_care_mother_bad"] * mother_bad_health
    # )
    # util_informal_care_by_period = (
    #     params["util_informal_care_period"] * period
    #     + params["util_informal_care_period_squared"] * period**2
    # )

    # util_joint_care = (
    #     params["util_joint_informal_care_low"] * (1 - education)
    #     + params["util_joint_informal_care_high"] * education
    #     # params["util_joint_informal_care_bad"] * bad_health
    #     # + params["util_joint_informal_care_good"] * good_health
    # )

    # util_joint_care_by_health = (
    #     params["util_joint_informal_care_bad"] * bad_health
    #     + params["util_joint_informal_care_good"] * good_health
    # )
    # util_joint_care_by_mother_health = (
    #     params["util_joint_informal_care_mother_good"] * mother_good_health
    #     + params["util_joint_informal_care_mother_medium"] * mother_medium_health
    #     + params["util_joint_informal_care_mother_bad"] * mother_bad_health
    # )
    # util_joint_care_by_period = (
    #     params["util_joint_informal_care_period"] * period
    #     + params["util_joint_informal_care_period_squared"] * period**2
    #     + params["util_joint_informal_care_period_cubic"] * period**3
    # )

    # util_ft_work_informal_care_by_health = (
    #     params["util_ft_work_informal_care_bad"] * bad_health * informal_care
    #     + params["util_ft_work_informal_care_good"] * good_health * informal_care
    # )
    # util_pt_work_informal_care_by_health = (
    #     params["util_pt_work_informal_care_bad"] * bad_health * informal_care
    #     + params["util_pt_work_informal_care_good"] * good_health * informal_care
    # )

    # util_informal_care_and_work = (
    #     util_unemployed_and_informal_care * unemployed
    #     + (util_pt_work_and_informal_care_good + util_pt_work_and_informal_care_bad)
    #     * working_part_time
    #     + (util_ft_work_and_informal_care_good + util_ft_work_and_informal_care_bad)
    #     * working_full_time
    # )

    util_informal = (
        util_informal_care_by_education
        # + util_informal_care_and_work
        # + util_solo_informal_care_by_health)
        # + util_informal_and_work_by_health
        + params["util_joint_informal_care"]
        * (care_demand == CARE_DEMAND_AND_OTHER_SUPPLY)
    ) * informal_care

    # util_nursing_home = (
    #     params["util_nursing_home_low"] * (1 - education) * nursing_home
    #     + params["util_nursing_home_high"] * education * nursing_home
    # )
    # util_formal_home_care = (
    #     params["util_home_care_low"] * (1 - education) * formal_home_care
    #     + params["util_home_care_high"] * education * formal_home_care
    # )
    # util_nursing_home = (
    #     params["util_nursing_home_bad"] * bad_health * nursing_home
    #     + params["util_nursing_home_good"] * good_health * nursing_home
    # )
    # util_formal_care = (
    #     params["util_formal_care_bad"] * bad_health * formal_care
    #     + params["util_formal_care_good"] * good_health * formal_care
    # )
    util_formal_care = params["util_formal_care"] * formal_care

    util_relative_to_only_other_family_provide_care = util_informal + util_formal_care

    return util_relative_to_only_other_family_provide_care  # * (care_demand >= 1)


def consumption_scale(partner_state, education, period, options):
    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
