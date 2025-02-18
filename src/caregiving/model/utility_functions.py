"""Utility functions for the model."""

import jax
import jax.numpy as jnp

from caregiving.model.shared import (
    SEX,
    is_child_age_0_to_3,
    is_child_age_4_to_6,
    is_child_age_7_to_9,
    is_full_time,
    is_part_time,
    is_unemployed,
)


def create_utility_functions():
    """Create dict of utility functions."""
    return {
        "utility": utility_func,
        "marginal_utility": marginal_utility,
        "inverse_marginal_utility": inverse_marginal_utility,
    }


# =====================================================================================
# Per-period utility
# =====================================================================================


def utility_func(
    consumption: jnp.array,
    choice: int,
    period: int,
    education: int,
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
    rho = params["rho"]

    has_partner = (partner_state > 0).astype(int)
    n_children = options["children_by_state"][SEX, education, has_partner, period]
    age_youngest_child = options["age_youngest_child_by_state"][
        SEX, education, has_partner, period
    ]

    cons_scale = consumption_scale(has_partner, n_children)
    utility_consumption = ((consumption / cons_scale) ** (1 - rho) - 1) / (1 - rho)

    eta = utility_of_labor_and_children(choice, education, n_children, params)
    zeta = utility_of_labor_and_caregiving(
        choice, age_youngest_child, education, params
    )

    utility_with_rho_not_one = utility_consumption * jnp.exp(eta) + jnp.exp(zeta)

    utility = jax.lax.select(
        jnp.allclose(rho, 1),
        jnp.log(consumption * jnp.exp(eta) / cons_scale),
        utility_with_rho_not_one,
    )

    return utility


def marginal_utility(
    consumption, choice, period, education, partner_state, params, options
):
    """Computes the marginal utility of consumption and labor.

    consumption ** (-params["rho"])
    marginal utility = (consumption^(-rho)) * (cons_scale^(rho-1)).
    """
    rho = params["rho"]

    has_partner = (partner_state > 0).astype(int)
    n_children = options["children_by_state"][SEX, education, has_partner, period]

    eta = utility_of_labor_and_children(choice, education, params)

    cons_scale = consumption_scale(has_partner, n_children)

    marg_util_with_rho_not_one = (
        consumption ** (-rho) * cons_scale ** (rho - 1)
    ) * jnp.exp(eta)

    marg_util = jax.lax.select(
        jnp.allclose(rho, 1),
        1 / consumption,
        marg_util_with_rho_not_one,
    )

    return marg_util


def inverse_marginal_utility(
    marg_util, choice, period, education, partner_state, params, options
):
    """Compute the inverse marginal utility of consumption and labor.

    marginal_utility ** (-1 / params["rho"])
    """
    rho = params["rho"]

    has_partner = (partner_state > 0).astype(int)
    n_children = options["children_by_state"][SEX, education, has_partner, period]

    eta = utility_of_labor_and_children(choice, education, params)
    cons_scale = consumption_scale(has_partner, n_children)

    inv_marg_util_with_rho_not_one = (
        marg_util ** (-1 / rho)
        * (cons_scale ** ((rho - 1) / rho))
        * (jnp.exp(eta) ** (1 / rho))
    )

    inv_marg_util = jax.lax.select(
        jnp.allclose(rho, 1),
        1 / marg_util,
        inv_marg_util_with_rho_not_one,
    )

    return inv_marg_util


# =====================================================================================
# Auxiliary
# =====================================================================================


def utility_of_labor_and_children(choice, education, n_children, params):
    """Compute utility of labor.

    Interacted with utility of consumption above.

    Reference category is 'retired'.

    """

    unemployed = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)

    util_unemployed = (
        params["util_cons_unemployed_low_educ"] * (1 - education)
        + params["util_cons_unemployed_high_educ"] * education
    )
    util_part_time = (
        params["util_cons_part_time_low_educ"] * (1 - education)
        + params["util_cons_part_time_high_educ"] * education
    )
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


def utility_of_labor_and_caregiving(choice, age_youngest_child, education, params):
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


def consumption_scale(has_partner, n_children):
    """Adjust for number of people living in household."""
    hh_size = 1 + has_partner + n_children
    return jnp.sqrt(hh_size)
