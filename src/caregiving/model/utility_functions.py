"""Utility functions for the model."""

from typing import Any

import jax.numpy as jnp

from caregiving.model.shared import (
    is_child_age_0_to_3,
    is_child_age_4_to_6,
    is_child_age_7_to_9,
    is_not_working,
    is_part_time,
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
    has_partner: int,
    n_children: int,
    age_youngest_child: int,
    education: int,
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

    eta = utility_of_labor(choice, education, params)

    cons_scale = consumption_scale(has_partner, n_children)
    utility_consumption = ((consumption / cons_scale) ** (1 - rho) - 1) / (1 - rho)

    zeta = utility_of_labor_and_caregiving(
        choice, age_youngest_child, education, params
    )

    return utility_consumption * jnp.exp(eta) + jnp.exp(zeta)


def marginal_utility(consumption, has_partner, n_children, choice, education, params):
    """Computes the marginal utility of consumption and labor.

    consumption ** (-params["rho"])
    """
    rho = params["rho"]

    eta = utility_of_labor(choice, education, params)

    cons_scale = consumption_scale(has_partner, n_children)

    # mu = e^eta * (1/cons_scale) * ( (c/cons_scale)^(-rho) )
    # return jnp.exp(eta) * (consumption / cons_scale) ** (-rho) / cons_scale
    # marginal utility = (consumption^(-rho)) * (cons_scale^(rho-1)).
    return jnp.exp(eta) * consumption ** (-rho) * cons_scale ** (rho - 1)


def inverse_marginal_utility(
    marg_util, has_partner, n_children, choice, education, params
):
    """Compute the inverse marginal utility of consumption and labor.

    marginal_utility ** (-1 / params["rho"])
    """
    rho = params["rho"]

    eta = utility_of_labor(choice, education, params)
    cons_scale = consumption_scale(has_partner, n_children)

    # c = ( [exp(eta) * cons_scale^(rho - 1)] / m )^(1/rho)
    # numerator = jnp.exp(eta) * cons_scale ** (rho - 1)
    # return (numerator / m) ** (1.0 / rho)

    # return (cons_scale ** ((rho - 1) / rho)) * (marg_util ** (-1 / rho))
    return (
        marg_util ** (-1 / rho)
        * (cons_scale ** ((rho - 1) / rho))
        * (jnp.exp(eta) ** (1 / rho))
    )


# =====================================================================================
# Auxiliary
# =====================================================================================


def utility_of_labor(choice, education, params):
    """Reference category is full-time work."""
    util_part_time = (
        params["dis_util_pt_work_low"] * (1 - education)
        + params["dis_util_pt_work_high"] * education
    )

    util_not_working = (
        params["dis_util_unemployed_low"] * (1 - education)
        + params["dis_util_unemployed_high"] * education
    )

    not_working = is_not_working(choice)
    working_part_time = is_part_time(choice)

    exp_factor = util_not_working * not_working + util_part_time * working_part_time

    return exp_factor


def utility_of_labor_and_caregiving(choice, age_youngest_child, education, params):
    """Compute utility of labor and caregiving."""
    # Define utility for unemployed based on child age and education
    util_unemployed_low_educ = (
        params["util_unemployed_low_educ_constant"]
        + params["util_unemployed_low_educ_child_bin_one"]
        * is_child_age_0_to_3(age_youngest_child)
        + params["util_unemployed_low_educ_child_bin_two"]
        * is_child_age_4_to_6(age_youngest_child)
        + params["util_unemployed_low_educ_child_bin_three"]
        * is_child_age_7_to_9(age_youngest_child)
    )
    util_part_time_low_educ = (
        params["util_part_time_low_educ_constant"]
        + params["util_part_time_low_educ_child_bin_one"]
        * is_child_age_0_to_3(age_youngest_child)
        + params["util_part_time_low_educ_child_bin_two"]
        * is_child_age_4_to_6(age_youngest_child)
        + params["util_part_time_low_educ_child_bin_three"]
        * is_child_age_7_to_9(age_youngest_child)
    )

    util_unemployed_high_educ = (
        params["util_unemployed_high_educ_constant"]
        + params["util_unemployed_high_educ_child_bin_one"]
        * is_child_age_0_to_3(age_youngest_child)
        + params["util_unemployed_high_educ_child_bin_two"]
        * is_child_age_4_to_6(age_youngest_child)
        + params["util_unemployed_high_educ_child_bin_three"]
        * is_child_age_7_to_9(age_youngest_child)
    )
    util_part_time_high_educ = (
        params["util_part_time_high_educ_constant"]
        + params["util_part_time_high_educ_child_bin_one"]
        * is_child_age_0_to_3(age_youngest_child)
        + params["util_part_time_high_educ_child_bin_two"]
        * is_child_age_4_to_6(age_youngest_child)
        + params["util_part_time_high_educ_child_bin_three"]
        * is_child_age_7_to_9(age_youngest_child)
    )

    not_working = is_not_working(choice)
    working_part_time = is_part_time(choice)

    util_not_working = (
        util_unemployed_low_educ * (1 - education)
        + util_unemployed_high_educ * education
    )
    util_part_time = (
        util_part_time_low_educ * (1 - education) + util_part_time_high_educ * education
    )

    return util_not_working * not_working + util_part_time * working_part_time


def consumption_scale(has_partner, n_children):
    """Compute the consumption equivalence scale."""
    hh_size = 1 + has_partner + n_children
    return jnp.sqrt(hh_size)
