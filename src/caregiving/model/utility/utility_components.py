"""Utility functions for the model."""

import jax
import jax.numpy as jnp

from caregiving.model.shared import (  # is_nursing_home_care,
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    CARE_DEMAND_AND_OTHER_SUPPLY,
    DEAD,
    PARENT_BAD_HEALTH,
    PARENT_GOOD_HEALTH,
    PARENT_MEDIUM_HEALTH,
    PARTNER_RETIRED,
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
    is_retired,
    is_unemployed,
)
from caregiving.model.utility.bequest_utility import (
    marginal_utility_final_consume_all,
    utility_final_consume_all,
)


def disutility_work(period, choice, education, partner_state, health, params, options):
    # choice booleans
    retired = is_retired(choice)
    unemployed = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)
    partner_retired = partner_state == PARTNER_RETIRED

    bad_health = is_bad_health(health)
    good_health = is_good_health(health)

    # informal_care = is_informal_care(choice)

    # t = period / PERIOD_SCALE

    # disutil_ft_work_women = (
    #     params["disutil_ft_work_bad_women"] * bad_health
    #     + params["disutil_ft_work_good_women"] * good_health
    #     + params["disutil_ft_work_low_women"] * (1 - education)
    #     + params["disutil_ft_work_high_women"] * education
    # )
    # disutil_pt_work_women = (
    #     params["disutil_pt_work_bad_women"] * bad_health
    #     + params["disutil_pt_work_good_women"] * good_health
    #     + params["disutil_pt_work_low_women"] * (1 - education)
    #     + params["disutil_pt_work_high_women"] * education
    # )
    disutil_ft_work_women = (
        params["disutil_ft_work_high_bad"] * bad_health * education
        + params["disutil_ft_work_low_bad"] * bad_health * (1 - education)
        + params["disutil_ft_work_high_good"] * good_health * education
        + params["disutil_ft_work_low_good"] * good_health * (1 - education)
    )
    disutil_pt_work_women = (
        params["disutil_pt_work_high_bad"] * bad_health * education
        + params["disutil_pt_work_low_bad"] * bad_health * (1 - education)
        + params["disutil_pt_work_high_good"] * good_health * education
        + params["disutil_pt_work_low_good"] * good_health * (1 - education)
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

    # disutil_unemployed_and_informal_care = (
    #     params["disutil_unemployed_and_informal_care_low"] * (1 - education)
    #     + params["disutil_unemployed_and_informal_care_high"] * education
    # )
    # # util_pt_work_and_informal_care = (
    # #     params["util_pt_work_and_informal_care_low"] * (1 - education)
    # #     + params["util_pt_work_and_informal_care_high"] * education
    # # )
    # # util_ft_work_and_informal_care = (
    # #     params["util_ft_work_and_informal_care_low"] * (1 - education)
    # #     + params["util_ft_work_and_informal_care_high"] * education
    # # )
    # # util_unemployed_and_informal_care = (
    # #     params["util_unemployed_and_informal_care_bad"] * bad_health
    # #     + params["util_unemployed_and_informal_care_good"] * good_health
    # # )
    # disutil_pt_work_and_informal_care = (
    #     params["disutil_pt_work_and_informal_care_bad"] * bad_health
    #     + params["disutil_pt_work_and_informal_care_good"] * good_health
    # )
    # disutil_ft_work_and_informal_care = (
    #     params["disutil_ft_work_and_informal_care_bad"] * bad_health
    #     + params["disutil_ft_work_and_informal_care_good"] * good_health
    # )

    # disutil_children_ft_informal_care_low = (
    #     params["disutil_children_ft_work_informal_care_low"] * nb_children
    # )
    # disutil_children_ft_informal_care_high = (
    #     params["disutil_children_ft_work_informal_care_high"] * nb_children
    # )

    # disutil_children_pt_informal_care_low = (
    #     params["disutil_children_pt_work_informal_care_low"] * nb_children
    # )
    # disutil_children_pt_informal_care_high = (
    #     params["disutil_children_pt_work_informal_care_high"] * nb_children
    # )

    # disutil_children_pt_informal_care = (
    #     disutil_children_pt_informal_care_low * (1 - education)
    #     + disutil_children_pt_informal_care_high * education
    # )
    # disutil_children_ft_informal_care = (
    #     disutil_children_ft_informal_care_low * (1 - education)
    #     + disutil_children_ft_informal_care_high * education
    # )

    # exp_factor_work_and_care = (
    #     disutil_unemployed_and_informal_care * unemployed
    #     + (disutil_pt_work_and_informal_care + disutil_children_pt_informal_care)
    #     * working_part_time
    #     + (disutil_ft_work_and_informal_care + disutil_children_ft_informal_care)
    #     * working_full_time
    # )

    # =================================================================================

    # Compute eta
    disutility = (
        -exp_factor_work  # * (1 - informal_care)
        # - exp_factor_work_and_care * informal_care
        - partner_retired * retired * params["disutil_partner_retired"]
    )

    return disutility


# def consumption_scale(has_partner, n_children):
#     """Adjust for number of people living in household."""
#     hh_size = 1 + has_partner + n_children
#     return jnp.sqrt(hh_size)


def consumption_scale(partner_state, education, period, options):
    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
