"""Utility functions for the model with care demand and caregiving."""

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
    is_formal_care,
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


def disutility_work(
    period, choice, education, partner_state, health, care_demand, params, options
):
    """Compute disutility of work."""
    # choice booleans
    retired = is_retired(choice)
    unemployed = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)
    partner_retired = partner_state == PARTNER_RETIRED

    bad_health = is_bad_health(health)
    good_health = is_good_health(health)

    # Check care arrangement type based on choice
    # NO_CARE (choices 0, 1, 2, 3):
    #   No formal care, someone else provides informal care.
    # INFORMAL_CARE (choices 4, 5, 6, 7):
    #   Agent provides informal care.
    # FORMAL_CARE (choices 8, 9, 10, 11):
    #   Formal care is organized.
    informal_care = is_informal_care(choice)  # Agent provides informal care
    formal_care = is_formal_care(choice)  # Formal care

    has_partner_int = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner_int, period]

    # =================================================================================
    # No caregiving
    # =================================================================================

    disutil_ft_work = (
        params["disutil_ft_work_high_bad"] * bad_health * education
        + params["disutil_ft_work_low_bad"] * bad_health * (1 - education)
        + params["disutil_ft_work_high_good"] * good_health * education
        + params["disutil_ft_work_low_good"] * good_health * (1 - education)
        # + params["disutil_ft_work_low"] * (1 - education)
    )
    disutil_pt_work = (
        params["disutil_pt_work_high_bad"] * bad_health * education
        + params["disutil_pt_work_low_bad"] * bad_health * (1 - education)
        + params["disutil_pt_work_high_good"] * good_health * education
        + params["disutil_pt_work_low_good"] * good_health * (1 - education)
        # + params["disutil_pt_work_low"] * (1 - education)
    )
    disutil_unemployed = (
        params["disutil_unemployed_low_women"] * (1 - education)
        + params["disutil_unemployed_high_women"] * education
        # params["disutil_unemployed_high_bad_women"] * bad_health * education
        # + params["disutil_unemployed_low_bad_women"] * bad_health * (1 - education)
        # + params["disutil_unemployed_high_good_women"] * good_health * education
        # + params["disutil_unemployed_low_good_women"] * good_health * (1 - education)
    )

    # disutil_children_ue_low = params["disutil_children_unemployed_low"]*nb_children
    # disutil_children_ue_high = params["disutil_children_unemployed_high"]*nb_children

    disutil_children_pt_low = params["disutil_children_pt_work_low"] * nb_children
    disutil_children_pt_high = params["disutil_children_pt_work_high"] * nb_children

    disutil_children_ft_low = params["disutil_children_ft_work_low"] * nb_children
    disutil_children_ft_high = params["disutil_children_ft_work_high"] * nb_children

    # disutil_children_ue = (
    #     disutil_children_ue_low * (1 - education)
    # + disutil_children_ue_high * education
    # )
    disutil_children_pt = (
        disutil_children_pt_low * (1 - education) + disutil_children_pt_high * education
    )
    disutil_children_ft = (
        disutil_children_ft_low * (1 - education) + disutil_children_ft_high * education
    )

    disutility_work_and_no_informal_care = (
        (disutil_unemployed) * unemployed
        + (disutil_pt_work + disutil_children_pt) * working_part_time
        + (disutil_ft_work + disutil_children_ft) * working_full_time
    )

    # =================================================================================
    # Caregiving
    # =================================================================================

    disutil_ft_work_informal_care = (
        params["disutil_ft_work_high_bad_informal_care"] * bad_health * education
        + params["disutil_ft_work_low_bad_informal_care"] * bad_health * (1 - education)
        + params["disutil_ft_work_high_good_informal_care"] * good_health * education
        + params["disutil_ft_work_low_good_informal_care"]
        * good_health
        * (1 - education)
        # + params["disutil_ft_work_low_informal_care"] * (1 - education)
    )
    disutil_pt_work_informal_care = (
        params["disutil_pt_work_high_bad_informal_care"] * bad_health * education
        + params["disutil_pt_work_low_bad_informal_care"] * bad_health * (1 - education)
        + params["disutil_pt_work_high_good_informal_care"] * good_health * education
        + params["disutil_pt_work_low_good_informal_care"]
        * good_health
        * (1 - education)
        # + params["disutil_pt_work_low_informal_care"] * (1 - education)
    )
    # disutil_ft_work_informal_care = params[
    #     "disutil_ft_work_high_informal_care"
    # ] * education + params["disutil_ft_work_low_informal_care"] * (1 - education)
    # disutil_pt_work_informal_care = params[
    #     "disutil_pt_work_high_informal_care"
    # ] * education + params["disutil_pt_work_low_informal_care"] * (1 - education)

    disutil_unemployed_informal_care = (
        params["disutil_unemployed_low_women_informal_care"] * (1 - education)
        + params["disutil_unemployed_high_women_informal_care"] * education
    )

    disutil_children_ft_low_informal_care = (
        params["disutil_children_ft_work_low_informal_care"] * nb_children
    )
    disutil_children_ft_high_informal_care = (
        params["disutil_children_ft_work_high_informal_care"] * nb_children
    )

    disutil_children_pt_low_informal_care = (
        params["disutil_children_pt_work_low_informal_care"] * nb_children
    )
    disutil_children_pt_high_informal_care = (
        params["disutil_children_pt_work_high_informal_care"] * nb_children
    )

    disutil_children_pt_informal_care = (
        disutil_children_pt_low_informal_care * (1 - education)
        + disutil_children_pt_high_informal_care * education
    )
    disutil_children_ft_informal_care = (
        disutil_children_ft_low_informal_care * (1 - education)
        + disutil_children_ft_high_informal_care * education
    )

    disutility_work_and_informal_care = (
        disutil_unemployed_informal_care * unemployed
        + (disutil_pt_work_informal_care + disutil_children_pt_informal_care)
        * working_part_time
        + (disutil_ft_work_informal_care + disutil_children_ft_informal_care)
        * working_full_time
    )

    # =================================================================================
    # Care utility (based on agent's own health)
    # =================================================================================
    #
    # Four care demand scenarios:
    # 1. care_demand == CARE_DEMAND_AND_OTHER_SUPPLY (1):
    #    - If agent chooses informal care → joint informal care
    #      (util_joint_informal_care_good/bad)
    #    - If agent chooses no informal care → only other family member
    #      provides care (utility = 0)
    # 2. care_demand == CARE_DEMAND_AND_NO_OTHER_SUPPLY (2):
    #    - If agent chooses informal care → solo informal care
    #      (util_informal_care_good/bad)
    #    - If agent chooses no care → formal care (util_formal_care_good/bad)

    # Compute utility from solo informal care (varies by agent's health and education)
    util_informal_care = (
        params["util_informal_care_high_good"] * good_health * education
        + params["util_informal_care_high_bad"] * bad_health * education
        + params["util_informal_care_low_good"] * good_health * (1 - education)
        + params["util_informal_care_low_bad"] * bad_health * (1 - education)
    )

    # Compute utility from formal care (varies by agent's health)
    util_formal_care = (
        params["util_formal_care_good"] * good_health
        + params["util_formal_care_bad"] * bad_health
    )

    # Utility from care arrangements.
    # When care_demand == 1:
    #   - FORMAL_CARE (choices 8, 9, 10, 11):
    #     Formal care is organized --> util_formal_care.
    #   - INFORMAL_CARE (choices 4, 5, 6, 7):
    #     Agent provides informal care --> util_informal_care.
    #   - NO_CARE (choices 0, 1, 2, 3):
    #     Someone else provides informal care --> reference category.
    utility_from_care = (
        informal_care * util_informal_care + formal_care * util_formal_care
    )

    # =================================================================================
    # Compute total disutility
    # =================================================================================
    # - NO_CARE (choices 0, 1, 2, 3):
    #   Someone else provides care → disutility_no_caregiving.
    # - INFORMAL_CARE (choices 4, 5, 6, 7):
    #   Agent provides care → disutility_informal_care.
    # - FORMAL_CARE (choices 8, 9, 10, 11):
    #   Formal care → disutility_no_caregiving.

    disutility = (
        -disutility_work_and_no_informal_care * (1 - informal_care)
        - disutility_work_and_informal_care * informal_care
        - partner_retired * retired * params["disutil_partner_retired"]
        + utility_from_care * (care_demand == 1)
    )

    return disutility


def consumption_scale(partner_state, education, period, options):
    """Compute the household consumption scale."""
    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][SEX, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
