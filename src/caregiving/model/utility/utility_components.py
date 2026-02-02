"""Utility functions for the model with care demand and caregiving."""

import jax.numpy as jnp

from caregiving.model.shared import (  # is_nursing_home_care,
    AGE_40,
    CARE_DEMAND_LIGHT,
    PARTNER_RETIRED,
    SEX,
    is_bad_health,
    is_formal_care,
    is_full_time,
    is_good_health,
    is_informal_care,
    is_intensive_informal_care,
    is_light_informal_care,
    is_part_time,
    is_retired,
    is_unemployed,
)


def disutility_work(
    period, choice, education, partner_state, health, care_demand, params, model_specs
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
    light_informal_care = is_light_informal_care(choice)
    intensive_informal_care = is_intensive_informal_care(choice)
    formal_care = is_formal_care(choice)  # Formal care

    has_partner_int = (partner_state > 0).astype(int)
    nb_children = model_specs["children_by_state"][
        SEX, education, has_partner_int, period
    ]

    # Calculate age for age-based parameters
    age = period + model_specs["start_age"]
    age_below_40 = (age < AGE_40).astype(int)
    age_above_40 = (age >= AGE_40).astype(int)

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

    # Age-based disutility from children (age < 40 vs age >= 40)
    disutil_children_pt_low = (
        params["disutil_children_pt_work_low_below_40"] * age_below_40
        + params["disutil_children_pt_work_low_above_40"] * age_above_40
    ) * nb_children
    disutil_children_pt_high = (
        params["disutil_children_pt_work_high_below_40"] * age_below_40
        + params["disutil_children_pt_work_high_above_40"] * age_above_40
    ) * nb_children

    disutil_children_ft_low = (
        params["disutil_children_ft_work_low_below_40"] * age_below_40
        + params["disutil_children_ft_work_low_above_40"] * age_above_40
    ) * nb_children
    disutil_children_ft_high = (
        params["disutil_children_ft_work_high_below_40"] * age_below_40
        + params["disutil_children_ft_work_high_above_40"] * age_above_40
    ) * nb_children

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

    # Level shifts for labor and care intensity by education type
    level_shift_work_and_light_informal_care = (
        params["disutil_unemployed_light_informal_care_high"] * education * unemployed
        + params["disutil_ft_work_light_informal_care_high"]
        * education
        * working_full_time
        + params["disutil_unemployed_light_informal_care_low"]
        * (1 - education)
        * unemployed
        + params["disutil_ft_work_light_informal_care_low"]
        * (1 - education)
        * working_full_time
    )
    level_shift_work_and_intensive_informal_care = (
        params["disutil_unemployed_intensive_informal_care_high"]
        * education
        * unemployed
        + params["disutil_ft_work_intensive_informal_care_high"]
        * education
        * working_full_time
        + params["disutil_unemployed_intensive_informal_care_low"]
        * (1 - education)
        * unemployed
        + params["disutil_ft_work_intensive_informal_care_low"]
        * (1 - education)
        * working_full_time
    )

    # =================================================================================
    # Care utility (based on agent's own health)
    # =================================================================================

    # Compute utility from solo informal care (varies by agent's health and education)
    util_light_informal_care = (
        params["util_light_informal_care_high_good"] * good_health * education
        + params["util_light_informal_care_high_bad"] * bad_health * education
        + params["util_light_informal_care_low_good"] * good_health * (1 - education)
        + params["util_light_informal_care_low_bad"] * bad_health * (1 - education)
    )
    util_intensive_informal_care = (
        params["util_intensive_informal_care_high_good"] * good_health * education
        + params["util_intensive_informal_care_high_bad"] * bad_health * education
        + params["util_intensive_informal_care_low_good"]
        * good_health
        * (1 - education)
        + params["util_intensive_informal_care_low_bad"] * bad_health * (1 - education)
    )
    # util_informal_care = (
    #     params["util_informal_care_high_good"] * good_health * education
    #     + params["util_informal_care_high_bad"] * bad_health * education
    #     + params["util_informal_care_low_good"] * good_health * (1 - education)
    #     + params["util_informal_care_low_bad"] * bad_health * (1 - education)
    # )

    # Compute utility from formal care (varies by agent's health and education)
    util_formal_care = (
        params["util_formal_care_high_good"] * good_health * education
        + params["util_formal_care_high_bad"] * bad_health * education
        + params["util_formal_care_low_good"] * good_health * (1 - education)
        + params["util_formal_care_low_bad"] * bad_health * (1 - education)
    )

    # Someone else provides informal care --> reference category.
    utility_from_care = (
        # informal_care * util_informal_care
        light_informal_care * util_light_informal_care
        + intensive_informal_care * util_intensive_informal_care
        + formal_care * util_formal_care
    )

    # =================================================================================
    # Compute total disutility
    # =================================================================================

    disutility = (
        -disutility_work_and_no_informal_care * (1 - informal_care)
        - disutility_work_and_informal_care * informal_care
        + level_shift_work_and_intensive_informal_care * intensive_informal_care
        + level_shift_work_and_light_informal_care * light_informal_care
        - partner_retired * retired * params["disutil_partner_retired"]
        + utility_from_care * (care_demand >= CARE_DEMAND_LIGHT)
    )

    return disutility


def consumption_scale(partner_state, education, period, model_specs):
    """Compute the household consumption scale."""
    has_partner = (partner_state > 0).astype(int)
    nb_children = model_specs["children_by_state"][SEX, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
