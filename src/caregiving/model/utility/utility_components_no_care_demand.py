"""Utility component helpers for the no-care-demand counterfactual.

This removes any dependencies on informal care choices and keeps only
labor disutility and consumption scale components.
"""

import jax.numpy as jnp

from caregiving.model.shared import (
    PARTNER_RETIRED,
    SEX,
    is_bad_health,
    is_good_health,
)
from caregiving.model.shared_no_care_demand import (
    is_full_time,
    is_part_time,
    is_retired,
    is_unemployed,
)


def disutility_work(
    period, choice, education, partner_state, health, params, model_specs
):
    retired = is_retired(choice)
    unemployed = is_unemployed(choice)
    working_part_time = is_part_time(choice)
    working_full_time = is_full_time(choice)
    partner_retired = partner_state == PARTNER_RETIRED

    bad_health = is_bad_health(health)
    good_health = is_good_health(health)

    has_partner_int = (partner_state > 0).astype(int)
    nb_children = model_specs["children_by_state"][
        SEX, education, has_partner_int, period
    ]

    disutil_ft_work = (
        params["disutil_ft_work_high_bad"] * bad_health * education
        + params["disutil_ft_work_low_bad"] * bad_health * (1 - education)
        + params["disutil_ft_work_high_good"] * good_health * education
        + params["disutil_ft_work_low_good"] * good_health * (1 - education)
    )
    disutil_pt_work = (
        params["disutil_pt_work_high_bad"] * bad_health * education
        + params["disutil_pt_work_low_bad"] * bad_health * (1 - education)
        + params["disutil_pt_work_high_good"] * good_health * education
        + params["disutil_pt_work_low_good"] * good_health * (1 - education)
    )
    disutil_unemployed = (
        params["disutil_unemployed_low_women"] * (1 - education)
        + params["disutil_unemployed_high_women"] * education
        # params["disutil_unemployed_high_bad_women"] * bad_health * education
        # + params["disutil_unemployed_low_bad_women"] * bad_health * (1 - education)
        # + params["disutil_unemployed_high_good_women"] * good_health * education
        # + params["disutil_unemployed_low_good_women"] * good_health * (1 - education)
    )

    disutil_children_pt_low = params["disutil_children_pt_work_low"] * nb_children
    disutil_children_pt_high = params["disutil_children_pt_work_high"] * nb_children

    disutil_children_ft_low = params["disutil_children_ft_work_low"] * nb_children
    disutil_children_ft_high = params["disutil_children_ft_work_high"] * nb_children

    disutil_children_pt = (
        disutil_children_pt_low * (1 - education) + disutil_children_pt_high * education
    )
    disutil_children_ft = (
        disutil_children_ft_low * (1 - education) + disutil_children_ft_high * education
    )

    disutility_no_caregiving = (
        (disutil_unemployed) * unemployed
        + (disutil_pt_work + disutil_children_pt) * working_part_time
        + (disutil_ft_work + disutil_children_ft) * working_full_time
    )

    disutility = (
        -disutility_no_caregiving
        - partner_retired * retired * params["disutil_partner_retired"]
    )
    return disutility


def consumption_scale(partner_state, education, period, model_specs):
    has_partner = (partner_state > 0).astype(int)
    nb_children = model_specs["children_by_state"][SEX, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
