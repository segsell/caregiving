import jax
from jax import numpy as jnp

from caregiving.model.shared import (
    PARENT_RECENTLY_DEAD,
    SEX,
    is_formal_care,
    is_informal_care,
    is_intensive_informal_care,
    is_light_informal_care,
    is_no_care,
)


def calc_child_benefits(sex, education, has_partner_int, period, model_specs):
    """Calculate the child benefits."""
    n_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]

    return n_children * model_specs["monthly_child_benefits"] * 12


def calc_unemployment_benefits(
    assets, sex, education, has_partner_int, period, model_specs
):
    # Unemployment benefits means test
    means_test = assets < model_specs["unemployment_wealth_thresh"]

    # Unemployment benefits for children living in the household
    nb_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]
    unemployment_benefits_children = (
        nb_children * model_specs["annual_child_unemployment_benefits"]
    )

    own_unemployemnt_benefits = (
        model_specs["annual_unemployment_benefits"]
        + model_specs["annual_unemployment_benefits_housing"]
    )

    partner_unemployment_benefits = has_partner_int * (
        model_specs["annual_unemployment_benefits"]
        + model_specs["annual_unemployment_benefits_housing"] * 0.5
    )

    # Total unemployment benefits
    total_unemployment_benefits = (
        own_unemployemnt_benefits
        + partner_unemployment_benefits
        + unemployment_benefits_children
    )

    # reduced benefits for savings slightly above threshold
    reduced_benefits_threshhold = (
        model_specs["unemployment_wealth_thresh"] + total_unemployment_benefits
    )
    reduced_benefits_means_test = (1 - means_test) * (
        assets < reduced_benefits_threshhold
    )
    reduced_benefits = reduced_benefits_threshhold - assets

    household_unemployment_benefits = (
        means_test * total_unemployment_benefits
        + reduced_benefits_means_test * reduced_benefits
    )
    return household_unemployment_benefits, own_unemployemnt_benefits


# def calc_unemployment_benefits(
#     assets, sex, education, has_partner_int, period, model_specs
# ):
#     # Unemployment benefits means test
#     means_test = assets < model_specs["unemployment_wealth_thresh"]

#     # Unemployment benefits for children living in the household
#     n_children = model_specs["children_by_state"][
#         sex, education, has_partner_int, period
#     ]
#     unemployment_benefits_children = (
#         n_children * model_specs["annual_child_unemployment_benefits"]
#     )

#     # Unemployment benefits for adults living in the household
#     unemployment_benefits_adults = (1 + has_partner_int) * model_specs[
#         "annual_unemployment_benefits"
#     ]
#     # For housing, second adult gets only half
#     unemployment_benefits_housing = (1 + 0.5 * has_partner_int) * model_specs[
#         "annual_unemployment_benefits_housing"
#     ]

#     # Total unemployment benefits
#     total_unemployment_benefits = (
#         unemployment_benefits_adults
#         + unemployment_benefits_children
#         + unemployment_benefits_housing
#     )

#     # reduced benefits for assets slightly above threshold
#     reduced_benefits_threshhold = (
#         model_specs["unemployment_wealth_thresh"] + total_unemployment_benefits
#     )
#     reduced_benefits_means_test = (1 - means_test) * (
#         assets < reduced_benefits_threshhold
#     )
#     reduced_benefits = reduced_benefits_threshhold - assets

#     unemployment_benefits = (
#         means_test * total_unemployment_benefits
#         + reduced_benefits_means_test * reduced_benefits
#     )

#     return unemployment_benefits


def calc_care_benefits_and_costs(lagged_choice, education, care_demand, model_specs):
    """Calculate the care benefits and costs."""

    informal_care_solo = is_informal_care(lagged_choice)
    formal_care = is_formal_care(lagged_choice)

    # # Care benefits
    # care_benefits = options["care_benefits"][education, has_sister]

    # # Care costs
    # care_costs = options["care_costs"][education, has_sister]

    annual_care_benefits = model_specs["informal_care_cash_benefits"] * 12
    annual_care_benefits_weighted = annual_care_benefits * informal_care_solo

    annual_care_costs_weighted = (
        model_specs["formal_care_costs"] * formal_care * 12 * 0.5
    )

    return annual_care_benefits_weighted - annual_care_costs_weighted


def calc_inheritance_amount(
    period,
    lagged_choice,
    education,
    model_specs,
):
    """Calculate inheritance amount for baseline model.

    Uses precomputed inheritance amount matrix from specs.
    Determines care type from lagged_choice to select appropriate column:
    - no_care (index 0): if lagged_choice is not in LIGHT_INFORMAL_CARE or INTENSIVE_INFORMAL_CARE
    - light_care (index 1): if lagged_choice is in LIGHT_INFORMAL_CARE
    - intensive_care (index 2): if lagged_choice is in INTENSIVE_INFORMAL_CARE

    Args:
        period: Current period
        lagged_choice: Choice from previous period (d_{t-1})
        education: Education level
        model_specs: Model specifications dictionary containing:
            - inheritance_amount_mat: Precomputed amount matrix of shape
              (n_sexes, n_periods, n_education, 3) where last dim is [no_care, light_care, intensive_care]

    Returns:
        Expected inheritance amount (conditional on positive inheritance).

    """
    sex_var = SEX

    # Get precomputed inheritance amount matrix
    # Shape: (n_sexes, n_periods, n_education, 3)
    # Last dimension: [no_care, light_care, intensive_care]
    inheritance_amount_mat = model_specs["inheritance_amount_mat"]

    # Determine care type index from lagged_choice
    is_light = is_light_informal_care(lagged_choice)
    is_intensive = is_intensive_informal_care(lagged_choice)

    # Select care type index: 0=no_care, 1=light_care, 2=intensive_care
    # Use jnp.where to select the appropriate index
    care_type_idx = jnp.where(
        is_intensive,
        2,  # intensive_care
        jnp.where(
            is_light,
            1,  # light_care
            0,  # no_care
        ),
    )

    # Look up amount at (sex, period, education, care_type_idx)
    inheritance_amount = inheritance_amount_mat[
        sex_var, period, education, care_type_idx
    ]

    return inheritance_amount
