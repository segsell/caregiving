import jax
from jax import numpy as jnp

from caregiving.model.shared import (
    SEX,
    is_formal_care,
    is_informal_care,
    is_intensive_informal_care,
    is_light_informal_care,
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


# =====================================================================================
# Inheritance
# =====================================================================================


def calc_inheritance_amount(
    period,
    lagged_choice,
    education,
    model_specs,
):
    """Calculate inheritance amount for baseline model.

    Uses precomputed inheritance amount matrix from specs.
    Determines care type from lagged_choice to select appropriate column:
    - no_care (index 0): if lagged_choice is not in any care type
    - formal_care (index 1): if lagged_choice is in FORMAL_CARE
    - light_care (index 2): if lagged_choice is in LIGHT_INFORMAL_CARE
    - intensive_care (index 3): if lagged_choice is in INTENSIVE_INFORMAL_CARE

    Args:
        period: Current period
        lagged_choice: Choice from previous period (d_{t-1})
        education: Education level
        model_specs: Model specifications dictionary containing:
            - inheritance_amount_mat: Precomputed amount matrix of shape
              (n_sexes, n_periods, n_education, 4) where last dim is [no_care,
                  formal_care, light_care, intensive_care]

    Returns:
        Expected inheritance amount (conditional on positive inheritance).

    """
    sex_var = SEX

    # Get precomputed inheritance amount matrix
    # Shape: (n_sexes, n_periods, n_education, 4)
    # Last dimension: [no_care, formal_care, light_care, intensive_care]
    inheritance_amount_mat = model_specs["inheritance_amount_mat"]

    # Determine care type index from lagged_choice
    # Care types are mutually exclusive
    is_light = is_light_informal_care(lagged_choice)
    is_intensive = is_intensive_informal_care(lagged_choice)
    is_formal = is_formal_care(lagged_choice)

    # Select care type index: 0=no_care, 1=formal_care, 2=light_care, 3=intensive_care
    # Care types are mutually exclusive, so use arithmetic instead of nested
    # conditionals. This is faster on GPU: is_formal*1 + is_light*2 + is_intensive*3
    care_type_idx = (
        is_formal.astype(int) * 1
        + is_light.astype(int) * 2
        + is_intensive.astype(int) * 3
    )

    # Look up amount at (sex, period, education, care_type_idx)
    inheritance_amount = inheritance_amount_mat[
        sex_var, period, education, care_type_idx
    ]

    return inheritance_amount


def draw_inheritance_outcome(
    period,
    lagged_choice,
    education,
    asset_end_of_previous_period,
    model_specs,
):
    """Draw inheritance outcome (0 or 1) using Bernoulli distribution for  # noqa: E501
    baseline model.

    Uses precomputed inheritance probability matrix and performs a Bernoulli draw
    with a deterministic seed based on state variables for reproducibility.

    Args:
        period: Current period
        lagged_choice: Choice from previous period (d_{t-1}) - used to  # noqa: E501
        determine care type
        education: Education level
        asset_end_of_previous_period: Assets at end of previous period  # noqa: E501
        (for seed variation)
        model_specs: Model specifications dictionary containing:
            - inheritance_prob_mat: Precomputed probability matrix of shape
              (n_sexes, n_periods, n_education, 4) where last dim is [no_care,
                  formal_care, light_care, intensive_care]
            - seed: Base random seed

    Returns:
        Binary outcome (0 or 1) as uint8: 1 if inheritance is received, 0 otherwise.
    """
    sex_var = SEX

    # Get probability of receiving inheritance from precomputed matrix
    # Shape: (n_sexes, n_periods, n_education, 4)
    # Last dimension: [no_care, formal_care, light_care, intensive_care]
    inheritance_prob_mat = model_specs["inheritance_prob_mat"]

    # Determine care type index from lagged_choice
    # Care types are mutually exclusive:
    # - Index 0: no_care (no informal care, no formal care)
    # - Index 1: formal_care (formal care only)
    # - Index 2: light_care (light informal care only)
    # - Index 3: intensive_care (intensive informal care only)
    is_light = is_light_informal_care(lagged_choice)
    is_intensive = is_intensive_informal_care(lagged_choice)
    is_formal = is_formal_care(lagged_choice)

    # Select care type index: 0=no_care, 1=formal_care, 2=light_care, 3=intensive_care
    # Care types are mutually exclusive, so use arithmetic instead of nested
    # conditionals. This is faster on GPU: is_formal*1 + is_light*2 + is_intensive*3
    care_type_idx = (
        is_formal.astype(int) * 1
        + is_light.astype(int) * 2
        + is_intensive.astype(int) * 3
    )

    prob_inheritance = inheritance_prob_mat[sex_var, period, education, care_type_idx]

    # Draw a uniform random number [0, 1] using a deterministic seed from model_specs
    # Create a deterministic key based on seed and state for reproducibility
    base_seed = model_specs["seed"]
    inheritance_seed = jnp.uint16(
        base_seed
        + period * 100
        + lagged_choice * 7
        + education * (3 + 1)
        + (1 - education)
        # + 100 * is_intensive_informal_care(lagged_choice)
        # + 50 * is_light_informal_care(lagged_choice)
        + asset_end_of_previous_period  # already scaled by wealth_unit
    )
    key = jax.random.PRNGKey(inheritance_seed)
    uniform_draw = jax.random.uniform(key, shape=(), minval=0.0, maxval=1.0)

    # Convert probability to discrete 0 or 1: if prob >= uniform_draw, then 1, else 0
    # Using uint8 for memory efficiency (values are only 0 or 1)
    gets_inheritance = (prob_inheritance >= uniform_draw).astype(jnp.uint8)

    return gets_inheritance
