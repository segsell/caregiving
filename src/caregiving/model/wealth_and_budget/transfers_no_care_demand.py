"""Transfer calculation functions for the no-care-demand counterfactual.

This module provides transfer calculation functions adapted for the reduced 4-state
choice space without informal caregiving model.
"""

import jax
import jax.numpy as jnp

from caregiving.model.shared import SEX


def calc_inheritance_amount_no_care_demand(
    period,
    education,
    model_specs,
):
    """Calculate inheritance amount for no care demand counterfactual.

    Uses precomputed inheritance amount matrix from specs.
    In the no care demand counterfactual, there is no caregiving, so uses no_care column (index 0).

    Args:
        period: Current period
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

    # Look up amount for no_care (index 0) at (sex, period, education)
    inheritance_amount = inheritance_amount_mat[sex_var, period, education, 0]

    return inheritance_amount


def draw_inheritance_outcome_no_care_demand(
    period,
    lagged_choice,
    education,
    asset_end_of_previous_period,
    model_specs,
):
    """Draw inheritance outcome (0 or 1) using Bernoulli distribution for no care demand model.

    Uses precomputed inheritance probability matrix and performs a Bernoulli draw
    with a deterministic seed based on state variables for reproducibility.

    Args:
        period: Current period
        education: Education level
        asset_end_of_previous_period: Assets at end of previous period (for seed variation)
        model_specs: Model specifications dictionary containing:
            - inheritance_prob_mat: Precomputed probability matrix of shape
              (n_sexes, n_periods, n_education, 3) where last dim is [no_care, light_care, intensive_care]
            - seed: Base random seed

    Returns:
        Binary outcome (0 or 1) as uint8: 1 if inheritance is received, 0 otherwise.
    """
    sex_var = SEX

    # Get probability of receiving inheritance from precomputed matrix
    # Shape: (n_sexes, n_periods, n_education, 3)
    # Last dimension: [no_care, light_care, intensive_care]
    inheritance_prob_mat = model_specs["inheritance_prob_mat"]
    prob_inheritance = inheritance_prob_mat[sex_var, period, education, 0]

    # Draw a uniform random number [0, 1] using a deterministic seed from model_specs
    # Create a deterministic key based on seed and state for reproducibility
    base_seed = model_specs["seed"]
    # inheritance_seed = jnp.int64(
    #     base_seed + period * 1000 + education * 200 + asset_end_of_previous_period * 3
    # )
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
