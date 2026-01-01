"""Transfer calculation functions for the no-care-demand counterfactual.

This module provides transfer calculation functions adapted for the reduced 4-state
choice space without informal caregiving model.
"""

from jax import numpy as jnp

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
