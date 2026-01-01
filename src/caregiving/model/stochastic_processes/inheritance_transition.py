"""Inheritance transition probability for baseline model.

Uses precomputed inheritance probability matrix from specs.
Determines care type from lagged_choice to select appropriate column.
"""

import jax.numpy as jnp

from caregiving.model.shared import (
    SEX,
    is_informal_care,
)


def inheritance_transition(period, education, lagged_choice, model_specs):
    """Transition probability for receiving inheritance (baseline model).

    Uses precomputed inheritance probability matrix from specs.
    Determines care type from lagged_choice:
    - no_care (index 0): if lagged_choice is not in LIGHT_INFORMAL_CARE or INTENSIVE_INFORMAL_CARE
    - light_care (index 1): if lagged_choice is in LIGHT_INFORMAL_CARE
    - intensive_care (index 2): if lagged_choice is in INTENSIVE_INFORMAL_CARE

    Parameters
    ----------
    period : int
        Current period
    education : int
        Education level (0=Low, 1=High)
    lagged_choice : int
        Previous period's choice (determines care type)
    model_specs : dict
        Model specifications dictionary containing:
        - inheritance_prob_mat: Precomputed probability matrix of shape
          (n_sexes, n_periods, n_education, 3) where last dim is [no_care, light_care, intensive_care]

    Returns
    -------
    jnp.ndarray
        Probability vector over gets_inheritance states [0, 1], where:
        - 0: no inheritance received
        - 1: inheritance received
        Shape: (2,)

    """
    sex_var = SEX

    # Get precomputed inheritance probability matrix
    # Shape: (n_sexes, n_periods, n_education, 3)
    # Last dimension: [no_care, light_care, intensive_care]
    inheritance_prob_mat = model_specs["inheritance_prob_mat"]

    # Determine care type from lagged_choice
    # Note: The probability model uses any_care (binary), so light_care and intensive_care
    # have the same probability value. We use light_care (index 1) for both.
    is_any_informal_care = is_informal_care(lagged_choice)

    # Select care type index: 0=no_care, 1=light_care
    # Since light and intensive have the same probability, we use index 1 for any informal care
    # Convert boolean to int: False -> 0 (no_care), True -> 1 (light_care)
    care_type_idx = is_any_informal_care.astype(int)

    # Look up probability at (sex, period, education, care_type_idx)
    prob_positive_inheritance = inheritance_prob_mat[
        sex_var, period, education, care_type_idx
    ]

    # Return probability vector: [prob_no_inheritance, prob_inheritance]
    prob_no_inheritance = 1.0 - prob_positive_inheritance

    return jnp.array([prob_no_inheritance, prob_positive_inheritance])
