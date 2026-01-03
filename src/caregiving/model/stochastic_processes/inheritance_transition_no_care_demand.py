"""Inheritance transition probability for no care demand counterfactual.

Uses precomputed inheritance probability matrix from specs.
In the no care demand counterfactual, there is no caregiving, so uses no_care column.
"""

import jax.numpy as jnp

from caregiving.model.shared import SEX


def inheritance_transition_no_care_demand(period, education, model_specs):
    """Transition probability for receiving inheritance (no care demand counterfactual).

    Uses precomputed inheritance probability matrix from specs.
    In the no care demand counterfactual, there is no caregiving, so uses no_care column (index 0).

    Parameters
    ----------
    period : int
        Current period
    education : int
        Education level (0=Low, 1=High)
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

    # Look up probability for no_care (index 0) at (sex, period, education)
    prob_positive_inheritance = inheritance_prob_mat[sex_var, period, education, 0]

    # Return probability vector: [prob_no_inheritance, prob_inheritance]
    prob_no_inheritance = 1.0 - prob_positive_inheritance

    return jnp.array([prob_no_inheritance, prob_positive_inheritance])
