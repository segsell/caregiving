"""State space for the no-inheritance model.

This module provides state space functions for the no-inheritance counterfactual.
The only difference from the baseline is that experience accumulation does not
include the intensive care bonus for part-time workers.
"""

from caregiving.model.experience_no_inheritance import get_next_period_experience
from caregiving.model.state_space import (
    next_period_deterministic_state,
    sparsity_condition,
    state_specific_choice_set_with_caregiving,
)


def create_state_space_functions():
    """Create state space functions for the no-inheritance model.

    Returns a dictionary of state space functions. All functions except
    `get_next_period_experience` are imported from the baseline `state_space.py`
    module, as they do not depend on inheritance calculations.

    The `get_next_period_experience` function is imported from
    `experience_no_inheritance.py`, which removes the intensive care bonus
    for part-time workers.

    Returns
    -------
    dict
        Dictionary containing:
        - "state_specific_choice_set": state_specific_choice_set_with_caregiving
        - "next_period_deterministic_state": next_period_deterministic_state
        - "next_period_experience": get_next_period_experience (from  # noqa: E501
        experience_no_inheritance)
        - "sparsity_condition": sparsity_condition
    """
    return {
        "state_specific_choice_set": state_specific_choice_set_with_caregiving,
        "next_period_deterministic_state": next_period_deterministic_state,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
    }
