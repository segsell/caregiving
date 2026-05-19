"""State space for the care pension credit model variant.

Identical to the baseline state space except the experience update function
grants unemployed intensive caregivers +0.5 pension credit (capped at 1.0).
"""

from caregiving.model.experience_baseline_model import (
    get_next_period_experience_care_pension_credit,
)
from caregiving.model.state_space import (
    next_period_deterministic_state,
    sparsity_condition,
    state_specific_choice_set_with_caregiving,
)


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set_with_caregiving,
        "next_period_deterministic_state": next_period_deterministic_state,
        "next_period_experience": get_next_period_experience_care_pension_credit,
        "sparsity_condition": sparsity_condition,
    }
