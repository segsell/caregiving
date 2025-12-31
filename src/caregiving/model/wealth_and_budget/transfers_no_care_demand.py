"""Transfer calculation functions for the no-care-demand counterfactual.

This module provides transfer calculation functions adapted for the reduced 4-state
choice space without informal caregiving model.
"""

import jax
from jax import numpy as jnp

from caregiving.model.shared import PARENT_RECENTLY_DEAD, SEX
from caregiving.model.wealth_and_budget.transfers import draw_inheritance_outcome


def calc_inheritance_no_care_demand(
    period,
    education,
    mother_dead,
    model_specs,
):
    """Calculate inheritance for no care demand counterfactual.

    In the no care demand counterfactual, there is no caregiving, so
    any_care, light_care, and intensive_care are always 0. This function
    explicitly sets these values instead of relying on choice-based functions.

    Args:
        period: Current period
        education: Education level
        mother_dead: Mother death status (0=alive, 1=recently died, 2=longer dead)
        model_specs: Model specifications dictionary containing inheritance parameters
                     and a "seed" key for deterministic Bernoulli draw.

    Returns:
        Inheritance amount: Either 0 or full amount (binary draw based on probability).
    """
    sex_var = SEX
    start_age = model_specs["start_age"]
    age = start_age + period

    # Only compute inheritance if mother recently died this period (state 1)
    mother_died_recently = mother_dead == PARENT_RECENTLY_DEAD

    # Get sex label for parameter lookup
    sex_label = model_specs["sex_labels"][sex_var]

    # Step 1: Compute probability of positive inheritance using spec7 parameters
    # In no care demand counterfactual, any_care is always 0
    any_care = 0
    age_sq = age**2

    # Get spec7 logit parameters (stored as spec5_params key for backward compatibility)
    inheritance_prob_params = model_specs["inheritance_prob_spec5_params"]

    # Compute logit linear predictor
    # X = [age, age_sq, any_care, education]
    logit_linear = (
        inheritance_prob_params.loc[sex_label, "age"] * age
        + inheritance_prob_params.loc[sex_label, "age_sq"] * age_sq
        + inheritance_prob_params.loc[sex_label, "any_care"] * any_care
        + inheritance_prob_params.loc[sex_label, "education"] * education
        + inheritance_prob_params.loc[sex_label, "const"]
    )

    # Compute probability using logistic function: P = 1 / (1 + exp(-X))
    prob_positive_inheritance = 1.0 / (1.0 + jnp.exp(-logit_linear))

    # Step 2: Compute expected inheritance amount using spec12 parameters
    # In no care demand counterfactual, light_care and intensive_care are always 0
    light_care = 0
    intensive_care = 0

    # Get spec12 OLS parameters (stored as spec5_params key for backward compatibility)
    inheritance_amount_params = model_specs["inheritance_amount_spec5_params"]

    # Compute OLS linear predictor for ln(inheritance_amount)
    # X = [age, age_sq, light_care_recent, intensive_care_recent, education]
    ln_inheritance_amount = (
        inheritance_amount_params.loc[sex_label, "age"] * age
        + inheritance_amount_params.loc[sex_label, "age_sq"] * age_sq
        + inheritance_amount_params.loc[sex_label, "light_care_recent"] * light_care
        + inheritance_amount_params.loc[sex_label, "intensive_care_recent"]
        * intensive_care
        + inheritance_amount_params.loc[sex_label, "education"] * education
        + inheritance_amount_params.loc[sex_label, "const"]
    )

    # Convert from log to level: amount = exp(ln(amount))
    inheritance_amount = jnp.exp(ln_inheritance_amount)

    # In practice, inheritance receipt is binary (0 or 1)
    # Use Bernoulli draw to determine if inheritance is received
    seed = model_specs["seed"]
    gets_inheritance = draw_inheritance_outcome(prob_positive_inheritance, seed)
    inheritance = gets_inheritance * inheritance_amount

    # Only return inheritance if mother is dead
    return mother_died_recently * inheritance
