from jax import numpy as jnp

from caregiving.model.shared import (
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
    n_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]
    unemployment_benefits_children = (
        n_children * model_specs["annual_child_unemployment_benefits"]
    )

    # Unemployment benefits for adults living in the household
    unemployment_benefits_adults = (1 + has_partner_int) * model_specs[
        "annual_unemployment_benefits"
    ]
    # For housing, second adult gets only half
    unemployment_benefits_housing = (1 + 0.5 * has_partner_int) * model_specs[
        "annual_unemployment_benefits_housing"
    ]

    # Total unemployment benefits
    total_unemployment_benefits = (
        unemployment_benefits_adults
        + unemployment_benefits_children
        + unemployment_benefits_housing
    )

    # reduced benefits for assets slightly above threshold
    reduced_benefits_threshhold = (
        model_specs["unemployment_wealth_thresh"] + total_unemployment_benefits
    )
    reduced_benefits_means_test = (1 - means_test) * (
        assets < reduced_benefits_threshhold
    )
    reduced_benefits = reduced_benefits_threshhold - assets

    unemployment_benefits = (
        means_test * total_unemployment_benefits
        + reduced_benefits_means_test * reduced_benefits
    )

    return unemployment_benefits


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


def calc_inheritance(
    period,
    lagged_choice,
    education,
    mother_dead,
    model_specs,
):
    """Calculate expected inheritance.

    This function computes inheritance in two steps:
    1. Probability of positive inheritance using spec5 logit parameters
       (uses any_care dummy, no distinction between light/intensive, no parent dummy)
    2. Expected inheritance amount using spec5 OLS parameters
       (distinguishes between light and intensive care)

    Both steps condition on mother_dead == 1 this period.

    Args:
        period: Current period
        lagged_choice: Choice from previous period (d_{t-1})
        education: Education level
        mother_dead: Mother death status (0=alive, 1=recently died, 2=longer dead)
        model_specs: Model specifications dictionary containing inheritance parameters

    Returns:
        Expected inheritance amount (probability * amount)

    """
    sex_var = SEX
    start_age = model_specs["start_age"]
    age = start_age + period

    # Only compute inheritance if mother recently died this period (state 1)
    # State 0 = alive, State 1 = recently died (inheritance paid), State 2 = longer dead
    mother_dead_int = mother_dead == 1

    # Get sex label for parameter lookup
    sex_label = model_specs["sex_labels"][sex_var]

    # Step 1: Compute probability of positive inheritance using spec7 parameters
    # Spec7 uses: any_care, age, age_sq, education
    # Filter: parent_died_this_year == 1
    # (which corresponds to mother_dead == 1 this period)
    # NO parent variable in the regression (parent_var = None)
    # Parameters: age, age_sq, any_care, education, const

    # Check if any informal care was provided (light or intensive)
    any_care = is_informal_care(lagged_choice).astype(int)
    light_care = is_light_informal_care(lagged_choice).astype(int)
    intensive_care = is_intensive_informal_care(lagged_choice).astype(int)

    # Get spec7 logit parameters (stored as spec5_params key for backward compatibility)
    inheritance_prob_params = model_specs["inheritance_prob_spec5_params"]
    age_sq = age**2

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
    # Spec12 uses: light_care_recent, intensive_care_recent, age, age_sq, education
    # Filter: parent_died_recent == 1
    # (which corresponds to mother_dead == 1 this period)
    # Parameters: age, age_sq, light_care_recent, intensive_care_recent,
    # education, const

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
    expected_inheritance_amount = jnp.exp(ln_inheritance_amount)

    # Expected inheritance = probability * amount
    expected_inheritance = prob_positive_inheritance * expected_inheritance_amount

    # Only return inheritance if mother is dead
    return mother_dead_int * expected_inheritance
