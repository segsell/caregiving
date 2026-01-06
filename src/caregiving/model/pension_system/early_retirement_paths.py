import jax
import numpy as np
from jax import numpy as jnp


def calc_early_retirement_pension_points(
    total_pension_points,
    retirement_age_difference,
    experience_years,
    sex,
    model_specs,
):
    """Calculate the pension points for early retirement for different paths. We implemented
    three ways to retire before the SRA:

        - Pension for long insured
        - Pension for very long insured
        - Disability pension

    We also check, which pension is best, given multiple are possible. There we use the
    following:
        - This function is only used, if the person freshly retires.
        - Disability pension is always better than pension for long insured, as the pension
          points fill up to the same level as if you would have worked until SRA and your
          deductions are limited to 3 years.
        - Pension for very long insured is always better than the pension for long insured.
          So we only check if very long insured path is better than disability pension.

    """

    # Check if the individual is eligible for very long insured pension
    very_long_insured_bool = check_very_long_insured(
        retirement_age_difference=retirement_age_difference,
        experience_years=experience_years,
        sex=sex,
        model_specs=model_specs,
    )

    # Choose deduction factor according to information state
    early_retirement_penalty = model_specs["early_retirement_penalty"]

    # Create early retirement factor
    early_retirement_factor = (
        1 - early_retirement_penalty * retirement_age_difference
    ).clip(min=0.0, max=1.0)

    # Early retirement pension points
    early_retirement_pension_points = early_retirement_factor * total_pension_points

    # Now first select by very long insured.
    pension_points_early_retired = jax.lax.select(
        very_long_insured_bool,
        on_true=total_pension_points,
        on_false=early_retirement_pension_points,
    )

    return pension_points_early_retired


def check_very_long_insured(
    retirement_age_difference, experience_years, sex, model_specs
):
    """Test if the individual qualifies for pension for very long insured
    (Rente besonders für langjährig Versicherte)."""

    policy_state_value = model_specs["min_SRA"]
    # Get diff of SRA value to 67
    sra_diff_67 = jnp.maximum(67 - policy_state_value, 0)

    experience_threshold = model_specs["experience_threshold_very_long_insured"][sex]
    enough_years = experience_years >= experience_threshold
    close_enough_to_SRA = retirement_age_difference <= 2

    return enough_years & close_enough_to_SRA


def retirement_age_long_insured(SRA, model_specs):
    """Everyone can retire 4 years before the SRA but must be at least at 63 y/o.
    That means that we assume 1) everyone qualifies for "Rente für langjährig Versicherte" and 2) that
    the threshhold for "Rente für langjährig Versicherte" moves with the SRA. "Rente für besonders langjährig
    Versicherte" only differs with respect to deductions. Not with respect to entry age. We introduce the
    lower bound of 63 as this is the current law, even for individuals with SRA below 67.
    """
    # if model_specs["ERA_moves"]:
    #     return jnp.maximum(
    #         SRA - model_specs["years_before_SRA_long_insured"],
    #         model_specs["min_long_insured_age"],
    #     )
    # else:
    return model_specs["min_long_insured_age"]
