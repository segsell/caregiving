"""Formal care costs precomputation functions."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd


def read_in_formal_care_costs_by_age(
    path_to_params: Path,
    specs: dict,
) -> jnp.ndarray:
    """Predict average formal care costs by age (period) using pooled parameters.

    Uses the estimated OLS parameters from pooled regression (no education/sex
    distinction) to predict formal care costs for each period (age).

    Parameters
    ----------
    path_to_params : Path
        Path to CSV file with formal care costs parameters (pooled).
        Expected columns: const, age, age_sq
        Expected index: "coefficient"
    specs : dict
        Master spec-dictionary containing:
        - start_age: Starting age
        - end_age: Ending age

    Returns
    -------
    jax.numpy.ndarray
        1D array of shape (n_periods,) with predicted formal care costs
        for each period (indexed by period, where period = age - start_age)
    """
    # Read parameters
    params_df = pd.read_csv(path_to_params, index_col=0)

    # Extract parameters
    const = params_df.loc["coefficient", "const"]
    age_coef = params_df.loc["coefficient", "age"]
    age_sq_coef = params_df.loc["coefficient", "age_sq"]

    # Create age range and initialize array
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    n_periods = end_age - start_age + 1

    # Initialize array with zeros (following pattern from family_specs)
    formal_care_costs = np.zeros(n_periods)

    # Loop over periods and predict formal care costs for each age
    for period in range(n_periods):
        age = period + start_age
        age_sq = age**2

        # Predict: const + age_coef * age + age_sq_coef * age^2
        predicted_cost = const + age_coef * age + age_sq_coef * age_sq

        # Ensure non-negative (formal care costs should be >= 0)
        formal_care_costs[period] = np.maximum(0, predicted_cost)

    return jnp.asarray(formal_care_costs)
