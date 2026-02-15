"""Functions for evaluating the MSM criterion function."""

import pickle
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.estimation.estimation_setup import (
    get_msm_optimization_function,
    simulate_moments_with_unobserved_type_shares,
)
from caregiving.model.shared import MACHINE_ZERO


def evaluate_criterion(
    params: Dict[str, Any],
    model_class: Dict[str, Any],
    model_specs: Dict[str, Any],
    simulate_scenario_func: Callable,
    simulate_moments_func: Callable,
    *,
    path_to_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_empirical_moments: Path = BLD / "moments" / "moments_full.csv",
    path_to_empirical_variance: Path = BLD / "moments" / "variances_full.csv",
    weighting_method: str = "identity",
    use_cholesky_weights: bool = True,
    relative_deviations: bool = False,
    least_squares: bool = True,
    random_seed: bool = False,
) -> np.ndarray:
    """Evaluate the MSM criterion function and return residuals.

    This function sets up the criterion function exactly as in
    estimate_model_with_unobserved_type_shares (lines 280-359) and evaluates
    it with the given parameters.

    Parameters
    ----------
    params : dict
        Parameter dictionary to evaluate
    model_class : dict
        Model class instance
    model_specs : dict
        Model specifications dictionary
    simulate_scenario_func : callable
        Function to simulate the model scenario given parameters
    simulate_moments_func : callable
        Function to compute moments from simulated data
    path_to_initial_states : Path, optional
        Path to initial states pickle file
    path_to_empirical_moments : Path, optional
        Path to empirical moments CSV file
    path_to_empirical_variance : Path, optional
        Path to empirical variance CSV file
    weighting_method : str, default "identity"
        Weighting method: "identity" or "diagonal"
    use_cholesky_weights : bool, default True
        Whether to use Cholesky decomposition of weights
    relative_deviations : bool, default False
        Whether to use relative deviations
    least_squares : bool, default True
        Whether to use least squares (returns residuals array).
        If False, returns scalar criterion value.
    random_seed : bool, default False
        If True, uses random seed generator. If False, uses fixed seed from model_specs.

    Returns
    -------
    np.ndarray
        Residuals array when least_squares=True, or scalar when least_squares=False
    """
    # Set up seed generator (exactly as in lines 280-285)
    if random_seed:
        seed_generator = np.random.default_rng()
        fixed_seed = None
    else:
        seed_generator = None
        fixed_seed = model_specs["seed"]

    # Load initial states (exactly as in line 287)
    initial_states = pickle.load(path_to_initial_states.open("rb"))

    # Load empirical moments and variances (exactly as in lines 289-294)
    empirical_moments = np.array(
        pd.read_csv(path_to_empirical_moments, index_col=0).squeeze()
    )
    empirical_variances = np.array(
        pd.read_csv(path_to_empirical_variance, index_col=0).squeeze()
    )

    # Compute weights based on weighting_method (exactly as in lines 302-324)
    if weighting_method == "identity":
        weights = np.identity(empirical_moments.shape[0])
    elif weighting_method == "diagonal":
        # Use robust diagonal weights to avoid numerical issues
        empirical_variances_reg = empirical_variances.copy()
        close_to_zero = empirical_variances_reg < MACHINE_ZERO
        close_to_zero = np.isnan(empirical_variances_reg) | close_to_zero
        # Replace zero variances with a small positive value to avoid division by zero
        weight_elements = 1 / empirical_variances_reg
        weight_elements[close_to_zero] = 0.0
        weights_sum = np.sum(weight_elements)
        weight_elements = np.sqrt(weight_elements / weights_sum)
        weights = np.diag(weight_elements)
    else:
        raise ValueError(f"Unknown weighting method: {weighting_method}")

    # Set up simulate_moments_given_params (exactly as in lines 340-349)
    simulate_moments_given_params = partial(
        simulate_moments_with_unobserved_type_shares,
        initial_states=initial_states,
        model_class=model_class,
        model_specs=model_specs,
        fixed_seed=fixed_seed,
        seed_generator=seed_generator,
        simulate_scenario_func=simulate_scenario_func,
        simulate_moments_func=simulate_moments_func,
    )

    # Set up criterion function (exactly as in lines 351-359)
    criterion_func = get_msm_optimization_function(
        simulate_moments=simulate_moments_given_params,
        empirical_moments=empirical_moments,
        weights=weights,
        cholesky=use_cholesky_weights,
        relative_deviations=relative_deviations,
        least_squares=least_squares,
    )

    # Call criterion function with parameters and return residuals
    residuals = criterion_func(params)

    return residuals
