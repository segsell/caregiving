"""Functions for pre and post estimation setup."""

import pickle
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import jax
import numpy as np
import optimagic as om
import pandas as pd
import yaml

from caregiving.config import BLD, SRC
from caregiving.model.shared import MACHINE_ZERO

jax.config.update("jax_enable_x64", True)


def estimate_model(
    model_for_simulation: Dict[str, Any],
    start_params: Dict[str, Any],
    solve_func: callable,
    options: Dict[str, Any],
    algo: str,
    algo_options: Dict[str, Any],
    lower_bounds: Dict[str, float],
    upper_bounds: Dict[str, float],
    simulate_scenario_func: callable,
    simulate_moments_func: callable,
    weighting_method: str = "identity",
    use_cholesky_weights: bool = True,
    relative_deviations: bool = False,
    least_squares: bool = True,
    *,
    path_to_discrete_states: str = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_wealth: str = BLD / "model" / "initial_conditions" / "wealth.csv",
    path_to_empirical_moments: str = BLD / "moments" / "moments_full.csv",
    path_to_empirical_variance: str = BLD / "moments" / "variances_full.csv",
    path_to_save_estimation_result: str = BLD / "estimation" / "result.pkl",
    path_to_save_estimation_params: str = BLD / "estimation" / "estimated_params.csv",
    select_fixed_params: Optional[Callable[[str, Any], bool]] = None,
    other_constraint: Optional[om.constraints.Constraint] = None,
    scaling: bool = False,
    scaling_options: Optional[Dict[str, Any]] = None,
    multistart: bool = False,
    multistart_options: Optional[Dict[str, Any]] = None,
    random_seed: bool = False,
    error_handling: str = "continue",
) -> None:
    """Estimate the model based on empirical data and starting parameters.

    Parameters:
    -----------
    simulate_scenario_func : callable
        Function to simulate the model scenario given parameters.
    simulate_moments_func : callable
        Function to compute moments from simulated data.
    least_squares : bool, default True
        If True, uses least squares optimization (returns residuals array).
        If False, uses scalar optimization (returns a single scalar value from
        the criterion function). When False, the criterion function returns
        the sum of squared residuals as a scalar value, which is suitable for
        scalar optimizers like 'scipy_lbfgsb', 'scipy_neldermead', etc.
    """

    # 1) set up a single RNG if we're doing truly random draws
    if random_seed:
        seed_generator = np.random.default_rng()  # draws come from system entropy
        fixed_seed = None
    else:
        seed_generator = None
        fixed_seed = options["model_params"]["seed"]  # same seed every call

    initial_states = pickle.load(path_to_discrete_states.open("rb"))
    wealth_agents = np.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

    # Load empirical data
    empirical_moments = np.array(
        pd.read_csv(path_to_empirical_moments, index_col=0).squeeze()
    )
    empirical_variances = np.array(
        pd.read_csv(path_to_empirical_variance, index_col=0).squeeze()
    )

    if weighting_method == "identity":
        weights = np.identity(empirical_moments.shape[0])
    elif weighting_method == "diagonal":
        # Use robust diagonal weights to avoid numerical issues
        empirical_variances_reg = empirical_variances.copy()
        close_to_zero = empirical_variances_reg < MACHINE_ZERO
        # Replace zero variances with a small positive value to avoid division by zero
        empirical_variances_reg[close_to_zero] = 1e-6
        weight_elements = 1 / empirical_variances_reg
        weight_elements = np.sqrt(weight_elements)
        weights = np.diag(weight_elements)
    else:
        raise ValueError(f"Unknown weighting method: {weighting_method}")

    # if method == "optimal":
    #     array_weights = robust_inverse(_internal_cov)
    # elif method == "diagonal":
    #     diagonal_values = 1 / np.clip(np.diagonal(_internal_cov), clip_value, np.inf)
    #     array_weights = np.diag(diagonal_values)
    # elif method == "identity":
    #     array_weights = np.identity(_internal_cov.shape[0])

    # start_params_all = yaml.safe_load(path_to_updated_start_params.open("rb"))

    # # Assign start params from before
    # if last_estimate is not None:
    #     for key in last_estimate.keys():
    #         if key in ("sigma", "interest_rate", "beta"):
    #             continue
    #         try:
    #             print(
    #                 f"Start params value of {key} was {start_params_all[key]} and is"
    #                 f"replaced by {last_estimate[key]}"
    #             )
    #         except KeyError as err:
    #             raise ValueError(f"Key {key} not found in start params.") from err
    #         start_params_all[key] = last_estimate[key]

    # start_params = {name: start_params_all[name] for name in start_params.keys()}

    # lower_bounds_all = yaml.safe_load(open(path_to_lower_bounds, "rb"))
    # lower_bounds = {name: lower_bounds_all[name] for name in start_params.keys()}

    # upper_bounds_all = yaml.safe_load(open(path_to_upper_bounds, "rb"))
    # upper_bounds = {name: upper_bounds_all[name] for name in start_params.keys()}

    # # --- Build constraints list (NEW) ------------------------------------------
    # constraints_list: List[Any] = []
    # if select_fixed_params is not None:
    #     constraints_list.append(om.FixedConstraint(selector=select_fixed_params))

    # if other_constraint is not None:
    #     if isinstance(other_constraint, (list, tuple)):
    #         constraints_list.extend(other_constraint)
    #     else:
    #         constraints_list.append(other_constraint)
    # # --------------------------------------------------------------------------

    # --- Adjust bounds if other_constraint is present --------------------------
    # constraints_list: List[Any] = []
    # if select_fixed_params is not None:
    #     constraints_list.append(om.FixedConstraint(selector=select_fixed_params))

    # if other_constraint is not None:
    #     if isinstance(other_constraint, (list, tuple)):
    #         constraints_to_apply = other_constraint
    #     else:
    #         constraints_to_apply = [other_constraint]

    #     # set bounds of selected params to (-inf, inf)
    #     for constr in constraints_to_apply:
    #         if hasattr(constr, "selector") and callable(constr.selector):
    #             selected = constr.selector(start_params)
    #             for pname in selected.keys():
    #                 lower_bounds[pname] = -np.inf
    #                 upper_bounds[pname] = np.inf

    #     constraints_list.extend(constraints_to_apply)

    constraints_list, lower_bounds, upper_bounds = (
        combine_constraints_and_update_bounds(
            select_fixed_params=select_fixed_params,
            other_constraint=other_constraint,
            start_params=start_params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
    )
    # --------------------------------------------------------------------------

    bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)

    simulate_moments_given_params = partial(
        simulate_moments,
        solve_func=solve_func,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        model_for_simulation=model_for_simulation,
        options=options,
        fixed_seed=fixed_seed,
        seed_generator=seed_generator,
        simulate_scenario_func=simulate_scenario_func,
        simulate_moments_func=simulate_moments_func,
    )

    criterion_func = get_msm_optimization_function(
        simulate_moments=simulate_moments_given_params,
        empirical_moments=empirical_moments,
        weights=weights,
        cholesky=use_cholesky_weights,
        relative_deviations=relative_deviations,
        least_squares=least_squares,
    )

    minimize_kwargs = {
        "fun": criterion_func,
        "params": start_params,
        "algorithm": algo,
        "algo_options": algo_options,
        "bounds": bounds,
        "error_handling": error_handling,
    }

    if constraints_list:
        minimize_kwargs["constraints"] = constraints_list

    if scaling:
        so_opts = scaling_options or {
            "method": "start_values",
            "clipping_value": 0.1,
            "magnitude": 1,
        }
        minimize_kwargs["scaling"] = so_opts

    if multistart:
        ms_opts = (
            om.MultistartOptions(**multistart_options)
            if multistart_options is not None
            else om.MultistartOptions(n_samples=100, seed=0, n_cores=4)
        )
        minimize_kwargs["multistart"] = ms_opts

    result = om.minimize(**minimize_kwargs)

    pickle.dump(result, open(path_to_save_estimation_result, "wb"))

    start_params.update(result.params)
    start_params_series = pd.Series(start_params, name="value")
    start_params_series.to_csv(path_to_save_estimation_params, header=True)

    return result


# =====================================================================================
# Criterion function
# =====================================================================================


def simulate_moments(
    params: np.ndarray,
    solve_func: callable,
    initial_states: Dict[str, Any],
    wealth_agents: np.ndarray,
    model_for_simulation: Dict[str, Any],
    options: Dict[str, Any],
    fixed_seed: Optional[int],
    seed_generator: Optional[np.random.Generator],
    simulate_scenario_func: callable,
    simulate_moments_func: callable,
):
    """Solve the model and simulate moments.

    - If seed_generator is not None, we draw a brand-new random seed from it.
    - Otherwise we just reuse fixed_seed every time.

    """

    if seed_generator is not None:
        # extremely fast: draws a uint64 from PCG64
        # seed = int(seed_generator.integers(0, 2**63, dtype=np.uint64))
        # seed = int(seed_generator.integers(0, 2**16, dtype=np.uint16))
        seed = int(seed_generator.integers(0, 2**32, dtype=np.uint32))
    else:
        seed = fixed_seed

    solution_dict = {}
    (
        solution_dict["value"],
        solution_dict["policy"],
        solution_dict["endog_grid"],
    ) = solve_func(params)

    sim_df = simulate_scenario_func(
        model=model_for_simulation,
        solution=solution_dict,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options,
        seed=seed,
    )

    simulated_moments = simulate_moments_func(sim_df, options=options)

    out = np.asarray(simulated_moments.to_numpy())
    # More aggressive NaN handling to prevent optimizer issues
    out = np.nan_to_num(out, nan=1e6, posinf=1e6, neginf=1e6)

    return out


def msm_criterion_scalar(
    params: np.ndarray,
    simulate_moments: callable,
    flat_empirical_moments: np.ndarray,
    chol_weights: np.ndarray,
    relative_deviations: bool = False,
) -> float:
    """Compute the MSM scalar criterion based on simulated and empirical moments."""

    simulated_flat = simulate_moments(params)

    deviations = simulated_flat - flat_empirical_moments

    if relative_deviations:
        np.divide(
            deviations,
            flat_empirical_moments,
            out=deviations,
            where=flat_empirical_moments != 0,
        )

    # Return the weighted sum of squared deviations as a scalar
    # This is equivalent to deviations @ weights @ deviations (Mahalanobis distance)
    scalar_criterion = deviations @ chol_weights @ deviations

    return scalar_criterion


def get_msm_optimization_function(
    simulate_moments: callable,
    empirical_moments: np.ndarray,
    weights: np.ndarray,
    cholesky: bool = True,
    relative_deviations: bool = False,
    least_squares: bool = True,
) -> np.ndarray:

    if cholesky:
        chol_weights = np.linalg.cholesky(weights)
    else:
        chol_weights = weights

    if least_squares:
        criterion = om.mark.least_squares(
            partial(
                msm_criterion,
                simulate_moments=simulate_moments,
                flat_empirical_moments=empirical_moments,
                chol_weights=chol_weights,
                relative_deviations=relative_deviations,
            )
        )
    else:
        criterion = om.mark.scalar(
            partial(
                msm_criterion_scalar,
                simulate_moments=simulate_moments,
                flat_empirical_moments=empirical_moments,
                chol_weights=chol_weights,
                relative_deviations=relative_deviations,
            )
        )

    return criterion


def msm_criterion(
    params: np.ndarray,
    simulate_moments: callable,
    flat_empirical_moments: np.ndarray,
    chol_weights: np.ndarray,
    relative_deviations: bool = False,
) -> np.ndarray:
    """Compute the MSM residuals based on simulated and empirical moments."""

    simulated_flat = simulate_moments(params)

    deviations = simulated_flat - flat_empirical_moments

    if relative_deviations:
        np.divide(
            deviations,
            flat_empirical_moments,
            out=deviations,
            where=flat_empirical_moments != 0,
        )

    residuals = deviations @ chol_weights

    return residuals


def combine_constraints_and_update_bounds(
    select_fixed_params, other_constraint, start_params, lower_bounds, upper_bounds
):
    """Select constraints for the optimization."""
    constraints_list: List[Any] = []

    if select_fixed_params is not None:
        constraints_list.append(om.FixedConstraint(selector=select_fixed_params))

    if other_constraint is not None:
        if isinstance(other_constraint, (list, tuple)):
            constraints_to_apply = other_constraint
        else:
            constraints_to_apply = [other_constraint]

        # set bounds of selected params to (-inf, inf)
        for constr in constraints_to_apply:
            if hasattr(constr, "selector") and callable(constr.selector):
                selected = constr.selector(start_params)
                for pname in selected.keys():
                    lower_bounds[pname] = -np.inf
                    upper_bounds[pname] = np.inf

        constraints_list.extend(constraints_to_apply)

    return constraints_list, lower_bounds, upper_bounds


# =====================================================================================
# Example
# =====================================================================================


def select_fixed_params_example(params):
    """Select fixed parameters for the optimization."""

    fixed_params = {
        "sigma": params["sigma"],
        "interest_rate": params["interest_rate"],
        "beta": params["beta"],
        "rho": params["rho"],
    }

    job_finding_params = {
        key: val for key, val in params.items() if key.startswith("job_finding")
    }
    fixed_params.update(job_finding_params)

    return fixed_params
