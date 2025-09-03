"""Functions for pre and post estimation setup."""

import pickle
import time
from functools import partial
from typing import Any, Callable, Dict, Optional

import jax
import numpy as np
import optimagic as om
import pandas as pd
import yaml
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.wealth_correction import adjust_observed_wealth

from caregiving.config import BLD, SRC
from caregiving.model.shared import MACHINE_ZERO, RETIREMENT
from caregiving.model.state_space import (
    create_state_space_functions,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.simulate import simulate_scenario
from caregiving.simulation.simulate_moments import (
    simulate_moments_jax,
    simulate_moments_pandas,
)

jax.config.update("jax_enable_x64", True)


def estimate_model(
    model_for_simulation: Dict[str, Any],
    start_params: Dict[str, Any],
    solve_func: callable,
    options: Dict[str, Any],
    algo: str,
    algo_options: Dict[str, Any],
    weighting_method: str = "identity",
    use_cholesky_weights: bool = True,
    relative_deviations: bool = False,
    *,
    path_to_discrete_states: str = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_wealth: str = BLD / "model" / "initial_conditions" / "wealth.csv",
    path_to_empirical_moments: str = BLD / "moments" / "moments_full.csv",
    path_to_empirical_variance: str = BLD / "moments" / "variances_full.csv",
    # path_to_updated_start_params: str = BLD
    # / "model"
    # / "params"
    # / "start_params_model.yaml",
    path_to_lower_bounds: str = SRC
    / "estimation"
    / "start_params_and_bounds"
    / "lower_bounds.yaml",
    path_to_upper_bounds: str = SRC
    / "estimation"
    / "start_params_and_bounds"
    / "upper_bounds.yaml",
    path_to_save_estimation_result: str = BLD / "estimation" / "result.pkl",
    path_to_save_estimation_params: str = BLD / "estimation" / "estimated_params.csv",
    # last_estimate: Optional[Dict[str, Any]] = None,
    select_fixed_params: Optional[Callable[[str, Any], bool]] = None,
    scaling: bool = False,
    scaling_options: Optional[Dict[str, Any]] = None,
    multistart: bool = False,
    multistart_options: Optional[Dict[str, Any]] = None,
    random_seed: bool = False,
    error_handling: str = "continue",
) -> None:
    """Estimate the model based on empirical data and starting parameters."""

    # 1) set up a single RNG if we’re doing truly random draws
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
        # empirical_variances_reg = np.maximum(empirical_variances, 1e-4)
        # weights = np.diag(1 / empirical_variances_reg)
        empirical_variances_reg = empirical_variances
        close_to_zero = empirical_variances_reg < MACHINE_ZERO
        weight_elements = 1 / empirical_variances_reg
        weight_elements[close_to_zero] = 0.0
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

    lower_bounds_all = yaml.safe_load(open(path_to_lower_bounds, "rb"))
    lower_bounds = {name: lower_bounds_all[name] for name in start_params.keys()}

    upper_bounds_all = yaml.safe_load(open(path_to_upper_bounds, "rb"))
    upper_bounds = {name: upper_bounds_all[name] for name in start_params.keys()}

    bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)
    fixed_constraint = om.FixedConstraint(selector=select_fixed_params)

    simulate_moments_given_params = partial(
        simulate_moments,
        solve_func=solve_func,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        model_for_simulation=model_for_simulation,
        options=options,
        fixed_seed=fixed_seed,
        seed_generator=seed_generator,
        pandas=True,
    )

    criterion_func = get_msm_optimization_function(
        simulate_moments=simulate_moments_given_params,
        empirical_moments=empirical_moments,
        weights=weights,
        cholesky=use_cholesky_weights,
        relative_deviations=relative_deviations,
    )

    minimize_kwargs = {
        "fun": criterion_func,
        "params": start_params,
        "algorithm": algo,
        "algo_options": algo_options,
        "bounds": bounds,
        "constraints": fixed_constraint,
        "error_handling": error_handling,
    }

    if select_fixed_params is not None:
        fixed_constraint = om.FixedConstraint(selector=select_fixed_params)
        minimize_kwargs["constraints"] = fixed_constraint

    if scaling:
        # Either use user-supplied dict or fall back to your defaults
        so_opts = scaling_options or {
            "method": "start_values",
            "clipping_value": 0.1,
            "magnitude": 1,
        }
        minimize_kwargs["scaling"] = so_opts

    if multistart:
        # allow custom options or fall back to your defaults
        ms_opts = (
            om.MultistartOptions(**multistart_options)
            if multistart_options is not None
            else om.MultistartOptions(n_samples=100, seed=0, n_cores=4)
        )
        minimize_kwargs["multistart"] = ms_opts

    result = om.minimize(**minimize_kwargs)

    pickle.dump(result, open(path_to_save_estimation_result, "wb"))

    start_params.update(result.params)
    # with open(path_to_save_estimation_params, "w") as yamlfile:
    #     yaml.dump(start_params, yamlfile)
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
    pandas: bool = False,
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

    sim_df = simulate_scenario(
        model=model_for_simulation,
        solution=solution_dict,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options,
        seed=seed,
    )

    if pandas:
        simulated_moments = simulate_moments_pandas(sim_df, options=options)
    else:
        simulated_moments = simulate_moments_jax(sim_df, options=options)

    out = np.asarray(simulated_moments.to_numpy())
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out


def get_msm_optimization_function(
    simulate_moments: callable,
    empirical_moments: np.ndarray,
    weights: np.ndarray,
    cholesky: bool = True,
    relative_deviations: bool = False,
) -> np.ndarray:

    if cholesky:
        chol_weights = np.linalg.cholesky(weights)
    else:
        chol_weights = weights

    criterion = om.mark.least_squares(
        partial(
            msm_criterion,
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

    # if relative_deviations:
    #     # with np.errstate(divide="ignore", invalid="ignore"):
    #     #     rel = deviations / flat_empirical_moments
    #     #     invalid = ~np.isfinite(rel)
    #     #     rel[invalid] = deviations[invalid]
    #     deviations /= flat_empirical_moments

    if relative_deviations:
        np.divide(
            deviations,
            flat_empirical_moments,
            out=deviations,
            where=flat_empirical_moments != 0,
        )

    residuals = deviations @ chol_weights

    return residuals


def select_fixed_params(params):
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


# =====================================================================================
# Preparation for estimation
# =====================================================================================


def load_and_setup_full_model_for_solution(options, path_to_model) -> Dict[str, Any]:
    """Load and setup full model for solution."""

    model_full = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_model,
        sim_model=False,
    )

    return model_full


def load_and_prep_data(data_emp, model, start_params, drop_retirees=True):
    """Load and prepare empirical data to compare with simulated data."""
    # specs = generate_derived_and_data_derived_specs(path_dict)
    # data_decision = pd.read_pickle(path_dict["struct_est_sample"])

    specs = model["options"]["model_params"]
    # We need to filter observations in period 0 because of job offer
    # weighting from last period
    data_emp = data_emp[data_emp["period"] > 0].copy()

    # Also already retired individuals hold no identification
    if drop_retirees:
        data_emp = data_emp[~data_emp["lagged_choice"].isin(RETIREMENT.tolist())]

    data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_emp.loc[:, "age_bin"] = np.floor(data_emp["age"] / 10)
    data_emp.loc[data_emp["age_bin"] > 6, "age_bin"] = 6  # noqa: PLR2004

    age_bin_av_size = data_emp.shape[0] / data_emp["age_bin"].nunique()
    data_emp.loc[:, "age_weights"] = 1.0
    data_emp.loc[:, "age_weights"] = age_bin_av_size / data_emp.groupby("age_bin")[
        "age_weights"
    ].transform("sum")

    # Transform experience
    max_init_exp = specs["max_exp_diffs_per_period"][data_emp["period"].values]
    exp_denominator = data_emp["period"].values + max_init_exp
    data_emp["experience"] = data_emp["experience"] / exp_denominator

    # We can adjust wealth outside, as it does not depend on estimated parameters
    # (only on interest rate)
    # Now transform for dcegm
    states_dict = {
        name: data_emp[name].values
        for name in model["model_structure"]["discrete_states_names"]
        if name not in ("mother_health", "care_demand", "care_supply")
    }
    states_dict["care_demand"] = np.zeros_like(data_emp["wealth"])
    states_dict["experience"] = data_emp["experience"].values
    states_dict["wealth"] = data_emp["wealth"].values / specs["wealth_unit"]

    adjusted_wealth = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=start_params,
        model=model,
    )
    data_emp.loc[:, "adjusted_wealth"] = adjusted_wealth
    states_dict["wealth"] = data_emp["adjusted_wealth"].values

    return data_emp, states_dict
