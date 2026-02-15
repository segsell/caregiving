"""Simulate moments of the model using estimated parameters."""

import pickle
from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import pandas as pd
import pytask
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal as aaae
from pytask import Product

from caregiving.config import BLD
from caregiving.simulation.simulate_moments import (
    plot_model_fit_labor_moments_by_education_pandas_jax,
    plot_model_fit_labor_moments_pandas_by_education,
    simulate_moments_jax,
    simulate_moments_pandas,
)
from caregiving.simulation.simulate_moments_restricted import (
    simulate_moments_pandas_mean_wealth,
)

jax.config.update("jax_enable_x64", True)


@pytask.mark.sim_estimated_params
def task_simulate_moments_estimated_params(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_empirical_moments: Path = BLD / "moments" / "moments_full.csv",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_pandas_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "simulated_moments_pandas_estimated_params.csv",
    path_to_save_jax_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "simulated_moments_jax_estimated_params.csv",
    path_to_save_labor_shares_pandas: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "simulated_labor_shares_pandas_estimated_params.png",
    path_to_save_labor_shares_jax: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "simulated_labor_shares_jax_estimated_params.png",
    path_to_save_labor_shares_with_caregivers_pandas: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "simulated_labor_shares_with_caregivers_pandas_estimated_params.png",
    path_to_save_labor_shares_with_caregivers_jax: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "simulated_labor_shares_with_caregivers_jax_estimated_params.png",
    # path_to_save_transitions_pandas: Annotated[Path, Product] = BLD
    # / "plots"
    # / "model_fit"
    # / "simulated_work_transitions_pandas_estimated_params.png",
) -> None:
    """Simulate moments using estimated parameters model specification."""

    specs = pickle.load(path_to_specs.open("rb"))
    sim_df = pd.read_pickle(path_to_simulated_data)
    emp_moms = pd.read_csv(path_to_empirical_moments, index_col=[0]).squeeze("columns")

    # Keep only columns needed for simulate_moments_pandas
    required_cols = [
        "care_demand",
        "health",
        "age",
        "education",
        "choice",
        "lagged_choice",
        "assets_begin_of_period",
    ]
    sim_df = sim_df[required_cols].copy()

    sim_moms_pandas = simulate_moments_pandas(sim_df, model_specs=specs)
    sim_moms_pandas_mean_wealth = simulate_moments_pandas_mean_wealth(
        sim_df, model_specs=specs
    )
    sim_moms_jax = simulate_moments_jax(sim_df, model_specs=specs)

    # Save moments
    sim_moms_pandas.to_csv(path_to_save_pandas_moments)
    np.savetxt(path_to_save_jax_moments, sim_moms_jax, delimiter=",")

    # Convert to numpy arrays for comparison
    sim_moms_jax_arr = np.asarray(sim_moms_jax[25:])
    sim_moms_pandas_arr = np.asarray(sim_moms_pandas[25:])

    # Round very small values near zero to exactly zero to handle floating point
    # precision. This is necessary because JAX and pandas may represent zeros
    # slightly differently
    tolerance = 1e-10
    sim_moms_jax_arr = np.where(
        np.abs(sim_moms_jax_arr) < tolerance, 0.0, sim_moms_jax_arr
    )
    sim_moms_pandas_arr = np.where(
        np.abs(sim_moms_pandas_arr) < tolerance, 0.0, sim_moms_pandas_arr
    )

    # Use assert_allclose with both absolute and relative tolerance
    # atol handles zeros and very small values, rtol handles larger values
    assert_allclose(
        sim_moms_jax_arr,
        sim_moms_pandas_arr,
        atol=1e-2,  # Absolute tolerance: 0.01
        rtol=1e-2,  # Relative tolerance: 1%
    )
    assert np.equal(emp_moms.shape, sim_moms_pandas.shape)
    assert np.equal(emp_moms.shape, sim_moms_jax.shape)
    assert np.equal(emp_moms.shape, sim_moms_pandas_mean_wealth.shape)

    # Non-wealth moments must match between median and mean-wealth versions
    wealth_prefixes = (
        "median_assets_begin_of_period_",
        "mean_assets_begin_of_period_",
    )
    non_wealth_keys = [
        k for k in sim_moms_pandas.index if not k.startswith(wealth_prefixes)
    ]
    aaae(
        sim_moms_pandas.loc[non_wealth_keys].values,
        sim_moms_pandas_mean_wealth.loc[non_wealth_keys].values,
        decimal=12,
    )

    # states = {
    #     "not_working": NOT_WORKING,
    #     "part_time": PART_TIME,
    #     "full_time": FULL_TIME,
    # }
    # state_labels = {
    #     "not_working": "Not Working",
    #     "part_time": "Part-time",
    #     "full_time": "Full-time",
    # }
    # states = {
    #     "not_working": NOT_WORKING,
    #     "working": WORK,
    # }
    # state_labels = {
    #     "not_working": "Not Working",
    #     "working": "Work",
    # }
    # plot_transition_shares_by_age_bins(
    #     moms_emp=emp_moms,
    #     moms_sim=sim_moms_pandas,
    #     specs=specs,
    #     states=states,
    #     state_labels=state_labels,
    #     path_to_save_plot=path_to_save_transitions_pandas,
    # )

    # Plot general labor moments (non-caregivers only)
    plot_model_fit_labor_moments_pandas_by_education(
        moms_emp=emp_moms,
        moms_sim=sim_moms_pandas,
        specs=specs,
        path_to_save_plot=path_to_save_labor_shares_pandas,
        include_caregivers=False,
    )
    plot_model_fit_labor_moments_by_education_pandas_jax(
        moms_emp=emp_moms,
        moms_sim=sim_moms_jax,
        specs=specs,
        path_to_save_plot=path_to_save_labor_shares_jax,
        include_caregivers=False,
    )

    # Plot labor moments including caregivers
    plot_model_fit_labor_moments_pandas_by_education(
        moms_emp=emp_moms,
        moms_sim=sim_moms_pandas,
        specs=specs,
        path_to_save_plot=path_to_save_labor_shares_with_caregivers_pandas,
        include_caregivers=True,
    )
    plot_model_fit_labor_moments_by_education_pandas_jax(
        moms_emp=emp_moms,
        moms_sim=sim_moms_jax,
        specs=specs,
        path_to_save_plot=path_to_save_labor_shares_with_caregivers_jax,
        include_caregivers=True,
    )
