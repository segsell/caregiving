"""Simulate moments of the no care demand model using estimated parameters."""

import pickle
from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.simulation.simulate_moments import (
    plot_model_fit_labor_moments_pandas_by_education,
)
from caregiving.simulation.simulate_moments_no_care_demand import (
    simulate_moments_pandas_no_care_demand,
)

jax.config.update("jax_enable_x64", True)


@pytask.mark.no_care_demand_model
@pytask.mark.sim_estimated_params_no_care_demand
def task_simulate_moments_estimated_params_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_empirical_moments: Path = BLD
    / "moments"
    / "soep_moments_no_care_demand.csv",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_save_pandas_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "simulated_moments_pandas_no_care_demand_estimated_params.csv",
    path_to_save_labor_shares_pandas: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_no_care_demand"
    / "simulated_labor_shares_pandas_estimated_params.png",
) -> None:
    """Simulate moments using estimated parameters for no care demand model."""

    specs = pickle.load(path_to_specs.open("rb"))
    sim_df = pd.read_pickle(path_to_simulated_data)
    emp_moms = pd.read_csv(path_to_empirical_moments, index_col=[0]).squeeze("columns")

    # Keep only columns needed for simulate_moments_pandas_no_care_demand
    required_cols = [
        "age",
        "education",
        "choice",
        "assets_begin_of_period",
    ]
    sim_df = sim_df[required_cols].copy()

    # Compute simulated moments (pandas version only for no care demand)
    sim_moms_pandas = simulate_moments_pandas_no_care_demand(sim_df, model_specs=specs)

    # Save moments
    sim_moms_pandas.to_csv(path_to_save_pandas_moments)
    # Validate shapes match
    assert np.equal(emp_moms.shape, sim_moms_pandas.shape)

    # Plot labor moments by education (no caregivers in no care demand model)
    plot_model_fit_labor_moments_pandas_by_education(
        moms_emp=emp_moms,
        moms_sim=sim_moms_pandas,
        specs=specs,
        path_to_save_plot=path_to_save_labor_shares_pandas,
        include_caregivers=False,
    )
