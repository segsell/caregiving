"""Simulate moments of the model ."""

import pickle
from pathlib import Path
from typing import Annotated

import jax
import pandas as pd
from numpy.testing import assert_array_equal as aae
from pytask import Product

from caregiving.config import BLD
from caregiving.simulation.simulate_moments import (
    plot_model_fit_labor_moments_pandas,
    plot_model_fit_labor_moments_pandas_jax,
    simulate_moments_jax,
    simulate_moments_pandas,
)

jax.config.update("jax_enable_x64", True)


def task_simulate_moments(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_empirical_moments: Path = BLD / "moments" / "soep_moments.csv",
    path_to_simulated_data: Path = BLD / "solve_and_simulate" / "simulated_data.pkl",
    path_to_save_pandas_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "simulated_moments_pandas.png",
    path_to_save_jax_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "simulated_moments_jax.png",
) -> None:

    options = pickle.load(path_to_options.open("rb"))
    df_sim = pd.read_pickle(path_to_simulated_data)

    emp_moms = pd.read_csv(path_to_empirical_moments, index_col=[0]).squeeze("columns")

    sim_moms_jax = simulate_moments_jax(df_sim, options=options)
    sim_moms_pandas = simulate_moments_pandas(df_sim, options=options)

    aae(sim_moms_jax, sim_moms_pandas)

    plot_model_fit_labor_moments_pandas(
        moms_emp=emp_moms,
        moms_sim=sim_moms_pandas,
        path_to_save_plot=path_to_save_pandas_plot,
    )

    # plot_model_fit_labor_moments_pandas_jax(
    #     moms_emp=emp_moms,
    #     moms_sim=sim_moms_jax,
    #     # path_to_save_plot=path_to_save_jax_plot,
    # )
