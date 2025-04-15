"""Simulate moments of the model ."""

import pickle
from pathlib import Path

import jax
import pandas as pd
from numpy.testing import assert_array_equal as aae

from caregiving.config import BLD
from caregiving.simulation.simulate_moments import (
    simulate_moments_jax,
    simulate_moments_pandas,
    simulate_moments_pandas_like_empirical,
)

jax.config.update("jax_enable_x64", True)


def task_simulate_moments(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_simulated_data: Path = BLD / "solve_and_simulate" / "simulated_data.pkl",
) -> None:

    options = pickle.load(path_to_options.open("rb"))
    df_sim = pd.read_pickle(path_to_simulated_data)

    # sim_moms_jax = simulate_moments_jax(df_sim, options=options)
    sim_moms_pandas = simulate_moments_pandas(df_sim, options=options)
    sim_moms_pandas_like_empirical = simulate_moments_pandas_like_empirical(
        df_sim, options=options
    )

    # aae(sim_moms_jax, sim_moms_pandas)
    aae(sim_moms_pandas_like_empirical, sim_moms_pandas)
