"""Simulate moments of the model ."""

import pickle
from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal as aae
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.shared import (
    FULL_TIME_CHOICES,
    NOT_WORKING,
    PART_TIME,
    WORK,
)
from caregiving.simulation.simulate_moments import (
    plot_model_fit_labor_moments_by_education_pandas_jax,
    plot_model_fit_labor_moments_pandas,
    plot_model_fit_labor_moments_pandas_by_education,
    plot_transition_shares_by_age_bins,
    simulate_moments_jax,
    simulate_moments_pandas,
)
from caregiving.specs.task_write_specs import read_and_derive_specs

jax.config.update("jax_enable_x64", True)


def task_simulate_moments(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_empirical_moments: Path = BLD / "moments" / "moments_full.csv",
    path_to_simulated_data: Path = BLD / "solve_and_simulate" / "simulated_data.pkl",
    path_to_save_labor_shares_pandas: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "simulated_labor_shares_pandas.png",
    path_to_save_labor_shares_jax: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "simulated_labor_shares_jax.png",
    path_to_save_transitions_pandas: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "simulated_work_transitions_pandas.png",
) -> None:

    specs = read_and_derive_specs(path_to_specs)

    options = pickle.load(path_to_options.open("rb"))
    df_sim = pd.read_pickle(path_to_simulated_data)

    emp_moms = pd.read_csv(path_to_empirical_moments, index_col=[0]).squeeze("columns")

    sim_moms_pandas = simulate_moments_pandas(df_sim, options=options)
    sim_moms_jax = simulate_moments_jax(df_sim, options=options)

    aae(sim_moms_jax, sim_moms_pandas)
    assert np.equal(emp_moms.shape, sim_moms_pandas.shape)

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

    plot_model_fit_labor_moments_pandas_by_education(
        moms_emp=emp_moms,
        moms_sim=sim_moms_pandas,
        specs=specs,
        path_to_save_plot=path_to_save_labor_shares_pandas,
    )

    plot_model_fit_labor_moments_by_education_pandas_jax(
        moms_emp=emp_moms,
        moms_sim=sim_moms_jax,
        specs=specs,
        path_to_save_plot=path_to_save_labor_shares_jax,
    )
