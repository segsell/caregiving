import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
import yaml
from pytask import Product

from caregiving.config import BLD
from caregiving.estimation.estimation_setup import (
    load_and_prep_data,
    load_and_setup_full_model_for_solution,
)
from caregiving.model.shared import SEX
from caregiving.simulation.plot_model_fit import plot_average_wealth


def task_plot_model_fit(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_empirical_data: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_simulated_data: Path = BLD / "solve_and_simulate" / "simulated_data.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "average_wealth.png",
) -> None:
    """Plot model fit between empirical and simulated data."""

    options = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))

    model_full = load_and_setup_full_model_for_solution(
        options, path_to_model=path_to_solution_model
    )

    df_emp = pd.read_csv(path_to_empirical_data, index_col=[0])
    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = 1

    df_emp_prep, _states_dict = load_and_prep_data(
        data_emp=df_emp,
        model=model_full,
        start_params=params,
        drop_retirees=False,
    )

    specs = model_full["options"]["model_params"]
    plot_average_wealth(df_emp_prep, df_sim, specs, path_to_save_plot)
