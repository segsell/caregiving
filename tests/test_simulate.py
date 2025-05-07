"""Test the simulation function."""

import pickle
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import pandas as pd
import pytest
import yaml
from pytask import Product

from caregiving.config import BLD, TESTS
from caregiving.model.shared import DEAD
from caregiving.model.state_space import (
    create_state_space_functions,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.simulate import simulate_scenario
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.solve import get_solve_func_for_model

jax.config.update("jax_enable_x64", True)


@pytest.mark.skip()
def test_solve_and_simulate(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
    path_to_save_solution: Annotated[Path, Product] = TESTS
    / "resources"
    / "solution.pkl",
):

    options = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))

    # 1) Solve
    model_full = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_solution_model,
        sim_model=False,
    )

    solution_dict = {}
    (
        solution_dict["value"],
        solution_dict["policy"],
        solution_dict["endog_grid"],
    ) = get_solve_func_for_model(model_full)(params)
    # pickle.dump(solution_dict, path_to_save_solution.open("wb"))

    # 2) Simulate
    initial_states = pickle.load(path_to_discrete_states.open("rb"))
    wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

    model_for_simulation = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_solution_model,
        sim_model=True,
    )

    sim_df = simulate_scenario(
        model_for_simulation,
        solution=solution_dict,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options,
        seed=options["model_params"]["seed"],
    )

    cols_no_value_choice = [
        col for col in sim_df.columns if not col.startswith("value_choice_")
    ]

    # end_period = options["model_params"]["n_periods"] - 1
    end_period = (
        options["model_params"]["end_age"] - options["model_params"]["start_age"]
    )

    # Alive indiviudals should have nan entries
    df_0_to_49 = sim_df.xs(slice(0, end_period - 1), level="period")
    df_filtered = df_0_to_49[df_0_to_49["health"] != DEAD]

    assert not df_filtered[cols_no_value_choice].isna().any(axis=None)

    # No income and savings decision in the last period
    df_50 = sim_df.xs(end_period, level="period")
    assert df_50["total_income"].isna().all()
    assert df_50["savings_dec"].isna().all()
