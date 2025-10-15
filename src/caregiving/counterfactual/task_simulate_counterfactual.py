import copy
import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import pandas as pd
import pytask
import yaml
from pytask import Product

from caregiving.config import BLD, TESTS
from caregiving.counterfactual.simulate_counterfactual import (
    compute_npv,
    simulate_counterfactual_npv,
)
from caregiving.estimation.prepare_estimation import (
    load_and_setup_full_model_for_solution,
)
from caregiving.model.shared import DEAD
from caregiving.model.state_space import (
    create_state_space_functions,
    create_state_space_functions_counterfactual,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.simulate import simulate_scenario
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.solve import get_solve_func_for_model


@pytask.mark.skip()
def task_simulate_counterfactual(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
    # path_to_options_counterfactual: Path = BLD / "model" / "options_counterfactual.pkl",
    # path_to_solution_model_counterfactual: Path = BLD
    # / "model"
    # / "model_for_solution_counterfactual.pkl",
    path_to_load_solution: Path = BLD / "solve_and_simulate" / "solution.pkl",
    path_to_load_solution_counterfactual: Path = BLD
    / "solve_and_simulate"
    / "solution_counterfactual.pkl",
    path_to_load_simulation_counterfactual: Path = BLD
    / "solve_and_simulate"
    / "sim_df_counterfactual.csv",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
) -> None:
    """Simulate the model for given parametrization and model solution."""

    options_baseline = pickle.load(path_to_options.open("rb"))
    # options_counterfactual = pickle.load(path_to_options_counterfactual.open("rb"))

    options_counterfactual = copy.deepcopy(options_baseline)
    options_counterfactual["model_params"]["formal_care_costs"] = 0

    params = yaml.safe_load(path_to_start_params.open("rb"))

    # =================================================================================
    # Baseline
    # =================================================================================

    # model_for_solution = load_and_setup_full_model_for_solution(
    #     options, path_to_model=path_to_solution_model
    # )

    # # 1) Solve
    # solution_dict = {}
    # (
    #     solution_dict["value"],
    #     solution_dict["policy"],
    #     solution_dict["endog_grid"],
    # ) = get_solve_func_for_model(model_for_solution)(params)
    solution_dict_baseline = pickle.load(path_to_load_solution.open("rb"))

    # 2) Simulate
    initial_states = pickle.load(path_to_discrete_states.open("rb"))
    wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

    model_for_simulation_baseline = load_and_setup_model(
        options=options_baseline,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_solution_model,
        sim_model=True,
    )

    # Simulate the model
    sim_df = simulate_counterfactual_npv(
        model=model_for_simulation_baseline,
        solution=solution_dict_baseline,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options_baseline,
        seed=options_baseline["model_params"]["seed"],
    )

    # Compute NPV
    AGE_MAX = 80
    c = sim_df.loc[
        (sim_df["care_ever"] == 1) & (sim_df["age"] == AGE_MAX), "npv_income_30_80"
    ].mean()
    nc = sim_df.loc[
        (sim_df["care_ever"] == 0) & (sim_df["age"] == AGE_MAX), "npv_income_30_80"
    ].mean()
    npv_care = 1 - (nc / c)
    print(f"NPV Care: {npv_care}")

    del sim_df

    # =================================================================================
    # Counterfactual
    # =================================================================================

    # # Counterfactual simulation: No informal care

    # model_for_solution_counterfactual = load_and_setup_model(
    #     options=options_counterfactual,
    #     state_space_functions=create_state_space_functions_counterfactual(),
    #     utility_functions=create_utility_functions(),
    #     utility_functions_final_period=create_final_period_utility_functions(),
    #     budget_constraint=budget_constraint,
    #     # shock_functions=shock_function_dict(),
    #     path=path_to_solution_model,
    #     sim_model=False,
    # )

    # solution_dict_counterfactual = {}
    # (
    #     solution_dict_counterfactual["value"],
    #     solution_dict_counterfactual["policy"],
    #     solution_dict_counterfactual["endog_grid"],
    # ) = get_solve_func_for_model(model_for_solution_counterfactual)(params)

    sim_df_counter = pd.read_csv(path_to_load_simulation_counterfactual)

    # Compare to care demand ever! and no informal care counterfactual
    npv_no_informal_care = sim_df_counter.loc[
        (sim_df_counter["care_demand_ever"] == 1) & (sim_df_counter["age"] == AGE_MAX),
        "npv_income_30_80",
    ].mean()
    print(f"NPV No Informal Care: {npv_no_informal_care}")


@pytask.mark.skip()
def task_solve_and_simulate_counterfactual(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
    path_to_save_solution: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "solution_counterfactual.pkl",
    path_to_save_simulation: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "sim_df_counterfactual.csv",
) -> None:
    """Solve the model for given parametrization and model solution."""

    options_baseline = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))
    # options_counterfactual = pickle.load(path_to_options_counterfactual.open("rb"))

    options_counterfactual = copy.deepcopy(options_baseline)
    options_counterfactual["model_params"]["formal_care_costs"] = 0

    model_for_solution_counterfactual = load_and_setup_model(
        options=options_counterfactual,
        state_space_functions=create_state_space_functions_counterfactual(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_solution_model,
        sim_model=False,
    )

    solution_dict_counterfactual = {}
    (
        solution_dict_counterfactual["value"],
        solution_dict_counterfactual["policy"],
        solution_dict_counterfactual["endog_grid"],
    ) = get_solve_func_for_model(model_for_solution_counterfactual)(params)

    pickle.dump(solution_dict_counterfactual, path_to_save_solution.open("wb"))

    # 2) Simulate
    initial_states = pickle.load(path_to_discrete_states.open("rb"))
    wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

    model_for_simulation_counterfactual = load_and_setup_model(
        options=options_counterfactual,
        state_space_functions=create_state_space_functions_counterfactual(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        # shock_functions=shock_function_dict(),
        path=path_to_solution_model,
        sim_model=True,
    )

    sim_df_counter = simulate_counterfactual_npv(
        model=model_for_simulation_counterfactual,
        solution=solution_dict_counterfactual,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options_counterfactual,
        seed=options_counterfactual["model_params"]["seed"],
    )
    sim_df_counter.to_csv(path_to_save_simulation, index=False)
