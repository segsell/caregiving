import copy
import pickle
from pathlib import Path

import jax.numpy as jnp
import pandas as pd
import yaml
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.solve import get_solve_func_for_model
from pytask import Product

from caregiving.config import BLD, TESTS
from caregiving.counterfactual.simulate_counterfactual import (
    compute_npv,
    simulate_counterfactual_npv,
)
from caregiving.estimation.estimation_setup import (
    load_and_setup_full_model_for_solution,
)
from caregiving.model.shared import DEAD
from caregiving.model.state_space import (  # create_state_space_functions_counterfactual,
    create_state_space_functions,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.simulate import simulate_scenario


def task_simulate_counterfactual(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_load_solution: Path = BLD / "solve_and_simulate" / "solution.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
) -> None:
    """Simulate the model for given parametrization and model solution."""

    options_baseline = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))

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
    # compare to care demand ever! and no informal care counterfactual

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

    # # Counterfactual simulation: No informal care
    # options_no_informal_care = copy.deepcopy(options_baseline)
    # options_no_informal_care["model_params"]["formal_care_costs"] = 0

    # model_for_simulation_no_informal_care = load_and_setup_model(
    #     options=options_no_informal_care,
    #     state_space_functions=create_state_space_functions_counterfactual(),
    #     utility_functions=create_utility_functions(),
    #     utility_functions_final_period=create_final_period_utility_functions(),
    #     budget_constraint=budget_constraint,
    #     # shock_functions=shock_function_dict(),
    #     path=path_to_solution_model,
    #     sim_model=True,
    # )

    # sim_df_no_informal_care = simulate_counterfactual_npv(
    #     model=model_for_simulation_no_informal_care,
    #     solution=solution_dict_baseline,
    #     initial_states=initial_states,
    #     wealth_agents=wealth_agents,
    #     params=params,
    #     options=options_no_informal_care,
    #     seed=options_no_informal_care["model_params"]["seed"],
    # )

    # npv_no_informal_care = sim_df_no_informal_care.loc[
    #     (sim_df_no_informal_care["care_demand_ever"] == 1)
    #     & (sim_df_no_informal_care["age"] == 80),
    #     "npv_income_30_80",
    # ].mean()
    # print(f"NPV No Informal Care: {npv_no_informal_care}")

    # breakpoint()
