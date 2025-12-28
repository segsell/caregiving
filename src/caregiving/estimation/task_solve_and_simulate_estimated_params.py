"""Solve and simulate the model for estimated parameters."""

import pickle
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
import yaml
from pytask import Product

import dcegm
from caregiving.config import BLD, SRC
from caregiving.model.shared import DEAD
from caregiving.model.state_space import (
    construct_experience_years,
    create_state_space_functions,
)

# from caregiving.simulation.simulate import simulate_scenario
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint

jax.config.update("jax_enable_x64", True)


@pytask.mark.solve_and_simulate
@pytask.mark.baseline_model
def task_solve_and_simulate_estimated_params(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model: Path = BLD / "model" / "model.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_estimated_params: Path = BLD
    / "model"
    / "params"
    / "estimated_params_model.yaml",
    path_to_initial_states: Path = (
        BLD / "model" / "initial_conditions" / "initial_states.pkl"
    ),
    path_to_save_solution: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "solution_estimated_params.pkl",
    # path_to_save_simulation_model: Annotated[Path, Product] = BLD
    # / "model"
    # / "model_for_simulation_estimated_params.pkl",
    path_to_save_simulated_data: Annotated[Path, Product] = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    # path_to_save_simulated_data_jax: Annotated[Path, Product] = BLD
    # / "solve_and_simulate"
    # / "simulated_data_jax_estimated_params.pkl",
) -> None:
    """Solve and simulate the model for estimated parameters."""

    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))
    params = yaml.safe_load(path_to_estimated_params.open("rb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_model,
    )
    # 1) Solve
    model_solved = model.solve(params, save_sol_path=path_to_save_solution)

    # 2) Simulate
    initial_states = pickle.load(path_to_initial_states.open("rb"))

    # =================================================================================
    sim_df = model_solved.simulate(
        states_initial=initial_states,
        seed=specs["seed"],
    )

    sim_df = sim_df[sim_df["health"] != DEAD].copy()
    sim_df.reset_index(inplace=True)

    sim_df["age"] = sim_df["period"] + specs["start_age"]
    # sim_df = create_additional_variables(sim_df, specs)
    # =================================================================================

    # sim_df.to_csv(path_to_save_simulated_data, index=True)
    sim_df.to_pickle(path_to_save_simulated_data)


# ==============================================================================
# Additional variables related to the budget equation (see budget_equation.py).
# ==============================================================================


def _create_income_variables(df, specs):
    """Create income related variables in the simulated dataframe.
    Note: check budget equation first! They may already be there (under "aux").
    """
    df = df.copy()

    # Create income vars:
    # First, total income as the difference between wealth at the beginning of
    # next period and savings.
    df.loc[:, "total_income"] = df["assets_begin_of_period"] - df.groupby("agent")[
        "savings"
    ].shift(1)

    # periodic savings and savings rate
    df.loc[:, "savings_dec"] = df["total_income"] - df["consumption"]
    df.loc[:, "savings_rate"] = df["savings_dec"] / df["total_income"]

    # Create gross own income (without pension income)
    df.loc[:, "gross_own_income"] = (
        (df["choice"] == 0) * df["gross_retirement_income"]  # Retired  # noqa: PLR2004
        + (df["choice"] == 1) * 0  # Unemployed  # noqa: PLR2004
        + ((df["choice"] == 2) | (df["choice"] == 3))  # noqa: PLR2004
        * df["gross_labor_income"]  # Part-time or full-time work
    )
    return df


def _transform_states_into_variables(df, specs):
    """Transform state variables into more interpretable variables."""
    df = df.copy()

    # Create additional variables
    df.loc[:, "age"] = df["period"] + specs["start_age"]

    # Create experience years
    df.loc[:, "exp_years"] = construct_experience_years(
        float_experience=df["experience"].values,
        period=df["period"].values,
        is_retired=df["lagged_choice"].values == 0,
        model_specs=specs,
    )

    return df


def _compute_working_hours(df, specs):
    """Compute working hours based on employment choice and demographics."""
    df = df.copy()

    # Initialize working_hours column
    df.loc[:, "working_hours"] = 0.0

    for sex_var in (0, 1):
        for edu_var in range(specs["n_education_types"]):
            # Full-time work
            mask_ft = (
                (df["choice"] == 3)  # noqa: PLR2004
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var)
            )
            df.loc[mask_ft, "working_hours"] = specs["av_annual_hours_ft"][
                sex_var, edu_var
            ]

            # Part-time work
            mask_pt = (
                (df["choice"] == 2)  # noqa: PLR2004
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var)
            )
            df.loc[mask_pt, "working_hours"] = specs["av_annual_hours_pt"][
                sex_var, edu_var
            ]

    return df


def _compute_actual_retirement_age(df):
    """Compute actual retirement age based on choice variable."""
    df = df.copy()

    df_retirement = df[df["choice"] == 0].copy()
    actual_retirement_ages = df_retirement.groupby("agent")["age"].min()
    df.loc[:, "actual_retirement_age"] = df["agent"].map(actual_retirement_ages)
    return df


def create_realized_taste_shock(df, specs):
    df.loc[:, "real_taste_shock"] = np.nan
    for choice in range(specs["n_choices"]):
        df.loc[df["choice"] == choice, "real_taste_shock"] = df.loc[
            df["choice"] == choice, f"taste_shocks_{choice}"
        ]
    return df


def create_real_utility(df, specs):
    df = create_realized_taste_shock(df, specs)
    df.loc[:, "real_util"] = df["utility"] + df["real_taste_shock"]
    return df


def create_additional_variables(df, specs):
    """Wrapper function to create additional variables in the simulated dataframe."""
    df = df.copy()

    df = _create_income_variables(df, specs)
    df = _transform_states_into_variables(df, specs)
    df = _compute_working_hours(df, specs)
    df = _compute_actual_retirement_age(df)
    df = create_real_utility(df, specs)
    return df
