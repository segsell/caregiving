"""Simulate summary statistics for counterfactual scenarios."""

from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.solve import get_solve_func_for_model
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.state_space_counterfactual import (
    create_state_space_functions_no_informal_care,
    create_state_space_functions_only_informal_care,
)
from caregiving.model.budget import (
    budget_constraint,
    calc_net_income_pensions,
    create_savings_grid,
)
from caregiving.model.shared import (
    AGE_BINS_SIM,
    BAD_HEALTH,
    BETA,
    COMBINATION_CARE,
    DEAD,
    FORMAL_CARE,
    FULL_TIME,
    INFORMAL_CARE,
    MAX_AGE_SIM,
    MIN_AGE_SIM,
    NO_CARE,
    NO_COMBINATION_CARE,
    NO_INFORMAL_CARE,
    NOT_WORKING,
    PART_TIME,
    PURE_INFORMAL_CARE,
    RETIREMENT,
    RETIREMENT_AGE,
    UNEMPLOYED,
)
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.task_specify_model import get_options_dict
from caregiving.model.utility_functions import (
    create_final_period_utility_functions,
    create_utility_functions,
)
from caregiving.simulation.initial_conditions import draw_initial_states
from caregiving.simulation.simulate import (
    _assign_working_hours_vectorized,
    create_simulation_array_from_df,
    create_simulation_df_from_dict,
    get_share_by_age,
    get_share_by_type_by_age_bin,
    simulate_moments,
)
from caregiving.utils import load_dict_from_pickle, save_dict_to_pickle

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

FOUR_YEARS = 4
PERIOD_40 = 40

# Still good!

PROGRESS = {
    "rho": 1.9822537050081712,
    "beta": 0.959,
    "sigma": 0.5722436158,
    "lambda": 1.0031254320374083,
    "interest_rate": 0.04,
    "disutility_part_time_constant": 0.1812520177040096,
    "disutility_part_time_age": -0.159966282487904,
    "disutility_part_time_age_squared": -0.005271341145737907,
    "disutility_full_time_constant": 1.0768449003853533,
    "disutility_full_time_age": 0.09372663929587333,
    "disutility_full_time_age_squared": -0.004756250524533101,
    "disutility_part_time_informal_care_constant": -0.5687555122901369,
    "disutility_full_time_informal_care_constant": -1.1756026407414506,
    "utility_no_care_parent_bad_health": -1.0574802998146797,
    "utility_informal_care_parent_bad_health": 1.1383169733068517,
    "utility_formal_care_parent_bad_health": -0.01162702405221161,
    "utility_combination_care_parent_bad_health": 0.05791895311391748,
    "part_time_constant": -2.5183517209,
    "part_time_not_working_last_period": 0.2908819469,
    "part_time_above_retirement_age": -1.9963829408,
    "full_time_constant": -2.3705285317,
    "full_time_not_working_last_period": -1.0331988175,
    "full_time_above_retirement_age": -2.6573652335,
    "wage_constant": 2.4424732229,
    "wage_experience": 0.0195435854,
    "wage_experience_squared": -0.0003597042,
    "wage_part_time": -0.1245776169,
}


@pytask.mark.skip()
def task_simulate_benchmark(
    path_to_save_result: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "benchmark_result.pkl",
    path_to_save_sim_dict: Annotated[Path, Product] = BLD  # noqa: ARG001
    / "counterfactual"
    / "benchmark_dict.pkl",
):
    """Debug simulate.

    path_to_sim_dict: Path = BLD / "debugging" / "sim_dict.pkl",

    """
    path_to_model = BLD / "model" / "model.pkl"
    options = get_options_dict()

    params = PROGRESS

    model_loaded = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_model,
    )

    exog_savings_grid = create_savings_grid()

    # results = load_dict_from_pickle(BLD / "counterfactual" / "benchmark_result.pkl")
    func = get_solve_func_for_model(
        model=model_loaded,
        exog_savings_grid=exog_savings_grid,
        options=options,
    )
    results = func(params)
    save_dict_to_pickle(results, path_to_save_result)

    n_agents = 100_000
    seed = 2024

    path_high_educ = f"{BLD}/moments/real_wealth_age_39_high_educ.csv"
    initial_wealth_high_educ = jnp.asarray(pd.read_csv(path_high_educ)).ravel()

    path_low_educ = f"{BLD}/moments/real_wealth_age_39_low_educ.csv"
    initial_wealth_low_educ = jnp.asarray(pd.read_csv(path_low_educ)).ravel()

    path = f"{BLD}/moments/initial_discrete_conditions_at_age_40.csv"
    initial_conditions = pd.read_csv(path, index_col=0)

    initial_resources, initial_states = draw_initial_states(
        initial_conditions,
        initial_wealth_low_educ=initial_wealth_low_educ,
        initial_wealth_high_educ=initial_wealth_high_educ,
        n_agents=n_agents,
        seed=seed,
    )

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=initial_resources,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        value_solved=results[0],
        policy_solved=results[1],
        endog_grid_solved=results[2],
        model=model_loaded,
    )

    # sim_dict = load_dict_from_pickle(path_to_save_sim_dict)
    options = get_options_dict()

    data_raw = create_simulation_df_from_dict(sim_dict)

    data = prepare_data_for_counterfactual_analysis(data_raw, options, params)

    arr, idx = create_simulation_array_from_df(
        data=data_raw,
        options=options,
        params=PROGRESS,
    )
    # out = simulate_moments(arr, idx)  # 159

    # summary statistics
    # Filter the DataFrame based on the given conditions
    filtered_data = data[
        (data["informal_care_ever"] == True)
        & (data["informal_care_years"] >= FOUR_YEARS)
        & (data.index.get_level_values("period") == PERIOD_40)
    ]
    mean_cumsum_total_npv_income = filtered_data["cumsum_total_NPV_income"].mean()

    filtered_data_raw = data[
        (data["informal_care_ever"] == True)
        & (data["informal_care_years"] >= FOUR_YEARS)
        & (data.index.get_level_values("period") == PERIOD_40)
    ]
    mean_cumsum_total_income = filtered_data_raw["cumsum_total_income"].mean()

    return mean_cumsum_total_income, mean_cumsum_total_npv_income


@pytask.mark.skip()
def task_simulate_counterfactual_no_informal_care(
    path_to_save_result: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "no_informal_care_result.pkl",
    path_to_save_sim_dict: Annotated[Path, Product] = BLD  # noqa: ARG001
    / "counterfactual"
    / "no_informal_care_dict.pkl",
):
    """Debug simulate.

    path_to_sim_dict: Path = BLD / "debugging" / "sim_dict.pkl",

    """
    path_to_model = BLD / "model" / "counterfactual_no_informal_care.pkl"
    options = get_options_dict()

    params = PROGRESS

    model_loaded = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions_no_informal_care(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_model,
    )

    exog_savings_grid = create_savings_grid()

    # results = load_dict_from_pickle(path_to_save_result)
    func = get_solve_func_for_model(
        model=model_loaded,
        exog_savings_grid=exog_savings_grid,
        options=options,
    )
    results = func(params)
    save_dict_to_pickle(results, path_to_save_result)

    n_agents = 100_000
    seed = 2024

    path_high_educ = f"{BLD}/moments/real_wealth_age_39_high_educ.csv"
    initial_wealth_high_educ = jnp.asarray(pd.read_csv(path_high_educ)).ravel()

    path_low_educ = f"{BLD}/moments/real_wealth_age_39_low_educ.csv"
    initial_wealth_low_educ = jnp.asarray(pd.read_csv(path_low_educ)).ravel()

    path = f"{BLD}/moments/initial_discrete_conditions_at_age_40.csv"
    initial_conditions = pd.read_csv(path, index_col=0)

    initial_resources, initial_states = draw_initial_states(
        initial_conditions,
        initial_wealth_low_educ=initial_wealth_low_educ,
        initial_wealth_high_educ=initial_wealth_high_educ,
        n_agents=n_agents,
        seed=seed,
    )

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=initial_resources,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        value_solved=results[0],
        policy_solved=results[1],
        endog_grid_solved=results[2],
        model=model_loaded,
    )

    # sim_dict = load_dict_from_pickle(path_to_save_sim_dict)
    options = get_options_dict()

    data_raw = create_simulation_df_from_dict(sim_dict)

    data = prepare_data_for_counterfactual_analysis(data_raw, options, params)

    arr, idx = create_simulation_array_from_df(
        data=data_raw,
        options=options,
        params=PROGRESS,
    )
    # out = simulate_moments(arr, idx)  # 159

    # summary statistics
    # Filter the DataFrame based on the given conditions
    filtered_data = data[
        (data["informal_care_ever"] == True)
        & (data["informal_care_years"] >= FOUR_YEARS)
        & (data.index.get_level_values("period") == PERIOD_40)
    ]
    mean_cumsum_total_npv_income = filtered_data["cumsum_total_NPV_income"].mean()

    filtered_data_raw = data[
        (data["informal_care_ever"] == True)
        & (data["informal_care_years"] >= FOUR_YEARS)
        & (data.index.get_level_values("period") == PERIOD_40)
    ]
    mean_cumsum_total_income = filtered_data_raw["cumsum_total_income"].mean()

    return mean_cumsum_total_income, mean_cumsum_total_npv_income


def prepare_data_for_counterfactual_analysis(data, options, params):  # noqa: PLR0915
    """Create simulation array from dict."""
    data = data.copy()  # Make a copy to avoid modifying a slice

    options = options["model_params"]
    n_agents = options["n_agents"]
    n_periods = options["n_periods"]

    # Assigning the 'agent' and age-related calculations
    data.loc[:, "agent"] = jnp.tile(jnp.arange(n_agents), n_periods)
    period_indices = jnp.tile(jnp.arange(n_periods)[:, None], (1, n_agents)).ravel()

    data.loc[:, "age"] = options["start_age"] + period_indices
    data.loc[:, "age_squared"] = data["age"] ** 2

    # Mother age bins
    data.loc[:, "mother_alive"] = data["mother_health"] < DEAD

    data.loc[:, "mother_age"] = options["mother_start_age"] + period_indices
    data.loc[data["mother_alive"] == False, "mother_age"] = np.nan

    # Define age bins
    bins = [65, 70, 75, 80, 85, np.inf]
    labels = ["65_70", "70_75", "75_80", "80_85", "85_plus"]

    data["mother_age_bin"] = pd.cut(
        data["mother_age"],
        bins=bins,
        labels=labels,
        right=False,
    )

    dummies = pd.get_dummies(data["mother_age_bin"], prefix="mother_age_bin")
    data = pd.concat([data, dummies], axis=1)
    data = data.drop(columns=["mother_age_bin"])

    # Financial calculations
    data.loc[:, "wealth"] = data["savings"] + data["consumption"]
    data.loc[:, "savings_rate"] = jnp.where(
        jnp.array(data["wealth"]) > 0,
        jnp.divide(jnp.array(data["savings"]), jnp.array(data["wealth"])),
        0,
    )

    # Squared experience
    data.loc[:, "experience"] = data["experience"] / 2
    data.loc[:, "experience_squared"] = data["experience"] ** 2

    # Employment status
    data.loc[:, "lagged_part_time"] = jnp.isin(
        jnp.array(data["lagged_choice"]),
        PART_TIME,
    )

    data.loc[:, "choice_retired"] = jnp.isin(
        jnp.array(data["choice"]),
        RETIREMENT,
    )
    data.loc[:, "choice_part_time"] = jnp.isin(
        jnp.array(data["choice"]),
        PART_TIME,
    )
    data.loc[:, "choice_full_time"] = jnp.isin(
        jnp.array(data["choice"]),
        FULL_TIME,
    )
    # data.loc[:, "choice_informal_care"] = jnp.isin(
    #     jnp.array(data["choice"]),
    #     INFORMAL_CARE,
    # )

    data.loc[:, "choice_no_care"] = jnp.isin(
        jnp.array(data["choice"]),
        NO_CARE,
    )
    data.loc[:, "choice_pure_informal_care"] = jnp.isin(
        jnp.array(data["choice"]),
        PURE_INFORMAL_CARE,
    )
    data.loc[:, "choice_combination_care"] = jnp.isin(
        jnp.array(data["choice"]),
        COMBINATION_CARE,
    )
    data.loc[:, "choice_formal_care"] = jnp.isin(
        jnp.array(data["choice"]),
        FORMAL_CARE,
    )

    data.loc[:, "choice_informal_care"] = jnp.isin(
        jnp.array(data["choice"]),
        INFORMAL_CARE,
    )

    # Wage calculations
    data.loc[:, "log_wage"] = (
        params["wage_constant"]
        + params["wage_experience"] * data["experience"]
        + params["wage_experience_squared"] * data["experience_squared"]
        # + params["wage_high_education"] * data["high_educ"]
        + params["wage_part_time"] * data["lagged_part_time"]
    )

    data.loc[:, "wage"] = jnp.exp(
        jnp.array(data["log_wage"]) + jnp.array(data["income_shock"]),
    )

    # Working hours and income calculation
    data.loc[:, "working_hours"] = jax.vmap(_assign_working_hours_vectorized)(
        data["lagged_choice"].values,
    )

    data.loc[:, "labor_income"] = data["working_hours"] * data["wage"]

    # =================================================================================

    # Unemployment benefits
    data.loc[:, "means_test"] = (
        data.loc[:, "savings"] < options["unemployment_wealth_thresh"]
    )
    data.loc[:, "unemployment_benefits"] = (
        data.loc[:, "means_test"] * options["unemployment_benefits"] * 12
    )

    # retirement benefits
    data.loc[:, "pension_factor"] = (
        1 - (data.loc[:, "age"] - RETIREMENT_AGE) * options["early_retirement_penalty"]
    )
    data.loc[:, "retirement_income_gross_one_year"] = (
        options["pension_point_value"]
        * data.loc[:, "experience"]
        * data.loc[:, "pension_factor"]
        * data.loc[:, "choice_retired"]  # only receive benefits if actually retired
        * 12
    )
    data.loc[:, "retirement_income"] = jax.vmap(calc_net_income_pensions)(
        data.loc[:, "retirement_income_gross_one_year"].values,
    )

    # Cumulative life time income
    data["cum_labor_income"] = data.groupby(level="agent")["labor_income"].transform(
        "cumsum",
    )
    data["cum_unemployment_benefits"] = data.groupby(level="agent")[
        "unemployment_benefits"
    ].transform("cumsum")
    data["cum_retirement_income"] = data.groupby(level="agent")[
        "retirement_income"
    ].transform("cumsum")

    # Compute the cumulative sum over all three variables by 'agent'
    data["cumsum_total_income"] = data[
        ["cum_labor_income", "cum_unemployment_benefits", "cum_retirement_income"]
    ].sum(axis=1)
    data["cumsum_total_income"] = data.groupby(level="agent")[
        "cumsum_total_income"
    ].transform("cumsum")

    # Discount factor
    data["beta"] = BETA ** data.index.get_level_values("period")

    # Net Present Value for each income stream
    data["NPV_labor_income"] = data["labor_income"] * data["beta"]
    data["NPV_unemployment_benefits"] = data["unemployment_benefits"] * data["beta"]
    data["NPV_retirement_income"] = data["retirement_income"] * data["beta"]

    # Sum up the NPV for each income stream
    data["cumsum_NPV_labor_income"] = data.groupby(level="agent")[
        "NPV_labor_income"
    ].transform("cumsum")
    data["cumsum_NPV_unemployment_benefits"] = data.groupby(level="agent")[
        "NPV_unemployment_benefits"
    ].transform("cumsum")
    data["cumsum_NPV_retirement_income"] = data.groupby(level="agent")[
        "NPV_retirement_income"
    ].transform("cumsum")

    # Compute the cumulative sum over all three variables by 'agent'
    data["cumsum_total_NPV_income"] = data[
        ["NPV_labor_income", "NPV_unemployment_benefits", "NPV_retirement_income"]
    ].sum(axis=1)
    data["cumsum_total_NPV_income"] = data.groupby(level="agent")[
        "cumsum_total_NPV_income"
    ].transform("cumsum")

    # Caregiving stuff
    data["informal_care_ever"] = data.groupby(level="agent")[
        "choice_informal_care"
    ].transform("max")
    data["pure_informal_care_ever"] = data.groupby(level="agent")[
        "choice_pure_informal_care"
    ].transform("max")
    data["combination_care_ever"] = data.groupby(level="agent")[
        "choice_combination_care"
    ].transform("max")
    data["formal_care_ever"] = data.groupby(level="agent")[
        "choice_formal_care"
    ].transform("max")

    data["informal_care_years"] = data.groupby(level="agent")[
        "choice_informal_care"
    ].cumsum()
    data["pure_informal_care_years"] = data.groupby(level="agent")[
        "choice_pure_informal_care"
    ].cumsum()
    data["combination_care_years"] = data.groupby(level="agent")[
        "choice_combination_care"
    ].cumsum()
    data["formal_care_years"] = data.groupby(level="agent")[
        "choice_formal_care"
    ].cumsum()

    # =================================================================================

    # Logit regressison: Trim subsample
    # Define the columns to be set to NaN
    columns_to_nan = [
        "choice_no_care",
        "choice_informal_care",
        "choice_pure_informal_care",
        "choice_combination_care",
        "choice_formal_care",
    ]
    columns_to_nan += [col for col in data.columns if col.startswith("mother_age_bin")]

    # Set the specified columns to NaN where mother_health != 1
    data.loc[data["mother_health"] != BAD_HEALTH, columns_to_nan] = np.nan

    # Create a mapping of column indices
    # column_indices = {col: idx for idx, col in enumerate(data.columns)}

    # data = data.dropna()

    return data
