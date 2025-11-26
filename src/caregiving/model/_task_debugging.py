# """Debugging tasks."""

# from pathlib import Path
# from typing import Annotated

# import jax
# import jax.numpy as jnp
# import pandas as pd
# import pytask
# from dcegm.pre_processing.setup_model import load_model_dict
# from dcegm.simulation.sim_utils import create_simulation_df
# from dcegm.simulation.simulate import simulate_all_periods
# from dcegm.solve import get_solve_func_for_model
# from pytask import Product

# from caregiving.config import BLD
# from caregiving.model.budget import budget_constraint, create_savings_grid
# from caregiving.model.shared import (
#     AGE_BINS_SIM,
#     ALL,
#     # FORMAL_CARE,
#     FULL_TIME,
#     # INFORMAL_CARE,
#     NOT_WORKING,
#     PART_TIME,
#     RETIREMENT,
#     UNEMPLOYED,
# )
# from caregiving.model.state_space import create_state_space_functions
# from caregiving.model.task_specify_model import get_options_dict
# from caregiving.model.utility_functions import (
#     create_final_period_utility_functions,
#     create_utility_functions,
# )
# from caregiving.simulation.initial_conditions import draw_initial_states
# from caregiving.simulation.simulate import (
#     create_simulation_array_from_df,
#     create_simulation_df_from_dict,
#     get_share_by_age,
#     get_share_by_type_by_age_bin,
#     simulate_moments,
# )
# from caregiving.utils import load_dict_from_pickle, save_dict_to_pickle

# jax.config.update("jax_enable_x64", True)  # noqa: FBT003


# PARAMS = {
#     "rho": 0.8,
#     "beta": 0.959,
#     "sigma": 0.5364562201,
#     "lambda": 1.0024565188638312,
#     "interest_rate": 0.04,
#     "disutility_part_time_constant": -1.1325945318518755,
#     "disutility_part_time_age": 2.959047551749629,
#     "disutility_part_time_age_squared": 0.08466169989356942,
#     "disutility_full_time_constant": 2.0114514159888124,
#     "disutility_full_time_age": 0.3052996873888407,
#     "disutility_full_time_age_squared": 0.5032327025372947,
#     "disutility_part_time_informal_care_constant": -1.1325945318518755,
#     "disutility_part_time_informal_care_age": 2.959047551749629,
#     "disutility_part_time_informal_care_age_squared": 0.08466169989356942,
#     "disutility_full_time_informal_care_constant": 1.0114514159888124,
#     "disutility_full_time_informal_care_age": 0.3052996873888407,
#     "disutility_full_time_informal_care_age_squared": 0.5032327025372947,
#     "part_time_constant": -2.108705825398208,
#     "part_time_not_working_last_period": -1.9513432887146251,
#     # "part_time_high_education": 3.0168909139283073,
#     # "part_time_above_retirement_age": -1.4626306391231516,
#     "full_time_constant": -3.509669706331148,
#     "full_time_not_working_last_period": -2.754513647434816,
#     # "full_time_high_education": 2.6489281603375474,
#     # "full_time_above_retirement_age": -3.0922669492895896,
#     #
#     "wage_constant": 2.036543017339472,
#     "wage_experience": 1.2298196564297121,
#     "wage_experience_squared": -0.04276311179999581,
#     "wage_high_education": 0.31239997233992,
#     "wage_part_time": -0.6683338057565962,
#     "wage_std_regression_residual": 0.41264342652997477,
#     "utility_no_care_parent_bad_health": -1,
#     "utility_informal_care_parent_bad_health": 0.5,
#     "utility_formal_care_parent_bad_health": 0.2,
#     "utility_combination_care_parent_bad_health": 0.4,
# }

# START_PARAMS = {
#     "rho": 1.96,  # Adda et al (2017)
#     "beta": 0.959,  # Adda et al (2017)
#     "sigma": 0.5364562201,  # 0.5584583071,
#     "lambda": 1,
#     "interest_rate": 0.04,  # Adda et al (2017)
#     "disutility_part_time_constant": 2,
#     "disutility_part_time_age": 0.1,
#     "disutility_part_time_age_squared": -0.06,
#     "disutility_full_time_constant": 1,
#     "disutility_full_time_age": 0.152,
#     "disutility_full_time_age_squared": -0.012,
#     "disutility_part_time_informal_care_constant": -1,
#     "disutility_full_time_informal_care_constant": -1.5,
#     # "disutility_part_time_informal_care_age": 0.1,
#     # "disutility_part_time_informal_care_age_squared": -0.01,
#     # "disutility_full_time_informal_care_age": 0.1,
#     # "disutility_full_time_informal_care_age_squared": -0.01,
#     # caregiving
#     "utility_no_care_parent_bad_health": 0,
#     "utility_informal_care_parent_bad_health": 0,
#     "utility_formal_care_parent_bad_health": 0,
#     "utility_combination_care_parent_bad_health": 0,
#     # part-time job offer
#     "part_time_constant": -2,
#     "part_time_not_working_last_period": -1,
#     "part_time_above_retirement_age": -2,
#     # "part_time_high_education": 0.5,
#     # full-time job offer
#     "full_time_constant": -2,
#     "full_time_not_working_last_period": -2,
#     "full_time_above_retirement_age": -3,
#     # "full_time_high_education": 1,
#     # wage equation
#     "wage_constant": 2,
#     "wage_experience": 0.1,
#     "wage_experience_squared": -0.01,
#     "wage_part_time": -0.1,
#     # "wage_high_education": 0.5,
# }


# @pytask.mark.skip()
# def task_debugging(
#     path_to_save_result: Annotated[Path, Product] = BLD / "debugging" / "result.pkl",
#     path_to_save_sim_dict: Annotated[Path, Product] = BLD
#     / "debugging"
#     / "sim_dict.pkl",
# ):
#     """Debugging task.

#     path_to_model: Path = BLD / "model" / "model.pkl",

#     results = load_dict_from_pickle(BLD / "debugging" / "result.pkl")


#     # check initial labor shares
#     jnp.unique(initial_states["lagged_choice"], return_counts=True)


#     _mother_health_probs = initial_conditions.loc[
#         ["mother_good_health", "mother_medium_health", "mother_bad_health"]
#     ].to_numpy()
#     mother_health_probs = jnp.array(_mother_health_probs).ravel()

#     """
#     path_to_model = BLD / "model" / "model.pkl"
#     options = get_options_dict()

#     params = START_PARAMS

#     model_loaded = load_model_dict(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         path=path_to_model,
#     )

#     exog_savings_grid = create_savings_grid()

#     # results = load_dict_from_pickle(BLD / "debugging" / "result.pkl")
#     func = get_solve_func_for_model(
#         model=model_loaded,
#         exog_savings_grid=exog_savings_grid,
#         options=options,
#     )
#     results = func(params)
#     save_dict_to_pickle(results, path_to_save_result)

#     n_agents = 100_000
#     seed = 2024

#     path_high_educ = f"{BLD}/moments/real_wealth_age_39_high_educ.csv"
#     initial_wealth_high_educ = jnp.asarray(pd.read_csv(path_high_educ)).ravel()

#     path_low_educ = f"{BLD}/moments/real_wealth_age_39_low_educ.csv"
#     initial_wealth_low_educ = jnp.asarray(pd.read_csv(path_low_educ)).ravel()

#     path = f"{BLD}/moments/initial_discrete_conditions_at_age_40.csv"
#     initial_conditions = pd.read_csv(path, index_col=0)

#     initial_resources, initial_states = draw_initial_states(
#         initial_conditions,
#         initial_wealth_low_educ=initial_wealth_low_educ,
#         initial_wealth_high_educ=initial_wealth_high_educ,
#         n_agents=n_agents,
#         seed=seed,
#     )

#     sim_dict = simulate_all_periods(
#         states_initial=initial_states,
#         resources_initial=initial_resources,
#         n_periods=options["model_params"]["n_periods"],
#         params=params,
#         seed=seed,
#         value_solved=results[0],
#         policy_solved=results[1],
#         endog_grid_solved=results[2],
#         model=model_loaded,
#     )
#     save_dict_to_pickle(sim_dict, path_to_save_sim_dict)

#     # df_raw = create_simulation_df(sim_dict)
#     # data = df_raw.notna()

#     data = create_simulation_df_from_dict(sim_dict)
#     arr, idx = create_simulation_array_from_df(
#         data=data,
#         options=options,
#         params=params,
#     )

#     # n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

#     # keys_to_drop = ["taste_shocks"]
#     # dict_to_df = {key: sim_dict[key] for key in sim_dict if key not in keys_to_drop}

#     # data = pd.DataFrame(
#     #     {key: val.ravel() for key, val in dict_to_df.items()},
#     #     index=pd.MultiIndex.from_product(
#     #         [np.arange(n_periods), np.arange(n_agents)],
#     #         names=["period", "agent"],
#     #     ),
#     # )
#     # data["age"] = data["period"] + options["model_params"]["start_age"]

#     # data_clean = data.dropna()
#     # column_indices = {col: idx for idx, col in enumerate(data_clean.columns)}
#     # idx = column_indices.copy()
#     # arr = jnp.asarray(data_clean)

#     share_not_working_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=UNEMPLOYED,
#     )  # 15
#     share_part_time_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=PART_TIME,
#     )  # 15
#     share_full_time_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=FULL_TIME,
#     )  # 15
#     share_retired_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=RETIREMENT,
#     )  # 15
#     share_out_of_labor_force = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=NOT_WORKING,
#     )  # 15

#     # share_informal_care_by_age_bin = get_share_by_type_by_age_bin(  # noqa: F841
#     #     arr,
#     #     ind=idx,
#     #     choice=INFORMAL_CARE,
#     #     care_type=ALL,
#     #     age_bins=AGE_BINS_SIM,
#     # )

#     # share_formal_care_by_age_bin = get_share_by_type_by_age_bin(  # noqa: F841
#     #     arr,
#     #     ind=idx,
#     #     choice=FORMAL_CARE,
#     #     care_type=ALL,
#     #     age_bins=AGE_BINS_SIM,
#     # )

#     # share_not_working_informal_care_by_age_bin = (  # noqa: F841
#     #     get_share_by_type_by_age_bin(
#     #         ind=idx,
#     #         choice=UNEMPLOYED,
#     #         care_type=INFORMAL_CARE,
#     #         age_bins=AGE_BINS_SIM,
#     #     )
#     # )
#     # share_part_time_informal_care_by_age_bin = (  # noqa: F841
#     #     get_share_by_type_by_age_bin(
#     #         arr,
#     #         ind=idx,
#     #         choice=PART_TIME,
#     #         care_type=INFORMAL_CARE,
#     #         age_bins=AGE_BINS_SIM,
#     #     )
#     # )
#     # share_full_time_informal_care_by_age_bin = (  # noqa: F841
#     #     get_share_by_type_by_age_bin(
#     #         arr,
#     #         ind=idx,
#     #         choice=FULL_TIME,
#     #         care_type=INFORMAL_CARE,
#     #         age_bins=AGE_BINS_SIM,
#     #     )
#     # )

#     return (
#         share_not_working_by_age,
#         share_part_time_by_age,
#         share_full_time_by_age,
#         share_retired_by_age,
#         share_out_of_labor_force,
#         # share_informal_care_by_age_bin,
#         # share_formal_care_by_age_bin,
#         # share_not_working_informal_care_by_age_bin,
#         # share_part_time_informal_care_by_age_bin,
#         # share_full_time_informal_care_by_age_bin,
#     )


# @pytask.mark.skip()
# def task_debug_simulate():
#     """Debug simulate.

#     path_to_sim_dict: Path = BLD / "debugging" / "sim_dict.pkl",

#     """
#     path_to_sim_dict = BLD / "debugging" / "sim_dict.pkl"

#     options = get_options_dict()
#     sim_dict = load_dict_from_pickle(path_to_sim_dict)

#     data = create_simulation_df_from_dict(sim_dict)

#     arr, idx = create_simulation_array_from_df(
#         data=data,
#         options=options,
#         params=PARAMS,
#     )
#     out = simulate_moments(arr, idx)  # 159

#     # breakpoint()

#     return out, arr
