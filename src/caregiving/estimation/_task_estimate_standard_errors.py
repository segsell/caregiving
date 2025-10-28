# """Analytical standard errors a la Della Vigna et al (2016)."""

# import jax.numpy as jnp
# import pandas as pd
# import pytask
# from dcegm.pre_processing.setup_model import load_and_setup_model
# from dcegm.solve import get_solve_func_for_model

# from caregiving.config import BLD
# from caregiving.estimation.standard_errors import get_analytical_standard_errors
# from caregiving.model.budget import budget_constraint, create_savings_grid
# from caregiving.model.state_space import create_state_space_functions
# from caregiving.model.task_specify_model import get_options_dict
# from caregiving.model.utility_functions import (
#     create_final_period_utility_functions,
#     create_utility_functions,
# )
# from caregiving.simulation.initial_conditions import draw_initial_states

# PARAMS = {
#     "beta": 0.959,
#     "rho": 0.8,
#     "lambda": 1,
#     "sigma": 0.5364562201,
#     "interest_rate": 0.04,
#     "utility_leisure_constant": 1,
#     "utility_leisure_age": 1,
#     "disutility_part_time": -5,
#     "disutility_full_time": -6,
#     # caregiving
#     "utility_informal_care_parent_medium_health": 2,
#     "utility_informal_care_parent_bad_health": 1,
#     "utility_formal_care_parent_medium_health": 0.7,
#     "utility_formal_care_parent_bad_health": 1,
#     "utility_combination_care_parent_medium_health": -0.8,
#     "utility_combination_care_parent_bad_health": -1.5,
#     # caregiving if sibling present
#     "utility_informal_care_medium_health_sibling": 2.5,
#     "utility_informal_care_bad_health_sibling": 2,
#     "utility_formal_care_medium_health_sibling": 1,
#     "utility_formal_care_bad_health_sibling": 1,
#     "utility_combination_care_medium_health_sibling": -0.2,
#     "utility_combination_care_bad_health_sibling": -0.4,
#     # part-time job offer
#     "part_time_constant": -2.568584,
#     "part_time_not_working_last_period": 0.3201395,
#     "part_time_high_education": 0.1691369,
#     "part_time_above_retirement_age": -1.9976496,
#     # full-time job offer
#     "full_time_constant": -2.445238,
#     "full_time_not_working_last_period": -0.9964007,
#     "full_time_high_education": 0.3019138,
#     "full_time_above_retirement_age": -2.6571659,
# }

# PROGRESS = {
#     "rho": 1.98,
#     "beta": 0.959,
#     "sigma": 0.5364562201,
#     "lambda": 1.0,
#     "interest_rate": 0.04,
#     "utility_leisure_constant": 3.2194001905695693,
#     "utility_leisure_age": 0.04691636386597703,
#     "utility_leisure_age_squared": -0.006495755962584587,
#     "disutility_part_time": -1.9337796413959372,
#     "disutility_full_time": -5.282394222692581,
#     "utility_informal_care_parent_medium_health": -0.4089331199248291,
#     "utility_informal_care_parent_bad_health": -0.3851096018741167,
#     "utility_formal_care_parent_medium_health": 0.4045081430627899,
#     "utility_formal_care_parent_bad_health": -0.4575685368898893,
#     "utility_combination_care_parent_medium_health": -4.112870982901066,
#     "utility_combination_care_parent_bad_health": -2.6130289452563393,
#     "utility_informal_care_medium_health_sibling": 1.815903823857372,
#     "utility_informal_care_bad_health_sibling": 2.439680402899742,
#     "utility_formal_care_medium_health_sibling": 0.6975043998652197,
#     "utility_formal_care_bad_health_sibling": 0.9263483706654374,
#     "utility_combination_care_medium_health_sibling": -1.6125665883276101,
#     "utility_combination_care_bad_health_sibling": -1.6952982282449929,
#     "part_time_constant": -2.568584,
#     "part_time_not_working_last_period": 0.3201395,
#     "part_time_high_education": 0.1691369,
#     "part_time_above_retirement_age": -1.9976496,
#     "full_time_constant": -2.445238,
#     "full_time_not_working_last_period": -0.9964007,
#     "full_time_high_education": 0.3019138,
#     "full_time_above_retirement_age": -2.6571659,
# }
# FIXED_PARAMS = ["beta", "rho", "lambda", "sigma", "interest_rate"]


# @pytask.mark.skip(reason="No local GPU.")
# def task_estimate_standard_errors():
#     """Estimate standard errors.

#     PARAMS = load_dict_from_pickle(BLD / "output" / "result_params.pkl")

#     """
#     path_to_emp_moments = BLD / "moments" / "empirical_moments_long.csv"
#     path_to_emp_var = BLD / "moments" / "empirical_moments_var_long.csv"
#     path_to_model = BLD / "model" / "model_short_exp.pkl"

#     emp_moments = jnp.asarray(
#         pd.read_csv(path_to_emp_moments, index_col=0).iloc[:, 0]
#     )
#     emp_var = jnp.asarray(pd.read_csv(path_to_emp_var, index_col=0).iloc[:, 0])

#     options = get_options_dict()

#     params = {key: val for key, val in PROGRESS.items() if key not in FIXED_PARAMS}
#     params_fixed = {key: val for key, val in PROGRESS.items() if key in FIXED_PARAMS}

#     model_loaded = load_and_setup_model(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         path=path_to_model,
#     )

#     exog_savings_grid = create_savings_grid()

#     func = get_solve_func_for_model(
#         model=model_loaded,
#         exog_savings_grid=exog_savings_grid,
#         options=options,
#     )

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

#     return get_analytical_standard_errors(
#         params=params,
#         params_fixed=params_fixed,
#         options=options,
#         emp_moments=emp_moments,
#         emp_var=emp_var,
#         model_loaded=model_loaded,
#         solve_func=func,
#         initial_states=initial_states,
#         initial_resources=initial_resources,
#     )
