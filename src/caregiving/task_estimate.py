# """Estimate the model using the method of simulated moments."""

# from functools import partial

# import estimagic as em
# import jax.numpy as jnp
# import pandas as pd
# import pytask
# from dcegm.pre_processing.setup_model import load_and_setup_model
# from dcegm.solve import get_solve_func_for_model

# from caregiving.config import BLD
# from caregiving.estimate import criterion_solve_and_simulate
# from caregiving.model.budget import budget_constraint, create_savings_grid
# from caregiving.model.state_space import create_state_space_functions
# from caregiving.model.task_specify_model import get_options_dict
# from caregiving.model.utility_functions import (
#     create_final_period_utility_functions,
#     create_utility_functions,
# )
# from caregiving.simulation.initial_conditions import draw_initial_states
# from caregiving.utils import save_dict_to_pickle

# START_PARAMS = {
#     "beta": 1.98,  # Adda et al (2017)
#     "rho": 0.959,  # Adda et al (2017)
#     "sigma": 0.55,
#     "lambda": 1,
#     "interest_rate": 0.04,  # Adda et al (2017)
#     "utility_leisure_constant": 2,
#     "utility_leisure_age": 1,
#     "disutility_part_time": -3,
#     "disutility_full_time": -5,
#     # caregiving
#     "utility_informal_care_parent_medium_health": 2,
#     "utility_informal_care_parent_bad_health": 1,
#     "utility_formal_care_parent_medium_health": 0.7,
#     "utility_formal_care_parent_bad_health": 2,
#     "utility_combination_care_parent_medium_health": -0.8,
#     "utility_combination_care_parent_bad_health": -1.5,
#     # caregiving if sibling present
#     "utility_informal_care_medium_health_sibling": 2.5,
#     "utility_informal_care_bad_health_sibling": 2,
#     "utility_formal_care_medium_health_sibling": 1,
#     "utility_formal_care_bad_health_sibling": 1,
#     "utility_combination_care_medium_health_sibling": -0.2,
#     "utility_combination_care_bad_health_sibling": -0.4,
# }

# FIXED_PARAMS = ["rho", "beta", "lambda", "sigma", "interest_rate"]


# @pytask.mark.skip(reason="Do estimation on Colab GPU")
# def task_estimate_msm():
#     path_to_save_params = BLD / "estimation" / "result_params.pkl"
#     path_to_save_criterion_value = BLD / "estimation" / "result_criterion.pkl"
#     path_to_save_convergence_report = BLD / "estimation" / "result_convergence.pkl"
#     path_to_save_history = BLD / "estimation" / "result_history.pkl"

#     path_to_model = BLD / "model" / "model.pkl"
#     path_to_empirical_moments = BLD / "moments" / "empirical_moments.csv"

#     path_to_discrete_initial = (
#         f"{BLD}/moments/initial_discrete_conditions_at_age_40.csv"
#     )
#     path_high_educ = f"{BLD}/moments/real_wealth_age_39_high_educ.csv"
#     path_low_educ = f"{BLD}/moments/real_wealth_age_39_low_educ.csv"

#     options = get_options_dict()
#     n_agents = 100_000
#     seed = 2024

#     model_loaded = load_and_setup_model(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         path=path_to_model,
#     )
#     exog_savings_grid = create_savings_grid()

#     solve_func = get_solve_func_for_model(
#         model=model_loaded,
#         exog_savings_grid=exog_savings_grid,
#         options=options,
#     )

#     initial_conditions = pd.read_csv(path_to_discrete_initial, index_col=0)
#     initial_wealth_high_educ = jnp.asarray(pd.read_csv(path_high_educ)).ravel()
#     initial_wealth_low_educ = jnp.asarray(pd.read_csv(path_low_educ)).ravel()

#     initial_resources, initial_states = draw_initial_states(
#         initial_conditions,
#         initial_wealth_low_educ=initial_wealth_low_educ,
#         initial_wealth_high_educ=initial_wealth_high_educ,
#         n_agents=n_agents,
#         seed=seed,
#     )

#     emp_moments = jnp.asarray(
#         pd.read_csv(path_to_empirical_moments, index_col=0).iloc[:, 0],
#     )

#     chol_weights = jnp.eye(len(emp_moments))

#     criterion = partial(
#         criterion_solve_and_simulate,
#         options=options,
#         chol_weights=chol_weights,
#         model_loaded=model_loaded,
#         emp_moments=emp_moments,
#         solve_func=solve_func,
#         initial_states=initial_states,
#         initial_resources=initial_resources,
#     )

#     algo_options = {
#         "convergence.relative_criterion_change": 1e-14,
#         "stopping.max_iterations": 2,
#         "noisy": True,
#         "n_cores": 1,
#         "batch_size": 4,
#         "logging": "my_log.db",
#         "log_options": {"fast_logging": True},
#     }

#     result = em.minimize(
#         criterion=criterion,
#         params=START_PARAMS,
#         lower_bounds=lower_bounds,
#         upper_bounds=upper_bounds,
#         algorithm="tranquilo_ls",
#         algo_options=algo_options,
#         multistart=True,
#         multistart_options=options,
#         constraints={
#             "selector": lambda params: {key: params[key] for key in FIXED_PARAMS},
#             "type": "fixed",
#         },
#     )

#     save_dict_to_pickle(result.params, path_to_save_params)
#     save_dict_to_pickle(result.criterion, path_to_save_criterion_value)
#     save_dict_to_pickle(result.convergence, path_to_save_convergence_report)
#     save_dict_to_pickle(result.history, path_to_save_history)


# # ====================================================================================
# # Bounds
# # ====================================================================================

# lower_bounds = {
#     # Fixed params
#     "rho": 0,
#     "beta": 0.94,
#     "lambda": 1e-17,
#     "sigma": 1.0 - 0.1,
#     "interest_rate": 0.04 - 0.003,
#     "utility_leisure_constant": -5,
#     "utility_leisure_age": -5,
#     "disutility_part_time": -10,
#     "disutility_full_time": -10,
#     # caregiving
#     "utility_informal_care_parent_medium_health": -5,
#     "utility_informal_care_parent_bad_health": -5,
#     "utility_formal_care_parent_medium_health": -5,
#     "utility_formal_care_parent_bad_health": -5,
#     "utility_combination_care_parent_medium_health": -5,
#     "utility_combination_care_parent_bad_health": -5,
#     # caregiving if sibling present
#     "utility_informal_care_medium_health_sibling": -5,
#     "utility_informal_care_bad_health_sibling": -5,
#     "utility_formal_care_medium_health_sibling": -5,
#     "utility_formal_care_bad_health_sibling": -5,
#     "utility_combination_care_medium_health_sibling": -5,
#     "utility_combination_care_bad_health_sibling": -5,
# }

# upper_bounds = {
#     # Fixed params
#     "rho": 1.95 + 0.1,
#     "beta": 0.95,
#     "lambda": 1e-16,
#     "sigma": 1.0,
#     "interest_rate": 0.05,
#     "utility_leisure_constant": 5,
#     "utility_leisure_age": 5,
#     "disutility_part_time": 0,
#     "disutility_full_time": 0,
#     # caregiving
#     "utility_informal_care_parent_medium_health": 5,
#     "utility_informal_care_parent_bad_health": 5,
#     "utility_formal_care_parent_medium_health": 5,
#     "utility_formal_care_parent_bad_health": 5,
#     "utility_combination_care_parent_medium_health": 5,
#     "utility_combination_care_parent_bad_health": 5,
#     # caregiving if sibling present
#     "utility_informal_care_medium_health_sibling": 5,
#     "utility_informal_care_bad_health_sibling": 5,
#     "utility_formal_care_medium_health_sibling": 5,
#     "utility_formal_care_bad_health_sibling": 5,
#     "utility_combination_care_medium_health_sibling": 5,
#     "utility_combination_care_bad_health_sibling": 5,
# }


# multistart_options = {
#     # Set the number of points at which criterion is evaluated
#     # in the exploration phase
#     "n_samples": 5 * len(START_PARAMS),
#     # Pass in a DataFrame or array with a custom sample
#     # for the exploration phase.
#     "sample": None,
#     # Determine number of optimizations, relative to n_samples
#     "share_optimizations": 0.1,
#     # Determine distribution from which sample is drawn
#     "sampling_distribution": "uniform",
#     # Determine sampling method. Allowed: ["sobol", "random",
#     # "halton", "hammersley", "korobov", "latin_hypercube"]
#     "sampling_method": "sobol",
#     # Determine how start parameters for local optimizations are
#     # calculated. Allowed: ["tiktak", "linear"] or a custom
#     # function with arguments iteration, n_iterations, min_weight,
#     # and max_weight
#     "mixing_weight_method": "tiktak",
#     # Determine bounds on mixing weights.
#     "mixing_weight_bounds": (0.1, 0.995),
#     # Determine after how many re-discoveries of the currently best
#     # local optimum the multistart optimization converges.
#     "convergence.max_discoveries": 2,
#     # Determine the maximum relative distance two parameter vectors
#     # can have to be considered equal for convergence purposes:
#     "convergence.relative_params_tolerance": 0.01,
#     # Determine how many cores are used
#     "n_cores": 1,
#     # Determine which batch_evaluator is used:
#     "batch_evaluator": "joblib",
#     # Determine the batch size. It must be larger than n_cores.
#     # Setting the batch size larger than n_cores allows to reproduce
#     # the exact results of a highly parallel optimization on a smaller
#     # machine.
#     "batch_size": 4,
#     # Set the random seed:
#     "seed": None,
#     # Set how errors are handled during the exploration phase:
#     "exploration_error_handling": "continue",
#     # Set how errors are handled during the optimization phase:
#     "optimization_error_handling": "continue",
# }
