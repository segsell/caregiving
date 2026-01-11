# """Plot model fit between empirical and simulated data.

# Uses higher-retirement-age model.
# """

# import pickle
# from pathlib import Path
# from typing import Annotated

# import numpy as np
# import pandas as pd
# import pytask
# from pytask import Product

# from caregiving.config import BLD

# from caregiving.estimation.prepare_estimation import (
#     load_and_setup_full_model_for_solution,
# )
# from caregiving.model.shared import (
#     DEAD,
#     INFORMAL_CARE,
#     INTENSIVE_INFORMAL_CARE,
#     LIGHT_INFORMAL_CARE,
#     NO_INFORMAL_CARE,
#     NOT_WORKING,
#     NOT_WORKING_CHOICES,
#     PARENT_DEAD,
#     RETIREMENT,
#     SCALE_CAREGIVER_SHARE,
#     SEX,
#     WEALTH_QUANTILE_CUTOFF,
#     WORK,
#     WORK_CHOICES,
# )
# from caregiving.moments.task_create_soep_moments import (
#     adjust_and_trim_wealth_data,
#     create_df_with_caregivers,
# )
# from caregiving.simulation.plot_model_fit import (
#     plot_average_savings_decision,
#     plot_caregiver_shares_by_age_bins,
#     plot_choice_shares_by_education,
#     plot_choice_shares_by_education_age_bins,
#     plot_job_offer_share_by_age,
#     plot_simulated_care_demand_by_age,
#     plot_transitions_by_age,
#     plot_transitions_by_age_bins,
#     plot_wealth_by_age_and_education,
#     plot_wealth_by_age_bins_and_education,
# )


# @pytask.mark.model_fit_higher_ret_age
# def task_plot_model_fit_higher_ret_age(  # noqa: PLR0915
#     path_to_options: Path = BLD / "model" / "options_higher_ret_age.pkl",
#     path_to_solution_model: Path = BLD / "model" / "model_higher_ret_age.pkl",
#     path_to_estimated_params: Path = BLD
#     / "model"
#     / "params"
#     / "estimated_params_model.yaml",
#     path_to_simulated_data: Path = BLD
#     / "solve_and_simulate"
#     / "simulated_data_higher_ret_age_estimated_params.pkl",
#     path_to_empirical_moments: Path = BLD / "moments" / "moments_full.csv",
#     path_to_empirical_data: Path = BLD
#     / "data"
#     / "soep_structural_estimation_sample.csv",
#     path_to_caregivers_sample: Path = BLD
#     / "data"
#     / "soep_structural_caregivers_sample.csv",
#     path_to_save_wealth_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "average_wealth.png",
#     path_to_save_wealth_age_bins_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "average_wealth_age_bins.png",
#     path_to_save_savings_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "average_savings.png",
#     path_to_save_labor_shares_by_educ_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "labor_shares_by_educ_and_age.png",
#     path_to_save_labor_shares_caregivers_by_age: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "labor_shares_caregivers_by_age.png",
#     save_labor_shares_caregivers_by_educ_and_age_bin: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "labor_shares_caregivers_by_educ_and_age_bin.png",
#     save_labor_shares_light_caregivers_by_educ_and_age_bin: Annotated[
#         Path, Product
#     ] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "labor_shares_light_caregivers_by_educ_and_age_bin.png",
#     save_labor_shares_intensive_caregivers_by_educ_and_age_bin: Annotated[
#         Path, Product
#     ] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "labor_shares_intensive_caregivers_by_educ_and_age_bin.png",
#     path_to_save_caregiver_share_by_age_bin_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "share_caregivers_by_age_bin.png",
#     path_to_save_care_demand_by_age_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "simulated_care_demand_by_age.png",
#     path_to_save_work_transition_age_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "work_transitions_by_edu_and_age.png",
#     path_to_save_work_transition_age_bin_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "work_transitions_by_edu_and_age_bin.png",
#     path_to_save_caregiving_transition_age_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "caregiving_transitions_by_age.png",
#     path_to_save_caregiving_transition_age_bin_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_higher_ret_age"
#     / "caregiving_transitions_by_age_bin.png",
# ) -> None:
#     """Plot model fit using higher-retirement-age model with estimated parameters."""

#     options = pickle.load(path_to_options.open("rb"))

#     model_full = load_and_setup_full_model_for_solution(
#         options, path_to_model=path_to_solution_model
#     )
#     specs = model_full["options"]["model_params"]

#     start_age = specs["start_age"]
#     start_age_caregivers = specs["start_age_caregiving"]
#     end_age = specs["end_age_msm"]
#     start_year = 2001
#     end_year = 2019

#     emp_moms = pd.read_csv(path_to_empirical_moments, index_col=[0]).squeeze("columns")  # noqa: E501

#     # Load full datasets
#     df_emp_full = pd.read_csv(path_to_empirical_data, index_col=[0])
#     # df_caregivers_full is loaded but not used in this function
#     _df_caregivers_full = pd.read_csv(path_to_caregivers_sample, index_col=[0])

#     # Create standardized subsamples using shared functions
#     df_emp_with_caregivers = create_df_with_caregivers(
#         df_full=df_emp_full,
#         specs=specs,
#         start_year=start_year,
#         end_year=end_year,
#         end_age=end_age,
#     )

#     df_emp = pd.read_csv(path_to_empirical_data, index_col=[0])
#     df_emp_wealth = df_emp[(df_emp["wealth"].notna()) & (df_emp["sex"] == 1)].copy()

#     df_caregivers = pd.read_csv(path_to_caregivers_sample, index_col=[0])
#     df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
#     df_sim["sex"] = SEX
#     df_sim = df_sim[df_sim["health"] != DEAD].copy()

#     # Subsets empirical
#     df_emp_caregivers = df_caregivers.loc[df_caregivers["any_care"] == 1].copy()

#     df_emp_light_caregivers = df_emp_caregivers.loc[
#         df_emp_caregivers["light_care"] == 1
#     ]
#     df_emp_intensive_caregivers = df_emp_caregivers.loc[
#         df_emp_caregivers["intensive_care"] == 1
#     ]

#     df_sim_caregivers = df_sim.loc[
#         df_sim["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
#     ]
#     df_sim_light_caregivers = df_sim.loc[
#         df_sim["choice"].isin(np.asarray(LIGHT_INFORMAL_CARE).tolist())
#     ]
#     df_sim_intensive_caregivers = df_sim.loc[
#         df_sim["choice"].isin(np.asarray(INTENSIVE_INFORMAL_CARE).tolist())
#     ]

#     df_emp_wealth = adjust_and_trim_wealth_data(df=df_emp_wealth, specs=specs)
#     plot_wealth_by_age_and_education(
#         data_emp=df_emp_wealth,
#         data_sim=df_sim,
#         specs=specs,
#         wealth_var_emp="adjusted_wealth",
#         wealth_var_sim="wealth_at_beginning",
#         median=False,
#         age_min=30,
#         age_max=100,
#         path_to_save_plot=path_to_save_wealth_plot,
#     )

#     # Wealth by age bins
#     plot_wealth_by_age_bins_and_education(
#         data_emp=df_emp_wealth,
#         data_sim=df_sim,
#         specs=specs,
#         wealth_var_emp="adjusted_wealth",
#         wealth_var_sim="wealth_at_beginning",
#         median=False,
#         age_min=30,
#         age_max=79,
#         bin_width=5,
#         path_to_save_plot=path_to_save_wealth_age_bins_plot,
#     )
#     plot_average_savings_decision(df_sim, path_to_save_savings_plot)

#     plot_choice_shares_by_education(
#         df_emp, df_sim, specs, path_to_save_plot=path_to_save_labor_shares_by_educ_plot  # noqa: E501
#     )

#     plot_job_offer_share_by_age(
#         df_sim,
#         path_to_save_plot=BLD
#         / "plots"
#         / "model_fit_higher_ret_age"
#         / "simulated_job_offer",
#     )

#     plot_choice_shares_by_education(
#         df_emp_caregivers,
#         df_sim_caregivers,
#         specs,
#         path_to_save_plot=path_to_save_labor_shares_caregivers_by_age,
#     )

#     # Define age range for caregivers and use 3-year bins (mirror baseline)
#     start_age = specs["start_age"]
#     start_age_caregivers = specs["start_age_caregiving"]
#     end_age = 69

#     plot_choice_shares_by_education_age_bins(
#         df_emp_caregivers,
#         df_sim_caregivers,
#         specs,
#         age_min=start_age_caregivers,
#         age_max=end_age,
#         bin_width=3,
#         age_bin_ticks=False,
#         path_to_save_plot=save_labor_shares_caregivers_by_educ_and_age_bin,
#     )
#     plot_choice_shares_by_education_age_bins(
#         df_emp_light_caregivers,
#         df_sim_light_caregivers,
#         specs,
#         age_min=start_age_caregivers,
#         age_max=end_age,
#         bin_width=3,
#         age_bin_ticks=False,
#         path_to_save_plot=save_labor_shares_light_caregivers_by_educ_and_age_bin,
#     )
#     plot_choice_shares_by_education_age_bins(
#         df_emp_intensive_caregivers,
#         df_sim_intensive_caregivers,
#         specs,
#         age_min=start_age_caregivers,
#         age_max=end_age,
#         bin_width=3,
#         age_bin_ticks=False,
#         path_to_save_plot=save_labor_shares_intensive_caregivers_by_educ_and_age_bin,
#     )

#     plot_caregiver_shares_by_age_bins(
#         emp_moms,
#         df_sim,
#         specs,
#         choice_set=INFORMAL_CARE,
#         age_min=40,
#         age_max=75,
#         scale=SCALE_CAREGIVER_SHARE,
#         path_to_save_plot=path_to_save_caregiver_share_by_age_bin_plot,
#     )

#     plot_simulated_care_demand_by_age(
#         df_sim,
#         specs,
#         age_min=40,
#         age_max=80,
#         path_to_save_plot=path_to_save_care_demand_by_age_plot,
#     )

#     # Transition probabilities: work transitions
#     states_sim = {
#         "not_working": NOT_WORKING,
#         "working": WORK,
#     }
#     states_emp = {
#         "not_working": NOT_WORKING_CHOICES,
#         "working": WORK_CHOICES,
#     }
#     state_labels = {
#         "not_working": "Not Working",
#         "working": "Work",
#     }
#     plot_transitions_by_age(
#         df_emp_with_caregivers,
#         df_sim,
#         specs,
#         state_labels,
#         states_emp=states_emp,
#         states_sim=states_sim,
#         one_way=True,
#         path_to_save_plot=path_to_save_work_transition_age_plot,
#     )
#     plot_transitions_by_age_bins(
#         df_emp_with_caregivers,
#         df_sim,
#         specs,
#         state_labels,
#         states_emp=states_emp,
#         states_sim=states_sim,
#         bin_width=5,
#         one_way=True,
#         path_to_save_plot=path_to_save_work_transition_age_bin_plot,
#     )

#     # =================================================================================  # noqa: E501
#     # Caregiving transitions
#     # =================================================================================  # noqa: E501
#     # Create temporary choice columns for empirical data (from any_care)
#     df_emp_care = df_emp_with_caregivers.copy()
#     if "any_care" in df_emp_care.columns and "lagged_any_care" in df_emp_care.columns:
#         df_emp_care["choice"] = df_emp_care["any_care"]
#         df_emp_care["lagged_choice"] = df_emp_care["lagged_any_care"]
#     else:
#         # If columns don't exist, create them (assuming they should be there)
#         raise ValueError(
#             "df_emp_with_caregivers must have 'any_care' and 'lagged_any_care' columns"  # noqa: E501
#         )

#     # Caregiving states: empirical uses binary (0/1), simulated uses choice codes
#     states_caregiving_emp = {
#         "no_informal_care": [0],  # any_care == 0
#         "informal_care": [1],  # any_care == 1
#     }
#     states_caregiving_sim = {
#         "no_informal_care": NO_INFORMAL_CARE,
#         "informal_care": INFORMAL_CARE,
#     }
#     state_labels_caregiving = {
#         "no_informal_care": "No Informal Care",
#         "informal_care": "Informal Care",
#     }

#     plot_transitions_by_age(
#         df_emp_care,
#         df_sim,
#         specs,
#         state_labels_caregiving,
#         states_emp=states_caregiving_emp,
#         states_sim=states_caregiving_sim,
#         age_min=start_age_caregivers,
#         one_way=True,
#         path_to_save_plot=path_to_save_caregiving_transition_age_plot,
#     )
#     plot_transitions_by_age_bins(
#         df_emp_care,
#         df_sim,
#         specs,
#         state_labels_caregiving,
#         states_emp=states_caregiving_emp,
#         states_sim=states_caregiving_sim,
#         age_min=start_age_caregivers,
#         bin_width=5,
#         one_way=True,
#         path_to_save_plot=path_to_save_caregiving_transition_age_bin_plot,
#     )
