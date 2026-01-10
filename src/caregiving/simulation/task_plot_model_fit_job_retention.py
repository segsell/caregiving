# """Plot model fit between empirical and simulated data using job retention model."""

# import pickle
# from pathlib import Path
# from typing import Annotated

# import numpy as np
# import pandas as pd
# import pytask
# import yaml
# from pytask import Product

# from caregiving.config import BLD

# from caregiving.estimation.prepare_estimation import (
#     load_and_prep_data,
#     load_and_setup_full_model_for_solution,
# )
# from caregiving.model.shared import (
#     DEAD,
#     INFORMAL_CARE,
#     INTENSIVE_INFORMAL_CARE,
#     LIGHT_INFORMAL_CARE,
#     NOT_WORKING,
#     NOT_WORKING_CHOICES,
#     PARENT_DEAD,
#     SCALE_CAREGIVER_SHARE,
#     SEX,
#     WEALTH_QUANTILE_CUTOFF,
#     WORK,
#     WORK_CHOICES,
# )
# from caregiving.moments.task_create_soep_moments import adjust_and_trim_wealth_data
# from caregiving.simulation.plot_model_fit import (
#     plot_average_savings_decision,
#     plot_average_wealth,
#     plot_caregiver_shares_by_age,
#     plot_caregiver_shares_by_age_bins,
#     plot_choice_shares,
#     plot_choice_shares_by_education,
#     plot_choice_shares_by_education_age_bins,
#     plot_choice_shares_overall,
#     plot_choice_shares_overall_age_bins,
#     plot_choice_shares_single,
#     plot_job_offer_share_by_age,
#     plot_simulated_care_demand_by_age,
#     plot_states,
#     plot_transitions_by_age,
#     plot_transitions_by_age_bins,
#     plot_wealth_by_age_and_education,
#     plot_wealth_by_age_bins_and_education,
# )


# @pytask.mark.model_fit_job_retention
# def task_plot_model_fit_job_retention(  # noqa: PLR0915
#     path_to_options: Path = BLD / "model" / "options_job_retention.pkl",
#     path_to_solution_model: Path = BLD / "model" / "model_job_retention.pkl",
#     path_to_estimated_params: Path = BLD
#     / "model"
#     / "params"
#     / "estimated_params_model.yaml",
#     path_to_simulated_data: Path = BLD
#     / "solve_and_simulate"
#     / "simulated_data_job_retention_estimated_params.pkl",
#     path_to_empirical_moments: Path = BLD / "moments" / "moments_full.csv",
#     path_to_empirical_data: Path = BLD
#     / "data"
#     / "soep_structural_estimation_sample.csv",
#     path_to_caregivers_sample: Path = BLD
#     / "data"
#     / "soep_structural_caregivers_sample.csv",
#     path_to_save_wealth_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "average_wealth.png",
#     path_to_save_wealth_age_bins_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "average_wealth_age_bins.png",
#     path_to_save_savings_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "average_savings.png",
#     path_to_save_labor_shares_by_educ_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "labor_shares_by_educ_and_age.png",
#     path_to_save_labor_shares_caregivers_by_age: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "labor_shares_caregivers_by_age.png",
#     save_labor_shares_caregivers_by_educ_and_age_bin: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "labor_shares_caregivers_by_educ_and_age_bin.png",
#     save_labor_shares_light_caregivers_by_educ_and_age_bin: Annotated[
#         Path, Product
#     ] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "labor_shares_light_caregivers_by_educ_and_age_bin.png",
#     save_labor_shares_intensive_caregivers_by_educ_and_age_bin: Annotated[
#         Path, Product
#     ] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "labor_shares_intensive_caregivers_by_educ_and_age_bin.png",
#     path_to_save_caregiver_share_by_age_bin_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "share_caregivers_by_age_bin.png",
#     path_to_save_care_demand_by_age_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "simulated_care_demand_by_age.png",
#     path_to_save_work_transition_age_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "work_transitions_by_edu_and_age.png",
#     path_to_save_work_transition_age_bin_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_job_retention"
#     / "work_transitions_by_edu_and_age_bin.png",
# ) -> None:
#     """Plot model fit using job retention model with estimated parameters.

#     This function generates plots comparing empirical and simulated data
#     for the job retention counterfactual model. The job retention model
#     implements a policy where caregivers can keep their jobs even when
#     they reduce hours or become unemployed due to caregiving activities.

#     Args:
#         path_to_options: Path to the job retention model options
#         path_to_solution_model: Path to the job retention model for solution
#         path_to_estimated_params: Path to the estimated parameters
#         path_to_simulated_data: Path to simulated data from job retention model
#         path_to_empirical_moments: Path to empirical moments
#         path_to_empirical_data: Path to empirical data
#         path_to_caregivers_sample: Path to caregivers sample data
#         path_to_save_wealth_plot: Path to save wealth plot
#         path_to_save_wealth_age_bins_plot: Path to save wealth age bins plot
#         path_to_save_savings_plot: Path to save savings plot
#         path_to_save_labor_shares_by_educ_plot: Path to save labor shares by education
#             plot
#         path_to_save_labor_shares_caregivers_by_age: Path to save caregivers labor
#             shares plot
#         save_labor_shares_caregivers_by_educ_and_age_bin: Path to save caregivers by
#             education and age bin plot
#         save_labor_shares_light_caregivers_by_educ_and_age_bin: Path to save light
#             caregivers plot
#         save_labor_shares_intensive_caregivers_by_educ_and_age_bin: Path to save
#             intensive caregivers plot
#         path_to_save_caregiver_share_by_age_bin_plot: Path to save caregiver share by
#             age bin plot
#         path_to_save_care_demand_by_age_plot: Path to save care demand by age plot
#         path_to_save_work_transition_age_plot: Path to save work transitions by age
#             plot
#         path_to_save_work_transition_age_bin_plot: Path to save work transitions by
#             age bin plot
#     """

#     options = pickle.load(path_to_options.open("rb"))

#     model_full = load_and_setup_full_model_for_solution(
#         options, path_to_model=path_to_solution_model
#     )
#     specs = model_full["options"]["model_params"]

#     emp_moms = pd.read_csv(path_to_empirical_moments, index_col=[0]).squeeze("columns")  # noqa: E501

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

#     test_choice_shares_sum_to_one(df_emp, df_sim, specs)

#     plot_job_offer_share_by_age(
#         df_sim,
#         path_to_save_plot=BLD
#         / "plots"
#         / "model_fit_job_retention"
#         / "simulated_job_offer",
#     )

#     plot_choice_shares_by_education(
#         df_emp_caregivers,
#         df_sim_caregivers,
#         specs,
#         path_to_save_plot=path_to_save_labor_shares_caregivers_by_age,
#     )

#     plot_choice_shares_by_education_age_bins(
#         df_emp_caregivers,
#         df_sim_caregivers,
#         specs,
#         path_to_save_plot=save_labor_shares_caregivers_by_educ_and_age_bin,
#     )
#     plot_choice_shares_by_education_age_bins(
#         df_emp_light_caregivers,
#         df_sim_light_caregivers,
#         specs,
#         path_to_save_plot=save_labor_shares_light_caregivers_by_educ_and_age_bin,
#     )
#     plot_choice_shares_by_education_age_bins(
#         df_emp_intensive_caregivers,
#         df_sim_intensive_caregivers,
#         specs,
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

#     df_sim["informal_care"] = df_sim["choice"].isin(np.asarray(INFORMAL_CARE))
#     share_informal_care = df_sim.loc[df_sim["care_demand"] == 1, "informal_care"].mean()  # noqa: E501
#     print(f"Share informal caregivers (cond. on care demand): {share_informal_care}")

#     share_caregivers_high_edu = df_sim.loc[
#         (df_sim["informal_care"] == 1), "education"
#     ].mean()
#     print(f"Share high education (cond. on informal care): {share_caregivers_high_edu}")  # noqa: E501

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
#         df_emp,
#         df_sim,
#         specs,
#         state_labels,
#         states_emp=states_emp,
#         states_sim=states_sim,
#         one_way=True,
#         path_to_save_plot=path_to_save_work_transition_age_plot,
#     )
#     plot_transitions_by_age_bins(
#         df_emp,
#         df_sim,
#         specs,
#         state_labels,
#         states_emp=states_emp,
#         states_sim=states_sim,
#         bin_width=5,
#         one_way=True,
#         path_to_save_plot=path_to_save_work_transition_age_bin_plot,
#     )


# def test_choice_shares_sum_to_one(data_emp, data_sim, specs):
#     """
#     Test that, for each age, the sum of choice-specific shares equals 1
#     in both empirical and simulated datasets.
#     Parameters
#     ----------
#     data_emp : pd.DataFrame
#         Empirical data with columns "period" and "choice".
#     data_sim : pd.DataFrame
#         Simulated data with columns "period" and "choice".
#     specs : dict
#         Must contain "start_age" (int).
#     """
#     for df, name in ((data_emp, "data_emp"), (data_sim, "data_sim")):

#         df_gender = df[df["sex"] == SEX].copy()
#         df_gender["age"] = df_gender["period"] + specs["start_age"]

#         # compute normalized choice shares by age
#         shares = (
#             df_gender.groupby("age")["choice"]
#             .value_counts(normalize=True)
#             .unstack(fill_value=0)
#         )

#         # sum across choices for each age
#         sum_per_age = shares.sum(axis=1)

#         # assert all sums are (approximately) 1
#         if not np.allclose(sum_per_age.values, 1.0, atol=1e-8):
#             bad = sum_per_age[~np.isclose(sum_per_age, 1.0)]
#             raise AssertionError(
#                 f"In {name}, choice shares do not sum to 1 at ages:\n{bad}"
#             )
