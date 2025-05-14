"""Plot model fit between empirical and simulated data."""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
import yaml
from pytask import Product

from caregiving.config import BLD
from caregiving.estimation.estimation_setup import (
    load_and_prep_data,
    load_and_setup_full_model_for_solution,
)
from caregiving.model.shared import (
    DEAD,
    INFORMAL_CARE,
    NOT_WORKING,
    PARENT_DEAD,
    SEX,
    WORK,
    WORK_CHOICES,
)
from caregiving.simulation.plot_model_fit import (
    plot_average_savings_decision,
    plot_average_wealth,
    plot_caregiver_shares_by_age,
    plot_choice_shares,
    plot_choice_shares_by_education,
    plot_choice_shares_by_education_age_bins,
    plot_choice_shares_overall,
    plot_choice_shares_overall_age_bins,
    plot_choice_shares_single,
    plot_simulated_care_demand_by_age,
    plot_states,
    plot_transitions_by_age,
    plot_transitions_by_age_bins,
)
from caregiving.simulation.task_plot_model_fit import test_choice_shares_sum_to_one


# def task_plot_model_fit_for_slides(  # noqa: PLR0915
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
#     path_to_empirical_data: Path = BLD
#     / "data"
#     / "soep_structural_estimation_sample.csv",
#     path_to_simulated_data: Path = BLD / "solve_and_simulate" / "simulated_data.pkl",
#     path_to_save_wealth_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "slides"
#     / "average_wealth.png",
#     path_to_save_savings_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "slides"
#     / "average_savings.png",
#     path_to_save_labor_shares_by_educ_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "slides"
#     / "labor_shares_by_educ_and_age.png",
#     path_to_save_labor_shares_caregivers_by_age: Annotated[Path, Product] = BLD
#     / "plots"
#     / "slides"
#     / "labor_shares_caregivers_by_age.png",
#     # path_to_save_labor_shares_caregivers_by_age_bin: Annotated[Path, Product] = BLD
#     # / "plots"
#     # / "slides"
#     # / "labor_shares_caregivers_by_age_bin.png",
#     path_to_save_labor_shares_caregivers_by_educ_and_age_plot: Annotated[
#         Path, Product
#     ] = BLD
#     / "plots"
#     / "slides"
#     / "labor_shares_caregivers_by_educ_and_age.png",
#     path_to_save_caregiver_share_by_age_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "slides"
#     / "share_caregivers_by_age.png",
#     path_to_save_care_demand_by_age_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "slides"
#     / "simulated_care_demand_by_age.png",
#     # path_to_save_work_transition_age_plot: Annotated[Path, Product] = BLD
#     # / "plots"
#     # / "slides"
#     # / "work_transitions_by_edu_and_age.png",
#     # path_to_save_work_transition_age_bin_plot: Annotated[Path, Product] = BLD
#     # / "plots"
#     # / "slides"
#     # / "work_transitions_by_edu_and_age_bin.png",
# ) -> None:
#     """Plot model fit between empirical and simulated data."""

#     options = pickle.load(path_to_options.open("rb"))
#     params = yaml.safe_load(path_to_start_params.open("rb"))

#     model_full = load_and_setup_full_model_for_solution(
#         options, path_to_model=path_to_solution_model
#     )

#     df_emp = pd.read_csv(path_to_empirical_data, index_col=[0])
#     df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
#     df_sim["sex"] = SEX

#     df_emp_prep, _states_dict = load_and_prep_data(
#         data_emp=df_emp,
#         model=model_full,
#         start_params=params,
#         drop_retirees=False,
#     )

#     specs = model_full["options"]["model_params"]

#     plot_average_wealth(df_emp_prep, df_sim, specs, path_to_save_wealth_plot)
#     plot_average_savings_decision(df_sim, path_to_save_savings_plot)

#     # plot_choice_shares_single(
#     #     df_emp, df_sim, specs, path_to_save_plot=path_to_save_single_choice_plot
#     # )
#     plot_choice_shares_by_education(
#         df_emp, df_sim, specs, path_to_save_plot=path_to_save_labor_shares_by_educ_plot
#     )
#     test_choice_shares_sum_to_one(df_emp, df_sim, specs)

#     df_emp_caregivers = df_emp.loc[df_emp["any_care"] == 1].copy()
#     df_sim_caregivers = df_sim.loc[
#         df_sim["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
#     ].copy()
#     start_age_care = 40
#     end_age_care = 80
#     plot_choice_shares_overall(
#         df_emp_caregivers,
#         df_sim_caregivers,
#         specs,
#         age_min=start_age_care,
#         age_max=specs["end_age_msm"],
#         path_to_save_plot=path_to_save_labor_shares_caregivers_by_age,
#     )

#     plot_choice_shares_by_education(
#         df_emp_caregivers,
#         df_sim_caregivers,
#         specs,
#         age_min=start_age_care,
#         age_max=specs["end_age_msm"],
#         path_to_save_plot=path_to_save_labor_shares_caregivers_by_educ_and_age_plot,
#     )

#     plot_caregiver_shares_by_age(
#         df_emp,
#         df_sim,
#         specs,
#         choice_set=INFORMAL_CARE,
#         age_min=start_age_care,
#         age_max=end_age_care,
#         path_to_save_plot=path_to_save_caregiver_share_by_age_plot,
#     )
#     plot_simulated_care_demand_by_age(
#         df_sim,
#         specs,
#         age_min=start_age_care,
#         age_max=end_age_care,
#         path_to_save_plot=path_to_save_care_demand_by_age_plot,
#     )
