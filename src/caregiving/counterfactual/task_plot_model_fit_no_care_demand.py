"""Plot model fit for the no-care-demand counterfactual."""

import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
import yaml
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import NOT_WORKING_CHOICES, SEX, WORK_CHOICES
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    NOT_WORKING_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)
from caregiving.moments.task_create_soep_moments import adjust_and_trim_wealth_data
from caregiving.simulation.plot_model_fit import (
    plot_average_savings_decision,
    plot_choice_shares_by_education,
    plot_transitions_by_age,
    plot_transitions_by_age_bins,
    plot_wealth_by_age_and_education,
    plot_wealth_by_age_bins_and_education,
)


@pytask.mark.skip()
@pytask.mark.model_fit
def task_plot_model_fit_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_empirical_moments: Path = BLD / "moments" / "moments_full.csv",
    path_to_empirical_data: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_save_wealth_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "counterfactual"
    / "average_wealth_no_care_demand.png",
    path_to_save_wealth_age_bins_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "counterfactual"
    / "average_wealth_age_bins_no_care_demand.png",
    path_to_save_savings_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "counterfactual"
    / "average_savings_no_care_demand.png",
    path_to_save_labor_shares_by_educ_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "counterfactual"
    / "labor_shares_by_educ_and_age_no_care_demand.png",
    path_to_save_work_transition_age_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "counterfactual"
    / "work_transitions_by_edu_and_age_no_care_demand.png",
    path_to_save_work_transition_age_bin_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "counterfactual"
    / "work_transitions_by_edu_and_age_bin_no_care_demand.png",
) -> None:

    specs = pickle.load(path_to_specs.open("rb"))

    df_emp = pd.read_csv(path_to_empirical_data)
    # df_emp_wealth = df_emp[["sex", "education", "age", "wealth"]].copy()
    df_emp_wealth = df_emp[(df_emp["wealth"].notna()) & (df_emp["sex"] == 1)].copy()

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    df_emp_wealth = adjust_and_trim_wealth_data(df=df_emp_wealth, specs=specs)

    # Wealth fit
    plot_wealth_by_age_and_education(
        data_emp=df_emp_wealth,
        data_sim=df_sim,
        specs=specs,
        wealth_var_emp="adjusted_wealth",
        wealth_var_sim="assets_begin_of_period",
        median=False,
        age_min=specs["start_age"],
        age_max=100,
        path_to_save_plot=path_to_save_wealth_plot,
    )

    # Wealth by age bins
    plot_wealth_by_age_bins_and_education(
        data_emp=df_emp_wealth,
        data_sim=df_sim,
        specs=specs,
        wealth_var_emp="adjusted_wealth",
        wealth_var_sim="assets_begin_of_period",
        median=False,
        age_min=specs["start_age"],
        age_max=79,
        bin_width=5,
        path_to_save_plot=path_to_save_wealth_age_bins_plot,
    )

    # Savings
    plot_average_savings_decision(df_sim, path_to_save_savings_plot)

    # Labor shares (no-care-demand has 4 choices; function handles generic mapping)

    choices_sim = {
        0: RETIREMENT_NO_CARE_DEMAND,
        1: UNEMPLOYED_NO_CARE_DEMAND,
        2: PART_TIME_NO_CARE_DEMAND,
        3: FULL_TIME_NO_CARE_DEMAND,
    }
    plot_choice_shares_by_education(
        df_emp,
        df_sim,
        specs,
        choice_groups_sim=choices_sim,
        path_to_save_plot=path_to_save_labor_shares_by_educ_plot,
    )

    states_sim = {
        "not_working": NOT_WORKING_NO_CARE_DEMAND,
        "working": WORK_NO_CARE_DEMAND,
        # "part_time": PART_TIME,
        # "full_time": FULL_TIME,
    }
    states_emp = {
        "not_working": NOT_WORKING_CHOICES,
        "working": WORK_CHOICES,
        # "part_time": PART_TIME,
        # "full_time": FULL_TIME,
    }
    state_labels = {
        "not_working": "Not Working",
        "working": "Work",
        # "part_time": "Part-time",
        # "full_time": "Full-time",
    }

    plot_transitions_by_age(
        df_emp,
        df_sim,
        specs,
        state_labels,
        states_emp=states_emp,
        states_sim=states_sim,
        one_way=True,
        path_to_save_plot=path_to_save_work_transition_age_plot,
    )
    plot_transitions_by_age_bins(
        df_emp,
        df_sim,
        specs,
        state_labels,
        states_emp=states_emp,
        states_sim=states_sim,
        bin_width=5,
        one_way=True,
        path_to_save_plot=path_to_save_work_transition_age_bin_plot,
    )
