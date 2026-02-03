"""Plot model fit for the no-care-demand counterfactual."""

import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

import dcegm
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
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.moments.task_create_soep_moments import create_df_wealth
from caregiving.simulation.plot_model_fit import (
    plot_average_savings_decision,
    plot_choice_shares_by_education,
    plot_transitions_by_age,
    plot_transitions_by_age_bins,
    plot_wealth_by_age_and_education,
    plot_wealth_by_age_bins_and_education,
)


@pytask.mark.model_fit
def task_plot_model_fit_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_model: Path = BLD / "model" / "model.pkl",
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
    model_config = pickle.load(path_to_model_config.open("rb"))

    model_class = dcegm.setup_model(
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

    df_emp = pd.read_csv(path_to_empirical_data)
    df_emp_full = df_emp.copy()

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX

    df_emp_wealth = create_df_wealth(
        df_full=df_emp_full,
        model_class=model_class,
    )

    # Wealth fit
    plot_wealth_by_age_and_education(
        data_emp=df_emp_wealth,
        data_sim=df_sim,
        specs=specs,
        wealth_var_emp="adjusted_wealth",
        wealth_var_sim="assets_begin_of_period",
        median=False,
        age_min=30,
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
        age_min=30,
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
