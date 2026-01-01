"""Plot model fit between empirical and simulated data for no care demand model."""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
import yaml
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    NOT_WORKING_CHOICES,
    SEX,
    WEALTH_QUANTILE_CUTOFF,
    WORK_CHOICES,
)
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
    plot_average_wealth,
    plot_caregiver_shares_by_age,
    plot_caregiver_shares_by_age_bins,
    plot_choice_shares,
    plot_choice_shares_by_education,
    plot_choice_shares_by_education_age_bins,
    plot_choice_shares_overall,
    plot_choice_shares_overall_age_bins,
    plot_choice_shares_single,
    plot_job_offer_share_by_age,
    plot_simulated_care_demand_by_age,
    plot_states,
    plot_transitions_by_age,
    plot_transitions_by_age_bins,
    plot_wealth_by_age_and_education,
    plot_wealth_by_age_bins_and_education,
)


@pytask.mark.no_care_demand_model
def task_plot_model_fit_no_care_demand(  # noqa: PLR0915
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
    / "model_fit_no_care_demand"
    / "average_wealth.png",
    path_to_save_wealth_age_bins_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_no_care_demand"
    / "average_wealth_age_bins.png",
    path_to_save_savings_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_no_care_demand"
    / "average_savings.png",
    path_to_save_labor_shares_by_educ_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_no_care_demand"
    / "labor_shares_by_educ_and_age.png",
    path_to_save_work_transition_age_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_no_care_demand"
    / "work_transitions_by_edu_and_age.png",
    path_to_save_work_transition_age_bin_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_no_care_demand"
    / "work_transitions_by_edu_and_age_bin.png",
) -> None:
    """Plot model fit between empirical and simulated data for no care demand model."""

    specs = pickle.load(path_to_specs.open("rb"))

    df_emp = pd.read_csv(path_to_empirical_data, index_col=[0])
    df_emp_wealth = df_emp[(df_emp["wealth"].notna()) & (df_emp["sex"] == 1)].copy()

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX
    df_sim["age"] = df_sim["period"] + specs["start_age"]

    df_emp_wealth = adjust_and_trim_wealth_data(df=df_emp_wealth, specs=specs)
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

    plot_average_savings_decision(
        df_sim, path_to_save_savings_plot, end_age=specs["end_age"]
    )

    # Use no care demand choice groups for plotting
    choice_groups_sim = {
        0: RETIREMENT_NO_CARE_DEMAND,
        1: UNEMPLOYED_NO_CARE_DEMAND,
        2: PART_TIME_NO_CARE_DEMAND,
        3: FULL_TIME_NO_CARE_DEMAND,
    }

    plot_choice_shares_by_education(
        df_emp,
        df_sim,
        specs,
        choice_groups_sim=choice_groups_sim,
        path_to_save_plot=path_to_save_labor_shares_by_educ_plot,
    )
    test_choice_shares_sum_to_one(df_emp, df_sim, specs)

    plot_job_offer_share_by_age(
        df_sim,
        path_to_save_plot=BLD
        / "plots"
        / "model_fit_no_care_demand"
        / "simulated_job_offer",
    )

    states_sim = {
        "not_working": NOT_WORKING_NO_CARE_DEMAND,
        "working": WORK_NO_CARE_DEMAND,
    }
    states_emp = {
        "not_working": NOT_WORKING_CHOICES,
        "working": WORK_CHOICES,
    }
    state_labels = {
        "not_working": "Not Working",
        "working": "Work",
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


def test_choice_shares_sum_to_one(data_emp, data_sim, specs):
    """
    Test that, for each age, the sum of choice-specific shares equals 1
    in both empirical and simulated datasets.
    Parameters
    ----------
    data_emp : pd.DataFrame
        Empirical data with columns "period" and "choice".
    data_sim : pd.DataFrame
        Simulated data with columns "period" and "choice".
    specs : dict
        Must contain "start_age" (int).
    """
    for df, name in ((data_emp, "data_emp"), (data_sim, "data_sim")):

        df_gender = df[df["sex"] == SEX].copy()
        df_gender["age"] = df_gender["period"] + specs["start_age"]

        # compute normalized choice shares by age
        shares = (
            df_gender.groupby("age")["choice"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

        # sum across choices for each age
        sum_per_age = shares.sum(axis=1)

        # assert all sums are (approximately) 1
        if not np.allclose(sum_per_age.values, 1.0, atol=1e-8):
            bad = sum_per_age[~np.isclose(sum_per_age, 1.0)]
            raise AssertionError(
                f"In {name}, choice shares do not sum to 1 at ages:\n{bad}"
            )
