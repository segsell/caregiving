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
from caregiving.model.shared import FULL_TIME, NOT_WORKING, PART_TIME, SEX, WORK
from caregiving.simulation.plot_model_fit import (
    plot_average_savings_decision,
    plot_average_wealth,
    plot_choice_shares,
    plot_choice_shares_by_education,
    plot_choice_shares_single,
    plot_states,
    plot_transitions_by_age,
    plot_transitions_by_age_bins,
)


def task_plot_model_fit(
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_empirical_data: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_simulated_data: Path = BLD / "solve_and_simulate" / "simulated_data.pkl",
    path_to_save_wealth_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "average_wealth.png",
    path_to_save_savings_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "average_savings.png",
    path_to_save_single_choice_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "labor_shares_by_educ_and_age.png",
    path_to_save_work_transition_age_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "work_transitions_by_edu_and_age.png",
    path_to_save_work_transition_age_bin_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit"
    / "work_transitions_by_edu_and_age_bin.png",
) -> None:
    """Plot model fit between empirical and simulated data."""

    options = pickle.load(path_to_options.open("rb"))
    params = yaml.safe_load(path_to_start_params.open("rb"))

    model_full = load_and_setup_full_model_for_solution(
        options, path_to_model=path_to_solution_model
    )

    df_emp = pd.read_csv(path_to_empirical_data, index_col=[0])
    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = 1

    df_emp_prep, _states_dict = load_and_prep_data(
        data_emp=df_emp,
        model=model_full,
        start_params=params,
        drop_retirees=False,
    )

    specs = model_full["options"]["model_params"]

    plot_average_wealth(df_emp_prep, df_sim, specs, path_to_save_wealth_plot)
    plot_average_savings_decision(df_sim, path_to_save_savings_plot)

    # plot_choice_shares_single(
    #     df_emp, df_sim, specs, path_to_save_plot=path_to_save_single_choice_plot
    # )
    plot_choice_shares_by_education(
        df_emp, df_sim, specs, path_to_save_plot=path_to_save_single_choice_plot
    )
    test_choice_shares_sum_to_one(df_emp, df_sim, specs)

    # plot_choice_shares(df_emp, df_sim, specs)
    # discrete_state_names = model_full["model_structure"]["discrete_states_names"]
    # plot_states(df_emp, df_sim, discrete_state_names, specs)

    states = {
        "not_working": NOT_WORKING,
        "working": WORK,
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
        states,
        state_labels,
        one_way=True,
        path_to_save_plot=path_to_save_work_transition_age_plot,
    )
    plot_transitions_by_age_bins(
        df_emp,
        df_sim,
        specs,
        states,
        state_labels,
        bin_width=10,
        one_way=True,
        path_to_save_plot=path_to_save_work_transition_age_bin_plot,
    )

    data_emp = df_emp.copy()
    mask = (
        data_emp["lagged_choice"].isin(WORK)
        & data_emp["choice"].isin(WORK)
        & data_emp["age"].between(60, 70)
    )
    df_60_70 = data_emp[mask]
    counts = (  # noqa: F841
        df_60_70.groupby("age")
        .size()
        .reindex(range(60, 71), fill_value=0)
        .rename("count")
        .reset_index()
    )

    mask2 = (
        data_emp["lagged_choice"].isin(WORK)
        # & data_emp["choice"].isin(WORK)
        & data_emp["age"].between(60, 70)
    )
    df_60_70_2 = data_emp[mask2]
    counts_2 = (  # noqa: F841
        df_60_70_2.groupby("age")
        .size()
        .reindex(range(60, 71), fill_value=0)
        .rename("count")
        .reset_index()
    )

    mask3 = (
        data_emp["age"].between(60, 70)
        # & data_emp["choice"].isin(WORK)
    )
    df_60_70_3 = data_emp[mask3]
    counts_3 = (  # noqa: F841
        df_60_70_3.groupby("age")
        .size()
        .reindex(range(60, 71), fill_value=0)
        .rename("count")
        .reset_index()
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

        df_gender = df[df["sex"] == SEX]
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
