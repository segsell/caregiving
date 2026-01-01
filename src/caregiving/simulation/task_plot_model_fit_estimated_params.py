"""Plot model fit between empirical and simulated data using estimated parameters."""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
import yaml
from pytask import Product

import dcegm
from caregiving.config import BLD
from caregiving.model.shared import (
    DEAD,
    INFORMAL_CARE,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_INFORMAL_CARE,
    NOT_WORKING,
    NOT_WORKING_CHOICES,
    RETIREMENT,
    SCALE_CAREGIVER_SHARE,
    SEX,
    WEALTH_QUANTILE_CUTOFF,
    WORK,
    WORK_CHOICES,
)
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.moments.task_create_soep_moments import (
    create_df_caregivers,
    create_df_non_caregivers,
    create_df_wealth,
    create_df_with_caregivers,
)
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
    plot_transition_counts_by_age,
    plot_transitions_by_age,
    plot_transitions_by_age_bins,
    plot_wealth_by_age_and_education,
    plot_wealth_by_age_bins_and_education,
)


@pytask.mark.baseline_model
@pytask.mark.model_fit_estimated_params
def task_plot_model_fit_estimated_params(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_empirical_moments: Path = BLD / "moments" / "moments_full.csv",
    path_to_empirical_data: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_caregivers_sample: Path = BLD
    / "data"
    / "soep_structural_caregivers_sample.csv",
    path_to_save_wealth_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "average_wealth.png",
    path_to_save_wealth_age_bins_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "average_wealth_age_bins.png",
    path_to_save_savings_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "average_savings.png",
    path_to_save_labor_shares_by_educ_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "labor_shares_by_educ_and_age.png",
    path_to_save_labor_shares_caregivers_by_age: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "labor_shares_caregivers_by_age.png",
    # path_to_save_labor_shares_caregivers_by_age_bin: Annotated[Path, Product] = BLD
    # / "plots"
    # / "model_fit_estimated_params"
    # / "labor_shares_caregivers_by_age_bin.png",
    save_labor_shares_caregivers_by_educ_and_age_bin: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "labor_shares_caregivers_by_educ_and_age_bin.png",
    save_labor_shares_light_caregivers_by_educ_and_age_bin: Annotated[
        Path, Product
    ] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "labor_shares_light_caregivers_by_educ_and_age_bin.png",
    save_labor_shares_intensive_caregivers_by_educ_and_age_bin: Annotated[
        Path, Product
    ] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "labor_shares_intensive_caregivers_by_educ_and_age_bin.png",
    # path_to_save_caregiver_share_by_age_plot: Annotated[Path, Product] = BLD
    # / "plots"
    # / "model_fit_estimated_params"
    # / "share_caregivers_by_age.png",
    path_to_save_caregiver_share_by_age_bin_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "share_caregivers_by_age_bin.png",
    path_to_save_care_demand_by_age_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "simulated_care_demand_by_age.png",
    path_to_save_work_transition_age_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "work_transitions_by_edu_and_age.png",
    path_to_save_work_transition_age_bin_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "work_transitions_by_edu_and_age_bin.png",
    path_to_save_work_transition_counts_age_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "work_transition_counts_by_edu_and_age.png",
    path_to_save_caregiving_transition_age_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "caregiving_transitions_by_age.png",
    path_to_save_caregiving_transition_age_bin_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "caregiving_transitions_by_age_bin.png",
    path_to_save_avg_caregiving_years: Annotated[Path, Product] = BLD
    / "plots"
    / "model_fit_estimated_params"
    / "average_caregiving_years.csv",
) -> None:
    """Plot model fit using estimated parameters."""

    specs = pickle.load(path_to_specs.open("rb"))

    start_age_caregivers = specs["start_age_caregiving"]
    end_age = specs["end_age_msm"]
    start_year = 2001
    end_year = 2019

    emp_moms = pd.read_csv(path_to_empirical_moments, index_col=[0]).squeeze("columns")

    # Load full datasets
    df_emp_full = pd.read_csv(path_to_empirical_data, index_col=[0])
    df_caregivers_full = pd.read_csv(path_to_caregivers_sample, index_col=[0])

    # Create standardized subsamples using shared functions
    df_emp = create_df_non_caregivers(
        df_full=df_emp_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age,
    )
    df_emp_with_caregivers = create_df_with_caregivers(
        df_full=df_emp_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age,
    )
    df_emp_caregivers = create_df_caregivers(
        df_caregivers_full=df_caregivers_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age,
    )
    df_emp_wealth = create_df_wealth(df_full=df_emp_full, specs=specs)

    # =================================================================================
    # Simulated data
    # =================================================================================

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX
    df_sim["age"] = df_sim["period"] + specs["start_age"]
    df_sim = df_sim[df_sim["health"] != DEAD].copy()

    # Subsets empirical
    df_emp_light_caregivers = df_emp_caregivers.loc[
        df_emp_caregivers["light_care"] == 1
    ]
    df_emp_intensive_caregivers = df_emp_caregivers.loc[
        df_emp_caregivers["intensive_care"] == 1
    ]

    df_sim_caregivers = df_sim.loc[
        df_sim["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
    ]
    df_sim_light_caregivers = df_sim.loc[
        df_sim["choice"].isin(np.asarray(LIGHT_INFORMAL_CARE).tolist())
    ]
    df_sim_intensive_caregivers = df_sim.loc[
        df_sim["choice"].isin(np.asarray(INTENSIVE_INFORMAL_CARE).tolist())
    ]

    # Wealth by age
    plot_wealth_by_age_and_education(
        data_emp=df_emp_wealth,
        data_sim=df_sim,
        specs=specs,
        wealth_var_emp="adjusted_wealth",
        wealth_var_sim="assets_begin_of_period",
        median=False,
        age_min=30,
        age_max=89,
        path_to_save_plot=path_to_save_wealth_plot,
    )
    plot_wealth_by_age_bins_and_education(
        data_emp=df_emp_wealth,
        data_sim=df_sim,
        specs=specs,
        wealth_var_emp="adjusted_wealth",
        wealth_var_sim="assets_begin_of_period",
        median=False,
        age_min=30,
        age_max=89,
        bin_width=5,
        path_to_save_plot=path_to_save_wealth_age_bins_plot,
    )
    plot_average_savings_decision(
        df_sim, path_to_save_savings_plot, end_age=specs["end_age"]
    )

    plot_choice_shares_by_education(
        df_emp,
        df_sim,
        specs,
        path_to_save_plot=path_to_save_labor_shares_by_educ_plot,
    )
    test_choice_shares_sum_to_one(df_emp, df_sim, specs)

    plot_job_offer_share_by_age(
        df_sim,
        path_to_save_plot=BLD
        / "plots"
        / "model_fit_estimated_params"
        / "simulated_job_offer",
    )

    plot_choice_shares_by_education(
        df_emp_caregivers,
        df_sim_caregivers,
        specs,
        path_to_save_plot=path_to_save_labor_shares_caregivers_by_age,
    )

    # Define age range for caregivers (40-75) and use 3-year bins
    end_age = 69

    plot_choice_shares_by_education_age_bins(
        df_emp_caregivers,
        df_sim_caregivers,
        specs,
        age_min=start_age_caregivers,
        age_max=end_age,
        bin_width=3,
        age_bin_ticks=False,
        path_to_save_plot=save_labor_shares_caregivers_by_educ_and_age_bin,
    )
    plot_choice_shares_by_education_age_bins(
        df_emp_light_caregivers,
        df_sim_light_caregivers,
        specs,
        age_min=start_age_caregivers,
        age_max=end_age,
        bin_width=3,
        age_bin_ticks=False,
        path_to_save_plot=save_labor_shares_light_caregivers_by_educ_and_age_bin,
    )
    plot_choice_shares_by_education_age_bins(
        df_emp_intensive_caregivers,
        df_sim_intensive_caregivers,
        specs,
        age_min=start_age_caregivers,
        age_max=end_age,
        bin_width=3,
        age_bin_ticks=False,
        path_to_save_plot=save_labor_shares_intensive_caregivers_by_educ_and_age_bin,
    )

    plot_caregiver_shares_by_age_bins(
        emp_moms,
        df_sim,
        specs,
        choice_set=INFORMAL_CARE,
        age_min=40,
        age_max=75,
        scale=SCALE_CAREGIVER_SHARE,
        path_to_save_plot=path_to_save_caregiver_share_by_age_bin_plot,
    )

    plot_simulated_care_demand_by_age(
        df_sim,
        specs,
        age_min=40,
        age_max=80,
        path_to_save_plot=path_to_save_care_demand_by_age_plot,
    )

    # Calculate average number of caregiving years, conditional on being a caregiver
    AGE_FOCUS = 75
    # 1. Agents alive / observed at the focus age
    ids_at_age = df_sim.loc[
        (df_sim["age"] == AGE_FOCUS) & (df_sim["health"] != DEAD), "agent"
    ].unique()

    # 2. Keep their entire life histories
    sub = df_sim.loc[df_sim["agent"].isin(ids_at_age)].copy()

    # 3. Flag caregiver years via choice ∈ INFORMAL_CARE
    care_codes = np.asarray(INFORMAL_CARE).tolist()
    sub["is_care"] = sub["choice"].isin(care_codes)

    # 4. Person-level aggregates
    agg = sub.groupby("agent")["is_care"].agg(
        care_sum="sum", care_ever="any"  # years with is_care == True
    )  # at least one caregiver year

    mean_care_years = agg.loc[agg["care_ever"], "care_sum"].mean()

    # Save to CSV
    result_df = pd.DataFrame(
        {
            "age_focus": [AGE_FOCUS],
            "average_caregiving_years_conditional_on_caregiver": [mean_care_years],
        }
    )
    result_df.to_csv(path_to_save_avg_caregiving_years, index=False)

    df_sim["informal_care"] = df_sim["choice"].isin(np.asarray(INFORMAL_CARE))
    share_informal_care = df_sim.loc[df_sim["care_demand"] == 1, "informal_care"].mean()
    print(f"Share informal caregivers (cond. on care demand): {share_informal_care}")

    share_caregivers_high_edu = df_sim.loc[
        (df_sim["informal_care"] == 1), "education"
    ].mean()
    print(f"Share high education (cond. on informal care): {share_caregivers_high_edu}")

    # =================================================================================
    # Transition probabilities
    # =================================================================================
    _working_counts = count_working_after_max_ret_age(df_sim, specs)

    # Drop all agents (entirely) who work after the maximum retirement age
    # Identify agents who work (not retired) after max_ret_age
    retirement_values = RETIREMENT.ravel().tolist()
    max_ret_age = specs["max_ret_age"]
    agents_working_after_ret = df_sim.loc[
        (~df_sim["choice"].isin(retirement_values)) & (df_sim["age"] > max_ret_age)
    ]["agent"].unique()

    # Drop all rows for those agents
    df_sim = df_sim[~df_sim["agent"].isin(agents_working_after_ret)]

    states_sim = {
        "not_working": NOT_WORKING,
        "working": WORK,
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
        df_emp_with_caregivers,
        df_sim,
        specs,
        state_labels,
        states_emp=states_emp,
        states_sim=states_sim,
        one_way=True,
        path_to_save_plot=path_to_save_work_transition_age_plot,
    )
    plot_transitions_by_age_bins(
        df_emp_with_caregivers,
        df_sim,
        specs,
        state_labels,
        states_emp=states_emp,
        states_sim=states_sim,
        bin_width=5,
        one_way=True,
        path_to_save_plot=path_to_save_work_transition_age_bin_plot,
    )
    plot_transition_counts_by_age(
        df_emp_with_caregivers,
        df_sim,
        specs,
        state_labels,
        states_emp=states_emp,
        states_sim=states_sim,
        one_way=True,
        path_to_save_plot=path_to_save_work_transition_counts_age_plot,
    )

    # =================================================================================
    # Caregiving transitions
    # =================================================================================
    # Create temporary choice columns for empirical data (from any_care)
    df_emp_care = df_emp_with_caregivers.copy()
    if "any_care" in df_emp_care.columns and "lagged_any_care" in df_emp_care.columns:
        df_emp_care["choice"] = df_emp_care["any_care"]
        df_emp_care["lagged_choice"] = df_emp_care["lagged_any_care"]
    else:
        # If columns don't exist, create them (assuming they should be there)
        raise ValueError(
            "df_emp_with_caregivers must have 'any_care' and 'lagged_any_care' columns"
        )

    # Caregiving states: empirical uses binary (0/1), simulated uses choice codes
    states_caregiving_emp = {
        "no_informal_care": [0],  # any_care == 0
        "informal_care": [1],  # any_care == 1
    }
    states_caregiving_sim = {
        "no_informal_care": NO_INFORMAL_CARE,
        "informal_care": INFORMAL_CARE,
    }
    state_labels_caregiving = {
        "no_informal_care": "No Informal Care",
        "informal_care": "Informal Care",
    }

    plot_transitions_by_age(
        df_emp_care,
        df_sim,
        specs,
        state_labels_caregiving,
        states_emp=states_caregiving_emp,
        states_sim=states_caregiving_sim,
        age_min=specs["start_age_caregiving"],
        one_way=True,
        path_to_save_plot=path_to_save_caregiving_transition_age_plot,
    )
    plot_transitions_by_age_bins(
        df_emp_care,
        df_sim,
        specs,
        state_labels_caregiving,
        states_emp=states_caregiving_emp,
        states_sim=states_caregiving_sim,
        age_min=specs["start_age_caregiving"],
        bin_width=5,
        one_way=True,
        path_to_save_plot=path_to_save_caregiving_transition_age_bin_plot,
    )


def count_working_after_max_ret_age(data_sim, specs, max_ret_age=None):
    """
    Count the number of working individuals per age and education type
    after the maximum retirement age.

    Parameters
    ----------
    data_sim : pd.DataFrame
        Simulated data with columns "age", "choice", "education", and "agent".
    specs : dict
        Model specifications. Must contain "max_ret_age" if max_ret_age is None.
    max_ret_age : int, optional
        Maximum retirement age. If None, uses specs["max_ret_age"].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns "age", "education", and "n_working_individuals",
        showing the count of unique working individuals per age and education type.
    """
    if max_ret_age is None:
        max_ret_age = specs["max_ret_age"]

    # Filter for ages after max_ret_age
    df_after_ret = data_sim[data_sim["age"] >= max_ret_age].copy()

    # Filter for working individuals (choice in WORK)
    work_codes = np.asarray(WORK).tolist()
    df_working = df_after_ret[df_after_ret["choice"].isin(work_codes)].copy()

    # Count unique individuals per age and education
    counts = (
        df_working.groupby(["age", "education"])["agent"]
        .nunique()
        .reset_index(name="n_working_individuals")
    )

    # Ensure both education types are represented for each age (fill with 0 if missing)
    if len(counts) > 0:
        all_ages = counts["age"].unique()
        all_educations = [0, 1]  # Low and high education
        full_index = pd.MultiIndex.from_product(
            [all_ages, all_educations], names=["age", "education"]
        )
        counts = (
            counts.set_index(["age", "education"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

    return counts


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
