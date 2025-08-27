"""Function that simulates the model for a given scenario."""

import numpy as np
import pandas as pd

from caregiving.model.shared import (
    DEAD,
    FULL_TIME_CHOICES,
    INFORMAL_CARE,
    PARENT_DEAD,
    PART_TIME,
    SEX,
)
from caregiving.model.state_space import construct_experience_years
from caregiving.utils import table
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods


def simulate_scenario(
    model,
    solution_endog_grid,
    solution_value,
    solution_policy,
    initial_states,
    wealth_agents,
    params,
    options,
    seed,
) -> pd.DataFrame:
    """Simulate the model for given parametrization and model solution."""

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        wealth_initial=wealth_agents,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        endog_grid_solved=solution_endog_grid,
        value_solved=solution_value,
        policy_solved=solution_policy,
        model=model,
        model_sim=model,
    )
    df = create_simulation_df(sim_dict)

    # Create additional variables
    model_params = options["model_params"]
    df["age"] = df.index.get_level_values("period") + model_params["start_age"]

    # Create experience years
    df["exp_years"] = construct_experience_years(
        experience=df["experience"].values,
        period=df.index.get_level_values("period").values,
        max_exp_diffs_per_period=model_params["max_exp_diffs_per_period"],
    )

    # Assign working hours for choice 1 (unemployed)
    df["working_hours"] = 0.0

    part_time_values = PART_TIME.ravel().tolist()
    full_time_values = FULL_TIME_CHOICES.ravel().tolist()

    sex_var = SEX

    for edu_var in range(model_params["n_education_types"]):

        # full-time
        df.loc[
            df["choice"].isin(full_time_values) & (df["education"] == edu_var),
            "working_hours",
        ] = model_params["av_annual_hours_ft"][sex_var, edu_var]

        # part-time
        df.loc[
            df["choice"].isin(part_time_values) & (df["education"] == edu_var),
            "working_hours",
        ] = model_params["av_annual_hours_pt"][sex_var, edu_var]

    # Create income vars:
    # First wealth at the beginning of period as the sum of savings and consumption
    df["wealth_at_beginning"] = df["savings"] + df["consumption"]

    # Then total income as the difference between wealth at the beginning
    # of next period and savings
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    df["income_wo_interest"] = df.groupby("agent")["wealth_at_beginning"].shift(
        -1
    ) - df["savings"] * (1 + params["interest_rate"])

    # periodic savings and savings rate
    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    # # Caregiving
    # df["informal_care"] = np.nan
    # df["formal_care"] = np.nan

    # alive_and_demand = (
    #     (df["health"] != DEAD)
    #     & (df["mother_health"] != PARENT_DEAD)
    #     & (df["care_demand"] == 1)
    # )

    # df.loc[alive_and_demand & (df["choice"].isin(INFORMAL_CARE)), "informal_care"] = 1
    # df.loc[alive_and_demand & (~df["choice"].isin(INFORMAL_CARE)),
    # "informal_care"] = 0

    # df.loc[alive_and_demand & (~df["choice"].isin(INFORMAL_CARE)), "formal_care"] = 1
    # df.loc[alive_and_demand & (df["choice"].isin(INFORMAL_CARE)), "formal_care"] = 0

    return df
