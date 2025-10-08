"""Simulation helper for the no-care-demand counterfactual.

Mirrors the baseline simulate_scenario but uses the reduced 4-state
choice arrays from shared_no_care_demand to assign working hours, etc.
"""

import numpy as np
import pandas as pd
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods

from caregiving.model.shared import (
    SEX,
)
from caregiving.model.shared_no_care_demand import (
    PART_TIME,
    FULL_TIME,
)
from caregiving.model.state_space import construct_experience_years


def simulate_scenario_no_care_demand(
    model,
    solution,
    initial_states,
    wealth_agents,
    params,
    options,
    seed,
) -> pd.DataFrame:
    """Simulate the counterfactual model and return a DataFrame."""

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        wealth_initial=wealth_agents,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        endog_grid_solved=solution["endog_grid"],
        value_solved=solution["value"],
        policy_solved=solution["policy"],
        model=model,
        model_sim=model,
    )
    df = create_simulation_df(sim_dict)

    # Add derived variables
    model_params = options["model_params"]
    df["age"] = df.index.get_level_values("period") + model_params["start_age"]

    df["exp_years"] = construct_experience_years(
        experience=df["experience"].values,
        period=df.index.get_level_values("period").values,
        max_exp_diffs_per_period=model_params["max_exp_diffs_per_period"],
    )

    # Assign working hours
    df["working_hours"] = 0.0
    part_time_values = PART_TIME.ravel().tolist()
    full_time_values = FULL_TIME.ravel().tolist()

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

    # Income variables
    df["wealth_at_beginning"] = df["savings"] + df["consumption"]
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    df["income_wo_interest"] = df.groupby("agent")["wealth_at_beginning"].shift(
        -1
    ) - df["savings"] * (1 + params["interest_rate"])

    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    # Mother age
    df["mother_age"] = (
        df["age"].to_numpy()
        + model_params["mother_age_diff"][
            df["has_sister"].to_numpy(), df["education"].to_numpy()
        ]
    )

    return df
