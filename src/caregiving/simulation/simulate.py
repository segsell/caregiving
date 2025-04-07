"""Function that simulates the model for a given scenario."""

import pandas as pd

from caregiving.model.shared import FULL_TIME, PART_TIME, SEX
from caregiving.model.state_space import construct_experience_years
from caregiving.utils import table
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods


def simulate_scenario(
    model,
    solution,
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
        endog_grid_solved=solution["endog_grid"],
        value_solved=solution["value"],
        policy_solved=solution["policy"],
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

    # Create income vars:
    # First wealth at the beginning of period as the sum of savings and consumption
    df["wealth_at_beginning"] = df["savings"] + df["consumption"]

    # Then total income as the difference between wealth at the beginning
    # of next period and savings
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    # Finally the savings decision
    df["savings_dec"] = df["total_income"] - df["consumption"]

    return df
