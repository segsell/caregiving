"""Function that simulates the model with forced care_demand == 2 at age 50."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods

from caregiving.model.shared import (
    CARE_DEMAND_AND_NO_OTHER_SUPPLY,
    DEAD,
    FULL_TIME,
    PARENT_BAD_HEALTH,
    PARENT_DEAD,
    PART_TIME,
    SEX,
)
from caregiving.model.state_space import construct_experience_years
from caregiving.model.wealth_and_budget.wages import calc_labor_income_after_ssc
from caregiving.simulation.simulation_utils import (
    convert_states_to_dtypes,
    extract_discrete_states_at_period,
    extract_exogenous_states_at_period,
    extract_lagged_choice_at_period,
    extract_wealth_at_period,
    find_continuous_state_names,
    override_forced_states,
)


def simulate_scenario_forced_care_demand_at_50(
    model,
    solution,
    initial_states,
    wealth_agents,
    params,
    options,
    seed,
    forced_age: int = 50,
) -> pd.DataFrame:
    """Simulate the model with forced care_demand == 2 and mother_health == 0 at age 50.

    This function:
    1. Runs baseline simulation from start_age to forced_age - 1 (age 49)
    2. Extracts all state variables at age 49, including labor supply shares (choices)
    3. Uses those states as initial conditions for a new simulation starting at age 50
    4. Forces care_demand = 2 and mother_health = 0 at age 50
    5. Continues simulation from age 50 onwards

    Args:
        model: Model structure for simulation
        solution: Model solution (endog_grid, value, policy)
        initial_states: Initial discrete states for all agents
        wealth_agents: Initial wealth for all agents
        params: Model parameters
        options: Model options
        seed: Random seed
        forced_age: Age to force care_demand == 2 and mother_health == 0 (default: 50)

    Returns:
        DataFrame with simulation results (only from age 50 onwards)
    """

    # Calculate the period corresponding to forced_age
    start_age = options["model_params"]["start_age"]
    forced_period = forced_age - start_age

    if forced_period <= 0 or forced_period >= options["model_params"]["n_periods"]:
        raise ValueError(
            f"forced_age {forced_age} corresponds to period {forced_period}, "
            f"which is out of bounds"
        )

    # Step 1: Run baseline simulation up to age 50 (forced_period)
    # Note: simulate_all_periods simulates n_periods periods (0 to n_periods-1)
    # So if forced_period = 20, we want periods 0-20, which is 21 periods
    n_periods_to_age_50 = forced_period + 1
    sim_dict_baseline = simulate_all_periods(
        states_initial=initial_states,
        wealth_initial=wealth_agents,
        n_periods=n_periods_to_age_50,
        params=params,
        seed=seed,
        endog_grid_solved=solution["endog_grid"],
        value_solved=solution["value"],
        policy_solved=solution["policy"],
        model=model,
        model_sim=model,
    )

    # Step 2: Extract states at age 50 (forced_period) from baseline simulation
    # Create DataFrame to extract all states including choice, lagged_choice,
    # and exogenous processes
    df_baseline = create_simulation_df(sim_dict_baseline)
    df_baseline["age"] = df_baseline.index.get_level_values("period") + start_age

    # Get states at age 50 (period forced_period)
    df_age_50 = df_baseline[
        df_baseline.index.get_level_values("period") == forced_period
    ].copy()

    # Extract all state variables at age 50
    model_structure = model["model_structure"]
    discrete_state_space = model_structure["state_space_dict"]

    # Check for continuous states (like experience)
    continuous_states = model["options"]["state_space"].get("continuous_states", {})
    continuous_state_names = find_continuous_state_names(continuous_states)

    # Extract discrete states
    states_at_age_50 = extract_discrete_states_at_period(
        sim_dict_baseline, df_age_50, forced_period, initial_states
    )

    # Extract lagged_choice
    states_at_age_50["lagged_choice"] = extract_lagged_choice_at_period(
        df_baseline, df_age_50, forced_period
    )

    # Extract exogenous processes realized at age 50
    exogenous_states_dict = extract_exogenous_states_at_period(
        df_age_50, ["care_demand", "mother_health", "job_offer"]
    )
    states_at_age_50.update(exogenous_states_dict)

    # Step 3: Override care_demand = 2 and mother_health = 0 at age 50
    states_at_age_50 = override_forced_states(
        states_at_age_50,
        CARE_DEMAND_AND_NO_OTHER_SUPPLY,
        PARENT_BAD_HEALTH,
        PARENT_DEAD,
    )

    # Get wealth at beginning of age 50
    wealth_at_age_50 = extract_wealth_at_period(
        sim_dict_baseline, forced_period, wealth_agents
    )

    # Convert discrete states to proper dtypes
    states_at_age_50_dtype = convert_states_to_dtypes(
        states_at_age_50,
        discrete_state_space,
        continuous_state_names=continuous_state_names,
    )

    # Step 4: Run simulation starting from age 50 (period 0 of new simulation)
    # We use the baseline model (no modification needed since care_demand
    # is set in initial states)
    # We need to simulate from age 50 to the end
    total_n_periods = options["model_params"]["n_periods"]
    n_periods_from_age_50 = total_n_periods - forced_period

    # Verify the calculation
    if n_periods_from_age_50 <= 0:
        raise ValueError(
            f"Invalid period calculation: forced_period={forced_period}, "
            f"total_n_periods={total_n_periods}, "
            f"n_periods_from_age_50={n_periods_from_age_50}"
        )

    # Run simulation from age 50 onwards
    # The period in the new simulation starts at 0, but corresponds to age 50
    sim_dict_from_age_50 = simulate_all_periods(
        states_initial=states_at_age_50_dtype,
        wealth_initial=wealth_at_age_50,
        n_periods=n_periods_from_age_50,
        params=params,
        seed=seed + forced_period,  # Use different seed for age 50+ simulation
        endog_grid_solved=solution["endog_grid"],
        value_solved=solution["value"],
        policy_solved=solution["policy"],
        model=model,
        model_sim=model,
    )

    # Step 6: Convert simulation results to DataFrame
    # We only have the simulation from age 50 onwards
    df = create_simulation_df(sim_dict_from_age_50)

    # Adjust periods to match baseline:
    # period 0 in new simulation = period 20 (forced_period) in baseline
    # Reset index to adjust period values
    df = df.reset_index()
    df["period"] = df["period"] + forced_period
    df = df.set_index(["period", "agent"])

    # Create age column
    model_params = options["model_params"]
    df["age"] = df.index.get_level_values("period") + start_age

    # Verify forced states are set correctly at age 50 (period 0)
    mask_at_age_50 = df["age"] == forced_age
    df.loc[mask_at_age_50, "care_demand"] = 2
    mother_alive_mask = mask_at_age_50 & (df["mother_health"] != PARENT_DEAD)
    df.loc[mother_alive_mask, "mother_health"] = PARENT_BAD_HEALTH

    # Add additional variables (similar to simulate_scenario)
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
        df.loc[
            df["choice"].isin(full_time_values) & (df["education"] == edu_var),
            "working_hours",
        ] = model_params["av_annual_hours_ft"][sex_var, edu_var]
        df.loc[
            df["choice"].isin(part_time_values) & (df["education"] == edu_var),
            "working_hours",
        ] = model_params["av_annual_hours_pt"][sex_var, edu_var]

    # Create income variables
    df["wealth_at_beginning"] = df["savings"] + df["consumption"]
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    df["income_wo_interest"] = df.groupby("agent")["wealth_at_beginning"].shift(
        -1
    ) - df["savings"] * (1 + params["interest_rate"])
    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    df["mother_age"] = (
        df["age"].to_numpy()
        + model_params["mother_age_diff"][
            df["has_sister"].to_numpy(), df["education"].to_numpy()
        ]
    )

    # ===============================================================================
    # Gross labor income computation
    # ===============================================================================
    work_values = part_time_values + full_time_values

    # Convert pandas Series to numpy arrays for JAX
    lagged_choice_array = np.asarray(df["lagged_choice"])
    experience_years_array = np.asarray(df["exp_years"])
    education_array = np.asarray(df["education"])
    income_shock_array = np.asarray(df["income_shock"])

    # Vectorized gross labor income calculation
    vectorized_calc_gross_labor_income = jax.vmap(
        lambda lc, exp, edu, shock: calculate_gross_labor_income(
            lagged_choice=lc,
            experience_years=exp,
            education=edu,
            sex=sex_var,
            income_shock=shock,
            options=model_params,
        )
    )
    gross_labor_income_array = vectorized_calc_gross_labor_income(
        lagged_choice_array,
        experience_years_array,
        education_array,
        income_shock_array,
    )
    df["gross_labor_income"] = gross_labor_income_array * df["lagged_choice"].isin(
        work_values
    )

    return df
