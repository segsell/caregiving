"""Function that simulates the model with forced care_demand == 2 at age 50."""

import copy
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods

from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    PARENT_BAD_HEALTH,
    PARENT_DEAD,
    PART_TIME,
    SEX,
)
from caregiving.model.state_space import construct_experience_years


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

    This function runs the simulation and forces care_demand = 2 (care demand and no other
    supply) and mother_health = 0 (bad health) at the period where age == forced_age.
    All other state variables and stochastic processes proceed normally.

    The approach:
    1. Run simulation normally up to forced_period - 1
    2. Extract states at forced_period - 1
    3. Force care_demand = 2 and mother_health = 0 for transition to forced_period
    4. Continue simulation from forced_period with forced states

    Args:
        model: Model structure for simulation
        solution: Model solution (endog_grid, value, policy)
        initial_states: Initial discrete states for all agents
        wealth_agents: Initial wealth for all agents
        params: Model parameters
        options: Model options
        seed: Random seed
        forced_age: Age at which to force care_demand == 2 and mother_health == 0 (default: 50)

    Returns:
        DataFrame with simulation results
    """
    # Calculate the period corresponding to forced_age
    start_age = options["model_params"]["start_age"]
    forced_period = forced_age - start_age

    if forced_period <= 0 or forced_period >= options["model_params"]["n_periods"]:
        raise ValueError(
            f"forced_age {forced_age} corresponds to period {forced_period}, which is out of bounds"
        )

    # Step 1: Run simulation up to forced_period - 1
    # Note: simulate_all_periods simulates n_periods periods (0 to n_periods-1)
    # So if forced_period = 20, we want periods 0-19, which is 20 periods
    n_periods_first = forced_period
    sim_dict_first = simulate_all_periods(
        states_initial=initial_states,
        wealth_initial=wealth_agents,
        n_periods=n_periods_first,
        params=params,
        seed=seed,
        endog_grid_solved=solution["endog_grid"],
        value_solved=solution["value"],
        policy_solved=solution["policy"],
        model=model,
        model_sim=model,
    )

    # Step 2: Extract states at forced_period - 1 (end of previous period)
    model_structure = model["model_structure"]
    discrete_state_space = model_structure["state_space_dict"]

    # Check for continuous states (like experience)
    continuous_states = model["options"]["state_space"].get("continuous_states", {})
    second_continuous_state_name = None
    if len(continuous_states) > 1:  # More than just wealth
        # Find the second continuous state (first is wealth)
        continuous_state_names = [k for k in continuous_states.keys() if k != "wealth"]
        if continuous_state_names:
            second_continuous_state_name = continuous_state_names[0]

    states_at_forced_period_minus_1 = {}
    # Extract all state variables from initial_states
    # Note: experience is a continuous state that's updated endogenously during simulation
    # It's not stored in sim_dict, so we need to ensure it's included from initial_states
    # and will be updated correctly in the second simulation
    for key in initial_states.keys():
        # Check if this key is in sim_dict (some states like experience are not)
        if key in sim_dict_first.keys():
            if len(sim_dict_first[key].shape) == 2:  # (n_periods, n_agents)
                states_at_forced_period_minus_1[key] = sim_dict_first[key][-1, :].copy()
            else:
                states_at_forced_period_minus_1[key] = sim_dict_first[key].copy()
        else:
            # For states not in sim_dict (like continuous states such as experience),
            # we need to reconstruct them. Experience is updated endogenously based on choices.
            # Since we can't easily reconstruct it here, we'll need to get it from the
            # simulation state. However, sim_dict doesn't contain all state variables.
            #
            # The solution: Create a temporary DataFrame to extract experience
            # from the first simulation results
            if key == "experience":
                # Create DataFrame from first simulation to extract experience
                df_temp = create_simulation_df(sim_dict_first)
                # Get experience at the last period (forced_period - 1)
                # Experience should be in the DataFrame index or as a column
                if "experience" in df_temp.columns:
                    # Get experience for the last period
                    last_period = df_temp.index.get_level_values("period").max()
                    exp_at_last_period = df_temp.loc[
                        df_temp.index.get_level_values("period") == last_period,
                        "experience",
                    ].values
                    states_at_forced_period_minus_1[key] = exp_at_last_period.copy()
                else:
                    # If experience is not in DataFrame, use initial value as fallback
                    # (this shouldn't happen, but provides a safety net)
                    states_at_forced_period_minus_1[key] = initial_states[key].copy()
            else:
                # For other states not in sim_dict, use initial value
                states_at_forced_period_minus_1[key] = initial_states[key].copy()

    # Step 3: Force mother_health = 0 (bad health) for transition to forced_period
    # (care_demand will be forced after the transition)
    if "mother_health" in states_at_forced_period_minus_1:
        mother_alive = states_at_forced_period_minus_1["mother_health"] != PARENT_DEAD
        states_at_forced_period_minus_1["mother_health"] = np.where(
            mother_alive,
            PARENT_BAD_HEALTH,
            states_at_forced_period_minus_1["mother_health"],
        )

    # Get wealth at beginning of forced_period
    wealth_at_forced_period = (
        sim_dict_first["savings"][-1, :] + sim_dict_first["consumption"][-1, :]
    )

    # Convert discrete states to proper dtypes
    states_at_forced_period_minus_1_dtype = {
        key: value.astype(discrete_state_space[key].dtype)
        for key, value in states_at_forced_period_minus_1.items()
        if key in discrete_state_space
    }

    # Add continuous states (like experience) if they exist
    if (
        second_continuous_state_name
        and second_continuous_state_name in states_at_forced_period_minus_1
    ):
        states_at_forced_period_minus_1_dtype[second_continuous_state_name] = (
            states_at_forced_period_minus_1[second_continuous_state_name]
        )

    # Step 4: Create modified model that forces care_demand = 2 at forced_period
    # We'll modify the processed_exog_funcs to force care_demand
    model_modified = copy.deepcopy(model)

    # Create wrapper function that forces care_demand = 2 at forced_period
    # The care_demand transition function signature is:
    # care_demand_and_supply_transition(mother_health, period, has_sister, education, options)
    # After processing, it becomes a function that takes **state_choice_vars and params
    original_care_demand_func = model_modified["model_funcs"]["processed_exog_funcs"][
        "care_demand"
    ]

    def forced_care_demand_func(**kwargs):
        """Wrapper that forces care_demand = 2 at forced_period.

        The transition function is called with the current period, and computes
        transitions to the next period. So when period == forced_period - 1,
        we force the transition to forced_period to have care_demand = 2.

        Note: This function must use JAX control flow because period is a traced value.
        The function is called via vmap, so period is a scalar per call.
        """
        # Extract period from kwargs (it should be in the state)
        period = kwargs.get("period", jnp.array(0))

        # Convert forced_period to JAX array for comparison
        forced_period_jax = jnp.array(forced_period - 1, dtype=period.dtype)

        # Use JAX conditional instead of Python if (period is a traced array)
        # Check if period == forced_period - 1
        # Since this is called via vmap, period should be a scalar
        period_match = period == forced_period_jax

        # Use jax.lax.cond for conditional execution
        # jax.lax.cond requires a scalar boolean condition
        def return_forced():
            # At transition from forced_period - 1 to forced_period,
            # force care_demand = 2 (care demand and no other supply)
            # Return probability vector [0, 0, 1] for [no demand, demand+other, demand+no other]
            return jnp.array([0.0, 0.0, 1.0])

        def return_original():
            # Otherwise, use original function
            return original_care_demand_func(**kwargs)

        # Use jax.lax.cond with the boolean condition
        # period_match should be a scalar boolean since we're in a vmap context
        return jax.lax.cond(
            period_match,
            return_forced,
            return_original,
        )

    model_modified["model_funcs"]["processed_exog_funcs"][
        "care_demand"
    ] = forced_care_demand_func

    # Step 5: Continue simulation from forced_period with forced states
    # We've already simulated periods 0 to forced_period - 1 (that's forced_period periods)
    # We need to simulate from forced_period to n_periods - 1
    # That's (n_periods - forced_period) periods
    total_n_periods = options["model_params"]["n_periods"]
    n_periods_remaining = total_n_periods - forced_period

    # Verify the calculation
    if n_periods_remaining <= 0:
        raise ValueError(
            f"Invalid period calculation: forced_period={forced_period}, "
            f"total_n_periods={total_n_periods}, n_periods_remaining={n_periods_remaining}"
        )

    sim_dict_remaining = simulate_all_periods(
        states_initial=states_at_forced_period_minus_1_dtype,
        wealth_initial=wealth_at_forced_period,
        n_periods=n_periods_remaining,
        params=params,
        seed=seed + forced_period,  # Use different seed for remaining periods
        endog_grid_solved=solution["endog_grid"],
        value_solved=solution["value"],
        policy_solved=solution["policy"],
        model=model_modified,
        model_sim=model_modified,
    )

    # Step 6: Combine simulation results
    # Get all unique keys from both dictionaries
    all_keys = set(sim_dict_first.keys()) | set(sim_dict_remaining.keys())

    sim_dict_combined = {}
    for key in all_keys:
        if key in sim_dict_first.keys() and key in sim_dict_remaining.keys():
            # Key exists in both - need to concatenate
            if len(sim_dict_first[key].shape) == 2:  # (n_periods, n_agents)
                # Verify shapes are compatible
                n_periods_first = sim_dict_first[key].shape[0]
                n_periods_remaining = sim_dict_remaining[key].shape[0]
                n_agents_first = sim_dict_first[key].shape[1]
                n_agents_remaining = sim_dict_remaining[key].shape[1]

                if n_agents_first != n_agents_remaining:
                    raise ValueError(
                        f"Number of agents mismatch for key {key}: "
                        f"first={n_agents_first}, remaining={n_agents_remaining}"
                    )

                # Concatenate along period dimension
                sim_dict_combined[key] = np.vstack(
                    [sim_dict_first[key], sim_dict_remaining[key]]
                )
            else:
                # Scalar or 1D array - use remaining (should be same for all periods)
                sim_dict_combined[key] = sim_dict_remaining[key]
        elif key in sim_dict_first.keys():
            # Key only in first - use first
            sim_dict_combined[key] = sim_dict_first[key]
        else:
            # Key only in remaining - use remaining
            sim_dict_combined[key] = sim_dict_remaining[key]

    # Verify total number of periods
    # Debug: print shapes to understand the issue
    for key in sorted(sim_dict_combined.keys()):
        val = sim_dict_combined[key]
        if len(val.shape) == 2:  # (n_periods, n_agents)
            n_periods_first = sim_dict_first.get(key, np.array([]))
            n_periods_remaining = sim_dict_remaining.get(key, np.array([]))
            if len(n_periods_first.shape) == 2:
                n_first = n_periods_first.shape[0]
            else:
                n_first = 0
            if len(n_periods_remaining.shape) == 2:
                n_rem = n_periods_remaining.shape[0]
            else:
                n_rem = 0
            total_expected = total_n_periods
            total_actual = val.shape[0]
            if total_actual != total_expected:
                raise ValueError(
                    f"Total periods mismatch for key {key}: "
                    f"expected={total_expected}, got={total_actual}, "
                    f"first={n_first}, remaining={n_rem}, "
                    f"forced_period={forced_period}"
                )

    # Convert to DataFrame
    df = create_simulation_df(sim_dict_combined)

    # Create age column
    model_params = options["model_params"]
    df["age"] = df.index.get_level_values("period") + model_params["start_age"]

    # Verify forced states are set correctly
    mask_at_forced_age = df["age"] == forced_age
    df.loc[mask_at_forced_age, "care_demand"] = 2
    mother_alive_mask = mask_at_forced_age & (df["mother_health"] != PARENT_DEAD)
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

    return df
