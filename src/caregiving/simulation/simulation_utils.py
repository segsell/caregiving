"""Helper functions for simulation state extraction and manipulation.

This module provides reusable functions to reduce code duplication in simulation tasks.
"""

import numpy as np
import pandas as pd


def extract_discrete_states_at_period(
    sim_dict: dict,
    df_period: pd.DataFrame,
    forced_period: int,
    initial_states: dict,
) -> dict:
    """Extract discrete states at a specific period from simulation dictionary.

    Args:
        sim_dict: Simulation dictionary with state arrays
        df_period: DataFrame for the specific period
        forced_period: Period to extract states from
        initial_states: Initial states dict (for fallback values)

    Returns:
        Dictionary of state arrays at the specified period
    """
    states_at_period = {}

    for key in initial_states.keys():
        if key in sim_dict:
            # Shape is (n_periods, n_agents)
            if len(sim_dict[key].shape) == 2:  # noqa: PLR2004
                # Get states at forced_period
                states_at_period[key] = sim_dict[key][forced_period, :].copy()
            else:
                states_at_period[key] = sim_dict[key].copy()
        elif key == "experience":
            # Extract experience from DataFrame
            if "experience" in df_period.columns:
                states_at_period[key] = df_period["experience"].values.copy()
            else:
                # Fallback to initial value
                states_at_period[key] = initial_states[key].copy()
        else:
            # For other states not in sim_dict, use initial value
            states_at_period[key] = initial_states[key].copy()

    return states_at_period


def extract_lagged_choice_at_period(
    df_baseline: pd.DataFrame,
    df_period: pd.DataFrame,
    forced_period: int,
) -> np.ndarray:
    """Extract lagged_choice at a specific period.

    Args:
        df_baseline: Full baseline simulation DataFrame
        df_period: DataFrame for the specific period
        forced_period: Period to extract lagged_choice from

    Returns:
        Array of lagged_choice values

    Raises:
        ValueError: If lagged_choice cannot be extracted
    """
    if "lagged_choice" in df_period.columns:
        return df_period["lagged_choice"].values.copy()

    if "choice" in df_period.columns:
        # If lagged_choice not available, get choice from previous period
        df_prev_period = df_baseline[
            df_baseline.index.get_level_values("period") == forced_period - 1
        ].copy()
        if "choice" in df_prev_period.columns:
            return df_prev_period["choice"].values.copy()

    raise ValueError(f"Could not extract lagged_choice at period {forced_period}")


def extract_exogenous_states_at_period(
    df_period: pd.DataFrame,
    state_names: list[str],
) -> dict:
    """Extract exogenous states at a specific period.

    Args:
        df_period: DataFrame for the specific period
        state_names: List of exogenous state names to extract

    Returns:
        Dictionary of exogenous state arrays
    """
    exogenous_states = {}
    for exog_state in state_names:
        if exog_state in df_period.columns:
            exogenous_states[exog_state] = df_period[exog_state].values.copy()
    return exogenous_states


def override_forced_states(
    states: dict,
    care_demand_value: int,
    parent_bad_health_value: int,
    parent_dead_value: int,
) -> dict:
    """Override care_demand and mother_health to forced values.

    Args:
        states: Dictionary of state arrays
        care_demand_value: Value to set for care_demand
        parent_bad_health_value: Value to set for mother_health when alive
        parent_dead_value: Value indicating parent is dead

    Returns:
        Dictionary with overridden states
    """
    n_agents = len(states.get("care_demand", states.get("mother_health", [])))
    if n_agents == 0:
        return states

    states["care_demand"] = np.full(n_agents, care_demand_value, dtype=np.uint8)

    if "mother_health" in states:
        mother_alive = states["mother_health"] != parent_dead_value
        states["mother_health"] = np.where(
            mother_alive,
            parent_bad_health_value,
            states["mother_health"],
        )

    return states


def extract_wealth_at_period(
    sim_dict: dict,
    forced_period: int,
    initial_wealth: np.ndarray,
) -> np.ndarray:
    """Extract wealth at beginning of a specific period.

    Args:
        sim_dict: Simulation dictionary with savings and consumption
        forced_period: Period to extract wealth from
        initial_wealth: Initial wealth array (used if forced_period == 0)

    Returns:
        Array of wealth values at the beginning of the period
    """
    if forced_period > 0:
        # Wealth at beginning of period = savings + consumption from previous period
        return (
            sim_dict["savings"][forced_period - 1, :]
            + sim_dict["consumption"][forced_period - 1, :]
        )
    # If forced_period == 0, use initial wealth
    return initial_wealth


def convert_states_to_dtypes(
    states: dict,
    discrete_state_space: dict,
    continuous_state_names: list[str] | None = None,
) -> dict:
    """Convert states to proper dtypes based on state space definition.

    Args:
        states: Dictionary of state arrays
        discrete_state_space: Dictionary mapping state names to dtype definitions
        continuous_state_names: Optional list of continuous state names

    Returns:
        Dictionary of states with proper dtypes
    """
    if continuous_state_names is None:
        continuous_state_names = []

    states_dtype = {}
    for key, value in states.items():
        if key in discrete_state_space:
            states_dtype[key] = value.astype(discrete_state_space[key].dtype)
        elif key in ("care_demand", "mother_health", "job_offer"):
            # Exogenous states - use appropriate dtype
            states_dtype[key] = value.astype(np.uint8)
        elif key in continuous_state_names:
            # Continuous states - keep as is
            states_dtype[key] = value

    return states_dtype


def find_continuous_state_names(
    continuous_states: dict,
) -> list[str]:
    """Find continuous state names (excluding wealth).

    Args:
        continuous_states: Dictionary of continuous state definitions

    Returns:
        List of continuous state names (excluding 'wealth')
    """
    if len(continuous_states) > 1:  # More than just wealth
        # Find the second continuous state (first is wealth)
        continuous_state_names = [k for k in continuous_states.keys() if k != "wealth"]
        return continuous_state_names
    return []
