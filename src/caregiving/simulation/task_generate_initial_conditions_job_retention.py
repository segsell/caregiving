"""Initial conditions for the job retention simulation.

This module creates initial conditions for the job retention counterfactual
by loading the baseline initial states and adding the job_before_caregiving variable.
"""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import pytask
from pytask import Product

from caregiving.config import BLD


def task_generate_start_states_for_solution_job_retention(
    path_to_baseline_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_save_discrete_states: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "states_job_retention.pkl",
) -> None:
    """Generate initial conditions for job retention model simulation.

    This function loads the baseline initial states and adds the
    job_before_caregiving state variable (initialized to zeros).
    Wealth is taken from the baseline wealth.csv file, so no wealth
    regeneration is needed.

    Args:
        path_to_baseline_states: Path to baseline initial states pickle file
        path_to_save_discrete_states: Path to save job retention initial states
    """
    # Load baseline states
    with path_to_baseline_states.open("rb") as f:
        states = pickle.load(f)

    # Add job_before_caregiving initialized to zeros
    # Use experience array as template for shape
    states["job_before_caregiving"] = jnp.zeros_like(
        states["experience"], dtype=jnp.uint8
    )

    # Save job retention states
    with path_to_save_discrete_states.open("wb") as f:
        pickle.dump(states, f)
