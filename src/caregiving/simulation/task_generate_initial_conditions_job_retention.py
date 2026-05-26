"""Initial conditions for the job retention and Beirat leave simulations.

This module creates initial conditions for the job retention counterfactual
by loading the baseline initial states and adding the job_before_caregiving variable.
For the Beirat leave model it also adds years_leave_used_total (partial leave only).
For the full Beirat model (max 1 year full leave) it adds years_leave_used_total
and full_leave_year_used.
For the Full-Beirat-no-total-cap variant it adds job_before_caregiving and
full_leave_year_used (the 3-year total cap is dropped, so years_leave_used_total
is NOT added).
"""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import pytask
from pytask import Product

from caregiving.config import BLD


@pytask.mark.initial_conditions
@pytask.mark.initial_conditions_job_retention
def task_generate_start_states_for_solution_job_retention(
    path_to_baseline_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_updated_states: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "initial_states_job_retention.pkl",
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
    with path_to_save_updated_states.open("wb") as f:
        pickle.dump(states, f)


@pytask.mark.initial_conditions
@pytask.mark.initial_conditions_beirat
def task_generate_start_states_for_solution_beirat(
    path_to_baseline_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_updated_states: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "initial_states_beirat.pkl",
) -> None:
    """Generate initial conditions for Beirat leave model simulation.

    Loads the baseline initial states and adds job_before_caregiving and
    years_leave_used_total (partial leave only), initialized to zeros.
    """
    with path_to_baseline_states.open("rb") as f:
        states = pickle.load(f)

    states["job_before_caregiving"] = jnp.zeros_like(
        states["experience"], dtype=jnp.uint8
    )
    states["years_leave_used_total"] = jnp.zeros_like(
        states["experience"], dtype=jnp.uint8
    )

    with path_to_save_updated_states.open("wb") as f:
        pickle.dump(states, f)


@pytask.mark.initial_conditions
@pytask.mark.initial_conditions_full_beirat
def task_generate_start_states_for_solution_full_beirat(
    path_to_baseline_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_updated_states: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "initial_states_full_beirat.pkl",
) -> None:
    """Generate initial conditions for full Beirat leave model simulation.

    Loads the baseline initial states and adds job_before_caregiving,
    years_leave_used_total, and full_leave_year_used (max 1 year full leave),
    all initialized to zeros.
    """
    with path_to_baseline_states.open("rb") as f:
        states = pickle.load(f)

    states["job_before_caregiving"] = jnp.zeros_like(
        states["experience"], dtype=jnp.uint8
    )
    states["years_leave_used_total"] = jnp.zeros_like(
        states["experience"], dtype=jnp.uint8
    )
    states["full_leave_year_used"] = jnp.zeros_like(
        states["experience"], dtype=jnp.uint8
    )

    with path_to_save_updated_states.open("wb") as f:
        pickle.dump(states, f)


@pytask.mark.initial_conditions
@pytask.mark.initial_conditions_full_beirat_no_total_cap
def task_generate_start_states_for_solution_full_beirat_no_total_cap(
    path_to_baseline_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states.pkl",
    path_to_save_updated_states: Annotated[Path, Product] = BLD
    / "model"
    / "initial_conditions"
    / "initial_states_full_beirat_no_total_cap.pkl",
) -> None:
    """Generate initial conditions for Full-Beirat-no-total-cap leave simulation.

    Loads the baseline initial states and adds job_before_caregiving and
    full_leave_year_used (1-year full-leave sub-cap retained), initialized to
    zeros. ``years_leave_used_total`` is intentionally NOT added because the
    new variant drops the 3-year cumulative cap and no longer carries that
    state. See task_specify_model_caregiving_leave_full_beirat_no_total_cap.
    """
    with path_to_baseline_states.open("rb") as f:
        states = pickle.load(f)

    states["job_before_caregiving"] = jnp.zeros_like(
        states["experience"], dtype=jnp.uint8
    )
    states["full_leave_year_used"] = jnp.zeros_like(
        states["experience"], dtype=jnp.uint8
    )

    with path_to_save_updated_states.open("wb") as f:
        pickle.dump(states, f)
