"""Sandbox for simulating moments."""

from pathlib import Path

import pandas as pd

from caregiving._simulate import simulate_moments
from caregiving.config import TESTS


def task_create_simulated_moments(
    path_to_one: Path = TESTS / "data" / "simulation" / "simulated_moments_trial.csv",
    path_to_two: Path = TESTS
    / "data"
    / "simulation"
    / "simulated_moments_trial_two.csv",
):
    """Create array of simulated moments.

    df2["income"] = df2["working_hours"] * df["wage"] column_indices = {col: idx for
    idx, col in enumerate(df2.columns)} idx = column_indices.copy() arr2 =
    jnp.asarray(df2)

    """
    sim = pd.read_csv(path_to_one)
    df2 = pd.read_csv(path_to_two)

    df2["income"] = df2["working_hours"] * df2["wage"]

    simulate_moments(sim)
