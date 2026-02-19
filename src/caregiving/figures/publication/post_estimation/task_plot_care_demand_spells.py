"""Plot average duration of care demand spells by age at first care demand."""

import pickle
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import DEAD, PARENT_ALIVE


def _compute_average_care_demand_duration_by_age_at_first(
    df: pd.DataFrame, start_age: int
) -> pd.DataFrame:
    """Compute average care demand duration by age at first care demand.

    For agents who first experience care_demand > 0 at a given age, compute
    the average number of years with positive care demand from that first
    occurrence until the mother dies (or end of observation). Care demands
    need not be consecutive.

    Returns:
        DataFrame with columns: age_at_first_care_demand, avg_duration, n_agents
    """
    df = df.copy()
    df["age"] = df["period"] + start_age

    # Positive care demand: light or intensive
    df["has_care_demand"] = (df["care_demand"] > 0).astype(int)

    # Mother dead: not PARENT_ALIVE (0)
    df["mother_is_dead"] = (df["mother_dead"] != PARENT_ALIVE).astype(int)

    # First care demand per agent
    first_care = (
        df.loc[df["has_care_demand"] == 1, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")
        .rename(columns={"period": "first_care_period", "age": "age_at_first_care"})
    )

    if len(first_care) == 0:
        return pd.DataFrame(
            columns=["age_at_first_care_demand", "avg_duration", "n_agents"]
        )

    # For each agent with care demand, compute duration from first care until mother dies
    results = []
    for agent in first_care["agent"].unique():
        agent_df = df[df["agent"] == agent].sort_values("period")
        first_row = first_care[first_care["agent"] == agent].iloc[0]
        first_period = first_row["first_care_period"]
        age_at_first = first_row["age_at_first_care"]

        # Restrict to periods from first care onward
        agent_from_first = agent_df[agent_df["period"] >= first_period].copy()

        # Find first period where mother is dead (if any); spell ends before that
        dead_periods = agent_from_first[agent_from_first["mother_is_dead"] == 1]
        if len(dead_periods) > 0:
            first_dead_period = dead_periods["period"].min()
            agent_from_first = agent_from_first[
                agent_from_first["period"] < first_dead_period
            ]

        # Duration = count of periods with care_demand > 0 in this window
        duration = agent_from_first["has_care_demand"].sum()
        results.append(
            {
                "agent": agent,
                "age_at_first_care": age_at_first,
                "duration": duration,
            }
        )

    res_df = pd.DataFrame(results)

    # Aggregate by age at first care
    agg = (
        res_df.groupby("age_at_first_care", observed=False)["duration"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "age_at_first_care": "age_at_first_care_demand",
                "mean": "avg_duration",
                "count": "n_agents",
            }
        )
    )
    return agg


def plot_average_care_demand_duration_by_age(
    df: pd.DataFrame,
    start_age: int,
    age_min: int = 40,
    age_max: int = 65,
    path_to_save: Optional[Path] = None,
) -> None:
    """Plot average care demand duration by age at first care demand.

    Y-axis: age (40 at top, 65 at bottom).
    X-axis: average duration in years.
    """
    agg = _compute_average_care_demand_duration_by_age_at_first(df, start_age)

    if len(agg) == 0:
        raise ValueError("No agents with care demand in the data.")

    # Filter to age range
    agg = agg[
        (agg["age_at_first_care_demand"] >= age_min)
        & (agg["age_at_first_care_demand"] <= age_max)
    ].sort_values("age_at_first_care_demand")

    if len(agg) == 0:
        raise ValueError(
            f"No agents with first care demand between ages {age_min} and {age_max}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))

    ages = agg["age_at_first_care_demand"].values
    durations = agg["avg_duration"].values

    ax.plot(durations, ages, marker="o", linewidth=2, markersize=6)
    ax.set_ylabel("Age at first care demand", fontsize=14)
    ax.set_xlabel("Average duration of care demand spell (years)", fontsize=14)
    ax.set_ylim(age_max + 1, age_min - 1)  # 40 at top, 65 at bottom
    ax.set_yticks(range(age_min, age_max + 1, 5))
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if path_to_save:
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save, dpi=150, bbox_inches="tight")
    plt.close()


@pytask.mark.post_estimation
@pytask.mark.care_demand_duration
def task_plot_care_demand_spells_by_age_at_first(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "care_demand_spell_duration_by_age_at_first.pdf",
) -> None:
    """Plot average care demand spell duration by age at first care demand.

    For each age (40–65), shows the average number of years with positive
    care demand among agents who first experienced care demand at that age.
    Duration is measured from first care demand until mother dies; care
    demands need not be consecutive.
    """
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    df = pd.read_pickle(path_to_simulated_data)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    df = df[df["health"] != DEAD].copy()
    start_age = specs["start_age"]

    plot_average_care_demand_duration_by_age(
        df=df,
        start_age=start_age,
        age_min=40,
        age_max=65,
        path_to_save=path_to_plot,
    )
