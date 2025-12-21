"""Plot care demand shares by age from simulated data."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    CARE_DEMAND_INTENSIVE,
    CARE_DEMAND_LIGHT,
    DEAD,
)


@pytask.mark.post_estimation
@pytask.mark.post_estimation_care_demand
def task_plot_care_demand_by_age_simulated(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "care_demand_by_age_simulated.png",
):
    """Plot share of care demand states by age from simulated data.

    Loads simulated data, conditions on agents with health != DEAD, and plots
    care demand shares (Light Care, Intensive Care, and Any Care Demand) by age.

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to simulated data pkl file
    path_to_save : Path
        Path to save the plot

    """
    # Load specs and simulated data
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()

    # Create age variable from start_age + period
    df_sim["age"] = df_sim["period"] + specs["start_age"]

    # Condition on agents with health != DEAD
    df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()

    # Calculate care demand shares by age
    # care_demand can take 4 states: 0=NO_CARE_DEMAND_DEAD, 1=NO_CARE_DEMAND_ALIVE,
    # 2=CARE_DEMAND_LIGHT, 3=CARE_DEMAND_INTENSIVE
    care_demand_by_age = df_sim.groupby("age", observed=False)["care_demand"].agg(
        light=lambda x: (x == CARE_DEMAND_LIGHT).mean(),
        intensive=lambda x: (x == CARE_DEMAND_INTENSIVE).mean(),
        any_care=lambda x: (x.isin([CARE_DEMAND_LIGHT, CARE_DEMAND_INTENSIVE])).mean(),
    )

    # Get age range and shares
    ages = care_demand_by_age.index.values
    share_light = care_demand_by_age["light"].values
    share_intensive = care_demand_by_age["intensive"].values
    share_any_care = care_demand_by_age["any_care"].values

    # Create plot - similar style to pre_estimation plots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot care demand categories
    ax.plot(
        ages,
        share_any_care,
        label="Any Care Demand",
        linewidth=2,
        color="blue",
        linestyle="--",
    )
    ax.plot(ages, share_light, label="Light Care", linewidth=2, color="green")
    ax.plot(ages, share_intensive, label="Intensive Care", linewidth=2, color="red")

    # Set y-axis limits and ticks (similar to pre_estimation plots)
    y_max = 0.16
    ax.set_ylim([0, y_max])
    # Set yticks every 0.02
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.02))

    ax.set_xlabel("Agent Age")
    ax.set_ylabel("Share in Population")
    ax.set_title("Share of Care Demand States by Age (Simulated Data)")
    ax.legend(title="Care level", fontsize=9, title_fontsize=10, loc="best")

    # Grid: major and minor ticks every 0.02
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.02), minor=True)
    ax.grid(True, which="minor", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.5)

    # Add vertical line at start age
    start_age = specs["start_age"]
    ax.axvline(x=start_age, color="gray", linestyle="--", alpha=0.5, label="Start Age")

    plt.tight_layout()
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save, dpi=300)
    plt.close(fig)

    print(f"Care demand by age plot (simulated data) saved to {path_to_save}")
