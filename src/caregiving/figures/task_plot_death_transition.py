"""Plot death transition shares over time."""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.model.stochastic_processes.adl_transition import death_transition


@pytask.mark.plot_mother_death
def task_plot_death_transition(  # noqa: PLR0915
    path_to_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "mother_death_transition.png",
):
    """Plot share of alive and dead parents over time.

    Starts at period 0 with distribution from initial states, then simulates
    forward using death transition probabilities.

    Parameters
    ----------
    path_to_states : Path
        Path to initial states pkl file
    path_to_options : Path
        Path to options pkl file containing model parameters
    path_to_save : Path
        Path to save the plot

    """

    # Load initial states and options
    with path_to_states.open("rb") as f:
        states = pickle.load(f)

    with path_to_options.open("rb") as f:
        options = pickle.load(f)

    specs = options["model_params"]
    n_periods = specs["n_periods"]

    # Convert to numpy for fast aggregation
    mother_dead_initial = np.asarray(states["mother_dead"], dtype=np.uint8)
    has_sister = np.asarray(states["has_sister"], dtype=np.uint8)
    education = np.asarray(states["education"], dtype=np.uint8)

    n_edu = specs["n_education_types"]

    # Initialize share arrays
    share_alive = np.zeros(n_periods)
    share_dead = np.zeros(n_periods)

    # Initial shares
    share_alive[0] = np.mean(mother_dead_initial == 0)
    share_dead[0] = 1 - share_alive[0]

    # Track alive and dead counts by group (hs, edu)
    alive_by_group = {}
    dead_by_group = {}
    for hs in (0, 1):
        for edu in range(n_edu):
            mask_alive = (
                (has_sister == hs) & (education == edu) & (mother_dead_initial == 0)
            )
            mask_dead = (
                (has_sister == hs) & (education == edu) & (mother_dead_initial == 1)
            )
            alive_by_group[(hs, edu)] = float(mask_alive.sum())
            dead_by_group[(hs, edu)] = float(mask_dead.sum())

    n_agents = len(mother_dead_initial)

    # Simulate forward deterministically using transition probs (no per-agent loop)
    for period in range(1, n_periods):
        alive_next = {}
        dead_next = {}
        for hs in (0, 1):
            for edu in range(n_edu):
                alive_curr = alive_by_group[(hs, edu)]
                dead_curr = dead_by_group[(hs, edu)]

                # Dead mothers stay dead
                dead_next[(hs, edu)] = dead_curr

                # Calculate transitions for alive mothers
                if alive_curr == 0:
                    alive_next[(hs, edu)] = 0.0
                else:
                    # death_transition returns [alive_prob, dead_prob]
                    # mother_dead=0 for currently alive mothers
                    prob_vector = death_transition(period - 1, 0, hs, edu, specs)
                    alive_prob = float(prob_vector[0])  # Extract alive probability
                    dead_prob = float(prob_vector[1])  # Extract death probability
                    alive_next[(hs, edu)] = alive_curr * alive_prob
                    dead_next[(hs, edu)] = dead_curr + alive_curr * dead_prob

        alive_by_group = alive_next
        dead_by_group = dead_next

        total_alive = sum(alive_by_group.values())
        total_dead = sum(dead_by_group.values())
        share_alive[period] = total_alive / n_agents
        share_dead[period] = total_dead / n_agents

    # Convert periods to ages for x-axis
    start_age = specs["start_age"]
    periods = np.arange(n_periods)
    ages = start_age + periods

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ages, share_alive, label="Alive", linewidth=2, color="green")
    ax.plot(ages, share_dead, label="Dead", linewidth=2, color="red")

    ax.set_xlabel("Age")
    ax.set_ylabel("Share of Population")
    ax.set_title("Share of Alive and Dead Parents Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0, 1.05, 0.05))

    # Add vertical line at start age if needed
    ax.axvline(x=start_age, color="gray", linestyle="--", alpha=0.5, label="Start Age")

    plt.tight_layout()
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save, dpi=300)
    plt.close(fig)

    print(f"Death transition plot saved to {path_to_save}")
