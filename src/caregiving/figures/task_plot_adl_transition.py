"""Plot ADL transition shares over time."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import ADL_1, ADL_2_3
from caregiving.model.stochastic_processes.adl_transition import (
    death_transition,
    limitations_with_adl_transition,
)


@pytask.mark.plot_mother_adl
def task_plot_adl_transition(  # noqa: PLR0915
    path_to_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "mother_adl_transition.png",
):
    """Plot share of ADL states (No ADL, ADL 1, ADL 2/3) over time.

    Starts at period 0 with distribution from initial states, then simulates
    forward using ADL transition probabilities. Only tracks alive mothers.

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
    mother_adl_initial = np.asarray(states["mother_adl"], dtype=np.uint8)
    education = np.asarray(states["education"], dtype=np.uint8)

    n_edu = specs["n_education_types"]
    n_adl_states = specs[
        "n_adl_states_light_intensive"
    ]  # 3 states: No ADL, ADL 1, ADL 2/3

    n_agents = len(mother_dead_initial)

    # Initialize share arrays
    share_no_adl = np.zeros(n_periods)
    share_adl_1 = np.zeros(n_periods)
    share_adl_2_3 = np.zeros(n_periods)
    share_any_adl = np.zeros(n_periods)

    # Initial shares
    alive_mask = mother_dead_initial == 0
    dead_mask = mother_dead_initial == 1

    # No ADL includes: alive with no ADL + dead (dead = no ADL)
    no_adl_alive = ((mother_adl_initial == 0) & alive_mask).sum()
    n_dead_initial = dead_mask.sum()
    share_no_adl[0] = (no_adl_alive + n_dead_initial) / n_agents

    share_adl_1[0] = np.mean((mother_adl_initial == ADL_1) & alive_mask)
    share_adl_2_3[0] = np.mean((mother_adl_initial == ADL_2_3) & alive_mask)
    share_any_adl[0] = share_adl_1[0] + share_adl_2_3[0]

    # Track ADL state counts by group (edu, adl_state) for alive mothers only
    adl_by_group = {}
    # Track dead mothers by group (edu)
    dead_by_group = {}
    for edu in range(n_edu):
        # Track alive mothers by ADL state
        for adl in range(n_adl_states):
            mask = (
                (education == edu)
                & (mother_dead_initial == 0)
                & (mother_adl_initial == adl)
            )
            adl_by_group[(edu, adl)] = float(mask.sum())
        # Track dead mothers
        mask_dead = (education == edu) & (mother_dead_initial == 1)
        dead_by_group[edu] = float(mask_dead.sum())

    # Simulate forward deterministically using transition probs
    for period in range(1, n_periods):
        adl_next = {}
        dead_next = {}
        for edu in range(n_edu):
            # Dead mothers stay dead (and count as "No ADL")
            dead_curr = dead_by_group.get(edu, 0.0)
            dead_next[edu] = dead_curr

            for adl_curr_state in range(n_adl_states):
                count_curr = adl_by_group.get((edu, adl_curr_state), 0.0)

                if count_curr == 0:
                    # No one in this state, nothing to transition
                    continue

                # First, account for death transitions
                # death_transition returns [alive_prob, dead_prob]
                death_prob_vector = death_transition(period - 1, 0, edu, specs)
                alive_prob = float(death_prob_vector[0])
                dead_prob = float(death_prob_vector[1])

                # Only alive mothers have ADL states
                count_alive = count_curr * alive_prob
                count_died = count_curr * dead_prob

                # Add newly dead to dead count (they count as "No ADL")
                dead_next[edu] = dead_next.get(edu, 0.0) + count_died

                if count_alive == 0:
                    # All died, nothing to transition
                    continue

                # Get ADL transition probabilities for alive mothers
                # Returns [prob_no_adl, prob_adl_1, prob_adl_2/3]
                adl_prob_vector = limitations_with_adl_transition(
                    adl_curr_state, period - 1, edu, specs
                )

                # Distribute alive count to next ADL states
                # according to transition probs
                for adl_next_state in range(n_adl_states):
                    prob = float(adl_prob_vector[adl_next_state])
                    key = (edu, adl_next_state)
                    adl_next[key] = adl_next.get(key, 0.0) + count_alive * prob

        # Update counts
        adl_by_group = adl_next
        dead_by_group = dead_next

        # Calculate total shares
        # No ADL includes: alive with no ADL + dead (dead = no ADL)
        total_no_adl_alive = sum(
            adl_by_group.get((edu, 0), 0.0) for edu in range(n_edu)
        )
        total_dead = sum(dead_by_group.get(edu, 0.0) for edu in range(n_edu))
        total_adl_1 = sum(adl_by_group.get((edu, 1), 0.0) for edu in range(n_edu))
        total_adl_2_3 = sum(adl_by_group.get((edu, 2), 0.0) for edu in range(n_edu))

        # Normalize by total agents (including dead ones)
        share_no_adl[period] = (total_no_adl_alive + total_dead) / n_agents
        share_adl_1[period] = total_adl_1 / n_agents
        share_adl_2_3[period] = total_adl_2_3 / n_agents
        share_any_adl[period] = share_adl_1[period] + share_adl_2_3[period]

    # Convert periods to agent ages for x-axis
    start_age = specs["start_age"]
    periods = np.arange(n_periods)
    ages = start_age + periods

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ages, share_no_adl, label="No ADL", linewidth=2, color="green")
    ax.plot(ages, share_adl_1, label="ADL 1", linewidth=2, color="orange")
    ax.plot(ages, share_adl_2_3, label="ADL 2/3", linewidth=2, color="red")
    ax.plot(
        ages,
        share_any_adl,
        label="Any ADL",
        linewidth=2,
        color="purple",
        linestyle="--",
    )

    ax.set_xlabel("Agent Age")
    ax.set_ylabel("Share of Population")
    ax.set_title("Share of ADL States Over Time (No ADL includes dead mothers)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Add vertical line at start age
    ax.axvline(x=start_age, color="gray", linestyle="--", alpha=0.5, label="Start Age")

    plt.tight_layout()
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save, dpi=300)
    plt.close(fig)

    print(f"ADL transition plot saved to {path_to_save}")


@pytask.mark.plot_mother_adl
def task_plot_adl_transition_adl_only(  # noqa: PLR0915
    path_to_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "mother_adl_transition_adl_only.png",
):
    """Plot share of ADL states (Any ADL, ADL 1, ADL 2/3) over time.

    Starts at period 0 with distribution from initial states, then simulates
    forward using ADL transition probabilities. Only shows ADL categories.

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
    mother_adl_initial = np.asarray(states["mother_adl"], dtype=np.uint8)
    education = np.asarray(states["education"], dtype=np.uint8)

    n_edu = specs["n_education_types"]
    n_adl_states = specs[
        "n_adl_states_light_intensive"
    ]  # 3 states: No ADL, ADL 1, ADL 2/3

    n_agents = len(mother_dead_initial)

    # Initialize share arrays
    share_no_adl = np.zeros(n_periods)
    share_adl_1 = np.zeros(n_periods)
    share_adl_2_3 = np.zeros(n_periods)
    share_any_adl = np.zeros(n_periods)

    # Initial shares
    alive_mask = mother_dead_initial == 0
    dead_mask = mother_dead_initial == 1

    # No ADL includes: alive with no ADL + dead (dead = no ADL)
    no_adl_alive = ((mother_adl_initial == 0) & alive_mask).sum()
    n_dead_initial = dead_mask.sum()
    share_no_adl[0] = (no_adl_alive + n_dead_initial) / n_agents

    share_adl_1[0] = np.mean((mother_adl_initial == ADL_1) & alive_mask)
    share_adl_2_3[0] = np.mean((mother_adl_initial == ADL_2_3) & alive_mask)
    share_any_adl[0] = share_adl_1[0] + share_adl_2_3[0]

    # Track ADL state counts by group (edu, adl_state) for alive mothers only
    adl_by_group = {}
    # Track dead mothers by group (edu)
    dead_by_group = {}
    for edu in range(n_edu):
        # Track alive mothers by ADL state
        for adl in range(n_adl_states):
            mask = (
                (education == edu)
                & (mother_dead_initial == 0)
                & (mother_adl_initial == adl)
            )
            adl_by_group[(edu, adl)] = float(mask.sum())
        # Track dead mothers
        mask_dead = (education == edu) & (mother_dead_initial == 1)
        dead_by_group[edu] = float(mask_dead.sum())

    # Simulate forward deterministically using transition probs
    for period in range(1, n_periods):
        adl_next = {}
        dead_next = {}
        for edu in range(n_edu):
            # Dead mothers stay dead (and count as "No ADL")
            dead_curr = dead_by_group.get(edu, 0.0)
            dead_next[edu] = dead_curr

            for adl_curr_state in range(n_adl_states):
                count_curr = adl_by_group.get((edu, adl_curr_state), 0.0)

                if count_curr == 0:
                    # No one in this state, nothing to transition
                    continue

                # First, account for death transitions
                # death_transition returns [alive_prob, dead_prob]
                death_prob_vector = death_transition(period - 1, 0, edu, specs)
                alive_prob = float(death_prob_vector[0])
                dead_prob = float(death_prob_vector[1])

                # Only alive mothers have ADL states
                count_alive = count_curr * alive_prob
                count_died = count_curr * dead_prob

                # Add newly dead to dead count (they count as "No ADL")
                dead_next[edu] = dead_next.get(edu, 0.0) + count_died

                if count_alive == 0:
                    # All died, nothing to transition
                    continue

                # Get ADL transition probabilities for alive mothers
                # Returns [prob_no_adl, prob_adl_1, prob_adl_2/3]
                adl_prob_vector = limitations_with_adl_transition(
                    adl_curr_state, period - 1, edu, specs
                )

                # Distribute alive count to next ADL states
                # according to transition probs
                for adl_next_state in range(n_adl_states):
                    prob = float(adl_prob_vector[adl_next_state])
                    key = (edu, adl_next_state)
                    adl_next[key] = adl_next.get(key, 0.0) + count_alive * prob

        # Update counts
        adl_by_group = adl_next
        dead_by_group = dead_next

        # Calculate total shares
        # No ADL includes: alive with no ADL + dead (dead = no ADL)
        total_no_adl_alive = sum(
            adl_by_group.get((edu, 0), 0.0) for edu in range(n_edu)
        )
        total_dead = sum(dead_by_group.get(edu, 0.0) for edu in range(n_edu))
        total_adl_1 = sum(adl_by_group.get((edu, 1), 0.0) for edu in range(n_edu))
        total_adl_2_3 = sum(adl_by_group.get((edu, 2), 0.0) for edu in range(n_edu))

        # Normalize by total agents (including dead ones)
        share_no_adl[period] = (total_no_adl_alive + total_dead) / n_agents
        share_adl_1[period] = total_adl_1 / n_agents
        share_adl_2_3[period] = total_adl_2_3 / n_agents
        share_any_adl[period] = share_adl_1[period] + share_adl_2_3[period]

    # Convert periods to agent ages for x-axis
    start_age = specs["start_age"]
    periods = np.arange(n_periods)
    ages = start_age + periods

    # Create plot - only ADL categories
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot with colors matching light_intensive_adl_transitions_and_shares.png style
    ax.plot(
        ages,
        share_any_adl,
        label="Any ADL",
        linewidth=2,
        color="blue",
        linestyle="--",
    )
    ax.plot(ages, share_adl_1, label="ADL 1", linewidth=2, color="green")
    ax.plot(ages, share_adl_2_3, label="ADL 2/3", linewidth=2, color="red")

    # Set y-axis limits and ticks
    y_max = 0.16
    ax.set_ylim([0, y_max])
    # Set yticks every 0.02
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.02))

    ax.set_xlabel("Agent Age")
    ax.set_ylabel("Share in Population")
    ax.set_title("Share of ADL States Over Time (ADL Categories Only)")
    ax.legend(title="Care level", fontsize=9, title_fontsize=10, loc="best")

    # Grid: major and minor ticks every 0.02
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.02), minor=True)
    ax.grid(True, which="minor", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.5)

    # Add vertical line at start age
    ax.axvline(x=start_age, color="gray", linestyle="--", alpha=0.5, label="Start Age")

    plt.tight_layout()
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save, dpi=300)
    plt.close(fig)

    print(f"ADL transition plot (ADL only) saved to {path_to_save}")
