"""Plot care demand transition shares over time."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    CARE_DEMAND_INTENSIVE,
    CARE_DEMAND_LIGHT,
    NO_CARE_DEMAND,
)
from caregiving.model.stochastic_processes.adl_transition import (
    death_transition,
    limitations_with_adl_transition,
)
from caregiving.model.stochastic_processes.caregiving_transition import (
    care_demand_transition_adl_light_intensive,
)


@pytask.mark.pre_estimation
def task_plot_care_demand_transition(  # noqa: PLR0912, PLR0915
    path_to_states: Path = BLD / "model" / "initial_conditions" / "initial_states.pkl",
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "pre_estimation"
    / "care_demand_transition.png",
):
    """Plot share of care demand states over time.

    Starts at period 0 with distribution from initial states, then simulates
    forward using care demand transition probabilities. Shows care demand
    categories: No Care Demand, Light Care, and Intensive Care.
    Uses the 3-state care demand system with separate mother_dead state variable.

    Parameters
    ----------
    path_to_states : Path
        Path to initial states pkl file
    path_to_full_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_save : Path
        Path to save the plot

    """
    # Load initial states and specs
    with path_to_states.open("rb") as f:
        states = pickle.load(f)

    with path_to_full_specs.open("rb") as f:
        specs = pickle.load(f)

    n_periods = specs["n_periods"]
    start_age = specs["start_age"]

    # Convert to numpy for fast aggregation
    mother_dead_initial = np.asarray(states["mother_dead"], dtype=np.uint8)
    mother_adl_initial = np.asarray(states["mother_adl"], dtype=np.uint8)
    education = np.asarray(states["education"], dtype=np.uint8)

    n_edu = specs["n_education_types"]
    n_adl_states = 3  # 0=No ADL, 1=ADL 1, 2=ADL 2/3 (for alive mothers)
    n_care_demand_states = (
        3  # 0=NO_CARE_DEMAND, 1=CARE_DEMAND_LIGHT, 2=CARE_DEMAND_INTENSIVE
    )

    n_agents = len(mother_dead_initial)

    # Initialize share arrays
    share_no_care = np.zeros(n_periods)
    share_light = np.zeros(n_periods)
    share_intensive = np.zeros(n_periods)
    share_any_care = np.zeros(n_periods)

    # Track state counts by group (edu, mother_adl, mother_dead)
    # mother_adl: 0=No ADL, 1=ADL 1, 2=ADL 2/3 (only for alive mothers)
    # mother_dead: 0=alive, 1=dead
    state_by_group = {}
    for edu in range(n_edu):
        for adl in range(n_adl_states):
            for dead in (0, 1):
                mask = (
                    (education == edu)
                    & (mother_adl_initial == adl)
                    & (mother_dead_initial == dead)
                )
                state_by_group[(edu, adl, dead)] = float(mask.sum())

    # Calculate initial care demand shares using
    # care_demand_transition_adl_light_intensive.
    care_demand_by_group_initial = {}
    for edu in range(n_edu):
        for adl in range(n_adl_states):
            for dead in (0, 1):
                count = state_by_group.get((edu, adl, dead), 0.0)
                if count == 0:
                    continue

                # Get care demand transition probabilities
                # Returns [p_no_care, p_light, p_intensive]
                care_demand_probs = care_demand_transition_adl_light_intensive(
                    adl, dead, 0, edu, specs
                )

                # Distribute to care demand states
                for care_demand_state in range(n_care_demand_states):
                    prob = float(care_demand_probs[care_demand_state])
                    key = (edu, care_demand_state)
                    care_demand_by_group_initial[key] = (
                        care_demand_by_group_initial.get(key, 0.0) + count * prob
                    )

    # Initial shares
    total_no_care = sum(
        care_demand_by_group_initial.get((edu, NO_CARE_DEMAND), 0.0)
        for edu in range(n_edu)
    )
    total_light = sum(
        care_demand_by_group_initial.get((edu, CARE_DEMAND_LIGHT), 0.0)
        for edu in range(n_edu)
    )
    total_intensive = sum(
        care_demand_by_group_initial.get((edu, CARE_DEMAND_INTENSIVE), 0.0)
        for edu in range(n_edu)
    )

    share_no_care[0] = total_no_care / n_agents
    share_light[0] = total_light / n_agents
    share_intensive[0] = total_intensive / n_agents
    share_any_care[0] = share_light[0] + share_intensive[0]

    # Simulate forward using death_transition and limitations_with_adl_transition
    # then compute care demand using care_demand_transition_adl_light_intensive
    for period in range(1, n_periods):
        state_next = {}
        for edu in range(n_edu):
            # Handle dead mothers - they stay dead
            count_dead = (
                state_by_group.get((edu, 0, 1), 0.0)
                + state_by_group.get((edu, 1, 1), 0.0)
                + state_by_group.get((edu, 2, 1), 0.0)
            )
            if count_dead > 0:
                # Dead mothers have ADL=0 (no ADL)
                state_next[(edu, 0, 1)] = count_dead

            # Handle alive mothers
            for adl_curr in range(n_adl_states):
                count_alive = state_by_group.get((edu, adl_curr, 0), 0.0)
                if count_alive == 0:
                    continue

                # Get death transition probabilities
                death_prob_vector = death_transition(period - 1, 0, edu, specs)
                alive_prob = float(death_prob_vector[0])
                dead_prob = float(death_prob_vector[1])

                # Count who dies
                count_dies = count_alive * dead_prob
                count_survives = count_alive * alive_prob

                # Add newly dead (they have ADL=0)
                state_next[(edu, 0, 1)] = state_next.get((edu, 0, 1), 0.0) + count_dies

                if count_survives == 0:
                    continue

                # Get ADL transition probabilities for survivors
                adl_prob_vector = limitations_with_adl_transition(
                    adl_curr, period - 1, edu, specs
                )
                # adl_prob_vector: [prob_no_adl, prob_adl_1, prob_adl_2/3]

                # Distribute survivors to next ADL states (all alive)
                count_no_adl = count_survives * float(adl_prob_vector[0])
                count_adl_1 = count_survives * float(adl_prob_vector[1])
                count_adl_2_3 = count_survives * float(adl_prob_vector[2])

                state_next[(edu, 0, 0)] = (
                    state_next.get((edu, 0, 0), 0.0) + count_no_adl
                )
                state_next[(edu, 1, 0)] = state_next.get((edu, 1, 0), 0.0) + count_adl_1
                state_next[(edu, 2, 0)] = (
                    state_next.get((edu, 2, 0), 0.0) + count_adl_2_3
                )

        # Update state counts
        state_by_group = state_next

        # Compute care demand shares using care_demand_transition_adl_light_intensive
        care_demand_by_group = {}
        for edu in range(n_edu):
            for adl in range(n_adl_states):
                for dead in (0, 1):
                    count = state_by_group.get((edu, adl, dead), 0.0)
                    if count == 0:
                        continue

                    # Get care demand transition probabilities
                    care_demand_probs = care_demand_transition_adl_light_intensive(
                        adl, dead, period, edu, specs
                    )

                    # Distribute to care demand states
                    for care_demand_state in range(n_care_demand_states):
                        prob = float(care_demand_probs[care_demand_state])
                        key = (edu, care_demand_state)
                        care_demand_by_group[key] = (
                            care_demand_by_group.get(key, 0.0) + count * prob
                        )

        # Calculate total shares
        total_no_care = sum(
            care_demand_by_group.get((edu, NO_CARE_DEMAND), 0.0) for edu in range(n_edu)
        )
        total_light = sum(
            care_demand_by_group.get((edu, CARE_DEMAND_LIGHT), 0.0)
            for edu in range(n_edu)
        )
        total_intensive = sum(
            care_demand_by_group.get((edu, CARE_DEMAND_INTENSIVE), 0.0)
            for edu in range(n_edu)
        )

        # Normalize by total agents
        share_no_care[period] = total_no_care / n_agents
        share_light[period] = total_light / n_agents
        share_intensive[period] = total_intensive / n_agents
        share_any_care[period] = share_light[period] + share_intensive[period]

    # Convert periods to agent ages for x-axis
    periods = np.arange(n_periods)
    ages = start_age + periods

    # Create plot - similar style to mother_adl_transition_adl_only.png
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

    # Set y-axis limits and ticks (similar to ADL plot)
    y_max = 0.16
    ax.set_ylim([0, y_max])
    # Set yticks every 0.02
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.02))

    ax.set_xlabel("Agent Age")
    ax.set_ylabel("Share in Population")
    ax.set_title("Share of Care Demand States Over Time")
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

    print(f"Care demand transition plot saved to {path_to_save}")


@pytask.mark.pre_estimation
def task_plot_care_demand_transition_adl_and_dead_state(  # noqa: PLR0912, PLR0915
    path_to_states: Path = BLD / "model" / "initial_conditions" / "initial_states.pkl",
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "pre_estimation"
    / "care_demand_by_age.png",
):
    """Plot share of care demand states by age using ADL-based transitions.

    Computes care demand shares by age using the
    care_demand_transition_adl_light_intensive function with yearly inputs from
    death_transition and limitations_with_adl_transition. Shows care demand
    categories: No Care Demand, Light Care, and Intensive Care.

    Parameters
    ----------
    path_to_states : Path
        Path to initial states pkl file
    path_to_full_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_save : Path
        Path to save the plot

    """
    # Load initial states and specs
    with path_to_states.open("rb") as f:
        states = pickle.load(f)

    with path_to_full_specs.open("rb") as f:
        specs = pickle.load(f)

    n_periods = specs["n_periods"]
    start_age = specs["start_age"]

    # Convert to numpy for fast aggregation
    mother_dead_initial = np.asarray(states["mother_dead"], dtype=np.uint8)
    mother_adl_initial = np.asarray(states["mother_adl"], dtype=np.uint8)
    education = np.asarray(states["education"], dtype=np.uint8)

    n_edu = specs["n_education_types"]
    n_adl_states = 3  # 0=No ADL, 1=ADL 1, 2=ADL 2/3 (for alive mothers)
    n_care_demand_states = 3  # 0=no care, 1=light, 2=intensive

    n_agents = len(mother_dead_initial)

    # Initialize share arrays
    share_no_care = np.zeros(n_periods)
    share_light = np.zeros(n_periods)
    share_intensive = np.zeros(n_periods)
    share_any_care = np.zeros(n_periods)

    # Track state counts by group (edu, mother_adl, mother_dead)
    # mother_adl: 0=No ADL, 1=ADL 1, 2=ADL 2/3 (only for alive mothers)
    # mother_dead: 0=alive, 1=dead
    state_by_group = {}
    for edu in range(n_edu):
        for adl in range(n_adl_states):
            for dead in (0, 1):
                mask = (
                    (education == edu)
                    & (mother_adl_initial == adl)
                    & (mother_dead_initial == dead)
                )
                state_by_group[(edu, adl, dead)] = float(mask.sum())

    # Calculate initial care demand shares using
    # care_demand_transition_adl_light_intensive
    # This function expects mother_adl (0,1,2) and mother_dead (0,1)
    # as separate inputs
    care_demand_by_group_initial = {}
    for edu in range(n_edu):
        for adl in range(n_adl_states):
            for dead in (0, 1):
                count = state_by_group.get((edu, adl, dead), 0.0)
                if count == 0:
                    continue

                # Get care demand transition probabilities
                # Returns [p_no_care, p_light, p_intensive]
                care_demand_probs = care_demand_transition_adl_light_intensive(
                    adl, dead, 0, edu, specs
                )

                # Distribute to care demand states
                for care_demand_state in range(n_care_demand_states):
                    prob = float(care_demand_probs[care_demand_state])
                    key = (edu, care_demand_state)
                    care_demand_by_group_initial[key] = (
                        care_demand_by_group_initial.get(key, 0.0) + count * prob
                    )

    # Initial shares
    total_no_care = sum(
        care_demand_by_group_initial.get((edu, 0), 0.0) for edu in range(n_edu)
    )
    total_light = sum(
        care_demand_by_group_initial.get((edu, 1), 0.0) for edu in range(n_edu)
    )
    total_intensive = sum(
        care_demand_by_group_initial.get((edu, 2), 0.0) for edu in range(n_edu)
    )

    share_no_care[0] = total_no_care / n_agents
    share_light[0] = total_light / n_agents
    share_intensive[0] = total_intensive / n_agents
    share_any_care[0] = share_light[0] + share_intensive[0]

    # Simulate forward using death_transition and limitations_with_adl_transition
    # then compute care demand using care_demand_transition_adl_light_intensive
    for period in range(1, n_periods):
        state_next = {}
        for edu in range(n_edu):
            # Handle dead mothers - they stay dead
            count_dead = (
                state_by_group.get((edu, 0, 1), 0.0)
                + state_by_group.get((edu, 1, 1), 0.0)
                + state_by_group.get((edu, 2, 1), 0.0)
            )
            if count_dead > 0:
                # Dead mothers have ADL=0 (no ADL)
                state_next[(edu, 0, 1)] = count_dead

            # Handle alive mothers
            for adl_curr in range(n_adl_states):
                count_alive = state_by_group.get((edu, adl_curr, 0), 0.0)
                if count_alive == 0:
                    continue

                # Get death transition probabilities
                death_prob_vector = death_transition(period - 1, 0, edu, specs)
                alive_prob = float(death_prob_vector[0])
                dead_prob = float(death_prob_vector[1])

                # Count who dies
                count_dies = count_alive * dead_prob
                count_survives = count_alive * alive_prob

                # Add newly dead (they have ADL=0)
                state_next[(edu, 0, 1)] = state_next.get((edu, 0, 1), 0.0) + count_dies

                if count_survives == 0:
                    continue

                # Get ADL transition probabilities for survivors
                adl_prob_vector = limitations_with_adl_transition(
                    adl_curr, period - 1, edu, specs
                )
                # adl_prob_vector: [prob_no_adl, prob_adl_1, prob_adl_2/3]

                # Distribute survivors to next ADL states (all alive)
                count_no_adl = count_survives * float(adl_prob_vector[0])
                count_adl_1 = count_survives * float(adl_prob_vector[1])
                count_adl_2_3 = count_survives * float(adl_prob_vector[2])

                state_next[(edu, 0, 0)] = (
                    state_next.get((edu, 0, 0), 0.0) + count_no_adl
                )
                state_next[(edu, 1, 0)] = state_next.get((edu, 1, 0), 0.0) + count_adl_1
                state_next[(edu, 2, 0)] = (
                    state_next.get((edu, 2, 0), 0.0) + count_adl_2_3
                )

        # Update state counts
        state_by_group = state_next

        # Compute care demand shares using care_demand_transition_adl_light_intensive
        care_demand_by_group = {}
        for edu in range(n_edu):
            for adl in range(n_adl_states):
                for dead in (0, 1):
                    count = state_by_group.get((edu, adl, dead), 0.0)
                    if count == 0:
                        continue

                    # Get care demand transition probabilities
                    care_demand_probs = care_demand_transition_adl_light_intensive(
                        adl, dead, period, edu, specs
                    )

                    # Distribute to care demand states
                    for care_demand_state in range(n_care_demand_states):
                        prob = float(care_demand_probs[care_demand_state])
                        key = (edu, care_demand_state)
                        care_demand_by_group[key] = (
                            care_demand_by_group.get(key, 0.0) + count * prob
                        )

        # Calculate total shares
        total_no_care = sum(
            care_demand_by_group.get((edu, 0), 0.0) for edu in range(n_edu)
        )
        total_light = sum(
            care_demand_by_group.get((edu, 1), 0.0) for edu in range(n_edu)
        )
        total_intensive = sum(
            care_demand_by_group.get((edu, 2), 0.0) for edu in range(n_edu)
        )

        # Normalize by total agents
        share_no_care[period] = total_no_care / n_agents
        share_light[period] = total_light / n_agents
        share_intensive[period] = total_intensive / n_agents
        share_any_care[period] = share_light[period] + share_intensive[period]

    # Convert periods to agent ages for x-axis
    periods = np.arange(n_periods)
    ages = start_age + periods

    # Create plot
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

    # Set y-axis limits and ticks
    y_max = 0.16
    ax.set_ylim([0, y_max])
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.02))

    ax.set_xlabel("Agent Age")
    ax.set_ylabel("Share in Population")
    ax.set_title(
        "Share of Care Demand States by Age\n"
        "(using care_demand_transition_adl_light_intensive)"
    )
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

    print(f"Care demand by age plot saved to {path_to_save}")
