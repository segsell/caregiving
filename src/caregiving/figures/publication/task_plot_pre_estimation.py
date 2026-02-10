"""Publication plot: pre-estimation care demand by age."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.stochastic_processes.adl_transition import (
    death_transition,
    limitations_with_adl_transition,
)
from caregiving.model.stochastic_processes.caregiving_transition import (
    care_demand_transition_adl_light_intensive,
)


@pytask.mark.publication
@pytask.mark.publication_pre_estimation
def task_plot_care_demand_transition_adl_and_dead_state(  # noqa: PLR0912, PLR0915
    path_to_states: Path = BLD / "model" / "initial_conditions" / "initial_states.pkl",
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "pre_estimation"
    / "care_demand_by_age_pre.pdf",
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

    start_age = specs["start_age"]
    # Extend simulation to age 75 for plot; otherwise care demand is forced to zero
    # at end_age_caregiving (often 70) by care_demand_transition_adl_light_intensive
    plot_max_age = 75
    n_periods = max(
        specs["n_periods"],
        plot_max_age - start_age + 1,
    )
    specs_plot = {**specs, "end_age_caregiving": plot_max_age + 1}

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
                    adl, dead, 0, edu, specs_plot
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
                        adl, dead, period, edu, specs_plot
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

    # Restrict to 40–75 so the line ends at 40 and 75 (whitespace to left/right of line)
    age_min_plot, age_max_plot = 40, 75
    mask = (ages >= age_min_plot) & (ages <= age_max_plot)
    ages_plot = ages[mask]
    share_any_plot = share_any_care[mask]
    share_light_plot = share_light[mask]
    share_intensive_plot = share_intensive[mask]

    # Create plot (narrower width so x-axis is less stretched)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Greyscale: any=dashed 0.8, light=solid 0.2, intensive=solid black (0)
    linewidth = 2.0
    offset = 0
    ax.plot(
        ages_plot,
        share_any_plot,
        color="0",  # black
        linewidth=linewidth,
        linestyle="-",
    )
    ax.plot(
        ages_plot,
        share_intensive_plot,
        color="0.3",
        linewidth=linewidth - offset,
        linestyle="--",
        # marker="s",
        # markersize=5,
        # markevery=1,
        # markerfacecolor="none",
        # markeredgewidth=1.5,
    )
    ax.plot(
        ages_plot,
        share_light_plot,
        color="0.7",
        linewidth=linewidth - offset,
        linestyle="-.",
        # marker="o",
        # markersize=5,
        # markevery=1,
        # markerfacecolor="none",
        # markeredgewidth=1.5,
    )

    # Axes: x 40–75 with whitespace on sides; y with small pad below 0
    ax.set_xlim(age_min_plot - 0.5, age_max_plot + 0.5)
    ax.set_ylim(-0.005, 0.2)

    # y ticks every 0.05
    ax.set_yticks(np.arange(0, 0.21, 0.05))

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Share", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8)

    # Horizontal grid only
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save, dpi=1200, bbox_inches="tight")
    plt.close(fig)

    print(f"Care demand by age plot saved to {path_to_save}")
