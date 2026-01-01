"""Plot mother death probability for no care demand model.

Visualizes the probability of mother being recently dead (PARENT_RECENTLY_DEAD)
by simulating forward from initial states using the death_transition function.
"""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.stochastic_processes.adl_transition import death_transition


@pytask.mark.pre_estimation_no_care_demand
def task_plot_mother_dead_probability_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "initial_states_no_care_demand.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "pre_estimation"
    / "mother_dead_probability_no_care_demand.png",
):
    """Plot mother recently dead probability by age and education for no care demand model.

    Simulates forward from initial states distribution, tracking the probability of
    mother being recently dead (PARENT_RECENTLY_DEAD) over time by education.

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_initial_states : Path
        Path to initial states pkl file for no care demand model
    path_to_save_plot : Path
        Path to save the plot

    """
    # Load specs and initial states
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    with path_to_initial_states.open("rb") as f:
        initial_states = pickle.load(f)

    n_periods = specs["n_periods"]
    start_age = specs["start_age"]

    # Convert initial states to numpy
    mother_dead_initial = np.asarray(initial_states["mother_dead"], dtype=np.uint8)
    education_initial = np.asarray(initial_states["education"], dtype=np.uint8)
    n_agents = len(mother_dead_initial)

    # Track probability distribution over states by education
    # States: 0=alive, 1=recently_dead, 2=longer_dead
    n_edu = specs["n_education_types"]
    n_states = 3

    # Initialize state distributions by education
    # prob_by_state[edu][state] = probability of being in that state
    prob_by_state = {}
    for edu in range(n_edu):
        mask_edu = education_initial == edu
        n_edu_agents = mask_edu.sum()
        if n_edu_agents == 0:
            # If no agents in this education group, initialize to 0
            prob_by_state[edu] = {0: 0.0, 1: 0.0, 2: 0.0}
        else:
            prob_by_state[edu] = {
                0: float((mother_dead_initial[mask_edu] == 0).sum() / n_edu_agents),
                1: float((mother_dead_initial[mask_edu] == 1).sum() / n_edu_agents),
                2: float((mother_dead_initial[mask_edu] == 2).sum() / n_edu_agents),
            }

    # Store probabilities over time
    prob_recently_dead_by_edu = {edu: [] for edu in range(n_edu)}

    # Add initial period
    for edu in range(n_edu):
        prob_recently_dead_by_edu[edu].append(prob_by_state[edu][1])

    # Simulate forward period by period
    for period in range(n_periods):
        prob_by_state_next = {}
        for edu in range(n_edu):
            prob_by_state_next[edu] = {0: 0.0, 1: 0.0, 2: 0.0}

            # Current state distribution
            prob_alive = prob_by_state[edu][0]
            prob_recently_dead = prob_by_state[edu][1]
            prob_longer_dead = prob_by_state[edu][2]

            # Transition from alive (state 0)
            if prob_alive > 0:
                prob_vector = death_transition(
                    period=period,
                    mother_dead=0,
                    education=edu,
                    model_specs=specs,
                )
                # prob_vector = [alive_prob, recently_died_prob, longer_dead_prob]
                prob_by_state_next[edu][0] += prob_alive * float(prob_vector[0])
                prob_by_state_next[edu][1] += prob_alive * float(prob_vector[1])
                prob_by_state_next[edu][2] += prob_alive * float(prob_vector[2])

            # Transition from recently_dead (state 1) -> longer_dead (state 2) with certainty
            prob_by_state_next[edu][2] += prob_recently_dead

            # Transition from longer_dead (state 2) -> longer_dead (state 2) with certainty
            prob_by_state_next[edu][2] += prob_longer_dead

        # Update state distribution
        prob_by_state = prob_by_state_next

        # Store probability of recently_dead for this period
        for edu in range(n_edu):
            prob_recently_dead_by_edu[edu].append(prob_by_state[edu][1])

    # Convert periods to ages
    periods_plot = np.arange(n_periods + 1)  # +1 for initial period
    ages_plot = start_age + periods_plot

    # Colors for education levels
    edu_colors = [plt.cm.tab10(i) for i in range(len(specs["education_labels"]))]

    # =================================================================================
    # Plot: Mother recently dead probability by age and education
    # =================================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for edu_var, edu_label in enumerate(specs["education_labels"]):
        color = edu_colors[edu_var]
        probabilities = np.array(prob_recently_dead_by_edu[edu_var])

        ax.plot(
            ages_plot,
            probabilities,
            linewidth=2,
            color=color,
            label=edu_label,
            alpha=0.8,
        )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Probability of Mother Recently Dead", fontsize=12)
    ax.set_title(
        "Mother Recently Dead Probability by Age and Education (No Care Demand Model)\n"
        "Simulated from Initial States Distribution",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)

    print(f"Mother dead probability plot saved to {path_to_save_plot}")
