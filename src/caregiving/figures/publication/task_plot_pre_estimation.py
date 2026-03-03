"""Publication plot: pre-estimation care demand by age."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import PUBLICATION_PLOT_STYLE
from caregiving.model.stochastic_processes.adl_transition import (
    death_transition,
    limitations_with_adl_transition,
)
from caregiving.model.stochastic_processes.caregiving_transition import (
    care_demand_transition_adl_light_intensive,
)


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


@pytask.mark.publication
@pytask.mark.publication_pre_estimation
def task_plot_care_demand_transition_adl_shaded(  # noqa: PLR0912, PLR0915
    path_to_states: Path = BLD / "model" / "initial_conditions" / "initial_states.pkl",
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "pre_estimation"
    / "care_demand_by_age_pre_shaded.pdf",
):
    """Plot care demand shares with shaded light/intensive areas beneath total line."""

    with path_to_states.open("rb") as f:
        states = pickle.load(f)

    with path_to_full_specs.open("rb") as f:
        specs = pickle.load(f)

    start_age = specs["start_age"]
    plot_max_age = 75
    n_periods = max(
        specs["n_periods"],
        plot_max_age - start_age + 1,
    )
    specs_plot = {**specs, "end_age_caregiving": plot_max_age + 1}

    mother_dead_initial = np.asarray(states["mother_dead"], dtype=np.uint8)
    mother_adl_initial = np.asarray(states["mother_adl"], dtype=np.uint8)
    education = np.asarray(states["education"], dtype=np.uint8)

    n_edu = specs["n_education_types"]
    n_adl_states = 3
    n_care_demand_states = 3

    n_agents = len(mother_dead_initial)

    share_no_care = np.zeros(n_periods)
    share_light = np.zeros(n_periods)
    share_intensive = np.zeros(n_periods)
    share_any_care = np.zeros(n_periods)

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

    care_demand_by_group_initial = {}
    for edu in range(n_edu):
        for adl in range(n_adl_states):
            for dead in (0, 1):
                count = state_by_group.get((edu, adl, dead), 0.0)
                if count == 0:
                    continue

                care_demand_probs = care_demand_transition_adl_light_intensive(
                    adl, dead, 0, edu, specs_plot
                )

                for care_demand_state in range(n_care_demand_states):
                    prob = float(care_demand_probs[care_demand_state])
                    key = (edu, care_demand_state)
                    care_demand_by_group_initial[key] = (
                        care_demand_by_group_initial.get(key, 0.0) + count * prob
                    )

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

    for period in range(1, n_periods):
        state_next = {}
        for edu in range(n_edu):
            count_dead = (
                state_by_group.get((edu, 0, 1), 0.0)
                + state_by_group.get((edu, 1, 1), 0.0)
                + state_by_group.get((edu, 2, 1), 0.0)
            )
            if count_dead > 0:
                state_next[(edu, 0, 1)] = count_dead

            for adl_curr in range(n_adl_states):
                count_alive = state_by_group.get((edu, adl_curr, 0), 0.0)
                if count_alive == 0:
                    continue

                death_prob_vector = death_transition(period - 1, 0, edu, specs)
                alive_prob = float(death_prob_vector[0])
                dead_prob = float(death_prob_vector[1])

                count_dies = count_alive * dead_prob
                count_survives = count_alive * alive_prob

                state_next[(edu, 0, 1)] = state_next.get((edu, 0, 1), 0.0) + count_dies

                if count_survives == 0:
                    continue

                adl_prob_vector = limitations_with_adl_transition(
                    adl_curr, period - 1, edu, specs
                )

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

        state_by_group = state_next

        care_demand_by_group = {}
        for edu in range(n_edu):
            for adl in range(n_adl_states):
                for dead in (0, 1):
                    count = state_by_group.get((edu, adl, dead), 0.0)
                    if count == 0:
                        continue

                    care_demand_probs = care_demand_transition_adl_light_intensive(
                        adl, dead, period, edu, specs_plot
                    )

                    for care_demand_state in range(n_care_demand_states):
                        prob = float(care_demand_probs[care_demand_state])
                        key = (edu, care_demand_state)
                        care_demand_by_group[key] = (
                            care_demand_by_group.get(key, 0.0) + count * prob
                        )

        total_no_care = sum(
            care_demand_by_group.get((edu, 0), 0.0) for edu in range(n_edu)
        )
        total_light = sum(
            care_demand_by_group.get((edu, 1), 0.0) for edu in range(n_edu)
        )
        total_intensive = sum(
            care_demand_by_group.get((edu, 2), 0.0) for edu in range(n_edu)
        )

        share_no_care[period] = total_no_care / n_agents
        share_light[period] = total_light / n_agents
        share_intensive[period] = total_intensive / n_agents
        share_any_care[period] = share_light[period] + share_intensive[period]

    periods = np.arange(n_periods)
    ages = start_age + periods

    age_min_plot, age_max_plot = 40, 75
    mask = (ages >= age_min_plot) & (ages <= age_max_plot)
    ages_plot = ages[mask]
    share_any_plot = share_any_care[mask]
    share_light_plot = share_light[mask]
    share_intensive_plot = share_intensive[mask]

    style = PUBLICATION_PLOT_STYLE
    # Use DejaVu Sans (original matplotlib default / fallback look)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        ages_plot,
        share_any_plot,
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )
    ax.plot(
        ages_plot,
        share_light_plot,
        color="0.3",
        linewidth=linewidth - 1.0,
        linestyle="-",
    )

    ax.fill_between(
        ages_plot,
        0,
        share_light_plot,
        color="0.9",
        alpha=0.5,
    )
    ax.fill_between(
        ages_plot,
        share_light_plot,
        share_any_plot,
        color="0.6",
        alpha=0.5,
    )

    share_intensive_area = share_intensive_plot

    ax.set_xlim(age_min_plot - 0.5, age_max_plot + 0.5)
    ax.set_ylim(-0.005, 0.2)

    light_idx = int(np.argmax(share_light_plot))
    intensive_idx = int(np.argmax(share_intensive_area))

    # Use fig.text with ax.transData so labels are not clipped by the axes
    if share_light_plot[light_idx] > 0:
        fig.text(
            ages_plot[light_idx],
            share_light_plot[light_idx] / 2,
            "Light",
            color="0.2",
            fontsize=14,
            ha="center",
            va="center",
            transform=ax.transData,
        )

    if share_intensive_area[intensive_idx] > 0:
        fig.text(
            ages_plot[intensive_idx],
            share_light_plot[intensive_idx] + share_intensive_area[intensive_idx] / 2,
            "Intensive",
            color="0.1",
            fontsize=14,
            ha="center",
            va="center",
            transform=ax.transData,
        )
    ax.set_yticks(np.arange(0, 0.21, 0.05))

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Share", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    # Embed TrueType fonts in PDF so the chosen serif font renders correctly
    if path_to_save.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_save,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_save.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Shaded care demand plot saved to {path_to_save}")


@pytask.mark.publication
@pytask.mark.publication_pre_estimation
def task_plot_ever_care_demand_by_age_shaded(  # noqa: PLR0912, PLR0915
    path_to_states: Path = BLD / "model" / "initial_conditions" / "initial_states.pkl",
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "pre_estimation"
    / "ever_care_demand_by_age_pre_shaded.pdf",
):
    """Plot share of agents who have EVER (up to that age) experienced care demand, with shaded light/intensive. Same style as care demand shaded plot."""

    with path_to_states.open("rb") as f:
        states = pickle.load(f)

    with path_to_full_specs.open("rb") as f:
        specs = pickle.load(f)

    start_age = specs["start_age"]
    plot_max_age = 75
    n_periods = max(
        specs["n_periods"],
        plot_max_age - start_age + 1,
    )
    specs_plot = {**specs, "end_age_caregiving": plot_max_age + 1}

    mother_dead_initial = np.asarray(states["mother_dead"], dtype=np.uint8)
    mother_adl_initial = np.asarray(states["mother_adl"], dtype=np.uint8)
    education = np.asarray(states["education"], dtype=np.uint8)

    n_edu = specs["n_education_types"]
    n_adl_states = 3
    n_care_demand_states = 3

    n_agents = len(mother_dead_initial)

    # state key: (edu, adl, dead, ever_light, ever_intensive); ever_* in {0, 1}
    state_by_group = {}
    for edu in range(n_edu):
        for adl in range(n_adl_states):
            for dead in (0, 1):
                mask = (
                    (education == edu)
                    & (mother_adl_initial == adl)
                    & (mother_dead_initial == dead)
                )
                state_by_group[(edu, adl, dead, 0, 0)] = float(mask.sum())

    ever_any = np.zeros(n_periods)
    ever_light_only = np.zeros(n_periods)
    ever_intensive_ever = np.zeros(n_periods)

    def _cumulative_shares(sbg):
        """Share with (e_l or e_i), (e_l and not e_i), (e_i).
        ever_any is truly cumulative (non-decreasing). ever_light_only can decrease
        because agents leave it when they experience intensive care (reclassified into
        ever_intensive_ever). ever_light_only + ever_intensive_ever = ever_any (disjoint).
        """
        share_any = 0.0
        share_light_only = 0.0
        share_intensive = 0.0
        for key, count in sbg.items():
            if len(key) == 5:
                _edu, _adl, _dead, e_l, e_i = key
            else:
                continue
            share_any += count if (e_l or e_i) else 0.0
            share_light_only += count if (e_l and not e_i) else 0.0
            share_intensive += count if e_i else 0.0
        return share_any / n_agents, share_light_only / n_agents, share_intensive / n_agents

    # Period 0: apply care demand to get initial ever_light / ever_intensive
    sbg_new = {}
    for (edu, adl, dead, e_l, e_i), count in state_by_group.items():
        if count == 0:
            continue
        probs = care_demand_transition_adl_light_intensive(
            adl, dead, 0, edu, specs_plot
        )
        p_no = float(probs[0])
        p_light = float(probs[1])
        p_intensive = float(probs[2])
        key_n = (edu, adl, dead, e_l, e_i)
        key_l = (edu, adl, dead, 1, e_i)
        key_i = (edu, adl, dead, e_l, 1)
        sbg_new[key_n] = sbg_new.get(key_n, 0.0) + count * p_no
        sbg_new[key_l] = sbg_new.get(key_l, 0.0) + count * p_light
        sbg_new[key_i] = sbg_new.get(key_i, 0.0) + count * p_intensive
    state_by_group = sbg_new
    ever_any[0], ever_light_only[0], ever_intensive_ever[0] = _cumulative_shares(
        state_by_group
    )

    for period in range(1, n_periods):
        # Step 1: transition (adl, dead); keep (edu, ever_light, ever_intensive)
        state_after_transition = {}
        for (edu, adl, dead, e_l, e_i), count in state_by_group.items():
            if count == 0:
                continue
            if dead == 1:
                state_after_transition[(edu, 0, 1, e_l, e_i)] = (
                    state_after_transition.get((edu, 0, 1, e_l, e_i), 0.0) + count
                )
                continue
            death_prob_vector = death_transition(period - 1, 0, edu, specs)
            alive_prob = float(death_prob_vector[0])
            dead_prob = float(death_prob_vector[1])
            count_dies = count * dead_prob
            count_survives = count * alive_prob
            state_after_transition[(edu, 0, 1, e_l, e_i)] = (
                state_after_transition.get((edu, 0, 1, e_l, e_i), 0.0) + count_dies
            )
            if count_survives == 0:
                continue
            adl_prob_vector = limitations_with_adl_transition(
                adl, period - 1, edu, specs
            )
            for adl_next, w in enumerate(
                (float(adl_prob_vector[0]), float(adl_prob_vector[1]), float(adl_prob_vector[2]))
            ):
                if w > 0:
                    state_after_transition[(edu, adl_next, 0, e_l, e_i)] = (
                        state_after_transition.get((edu, adl_next, 0, e_l, e_i), 0.0)
                        + count_survives * w
                    )

        # Step 2: apply care demand and update ever_light, ever_intensive
        state_next = {}
        for (edu, adl, dead, e_l, e_i), count in state_after_transition.items():
            if count == 0:
                continue
            probs = care_demand_transition_adl_light_intensive(
                adl, dead, period, edu, specs_plot
            )
            p_no = float(probs[0])
            p_light = float(probs[1])
            p_intensive = float(probs[2])
            state_next[(edu, adl, dead, e_l, e_i)] = (
                state_next.get((edu, adl, dead, e_l, e_i), 0.0) + count * p_no
            )
            state_next[(edu, adl, dead, 1, e_i)] = (
                state_next.get((edu, adl, dead, 1, e_i), 0.0) + count * p_light
            )
            state_next[(edu, adl, dead, e_l, 1)] = (
                state_next.get((edu, adl, dead, e_l, 1), 0.0) + count * p_intensive
            )

        state_by_group = state_next
        ever_any[period], ever_light_only[period], ever_intensive_ever[period] = (
            _cumulative_shares(state_by_group)
        )

    periods = np.arange(n_periods)
    ages = start_age + periods

    age_min_plot, age_max_plot = 40, 75
    mask = (ages >= age_min_plot) & (ages <= age_max_plot)
    ages_plot = ages[mask]
    share_any_plot = ever_any[mask]
    share_light_only_plot = ever_light_only[mask]
    share_intensive_plot = ever_intensive_ever[mask]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        ages_plot,
        share_any_plot,
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )
    ax.plot(
        ages_plot,
        share_light_only_plot,
        color="0.3",
        linewidth=linewidth - 1.0,
        linestyle="-",
    )

    ax.fill_between(
        ages_plot,
        0,
        share_light_only_plot,
        color="0.9",
        alpha=0.5,
    )
    ax.fill_between(
        ages_plot,
        share_light_only_plot,
        share_any_plot,
        color="0.6",
        alpha=0.5,
    )

    y_max = min(1.0, max(share_any_plot.max() * 1.05, 0.2))
    ax.set_xlim(age_min_plot - 0.5, age_max_plot + 0.5)
    ax.set_ylim(-0.005, y_max)
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.1 if y_max > 0.5 else 0.05))

    light_idx = int(np.argmax(share_light_only_plot)) if share_light_only_plot.max() > 0 else 0
    intensive_idx = (
        int(np.argmax(share_intensive_plot)) if share_intensive_plot.max() > 0 else 0
    )

    if share_light_only_plot[light_idx] > 0:
        fig.text(
            ages_plot[light_idx],
            share_light_only_plot[light_idx] / 2,
            "Light",
            color="0.2",
            fontsize=14,
            ha="center",
            va="center",
            transform=ax.transData,
        )

    if share_intensive_plot[intensive_idx] > 0:
        mid_idx = len(ages_plot) // 2
        fig.text(
            ages_plot[mid_idx],
            share_light_only_plot[mid_idx]
            + share_intensive_plot[mid_idx] / 2,
            "Intensive",
            color="0.1",
            fontsize=14,
            ha="center",
            va="center",
            transform=ax.transData,
        )

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Share", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    if path_to_save.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_save,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_save.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Ever care demand (shaded) plot saved to {path_to_save}")


@pytask.mark.publication
@pytask.mark.publication_pre_estimation
def task_plot_mother_alive_share_by_age(  # noqa: PLR0912, PLR0915
    path_to_states: Path = BLD / "model" / "initial_conditions" / "initial_states.pkl",
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "pre_estimation"
    / "mother_alive_share_by_age_pre.pdf",
):
    """Plot share of agents with mother alive (mother_dead == 0) by age. Same style as care demand shaded plot."""

    with path_to_states.open("rb") as f:
        states = pickle.load(f)

    with path_to_full_specs.open("rb") as f:
        specs = pickle.load(f)

    start_age = specs["start_age"]
    plot_max_age = 75
    n_periods = max(
        specs["n_periods"],
        plot_max_age - start_age + 1,
    )

    mother_dead_initial = np.asarray(states["mother_dead"], dtype=np.uint8)
    mother_adl_initial = np.asarray(states["mother_adl"], dtype=np.uint8)
    education = np.asarray(states["education"], dtype=np.uint8)

    n_edu = specs["n_education_types"]
    n_adl_states = 3

    n_agents = len(mother_dead_initial)

    share_mother_alive = np.zeros(n_periods)

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

    # Period 0: share with mother_dead == 0
    count_alive_0 = sum(
        state_by_group.get((edu, adl, 0), 0.0)
        for edu in range(n_edu)
        for adl in range(n_adl_states)
    )
    share_mother_alive[0] = count_alive_0 / n_agents

    for period in range(1, n_periods):
        state_next = {}
        for edu in range(n_edu):
            count_dead = (
                state_by_group.get((edu, 0, 1), 0.0)
                + state_by_group.get((edu, 1, 1), 0.0)
                + state_by_group.get((edu, 2, 1), 0.0)
            )
            if count_dead > 0:
                state_next[(edu, 0, 1)] = count_dead

            for adl_curr in range(n_adl_states):
                count_alive = state_by_group.get((edu, adl_curr, 0), 0.0)
                if count_alive == 0:
                    continue

                death_prob_vector = death_transition(period - 1, 0, edu, specs)
                alive_prob = float(death_prob_vector[0])
                dead_prob = float(death_prob_vector[1])

                count_dies = count_alive * dead_prob
                count_survives = count_alive * alive_prob

                state_next[(edu, 0, 1)] = state_next.get((edu, 0, 1), 0.0) + count_dies

                if count_survives == 0:
                    continue

                adl_prob_vector = limitations_with_adl_transition(
                    adl_curr, period - 1, edu, specs
                )

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

        state_by_group = state_next

        count_alive = sum(
            state_by_group.get((edu, adl, 0), 0.0)
            for edu in range(n_edu)
            for adl in range(n_adl_states)
        )
        share_mother_alive[period] = count_alive / n_agents

    periods = np.arange(n_periods)
    ages = start_age + periods

    age_min_plot, age_max_plot = 40, 75
    mask = (ages >= age_min_plot) & (ages <= age_max_plot)
    ages_plot = ages[mask]
    share_plot = share_mother_alive[mask]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        ages_plot,
        share_plot,
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    ax.set_xlim(age_min_plot - 0.5, age_max_plot + 0.5)
    y_max = max(share_plot.max(), 0.2) * 1.05
    ax.set_ylim(-0.005, min(y_max, 1.0))
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Share", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    if path_to_save.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_save,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_save.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Mother alive share plot saved to {path_to_save}")
