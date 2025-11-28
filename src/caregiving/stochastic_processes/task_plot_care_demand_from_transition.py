"""Plot expected share of agents facing care demand implied by the transition.

This version uses the *simulated initial conditions* for mothers'
ADL status (from ``states.pkl``) as the starting distribution and
then propagates this distribution forward using the ADL state transition
matrix (``adl_state_transition_mat_with_death``). At each age, we combine
the current mother ADL distribution with the care-demand transition to
obtain the expected share of women who face any care demand.

We do this separately for all combinations of ``has_sister`` in {0, 1}
and education in {0, 1}, and plot the resulting four age profiles in
one figure.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from matplotlib.ticker import FuncFormatter
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.model.shared import ADL_DEAD, MOTHER
from caregiving.model.stochastic_processes.caregiving_transition import (
    care_demand_and_supply_transition,
)
from caregiving.specs.derive_specs import read_and_derive_specs


def plot_expected_care_demand_share_by_age(
    *,
    options: dict,
    path_to_states: Path,
    path_to_save_plot: Path | None = None,
) -> None:
    """Create line plot of expected care-demand share by age.

    The resulting figure has age on the x-axis and the expected share
    of agents facing any care demand on the y-axis, with four lines
    corresponding to all combinations of has_sister in {0, 1} and
    education in {0, 1}.
    """
    with path_to_states.open("rb") as f:
        states = pickle.load(f)

    df = _expected_care_demand_share_by_age(
        options=options,
        states=states,
        has_sister_values=(0, 1),
        education_values=(0, 1),
    )

    specs = options["model_params"]
    edu_labels = specs.get("education_labels", ["Low edu", "High edu"])

    start_age = int(specs["start_age"])
    end_age = int(specs["end_age_caregiving"])

    fig, ax = plt.subplots(figsize=(8, 4))

    for has_sister in (0, 1):
        linestyle = "--" if has_sister == 0 else "-"
        sister_lbl = "No sister" if has_sister == 0 else "Has sister"

        for edu in (0, 1):
            colour = JET_COLOR_MAP[edu]
            edu_lbl = edu_labels[edu] if edu < len(edu_labels) else f"Edu {edu}"

            grp = df[(df["has_sister"] == has_sister) & (df["education"] == edu)]
            if len(grp) == 0:
                continue

            grp_sorted = grp.sort_values("age")
            label = f"{sister_lbl}, {edu_lbl}"

            ax.plot(
                grp_sorted["age"],
                grp_sorted["share_care_demand"],
                label=label,
                color=colour,
                linestyle=linestyle,
            )

    pad = 1
    ax.set_xlabel("Age")
    ax.set_ylabel("Expected share with care demand")
    ax.set_xlim(start_age - pad, end_age + pad)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if path_to_save_plot is not None:
        path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_to_save_plot, dpi=300)

    plt.close(fig)


def plot_expected_care_demand_share_by_age_from_health_tables(
    *,
    options: dict,
    path_to_health_by_age: Path,
    path_to_survival_by_age: Path,
    path_to_save_plot: Path | None = None,
) -> None:
    """Alternative plot using health-by-age and survival tables.

    This reproduces the earlier implementation which did *not* start
    from the simulated initial states, but instead combined:

    - P(alive at mother age) from ``survival_by_age.csv``, and
    - P(health | alive, mother age) from ``health_by_age.csv``,

    plus the care-demand transition. It is useful for comparing against
    the states-based implementation above.
    """
    health_prob_by_age = pd.read_csv(path_to_health_by_age, index_col=0)
    health_prob_by_age.index = health_prob_by_age.index.astype(int)

    survival_by_age = pd.read_csv(path_to_survival_by_age, index_col=0).squeeze(
        "columns"
    )
    survival_by_age.index = survival_by_age.index.astype(int)

    df = _expected_care_demand_share_by_age_from_health_tables(
        options=options,
        health_prob_by_age=health_prob_by_age,
        survival_by_age=survival_by_age,
        has_sister_values=(0, 1),
        education_values=(0, 1),
    )

    specs = options["model_params"]
    edu_labels = specs.get("education_labels", ["Low edu", "High edu"])

    fig, ax = plt.subplots(figsize=(8, 4))

    for (has_sister, education), grp in df.groupby(["has_sister", "education"]):
        label_has_sister = "Has sister" if has_sister == 1 else "No sister"
        label_edu = (
            edu_labels[education] if education < len(edu_labels) else f"Edu {education}"
        )
        label = f"{label_has_sister}, {label_edu}"

        grp_sorted = grp.sort_values("age")
        ax.plot(
            grp_sorted["age"],
            grp_sorted["share_care_demand"],
            label=label,
        )

    ax.set_xlabel("Age")
    ax.set_ylabel("Expected share with care demand")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if path_to_save_plot is not None:
        path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_to_save_plot, dpi=300)

    plt.close(fig)


def _expected_care_demand_share_by_age(
    *,
    options: dict,
    states: dict,
    has_sister_values: Iterable[int] = (0, 1),
    education_values: Iterable[int] = (0, 1),
) -> pd.DataFrame:
    """Compute expected share with care demand by age using mother_adl.

    Parameters
    ----------
    options :
        Full options dictionary as loaded from ``options.pkl``.
    states :
        Initial discrete state arrays as stored in
        ``bld/model/initial_conditions/states.pkl``. Must contain
        ``period``, ``education``, ``has_sister`` and ``mother_adl``.
    """
    specs = options["model_params"]

    start_age_child = int(specs["start_age"])
    end_age_child = int(specs["end_age_caregiving"])
    ages_child = np.arange(start_age_child, end_age_child + 1, dtype=int)

    # Extract initial distributions from states.pkl (100_000 agents).
    edu_agents = np.asarray(states["education"])
    sister_agents = np.asarray(states["has_sister"])
    mother_adl_agents = np.asarray(states["mother_adl"])

    # Get ADL transition matrix: [sex, period, adl_lag_state, adl_next_state]
    # adl_next_state: 0=No ADL, 1=ADL 1, 2=ADL 2, 3=ADL 3, 4=Death
    adl_trans_mat = np.asarray(specs["adl_state_transition_mat_with_death"])

    records: list[dict] = []

    for has_sister in has_sister_values:
        for education in education_values:
            # Subset of agents of this type
            mask_type = (edu_agents == education) & (sister_agents == has_sister)
            if not np.any(mask_type):
                continue

            # Initial mother ADL distribution at period 0 for this type
            adl_sub = mother_adl_agents[mask_type]
            counts = np.bincount(adl_sub, minlength=5).astype(float)
            dist_adl = counts / counts.sum()

            # Simulate forward mother ADL distribution and care-demand share
            pi = dist_adl.copy()

            for age_child in ages_child:
                period = age_child - start_age_child

                # Calculate mother's period for ADL transition matrix
                mother_period = (
                    period
                    + (specs["start_age"] - specs["start_age_parents"])
                    + specs["mother_age_diff"][has_sister, education]
                )

                # Clamp mother_period to valid range
                n_periods = adl_trans_mat.shape[1]
                mother_period = np.clip(mother_period, 0, n_periods - 1)

                # Expected care demand at this period for this type
                expected_share = 0.0
                for mother_adl_state, weight in enumerate(pi):
                    if weight == 0:
                        continue

                    prob_vec = care_demand_and_supply_transition(
                        mother_adl=mother_adl_state,
                        period=period,
                        has_sister=has_sister,
                        education=education,
                        options=specs,
                    )
                    # prob_vec = [no demand, demand+other supply, demand+no other]
                    p_demand = float(prob_vec[1] + prob_vec[2])
                    expected_share += weight * p_demand

                records.append(
                    {
                        "age": age_child,
                        "has_sister": has_sister,
                        "education": education,
                        "share_care_demand": expected_share,
                    }
                )

                # Update mother ADL distribution for next period using transition matrix
                # Death is now included as a lag state in the matrix (absorbing: death->death=1)
                new_pi = np.zeros_like(pi)
                for mother_adl_state, weight in enumerate(pi):
                    if weight == 0:
                        continue
                    # Get transition probabilities from current ADL state
                    # The matrix now includes death as a lag state (index 4)
                    trans = adl_trans_mat[MOTHER, mother_period, mother_adl_state, :]
                    new_pi += weight * np.asarray(trans)
                pi = new_pi

    return pd.DataFrame.from_records(records)


def _expected_care_demand_share_by_age_from_health_tables(
    *,
    options: dict,
    health_prob_by_age: pd.DataFrame,
    survival_by_age: pd.Series,
    has_sister_values: Iterable[int] = (0, 1),
    education_values: Iterable[int] = (0, 1),
) -> pd.DataFrame:
    """Compute expected share using health-by-age and survival tables.

    This mirrors the original implementation that worked directly with
    ``health_by_age.csv`` and ``survival_by_age.csv`` instead of the
    simulated initial states.
    """
    specs = options["model_params"]

    start_age_child = int(specs["start_age"])
    end_age_child = int(specs["end_age_caregiving"])
    ages_child = np.arange(start_age_child, end_age_child + 1, dtype=int)

    mother_age_diff = np.asarray(specs["mother_age_diff"])

    records: list[dict] = []

    for has_sister in has_sister_values:
        for education in education_values:
            diff = mother_age_diff[has_sister, education]

            for age_child in ages_child:
                period = age_child - start_age_child
                mother_age = int(round(age_child + diff))

                if mother_age not in health_prob_by_age.index:
                    # Outside empirical support of health-by-age; skip.
                    continue

                if mother_age not in survival_by_age.index:
                    continue

                prob_alive = float(survival_by_age.loc[mother_age])

                # Health distribution among ALIVE mothers at this age.
                # Columns are 0,1,2 for Bad/Medium/Good health.
                probs_alive = health_prob_by_age.loc[mother_age].to_numpy(
                    dtype=float
                )  # shape (3,)

                # Expected care demand share among *alive* mothers at
                # this age:
                expected_share_alive = 0.0
                for mother_health_state, weight in enumerate(probs_alive):
                    if weight == 0:
                        continue

                    prob_vec = care_demand_and_supply_transition(
                        mother_health=mother_health_state,
                        period=period,
                        has_sister=has_sister,
                        education=education,
                        options=specs,
                    )
                    # prob_vec = [no demand, demand+other supply, demand+no other]
                    p_demand = float(prob_vec[1] + prob_vec[2])
                    expected_share_alive += weight * p_demand

                # Unconditional share among all mothers (including
                # death) at this age:
                expected_share = prob_alive * expected_share_alive

                records.append(
                    {
                        "age": age_child,
                        "has_sister": has_sister,
                        "education": education,
                        "share_care_demand": expected_share,
                    }
                )

    return pd.DataFrame.from_records(records)


@pytask.mark.stochastic_processes_expected_care_demand
def task_plot_expected_care_demand_shares(  # noqa: D103
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "expected_care_demand_shares_by_age.png",
) -> None:
    specs = pickle.load(path_to_specs.open("rb"))
    # Create options-like dict with model_params key for compatibility
    options = {"model_params": specs}
    plot_expected_care_demand_share_by_age(
        options=options,
        path_to_states=path_to_states,
        path_to_save_plot=path_to_save_plot,
    )


def plot_adl_state_transitions_with_death(
    adl_state_transition_matrix: pd.DataFrame,
    specs: dict,
    path_to_save_plot: Path | None = None,
) -> plt.Figure:
    """
    Plot ADL state transition probabilities including death.

    2 x 5 grid:
        ┌───────────────┬───────────────┬───────────────┬───────────────┬───────────────┐
        │ Men / to 0    │ Men / to 1    │ Men / to 2    │ Men / to 3    │ Men / to Death│
        ├───────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
        │ Women / to 0  │ Women / to 1  │ Women / to 2  │ Women / to 3  │ Women / to Death│
        └───────────────┴───────────────┴───────────────┴───────────────┴───────────────┘

    Each subplot shows 5 lines (one for each adl_lag state: No ADL, ADL 1, ADL 2, ADL 3, Death).

    Parameters
    ----------
    adl_state_transition_matrix : pd.DataFrame
        DataFrame with columns: sex, age, adl_lag, adl_next, transition_prob
    specs : dict
        Full specification dictionary.
    path_to_save_plot : Path | None
        Path to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Prepare data
    df = adl_state_transition_matrix.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # ADL labels including death
    adl_labels = ["No ADL", "ADL 1", "ADL 2", "ADL 3", "Death"]
    adl_colors = {
        "No ADL": "blue",
        "ADL 1": "green",
        "ADL 2": "orange",
        "ADL 3": "red",
        "Death": "black",
    }

    start_age = specs["start_age_parents"]
    end_age = specs["end_age"]
    sex_labels = specs["sex_labels"]

    # Setup plot: 2 rows (Men/Women) x 5 columns (destination states)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(20, 6),
    )

    # Plot: 2 rows (Men/Women) x 5 columns (adl_next)
    for row, sex in enumerate(sex_labels):  # ["Men", "Women"]
        for col, adl_next in enumerate(adl_labels):
            ax = axes[row, col]

            # Plot one line for each adl_lag
            for adl_lag in adl_labels:
                dat = df[
                    (df["sex"] == sex)
                    & (df["adl_lag"] == adl_lag)
                    & (df["adl_next"] == adl_next)
                ].copy()

                if not dat.empty:
                    dat = dat.sort_values("age")
                    ax.plot(
                        dat["age"].values,
                        dat["transition_prob"].values,
                        label=adl_lag,
                        color=adl_colors[adl_lag],
                        linewidth=2,
                    )

            # Cosmetics
            if row == 1:
                ax.set_xlabel("Age")
            if col == 0:
                ax.set_ylabel("Probability")
            ax.set_title(f"{sex} - to {adl_next}")

            # Set limits with padding
            age_range = end_age - start_age
            y_range = 1.0 - 0.0
            padding_x = age_range * 0.05
            padding_y = y_range * 0.05

            ax.set_xlim(start_age - padding_x, end_age + padding_x)
            ax.set_xticks(np.arange(start_age, end_age + 1, 5))
            ax.set_ylim(0 - padding_y, 1 + padding_y)
            ax.grid(True, alpha=0.3)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(adl_labels),
        title="From ADL state",
    )

    if path_to_save_plot:
        path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_to_save_plot, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {path_to_save_plot}")

    return fig


@pytask.mark.stochastic_processes_expected_care_demand
def task_plot_adl_state_transitions_with_death(  # noqa: D103
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_adl_state_mat_with_death: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_state_transition_matrix_with_death.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "adl_state_transitions_with_death.png",
) -> None:
    """Plot ADL state transitions with death from CSV file."""
    specs = read_and_derive_specs(path_to_specs)
    adl_mat = pd.read_csv(path_to_adl_state_mat_with_death)

    plot_adl_state_transitions_with_death(
        adl_state_transition_matrix=adl_mat,
        specs=specs,
        path_to_save_plot=path_to_save_plot,
    )


# @pytask.mark.stochastic_processes_expected_care_demand
# def task_plot_expected_care_demand_shares_from_health_tables(  # noqa: D103
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_health_by_age: Path = BLD
#     / "model"
#     / "initial_conditions"
#     / "health_by_age.csv",
#     path_to_survival_by_age: Path = BLD
#     / "model"
#     / "initial_conditions"
#     / "survival_by_age.csv",
#     path_to_save_plot: Annotated[Path, Product] = BLD
#     / "estimation"
#     / "stochastic_processes"
#     / "expected_care_demand_shares_by_age_from_health_tables.png",
# ) -> None:
#     options = pickle.load(path_to_options.open("rb"))
#     plot_expected_care_demand_share_by_age_from_health_tables(
#         options=options,
#         path_to_health_by_age=path_to_health_by_age,
#         path_to_survival_by_age=path_to_survival_by_age,
#         path_to_save_plot=path_to_save_plot,
#     )
