"""Plot expected share of agents facing care demand implied by the transition.

This version uses the *simulated initial conditions* for mothers'
health status (from ``states.pkl``) as the starting distribution and
then propagates this distribution forward using the structural health
transition. At each age, we combine the current mother-health
distribution with the care-demand transition to obtain the expected
share of women who face any care demand.

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

from caregiving.config import BLD
from caregiving.model.stochastic_processes.caregiving_transition import (
    care_demand_and_supply_transition,
    health_transition_good_medium_bad,
)


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
    """Compute expected share with care demand by age.

    Parameters
    ----------
    options :
        Full options dictionary as loaded from ``options.pkl``.
    states :
        Initial discrete state arrays as stored in
        ``bld/model/initial_conditions/states.pkl``. Must contain
        ``period``, ``education``, ``has_sister`` and ``mother_health``.
    """
    specs = options["model_params"]

    start_age_child = int(specs["start_age"])
    end_age_child = int(specs["end_age_caregiving"])
    ages_child = np.arange(start_age_child, end_age_child + 1, dtype=int)

    # Extract initial distributions from states.pkl (100_000 agents).
    edu_agents = np.asarray(states["education"])
    sister_agents = np.asarray(states["has_sister"])
    mh_agents = np.asarray(states["mother_health"])

    # We treat the observed period in states as t = 0.
    # (If not all zeros, this is still consistent as long as
    # care_demand_and_supply_transition and health transitions use
    # "period" in the same way.)

    records: list[dict] = []

    for has_sister in has_sister_values:
        for education in education_values:
            # Subset of agents of this type
            mask_type = (edu_agents == education) & (sister_agents == has_sister)
            if not np.any(mask_type):
                continue

            # Initial mother-health distribution at period 0 for this type
            mh_sub = mh_agents[mask_type]
            counts = np.bincount(mh_sub, minlength=4).astype(float)
            dist_mh = counts / counts.sum()

            # Simulate forward mother-health distribution and care-demand share
            pi = dist_mh.copy()

            for age_child in ages_child:
                period = age_child - start_age_child

                # Expected care demand at this period for this type
                expected_share = 0.0
                for mother_health_state, weight in enumerate(pi):
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
                    expected_share += weight * p_demand

                records.append(
                    {
                        "age": age_child,
                        "has_sister": has_sister,
                        "education": education,
                        "share_care_demand": expected_share,
                    }
                )

                # Update mother-health distribution for next period
                new_pi = np.zeros_like(pi)
                for mother_health_state, weight in enumerate(pi):
                    if weight == 0:
                        continue
                    trans = health_transition_good_medium_bad(
                        mother_health=mother_health_state,
                        education=education,
                        has_sister=has_sister,
                        period=period,
                        options=specs,
                    )
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
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "expected_care_demand_shares_by_age.png",
) -> None:
    options = pickle.load(path_to_options.open("rb"))
    plot_expected_care_demand_share_by_age(
        options=options,
        path_to_states=path_to_states,
        path_to_save_plot=path_to_save_plot,
    )


@pytask.mark.stochastic_processes_expected_care_demand
def task_plot_expected_care_demand_shares_from_health_tables(  # noqa: D103
    path_to_options: Path = BLD / "model" / "options.pkl",
    path_to_health_by_age: Path = BLD
    / "model"
    / "initial_conditions"
    / "health_by_age.csv",
    path_to_survival_by_age: Path = BLD
    / "model"
    / "initial_conditions"
    / "survival_by_age.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "expected_care_demand_shares_by_age_from_health_tables.png",
) -> None:
    options = pickle.load(path_to_options.open("rb"))
    plot_expected_care_demand_share_by_age_from_health_tables(
        options=options,
        path_to_health_by_age=path_to_health_by_age,
        path_to_survival_by_age=path_to_survival_by_age,
        path_to_save_plot=path_to_save_plot,
    )
