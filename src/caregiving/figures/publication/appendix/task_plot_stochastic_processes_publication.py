"""Publication-quality plots for stochastic processes (appendix).

Redoes the plots in BLD/plots/stochastic_processes with PUBLICATION_PLOT_STYLE,
saving to BLD/figures/publication/stochastic_processes. Women only.
Excludes inheritance probability and amount plots.
"""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
import numpy as np
import pandas as pd
import pytask
from linearmodels.panel import PanelOLS
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    PUBLICATION_PLOT_STYLE,
    publication_savefig,
)
from caregiving.model.shared import MAX_AGE_SIM

_OUT = BLD / "figures" / "publication" / "stochastic_processes"
_EST = BLD / "estimation" / "stochastic_processes"

EDU_COLOR_HIGH = "black"
EDU_COLOR_LOW = "0.6"

_S = PUBLICATION_PLOT_STYLE
_X_PAD = 1


def _edu_color(edu_idx: int) -> str:
    """High education (1) = black, Low (0) = gray."""
    return EDU_COLOR_HIGH if edu_idx == 1 else EDU_COLOR_LOW


def _markov_simulator(initial_dist, trans_probs, n_periods=None):
    """Simulate a Markov process."""
    if n_periods is None:
        n_periods = trans_probs.shape[0]
    n_states = initial_dist.shape[0]
    final_dist = np.zeros((n_periods, n_states))
    final_dist[0, :] = initial_dist
    for t in range(n_periods - 1):
        current_dist = final_dist[t, :]
        for state in range(n_states - 1):
            final_dist[t + 1, state] = current_dist @ trans_probs[t, :, state]
        final_dist[t + 1, -1] = 1 - final_dist[t + 1, :-1].sum()
    return final_dist


def _apply_style(ax):
    """Apply publication style to axis — matches event-study layout exactly.

    Mirrors plotting_helpers.py lines 605-624: grid, spines, tick sizes,
    and per-label fontsize + font family.
    """
    ax.grid(True, axis="y", alpha=_S["grid_alpha"], linewidth=_S["grid_linewidth"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=_S["tick_length"], width=_S["tick_width"])
    for label in ax.get_xticklabels():
        label.set_fontsize(_S["xtick_fontsize"])
        label.set_fontfamily(_S["font_family"])
    for label in ax.get_yticklabels():
        label.set_fontsize(_S["ytick_fontsize"])
        label.set_fontfamily(_S["font_family"])
    ax.xaxis.get_label().set_fontsize(_S["label_fontsize"])
    ax.xaxis.get_label().set_fontfamily(_S["font_family"])
    ax.yaxis.get_label().set_fontsize(_S["label_fontsize"])
    ax.yaxis.get_label().set_fontfamily(_S["font_family"])
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(_S["label_fontsize"])
            text.set_fontfamily(_S["font_family"])


def _finalize(ax, path, ymin=None, ymax=None, xmin=None, xmax=None):
    """Apply style, x/y-margins, tight_layout, subplots_adjust, save, close."""
    _apply_style(ax)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin - _X_PAD, xmax + _X_PAD)
    if ymin is None or ymax is None:
        ymin, ymax = ax.get_ylim()
    pad = _S["y_axis_margin_factor"] * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax)
    plt.tight_layout()
    plt.subplots_adjust(bottom=_S["subplots_adjust_bottom"])
    path.parent.mkdir(parents=True, exist_ok=True)
    publication_savefig(path)
    plt.close()


# ---------------------------------------------------------------------------
# Partner state (3 separate figures) — end at age 90 inclusive
# ---------------------------------------------------------------------------

_PARTNER_STATE_MAX_AGE = 90


def _plot_partner_state_one_panel(specs, df, partner_state_idx, path_to_save):
    start_age = specs["start_age"]
    ages = np.arange(start_age, _PARTNER_STATE_MAX_AGE + 1)
    n_partner_states = specs["n_partner_states"]
    sex_var = 1
    trans_mat = np.asarray(specs["partner_trans_mat"])
    n_periods = min(len(ages), trans_mat.shape[2])
    ages = ages[:n_periods]
    initial_dist = np.zeros(n_partner_states)
    grouped = df.groupby(["sex", "education", "age"])["partner_state"].value_counts(
        normalize=True
    )
    fig, ax = plt.subplots(figsize=_S["figsize"])
    for edu, edu_label in enumerate(specs["education_labels"]):
        try:
            p_single = grouped.loc[(sex_var, edu, start_age, 0)]
        except Exception:
            p_single = 0.5
        initial_dist[0] = float(p_single)
        initial_dist[1] = 1 - initial_dist[0]
        if n_partner_states == 3:
            initial_dist[2] = 0.0
        trans_probs = trans_mat[sex_var, edu, :n_periods, :, :]
        shares = _markov_simulator(initial_dist, trans_probs, n_periods)
        try:
            obs_series = grouped.loc[(sex_var, edu, slice(None), partner_state_idx)]
            obs_ages = np.asarray(obs_series.index)
            mask = obs_ages <= _PARTNER_STATE_MAX_AGE
            if np.any(mask):
                ax.plot(
                    obs_ages[mask],
                    obs_series.values[mask],
                    color=_edu_color(edu),
                    linestyle="--",
                    label=f"{edu_label} (obs.)",
                    linewidth=_S["linewidth"],
                )
        except Exception:
            pass
        ax.plot(
            ages,
            shares[:, partner_state_idx],
            color=_edu_color(edu),
            label=f"{edu_label} (est.)",
            linewidth=_S["linewidth"],
        )
    ax.set_xlabel("Age", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.set_ylabel("Share", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.legend(fontsize=_S["label_fontsize"], frameon=False, loc="best")
    _finalize(
        ax,
        path_to_save,
        ymin=0,
        ymax=1,
        xmin=start_age,
        xmax=_PARTNER_STATE_MAX_AGE,
    )


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_partner_state_no_partner_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = _OUT
    / "partner_state_no_partner_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])
    _plot_partner_state_one_panel(specs, df, 0, path_to_save)


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_partner_state_working_partner_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = _OUT
    / "partner_state_working_partner_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])
    _plot_partner_state_one_panel(specs, df, 1, path_to_save)


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_partner_state_retired_partner_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = _OUT
    / "partner_state_retired_partner_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])
    _plot_partner_state_one_panel(specs, df, 2, path_to_save)


# ---------------------------------------------------------------------------
# Partner wage — women agents, male partners
# Uses partner_wage_eq_params_women.csv (sex==1 in data = women agents)
# Reference plot: wages_partner_women.png
# ---------------------------------------------------------------------------


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_partner_wage_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_params: Path = _EST / "partner_wage_eq_params_women.csv",
    path_to_data: Path = BLD / "data" / "soep_partner_wage_data.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "partner_wage_women.pdf",
) -> None:
    """Partner (male) wage for women agents: observed (dashed) + predicted (solid)."""
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    params = pd.read_csv(path_to_params, index_col=0)
    start_age = specs["start_age"]
    x_max = 65

    df = pd.read_csv(path_to_data, index_col=0)
    df = df.loc[df["sex"] == 1].copy()
    df["period"] = df["age"] - start_age
    df["period_sq"] = df["period"] ** 2
    df = df.loc[(df["age"] >= start_age) & (df["age"] <= x_max)].copy()

    wage_col = "wage_p" if "wage_p" in df.columns else "wage"

    fig, ax = plt.subplots(figsize=_S["figsize"])
    for edu, edu_label in enumerate(specs["education_labels"]):
        row = params.loc[edu_label] if edu_label in params.index else params.iloc[edu]
        const = float(row["constant"])
        p1 = float(row["period"])
        p2 = float(row["period_sq"])

        sub = df.loc[df["education"] == edu].copy()
        if len(sub) == 0:
            continue
        sub["wage_pred"] = const + p1 * sub["period"] + p2 * sub["period_sq"]

        obs = sub.groupby("age")[wage_col].mean()
        ax.plot(
            obs.index,
            obs.values,
            color=_edu_color(edu),
            linestyle="--",
            label=f"{edu_label} (obs.)",
            linewidth=_S["linewidth"],
        )
        pred = sub.groupby("age")["wage_pred"].mean()
        ax.plot(
            pred.index,
            pred.values,
            color=_edu_color(edu),
            label=f"{edu_label} (est.)",
            linewidth=_S["linewidth"],
        )
    ax.set_xlabel("Age", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.set_ylabel(
        "Monthly gross income", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"]
    )
    ax.legend(fontsize=_S["label_fontsize"], frameon=False, loc="best")
    _finalize(ax, path_to_save, ymin=1000, xmin=start_age, xmax=x_max)


# ---------------------------------------------------------------------------
# Number of children (2 separate figures: Single / Partnered)
# ---------------------------------------------------------------------------

_CHILDREN_YLIM = (0, 2.5)
_CHILDREN_XMAX = 70


def _plot_children_one_panel(specs, df, has_partner, path_to_save):
    start_age = specs["start_age"]
    ages = np.arange(start_age, _CHILDREN_XMAX + 1)
    n_periods = len(ages)
    sex_var = 1
    children = np.asarray(specs["children_by_state"])
    nb_children_data = df.groupby(["sex", "education", "has_partner", "age"])[
        "children"
    ].mean()
    fig, ax = plt.subplots(figsize=_S["figsize"])
    for edu, edu_label in enumerate(specs["education_labels"]):
        vals = children[sex_var, edu, has_partner, :n_periods]
        try:
            obs = nb_children_data.loc[(sex_var, edu, has_partner, slice(None))]
            obs_ages = np.asarray(obs.index)
            mask = obs_ages <= _CHILDREN_XMAX
            if np.any(mask):
                ax.plot(
                    obs_ages[mask],
                    obs.values[mask],
                    color=_edu_color(edu),
                    linestyle="--",
                    label=f"{edu_label} (obs.)",
                    linewidth=_S["linewidth"],
                )
        except Exception:
            pass
        ax.plot(
            ages,
            np.maximum(0, vals),
            color=_edu_color(edu),
            label=f"{edu_label} (est.)",
            linewidth=_S["linewidth"],
        )
    ax.set_xlabel("Age", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.set_ylabel(
        "Number of children in household",
        fontsize=_S["label_fontsize"],
        labelpad=_S["labelpad"],
    )
    ax.legend(fontsize=_S["label_fontsize"], frameon=False, loc="best")
    _finalize(
        ax,
        path_to_save,
        ymin=_CHILDREN_YLIM[0],
        ymax=_CHILDREN_YLIM[1],
        xmin=start_age,
        xmax=_CHILDREN_XMAX,
    )


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_children_single_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "children_single_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])
    df = df.loc[df["sex"] == 1].copy()
    df["has_partner"] = (df["partner_state"] > 0).astype(int)
    df = df.loc[df["age"] <= _CHILDREN_XMAX]
    _plot_children_one_panel(specs, df, 0, path_to_save)


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_children_partnered_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "children_partnered_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])
    df = df.loc[df["sex"] == 1].copy()
    df["has_partner"] = (df["partner_state"] > 0).astype(int)
    df = df.loc[df["age"] <= _CHILDREN_XMAX]
    _plot_children_one_panel(specs, df, 1, path_to_save)


# ---------------------------------------------------------------------------
# Job separation
# ---------------------------------------------------------------------------


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_job_separation_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_probs: Path = _EST / "job_sep_probs.pkl",
    path_to_data: Path = BLD / "data" / "soep_job_separation_data.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "job_separation_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    with path_to_probs.open("rb") as f:
        job_sep_probs = pkl.load(f)
    sex_var = 1
    start_age = specs["start_age"]
    x_max = 65
    n_working = min(x_max - start_age + 1, specs["max_ret_age"] - start_age + 1)
    working_ages = np.arange(n_working) + start_age
    probs = job_sep_probs[sex_var, :, :n_working]
    df = pd.read_csv(path_to_data)
    df = df.loc[df["sex"] == sex_var].copy()
    df = df.loc[df["age"] <= x_max]
    obs_shares = df.groupby(["sex", "education", "age"])["job_sep"].mean()
    fig, ax = plt.subplots(figsize=_S["figsize"])
    for edu, edu_label in enumerate(specs["education_labels"]):
        try:
            obs = obs_shares.loc[(sex_var, edu, slice(None))]
            obs_ages = np.asarray(obs.index)
            mask = obs_ages <= x_max
            if np.any(mask):
                ax.plot(
                    obs_ages[mask],
                    obs.values[mask],
                    color=_edu_color(edu),
                    linestyle="--",
                    label=f"{edu_label} (obs.)",
                    linewidth=_S["linewidth"],
                )
        except Exception:
            pass
        ax.plot(
            working_ages[: probs.shape[1]],
            probs[edu, :],
            color=_edu_color(edu),
            label=f"{edu_label} (est.)",
            linewidth=_S["linewidth"],
        )
    ax.set_xlabel("Age", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.set_ylabel(
        "Job separation probability",
        fontsize=_S["label_fontsize"],
        labelpad=_S["labelpad"],
    )
    ax.legend(fontsize=_S["label_fontsize"], frameon=False, loc="best")
    _finalize(ax, path_to_save, ymin=0, ymax=0.1, xmin=start_age, xmax=x_max)


# ---------------------------------------------------------------------------
# Survival probability (women only)
# ---------------------------------------------------------------------------


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_survival_women_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_mortality: Path = _EST / "mortality_transition_matrix_logit.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "survival_probabilities_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    df = pd.read_csv(path_to_mortality)
    df = df.loc[df["sex"] == 1].copy()
    df["survival_prob_year"] = 1 - df["death_prob"]
    df = df.sort_values(["education", "health", "age"])
    df["survival_prob"] = np.nan
    for (edu, health), g in df.groupby(["education", "health"]):
        sp = g["survival_prob_year"].cumprod()
        sp = sp.shift(1)
        sp.iloc[0] = 1.0
        df.loc[g.index, "survival_prob"] = sp.values
    age_min = int(df["age"].min())
    age_max = int(df["age"].max())
    fig, ax = plt.subplots(figsize=_S["figsize"])
    for edu, edu_label in enumerate(specs["education_labels"]):
        for health in [1, 0]:
            sub = df[(df["education"] == edu) & (df["health"] == health)]
            if len(sub) == 0:
                continue
            if health == 1:
                ls = "-"
                h_tag = "good health"
            else:
                ls = (0, (1, 1))
                h_tag = "bad health"
            ax.plot(
                sub["age"],
                sub["survival_prob"],
                color=_edu_color(edu),
                linestyle=ls,
                label=f"{edu_label} ({h_tag})",
                linewidth=_S["linewidth"],
            )
    ax.set_xlabel("Age", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.set_ylabel("Share alive", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.legend(fontsize=_S["label_fontsize"], frameon=False, loc="best")
    _finalize(ax, path_to_save, ymin=0, ymax=1, xmin=age_min, xmax=age_max)


# ---------------------------------------------------------------------------
# Health: P(healthy | alive)
# ---------------------------------------------------------------------------


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_health_probability_healthy_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "health_transition_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = _OUT / "health_prob_healthy_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    df = pd.read_pickle(path_to_data)
    df = df.loc[df["sex"] == 1].copy()
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    est_ages = np.arange(start_age, min(end_age + 1, 90))
    max_period = len(est_ages)
    edu_shares_data = df.groupby(["sex", "education", "age"])["health"].mean()
    health_mat = np.asarray(specs["health_trans_mat"])
    sex_var = 1
    fig, ax = plt.subplots(figsize=_S["figsize"])
    initial_dist = np.zeros(specs["n_health_states"])
    for edu, edu_label in enumerate(specs["education_labels"]):
        try:
            initial_dist[1] = float(edu_shares_data.loc[(1, edu, start_age)])
        except Exception:
            initial_dist[1] = 0.5
        initial_dist[0] = 1 - initial_dist[1]
        initial_dist[2] = 0.0
        trans = health_mat[sex_var, edu, :max_period, :, :]
        shares = _markov_simulator(initial_dist, trans, max_period)
        alive = 1 - shares[:, 2]
        healthy_cond = np.where(alive > 1e-10, shares[:, 1] / alive, np.nan)
        obs = (
            df.loc[
                (df["education"] == edu) & (df["age"] >= start_age) & (df["age"] < 90)
            ]
            .groupby("age")["health"]
            .mean()
        )
        ax.plot(
            obs.index,
            obs.values,
            color=_edu_color(edu),
            linestyle="--",
            label=f"{edu_label} (obs.)",
            linewidth=_S["linewidth"],
        )
        ax.plot(
            est_ages[:max_period],
            healthy_cond,
            color=_edu_color(edu),
            label=f"{edu_label} (est.)",
            linewidth=_S["linewidth"],
        )
    ax.set_xlabel("Age", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.set_ylabel(
        "Share good health", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"]
    )
    ax.legend(fontsize=_S["label_fontsize"], frameon=False, loc="best")
    _finalize(ax, path_to_save, ymin=0, ymax=1, xmin=start_age, xmax=89)


# ---------------------------------------------------------------------------
# Health: P(bad health | alive)
# ---------------------------------------------------------------------------


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_health_probability_bad_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "health_transition_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = _OUT / "health_prob_bad_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    df = pd.read_pickle(path_to_data)
    df = df.loc[df["sex"] == 1].copy()
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    est_ages = np.arange(start_age, min(end_age + 1, 90))
    max_period = len(est_ages)
    health_mat = np.asarray(specs["health_trans_mat"])
    sex_var = 1
    fig, ax = plt.subplots(figsize=_S["figsize"])
    initial_dist = np.zeros(specs["n_health_states"])
    for edu, edu_label in enumerate(specs["education_labels"]):
        try:
            initial_dist[1] = float(
                df.loc[
                    (df["education"] == edu) & (df["age"] == start_age), "health"
                ].mean()
            )
        except Exception:
            initial_dist[1] = 0.5
        initial_dist[0] = 1 - initial_dist[1]
        initial_dist[2] = 0.0
        trans = health_mat[sex_var, edu, :max_period, :, :]
        shares = _markov_simulator(initial_dist, trans, max_period)
        alive = 1 - shares[:, 2]
        bad_cond = np.where(alive > 1e-10, shares[:, 0] / alive, np.nan)
        obs_bad = (
            1
            - df.loc[
                (df["education"] == edu) & (df["age"] >= start_age) & (df["age"] < 90)
            ]
            .groupby("age")["health"]
            .mean()
        )
        ax.plot(
            obs_bad.index,
            obs_bad.values,
            color=_edu_color(edu),
            linestyle="--",
            label=f"{edu_label} (obs.)",
            linewidth=_S["linewidth"],
        )
        ax.plot(
            est_ages[:max_period],
            bad_cond,
            color=_edu_color(edu),
            label=f"{edu_label} (est.)",
            linewidth=_S["linewidth"],
        )
    ax.set_xlabel("Age", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.set_ylabel(
        "Share bad health", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"]
    )
    ax.legend(fontsize=_S["label_fontsize"], frameon=False, loc="best")
    _finalize(ax, path_to_save, ymin=0, ymax=1, xmin=start_age, xmax=89)


# ---------------------------------------------------------------------------
# Own wage — uses PanelOLS in-sample prediction (matches original)
# ---------------------------------------------------------------------------


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_own_wage_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_wage_data.csv",
    path_to_save: Annotated[Path, Product] = _OUT / "own_wage_women.pdf",
) -> None:
    """Own (log) wage by age and education. Women only. PanelOLS in-sample prediction."""
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    start_age = specs["start_age"]
    x_max = 65
    regressors = ["constant", "ln_exp"]

    wage_data = pd.read_csv(path_to_data, index_col=0)
    wage_data = wage_data.loc[wage_data["sex"] == 1].copy()
    wage_data["ln_wage"] = np.log(wage_data["hourly_wage"])
    wage_data["ln_exp"] = np.log(wage_data["experience"] + 1)
    wage_data["constant"] = 1.0
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])

    fig, ax = plt.subplots(figsize=_S["figsize"])
    for edu, edu_label in enumerate(specs["education_labels"]):
        sub = wage_data.loc[wage_data["education"] == edu].copy()
        if len(sub) == 0:
            continue
        model = PanelOLS(
            dependent=sub["ln_wage"],
            exog=sub[regressors + ["year"]],
            entity_effects=True,
        )
        fitted = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        sub["predicted_ln_wage"] = fitted.predict()
        sub = sub.loc[(sub["age"] >= start_age) & (sub["age"] <= x_max)]
        obs = sub.groupby("age")["ln_wage"].mean()
        ax.plot(
            obs.index,
            obs.values,
            color=_edu_color(edu),
            linestyle="--",
            label=f"{edu_label} (obs.)",
            linewidth=_S["linewidth"],
        )
        pred = sub.groupby("age")["predicted_ln_wage"].mean()
        ax.plot(
            pred.index,
            pred.values,
            color=_edu_color(edu),
            label=f"{edu_label} (est.)",
            linewidth=_S["linewidth"],
        )
    ax.set_xlabel("Age", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.set_ylabel(
        "Log hourly wage", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"]
    )
    ax.set_yticks(np.arange(1.5, 4.0, 0.5))
    ax.legend(fontsize=_S["label_fontsize"], frameon=False, loc="best")
    _finalize(ax, path_to_save, ymin=1.5, ymax=3.5, xmin=start_age, xmax=x_max)


# ---------------------------------------------------------------------------
# Formal care costs (pooled)
# ---------------------------------------------------------------------------


@pytask.mark.figures
@pytask.mark.stochastic_processes
@pytask.mark.publication_stochastic_processes
def task_plot_formal_care_costs_publication(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_params: Path = _EST / "formal_care_costs_params_pooled.csv",
    path_to_data: Path = BLD / "data" / "formal_care_costs_sample.pkl",
    path_to_save: Annotated[Path, Product] = _OUT / "formal_care_costs_women.pdf",
) -> None:
    with path_to_specs.open("rb") as f:
        specs = pkl.load(f)
    params_df = pd.read_csv(path_to_params, index_col=0)
    df_raw = pd.read_pickle(path_to_data)
    if isinstance(df_raw.index, pd.MultiIndex):
        df_raw = df_raw.reset_index()
    if "sex" in df_raw.columns:
        df_raw = df_raw.loc[df_raw["sex"] == 1].copy()
    df_raw = df_raw.loc[df_raw["age"] <= MAX_AGE_SIM].copy()

    obs_by_age = df_raw.groupby("age")["formal_care_costs"].mean()

    const = params_df.loc["coefficient", "const"]
    age_coef = params_df.loc["coefficient", "age"]
    age_sq_coef = params_df.loc["coefficient", "age_sq"]

    start_age = specs["start_age"]
    end_age = MAX_AGE_SIM
    ages = np.arange(start_age, end_age + 1)
    pred = const + age_coef * ages + age_sq_coef * ages**2

    fig, ax = plt.subplots(figsize=_S["figsize"])
    if len(obs_by_age) > 0:
        ax.scatter(
            obs_by_age.index,
            obs_by_age.values,
            color=EDU_COLOR_LOW,
            label="Raw data (obs.)",
            s=20,
            zorder=3,
        )
    ax.plot(
        ages,
        pred,
        color=EDU_COLOR_HIGH,
        label="OLS fit (est.)",
        linewidth=_S["linewidth"],
    )
    ax.set_xlabel("Age", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"])
    ax.set_ylabel(
        "Formal care costs (in euros)", fontsize=_S["label_fontsize"], labelpad=_S["labelpad"]
    )
    ax.legend(fontsize=_S["label_fontsize"], frameon=False, loc="best")
    _finalize(ax, path_to_save, xmin=start_age, xmax=end_age)
