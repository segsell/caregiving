"""Publication: caregiving leave top-up by age (full vs normal leave scenarios)."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_utils import prepare_dfs_for_comparison
from caregiving.model.shared import INFORMAL_CARE
from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_inc_tax_for_single_income,
)

MAX_AGE_PLOT = 90


def compute_net_caregiving_leave_top_up_cost(
    caregiving_leave_top_up,
    own_income_after_ssc,
    income_tax_single,
    model_specs,
):
    """Compute net caregiving leave top-up cost for the planner (in currency).

    The planner pays the gross top-up but receives income tax on it (top-up is
    part of the tax base). Net cost = gross top-up - extra income tax raised
    because of the top-up.

    Uses the same ingredients as returned by the caregiving-leave budget
    equations (full and normal): caregiving_leave_top_up, own_income_after_ssc,
    income_tax_single. Both budget_equation_full_caregiving_leave_with_job_
    retention and budget_equation_caregiving_leave_with_job_retention return
    the same aux keys (verified).

    All income/tax inputs are in model units (wealth_unit). model_specs must
    contain "wealth_unit" and the tax schedule (income_tax_brackets, etc.).
    Returns net cost in currency (float or array, same shape as inputs).

    For couples, the actual household tax is split; this uses the tax on own
    income (income_tax_single) so the implied tax clawback is exact for singles
    and an approximation for couples.
    """
    wealth_unit = float(model_specs["wealth_unit"])
    top_up_currency = np.asarray(caregiving_leave_top_up, dtype=float) * wealth_unit
    own_currency = np.asarray(own_income_after_ssc, dtype=float) * wealth_unit
    tax_single_with_currency = np.asarray(income_tax_single, dtype=float) * wealth_unit

    own_no_top_up_currency = own_currency - top_up_currency
    tax_single_without = np.asarray(
        calc_inc_tax_for_single_income(own_no_top_up_currency, model_specs)
    )
    tax_on_top_up = tax_single_with_currency - tax_single_without
    net_cost = top_up_currency - tax_on_top_up
    return net_cost


def _average_caregiving_leave_top_up_by_age(
    df: pd.DataFrame,
    wealth_unit: float,
    start_age: int,
    name: str,
    only_caregivers: bool = True,
) -> pd.Series:
    """Compute average caregiving_leave_top_up (currency) by age.

    If only_caregivers is True (default), restrict to current caregivers
    (choice in INFORMAL_CARE). Otherwise use all rows.
    """
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "caregiving_leave_top_up" not in df.columns:
        return pd.Series(dtype=float)
    if only_caregivers and "choice" in df.columns:
        care_choices = np.asarray(INFORMAL_CARE)
        df = df[df["choice"].isin(care_choices)].copy()
    if "age" not in df.columns:
        df["age"] = start_age + df["period"]
    df["caregiving_leave_top_up_currency"] = df["caregiving_leave_top_up"] * wealth_unit
    return (
        df.groupby("age", observed=False)["caregiving_leave_top_up_currency"]
        .mean()
        .rename(name)
        .sort_index()
    )


def _average_net_caregiving_leave_top_up_by_age(
    df: pd.DataFrame,
    model_specs: dict,
    start_age: int,
    name: str,
    only_caregivers: bool = True,
) -> pd.Series:
    """Compute average net caregiving leave top-up cost (planner, currency) by age.

    If only_caregivers is True (default), restrict to current caregivers
    (choice in INFORMAL_CARE). Requires columns caregiving_leave_top_up,
    own_income_after_ssc, income_tax_single (model units).
    """
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    required = ["caregiving_leave_top_up", "own_income_after_ssc", "income_tax_single"]
    if not all(c in df.columns for c in required):
        return pd.Series(dtype=float)
    if only_caregivers and "choice" in df.columns:
        care_choices = np.asarray(INFORMAL_CARE)
        df = df[df["choice"].isin(care_choices)].copy()
    if df.empty:
        return pd.Series(dtype=float)
    if "age" not in df.columns:
        df["age"] = start_age + df["period"]
    df["net_top_up_cost"] = compute_net_caregiving_leave_top_up_cost(
        caregiving_leave_top_up=df["caregiving_leave_top_up"].values,
        own_income_after_ssc=df["own_income_after_ssc"].values,
        income_tax_single=df["income_tax_single"].values,
        model_specs=model_specs,
    )
    return (
        df.groupby("age", observed=False)["net_top_up_cost"]
        .mean()
        .rename(name)
        .sort_index()
    )


@pytask.mark.tables
@pytask.mark.top_up
@pytask.mark.publication
def task_caregiving_leave_top_up_by_age_full_vs_normal(
    path_to_full_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_normal_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_plot_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_top_up_by_age_levels.pdf",
    path_to_save_plot_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_top_up_by_age_difference.pdf",
    only_caregivers: bool = True,
    monthly: bool = False,
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Compare average caregiving_leave_top_up paid out by age: full vs normal leave.

    If only_caregivers is True (default), restrict to current caregivers
    (choice in INFORMAL_CARE) in each scenario. Uses the same two caregiving
    leave scenarios (full vs normal/partial). Produces:
    1. Average caregiving_leave_top_up by age for both scenarios (two lines).
    2. Difference by age (full minus normal).
    """
    full_df = pd.read_pickle(path_to_full_policy_sim)
    normal_df = pd.read_pickle(path_to_normal_policy_sim)
    full_df, normal_df = prepare_dfs_for_comparison(
        full_df,
        normal_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = specs["wealth_unit"]
    start_age = int(specs.get("start_age", 30))

    full_by_age = _average_caregiving_leave_top_up_by_age(
        full_df, wealth_unit, start_age, "full_leave", only_caregivers
    )
    normal_by_age = _average_caregiving_leave_top_up_by_age(
        normal_df, wealth_unit, start_age, "normal_leave", only_caregivers
    )

    by_age = pd.concat([full_by_age, normal_by_age], axis=1)
    by_age = by_age.dropna(how="all").sort_index()
    by_age = by_age.fillna(0)
    by_age = by_age.loc[by_age.index <= MAX_AGE_PLOT]
    if monthly:
        by_age = by_age / 12

    path_to_save_plot_levels.parent.mkdir(parents=True, exist_ok=True)

    ylabel_levels = (
        "Average caregiving leave top-up (currency, monthly)"
        if monthly
        else "Average caregiving leave top-up (currency)"
    )
    ylabel_diff = (
        "Difference in average top-up (currency, monthly)"
        if monthly
        else "Difference in average top-up (currency)"
    )

    # Plot 1: Average caregiving_leave_top_up by age (both scenarios)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        by_age.index,
        by_age["full_leave"],
        label="Full caregiving leave with job retention",
        color="C0",
        linewidth=2,
    )
    ax1.plot(
        by_age.index,
        by_age["normal_leave"],
        label="Caregiving leave with job retention",
        color="C1",
        linewidth=2,
    )
    ax1.set_xlabel("Age")
    ax1.set_ylabel(ylabel_levels)
    ax1.set_title("Average caregiving leave top-up paid out by age")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(path_to_save_plot_levels)
    plt.close(fig1)

    # Plot 2: Difference (full minus normal) by age
    diff = by_age["full_leave"] - by_age["normal_leave"]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(diff.index, diff.values, color="C2", linewidth=2)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Age")
    ax2.set_ylabel(ylabel_diff)
    ax2.set_title("Caregiving leave top-up difference by age (full minus normal)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(path_to_save_plot_difference)
    plt.close(fig2)


@pytask.mark.tables
@pytask.mark.top_up
@pytask.mark.publication
def task_net_caregiving_leave_top_up_by_age_full_vs_normal(
    path_to_full_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_normal_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_plot_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_net_top_up_by_age_levels.pdf",
    path_to_save_plot_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_net_top_up_by_age_difference.pdf",
    only_caregivers: bool = True,
    monthly: bool = False,
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Compare average net caregiving leave top-up cost (planner) by age.

    Full vs normal leave. If only_caregivers is True (default), restrict to
    caregivers (choice in INFORMAL_CARE). Net cost
    = gross top-up minus
    income tax raised on the top-up. Produces:
    1. Average net top-up cost by age for both scenarios (two lines).
    2. Difference by age (full minus normal).
    """
    full_df = pd.read_pickle(path_to_full_policy_sim)
    normal_df = pd.read_pickle(path_to_normal_policy_sim)
    full_df, normal_df = prepare_dfs_for_comparison(
        full_df,
        normal_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    specs = pickle.load(path_to_specs.open("rb"))
    start_age = int(specs.get("start_age", 30))

    full_by_age = _average_net_caregiving_leave_top_up_by_age(
        full_df, specs, start_age, "full_leave", only_caregivers
    )
    normal_by_age = _average_net_caregiving_leave_top_up_by_age(
        normal_df, specs, start_age, "normal_leave", only_caregivers
    )

    by_age = pd.concat([full_by_age, normal_by_age], axis=1)
    by_age = by_age.dropna(how="all").sort_index()
    by_age = by_age.fillna(0)
    by_age = by_age.loc[by_age.index <= MAX_AGE_PLOT]
    if monthly:
        by_age = by_age / 12

    path_to_save_plot_levels.parent.mkdir(parents=True, exist_ok=True)

    ylabel_levels_net = (
        "Average net top-up cost (currency, monthly)"
        if monthly
        else "Average net caregiving leave top-up cost (currency)"
    )
    ylabel_diff_net = (
        "Difference in average net top-up cost (currency, monthly)"
        if monthly
        else "Difference in average net top-up cost (currency)"
    )

    # Plot 1: Average net top-up cost by age (both scenarios)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        by_age.index,
        by_age["full_leave"],
        label="Full caregiving leave with job retention",
        color="C0",
        linewidth=2,
    )
    ax1.plot(
        by_age.index,
        by_age["normal_leave"],
        label="Caregiving leave with job retention",
        color="C1",
        linewidth=2,
    )
    ax1.set_xlabel("Age")
    ax1.set_ylabel(ylabel_levels_net)
    ax1.set_title("Net caregiving leave top-up cost by age (caregivers, planner)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(path_to_save_plot_levels)
    plt.close(fig1)

    # Plot 2: Difference (full minus normal) by age
    diff = by_age["full_leave"] - by_age["normal_leave"]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(diff.index, diff.values, color="C2", linewidth=2)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Age")
    ax2.set_ylabel(ylabel_diff_net)
    ax2.set_title(
        "Net caregiving leave top-up cost difference by age (full minus normal)"
    )
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(path_to_save_plot_difference)
    plt.close(fig2)
