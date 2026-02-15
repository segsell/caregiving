"""Publication: income tax, working share, net top-up (normal/full leave vs baseline).

Note: In normal and full caregiving leave scenarios, the simulated data does not
contain care_benefits_and_costs (informal cash benefits); only caregiving_leave_top_up
is in the budget/aux. For leave scenarios we use net caregiving leave top-up as
government cost.
"""

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
from caregiving.model.shared import INFORMAL_CARE, WORK
from caregiving.tables.publication.task_government_budget_caregiving_leave import (
    _simple_income_tax_by_age,
)
from caregiving.tables.publication.task_government_budget_caregiving_leave_labor_supply import (  # noqa: E501
    _share_by_age,
)
from caregiving.tables.publication.task_government_budget_caregiving_leave_top_up import (  # noqa: E501
    _average_net_caregiving_leave_top_up_by_age,
)

MAX_AGE_PLOT = 90


def _average_baseline_care_benefits_by_age(
    df: pd.DataFrame,
    wealth_unit: float,
    start_age: int,
    name: str,
    only_caregivers: bool = True,
) -> pd.Series:
    """Average govt expenditure on informal care (care_benefits_and_costs > 0) by age.

    Baseline has no caregiving leave; positive care_benefits_and_costs = benefits paid
    for informal care. If only_caregivers, restrict to choice in INFORMAL_CARE.
    """
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "care_benefits_and_costs" not in df.columns:
        return pd.Series(dtype=float)
    if only_caregivers and "choice" in df.columns:
        care_choices = np.asarray(INFORMAL_CARE)
        df = df[df["choice"].isin(care_choices)].copy()
    if "age" not in df.columns:
        df["age"] = start_age + df["period"]
    care_benefits = np.maximum(df["care_benefits_and_costs"].values, 0.0) * wealth_unit
    df = df.assign(care_benefits_currency=care_benefits)
    return (
        df.groupby("age", observed=False)["care_benefits_currency"]
        .mean()
        .rename(name)
        .sort_index()
    )


def _average_net_government_budget_by_age(
    df: pd.DataFrame,
    wealth_unit: float,
    start_age: int,
    name: str,
) -> pd.Series:
    """Average net_government_budget (currency) by age. Uses all rows."""
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "net_government_budget" not in df.columns:
        return pd.Series(dtype=float)
    if "age" not in df.columns:
        df["age"] = start_age + df["period"]
    df["net_gov_budget_currency"] = df["net_government_budget"] * wealth_unit
    return (
        df.groupby("age", observed=False)["net_gov_budget_currency"]
        .mean()
        .rename(name)
        .sort_index()
    )


@pytask.mark.tables
@pytask.mark.publication
def task_income_tax_by_age_normal_leave_vs_baseline(
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_plot_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "government_budget_income_tax_by_age_normal_leave_vs_baseline_levels.pdf",
    path_to_save_plot_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "government_budget_income_tax_by_age_normal_leave_vs_baseline_difference.pdf",
    only_caregivers: bool = False,
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Compare average income tax by age: normal caregiving leave vs baseline.

    income_tax_single x working dummy; only_caregivers=True (current caregivers only).
    Yearly values. Produces levels (two lines) and difference (normal minus baseline).
    """
    normal_df = pd.read_pickle(path_to_normal_leave_sim)
    baseline_df = pd.read_pickle(path_to_baseline_sim)
    normal_df, baseline_df = prepare_dfs_for_comparison(
        normal_df,
        baseline_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = specs["wealth_unit"]
    start_age = int(specs.get("start_age", 30))

    normal_by_age = _simple_income_tax_by_age(
        normal_df,
        wealth_unit,
        start_age,
        "normal_leave",
        only_caregivers=only_caregivers,
    )
    baseline_by_age = _simple_income_tax_by_age(
        baseline_df, wealth_unit, start_age, "baseline", only_caregivers=only_caregivers
    )

    by_age = pd.concat([normal_by_age, baseline_by_age], axis=1)
    by_age = by_age.dropna(how="all").sort_index()
    by_age = by_age.loc[by_age.index <= MAX_AGE_PLOT]

    path_to_save_plot_levels.parent.mkdir(parents=True, exist_ok=True)

    # Plot 1: Levels
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        by_age.index,
        by_age["normal_leave"],
        label="Normal caregiving leave with job retention",
        color="C0",
        linewidth=2,
    )
    ax1.plot(
        by_age.index,
        by_age["baseline"],
        label="Baseline",
        color="C1",
        linewidth=2,
    )
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Average income tax (currency)")
    ax1.set_title("Average income tax by age (workers in num., all in denom.)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(path_to_save_plot_levels)
    plt.close(fig1)

    # Plot 2: Difference (normal minus baseline)
    diff = by_age["normal_leave"] - by_age["baseline"]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(diff.index, diff.values, color="C2", linewidth=2)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Difference in average income tax (currency)")
    ax2.set_title("Income tax difference by age (normal leave minus baseline)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(path_to_save_plot_difference)
    plt.close(fig2)


@pytask.mark.tables
@pytask.mark.publication
def task_working_share_by_age_full_normal_leave_vs_baseline(  # noqa: PLR0915
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_working_share_by_age_full_normal_vs_baseline_levels.pdf",
    path_to_save_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_working_share_by_age_full_normal_vs_baseline_difference.pdf",
    only_caregivers: bool = False,
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Compare share working (PT + FT) by age: full leave, normal leave, baseline.

    If only_caregivers is True (default False), restrict to current caregivers.
    Produces levels (three lines) and difference plot (full minus baseline,
    normal minus baseline).
    """
    full_df = pd.read_pickle(path_to_full_leave_sim)
    normal_df = pd.read_pickle(path_to_normal_leave_sim)
    baseline_df = pd.read_pickle(path_to_baseline_sim)
    if isinstance(full_df.index, pd.MultiIndex):
        full_df = full_df.reset_index()
    if isinstance(normal_df.index, pd.MultiIndex):
        normal_df = normal_df.reset_index()
    if isinstance(baseline_df.index, pd.MultiIndex):
        baseline_df = baseline_df.reset_index()
    full_df = full_df[full_df["age"] <= MAX_AGE_PLOT].copy()
    normal_df = normal_df[normal_df["age"] <= MAX_AGE_PLOT].copy()
    baseline_df = baseline_df[baseline_df["age"] <= MAX_AGE_PLOT].copy()
    full_df, normal_df, baseline_df = prepare_dfs_for_comparison(
        full_df,
        normal_df,
        baseline_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    if only_caregivers and "choice" in normal_df.columns:
        care_choices = np.asarray(INFORMAL_CARE)
        full_df = full_df[full_df["choice"].isin(care_choices)].copy()
        normal_df = normal_df[normal_df["choice"].isin(care_choices)].copy()
        baseline_df = baseline_df[baseline_df["choice"].isin(care_choices)].copy()

    specs = pickle.load(path_to_specs.open("rb"))
    start_age = int(specs.get("start_age", 30))

    full_working = _share_by_age(full_df, WORK, start_age, "full_leave")
    normal_working = _share_by_age(normal_df, WORK, start_age, "normal_leave")
    baseline_working = _share_by_age(baseline_df, WORK, start_age, "baseline")

    age_index = np.arange(start_age, MAX_AGE_PLOT + 1, dtype=int)
    levels = pd.concat([full_working, normal_working, baseline_working], axis=1)
    levels = levels.reindex(age_index).fillna(0)
    levels.columns = ["full_leave", "normal_leave", "baseline"]

    path_to_save_levels.parent.mkdir(parents=True, exist_ok=True)

    # --- Levels: share working (PT + FT), three lines ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        levels.index,
        levels["full_leave"],
        label="Full caregiving leave",
        color="C0",
        lw=2,
    )
    ax.plot(
        levels.index,
        levels["normal_leave"],
        label="Normal caregiving leave",
        color="C1",
        lw=2,
    )
    ax.plot(
        levels.index,
        levels["baseline"],
        label="Baseline",
        color="C2",
        lw=2,
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Working share (PT + FT)")
    ax.set_title("Working share by age (full / normal leave vs baseline)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path_to_save_levels)
    plt.close(fig)

    # --- Difference: full minus baseline, normal minus baseline (two lines) ---
    diff_full = levels["full_leave"] - levels["baseline"]
    diff_normal = levels["normal_leave"] - levels["baseline"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        diff_full.index,
        diff_full.values,
        label="Full leave minus baseline",
        color="C0",
        linewidth=2,
    )
    ax.plot(
        diff_normal.index,
        diff_normal.values,
        label="Normal leave minus baseline",
        color="C1",
        linewidth=2,
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Age")
    ax.set_ylabel("Difference in working share")
    ax.set_title("Working share difference by age (leave minus baseline)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_to_save_difference)
    plt.close(fig)


@pytask.mark.tables
@pytask.mark.publication
def task_net_caregiving_top_up_by_age_normal_full_leave_vs_baseline(  # noqa: PLR0915
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_levels_baseline_care_benefits: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "net_caregiving_top_up_by_age_leave_vs_baseline_levels_baseline_care_benefits.pdf",  # noqa: E501
    path_to_save_levels_baseline_zero: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "net_caregiving_top_up_by_age_leave_vs_baseline_levels_baseline_zero.pdf",
    path_to_save_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "net_caregiving_top_up_by_age_leave_vs_baseline_difference.pdf",
    only_caregivers: bool = True,
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Plot net caregiving top-up (govt cost) by age: full/normal leave vs baseline.

    In normal and full leave scenarios, the outcome is the average net caregiving leave
    top-up cost (planner; gross top-up minus income tax on top-up). Informal cash
    benefits (care_benefits_and_costs) are not in the leave-scenario simulated data;
    only the leave top-up is reported.

    Baseline has no caregiving leave (top-up is zero). Two level plots:
    1. Baseline = care_benefits_and_costs when positive (government expenditure on
       informal care in baseline; positive values mean benefits paid for informal care).
    2. Baseline = 0 (no informal cash benefits in this picture).

    Also produces a difference plot: (full minus baseline) and (normal minus baseline)
    with cash benefits. If only_caregivers is True (default), restrict to INFORMAL_CARE.
    """
    full_df = pd.read_pickle(path_to_full_leave_sim)
    normal_df = pd.read_pickle(path_to_normal_leave_sim)
    baseline_df = pd.read_pickle(path_to_baseline_sim)
    full_df = full_df[full_df["age"] <= MAX_AGE_PLOT].copy()
    normal_df = normal_df[normal_df["age"] <= MAX_AGE_PLOT].copy()
    baseline_df = baseline_df[baseline_df["age"] <= MAX_AGE_PLOT].copy()
    full_df, normal_df, baseline_df = prepare_dfs_for_comparison(
        full_df,
        normal_df,
        baseline_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = specs["wealth_unit"]
    start_age = int(specs.get("start_age", 30))

    full_by_age = _average_net_caregiving_leave_top_up_by_age(
        full_df, specs, start_age, "full_leave", only_caregivers=only_caregivers
    )
    normal_by_age = _average_net_caregiving_leave_top_up_by_age(
        normal_df, specs, start_age, "normal_leave", only_caregivers=only_caregivers
    )
    baseline_care_by_age = _average_baseline_care_benefits_by_age(
        baseline_df, wealth_unit, start_age, "baseline", only_caregivers=only_caregivers
    )

    age_index = np.arange(start_age, MAX_AGE_PLOT + 1, dtype=int)

    def _to_levels(full_s: pd.Series, normal_s: pd.Series, baseline_s: pd.Series):
        out = pd.concat([full_s, normal_s, baseline_s], axis=1)
        out = out.reindex(age_index).fillna(0)
        out.columns = ["full_leave", "normal_leave", "baseline"]
        return out

    levels_with_baseline_care = _to_levels(
        full_by_age, normal_by_age, baseline_care_by_age
    )
    levels_baseline_zero = levels_with_baseline_care.copy()
    levels_baseline_zero["baseline"] = 0.0

    path_to_save_levels_baseline_care_benefits.parent.mkdir(parents=True, exist_ok=True)

    ylabel = "Average government expenditure (currency)"

    # Plot 1: Full, Normal, Baseline = care_benefits (positive only)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        levels_with_baseline_care.index,
        levels_with_baseline_care["full_leave"],
        label="Full caregiving leave",
        color="C0",
        linewidth=2,
    )
    ax1.plot(
        levels_with_baseline_care.index,
        levels_with_baseline_care["normal_leave"],
        label="Normal caregiving leave",
        color="C1",
        linewidth=2,
    )
    ax1.plot(
        levels_with_baseline_care.index,
        levels_with_baseline_care["baseline"],
        label="Baseline (care benefits when positive)",
        color="C2",
        linewidth=2,
    )
    ax1.set_xlabel("Age")
    ax1.set_ylabel(ylabel)
    ax1.set_title(
        "Net caregiving leave top-up / care benefits by age (leave vs baseline)"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(path_to_save_levels_baseline_care_benefits)
    plt.close(fig1)

    # Plot 2: Full, Normal, Baseline = 0
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        levels_baseline_zero.index,
        levels_baseline_zero["full_leave"],
        label="Full caregiving leave",
        color="C0",
        linewidth=2,
    )
    ax2.plot(
        levels_baseline_zero.index,
        levels_baseline_zero["normal_leave"],
        label="Normal caregiving leave",
        color="C1",
        linewidth=2,
    )
    ax2.axhline(
        0, color="gray", linestyle="--", alpha=0.5, label="Baseline (no cash benefits)"
    )
    ax2.set_xlabel("Age")
    ax2.set_ylabel(ylabel)
    ax2.set_title(
        "Net caregiving leave top-up by age (baseline = 0, no informal cash benefits)"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(path_to_save_levels_baseline_zero)
    plt.close(fig2)

    # Plot 3: Differences vs baseline (with cash benefits) — two lines
    diff_full = (
        levels_with_baseline_care["full_leave"] - levels_with_baseline_care["baseline"]
    )
    diff_normal = (
        levels_with_baseline_care["normal_leave"]
        - levels_with_baseline_care["baseline"]
    )
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(
        diff_full.index,
        diff_full.values,
        label="Full leave minus baseline (care benefits)",
        color="C0",
        linewidth=2,
    )
    ax3.plot(
        diff_normal.index,
        diff_normal.values,
        label="Normal leave minus baseline (care benefits)",
        color="C1",
        linewidth=2,
    )
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Age")
    ax3.set_ylabel("Difference in average government expenditure (currency)")
    ax3.set_title(
        "Net top-up / care benefits: leave minus baseline (baseline = care benefits)"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(path_to_save_difference)
    plt.close(fig3)


@pytask.mark.tables
@pytask.mark.publication
def task_net_government_budget_by_age_full_normal_leave_vs_baseline(
    path_to_full_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_normal_leave_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "net_government_budget_by_age_leave_vs_baseline_levels.pdf",
    path_to_save_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "net_government_budget_by_age_leave_vs_baseline_difference.pdf",
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Plot mean net_government_budget by age: full leave, normal leave, baseline.

    Outcome is net_government_budget (revenue minus expenditure, in currency).
    Produces: (1) levels = three lines by age; (2) differences = full minus baseline,
    normal minus baseline (two lines).
    """
    full_df = pd.read_pickle(path_to_full_leave_sim)
    normal_df = pd.read_pickle(path_to_normal_leave_sim)
    baseline_df = pd.read_pickle(path_to_baseline_sim)
    full_df = full_df[full_df["age"] <= MAX_AGE_PLOT].copy()
    normal_df = normal_df[normal_df["age"] <= MAX_AGE_PLOT].copy()
    baseline_df = baseline_df[baseline_df["age"] <= MAX_AGE_PLOT].copy()
    full_df, normal_df, baseline_df = prepare_dfs_for_comparison(
        full_df,
        normal_df,
        baseline_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = specs["wealth_unit"]
    start_age = int(specs.get("start_age", 30))

    full_by_age = _average_net_government_budget_by_age(
        full_df, wealth_unit, start_age, "full_leave"
    )
    normal_by_age = _average_net_government_budget_by_age(
        normal_df, wealth_unit, start_age, "normal_leave"
    )
    baseline_by_age = _average_net_government_budget_by_age(
        baseline_df, wealth_unit, start_age, "baseline"
    )

    age_index = np.arange(start_age, MAX_AGE_PLOT + 1, dtype=int)
    levels = pd.concat([full_by_age, normal_by_age, baseline_by_age], axis=1)
    levels = levels.reindex(age_index).fillna(0)
    levels.columns = ["full_leave", "normal_leave", "baseline"]

    path_to_save_levels.parent.mkdir(parents=True, exist_ok=True)

    # Plot 1: Mean net government budget by age (three lines)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        levels.index,
        levels["full_leave"],
        label="Full caregiving leave",
        color="C0",
        linewidth=2,
    )
    ax1.plot(
        levels.index,
        levels["normal_leave"],
        label="Normal caregiving leave",
        color="C1",
        linewidth=2,
    )
    ax1.plot(
        levels.index,
        levels["baseline"],
        label="Baseline",
        color="C2",
        linewidth=2,
    )
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Mean net government budget (currency)")
    ax1.set_title("Net government budget by age (full / normal leave vs baseline)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(path_to_save_levels)
    plt.close(fig1)

    # Plot 2: Differences vs baseline (two lines)
    diff_full = levels["full_leave"] - levels["baseline"]
    diff_normal = levels["normal_leave"] - levels["baseline"]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        diff_full.index,
        diff_full.values,
        label="Full leave minus baseline",
        color="C0",
        linewidth=2,
    )
    ax2.plot(
        diff_normal.index,
        diff_normal.values,
        label="Normal leave minus baseline",
        color="C1",
        linewidth=2,
    )
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Difference in mean net government budget (currency)")
    ax2.set_title("Net government budget: leave minus baseline")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(path_to_save_difference)
    plt.close(fig2)
