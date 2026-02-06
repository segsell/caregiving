"""Publication: income tax by age for caregiving leave policy scenarios."""

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
from caregiving.tables.publication.task_government_budget import convert_to_currency

MAX_AGE_PLOT = 90


def _simple_income_tax_by_age(
    df: pd.DataFrame,
    wealth_unit: float,
    start_age: int,
    name: str,
    only_caregivers: bool = True,
) -> pd.Series:
    """Compute average income tax by age; non-workers get 0 tax but remain in sample.

    income_tax_single is multiplied by a working dummy (1 if choice in WORK, 0 else)
    so that potential income tax for non-workers is zeroed; everyone stays in sample.
    If only_caregivers is True (default), restrict to current caregivers (choice in
    INFORMAL_CARE). Otherwise use all rows.
    """
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if only_caregivers and "choice" in df.columns:
        care_choices = np.asarray(INFORMAL_CARE)
        df = df[df["choice"].isin(care_choices)].copy()
    df = convert_to_currency(df, wealth_unit)
    if "age" not in df.columns:
        df["age"] = start_age + df["period"]
    work_choices = np.asarray(WORK)
    is_working = df["choice"].isin(work_choices).astype(np.float64)
    df["income_tax_working"] = df["income_tax_single"] * is_working
    return (
        df.groupby("age", observed=False)["income_tax_working"]
        .mean()
        .rename(name)
        .sort_index()
    )


@pytask.mark.tables
@pytask.mark.publication
def task_income_tax_by_age_caregiving_leave_policy_scenarios(
    path_to_full_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_partial_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_plot_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "government_budget_income_tax_by_age_caregiving_leave_levels.pdf",
    path_to_save_plot_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "government_budget_income_tax_by_age_caregiving_leave_difference.pdf",
    monthly: bool = False,
    only_caregivers: bool = False,
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Compare absolute average income tax by age: full vs partial caregiving leave.

    income_tax_single is multiplied by working dummy (non-workers contribute 0).
    If only_caregivers is True (default), restrict to current caregivers (choice in
    INFORMAL_CARE). Compares:
    - Full caregiving leave with job retention
    - (Partial) caregiving leave with job retention

    Produces two publication plots:
    1. Average income tax by age for both policy scenarios (two lines).
    2. Difference in average income tax by age (full minus partial).
    """
    full_df = pd.read_pickle(path_to_full_policy_sim)
    partial_df = pd.read_pickle(path_to_partial_policy_sim)
    full_df, partial_df = prepare_dfs_for_comparison(
        full_df,
        partial_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = specs["wealth_unit"]
    start_age = int(specs.get("start_age", 30))

    full_by_age = _simple_income_tax_by_age(
        full_df, wealth_unit, start_age, "full_leave", only_caregivers=only_caregivers
    )
    partial_by_age = _simple_income_tax_by_age(
        partial_df,
        wealth_unit,
        start_age,
        "partial_leave",
        only_caregivers=only_caregivers,
    )

    by_age = pd.concat([full_by_age, partial_by_age], axis=1)
    by_age = by_age.dropna(how="all").sort_index()
    by_age = by_age.loc[by_age.index <= MAX_AGE_PLOT]
    if monthly:
        by_age = by_age / 12

    path_to_save_plot_levels.parent.mkdir(parents=True, exist_ok=True)

    unit_label = "currency, monthly" if monthly else "currency"

    # Plot 1: Absolute average income tax by age (both scenarios)
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
        by_age["partial_leave"],
        label="Caregiving leave with job retention",
        color="C1",
        linewidth=2,
    )
    ax1.set_xlabel("Age")
    ax1.set_ylabel(f"Average income tax ({unit_label})")
    ax1.set_title("Average income tax by age (workers in num., all in denom.)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(path_to_save_plot_levels)
    plt.close(fig1)

    # Plot 2: Difference (full minus partial) by age
    diff = by_age["full_leave"] - by_age["partial_leave"]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(diff.index, diff.values, color="C2", linewidth=2)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Age")
    ax2.set_ylabel(f"Difference in average income tax ({unit_label})")
    ax2.set_title("Income tax difference by age (full minus partial)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(path_to_save_plot_difference)
    plt.close(fig2)
