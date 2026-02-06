"""Publication task: government budget comparison (income tax by age)."""

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
from caregiving.model.shared import WORK


# @pytask.mark.tables
# @pytask.mark.publication
def task_compute_government_budget_baseline_vs_no_care_demand(
    path_to_baseline_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_demand_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_plot_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "government_budget_income_tax_by_age_levels.pdf",
    path_to_save_plot_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "government_budget_income_tax_by_age_difference.pdf",
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Compare average income tax (workers) by age: baseline vs no care demand.

    Produces two publication plots:
    1. Average income tax by age for both scenarios (two lines).
    2. Difference in average income tax by age (no care demand minus baseline).
    """
    # Load simulated data and specs
    baseline_df = pd.read_pickle(path_to_baseline_sim)
    no_care_demand_df = pd.read_pickle(path_to_no_care_demand_sim)
    baseline_df, no_care_demand_df = prepare_dfs_for_comparison(
        baseline_df,
        no_care_demand_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    specs = pickle.load(path_to_specs.open("rb"))
    wealth_unit = specs["wealth_unit"]
    start_age = int(specs.get("start_age", 30))

    baseline_df = convert_to_currency(baseline_df, wealth_unit)
    no_care_demand_df = convert_to_currency(no_care_demand_df, wealth_unit)

    # Ensure age column exists
    if "age" not in baseline_df.columns:
        baseline_df["age"] = start_age + baseline_df["period"]
    if "age" not in no_care_demand_df.columns:
        no_care_demand_df["age"] = start_age + no_care_demand_df["period"]

    # Agent-period comparison: merge on (agent, period), keep only rows in both
    work_choices = np.asarray(WORK)
    baseline_worker = baseline_df.copy()
    baseline_worker["is_working"] = np.isin(baseline_worker["choice"], work_choices)
    no_care_worker = no_care_demand_df.copy()
    no_care_worker["is_working"] = np.isin(no_care_worker["choice"], work_choices)

    merged = baseline_worker[
        ["agent", "period", "age", "income_tax_single", "is_working"]
    ].merge(
        no_care_worker[["agent", "period", "income_tax_single", "is_working"]],
        on=["agent", "period"],
        how="inner",
        suffixes=("_baseline", "_no_care_demand"),
    )

    # Per agent-period difference (no care demand minus baseline)
    merged["income_tax_diff"] = (
        merged["income_tax_single_no_care_demand"]
        - merged["income_tax_single_baseline"]
    )
    # Restrict to agent-periods where agent is working in BOTH scenarios
    merged_workers = merged[
        merged["is_working_baseline"] & merged["is_working_no_care_demand"]
    ].copy()

    # Average difference by age (across agent-period comparisons)
    avg_diff_by_age = (
        merged_workers.groupby("age")["income_tax_diff"].mean().sort_index()
    )

    # Levels for plot 1: same merged worker set, average tax by age per scenario
    baseline_by_age = (
        merged_workers.groupby("age")["income_tax_single_baseline"]
        .mean()
        .rename("baseline")
    )
    no_care_demand_by_age = (
        merged_workers.groupby("age")["income_tax_single_no_care_demand"]
        .mean()
        .rename("no_care_demand")
    )
    by_age = pd.concat([baseline_by_age, no_care_demand_by_age], axis=1)
    by_age = by_age.dropna(how="all").sort_index()

    path_to_save_plot_levels.parent.mkdir(parents=True, exist_ok=True)

    # Plot 1: Average income tax by age (both scenarios)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        by_age.index,
        by_age["baseline"],
        label="Baseline (no inheritance)",
        color="C0",
        linewidth=2,
    )
    ax1.plot(
        by_age.index,
        by_age["no_care_demand"],
        label="No care demand",
        color="C1",
        linewidth=2,
    )
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Average income tax (currency)")
    ax1.set_title("Average income tax (working individuals) by age")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(path_to_save_plot_levels)
    plt.close(fig1)

    # Plot 2: Average by age of agent-period differences (no care demand minus baseline)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        avg_diff_by_age.index,
        avg_diff_by_age.values,
        color="C2",
        linewidth=2,
    )
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Average income tax difference (currency)")
    ax2.set_title(
        "Income tax difference by age (no care demand minus baseline, same agents, "
        "workers in both scenarios)"
    )
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(path_to_save_plot_difference)
    plt.close(fig2)


def convert_to_currency(df: pd.DataFrame, wealth_unit: float) -> pd.DataFrame:
    """Convert model monetary columns to actual currency."""
    df = df.copy()
    monetary_cols = [
        "income_tax",
        "income_tax_single",
        "total_tax_revenue",
        "own_ssc",
        "partner_ssc",
        "child_benefits",
        "care_benefits_and_costs",
        "household_unemployment_benefits",
        "gross_retirement_income",
        "gross_partner_pension",
        "gross_labor_income",
        "gross_partner_income",
    ]
    for col in monetary_cols:
        if col in df.columns:
            df[col] = df[col] * wealth_unit
    return df
