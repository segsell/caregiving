"""Compute career costs by comparing NPV of incomes between scenarios.

Baseline and no-care demand counterfactual.
"""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
import yaml
from pytask import Product

import dcegm
from caregiving.config import BLD, SRC
from caregiving.model.shared import (
    BETA_NPV,
    DEAD,
    INFORMAL_CARE,
    NPV_END_AGE,
    NPV_START_AGE,
)
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation import simulate_no_care_demand
from caregiving.simulation.simulate import simulate_scenario


@pytask.mark.career_costs
def task_compute_career_costs(
    # Baseline
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_career_costs.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_career_costs.pkl",
    path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_no_care_demand.csv",
    path_to_baseline_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_baseline.csv",
    path_to_npv_care_ratios: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios.csv",
    path_to_npv_summary: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary.csv",
    restrict_to_ever_caregivers: bool = True,
    restrict_to_ever_care_demand: bool = False,
) -> None:
    """Compute career costs as NPV difference between baseline and counterfactual."""

    # ===============================================================================
    # Baseline
    # ===============================================================================

    specs = pkl.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Rename dataframes to match the pattern from task_plot_labor_supply_differences.py
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    # # Restrict to alive periods (same as task_plot_labor_supply_differences.py)
    # df_original = df_original[df_original["health"] != DEAD].copy()
    # df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure 'agent' column exists (same pattern as
    # task_plot_labor_supply_differences.py)
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Restrict to ever-caregivers (same pattern as
    # task_plot_labor_supply_differences.py)

    if restrict_to_ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()
    elif restrict_to_ever_care_demand:
        care_demand_ids = df_original.loc[
            df_original["care_demand"] > 0, "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(care_demand_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(care_demand_ids)
        ].copy()
    else:
        df_original = df_original.copy()
        df_counterfactual = df_no_care_demand.copy()

    # Create care flags for the restricted original data
    # df_original_care = create_care_flags(df_original)

    # Compute NPV for original scenario
    original_npv = compute_career_npv(df_original, BETA_NPV)
    original_npv.to_csv(path_to_baseline_npv, index=False)

    # Compute NPV for no-care-demand scenario
    no_care_demand_npv = compute_career_npv(df_counterfactual, BETA_NPV)
    no_care_demand_npv.to_csv(path_to_no_care_demand_npv, index=False)

    # Compute career costs (difference in NPV)
    # career_costs = compute_career_costs(original_npv, no_care_demand_npv)

    # _npv_care = (
    #     1
    #     - career_costs["career_npv_no_care_demand"]
    #     / career_costs["career_npv_baseline"]
    # )
    # Align by agent to ensure correct ratios
    _merged_npv = pd.merge(
        original_npv,
        no_care_demand_npv,
        on="agent",
        suffixes=("_baseline", "_no_care_demand"),
    )

    # Filter out agents with zero or very small baseline NPV to avoid division by zero
    _merged_npv = _merged_npv[_merged_npv["career_npv_baseline"] > 1e-10].copy()

    _npv_care = 1 - (
        _merged_npv["career_npv_no_care_demand"] / _merged_npv["career_npv_baseline"]
    )
    _npv_mean = 1 - (
        _merged_npv["career_npv_no_care_demand"].mean()
        / _merged_npv["career_npv_baseline"].mean()
    )

    # Save results to CSV files
    npv_care_mean = _npv_care.mean()

    # Save individual NPV care ratios
    pd.DataFrame({"agent": _merged_npv["agent"], "npv_care_ratio": _npv_care}).to_csv(
        path_to_npv_care_ratios, index=False
    )
    # Save summary statistics
    pd.DataFrame(
        {"metric": ["npv_care_mean", "npv_mean"], "value": [npv_care_mean, _npv_mean]}
    ).to_csv(path_to_npv_summary, index=False)

    print(f"NPV care mean: {npv_care_mean:.4f}")
    print(f"NPV mean: {_npv_mean:.4f}")
    print(f"Results saved to {path_to_npv_care_ratios}")
    print(f"Summary saved to {path_to_npv_summary}")


@pytask.mark.career_costs
@pytask.mark.career_costs_caregiving_leave_with_job_retention
def task_compute_career_costs_caregiving_leave_with_job_retention(
    # Baseline
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params_career_costs.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_career_costs.pkl",
    path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_caregiving_leave_with_job_retention.csv",
    path_to_baseline_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_caregiving_leave_with_job_retention.csv",
    path_to_npv_care_ratios: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios_caregiving_leave_with_job_retention.csv",
    path_to_npv_summary: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary_caregiving_leave_with_job_retention.csv",
    restrict_to_ever_caregivers: bool = True,
    restrict_to_ever_care_demand: bool = False,
) -> None:
    """Compute career costs as NPV difference between baseline and counterfactual."""

    # ===============================================================================
    # Baseline
    # ===============================================================================

    specs = pkl.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_caregiving_leave_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Rename dataframes to match the pattern from task_plot_labor_supply_differences.py
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    # # Restrict to alive periods (same as task_plot_labor_supply_differences.py)
    # df_original = df_original[df_original["health"] != DEAD].copy()
    # df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure 'agent' column exists (same pattern as
    # task_plot_labor_supply_differences.py)
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Restrict to ever-caregivers (same pattern as
    # task_plot_labor_supply_differences.py)

    if restrict_to_ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()
    elif restrict_to_ever_care_demand:
        care_demand_ids = df_original.loc[
            df_original["care_demand"] > 0, "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(care_demand_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(care_demand_ids)
        ].copy()
    else:
        df_original = df_original.copy()
        df_counterfactual = df_no_care_demand.copy()

    # Create care flags for the restricted original data
    # df_original_care = create_care_flags(df_original)

    # Compute NPV for original scenario
    original_npv = compute_career_npv(df_original, BETA_NPV)
    original_npv.to_csv(path_to_baseline_npv, index=False)

    # Compute NPV for no-care-demand scenario
    no_care_demand_npv = compute_career_npv(df_counterfactual, BETA_NPV)
    no_care_demand_npv.to_csv(path_to_no_care_demand_npv, index=False)

    # Compute career costs (difference in NPV)
    # career_costs = compute_career_costs(original_npv, no_care_demand_npv)

    # _npv_care = (
    #     1
    #     - career_costs["career_npv_no_care_demand"]
    #     / career_costs["career_npv_baseline"]
    # )
    # Align by agent to ensure correct ratios
    _merged_npv = pd.merge(
        original_npv,
        no_care_demand_npv,
        on="agent",
        suffixes=("_baseline", "_no_care_demand"),
    )

    # Filter out agents with zero or very small baseline NPV to avoid division by zero
    _merged_npv = _merged_npv[_merged_npv["career_npv_baseline"] > 1e-10].copy()

    _npv_care = 1 - (
        _merged_npv["career_npv_no_care_demand"] / _merged_npv["career_npv_baseline"]
    )
    _npv_mean = 1 - (
        _merged_npv["career_npv_no_care_demand"].mean()
        / _merged_npv["career_npv_baseline"].mean()
    )

    # Save results to CSV files
    npv_care_mean = _npv_care.mean()

    # Save individual NPV care ratios
    pd.DataFrame({"agent": _merged_npv["agent"], "npv_care_ratio": _npv_care}).to_csv(
        path_to_npv_care_ratios, index=False
    )
    # Save summary statistics
    pd.DataFrame(
        {"metric": ["npv_care_mean", "npv_mean"], "value": [npv_care_mean, _npv_mean]}
    ).to_csv(path_to_npv_summary, index=False)

    print(f"NPV care mean: {npv_care_mean:.4f}")
    print(f"NPV mean: {_npv_mean:.4f}")
    print(f"Results saved to {path_to_npv_care_ratios}")
    print(f"Summary saved to {path_to_npv_summary}")


@pytask.mark.career_costs
@pytask.mark.career_costs_full_caregiving_leave_with_job_retention
def task_compute_career_costs_full_caregiving_leave_with_job_retention(
    # Baseline
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params_career_costs.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_career_costs.pkl",
    path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_full_caregiving_leave_with_job_retention.csv",
    path_to_baseline_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_full_caregiving_leave_with_job_retention.csv",
    path_to_npv_care_ratios: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios_full_caregiving_leave_with_job_retention.csv",
    path_to_npv_summary: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary_full_caregiving_leave_with_job_retention.csv",
    restrict_to_ever_caregivers: bool = True,
    restrict_to_ever_care_demand: bool = False,
) -> None:
    """Compute career costs as NPV difference between baseline and counterfactual."""

    # ===============================================================================
    # Baseline
    # ===============================================================================

    specs = pkl.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_caregiving_leave_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Rename dataframes to match the pattern from task_plot_labor_supply_differences.py
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    # # Restrict to alive periods (same as task_plot_labor_supply_differences.py)
    # df_original = df_original[df_original["health"] != DEAD].copy()
    # df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure 'agent' column exists (same pattern as
    # task_plot_labor_supply_differences.py)
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Restrict to ever-caregivers (same pattern as
    # task_plot_labor_supply_differences.py)

    if restrict_to_ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()
    elif restrict_to_ever_care_demand:
        care_demand_ids = df_original.loc[
            df_original["care_demand"] > 0, "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(care_demand_ids)].copy()
        df_counterfactual = df_no_care_demand[
            df_no_care_demand["agent"].isin(care_demand_ids)
        ].copy()
    else:
        df_original = df_original.copy()
        df_counterfactual = df_no_care_demand.copy()

    # Create care flags for the restricted original data
    # df_original_care = create_care_flags(df_original)

    # Compute NPV for original scenario
    original_npv = compute_career_npv(df_original, BETA_NPV)
    original_npv.to_csv(path_to_baseline_npv, index=False)

    # Compute NPV for no-care-demand scenario
    no_care_demand_npv = compute_career_npv(df_counterfactual, BETA_NPV)
    no_care_demand_npv.to_csv(path_to_no_care_demand_npv, index=False)

    # Compute career costs (difference in NPV)
    # career_costs = compute_career_costs(original_npv, no_care_demand_npv)

    # _npv_care = (
    #     1
    #     - career_costs["career_npv_no_care_demand"]
    #     / career_costs["career_npv_baseline"]
    # )
    # Align by agent to ensure correct ratios
    _merged_npv = pd.merge(
        original_npv,
        no_care_demand_npv,
        on="agent",
        suffixes=("_baseline", "_no_care_demand"),
    )

    # Filter out agents with zero or very small baseline NPV to avoid division by zero
    _merged_npv = _merged_npv[_merged_npv["career_npv_baseline"] > 1e-10].copy()

    _npv_care = 1 - (
        _merged_npv["career_npv_no_care_demand"] / _merged_npv["career_npv_baseline"]
    )
    _npv_mean = 1 - (
        _merged_npv["career_npv_no_care_demand"].mean()
        / _merged_npv["career_npv_baseline"].mean()
    )

    # Save results to CSV files
    npv_care_mean = _npv_care.mean()

    # Save individual NPV care ratios
    pd.DataFrame({"agent": _merged_npv["agent"], "npv_care_ratio": _npv_care}).to_csv(
        path_to_npv_care_ratios, index=False
    )
    # Save summary statistics
    pd.DataFrame(
        {"metric": ["npv_care_mean", "npv_mean"], "value": [npv_care_mean, _npv_mean]}
    ).to_csv(path_to_npv_summary, index=False)

    print(f"NPV care mean: {npv_care_mean:.4f}")
    print(f"NPV mean: {_npv_mean:.4f}")
    print(f"Results saved to {path_to_npv_care_ratios}")
    print(f"Summary saved to {path_to_npv_summary}")


def compute_career_npv(
    df: pd.DataFrame,
    beta: float,
    include_bequest_from_parent: bool = False,
    only_own_income: bool = True,
) -> pd.DataFrame:
    """Compute net present value of total income."""

    df_filtered = df[(df["age"] >= NPV_START_AGE) & (df["age"] <= NPV_END_AGE)].copy()

    # Divide household_unemployment_benefits by 2 if has partner (partner_state > 0)
    household_unemployment = df_filtered["household_unemployment_benefits"].copy()
    if "partner_state" in df_filtered.columns:
        household_unemployment = np.where(
            df_filtered["partner_state"] > 0,
            household_unemployment / 2,
            household_unemployment,
        )

    if only_own_income:
        base_sum = df_filtered["own_income_after_ssc"]
    else:
        # Efficiently handle missing variables: compute sum directly without creating intermediate Series
        # This avoids memory overhead from creating full columns with zeros
        base_sum = (
            df_filtered["own_income_after_ssc"]
            + household_unemployment
            + df_filtered["child_benefits"]
        )
        # Add optional columns only if they exist (avoid creating zero columns)
        if "care_benefits_and_costs" in df_filtered.columns:
            base_sum = base_sum + df_filtered["care_benefits_and_costs"].fillna(0)
        if "caregiving_leave_top_up" in df_filtered.columns:
            base_sum = base_sum + df_filtered["caregiving_leave_top_up"].fillna(0)
        if include_bequest_from_parent:
            base_sum = base_sum + df_filtered["bequest_from_parent"].fillna(0)

    df_filtered["total_gross_income"] = base_sum

    # Apply maximum with unemployment benefits (following budget equation)
    df_filtered["total_npv_income"] = np.where(
        df_filtered["health"] != DEAD,
        np.maximum(df_filtered["total_gross_income"], 0),
        0,
    )

    # Create discount factors (beta^(age-40))
    df_filtered["discount_factor"] = beta ** (df_filtered["age"] - NPV_START_AGE)

    # Compute discounted income using total_income (individual income components)
    df_filtered["discounted_income"] = (
        df_filtered["total_npv_income"] * df_filtered["discount_factor"]
    )

    # Group by agent and sum discounted income to get NPV
    npv_by_agent = df_filtered.groupby("agent")["discounted_income"].sum().reset_index()
    npv_by_agent.columns = ["agent", "career_npv"]

    # # Merge with agent characteristics (education, sex, etc.)
    # agent_chars = (
    #     df_filtered.groupby("agent")
    #     .first()[["education", "sex", "partner", "children"]]
    #     .reset_index()
    # )

    # result = pd.merge(npv_by_agent, agent_chars, on="agent")

    return npv_by_agent


def compute_career_costs(
    original_npv: pd.DataFrame, no_care_demand_npv: pd.DataFrame
) -> pd.DataFrame:
    """Compute career costs as difference in NPV between scenarios."""

    # Merge the two NPV datasets
    merged = pd.merge(
        original_npv,
        no_care_demand_npv,
        on=["agent"],
        suffixes=("_original", "_no_care_demand"),
    )

    merged["career_costs"] = (
        merged["career_npv_no_care_demand"] - merged["career_npv_original"]
    )

    # Select relevant columns
    result = merged[
        [
            "agent",
            "career_npv_original",
            "career_npv_no_care_demand",
            "career_costs",
        ]
    ].copy()

    return result


def create_care_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create care ever and sum care variables."""
    # Caregiving
    df["informal_care"] = df["choice"].isin(np.asarray(INFORMAL_CARE).ravel().tolist())

    # Care ever - use agent column for grouping
    df["care_ever"] = df.groupby("agent")["informal_care"].transform(
        lambda x: x.cumsum().clip(upper=1)
    )

    # # Sum care
    # df["sum_informal_care"] = df.groupby("agent")["informal_care"].transform(
    #     lambda x: x.cumsum()
    # )

    # # Care demand ever
    # df["care_demand_ever"] = df.groupby("agent")["care_demand"].transform(
    #     lambda x: x.cumsum().clip(upper=1)
    # )

    return df


@pytask.mark.government_budget
def task_compute_government_budget_caregiving_leave(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_output: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "government_budget_caregiving_leave.csv",
) -> None:
    """Compute sum of government budget variables for caregiving leave scenario.

    Sums net_government_budget, caregiving_leave_top_up, and income_tax_single
    over all agents and periods up to and including max_ret_age.
    Each period-specific outcome is divided by the number of agents in that period
    before summing to account for deaths.
    """
    specs = pkl.load(path_to_specs.open("rb"))
    max_ret_age = specs["max_ret_age"]

    df = pd.read_pickle(path_to_caregiving_leave_data)
    df_baseline = pd.read_pickle(path_to_baseline_data)
    # net = (
    #     (df["income_tax_single"].sum() - df_baseline["income_tax_single"].sum())
    #     + (
    #         df["caregiving_leave_top_up"].sum()
    #         - df_baseline["caregiving_leave_top_up"].sum()
    #     )
    #     + (
    #         df["net_government_budget"].sum()
    #         - df_baseline["net_government_budget"].sum()
    #     )
    # )
    # breakpoint()

    # Ensure agent and period columns exist
    if "agent" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("agent" in df.index.names):
            df = df.reset_index(level=["agent"])
        else:
            df = df.reset_index()

    # Ensure age column exists
    if "age" not in df.columns:
        if "period" in df.columns:
            df["age"] = df["period"] + specs["start_age"]
        else:
            raise ValueError("Neither 'age' nor 'period' column found in data")

    # Filter to periods up to and including max_ret_age
    df_filtered = df[df["age"] <= max_ret_age].copy()

    # Ensure required columns exist, set to zero if missing
    if "net_government_budget" not in df_filtered.columns:
        df_filtered["net_government_budget"] = 0
    if "caregiving_leave_top_up" not in df_filtered.columns:
        df_filtered["caregiving_leave_top_up"] = 0
    if "income_tax_single" not in df_filtered.columns:
        df_filtered["income_tax_single"] = 0

    # Fill NaN values with 0
    df_filtered["net_government_budget"] = df_filtered["net_government_budget"].fillna(
        0
    )
    df_filtered["caregiving_leave_top_up"] = df_filtered[
        "caregiving_leave_top_up"
    ].fillna(0)
    df_filtered["income_tax_single"] = df_filtered["income_tax_single"].fillna(0)

    # Group by period and compute per-agent averages (dividing by number of agents)
    period_stats = df_filtered.groupby("period").agg(
        {
            "net_government_budget": ["sum", "count"],
            "caregiving_leave_top_up": "sum",
            "income_tax_single": "sum",
        }
    )

    # Compute per-agent values for each period (divide by number of agents)
    period_stats["net_government_budget_per_agent"] = (
        period_stats[("net_government_budget", "sum")]
        / period_stats[("net_government_budget", "count")]
    )
    period_stats["caregiving_leave_top_up_per_agent"] = (
        period_stats[("caregiving_leave_top_up", "sum")]
        / period_stats[("net_government_budget", "count")]
    )
    period_stats["income_tax_single_per_agent"] = (
        period_stats[("income_tax_single", "sum")]
        / period_stats[("net_government_budget", "count")]
    )

    # Sum across all periods
    total_net_government_budget = period_stats["net_government_budget_per_agent"].sum()
    total_caregiving_leave_top_up = period_stats[
        "caregiving_leave_top_up_per_agent"
    ].sum()
    total_income_tax_single = period_stats["income_tax_single_per_agent"].sum()

    # Save to CSV
    result = pd.DataFrame(
        {
            "variable": [
                "net_government_budget",
                "caregiving_leave_top_up",
                "income_tax_single",
            ],
            "total_sum": [
                total_net_government_budget,
                total_caregiving_leave_top_up,
                total_income_tax_single,
            ],
        }
    )
    result.to_csv(path_to_output, index=False)
    breakpoint()

    print(f"Government budget sums saved to {path_to_output}")
    print(f"Net government budget: {total_net_government_budget:.2f}")
    print(f"Caregiving leave top-up: {total_caregiving_leave_top_up:.2f}")
    print(f"Income tax single: {total_income_tax_single:.2f}")


@pytask.mark.government_budget
def task_compute_government_budget_full_caregiving_leave(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_full_caregiving_leave_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_output: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "government_budget_full_caregiving_leave.csv",
) -> None:
    """Compute sum of government budget variables for full caregiving leave scenario.

    Sums net_government_budget, caregiving_leave_top_up, and income_tax_single
    over all agents and periods up to and including max_ret_age.
    Each period-specific outcome is divided by the number of agents in that period
    before summing to account for deaths.
    """
    specs = pkl.load(path_to_specs.open("rb"))
    max_ret_age = specs["max_ret_age"]

    df = pd.read_pickle(path_to_full_caregiving_leave_data)

    # Ensure agent and period columns exist
    if "agent" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("agent" in df.index.names):
            df = df.reset_index(level=["agent"])
        else:
            df = df.reset_index()

    # Ensure age column exists
    if "age" not in df.columns:
        if "period" in df.columns:
            df["age"] = df["period"] + specs["start_age"]
        else:
            raise ValueError("Neither 'age' nor 'period' column found in data")

    # Filter to periods up to and including max_ret_age
    df_filtered = df[df["age"] <= max_ret_age].copy()

    # Ensure required columns exist, set to zero if missing
    if "net_government_budget" not in df_filtered.columns:
        df_filtered["net_government_budget"] = 0
    if "caregiving_leave_top_up" not in df_filtered.columns:
        df_filtered["caregiving_leave_top_up"] = 0
    if "income_tax_single" not in df_filtered.columns:
        df_filtered["income_tax_single"] = 0

    # Fill NaN values with 0
    df_filtered["net_government_budget"] = df_filtered["net_government_budget"].fillna(
        0
    )
    df_filtered["caregiving_leave_top_up"] = df_filtered[
        "caregiving_leave_top_up"
    ].fillna(0)
    df_filtered["income_tax_single"] = df_filtered["income_tax_single"].fillna(0)

    # Group by period and compute per-agent averages (dividing by number of agents)
    period_stats = df_filtered.groupby("period").agg(
        {
            "net_government_budget": ["sum", "count"],
            "caregiving_leave_top_up": "sum",
            "income_tax_single": "sum",
        }
    )

    # Compute per-agent values for each period (divide by number of agents)
    period_stats["net_government_budget_per_agent"] = (
        period_stats[("net_government_budget", "sum")]
        / period_stats[("net_government_budget", "count")]
    )
    period_stats["caregiving_leave_top_up_per_agent"] = (
        period_stats[("caregiving_leave_top_up", "sum")]
        / period_stats[("net_government_budget", "count")]
    )
    period_stats["income_tax_single_per_agent"] = (
        period_stats[("income_tax_single", "sum")]
        / period_stats[("net_government_budget", "count")]
    )

    # Sum across all periods
    total_net_government_budget = period_stats["net_government_budget_per_agent"].sum()
    total_caregiving_leave_top_up = period_stats[
        "caregiving_leave_top_up_per_agent"
    ].sum()
    total_income_tax_single = period_stats["income_tax_single_per_agent"].sum()

    # Save to CSV
    result = pd.DataFrame(
        {
            "variable": [
                "net_government_budget",
                "caregiving_leave_top_up",
                "income_tax_single",
            ],
            "total_sum": [
                total_net_government_budget,
                total_caregiving_leave_top_up,
                total_income_tax_single,
            ],
        }
    )
    result.to_csv(path_to_output, index=False)

    print(f"Government budget sums saved to {path_to_output}")
    print(f"Net government budget: {total_net_government_budget:.2f}")
    print(f"Caregiving leave top-up: {total_caregiving_leave_top_up:.2f}")
    print(f"Income tax single: {total_income_tax_single:.2f}")
