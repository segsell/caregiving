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

from caregiving.config import BLD, SRC
from caregiving.model.shared import (
    BETA_NPV,
    DEAD,
    INFORMAL_CARE,
    NPV_END_AGE,
    NPV_START_AGE,
)
from caregiving.simulation.simulate import (
    setup_model_for_simulation_baseline,
    setup_model_for_simulation_job_retention,
    simulate_career_costs,
)
from caregiving.simulation.simulate_no_care_demand import (
    setup_model_for_simulation_no_care_demand,
    simulate_career_costs_no_care_demand,
)


@pytask.mark.career_costs
def task_compute_career_costs(
    # Baseline
    path_to_baseline_options: Path = BLD / "model" / "options.pkl",
    path_to_baseline_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_baseline_solution: Path = BLD
    / "solve_and_simulate"
    / "solution_estimated_params.pkl",
    path_to_baseline_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "states.pkl",
    path_to_baseline_wealth_agents: Path = BLD
    / "model"
    / "initial_conditions"
    / "wealth.csv",
    path_to_baseline_params: Path = BLD
    / "model"
    / "params"
    / "estimated_params_model.yaml",
    # Counterfactual: no care demand
    path_to_no_care_demand_options: Path = BLD / "model" / "options_no_care_demand.pkl",
    path_to_no_care_demand_model: Path = BLD / "model" / "model_no_care_demand.pkl",
    path_to_no_care_demand_solution: Path = BLD
    / "solve_and_simulate"
    / "solution_no_care_demand.pkl",
    path_to_no_care_demand_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "states_no_care_demand.pkl",
    path_to_no_care_demand_wealth_agents: Path = BLD
    / "model"
    / "initial_conditions"
    / "wealth_no_care_demand.csv",
    # path_to_original_npv: Annotated[Path, Product] = BLD
    # / "counterfactual"
    # / "original_career_npv.csv",
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
) -> None:
    """Compute career costs as NPV difference between baseline and counterfactual."""

    # ===============================================================================
    # Baseline
    # ===============================================================================

    options = pkl.load(path_to_baseline_options.open("rb"))
    options_no_care_demand = pkl.load(path_to_no_care_demand_options.open("rb"))

    params = yaml.safe_load(path_to_baseline_params.open("rb"))
    solution_dict = pkl.load(path_to_baseline_solution.open("rb"))
    initial_states = pkl.load(path_to_baseline_initial_states.open("rb"))
    wealth_agents = np.array(
        pd.read_csv(path_to_baseline_wealth_agents, usecols=["wealth"]).squeeze()
    )

    model_for_simulation = setup_model_for_simulation_baseline(
        path_to_model=path_to_baseline_model,
        options=options,
    )

    # Simulate using the provided function
    df_baseline = simulate_career_costs(
        # df_baseline = simulate_scenario(
        model=model_for_simulation,
        solution=solution_dict,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options,
        seed=options["model_params"]["seed"],
    )

    # ===============================================================================
    # Counterfactual: no care demand
    # ===============================================================================

    # params_no_care_demand = yaml.safe_load(path_to_no_care_demand_params.open("rb"))
    solution_no_care_demand = pkl.load(path_to_no_care_demand_solution.open("rb"))
    initial_states_no_care_demand = pkl.load(
        path_to_no_care_demand_initial_states.open("rb")
    )
    wealth_agents_no_care_demand = np.array(
        pd.read_csv(path_to_no_care_demand_wealth_agents, usecols=["wealth"]).squeeze()
    )

    # Setup no-care-demand model for simulation
    model_no_care_demand_for_simulation = setup_model_for_simulation_no_care_demand(
        path_to_model=path_to_no_care_demand_model,
        options=options_no_care_demand,
    )

    # Simulate no-care-demand scenario using the specialized function
    df_no_care_demand = simulate_career_costs_no_care_demand(
        # df_no_care_demand = simulate_scenario_no_care_demand(
        model=model_no_care_demand_for_simulation,
        solution=solution_no_care_demand,
        initial_states=initial_states_no_care_demand,
        wealth_agents=wealth_agents_no_care_demand,
        params=params,
        options=options_no_care_demand,
        seed=options_no_care_demand["model_params"]["seed"],
    )

    # Rename dataframes to match the pattern from task_plot_labor_supply_differences.py
    df_original = df_baseline.copy()
    df_no_care_demand = df_no_care_demand.copy()

    # Restrict to alive periods (same as task_plot_labor_supply_differences.py)
    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

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
    informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiver_ids = df_original.loc[
        df_original["choice"].isin(informal_care_codes), "agent"
    ].unique()

    df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
    df_counterfactual = df_no_care_demand[
        df_no_care_demand["agent"].isin(caregiver_ids)
    ].copy()

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
def task_compute_career_costs_job_retention(
    # Baseline
    path_to_baseline_options: Path = BLD / "model" / "options.pkl",
    path_to_baseline_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_baseline_solution: Path = BLD
    / "solve_and_simulate"
    / "solution_estimated_params.pkl",
    path_to_baseline_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "states.pkl",
    path_to_baseline_wealth_agents: Path = BLD
    / "model"
    / "initial_conditions"
    / "wealth.csv",
    path_to_baseline_params: Path = BLD
    / "model"
    / "params"
    / "estimated_params_model.yaml",
    # Counterfactual: job retention
    path_to_job_retention_options: Path = BLD / "model" / "options_job_retention.pkl",
    path_to_job_retention_model: Path = BLD / "model" / "model_job_retention.pkl",
    path_to_job_retention_solution: Path = BLD
    / "solve_and_simulate"
    / "solution_job_retention_estimated_params.pkl",
    path_to_job_retention_initial_states: Path = BLD
    / "model"
    / "initial_conditions"
    / "states_job_retention.pkl",
    path_to_job_retention_wealth_agents: Path = BLD
    / "model"
    / "initial_conditions"
    / "wealth.csv",
    path_to_job_retention_npv: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_job_retention.csv",
    path_to_baseline_npv_jr: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "career_npv_baseline_for_job_retention.csv",
    path_to_npv_care_ratios_jr: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_care_ratios_job_retention.csv",
    path_to_npv_summary_jr: Annotated[Path, Product] = BLD
    / "counterfactual"
    / "npv_summary_job_retention.csv",
) -> None:
    """Compute career costs as NPV difference between baseline and job retention counterfactual.

    The job retention counterfactual introduces a policy where caregivers can keep their jobs
    (job offer probability = 1) if they provided care in the previous period.
    """

    # ===============================================================================
    # Baseline
    # ===============================================================================

    options = pkl.load(path_to_baseline_options.open("rb"))
    options_job_retention = pkl.load(path_to_job_retention_options.open("rb"))

    params = yaml.safe_load(path_to_baseline_params.open("rb"))
    solution_dict = pkl.load(path_to_baseline_solution.open("rb"))
    initial_states = pkl.load(path_to_baseline_initial_states.open("rb"))
    wealth_agents = np.array(
        pd.read_csv(path_to_baseline_wealth_agents, usecols=["wealth"]).squeeze()
    )

    model_for_simulation = setup_model_for_simulation_baseline(
        path_to_model=path_to_baseline_model,
        options=options,
    )

    # Simulate using the provided function
    df_baseline = simulate_career_costs(
        model=model_for_simulation,
        solution=solution_dict,
        initial_states=initial_states,
        wealth_agents=wealth_agents,
        params=params,
        options=options,
        seed=options["model_params"]["seed"],
    )

    # ===============================================================================
    # Counterfactual: job retention
    # ===============================================================================

    solution_job_retention = pkl.load(path_to_job_retention_solution.open("rb"))
    initial_states_job_retention = pkl.load(
        path_to_job_retention_initial_states.open("rb")
    )
    wealth_agents_job_retention = np.array(
        pd.read_csv(path_to_job_retention_wealth_agents, usecols=["wealth"]).squeeze()
    )

    # Setup job-retention model for simulation
    model_job_retention_for_simulation = setup_model_for_simulation_job_retention(
        path_to_model=path_to_job_retention_model,
        options=options_job_retention,
    )

    # Simulate job-retention scenario using the same simulate_career_costs function
    df_job_retention = simulate_career_costs(
        model=model_job_retention_for_simulation,
        solution=solution_job_retention,
        initial_states=initial_states_job_retention,
        wealth_agents=wealth_agents_job_retention,
        params=params,
        options=options_job_retention,
        seed=options_job_retention["model_params"]["seed"],
    )

    # Rename dataframes to match the pattern
    df_original = df_baseline.copy()
    df_job_retention_counterfactual = df_job_retention.copy()

    # Restrict to alive periods
    df_original = df_original[df_original["health"] != DEAD].copy()
    df_job_retention_counterfactual = df_job_retention_counterfactual[
        df_job_retention_counterfactual["health"] != DEAD
    ].copy()

    # Ensure 'agent' column exists
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_job_retention_counterfactual.columns:
        if isinstance(df_job_retention_counterfactual.index, pd.MultiIndex) and (
            "agent" in df_job_retention_counterfactual.index.names
        ):
            df_job_retention_counterfactual = (
                df_job_retention_counterfactual.reset_index(level=["agent"])
            )  # keep period indexed
        else:
            df_job_retention_counterfactual = (
                df_job_retention_counterfactual.reset_index()
            )

    # Restrict to ever-caregivers
    informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiver_ids = df_original.loc[
        df_original["choice"].isin(informal_care_codes), "agent"
    ].unique()

    df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
    df_counterfactual = df_job_retention_counterfactual[
        df_job_retention_counterfactual["agent"].isin(caregiver_ids)
    ].copy()

    # Compute NPV for original scenario
    original_npv = compute_career_npv(df_original, BETA_NPV)
    original_npv.to_csv(path_to_baseline_npv_jr, index=False)

    # Compute NPV for job-retention scenario
    job_retention_npv = compute_career_npv(df_counterfactual, BETA_NPV)
    job_retention_npv.to_csv(path_to_job_retention_npv, index=False)

    # Compute career costs (difference in NPV)
    # Align by agent to ensure correct ratios
    _merged_npv = pd.merge(
        original_npv,
        job_retention_npv,
        on="agent",
        suffixes=("_baseline", "_job_retention"),
    )
    _npv_care = 1 - (
        _merged_npv["career_npv_job_retention"] / _merged_npv["career_npv_baseline"]
    )
    _npv_mean = 1 - (
        _merged_npv["career_npv_job_retention"].mean()
        / _merged_npv["career_npv_baseline"].mean()
    )

    # Save results to CSV files
    npv_care_mean = _npv_care.mean()

    # Save individual NPV care ratios
    pd.DataFrame({"agent": _merged_npv["agent"], "npv_care_ratio": _npv_care}).to_csv(
        path_to_npv_care_ratios_jr, index=False
    )
    # Save summary statistics
    pd.DataFrame(
        {"metric": ["npv_care_mean", "npv_mean"], "value": [npv_care_mean, _npv_mean]}
    ).to_csv(path_to_npv_summary_jr, index=False)

    print(f"NPV care mean (job retention): {npv_care_mean:.4f}")
    print(f"NPV mean (job retention): {_npv_mean:.4f}")
    print(f"Results saved to {path_to_npv_care_ratios_jr}")
    print(f"Summary saved to {path_to_npv_summary_jr}")


def compute_career_npv(df: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Compute net present value of total income."""

    df_filtered = df[(df["age"] >= NPV_START_AGE) & (df["age"] <= NPV_END_AGE)].copy()

    # Create discount factors (beta^(age-40))
    df_filtered["discount_factor"] = beta ** (df_filtered["age"] - NPV_START_AGE)

    # Compute discounted income using total_income (individual income components)
    df_filtered["discounted_income"] = (
        df_filtered["total_income"] * df_filtered["discount_factor"]
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
