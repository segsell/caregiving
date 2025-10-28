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
from caregiving.model.shared import DEAD, INFORMAL_CARE, NPV_END_AGE, NPV_START_AGE
from caregiving.simulation.simulate import (
    setup_model_for_simulation_baseline,
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
    path_to_no_care_demand_params: Path = BLD
    / "model"
    / "params"
    / "estimated_params_model.yaml",
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
    # Counterfactual: no care demand
    # ===============================================================================

    options_no_care_demand = pkl.load(path_to_no_care_demand_options.open("rb"))

    params_no_care_demand = yaml.safe_load(path_to_no_care_demand_params.open("rb"))
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
        model=model_no_care_demand_for_simulation,
        solution=solution_no_care_demand,
        initial_states=initial_states_no_care_demand,
        wealth_agents=wealth_agents_no_care_demand,
        params=params_no_care_demand,
        options=options_no_care_demand,
        seed=options_no_care_demand["model_params"]["seed"],
    )

    # Load beta
    beta = 0.95

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
    df_no_care_demand = df_no_care_demand[
        df_no_care_demand["agent"].isin(caregiver_ids)
    ].copy()

    # Create care flags for the restricted original data
    df_original_care_ever = create_care_flags(df_original)

    # Compute NPV for original scenario
    original_npv = compute_career_npv(df_original_care_ever, beta)
    # original_npv.to_csv(path_to_original_npv)

    # Compute NPV for no-care-demand scenario
    no_care_demand_npv = compute_career_npv(df_no_care_demand, beta)
    # no_care_demand_npv.to_csv(path_to_no_care_demand_npv)

    # Compute career costs (difference in NPV)
    # career_costs = compute_career_costs(original_npv, no_care_demand_npv)

    # _npv_care = (
    #     1
    #     - career_costs["career_npv_no_care_demand"]
    #     / career_costs["career_npv_baseline"]
    # )
    _npv_care = 1 - no_care_demand_npv["career_npv"] / original_npv["career_npv"]
    _npv_mean = (
        1 - no_care_demand_npv["career_npv"].mean() / original_npv["career_npv"].mean()
    )

    # Save results to CSV files
    npv_care_mean = _npv_care.mean()

    # Save individual NPV care ratios
    pd.DataFrame({"agent": original_npv["agent"], "npv_care_ratio": _npv_care}).to_csv(
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


def compute_career_npv(df: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Compute net present value of total income from age 30 to 70."""

    # Filter data for ages 30-80
    df_filtered = df[(df["age"] >= NPV_START_AGE) & (df["age"] <= NPV_END_AGE)].copy()

    # Create discount factors (beta^(age-30))
    df_filtered["discount_factor"] = beta ** (df_filtered["age"] - NPV_START_AGE)

    # Compute discounted income using total_income (individual income components)
    df_filtered["discounted_income"] = (
        df_filtered["total_income"] * df_filtered["discount_factor"]
    )

    # Group by agent and sum discounted income to get NPV
    npv_by_agent = (
        df_filtered.groupby(level="agent")["discounted_income"].sum().reset_index()
    )
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
