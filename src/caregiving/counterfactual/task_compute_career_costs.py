"""Compute career costs by comparing NPV of incomes between scenarios.

Baseline and no-care demand counterfactual.
"""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import yaml
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.shared import INFORMAL_CARE, NPV_END_AGE, NPV_START_AGE
from caregiving.simulation.simulate import (
    setup_model_for_simulation_baseline,
    simulate_career_costs,
)
from caregiving.simulation.simulate_no_care_demand import (
    setup_model_for_simulation_no_care_demand,
    simulate_career_costs_no_care_demand,
)


def task_compute_career_costs(
    # Baseline
    path_to_baseline_options: Path = BLD / "model" / "options.pkl",
    path_to_baseline_model: Path = BLD / "model" / "model_for_solution.pkl",
    path_to_baseline_solution: Path = BLD / "solve_and_simulate" / "solution.pkl",
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
    / "start_params_model.yaml",
    # Counterfactual: no care demand
    path_to_no_care_demand_options: Path = BLD / "model" / "options_no_care_demand.pkl",
    path_to_no_care_demand_params: Path = BLD / "model" / "params"
    # / "start_params_model_no_care_demand.yaml",
    / "params_estimated_no_care_demand.yaml",
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
    # path_to_no_care_demand_npv: Annotated[Path, Product] = BLD
    # / "counterfactual"
    # / "no_care_demand_career_npv.csv",
    # path_to_career_costs: Annotated[Path, Product] = BLD
    # / "counterfactual"
    # / "career_costs.csv",
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

    df_baseline_care_ever = create_care_flags(df_baseline)

    # Get agent IDs that have care_ever == 1 in baseline
    agents_with_care = (
        df_baseline_care_ever[df_baseline_care_ever["care_ever"] == 1]
        .index.get_level_values("agent")
        .unique()
    )

    # Subset baseline data to only include agents who provided care
    df_baseline_care_ever_subset = df_baseline_care_ever[
        df_baseline_care_ever.index.get_level_values("agent").isin(agents_with_care)
    ]

    # Subset no-care-demand data to only include agents who provided care in baseline
    df_no_care_demand_subset = df_no_care_demand[
        df_no_care_demand.index.get_level_values("agent").isin(agents_with_care)
    ]

    # Compute NPV for baseline scenario
    baseline_npv = compute_career_npv(df_baseline_care_ever_subset, beta)
    # baseline_npv.to_csv(path_to_baseline_npv)

    # Compute NPV for no-care-demand scenario (subsetted to care providers)
    no_care_demand_npv = compute_career_npv(df_no_care_demand_subset, beta)
    # no_care_demand_npv.to_csv(path_to_no_care_demand_npv)

    # Compute career costs (difference in NPV)
    # career_costs = compute_career_costs(baseline_npv, no_care_demand_npv)

    # _npv_care = (
    #     1
    #     - career_costs["career_npv_no_care_demand"]
    #     / career_costs["career_npv_baseline"]
    # )
    _npv_care = 1 - no_care_demand_npv["career_npv"] / baseline_npv["career_npv"]
    _npv_mean = (
        1 - no_care_demand_npv["career_npv"].mean() / baseline_npv["career_npv"].mean()
    )


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
    baseline_npv: pd.DataFrame, no_care_demand_npv: pd.DataFrame
) -> pd.DataFrame:
    """Compute career costs as difference in NPV between scenarios."""

    # Merge the two NPV datasets
    merged = pd.merge(
        baseline_npv,
        no_care_demand_npv,
        on=["agent"],
        suffixes=("_baseline", "_no_care_demand"),
    )

    merged["career_costs"] = (
        merged["career_npv_no_care_demand"] - merged["career_npv_baseline"]
    )

    # Select relevant columns
    result = merged[
        [
            "agent",
            "career_npv_baseline",
            "career_npv_no_care_demand",
            "career_costs",
        ]
    ].copy()

    return result


def create_care_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create care ever and sum care variables."""
    # Caregiving
    df["informal_care"] = df["choice"].isin(np.asarray(INFORMAL_CARE).ravel().tolist())

    # Care ever - use MultiIndex level for grouping
    df["care_ever"] = df.groupby(level="agent")["informal_care"].transform(
        lambda x: x.cumsum().clip(upper=1)
    )

    # # Sum care
    # df["sum_informal_care"] = df.groupby(level="agent")["informal_care"].transform(
    #     lambda x: x.cumsum()
    # )

    # # Care demand ever
    # df["care_demand_ever"] = df.groupby(level="agent")["care_demand"].transform(
    #     lambda x: x.cumsum().clip(upper=1)
    # )

    return df
