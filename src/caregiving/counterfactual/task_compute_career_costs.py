"""Compute career costs by comparing NPV of incomes between scenarios.

Baseline and no-care demand counterfactual.
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import yaml
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.shared import NPV_END_AGE, NPV_START_AGE


def task_compute_career_costs(
    path_to_original_data: Path = BLD / "solve_and_simulate" / "simulated_data.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
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

    # Load original simulated data
    df_original = pd.read_pickle(path_to_original_data)

    # Load no-care-demand simulated data
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Load beta
    beta = 0.98

    # Compute NPV for original scenario
    original_npv = compute_career_npv(df_original, beta)
    # original_npv.to_csv(path_to_original_npv)

    # Compute NPV for no-care-demand scenario
    no_care_demand_npv = compute_career_npv(df_no_care_demand, beta)
    # no_care_demand_npv.to_csv(path_to_no_care_demand_npv)

    # Compute career costs (difference in NPV)
    career_costs = compute_career_costs(original_npv, no_care_demand_npv)

    _npv_care = (
        1
        - career_costs["career_npv_no_care_demand"]
        / career_costs["career_npv_original"]
    )


def compute_career_npv(df: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Compute net present value of total income from age 30 to 80."""

    # Filter data for ages 30-80
    df_filtered = df[(df["age"] >= NPV_START_AGE) & (df["age"] <= NPV_END_AGE)].copy()

    # Create discount factors (beta^(age-30))
    df_filtered["discount_factor"] = beta ** (df_filtered["age"] - NPV_START_AGE)

    # Compute discounted income
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
