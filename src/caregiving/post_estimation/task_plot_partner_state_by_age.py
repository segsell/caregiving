"""Plot partner state shares by age and education from simulated data.

Creates plots showing the share (proportion) of agents in each partner state category:
- No partner (Single)
- Working partner
- Retired partner

By age (x-axis) and separately for low and high education, resulting in 6 lines total.

Available for both baseline and no care demand counterfactual scenarios.
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
from caregiving.model.shared import (
    NO_PARTNER,
    PARTNER_RETIRED,
    PARTNER_WORKING,
)


@pytask.mark.baseline_model
@pytask.mark.post_estimation
def task_plot_partner_state_by_age_baseline(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "partner_state_by_age_baseline.png",
):
    """Plot partner state shares by age and education from baseline simulated data.

    Creates a plot with 6 lines:
    - Low education: No partner, Working partner, Retired partner
    - High education: No partner, Working partner, Retired partner

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to baseline simulated data pkl file
    path_to_plot : Path
        Path to save the plot
    """
    # Load specs and simulated data
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    df_sim = pd.read_pickle(path_to_simulated_data)

    # Check if agent and period are in the index (MultiIndex) or already columns
    if isinstance(df_sim.index, pd.MultiIndex):
        df_sim = df_sim.reset_index()
    elif "agent" not in df_sim.columns:
        if hasattr(df_sim.index, "names") and "agent" in df_sim.index.names:
            df_sim = df_sim.reset_index()
        else:
            raise ValueError(
                f"Cannot find 'agent' column or index level. "
                f"Available columns: {df_sim.columns.tolist()}, "
                f"Index names: {df_sim.index.names if hasattr(df_sim.index, 'names') else 'N/A'}"
            )

    # Verify agent and period columns exist
    if "agent" not in df_sim.columns or "period" not in df_sim.columns:
        raise ValueError(
            f"Missing required columns 'agent' or 'period'. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Create age variable if not already present
    if "age" not in df_sim.columns:
        df_sim["age"] = df_sim["period"] + specs["start_age"]

    # Verify required columns exist
    required_cols = ["education", "partner_state"]
    missing_cols = [col for col in required_cols if col not in df_sim.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Create binary indicators for each partner state
    df_sim["no_partner"] = (df_sim["partner_state"] == NO_PARTNER).astype(float)
    df_sim["working_partner"] = (df_sim["partner_state"] == PARTNER_WORKING).astype(
        float
    )
    df_sim["retired_partner"] = (df_sim["partner_state"] == PARTNER_RETIRED).astype(
        float
    )

    # Calculate share (mean) by age and education for each partner state
    partner_state_cols = ["no_partner", "working_partner", "retired_partner"]
    share_by_group = (
        df_sim.groupby(["age", "education"], observed=False)[partner_state_cols]
        .mean()
        .reset_index()
    )

    # Get all unique ages
    ages = np.sort(df_sim["age"].unique())

    # Education labels
    education_labels = specs.get("education_labels", ["Low", "High"])
    partner_labels = ["No Partner", "Working Partner", "Retired Partner"]

    # Colors for education levels (using tab10 colormap)
    edu_colors = [plt.cm.tab10(i) for i in range(len(education_labels))]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot lines for each combination of education and partner state
    linestyles = ["-", "--", "-."]  # Different line styles for partner states
    for edu_var, edu_label in enumerate(education_labels):
        color = edu_colors[edu_var]
        df_edu = share_by_group.loc[share_by_group["education"] == edu_var].copy()

        if len(df_edu) > 0:
            for partner_idx, (partner_col, partner_label) in enumerate(
                zip(partner_state_cols, partner_labels, strict=True)
            ):
                values = df_edu.set_index("age")[partner_col].reindex(ages).values
                mask = ~np.isnan(values)
                if mask.sum() > 0:
                    label = f"{edu_label} - {partner_label}"
                    ax.plot(
                        ages[mask],
                        values[mask],
                        linewidth=2,
                        color=color,
                        linestyle=linestyles[partner_idx],
                        label=label,
                        alpha=0.8,
                    )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Share of Agents", fontsize=12)
    ax.set_title(
        "Partner State Shares by Age and Education (Baseline Model)", fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, ncol=2)
    ax.set_ylim(0, 1)  # Shares are between 0 and 1

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {path_to_plot}")


@pytask.mark.no_care_demand_model
@pytask.mark.post_estimation
def task_plot_partner_state_by_age_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "partner_state_by_age_no_care_demand.png",
):
    """Plot partner state shares by age and education from no care demand counterfactual simulated data.

    Creates a plot with 6 lines:
    - Low education: No partner, Working partner, Retired partner
    - High education: No partner, Working partner, Retired partner

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to no care demand counterfactual simulated data pkl file
    path_to_plot : Path
        Path to save the plot
    """
    # Load specs and simulated data
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    df_sim = pd.read_pickle(path_to_simulated_data)

    # Check if agent and period are in the index (MultiIndex) or already columns
    if isinstance(df_sim.index, pd.MultiIndex):
        df_sim = df_sim.reset_index()
    elif "agent" not in df_sim.columns:
        if hasattr(df_sim.index, "names") and "agent" in df_sim.index.names:
            df_sim = df_sim.reset_index()
        else:
            raise ValueError(
                f"Cannot find 'agent' column or index level. "
                f"Available columns: {df_sim.columns.tolist()}, "
                f"Index names: {df_sim.index.names if hasattr(df_sim.index, 'names') else 'N/A'}"
            )

    # Verify agent and period columns exist
    if "agent" not in df_sim.columns or "period" not in df_sim.columns:
        raise ValueError(
            f"Missing required columns 'agent' or 'period'. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Create age variable if not already present
    if "age" not in df_sim.columns:
        df_sim["age"] = df_sim["period"] + specs["start_age"]

    # Verify required columns exist
    required_cols = ["education", "partner_state"]
    missing_cols = [col for col in required_cols if col not in df_sim.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Create binary indicators for each partner state
    df_sim["no_partner"] = (df_sim["partner_state"] == NO_PARTNER).astype(float)
    df_sim["working_partner"] = (df_sim["partner_state"] == PARTNER_WORKING).astype(
        float
    )
    df_sim["retired_partner"] = (df_sim["partner_state"] == PARTNER_RETIRED).astype(
        float
    )

    # Calculate share (mean) by age and education for each partner state
    partner_state_cols = ["no_partner", "working_partner", "retired_partner"]
    share_by_group = (
        df_sim.groupby(["age", "education"], observed=False)[partner_state_cols]
        .mean()
        .reset_index()
    )

    # Get all unique ages
    ages = np.sort(df_sim["age"].unique())

    # Education labels
    education_labels = specs.get("education_labels", ["Low", "High"])
    partner_labels = ["No Partner", "Working Partner", "Retired Partner"]

    # Colors for education levels (using tab10 colormap)
    edu_colors = [plt.cm.tab10(i) for i in range(len(education_labels))]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot lines for each combination of education and partner state
    linestyles = ["-", "--", "-."]  # Different line styles for partner states
    for edu_var, edu_label in enumerate(education_labels):
        color = edu_colors[edu_var]
        df_edu = share_by_group.loc[share_by_group["education"] == edu_var].copy()

        if len(df_edu) > 0:
            for partner_idx, (partner_col, partner_label) in enumerate(
                zip(partner_state_cols, partner_labels, strict=True)
            ):
                values = df_edu.set_index("age")[partner_col].reindex(ages).values
                mask = ~np.isnan(values)
                if mask.sum() > 0:
                    label = f"{edu_label} - {partner_label}"
                    ax.plot(
                        ages[mask],
                        values[mask],
                        linewidth=2,
                        color=color,
                        linestyle=linestyles[partner_idx],
                        label=label,
                        alpha=0.8,
                    )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Share of Agents", fontsize=12)
    ax.set_title(
        "Partner State Shares by Age and Education (No Care Demand Counterfactual)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, ncol=2)
    ax.set_ylim(0, 1)  # Shares are between 0 and 1

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {path_to_plot}")
